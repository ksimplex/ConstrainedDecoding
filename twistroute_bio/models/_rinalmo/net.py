"""
RiNALMo encoder as used by mRNAutilus, re-expressed so it loads the published checkpoint without flash-attn.

Provenance: mirrors `src/rinalmo/model/{model,modules,attention,rope}.py` of atombio/mrnautilus (HF,
CC BY-NC 4.0). Parameter names and shapes are those of the checkpoint (`model.net.rm.*`), so weights load
directly. Two attention implementations:

  "torch"  the upstream class's own non-flash branch (FlashMultiHeadSelfAttention.forward with
           return_attn_probs=True): the same fused Wqkv, the same rotary convention, explicit softmax
           attention. Runs anywhere.
  "flash"  flash_attn_qkvpacked_func, the path used for training and for the authors' sampling. Requires a
           GPU and flash-attn.

The rotary convention matches flash-attn's RotaryEmbedding(interleaved=False): with cos/sin built from
cat((freqs, freqs)), the half-rotation below is the same map. The two implementations should agree to
floating-point tolerance; `scripts/check_flash_equivalence.py` tests that on a GPU, and until it has been run
the "torch" path should be treated as unverified against the real kernel.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn


def _rotary_cos_sin(seq_len, dim, device, dtype, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)          # positions in fp32
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


class SelfAttention(nn.Module):
    """Fused-QKV multi-head self-attention with rotary embeddings (checkpoint names: Wqkv, out_proj)."""

    def __init__(self, embed_dim, num_heads, bias=False, impl="torch"):
        super().__init__()
        self.h, self.d, self.impl = num_heads, embed_dim // num_heads, impl
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, key_padding_mask=None):
        B, S, _ = x.shape
        qkv = self.Wqkv(x).view(B, S, 3, self.h, self.d)                   # 'b s (three h d) -> b s three h d'
        cos, sin = _rotary_cos_sin(S, self.d, x.device, qkv.dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        q = qkv[:, :, 0] * cos + _rotate_half(qkv[:, :, 0]) * sin
        k = qkv[:, :, 1] * cos + _rotate_half(qkv[:, :, 1]) * sin
        v = qkv[:, :, 2]
        if self.impl == "flash":
            from flash_attn import flash_attn_qkvpacked_func
            out = flash_attn_qkvpacked_func(torch.stack([q, k, v], 2), causal=False)
        else:
            qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))            # (B, h, S, d)
            attn = qh @ kh.transpose(-1, -2) / math.sqrt(self.d)
            if key_padding_mask is not None:
                attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            out = (attn.softmax(-1) @ vh).transpose(1, 2)                  # (B, S, h, d)
        return self.out_proj(out.reshape(B, S, self.h * self.d))


class SwiGLU(nn.Module):
    """Checkpoint names: transition.0.linear, transition.0.linear_gate, transition.0.beta."""

    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        self.linear_gate = nn.Linear(c_in, c_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        lin = self.linear(x)                                               # swish applies to `linear`,
        return lin * torch.sigmoid(self.beta * lin) * self.linear_gate(x)  # the gate multiplies it


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, transition_factor=4, bias=False, impl="torch"):
        super().__init__()
        hidden = int(2 / 3 * transition_factor * embed_dim)
        self.mh_attn = SelfAttention(embed_dim, num_heads, bias=bias, impl=impl)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.transition = nn.Sequential(SwiGLU(embed_dim, hidden), nn.Dropout(0.0),
                                        nn.Linear(hidden, embed_dim, bias=True))
        self.out_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        x = self.attn_layer_norm(x)
        x = x + self.mh_attn(x, key_padding_mask)
        return x + self.transition(self.out_layer_norm(x))


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, transition_factor=4, impl="torch"):
        super().__init__()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, transition_factor, impl=impl)
                                     for _ in range(num_blocks)])
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        for b in self.blocks:
            x = b(x, key_padding_mask)
        return self.final_layer_norm(x)


class MaskedLanguageModelHead(nn.Module):
    def __init__(self, embed_dim, alphabet_size):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, alphabet_size)

    def forward(self, x):
        return self.linear2(self.layer_norm(F.gelu(self.linear1(x))))


class TokenDropout(nn.Module):
    """ESM-style token dropout (upstream `TokenDropout`, active=True in the RiNALMo config, so it applies at
    inference as well as training): zero the embedding at masked positions, then rescale the whole sequence by
    (1 - mask_ratio_train) / (1 - mask_ratio_observed). It has no parameters, so it is invisible in the
    checkpoint; omitting it changes every prediction."""

    def __init__(self, mask_idx, pad_idx, mask_ratio=0.15, mask_tkn_prob=0.8, active=True):
        super().__init__()
        self.mask_idx, self.pad_idx, self.active = mask_idx, pad_idx, active
        self.mask_ratio_train = mask_ratio * mask_tkn_prob

    def forward(self, x, tokens):
        if not self.active:
            return x
        is_mask = tokens == self.mask_idx
        src_lens = (~tokens.eq(self.pad_idx)).sum(-1)
        x = torch.where(is_mask[..., None], torch.zeros_like(x), x)
        observed = is_mask.sum(-1) / src_lens
        return x * (1 - self.mask_ratio_train) / (1 - observed[..., None, None])


class RiNALMo(nn.Module):
    """Encoder with the checkpoint's module names (embedding, transformer, lm_mask_head), plus the
    parameter-free TokenDropout."""

    def __init__(self, alphabet_size=86, embed_dim=640, num_blocks=30, num_heads=20, pad_idx=1, mask_idx=4,
                 impl="torch"):
        super().__init__()
        self.pad_idx, self.mask_idx = pad_idx, mask_idx
        self.embedding = nn.Embedding(alphabet_size, embed_dim, padding_idx=pad_idx)
        self.token_dropout = TokenDropout(mask_idx, pad_idx)
        self.transformer = Transformer(embed_dim, num_blocks, num_heads, impl=impl)
        self.lm_mask_head = MaskedLanguageModelHead(embed_dim, alphabet_size)

    def forward(self, tokens):
        pad = tokens.eq(self.pad_idx)
        x = self.token_dropout(self.embedding(tokens), tokens)
        x = self.transformer(x, pad if pad.any() else None)
        return self.lm_mask_head(x)


def load_rinalmo(ckpt_path, impl="torch", device="cpu", dtype=torch.float32):
    """Build the encoder and load `model.net.rm.*` from an mRNAutilus Lightning checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    pre = "model.net.rm."
    got = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if not got:
        raise ValueError(f"no tensors under {pre!r} in the checkpoint")
    emb = got["embedding.weight"]
    nblocks = 1 + max(int(k.split(".")[2]) for k in got if k.startswith("transformer.blocks."))
    head_dim = 32                                                          # RiNALMo-150M: 640 / 20 heads
    net = RiNALMo(alphabet_size=emb.shape[0], embed_dim=emb.shape[1], num_blocks=nblocks,
                  num_heads=emb.shape[1] // head_dim, impl=impl)
    missing, unexpected = net.load_state_dict(got, strict=False)
    missing = [k for k in missing if "token_dropout" not in k]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing {missing[:6]}, unexpected {unexpected[:6]}")
    return net.to(device=device, dtype=dtype).eval()
