"""
OWT-130M masked diffusion LM (Sahoo et al., "Simple and Effective Masked Diffusion Language Models",
arXiv:2406.07524; checkpoint kuleshov-group/mdlm-owt). Requires flash_attn (imported by the remote code).

Denoiser interface used by twistroute.decode (any model exposing the same three attributes works):
    mask_id          token id of the mask
    V                vocabulary size including the mask
    logits(x)        (B, N) token ids -> (B, N, V) float32 log-probabilities, mask excluded
"""
import torch


class MDLM:
    def __init__(self, device="cuda", dtype=torch.bfloat16, model_id="kuleshov-group/mdlm-owt"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.m = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        cfg = self.m.config
        self.V = int(getattr(cfg, "vocab_size", 50258))
        self.mask_id = self.V - 1 if self.V > len(self.tok) else int(getattr(cfg, "mask_index", 50257))
        # The released checkpoint is not time-conditioned (config.time_conditioning = False), so sigma = 0
        # matches training. If it were, sigma would follow the log-linear schedule at the masked fraction.
        self.time_cond = bool(getattr(cfg, "time_conditioning", False))
        self.device, self.dtype = device, dtype

    def info(self):
        return dict(V=self.V, mask_id=self.mask_id, time_conditioning=self.time_cond)

    @torch.no_grad()
    def logits(self, x):
        t = (x == self.mask_id).float().mean(-1)
        sig = -torch.log1p(-(1 - 1e-3) * t) if self.time_cond else torch.zeros_like(t)
        with torch.autocast("cuda", dtype=self.dtype, enabled=(self.device == "cuda")):
            out = self.m(input_ids=x, timesteps=sig)
        lg = (out.logits if hasattr(out, "logits") else out).float()
        lg[..., self.mask_id] = float("-inf")
        return torch.log_softmax(lg, -1)
