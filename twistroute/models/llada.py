"""
LLaDA-8B (Nie et al., "Large Language Diffusion Models", arXiv:2502.09992; checkpoint GSAI-ML/LLaDA-8B-Instruct), a
masked diffusion language model trained from scratch, with a chat prompt as a fixed prefix.

Denoiser interface used by twistroute.decode (see models/mdlm_owt.py):
    mask_id          token id of the mask (126336)
    V                vocabulary size of the output layer (126464)
    logits(x)        (B, N) token ids of the generation region -> (B, N, V) float32 log-probabilities, mask excluded

set_prompt(messages) fixes the prefix: the chat template applied to `messages` with the assistant header appended.
The model is called on [prefix, x]; only the generation region is returned. Rows are processed `max_rows` at a time.
"""
import torch


class LLaDA:
    def __init__(self, device="cuda", dtype=torch.bfloat16, model_id="GSAI-ML/LLaDA-8B-Instruct", max_rows=16):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.m = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.mask_id = int(getattr(self.m.config, "mask_token_id", 126336)); self.V = int(self.m.config.vocab_size)
        self.end_ids = (int(self.tok.convert_tokens_to_ids("<|eot_id|>")), int(self.tok.eos_token_id))
        self.device, self.dtype, self.max_rows, self.prefix, self.calls = device, dtype, max_rows, None, 0

    def info(self):
        return dict(V=self.V, mask_id=self.mask_id, end_ids=list(self.end_ids), prefix_tokens=None if self.prefix is None else int(self.prefix.numel()))

    def set_prompt(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        self.prefix = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return int(self.prefix.numel())

    @torch.no_grad()
    def logits(self, x):
        if self.prefix is None:
            raise RuntimeError("call set_prompt first")
        out = []
        for lo in range(0, x.shape[0], self.max_rows):
            xb = x[lo:lo + self.max_rows]
            inp = torch.cat([self.prefix[None].expand(xb.shape[0], -1), xb], 1)
            lg = self.m(input_ids=inp).logits[:, self.prefix.numel():, :].float()
            lg[..., self.mask_id] = float("-inf")
            out.append(torch.log_softmax(lg, -1)); self.calls += 1
        return torch.cat(out, 0)
