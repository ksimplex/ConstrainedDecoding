"""
Token-class projection: map a model vocabulary onto a small constraint alphabet.

Constraints are defined over A symbol classes; the decoder works with class log-probabilities
    lw[b, i, c] = log sum_{v : class(v) = c} p(v | x)
and draws the token within the chosen class proportionally to p(v | x). Because the constraint depends only
on classes, this is exact: the Doob law over tokens is the native law conditioned on the class sequence
lying in C. Excluded tokens (the mask token) belong to no class and are never drawn.

Two constructions:
  semantic_classes  4 classes from the GPT-2 token string (word start / continuation / space+non-letter /
                    other), used in E2; .merge(map) gives coarser alphabets.
  ranked_classes    TWISTER Sec. 4.3 (arXiv:2609.35609): rank tokens by probability under a fully masked
                    input, split the top pool equally (after a random permutation) into n classes, and put
                    every other token in an extra class "x" (index n).
"""
import torch
import torch.nn.functional as F


class TokenClasses:
    def __init__(self, token_class, A, excluded=()):
        """token_class: (V,) long tensor of class ids in [0, A); `excluded` token ids are never drawn."""
        self.token_class = token_class.clone().long()
        self.A = int(A)
        self.onehot = F.one_hot(self.token_class.clamp_min(0), self.A).float()   # (V, A)
        for v in excluded:
            self.onehot[v] = 0.0
        self.excluded = tuple(excluded)

    def to(self, device):
        self.token_class, self.onehot = self.token_class.to(device), self.onehot.to(device)
        return self

    def class_logw(self, logp):
        """(B, N, V) token log-probs -> (B, N, A) class log-probs."""
        return torch.log((torch.exp(logp) @ self.onehot).clamp_min(1e-38))

    def merge(self, mapping):
        """Coarsen: new class of token v is mapping[old class of v]."""
        m = torch.tensor(mapping, device=self.token_class.device)
        return TokenClasses(m[self.token_class], max(mapping) + 1, self.excluded)

    def sizes(self):
        return torch.bincount(self.token_class, minlength=self.A).tolist()


def semantic_classes(tokenizer, V, mask_id):
    """E2 classes from the token string: 0 space+letter, 1 letter, 2 space+non-letter, 3 other."""
    cls = torch.full((V,), 3, dtype=torch.long)
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        sp = s.startswith(" ")
        alpha = (s[1:] if sp else s)[:1].isalpha()
        cls[i] = (0 if alpha else 2) if sp else (1 if alpha else 3)
    return TokenClasses(cls, 4, excluded=(mask_id,))


def ranked_classes(p_masked, n_classes, pool_size, mask_id, seed=0):
    """TWISTER Sec. 4.3 classes. p_masked: (V,) token probabilities under the fully masked input (e.g.
    averaged over positions). The top `pool_size` tokens are permuted with `seed` and split equally into
    classes 0..n_classes-1; all other tokens go to class n_classes ("x"). Returns TokenClasses with A = n+1."""
    p = p_masked.detach().float().cpu().clone()
    p[mask_id] = -1.0
    top = torch.argsort(p, descending=True)[:pool_size]
    g = torch.Generator().manual_seed(seed)
    top = top[torch.randperm(len(top), generator=g)]
    cls = torch.full((len(p),), n_classes, dtype=torch.long)
    per = pool_size // n_classes
    for c in range(n_classes):
        cls[top[c * per:(c + 1) * per]] = c
    return TokenClasses(cls, n_classes + 1, excluded=(mask_id,))
