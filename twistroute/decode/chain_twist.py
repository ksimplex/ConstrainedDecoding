"""
A first-order chain twist for automaton constraints: an intermediate twist that is not mean-field.

The sampler's standard twist is the constraint's probability when the masked positions are independent with the
model's current marginals mu_i (the product twist). This twist keeps those marginals and adds the dependence between
adjacent positions, taken from one table K(a, b) = p(class b follows class a) fitted before the run:

    transition   Q_i(a, b)  proportional to  mu_{i+1}(b) K(a, b) / sum_a' mu_i(a') K(a', b),   normalised over b
    chain law    P(y) = mu_0(y_0) prod_i Q_i(y_i, y_{i+1})
    twist        psi_chain = P(y in C)

Q_i reweights mu_{i+1} by how much more likely b is after a than after a typical predecessor, so the chain keeps
the model's marginals to first order and needs no extra denoiser call. A revealed position has a one-hot mu, and the
formula then returns mu_{i+1} unchanged: the denoiser has already conditioned on that token, so nothing is added.

Exactness does not depend on this twist being right. It enters the weights as an intermediate potential only: the
proposal and the terminal twist (the exact indicator) are untouched, so Z_hat stays unbiased for any positive twist.
`log_psi` mixes in a share eps of the product twist, which keeps the twist positive wherever the product twist is.
With `temp` < 1 the chain in `log_psi` is built on flattened marginals (the calibrated twist of the product case,
applied to the chain); the twist is still positive wherever the product twist is, so exactness is unaffected.
Without resampling the final weights are identical to those of the product twist (the intermediate twists
telescope); what changes is how well the weights part-way through predict the final ones, which is what resampling
acts on.

`marginals` returns the constrained class marginals under the chain law, at every position. Their KL divergence from
the product twist's constrained marginals is an estimate of the product twist's error at a position.

The automaton may be restricted to a subset of positions (`active`), as in torch_routes.TorchSubset; the chain still
runs through every position.
"""
import math
import torch


def fit_pair_kernel(den, classes, n, gen=None, smooth=1e-3, N=None):
    """K (A, A), row-stochastic: the class at position i + 1 given the class at position i, pooled over positions,
    from n sequences drawn from the model (den.sample when the model has an exact sampler, else one token per step,
    which costs N denoiser calls on a batch of n; N defaults to den.N)."""
    if hasattr(den, "sample"):
        x = den.sample(n, gen)
    else:
        from .native import native_sample
        N = den.N if N is None else N
        x = native_sample(den, classes, N, N, n, gen)
    y = classes.token_class[x]
    A = classes.A
    J = torch.bincount((y[:, :-1] * A + y[:, 1:]).reshape(-1), minlength=A * A).double().view(A, A) + smooth
    return J / J.sum(-1, keepdim=True)


class ChainTwist:
    def __init__(self, wdfa, kernel, eps=0.05, active=None, device="cpu", temp=1.0):
        """wdfa: compile.WDFA; kernel: (A, A) row-stochastic with positive entries; eps in (0, 1]: share of the
        product twist in `log_psi`; active: optional (N,) bool, the positions the automaton reads; temp in (0, 1]:
        calibration, the chain in `log_psi` is built on the marginals raised to this power and renormalised
        (mu_i^temp at masked positions; a revealed position is one-hot and is unchanged). 1 = no calibration."""
        e = torch.tensor(wdfa.edges, dtype=torch.long).reshape(-1, 3)
        self.src, self.sym, self.dst = (e[:, k].to(device) for k in range(3))
        self.Q, self.q0, self.A = wdfa.n, wdfa.q0, wdfa.A
        self.F = torch.zeros(self.Q, dtype=torch.float64, device=device)
        self.F[sorted(wdfa.F)] = 1.0
        self.K = torch.as_tensor(kernel, dtype=torch.float64, device=device)
        if self.K.shape != (self.A, self.A) or bool((self.K <= 0).any()):
            raise ValueError("kernel must be (A, A) with positive entries")
        if not 0.0 < eps <= 1.0:
            raise ValueError("eps must lie in (0, 1]")
        self.eps = float(eps)
        if not 0.0 < temp <= 1.0:
            raise ValueError("temp must lie in (0, 1]")
        self.temp = float(temp)
        self.active = None if active is None else torch.as_tensor(active, dtype=torch.bool, device=device)

    def transitions(self, mu):
        """(B, N, A) marginals -> (B, N - 1, A, A) row-stochastic transitions Q_i(a, b)."""
        kbar = torch.einsum("bia,ac->bic", mu[:, :-1], self.K).clamp_min(1e-300)
        q = mu[:, 1:, None, :] * self.K[None, None] / kbar[:, :, None, :]
        return q / q.sum(-1, keepdim=True).clamp_min(1e-300)

    def _read(self, g, i):
        """g[b, c, q]: mass with class c at position i and the automaton in state q before reading it. Returns the
        mass after the automaton has read c (unchanged at a position it does not read)."""
        if self.active is not None and not bool(self.active[i]):
            return g
        v = g[:, self.sym, self.src]                                        # (B, E)
        out = torch.zeros(g.shape[0], self.A * self.Q, dtype=g.dtype, device=g.device)
        return out.index_add(1, self.sym * self.Q + self.dst, v).view(g.shape)

    def log_chain(self, lwc, theta=None, Qc=None):
        """(B, N, A) class log-weights with revealed positions clamped -> (B,) float64 log psi_chain.
        `theta` (B, N, A) adds a log-weight per position and class (used for the marginals). `Qc` (B, N - 1, A, A)
        replaces the transitions (for instance the model's own pair conditionals, where they are available)."""
        w = torch.exp(lwc.double())
        mu = w / w.sum(-1, keepdim=True).clamp_min(1e-300)
        if Qc is None:
            Qc = self.transitions(mu)
        B, N, A = mu.shape
        first = mu[:, 0] if theta is None else mu[:, 0] * torch.exp(theta[:, 0])
        start = torch.zeros(self.Q, dtype=torch.float64, device=mu.device); start[self.q0] = 1.0
        f = self._read(first[:, :, None] * start[None, None, :], 0)         # (B, A, Q)
        logs = torch.zeros(B, dtype=torch.float64, device=mu.device)
        for i in range(N - 1):
            s = f.sum((1, 2))
            logs = logs + torch.log(s)
            f = f / s.clamp_min(1e-300)[:, None, None]
            g = torch.einsum("baq,bac->bcq", f, Qc[:, i])
            if theta is not None:
                g = g * torch.exp(theta[:, i + 1])[:, :, None]
            f = self._read(g, i + 1)
        return logs + torch.log((f.sum(1) * self.F[None]).sum(-1))

    def log_psi(self, lwc, log_product):
        """The twist used in the weights: log((1 - eps) psi_chain + eps psi_product), (B,) float64."""
        if self.eps >= 1.0:
            return log_product
        if self.temp != 1.0:                       # calibrated: flatter marginals at the masked positions
            lwc = torch.log_softmax(self.temp * lwc.double(), -1)
        return torch.logaddexp(self.log_chain(lwc) + math.log1p(-self.eps), log_product + math.log(self.eps))

    def marginals(self, lwc, Qc=None):
        """(B, N, A) float64: class marginals of the chain law given the constraint; zeros where it has no mass."""
        with torch.enable_grad():
            theta = torch.zeros(lwc.shape, dtype=torch.float64, device=lwc.device, requires_grad=True)
            g, = torch.autograd.grad(self.log_chain(lwc.detach(), theta, Qc).sum(), theta)
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
