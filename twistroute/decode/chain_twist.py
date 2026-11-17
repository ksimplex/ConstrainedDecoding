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
    def __init__(self, wdfa, kernel, eps=0.05, active=None, device="cpu", temp=1.0, match=0):
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
        self.match = int(match)                             # sweeps of marginal matching in `transitions` (0 = the ratio formula)
        self.active = None if active is None else torch.as_tensor(active, dtype=torch.bool, device=device)
        self.delta = torch.full((self.Q, self.A), -1, dtype=torch.long, device=device)   # next state, -1 = no move
        self.delta[self.src, self.sym] = self.dst

    def transitions(self, mu):
        """(B, N, A) marginals -> (B, N - 1, A, A) row-stochastic transitions Q_i(a, b).

        match = 0: the ratio formula of the module docstring. It keeps the marginals to first order only, and it is
        not exact for a first-order chain once tokens elsewhere are revealed.
        match > 0: for every adjacent pair, the joint J_i(a, b) = u(a) K(a, b) v(b) whose two marginals are exactly
        mu_i and mu_{i+1}, found by `match` sweeps of row and column scaling; Q_i(a, b) = J_i(a, b) / mu_i(a). Of
        all joints with those marginals it is the closest to the table's dependence (in KL), the chain law then has
        the model's marginal at every position, and for a first-order Markov model with table K it is the exact
        joint of the masked positions given any revealed tokens. No denoiser call."""
        if self.match <= 0:
            kbar = torch.einsum("bia,ac->bic", mu[:, :-1], self.K).clamp_min(1e-300)
            q = mu[:, 1:, None, :] * self.K[None, None] / kbar[:, :, None, :]
            return q / q.sum(-1, keepdim=True).clamp_min(1e-300)
        r, c = mu[:, :-1], mu[:, 1:]                                         # row and column marginals, (B, N - 1, A)
        v = torch.ones_like(c)
        for _ in range(self.match):
            u = r / torch.einsum("ac,bic->bia", self.K, v).clamp_min(1e-300)
            v = c / torch.einsum("bia,ac->bic", u, self.K).clamp_min(1e-300)
        u = r / torch.einsum("ac,bic->bia", self.K, v).clamp_min(1e-300)     # rows exact: Q is row-stochastic where mu_i(a) > 0
        q = self.K[None, None] * v[:, :, None, :] / torch.einsum("ac,bic->bia", self.K, v).clamp_min(1e-300)[..., None]
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

    def sample(self, lwc, gen=None, chunk=1024):
        """Draw class sequences from the chain law given the constraint. Returns (y (B, N) long, log P_chain(C) (B,)).
        One backward pass stores, for every position, class and automaton state, the chance of ending in an
        accepting state; a forward sweep then draws position by position. A row whose constraint has no mass under
        the chain law returns log P_chain(C) = -inf and an arbitrary y."""
        ys, lcs = [], []
        for lo in range(0, lwc.shape[0], chunk):
            y, lc = self._sample(lwc[lo:lo + chunk], gen); ys.append(y); lcs.append(lc)
        return torch.cat(ys), torch.cat(lcs)

    def _sample(self, lwc, gen):
        w = torch.exp(lwc.double()); mu = w / w.sum(-1, keepdim=True).clamp_min(1e-300)
        Qc = self.transitions(mu); B, N, A = mu.shape; Q, dev = self.Q, mu.device
        nxt = self.delta.T                                                   # (A, Q): state after reading class a in state q
        ok = (nxt >= 0).to(torch.float64)[None]; idx = nxt.clamp_min(0)[None].expand(B, A, Q)

        def read(v, i):                 # v[b, a, q']: value by the state after reading a at i -> by the state before
            if self.active is not None and not bool(self.active[i]):
                return v
            return torch.gather(v, 2, idx) * ok

        bm = torch.zeros(B, N, A, Q, dtype=torch.float64, device=dev)       # P(accept | class a at i, state q before reading it), scaled
        bm[:, N - 1] = read(self.F[None, None, :].expand(B, A, Q), N - 1)
        logs = torch.zeros(B, dtype=torch.float64, device=dev)
        for i in range(N - 2, -1, -1):
            g = torch.einsum("bac,bcq->baq", Qc[:, i], bm[:, i + 1])
            sc = g.amax((1, 2)); logs = logs + torch.log(sc); g = g / sc.clamp_min(1e-300)[:, None, None]
            bm[:, i] = read(g, i)
        rows = torch.arange(B, device=dev)

        def draw(p):
            tot = p.sum(-1, keepdim=True)
            p = torch.where(tot > 0, p, torch.ones_like(p))                  # dead row: arbitrary class
            return torch.multinomial(p, 1, generator=gen).squeeze(-1)

        def move(q, c, i):
            if self.active is not None and not bool(self.active[i]):
                return q
            return self.delta[q, c].clamp_min(0)

        p0 = mu[:, 0] * bm[:, 0, :, self.q0]
        logc = logs + torch.log(p0.sum(-1))
        y = torch.zeros(B, N, dtype=torch.long, device=dev)
        q = torch.full((B,), self.q0, dtype=torch.long, device=dev)
        y[:, 0] = draw(p0); q = move(q, y[:, 0], 0)
        for i in range(N - 1):
            y[:, i + 1] = draw(Qc[rows, i, y[:, i]] * bm[rows, i + 1, :, q]); q = move(q, y[:, i + 1], i + 1)
        return y, logc

    def free(self):
        """The same chain law with no constraint (an automaton that accepts everything): `free().log_clamped` is then
        the chain law's joint log-probability of classes at a set of positions."""
        if getattr(self, "_free", None) is None:
            from types import SimpleNamespace
            d = SimpleNamespace(edges=[(0, a, 0) for a in range(self.A)], n=1, q0=0, A=self.A, F={0})
            self._free = ChainTwist(d, self.K, eps=self.eps, device=self.K.device, match=self.match)
        return self._free

    def log_reference(self, lwc, lw, r, cr, rvalid=None):
        """log of the reference law of the classes cr revealed at positions r in one step: the model's own class
        probability when a particle reveals one position, the chain law's joint when it reveals two or more.
        lw (B, N, A) are the model's class log-probabilities (unclamped). Returns (B,) float64."""
        keep = torch.ones_like(r, dtype=torch.bool) if rvalid is None else rvalid
        native = torch.gather(lw, 1, r[..., None].expand(-1, -1, lw.shape[-1])).gather(2, cr[..., None]).squeeze(-1).double()
        native = torch.where(keep, native, torch.zeros_like(native)).sum(-1)
        many = keep.sum(-1) > 1
        if not bool(many.any()):
            return native
        out = native.clone()
        out[many] = self.free().log_clamped(lwc[many], r[many], cr[many], keep[many])
        return out

    def log_clamped(self, lwc, r, cr, rvalid=None):
        """log P_chain(classes cr at positions r, and C): (B,) float64. r, cr (B, n) long; rvalid (B, n) bool marks
        the columns that count (None = all). The chain law itself is unchanged: its transitions come from lwc."""
        theta = torch.zeros(lwc.shape, dtype=torch.float64, device=lwc.device)
        keep = torch.ones_like(r, dtype=torch.bool) if rvalid is None else rvalid
        b = torch.arange(r.shape[0], device=r.device)[:, None].expand_as(r)[keep]
        theta[b, r[keep]] = float("-inf"); theta[b, r[keep], cr[keep]] = 0.0
        return self.log_chain(lwc, theta)
