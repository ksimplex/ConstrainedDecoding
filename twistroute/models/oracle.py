"""
Oracle denoisers: exact single-position conditionals of a known hidden Markov model, with the same
interface as models.mdlm_owt.MDLM and models.synthetic.SyntheticMDLM.

    logits(x)[b, i, v] = log p(x_i = v | the revealed tokens of x[b] other than position i)

Because these are the exact conditionals of one joint law p, revealing positions one at a time in ANY
order (adaptive orders included) samples p exactly: the one-at-a-time native law is p itself and does not
depend on the schedule. That makes three things exact that are otherwise only estimable:
    the dependence structure   set by construction (chain, hidden chain, independent segments, mean-field);
    the constrained target     p(. | C) for an automaton constraint, by a forward pass over (hidden state,
                               automaton state) -- see `constrained_truth`;
    the parallel-reveal gap    r(x) = log p(x) - sum over reveal steps of the product-form log-probability
                               the schedule actually used -- see `log_joint` and `parallel_log_prob`.
A schedule that reveals several positions per step samples them independently given the context, which
differs from p whenever those positions are dependent under p; the oracle is the testbed for schedules that
choose which positions to reveal together.

Model. Hidden chain h_0..h_{N-1} over H states with initial law `pi`, transitions `trans[i]` (h_i -> h_{i+1},
one matrix per gap, so the chain may be inhomogeneous) and emissions `emit[i]` (h_i -> token). With H = S and
identity emissions it is a Markov chain over tokens, where a revealed position screens its two sides exactly.
With noisy emissions the dependence between tokens still decays with distance but no single revealed token
screens (the hidden state is never observed).

All inference is the scaled forward-backward recursion in float64 (Rabiner, Proc. IEEE 77(2):257-286, 1989,
doi:10.1109/5.18626), batched over rows: O(B N H^2) per call.
"""
import numpy as np
import torch

from .classes import TokenClasses

NEG_INF = float("-inf")


class OracleHMM:
    def __init__(self, pi, trans, emit, device="cpu", name="oracle_hmm"):
        """pi (H,), trans (N-1, H, H) row-stochastic, emit (N, H, S) row-stochastic. Probabilities, not logs."""
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)
        self.pi, self.trans, self.emit = f(pi), f(trans), f(emit)
        self.N, self.H, self.S = self.emit.shape
        if self.trans.shape != (self.N - 1, self.H, self.H) or self.pi.shape != (self.H,):
            raise ValueError("shapes must be pi (H,), trans (N-1, H, H), emit (N, H, S)")
        for nm, a in (("pi", self.pi), ("trans", self.trans), ("emit", self.emit)):
            if (a < 0).any() or not torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-9):
                raise ValueError(f"{nm} must be non-negative with rows summing to 1")
        self.V, self.mask_id = self.S + 1, self.S
        # evidence table with one extra token column (the mask) that carries no information
        ones = torch.ones(self.N, self.H, 1, dtype=torch.float64, device=device)
        self._emit_ext = torch.cat([self.emit, ones], -1).permute(0, 2, 1).contiguous()     # (N, S+1, H)
        self.device, self.name = device, name

    # ------------------------------------------------------------------------------------------ inference
    def _evidence(self, x):
        """(B, N) tokens (mask_id = no evidence) -> (B, N, H) likelihood of each hidden state."""
        return self._emit_ext[torch.arange(self.N, device=x.device)[None], x]

    def _forward(self, e):
        """Predicted filters ahat[:, i] = p(h_i | evidence before i) and log p(evidence)."""
        B = e.shape[0]
        ahat = torch.empty(B, self.N, self.H, dtype=torch.float64, device=e.device)
        logp = torch.zeros(B, dtype=torch.float64, device=e.device)
        cur = self.pi.expand(B, -1)
        for i in range(self.N):
            ahat[:, i] = cur
            a = cur * e[:, i]
            s = a.sum(-1, keepdim=True)
            logp = logp + torch.log(s.squeeze(-1))
            a = a / s.clamp_min(1e-300)
            if i < self.N - 1:
                cur = a @ self.trans[i]
        return ahat, logp

    def _backward(self, e):
        """b[:, i] proportional to p(evidence after i | h_i), rows scaled to sum to 1."""
        B = e.shape[0]
        b = torch.empty(B, self.N, self.H, dtype=torch.float64, device=e.device)
        cur = torch.full((B, self.H), 1.0 / self.H, dtype=torch.float64, device=e.device)
        for i in range(self.N - 1, -1, -1):
            b[:, i] = cur
            if i > 0:
                nxt = (cur * e[:, i]) @ self.trans[i - 1].T
                cur = nxt / nxt.sum(-1, keepdim=True).clamp_min(1e-300)
        return b

    def log_cond(self, x):
        """(B, N) -> (B, N, S) float64: log p(x_i = v | revealed tokens other than i), at every position."""
        e = self._evidence(x)
        ahat, _ = self._forward(e)
        g = ahat * self._backward(e)
        g = g / g.sum(-1, keepdim=True).clamp_min(1e-300)                 # p(h_i | evidence except i)
        return torch.log(torch.einsum("bnh,nhs->bns", g, self.emit))

    def logits(self, x):
        """Denoiser interface: (B, N) -> (B, N, V) float32 log-probs, mask column -inf."""
        lc = self.log_cond(x)
        pad = torch.full(lc.shape[:-1] + (1,), NEG_INF, dtype=lc.dtype, device=lc.device)
        return torch.cat([lc, pad], -1).float()

    def log_joint(self, x):
        """(B, N) -> (B,) float64 log-probability of the revealed tokens of each row (masked ones summed out).
        For a fully revealed row this is log p(x), the one-at-a-time law of this denoiser."""
        return self._forward(self._evidence(x))[1]

    # ------------------------------------------------------------------------------------------ sampling
    def sample(self, B, gen=None):
        """B exact draws from p, shape (B, N)."""
        def draw(p):
            return torch.multinomial(p.float(), 1, generator=gen).squeeze(-1)
        h = draw(self.pi.expand(B, -1))
        x = torch.empty(B, self.N, dtype=torch.long, device=self.device)
        for i in range(self.N):
            x[:, i] = draw(self.emit[i][h])
            if i < self.N - 1:
                h = draw(self.trans[i][h])
        return x

    # ------------------------------------------------------------------------------------------ helpers
    def classes(self, token_class=None, A=None):
        """Token classes; by default every token is its own class (A = S)."""
        if token_class is None:
            token_class, A = torch.arange(self.V) % self.S, self.S
        tc = torch.as_tensor(token_class).long()
        return TokenClasses(tc, int(A if A is not None else tc[: self.S].max() + 1),
                            excluded=(self.mask_id,)).to(self.device)

    def pair_mi(self, i, j):
        """Exact mutual information (nats) between the tokens at positions i < j with nothing revealed."""
        if not 0 <= i < j < self.N:
            raise ValueError("need 0 <= i < j < N")
        hi = self.pi
        for k in range(i):
            hi = hi @ self.trans[k]
        m = torch.eye(self.H, dtype=torch.float64, device=self.device)
        for k in range(i, j):
            m = m @ self.trans[k]
        joint = self.emit[i].T @ (hi[:, None] * m) @ self.emit[j]         # (S, S)
        pi_, pj_ = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
        nz = joint > 0
        return float((joint[nz] * torch.log(joint[nz] / (pi_ * pj_)[nz])).sum())


def parallel_log_prob(den, x_before, x_after):
    """log-probability a product-form (parallel) reveal assigns to the tokens revealed between two states:
    sum over positions masked in x_before and revealed in x_after of log p(x_i | revealed tokens of x_before).
    Summed over the steps of a trajectory this is log p~(x) for that trajectory's reveal sets, and
    r = den.log_joint(x_final) - that sum is the pointwise parallel-reveal gap (0 for one reveal per step)."""
    new = (x_before == den.mask_id) & (x_after != den.mask_id)
    lc = den.log_cond(x_before)
    tok = torch.where(new, x_after, torch.zeros_like(x_after))
    return torch.where(new, torch.gather(lc, -1, tok[..., None]).squeeze(-1), torch.zeros_like(lc[..., 0])).sum(-1)


# ---------------------------------------------------------------------------------------------- constructors
def _stationary(S, seed, concentration):
    if concentration is None:
        return np.full(S, 1.0 / S)
    return np.random.default_rng(seed).dirichlet(np.full(S, float(concentration)))


def lazy_chain(N, S, lam, seed=0, concentration=None, breaks=(), device="cpu", pi=None):
    """Markov chain over S tokens with transition (1 - lam) 1 pi^T + lam I: keep the token with probability
    lam, otherwise redraw from pi. The d-step transition is (1 - lam^d) 1 pi^T + lam^d I, so the correlation
    between positions at distance d is exactly lam^d (correlation length -1 / log lam). `breaks` lists gaps
    i (between positions i and i + 1) where the chain restarts from pi: independent segments.
    pi is uniform, or Dirichlet(concentration) with the given seed, or given explicitly."""
    if not 0.0 <= lam < 1.0:
        raise ValueError("lam must lie in [0, 1)")
    pi = _stationary(S, seed, concentration) if pi is None else np.asarray(pi, dtype=float)
    if pi.shape != (S,) or (pi <= 0).any() or abs(pi.sum() - 1) > 1e-9:
        raise ValueError("pi must be a positive probability vector of length S")
    T = (1 - lam) * np.tile(pi, (S, 1)) + lam * np.eye(S)
    trans = np.tile(T, (N - 1, 1, 1))
    for g in breaks:
        trans[g] = np.tile(pi, (S, 1))
    return OracleHMM(pi, trans, np.tile(np.eye(S), (N, 1, 1)), device, name=f"lazy_chain(lam={lam})")


def repel_chain(N, S, mu, seed=0, concentration=None, device="cpu"):
    """Markov chain over S tokens whose neighbours are anticorrelated: transition (1 + mu) 1 pi^T - mu I, the
    lazy chain continued to a negative stay weight. A token is repeated less often than under independence;
    the stationary law is pi and the correlation at distance d is (-mu)^d, alternating in sign. Entries stay
    positive only for mu < min(pi) / (1 - min(pi)) (1 / (S - 1) for uniform pi); the bound itself would give
    a token probability zero after itself, which the sampler's positivity assumption excludes."""
    pi = _stationary(S, seed, concentration)
    cap = pi.min() / (1 - pi.min())
    if not 0.0 <= mu < cap:
        raise ValueError(f"mu must lie in [0, {cap:.4f}) for this stationary law")
    T = (1 + mu) * np.tile(pi, (S, 1)) - mu * np.eye(S)
    return OracleHMM(pi, np.tile(T, (N - 1, 1, 1)), np.tile(np.eye(S), (N, 1, 1)), device, name=f"repel_chain(mu={mu})")


def noisy_hmm(N, S, lam, noise, seed=0, concentration=None, device="cpu", pi=None):
    """Hidden lazy chain over S states (as in `lazy_chain`) observed through a noisy channel: the token equals
    the hidden state with probability 1 - noise, otherwise it is uniform. Token dependence decays with distance
    but a single revealed token does not screen its two sides."""
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must lie in [0, 1]")
    base = lazy_chain(N, S, lam, seed, concentration, pi=pi)
    E = (1 - noise) * np.eye(S) + noise / S
    return OracleHMM(base.pi.numpy(), base.trans.numpy(), np.tile(E, (N, 1, 1)), device,
                     name=f"noisy_hmm(lam={lam}, noise={noise})")


def independent(probs, device="cpu"):
    """Mean-field model: position i is drawn from probs[i] independently of the others. probs (N, S)."""
    probs = np.asarray(probs, float)
    N, S = probs.shape
    return OracleHMM(np.array([1.0]), np.ones((N - 1, 1, 1)), probs[:, None, :], device, name="independent")


# ---------------------------------------------------------------------------------------------- ground truth
class ConstrainedTruth:
    """Exact p(. | C) over class sequences for an OracleHMM and an automaton constraint (compile.WDFA).

    Forward-backward over the product state (hidden state, automaton state), O(N H^2 Q + N H |E|), float64
    with per-step scaling. Attributes: `log_pC` = log p(C); `marginals` (N, A) = p(class at i | C).
    `sample(n, rng)` draws class sequences from p(. | C) exactly by backward sampling.

    `active` (N,) bool restricts the constraint to a subset of positions: the automaton reads the classes at
    the active positions, in order, and ignores the others (the constraint of torch_routes.TorchSubset over
    TorchChain). Default: every position is active."""

    def __init__(self, den, classes, wdfa, active=None):
        H, N, A, Q = den.H, den.N, classes.A, wdfa.n
        onehot = classes.onehot[: den.S].double().cpu().numpy()             # (S, A); the mask token has no class
        self.Ec = den.emit.cpu().numpy() @ onehot                           # (N, H, A) class emissions
        self.T, self.pi = den.trans.cpu().numpy(), den.pi.cpu().numpy()
        e = np.asarray(wdfa.edges, dtype=np.int64).reshape(-1, 3)
        self.src, self.sym, self.dst = e[:, 0], e[:, 1], e[:, 2]
        self.active = np.ones(N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        if self.active.shape != (N,):
            raise ValueError("active must have one entry per position")
        qs = np.repeat(np.arange(Q), A)                                     # an inactive position leaves the
        self._idle = (qs, np.tile(np.arange(A), Q), qs)                     # automaton state where it is
        self.N, self.H, self.A, self.Q = N, H, A, Q
        self._q0 = wdfa.q0
        Fm = np.zeros(Q); Fm[sorted(wdfa.F)] = 1.0
        # alpha[i][h, q]: p(h_i = h, automaton state q after reading classes 0..i), scaled
        self.alpha, logscale = np.zeros((N, H, Q)), 0.0
        pred = np.zeros((H, Q)); pred[:, wdfa.q0] = self.pi                 # (h_0, state before position 0)
        for i in range(N):
            src, sym, dst = self._edges(i)
            a = np.zeros((H, Q))
            np.add.at(a, (slice(None), dst), pred[:, src] * self.Ec[i][:, sym])
            s = a.sum()
            if s <= 0:
                raise ValueError("the constraint has probability zero under the model")
            logscale += np.log(s); a /= s
            self.alpha[i] = a
            if i < N - 1:
                pred = self.T[i].T @ a
        tail = (self.alpha[-1] * Fm[None]).sum()
        if tail <= 0:
            raise ValueError("the constraint has probability zero under the model")
        self.log_pC = float(logscale + np.log(tail))
        # beta[i][h, q]: p(accept | h_i = h, automaton state q after position i), scaled
        self.beta = np.zeros((N, H, Q)); self.beta[-1] = Fm[None]
        self.beta[-1] /= self.beta[-1].sum()
        for i in range(N - 2, -1, -1):
            src, sym, dst = self._edges(i + 1)
            w = np.zeros((H, Q))                                            # weight of (h_{i+1}, state after i)
            np.add.at(w, (slice(None), src), self.Ec[i + 1][:, sym] * self.beta[i + 1][:, dst])
            b = self.T[i] @ w
            self.beta[i] = b / b.sum()
        self.marginals = np.zeros((N, A))
        pred = np.zeros((H, Q)); pred[:, wdfa.q0] = self.pi
        for i in range(N):
            src, sym, dst = self._edges(i)
            contrib = pred[:, src] * self.Ec[i][:, sym] * self.beta[i][:, dst]   # (H, E)
            np.add.at(self.marginals[i], sym, contrib.sum(0))
            self.marginals[i] /= self.marginals[i].sum()
            if i < N - 1:
                pred = self.T[i].T @ self.alpha[i]

    def _edges(self, i):
        """(src, sym, dst) of the automaton edges that apply at position i."""
        return (self.src, self.sym, self.dst) if self.active[i] else self._idle

    def pair_marginals(self):
        """(N - 1, A, A): exact joint law of the classes at positions (i, i + 1) under p(. | C)."""
        N, A, H, Q = self.N, self.A, self.H, self.Q
        out = np.zeros((N - 1, A, A))
        pred = np.zeros((H, Q)); pred[:, self._q0] = self.pi
        for i in range(N - 1):
            s1, y1, d1 = self._edges(i)
            s2, y2, d2 = self._edges(i + 1)
            left = np.zeros((H, Q, A))                                      # [h_i, state after i, class at i]
            np.add.at(left, (slice(None), d1, y1), pred[:, s1] * self.Ec[i][:, y1])
            right = np.zeros((H, Q, A))                                     # [h_{i+1}, state after i, class at i+1]
            np.add.at(right, (slice(None), s2, y2), self.Ec[i + 1][:, y2] * self.beta[i + 1][:, d2])
            out[i] = np.einsum("hqy,hk,kqz->yz", left, self.T[i], right)
            out[i] /= out[i].sum()
            pred = self.T[i].T @ self.alpha[i]
        return out

    def sample(self, n, rng):
        """(n, N) class sequences drawn exactly from p(. | C)."""
        N, H, Q = self.N, self.H, self.Q
        y = np.zeros((n, N), dtype=np.int64)
        # backward sampling: draw (h, q) after the last position, then (h_prev, edge) position by position
        w = (self.alpha[-1] * (self.beta[-1] > 0)).reshape(-1)
        hq = rng.choice(H * Q, size=n, p=w / w.sum())
        h, q = hq // Q, hq % Q
        for i in range(N - 1, -1, -1):
            src, sym, dst = self._edges(i)
            if i > 0:
                pred_all = self.alpha[i - 1]                                 # (H', Q') state after position i-1
            for r in range(n):
                ed = np.nonzero(dst == q[r])[0]                              # edges entering q
                if i == 0:
                    sc = self.pi[h[r]] * (src[ed] == self._q0) * self.Ec[0][h[r], sym[ed]]
                    k = rng.choice(len(ed), p=sc / sc.sum())
                    y[r, 0] = sym[ed[k]]
                else:
                    # joint over (previous hidden state h', edge): alpha[i-1][h', src] T[h', h] Ec[i][h, sym]
                    sc = pred_all[:, src[ed]] * self.T[i - 1][:, h[r]][:, None] * self.Ec[i][h[r], sym[ed]][None]
                    k = rng.choice(sc.size, p=(sc / sc.sum()).reshape(-1))
                    hp, ke = k // len(ed), k % len(ed)
                    y[r, i] = sym[ed[ke]]
                    h[r], q[r] = hp, src[ed[ke]]
        return y


class GrammarTruth:
    """Exact p(. | C) over class sequences for an OracleHMM and a context-free constraint in Chomsky normal form:
    `grammar` = (start, binary rules (X, Y, Z), lexical rules (X, class), A) as in compile.grammars, the constraint
    of torch_routes.TorchChart. Attributes: `log_pC` = log p(C); `marginals` (N, A) = p(class at i | C).

    Inside pass over the product of the grammar and the hidden chain. I[i, j, X][a, b] is the probability that
    positions i..j-1 emit a class string derived from X while the chain goes from h_i = a to h_j = b:
        I[i, i+1, X][a, b] = sum_{y : X -> y} emit_i[a, y] T_i[a, b]
        I[i, j, X]         = sum_{X -> Y Z} sum_k I[i, k, Y] @ I[k, j, Z]
        p(C)               = sum_{a, b} pi[a] I[0, N, start][a, b]            (T_{N-1} := identity)
    Cost O(N^3 |R| H^3), float64, emissions rescaled per position (every derivation covers each position once, so
    the scale factors out exactly). The marginals are the gradient of log p(C) with respect to per-position class
    log-weights, by autograd; its graph holds every span, about 8 N^3 |R| H^2 / 6 bytes (0.2 GB at N = 64 for
    Dyck-2 on a 4-state chain, 12 GB at N = 256), so pass with_marginals=False for long sequences.

    The grammar must be unambiguous. Otherwise each string counts once per parse and `log_pC` is not a probability.
    That is a property of the grammar, checked for the grammars used here in tests/test_grammar_truth.py."""

    def __init__(self, den, classes, grammar, with_marginals=True):
        start, binary, lexical, A = grammar[0], grammar[1], grammar[2], grammar[3]
        N, H = den.N, den.H
        if classes.A != A:
            raise ValueError("the classes and the grammar must share an alphabet")
        f8 = torch.float64
        onehot = classes.onehot[: den.S].double().cpu()
        Ec = den.emit.cpu().double() @ onehot                                           # (N, H, A) class emissions
        T = torch.cat([den.trans.cpu().double(), torch.eye(H, dtype=f8)[None]])        # (N, H, H)
        pi = den.pi.cpu().double()
        nts = sorted({x for r in binary for x in r} | {x for x, _ in lexical} | {start})
        ix = {x: k for k, x in enumerate(nts)}
        rA, rB, rC = (torch.tensor([ix[r[k]] for r in binary], dtype=torch.long) for k in range(3))
        lex = torch.zeros(len(nts), A, dtype=f8)
        for x, y in lexical:
            lex[ix[x], y] = 1.0
        theta = torch.zeros(N, A, dtype=f8, requires_grad=with_marginals)
        scale = 1.0 / Ec.mean((1, 2))                                                   # (N,)
        with torch.set_grad_enabled(with_marginals):
            leaf = torch.einsum("iha,xa->ixh", Ec * torch.exp(theta)[:, None, :] * scale[:, None, None], lex)
            chart = torch.zeros(N, N + 1, len(nts), H, H, dtype=f8)
            ar = torch.arange(N)
            chart[ar, ar + 1] = leaf[:, :, :, None] * T[:, None, :, :]
            for span in range(2, N + 1):
                n = N - span + 1
                i = torch.arange(n)[:, None]; m = torch.arange(1, span)[None]
                left = chart[i, i + m][:, :, rB]                                        # (n, span - 1, R, H, H)
                right = chart[i + m, i + span][:, :, rC]
                vals = torch.einsum("nmrab,nmrbc->nrac", left, right)
                out = torch.zeros(n, len(nts), H, H, dtype=f8).index_add(1, rA, vals)
                chart[torch.arange(n), torch.arange(n) + span] = out
            tot = (pi[:, None] * chart[0, N, ix[start]]).sum()
            if not bool(tot > 0):
                raise ValueError("the constraint has probability zero under the model")
            logtot = torch.log(tot)
        self.log_pC = float(logtot.detach() - torch.log(scale).sum())
        self.N, self.H, self.A = N, H, A
        self.marginals = None
        if with_marginals:
            g, = torch.autograd.grad(logtot, theta)
            self.marginals = g.clamp_min(0.0).numpy()
            self.marginals /= self.marginals.sum(-1, keepdims=True)


class CopyTruth:
    """Exact p(. | C) for an OracleHMM and the same-order copy constraint class(x_i) = class(x_{n + i}),
    n = N / 2, for every i < n or for the i listed in `pairs` (the constraint of torch_routes.TorchCopy with
    those pairs; the other positions are free). Attributes: `log_pC`, and `marginals` (N, A) =
    p(class at i | C); constrained positions i and n + i have the same row.

    The hidden chain runs through positions 0..N-1, so the two halves form a pair chain (h_i, h_{n+i}) joined by
    one extra transition from h_{n-1} to h_n. Conditioning on h_n makes the pair chain a plain forward-backward
    over H^2 states; summing over the H values of h_n gives the result. Cost O(n H^5)."""

    def __init__(self, den, classes, pairs=None):
        N, H, A = den.N, den.H, classes.A
        if N % 2:
            raise ValueError("N must be even")
        n = N // 2
        tied = np.ones(n, dtype=bool)
        if pairs is not None:
            tied[:] = False; tied[np.asarray(list(pairs), dtype=np.int64)] = True
        onehot = classes.onehot[: den.S].double().cpu().numpy()
        Ec = den.emit.cpu().numpy() @ onehot                                # (N, H, A)
        T, pi = den.trans.cpu().numpy(), den.pi.cpu().numpy()
        g = np.einsum("iac,ibc->iabc", Ec[:n], Ec[n:])                      # [i, a, b, c]: both emit class c
        one = np.ones((n, H, H))
        gs = np.where(tied[:, None, None], g.sum(-1), one)                  # [i, a, b]; a free pair has factor 1
        g1 = np.where(tied[:, None, None, None], g, Ec[:n, :, None, :] * one[..., None])   # class c at position i
        g2 = np.where(tied[:, None, None, None], g, Ec[n:, None, :, :] * one[..., None])   # class c at position n + i
        total, m1, m2 = 0.0, np.zeros((n, A)), np.zeros((n, A))
        for b0 in range(H):                                                 # condition on h_n = b0
            F = np.zeros((n, H, H)); F[0, :, b0] = pi                       # before the factor at i
            for i in range(1, n):
                F[i] = np.einsum("ab,ax,by->xy", F[i - 1] * gs[i - 1], T[i - 1], T[n + i - 1])
            B = np.zeros((n, H, H)); B[n - 1] = T[n - 1][:, b0][:, None]    # after the factor at i: link h_{n-1} -> h_n
            for i in range(n - 2, -1, -1):
                B[i] = np.einsum("xy,ax,by->ab", gs[i + 1] * B[i + 1], T[i], T[n + i])
            total += (F[n - 1] * gs[n - 1] * B[n - 1]).sum()
            m1 += np.einsum("iab,iabc,iab->ic", F, g1, B)
            m2 += np.einsum("iab,iabc,iab->ic", F, g2, B)
        if total <= 0:
            raise ValueError("the constraint has probability zero under the model")
        self.log_pC = float(np.log(total))
        self.marginals = np.concatenate([m1 / m1.sum(-1, keepdims=True), m2 / m2.sum(-1, keepdims=True)], 0)
        self.N, self.A = N, A
