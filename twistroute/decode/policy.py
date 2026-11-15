"""
Constraint-aware parallel reveal policy: a scheduler s(r | x) that reveals, at each step, as many positions as
it can while keeping every co-revealed pair weakly coupled (docs/parallel_unmasking_plan.md, Sec. 3).

Why this is a legal scheduler. The reveal set is a function of the particle's own state (revealed tokens and
the current predictions), of a kernel frozen before the run, and of independent randomness; it never looks
at the values drawn in the same step. The native kernel therefore keeps the form s(r | x) prod_{i in r} p_i,
the scheduler cancels in the potential, and twisted SMC is exact for the native law this scheduler defines
(notes, Thm. "route-generic trajectory exactness"). What changes with the schedule is WHICH native law that is.

Coupling between two masked positions a, b has two parts:
    M(a, b)                          direct model coupling, from a kernel fitted in a pilot (`DistanceKernel`);
    sum_j M(a, j) rho(j, b)          a revision of the model's prediction at a still-masked j, passed on to b by
                                     the constraint; rho is the normalised mutual information between the
                                     classes at j and b under the constrained product law q_w(. | C).
Direct constraint coupling rho(a, b) is not penalised: positions revealed in one step are drawn jointly by the
route.

Forced positions (class determined by the constraint, single token) are considered first, because their value
needs nothing from the model. They are NOT exempt from the coupling test. The native kernel draws a forced
position from the model's marginal like any other, so revealing it together with a position the model couples
it to drops the factor p(x_a | x_forced) / p(x_a): the model never gets to condition a on the forced value.
(An earlier version revealed forced positions unconditionally; tests/test_policy.py caught the resulting gap
on a Markov chain, where several adjacent positions become forced together once a count saturates.)

The constrained marginals are exact: they are the gradient of route.logZ with respect to the class log-weights
(one backward pass). Pairwise terms come from the same pass with one position clamped, N * A clamps per
particle. This needs a route whose logZ is differentiable (TorchChain and TorchCopy are; TorchChart is not as
it stands).
"""
import math
import torch

from .schedule import NEG_INF, clamp_revealed

PRIORITIES = ("bound", "confident", "random")


# ---------------------------------------------------------------------------------------------- route queries
def constrained_marginals(route, lwc):
    """(B, N, A) clamped class log-weights -> (B, N, A) float64 marginals of q_w(. | C). Rows whose constraint
    mass is zero come back as zeros."""
    with torch.enable_grad():
        v = lwc.detach().double().clone().requires_grad_(True)
        g, = torch.autograd.grad(route.logZ(v).sum(), v)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)    # round-off can leave -1e-16


def pairwise_mi(route, lwc, mu, max_rows=40000):
    """(P, N, N) mutual information (nats) between the classes at every pair of positions under q_w(. | C).
    MI(j; b) = sum_c mu_j(c) KL(q(y_b | y_j = c, C) || q(y_b | C)), with the conditionals from clamped passes."""
    P, N, A = lwc.shape
    dev = lwc.device
    lw = lwc.double()
    logmu = torch.log(mu.clamp_min(1e-300))
    pos = torch.arange(N, device=dev)
    clampjc = torch.full((N, A, A), NEG_INF, dtype=torch.float64, device=dev)       # [j, c, :] one-hot log at c
    clampjc[:, torch.arange(A), torch.arange(A)] = 0.0
    out = torch.zeros(P, N, N, dtype=torch.float64, device=dev)
    step = max(1, max_rows // (N * A))
    for lo in range(0, P, step):
        base = lw[lo:lo + step]
        p = base.shape[0]
        rep = base[:, None, None].expand(p, N, A, N, A).clone()                    # [particle, j, c, position, class]
        rep[:, pos, :, pos] = clampjc[:, None]                                     # clamp position j to class c
        cond = constrained_marginals(route, rep.view(p * N * A, N, A)).view(p, N, A, N, A)
        kl = (torch.xlogy(cond, cond) - cond * logmu[lo:lo + step][:, None, None]).sum(-1)   # (p, j, c, b)
        out[lo:lo + step] = (mu[lo:lo + step][..., None] * kl).sum(2)
    out = 0.5 * (out + out.transpose(1, 2)).clamp_min(0.0)
    out[:, pos, pos] = 0.0
    return out


# ---------------------------------------------------------------------------------------------- model kernel
class DistanceKernel:
    """Model coupling pooled by distance: M(a, b) = unscreened[|a - b|], or screened[|a - b|] when the particle
    has a revealed position strictly between a and b. Both arrays have length N (index 0 unused)."""

    def __init__(self, unscreened, screened=None):
        self.ku = torch.as_tensor(unscreened, dtype=torch.float64)
        self.ks = self.ku.clone() if screened is None else torch.as_tensor(screened, dtype=torch.float64)

    def matrix(self, masked):
        P, N = masked.shape
        dev = masked.device
        rev = (~masked).long()
        upto = rev.cumsum(-1)                                              # revealed among positions <= i
        before = upto - rev                                                # revealed among positions < i
        between = before[:, None, :] - upto[:, :, None]                    # [p, i, j]: revealed strictly inside (i, j), i < j
        i = torch.arange(N, device=dev)
        d = (i[None] - i[:, None]).abs()
        scr = torch.triu(between > 0, 1)
        scr = scr | scr.transpose(1, 2)
        M = torch.where(scr, self.ks.to(dev)[d][None], self.ku.to(dev)[d][None]).clone()
        M[:, i, i] = 0.0
        return M

    def counts(self, masked):
        """Per-particle revealed counts used by `row`: (revealed at positions <= i, revealed at positions < i)."""
        rev = (~masked).long()
        upto = rev.cumsum(-1)
        return upto, upto - rev

    def row(self, counts, c):
        """(P, N): row c[p] of matrix(masked)[p] for every particle, from `counts(masked)`, without forming
        the (P, N, N) matrix. Memory O(P N), so long sequences do not need N^2 storage per particle."""
        upto, before = counts
        P, N = upto.shape
        dev = upto.device
        j = torch.arange(N, device=dev)[None]
        cc = c[:, None]
        d = (j - cc).abs()
        lo_upto = torch.where(j > cc, upto.gather(1, cc).expand(P, N), upto)           # revealed at <= min(c, j)
        hi_before = torch.where(j > cc, before, before.gather(1, cc).expand(P, N))     # revealed at <  max(c, j)
        scr = (hi_before - lo_upto) > 0
        r = torch.where(scr, self.ks.to(dev)[d], self.ku.to(dev)[d])
        return torch.where(d == 0, torch.zeros_like(r), r)


@torch.no_grad()
def fit_distance_kernel(den, N, n_traces, gen=None, device="cpu"):
    """Pilot: `n_traces` unconstrained traces of length N, revealing one uniformly chosen position per denoiser
    call. After each reveal of position a, the KL divergence between the new and the old prediction at every
    other masked position j is recorded; its mean over the token drawn at a is the conditional mutual
    information between a and j given the context. Returns a DistanceKernel of the per-distance means, split by
    whether a revealed position lay strictly between a and j, and the per-bin counts (2, N): row 0 unscreened,
    row 1 screened. Costs N + 1 denoiser calls of batch `n_traces`."""
    rows, idx = torch.arange(n_traces, device=device), torch.arange(N, device=device)
    x = torch.full((n_traces, N), den.mask_id, dtype=torch.long, device=device)
    lp = den.logits(x).double()
    sums = torch.zeros(2, N, dtype=torch.float64, device=device)
    cnts = torch.zeros(2, N, dtype=torch.float64, device=device)
    for _ in range(N):
        masked = x == den.mask_id
        u = torch.rand(masked.shape, generator=gen, device=device)
        a = torch.where(masked, u, torch.full_like(u, 2.0)).argmin(-1)                  # one masked position per row
        tok = torch.multinomial(torch.exp(lp[rows, a]).float(), 1, generator=gen).squeeze(-1)
        x2 = x.clone(); x2[rows, a] = tok
        lp2 = den.logits(x2).double()
        fin = torch.isfinite(lp2) & torch.isfinite(lp)
        kl = torch.where(fin, torch.exp(lp2) * (lp2 - lp), torch.zeros_like(lp2)).sum(-1)    # (B, N)
        rev = (~masked).long(); upto = rev.cumsum(-1); before = upto - rev
        lo, hi = torch.minimum(a[:, None], idx[None]), torch.maximum(a[:, None], idx[None])
        between = torch.gather(before, 1, hi) - torch.gather(upto, 1, lo)                  # revealed strictly inside
        still = x2 == den.mask_id
        d = (idx[None] - a[:, None]).abs()
        for k, sel in ((0, still & (between <= 0)), (1, still & (between > 0))):
            sums[k].index_add_(0, d[sel], kl[sel])
            cnts[k].index_add_(0, d[sel], torch.ones_like(kl[sel]))
        x, lp = x2, lp2
    mean = torch.where(cnts > 0, sums / cnts.clamp_min(1), torch.zeros_like(sums)).clamp_min(0.0)
    return DistanceKernel(mean[0].cpu(), mean[1].cpu()), cnts.cpu()


# ---------------------------------------------------------------------------------------------- the policy
class RevealState:
    """What a policy may look at: the particle states and their current predictions. Never the values that are
    about to be drawn."""
    __slots__ = ("x", "ycls", "lw", "mask_id", "classes", "route")

    def __init__(self, x, ycls, lw, mask_id, classes, route):
        self.x, self.ycls, self.lw, self.mask_id, self.classes, self.route = x, ycls, lw, mask_id, classes, route


class CouplingPolicy:
    """Greedy weakly-coupled reveal sets.

    kernel     model coupling, an object with .matrix(masked) -> (P, N, N) (e.g. DistanceKernel)
    theta      a position joins the reveal set if its summed coupling to the positions already chosen is <= theta
               (nats). theta = 0 admits only positions with exactly zero coupling to the set.
    priority   order in which positions are considered: "bound" (largest KL(mu_i || w_i): where the constraint
               distorts the model most), "confident" (lowest entropy of mu_i), or "random"
    mixed      include the term sum_j M(a, j) rho(j, b). False drops it (and its N * A clamped route passes).
    forced_first   consider positions whose class is determined (single token) before all others; they still
               pass through the coupling test

    One call returns padded reveal indices and a validity mask, the convention of schedule.reveal_indices.
    At the last step (t = T - 1) every remaining position is revealed, so T is a cap on the number of steps.
    `route_passes` counts batched logZ evaluations per particle requested by the policy (for cost accounting).
    """

    def __init__(self, kernel, theta, priority="bound", mixed=True, forced_first=True):
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        if theta < 0:
            raise ValueError("theta must be >= 0")
        self.kernel, self.theta, self.priority, self.mixed = kernel, float(theta), priority, mixed
        self.forced_first = forced_first
        self.route_passes = 0.0
        self.calls = 0

    def coupling(self, st, masked, mu, entropy):
        """Effective coupling E (P, N, N) among masked positions."""
        E = self.kernel.matrix(masked).to(mu.device)
        if self.mixed:
            lwc = clamp_revealed(st.lw, st.x, st.ycls, st.mask_id)
            mi = pairwise_mi(st.route, lwc, mu)
            hmin = torch.minimum(entropy[:, :, None], entropy[:, None, :])
            rho = torch.where(hmin > 1e-9, mi / hmin.clamp_min(1e-9), torch.zeros_like(mi)).clamp(0.0, 1.0)
            both = masked[:, :, None] & masked[:, None, :]
            rho = torch.where(both, rho, torch.zeros_like(rho))
            Mm = torch.where(both, E, torch.zeros_like(E))
            mix = Mm @ rho
            E = E + mix + mix.transpose(1, 2)
        return E

    @torch.no_grad()
    def reveal(self, st, t, T, gen=None):
        masked = st.x == st.mask_id
        P, N = masked.shape
        dev = masked.device
        if t >= T - 1:
            chosen = masked
        else:
            lwc = clamp_revealed(st.lw, st.x, st.ycls, st.mask_id)
            mu = constrained_marginals(st.route, lwc)
            self.route_passes += 1 + (N * mu.shape[-1] if self.mixed else 0)
            ent = -torch.xlogy(mu, mu).sum(-1)
            cand = masked
            lean = not self.mixed and hasattr(self.kernel, "row")     # kernel rows on demand: no (P, N, N) storage
            if lean:
                counts = self.kernel.counts(masked)
            else:
                E = self.coupling(st, masked, mu, ent)
                E = torch.where((cand[:, :, None] & cand[:, None, :]), E, torch.zeros_like(E))
            if self.priority == "bound":
                w = torch.exp(st.lw.double())
                score = (torch.xlogy(mu, mu) - mu * torch.log(w.clamp_min(1e-300))).sum(-1)
            elif self.priority == "confident":
                score = -ent
            else:
                score = torch.rand(masked.shape, generator=gen, device=dev).double()
            if self.forced_first:
                sizes = (st.classes.onehot > 0).sum(0).to(dev)
                forced = masked & (ent < 1e-9) & (sizes[mu.argmax(-1)] == 1)
                score = torch.where(forced, torch.full_like(score, float("inf")), score)
            score = torch.where(cand, score, torch.full_like(score, NEG_INF))
            order = torch.argsort(score, dim=-1, descending=True, stable=True)
            chosen = torch.zeros_like(masked)
            load = torch.zeros(P, N, dtype=torch.float64, device=dev)
            rows = torch.arange(P, device=dev)
            for k in range(N):
                c = order[:, k]
                ok = cand[rows, c] & (load[rows, c] <= self.theta)
                chosen[rows, c] |= ok
                if lean:                         # rows only for the particles that admitted their candidate
                    sel = ok.nonzero(as_tuple=True)[0]
                    if sel.numel():
                        r = self.kernel.row((counts[0][sel], counts[1][sel]), c[sel]).to(dev)
                        load[sel] += torch.where(cand[sel], r, torch.zeros_like(r))
                else:
                    load = load + torch.where(ok[:, None], E[rows, c], torch.zeros_like(load))
        self.calls += 1
        cnt = chosen.sum(-1)
        k = int(cnt.max().clamp_min(1))
        ordr = torch.argsort(chosen.to(torch.int8), dim=-1, descending=True, stable=True)
        return ordr[:, :k], torch.arange(k, device=dev)[None] < cnt[:, None]
