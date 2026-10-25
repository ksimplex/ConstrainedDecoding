"""
Route-generic twisted SMC for masked diffusion: TWISTER (Thimmaiah et al., arXiv:2609.35609, Alg. 1/2) with
the automaton replaced by any route that exposes logZ / sample / accepts (see twistroute_notes.tex, Sec. 6).

Notation. For a partial state x (masked positions hold mask_id), lw are the class log-probabilities of den(x)
with revealed positions clamped one-hot:
    Z(x'; x)   = route.logZ(lw of x, clamped at x')      clamped sum (notes Eq. 4; TWISTER Eq. 11)
    eta_t(x)   = Z(x; x)                                  frozen twist
    psi_t      = eta_t^beta for t < T, psi_T = 1_C       twist used for weighting and resampling
    M(x'|x)    = kappa(x'|x) Z(x'; x) / Z(x; x)          step-exact proposal (TWISTER Prop. 3.6): reveal set
                                                          from the native scheduler, y ~ route.sample, keep y_r
    G          = eta_t(x)^(1-beta) psi_{t+1}(x') / Z(x'; x)   potential (notes Eq. 7; beta = 1 is TWISTER)
The weights telescope, q(path) W(path) = p_native(path) 1_C(x_T), so E[Z_hat] = h0 = P_native(C) at every K
and the weighted particles target the native conditional. This holds for any route whose sampler draws from
the measure its logZ sums, including ambiguous grammars, because only the terminal twist must be exact.

Native kernel (TWISTER Sec. 4.3, random remasking): T steps; each reveals N/T positions chosen uniformly among
the masked ones; tokens at temperature 1, no truncation.

Resampling fires per run when ESS < ess_min * K (never at the last step). Ancestors are drawn from
pi = (1 - eps) w~ + eps / K and offspring get weight w~ / pi; eps = 0 is multinomial resampling, eps > 0 is
defensive resampling (notes Prop. 7.6: conditional relative variance at most 1 / eps). Z_hat multiplies the
mean weight at every resampling and at the end (Del Moral, Feynman-Kac Formulae, 2004).
"""
import math
import torch
from ..torch_routes.routes import onehot_log

NEG_INF = float("-inf")
RESAMPLERS = ("multinomial", "cpu", "systematic")


# ----------------------------------------------------------------------------------------------- primitives
def reveal_sets(masked, n_t, gen=None):
    """Uniform random subset of size n_t among masked positions, per row. Returns (B, n_t) indices."""
    u = torch.rand(masked.shape, device=masked.device, generator=gen)
    u = torch.where(masked, u, torch.full_like(u, 2.0))
    return torch.topk(-u, n_t, dim=-1).indices


def clamp_revealed(lw, x, ycls, mask_id):
    """Replace class log-weights at revealed positions of x by the one-hot of their class."""
    return torch.where((x != mask_id)[..., None], onehot_log(ycls, lw.shape[-1]), lw)


def _lmexp(x, dim):
    """log mean exp along dim."""
    return torch.logsumexp(x, dim) - math.log(x.shape[dim])


def resample_indices(probs, K, resampler="multinomial", gen=None, res_gen=None):
    """(nrun, K) ancestor indices; row r is drawn from the distribution probs[r].

    multinomial  torch.multinomial on probs' device with generator gen
    cpu          torch.multinomial on the CPU with its own generator res_gen
    systematic   one uniform per row (from res_gen, else gen); offspring counts are floor or ceil of K * probs
    """
    if resampler == "multinomial":
        return torch.multinomial(probs, K, replacement=True, generator=gen)
    if resampler == "cpu":
        return torch.multinomial(probs.double().cpu(), K, replacement=True, generator=res_gen).to(probs.device)
    if resampler == "systematic":
        g = res_gen if res_gen is not None else gen
        u = torch.rand(probs.shape[0], 1, generator=g, dtype=torch.float64,
                       device=g.device if g is not None else "cpu").to(probs.device)
        pts = (u + torch.arange(K, device=probs.device, dtype=torch.float64)[None]) / K
        cdf = torch.cumsum(probs.double(), -1)
        cdf = cdf / cdf[:, -1:]
        return torch.searchsorted(cdf, pts).clamp_max(K - 1)
    raise ValueError(f"resampler must be one of {RESAMPLERS}, got {resampler!r}")


_resample_idx = resample_indices                     # backwards-compatible name


def defensive_mixture(wn, defensive):
    """Ancestor distribution pi = (1 - eps) wn + eps / K for normalised weights wn (nrun, K)."""
    return (1.0 - defensive) * wn + defensive / wn.shape[-1]


def offspring_logweights(wn, pi, idx, defensive):
    """Log-weights log(wn_a / pi_a) of the offspring with ancestors idx; all zero when eps = 0."""
    if defensive > 0:
        return (torch.log(torch.gather(wn, 1, idx)) - torch.log(torch.gather(pi, 1, idx))).float()
    return torch.zeros_like(wn, dtype=torch.float32)


# ----------------------------------------------------------------------------------------------- SMC steps
class _Particles:
    """Per-particle state, one row per particle (nrun * K rows, run-major)."""

    def __init__(self, x, ycls, lp, lw, eta, psi=None):
        self.x, self.ycls, self.lp, self.lw, self.eta, self.psi = x, ycls, lp, lw, eta, psi

    def take(self, flat):
        self.x, self.ycls, self.lp, self.lw, self.eta = (self.x[flat], self.ycls[flat], self.lp[flat],
                                                         self.lw[flat], self.eta[flat])
        if self.psi is not None:
            self.psi = self.psi[flat]


def _propose(den, classes, route, s, nt, gen, mix_native=0.0):
    """One proposal step from state s: the step-exact proposal, or with probability mix_native per particle the
    native kernel (same reveal set, tokens from the unrestricted predictions). Returns (x2, y2, z_old, log_ratio)
    with z_old = Z(x2; x) and log_ratio = log(M(x2|x) / kappa(x2|x)), which is z_old - Z(x; x) for the
    step-exact proposal and log((1 - a) exp(z_old - Z(x; x)) + a) for the mixture (None when mix_native = 0)."""
    P = s.x.shape[0]
    lwc = clamp_revealed(s.lw, s.x, s.ycls, den.mask_id)
    y = route.sample(lwc, gen)                                              # y ~ q(. | C; den(x))
    r = reveal_sets(s.x == den.mask_id, nt, gen)                            # native scheduler
    cr = torch.gather(y, 1, r)                                              # classes revealed this step
    rows_all = torch.gather(s.lp, 1, r[..., None].expand(-1, -1, s.lp.shape[-1]))
    rows = torch.where(classes.onehot.T[cr] > 0, rows_all, torch.full_like(rows_all, NEG_INF))
    tok = torch.multinomial(torch.softmax(rows, -1).reshape(-1, rows.shape[-1]), 1,
                            generator=gen).reshape(P, nt)                   # token within class
    if mix_native > 0:
        tok_nat = torch.multinomial(torch.softmax(rows_all, -1).reshape(-1, rows_all.shape[-1]), 1,
                                    generator=gen).reshape(P, nt)           # native tokens at the same reveal set
        use = torch.rand(P, generator=gen, device=s.x.device) < mix_native
        tok = torch.where(use[:, None], tok_nat, tok)
        cr = torch.where(use[:, None], classes.token_class[tok_nat], cr)
    x2 = s.x.clone(); x2.scatter_(1, r, tok)
    y2 = s.ycls.clone(); y2.scatter_(1, r, cr)
    z_old = route.logZ(clamp_revealed(s.lw, x2, y2, den.mask_id))           # Z(x'; x)
    if mix_native <= 0:
        return x2, y2, z_old, None
    step = torch.nan_to_num(z_old - s.eta, nan=NEG_INF, neginf=NEG_INF)
    log_ratio = torch.logaddexp(math.log1p(-mix_native) + step, torch.full_like(step, math.log(mix_native)))
    return x2, y2, z_old, log_ratio


def _successor_twist(den, classes, route, x2, y2, z_old, last, beta):
    """New predictions at x2 and the twist there. Returns (lp, lw, eta2, psi2); lp, lw are None at the last step,
    where the twist is the exact indicator (0 or -inf in log space)."""
    if last:
        ok = route.accepts(classes.token_class[x2])
        eta2 = torch.where(ok, torch.zeros_like(z_old), torch.full_like(z_old, NEG_INF))
        return None, None, eta2, eta2
    lp = den.logits(x2)
    lw = classes.class_logw(lp)
    eta2 = route.logZ(clamp_revealed(lw, x2, y2, den.mask_id))              # Z(x'; x')
    return lp, lw, eta2, beta * eta2


def _log_potential(eta_cur, psi2, z_old, beta):
    """log G = (1 - beta) log eta_t(x) + log psi_{t+1}(x') - log Z(x'; x)."""
    if beta == 1.0:
        return psi2 - z_old
    return (1.0 - beta) * eta_cur + psi2 - z_old


def _lookahead_log_gain(den, classes, route, x2, y2, lp2, lw2, eta2, t, T, nt, horizon, m, gen):
    """log of the mean over m speculative step-exact rollouts of j = horizon steps from x2 (which is x_{t+1}) of
    the TWISTER potential product prod G = prod eta(x'') / Z(x''; x'). eta(x2) times this gain is an unbiased,
    positive estimate of E_kappa[eta_{t+1+j}(X) | x2]; when the rollout reaches step T the final twist is the
    indicator and the estimate is unbiased for the Doob function h_{t+1}(x2). Returns (P,) log gains."""
    P = x2.shape[0]
    rep = lambda v: v.repeat_interleave(m, 0)
    sp = _Particles(rep(x2), rep(y2), rep(lp2), rep(lw2), rep(eta2))
    acc = torch.zeros(P * m, device=x2.device, dtype=eta2.dtype)
    for k in range(horizon):
        tt = t + 1 + k
        if tt > T - 1:
            break
        last = tt == T - 1
        x3, y3, z3, _ = _propose(den, classes, route, sp, nt, gen)
        lp3, lw3, eta3, _ = _successor_twist(den, classes, route, x3, y3, z3, last, 1.0)
        acc = acc + torch.nan_to_num(eta3 - z3, nan=NEG_INF, neginf=NEG_INF)
        sp.x, sp.ycls, sp.eta = x3, y3, eta3
        if last:
            break
        sp.lp, sp.lw = lp3, lw3
    return _lmexp(acc.view(P, m), -1)


# ----------------------------------------------------------------------------------------------- driver
@torch.no_grad()
def twisted_smc(den, classes, route, N, T, nrun, K, gen=None, ess_min=0.5, return_particles=False,
                resampler="multinomial", res_gen=None, return_trace=False, defensive=0.0, twist_power=1.0,
                lookahead=0, lookahead_draws=1, mix_native=0.0):
    """Run `nrun` independent SMC runs of K particles each, batched as nrun * K rows.

    den          denoiser with .mask_id and .logits(x) -> (B, N, V) log-probs (models.mdlm_owt.MDLM interface)
    classes      models.classes.TokenClasses on the same device as den
    route        torch_routes route over classes.A symbols
    ess_min      resample a run when its ESS < ess_min * K (0 disables resampling)
    resampler    "multinomial", "cpu" or "systematic" (see resample_indices); all unbiased
    defensive    eps in [0, 1): defensive mixing of the ancestor distribution (0 = multinomial resampling)
    twist_power  beta in (0, 1]: intermediate twists eta_t^beta (1 = TWISTER)
    lookahead    horizon j >= 0 of the look-ahead twist psi = (eta * gain)^beta, where gain is the mean potential
                 product of `lookahead_draws` speculative rollouts of j steps (j >= T covers the whole remaining
                 trajectory, giving an unbiased estimate of the Doob function). Costs j * lookahead_draws extra
                 denoiser calls per particle per step. 0 = frozen twist.
    mix_native   a in [0, 1): propose each step from (1 - a) M + a kappa, mixing in the native kernel; the
                 potential then uses the mixture density and is bounded by psi' / (a psi).
    Returns a dict with per-run 'logZ' (log Z_hat), 'ess_min', 'n_res', validity checks, 'eta0'; with
    return_trace the per-step log-weights before resampling, shape (T, nrun, K); with return_particles the
    final class sequences (nrun, K, N) and log-weights (nrun, K). K = 1 without the weights is step-exact
    decoding (Dang & Ermon, arXiv:2607.07026).
    """
    if not 0.0 <= defensive < 1.0:
        raise ValueError("defensive must lie in [0, 1)")
    if not 0.0 < twist_power <= 1.0:
        raise ValueError("twist_power must lie in (0, 1]")
    if not 0.0 <= mix_native < 1.0:
        raise ValueError("mix_native must lie in [0, 1)")
    if lookahead < 0 or lookahead_draws < 1:
        raise ValueError("lookahead must be >= 0 and lookahead_draws >= 1")
    general = lookahead > 0 or mix_native > 0           # otherwise the TWISTER / tempered update below
    dev = classes.onehot.device
    P, nt, beta = nrun * K, N // T, float(twist_power)
    run_offset = K * torch.arange(nrun, device=dev)[:, None]

    x = torch.full((P, N), den.mask_id, dtype=torch.long, device=dev)
    ycls = torch.zeros(P, N, dtype=torch.long, device=dev)                  # classes at revealed positions
    lp = den.logits(x)
    lw = classes.class_logw(lp)
    eta0 = route.logZ(clamp_revealed(lw, x, ycls, den.mask_id))
    logw = beta * eta0                                                      # log psi_0(x_0)
    psi0 = None
    if general:
        psi0 = logw
        if lookahead > 0:
            psi0 = beta * (eta0 + _lookahead_log_gain(den, classes, route, x, ycls, lp, lw, eta0, -1, T, nt,
                                                      lookahead, lookahead_draws, gen))
            logw = psi0
    s = _Particles(x, ycls, lp, lw, eta0.clone(), None if psi0 is None else psi0.clone())
    logZ = torch.zeros(nrun, dtype=torch.float64, device=dev)
    ess_track = torch.full((nrun,), float(K), device=dev, dtype=torch.float64)
    n_res = torch.zeros(nrun, device=dev)
    trace = []

    for t in range(T):
        last = t == T - 1
        x2, y2, z_old, log_ratio = _propose(den, classes, route, s, nt, gen, mix_native)
        lp2, lw2, eta2, psi2 = _successor_twist(den, classes, route, x2, y2, z_old, last, beta)
        if not general:
            logw = logw + _log_potential(s.eta, psi2, z_old, beta)
        else:
            if lookahead > 0 and not last:
                psi2 = beta * (eta2 + _lookahead_log_gain(den, classes, route, x2, y2, lp2, lw2, eta2, t, T, nt,
                                                          lookahead, lookahead_draws, gen))
            if log_ratio is None:
                log_ratio = z_old - s.eta
            # log G = log psi_{t+1}(x') - log psi_t(x) - log(M(x'|x) / kappa(x'|x))
            inc = psi2 - s.psi - log_ratio
            dead = torch.isneginf(s.psi) | torch.isnan(inc)                 # a dead particle stays dead
            logw = logw + torch.where(dead, torch.full_like(inc, NEG_INF), inc)
            s.psi = psi2
        s.x, s.ycls, s.eta = x2, y2, eta2
        if not last:
            s.lp, s.lw = lp2, lw2
        if return_trace:
            trace.append(logw.float().cpu())

        lwr = logw.view(nrun, K)
        wn = torch.nan_to_num(torch.softmax(lwr, -1), nan=1.0 / K)
        ess = 1.0 / (wn.double() ** 2).sum(-1)
        ess_track = torch.minimum(ess_track, torch.nan_to_num(ess, nan=0.0))
        if last or K == 1:
            continue
        fire = ess < ess_min * K
        if not fire.any():
            continue
        logZ = logZ + torch.where(fire, _lmexp(lwr, -1), torch.zeros_like(logZ))
        pi = defensive_mixture(wn, defensive)
        probs = torch.where(fire[:, None], pi, torch.full_like(wn, 1.0 / K)).float()
        idx = resample_indices(probs, K, resampler, gen, res_gen)
        idx = torch.where(fire[:, None], idx, torch.arange(K, device=dev)[None])
        s.take((idx + run_offset).reshape(-1))
        logw = torch.where(fire[:, None], offspring_logweights(wn, pi, idx, defensive), lwr).reshape(-1)
        n_res = n_res + fire.float()

    lwr = logw.view(nrun, K)
    logZ = logZ + _lmexp(lwr, -1)
    final = classes.token_class[s.x]
    valid = route.accepts(final).view(nrun, K)
    wfin = torch.isfinite(lwr)
    out = dict(logZ=logZ.cpu().numpy(), ess_min=ess_track.cpu().numpy(), n_res=n_res.cpu().numpy(),
               frac_valid_finite=float((valid & wfin).sum() / wfin.sum().clamp_min(1)),
               any_finite_invalid=bool((wfin & ~valid).any()), eta0=float(eta0[0]))
    if return_trace:
        out["trace"] = torch.stack(trace).view(T, nrun, K).numpy()
    if return_particles:
        out["classes"] = final.view(nrun, K, N).cpu().numpy()
        out["logw"] = lwr.cpu().numpy()
    return out


def select_particles(out, rng):
    """One output per run, drawn in proportion to the run's final weights (TWISTER returns the weighted particle
    set; one draw per run samples from the SMC's terminal estimate)."""
    import numpy as np
    lw = out["logw"]
    w = np.exp(lw - lw.max(1, keepdims=True)); w = np.nan_to_num(w); w /= w.sum(1, keepdims=True)
    idx = np.array([rng.choice(w.shape[1], p=wi) for wi in w])
    return out["classes"][np.arange(len(idx)), idx]
