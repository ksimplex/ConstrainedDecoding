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
import warnings
import torch

from ..torch_routes.routes import onehot_log
from .resampling import RESAMPLERS, _lmexp, defensive_mixture, offspring_logweights, resample_indices
from .schedule import NEG_INF, SCHEDULES, clamp_revealed, reveal_indices, reveal_sets
from .twist import (SUPPORT_CHECKS, _Particles, _lookahead_log_gain, _log_potential, _propose,
                    _successor_twist, tempered_class_logw)

__all__ = ["twisted_smc", "select_particles", "reveal_sets", "reveal_indices", "clamp_revealed",
           "resample_indices", "defensive_mixture", "offspring_logweights", "tempered_class_logw",
           "SCHEDULES", "RESAMPLERS", "SUPPORT_CHECKS", "NEG_INF"]


# ----------------------------------------------------------------------------------------------- driver
@torch.no_grad()
def twisted_smc(den, classes, route, N, T, nrun, K, gen=None, ess_min=0.5, return_particles=False,
                resampler="multinomial", res_gen=None, return_trace=False, defensive=0.0, twist_power=1.0,
                lookahead=0, lookahead_draws=1, lookahead_mean="arith", lookahead_shrink=1.0, mix_native=0.0,
                twist_temp=1.0, guard="warn", schedule="fixed"):
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
    lookahead_mean    "arith" (unbiased gain) or "geom" (mean of log gains over the draws)
    lookahead_shrink  lambda in [0, 1]: log psi = beta * (log eta + lambda * log gain)
    mix_native   a in [0, 1): propose each step from (1 - a) M + a kappa, mixing in the native kernel; the
                 potential then uses the mixture density and is bounded by psi' / (a psi).
    twist_temp   tau in (0, 1]: calibrated twist, the route sum under future-token law q ∝ p^tau (1 = TWISTER);
                 the proposal is locally optimal for it and the weights correct exactly (see _propose)
    schedule     "fixed" (N / T positions per step, the default and TWISTER's) or "independent" (each masked
                 position reveals with probability 1 / (T - t), the MDLM reverse step; see reveal_indices).
                 The native kernel must use the same schedule for the result to target that process.
    guard        "warn" (default), "raise" or "off": support invariants checked at every step (see
                 SUPPORT_CHECKS); counts are returned under 'guard'. They consume no randomness.
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
    tau = float(twist_temp)
    if not 0.0 < tau <= 1.0:
        raise ValueError("twist_temp must lie in (0, 1]")
    if tau != 1.0 and (lookahead > 0 or mix_native > 0):
        raise ValueError("twist_temp != 1 is not combined with lookahead or mix_native")
    if lookahead < 0 or lookahead_draws < 1:
        raise ValueError("lookahead must be >= 0 and lookahead_draws >= 1")
    if lookahead_mean not in ("arith", "geom") or not 0.0 <= lookahead_shrink <= 1.0:
        raise ValueError("lookahead_mean must be 'arith' or 'geom' and lookahead_shrink in [0, 1]")
    def la(xx, yy, lpp, lww, ee, tt):
        g = _lookahead_log_gain(den, classes, route, xx, yy, lpp, lww, ee, tt, T, schedule, lookahead,
                                lookahead_draws, gen, lookahead_mean)
        return torch.where(torch.isneginf(g), g, lookahead_shrink * g)
    general = lookahead > 0 or mix_native > 0 or tau != 1.0           # otherwise the TWISTER / tempered update below
    dev = classes.onehot.device
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    P, beta = nrun * K, float(twist_power)
    run_offset = K * torch.arange(nrun, device=dev)[:, None]

    x = torch.full((P, N), den.mask_id, dtype=torch.long, device=dev)
    ycls = torch.zeros(P, N, dtype=torch.long, device=dev)                  # classes at revealed positions
    lp = den.logits(x)
    lw = classes.class_logw(lp)
    eta0 = route.logZ(clamp_revealed(tempered_class_logw(lw, tau), x, ycls, den.mask_id))
    logw = beta * eta0                                                      # log psi_0(x_0)
    psi0 = None
    if general:
        psi0 = logw
        if lookahead > 0:
            psi0 = beta * (eta0 + la(x, ycls, lp, lw, eta0, -1))
            logw = psi0
    s = _Particles(x, ycls, lp, lw, eta0.clone(), None if psi0 is None else psi0.clone())
    logZ = torch.zeros(nrun, dtype=torch.float64, device=dev)
    ess_track = torch.full((nrun,), float(K), device=dev, dtype=torch.float64)
    n_res = torch.zeros(nrun, device=dev)
    trace = []
    if guard not in ("warn", "raise", "off"):
        raise ValueError("guard must be 'warn', 'raise' or 'off'")
    checks = {k: 0 for k in SUPPORT_CHECKS}

    for t in range(T):
        last = t == T - 1
        x2, y2, z_old, log_ratio = _propose(den, classes, route, s, t, T, schedule, gen, mix_native, tau)
        lp2, lw2, eta2, psi2 = _successor_twist(den, classes, route, x2, y2, z_old, last, beta, tau)
        if guard != "off":
            live = torch.isfinite(logw) & torch.isfinite(s.eta)
            on = torch.isfinite(z_old)
            if mix_native <= 0:                     # native-mixture proposals may leave the support by design
                checks["proposal_off_support"] += int((live & ~on).sum())
            if not last:
                checks["twist_collapse"] += int((live & on & ~torch.isfinite(psi2)).sum())
        if not general:
            logw = logw + _log_potential(s.eta, psi2, z_old, beta)
        else:
            if lookahead > 0 and not last:
                psi2 = beta * (eta2 + la(x2, y2, lp2, lw2, eta2, t))
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
        if guard != "off":
            checks["nan_weight"] += int(torch.isnan(logw).sum())
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
    out["guard"] = checks
    bad = {k: v for k, v in checks.items() if v}
    if bad and guard != "off":
        msg = "support invariants violated: " + "; ".join(f"{k} x{v} ({SUPPORT_CHECKS[k]})" for k, v in bad.items())
        if guard == "raise":
            raise FloatingPointError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    if return_trace:
        out["trace"] = torch.stack(trace).view(T, nrun, K).numpy()
    if return_particles:
        out["classes"] = final.view(nrun, K, N).cpu().numpy()
        out["logw"] = lwr.cpu().numpy()
    return out


def select_particles(out, rng):
    """One output per run, drawn in proportion to the run's final weights (TWISTER returns the weighted particle
    set; one draw per run samples from the SMC's terminal estimate). A run whose particles all have weight zero
    has no output: its row is drawn uniformly and reported in out["dead_runs"] (a count), so callers can drop it.
    With an exact route this does not happen; the support guard reports the cause when it does."""
    import numpy as np
    lw = out["logw"]
    dead = ~np.isfinite(lw).any(1)
    mx = np.where(dead[:, None], 0.0, lw.max(1, keepdims=True))
    w = np.where(np.isfinite(lw), np.exp(lw - mx), 0.0)
    w[dead] = 1.0
    w /= w.sum(1, keepdims=True)
    out["dead_runs"] = int(dead.sum())
    idx = np.array([rng.choice(w.shape[1], p=wi) for wi in w])
    return out["classes"][np.arange(len(idx)), idx]
