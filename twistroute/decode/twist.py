"""
One twisted-SMC step: the proposal, the twist, and the potential.

For a partial state x (masked positions hold mask_id), write lw for the class log-probabilities of
den(x) with revealed positions clamped one-hot, and

    Z(x'; x) = route.logZ(lw of x, clamped at x')    clamped route sum (notes Eq. 4; TWISTER Eq. 11)
    eta_t(x) = Z(x; x)                               frozen twist
    psi_t    = eta_t^beta for t < T,  psi_T = 1_C    twist used for weighting and resampling
    M(x'|x)  = kappa(x'|x) Z(x'; x) / Z(x; x)        step-exact proposal (TWISTER Prop. 3.6)
    G        = eta_t(x)^(1-beta) psi_{t+1}(x') / Z(x'; x)                   potential (notes Eq. 7)

Three variations on the twist live here, all of which leave the estimator exact because only the
terminal twist must be:

  tempering (`twist_power` = beta in (0, 1]) flattens the intermediate twists, the standard device
  for keeping a particle filter from committing early (Del Moral, Feynman-Kac Formulae, 2004,
  doi:10.1007/978-1-4684-9393-1, Ch. 12);

  calibration (`twist_temp` = tau in (0, 1], `tempered_class_logw`) computes the route sum under the
  future-token law q proportional to p^tau while weighting under p. The proposal is locally optimal
  for q and the weights correct exactly. This is a twist-function choice in the sense of controlled
  SMC (Heng, Bishop, Deligiannidis & Doucet, Ann. Statist. 48(5):2904-2929, 2020,
  doi:10.1214/19-AOS1914), with tau as a one-parameter family rather than a learned twist;

  look-ahead (`lookahead` = j, `_lookahead_log_gain`) multiplies eta by the mean potential product
  of j-step speculative rollouts, an unbiased estimate of the Doob function when j covers the
  remaining trajectory. Measured to hurt: the rollout noise widens the spread of log(psi / h) more
  than the bias it removes (docs/twist_design_findings.md). Retained for the record, not a default.
"""
import math
import torch

from ..torch_routes.routes import onehot_log
from .resampling import log_mean_exp
from .policy import RevealState
from .schedule import NEG_INF, clamp_revealed, choose_reveals

# Support invariants. With a denoiser that gives every token positive probability, the set of
# completable states does not depend on the predictions, so a live particle must keep a finite
# clamped sum and twist. A violation means numerical support loss (underflow in a route pass, or
# exact zeros from the model), and the weights are then no longer those of the theorem.
SUPPORT_CHECKS = {
    "proposal_off_support": "live particle whose proposed state has Z(x'; x) = 0: the route sampler and its "
                            "mass pass disagree",
    "twist_collapse": "live particle with Z(x'; x) > 0 but twist(x') = 0 under the new predictions",
    "nan_weight": "NaN log-weight",
}

def tempered_class_logw(lw, tau):
    """Class log-weights of the future-token law q_i(c) ∝ p_i(c)^tau used inside the twist (tau = 1: p itself).
    Any positive product law keeps the route sum exact and the twist valid; tau < 1 flattens over-confident
    marginals so the twist cannot assign a near-zero value to completions the denoiser may later revise toward."""
    return lw if tau == 1.0 else torch.log_softmax(tau * lw, -1)




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


def _propose(den, classes, route, s, t, T, schedule, gen, mix_native=0.0, tau=1.0, chain=None, reference=None, chain_mix=0.0):
    """One proposal step from state s: the step-exact proposal, or with probability mix_native per particle the
    native kernel (same reveal set, tokens from the unrestricted predictions). Returns (x2, y2, z_old, log_ratio)
    with z_old = Z(x2; x) and log_ratio = log(M(x2|x) / kappa(x2|x)), which is z_old - Z(x; x) for the
    step-exact proposal and log((1 - a) exp(z_old - Z(x; x)) + a) for the mixture (None when mix_native = 0).

    tau != 1 (calibrated twist): the twist sums the route under q ∝ p^tau at masked positions. The proposal is the
    locally optimal one for that twist, M(x'|x) ∝ kappa(x'|x) Z_q(x'; x): y is drawn with the true class law p at
    the reveal set r and q elsewhere, so log(M / kappa) = Z_q(x'; x) - Z_mix(x; x) and z_old = Z_q(x'; x).

    chain (a decode.chain_twist.ChainTwist): the classes at the reveal set are drawn from the chain law given the
    constraint instead of the product law, M(x'|x) = P_chain(classes at r, C) / P_chain(C) times the model's law of
    the token within its class. Then log(M / kappa) = log P_chain(classes at r, C) - log P_chain(C) - the sum over r
    of the model's class log-probabilities. The reveal set is chosen without seeing the draw, as for the product.

    reference (a ChainTwist): kappa, the law of the revealed classes with no constraint, is the chain law's joint for
    a particle that reveals two or more positions in this step (the model's marginal for one position, as always).
    This changes what the parallel process is, and so what the weights converge to; it leaves one token per step
    untouched. The returned log_ratio is then log(M / kappa) for that kappa.

    chain_mix (d in [0, 1), with chain): each particle draws its classes from the product law given the constraint
    with probability d and from the chain law otherwise; the reveal set is chosen as before. The proposal is the
    mixture, M / kappa = (1 - d) M_chain / kappa + d Z(x'; x) / Z(x; x), so a class sequence the constraint and the
    model allow has at least the share d of its product-proposal density whatever the pair table says."""
    P = s.x.shape[0]
    view = lambda: RevealState(s.x, s.ycls, s.lw, den.mask_id, classes, route, lp=getattr(s, "lp", None))   # for policy schedules only
    log_chain_c = None
    if chain is not None:
        lwc = clamp_revealed(s.lw, s.x, s.ycls, den.mask_id)
        y, log_chain_c = chain.sample(lwc, gen)                             # y ~ chain law given C
        if chain_mix > 0:                                                   # defensive share of the product proposal
            from_product = torch.rand(P, generator=gen, device=s.x.device) < chain_mix
            y = torch.where(from_product[:, None], route.sample(lwc, gen), y)
        r, rvalid = choose_reveals(schedule, s.x == den.mask_id, t, T, gen, view)
    elif tau != 1.0:
        r, rvalid = choose_reveals(schedule, s.x == den.mask_id, t, T, gen, view)
        lwq = tempered_class_logw(s.lw, tau)
        inr = torch.zeros(s.x.shape, dtype=torch.bool, device=s.x.device)
        inr.scatter_(1, r, rvalid if rvalid is not None else torch.ones_like(r, dtype=torch.bool))
        lwc = clamp_revealed(torch.where(inr[..., None], s.lw, lwq), s.x, s.ycls, den.mask_id)
        y = route.sample(lwc, gen)
    else:
        lwc = clamp_revealed(s.lw, s.x, s.ycls, den.mask_id)
        y = route.sample(lwc, gen)                                          # y ~ q(. | C; den(x))
        r, rvalid = choose_reveals(schedule, s.x == den.mask_id, t, T, gen, view)  # native scheduler
    nt = r.shape[1]
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
    if rvalid is not None:                                                  # padded columns keep their value
        tok = torch.where(rvalid, tok, torch.gather(s.x, 1, r))
        cr = torch.where(rvalid, cr, torch.gather(s.ycls, 1, r))
    x2 = s.x.clone(); x2.scatter_(1, r, tok)
    y2 = s.ycls.clone(); y2.scatter_(1, r, cr)
    shift = None
    if reference is not None:                                              # log kappa_product - log kappa_reference
        if mix_native > 0 or tau != 1.0:
            raise ValueError("the chain reference is not combined with mix_native or twist_temp")
        keep = torch.ones_like(r, dtype=torch.bool) if rvalid is None else rvalid
        nat = torch.gather(s.lw, 1, r[..., None].expand(-1, -1, s.lw.shape[-1])).gather(2, cr[..., None]).squeeze(-1).double()
        nat = torch.where(keep, nat, torch.zeros_like(nat)).sum(-1)
        shift = nat - reference.log_reference(clamp_revealed(s.lw, s.x, s.ycls, den.mask_id), s.lw, r, cr, rvalid)
    if tau != 1.0:
        z_old = route.logZ(clamp_revealed(lwq, x2, y2, den.mask_id))        # Z_q(x'; x)
        return x2, y2, z_old, z_old - route.logZ(lwc)
    z_old = route.logZ(clamp_revealed(s.lw, x2, y2, den.mask_id))           # Z(x'; x)
    if chain is not None:
        keep = torch.ones_like(r, dtype=torch.bool) if rvalid is None else rvalid
        native = torch.gather(s.lw, 1, r[..., None].expand(-1, -1, s.lw.shape[-1])).gather(2, cr[..., None]).squeeze(-1).double()
        native = torch.where(keep, native, torch.zeros_like(native)).sum(-1)   # the model's class log-probabilities at r
        step = torch.nan_to_num(chain.log_clamped(lwc, r, cr, rvalid) - log_chain_c - native, nan=NEG_INF, neginf=NEG_INF)
        if chain_mix > 0:
            prod = torch.nan_to_num(z_old.double() - s.eta.double(), nan=NEG_INF, neginf=NEG_INF)
            step = torch.logaddexp(math.log1p(-chain_mix) + step, math.log(chain_mix) + prod)
        if mix_native <= 0:
            return x2, y2, z_old, step if shift is None else step + shift
    elif mix_native <= 0:
        if shift is None:
            return x2, y2, z_old, None
        return x2, y2, z_old, torch.nan_to_num(z_old.double() - s.eta.double(), nan=NEG_INF, neginf=NEG_INF) + shift
    else:
        step = torch.nan_to_num(z_old - s.eta, nan=NEG_INF, neginf=NEG_INF)
    log_ratio = torch.logaddexp(math.log1p(-mix_native) + step, torch.full_like(step, math.log(mix_native)))
    return x2, y2, z_old, log_ratio


def _successor_twist(den, classes, route, x2, y2, z_old, last, beta, tau=1.0):
    """New predictions at x2 and the twist there. Returns (lp, lw, eta2, psi2); lp, lw are None at the last step,
    where the twist is the exact indicator (0 or -inf in log space)."""
    if last:
        ok = route.accepts(classes.token_class[x2])
        eta2 = torch.where(ok, torch.zeros_like(z_old), torch.full_like(z_old, NEG_INF))
        return None, None, eta2, eta2
    lp = den.logits(x2)
    lw = classes.class_logw(lp)
    eta2 = route.logZ(clamp_revealed(tempered_class_logw(lw, tau), x2, y2, den.mask_id))  # Z(x'; x')
    return lp, lw, eta2, beta * eta2


def _log_potential(eta_cur, psi2, z_old, beta):
    """log G = (1 - beta) log eta_t(x) + log psi_{t+1}(x') - log Z(x'; x)."""
    if beta == 1.0:
        return psi2 - z_old
    return (1.0 - beta) * eta_cur + psi2 - z_old


def _lookahead_log_gain(den, classes, route, x2, y2, lp2, lw2, eta2, t, T, schedule, horizon, m, gen, mean="arith"):
    """log of the mean over m speculative step-exact rollouts of j = horizon steps from x2 (which is x_{t+1}) of
    the TWISTER potential product prod G = prod eta(x'') / Z(x''; x'). With mean="arith", eta(x2) times this gain
    is an unbiased, positive estimate of E_kappa[eta_{t+1+j}(X) | x2] (of the Doob function h_{t+1}(x2) when
    the rollout reaches step T). mean="geom" averages log gains instead: biased low but far less dispersed,
    which is what resampling needs (twist_design_findings.md). Returns (P,) log gains."""
    P = x2.shape[0]
    rep = lambda v: v.repeat_interleave(m, 0)
    sp = _Particles(rep(x2), rep(y2), rep(lp2), rep(lw2), rep(eta2))
    acc = torch.zeros(P * m, device=x2.device, dtype=eta2.dtype)
    for k in range(horizon):
        tt = t + 1 + k
        if tt > T - 1:
            break
        last = tt == T - 1
        x3, y3, z3, _ = _propose(den, classes, route, sp, tt, T, schedule, gen)
        lp3, lw3, eta3, _ = _successor_twist(den, classes, route, x3, y3, z3, last, 1.0)
        acc = acc + torch.nan_to_num(eta3 - z3, nan=NEG_INF, neginf=NEG_INF)
        sp.x, sp.ycls, sp.eta = x3, y3, eta3
        if last:
            break
        sp.lp, sp.lw = lp3, lw3
    acc = acc.view(P, m)
    if mean == "geom":
        return torch.where(torch.isneginf(acc).any(-1), torch.full_like(acc[:, 0], NEG_INF), acc.mean(-1))
    return log_mean_exp(acc, -1)
