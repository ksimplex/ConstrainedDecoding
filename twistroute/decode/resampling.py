"""
Resampling for the particle filter: which ancestors survive, and with what weights.

Three unbiased resamplers (`resample_indices`) and the defensive mixture that makes the weight
after resampling bounded.

Plain multinomial resampling draws ancestors from the normalised weights w~. That discards
particles whose current weight is small but whose eventual contribution is not, and on OWT-130M it
cost 7% of h0 (docs/status.md Sec. 5). Defensive resampling draws instead from

    pi = (1 - eps) w~ + eps / K,

so every particle keeps probability at least eps / K and the offspring weight w~ / pi is bounded by
1 / eps. This is defensive importance sampling (Hesterberg, Technometrics 37(2):185-194, 1995,
doi:10.1080/00401706.1995.10484303) applied to the ancestor distribution; the resulting estimator is
still unbiased because the weight correction is exact.

Systematic resampling (Kitagawa, J. Comput. Graph. Statist. 5(1):1-25, 1996,
doi:10.1080/10618600.1996.10474692) uses one uniform and a regular grid, giving lower resampling
variance at equal cost; it is offered as a control rather than a default because its particles are
not independent given the weights.
"""
import math
import torch

RESAMPLERS = ("multinomial", "cpu", "systematic")

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
