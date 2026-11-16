"""Gated resampling: a pilot without resampling says at which steps resampling would lower the variance."""
import numpy as np
import torch

GATE_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def resampling_gate(den, cl, route, N, schedule, n, gen, threshold=0.8):
    """Audit for resampling. Runs n single particles without resampling and, at every step t, estimates for each
    exponent alpha the second moment of the final weights if ancestors were drawn there by (weight so far)^alpha,
    relative to not resampling: rho_t(alpha) = E[w_T^2 / w_t^alpha] E[w_t^alpha] / E[w_T^2] (many-particle limit,
    one resampling event). Returns a gate f(revealed fraction) -> alpha for twisted_smc(resample_power=f): the
    minimising alpha where rho < threshold, 0 (do not resample) elsewhere; and the curve, for the record."""
    from twistroute.decode import twisted_smc
    fr = []
    def ps(xb, xa):
        fr.append(float((xa != den.mask_id).float().mean())); return torch.zeros(xb.shape[0], dtype=torch.float64)
    out = twisted_smc(den, cl, route, N, N, nrun=n, K=1, gen=gen, schedule=schedule, ess_min=0.0, return_trace=True, path_stat=ps)
    tr = np.asarray(out["trace"], dtype=np.float64)[:, :, 0]
    T = tr.shape[0]; fin = tr[-1]; ok = np.isfinite(fin)
    wf2 = np.exp(2 * (fin[ok] - fin[ok].max()))
    frac, alpha, rho = np.array(fr[:T]), np.zeros(T), np.ones(T)
    for t in range(T - 1):
        now = tr[t][ok]; m = np.isfinite(now)
        if m.sum() < 10 or now[m].std() < 1e-12:
            continue
        wn = np.exp(now[m] - now[m].max())
        r = [float((wf2[m] / wn ** al).mean() * (wn ** al).mean() / wf2.mean()) for al in GATE_ALPHAS]
        k = int(np.argmin(r))
        if r[k] < threshold:
            alpha[t], rho[t] = GATE_ALPHAS[k], r[k]
    ft, at = torch.tensor(frac), torch.tensor(alpha)
    def gate(f):
        i = torch.searchsorted(ft, f.double().cpu().clamp(max=float(ft[-1])), right=False).clamp(max=T - 1)
        return at[i]
    return gate, dict(frac=frac.round(4).tolist(), alpha=alpha.tolist(), rho=rho.round(4).tolist(), n=int(ok.sum()))
