"""
Native (unconstrained) decoding under the same kernel as decode.smc, plus the two-sided identity check.

native_sample        rejection-sampling references: unconstrained samples whose class sequences give h0 and the
                     Doob law p(. | C) (TWISTER Sec. 4.3, Eq. 7) by filtering.
native_path_weights  along native trajectories, the importance weight that K=1 twisted SMC would assign,
                         w = prod_t eta_t / Z(x_{t+1}; x_t)  with eta_T = 1_C.
                     For a correct implementation, q(path) w(path) = p(path) 1_C(x_T) for every path, so
                         E_q[w] = h0            and          E_p[1_C / w] = q(valid) = 1,
                     and the law of log w under p(. | C) equals its law under q tilted by w / h0. The native
                     side samples exactly where p has mass, so it exposes high-weight regions that q rarely
                     visits; a mismatch between the native kernel and the proposal breaks E_p[1_C / w] = 1.
"""
import torch
from .smc import reveal_sets, clamp_revealed


@torch.no_grad()
def native_sample(den, classes, N, T, B, gen=None, return_tokens=False):
    """B native samples of length N in T steps. Returns class sequences (B, N) (and tokens if requested)."""
    dev = classes.onehot.device
    x = torch.full((B, N), den.mask_id, dtype=torch.long, device=dev)
    for t in range(T):
        r = reveal_sets(x == den.mask_id, N // T, gen)
        lp = den.logits(x)
        rows = torch.gather(lp, 1, r[..., None].expand(-1, -1, lp.shape[-1]))
        tok = torch.multinomial(torch.exp(rows).reshape(-1, rows.shape[-1]), 1, generator=gen).reshape(B, -1)
        x.scatter_(1, r, tok)
    cls = classes.token_class[x]
    return (cls, x) if return_tokens else cls


@torch.no_grad()
def native_path_weights(den, classes, route, N, T, B, gen=None):
    """B native trajectories; returns (log w along each path, validity of its final class sequence).
    log w = -inf for paths that leave C."""
    dev = classes.onehot.device
    ohA, tokA = classes.onehot, classes.token_class
    x = torch.full((B, N), den.mask_id, dtype=torch.long, device=dev)
    ycls = torch.zeros(B, N, dtype=torch.long, device=dev)
    lp = den.logits(x)
    lw = classes.class_logw(lp)
    logw = route.logZ(clamp_revealed(lw, x, ycls, den.mask_id))
    nt = N // T
    for t in range(T):
        r = reveal_sets(x == den.mask_id, nt, gen)
        rows = torch.gather(lp, 1, r[..., None].expand(-1, -1, lp.shape[-1]))
        tok = torch.multinomial(torch.exp(rows).reshape(-1, rows.shape[-1]), 1, generator=gen).reshape(B, nt)
        x2 = x.clone(); x2.scatter_(1, r, tok)
        y2 = ycls.clone(); y2.scatter_(1, r, tokA[tok])
        z_old = route.logZ(clamp_revealed(lw, x2, y2, den.mask_id))
        if t < T - 1:
            lp = den.logits(x2)
            lw = classes.class_logw(lp)
            eta2 = route.logZ(clamp_revealed(lw, x2, y2, den.mask_id))
        else:
            ok = route.accepts(tokA[x2])
            eta2 = torch.where(ok, torch.zeros_like(z_old), torch.full_like(z_old, float("-inf")))
        dead = ~torch.isfinite(z_old)
        logw = torch.where(dead, torch.full_like(logw, float("-inf")), logw + (eta2 - z_old))
        x, ycls = x2, y2
    return logw.cpu().numpy(), route.accepts(tokA[x]).cpu().numpy()
