"""Attribute a sequential order's KL(sampler || exact law) to the positions it reveals: the sum over steps of the
true error KL(proposal marginal || exact conditional) at the revealed position, split by distance from the window.
Also the estimate (chain-vs-product disagreement) at the same position, to see where it is wrong."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, '.')
import twist_error as te
from pulib.cells import make_model, make_constraint
from pulib.orders import cheapest_order, estimate_order, kl
from twistroute.decode import CouplingPolicy, ChainTwist, fit_pair_kernel, fit_distance_kernel, COUNT_EDGES, constrained_marginals
from twistroute.decode.policy import RevealState
from twistroute.decode.schedule import clamp_revealed
torch.set_num_threads(2)
N, P = 64, 400
out = {}
for cname in ('window', 'two_window'):
    den = make_model('hidden_0.85', N, 3); cl, route, truth = make_constraint(cname, den, N)
    act = np.asarray(truth.active, dtype=bool); win = torch.tensor(act)
    wpos = np.nonzero(act)[0]; dist = torch.tensor([int(np.abs(wpos - i).min()) for i in range(N)])
    group = torch.where(dist == 0, 0, torch.where(dist <= 6, 1, 2))            # inside, within 6 (one correlation length), farther
    G = lambda s: torch.Generator().manual_seed(s)
    K = fit_pair_kernel(den, cl, 300, G(2), N=N); tw = ChainTwist(truth.wdfa, K, active=act)
    kern, _ = fit_distance_kernel(den, N, 64, G(1), count_edges=COUNT_EDGES)
    ref = te.Twists(den, truth.wdfa, act)
    res = {}
    for oname, order in (('estimate', estimate_order(tw)), ('cheapest', cheapest_order(ref, den)), ('lex', 'lex'), ('random', 'random')):
        t0 = time.time()
        pol = CouplingPolicy(kern, 0.0, priority=order, mixed=False, max_set=1)
        x = torch.full((P, N), den.mask_id); g = G(7); rows = torch.arange(P)
        true_g = torch.zeros(3, dtype=torch.float64); est_g = torch.zeros(3, dtype=torch.float64); cnt_g = torch.zeros(3)
        step_g = torch.zeros(3, dtype=torch.float64); logw = torch.zeros(P, dtype=torch.float64)
        for t in range(N):
            masked = x == den.mask_id
            ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
            lw = cl.class_logw(den.logits(x)); st = RevealState(x, ycls, lw, den.mask_id, cl, route)
            idx, valid = pol.reveal(st, t, N, g); i = idx[:, 0]
            lwc = clamp_revealed(lw, x, ycls, den.mask_id)
            q0 = constrained_marginals(route, lwc)[rows, i]                       # the proposal at the revealed position
            q1 = tw.marginals(lwc)[rows, i]
            pm = torch.exp(den.log_cond(x))[rows, i]                              # model conditional at i
            pi = torch.zeros(P, den.S, dtype=torch.float64)
            for a in range(den.S):
                xa = x.clone(); xa[rows, i] = a
                pi[:, a] = pm[:, a] * torch.exp(ref.true(xa))
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-300)
            e_true = kl(q0, pi); e_est = kl(q0, q1)
            gi = group[i]
            for k in range(3):
                m = gi == k
                true_g[k] += e_true[m].sum(); est_g[k] += e_est[m].sum(); cnt_g[k] += m.sum(); step_g[k] += float(m.sum()) * t
            a = torch.multinomial(q0.clamp_min(0), 1, generator=g).squeeze(-1)
            logw += torch.log(pi[rows, a].clamp_min(1e-300)) - torch.log(q0[rows, a].clamp_min(1e-300))
            x[rows, i] = a
        r = dict(kl_from_steps=float(true_g.sum() / P), kl_from_weights=float(-logw.mean()), kl_weights_se=float(logw.std() / P ** 0.5),
                 true_inside=float(true_g[0] / P), true_near=float(true_g[1] / P), true_far=float(true_g[2] / P),
                 est_inside=float(est_g[0] / P), est_near=float(est_g[1] / P), est_far=float(est_g[2] / P),
                 mean_step_inside=float(step_g[0] / cnt_g[0]), mean_step_near=float(step_g[1] / cnt_g[1]), mean_step_far=float(step_g[2] / cnt_g[2]),
                 positions=[int((group == k).sum()) for k in range(3)], seconds=time.time() - t0)
        res[oname] = r
        print(cname, oname, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out[cname] = res
json.dump(out, open('/tmp/est_attrib.json', 'w'), indent=1)
