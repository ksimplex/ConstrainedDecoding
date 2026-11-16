"""Are sequences that satisfy a forbidden-motif constraint unusually regular? Exact draws from p(. | C) against exact
draws from the model p, 64 positions, default hidden chain. Also the size of the allowed set."""
import sys, json, itertools, numpy as np, torch
sys.path.insert(0, '.')
from pulib.cells import make_model, make_constraint, FORBIDDEN_MOTIFS
torch.set_num_threads(2)
N, n = 64, 40000

def count_allowed(motifs, A, n):
    """Number of length-n strings over A symbols with none of the motifs, by a walk over suffix states."""
    L = max(len(m) for m in motifs); bad = lambda s: any(s[-len(m):] == m for m in motifs if len(s) >= len(m))
    states = {(): 1}
    for _ in range(n):
        new = {}
        for s, c in states.items():
            for a in range(A):
                t = s + (a,)
                if bad(t): continue
                t = t[-(L - 1):] if L > 1 else ()
                new[t] = new.get(t, 0) + c
        states = new
    return sum(states.values())

den = make_model('hidden_0.85', N, 3)
pi, T, E = den.pi.numpy(), den.trans.numpy(), den.emit.numpy()
# exact model law of token pairs (i, i + d), and the model's pointwise log dependence log p(a, b) / (p(a) p(b))
alpha = [pi]
for i in range(N - 1): alpha.append(alpha[-1] @ T[i])
def pair_law(i, d):
    M = np.eye(den.H)
    for k in range(i, i + d): M = M @ T[k]
    return np.einsum('h,ha,hg,gb->ab', alpha[i], E[i], M, E[i + d])
D = (2, 4, 8, 16, 32)
pmi = {d: [np.log(pair_law(i, d) / np.outer(pair_law(i, d).sum(1), pair_law(i, d).sum(0))) for i in range(N - d)] for d in (1,) + D}
mi = {d: float(np.mean([(pair_law(i, d) * pmi[d][i]).sum() for i in range(N - d)])) for d in (1,) + D}

def describe(x):
    x = np.asarray(x); same = x[:, 1:] == x[:, :-1]
    runs = (~same).sum(1) + 1
    o = dict(freq=[float((x == a).mean()) for a in range(3)], adjacent_agree=float(same.mean()), mean_run=float(N / runs.mean()), distinct=int(len({tuple(r) for r in x[:5000]})))
    for d in D:
        o[f'agree_{d}'] = float((x[:, d:] == x[:, :-d]).mean())
        o[f'dep_{d}'] = float(np.mean([pmi[d][i][x[:, i], x[:, i + d]].mean() for i in range(N - d)]))    # model's pointwise dependence, averaged on these sequences
    return o

res = {'model (no constraint)': dict(describe(den.sample(n, torch.Generator().manual_seed(0)).numpy()), log_inv_pC=0.0, log3_share=1.0)}
for name, motifs in FORBIDDEN_MOTIFS.items():
    cl, route, truth = make_constraint(name, den, N)
    cnt = count_allowed(motifs, 3, N)
    o = describe(truth.sample(n, np.random.default_rng(0)))
    o.update(log_inv_pC=float(-truth.log_pC), allowed_log=float(np.log(float(cnt))), log3_share=float(np.log(float(cnt)) / (N * np.log(3))), motifs=[''.join(map(str, m)) for m in motifs])
    res[name] = o
res['toy: 2 symbols, ban "ab"'] = dict(allowed=count_allowed([(0, 1)], 2, N), total_log2=N)
json.dump(dict(mi=mi, rows=res), open('/tmp/regularity.json', 'w'), indent=1)
print('toy check: 2 symbols, ban one switch, length %d: %d allowed sequences (n + 1 = %d) of 2^%d' % (N, res['toy: 2 symbols, ban "ab"']['allowed'], N + 1, N))
print('model mutual information by distance (nats):', {d: round(v, 4) for d, v in mi.items()})
print('%-22s %-11s %9s %8s | %-17s %7s %6s | agree at distance 2/4/8/16/32          | model dependence on these sequences minus its average, x1000, d=2/4/8/16/32' % ('case', 'motifs', 'log1/p(C)', 'entropy', 'symbol freq', 'adj=', 'run'))
for k, o in res.items():
    if 'freq' not in o: continue
    print('%-22s %-11s %9.2f %8.3f | %s %7.3f %6.2f | %s | %s' % (k, ','.join(o.get('motifs', [])), o['log_inv_pC'], o['log3_share'], ' '.join('%.3f' % f for f in o['freq']), o['adjacent_agree'], o['mean_run'],
          ' '.join('%.3f' % o[f'agree_{d}'] for d in D), ' '.join('%+7.1f' % (1000 * (o[f'dep_{d}'] - mi[d])) for d in D)))
