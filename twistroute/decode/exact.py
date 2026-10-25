"""
Exact path laws by forward enumeration, for small toy problems (CPU, numpy).

For a route and a state-dependent toy denoiser, enumerates every single-reveal trajectory and returns
    doob   native law conditioned on C (the target; TWISTER Eq. 6)
    step   law of uncorrected step-exact decoding (Dang & Ermon; biased in general, TWISTER Prop. 3.9)
    fk     terminal law of the Feynman-Kac (twisted) model; equals doob exactly (TWISTER Thm. 4.2,
           extended to any route by Theorem 3 of the theory notes)
    zhat_over_h0, path_ess   exact first and second moments of the K=1 weight
plus finite-K SMC with exact route sampling. Used by tests/test_exact_laws.py as the strongest available
check: equality to machine precision, no sampling noise.
"""
import hashlib
import numpy as np
from scipy.special import logsumexp

MASK = -1
NEG = -np.inf


def clamp(logW, x):
    lw = np.array(logW, float, copy=True)
    for i, v in enumerate(x):
        if v != MASK:
            row = np.full(lw.shape[1], NEG)
            row[v] = 0.0
            lw[i] = row
    return lw


def sample_exact(route, logW, rng):
    """Exact draw from q(y | C) ∝ prod_i exp(logW[i, y_i]) * 1_C(y), weighted by derivations for ambiguous routes."""
    if hasattr(route, "sample"):
        return route.sample(logW, rng)
    L, A = logW.shape
    lw = np.array(logW, float, copy=True)
    out = []
    for i in range(L):
        sc = np.full(A, NEG)
        for s in range(A):
            if not np.isfinite(lw[i, s]):
                continue
            t = lw.copy()
            row = np.full(A, NEG); row[s] = lw[i, s]; t[i] = row
            sc[s] = route.logZ(t)
        p = np.exp(sc - logsumexp(sc))
        s = int(rng.choice(A, p=p / p.sum()))
        row = np.full(A, NEG); row[s] = lw[i, s]; lw[i] = row
        out.append(s)
    return tuple(out)


class ToyDenoiser:
    """State-dependent mean-field denoiser. eps controls re-conditioning on revealed tokens (eps=0: frozen)."""
    def __init__(self, L, A, eps=1.0, seed=0, scale=1.0):
        self.L, self.A, self.eps, self.seed = L, A, eps, seed
        self.base = np.random.default_rng(10_000 + seed).normal(0, scale, (L, A))
        self._cache = {}

    def __call__(self, x):
        x = tuple(x)
        if x not in self._cache:
            h = int(hashlib.md5(f"{self.seed}|{x}".encode()).hexdigest()[:8], 16)
            lg = self.base + self.eps * np.random.default_rng(h).normal(0, 1, (self.L, self.A))
            self._cache[x] = lg - logsumexp(lg, axis=1, keepdims=True)
        return self._cache[x]


class Decoder:
    def __init__(self, route, den, L, A):
        self.route, self.den, self.L, self.A = route, den, L, A
        self._zloc = {}

    def z(self, xp, x):
        return float(self.route.logZ(clamp(self.den(x), xp)))

    def zloc(self, x):
        x = tuple(x)
        if x not in self._zloc:
            self._zloc[x] = self.z(x, x)
        return self._zloc[x]

    def eta(self, x):
        if MASK in x:
            return self.zloc(x)
        return 0.0 if self.route.accepts(tuple(x)) else NEG

    def actions(self, x):
        """All single reveals from x: (x', log kappa, log M, log G)."""
        x = tuple(x)
        lw = self.den(x)
        ms = [i for i, v in enumerate(x) if v == MASK]
        zl = self.zloc(x)
        out = []
        for i in ms:
            for s in range(self.A):
                xp = list(x); xp[i] = s; xp = tuple(xp)
                lk = -np.log(len(ms)) + lw[i, s]
                zc = self.z(xp, x)
                if not np.isfinite(zc):
                    out.append((xp, lk, NEG, NEG))
                    continue
                out.append((xp, lk, lk + zc - zl, self.eta(xp) - zc))
        return out

    # ---------------------------------------------------------------- exact laws by forward recursion
    def exact_laws(self):
        x0 = (MASK,) * self.L
        nat, comp = {x0: 0.0}, {x0: 0.0}
        fk, fk2 = {x0: self.eta(x0)}, {x0: 2 * self.eta(x0)}
        for _ in range(self.L):
            nn, cc, ff, f2 = {}, {}, {}, {}
            acts = {x: self.actions(x) for x in set(nat) | set(comp)}
            for x, lp in nat.items():
                for xp, lk, _, _ in acts[x]:
                    nn[xp] = np.logaddexp(nn.get(xp, NEG), lp + lk)
            for x, lp in comp.items():
                for xp, _, lm, lg in acts[x]:
                    if lm == NEG:
                        continue
                    cc[xp] = np.logaddexp(cc.get(xp, NEG), lp + lm)
                    if x in fk:
                        ff[xp] = np.logaddexp(ff.get(xp, NEG), fk[x] + lm + lg)
                        f2[xp] = np.logaddexp(f2.get(xp, NEG), fk2[x] + lm + 2 * lg)
            nat, comp, fk, fk2 = nn, cc, ff, f2

        def norm(d):
            v = np.array(list(d.values())); z = logsumexp(v)
            return {k: float(np.exp(val - z)) for k, val in d.items() if np.isfinite(val)}
        valid = {k: v for k, v in nat.items() if self.route.accepts(k)}
        doob = norm(valid)
        step = norm({k: v for k, v in comp.items() if self.route.accepts(k)})
        fkn = norm({k: v for k, v in fk.items() if np.isfinite(v)})
        log_h0 = float(logsumexp(list(valid.values())))
        log_zhat = float(logsumexp([v for v in fk.values() if np.isfinite(v)]))
        log_m2 = float(logsumexp([v for v in fk2.values() if np.isfinite(v)]))
        return dict(doob=doob, step=step, fk=fkn, h0=float(np.exp(log_h0)),
                    zhat_over_h0=float(np.exp(log_zhat - log_h0)),
                    path_ess=float(np.exp(2 * log_zhat - log_m2)),
                    support_ok=set(step) == set(doob))

    # ---------------------------------------------------------------- finite-K SMC with exact sampling
    def smc(self, K, rng, ess_min=0.5):
        x0 = (MASK,) * self.L
        X = [x0] * K
        logw = np.full(K, self.eta(x0))
        for t in range(self.L):
            new, lg = [], np.zeros(K)
            for k, x in enumerate(X):
                y = sample_exact(self.route, clamp(self.den(x), x), rng)
                ms = [i for i, v in enumerate(x) if v == MASK]
                i = ms[int(rng.integers(len(ms)))]
                xp = list(x); xp[i] = y[i]; xp = tuple(xp)
                lg[k] = self.eta(xp) - self.z(xp, x)
                new.append(xp)
            X, logw = new, logw + lg
            fin = np.isfinite(logw)
            if not fin.any():
                return None
            w = np.where(fin, np.exp(logw - logw[fin].max()), 0.0); w /= w.sum()
            if t < self.L - 1 and 1.0 / (w ** 2).sum() < ess_min * K:
                idx = rng.choice(K, size=K, p=w)
                X = [X[j] for j in idx]
                logw = np.zeros(K)
        fin = np.isfinite(logw)
        w = np.where(fin, np.exp(logw - logw[fin].max()), 0.0); w /= w.sum()
        return X[int(rng.choice(K, p=w))]


def tv(p, q):
    ks = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in ks)
