"""
Batched factor-graph route: exact logZ by variable elimination and exact sampling by backward sampling, for a
fixed elimination order. The plan (which tables meet in which bucket) depends only on the factor scopes, so it
is computed once; at run time every bucket is one broadcast sum and one logsumexp over a (B, A^k) table, with
k the bucket's scope size. Cost per pass: O(B * sum over buckets of A^k), exponential only in the induced
width of the order (compile.induced_width). Hard factors enter as log tables with -inf for forbidden tuples.

TorchLifted wraps a route over an augmented alphabet (each augmented symbol maps to one base symbol), as built
by compile.DFAFactorProduct, so that it reads and returns base-alphabet weights and sequences.
"""
import torch
from .routes import _gumbel_argmax, onehot_log


class TorchFactorGraph:
    def __init__(self, fg, device="cpu", order=None):
        self.L, self.A, self.name, self.device = fg.L, fg.A, getattr(fg, "name", "fg"), device
        order = list(order if order is not None else fg.order)
        tables = [((i,), None) for i in range(self.L)]                      # unaries come from lw
        self.consts = []
        for scope, logt in fg.factors:
            perm = sorted(range(len(scope)), key=lambda k: scope[k])
            t = torch.as_tensor(logt, dtype=torch.float32).permute(*perm).contiguous()
            self.consts.append(t.to(device)[None])                          # (1, A, ..., A), sorted scope
            tables.append((tuple(sorted(scope)), len(self.consts) - 1))
        live = list(range(len(tables)))
        scopes = [s for s, _ in tables]
        self.src = tables
        self.buckets = []                                                   # (v, scope, input table ids)
        for v in order:
            ins = [k for k in live if v in scopes[k]]
            if not ins:
                continue
            scope = tuple(sorted(set().union(*[scopes[k] for k in ins])))
            live = [k for k in live if k not in ins] + [len(scopes)]
            scopes.append(tuple(x for x in scope if x != v))
            self.buckets.append((v, scope, ins, len(scopes) - 1))
        self.final = live
        self.scopes = scopes
        if set(order) != set(range(self.L)) or len(self.buckets) != self.L:
            raise ValueError("elimination order must list every variable exactly once")

    def _expand(self, t, s, full):
        A = self.A
        return t.reshape(t.shape[0], *[A if x in s else 1 for x in full])

    def _eliminate(self, lw, keep=False):
        vals = {}
        for k, (s, c) in enumerate(self.src):
            vals[k] = lw[:, s[0], :] if c is None else self.consts[c].to(lw.dtype)
        kept = []
        for v, scope, ins, out in self.buckets:
            acc = sum(self._expand(vals[k], self.scopes[k], scope) for k in ins)
            vals[out] = torch.logsumexp(acc, dim=1 + scope.index(v))
            if keep:
                kept.append(acc)
            for k in ins:
                del vals[k]
        z = sum(vals[k].reshape(vals[k].shape[0]) for k in self.final)
        return z, kept

    def logZ(self, lw):
        return self._eliminate(lw.float())[0].double()

    def sample(self, lw, gen=None):
        lw = lw.float()
        B = lw.shape[0]
        _, kept = self._eliminate(lw, keep=True)
        y = torch.empty(B, self.L, dtype=torch.long, device=lw.device)
        ar = torch.arange(B, device=lw.device)
        for (v, scope, _, _), acc in zip(reversed(self.buckets), reversed(kept)):
            idx = (ar,) + tuple(slice(None) if x == v else y[:, x] for x in scope)
            y[:, v] = _gumbel_argmax(acc[idx], gen)                         # (B, A): exact conditional of v
        return y

    def accepts(self, y):
        return torch.isfinite(self.logZ(onehot_log(y, self.A)))


class TorchLifted:
    """Route over an augmented alphabet seen through the base alphabet: augmented symbol a has base symbol
    base_of[a], its weight is the base weight, and samples are mapped back."""

    def __init__(self, inner, base_of, A, device="cpu", name=None):
        self.inner, self.A = inner, A
        self.base_of = torch.as_tensor(base_of, dtype=torch.long, device=device)
        self.name = name or getattr(inner, "name", "lifted")

    def logZ(self, lw):
        return self.inner.logZ(lw[..., self.base_of])

    def sample(self, lw, gen=None):
        return self.base_of[self.inner.sample(lw[..., self.base_of], gen)]

    def accepts(self, y):
        return torch.isfinite(self.logZ(onehot_log(y, self.A)))
