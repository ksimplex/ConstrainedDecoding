"""Construct a batched route from a JSON-serialisable constraint spec (as used by experiments/configs).

spec keys: "kind" in {"forbidden", "exactly_k", "prefix", "copy", "cfg", "factor_graph", "and"}, "A" (alphabet size), "N"
(length), plus kind parameters: patterns | cls, count | cls | pairs (first `pairs` positions copied to N/2 + i)
| grammar in {"dyck1_unamb", "dyck1_amb", "dyck2", "rna_pair"} with optional span_limits {nonterminal: W} |
parts (a list of specs, all with the same A and N).

"and" compiles a conjunction to one route: automata are combined by the product construction, and an
automaton with one grammar by the Bar-Hillel product (still a grammar, so it runs on the batched chart), and
an automaton with a factor graph through augmented variables (symbol, automaton state). Two grammars, or a
grammar with a factor graph, are refused.
"""
from .. import compile as C
from .routes import TorchChain, TorchChart, TorchCopy
from .factor_graph import TorchFactorGraph, TorchLifted

GRAMMARS = {"dyck1_unamb": C.DYCK1_UNAMBIGUOUS, "dyck1_amb": C.DYCK1_AMBIGUOUS, "dyck2": C.DYCK2}


def cpu_route(spec):
    """CPU reference route for the same spec (used by tests and small-scale checks)."""
    k, A, N = spec["kind"], spec["A"], spec["N"]
    if k == "and":
        return _conjunction(spec)
    if k == "forbidden":
        return C.dfa_forbidden([tuple(p) for p in spec["patterns"]], A)
    if k == "exactly_k":
        return C.dfa_exactly_k(spec["cls"], spec["count"], A)
    if k == "prefix":
        return C.dfa_prefix_class(spec["cls"], A)
    if k == "factor_graph":
        fs = [(tuple(f["scope"]), f["table"]) for f in spec["factors"]]
        order = spec.get("order") or C.min_degree_order(N, [s for s, _ in fs])
        return C.FactorGraph(N, A, fs, order=order, name=spec.get("name", "fg"))
    if k == "cfg":
        g = C.rna_pair_grammar() if spec["grammar"] == "rna_pair" else GRAMMARS[spec["grammar"]]
        return C.Inside(*g, span_limits=spec.get("span_limits"))
    raise ValueError(f"no CPU route object for kind {k!r}")


def _conjunction(spec):
    parts = spec["parts"]
    if any(p["A"] != spec["A"] or p["N"] != spec["N"] for p in parts):
        raise ValueError("conjunction parts must share A and N")
    if any(p["kind"] == "copy" for p in parts):
        raise C.CompilerRefuse("conjunctions with the closed-form copy route are not supported")
    routes = [cpu_route(p) for p in parts]
    grams = [r for r in routes if isinstance(r, C.Inside)]
    fgs = [r for r in routes if isinstance(r, C.FactorGraph)]
    dfas = [r for r in routes if isinstance(r, C.WDFA)]
    if len(grams) > 1 or len(fgs) > 1 or (grams and fgs):
        raise C.CompilerRefuse("conjunction needs at most one grammar or one factor graph besides automata")
    d = None
    for r in dfas:
        d = r if d is None else C.product_dfa(d, r, spec["A"])
    other = (grams or fgs or [None])[0]
    if other is None:
        return d
    if d is None:
        return other
    return C.intersect_dfa_cfg(d, other) if grams else C.DFAFactorProduct(d, other)


def build_route(spec, device="cpu"):
    k = spec["kind"]
    if k == "copy":
        N = spec["N"]
        return TorchCopy([(i, N // 2 + i) for i in range(spec["pairs"])], N, spec["A"], device)
    r = cpu_route(spec)
    return torch_route(r, device)


def torch_route(r, device="cpu"):
    """Batched route for a CPU route object."""
    if isinstance(r, C.Inside):
        return TorchChart(r, device)
    if isinstance(r, C.FactorGraph):
        return TorchFactorGraph(r, device)
    if isinstance(r, C.DFAFactorProduct):
        return TorchLifted(TorchFactorGraph(r._aug, device), r.base_of, r.A, device, r.name, valid=r.valid)
    return TorchChain(r, device)
