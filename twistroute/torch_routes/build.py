"""Construct a batched route from a JSON-serialisable constraint spec (as used by experiments/configs).

spec keys: "kind" in {"forbidden", "exactly_k", "prefix", "copy", "cfg", "and"}, "A" (alphabet size), "N"
(length), plus kind parameters: patterns | cls, count | cls | pairs (first `pairs` positions copied to N/2 + i)
| grammar in {"dyck1_unamb", "dyck1_amb", "dyck2", "rna_pair"} with optional span_limits {nonterminal: W} |
parts (a list of specs, all with the same A and N).

"and" compiles a conjunction to one route: automata are combined by the product construction, and an
automaton with one grammar by the Bar-Hillel product (still a grammar, so it runs on the batched chart).
Conjunctions of two grammars are not context-free in general and are refused.
"""
from .. import compile as C
from .routes import TorchChain, TorchChart, TorchCopy

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
    dfas = [r for r in routes if not isinstance(r, C.Inside)]
    if len(grams) > 1:
        raise C.CompilerRefuse("conjunction of two grammars is not context-free in general")
    d = None
    for r in dfas:
        d = r if d is None else C.product_dfa(d, r, spec["A"])
    if not grams:
        return d
    return grams[0] if d is None else C.intersect_dfa_cfg(d, grams[0])


def build_route(spec, device="cpu"):
    k = spec["kind"]
    if k == "copy":
        N = spec["N"]
        return TorchCopy([(i, N // 2 + i) for i in range(spec["pairs"])], N, spec["A"], device)
    r = cpu_route(spec)
    return TorchChart(r, device) if isinstance(r, C.Inside) else TorchChain(r, device)
