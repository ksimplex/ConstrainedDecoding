"""Construct a batched route from a JSON-serialisable constraint spec (as used by experiments/configs).

spec keys: "kind" in {"forbidden", "exactly_k", "prefix", "copy", "cfg"}, "A" (alphabet size), "N" (length),
plus kind parameters: patterns | cls, count | cls | pairs (first `pairs` positions copied to N/2 + i) |
grammar in {"dyck1_unamb", "dyck1_amb", "dyck2", "rna_pair"}.
"""
from .. import compile as C
from .routes import TorchChain, TorchChart, TorchCopy

GRAMMARS = {"dyck1_unamb": C.DYCK1_UNAMBIGUOUS, "dyck1_amb": C.DYCK1_AMBIGUOUS, "dyck2": C.DYCK2}


def cpu_route(spec):
    """CPU reference route for the same spec (used by tests and small-scale checks)."""
    k, A, N = spec["kind"], spec["A"], spec["N"]
    if k == "forbidden":
        return C.dfa_forbidden([tuple(p) for p in spec["patterns"]], A)
    if k == "exactly_k":
        return C.dfa_exactly_k(spec["cls"], spec["count"], A)
    if k == "prefix":
        return C.dfa_prefix_class(spec["cls"], A)
    if k == "cfg":
        g = C.rna_pair_grammar() if spec["grammar"] == "rna_pair" else GRAMMARS[spec["grammar"]]
        return C.Inside(*g)
    raise ValueError(f"no CPU route object for kind {k!r}")


def build_route(spec, device="cpu"):
    k = spec["kind"]
    if k == "copy":
        N = spec["N"]
        return TorchCopy([(i, N // 2 + i) for i in range(spec["pairs"])], N, spec["A"], device)
    r = cpu_route(spec)
    return TorchChart(r, device) if k == "cfg" else TorchChain(r, device)
