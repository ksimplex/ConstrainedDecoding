"""
Exact regression for the parallel-unmasking comparison (experiments/parallel_unmasking/compare.py).

Every arm set is run on one or two small cells with fixed seeds and compared, field by field, with reference
outputs frozen in tests/golden/pu_compare.json. Each arm draws from its own named random stream, so a change in
any arm's numbers means the sampler, a schedule, the policy or the metrics changed behaviour. Timing and memory
fields are not compared.

The reference was produced on one platform and torch version (recorded in the file). Multinomial draws can flip
on last-bit differences in floating point, so on another platform the comparison is skipped, not loosened.

Regenerate (only when a change in behaviour is intended, and say so in the commit):
    python tests/test_pu_regression.py
"""
import json, math, os, platform, sys, time
import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "experiments", "parallel_unmasking")]
GOLDEN = os.path.join(HERE, "golden", "pu_compare.json")
BASE = dict(nrun=6, nrun_mixed=4, K=4, pilot=60, pilot_small=20, seed=0, order_theta=0.0, gate_pilot=200,
            resample_ks=[4, 16], resample_modes=None, only=None, partial=None)
CASES = [("full", "hidden_0.85", "copy", 32), ("core", "chain_0.85", "dyck", 32),
         ("order", "hidden_0.85", "counting", 32), ("order", "segments_0.85", "forbidden", 32),
         ("budget", "hidden_0.85", "forbidden", 32), ("budget", "repel_0.8", "counting", 32),
         ("interact", "hidden_0.85", "copy", 32), ("interact", "chain_0.6", "dyck", 32),
         ("thick", "hidden_0.85", "forbidden", 64), ("seqorder", "hidden_0.85_n0.15", "copy_block", 80),
         ("seqorder", "hidden_0.85", "two_window", 80), ("fix", "hidden_0.85", "copy_block", 80),
         ("fix", "chain_0.85", "copy_spread", 80), ("fix", "hidden_0.85", "window", 80),
         ("resample", "chain_0.85", "counting", 32)]
SKIP = {"seconds"}


def stamp():
    return dict(torch=torch.__version__, machine=platform.machine(), system=platform.system())


def case_id(c):
    return "|".join(str(x) for x in c)


def run_case(arms, model, constraint, N):
    import compare
    rows, meta, zlog = compare.run_cell((model, constraint, N, dict(BASE, arms=arms)))
    return dict(rows=[{k: v for k, v in r.items() if k not in SKIP} for r in rows],
                logZ={k: np.asarray(v, dtype=np.float64).tolist() for k, v in sorted(zlog.items())},
                kernel=[meta["kernel_unscreened"], meta["kernel_screened"]], gate=meta.get("resampling_gate"))


def differences(a, b, path=""):
    """Paths at which two nested structures differ (numbers: relative 1e-9, NaN equals NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        return out + [d for k in sorted(set(a) & set(b)) for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    a, b = float(a), float(b)
    same = (math.isnan(a) and math.isnan(b)) or a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b)) + 1e-12
    return [] if same else [f"{path}: {a!r} != {b!r}"]


def _golden():
    return json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else None


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_compare_matches_frozen_reference(case):
    g = _golden()
    if g is None:
        pytest.skip("no reference file; run python tests/test_pu_regression.py")
    if g["stamp"] != stamp():
        pytest.skip(f"reference made on {g['stamp']}, this is {stamp()}")
    got = json.loads(json.dumps(run_case(*case)))          # the same JSON round trip as the reference
    diff = differences(g["cases"][case_id(case)], got)
    assert not diff, f"{len(diff)} differences, first: {diff[:5]}"


if __name__ == "__main__":
    out, t0 = dict(stamp=stamp(), base=BASE, cases={}), time.time()
    for c in CASES:
        t = time.time(); out["cases"][case_id(c)] = run_case(*c)
        print(f"{case_id(c):45s} {len(out['cases'][case_id(c)]['rows']):3d} arms {time.time() - t:6.1f}s", flush=True)
    json.dump(out, open(GOLDEN, "w"))
    print(f"wrote {GOLDEN} in {time.time() - t0:.0f}s, {os.path.getsize(GOLDEN) / 1e3:.0f} kB")
