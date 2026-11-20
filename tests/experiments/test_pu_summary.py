"""Suites, the standard summary and the results log of the parallel-unmasking comparison."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", ".."), os.path.join(HERE, "..", "..", "experiments", "studies")]


def test_spread_and_its_interval():
    from pulib.summary import spread
    z = np.log(np.random.default_rng(0).lognormal(0.0, 0.3, 400))
    s, lo, hi = spread(z)
    w = np.exp(z)
    assert abs(s - w.std(ddof=1) / w.mean()) < 1e-12 and lo < s < hi and hi - lo < 0.1
    assert spread(np.zeros(50))[0] == 0.0                                 # constant weights
    assert spread([0.0])[0] is None and spread([0.0, float("-inf")])[0] is not None


def test_suite_run_summary_and_log(tmp_path, monkeypatch):
    import compare
    from pulib import summary as S
    from pulib.suites import SUITES, cells
    out = str(tmp_path / "run.json")
    monkeypatch.setattr(sys, "argv", ["compare.py", "--suite", "smoke", "--arms", "fix", "--nrun", "6", "--only", "lex B=0.03", "--out", out])
    compare.main()
    d = json.load(open(out))
    assert d["args"]["suite"] == "smoke" and d["args"]["nrun"] == 6       # an explicit flag overrides the suite
    assert d["args"]["models"] == SUITES["smoke"]["models"] and d["args"]["pilot"] == SUITES["smoke"]["pilot"]
    assert {"commit", "dirty", "torch", "platform", "started"} <= set(d["provenance"])
    rows = S.summarise(out)
    assert len(rows) == 3 * len(cells("smoke")) and {r["suite"] for r in rows} == {"smoke"}
    z = dict(np.load(str(tmp_path / "run.logZ.npz")))
    for r in rows:
        w = np.exp(z[f"{r['model']}|{r['constraint']}|{r['N']}|{r['arm']}"].astype(np.float64))
        assert abs(r["spread"] - w.std(ddof=1) / w.mean()) < 1e-9 and r["spread_lo"] <= r["spread"] <= r["spread_hi"]
        assert abs(r["cost"] - r["calls"] * r["spread"] ** 2) < 1e-12 and r["commit"] == d["provenance"]["commit"]
    log = str(tmp_path / "log.jsonl")
    assert S.append_log(rows, log) == len(rows) and S.append_log(rows, log) == 0      # logging twice adds nothing
    other = [dict(r, nrun=7) for r in rows]                                           # another setting is another line
    assert S.append_log(other, log) == len(rows) and len(S.read_log(log)) == 2 * len(rows)
    v = S.versus(rows, rows, "lex B=0.03", "lex B=0.03 +pull")
    assert len(v) == len(cells("smoke")) and all(abs(x["spread_ratio"] - x["spread_b"] / x["spread_a"]) < 1e-12 for x in v)
    assert "relative to" in S.table(rows, "lex B=0.03")
