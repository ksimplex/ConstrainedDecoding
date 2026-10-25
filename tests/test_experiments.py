"""Experiment plumbing and end-to-end dry runs of the experiment scripts on the synthetic denoiser.

* run_smc gives the same result as one direct twisted_smc call, however the runs are chunked into batches when
  the chunk holds all runs, and concatenates consistently otherwise;
* resampling_arms.py and e1_families.py run end to end, write the expected arrays, and produce only valid
  samples; e1_analysis summarises the E1 output.
"""
import importlib.util, pathlib
import numpy as np
import pytest
import torch
from twistroute.decode import twisted_smc
from twistroute.experiment import load_denoiser, fig1a_setup, run_smc, z_summary
from twistroute.torch_routes import build_route

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(rel):
    spec = importlib.util.spec_from_file_location(pathlib.Path(rel).stem, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_run_smc_single_chunk_equals_direct_call():
    M, dev = load_denoiser(16, synthetic=True)
    cl, route = fig1a_setup(M, 16, dev, pool=8, seed=1)
    a = twisted_smc(M, cl, route, 16, 4, 64, 4, torch.Generator().manual_seed(9), ess_min=0.5)
    b = run_smc(M, cl, route, 16, 4, 64, 4, torch.Generator().manual_seed(9), chunk=256, ess_min=0.5)
    assert np.array_equal(a["logZ"], b["logZ"]) and np.array_equal(a["n_res"], b["n_res"])


def test_run_smc_chunks_concatenate_and_select():
    M, dev = load_denoiser(16, synthetic=True)
    cl, route = fig1a_setup(M, 16, dev)
    r = run_smc(M, cl, route, 16, 4, 70, 4, torch.Generator().manual_seed(1), chunk=64,
                select_seed=lambda j, i: j, return_trace=True)
    assert r["logZ"].shape == (70,) and r["samples"].shape == (70, 16) and r["trace"].shape == (4, 70, 4)
    assert route.accepts(torch.tensor(r["samples"])).all() and not r["any_finite_invalid"]
    s = z_summary(r["logZ"], 0.5)
    assert s["ratio"] > 0 and s["se_rel"] > 0


def test_resampling_arms_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load("experiments/resampling_arms.py")
    mod.main(["--synthetic", "--arms", "K4_plain,K4_def25,K4_off_trace", "--runs", "128", "--native", "2048",
              "--out", "ra"])
    Z = np.load("ra.npz")
    assert {"h0", "logZ_K4_plain", "logZ_K4_def25", "trace_K4_off_trace"} <= set(Z)
    assert Z["logZ_K4_plain"].shape == (128,) and Z["trace_K4_off_trace"].shape == (8, 128, 4)
    assert 0 < Z["h0"][0] < 1


def test_resampling_arms_rejects_unknown_arm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load("experiments/resampling_arms.py")
    with pytest.raises(SystemExit):
        mod.main(["--synthetic", "--arms", "K4_nonsense"])


def test_rng_tests_report_calibrated_multinomial():
    mod = _load("experiments/resampling_arms.py")
    r = mod.rng_tests("cpu", B=20_000)
    assert r["after_equals_unadvanced"] < 0.01 and r["max_abs_corr_idx_next_uniform"] < 0.05
    assert all(abs(p - f) < 0.02 for p, f in r["calibration"])


def test_e1_families_and_analysis_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load("experiments/e1_families.py")
    mod.main(["--cells", str(ROOT / "experiments/configs/e2_cells.json"), "--synthetic",
              "--families", "regular,dyck1_amb", "--runs", "step_exact:64,K4_def:32,K16_off:16",
              "--min-ref", "40", "--max-topup", "4096", "--out", "e1"])
    Z = np.load("e1.npz")
    specs = mod.family_specs(str(ROOT / "experiments/configs/e2_cells.json"), 32)
    for fam in ("regular", "dyck1_amb"):
        route = build_route(specs[fam])
        for m, n in (("step_exact", 64), ("K4_def", 32), ("K16_off", 16)):
            S = Z[f"con_{fam}_{m}"]
            assert S.shape == (n, 32) and route.accepts(torch.tensor(S.astype(np.int64))).all()
        assert route.accepts(torch.tensor(Z[f"ref_{fam}"].astype(np.int64))).all()
    df = _load("experiments/analysis/e1_analysis.py").analyse(["e1.npz"], n_floor=2)
    assert len(df) == 6 and np.isfinite(df.tv).all() and (df.z_lo <= df.z_hi).all()
