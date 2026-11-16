"""One standard summary of a comparison output, and an append-only log of summaries across runs and settings.

Per (cell, arm) the summary holds
    calls      denoiser calls per particle (reveal steps)
    spread     sd / mean of Z_hat over runs, with a 90% bootstrap interval over runs (the interval is too narrow
               when the weights are heavy-tailed; `z_se_unreliable` flags that)
    cost       calls * spread^2: calls needed for a given relative error of Z_hat, up to a constant
    z_ratio    mean Z_hat / exact p(C), with its standard error (1 for an exact schedule; under a budget it
               estimates the native hit rate of that schedule instead)
    marg_tv    mean over positions of TV(weighted class marginal, exact constrained marginal)
    sd_r       spread of the parallel-reveal gap r (0 for an exact schedule)
together with the run settings and the provenance of the code. `spread` needs the per-run file <out>.logZ.npz.

The log is JSON lines, one per (run, cell, arm). A run is identified by its settings and code version, so logging
the same output twice adds nothing, and the same arm under two settings or two commits gives two comparable lines.
"""
import json, os, platform, subprocess, time, zlib
import numpy as np

KEY = ("suite", "model", "constraint", "N", "arm", "arms", "nrun", "K", "pilot", "seed", "commit")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "parallel_unmasking", "log.jsonl")


def provenance():
    """Code version and platform of the running process (commit is None outside a git checkout)."""
    here = os.path.dirname(os.path.abspath(__file__))
    def git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=here, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:
            return None
    try:
        import torch; tv = torch.__version__
    except Exception:
        tv = None
    return dict(commit=git("rev-parse", "--short", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                torch=tv, platform=f"{platform.system()}-{platform.machine()}", started=time.strftime("%Y-%m-%d %H:%M"))


def spread(logz, n_boot=1000, seed=0):
    """sd / mean of Z_hat over runs from per-run log Z_hat, with a 90% bootstrap interval -> (value, low, high)."""
    w = np.exp(np.asarray(logz, dtype=np.float64).reshape(-1)); n = len(w)
    if n < 2 or not np.isfinite(w).all() or w.mean() == 0:
        return None, None, None
    f = lambda v: float(v.std(ddof=1) / v.mean())
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    b = w[idx]
    with np.errstate(invalid="ignore", divide="ignore"):                 # a resample of runs that all have Z_hat = 0
        bs = b.std(axis=1, ddof=1) / b.mean(axis=1)                       # has no spread; it is left out
    return f(w), float(np.nanquantile(bs, 0.05)), float(np.nanquantile(bs, 0.95))


def summarise(path):
    """Standard summary rows of one comparison output (see the module docstring)."""
    d = json.load(open(path)); a = d["args"]; prov = d.get("provenance") or {}
    zpath = os.path.splitext(path)[0] + ".logZ.npz"
    z = dict(np.load(zpath)) if os.path.exists(zpath) else {}
    out = []
    for r in d["rows"]:
        key = f"{r['model']}|{r['constraint']}|{r['N']}|{r['arm']}"
        s, lo, hi = spread(z[key], seed=zlib.crc32(key.encode())) if key in z else (None, None, None)
        out.append(dict(suite=a.get("suite"), model=r["model"], constraint=r["constraint"], N=r["N"], arm=r["arm"],
                        arms=a.get("arms"), nrun=r["nrun"], K=r["K"], pilot=a.get("pilot"), seed=a.get("seed"),
                        commit=prov.get("commit"), dirty=prov.get("dirty"), started=prov.get("started"),
                        calls=r["steps"], spread=s, spread_lo=lo, spread_hi=hi,
                        cost=None if s is None else r["steps"] * s * s,
                        z_ratio=r["z_ratio"], z_ratio_se=r["z_ratio_se"], z_se_unreliable=r.get("z_se_unreliable"),
                        marg_tv=r["marg_tv"], sd_r=r.get("sd_r"), n_res=r.get("n_res"), ess=r.get("ess"),
                        source=os.path.basename(path)))
    return out


def read_log(path=LOG):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def append_log(rows, path=LOG):
    """Append the rows that are not in the log yet (same settings, code version and source). Returns how many."""
    ident = lambda r: tuple(r.get(k) for k in KEY) + (r.get("source"),)
    have = {ident(r) for r in read_log(path)}
    new = [r for r in rows if ident(r) not in have]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    return len(new)


def by_cell(rows):
    out = {}
    for r in rows:
        out.setdefault((r["model"], r["constraint"], r["N"]), []).append(r)
    return out


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if abs(x) >= 10 ** -nd or x == 0 else f"{x:.1e}")


def table(rows, baseline=None):
    """Text table, one block per cell. With `baseline` (an arm name) each arm also gets its spread and cost
    relative to that arm in the same cell."""
    lines = []
    for (m, c, N), rs in by_cell(rows).items():
        base = next((r for r in rs if r["arm"] == baseline), None)
        lines.append(f"\n{m} | {c} | N={N}" + (f"   (relative to: {baseline})" if base else ""))
        lines.append(f"  {'arm':38s} {'calls':>7s} {'spread [90%]':>22s} {'Z/p(C)':>14s} {'TV %':>6s} {'sd r':>6s} {'cost':>8s}" + ("   spread x   cost x" if base else ""))
        for r in rs:
            sp = "-" if r["spread"] is None else f"{fmt(r['spread'])} [{fmt(r['spread_lo'])}, {fmt(r['spread_hi'])}]"
            line = (f"  {r['arm'][:38]:38s} {r['calls']:7.1f} {sp:>22s} {fmt(r['z_ratio'])+' ± '+fmt(r['z_ratio_se']):>14s} "
                    f"{100 * r['marg_tv']:6.2f} {fmt(r['sd_r']):>6s} {fmt(r['cost'], 3):>8s}")
            if base and r["spread"] is not None and base["spread"]:
                line += f"   {r['spread'] / base['spread']:8.2f} {r['cost'] / base['cost']:8.2f}"
            lines.append(line)
    return "\n".join(lines)


def versus(rows_a, rows_b, arm_a=None, arm_b=None):
    """Cell-by-cell comparison of two settings: the same arm in two outputs, or two arms (arm_a in rows_a against
    arm_b in rows_b). Returns one dict per matched (cell, arm pair) with the ratio b / a of spread, cost and calls,
    and whether the two 90% intervals of the spread are disjoint."""
    idx = {(r["model"], r["constraint"], r["N"], r["arm"]): r for r in rows_b}
    out = []
    for r in rows_a:
        if arm_a is not None and r["arm"] != arm_a:
            continue
        o = idx.get((r["model"], r["constraint"], r["N"], arm_b if arm_b is not None else r["arm"]))
        if o is None or r["spread"] is None or o["spread"] is None:
            continue
        out.append(dict(model=r["model"], constraint=r["constraint"], N=r["N"], arm_a=r["arm"], arm_b=o["arm"],
                        spread_a=r["spread"], spread_b=o["spread"], spread_ratio=o["spread"] / r["spread"],
                        calls_a=r["calls"], calls_b=o["calls"], cost_ratio=o["cost"] / r["cost"] if r["cost"] else None,
                        tv_a=r["marg_tv"], tv_b=o["marg_tv"],
                        clear=bool(o["spread_lo"] > r["spread_hi"] or o["spread_hi"] < r["spread_lo"])))
    return out
