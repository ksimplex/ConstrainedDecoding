"""Standard report of comparison outputs, and the log of results across settings.

    report.py OUT.json [--baseline ARM] [--log]          table per cell; --log appends the summary to the log
    report.py A.json --against B.json                    the same arms in two outputs, cell by cell (b / a)
    report.py OUT.json --arm-a ARM --arm-b ARM           two arms of one output, cell by cell (b / a)
    report.py --query model=hidden_0.85 constraint=copy_block N=256 [arm=lex]
                                                         every logged setting of the matching cells, cheapest first
The log is results/parallel_unmasking/log.jsonl (pulib/summary.py describes the columns).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulib import summary as S                                                                    # noqa: E402

CONTROL = "fixed 1/step, no resampling"


def print_versus(v):
    if not v:
        print("no matching (cell, arm) pairs with per-run data"); return
    print(f"{'cell':44s} {'arm a -> arm b':52s} {'spread a':>9s} {'spread b':>9s} {'b / a':>7s} {'calls a':>8s} {'calls b':>8s} {'cost b/a':>9s}  clear")
    for r in v:
        arms = r["arm_a"] if r["arm_a"] == r["arm_b"] else f"{r['arm_a']} -> {r['arm_b']}"
        print(f"{r['model'] + ' | ' + r['constraint'] + ' | ' + str(r['N']):44s} {arms[:52]:52s} {r['spread_a']:9.3f} {r['spread_b']:9.3f} "
              f"{r['spread_ratio']:7.2f} {r['calls_a']:8.1f} {r['calls_b']:8.1f} {S.fmt(r['cost_ratio']):>9s}  {'yes' if r['clear'] else 'no'}")
    better = sum(r["clear"] and r["spread_ratio"] < 1 for r in v); worse = sum(r["clear"] and r["spread_ratio"] > 1 for r in v)
    print(f"{len(v)} pairs: b clearly lower spread in {better}, clearly higher in {worse}, within the intervals in {len(v) - better - worse}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?"); ap.add_argument("--baseline", default=None)
    ap.add_argument("--log", action="store_true"); ap.add_argument("--log-path", dest="log_path", default=S.LOG)
    ap.add_argument("--against", default=None); ap.add_argument("--arm-a", dest="arm_a", default=None); ap.add_argument("--arm-b", dest="arm_b", default=None)
    ap.add_argument("--query", nargs="+", default=None, metavar="FIELD=VALUE")
    a = ap.parse_args()
    if a.query:
        want = dict(q.split("=", 1) for q in a.query)
        hit = lambda r: all((str(r.get(k)) == v) if k != "arm" else str(r.get(k)).startswith(v) for k, v in want.items())
        rows = sorted((r for r in S.read_log(a.log_path) if hit(r)), key=lambda r: (r["model"], r["constraint"], r["N"], r["cost"] if r["cost"] is not None else 1e99))
        print(f"{'cell':40s} {'arm':34s} {'calls':>7s} {'spread [90%]':>22s} {'cost':>8s} {'Z/p(C)':>7s} {'TV %':>6s} {'runs x K':>9s} {'commit':>8s}  source")
        for r in rows:
            sp = "-" if r["spread"] is None else f"{S.fmt(r['spread'])} [{S.fmt(r['spread_lo'])}, {S.fmt(r['spread_hi'])}]"
            print(f"{r['model'] + ' | ' + r['constraint'] + ' | ' + str(r['N']):40s} {r['arm'][:34]:34s} {r['calls']:7.1f} {sp:>22s} {S.fmt(r['cost'], 3):>8s} "
                  f"{S.fmt(r['z_ratio']):>7s} {100 * r['marg_tv']:6.2f} {str(r['nrun']) + ' x ' + str(r['K']):>9s} {str(r['commit'] or '-'):>8s}  {r['source']}")
        print(f"{len(rows)} logged results")
        return
    if not a.out:
        ap.error("an output file or --query is needed")
    rows = S.summarise(a.out)
    if a.against or a.arm_a:
        print_versus(S.versus(rows, S.summarise(a.against) if a.against else rows, a.arm_a, a.arm_b))
    else:
        base = a.baseline or (CONTROL if any(r["arm"] == CONTROL for r in rows) else None)
        print(S.table(rows, base))
    if a.log:
        print(f"\nlogged {S.append_log(rows, a.log_path)} new results to {os.path.relpath(a.log_path)}")


if __name__ == "__main__":
    main()
