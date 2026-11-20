"""Score stored BFCL outputs (llada_json.py --stage baseline / compare on routes from bfcl_compile.py) with the official checker.

Per arm, format (python, json) and group (non-live, live): accuracy = share of returned outputs the checker accepts, averaged
over examples; for the unconstrained baseline also the share that satisfies the constraint. `--self-check` scores the reference
texts of the manifest instead of model outputs (should be near 1).
"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_score import score                                                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--routes", required=True); ap.add_argument("--results", nargs="*", default=[]); ap.add_argument("--self-check", dest="self_check", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); man = {str(r["index"]): r for r in json.load(open(a.routes + ".json"))["rows"]}
    acc = collections.defaultdict(list); cs = collections.defaultdict(list)
    def add(arm, r, outputs, matches=None):
        q = dict(function=r["function"]); y = dict(ground_truth=r["ground_truth"])
        key = (arm, r["format"], "live" if r["split"].startswith("live") else "non-live")
        acc[key].append(float(np.mean([score(t.split("<|")[0].strip(), r["format"], q, y, r["split"]) for t in outputs])) if outputs else 0.0)
        if matches is not None:
            cs[key].append(matches)
    if a.self_check:
        for r in man.values():
            add("reference text", r, [r["completion"]])
    for f in a.results:
        d = json.load(open(f))
        for k, v in d.get("baseline", {}).items():
            add("unconstrained, shipped setting", man[k], v["outputs"], v["matches"])
        for k, v in d.get("compare", {}).items():
            for arm, r in v.items():
                if isinstance(r, dict) and "outputs" in r:
                    add(f"{arm} | K={r.get('K')}", man[k], r["outputs"])
    out = {}
    for key in sorted(acc):
        v = np.array(acc[key]); out[" | ".join(key)] = dict(examples=len(v), accuracy=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                                                             constraint_satisfied=float(np.mean(cs[key])) if key in cs else None)
        r = out[" | ".join(key)]; print(f"{key[0]:52s} {key[1]:6s} {key[2]:8s} examples {r['examples']:4d}  accuracy {r['accuracy']:.3f}" + (f" ± {r['se']:.3f}" if r["se"] is not None else "")
                                        + (f"  constraint satisfied {r['constraint_satisfied']:.3f}" if r["constraint_satisfied"] is not None else ""))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
