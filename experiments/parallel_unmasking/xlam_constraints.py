"""xLAM (Salesforce/xlam-function-calling-60k) in the form bfcl_constraints.py expects.

Each tool's flat parameter list with Python-style type strings ("str", "int, optional", "List[float]", ...) becomes a BFCL-style
function description, so the same call-format constraint and the same two output formats apply. The dataset has no list of
required arguments; `required` chooses the rule:
    "unmarked"   an argument is mandatory unless its type string says optional
    "none"       every argument may be left out
`--stats` draws the fixed-seed subset and reports how many reference calls each rule's constraint accepts.
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_constraints import call_regex                                    # noqa: E402

BASE = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "float", "number": "float", "bool": "boolean", "boolean": "boolean",
        "list": "array", "set": "array", "tuple": "array", "dict": "dict", "object": "dict", "callable": "string", "any": "any"}


def convert_type(t):
    """'List[Tuple[int, int]], optional' -> ({'type': 'array', 'items': {...}}, optional flag)."""
    t = str(t); opt = "optional" in t.lower()
    head = re.split(r",\s*(optional|default)", t, flags=re.I)[0].strip()
    def conv(x):
        x = x.strip(); m = re.match(r"^(\w+)\[(.*)\]$", x)
        if m:
            outer = BASE.get(m.group(1).lower(), "any"); inner = m.group(2)
            if outer == "array":
                first = re.split(r",(?![^\[]*\])", inner)[0]
                same = len({p.strip() for p in re.split(r",(?![^\[]*\])", inner)}) == 1
                return {"type": "array", "items": conv(first) if (m.group(1).lower() != "tuple" or same) else {"type": "any"}}
            return {"type": outer}
        return {"type": BASE.get(x.lower(), "any")}
    return conv(head), opt


def to_function(tool, required="unmarked"):
    props, req = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        p, opt = convert_type(spec.get("type", "any")); p["description"] = spec.get("description", ""); props[name] = p
        if required == "unmarked" and not opt:
            req.append(name)
    return {"name": tool["name"], "description": tool.get("description", ""), "parameters": {"type": "dict", "properties": props, "required": req}}


def reference_text(functions, answers, fmt):
    """The reference calls as text, arguments in schema order."""
    by = {f["name"]: f for f in functions}; dump = (lambda v: json.dumps(v, ensure_ascii=False)) if fmt == "json" else repr; out = []
    for a in answers:
        props = (by.get(a["name"], {}).get("parameters") or {}).get("properties") or {}; args = a["arguments"]
        kept = [(k, args[k]) for k in props if k in args] + [(k, v) for k, v in args.items() if k not in props]
        if fmt == "python":
            out.append(f"{a['name']}(" + ", ".join(f"{k}={dump(v)}" for k, v in kept) + ")")
        else:
            out.append('{"name": "' + a["name"] + '", "arguments": {' + ", ".join(f'"{k}": {dump(v)}' for k, v in kept) + "}}")
    return "[" + ", ".join(out) + "]"


def subset(data, n=1000, seed=0):
    idx = np.random.default_rng(seed).choice(len(data), size=n, replace=False)
    return [data[int(i)] for i in sorted(idx)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out")
    a = ap.parse_args(); rows = subset(json.load(open(a.data)), a.n, a.seed); res = {}
    for rule in ("unmarked", "none"):
        for fmt in ("python", "json"):
            ok, bad, why = 0, [], {"missing mandatory argument": 0, "other": 0}
            for r in rows:
                tools = json.loads(r["tools"]); ans = json.loads(r["answers"]); fs = [to_function(t, rule) for t in tools]; t = reference_text(fs, ans, fmt)
                hit = re.fullmatch(call_regex(fs, fmt), t) is not None; ok += hit
                if not hit:
                    by = {f["name"]: f for f in fs}
                    miss = any(k not in x["arguments"] for x in ans for k in by[x["name"]]["parameters"]["required"])
                    why["missing mandatory argument" if miss else "other"] += 1; bad.append((r["id"], t[:160]))
            res[f"{rule}|{fmt}"] = dict(n=len(rows), accepted=ok, reasons=why, rejected=bad[:60])
            print(f"mandatory rule {rule:9s} {fmt:6s} reference accepted {ok:4d} of {len(rows)}   rejected: {why}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
