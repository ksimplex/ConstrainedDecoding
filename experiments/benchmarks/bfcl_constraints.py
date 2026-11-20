"""Regular-expression constraints for BFCL function calls, in the two output formats of Dang & Ermon (2026).

  python   [name(arg=value, arg=value), name(...)]
  json     [{"name": "name", "arguments": {"arg": value, "arg": value}}, {...}]

The constraint fixes: the function name (one of those offered), the argument names (schema order; required ones
mandatory, optional ones may be left out), the type of every value (enumerated values listed literally), and the
surface syntax (one space after a comma or colon, as in the BFCL prompt). One or more calls. Free strings are
unrestricted apart from the closing quote and a newline; a backslash escapes the next character. Untyped or schema-less containers are
allowed one level of scalar content, since a regular language cannot nest without bound.

`--stats` checks, per split, how many reference answers the constraint accepts (first acceptable value of each
argument, arguments in schema order).
"""
import argparse, json, re

INT = r"-?(0|[1-9][0-9]*)"
FLOAT = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
NAMES = {"integer": "integer", "int": "integer", "float": "float", "number": "float", "double": "float", "boolean": "boolean", "bool": "boolean",
         "string": "string", "str": "string", "array": "array", "list": "array", "tuple": "array", "dict": "dict", "object": "dict", "any": "any"}


def esc(s):
    return re.sub(r"([\\.^$*+?{}\[\]()|/\-\"'])", r"\\\1", s)


def string(fmt):
    return r'"([^"\\\n]|\\.)*"' if fmt == "json" else r"""('([^'\\\n]|\\.)*'|"([^"\\\n]|\\.)*")"""


def literal(v, fmt):
    if isinstance(v, bool):
        return ("true" if v else "false") if fmt == "json" else ("True" if v else "False")
    if isinstance(v, (int, float)):
        return esc(repr(v))
    if v is None:
        return "null" if fmt == "json" else "None"
    v = esc(str(v))
    return f'"{v}"' if fmt == "json" else f"""('{v}'|"{v}")"""


def scalar(fmt):
    b = "(true|false)" if fmt == "json" else "(True|False)"
    return f"({string(fmt)}|{FLOAT}|{b})"


def seq(items, optional):
    """Regex for the items in order, comma-separated, where optional[i] items may be left out."""
    n = len(items)
    rest = ""                                                       # after at least one item has been written
    rests = [""] * (n + 1)
    for i in range(n - 1, -1, -1):
        rests[i] = (f"(, {items[i]})?" if optional[i] else f", {items[i]}") + rests[i + 1]
    alts = []
    for i in range(n):                                              # item i is the first one written
        alts.append(items[i] + rests[i + 1])
        if not optional[i]:
            break
    else:
        alts.append("")                                             # every item optional: nothing written
    return "(" + "|".join(alts) + ")"


def value(p, fmt, depth=0):
    p = p if isinstance(p, dict) else {}
    t = NAMES.get(str(p.get("type", "any")).lower(), "any")
    if p.get("enum"):
        return "(" + "|".join(literal(v, fmt) for v in p["enum"]) + ")"
    if t == "integer":
        return INT
    if t == "float":
        return FLOAT
    if t == "boolean":
        return "(true|false)" if fmt == "json" else "(True|False)"
    if t == "string":
        return string(fmt)
    if t == "array":
        it = value(p["items"], fmt, depth + 1) if isinstance(p.get("items"), dict) and depth < 2 else scalar(fmt)
        return rf"\[({it}(, {it})*)?\]"
    if t == "dict":
        props = p.get("properties")
        if isinstance(props, dict) and props and depth < 2:
            return r"\{" + members(p, fmt, depth + 1, sep=": ", quote=True) + r"\}"
        kv = f"{string(fmt)}: {scalar(fmt)}"
        return rf"\{{({kv}(, {kv})*)?\}}"
    return scalar(fmt) if depth else f"({scalar(fmt)}|" + rf"\[({scalar(fmt)}(, {scalar(fmt)})*)?\])"


def members(params, fmt, depth, sep, quote):
    props = params.get("properties") or {}; req = set(params.get("required") or [])
    def key(k):
        k = esc(k)
        return (f'"{k}"' if fmt == "json" else f"""('{k}'|"{k}")""") if quote else k
    return seq([key(k) + sep + value(v, fmt, depth) for k, v in props.items()], [k not in req for k in props])


def call_regex(functions, fmt):
    calls = []
    for f in functions:
        name = esc(f["name"]); par = f.get("parameters") or {}
        if fmt == "python":
            calls.append(name + r"\(" + members(par, fmt, 0, sep="=", quote=False) + r"\)")
        else:
            calls.append(r'\{"name": "' + name + r'", "arguments": \{' + members(par, fmt, 0, sep=": ", quote=True) + r"\}\}")
    c = "(" + "|".join(calls) + ")"
    return rf"\[{c}(, {c})*\]"


def unwrap(v):
    """BFCL stores a list of acceptable values at every level of a nested dictionary; keep the first of each."""
    if isinstance(v, dict):
        v = {k: (x[0] if isinstance(x, list) and x else x) for k, x in v.items()}
        return {k: unwrap(x) for k, x in v.items() if not (isinstance(x, str) and x == "")}        # "" marks an argument left out
    if isinstance(v, list):
        return [unwrap(x) for x in v]
    return v


def gold_text(functions, ground_truth, fmt):
    """Reference answer as text: first acceptable value of each argument, arguments in schema order."""
    by = {f["name"]: f for f in functions}
    dump = (lambda v: json.dumps(v, ensure_ascii=False)) if fmt == "json" else repr
    out = []
    for call in ground_truth:
        (name, args), = call.items()
        props = (by.get(name, {}).get("parameters") or {}).get("properties") or {}
        order = [k for k in props if k in args] + [k for k in args if k not in props]
        kept = [(k, unwrap(args[k][0])) for k in order if len(args[k]) and args[k][0] != ""]
        if fmt == "python":
            out.append(f"{name}(" + ", ".join(f"{k}={dump(v)}" for k, v in kept) + ")")
        else:
            out.append('{"name": "' + name + '", "arguments": {' + ", ".join(f'"{k}": {dump(v)}' for k, v in kept) + "}}")
    return "[" + ", ".join(out) + "]"


SPLITS = ["simple", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--stats", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); res = {}
    for c in SPLITS:
        q = load(f"{a.data}/BFCL_v3_{c}.json"); g = load(f"{a.data}/possible_answer/BFCL_v3_{c}.json")
        for fmt in ("python", "json"):
            ok, bad, size = 0, [], []
            for x, y in zip(q, g):
                rx = call_regex(x["function"], fmt); size.append(len(rx)); t = gold_text(x["function"], y["ground_truth"], fmt)
                try:
                    hit = re.fullmatch(rx, t) is not None
                except re.error as e:
                    hit = False; t = f"regex error {e}"
                ok += hit
                if not hit:
                    bad.append((x["id"], t[:150]))
            size.sort(); res[f"{c}|{fmt}"] = dict(n=len(q), accepted=ok, regex_chars_median=size[len(size) // 2], regex_chars_max=size[-1], rejected=bad[:400])
            print(f"{c:24s} {fmt:6s} reference accepted {ok:5d} of {len(q):5d}   regex length median {size[len(size) // 2]:6d} max {size[-1]:7d}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
