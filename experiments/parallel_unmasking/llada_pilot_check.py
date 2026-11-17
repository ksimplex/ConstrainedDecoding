"""Score the unconstrained LLaDA pilot samples (llada_json.py --stage pilot) on JSON-Mode-Eval.

Per datapoint: how many samples parse as JSON as written, parse after keeping only the outermost {...},
validate against the datapoint's schema (jsonschema), and equal the reference object."""
import argparse, csv, json, re
import numpy as np, pandas as pd, jsonschema
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", required=True); ap.add_argument("--tokenizer", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--first-special", type=int, default=126080)
a = ap.parse_args()
tok = Tokenizer.from_file(a.tokenizer); df = pd.read_parquet(a.data); d = np.load(a.pilot)
cols = ["datapoint", "samples", "parse_as_written", "parse_after_stripping", "schema_valid", "equal_reference"]
rows = []
for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
    i = int(k.split("_")[1]); schema = json.loads(df.schema[i]); ref = json.loads(df.completion[i])
    c = dict.fromkeys(cols, 0); c["datapoint"] = i
    for r in d[k]:
        s = tok.decode([int(t) for t in r if int(t) < a.first_special]); c["samples"] += 1
        try:
            json.loads(s); c["parse_as_written"] += 1
        except Exception:
            pass
        m = re.search(r"\{.*\}", s, re.S)
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        c["parse_after_stripping"] += 1
        try:
            jsonschema.validate(obj, schema); c["schema_valid"] += 1
        except Exception:
            pass
        c["equal_reference"] += obj == ref
    rows.append(c)
tot = {q: sum(r[q] for r in rows) for q in cols}; tot["datapoint"] = "total"; rows.append(tot)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
print(tot)
