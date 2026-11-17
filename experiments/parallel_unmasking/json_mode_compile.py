"""Compile JSON-Mode-Eval (NousResearch) into token-level routes for one tokenizer, once, on CPU.

For every datapoint whose schema Outlines turns into a regular expression (94 of 100; TWISTER, appendix F, reports the
same six failures): the chat messages (with the instruction TWISTER appends to the last user turn), the generation
length N = ceil(1.5 x the reference answer's length in tokens) as in TWISTER, the token classes and the automaton over
classes (twistroute.compile.tokens.lift_pattern, end tokens allowed after the JSON closes), and whether the reference
answer padded with end tokens is accepted. Written as one .npz (arrays) and one .json (manifest), so the GPU job needs
neither Outlines nor interegular.
"""
import argparse, json, math, time
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from outlines_core.json_schema import build_regex_from_schema
from twistroute.compile.tokens import lift_pattern, token_codes

INSTRUCTION = "Only output the JSON object, no other text or comments."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--V", type=int, default=126464); ap.add_argument("--n-text", dest="n_text", type=int, default=126080, help="ids below this are ordinary tokens")
    ap.add_argument("--end-ids", dest="end_ids", type=int, nargs="+", default=[126348, 126081]); ap.add_argument("--out", default="json_mode_routes")
    a = ap.parse_args(); tok = Tokenizer.from_file(a.tokenizer); t0 = time.time()
    strings = [tok.decode([i], skip_special_tokens=False) for i in range(a.n_text)] + [""] * (a.V - a.n_text)     # special tokens: dead unless end tokens
    codes = token_codes(strings); df = pd.read_parquet(a.dataset); arrays, manifest = {}, dict(args=vars(a), instruction=INSTRUCTION, rows=[], failed=[])
    for i, row in df.iterrows():
        try:
            rx = build_regex_from_schema(row["schema"])
        except BaseException as e:
            manifest["failed"].append(dict(index=int(i), error=f"{type(e).__name__}: {str(e)[:80]}")); continue
        cls, d, info = lift_pattern(rx, strings, end_ids=a.end_ids, codes=codes)
        msgs = [dict(role=str(m["role"]), content=str(m["content"])) for m in list(row["prompt"])]
        last = max(k for k, m in enumerate(msgs) if m["role"] == "user"); msgs[last]["content"] = msgs[last]["content"].rstrip() + "\n" + INSTRUCTION
        gold = tok.encode(row["completion"], add_special_tokens=False).ids; N = int(math.ceil(1.5 * len(gold)))
        padded = gold + [a.end_ids[0]] + [a.end_ids[-1]] * (N - len(gold) - 1)
        ok = bool(d.accepts([int(cls[t]) for t in padded]))
        arrays[f"cls_{i}"] = cls.astype(np.int16); arrays[f"edges_{i}"] = np.asarray(d.edges, dtype=np.int32); arrays[f"final_{i}"] = np.asarray(sorted(d.F), dtype=np.int32); arrays[f"gold_{i}"] = np.asarray(gold, dtype=np.int32)
        manifest["rows"].append(dict(index=int(i), N=N, gold_tokens=len(gold), states=d.n, classes=d.A, edges=len(d.edges), end_class=info["end_class"], gold_accepted=ok,
                                     regex=rx, schema=row["schema"], completion=row["completion"], messages=msgs))
        if len(manifest["rows"]) % 20 == 0:
            print(f"[{time.time() - t0:5.0f}s] {len(manifest['rows'])} compiled; gold accepted so far {sum(r['gold_accepted'] for r in manifest['rows'])}", flush=True)
    np.savez_compressed(a.out + ".npz", **arrays); json.dump(manifest, open(a.out + ".json", "w"), indent=1)
    r = manifest["rows"]; print(f"{len(r)} datapoints compiled, {len(manifest['failed'])} failed; reference answer accepted for {sum(x['gold_accepted'] for x in r)}; "
                                f"N median {int(np.median([x['N'] for x in r]))}, max {max(x['N'] for x in r)}; classes median {int(np.median([x['classes'] for x in r]))}; states median {int(np.median([x['states'] for x in r]))}")
    print("not accepted:", [(x["index"], x["completion"][:70]) for x in r if not x["gold_accepted"]][:6])


if __name__ == "__main__":
    main()
