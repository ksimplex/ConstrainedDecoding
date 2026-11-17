"""Compile a sample of BFCL single-turn examples into token-level routes for one tokenizer (CPU), and report sizes.

Per example and output format (python, json): token classes and the automaton over classes for the call-format
constraint of bfcl_constraints.py (end tokens allowed after the call closes), at a fixed generation length (256 as in
Dang & Ermon, 2026), and whether the reference answer padded with end tokens is accepted. With --routes, also the chat
messages (BFCL default system prompt with the function list) and everything the GPU job and the scorer need.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tokenizers import Tokenizer
from twistroute.compile.tokens import lift_pattern, token_codes            # noqa: E402
from bfcl_constraints import call_regex, gold_text, load, SPLITS          # noqa: E402

# The BFCL default system prompt (bfcl-eval, constants/default_prompts.py), as used by Dang & Ermon (2026, Table 6).
SYSTEM = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of {call_format}
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""
CALL_FORMAT = dict(python="[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]",
                   json='a JSON array: [{"name": "func_name1", "arguments": {"params_name1": params_value1, ...}}, ...]')     # the JSON wording is Dang & Ermon's


def messages(question, functions, fmt):
    """Chat messages for one example: the system prompt with the function list, then the example's own turns."""
    turns = [dict(role=str(m["role"]), content=str(m["content"])) for m in question[0]]
    system = SYSTEM.format(call_format=CALL_FORMAT[fmt], functions=json.dumps(functions, indent=4))
    if turns and turns[0]["role"] == "system":
        turns[0]["content"] = system + "\n\n" + turns[0]["content"]; return turns
    return [dict(role="system", content=system)] + turns


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--V", type=int, default=126464); ap.add_argument("--n-text", dest="n_text", type=int, default=126080)
    ap.add_argument("--end-ids", dest="end_ids", type=int, nargs="+", default=[126348, 126081]); ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--per-split", dest="per", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--routes", default=None, help="write token classes, automata, prompts and references to <routes>.npz and <routes>.json for the GPU job")
    a = ap.parse_args(); tok = Tokenizer.from_file(a.tokenizer); rng = np.random.default_rng(a.seed); t00 = time.time()
    strings = [tok.decode([i], skip_special_tokens=False) for i in range(a.n_text)] + [""] * (a.V - a.n_text)
    codes = token_codes(strings); rows = []; arrays = {}; manifest = dict(args=vars(a), rows=[])
    for c in SPLITS:
        q = load(f"{a.data}/BFCL_v3_{c}.json"); g = load(f"{a.data}/possible_answer/BFCL_v3_{c}.json")
        for i in sorted(rng.choice(len(q), size=min(a.per, len(q)), replace=False).tolist()):
            for fmt in ("python", "json"):
                rx = call_regex(q[i]["function"], fmt); t0 = time.time()
                try:
                    cls, d, info = lift_pattern(rx, strings, end_ids=a.end_ids, codes=codes)
                except Exception as e:
                    rows.append(dict(split=c, id=q[i]["id"], format=fmt, error=f"{type(e).__name__}: {str(e)[:200]}")); print(rows[-1], flush=True); continue
                gold = tok.encode(gold_text(q[i]["function"], g[i]["ground_truth"], fmt), add_special_tokens=False).ids
                ok = None
                if len(gold) < a.N:
                    ok = bool(d.accepts([int(cls[t]) for t in gold + [a.end_ids[0]] + [a.end_ids[-1]] * (a.N - len(gold) - 1)]))
                sizes = np.bincount(cls.astype(np.int64), minlength=d.A)
                rows.append(dict(split=c, id=q[i]["id"], format=fmt, functions=len(q[i]["function"]), regex_chars=len(rx), gold_tokens=len(gold), states=int(d.n), classes=int(d.A),
                                 edges=len(d.edges), largest_class_share=float(sizes.max() / a.n_text), gold_accepted=ok, seconds=round(time.time() - t0, 1)))
                if a.routes:
                    k = len(manifest["rows"]); msgs = messages(q[i]["question"], q[i]["function"], fmt)
                    arrays[f"cls_{k}"] = cls.astype(np.int16); arrays[f"edges_{k}"] = np.asarray(d.edges, dtype=np.int32); arrays[f"final_{k}"] = np.asarray(sorted(d.F), dtype=np.int32)
                    arrays[f"gold_{k}"] = np.asarray(gold, dtype=np.int32)
                    manifest["rows"].append(dict(index=k, id=q[i]["id"], split=c, format=fmt, N=a.N, gold_tokens=len(gold), states=int(d.n), classes=int(d.A), edges=len(d.edges), end_class=info["end_class"],
                                                 gold_accepted=ok, regex=rx, completion=gold_text(q[i]["function"], g[i]["ground_truth"], fmt), messages=msgs, function=q[i]["function"],
                                                 ground_truth=g[i]["ground_truth"], prompt_tokens=sum(len(tok.encode(m["content"], add_special_tokens=False).ids) for m in msgs)))
                    if k % 8 == 7:
                        np.savez_compressed(a.routes + ".tmp.npz", **arrays); os.replace(a.routes + ".tmp.npz", a.routes + ".npz"); json.dump(manifest, open(a.routes + ".json", "w"), indent=1)
                r = rows[-1]; print(f"[{time.time() - t00:5.0f}s] {r['id']:28s} {fmt:6s} functions {r['functions']:2d} gold {r['gold_tokens']:3d} states {r['states']:5d} classes {r['classes']:4d} edges {r['edges']:6d} "
                                    f"accepted {ok} {r['seconds']} s", flush=True)
                json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"), indent=1)
    if a.routes:
        np.savez_compressed(a.routes + ".tmp.npz", **arrays); os.replace(a.routes + ".tmp.npz", a.routes + ".npz"); json.dump(manifest, open(a.routes + ".json", "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    for fmt in ("python", "json"):
        s = [r for r in ok if r["format"] == fmt]
        if s:
            print(f"{fmt}: {len(s)} compiled; states median {int(np.median([r['states'] for r in s]))} max {max(r['states'] for r in s)}; classes median {int(np.median([r['classes'] for r in s]))} "
                  f"max {max(r['classes'] for r in s)}; edges median {int(np.median([r['edges'] for r in s]))} max {max(r['edges'] for r in s)}; reference accepted {sum(bool(r['gold_accepted']) for r in s)}; "
                  f"seconds median {np.median([r['seconds'] for r in s]):.1f} max {max(r['seconds'] for r in s):.1f}")
    print("errors:", len(rows) - len(ok))


if __name__ == "__main__":
    main()
