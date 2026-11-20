"""Token classes of the JSON-Mode-Eval schemas under a real tokenizer (the setting of TWISTER, appendix F).

Each schema is compiled to a regular expression with Outlines (outlines_core), the expression to an automaton
(interegular), and every vocabulary token to its transition function on the automaton's states (json_classes.py).
Reported per schema: automaton states reachable at token boundaries, classes, live (state, class) edges, and the
work of one chain-twist step per position done densely (classes^2 x states) and over automaton edges only
(sum over live edges of the number of live classes at the edge's target state).
"""
import argparse, json, time
import numpy as np
import pandas as pd
import interegular
from tokenizers import Tokenizer
from outlines_core.json_schema import build_regex_from_schema


def token_codes(tok, V):
    strings = [tok.decode([i], skip_special_tokens=False) for i in range(V)]
    L = max(len(s) for s in strings); codes = np.full((V, L), -1, dtype=np.int64)
    for i, s in enumerate(strings):
        codes[i, :len(s)] = [ord(ch) for ch in s]
    return codes, np.array([len(s) for s in strings])


def classes_of(pattern, codes, lens):
    f = interegular.parse_pattern(pattern).to_fsm().reduce()
    states = sorted(f.states); idx = {s: k for k, s in enumerate(states)}; Q = len(states); C = len(f.alphabet.by_transition)
    delta = np.full((Q + 1, C + 1), Q, dtype=np.int32)                   # state Q is dead; symbol C = no such character class
    for s, row in f.map.items():
        for c, t in row.items():
            delta[idx[s], c] = idx[t]
    sym = f.alphabet; other = sym[interegular.fsm.anything_else] if interegular.fsm.anything_else in sym else C
    uniq = np.unique(codes[codes >= 0]); lut = np.array([sym[chr(u)] if chr(u) in sym else other for u in uniq], dtype=np.int64)
    V, L = codes.shape; T = np.tile(np.arange(Q, dtype=np.int32), (V, 1)); T[lens == 0] = Q
    for p in range(L):
        rows = np.nonzero(lens > p)[0]
        if len(rows) == 0:
            break
        c = lut[np.searchsorted(uniq, codes[rows, p])]
        T[rows] = delta[T[rows], c[:, None]]
    q0 = idx[f.initial]; reach = {q0}; frontier = [q0]
    while frontier:
        q = frontier.pop(); nxt = np.unique(T[:, q]); nxt = nxt[nxt < Q]
        for t in nxt.tolist():
            if t not in reach:
                reach.add(t); frontier.append(t)
    R = np.array(sorted(reach)); Tr = T[:, R]
    u, cnt = np.unique(Tr, axis=0, return_counts=True)                   # one row per class: target state from each reachable state
    live = u < Q; A = len(u); pos = {int(q): k for k, q in enumerate(R)}
    out_deg = live.sum(0)                                               # live classes at each reachable state
    edges = int(live.sum()); pairs = int(sum(out_deg[pos[int(t)]] for t in u[live].tolist()))
    return dict(char_states=Q, states=len(R), classes=A, edges=edges, edge_pairs=pairs, dense=int(A * A * len(R)), largest_class_share=float(cnt.max() / V),
                max_out_degree=int(out_deg.max()), mean_out_degree=float(out_deg.mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", required=True); ap.add_argument("--name", default="tokenizer")
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", default="json_mode_classes.json")
    a = ap.parse_args(); tok = Tokenizer.from_file(a.tokenizer); V = tok.get_vocab_size(with_added_tokens=False); codes, lens = token_codes(tok, V)
    df = pd.read_parquet(a.dataset); res = dict(tokenizer=a.name, vocab=V, rows=[], failed=[]); t0 = time.time()
    for i, row in df.iterrows():
        try:
            rx = build_regex_from_schema(row["schema"])
        except BaseException as e:
            res["failed"].append(dict(index=int(i), error=f"{type(e).__name__}: {str(e)[:80]}")); continue
        r = classes_of(rx, codes, lens); r.update(index=int(i), regex_length=len(rx), completion_tokens=len(tok.encode(row["completion"], add_special_tokens=False).ids)); res["rows"].append(r)
        if len(res["rows"]) % 10 == 0:
            print(f"[{time.time() - t0:5.0f}s] {len(res['rows'])} schemas; last: states {r['states']}, classes {r['classes']}, edges {r['edges']}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)
    q = lambda k: np.percentile([r[k] for r in res["rows"]], [0, 25, 50, 75, 100]).round(1).tolist()
    print(f"{a.name}: {len(res['rows'])} schemas compiled, {len(res['failed'])} failed")
    for k in ("completion_tokens", "states", "classes", "edges", "mean_out_degree", "max_out_degree", "edge_pairs", "dense", "largest_class_share"):
        print(f"  {k:20s} min / quartiles / max: {q(k)}")
    ratio = [r["dense"] / r["edge_pairs"] for r in res["rows"]]; print("  dense / sparse work: median %.0f, min %.0f, max %.0f" % (np.median(ratio), min(ratio), max(ratio)))


if __name__ == "__main__":
    main()
