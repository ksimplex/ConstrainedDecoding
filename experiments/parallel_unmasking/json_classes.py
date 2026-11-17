"""How many token classes does a JSON constraint have under a real tokenizer?

A constraint written as a regular expression over characters is compiled to an automaton (interegular), and every
vocabulary token is mapped to its transition function on the automaton's states (the state reached from each state
after reading the token's characters, or dead). Two tokens are in the same class iff their transition functions are
equal: the constraint cannot tell them apart. Reported: automaton states, classes, the share of tokens that are dead
everywhere, and the sizes of the largest classes. The chain twist costs about classes^2 x states per position and its
pair table has classes^2 entries; the product twist costs about classes x states.

Tokens are compared through their decoded strings; a token that is a fragment of a multi-byte character decodes to
U+FFFD and is treated as one ordinary character (an approximation that matters only inside strings).
"""
import argparse, json, time
import numpy as np
import interegular
from tokenizers import Tokenizer

WS = r"[ \n\t]*"
STR = r'"([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
INT = r"-?(0|[1-9][0-9]*)"
NUM = INT + r"(\.[0-9]+)?([eE][+-]?[0-9]+)?"
BOOL = r"(true|false)"


def obj(fields):
    body = (WS + "," + WS).join(f'"{k}"' + WS + ":" + WS + v for k, v in fields)
    return r"\{" + WS + body + WS + r"\}"


SCALAR = f"({STR}|{NUM}|{BOOL}|null)"
ARR = r"\[" + WS + f"({SCALAR}({WS},{WS}{SCALAR})*)?" + WS + r"\]"
PATTERNS = {
    "flat_schema": obj([("name", STR), ("age", INT), ("email", STR), ("is_member", BOOL)]),
    "nested_schema": obj([("id", INT), ("user", obj([("name", STR), ("tags", ARR)])), ("score", NUM), ("status", r'"(active|inactive|pending)"')]),
    "any_flat_object": r"\{" + WS + f"({STR}{WS}:{WS}({SCALAR}|{ARR})({WS},{WS}{STR}{WS}:{WS}({SCALAR}|{ARR}))*)?" + WS + r"\}",
}


def compile_fsm(pattern):
    f = interegular.parse_pattern(pattern).to_fsm().reduce()
    states = sorted(f.states); idx = {s: k for k, s in enumerate(states)}; Q = len(states); C = len(f.alphabet.by_transition)
    delta = np.full((Q + 1, C), Q, dtype=np.int32)                       # state Q is dead
    for s, row in f.map.items():
        for c, t in row.items():
            delta[idx[s], c] = idx[t]
    return f, delta, idx[f.initial], np.array([idx[s] for s in f.finals])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", required=True); ap.add_argument("--name", default="tokenizer"); ap.add_argument("--out", default="json_classes.json")
    a = ap.parse_args(); tok = Tokenizer.from_file(a.tokenizer); V = tok.get_vocab_size(with_added_tokens=False)
    strings = [tok.decode([i], skip_special_tokens=False) for i in range(V)]
    res = dict(tokenizer=a.name, vocab=V, patterns={})
    for pname, pat in PATTERNS.items():
        t0 = time.time(); f, delta, q0, fin = compile_fsm(pat); Q = delta.shape[0] - 1
        sym = f.alphabet; other = sym[interegular.fsm.anything_else] if interegular.fsm.anything_else in sym else None
        cid = lambda ch: sym[ch] if ch in sym else other
        T = np.empty((V, Q), dtype=np.int32); base = np.arange(Q, dtype=np.int32)
        for i, s in enumerate(strings):
            cur = base
            if not s:
                cur = np.full(Q, Q, dtype=np.int32)                     # empty string: a special token, never allowed here
            for ch in s:
                c = cid(ch)
                cur = delta[cur, c] if c is not None else np.full(Q, Q, dtype=np.int32)
            T[i] = cur
        # only states reachable at token boundaries matter
        reach = {q0}; frontier = [q0]
        live_cols = None
        while frontier:
            q = frontier.pop(); nxt = np.unique(T[:, q]); nxt = nxt[nxt < Q]
            for t in nxt.tolist():
                if t not in reach:
                    reach.add(t); frontier.append(t)
        R = sorted(reach); Tr = T[:, R]
        uniq, inv, cnt = np.unique(Tr, axis=0, return_inverse=True, return_counts=True)
        dead = int((Tr == Q).all(1).sum()); order = np.argsort(-cnt)
        res["patterns"][pname] = dict(char_states=Q, states_at_token_boundaries=len(R), classes=int(len(uniq)), dead_tokens=dead, dead_share=dead / V,
                                      largest_classes=[int(c) for c in cnt[order][:6]], singleton_classes=int((cnt == 1).sum()), seconds=round(time.time() - t0, 1))
        r = res["patterns"][pname]
        print(f"{a.name:8s} {pname:16s} char states {Q:4d}  states at token boundaries {len(R):4d}  classes {r['classes']:6d}  dead tokens {100 * r['dead_share']:.1f}%  largest classes {r['largest_classes']}  singletons {r['singleton_classes']}  {r['seconds']} s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
