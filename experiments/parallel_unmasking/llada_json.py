"""Constrained decoding of JSON-Mode-Eval on LLaDA-8B-Instruct: product against matched-chain proposal.

Routes come from json_mode_compile.py (.npz + .json). Stages, each resumable and written as it goes:

  smoke    a few datapoints: seconds per denoiser call, one unconstrained sample, one constrained decode with the
           product twist; the decoded text, whether it matches the pattern and parses.
  pilot    unconstrained samples of the model for every selected prompt, one token per step in uniformly random order
           (the reference process): the corpus the pair tables are collapsed from. Saved as token ids.
  compare  per datapoint, one token per step, K particles, nrun runs, for each order in --orders (uniformly random, or
           a CouplingPolicy priority such as `sum`, whose distance kernel is fitted per prompt from --traces
           unconstrained traces): product proposal and
           twist; chain proposal and twist with matched marginals and a defensive share of the product proposal, with
           the pair table (a) collapsed from the pilot corpus of the OTHER prompts (lam = 1) and (b) probed from the
           denoiser's own conditionals in this prompt (decode.chain_twist.probe_pair_kernel; its cost is recorded). Reported: mean and spread of
           log Z_hat over runs, relative standard deviation of Z_hat, validity of the returned sample, seconds.
           There is no exact p(C) here; arms of one datapoint estimate the same quantity.
           The table check printed with every datapoint compares the pooled table with the one collapsed from the
           reference answers of the other datapoints.

--synthetic replaces the model by a stand-in denoiser over the same vocabulary (a code check on CPU, not a result).
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twistroute.compile.automata import WDFA                                                       # noqa: E402
from twistroute.compile.tokens import pair_ratio_table                                             # noqa: E402
from twistroute.decode import twisted_smc, ChainTwist, probe_pair_kernel, CouplingPolicy, ConfidenceSchedule, fit_distance_kernel, COUNT_EDGES   # noqa: E402
from twistroute.decode.native import native_sample                                                 # noqa: E402
from twistroute.models.classes import TokenClasses                                                 # noqa: E402
from twistroute.torch_routes import torch_route                                                    # noqa: E402


class StandIn:
    """A denoiser over the real vocabulary that favours the reference answer; no context dependence."""
    def __init__(self, V, mask_id, end_ids, device="cpu"):
        self.V, self.mask_id, self.end_ids, self.device, self.calls, self.base, self.tok = V, mask_id, tuple(end_ids), device, 0, None, None
    def set_gold(self, gold, N, seed=0):
        g = torch.Generator().manual_seed(seed); self.base = torch.randn(N, self.V, generator=g) * 0.5
        pad = list(gold) + [self.end_ids[0]] + [self.end_ids[-1]] * (N - len(gold) - 1)
        self.base[torch.arange(N), torch.tensor(pad[:N])] += 9.0; self.base[:, self.mask_id] = float("-inf")
        self.base = torch.log_softmax(self.base, -1)
    def set_prompt(self, messages):
        return 0
    def logits(self, x):
        self.calls += 1; return self.base[None].expand(x.shape[0], -1, -1).clone()


def load_routes(path):
    man = json.load(open(path + ".json")); arr = np.load(path + ".npz"); return man, arr


def build(row, arr, V, mask_id, dev):
    i = row["index"]; cls = torch.as_tensor(arr[f"cls_{i}"].astype(np.int64))
    d = WDFA(row["states"], [tuple(int(v) for v in e) for e in arr[f"edges_{i}"]], 0, [int(q) for q in arr[f"final_{i}"]], row["classes"], name=f"json-{i}")
    full = torch.zeros(V, dtype=torch.long); full[:len(cls)] = cls; dead = int(cls[mask_id]) if mask_id < len(cls) else 0; full[len(cls):] = dead
    return TokenClasses(full, row["classes"], excluded=(mask_id,)).to(dev), d, torch_route(d, dev)


def decode_text(den, tokfile, ids, end_ids):
    ids = [int(t) for t in ids]; k = len(ids)
    while k > 0 and ids[k - 1] in end_ids:
        k -= 1
    return tokfile.decode(ids[:k], skip_special_tokens=False), all(t not in end_ids for t in ids[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True); ap.add_argument("--tokenizer", default=None, help="tokenizer.json for decoding text (optional)")
    ap.add_argument("--stage", choices=("smoke", "pilot", "baseline", "compare"), required=True); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--datapoints", type=int, nargs="*", default=None, help="dataset indices; default: all with N <= --max-n")
    ap.add_argument("--max-n", dest="max_n", type=int, default=160); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot-n", dest="pilot_n", type=int, default=8); ap.add_argument("--pilot", default="llada_pilot.npz")
    ap.add_argument("--nrun", type=int, default=16); ap.add_argument("--K", type=int, default=4); ap.add_argument("--runs-per-batch", dest="rpb", type=int, default=4)
    ap.add_argument("--probe-stride", dest="probe_stride", type=int, default=1); ap.add_argument("--probe-top", dest="probe_top", type=int, default=4)
    ap.add_argument("--tables", nargs="+", default=["counted", "probed"], choices=("counted", "probed"))
    ap.add_argument("--orders", nargs="+", default=["random", "sum"], choices=("random", "sum", "lex", "bound", "confident", "ltr", "token_confident", "block_confident"),
                    help="random = uniformly random, one token per step; the others are CouplingPolicy priorities, one token per step (confident = lowest entropy of the constrained class marginals first, "
                         "the confidence rule of Dang & Ermon at class level; ltr = left-most masked position first, after any forced positions; token_confident = highest top-token probability under the constrained marginals, "
                         "LLaDA's rule as in Dang & Ermon; block_confident = the same inside blocks of --block from the left, --per-step tokens per step: their decoding setting)")
    ap.add_argument("--block", type=int, default=32); ap.add_argument("--per-step", dest="per_step", type=int, default=2)
    ap.add_argument("--traces", type=int, default=8, help="unconstrained traces for the schedule's distance kernel (one decode of this batch per prompt)")
    ap.add_argument("--match", type=int, default=5); ap.add_argument("--mix", type=float, default=0.1); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-rows", dest="max_rows", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="llada_json.json")
    a = ap.parse_args(); man, arr = load_routes(a.routes); t00 = time.time()
    V, mask_id, end_ids = man["args"]["V"], 126336, tuple(man["args"]["end_ids"])
    rows = [r for r in man["rows"] if (a.datapoints is None and r["N"] <= a.max_n) or (a.datapoints is not None and r["index"] in a.datapoints)]
    rows = rows[:a.limit] if a.limit else rows
    tokfile = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tokfile = Tokenizer.from_file(a.tokenizer)
    if a.synthetic:
        dev = "cpu"; den = StandIn(V, mask_id, end_ids)
    else:
        from twistroute.models.llada import LLaDA
        dev = "cuda"; den = LLaDA(dev, max_rows=a.max_rows); assert den.V == V and den.mask_id == mask_id, (den.V, den.mask_id)
    G = lambda s: torch.Generator(device=dev).manual_seed(a.seed * 100003 + s)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res["args"] = vars(a); res.setdefault(a.stage, {})
    save = lambda: (json.dump(res, open(a.out + ".tmp", "w"), indent=1), os.replace(a.out + ".tmp", a.out))

    def prepare(r):
        den.set_gold(arr[f"gold_{r['index']}"].tolist(), r["N"], seed=r["index"]) if a.synthetic else den.set_prompt(r["messages"])
        return build(r, arr, V, mask_id, dev)

    def check(r, ids):
        text, clean = decode_text(den, tokfile, ids, end_ids) if tokfile is not None else ("", True)
        ok = None
        if tokfile is not None:
            ok = bool(clean and re.fullmatch(r["regex"], text) is not None)
            try:
                json.loads(text); parses = True
            except Exception:
                parses = False
            return text, ok, parses
        return text, ok, None

    if a.stage == "smoke":
        for r in rows:
            key = str(r["index"]); N = r["N"]
            if key in res["smoke"]:
                continue
            cl, d, route = prepare(r); x0 = torch.full((a.max_rows, N), mask_id, dtype=torch.long, device=dev)
            t0 = time.time(); lp = den.logits(x0); t_call = time.time() - t0
            t0 = time.time(); lp = den.logits(x0); t_call2 = time.time() - t0
            lw0 = cl.class_logw(lp[:1]); t0 = time.time(); z0 = float(route.logZ(lw0)[0]); t_route = time.time() - t0
            t0 = time.time(); _, toks = native_sample(den, cl, N, N, 2, G(1), return_tokens=True); t_native = time.time() - t0
            free_text, free_ok, free_parses = check(r, toks[0].tolist())
            t0 = time.time(); o = twisted_smc(den, cl, route, N, N, nrun=2, K=a.K, gen=G(2), return_particles=True); t_smc = time.time() - t0
            lw = o["logw"][0]; pick = int(np.argmax(lw)); text, ok, parses = check(r, o["tokens"][0, pick].tolist())
            res["smoke"][key] = dict(N=N, classes=r["classes"], states=r["states"], prefix_tokens=None if a.synthetic else int(den.prefix.numel()), rows=a.max_rows,
                                     seconds_first_call=round(t_call, 3), seconds_call=round(t_call2, 3), seconds_route_pass=round(t_route, 4), log_product_twist_at_start=z0,
                                     seconds_unconstrained_decode_2=round(t_native, 1), unconstrained_text=free_text[:400], unconstrained_matches=free_ok, unconstrained_parses=free_parses,
                                     seconds_constrained_2runs=round(t_smc, 1), logZ=[float(v) for v in o["logZ"]], constrained_text=text[:400], constrained_matches=ok, constrained_parses=parses,
                                     reference=r["completion"][:400])
            save(); s = res["smoke"][key]
            print(f"[{time.time() - t00:6.0f}s] datapoint {key}: N {N}, classes {r['classes']}, states {r['states']}; call {s['seconds_call']} s for {a.max_rows} rows; log twist at start {z0:.2f}; "
                  f"unconstrained matches {free_ok}; constrained matches {ok}, parses {parses}, log Z {s['logZ']}\n    unconstrained: {free_text[:160]!r}\n    constrained:   {text[:160]!r}\n    reference:     {r['completion'][:160]!r}", flush=True)

    if a.stage == "pilot":
        store = dict(np.load(a.pilot)) if os.path.exists(a.pilot) else {}
        for r in rows:
            key = f"pilot_{r['index']}"
            if key in store:
                continue
            cl, d, route = prepare(r); t0 = time.time()
            _, toks = native_sample(den, cl, r["N"], r["N"], a.pilot_n, G(10 + r["index"]), return_tokens=True)
            store[key] = toks.cpu().numpy().astype(np.int32); np.savez_compressed(a.pilot + ".tmp.npz", **store); os.replace(a.pilot + ".tmp.npz", a.pilot)
            texts = [check(r, t.tolist()) for t in toks[:a.pilot_n]]
            res["pilot"][str(r["index"])] = dict(N=r["N"], n=a.pilot_n, seconds=round(time.time() - t0, 1), matches=float(np.mean([bool(t[1]) for t in texts])) if tokfile is not None else None, example=texts[0][0][:300])
            save(); print(f"[{time.time() - t00:6.0f}s] pilot {r['index']}: {a.pilot_n} samples of {r['N']} tokens in {time.time() - t0:.0f} s; unconstrained samples matching the schema: {res['pilot'][str(r['index'])]['matches']}", flush=True)

    if a.stage == "baseline":
        for r in rows:
            key = str(r["index"])
            if key in res.setdefault("baseline", {}):
                continue
            cl, d, route = prepare(r); N = r["N"]; t0 = time.time(); c0 = den.calls
            _, toks = native_sample(den, cl, N, -(-N // a.per_step), a.pilot_n, G(20 + r["index"]), return_tokens=True, schedule=ConfidenceSchedule(a.per_step, a.block, "model"), route=route)
            chk = [check(r, t.tolist()) for t in toks]
            res["baseline"][key] = dict(N=N, n=len(chk), matches=float(np.mean([bool(c[1]) for c in chk])), outputs=[c[0] for c in chk], seconds=round(time.time() - t0, 1), denoiser_batches=den.calls - c0,
                                        setting=dict(block=a.block, per_step=a.per_step, rule="model's top-token probability", temperature=1))
            save(); print(f"[{time.time() - t00:6.0f}s] baseline {key}: {len(chk)} unconstrained samples of {N} tokens, blocks of {a.block}, {a.per_step} per step; matching the constraint {res['baseline'][key]['matches']:.2f}; "
                          f"{res['baseline'][key]['seconds']:.0f} s\n    {chk[0][0][:200]!r}", flush=True)

    if a.stage == "compare":
        counted = "counted" in a.tables
        if counted:
            store = dict(np.load(a.pilot)); pilot_rows = {int(k.split("_")[1]): v for k, v in store.items()}
            gold_rows = {x["index"]: np.asarray(arr[f"gold_{x['index']}"].tolist() + [end_ids[0]] + [end_ids[-1]] * (x["N"] - x["gold_tokens"] - 1)) for x in man["rows"] if x["gold_tokens"] < x["N"]}
        for r in rows:
            key = str(r["index"]); N = r["N"]; res["compare"].setdefault(key, {})
            cl, d, route = prepare(r); tc = cl.token_class.cpu().numpy()
            agree = float("nan")
            if counted:
                others = [s for i, v in pilot_rows.items() if i != r["index"] for s in v]
                Kt, tinfo = pair_ratio_table(others, tc, r["classes"], lam=a.lam)
                Kg, _ = pair_ratio_table([v for i, v in gold_rows.items() if i != r["index"]], tc, r["classes"], lam=a.lam)
                seen = np.array([tc[np.asarray(s)] for s in others[:1]]).size; rowsum = np.zeros(r["classes"])
                for s in others:
                    np.add.at(rowsum, tc[np.asarray(s)], 1.0)
                big = rowsum >= 50; lr = np.log(Kt[np.ix_(big, big)] / Kt[np.ix_(big, big)].mean(1, keepdims=True)); lg = np.log(Kg[np.ix_(big, big)] / Kg[np.ix_(big, big)].mean(1, keepdims=True))
                agree = float(np.corrcoef(lr.ravel(), lg.ravel())[0, 1]) if big.sum() >= 2 else float("nan")
                res["compare"][key]["table"] = dict(pilot_sequences=len(others), pairs=tinfo["pairs"], seen_entries=tinfo["seen_entries"], classes=r["classes"], frequent_classes=int(big.sum()),
                                                    agreement_with_reference_answers=agree, log10_min_entry=float(np.log10(Kt.min())), max_over_min_in_row=float((Kt.max(1) / Kt.min(1)).max()))
            arms = [("product", dict())]
            if counted:
                tw = ChainTwist(d, torch.as_tensor(Kt), device=dev, match=a.match)
                arms.append(("matched chain, counted table", dict(twist=tw, proposal_twist=tw, proposal_mix=a.mix)))
            if "probed" in a.tables:
                t0 = time.time(); c0 = den.calls
                Kp, pinfo = probe_pair_kernel(den, cl, N, G(5), stride=a.probe_stride, top=a.probe_top, rows=a.max_rows)
                pinfo.update(seconds=round(time.time() - t0, 1), denoiser_calls=den.calls - c0); res["compare"][key]["probe"] = pinfo
                tp = ChainTwist(d, Kp, device=dev, match=a.match)
                arms.append(("matched chain, probed table", dict(twist=tp, proposal_twist=tp, proposal_mix=a.mix)))
            kern = None
            if any(o in ("sum", "lex", "bound") for o in a.orders):
                t0 = time.time(); c0 = den.calls
                kern, _ = fit_distance_kernel(den, N, a.traces, G(7), device=dev, count_edges=COUNT_EDGES)
                res["compare"][key]["distance_kernel"] = dict(traces=a.traces, seconds=round(time.time() - t0, 1), denoiser_calls=den.calls - c0)
            sched = {"random": lambda: "fixed", "ltr": lambda: ConfidenceSchedule(1, None, "left", forced_first=True),
                     "confident": lambda: ConfidenceSchedule(1, None, "entropy", forced_first=True), "token_confident": lambda: ConfidenceSchedule(1, None, "token"),
                     "block_confident": lambda: ConfidenceSchedule(a.per_step, a.block, "token")}
            steps = {"block_confident": -(-N // a.per_step)}                    # every other order: one token per step
            for o in a.orders:
                if o in ("sum", "lex", "bound"):
                    sched[o] = lambda o=o: CouplingPolicy(kern, 0.0, priority=o, mixed=False, max_set=1)
            arms = [(f"{o} | {pn}", dict(kw, schedule=sched[o]())) for o in a.orders for pn, kw in arms]
            for name, kw in arms:
                if name in res["compare"][key]:
                    continue
                t0 = time.time(); lz, toks, lws, nres = [], [], [], []; c0 = den.calls
                try:
                    for b in range(0, a.nrun, a.rpb):
                        if "schedule" in kw and not isinstance(kw["schedule"], str):
                            kw = dict(kw, schedule=sched[name.split(" | ")[0]]())           # a fresh policy object per batch
                        o = twisted_smc(den, cl, route, N, steps.get(name.split(" | ")[0], N), nrun=min(a.rpb, a.nrun - b), K=a.K, gen=G(1000 + b), return_particles=True, **kw)
                        nres.append(float(np.mean(o["n_res"])) if "n_res" in o else float("nan"))
                        lz.append(np.asarray(o["logZ"], dtype=np.float64)); toks.append(o["tokens"]); lws.append(o["logw"])
                except Exception as e:
                    res["compare"][key][name + "|error"] = f"{type(e).__name__}: {str(e)[:300]}"; save(); print(f"[{time.time() - t00:6.0f}s] {key} {name} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
                lz = np.concatenate(lz); toks = np.concatenate(toks); lws = np.concatenate(lws); fin = np.isfinite(lz); z = np.exp(lz[fin] - lz[fin].max())
                picks = [toks[i, int(np.argmax(lws[i] + np.random.default_rng(i).gumbel(size=lws.shape[1])))] for i in range(len(toks)) if fin[i]]
                chk = [check(r, p.tolist()) for p in picks]
                def same_object(text):
                    try:
                        return json.loads(text) == json.loads(r["completion"])
                    except Exception:
                        return False
                res["compare"][key][name] = dict(N=N, n=int(fin.sum()), log_mean_Z=float(lz[fin].max() + math.log(z.mean())), mean_logZ=float(lz[fin].mean()), sd_logZ=float(lz[fin].std(ddof=1)),
                                                 spread=float(z.std(ddof=1) / z.mean()), matches=float(np.mean([bool(c[1]) for c in chk])) if tokfile is not None else None,
                                                 parses=float(np.mean([bool(c[2]) for c in chk])) if tokfile is not None else None, exact_reference=float(np.mean([c[0].strip() == r["completion"].strip() for c in chk])) if tokfile is not None else None,
                                                 seconds=round(time.time() - t0, 1), denoiser_batches=den.calls - c0, resamplings=float(np.nanmean(nres)) if nres else None,
                                                 logZ=[float(v) for v in lz], example=chk[0][0][:300] if chk else "", K=a.K,
                                                 equal_object=float(np.mean([same_object(c[0]) for c in chk])) if tokfile is not None and chk else None,
                                                 outputs=[c[0] for c in chk])
                save(); s = res["compare"][key][name]
                print(f"[{time.time() - t00:6.0f}s] {key:>3s} N {N:3d} A {r['classes']:3d} {name:38s} log Z {s['log_mean_Z']:8.2f}  mean log Z {s['mean_logZ']:8.2f}  sd log Z {s['sd_logZ']:.2f}  spread {s['spread']:.2f}  "
                      f"matches {s['matches']}  equals reference {s['exact_reference']} (as object {s['equal_object']})  {s['seconds']:.0f} s  (table agreement {agree:.2f})", flush=True)
    print(f"done in {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
