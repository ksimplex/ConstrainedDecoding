"""E4 (results: e4_results.json; DONE 2026-10-13, job 545cda82): per-pass cost of every constraint route vs length, compile-size growth to refusal, denoiser-forward units.
Writes e4_results.json incrementally (a later failure does not lose earlier sections)."""
import json, time, inspect, traceback, itertools
import numpy as np, torch
from types import SimpleNamespace
from twistroute import compile as C
from twistroute.torch_routes import TorchChain, inside_log, cnf_from_inside, copy_logZ, det_logZ

# E4 was written against the pre-package kernel module; map its names onto the package.
RK = SimpleNamespace(ChainKernel=TorchChain, inside_log=inside_log, to_cnf=cnf_from_inside, copy_logZ=copy_logZ,
                     det_logZ=det_logZ, dfa_band=C.dfa_window_alldiff, dfa_dyck2_depth=C.dfa_dyck2_depth)

dev = "cuda" if torch.cuda.is_available() else "cpu"
R = {"device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
     "torch": torch.__version__, "routes": [], "compile": [], "denoiser": [], "errors": []}
LS, BS = [32, 64, 128, 256, 512], [1, 4, 16]
rng = np.random.default_rng(0)


def save():
    json.dump(R, open("e4_results.json", "w"), indent=1, default=float)


def timeit(fn, reps=3):
    fn()
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps): out = fn()
    if dev == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / reps * 1e3, out


def rand_logW(B, L, A):
    return torch.log_softmax(torch.randn(B, L, A, device=dev), -1)


def run(family, route, param, L, B, fn, A, compile_states=None, compile_s=None):
    try:
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        ms, z = timeit(lambda: fn(rand_logW(B, L, A)))
        row = dict(family=family, route=route, param=param, L=L, B=B, ms=round(ms, 3),
                   finite=bool(torch.isfinite(z).all()), compile_states=compile_states, compile_s=compile_s)
        if dev == "cuda": row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception as e:
        row = dict(family=family, route=route, param=param, L=L, B=B, error=repr(e)[:200])
    R["routes"].append(row); print(row, flush=True)
    return row


def build(fn):
    t = time.time()
    try:
        obj = fn(); return obj, round(time.time() - t, 2), None
    except C.CompilerRefuse as e:
        return None, round(time.time() - t, 2), str(e)

# ---------------------------------------------------------------- 1. chain families
fb, fs, _ = build(lambda: C.dfa_forbidden([tuple(rng.integers(0, 4, 4)) for _ in range(5)], 4))
K = RK.ChainKernel(fb, dev)
for L in LS:
    for B in BS: run("regular: 5 forbidden motifs", "chain", f"Q={fb.n},E={K.E}", L, B, K.logZ, 4, fb.n, fs)
for L in LS:
    ek, es, _ = build(lambda: C.dfa_exactly_k(0, L // 4, 4)); Kk = RK.ChainKernel(ek, dev)
    for B in BS: run("counting: exactly L/4", "chain", f"Q={ek.n},E={Kk.E}", L, B, Kk.logZ, 4, ek.n, es)
for w in (1, 2, 3, 4):
    bd, bs, _ = build(lambda: RK.dfa_band(w, 5)); Kb = RK.ChainKernel(bd, dev)
    for L in LS:
        for B in BS: run(f"width {w}: all-different in window {w + 1}", "chain", f"w={w},Q={bd.n},E={Kb.E}", L, B, Kb.logZ, 5, bd.n, bs)
save()

# ---------------------------------------------------------------- 2. copy: factor graph vs automaton
for L in LS:
    for B in BS: run("relational: copy k=L/2", "factor graph", "closed form", L, B, lambda W, k=L // 2: RK.copy_logZ(W, k), 4)
for k in range(2, 16):
    d, s, refused = build(lambda: C.dfa_copy_positional(k, 2))
    R["compile"].append(dict(family="copy as automaton", param=f"k={k}", states=None if d is None else d.n, seconds=s, refused=refused))
    if d is not None:
        Kc = RK.ChainKernel(d, dev)
        for B in BS: run("relational: copy k=L/2", "automaton", f"k={k},Q={d.n}", 2 * k, B, Kc.logZ, 2, d.n, s)
        for B in BS: run("relational: copy k=L/2 (A=2 ref)", "factor graph", "closed form", 2 * k, B, lambda W, k=k: RK.copy_logZ(W, k), 2)
    else:
        print("copy automaton refused at k =", k, refused, flush=True); break
save()

# ---------------------------------------------------------------- 3. Dyck-2: exact depth-bounded automaton vs chart
dyg = RK.to_cnf(C.Inside(*C.DYCK2)); dyg.to(dev)
DL = [8, 12, 16, 20, 24, 28, 32, 34, 36, 64, 128, 256, 512]
dfa_alive = True
for L in DL:
    if dfa_alive:
        d, s, refused = build(lambda: RK.dfa_dyck2_depth(L // 2))
        R["compile"].append(dict(family="Dyck-2 exact automaton (D=L/2)", param=f"L={L}", states=None if d is None else d.n,
                                 seconds=s, refused=refused))
        if d is None:
            dfa_alive = False; print("Dyck-2 automaton refused at L =", L, flush=True)
        else:
            Kd = RK.ChainKernel(d, dev)
            for B in BS: run("structural: Dyck-2", "automaton", f"D={L // 2},Q={d.n},E={Kd.E}", L, B, Kd.logZ, 4, d.n, s)
    for B in BS: run("structural: Dyck-2", "chart", f"R={len(C.DYCK2[1])}", L, B, lambda W: RK.inside_log(dyg, W)[0], 4)
    save()

# ---------------------------------------------------------------- 4. generic CFGs: chart only, growing grammar
for nN, nB in [(8, 16), (16, 64), (32, 256)]:
    st, bi, lx = C.random_cnf(nN, nB, 2 * nN, 4, seed=1)
    g = RK.to_cnf(C.Inside(st, bi, lx, 4)); g.to(dev)
    last = None
    for L in LS:
        if last is not None and last["ms"] * 8 > 25_000:
            R["routes"].append(dict(family=f"generic CFG N={nN}", route="chart", param=f"R={len(bi)}", L=L, B=16,
                                    skipped=f"predicted {last['ms'] * 8 / 1e3:.0f}s/pass")); continue
        for B in BS:
            row = run(f"generic CFG N={nN}", "chart", f"R={len(bi)}", L, B, lambda W: RK.inside_log(g, W)[0], 4)
        last = row if "ms" in row else None
    save()

# ---------------------------------------------------------------- 5. trees: determinant
for n in LS:
    for B in BS: run("trees: arborescence", "determinant", f"n={n}", n, B, RK.det_logZ, n + 1)
save()

# ---------------------------------------------------------------- 6. boundary: 3-CNF compile growth until refusal
def clause_dfa(vs, sg, L):
    def step(st, y):
        if st == ("acc",): return None
        pos, seen = st; nv = dict(seen)
        if pos in vs: nv[pos] = y
        if pos + 1 == L:
            return ("acc",) if any(nv.get(v) == s for v, s in zip(vs, sg)) else None
        return (pos + 1, tuple(sorted(nv.items())))
    return C.build_dfa((0, ()), step, lambda s: s == ("acc",), 2, max_states=10_000, name="clause")

Lc = 24; prod = None; t0 = time.time()
clauses = [(tuple(int(v) for v in rng.choice(Lc, 3, replace=False)), tuple(int(s) for s in rng.integers(0, 2, 3))) for _ in range(60)]
for m, (vs, sg) in enumerate(clauses, 1):
    try:
        d = clause_dfa(vs, sg, Lc)
        prod = d if prod is None else C.product_dfa(prod, d, 2, max_states=200_000)
        R["compile"].append(dict(family="3-CNF product", param=f"clauses={m},L={Lc}", states=prod.n, seconds=round(time.time() - t0, 1), refused=None))
    except C.CompilerRefuse as e:
        R["compile"].append(dict(family="3-CNF product", param=f"clauses={m},L={Lc}", states=None, seconds=round(time.time() - t0, 1), refused=str(e)))
        print("3-CNF refused at clauses =", m, flush=True); break
save()

# ---------------------------------------------------------------- 7. denoiser forward passes (the cost unit)
def bench_model(name, load, call):
    try:
        t = time.time(); m = load(); load_s = round(time.time() - t, 1)
        R["denoiser"].append(dict(model=name, load_s=load_s, signature=str(inspect.signature(m.forward))[:300]))
        for L in LS:
            for B in BS:
                try:
                    ms, _ = timeit(lambda: call(m, B, L))
                    R["denoiser"].append(dict(model=name, L=L, B=B, ms=round(ms, 3))); print(name, L, B, round(ms, 2), flush=True)
                except Exception as e:
                    R["denoiser"].append(dict(model=name, L=L, B=B, error=repr(e)[:200]))
        del m; torch.cuda.empty_cache()
    except Exception:
        R["errors"].append(dict(model=name, trace=traceback.format_exc()[-1500:]))
    save()

from transformers import AutoModel, AutoModelForMaskedLM

def owt_call(m, B, L):
    ids = torch.randint(0, 50257, (B, L), device=dev)
    kw = {}
    params = inspect.signature(m.forward).parameters
    for nm in ("timesteps", "sigma"):
        if nm in params: kw[nm] = torch.zeros(B, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m(input_ids=ids, **kw)

bench_model("OWT-130M MDLM",
            lambda: AutoModelForMaskedLM.from_pretrained("kuleshov-group/mdlm-owt", trust_remote_code=True).to(dev).eval(),
            owt_call)

def llada_call(m, B, L):
    ids = torch.randint(0, 126000, (B, L), device=dev)
    with torch.no_grad():
        return m(ids).logits

bench_model("LLaDA-8B-Instruct",
            lambda: AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
                                              torch_dtype=torch.bfloat16).to(dev).eval(),
            llada_call)
print("DONE", flush=True)
