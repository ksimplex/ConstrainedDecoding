"""Metrics of one arm against the exact constrained law, and the memory footprint of a worker."""
import math, time
import numpy as np
import torch


def metrics(out, den, truth, N, A, K, pair_truth, nb=10):
    logZ = out["logZ"]; nrun = len(logZ)
    lw = out["logw"]; w = np.exp(lw - lw.max(1, keepdims=True)); w = w / w.sum(1, keepdims=True)
    cls = out["classes"]
    r = den.log_joint(torch.tensor(out["tokens"]).view(-1, N)).numpy().reshape(lw.shape) - out["path_stat"]
    oh = np.eye(A)[cls]                                                   # (nrun, K, N, A)
    pair = oh[:, :, :-1, :, None] * oh[:, :, 1:, None, :] if pair_truth is not None else None
    zrel = np.exp(logZ - truth.log_pC)
    zs = np.sort(zrel)[::-1]; m = min(max(10, len(zs) // 50), len(zs) - 1)                  # upper tail of Z_hat over runs
    tail = dict(top1_share=float(zs[:max(1, len(zs) // 100)].sum() / zs.sum()),
                hill_k=float(np.mean(np.log(zs[:m] / zs[m]))) if zs[m] > 0 else float("nan"))

    def stat(idx):
        rw = (np.exp(logZ[idx] - logZ.max())[:, None] * w[idx]); rw = rw / rw.sum()
        rr = r[idx]; m = (rw * rr).sum(); var = (rw * (rr - m) ** 2).sum()
        e = np.exp(rr - rr.max()); cw = rw * e; kept = cw.sum() ** 2 / (rw * e * e).sum(); cw = cw / cw.sum()
        o = dict(steps=out["steps"][idx].mean(), z_ratio=zrel[idx].mean(), sd_r=math.sqrt(max(var, 0.0)), kl_r=var / 2,
                 mean_r=m, ess=(out["ess_min"][idx] / K).mean(), kept=kept, var_logZ=np.log(zrel[idx]).var())
        for tag, ww in (("", rw), ("_corr", cw)):
            mg = np.einsum("rk,rkna->na", ww, oh[idx])
            o["marg_tv" + tag] = 0.5 * np.abs(mg - truth.marginals).sum(-1).mean()
            if pair is not None:
                pm = np.einsum("rk,rknab->nab", ww, pair[idx])
                o["pair_tv" + tag] = 0.5 * np.abs(pm - pair_truth).sum((-1, -2)).mean()
        return o

    full = stat(np.arange(nrun))
    nb = min(nb, nrun)
    batches = np.array_split(np.arange(nrun), nb)
    jk = [stat(np.concatenate([b for j, b in enumerate(batches) if j != i])) for i in range(nb)]
    res = {}
    for k, v in full.items():
        vals = np.array([j[k] for j in jk])
        res[k] = float(v); res[k + "_se"] = float(math.sqrt((nb - 1) / nb * ((vals - vals.mean()) ** 2).sum()))
    res["z_ratio_se"] = float(zrel.std(ddof=1) / math.sqrt(nrun))         # runs are independent: the plain standard error
    res.update(tail)                                                      # is exact here and steadier than 10 batches
    res["z_se_unreliable"] = bool(not tail["hill_k"] <= 0.5)               # tail index above 1/2: no finite variance
    return res


class Footprint:
    """Peak memory footprint of this process in MB, sampled in a thread. On macOS the footprint counts compressed
    pages, which the peak resident size does not; without psutil it stays at -1."""
    def __init__(self, every=0.2):
        import threading
        self.peak, self._stop = -1.0, False
        try:
            import psutil
            self._p = psutil.Process()
        except Exception:
            return
        def watch():
            while not self._stop:
                try:
                    self.peak = max(self.peak, self._p.memory_full_info().uss / 1e6)
                except Exception:
                    return
                time.sleep(every)
        threading.Thread(target=watch, daemon=True).start()

    def stop(self):
        """End the sampling thread (one is started per cell; without this they accumulate in a long-lived process)."""
        self._stop = True
