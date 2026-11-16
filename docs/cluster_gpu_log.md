# GPU job incident log (MIT ORCD Engaging, `mit_normal_gpu`)

One row per GPU job that failed or behaved abnormally, and per diagnostic probe. Times are cluster local time
(US Eastern). "Job" is the first 8 characters of the app's job id; "Slurm" is the scheduler's id. Append a row for
every new incident: what was submitted, where it ran, what happened, the evidence, and what was concluded.
The evidence for a failed job is `slurm-<id>.out` in its working directory
(`~/.claude-science-scratch/.claude-science/jobs/<job id>/`) and `sacct -j <Slurm id>`.

| Date, start | Job | Slurm | Node | What was submitted | Outcome | Error and evidence | Conclusion |
|---|---|---|---|---|---|---|---|
| 2026-10-22 19:55 | (earlier session) | 26628762 | node3006 | GPU job from this project | FAILED after 0:31 | `RuntimeError: CUDA error: unknown error` at the first CUDA call (from the cluster notes) | Bad GPU node at the time; excluded with `--exclude=node3006` since. Not re-tested. |
| 2026-11-16 01:43 | 1fc965f6 | 29283420 | node2804 | MDLM step timing, `mdlm_timing.py` at `5c12f04`, N 128/256, 8/64/256 particles | FAILED (exit 1) after 1:54; 10 of 12 cells written | `torch.OutOfMemoryError`: 12.27 GiB requested with 35 GiB held, in `native_sample` at N = 256, 256 particles | Our script: an unchunked full-vocabulary log-softmax. Fixed in `7395e53`. Storage and GPU were fine on this node. |
| 2026-11-16 02:01 | 500498b5 | 29285277 | node4204 | Same timing at `7395e53`, with per-stage timers | TIMEOUT at 45:24 (reported as exit 143, 0 files) | Log contains only the cancellation line; the first command to print was the Python GPU check | Hung before any of our code ran. Cause identified later: pool directory unreachable from this node. |
| 2026-11-16 03:45 | 0ef5c662 | 29296067 | node3403 | Same, excluding node4204, GPU check under `timeout 300` | TIMEOUT at 30:13 (exit 143, 0 files) | Log: host and extraction lines at 03:45:03, then nothing; `import torch` never printed; the 300 s `timeout` (SIGTERM) did not end it | Not node4204-specific. A process blocked on storage ignores SIGTERM. |
| 2026-11-16 04:45 | 48847543 | 29300370 | node3502 | Same, start-up staged with `timeout -s KILL 120` per stage | FAILED (exit 137) after 2:00 | First stage, `ls -la ~/orcd/pool/cfg-twister/envs/cfgtw/bin/python`, killed at 120 s | The hang is access to the pool directory, before Python or the GPU. SIGKILL does end it. |
| 2026-11-16 04:57 | b26d8142 | 29301527 | node3203 | Storage probe (per-path `ls -d`, `stat`, `md5sum`, `df`, 20 s hard kill) | COMPLETED in 1 s | Every path instant; pool mounted from `hstor012-n2:/pool/005/ksanj` | Pool reachable from this GPU node. |
| 2026-11-16 04:57 | baad2f03 | 29301529 | node1618 (CPU) | Same probe on `mit_quicktest` | COMPLETED in 1 s | Every path instant | Pool reachable from this CPU node. |
| 2026-11-16 04:59 | c7c81c41 | 29301739 | node3005 | Timing with a 30 s hard-killed storage check first; node3006, 3502, 3403, 4204 excluded | FAILED (exit 1) after 0:30 | Guard message: `pool storage unreachable on node3005` | A fourth bad node. The guard works: the failure cost 30 s. |
| 2026-11-16 04:59 | edc21df2 | 29301714 | node3403 | Storage probe pinned to the node | COMPLETED; pool tests hung | `/orcd/pool` and `/orcd/pool/005` instant; `/orcd/pool/005/ksanj` hangs for `ls -d`, `stat`, `md5sum`, `df`, Python; home and scratch instant | Still bad 74 min after the first failure on this node. Only the user's pool directory is affected. |
| 2026-11-16 05:00 | e2f6bade | 29301690 | node3502 | Storage probe pinned to the node | COMPLETED; pool tests hung | Same pattern as node3403 | Still bad. |
| 2026-11-16 05:02 | 165e88ec | 29301723 | node4204 | Storage probe pinned to the node | COMPLETED; pool tests hung | `ls -d /orcd/pool/005/ksanj` instant, but `stat`, `md5sum`, `df` inside it and Python hang | Still bad three hours after the first failure; the directory appears mounted but unresponsive (looks like a stale mount). |
| 2026-11-16 05:08 | b41c1480 | 29302027 | node3203 | Timing at `7395e53`, pinned with `-w node3203`, storage check first | COMPLETED, 12 of 12 cells | none | Pinning to a node that just passed the probe works. |
| 2026-11-16 06:36 | b1ffcca2 | not recorded | node3203 | MDLM order consistency at `cc24eb5`, pinned with `-w node3203`, storage check first | COMPLETED in about 1 min | none | Pinned node still good. |
| 2026-11-16 17:42 | 8fabd084 | not recorded | node3203 | MDLM order consistency follow-up at `00121c0` (real text, lengths to 1,024), pinned, storage check first | COMPLETED in about 10 min, 6 of 6 files | none | Pinned node still good eleven hours later. Other nodes not re-probed. |

## Summary of the 2026-11-16 episode

- Symptom: any access to `/orcd/pool/005/ksanj` (NFS, `hstor012-n2:/pool/005/ksanj`, automounted) hangs on some GPU
  nodes: node4204, node3403, node3502, node3005. Fine on node2804 (01:43), node3203 (04:57, 05:08), CPU node1618 and
  the login node (login006).
- Home (`/home`) and scratch (`/orcd/scratch/orcd/014`) are fine on the bad nodes.
- Persistent per node for at least one to three hours. The same nodes ran this project's jobs normally in October
  (node3403 on 10-11, node3502 on 10-13).
- Nothing on our side changed: same environment path, partition and directives as the successful run on 2026-11-15
  17:29 (node4202). Quotas are far from full (home 4.9 of 195 GB, pool 38 GB of 1 TB) and no file in the environment
  was modified since before 2026-11-15.
- Underlying cause not known; reported to the user for ORCD support. Probe outputs are saved as artifacts
  (`storage_probe_node*.txt`).

## Guard to put first in every GPU job while this lasts

```
echo "[$(date +%H:%M:%S)] host $(hostname)"
timeout -s KILL 30 ls -la $HOME/orcd/pool/cfg-twister/envs/cfgtw/bin/python \
  || { echo "[$(date +%H:%M:%S)] pool storage unreachable on $(hostname)"; exit 1; }
```

Earlier failed GPU jobs appear in the scheduler history (`sacct -u $USER -S 2026-10-01 -r mit_normal_gpu -X`); their
causes were not traced for this log.
