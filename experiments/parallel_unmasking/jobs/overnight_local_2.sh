#!/bin/bash
# Revised local queue (after the first one was stopped at stage s5): two lanes, at most 5 workers in total.
ROOT="$(cd "$(dirname "$0")" && pwd)"; REPO="$ROOT/cd_repo_tmp"; OUT="$ROOT/overnight"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CMP="$REPO/experiments/parallel_unmasking/compare.py"
stage() { name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then echo "skip $name (done)" | tee -a "$OUT/progress.txt"; return; fi
  echo "start $name $(date '+%H:%M:%S')" | tee -a "$OUT/progress.txt"
  "$@" >> "$OUT/$name.log" 2>&1; rc=$?
  echo "end   $name $(date '+%H:%M:%S') rc=$rc" | tee -a "$OUT/progress.txt"
  [ $rc -eq 0 ] && touch "$OUT/$name.done"; }
CH="chain_0.6 chain_0.85 hidden_0.85"
MODES="off defensive plain gated systematic"; MODES2="power 0.5"
laneA() {
  stage s5a_resample_N64_local python "$CMP" --N 64 --nrun 1600 --K 4 --arms resample --models chain_0.85 hidden_0.85 --constraints forbidden copy --pilot 2000 --workers 2 --resume --out "$OUT/resample_N64.json"
  stage s7_thick_N1024 python "$CMP" --N 1024 --nrun 100 --K 4 --arms thick --models hidden_0.85 chain_0.85 --constraints window --pilot 300 --workers 2 --resume --out "$OUT/thick_N1024.json"
  stage s9_interact_N1024 python "$CMP" --N 1024 --nrun 50 --K 16 --arms interact --only separator sum weighted lex bound random least_tie --models chain_0.85 hidden_0.85 --constraints copy_spread two_window --pilot 300 --workers 2 --resume --out "$OUT/interact_N1024_local.json"
}
laneB() {
  stage s5b_resample_N64_global python "$CMP" --N 64 --nrun 800 --K 4 --resample-ks 4 16 64 --resample-modes off defensive plain gated systematic "power 0.5" --arms resample --models chain_0.85 hidden_0.85 --constraints counting dyck --pilot 2000 --workers 3 --resume --out "$OUT/resample_N64_global.json"
  for sd in 0 1; do
  stage s6_interact_dyck_N256_seed$sd python "$CMP" --N 256 --nrun 25 --K 16 --seed $sd --arms interact --models $CH repel_0.8 --constraints dyck --pilot 2000 --workers 3 --resume --out "$OUT/interact_dyck_N256_seed$sd.json"
  done
  stage s8_resample_N256 python "$CMP" --N 256 --nrun 800 --K 4 --resample-ks 4 16 64 --resample-modes off defensive plain gated systematic "power 0.5" --arms resample --models chain_0.85 hidden_0.85 --constraints window copy_spread --pilot 2000 --workers 3 --resume --out "$OUT/resample_N256.json"
}
laneA & laneB & wait
echo "queue 2 finished $(date '+%H:%M:%S')" | tee -a "$OUT/progress.txt"
