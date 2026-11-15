#!/bin/bash
# Overnight local queue (laptop, 8 cores / 16 GB). Every compare.py stage resumes cell by cell and arm by arm.
ROOT="$(cd "$(dirname "$0")" && pwd)"; REPO="$ROOT/cd_repo_tmp"; OUT="$ROOT/overnight"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CMP="$REPO/experiments/parallel_unmasking/compare.py"; WT="$REPO/experiments/parallel_unmasking/weight_trust.py"
stage() { name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then echo "skip $name (done)" | tee -a "$OUT/progress.txt"; return; fi
  echo "start $name $(date '+%H:%M:%S')" | tee -a "$OUT/progress.txt"
  "$@" >> "$OUT/$name.log" 2>&1; rc=$?
  echo "end   $name $(date '+%H:%M:%S') rc=$rc" | tee -a "$OUT/progress.txt"
  [ $rc -eq 0 ] && touch "$OUT/$name.done"; }
CH="chain_0.6 chain_0.85 hidden_0.85"
stage s1_weight_trust python "$WT" --N 64 --n 4000 --workers 5 --out "$OUT/weight_trust_N64.json"
stage s2_thick_N256 python "$CMP" --N 256 --nrun 500 --K 4 --arms thick --models hidden_0.85 chain_0.85 chain_0.6 --constraints window --pilot 2000 --workers 3 --resume --out "$OUT/thick_N256.json"
stage s3_order_extra python "$CMP" --N 64 --nrun 2000 --arms order --order-theta 0.0 --only least_tie lex --workers 5 --resume --out "$OUT/order_N64_extra.json"
stage s4_interact_N256 python "$CMP" --N 256 --nrun 100 --K 16 --arms interact --models $CH repel_0.8 --constraints window two_window copy_spread copy_block --pilot 2000 --workers 5 --resume --out "$OUT/interact_N256.json"
stage s5_resample_N64 python "$CMP" --N 64 --nrun 1600 --K 4 --arms resample --models chain_0.85 hidden_0.85 --constraints forbidden counting copy dyck --pilot 2000 --workers 5 --resume --out "$OUT/resample_N64.json"
for sd in 0 1; do
stage s6_interact_dyck_N256_seed$sd python "$CMP" --N 256 --nrun 25 --K 16 --seed $sd --arms interact --models $CH repel_0.8 --constraints dyck --pilot 2000 --workers 3 --resume --out "$OUT/interact_dyck_N256_seed$sd.json"
done
stage s7_thick_N1024 python "$CMP" --N 1024 --nrun 100 --K 4 --arms thick --models hidden_0.85 chain_0.85 --constraints window --pilot 300 --workers 2 --resume --out "$OUT/thick_N1024.json"
stage s8_resample_N256 python "$CMP" --N 256 --nrun 800 --K 4 --resample-ks 4 16 64 --arms resample --models chain_0.85 hidden_0.85 --constraints window copy_spread --pilot 2000 --workers 4 --resume --out "$OUT/resample_N256.json"
stage s9_interact_N1024 python "$CMP" --N 1024 --nrun 50 --K 16 --arms interact --only separator sum weighted lex bound random least_tie --models chain_0.85 hidden_0.85 --constraints copy_spread two_window --pilot 300 --workers 2 --resume --out "$OUT/interact_N1024_local.json"
echo "queue finished $(date '+%H:%M:%S')" | tee -a "$OUT/progress.txt"
