#!/bin/bash
# PLANNED, NOT SUBMITTED: constraint-interaction study at 1,024 positions (needs approval before submission).
# 3 models x 4 constraints, 7 orders x (exact threshold + 2 budgets) + one-per-step control, resampling off, K = 16.
# Expected: about 2 h per cell on one core (22 arms at ~2 min, control ~1 h, pilot ~15 min); 12 workers -> ~2.5 h.
# Memory: 800 particles x 1,024 positions; request 64G as for the budget sweep (12 workers ran in that).
#SBATCH -p mit_preemptable
#SBATCH -t 480
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --requeue

PY=$HOME/orcd/pool/cfg-twister/envs/cfgtw/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
tar --warning=no-unknown-keyword -xzf pu_code8.tar.gz
export PYTHONPATH=$PWD
CMP=experiments/parallel_unmasking/compare.py
echo "start $(date)  job ${SLURM_JOB_ID:-none}  restarts ${SLURM_RESTART_COUNT:-0}" | tee -a progress.txt
if [ ! -f tests_ok ]; then
  $PY -m pytest -q -x -p no:cacheprovider tests/test_longchain.py tests/test_resampling.py > unit_tests.txt 2>&1 || { tail -30 unit_tests.txt; echo "UNIT TESTS FAILED" | tee -a progress.txt; exit 1; }
  tail -1 unit_tests.txt | tee -a progress.txt
  $PY $CMP --N 64 --models hidden_0.85 --constraints window copy_block --nrun 10 --K 16 --pilot 100 --arms interact --workers 2 --out dry_run.json > dry_run.txt 2>&1 || { tail -30 dry_run.txt; echo "DRY RUN FAILED" | tee -a progress.txt; exit 1; }
  echo "dry run ok $(date)" | tee -a progress.txt; touch tests_ok
fi
$PY $CMP --N 1024 --models chain_0.6 chain_0.85 hidden_0.85 --constraints window two_window copy_spread copy_block --nrun 50 --K 16 --pilot 1000 \
    --arms interact --workers 12 --resume --out interact_N1024.json 2>&1 | grep --line-buffered -v Warning | tee -a log_N1024.txt
echo "N=1024 done $(date)" | tee -a progress.txt
