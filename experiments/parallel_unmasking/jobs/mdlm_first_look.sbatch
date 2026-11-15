#!/bin/bash
# DRAFT, NOT SUBMITTED: first look at parallel unmasking on MDLM (OpenWebText, 130M), one L40S GPU.
# Parts: pilot kernel, unconstrained audit (calls against co-reveal gap), order inconsistency, a small constrained
# comparison (random sequential against smart sequential and smart parallel). N = 128, then kernel + audit at N = 256.
# Expected: under 1 h. Results are written part by part; a rerun skips finished arms.
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 150
#SBATCH --exclude=node3006

PY=$HOME/orcd/pool/cfg-twister/envs/cfgtw/bin/python
tar --warning=no-unknown-keyword -xzf pu_code9.tar.gz
export PYTHONPATH=$PWD HF_HOME=$HOME/orcd/pool/cfg-twister/hf
echo "start $(date) job ${SLURM_JOB_ID:-none} node $(hostname)" | tee -a progress.txt
$PY -c "import torch, flash_attn; torch.ones(1, device='cuda'); print('cuda ok', torch.__version__, torch.cuda.get_device_name(0))" 2>&1 | tee -a progress.txt || { echo "CUDA CHECK FAILED" | tee -a progress.txt; exit 1; }
$PY experiments/parallel_unmasking/mdlm_first_look.py --synthetic --N 32 --traces 8 --pilot-reps 1 --audit 16 --rows 16 --nrun 8 --K 4 --out dry_run.json > dry_run.txt 2>&1 || { tail -30 dry_run.txt; echo "DRY RUN FAILED" | tee -a progress.txt; exit 1; }
echo "dry run ok $(date)" | tee -a progress.txt
$PY experiments/parallel_unmasking/mdlm_first_look.py --N 128 --traces 32 --pilot-reps 4 --audit 256 --rows 96 --nrun 48 --K 8 --out mdlm_N128.json 2>&1 | grep --line-buffered -v Warning | tee -a log_N128.txt
echo "N=128 done $(date)" | tee -a progress.txt
$PY experiments/parallel_unmasking/mdlm_first_look.py --N 256 --traces 16 --pilot-reps 4 --audit 128 --rows 48 --parts AB --out mdlm_N256.json 2>&1 | grep --line-buffered -v Warning | tee -a log_N256.txt
echo "N=256 done $(date)" | tee -a progress.txt
