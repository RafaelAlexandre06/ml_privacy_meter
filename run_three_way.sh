#!/bin/bash
#SBATCH --job-name=mia_3way
#SBATCH --account=ldap_users
#SBATCH --partition=compute
#SBATCH --nodelist=lotus
#SBATCH --output=/mnt/storage/admindi/home/rafigueiredo/slurm_outputs/mia-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#
# Attack-grade three-way test (Hayes et al. 2024, App. D.2): one level-3 leg,
# neggrad_plus on cifar10, with audit.three_way on. Persists every reference's
# pre-unlearn checkpoint and fits the trained / unlearned / never worlds from
# references alone. One leg only, not a sweep.
#
#   sbatch run_three_way.sh
#
# Writes to a fresh log_dir (three_way_cifar10_neggrad_plus): a sweep dir made
# before audit.three_way has no base checkpoints and cannot be upgraded in
# place. Resumable like every other leg.
#
# The second line re-reads the finished run from cache and prints the
# three-way table into this job's output; the third builds the PDF for this
# one run. Neither trains anything. The PDF is separate from the sweep report
# on purpose: launch_sweep.sh report globs sweep_* and this dir is not one.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
ulimit -n 65535
ulimit -n

cd /mnt/storage/admindi/home/rafigueiredo/ml_privacy_meter

source venv/bin/activate

python3 run_urmia_online.py --cf configs/sweep/l3_cifar10_neggrad_plus_three_way.yaml
python3 three_way_attack.py three_way_cifar10_neggrad_plus
python3 make_report.py --dirs three_way_cifar10_neggrad_plus --out pdf_reports/three_way_cifar10_neggrad_plus.pdf
