#!/bin/bash
# SLURM job array: one task = one row of the plan = one IFW case (6 MPI ranks on one node).
# Submit from the hpc/ directory of the clone, AFTER setup_cluster.sh and a smoke test:
#
#   mkdir -p logs
#   sbatch --array=1-37%8 [-p <partition>] [-A <account>] [--qos <qos>] oat_array.sbatch oat_plan.csv
#
# %8 = at most 8 cases at the same time (tune to the allowed share). Per case on the laptop (L0, 6 ranks):
# 2.2 M cells, peak RSS 4.9 GB, about 1-1.5 h. The autostop usually ends the solver well before NITER.
# Partition, account and QoS are NOT set here: pass them on the command line (see probe.txt).
#SBATCH --job-name=ifw-oat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

set -uo pipefail
PLAN=${1:?plan csv}
HPC_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname), $(nproc) cpus, start $(date -Is)"
NP=${NP:-${SLURM_CPUS_PER_TASK:-6}} bash "$HPC_DIR/run_case_hpc.sh" "$HPC_DIR/$PLAN" "$SLURM_ARRAY_TASK_ID"
rc=$?
echo "end $(date -Is), rc $rc"
exit $rc
