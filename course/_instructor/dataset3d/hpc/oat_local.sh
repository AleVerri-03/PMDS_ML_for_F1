#!/bin/bash
# Fallback WITHOUT a scheduler (a shared compute server you may load directly). NEVER on a login node.
# Runs the rows of the plan, PAR cases at a time, each on NP ranks:
#
#   nohup bash oat_local.sh oat_plan.csv [first] [last] > oat_local.log 2>&1 &
#
# PAR defaults to (free cores) / NP, at least 1; set PAR explicitly on shared machines.
set -uo pipefail
cd "$(dirname "$0")"
PLAN=${1:?plan csv}
N=$(($(wc -l < "$PLAN") - 1))
FIRST=${2:-1}; LAST=${3:-$N}
NP=${NP:-6}
PAR=${PAR:-$(( $(nproc) / NP ))}; [ "$PAR" -ge 1 ] || PAR=1
echo "rows $FIRST-$LAST, $PAR at a time, $NP ranks each, start $(date -Is)"
mkdir -p logs
seq "$FIRST" "$LAST" | xargs -P "$PAR" -I{} sh -c "NP=$NP bash ./run_case_hpc.sh '$PLAN' {} > logs/local_{}.out 2>&1; echo \"row {}: rc \$?\""
echo "end $(date -Is)"
