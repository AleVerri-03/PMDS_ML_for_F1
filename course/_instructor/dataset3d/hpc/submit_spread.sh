#!/bin/bash
# Spread the rows of a plan over several nodes: one PBS array per node, at most PER cases running on each.
# Reason (oat_v1, 11-13/11/2026): PBS packed every job on cpu01; with 6-8 cases on one node a case runs at
# 5-9 s/iteration instead of 3.8 (memory bandwidth), while cpu02-05 stayed free.
#
#   DRY=1 bash submit_spread.sh ds_plan.csv 1-32 "cpu01 cpu02 cpu03 cpu04 cpu05" 4 ds_v1   # print the qsub only
#   bash submit_spread.sh ds_plan.csv 1-32 "cpu01 cpu02 cpu03 cpu04 cpu05" 4 ds_v1
#
# Host names as PBS knows them: pbsnodes -a | grep -E "^[a-z]|resources_available.host"
# Extra variables for run_case_hpc.sh: EXTRA_V, e.g. EXTRA_V=CLEAN=0 keeps the fields of every case.
# Walltime per case: WALLTIME (default 16:00:00; a case capped at 3000 iterations on a loaded node ~8 h).
# Rows are dealt round-robin; the lists are in rows/<campaign>.<host>.txt. Re-submitting the same range is
# safe: rows already done are skipped by run_case_hpc.sh.
set -euo pipefail
cd "$(dirname "$0")"
PLAN=${1:?plan csv}; RANGE=${2:?first-last}; read -r -a HOSTS <<< "${3:?\"host1 host2 ...\"}"; PER=${4:?cases per node}
CAMP=${5:-ds_v1}
i0=${RANGE%-*}; i1=${RANGE#*-}
mkdir -p rows
for h in "${HOSTS[@]}"; do : > "rows/$CAMP.$h.txt"; done
k=0
for ((r = i0; r <= i1; r++)); do echo "$r" >> "rows/$CAMP.${HOSTS[$((k % ${#HOSTS[@]}))]}.txt"; k=$((k + 1)); done
for h in "${HOSTS[@]}"; do
    f=rows/$CAMP.$h.txt; n=$(wc -l < "$f")
    [ "$n" -ge 1 ] || continue
    sel="select=1:ncpus=6:mem=16gb:host=$h"
    if [ "$n" -ge 2 ]; then
        cmd=(qsub -J "1-$n%$PER" -v "PLAN=$PLAN,CAMPAIGN=$CAMP,ROWS_FILE=$f${EXTRA_V:+,$EXTRA_V}" -l "$sel" -l "walltime=${WALLTIME:-16:00:00}" oat_array.pbs)
    else
        cmd=(qsub -v "PLAN=$PLAN,CAMPAIGN=$CAMP,IDX=$(cat "$f")${EXTRA_V:+,$EXTRA_V}" -l "$sel" -l "walltime=${WALLTIME:-16:00:00}" oat_array.pbs)
    fi
    echo "$h: $n rows ($(head -1 "$f")..$(tail -1 "$f")): ${cmd[*]}"
    [ "${DRY:-0}" = 1 ] || "${cmd[@]}"
done
