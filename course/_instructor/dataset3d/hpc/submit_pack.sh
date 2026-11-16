#!/bin/bash
# Submit a range of rows as packed jobs (pack.pbs): rows dealt round-robin over the hosts, cut into blocks of
# ROWS_PER_JOB rows; one PBS array per host, one block running at a time on each host (%1), so the user limit
# of 4 running jobs puts at most one packed job (SLOTS cases) on each of 4 nodes.
#
#   DRY=1 bash submit_pack.sh ds_plan.csv 33-40 "cpu02 cpu03" 4 4 ds_v1     # print the qsub only
#   bash submit_pack.sh ds_plan.csv 33-1024 "cpu01 cpu02 cpu03 cpu04 cpu05" 16 4 ds_v1
#
# Args: plan, first-last, hosts, ROWS_PER_JOB, SLOTS (cases at once per job), campaign.
# Env: EXTRA_V (e.g. CLEAN=0), WALLTIME (default 16:00:00: 16 rows / 4 slots ~ 7-9 h).
# Re-submitting the same range is safe: rows already done are skipped by run_case_hpc.sh.
set -euo pipefail
cd "$(dirname "$0")"
PLAN=${1:?plan}; RANGE=${2:?first-last}; read -r -a HOSTS <<< "${3:?hosts}"; RPJ=${4:?rows per job}; SLOTS=${5:?slots}
CAMP=${6:-ds_v1}
i0=${RANGE%-*}; i1=${RANGE#*-}
mkdir -p rows
for h in "${HOSTS[@]}"; do rm -f "rows/$CAMP.$h".blk*.txt; : > "rows/$CAMP.$h.all"; done
k=0
for ((r = i0; r <= i1; r++)); do echo "$r" >> "rows/$CAMP.${HOSTS[$((k % ${#HOSTS[@]}))]}.all"; k=$((k + 1)); done
for h in "${HOSTS[@]}"; do
    split -l "$RPJ" -d -a 3 --additional-suffix=.tmp "rows/$CAMP.$h.all" "rows/$CAMP.$h.blk"
    n=0
    for f in rows/"$CAMP.$h".blk*.tmp; do [ -e "$f" ] || continue; n=$((n + 1)); mv "$f" "rows/$CAMP.$h.blk$n.txt"; done
    [ "$n" -ge 1 ] || continue
    sel="select=1:ncpus=$((6 * SLOTS)):mem=$((8 * SLOTS))gb:host=$h"
    v="PLAN=$PLAN,CAMPAIGN=$CAMP,SLOTS=$SLOTS,ROWS_PREFIX=rows/$CAMP.$h${EXTRA_V:+,$EXTRA_V}"
    if [ "$n" -ge 2 ]; then cmd=(qsub -J "1-$n%1" -v "$v" -l "$sel" -l "walltime=${WALLTIME:-16:00:00}" pack.pbs)
    else cmd=(qsub -v "$v,BLK=1" -l "$sel" -l "walltime=${WALLTIME:-16:00:00}" pack.pbs); fi
    echo "$h: $(wc -l < "rows/$CAMP.$h.all") rows in $n blocks: ${cmd[*]}"
    [ "${DRY:-0}" = 1 ] || "${cmd[@]}"
done
