#!/bin/bash
# One IFW case of a plan on a cluster node: variant geometry -> OpenFOAM case -> mesh + solve in the .sif
# (with the autostop watcher) -> report. Cluster twin of ../run_ifw.sh (which uses podman on the laptop).
#
#   bash run_case_hpc.sh <plan.csv> <idx>            # idx = column idx of the plan (= SLURM_ARRAY_TASK_ID)
#   LEVEL=smoke bash run_case_hpc.sh oat_plan.csv 1  # quick check of the chain (~1 M cells, 300 iterations)
#
# Env: LEVEL (L0), NP (6), NITER (3000; smoke 300), CAMPAIGN (oat_v1), CLEAN (1: delete processor* after the
# report unless the plan row has keep_fields=1), FORCE (1: redo a case already done).
# Output: $FW3D_RUNS/$CAMPAIGN/<case_id>_<LEVEL>_np<NP>/ with STATUS = done | failed:<step> | invalid | running.
# Resume-safe: a case with STATUS=done is skipped; an unfinished one is moved aside and redone.
set -uo pipefail
. "$(dirname "$0")/lib.sh"
load_modules
PLAN=$(realpath "${1:?usage: run_case_hpc.sh <plan.csv> <idx>}")
IDX=${2:?idx}
LEVEL=${LEVEL:-L0}; NP=${NP:-6}; CAMPAIGN=${CAMPAIGN:-oat_v1}; CLEAN=${CLEAN:-1}
case $LEVEL in smoke) NITER=${NITER:-300};; L0|L1) NITER=${NITER:-3000};; *) echo "LEVEL smoke|L0|L1"; exit 2;; esac
[ -n "${FW3D_SIF:-}${FW3D_FOAM_BASHRC:-}" ] || { echo "run setup_cluster.sh first (no fw3d.env)"; exit 2; }

# --- the plan row -> shell variables (CASE_ID, PARAM, VALUE, SHAPE_JSON, DZ_MM, DPITCH_DEG, KEEP_FIELDS)
eval "$(py - "$PLAN" "$IDX" <<'EOF'
import csv, shlex, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["idx"] == sys.argv[2]]
if not rows:
    sys.exit(f"idx {sys.argv[2]} not in {sys.argv[1]}")
for k, v in rows[0].items():
    print(f"{k.upper()}={shlex.quote(v)}")
EOF
)"
[ -n "${CASE_ID:-}" ] || { echo "cannot read row $IDX of $PLAN"; exit 2; }

OUT=$FW3D_RUNS/$CAMPAIGN
CASE=$OUT/${CASE_ID}_${LEVEL}_np${NP}
mkdir -p "$OUT/logs"
if [ -f "$CASE/STATUS" ] && [ "$(cat "$CASE/STATUS")" = done ] && [ "${FORCE:-0}" != 1 ]; then
    echo "$CASE_ID: already done, skipped"; exit 0
fi
[ -e "$CASE" ] && mv "$CASE" "$CASE.old_$(date +%s)"
echo "== [$IDX] $CASE_ID  ($PARAM = $VALUE)  level $LEVEL, $NP ranks, host $(hostname), job ${SLURM_JOB_ID:-none}"

# --- geometry: the original STL, or a deformed variant built in a scratch dir (deleted after case setup)
GEOM=$FW3D_STL
if [ "$SHAPE_JSON" != "{}" ] || [ "$DZ_MM" != 0 ] || [ "$DPITCH_DEG" != 0 ]; then
    GEOM=$OUT/geom/$CASE_ID
    rm -rf "$GEOM"
    py "$KIT/ifw/deform_ifw.py" "$FW3D_STL" "$FW3D_STL/ifw_elements.json" "$GEOM" \
        --params "$SHAPE_JSON" --dz-mm "$DZ_MM" --dpitch-deg "$DPITCH_DEG" > "$OUT/logs/deform_$CASE_ID.json" 2>&1 \
        || { mkdir -p "$CASE"; cp "$OUT/logs/deform_$CASE_ID.json" "$CASE/"; echo failed:deform > "$CASE/STATUS"; echo "deform failed"; exit 1; }
    if ! py -c "import json, sys; sys.exit(0 if json.load(open('$GEOM/deform.json'))['valid'] else 1)"; then
        mkdir -p "$CASE"; cp "$GEOM/deform.json" "$CASE/"; echo invalid > "$CASE/STATUS"
        echo "invalid variant (see $CASE/deform.json): not run"; rm -rf "$GEOM"; exit 0
    fi
fi

# --- OpenFOAM case
py "$KIT/ifw/case_ifw.py" "$CASE" --stl "$GEOM" --level "$LEVEL" --np "$NP" --n-iter "$NITER" > /dev/null \
    || { mkdir -p "$CASE"; echo failed:case_ifw > "$CASE/STATUS"; exit 1; }
[ "$GEOM" != "$FW3D_STL" ] && { cp "$GEOM/deform.json" "$CASE/deform.json"; rm -rf "$GEOM"; }
echo running > "$CASE/STATUS"
{
    echo "case_id $CASE_ID"; echo "param $PARAM"; echo "value $VALUE"; echo "shape_json $SHAPE_JSON"
    echo "dz_mm $DZ_MM"; echo "dpitch_deg $DPITCH_DEG"
    echo "host $(hostname)"; echo "cpu $(lscpu | sed -n 's/^Model name: *//p' | head -1)"; echo "nproc $(env -u OMP_NUM_THREADS nproc)"
    echo "mem_total_GB $(free -g | awk '/^Mem/ {print $2}')"; echo "slurm_job ${SLURM_JOB_ID:-} ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"
    echo "foam $FW3D_FOAM ${FW3D_SIF:-$FW3D_FOAM_BASHRC}"; echo "started_utc $(date -u +%FT%TZ)"
} > "$CASE/platform.txt"

# --- autostop watcher next to the solver (stops at the regime: converged or steady oscillation)
py "$KIT/zerihan3d/autostop3d.py" "$CASE" --watch > "$CASE/autostop.log" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

echo "== mesh + solve; progress: tail -f $CASE/log.Allrun"
foam "$CASE" "./Allrun $NP"
RC=$?
echo "finished_utc $(date -u +%FT%TZ)" >> "$CASE/platform.txt"
kill $WATCH 2>/dev/null

echo "== report"
py "$KIT/zerihan3d/report3d.py" "$CASE" > "$CASE/report3d.log" 2>&1 || echo "report3d failed, see $CASE/report3d.log"
if [ $RC -eq 0 ] && grep -q "\] done" "$CASE/log.Allrun" 2>/dev/null; then
    echo done > "$CASE/STATUS"
else
    echo "failed:$(sed -n 's/^FAILED at \([^,]*\).*/\1/p' "$CASE/log.Allrun" 2>/dev/null | head -1)" > "$CASE/STATUS"
fi
# space: a L0 case is ~2 M cells; fields of all ranks ~1-2 GB. Keep logs, postProcessing, reports.
if [ "$CLEAN" = 1 ] && [ "$KEEP_FIELDS" != 1 ] && [ "$(cat "$CASE/STATUS")" = done ]; then
    rm -rf "$CASE"/processor* "$CASE"/constant/triSurface "$CASE"/constant/extendedFeatureEdgeMesh
fi
echo "== $CASE_ID: $(cat "$CASE/STATUS")  ($(du -sh "$CASE" | cut -f1))"
[ "$(cat "$CASE/STATUS")" = done ]
