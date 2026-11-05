#!/bin/bash
# 3D feasibility trial on the production desktop (WSL2 + rootless podman + pinned OpenFOAM v2606).
#
#   cd ~/frontwing/F1-Front-Wing-Physics-Informed-NN/course/_instructor/dataset3d
#   ./run_trial.sh smoke            # ~1 M cells: pipeline check, 300 iterations (minutes to ~1 h)
#   ./run_trial.sh L0               # the trial: ~2.5-3 M cells, up to 3000 iterations (hours)
#   ./run_trial.sh L1               # optional: ~8-10 M cells (overnight, needs ~16 GB RAM in WSL)
#
# Options (env): NP=6 (MPI ranks), H=0.211 (h/c, Zerihan definition), NITER (cap), NAME (case dir name).
# Never overwrites: every run is a new directory in ../../../data/frontwing_runs3d/.
set -euo pipefail
cd "$(dirname "$0")"
LEVEL=${1:?usage: ./run_trial.sh smoke|L0|L1}
NP=${NP:-6}
H=${H:-0.211}
case $LEVEL in smoke) NITER=${NITER:-300};; L0|L1) NITER=${NITER:-3000};; *) echo "level: smoke|L0|L1"; exit 2;; esac

DS=../dataset
PY=$DS/.venv/bin/python
IMAGE=$(grep '^OPENFOAM_IMAGE=' $DS/pipeline_versions.env | cut -d= -f2)
RUNS=../../../data/frontwing_runs3d
NAME=${NAME:-zerihan3d_${LEVEL}_h${H}_np${NP}_$(date -u +%Y%m%dT%H%M%SZ)}
CASE=$RUNS/$NAME

[ -x "$PY" ] || { echo "missing $PY (the 2D pipeline venv)"; exit 1; }
command -v podman > /dev/null || { echo "podman not found"; exit 1; }
podman image exists "$IMAGE" || { echo "image $IMAGE not present: podman pull it (see ../dataset/SETUP_DESKTOP.md)"; exit 1; }
[ -e "$RUNS/../frontwing_runs/v1/production.pid" ] && kill -0 "$(cat "$RUNS/../frontwing_runs/v1/production.pid")" 2> /dev/null \
    && { echo "the 2D production driver is running: do not start the 3D trial on top of it"; exit 1; }

mkdir -p "$RUNS"
echo "== case $CASE"
$PY zerihan3d/case3d.py "$CASE" --level "$LEVEL" --h "$H" --np "$NP" --n-iter "$NITER"
{
    echo "host $(hostname)"; echo "cpu $(lscpu | sed -n 's/^Model name: *//p')"; echo "nproc $(nproc)"
    echo "mem_total_GB $(free -g | awk '/^Mem/ {print $2}')"; echo "image $IMAGE"; echo "started_utc $(date -u +%FT%TZ)"
} > "$CASE/platform.txt"

echo "== mesh + solve in the container ($NP ranks); progress: tail -f $CASE/log.Allrun"
podman run --rm --network=none --init --name "fw3d_$$" \
    -v "$(realpath "$CASE")":/case -w /case \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    "$IMAGE" bash -c "cd /case && ./Allrun $NP" || echo "Allrun failed: see $CASE/log.*"

echo "finished_utc $(date -u +%FT%TZ)" >> "$CASE/platform.txt"
echo "== report"
$PY zerihan3d/report3d.py "$CASE" | tail -40
echo "report: $CASE/report3d.json, $CASE/report3d.png"
