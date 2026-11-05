#!/bin/bash
# Imperial Front Wing: STEP -> STL (once) -> OpenFOAM case -> mesh + solve in the pinned container -> report.
#
#   cd ~/Documents/GitHub/F1-Front-Wing-Physics-Informed-NN/course/_instructor/dataset3d
#   NP=6 bash run_ifw.sh smoke     # ~1 M cells: checks the chain (minutes)
#   NP=6 bash run_ifw.sh L0        # first look at the model (laptop: ~3-4 M cells, about 1-3 h)
#
# Options (env): NP=6, NITER (cap, default 3000; smoke 300), AUTOSTOP=1, STEP (default ../../../imperial/ifw.stp).
# Variants (ifw/deform_ifw.py), all 0 = the original CAD:
#   D_FLAP1 (deg, flap 1 + flap 2 about the flap-1 leading edge), D_FLAP2 (deg, flap 2 alone),
#   DZ_MM (ride height change), DPITCH (deg, pitch change about the footplate trailing edge).
#   SHAPE='{"camber_main": 1.5, "thick_flap1": 1.1}' (any mode of ifw/shape_modes.py, JSON)
#   e.g.  D_FLAP2=4 NP=6 bash run_ifw.sh L0
# The STL are generated once into ../../../imperial/stl/ (gmsh, about 5 min) and reused.
set -euo pipefail
cd "$(dirname "$0")"
LEVEL=${1:?usage: run_ifw.sh smoke|L0|L1}
NP=${NP:-6}
case $LEVEL in smoke) NITER=${NITER:-300};; L0|L1) NITER=${NITER:-3000};; *) echo "level: smoke|L0|L1"; exit 2;; esac
DS=../dataset
PY=$DS/.venv/bin/python
IMAGE=$(grep '^OPENFOAM_IMAGE=' $DS/pipeline_versions.env | cut -d= -f2)
ROOT=../../..
STEP=${STEP:-$ROOT/imperial/ifw.stp}
STL=$ROOT/imperial/stl
RUNS=$ROOT/data/frontwing_runs3d
D_FLAP1=${D_FLAP1:-0}; D_FLAP2=${D_FLAP2:-0}; DZ_MM=${DZ_MM:-0}; DPITCH=${DPITCH:-0}; SHAPE=${SHAPE:-"{}"}
TAG=""
[ "$D_FLAP1$D_FLAP2$DZ_MM$DPITCH" != "0000" ] && TAG="_f1${D_FLAP1}_f2${D_FLAP2}_dz${DZ_MM}_dp${DPITCH}"
if [ "$SHAPE" != "{}" ]; then
    TAG="${TAG}_s$(printf '%s' "$SHAPE" | sha1sum | cut -c1-8)"     # shape modes: short hash, full JSON in deform.json
fi
CASE=$RUNS/ifw${TAG}_${LEVEL}_np${NP}_$(date -u +%Y%m%dT%H%M%SZ)

[ -x "$PY" ] || { echo "missing $PY (the 2D pipeline venv)"; exit 1; }
podman image exists "$IMAGE" || { echo "image $IMAGE not present"; exit 1; }
# wing.stl travels in parts of 25 MB (file-size limit of the transfer): reassemble and verify once
if [ ! -f "$STL/wing.stl" ] && ls "$STL"/wing.stl.part* > /dev/null 2>&1; then
    cat "$STL"/wing.stl.part* > "$STL/wing.stl"
    (cd "$STL" && sha256sum -c SHA256SUMS) || { echo "STL checksum mismatch"; exit 1; }
fi
if [ ! -f "$STL/ifw_geometry.json" ] || [ ! -f "$STL/wing.stl" ]; then
    echo "== STEP -> STL ($STEP), about 5 min"
    $PY ifw/prepare_ifw.py "$STEP" "$STL"
fi
mkdir -p "$RUNS"
GEOM=$STL
if [ -n "$TAG" ]; then
    [ -f "$STL/wing_mesh.npz" ] && [ -f "$STL/ifw_elements.json" ] || { echo "variants need $STL/wing_mesh.npz and ifw_elements.json"; exit 1; }
    GEOM=$ROOT/imperial/variants/${TAG#_}
    echo "== variant $TAG -> $GEOM"
    $PY ifw/deform_ifw.py "$STL" "$STL/ifw_elements.json" "$GEOM" --d-flap1 "$D_FLAP1" --d-flap2 "$D_FLAP2" \
        --dz-mm "$DZ_MM" --dpitch-deg "$DPITCH" --params "$SHAPE" | tail -25
    $PY -c "import json,sys; sys.exit(0 if json.load(open('$GEOM/deform.json'))['valid'] else 1)" \
        || { echo "invalid variant (slot closed or ground contact): see $GEOM/deform.json"; exit 1; }
fi
echo "== case $CASE"
$PY ifw/case_ifw.py "$CASE" --stl "$GEOM" --level "$LEVEL" --np "$NP" --n-iter "$NITER"
[ -n "$TAG" ] && cp "$GEOM/deform.json" "$CASE/deform.json"
{
    echo "host $(hostname)"; echo "cpu $(lscpu | sed -n 's/^Model name: *//p')"; echo "nproc $(nproc)"
    echo "mem_total_GB $(free -g | awk '/^Mem/ {print $2}')"; echo "image $IMAGE"; echo "started_utc $(date -u +%FT%TZ)"
} > "$CASE/platform.txt"
if [ "${AUTOSTOP:-1}" = 1 ]; then
    $PY zerihan3d/autostop3d.py "$CASE" --watch > "$CASE/autostop.log" 2>&1 &
    WATCH=$!
    trap 'kill $WATCH 2>/dev/null' EXIT
fi
echo "== mesh + solve ($NP ranks); progress: tail -f $CASE/log.Allrun"
podman run --rm --network=none --init --name "fwifw_$$" \
    -v "$(realpath "$CASE")":/case -w /case \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    "$IMAGE" bash -c "cd /case && ./Allrun $NP" || echo "Allrun failed: see $CASE/log.*"
echo "finished_utc $(date -u +%FT%TZ)" >> "$CASE/platform.txt"
echo "== report"
$PY zerihan3d/report3d.py "$CASE" | tail -40
echo "report: $CASE/report3d.json, $CASE/report3d.png"
