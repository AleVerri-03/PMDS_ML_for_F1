#!/bin/bash
# One-time setup of the IFW 3D kit on a cluster, from the LOGIN node. Idempotent: re-run any step any time.
#
#   bash setup_cluster.sh probe       # 1. report of the machine -> probe.txt (changes nothing)
#   bash setup_cluster.sh container   # 2. OpenFOAM v2606 .sif   (route A: build with python; B: pull bare image)
#   bash setup_cluster.sh python      # 3. python side           (inside the .sif, or a host venv)
#   bash setup_cluster.sh geometry    # 4. STEP -> STL + elements (or check STL copied from the laptop)
#   bash setup_cluster.sh test        # 5. OpenFOAM, mpirun and python imports
#   bash setup_cluster.sh all         # 2 -> 5
#
# Before step 2 you may set (they are saved in fw3d.env):
#   FW3D_HOME     big storage for .sif, venv, STL and runs (default: $SCRATCH, else $WORK, else $HOME, + /fw3d)
#   FW3D_MODULES  modules to load first, e.g. "apptainer" or "singularity python/3.11"
# All settings end in hpc/fw3d.env, sourced by run_case_hpc.sh.
set -uo pipefail
. "$(dirname "$0")/lib.sh"
load_modules
STEP=${1:?usage: setup_cluster.sh probe|container|python|geometry|test|all}
say() { echo "== $*"; }
die() { echo "!! $*" >&2; exit 1; }

save_base() {
    setv FW3D_HOME "$FW3D_HOME"; setv FW3D_STL "$FW3D_STL"; setv FW3D_RUNS "$FW3D_RUNS"
    setv FW3D_FOAM "$FW3D_FOAM"; setv FW3D_MODULES "${FW3D_MODULES:-}"
    mkdir -p "$FW3D_HOME" "$FW3D_STL" "$FW3D_RUNS" || die "cannot create $FW3D_HOME"
}

find_apptainer() {
    local c
    for c in apptainer singularity; do command -v $c > /dev/null && { setv FW3D_APPTAINER "$(command -v $c)"; return 0; }; done
    return 1
}

probe() {
    {
        echo "date        $(date -Is)"; echo "host        $(hostname -f 2>/dev/null || hostname)"
        echo "os          $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")"; echo "kernel      $(uname -r)"
        echo "cpu         $(lscpu | sed -n 's/^Model name: *//p' | head -1)"; echo "nproc_login $(nproc)"
        echo "mem_login   $(free -g | awk '/^Mem/ {print $2 " GB"}')"
        echo "user        $(id -un)  groups: $(id -Gn)"
        echo "HOME=$HOME SCRATCH=${SCRATCH:-} WORK=${WORK:-} TMPDIR=${TMPDIR:-}"
        echo "FW3D_HOME (default) $FW3D_HOME"
        echo; echo "--- scheduler"
        for c in sbatch squeue sinfo qsub qstat bsub; do command -v $c > /dev/null && echo "found $c: $(command -v $c)"; done
        command -v sinfo > /dev/null && { sinfo -s 2>&1 | head -30; echo; sinfo -o "%P %c %m %l %D %N" 2>&1 | head -30; }
        command -v sacctmgr > /dev/null && sacctmgr -nP show assoc user="$(id -un)" format=account,partition,qos 2>&1 | head -20
        command -v qstat > /dev/null && qstat -Q 2>&1 | head -20
        echo; echo "--- containers"
        for c in apptainer singularity podman docker; do command -v $c > /dev/null && echo "$c: $($c --version 2>&1 | head -1)"; done
        type module > /dev/null 2>&1 && { echo "module system: yes"; module -t avail 2>&1 | grep -iE "apptainer|singularity|openfoam|python|openmpi" | head -40; }
        echo; echo "--- python on host"
        for c in python3 python3.13 python3.12 python3.11 python3.10; do command -v $c > /dev/null && echo "$c: $($c --version 2>&1)"; done
        echo; echo "--- internet from login node"
        for u in https://registry-1.docker.io/v2/ https://pypi.org/simple/ https://github.com; do
            printf '%s %s\n' "$u" "$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u")"; done   # 000 = unreachable
        echo; echo "--- storage"
        df -h "$HOME" ${SCRATCH:+"$SCRATCH"} ${WORK:+"$WORK"} 2>&1
        command -v quota > /dev/null && quota -s 2>&1 | head -20
        command -v lfs > /dev/null && lfs quota -h -u "$(id -un)" "${SCRATCH:-$HOME}" 2>&1 | head -10
    } 2>&1 | tee "$HPC/probe.txt"
    say "written $HPC/probe.txt"
}

container() {
    save_base
    find_apptainer || die "no apptainer/singularity on PATH (try FW3D_MODULES=apptainer, see probe.txt)"
    export APPTAINER_CACHEDIR=$FW3D_HOME/.cache/apptainer SINGULARITY_CACHEDIR=$FW3D_HOME/.cache/apptainer
    export APPTAINER_TMPDIR=$FW3D_HOME/.cache/tmp SINGULARITY_TMPDIR=$FW3D_HOME/.cache/tmp
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
    local A=$FW3D_HOME/ifw_v2606.sif B=$FW3D_HOME/openfoam-run_v2606.sif
    if [ -f "$A" ]; then
        say "route A: found $A"; setv FW3D_SIF "$A"; setv FW3D_SIF_HAS_PY 1
    elif [ "${FW3D_ROUTE:-A}" = A ] && say "route A: building $A from ifw_v2606.def (fakeroot, 10-30 min)..." \
         && "$FW3D_APPTAINER" build --fakeroot "$A" "$HPC/ifw_v2606.def" > "$FW3D_HOME/build_sif.log" 2>&1; then
        say "route A OK"; setv FW3D_SIF "$A"; setv FW3D_SIF_HAS_PY 1
    else
        rm -f "$A"
        say "route A not possible (see $FW3D_HOME/build_sif.log); route B: pulling the bare OpenFOAM image"
        [ -f "$B" ] || "$FW3D_APPTAINER" pull "$B" "$IMAGE_REF" || die "pull failed: no internet/registry access? see probe.txt"
        setv FW3D_SIF "$B"; setv FW3D_SIF_HAS_PY 0
    fi
    sha256sum "$FW3D_SIF" | tee "$FW3D_SIF.sha256"
}

python_side() {
    save_base
    if [ "${FW3D_SIF_HAS_PY:-0}" = 1 ]; then
        setv FW3D_PY sif
        py -c "import numpy, scipy, matplotlib, gmsh; print('python in .sif OK, gmsh', gmsh.__version__)" \
            && { setv FW3D_HAS_GMSH 1; return 0; }
        die "python inside $FW3D_SIF does not work"
    fi
    local c host=""
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
        command -v $c > /dev/null && $c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' && { host=$(command -v $c); break; }
    done
    [ -n "$host" ] || die "no python >= 3.10 on the host (try FW3D_MODULES='... python/3.11')"
    local V=$FW3D_HOME/venv
    [ -x "$V/bin/python" ] || "$host" -m venv "$V" || die "venv creation failed"
    "$V/bin/pip" install -q --upgrade pip > /dev/null 2>&1
    "$V/bin/pip" install -q numpy scipy matplotlib || die "pip install failed (no internet from login node?)"
    setv FW3D_PY "$V/bin/python"
    "$V/bin/pip" install -q gmsh==4.15.2 > /dev/null 2>&1 || "$V/bin/pip" install -q gmsh > /dev/null 2>&1
    if "$V/bin/python" -c "import gmsh" > /dev/null 2>&1; then setv FW3D_HAS_GMSH 1; else setv FW3D_HAS_GMSH 0; fi
    "$V/bin/python" -c "import numpy, scipy, matplotlib; print('host venv OK', numpy.__version__, scipy.__version__)"
    echo "gmsh usable: $FW3D_HAS_GMSH"
}

geometry() {
    save_base
    local S=$FW3D_STL f missing=0
    for f in wing.stl nose.stl hanger.stl ifw_geometry.json wing_mesh.npz; do [ -f "$S/$f" ] || missing=1; done
    if [ $missing = 1 ]; then
        if [ "${FW3D_HAS_GMSH:-0}" = 1 ]; then
            say "STEP -> STL with gmsh (about 5 min; if the login node kills it, run this step through srun)"
            py "$KIT/ifw/prepare_ifw.py" "$ROOT/imperial/ifw.stp" "$S" || die "prepare_ifw failed"
        else
            die "no gmsh here: copy the STL from the laptop instead, from the repository root on the laptop:
    rsync -av imperial/stl/{wing.stl,nose.stl,hanger.stl,ifw_geometry.json,wing_mesh.npz,ifw_elements.json} <cluster>:$S/
then re-run: bash setup_cluster.sh geometry"
        fi
    fi
    if [ ! -f "$S/ifw_elements.json" ]; then
        [ "${FW3D_HAS_GMSH:-0}" = 1 ] || die "ifw_elements.json missing and no gmsh: copy it from the laptop (imperial/stl/)"
        py "$KIT/ifw/classify_elements.py" "$ROOT/imperial/ifw.stp" "$S/ifw_elements.json" || die "classify_elements failed"
    fi
    local h; h=$(sha256sum "$S/wing.stl" | cut -c1-8)
    if [ "$h" = "$WING_STL_SHA256" ]; then say "wing.stl identical to the laptop one ($h)"
    else echo "WARNING: wing.stl sha256 $h differs from the laptop ($WING_STL_SHA256): different tessellation;"
         echo "         fine for the sensitivity test (all cases share it), note it in the results"; fi
    py -c "import json; e = json.load(open('$S/ifw_elements.json'))['elements']; print({k: len(v) for k, v in e.items()})"
}

test_all() {
    [ -n "${FW3D_SIF:-}" ] || die "run: setup_cluster.sh container"
    say "OpenFOAM in the container"
    foam "$FW3D_HOME" 'echo "OpenFOAM $WM_PROJECT_VERSION"; command -v simpleFoam snappyHexMesh; mpirun --version | head -1' \
        || die "OpenFOAM does not start in $FW3D_SIF"
    say "mpirun with 2 ranks inside the container"
    foam "$FW3D_HOME" 'mpirun $FW3D_MPIRUN_OPTS -np 2 hostname' || die "mpirun failed: try FW3D_MPIRUN_OPTS='--oversubscribe' (setv in fw3d.env)"
    say "python"
    py -c "import numpy, scipy, matplotlib; print('python OK')" || die "python side broken"
    py "$HPC/oat_plan.py" "$HPC/oat_plan.csv" | tail -1
    say "all good; fw3d.env:"; cat "$ENVF"
}

case $STEP in
    probe) probe;;
    container) container;;
    python) python_side;;
    geometry) geometry;;
    test) test_all;;
    all) container && python_side && geometry && test_all;;
    *) die "unknown step $STEP";;
esac
