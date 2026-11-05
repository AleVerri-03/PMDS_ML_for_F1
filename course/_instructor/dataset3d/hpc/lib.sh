# Shared helpers for the cluster scripts (sourced, not executed).
# Paths: HPC = this directory, KIT = dataset3d, ROOT = repository root.
HPC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
KIT=$(dirname "$HPC")
ROOT=$(cd "$KIT/../../.." && pwd -P)
ENVF=$HPC/fw3d.env                      # written by setup_cluster.sh, local to the clone (not in git)
IMAGE_REF="docker://opencfd/openfoam-run@sha256:0b0892f2fadbf5602f8fad8f8e507a81b6995863c7b021886a58ddce099e7a77"
WING_STL_SHA256=ae3d2579        # first 8 hex of sha256(wing.stl) on the laptop (gmsh 4.15.2, 04/11/2026)

[ -f "$ENVF" ] && . "$ENVF"
: "${FW3D_HOME:=${SCRATCH:-${WORK:-$HOME}}/fw3d}"
FW3D_HOME=$(realpath -m "$FW3D_HOME")   # physical paths only: symlinked /scratch would not exist in the container
: "${FW3D_STL:=$FW3D_HOME/stl}"
: "${FW3D_RUNS:=$FW3D_HOME/runs}"
FW3D_STL=$(realpath -m "$FW3D_STL"); FW3D_RUNS=$(realpath -m "$FW3D_RUNS")
: "${FW3D_FOAM:=container}"             # container | native
: "${FW3D_MPIRUN_OPTS:=--oversubscribe --bind-to none}"
# single node: shared memory only, no UCX/InfiniBand probing inside the container
: "${FW3D_OMPI_ENV:=OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader OMPI_MCA_btl_vader_single_copy_mechanism=none}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1

load_modules() {                        # FW3D_MODULES="apptainer python/3.11" etc., if the cluster uses modules
    if [ -n "${FW3D_MODULES:-}" ] && type module > /dev/null 2>&1; then module load $FW3D_MODULES; fi
}

setv() {                                # setv KEY VALUE: set a line KEY='VALUE' in fw3d.env
    touch "$ENVF"
    grep -v "^$1=" "$ENVF" > "$ENVF.tmp" || true
    printf "%s='%s'\n" "$1" "$2" >> "$ENVF.tmp"
    mv "$ENVF.tmp" "$ENVF"
    eval "$1='$2'"
}

binds() {                              # repository + FW3D_HOME (+ runs/stl when they live elsewhere), same paths inside
    local b=$ROOT,$FW3D_HOME d
    for d in "$FW3D_RUNS" "$FW3D_STL"; do case $d in "$FW3D_HOME"/*) ;; *) b=$b,$d;; esac; done
    printf '%s' "$b"
}

py() {                                  # python of the kit: inside the .sif (/opt/venv) or a host venv
    if [ "${FW3D_PY:-}" = sif ]; then
        "$FW3D_APPTAINER" exec --cleanenv --bind "$(binds)" "$FW3D_SIF" /opt/venv/bin/python "$@"
    else
        "${FW3D_PY:?run setup_cluster.sh first}" "$@"
    fi
}

foam() {                                # foam <dir> <command string>: run in <dir> with OpenFOAM v2606 loaded
    local dir=$1 cmd=$2
    local pre="export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 $FW3D_OMPI_ENV FW3D_MPIRUN_OPTS='$FW3D_MPIRUN_OPTS' FW3D_SOLVER_OPTS='${FW3D_SOLVER_OPTS:-}'"
    if [ "$FW3D_FOAM" = container ]; then
        "$FW3D_APPTAINER" exec --cleanenv --bind "$(binds)" "$FW3D_SIF" \
            bash -c "$pre; . /usr/lib/openfoam/openfoam2606/etc/bashrc; cd '$dir' && $cmd"
    else                                # native install: FW3D_FOAM_BASHRC=/path/to/OpenFOAM-vXXXX/etc/bashrc
        bash -c "$pre; . '${FW3D_FOAM_BASHRC:?set FW3D_FOAM_BASHRC}'; cd '$dir' && $cmd"
    fi
}
