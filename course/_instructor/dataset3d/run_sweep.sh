#!/bin/bash
# Ride-height sweep at L0 with automatic stop, then the CL(h/c) summary against Zerihan Fig. 36b.
#
#   cd ~/Documents/GitHub/F1-Front-Wing-Physics-Informed-NN/course/_instructor/dataset3d
#   NP=6 bash run_sweep.sh                          # h/c 0.079 0.132 0.211 0.395 (~45-60 min per case)
#   SWEEP_H="0.105 0.158" NP=6 bash run_sweep.sh    # other heights
# Cases run one after the other; a failed case does not stop the sweep. Summary: ../../../data/frontwing_runs3d/sweep_L0.{csv,png}
set -uo pipefail
cd "$(dirname "$0")"
SWEEP_H=${SWEEP_H:-"0.079 0.132 0.211 0.395"}
LEVEL=${LEVEL:-L0}
for h in $SWEEP_H; do
    echo "######## h/c $h  ($(date +%T))"
    H=$h bash run_trial.sh "$LEVEL" || echo "case h/c $h failed"
done
../dataset/.venv/bin/python zerihan3d/sweep_summary.py ../../../data/frontwing_runs3d --level "$LEVEL"
