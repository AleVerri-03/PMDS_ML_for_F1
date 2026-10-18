#!/usr/bin/env python3
"""
Phase 1-2 artifact: critical replication of

    N. Shah, "Computational Fluid Dynamics Optimization of F1 Front Wing using
    Physics Informed Neural Networks", arXiv:2509.01963 (2025)

Data/code: https://github.com/NavalShah/F1-Front-Wing-Physics-Informed-NN

By default it prints a SHORT summary of four checks (about 30 lines): dataset,
identity, baseline, physics constant. Add --full for the long explanations and a
fifth check (train/validation leakage). Re-derive every number by hand before
believing the conclusions.

Usage (from the root of the cloned repo, which already contains the three CSVs):
    python course/L1/audit_replication.py
    python course/L1/audit_replication.py --full     # long output + leakage check
    python course/L1/audit_replication.py --plot
    python course/L1/audit_replication.py --data-dir <folder with the CSVs>   # elsewhere

Requires: numpy, pandas, scikit-learn, matplotlib (optional, for --plot)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Values reported in the paper, Section 4.1
PAPER_R2_CD = 0.96753
PAPER_R2_CL = 0.98102
PAPER_MSE_CD = 0.031
PAPER_MSE_CL = 0.024

# Constants hardcoded in the paper's physics_loss() (main.py, line 103)
PAPER_RHO = 1.225   # kg/m^3
PAPER_V = 50.0      # m/s

FORCE_COLS = ["PRESSURE_FORCE_X", "PRESSURE_FORCE_Y", "PRESSURE_FORCE_Z",
              "VISCOUS_FORCE_X", "VISCOUS_FORCE_Y", "VISCOUS_FORCE_Z"]
MOMENT_COLS = ["PRESSURE_MOMENT_X", "PRESSURE_MOMENT_Y", "PRESSURE_MOMENT_Z",
               "VISCOUS_MOMENT_X", "VISCOUS_MOMENT_Y", "VISCOUS_MOMENT_Z"]


def banner(n, title):
    print("\n" + "=" * 78)
    print(f"CHECK {n}: {title}")
    print("=" * 78)


def load(data_dir):
    """Load the three CSVs exactly as the paper's WingDataset does."""
    coeff = pd.read_csv(os.path.join(data_dir, "Force_and_moment_coefficients_2.csv"))
    moments = pd.read_csv(os.path.join(data_dir, "Forces_and_moments_1 (1).csv"))
    forces = pd.read_csv(os.path.join(data_dir, "Forces_and_moments_1 (2).csv"))

    X = np.c_[forces[FORCE_COLS].values, moments[MOMENT_COLS].values]
    y = coeff[["FORCE_COEFFICIENT_CD", "FORCE_COEFFICIENT_CL"]].values
    return coeff, forces, moments, X, y


# ---------------------------------------------------------------------------
# CHECK 1 - What is actually in the dataset?
# ---------------------------------------------------------------------------
def check_dataset_structure(coeff, forces):
    banner(1, "What does the dataset actually contain?")

    n = len(coeff)
    print(f"Rows in each CSV               : {n}")
    print(f"Index column                   : {coeff.columns[0]!r} "
          f"(range {coeff.iloc[0, 0]:.0f} .. {coeff.iloc[-1, 0]:.0f}, step 1)")
    print(f"Coefficient columns available  : {list(coeff.columns[1:])}")
    print(f"  -> the paper uses only 2 of these {len(coeff.columns) - 1}; "
          f"CLF/CLR (front/rear lift split) and CM are discarded.")

    cd = coeff["FORCE_COEFFICIENT_CD"].values
    cl = coeff["FORCE_COEFFICIENT_CL"].values

    print("\nHistory of Cd over the integer counter in the 'Time (s)' column "
          "(solver iterations, not a set of designs):")
    print(f"  first 10 rows  : {np.round(cd[:10], 4)}")
    print(f"  last  10 rows  : {np.round(cd[-10:], 5)}")
    print(f"\n  Cd  full range   : mean {cd.mean():.5f}  std {cd.std():.5f}")
    print(f"  Cd  rows  >2000  : mean {cd[2000:].mean():.5f}  std {cd[2000:].std():.5f}")
    print(f"  Cl  rows  >2000  : mean {cl[2000:].mean():.5f}  std {cl[2000:].std():.5f}")

    print(f"\nVERDICT: all {n} rows come from ONE simulation of")
    print("ONE geometry, converging to a single operating point (the counter is 1, 2, 3, ...")
    print("in unit steps: most likely solver iterations of a steady run, not physical time). There is no")
    print("geometric variation anywhere in the data. The paper's abstract describes")
    print("'12 aerodynamic features' -- these are the 12 input COLUMNS, not 12")
    print("configurations. A surrogate trained on this cannot explore a design")
    print("space, because it has never seen a second design.")


# ---------------------------------------------------------------------------
# CHECK 2 - Are inputs and targets algebraically identical?
# ---------------------------------------------------------------------------
def check_algebraic_identity(coeff, forces):
    banner(2, "Are the inputs and the targets the same quantity?")

    cd = coeff["FORCE_COEFFICIENT_CD"].values
    cl = coeff["FORCE_COEFFICIENT_CL"].values

    # The paper feeds pressure and viscous components separately.
    # Their sum is the total force, which is what a coefficient is defined from.
    fx = forces["PRESSURE_FORCE_X"].values + forces["VISCOUS_FORCE_X"].values
    fy = forces["PRESSURE_FORCE_Y"].values + forces["VISCOUS_FORCE_Y"].values

    resid_x = np.abs(fx - forces["TOTAL_FORCE_X"].values).max()
    resid_y = np.abs(fy - forces["TOTAL_FORCE_Y"].values).max()
    print(f"max |PRESSURE_X + VISCOUS_X - TOTAL_X| = {resid_x:.3e}")
    print(f"max |PRESSURE_Y + VISCOUS_Y - TOTAL_Y| = {resid_y:.3e}")
    print("  -> total force is exactly the sum of two of the 12 input features.\n")

    k_d = fx / cd
    k_l = fy / cl
    print(f"Fx / Cd : mean {k_d.mean():+.6f}   std {k_d.std():.4e}   "
          f"rel.std {k_d.std() / abs(k_d.mean()):.2e}")
    print(f"Fy / Cl : mean {k_l.mean():+.6f}   std {k_l.std():.4e}   "
          f"rel.std {k_l.std() / abs(k_l.mean()):.2e}")

    print("\nVERDICT: the ratio is constant to ~1e-13, i.e. to floating-point")
    print("precision. Cd and Cl are not physical quantities to be *predicted* from")
    print("the forces -- they are the forces, divided by a fixed constant. The")
    print("'learning' task is a division by a scalar.")
    return abs(k_d.mean())


# ---------------------------------------------------------------------------
# CHECK 3 - Beat the network with a linear model
# ---------------------------------------------------------------------------
def check_linear_baseline(X, y):
    banner(3, "The baseline the paper never ran")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    lin = LinearRegression().fit(Xtr, ytr)
    pred = lin.predict(Xte)

    r2_cd = r2_score(yte[:, 0], pred[:, 0])
    r2_cl = r2_score(yte[:, 1], pred[:, 1])
    mse_cd = float(np.mean((yte[:, 0] - pred[:, 0]) ** 2))
    mse_cl = float(np.mean((yte[:, 1] - pred[:, 1]) ** 2))

    print("Ordinary least squares, 12 inputs -> 2 outputs, same 80/20 split.")
    print("Model complexity: 26 parameters. Training time: milliseconds. CPU only.\n")
    print(f"{'':22s} {'R2 (Cd)':>14s} {'R2 (Cl)':>14s} {'MSE (Cd)':>12s} {'MSE (Cl)':>12s}")
    print(f"{'paper PINN (12,066 p)':22s} {PAPER_R2_CD:>14.5f} {PAPER_R2_CL:>14.5f} "
          f"{PAPER_MSE_CD:>12.3g} {PAPER_MSE_CL:>12.3g}")
    print(f"{'linear reg. (26 p)':22s} {r2_cd:>14.10f} {r2_cl:>14.10f} "
          f"{mse_cd:>12.3g} {mse_cl:>12.3g}")

    print(f"\nRecovered coefficients for Cd (should be ~1/k on the two X-force inputs):")
    for name, w in zip(FORCE_COLS + MOMENT_COLS, lin.coef_[0]):
        if abs(w) > 1e-9:
            print(f"    {name:20s} {w:+.8e}")

    print("\nVERDICT: R2 = 1.0 to ten decimal places, MSE at the level of numerical")
    print("noise. The 12,066-parameter 'PINN' does measurably WORSE than ordinary")
    print("least squares on a problem whose exact solution is linear. Reporting")
    print("R2 = 0.968 as a success is only possible if no baseline was ever run.")


# ---------------------------------------------------------------------------
# CHECK 4 - Is the physics term dimensionally correct?
# ---------------------------------------------------------------------------
def check_physics_term(k_true):
    banner(4, "Is the physics-informed loss physically correct?")

    print("The paper writes the constraint as   F = 0.5 * rho * V^2 * C")
    print("The definition of a force coefficient is  F = 0.5 * rho * V^2 * A * C")
    print("                                                              ^^^")
    print("The reference area A is missing. Check the units:")
    print("  0.5 * rho * V^2        -> [kg/m^3][m^2/s^2] = kg/(m s^2) = Pa")
    print("  a force               -> N = kg m/s^2")
    print("  Pa != N, so the stated equation cannot be dimensionally valid.\n")

    k_paper = 0.5 * PAPER_RHO * PAPER_V ** 2
    print(f"Constant hardcoded in the paper's loss : 0.5*{PAPER_RHO}*{PAPER_V}^2 "
          f"= {k_paper:.4f}")
    print(f"Constant actually present in the data  : {k_true:.4f}")
    print(f"Ratio (paper / data)                  : {k_paper / k_true:.4f}x\n")

    print("Solving 0.5 * rho * V^2 * A = k_true for the two unknowns:")
    print(f"  if A = 1.00 m^2 (unit reference area) -> V = "
          f"{np.sqrt(k_true / (0.5 * PAPER_RHO)):.2f} m/s "
          f"({3.6 * np.sqrt(k_true / (0.5 * PAPER_RHO)):.0f} km/h)")
    print(f"  if V = 50 m/s (as claimed)            -> A = "
          f"{k_true / k_paper:.4f} m^2")
    print("  An F1 front wing reference area is order 1 m^2, so V = 50 m/s is")
    print("  inconsistent with this data by roughly an order of magnitude.\n")

    print(f"VERDICT: the physics term penalises the network by a factor "
          f"{k_paper / k_true:.1f}x away from")
    print("the true relation. It does not encode physics -- it encodes a wrong")
    print("constant, and it actively pulls the prediction away from the correct")
    print("answer. The reported lambda_phys = 0.1 is what keeps the damage small.")
    print("\nSeparately: there is no call to torch.autograd.grad anywhere in main.py.")
    print("No derivative of the network is ever taken with respect to its inputs, so")
    print("no PDE residual is ever formed. By the definition in Raissi et al. (2019)")
    print("this model is not a physics-informed neural network. It is a multilayer")
    print("perceptron with an algebraic penalty term.")


# ---------------------------------------------------------------------------
# CHECK 5 - Does the evaluation protocol hold up?
# ---------------------------------------------------------------------------
def check_evaluation_protocol(X, y):
    banner(5, "Does the train/validation protocol measure generalisation?")

    # Autocorrelation of the target: adjacent timesteps are near-duplicates.
    cd = y[:, 0]
    ac1 = np.corrcoef(cd[:-1], cd[1:])[0, 1]
    print(f"Lag-1 autocorrelation of Cd across timesteps: {ac1:.6f}")
    print("Adjacent rows are near-duplicates of each other. A uniformly random")
    print("80/20 split therefore places, for almost every validation row, its two")
    print("immediate neighbours in the training set.\n")

    # Nearest-neighbour distance in time from each val point to a train point.
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    tr_sorted = np.sort(tr)
    gaps = np.abs(te[:, None] - tr_sorted[None, :]).min(axis=1)
    print(f"Distance (in timesteps) from each validation row to the nearest")
    print(f"training row:  mean {gaps.mean():.3f}   max {gaps.max()}   "
          f"share at distance 1: {100 * (gaps == 1).mean():.1f}%")

    print("\nThe paper states the split was stratified to preserve the distribution")
    print("of force magnitudes. main.py line 133 calls train_test_split(indices,")
    print("test_size=0.2, random_state=42) with no `stratify` argument. The stated")
    print("protocol is not the implemented one.\n")

    print("Finally, main.py lines 229-233 fit a LinearRegression calibration on")
    print("y_pred_all / y_true_all -- the FULL dataset, validation rows included --")
    print("and the paper reports the resulting improvement as a result. Any metric")
    print("computed after that step is fitted on its own test set.\n")

    print("VERDICT: three compounding protocol errors. Even if the task were not")
    print("trivial, the reported numbers would not measure generalisation.")


# ---------------------------------------------------------------------------
# COMPACT MODE (default): the four numbers that matter, a few lines each
# ---------------------------------------------------------------------------
def compact_report(coeff, forces, X, y):
    cd = coeff["FORCE_COEFFICIENT_CD"].values
    cl = coeff["FORCE_COEFFICIENT_CL"].values
    n = len(coeff)

    print("\n[1] DATASET")
    print(f"    {n} rows per CSV, counter 1..{n}: one simulation, one geometry.")
    print(f"    Cd: first rows {np.round(cd[:4], 3)} ... mean {cd[2000:].mean():.4f} after row 2000 "
          f"(std {cd[2000:].std():.4f})")

    fx = forces["PRESSURE_FORCE_X"].values + forces["VISCOUS_FORCE_X"].values
    fy = forces["PRESSURE_FORCE_Y"].values + forces["VISCOUS_FORCE_Y"].values
    k_d, k_l = fx / cd, fy / cl
    print("\n[2] IDENTITY  (force / coefficient)")
    print(f"    Fx / Cd = {k_d.mean():+.4f}   relative spread {k_d.std() / abs(k_d.mean()):.1e}")
    print(f"    Fy / Cl = {k_l.mean():+.4f}   relative spread {k_l.std() / abs(k_l.mean()):.1e}")
    print("    -> constant to floating-point precision: the target is the input / a constant.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
    r2_cd = r2_score(yte[:, 0], pred[:, 0])
    r2_cl = r2_score(yte[:, 1], pred[:, 1])
    print("\n[3] BASELINE  (ordinary least squares, paper's own 80/20 split)")
    print(f"    {'':24s}{'params':>8s}{'R2 Cd':>10s}{'R2 Cl':>10s}")
    print(f"    {'paper PINN':24s}{'12066':>8s}{PAPER_R2_CD:>10.3f}{PAPER_R2_CL:>10.3f}")
    print(f"    {'linear regression':24s}{'26':>8s}{r2_cd:>10.6f}{r2_cl:>10.6f}")

    k_true = abs(k_d.mean())
    k_paper = 0.5 * PAPER_RHO * PAPER_V ** 2
    print("\n[4] PHYSICS TERM  (constant 1/2 rho V^2 used in the paper's loss)")
    print(f"    paper: 0.5*{PAPER_RHO}*{PAPER_V:g}^2 = {k_paper:.2f}    data: {k_true:.4f}    "
          f"ratio {k_paper / k_true:.1f}x")
    print("    The area A is missing from the formula (Pa != N), and no derivative of the")
    print("    network is ever taken: no residual, so it is not a PINN.")

    print("\nSummary: D1 one simulation | D2 target = input / constant | D3 26-parameter model "
          "beats the network")
    print("         D4 physics term wrong by 11.5x | D5 no derivative, not a PINN")
    print("Run with --full for the long explanations and the train/validation leakage check.")



def maybe_plot(coeff, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not available, skipping plot]")
        return
    cd = coeff["FORCE_COEFFICIENT_CD"].values
    cl = coeff["FORCE_COEFFICIENT_CL"].values
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, v, lab in ((ax[0], cd, "$C_d$"), (ax[1], cl, "$C_l$")):
        a.plot(v, lw=0.7)
        a.set_xlabel("solver iteration")
        a.set_ylabel(lab)
        a.set_title(f"{lab}: one run, converging")
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "dataset_is_one_simulation.png")
    fig.savefig(path, dpi=140)
    print(f"\n[wrote {path}]")


def find_data_dir():
    """Default: first folder (cwd, then the script's parents) that holds the CSVs."""
    marker = "Force_and_moment_coefficients_2.csv"
    d = os.path.dirname(os.path.abspath(__file__))
    cands = [os.getcwd()]
    for _ in range(4):
        cands.append(d)
        d = os.path.dirname(d)
    for c in cands:
        if os.path.exists(os.path.join(c, marker)):
            return c
    return "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="directory containing the three CSVs (default: auto-detect the repo root)")
    ap.add_argument("--full", "--verbose", dest="full", action="store_true",
                    help="long output with explanations, plus the train/validation leakage check")
    ap.add_argument("--plot", action="store_true", help="also write a PNG of Cd/Cl history")
    args = ap.parse_args()
    if args.data_dir is None:
        args.data_dir = find_data_dir()

    try:
        coeff, forces, moments, X, y = load(args.data_dir)
    except FileNotFoundError as e:
        sys.exit(f"Could not find the data: {e}\n"
                 f"Clone https://github.com/NavalShah/F1-Front-Wing-Physics-Informed-NN "
                 f"or pass --data-dir <folder with the CSVs>")

    print("CRITICAL REPLICATION OF arXiv:2509.01963")
    print("Polimi Data Scientists - F1 PINN project, Phase 1-2")

    if args.full:
        check_dataset_structure(coeff, forces)
        k_true = check_algebraic_identity(coeff, forces)
        check_linear_baseline(X, y)
        check_physics_term(k_true)
        check_evaluation_protocol(X, y)
    else:
        compact_report(coeff, forces, X, y)

    if args.plot:
        maybe_plot(coeff, args.data_dir if os.access(args.data_dir, os.W_OK) else ".")

    if args.full:
        print("\n" + "=" * 78)
        print("SUMMARY OF DEFECTS")
        print("=" * 78)
        print("D1  Dataset is one simulation of one geometry; no design variation.")
        print("D2  Targets are the inputs divided by a constant (exact to ~1e-13).")
        print("D3  A 26-parameter linear model reaches R2 = 1.0; the network gets 0.968.")
        print("D4  Physics term omits the reference area and is wrong by ~11.5x.")
        print("D5  No autograd-based PDE residual: the model is not a PINN.")
        print("D6  Random split on autocorrelated timesteps + calibration fitted on")
        print("    the test set: reported metrics do not measure generalisation.")
        print()
        print("None of this means the research direction is wrong. Surrogate models for")
        print("aerodynamic design ARE valuable and PINNs ARE a real method. It means the")
        print("paper does not provide evidence for either. Our project supplies it.")


if __name__ == "__main__":
    main()
