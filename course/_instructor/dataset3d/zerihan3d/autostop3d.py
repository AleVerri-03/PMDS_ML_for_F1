"""Automatic stop for the 3D cases (recipe conv3d-v1), on the model of the 2D monitor (conv-v1).

Why: on the L0 trial (03/11/2026) CL was flat to <0.01 % per 100 iterations from iteration ~900, but the
initial residuals plateau (p 2e-5...4e-4, U 5e-5) so residualControl never fires and the run went on to 2500.

Rule, applied only at checkpoints k = CHECK_EVERY * m (fields are written at every checkpoint, writeInterval
= CHECK_EVERY), on the history truncated at k:
  W = last WINDOW iterations up to k, previous window = the WINDOW before it
  half_amp = (max - min) / 2 over W; drift = mean(W) - mean(previous window)
  CL total: half_amp <= 5e-4 |CL|, |drift| <= 2e-4 |CL|   (same limits as the 2D conv-v1)
  CD total: half_amp <= 6e-4 |CD|, |drift| <= 2e-4 |CD|   (v1, relative; v0 used 1.5e-4 / 5e-5 absolute as in
            2D, unreachable for the IFW whose CD is 0.59 and oscillates by +-0.4 %)
  residual drop >= 3 orders for U, k, p (max initial residual over the first 50 iterations / over the last
  50 up to k); the 2D asks 4, unreachable here (Uz plateaus at 3.8 orders). omega starts from ~1e-4 with the
  uniform field, so it is checked in absolute terms: max initial residual over the last 50 <= 1e-5
  -> flag "converged".
Bounded oscillation (v1, the 2D "oscillating" class): CL drift <= 5e-4 |CL|, CD drift <= 1e-3 |CD|, half_amp
  CL <= 2.5e-3 |CL| and CD <= 1e-2 |CD|, residuals >= 2.5 orders (as the 2D class B; on the IFW the p
  residual plateaus at 3.0 +- 0.05 orders, oscillating with the flow), held at OSC_HOLD consecutive checkpoints
  -> flag "oscillating"; the mean is the label and the half amplitude its declared uncertainty.
  (IFW L0, 04/11/2026: CL +-0.07 %, CD +-0.4 % at iteration 900, a steady RANS on a vortex-dominated flow
  with blunt trailing edges and Gurney flaps.)
When the rule holds, endTime is set to k in system/controlDict (re-read by the running solver): the state at
k is on disk whatever the solver did beyond k. The decision is written to autostop.json.

Usage: python autostop3d.py <case> [--watch]   (without --watch: replay on the finished history)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

RECIPE = "conv3d-v1"
CHECK_EVERY = 100
MIN_ITERS = 600
WINDOW = 200            # replay on the L0 trial: stop at 1300, CL within 0.003 % of the value at 2504
RESID_ORDERS = 3.0
RESID_REF_ITERS = 50
OMEGA_ABS = 1e-5        # omega starts from a small residual (1e-4 with the uniform field): absolute limit instead
HALFAMP_CL_REL, DRIFT_CL_REL = 5e-4, 2e-4
HALFAMP_CD_REL, DRIFT_CD_REL = 6e-4, 2e-4
OSC_HALFAMP_CL_REL, OSC_HALFAMP_CD_REL, OSC_DRIFT_CD_REL = 2.5e-3, 1e-2, 1e-3
OSC_DRIFT_CL_REL = 5e-4  # window means of an oscillating signal wander: IFW L0 CL within +-0.016 % from it 900 to 1500
                         # but window-to-window drift 1.1-2.4e-4
OSC_HOLD = 3            # consecutive checkpoints (300 iterations)
RESID_ORDERS_OSC = 2.5  # as the 2D class B (oscillating, usable with declared uncertainty): residuals >= 2.5 orders


def _table(path: Path) -> tuple[list[str], np.ndarray]:
    names, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            t = line[1:].split()
            if t and t[0] == "Time":
                names = t
            continue
        t = line.split()
        if t:
            rows.append([_num(x) for x in t])
    n = min(len(r) for r in rows) if rows else 0
    return names or [], np.array([r[:n] for r in rows])


def _num(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def history(case: Path):
    """Iterations, CL, CD of coeffs_total and the initial residual of every equation (first solve per iteration)."""
    f = sorted(case.glob("postProcessing/coeffs_total/*/coefficient.dat"))[-1]
    names, a = _table(f)
    it, cl, cd = a[:, 0], a[:, names.index("Cl")], a[:, names.index("Cd")]
    res = {}
    fr = sorted(case.glob("postProcessing/residuals/*/solverInfo.dat"))
    if fr:
        rn, ra = _table(fr[-1])
        for j, n in enumerate(rn):
            if n.endswith("_initial"):
                res[n[:-8]] = (ra[:, 0], ra[:, j])
    return it, cl, cd, res


def evaluate(it, cl, cd, res, k: int) -> dict:
    m = it <= k
    it, cl, cd = it[m], cl[m], cd[m]
    out = {"k": k, "stop": False}
    if k < MIN_ITERS or len(it) < 2 * WINDOW:
        return out
    w, p = slice(-WINDOW, None), slice(-2 * WINDOW, -WINDOW)
    clm = abs(cl[w].mean())
    out.update(CL=float(cl[w].mean()), CD=float(cd[w].mean()),
               cl_halfamp_rel=float((cl[w].max() - cl[w].min()) / 2 / clm),
               cl_drift_rel=float((cl[w].mean() - cl[p].mean()) / clm),
               cd_halfamp=float((cd[w].max() - cd[w].min()) / 2),
               cd_drift=float(cd[w].mean() - cd[p].mean()))
    cdm = abs(cd[w].mean())
    out.update(cd_halfamp_rel=out["cd_halfamp"] / cdm, cd_drift_rel=out["cd_drift"] / cdm)
    drops = {}
    for name, (ri, rv) in res.items():
        rv = rv[ri <= k]
        if len(rv) > 2 * RESID_REF_ITERS and np.all(np.isfinite(rv)):
            drops[name] = float(np.log10(rv[:RESID_REF_ITERS].max() / max(rv[-RESID_REF_ITERS:].max(), 1e-300)))
    out["resid_drop_orders"] = drops
    ok_coef = (out["cl_halfamp_rel"] <= HALFAMP_CL_REL and abs(out["cl_drift_rel"]) <= DRIFT_CL_REL
               and out["cd_halfamp_rel"] <= HALFAMP_CD_REL and abs(out["cd_drift_rel"]) <= DRIFT_CD_REL)
    ok_osc = (out["cl_halfamp_rel"] <= OSC_HALFAMP_CL_REL and abs(out["cl_drift_rel"]) <= OSC_DRIFT_CL_REL
              and out["cd_halfamp_rel"] <= OSC_HALFAMP_CD_REL and abs(out["cd_drift_rel"]) <= OSC_DRIFT_CD_REL)
    om = res.get("omega")
    om_last = float(om[1][om[0] <= k][-RESID_REF_ITERS:].max()) if om is not None else None
    out["omega_initial_last"] = om_last
    ok_res = (bool(drops) and min(v for n, v in drops.items() if n != "omega") >= RESID_ORDERS
              and (om_last is None or om_last <= OMEGA_ABS))
    out["stop"] = bool(ok_coef and ok_res)
    ok_res_osc = (bool(drops) and min(v for n, v in drops.items() if n != "omega") >= RESID_ORDERS_OSC
                  and (om_last is None or om_last <= OMEGA_ABS))
    out["osc_ok"] = bool(ok_osc and ok_res_osc and not out["stop"])
    out["flag"] = "converged" if out["stop"] else ("oscillating_candidate" if out["osc_ok"] else "running")
    return out


def decide(it, cl, cd, res, ks) -> dict | None:
    """First checkpoint in ks where the case may stop: converged, or bounded oscillation held OSC_HOLD times."""
    held = 0
    for k in ks:
        v = evaluate(it, cl, cd, res, k)
        held = held + 1 if v.get("osc_ok") else 0
        if v["stop"]:
            return v
        if held >= OSC_HOLD:
            v.update(stop=True, flag="oscillating", osc_held_checks=held)
            return v
    return None


def set_end_time(case: Path, k: int):
    cd = case / "system/controlDict"
    cd.write_text(re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {k};   // autostop3d {RECIPE}", cd.read_text()))


def replay(case: Path) -> dict | None:
    it, cl, cd, res = history(case)
    return decide(it, cl, cd, res, range(MIN_ITERS, int(it[-1]) + 1, CHECK_EVERY))
    return None


def watch(case: Path, poll_s: float = 30.0):
    """Run next to the solver: decide at checkpoints, set endTime, record the decision. Exits when the case
    reports 'done'/'FAILED' in log.Allrun or once a decision is taken."""
    while True:
        time.sleep(poll_s)
        log = case / "log.Allrun"
        if log.exists() and re.search(r"\] done|FAILED", log.read_text()):
            return
        try:
            it, cl, cd, res = history(case)
        except Exception:          # files not there yet / partial line
            continue
        last = int(it[-1]) // CHECK_EVERY * CHECK_EVERY
        if last >= MIN_ITERS:
            v = decide(it, cl, cd, res, range(MIN_ITERS, last + 1, CHECK_EVERY))   # re-evaluated from the start:
            if v is not None:                                                   # deterministic, holds included
                set_end_time(case, v["k"])
                (case / "autostop.json").write_text(json.dumps({"recipe": RECIPE, **v}, indent=1))
                print(f"autostop: {v['flag']}, endTime set to {v['k']} (CL {v['CL']:.4f}, CD {v['CD']:.5f})", flush=True)
                return


if __name__ == "__main__":
    case = Path(sys.argv[1])
    if "--watch" in sys.argv:
        watch(case)
    else:
        print(json.dumps(replay(case), indent=1))
