"""CL(h/c) and CD(h/c) of the 3D cases against Zerihan Fig. 36b (digitised) and the 2D reference point.

Usage: python sweep_summary.py <runs_dir> [--level L0] [--since YYYYMMDD]
Takes, for every h/c, the most recent case of that level that has a report3d.json with coeffs_ok.
Writes <runs_dir>/sweep_<level>.csv and .png.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from report3d import CL_2D_REF, EXP_CL_VALUES, EXP_H, exp_cl

ap = argparse.ArgumentParser()
ap.add_argument("runs", type=Path)
ap.add_argument("--level", default="L0")
ap.add_argument("--since", default="", help="only cases whose directory timestamp is >= this (YYYYMMDD)")
a = ap.parse_args()

best = {}
for rj in sorted(a.runs.glob(f"zerihan3d_{a.level}_h*/report3d.json")):
    stamp = rj.parent.name.rsplit("_", 1)[-1]
    if stamp[:8] < a.since:
        continue
    r = json.loads(rj.read_text())
    if not r.get("coeffs_ok"):
        continue
    best[r["h_over_c"]] = (stamp, rj.parent.name, r)          # sorted by name: the latest wins

rows = []
for h, (stamp, name, r) in sorted(best.items()):
    c = r["coeffs"]
    e = exp_cl(h)
    rows.append({"h_over_c": h, "CL_3D": c["coeffs_total"]["Cl"]["mean"], "CD_3D": c["coeffs_total"]["Cd"]["mean"],
                 "CL_main": c.get("coeffs_main", {}).get("Cl", {}).get("mean"),
                 "CL_flap": c.get("coeffs_flap", {}).get("Cl", {}).get("mean"),
                 "CL_exp_fig36b": e, "rel_diff_pct": None if e is None else 100 * (c["coeffs_total"]["Cl"]["mean"] / e - 1),
                 "Cl_2D": CL_2D_REF.get(h), "iterations": r["solver"]["iterations"],
                 "stop_iter": r.get("autostop", {}).get("k"), "cells": r["mesh"]["cells"],
                 "t_total_min": r["timings_s"].get("total", float("nan")) / 60, "case": name})
out = a.runs / f"sweep_{a.level}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["h_over_c"])
    w.writeheader(); w.writerows(rows)
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(EXP_H, EXP_CL_VALUES, "s-", c="k", ms=4, lw=1, label="Zerihan 2001, Fig. 36b (balance, digitised)")
    ax[0].plot([r["h_over_c"] for r in rows], [r["CL_3D"] for r in rows], "o-", c="C0", label=f"3D RANS {a.level}")
    h2 = [h for h in CL_2D_REF]
    ax[0].plot(h2, [CL_2D_REF[h] for h in h2], "^", c="C3", label="2D dataset (reference case)")
    ax[0].set(xlabel="h/c", ylabel="CL (downforce +)", title="Datum flap, alpha 1 deg"); ax[0].legend(fontsize=8)
    ax[1].plot([r["h_over_c"] for r in rows], [r["CD_3D"] for r in rows], "o-", c="C0")
    ax[1].set(xlabel="h/c", ylabel="CD (3D RANS)")
    fig.tight_layout(); fig.savefig(a.runs / f"sweep_{a.level}.png", dpi=120)
    print(f"written {out} and sweep_{a.level}.png")
except ImportError:
    print(f"written {out}")
