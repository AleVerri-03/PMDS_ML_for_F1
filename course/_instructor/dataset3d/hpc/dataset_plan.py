"""IFW 3D dataset plan: 8 parameters sampled jointly with a scrambled Sobol sequence.

The parameters and ranges come from the one-at-a-time study (oat_v1, 37 cases) and the boundary check (bnd_v1,
7 cases), 11-15/11/2026: the 8 parameters with an effect on CL well above the remeshing noise (0.2-0.4 %),
each range cut back to where the steady RANS solution exists, with a margin:
  dz_mm        -15 .. +60   h/c 0.296 .. 0.596; -20 is steady (CL +9.6 %, the maximum) but -30 is unsteady
                            (CL -24 %): a cliff, and -20 went through a transient dip, so -15 for margin
  dpitch_deg   -1 .. +1
  rot_flap1    -6 .. +3     +4 steady but slow wander +-0.8 %; +6 stalls after ~850 iterations
  rot_flap2    -6 .. +6     steady (slow wander +-0.5 % at +6)
  twist_flap1  -4 .. +4
  twist_flap2  -4 .. +4
  camber_main  -2 .. +2
  camber2_main -1.5 .. +0.5 +0.5 steady, +1.0 unsteady (main plane separates)
The 10 dropped parameters stay at their neutral value.

The rows are the first N points of ONE Sobol sequence (seed fixed): a pilot of 32 = rows 1-32, and the
dataset of 1024 = rows 1-1024 (the pilot cases are reused). Powers of 2 keep the Sobol balance.

    python dataset_plan.py ds_plan.csv --n 1024                    # the plan
    python dataset_plan.py ds_plan.csv --n 1024 --check STL --rows 1-32   # + geometric validity of rows 1-32
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RANGES = {"dz_mm": (-15.0, 60.0), "dpitch_deg": (-1.0, 1.0),
          "rot_flap1": (-6.0, 3.0), "rot_flap2": (-6.0, 6.0),
          "twist_flap1": (-4.0, 4.0), "twist_flap2": (-4.0, 4.0),
          "camber_main": (-2.0, 2.0), "camber2_main": (-1.5, 0.5)}
OPERATING = ("dz_mm", "dpitch_deg")
SEED = 20261115
COLS = ["idx", "case_id", "param", "value", "shape_json", "dz_mm", "dpitch_deg", "keep_fields"] + [k for k in RANGES if k not in OPERATING]


def plan(n: int) -> list[dict]:
    import numpy as np
    from scipy.stats import qmc
    names = list(RANGES)
    lo = np.array([RANGES[k][0] for k in names])
    hi = np.array([RANGES[k][1] for k in names])
    u = qmc.Sobol(len(names), scramble=True, seed=SEED).random(n)
    x = (lo + u * (hi - lo)).round(3)
    rows = []
    for i, v in enumerate(x, 1):
        p = dict(zip(names, v.tolist()))
        shape = {k: p[k] for k in names if k not in OPERATING}
        rows.append({"idx": i, "case_id": f"s{i:04d}", "param": "sobol", "value": i, "shape_json": json.dumps(shape),
                     "dz_mm": p["dz_mm"], "dpitch_deg": p["dpitch_deg"], "keep_fields": 0, **p})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--check", type=Path, metavar="STL_DIR")
    ap.add_argument("--rows", default=None, help="rows to check, e.g. 1-32 (default: all)")
    a = ap.parse_args()
    rows = plan(a.n)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases -> {a.out}")
    if a.check:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from oat_plan import check
        sel = rows
        if a.rows:
            i0, i1 = map(int, a.rows.split("-"))
            sel = [r for r in rows if i0 <= r["idx"] <= i1]
        res = check(sel, a.check)
        bad = [r["case_id"] for r in res if not r["valid"]]
        a.out.with_suffix(".check.json").write_text(json.dumps(res, indent=1))
        print(f"valid {len(res) - len(bad)}/{len(res)}" + (f"; INVALID: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
