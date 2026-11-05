"""One-at-a-time (OAT) sensitivity plan for the Imperial Front Wing (IFW).

Each row is one CFD case: the baseline, then every parameter alone at the low and the high end of its range,
all the others neutral. 16 shape modes (ifw/shape_modes.py) + 2 operating parameters (ride height, pitch)
= 1 + 2 x 18 = 37 cases. The shape ranges are the ones validated jointly by ifw/check_shape_space.py
(32/32 Sobol samples valid with all 16 modes at once, 05/11/2026).

    python oat_plan.py oat_plan.csv                      # write the plan
    python oat_plan.py oat_plan.csv --check <stl_dir>    # + geometric validity of every row (no CFD, ~1-2 min)

Columns: idx (1-based = SLURM array index), case_id, param, value, shape_json, dz_mm, dpitch_deg, keep_fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SHAPE_RANGES = {"rot_flap2": (-6, 6), "rot_flap1": (-6, 6), "twist_flap2": (-4, 4), "twist_flap1": (-4, 4),
                "camber_main": (-2, 2), "camber_flap1": (-2, 2), "camber_flap2": (-2, 2),
                "camber2_main": (-1.5, 1.5), "camber2_flap1": (-1.5, 1.5), "camber2_flap2": (-1.5, 1.5),
                "thick_main": (0.85, 1.15), "thick_flap1": (0.85, 1.15), "thick_flap2": (0.85, 1.15),
                "chord_flap2": (0.9, 1.1), "gap_flap2": (-1.5, 2.0), "overlap_flap2": (-4, 4)}
# operating: ride height change (mm; baseline h/c 0.356 -> 0.196 / 0.596) and pitch change (deg)
OPERATING_RANGES = {"dz_mm": (-40.0, 60.0), "dpitch_deg": (-1.0, 1.0)}


def plan() -> list[dict]:
    rows = [{"case_id": "base", "param": "baseline", "value": 0, "shape_json": "{}", "dz_mm": 0, "dpitch_deg": 0,
             "keep_fields": 1}]
    for p, (lo, hi) in {**SHAPE_RANGES, **OPERATING_RANGES}.items():
        for end, v in (("lo", lo), ("hi", hi)):
            r = {"case_id": f"{p}_{end}", "param": p, "value": v, "shape_json": "{}", "dz_mm": 0, "dpitch_deg": 0,
                 "keep_fields": 0}
            if p in SHAPE_RANGES:
                r["shape_json"] = json.dumps({p: v})
            else:
                r[p] = v
            rows.append(r)
    for i, r in enumerate(rows, 1):
        r["idx"] = i
    return rows


def check(rows: list[dict], stl: Path) -> list[dict]:
    """Geometric validity of every row with the same rules as deform_ifw.py (slots, folds, hanger root, ground)."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ifw"))
    import shape_modes as sm
    from deform_ifw import MIN_SLOT_MM, hanger_band, measure
    m = np.load(stl / "wing_mesh.npz")
    xyz0, tri, face = m["xyz"], m["tri"], m["face"]
    el = json.loads((stl / "ifw_elements.json").read_text())
    face_el = {f: n for n, fs in el["elements"].items() for f in fs}
    tri_el = np.array([face_el[int(f)] for f in face])
    node_el = {n: np.unique(tri[tri_el == n]) for n in el["elements"]}
    fixed = np.zeros(len(xyz0), bool)
    fixed[node_el["endplate"]] = True
    hb = hanger_band(stl)
    out = []
    for r in rows:
        params = json.loads(r["shape_json"])
        xyz = sm.apply(xyz0, node_el, el["junction"], fixed, params, hb) if params else xyz0
        c = sm.checks(xyz0, xyz, tri, node_el, MIN_SLOT_MM, hanger_band=hb)
        dz, dp = float(r["dz_mm"]), float(r["dpitch_deg"])
        if dz or dp:                                   # same rigid motion as deform_ifw.py, wing only (lowest part)
            ob = xyz[xyz[:, 1] < -0.55]
            pv = ob[np.argmax(ob[:, 0])]
            th = -np.radians(dp)
            X = xyz.copy()
            dx, dzz = X[:, 0] - pv[0], X[:, 2] - pv[2]
            X[:, 0] = pv[0] + dx * np.cos(th) + dzz * np.sin(th)
            X[:, 2] = pv[2] - dx * np.sin(th) + dzz * np.cos(th) + dz / 1e3
            xyz = X
        g = measure(xyz)
        ok = bool(c["valid_slots"] and c["valid_folds"] and c["valid_hanger"] and g["z_min_m"] > 0.005)
        out.append({"idx": r["idx"], "case_id": r["case_id"], "valid": ok, "h_te_over_c": round(g["h_te_over_c"], 4),
                    "pitch_deg": round(g["pitch_deg"], 3), "z_min_mm": round(1e3 * g["z_min_m"], 2),
                    **{k: c[k] for k in ("valid_slots", "valid_folds", "valid_hanger")}})
        print(json.dumps(out[-1]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", type=Path, metavar="STL_DIR", help="also check every row geometrically")
    a = ap.parse_args()
    rows = plan()
    cols = ["idx", "case_id", "param", "value", "shape_json", "dz_mm", "dpitch_deg", "keep_fields"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases -> {a.out}  (sbatch --array=1-{len(rows)})")
    if a.check:
        res = check(rows, a.check)
        bad = [r["case_id"] for r in res if not r["valid"]]
        (a.out.with_suffix(".check.json")).write_text(json.dumps(res, indent=1))
        print(f"valid {len(res) - len(bad)}/{len(res)}" + (f"; INVALID: {bad}" if bad else ""))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
