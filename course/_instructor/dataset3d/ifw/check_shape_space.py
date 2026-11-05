"""Validity of random IFW shape variants: Sobol samples over the shape modes of shape_modes.py, checked for
slots, folds and the hanger root (no STL written, no mesh). Usage: python check_shape_space.py <stl_dir> <N> <out.json>
Result 05/11/2026: 32/32 valid with the RANGES below (16 modes at once)."""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape_modes as sm
STL = Path(sys.argv[1])
from deform_ifw import hanger_band
m = np.load(STL / "wing_mesh.npz"); xyz0, tri, face = m["xyz"], m["tri"], m["face"]
el = json.load(open(STL / "ifw_elements.json"))
face_el = {f: n for n, fs in el["elements"].items() for f in fs}
tri_el = np.array([face_el[int(f)] for f in face]); node_el = {n: np.unique(tri[tri_el == n]) for n in el["elements"]}
fixed = np.zeros(len(xyz0), bool); fixed[node_el["endplate"]] = True
HB = hanger_band(STL)
RANGES = {"rot_flap2": (-6, 6), "rot_flap1": (-6, 6), "twist_flap2": (-4, 4), "twist_flap1": (-4, 4),
          "camber_main": (-2, 2), "camber_flap1": (-2, 2), "camber_flap2": (-2, 2),
          "camber2_main": (-1.5, 1.5), "camber2_flap1": (-1.5, 1.5), "camber2_flap2": (-1.5, 1.5),
          "thick_main": (0.85, 1.15), "thick_flap1": (0.85, 1.15), "thick_flap2": (0.85, 1.15),
          "chord_flap2": (0.9, 1.1), "gap_flap2": (-1.5, 2.0), "overlap_flap2": (-4, 4)}
names = list(RANGES); lo = np.array([RANGES[k][0] for k in names]); hi = np.array([RANGES[k][1] for k in names])
N = int(sys.argv[2])
U = qmc.Sobol(len(names), scramble=True, seed=0).random(N)
rows = []
for i, u in enumerate(U):
    P = dict(zip(names, (lo + u * (hi - lo)).round(3).tolist()))
    t = time.time()
    X = sm.apply(xyz0, node_el, el["junction"], fixed, P, HB); c = sm.checks(xyz0, X, tri, node_el, hanger_band=HB)
    ok = c["valid_slots"] and c["valid_folds"] and c["valid_hanger"]
    rows.append({"i": i, "params": P, **{k: c[k] for k in ("max_normal_rotation_deg", "p999_normal_rotation_deg", "flipped_triangles", "max_node_displacement_mm", "hanger_root_motion_mm")},
                 "slot_m_f1": c["slot_main_flap1_mm"]["after"], "slot_f1_f2": c["slot_flap1_flap2_mm"]["after"], "valid": ok})
    print(i, ok, "slots %.2f %.2f" % (rows[-1]["slot_m_f1"], rows[-1]["slot_f1_f2"]), "rot max %.1f p999 %.1f flips %d" % (c["max_normal_rotation_deg"], c["p999_normal_rotation_deg"], c["flipped_triangles"]), "disp %.1f" % c["max_node_displacement_mm"], "%.0fs" % (time.time() - t), flush=True)
json.dump({"ranges": RANGES, "rows": rows}, open(sys.argv[3], "w"), indent=1)
print("valid", sum(r["valid"] for r in rows), "/", N)
