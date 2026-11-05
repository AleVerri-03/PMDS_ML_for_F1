"""Parametric variants of the Imperial Front Wing by deforming the tessellated CAD (no CAD rebuild).

Shape modes: see shape_modes.py (16 modes, passed as --params JSON); --d-flap1 / --d-flap2 are kept as
shortcuts for rot_flap1 / rot_flap2. Original description of the two rotations:
Shape parameters (deg), applied on the wing tessellation, each spanwise section about a local leading edge,
positive = trailing edge up = more incidence/camber of the inverted wing (more downforce expected):
  d_flap2: flap 2 alone, about the flap-2 leading edge;
  d_flap1: flap 1 AND flap 2 together (chained, slot flap1-flap2 unchanged), about the flap-1 leading edge;
  d_main:  main plane alone (diagnostic only: +3 deg already closes the main-flap1 slot, 04/11/2026).
  Tested 04/11/2026 with element-alone rotations: flap 2 +-6 deg keeps the slot flap1-flap2 at 3.7-4.7 mm
  (4.5 mm nominal); flap 1 alone at +6 deg closes it (0.04 mm), hence the chaining.
  The elements are attached ONLY to the endplate assembly (classify_elements.py), so the rotation is blended
  to zero towards the junction: w(y) = smoothstep((y - y_j) / L), y_j = innermost junction edge of the
  element, L = BLEND_MM. The outboard L of the element is twisted smoothly and the junction stays watertight
  (nodes of the endplate assembly never move).
Operating parameters (whole model, wing + nose + hanger, rigid):
  dz_mm: ride height change (h/c changes by dz/250 mm);
  dpitch_deg: pitch change about the footplate trailing edge, sign as the measured pitch (the slope of the
  footplate lower edge, 1.09 deg nominal): > 0 lowers the leading edge relative to the trailing edge.
Checks written to deform.json: minimum node distance in the slots main-flap1 and flap1-flap2 (before/after),
ground clearance, measured h/c and pitch of the result. A variant is flagged invalid when a slot closes below
MIN_SLOT_MM or the model touches the ground.

Usage: python deform_ifw.py <stl_dir> <elements.json> <out_dir> [--params '{"camber_main": 1.5, ...}']
                            [--d-flap1 0] [--d-flap2 0] [--dz-mm 0] [--dpitch-deg 0]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

BLEND_MM = 100.0
MIN_SLOT_MM = 1.0
BIN_MM = 5.0


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def read_stl(path: Path):
    b = path.read_bytes()
    n = int(np.frombuffer(b[80:84], "<u4")[0])
    rec = np.frombuffer(b[84:84 + 50 * n], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    v = rec["v"].astype(float).reshape(-1, 3)
    xyz, inv = np.unique(v, axis=0, return_inverse=True)
    return xyz, inv.reshape(-1, 3)


def write_stl(path: Path, name: str, xyz, tri):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_ifw import write_stl_binary
    write_stl_binary(path, name, np.asarray(xyz, float), np.asarray(tri))


def le_line(xyz_el: np.ndarray):
    """Leading edge of an element as a function of y: in each spanwise bin the most upstream node (x min)."""
    y = xyz_el[:, 1]
    edges = np.arange(y.min(), y.max() + BIN_MM / 1e3, BIN_MM / 1e3)
    ys, xs, zs = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        if m.sum() < 3:
            continue
        p = xyz_el[m][np.argmin(xyz_el[m, 0])]
        ys.append(0.5 * (a + b)); xs.append(p[0]); zs.append(p[2])
    return np.array(ys), np.array(xs), np.array(zs)


def rotate_sections(xyz, sel, angle_deg, y_j, le):
    """Rotate nodes `sel` about the leading-edge line `le` = (ys, xs, zs) (axis parallel to y at each y),
    weighted by the junction blend."""
    p = xyz[sel]
    ys, xs, zs = le
    x0, z0 = np.interp(p[:, 1], ys, xs), np.interp(p[:, 1], ys, zs)
    w = smoothstep((p[:, 1] - y_j) / (BLEND_MM / 1e3))
    th = -np.radians(angle_deg) * w                 # rotation about +y by -delta: trailing edge (x > x0) goes UP
    dx, dz = p[:, 0] - x0, p[:, 2] - z0
    out = xyz.copy()
    out[sel, 0] = x0 + dx * np.cos(th) + dz * np.sin(th)
    out[sel, 2] = z0 - dx * np.sin(th) + dz * np.cos(th)
    return out


def min_dist(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    return float(cKDTree(b).query(a)[0].min())


def measure(xyz_wing, c=0.25):
    """Footplate trailing-edge clearance and pitch from the outboard lower edge, as in prepare_ifw.py."""
    ob = xyz_wing[xyz_wing[:, 1] < -0.55]
    xs = np.linspace(ob[:, 0].min(), ob[:, 0].max(), 12)
    zmin = np.array([ob[(ob[:, 0] >= a) & (ob[:, 0] < b), 2].min() for a, b in zip(xs[:-1], xs[1:])])
    slope = np.polyfit(0.5 * (xs[:-1] + xs[1:]), zmin, 1)[0]
    return {"h_te_m": float(zmin[-1]), "h_te_over_c": float(zmin[-1] / c), "pitch_deg": float(np.degrees(np.arctan(slope))),
            "z_min_m": float(xyz_wing[:, 2].min())}


def hanger_band(stl_dir: Path) -> tuple:
    """(y_centre, half_width) of the hanger root on the main plane (lowest 10 mm of the hanger), +4 mm margin."""
    X, _ = read_stl(stl_dir / "hanger.stl")
    root = X[X[:, 2] < X[:, 2].min() + 0.01]
    return float(0.5 * (root[:, 1].min() + root[:, 1].max())), float(0.5 * (root[:, 1].max() - root[:, 1].min()) + 0.004)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import shape_modes as sm
    ap = argparse.ArgumentParser(description="IFW variants; shape modes in shape_modes.py")
    ap.add_argument("stl_dir", type=Path)
    ap.add_argument("elements", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--params", default="{}", help='shape modes as JSON, e.g. \'{"rot_flap2": 4, "camber_main": 1.5}\'')
    for k in ("d-flap1", "d-flap2", "dz-mm", "dpitch-deg"):
        ap.add_argument(f"--{k}", type=float, default=0.0)
    a = ap.parse_args()
    params = json.loads(a.params)
    unknown = set(params) - set(sm.MODES)
    if unknown:
        raise SystemExit(f"unknown shape modes {sorted(unknown)}; available: {sorted(sm.MODES)}")
    if a.d_flap1:
        params["rot_flap1"] = a.d_flap1
    if a.d_flap2:
        params["rot_flap2"] = a.d_flap2
    m = np.load(a.stl_dir / "wing_mesh.npz")
    xyz0, tri, face = m["xyz"], m["tri"], m["face"]
    el = json.loads(a.elements.read_text())
    face_el = {f: n for n, fs in el["elements"].items() for f in fs}
    tri_el = np.array([face_el[int(f)] for f in face])
    node_el = {n: np.unique(tri[tri_el == n]) for n in el["elements"]}
    fixed = np.zeros(len(xyz0), bool)
    fixed[node_el["endplate"]] = True                 # the endplate assembly never moves (junction nodes included)
    hb = hanger_band(a.stl_dir)
    xyz = sm.apply(xyz0, node_el, el["junction"], fixed, params, hb)
    checks = sm.checks(xyz0, xyz, tri, node_el, MIN_SLOT_MM, hanger_band=hb)

    # operating parameters: rigid motion of the whole model (wing, nose, hanger)
    parts = {"wing": (xyz, tri)}
    for n in ("nose", "hanger"):
        parts[n] = read_stl(a.stl_dir / f"{n}.stl")
    geo0 = measure(xyz)
    if a.dpitch_deg or a.dz_mm:
        ob = xyz[xyz[:, 1] < -0.55]
        pivot = ob[np.argmax(ob[:, 0])]                    # footplate trailing edge
        th = -np.radians(a.dpitch_deg)                      # > 0: trailing edge up = measured pitch increases by dpitch
        for n, (X, T) in parts.items():
            X = X.copy()
            dx, dz = X[:, 0] - pivot[0], X[:, 2] - pivot[2]
            X[:, 0] = pivot[0] + dx * np.cos(th) + dz * np.sin(th)
            X[:, 2] = pivot[2] - dx * np.sin(th) + dz * np.cos(th) + a.dz_mm / 1e3
            parts[n] = (X, T)
    geo = measure(parts["wing"][0])
    valid = checks["valid_slots"] and checks["valid_folds"] and checks["valid_hanger"] and geo["z_min_m"] > 0.005
    a.out.mkdir(parents=True, exist_ok=True)
    for n, (X, T) in parts.items():
        write_stl(a.out / f"{n}.stl", n, X, T)
    rec = json.loads((a.stl_dir / "ifw_geometry.json").read_text())
    allx = np.vstack([X for X, _ in parts.values()])
    full = {k: params.get(k, sm.NEUTRAL[k]) for k in sm.MODES}
    rec.update({"bbox_model_min_m": allx.min(0).tolist(), "bbox_model_max_m": allx.max(0).tolist(),
                "h_te_m": geo["h_te_m"], "h_te_over_c": geo["h_te_over_c"], "pitch_deg_lower_edge": geo["pitch_deg"],
                "variant": {**full, "dz_mm": a.dz_mm, "dpitch_deg": a.dpitch_deg, "blend_mm": sm.BLEND_MM}})
    (a.out / "ifw_geometry.json").write_text(json.dumps(rec, indent=1))
    out = {"params": rec["variant"], "checks": checks, "geometry_before_operating": geo0, "geometry": geo,
           "hanger_band_m": hb, "valid": bool(valid)}
    (a.out / "deform.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
