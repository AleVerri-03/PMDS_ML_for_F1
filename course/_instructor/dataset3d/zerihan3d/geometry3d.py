"""3D Zerihan (2001) double-element wing with endplates, as STL surfaces for snappyHexMesh.

Half model: symmetry plane at the wing mid-span (z = 0), wing tip and endplate at z = +b/2.

Sources (never nominal values where a measured one exists):
  * section: frontwing.geometry.zerihan_wing (frozen smooth_te contours of the 2D dataset, Tables 3-4);
  * span b = 1100 mm, endplates 400 x 170 x 4 mm, rectangular, parallel to the ground at the
    reference incidence (alpha nominal 1 deg) -- thesis Sec. 2.2.1-2.2.2 and 2.7;
  * endplate POSITION relative to the wing: not given in the text. Read from Fig. 1c (to-scale schematic)
    by fitting the true wing outline to the drawing (fit_endplate_from_fig1c.py, rms 0.87 px = 0.5 mm;
    the fitted vertical scale gives an endplate height of 170.3 mm against the 170 mm of the text).
    ASSUMPTION, declared in the README: front edge 3.1 mm ahead of the main LE, bottom edge 12.5 mm
    below the lowest point of the wing (alpha 1), +-1 mm.
  * the endplate is rigidly attached to the wing: for alpha != 1 it rotates with it about the same
    pivot (main-element quarter chord) used by zerihan_wing.

Frame of the CFD case: x downstream, y up, ground at y = 0, z spanwise (0 = mid-span), metres.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET_2D = HERE.parent.parent / "dataset"          # course/_instructor/dataset (2D pipeline, read-only)
sys.path.insert(0, str(DATASET_2D))
from frontwing.geometry import ground_y_zerihan, rotate, zerihan_wing  # noqa: E402

C_REF = 0.38            # m
SPAN = 1.10             # m, full span
ENDPLATE_T = 0.004      # m
AFT_FRACTION = 0.75     # STL region "aft": faces beyond 75 % of the element chord
EP = json.loads((HERE / "endplate_fig1c.json").read_text())
# endplate rectangle at alpha nominal 1 deg, wing frame (main LE at the origin), metres
EP_X0, EP_X1 = EP["endplate_x_front_mm"] / 1e3, EP["endplate_x_front_mm"] / 1e3 + 0.400
EP_Y0, EP_Y1 = EP["endplate_y_bottom_mm"] / 1e3, EP["endplate_y_bottom_mm"] / 1e3 + 0.170


def _sides(el, n_side: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Suction and pressure surfaces, each LE -> TE, resampled with the SAME number of points at the same
    cosine-spaced fractions of their own arc length (units of c). Index i on both sides is used to build the
    end caps as a ladder of quads, so no external triangulation library is needed."""
    s, p = el.dense(4000)
    out = []
    for side in (s, p):
        L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))])
        u = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_side))) * L[-1]
        out.append(np.column_stack([np.interp(u, L, side[:, 0]), np.interp(u, L, side[:, 1])]))
    return out[0], out[1]


def _area(poly: np.ndarray) -> float:
    return 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))


def _extrude_sides(S: np.ndarray, P: np.ndarray, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed prism of the section bounded by S (LE->TE) and P (LE->TE), S[0] == P[0] (leading edge).
    Polygon: S[0..m], P[m..1] (the blunt TE base closes it). Caps: nose triangle + quads (S_i, S_i+1, P_i+1, P_i).
    The cap triangulation is accepted only if its unsigned area equals the polygon area (no overlap, no flip)."""
    m = len(S) - 1
    poly = np.vstack([S, P[::-1][:-1]])                      # indices: S_i -> i, P_j -> 2m+1-j (j = 1..m)
    n = len(poly)
    sid = np.arange(m + 1); pid = np.concatenate([[0], 2 * m + 1 - np.arange(1, m + 1)])
    tri = [[sid[0], sid[1], pid[1]]]
    for i in range(1, m):
        tri += [[sid[i], sid[i + 1], pid[i + 1]], [sid[i], pid[i + 1], pid[i]]]
    tri = np.array(tri)
    A = _area(poly)
    ta = [_area(poly[t]) for t in tri]
    if np.any(np.sign(ta) != np.sign(A)) or abs(sum(abs(x) for x in ta) - abs(A)) > 1e-9 * abs(A):
        raise RuntimeError("end-cap triangulation invalid (flipped or overlapping triangles)")
    if A < 0:                                                # make the polygon counter-clockwise
        tri = tri[:, ::-1]
    V = np.vstack([np.column_stack([poly, np.full(n, z0)]), np.column_stack([poly, np.full(n, z1)])])
    F = [tri[:, ::-1], tri + n]                              # bottom cap (outward -z), top cap (+z)
    i = np.arange(n); j = (i + 1) % n
    side = [np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n])]
    if A < 0:
        side = [f[:, ::-1] for f in side]
    return V, np.vstack(F + side)


def _box(rect: np.ndarray, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Prism of a convex quadrilateral (endplate)."""
    if _area(rect) < 0:
        rect = rect[::-1]
    n = 4
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    V = np.vstack([np.column_stack([rect, np.full(n, z0)]), np.column_stack([rect, np.full(n, z1)])])
    i = np.arange(n); j = (i + 1) % n
    F = [tri[:, ::-1], tri + n, np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n])]
    return V, np.vstack(F)


def _write_stl(path: Path, name: str, V: np.ndarray, F: np.ndarray, region: np.ndarray | None = None):
    """ASCII STL; with `region` (one label per face) one solid per label, named <label>."""
    groups = {name: F} if region is None else {lab: F[region == lab] for lab in dict.fromkeys(region)}
    with open(path, "w") as f:
        for lab, FF in groups.items():
            _write_solid(f, lab, V, FF)


def _write_solid(f, name, V, F):
        f.write(f"solid {name}\n")
        for a, b, c in F:
            nrm = np.cross(V[b] - V[a], V[c] - V[a]); nrm /= np.linalg.norm(nrm) + 1e-300
            f.write(f" facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}\n  outer loop\n")
            for k in (a, b, c):
                f.write(f"   vertex {V[k, 0]:.9e} {V[k, 1]:.9e} {V[k, 2]:.9e}\n")
            f.write("  endloop\n endfacet\n")
        f.write(f"endsolid {name}\n")


def build(out_dir: Path, h_over_c: float = 0.211, alpha_deg: float = 1.0, flap_deg: float = 0.0,
          n_side: int = 400) -> dict:
    """Write constant/triSurface/{main,flap,endplate}.stl and return the geometry record (metres, case frame)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wing = zerihan_wing(flap_deg, alpha_deg, contour="smooth_te")
    y_ground = ground_y_zerihan(wing, h_over_c, alpha_deg)            # in c, wing frame
    dy = -y_ground * C_REF                                            # shift so that the ground is y = 0

    half = SPAN / 2
    # wing elements: from slightly below the symmetry plane (clean cut) into the endplate (watertight junction)
    z0, z1 = -0.01, half + 0.5 * ENDPLATE_T
    rec = {"h_over_c": h_over_c, "alpha_nominal_deg": alpha_deg, "flap_deflection_deg": flap_deg,
           "h_definition": "zerihan (lowest point at alpha nominal 0)", "c_ref_m": C_REF, "span_m": SPAN,
           "half_span_m": half, "endplate_t_m": ENDPLATE_T, "n_side": n_side, "surfaces": {}}
    lo = np.array([np.inf] * 3); hi = -lo
    for el in wing.elements:
        S, P = _sides(el, n_side)
        off = np.array([0.0, dy])
        V, F = _extrude_sides(S * C_REF + off, P * C_REF + off, z0, z1)
        poly = V[:, :2]
        # STL regions "fore" / "aft" (aft = last 25 % of the element chord): the aft region gets a finer
        # surface level so that the thin aft part (flap: ~35 mm at 1 mm thickness) is resolved by more than
        # one cell, without a volumetric refinement box (19% of the smoke mesh was spent in such boxes)
        le, te = el.le * C_REF + off, el.te_mid * C_REF + off
        u = (V[:, :2] - le) @ (te - le) / np.dot(te - le, te - le)
        region = np.where(u[F].min(axis=1) >= AFT_FRACTION, "aft", "fore")
        _write_stl(out_dir / f"{el.name}.stl", el.name, V, F, region)
        aft = poly[u >= AFT_FRACTION]
        te_box = [(aft.min(0) - 0.003).tolist() + [0.0], (aft.max(0) + 0.003).tolist() + [half]]
        rec["surfaces"][el.name] = {"n_tri": int(len(F)), "n_tri_aft": int((region == "aft").sum()), "x_le": float(poly[:, 0].min()), "x_te": float(poly[:, 0].max()),
                                    "y_min": float(poly[:, 1].min()), "y_max": float(poly[:, 1].max()),
                                    "te_box_m": te_box}
        lo = np.minimum(lo, V.min(0)); hi = np.maximum(hi, V.max(0))

    # endplate: rectangle defined at alpha 1 (wing frame, c units), rotated with the wing, then shifted
    rect = np.array([[EP_X0, EP_Y0], [EP_X1, EP_Y0], [EP_X1, EP_Y1], [EP_X0, EP_Y1]]) / C_REF
    pivot = np.asarray(wing.meta["incidence_pivot"])
    rect = rotate(rect, alpha_deg - 1.0, pivot) * C_REF + np.array([0.0, dy])
    V, F = _box(rect, half, half + ENDPLATE_T)
    _write_stl(out_dir / "endplate.stl", "endplate", V, F)
    rec["surfaces"]["endplate"] = {"n_tri": int(len(F)), "corners_xy_m": rect.tolist(), "z": [half, half + ENDPLATE_T],
                                   "y_min": float(rect[:, 1].min())}
    lo = np.minimum(lo, V.min(0)); hi = np.maximum(hi, V.max(0))

    lp = wing.lowest_point()
    rec.update({
        "wing_lowest_point_y_m": float((lp[1] - y_ground) * C_REF),
        "endplate_ground_clearance_m": float(rect[:, 1].min()),
        "bbox_min_m": lo.tolist(), "bbox_max_m": hi.tolist(),
        "measure_2d": wing.measure(),
        # reference quantities for forceCoeffs (half model: Aref = c * b/2)
        "lRef_m": C_REF, "Aref_half_m2": C_REF * half,
        "CofR_m": [float(wing.main.quarter_chord()[0] * C_REF), float((wing.main.quarter_chord()[1] - y_ground) * C_REF), 0.0],
        "endplate_source": "Zerihan 2001 Fig. 1c, fitted (endplate_fig1c.json); size from Sec. 2.2.2",
    })
    if rec["endplate_ground_clearance_m"] <= 0:
        raise ValueError(f"endplate touches the ground (clearance {rec['endplate_ground_clearance_m'] * 1e3:.1f} mm)")
    return rec


def check_watertight(out_dir: Path) -> dict:
    import trimesh
    res = {}
    for name in ("main", "flap", "endplate"):
        m = trimesh.load(out_dir / f"{name}.stl", force="mesh")   # merge the STL regions into one surface
        m.merge_vertices()
        res[name] = {"watertight": bool(m.is_watertight), "volume_m3": float(m.volume), "winding_ok": bool(m.is_winding_consistent)}
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--h", type=float, default=0.211)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--flap", type=float, default=0.0)
    a = ap.parse_args()
    r = build(a.out_dir, a.h, a.alpha, a.flap)
    print(json.dumps({k: v for k, v in r.items() if k != "measure_2d"}, indent=1))
    try:
        print(json.dumps(check_watertight(a.out_dir), indent=1))
    except ImportError:
        print("trimesh not installed: watertightness not checked (snappyHexMesh will report open surfaces)")
