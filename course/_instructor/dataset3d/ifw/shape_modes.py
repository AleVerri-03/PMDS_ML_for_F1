"""Shape modes of the Imperial Front Wing, applied to the tessellated CAD element by element.

Every mode acts on the spanwise sections of one element (or of a chained set), in the local chord frame of the
section: s in [0, 1] from the leading edge (most upstream node of the section) to the trailing edge (most
downstream node), n = distance from the chord line, positive upwards (away from the ground). All modes are
multiplied by the junction blend w(y) = smoothstep((y - y_j) / BLEND), y_j = innermost junction of the element
with the endplate assembly, so the nodes shared with the endplate never move and the shell stays watertight.
The chord frame is measured only where the element is a clean airfoil (y > y_j + BLEND/2) and held constant
outboard of it.

Modes (name -> unit, sign, what changes); E in main, flap1, flap2:
  rot_flap2      deg  flap 2 about its leading edge, + = trailing edge up (more incidence)
  rot_flap1      deg  flap 1 + flap 2 together about the flap-1 leading edge (slot flap1-flap2 kept)
  twist_flap2    deg  rotation growing linearly from 0 at the start of the blend to the value at the
                      symmetry plane (root vs tip incidence of flap 2)
  twist_flap1    deg  same for flap 1 + flap 2 together
  camber_E       % c  bump  -A c sin(pi s) along n: + = mean line bulges towards the ground (more camber of the
                      inverted wing); thickness unchanged (upper and lower node at the same s move together)
  camber2_E      % c  -A c sin(2 pi s): moves the camber forward (+) or aft (-)
  thick_E        -    thickness factor k: n -> n_c + [1 + (k-1) sin(pi s)] (n - n_c), n_c = local mean line
                      (leading-edge radius and trailing edge untouched)
  chord_flap2    -    chord factor k: points move along the chord by (k-1) s c (trailing edge extends)
  gap_flap2      mm   flap 2 translated along the normal of the flap-1 chord at its trailing edge (+ = wider)
  overlap_flap2  mm   flap 2 translated along the flap-1 chord (+ = more overlap, i.e. upstream)
Operating parameters (rigid motion of the whole model) stay in deform_ifw.py.
"""
from __future__ import annotations

import numpy as np

BLEND_MM = 100.0
BIN_MM = 5.0

MODES = {
    # name: (elements moved, frame element, kind)
    "rot_flap2": (("flap2",), "flap2", "rot"),
    "rot_flap1": (("flap1", "flap2"), "flap1", "rot"),
    "twist_flap2": (("flap2",), "flap2", "twist"),
    "twist_flap1": (("flap1", "flap2"), "flap1", "twist"),
    "camber_main": (("main",), "main", "camber"), "camber_flap1": (("flap1",), "flap1", "camber"),
    "camber_flap2": (("flap2",), "flap2", "camber"),
    "camber2_main": (("main",), "main", "camber2"), "camber2_flap1": (("flap1",), "flap1", "camber2"),
    "camber2_flap2": (("flap2",), "flap2", "camber2"),
    "thick_main": (("main",), "main", "thick"), "thick_flap1": (("flap1",), "flap1", "thick"),
    "thick_flap2": (("flap2",), "flap2", "thick"),
    "chord_flap2": (("flap2",), "flap2", "chord"),
    "gap_flap2": (("flap2",), "flap1", "gap"),
    "overlap_flap2": (("flap2",), "flap1", "overlap"),
}
NEUTRAL = {k: (1.0 if MODES[k][2] in ("thick", "chord") else 0.0) for k in MODES}
# order: the modes that move whole elements last, so that shape modes use the element's own chord frame
ORDER = [k for k in MODES if MODES[k][2] in ("camber", "camber2", "thick", "chord")] + \
        ["gap_flap2", "overlap_flap2", "twist_flap2", "rot_flap2", "twist_flap1", "rot_flap1"]


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def chord_frame(p_frame: np.ndarray, y_clean: float):
    """Leading- and trailing-edge lines of an element, y -> (x, z), from its nodes with y > y_clean."""
    q = p_frame[p_frame[:, 1] > y_clean]
    edges = np.arange(q[:, 1].min(), q[:, 1].max() + BIN_MM / 1e3, BIN_MM / 1e3)
    ys, le, te = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (q[:, 1] >= a) & (q[:, 1] < b)
        if m.sum() < 6:
            continue
        r = q[m]
        ys.append(0.5 * (a + b)); le.append(r[np.argmin(r[:, 0])][[0, 2]]); te.append(r[np.argmax(r[:, 0])][[0, 2]])
    return np.array(ys), np.array(le), np.array(te)


def _local(p, frame):
    ys, le, te = frame
    LE = np.column_stack([np.interp(p[:, 1], ys, le[:, 0]), np.interp(p[:, 1], ys, le[:, 1])])
    TE = np.column_stack([np.interp(p[:, 1], ys, te[:, 0]), np.interp(p[:, 1], ys, te[:, 1])])
    cv = TE - LE
    c = np.linalg.norm(cv, axis=1)
    t = cv / c[:, None]
    nv = np.column_stack([-t[:, 1], t[:, 0]])          # chord rotated +90 deg in the (x, z) plane: points up
    d = p[:, [0, 2]] - LE
    s = (d * t).sum(1) / c
    n = (d * nv).sum(1)
    return LE, t, nv, c, s, n


def _mean_line(y, s, n, nbin_s=40):
    """Local mean line n_c(y, s): midpoint between the highest and lowest node in each (y, s) cell of a grid
    (BIN_MM x 1/nbin_s), filled along s where a cell is empty, then interpolated bilinearly to the nodes
    (a piecewise-constant value made steps between cells, which tilted the smallest triangles)."""
    from scipy.interpolate import RegularGridInterpolator
    y0, ny = y.min(), int(np.ceil((y.max() - y.min()) / (BIN_MM / 1e3))) + 1
    iy = np.clip(((y - y0) / (BIN_MM / 1e3)).astype(int), 0, ny - 1)
    is_ = np.clip((np.clip(s, 0, 1) * nbin_s).astype(int), 0, nbin_s - 1)
    hi = np.full((ny, nbin_s), -np.inf); lo = np.full((ny, nbin_s), np.inf)
    np.maximum.at(hi, (iy, is_), n); np.minimum.at(lo, (iy, is_), n)
    with np.errstate(invalid="ignore"):
        g = 0.5 * (hi + lo)                   # inf - inf in empty cells -> nan, filled below
    ok = np.isfinite(g)
    sc = (np.arange(nbin_s) + 0.5) / nbin_s
    for i in range(ny):                       # fill empty cells along s, then empty rows from neighbours
        if ok[i].sum() >= 2:
            g[i] = np.interp(sc, sc[ok[i]], g[i, ok[i]])
        else:
            g[i] = np.nan
    good = ~np.isnan(g[:, 0])
    yc = y0 + (np.arange(ny) + 0.5) * BIN_MM / 1e3
    for j in range(nbin_s):
        g[:, j] = np.interp(yc, yc[good], g[good, j])
    f = RegularGridInterpolator((yc, sc), g, bounds_error=False, fill_value=None)
    return f(np.column_stack([y, np.clip(s, 0, 1)]))


HANGER_BLEND_MM = 60.0


def apply(xyz0: np.ndarray, node_el: dict, junction: dict, fixed: np.ndarray, params: dict,
          hanger_band: tuple | None = None) -> np.ndarray:
    """hanger_band = (y_centre, half_width) in m of the hanger root on the main plane: modes acting on the main
    are blended to zero there too (smoothstep over HANGER_BLEND_MM), because the hanger STL does not move and
    its root sits in a hole of the main-plane upper surface (found 05/11/2026: a 20 % thickness change of the
    main opened a 3 mm step at the hanger root)."""
    xyz = xyz0.copy()
    L = BLEND_MM / 1e3
    for name in ORDER:
        val = params.get(name, NEUTRAL[name])
        if val == NEUTRAL[name]:
            continue
        moved, frame_el, kind = MODES[name]
        sel = np.zeros(len(xyz), bool)
        for e in moved:
            sel[node_el[e]] = True
        sel &= ~fixed
        y_j = max(junction[e]["y_mm"][1] for e in moved) / 1e3
        fr = np.zeros(len(xyz), bool)
        fr[node_el[frame_el]] = True
        frame = chord_frame(xyz[fr & ~fixed], y_j + 0.5 * L)
        p = xyz[sel]
        w = smoothstep((p[:, 1] - y_j) / L)
        if "main" in moved and hanger_band is not None:
            yc, hw = hanger_band
            w = w * smoothstep((np.abs(p[:, 1] - yc) - hw) / (HANGER_BLEND_MM / 1e3))
        LE, t, nv, c, s, n = _local(p, frame)
        if kind in ("rot", "twist"):
            ang = np.radians(val) * w
            if kind == "twist":                              # full value at the root (y = 0), 0 at y_j + L
                ang = ang * np.clip((p[:, 1] - (y_j + L)) / (0.0 - (y_j + L)), 0.0, 1.0)
            th = -ang                                        # rotation about +y by -angle: trailing edge up
            dx, dz = p[:, 0] - LE[:, 0], p[:, 2] - LE[:, 1]
            p[:, 0] = LE[:, 0] + dx * np.cos(th) + dz * np.sin(th)
            p[:, 2] = LE[:, 1] - dx * np.sin(th) + dz * np.cos(th)
        elif kind in ("camber", "camber2"):
            sc = np.clip(s, 0, 1)
            shape = np.sin(np.pi * sc) if kind == "camber" else np.sin(2 * np.pi * sc)
            dn = -(val / 100.0) * c * shape * w
            p[:, 0] += dn * nv[:, 0]; p[:, 2] += dn * nv[:, 1]
        elif kind == "thick":
            nc = _mean_line(p[:, 1], s, n)
            k = 1 + (val - 1) * np.sin(np.pi * np.clip(s, 0, 1)) * w
            dn = (k - 1) * (n - nc)
            p[:, 0] += dn * nv[:, 0]; p[:, 2] += dn * nv[:, 1]
        elif kind == "chord":
            ds = (val - 1) * np.clip(s, 0, 1) * c * w
            p[:, 0] += ds * t[:, 0]; p[:, 2] += ds * t[:, 1]
        elif kind in ("gap", "overlap"):
            # frame of flap 1 at its trailing edge: chord direction t, normal nv (taken at each node's y)
            dvec = (nv if kind == "gap" else -t) * (val / 1e3) * w[:, None]
            p[:, 0] += dvec[:, 0]; p[:, 2] += dvec[:, 1]
        xyz[sel] = p
    return xyz


def checks(xyz0, xyz, tri, node_el, min_slot_mm=1.0, max_fold_deg=60.0, hanger_band=None) -> dict:
    """Slots between consecutive elements (minimum node distance, inboard of the endplate region) and folds
    (rotation of the triangle normals, triangles with an edge below 0.05 mm ignored; invalid if any triangle
    flips or rotates by more than max_fold_deg)."""
    from scipy.spatial import cKDTree
    out = {}
    for pa, pb in (("main", "flap1"), ("flap1", "flap2")):
        A, B = node_el[pa], node_el[pb]
        inb = lambda idx, X: idx[X[idx, 1] > -0.55]
        f = lambda X: 1e3 * float(cKDTree(X[inb(B, X)]).query(X[inb(A, X)])[0].min())
        out[f"slot_{pa}_{pb}_mm"] = {"before": f(xyz0), "after": f(xyz)}

    def normals(X):
        v = X[tri]
        nn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return nn / (np.linalg.norm(nn, axis=1)[:, None] + 1e-300)
    v0 = xyz0[tri]
    emin = np.linalg.norm(v0[:, [1, 2, 0]] - v0, axis=2).min(1)
    moved = (np.linalg.norm(xyz[tri] - v0, axis=2).max(1) > 0) & (emin > 5e-5)   # ignore slivers < 0.05 mm
    cosang = (normals(xyz0)[moved] * normals(xyz)[moved]).sum(1) if moved.any() else np.ones(1)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    out["max_normal_rotation_deg"] = float(ang.max())
    out["p999_normal_rotation_deg"] = float(np.percentile(ang, 99.9))
    out["flipped_triangles"] = int((ang > 90).sum())
    if hanger_band is not None:                      # the main plane must not move under the hanger root
        yc, hw = hanger_band
        mn = node_el["main"]
        under = mn[np.abs(xyz0[mn, 1] - yc) <= hw]
        out["hanger_root_motion_mm"] = float(1e3 * np.linalg.norm(xyz[under] - xyz0[under], axis=1).max()) if len(under) else 0.0
    out["valid_slots"] = all(v["after"] >= min_slot_mm for k, v in out.items() if k.startswith("slot"))
    out["valid_folds"] = out["flipped_triangles"] == 0 and out["max_normal_rotation_deg"] <= max_fold_deg
    out["valid_hanger"] = out.get("hanger_root_motion_mm", 0.0) <= 0.05
    out["max_node_displacement_mm"] = float(1e3 * np.linalg.norm(xyz - xyz0, axis=1).max())
    return out
