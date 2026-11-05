"""Imperial Front Wing (IFW): STEP -> binary STL surfaces for snappyHexMesh + geometry record.

Source: Buscariolo F. F., "Imperial Front Wing (IFW) geometry", Imperial College London HPC repository,
DOI 10.14469/hpc/6049, CC-BY 4.0. File ifw.stp = CATIA export "IFW_HC036_NOWHEEL_V4" (2018), in mm.

What the STEP contains (inspected 04/11/2026): 209 trimmed faces, no solids, as open shells:
  * the CFD domain box used at Imperial (x -7.53...12.03 m, y -3.05...0, z 0.004...3.05 m) and two auxiliary planar
    patches (on the ground and on the symmetry plane): NOT used, our domain is built by blockMesh;
  * the model, half car at y <= 0 (symmetry at y = 0): the front wing assembly (170 faces: three elements,
    canard, Gurney flaps, endplate, footplate, all in one connected shell), the nose cone (9 faces) and the
    hanger joining them (22 faces). The shells are open only on the symmetry plane and at the hanger junctions
    (geometric, not topological, contact): the union is closed up to sub-millimetre gaps, closed in practice
    by the symmetry plane of the background mesh and by snappyHexMesh.
The model is already in the test position: ground at z = 4.066 mm, footplate trailing-edge clearance 89.9 mm
(h/c = 0.36 with c = 250 mm), pitch 1.08 deg measured on the footplate lower edge (paper: 1.094 deg).

Output frame: the CAD frame (x downstream, y spanwise, z up) in METRES, shifted so that the ground is z = 0.
Usage: python prepare_ifw.py <ifw.stp> <out_dir> [--size-mm 2.5]
Writes <out_dir>/{wing,nose,hanger}.stl (binary), ifw_geometry.json and wing_mesh.npz (the wing tessellation
with the CAD face tag of every triangle, input of classify_elements.py / deform_ifw.py).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

Z_GROUND_MM = 4.06596960867      # domain box bottom in the CAD (= ground), from the STEP
C_MAIN = 0.25                    # m, main-element chord (paper)
SPAN_HALF = 0.70                 # m, W_s in the paper (reference area c * W_s for the half model)


def _components(gmsh):
    """Faces grouped by shared edges."""
    faces = [t for _, t in gmsh.model.getEntities(2)]
    e2f = defaultdict(set)
    for f in faces:
        for c in gmsh.model.getAdjacencies(2, f)[1]:
            e2f[c].add(f)
    parent = {f: f for f in faces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for fs in e2f.values():
        fs = list(fs)
        for o in fs[1:]:
            parent[find(o)] = find(fs[0])
    comp = defaultdict(list)
    for f in faces:
        comp[find(f)].append(f)
    return list(comp.values())


def _bbox(gmsh, faces):
    b = np.array([gmsh.model.getBoundingBox(2, f) for f in faces])
    return np.r_[b[:, :3].min(0), b[:, 3:].max(0)]


def classify(gmsh) -> dict[str, list[int]]:
    """Name the model components by their extent (mm); drop the domain box and the auxiliary planar patches."""
    out = {}
    for faces in _components(gmsh):
        bb = _bbox(gmsh, faces)
        size = bb[3:] - bb[:3]
        if size.max() > 5000 or size.min() < 1e-3:          # domain box faces, flat auxiliary patches
            continue
        if len(faces) > 100:
            out["wing"] = faces
        elif bb[2] > 200:                                     # highest part: nose cone
            out["nose"] = faces
        else:
            out["hanger"] = faces
    if set(out) != {"wing", "nose", "hanger"}:
        raise RuntimeError(f"unexpected STEP content: found {sorted(out)}")
    return out


def write_stl_binary(path: Path, name: str, xyz: np.ndarray, tri: np.ndarray):
    v = xyz[tri]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None] + 1e-300
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size-mm", type=float, default=2.5, help="maximum triangle size (curvature refines below)")
    a = ap.parse_args()
    import gmsh
    a.out.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(a.step))
    gmsh.model.occ.synchronize()
    parts = classify(gmsh)
    keep = {f for fs in parts.values() for f in fs}
    gmsh.model.removeEntities([(2, t) for _, t in gmsh.model.getEntities(2) if t not in keep])
    gmsh.option.setNumber("Mesh.MeshSizeMax", a.size_mm)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)    # >= 36 segments per full turn of curvature
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    nodes, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    idx = np.full(int(nodes.max()) + 1, -1)
    idx[nodes.astype(int)] = np.arange(len(nodes))
    xyz_m = (xyz - np.array([0, 0, Z_GROUND_MM])) / 1000.0
    rec = {"source": "ifw.stp (IFW_HC036_NOWHEEL_V4), DOI 10.14469/hpc/6049, CC-BY 4.0",
           "frame": "CAD frame in metres, ground at z = 0, symmetry plane y = 0, model at y < 0, flow +x",
           "z_ground_cad_mm": Z_GROUND_MM, "triangle_size_max_mm": a.size_mm, "parts": {}}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    tris = {}
    for name, faces in parts.items():
        tri, ftag = [], []
        for f in faces:
            types, _, conn = gmsh.model.mesh.getElements(2, f)
            for ty, cn in zip(types, conn):
                if ty == 2:
                    t = idx[cn.astype(int)].reshape(-1, 3)
                    tri.append(t)
                    ftag.append(np.full(len(t), f))
        tri = tris[name] = np.vstack(tri)
        if name == "wing":   # tessellation with CAD face tags, for the element-wise deformation (deform_ifw.py)
            used = np.unique(tri)
            remap = np.full(len(xyz_m), -1)
            remap[used] = np.arange(len(used))
            np.savez_compressed(a.out / "wing_mesh.npz", xyz=xyz_m[used], tri=remap[tri], face=np.concatenate(ftag))
        write_stl_binary(a.out / f"{name}.stl", name, xyz_m, tri)
        p = xyz_m[np.unique(tri)]
        e = np.linalg.norm(xyz_m[tri[:, [1, 2, 0]]] - xyz_m[tri], axis=2)
        rec["parts"][name] = {"faces_cad": len(faces), "triangles": int(len(tri)), "bbox_min_m": p.min(0).tolist(),
                              "bbox_max_m": p.max(0).tolist(), "edge_min_mm": float(e.min() * 1e3),
                              "edge_median_mm": float(np.median(e) * 1e3)}
        lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
    # ride height and pitch, measured on the wing assembly (lower edge of endplate/footplate, outboard part)
    w = xyz_m[np.unique(tris["wing"])]
    ob = w[w[:, 1] < -0.55]
    xs = np.linspace(ob[:, 0].min(), ob[:, 0].max(), 12)
    zmin = np.array([ob[(ob[:, 0] >= x0) & (ob[:, 0] < x1), 2].min() for x0, x1 in zip(xs[:-1], xs[1:])])
    xc = 0.5 * (xs[:-1] + xs[1:])
    slope = np.polyfit(xc, zmin, 1)[0]
    rec.update({"bbox_model_min_m": lo.tolist(), "bbox_model_max_m": hi.tolist(),
                "h_te_m": float(zmin[-1]), "h_te_over_c": float(zmin[-1] / C_MAIN),
                "pitch_deg_lower_edge": float(np.degrees(np.arctan(slope))),
                "c_ref_m": C_MAIN, "span_half_m": SPAN_HALF, "Aref_m2": C_MAIN * SPAN_HALF})
    (a.out / "ifw_geometry.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "parts"}, indent=1))
    for k, v in rec["parts"].items():
        print(k, v["triangles"], "triangles, edge min/median %.3f/%.2f mm" % (v["edge_min_mm"], v["edge_median_mm"]))
    gmsh.finalize()


if __name__ == "__main__":
    main()
