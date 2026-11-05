"""Split the IFW wing shell (one connected CAD shell of 170 faces) into its elements.

Method (deterministic, no hand-picked tags):
  1. smooth patches: faces joined across shared edges whose dihedral angle is below 40 deg;
  2. the three patches that run from the symmetry plane (y = 0) over more than 300 mm of span are the main
     plane, flap 1 and flap 2, ordered by streamwise position;
  3. the largest patch confined to the outboard region (CAD box y < -400 mm) is the endplate assembly (endplate,
     footplate, canard, fillets);
  4. every other patch (trailing-edge bases, Gurney strips, footplate plates) joins the element patch with
     which it shares the longest edge length; patches touching no element join the endplate assembly.
  5. junctions: the edges shared by an element and the endplate assembly; their y range is recorded, because
     the parametric deformation must vanish there (the elements are attached to the endplate only).
Result on ifw.stp (04/11/2026): main 12 faces, flap 1 20, flap 2 20, endplate assembly 118; junctions at
y = -631...-600 mm (main), -601...-585 (flap 1), -586...-569 (flap 2). No edge is shared between elements.

Usage: python classify_elements.py <ifw.stp> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

ANGLE_DEG = 40.0


def main():
    import gmsh
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from prepare_ifw import classify
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(sys.argv[1])
    gmsh.model.occ.synchronize()
    wing = [int(f) for f in classify(gmsh)["wing"]]
    e2f = defaultdict(list)
    for f in wing:
        for c in gmsh.model.getAdjacencies(2, f)[1]:
            e2f[abs(int(c))].append(f)

    def normal_at(f, p):
        par = gmsh.model.getClosestPoint(2, f, p)[1]
        n = np.array(gmsh.model.getNormal(f, par))
        return n / (np.linalg.norm(n) + 1e-300)

    parent = {f: f for f in wing}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edge_len, shared = {}, {}
    for c, fs in e2f.items():
        if len(fs) != 2:
            continue
        lo, hi = gmsh.model.getParametrizationBounds(1, c)
        pm = gmsh.model.getValue(1, c, [0.5 * (lo[0] + hi[0])])
        ang = np.degrees(np.arccos(np.clip(abs(normal_at(fs[0], pm) @ normal_at(fs[1], pm)), -1, 1)))
        edge_len[c], shared[c] = gmsh.model.occ.getMass(1, c), tuple(fs)
        if ang < ANGLE_DEG:
            parent[find(fs[0])] = find(fs[1])
    patches = defaultdict(list)
    for f in wing:
        patches[find(f)].append(f)
    patches = list(patches.values())
    bbox = {i: np.r_[np.min([gmsh.model.getBoundingBox(2, f)[:3] for f in p], 0),
                     np.max([gmsh.model.getBoundingBox(2, f)[3:] for f in p], 0)] for i, p in enumerate(patches)}
    area = {i: sum(gmsh.model.occ.getMass(2, f) for f in p) for i, p in enumerate(patches)}
    span = [i for i in bbox if bbox[i][4] > -1.0 and bbox[i][4] - bbox[i][1] > 300]
    span = sorted(sorted(span, key=lambda i: -area[i])[:3], key=lambda i: bbox[i][0] + bbox[i][3])
    outb = [i for i in bbox if bbox[i][4] < -400 and i not in span]   # (OCC boxes are conservative)
    ep = max(outb, key=lambda i: area[i])
    owner = {span[0]: "main", span[1]: "flap1", span[2]: "flap2", ep: "endplate"}
    face_patch = {f: i for i, p in enumerate(patches) for f in p}
    for i in range(len(patches)):
        if i in owner:
            continue
        touch = defaultdict(float)
        for c, (a, b) in shared.items():
            pa, pb = face_patch[a], face_patch[b]
            if i in (pa, pb):
                other = pb if pa == i else pa
                if other in owner and owner[other] != "endplate":
                    touch[owner[other]] += edge_len[c]
        owner[i] = max(touch, key=touch.get) if touch else "endplate"
    elements = defaultdict(list)
    for i, p in enumerate(patches):
        elements[owner[i]] += p
    face_el = {f: n for n, fs in elements.items() for f in fs}
    for i in span + [ep]:
        print("seed", owner[i], len(patches[i]), "faces bbox", np.round(bbox[i]).tolist(), file=sys.stderr)
    for k, v in elements.items():
        print(k, len(v), "faces", file=sys.stderr)
    junction = {}
    for n in ("main", "flap1", "flap2"):
        cs = [c for c, (a, b) in shared.items() if {face_el[a], face_el[b]} == {n, "endplate"}]
        bb = np.array([gmsh.model.getBoundingBox(1, c) for c in cs])
        junction[n] = {"edges": len(cs), "y_mm": [float(bb[:, 1].min()), float(bb[:, 4].max())]}
    cross = [c for c, (a, b) in shared.items() if face_el[a] != face_el[b] and "endplate" not in (face_el[a], face_el[b])]
    out = {"method": __doc__.split("\n\n")[1], "elements": {k: sorted(v) for k, v in elements.items()},
           "junction": junction, "edges_shared_between_elements": len(cross)}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for k, v in elements.items():
        print(k, len(v), "faces")
    print(json.dumps(junction), "edges shared between elements:", len(cross))


if __name__ == "__main__":
    main()
