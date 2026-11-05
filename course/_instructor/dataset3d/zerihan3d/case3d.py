"""Write an OpenFOAM case (snappyHexMesh + simpleFoam k-omega SST, wall functions) for the 3D Zerihan wing.

Half model over a moving ground. Physics as in the 2D dataset (Re 7.5e5 on c = 0.38 m, U = 30 m/s,
k-omega SST, Tu 0.2 %, nu_t/nu 10 at the inlet), EXCEPT the wall treatment: the 2D cases resolve the
viscous sublayer (y+ ~ 1, nutLowReWallFunction); here the first cell is at y+ ~ 30-150 and the walls use
nutUSpaldingWallFunction. A resolved 3D boundary layer would cost one to two orders of magnitude more cells.

Mesh levels (background cell size d0; surface refinement levels are the same, so every level is the
previous one with all cells scaled):
  smoke : d0 = 0.10 m   pipeline check (run in the cloud sandbox: cell count and cost per cell-iteration)
  L0    : d0 = 0.075 m  the feasibility trial on the desktop (cells x ~2.4 with respect to smoke)
  L1    : d0 = 0.05 m   optional, one refinement step (cells x ~3.4 with respect to L0) for a first
                        mesh-sensitivity number
Surface cells: level 5 = d0/32 (L0: 2.3 mm), level 6 on curvature and on the aft 25 % of each element
(L0: 1.2 mm), level 7 on the trailing-edge feature lines (L0: 0.59 mm).
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from geometry3d import ENDPLATE_T, build

LEVELS = {"smoke": 0.10, "L0": 0.075, "L1": 0.05}

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass
class Params3D:
    h_over_c: float = 0.211          # Zerihan definition (same as the 2D reference case)
    alpha_deg: float = 1.0
    flap_deg: float = 0.0
    level: str = "L0"
    U: float = 30.0
    c: float = 0.38
    Re: float = 7.5e5
    Tu: float = 0.002
    nut_ratio: float = 10.0
    n_procs: int = 6
    n_iter: int = 3000               # hard cap of SIMPLE iterations
    write_every: int = 100            # = autostop3d.CHECK_EVERY: the state at every checkpoint is on disk
    # domain (m), ground at y = 0, symmetry plane z = 0, wing LE near x = 0
    x_min: float = -1.5
    x_max: float = 4.0
    y_max: float = 2.0
    z_max: float = 1.75
    n_layers: int = 3
    of_version: str = "v2606"        # recorded only ("v1912" = the cloud syntax check)

    @property
    def nu(self):
        return self.U * self.c / self.Re

    @property
    def k_in(self):
        return 1.5 * (self.Tu * self.U) ** 2

    @property
    def omega_in(self):
        return self.k_in / (self.nut_ratio * self.nu)


def _w(path: Path, cls: str, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=path.name) + body)


WALLS = ("main", "flap", "endplate")
# snappy names the patches <surface>_<region>: main_fore, main_aft, flap_fore, flap_aft, endplate
# explicit names in function objects (a regex there breaks the function-object digest in v1912)
PATCHES = {"main": "main_fore main_aft", "flap": "flap_fore flap_aft", "endplate": "endplate"}


def _field(internal: tuple, inlet: str, outlet: str, far: str, ground: str, wall: str) -> str:
    bf = {"inlet": inlet, "outlet": outlet, "top": far, "side": far, "symmetry": "type symmetryPlane;",
          "ground": ground, '"(main|flap|endplate).*"': wall}
    body = f"dimensions {internal[0]};\n\ninternalField {internal[1]};\n\nboundaryField\n{{\n"
    body += '    #includeEtc "caseDicts/setConstraintTypes"\n'
    for patch, bc in bf.items():
        body += f"    {patch}\n    {{\n        {bc}\n    }}\n"
    return body + "}\n"


def _box(lo, hi):
    return f"type searchableBox; min ({lo[0]:.5f} {lo[1]:.5f} {lo[2]:.5f}); max ({hi[0]:.5f} {hi[1]:.5f} {hi[2]:.5f});"


def write_case(case: Path, p: Params3D) -> dict:
    if case.exists():
        raise FileExistsError(f"{case} exists: choose a new case directory (nothing is overwritten)")
    geo = build(case / "constant/triSurface", p.h_over_c, p.alpha_deg, p.flap_deg)
    d0 = LEVELS[p.level]
    nx = round((p.x_max - p.x_min) / d0); ny = round(p.y_max / d0); nz = round(p.z_max / d0)
    U, k, om = p.U, p.k_in, p.omega_in
    half = geo["half_span_m"]
    wy = geo["surfaces"]["flap"]["y_max"]     # top of the wing (flap TE region)

    # ------------------------------------------------------------------ fields
    vec = f"uniform ({U} 0 0)"
    wf_k = "type kqRWallFunction; value uniform 1e-10;"
    wf_om = f"type omegaWallFunction; value uniform {om:.6e};"
    wf_nut = "type nutUSpaldingWallFunction; value uniform 0;"
    _w(case / "0.orig/U", "volVectorField", _field(("[0 1 -1 0 0 0 0]", vec), f"type fixedValue; value {vec};",
       f"type inletOutlet; inletValue uniform (0 0 0); value {vec};", "type slip;",
       f"type fixedValue; value {vec};   // moving ground at the free-stream speed", "type noSlip;"))
    _w(case / "0.orig/p", "volScalarField", _field(("[0 2 -2 0 0 0 0]", "uniform 0"), "type zeroGradient;",
       "type fixedValue; value uniform 0;", "type slip;", "type zeroGradient;", "type zeroGradient;"))
    _w(case / "0.orig/k", "volScalarField", _field(("[0 2 -2 0 0 0 0]", f"uniform {k:.6e}"),
       f"type fixedValue; value uniform {k:.6e};", f"type inletOutlet; inletValue uniform {k:.6e}; value uniform {k:.6e};",
       "type slip;", wf_k, wf_k))
    _w(case / "0.orig/omega", "volScalarField", _field(("[0 0 -1 0 0 0 0]", f"uniform {om:.6e}"),
       f"type fixedValue; value uniform {om:.6e};", f"type inletOutlet; inletValue uniform {om:.6e}; value uniform {om:.6e};",
       "type slip;", wf_om, wf_om))
    _w(case / "0.orig/nut", "volScalarField", _field(("[0 2 -1 0 0 0 0]", "uniform 0"), "type calculated; value uniform 0;",
       "type calculated; value uniform 0;", "type slip;", wf_nut, wf_nut))

    _w(case / "constant/transportProperties", "dictionary", f"transportModel Newtonian;\n\nnu {p.nu:.8e};\n")
    _w(case / "constant/turbulenceProperties", "dictionary",
       "simulationType RAS;\n\nRAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs on;\n}\n")

    # ------------------------------------------------------------------ mesh
    _w(case / "system/blockMeshDict", "dictionary", f"""scale 1;

vertices
(
    ({p.x_min} 0 0) ({p.x_max} 0 0) ({p.x_max} {p.y_max} 0) ({p.x_min} {p.y_max} 0)
    ({p.x_min} 0 {p.z_max}) ({p.x_max} 0 {p.z_max}) ({p.x_max} {p.y_max} {p.z_max}) ({p.x_min} {p.y_max} {p.z_max})
);

blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );

boundary
(
    inlet    {{ type patch;         faces ((0 4 7 3)); }}
    outlet   {{ type patch;         faces ((1 2 6 5)); }}
    ground   {{ type wall;          faces ((0 1 5 4)); }}
    top      {{ type patch;         faces ((3 7 6 2)); }}
    symmetry {{ type symmetryPlane; faces ((0 3 2 1)); }}
    side     {{ type patch;         faces ((4 5 6 7)); }}
);
""")
    _w(case / "system/surfaceFeatureExtractDict", "dictionary", "".join(
        f"{n}.stl\n{{\n    extractionMethod extractFromSurface;\n    includedAngle 150;\n    subsetFeatures {{ nonManifoldEdges no; openEdges yes; }}\n    writeObj no;\n}}\n"
        for n in WALLS))

    x_le, x_te = geo["bbox_min_m"][0], geo["bbox_max_m"][0]
    # refinement boxes (m): around the wing incl. the ground gap; wake; tip-vortex/endplate region
    near_lo, near_hi = (x_le - 0.06, 0.0, 0.0), (x_te + 0.08, wy + 0.08, half + 0.06)
    wake_lo, wake_hi = (x_te - 0.05, 0.0, 0.0), (x_te + 1.2, wy + 0.25, half + 0.25)
    far_lo, far_hi = (x_le - 0.4, 0.0, 0.0), (x_te + 2.5, wy + 0.6, half + 0.6)
    # gap between the endplate lower edge and the ground, where the tip vortex forms (17.5 mm at h/c 0.079):
    # level 5 (L0: 2.3 mm), from just ahead of the endplate to 0.1 m behind it, +-0.05 m across it
    ep = np.array(geo["surfaces"]["endplate"]["corners_xy_m"])
    gap_lo = (ep[:, 0].min() - 0.02, 0.0, half - 0.05)
    gap_hi = (ep[:, 0].max() + 0.10, ep[:, 1].min() + 0.01, half + ENDPLATE_T + 0.05)
    _w(case / "system/snappyHexMeshDict", "dictionary", f"""castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    main.stl     {{ type triSurfaceMesh; name main; }}
    flap.stl     {{ type triSurfaceMesh; name flap; }}
    endplate.stl {{ type triSurfaceMesh; name endplate; }}
    nearWing     {{ {_box(near_lo, near_hi)} }}
    wake         {{ {_box(wake_lo, wake_hi)} }}
    farWake      {{ {_box(far_lo, far_hi)} }}
    endplateGap  {{ {_box(gap_lo, gap_hi)} }}
}};

castellatedMeshControls
{{
    maxLocalCells       20000000;
    maxGlobalCells      80000000;
    minRefinementCells  500;   // stops the long tail of shell iterations adding ~250 cells each
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;

    features
    (
        {{ file "main.eMesh";     level 7; }}
        {{ file "flap.eMesh";     level 7; }}
        {{ file "endplate.eMesh"; level 5; }}
    );

    refinementSurfaces
    {{
        main     {{ level (5 6); patchInfo {{ type wall; }} regions {{ aft {{ level (6 6); patchInfo {{ type wall; }} }} }} }}
        flap     {{ level (5 6); patchInfo {{ type wall; }} regions {{ aft {{ level (6 6); patchInfo {{ type wall; }} }} }} }}
        endplate {{ level (4 5); patchInfo {{ type wall; }} }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        nearWing {{ mode inside; levels ((1e15 3)); }}
        wake     {{ mode inside; levels ((1e15 2)); }}
        farWake  {{ mode inside; levels ((1e15 1)); }}
        endplateGap {{ mode inside; levels ((1e15 5)); }}
    }}

    // a point in the fluid: above the wing, inboard, away from every surface
    locationInMesh ({x_le - 0.3:.4f} {wy + 0.3:.4f} {0.5 * half:.4f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers
    {{
        "(main|flap|endplate).*" {{ nSurfaceLayers {p.n_layers}; }}
    }}
    expansionRatio      1.2;
    finalLayerThickness 0.5;
    minThickness        0.1;
    nGrow               0;
    featureAngle        130;
    slipFeatureAngle    30;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    #includeEtc "caseDicts/meshQualityDict"
    nSmoothScale    4;
    errorReduction  0.75;
}}

writeFlags ( scalarLevels layerSets layerFields );
mergeTolerance 1e-6;
""")

    # ------------------------------------------------------------------ numerics
    _w(case / "system/fvSchemes", "dictionary", """ddtSchemes { default steadyState; }

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(omega)     cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;     // first order for k, omega as in the OpenFOAM external-aero tutorials
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div(div(phi,U)) Gauss linear;
}

laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.33; }
wallDist { method meshWave; }
""")
    _w(case / "system/fvSolution", "dictionary", """solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }
    Phi
    {
        $p;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

potentialFlow
{
    nNonOrthogonalCorrectors 3;
}

SIMPLE
{
    // robust SIMPLE (the 2D "robust" preset): snappy meshes reach ~65 deg non-orthogonality
    nNonOrthogonalCorrectors 1;
    consistent      no;
    residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-6; }
}

relaxationFactors
{
    fields      { p 0.3; }
    equations   { U 0.7; "(k|omega)" 0.5; }
}
""")

    lRef, Aref = geo["lRef_m"], geo["Aref_half_m2"]
    cofr = geo["CofR_m"]

    def fc(name, patches):
        return f"""    {name}
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         ({' '.join(PATCHES[x] for x in patches)});
        rho             rhoInf;
        rhoInf          1;
        CofR            ({cofr[0]:.6f} {cofr[1]:.6f} 0);
        dragDir         (1 0 0);
        liftDir         (0 -1 0);   // positive CL = DOWNFORCE (Zerihan)
        pitchAxis       (0 0 -1);
        magUInf         {U};
        lRef            {lRef};     // c = 0.38 m
        Aref            {Aref:.6f}; // c * b/2 (half model): CL comparable with the balance data of the full wing
    }}
"""
    # spanwise load distribution (binField) left out on purpose: its dictionary changed across recent versions
    # and could not be tested here; it can be computed afterwards in ParaView from the wall pressure
    bin_fo = ""
    _w(case / "system/controlDict", "dictionary", f"""application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {p.n_iter};
deltaT          1;
writeControl    timeStep;
writeInterval   {p.write_every};
purgeWrite      2;
writeFormat     binary;
writePrecision  10;
timePrecision   6;
runTimeModifiable true;

functions
{{
{fc('coeffs_total', WALLS)}{fc('coeffs_wing', ['main', 'flap'])}{fc('coeffs_main', ['main'])}{fc('coeffs_flap', ['flap'])}{fc('coeffs_endplate', ['endplate'])}
    residuals {{ type solverInfo; libs (utilityFunctionObjects); fields (U p k omega); writeResidualFields no; }}
    yPlus     {{ type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }}
    midspan
    {{
        // surface pressure on the walls at the end (Cp at mid-span is cut from here, vs the 2D case and Fig. 8)
        type            surfaces;
        libs            (sampling);
        writeControl    writeTime;
        surfaceFormat   raw;
        fields          (p);
        interpolationScheme cellPoint;
        surfaces
        {{
            walls {{ type patch; patches (main_fore main_aft flap_fore flap_aft); interpolate false; }}
        }}
    }}
{bin_fo}}}
""")
    # hierarchical (geometric, deterministic) as in the 2D pipeline; split along x (long) and z (span)
    nz_ = max(d for d in range(1, int(p.n_procs ** 0.5) + 1) if p.n_procs % d == 0)
    _w(case / "system/decomposeParDict", "dictionary",
       f"numberOfSubdomains {p.n_procs};\n\nmethod hierarchical;\n\ncoeffs\n{{\n    n ({p.n_procs // nz_} 1 {nz_});\n    order xzy;\n}}\n")
    (case / "case.foam").touch()

    meta = {"params": asdict(p), "d0_m": d0, "background_cells": nx * ny * nz, "geometry": geo,
            "nu": p.nu, "k_in": k, "omega_in": om}
    (case / "case3d.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(Path(__file__).resolve().parent / "Allrun", case / "Allrun")
    (case / "Allrun").chmod(0o755)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--level", choices=list(LEVELS), default="L0")
    ap.add_argument("--h", type=float, default=0.211)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--flap", type=float, default=0.0)
    ap.add_argument("--np", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--of-version", default="v2606")
    a = ap.parse_args()
    m = write_case(a.case, Params3D(h_over_c=a.h, alpha_deg=a.alpha, flap_deg=a.flap, level=a.level, n_procs=a.np,
                                    n_iter=a.n_iter, of_version=a.of_version))
    print(f"case {a.case}: level {a.level}, d0 {m['d0_m']} m, background {m['background_cells']} cells, "
          f"endplate clearance {m['geometry']['endplate_ground_clearance_m'] * 1e3:.1f} mm")
