"""OpenFOAM case (snappyHexMesh + simpleFoam k-omega SST, wall functions) for the Imperial Front Wing.

Same physics recipe and numerics as the 3D Zerihan kit (../zerihan3d/case3d.py), CAD frame (z up):
  half model, symmetry plane y = 0, model at y < 0, moving ground z = 0 at U, flow +x;
  Re = 2.2e5 on the main-element chord c = 0.25 m with U = 25 m/s (Buscariolo et al. 2022, the reference
  Nektar++ iLES/uDNS: CL -5.54, CD 0.575 with Aref = c * W_s = 0.25 * 0.70 m2, NM54);
  walls: nutUSpaldingWallFunction, 3 layers (y+ ~ 30-150, NOT resolved as in the iLES);
  geometry = STL from prepare_ifw.py (wing assembly, nose, hanger), already in the test position
  (footplate TE clearance h/c = 0.356, pitch 1.09 deg).

Levels (background cell d0; wing surface level 5..6 = d0/32..d0/64, feature lines level 7 = d0/128):
  smoke d0 = 0.14 m (wing 4.4/2.2 mm, edges 1.1 mm)   pipeline check
  L0    d0 = 0.10 m (wing 3.1/1.6 mm, edges 0.8 mm)   first look at the model, laptop
  L1    d0 = 0.07 m (wing 2.2/1.1 mm, edges 0.55 mm)  refinement step
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

LEVELS = {"smoke": 0.14, "L0": 0.10, "L1": 0.07}
PARTS = ("wing", "nose", "hanger")
REFERENCE = {"CL": 5.54, "CD": 0.575, "sign": "CL > 0 = downforce (paper: Cl = -5.54)",
             "source": "Buscariolo et al., J. Wind Eng. Ind. Aerodyn. 2022 (arXiv 1909.06701), Nektar++ NM54, "
                       "Re 2.2e5, Aref = 0.25 x 0.70 m2; patches included in the paper's coefficient not stated"}

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass
class ParamsIFW:
    level: str = "L0"
    U: float = 25.0
    c: float = 0.25
    Re: float = 2.2e5
    Tu: float = 0.002
    nut_ratio: float = 10.0
    n_procs: int = 6
    n_iter: int = 3000
    write_every: int = 100                 # = autostop3d.CHECK_EVERY
    x_min: float = -3.0
    x_max: float = 4.5
    y_min: float = -2.2
    z_max: float = 1.8
    n_layers: int = 3
    h_over_c: float = 0.356                # measured (footplate trailing edge), recorded for the reports

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


def _field(internal, inlet, outlet, far, ground, wall) -> str:
    bf = {"inlet": inlet, "outlet": outlet, "top": far, "side": far, "symmetry": "type symmetryPlane;",
          "ground": ground, '"(wing|nose|hanger).*"': wall}
    body = f"dimensions {internal[0]};\n\ninternalField {internal[1]};\n\nboundaryField\n{{\n"
    body += '    #includeEtc "caseDicts/setConstraintTypes"\n'
    for patch, bc in bf.items():
        body += f"    {patch}\n    {{\n        {bc}\n    }}\n"
    return body + "}\n"


def _box(lo, hi):
    return f"type searchableBox; min ({lo[0]:.5f} {lo[1]:.5f} {lo[2]:.5f}); max ({hi[0]:.5f} {hi[1]:.5f} {hi[2]:.5f});"


def write_case(case: Path, stl_dir: Path, p: ParamsIFW) -> dict:
    if case.exists():
        raise FileExistsError(f"{case} exists: choose a new case directory (nothing is overwritten)")
    geo = json.loads((stl_dir / "ifw_geometry.json").read_text())
    p.h_over_c = round(geo["h_te_over_c"], 4)          # measured on the (possibly deformed) geometry
    (case / "constant/triSurface").mkdir(parents=True)
    for n in PARTS:
        shutil.copy(stl_dir / f"{n}.stl", case / "constant/triSurface" / f"{n}.stl")
    d0 = LEVELS[p.level]
    nx, ny, nz = round((p.x_max - p.x_min) / d0), round(-p.y_min / d0), round(p.z_max / d0)
    U, k, om = p.U, p.k_in, p.omega_in
    lo, hi = geo["bbox_model_min_m"], geo["bbox_model_max_m"]
    x_le, x_te, y_tip, z_top = lo[0], hi[0], lo[1], hi[2]

    # ---------------------------------------------------------------- fields
    vec = f"uniform ({U} 0 0)"
    wf_k = "type kqRWallFunction; value uniform 1e-10;"
    wf_om = f"type omegaWallFunction; value uniform {om:.6e};"
    wf_nut = "type nutUSpaldingWallFunction; value uniform 0;"
    _w(case / "0.orig/U", "volVectorField", _field(("[0 1 -1 0 0 0 0]", vec), f"type fixedValue; value {vec};",
       f"type inletOutlet; inletValue uniform (0 0 0); value {vec};", "type slip;",
       f"type fixedValue; value {vec};   // moving ground (rolling road)", "type noSlip;"))
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

    # ---------------------------------------------------------------- mesh (z up)
    X0, X1, Y0, Z1 = p.x_min, p.x_max, p.y_min, p.z_max
    _w(case / "system/blockMeshDict", "dictionary", f"""scale 1;

vertices
(
    ({X0} {Y0} 0) ({X1} {Y0} 0) ({X1} 0 0) ({X0} 0 0)
    ({X0} {Y0} {Z1}) ({X1} {Y0} {Z1}) ({X1} 0 {Z1}) ({X0} 0 {Z1})
);

blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );

boundary
(
    inlet    {{ type patch;         faces ((0 4 7 3)); }}
    outlet   {{ type patch;         faces ((1 2 6 5)); }}
    ground   {{ type wall;          faces ((0 3 2 1)); }}
    top      {{ type patch;         faces ((4 5 6 7)); }}
    side     {{ type patch;         faces ((0 1 5 4)); }}
    symmetry {{ type symmetryPlane; faces ((3 7 6 2)); }}
);
""")
    _w(case / "system/surfaceFeatureExtractDict", "dictionary", "".join(
        f"{n}.stl\n{{\n    extractionMethod extractFromSurface;\n    includedAngle 150;\n"
        f"    subsetFeatures {{ nonManifoldEdges no; openEdges no; }}\n    writeObj no;\n}}\n" for n in PARTS))
    near_lo, near_hi = (x_le - 0.06, y_tip - 0.06, 0.0), (x_te + 0.10, 0.0, z_top + 0.06)
    gap_lo, gap_hi = (x_le - 0.03, y_tip - 0.04, 0.0), (x_te + 0.15, y_tip + 0.15, 0.10)   # footplate/endplate - ground
    wake_lo, wake_hi = (x_te - 0.05, y_tip - 0.20, 0.0), (x_te + 1.0, 0.0, z_top + 0.15)
    far_lo, far_hi = (x_le - 0.4, y_tip - 0.5, 0.0), (x_te + 2.5, 0.0, z_top + 0.5)
    _w(case / "system/snappyHexMeshDict", "dictionary", f"""castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    wing.stl    {{ type triSurfaceMesh; name wing; }}
    nose.stl    {{ type triSurfaceMesh; name nose; }}
    hanger.stl  {{ type triSurfaceMesh; name hanger; }}
    nearModel   {{ {_box(near_lo, near_hi)} }}
    footplateGap {{ {_box(gap_lo, gap_hi)} }}
    wake        {{ {_box(wake_lo, wake_hi)} }}
    farWake     {{ {_box(far_lo, far_hi)} }}
}};

castellatedMeshControls
{{
    maxLocalCells       20000000;
    maxGlobalCells      80000000;
    minRefinementCells  500;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;

    features
    (
        {{ file "wing.eMesh";   level 7; }}
        {{ file "nose.eMesh";   level 4; }}
        {{ file "hanger.eMesh"; level 5; }}
    );

    refinementSurfaces
    {{
        wing    {{ level (5 6); patchInfo {{ type wall; }} }}
        nose    {{ level (3 4); patchInfo {{ type wall; }} }}
        hanger  {{ level (4 5); patchInfo {{ type wall; }} }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        nearModel    {{ mode inside; levels ((1e15 3)); }}
        footplateGap {{ mode inside; levels ((1e15 4)); }}
        wake         {{ mode inside; levels ((1e15 2)); }}
        farWake      {{ mode inside; levels ((1e15 1)); }}
    }}

    // a point in the fluid, far above and outboard of the model
    locationInMesh ({x_le - 0.5:.4f} {y_tip - 0.5:.4f} {z_top + 0.6:.4f});
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
        "(wing|nose|hanger).*" {{ nSurfaceLayers {p.n_layers}; }}
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

    # ---------------------------------------------------------------- numerics (same as the Zerihan kit)
    zdir = Path(__file__).resolve().parent.parent / "zerihan3d"
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
    div(phi,k)      bounded Gauss upwind;
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
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
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
    Aref, lRef = geo["Aref_m2"], geo["c_ref_m"]
    cofr = [x_le + 0.25 * lRef, -0.35, 0.1]

    def fc(name, patches):
        return f"""    {name}
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         ({' '.join(patches)});
        rho             rhoInf;
        rhoInf          1;
        CofR            ({cofr[0]:.4f} {cofr[1]:.4f} {cofr[2]:.4f});
        dragDir         (1 0 0);
        liftDir         (0 0 -1);   // positive CL = DOWNFORCE
        pitchAxis       (0 1 0);
        magUInf         {U};
        lRef            {lRef};
        Aref            {Aref:.6f}; // c * W_s = 0.25 * 0.70 (half model), as in Buscariolo et al.
    }}
"""
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
{fc('coeffs_total', PARTS)}{fc('coeffs_wing', ['wing'])}{fc('coeffs_nose', ['nose'])}{fc('coeffs_hanger', ['hanger'])}
    residuals {{ type solverInfo; libs (utilityFunctionObjects); fields (U p k omega); writeResidualFields no; }}
    yPlus     {{ type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }}
}}
""")
    nyz = max(d for d in range(1, int(p.n_procs ** 0.5) + 1) if p.n_procs % d == 0)
    _w(case / "system/decomposeParDict", "dictionary",
       f"numberOfSubdomains {p.n_procs};\n\nmethod hierarchical;\n\ncoeffs\n{{\n    n ({p.n_procs // nyz} {nyz} 1);\n    order xyz;\n}}\n")
    (case / "case.foam").touch()
    meta = {"case_type": "ifw", "params": asdict(p), "d0_m": d0, "background_cells": nx * ny * nz,
            "geometry": geo, "nu": p.nu, "k_in": k, "omega_in": om, "variant": geo.get("variant"),
            # the Nektar++ reference applies to the original CAD only
            "reference": None if geo.get("variant") else REFERENCE}
    (case / "case3d.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(zdir / "Allrun", case / "Allrun")
    (case / "Allrun").chmod(0o755)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--stl", type=Path, required=True, help="directory written by prepare_ifw.py")
    ap.add_argument("--level", choices=list(LEVELS), default="L0")
    ap.add_argument("--np", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=3000)
    a = ap.parse_args()
    m = write_case(a.case, a.stl, ParamsIFW(level=a.level, n_procs=a.np, n_iter=a.n_iter))
    print(f"case {a.case}: IFW level {a.level}, d0 {m['d0_m']} m, background {m['background_cells']} cells")
