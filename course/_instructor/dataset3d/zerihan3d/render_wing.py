"""Representative views of the 3D Zerihan wing case (OpenFOAM, decomposed), for ParaView 5.11+.

Run in batch:   pvbatch render_wing.py /path/to/case/case.foam [out_dir]
or in the GUI:  View > Python Shell > Run Script (edit CASE below first): the pipeline stays in the GUI to play with.

Views written (1920x1080 PNG):
  1_underside_cp.png   full wing (half model mirrored about the symmetry plane), seen from BELOW, where the
                       suction side is: surface pressure coefficient Cp
  2_rear_underside_cp.png  from below and behind, through the semi-transparent moving ground
  3_tip_vortex.png     streamlines from the endplate lower edge (|U|) + streamwise vorticity on a plane at x = 0.45 m
  4_midspan_U.png      velocity magnitude on the symmetry plane (mid-span section, the "2D" cut)
"""
import os
import sys

from paraview.simple import *  # noqa: F401,F403

CASE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/GitHub/F1-Front-Wing-Physics-Informed-NN/data/frontwing_runs3d/"
    "zerihan3d_L0_h0.211_np6_20261103T150218Z/case.foam")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(CASE), "views")
os.makedirs(OUT, exist_ok=True)
U_INF = 30.0
Q_INF = 0.5 * U_INF ** 2          # kinematic pressure: Cp = p / (0.5 U^2)
WALLS = ["patch/main_fore", "patch/main_aft", "patch/flap_fore", "patch/flap_aft", "patch/endplate"]
RES = [1920, 1080]

paraview.simple._DisableFirstRenderCameraReset()
src = OpenFOAMReader(registrationName="case.foam", FileName=CASE)
src.CaseType = "Decomposed Case"
src.UpdatePipelineInformation()
src.MeshRegions = ["internalMesh"] + WALLS + ["patch/ground", "patch/symmetry"]
src.CellArrays = ["p", "U", "yPlus"]
src.SkipZeroTime = 1
src.UpdatePipeline(max(src.TimestepValues))
anim = GetAnimationScene(); anim.UpdateAnimationUsingDataTimeSteps(); anim.AnimationTime = max(src.TimestepValues)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = RES
view.OrientationAxesVisibility = 1
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.EnableRenderOnInteraction = 0


def block(names, name):
    e = ExtractBlock(registrationName=name, Input=src)
    try:                                   # ParaView >= 5.10
        e.Selectors = ["/Root/" + n.replace("patch/", "boundary/") if n.startswith("patch/") else "/Root/" + n
                       for n in names]
    except AttributeError:
        e.BlockIndices = []
    return MergeBlocks(registrationName=name + "_m", Input=e)


def hide_all():
    for s in GetSources().values():
        Hide(s, view)


# ---------------------------------------------------------------- wing surfaces with Cp, mirrored
walls = block(WALLS, "walls")
cp = Calculator(registrationName="Cp", Input=walls)
cp.AttributeType = "Cell Data"
cp.ResultArrayName = "Cp"
cp.Function = f"p/{Q_INF}"
full = Reflect(registrationName="fullWing", Input=cp)
full.Plane = "Z Min"                     # symmetry plane is z = 0 = the minimum of the half model
full.CopyInput = 1
cpLUT = GetColorTransferFunction("Cp")
cpLUT.ApplyPreset("Cool to Warm (Extended)", True)
cpLUT.AutomaticRescaleRangeMode = "Never"
cpLUT.RescaleTransferFunction(-6.0, 1.0)
cpLUT.InvertTransferFunction()           # suction (negative Cp) in red

ground = block(["patch/ground"], "ground")
gclip = Clip(registrationName="groundNear", Input=ground)
gclip.ClipType = "Box"
gclip.ClipType.Position = [-0.3, -0.01, -0.9]
gclip.ClipType.Length = [1.3, 0.02, 1.8]
gclip.Invert = 1
gfull = Reflect(registrationName="groundFull", Input=gclip); gfull.Plane = "Z Min"; gfull.CopyInput = 1


def show_wing(edges=False):
    d = Show(full, view)
    ColorBy(d, ("CELLS", "Cp"))
    d.LookupTable = cpLUT
    cpLUT.AutomaticRescaleRangeMode = "Never"; cpLUT.RescaleTransferFunction(-6.0, 1.0)
    d.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(cpLUT, view)
    bar.Title = "Cp"; bar.ComponentTitle = ""; bar.TitleColor = bar.LabelColor = [0.1, 0.1, 0.1]
    bar.TitleFontSize = 20; bar.LabelFontSize = 16; bar.ScalarBarLength = 0.4
    bar.WindowLocation = "Any Location"; bar.Position = [0.9, 0.3]
    d.Specular = 0.3
    if edges:
        d.SetRepresentationType("Surface With Edges"); d.EdgeColor = [0.3, 0.3, 0.3]
    return d


def cam(direction, up, focal=None, zoom=1.0, angle=30.0):
    """Look at `focal` (default: centre of what is shown) from `direction`, fit everything shown, then zoom."""
    c = GetActiveCamera()
    f = focal or [0.2, 0.13, 0.0]
    c.SetFocalPoint(*f); c.SetPosition(*[f[i] + direction[i] for i in range(3)]); c.SetViewUp(*up)
    c.SetViewAngle(angle); c.OrthogonalizeViewUp()
    view.ResetCamera(False)
    c.SetFocalPoint(*f) if focal else None
    c.Dolly(zoom)


def save(name):
    Render(view)
    SaveScreenshot(os.path.join(OUT, name), view, ImageResolution=RES, TransparentBackground=0)
    print("written", os.path.join(OUT, name))


# 1 -- from below (the suction surface faces the ground), slightly in front and to the side
hide_all(); show_wing()
cam([-0.55, -1.0, 0.45], [0.6, -0.55, 0.0], zoom=1.15)
save("1_underside_cp.png")

# 2 -- from below and behind, through the semi-transparent moving ground: flap suction side and endplate
hide_all(); show_wing()
g = Show(gfull, view); ColorBy(g, None); g.DiffuseColor = [0.55, 0.55, 0.6]; g.Opacity = 0.18
cam([0.9, -0.75, 0.75], [-0.4, -0.5, 0.0], focal=[0.2, 0.1, 0.05], zoom=2.6)
save("2_rear_underside_cp.png")

# 3 -- tip vortex: streamlines seeded along the endplate lower edge, cross-flow plane behind the wing
hide_all()
fluid = block(["internalMesh"], "fluid")
c2p = CellDatatoPointData(registrationName="c2p", Input=fluid)
st = StreamTracer(registrationName="tipVortex", Input=c2p, SeedType="Line")
st.Vectors = ["POINTS", "U"]
st.SeedType.Point1 = [-0.02, 0.06, 0.53]
st.SeedType.Point2 = [-0.02, 0.06, 0.575]
st.SeedType.Resolution = 40
st.MaximumStreamlineLength = 2.5
tube = Tube(registrationName="tubes", Input=st); tube.Radius = 0.0012
plane = Slice(registrationName="crossflow", Input=c2p)
plane.SliceType = "Plane"; plane.SliceType.Origin = [0.45, 0, 0]; plane.SliceType.Normal = [1, 0, 0]
pclip = Clip(registrationName="crossflowNear", Input=plane)
pclip.ClipType = "Box"; pclip.ClipType.Position = [0.4, 0.0, 0.3]; pclip.ClipType.Length = [0.1, 0.3, 0.5]
pclip.Invert = 1
vort = Gradient(registrationName="vorticity", Input=pclip)   # in-plane derivatives are enough for omega_x
try:
    vort.ScalarArray = ["POINTS", "U"]
except AttributeError:
    vort.InputArray = ["POINTS", "U"]
vort.ComputeVorticity = 1
show_wing()
uLUT = GetColorTransferFunction("U"); uLUT.ApplyPreset("Viridis (matplotlib)", True)
t = Show(tube, view); ColorBy(t, ("POINTS", "U", "Magnitude")); t.LookupTable = uLUT
uLUT.AutomaticRescaleRangeMode = "Never"; uLUT.RescaleTransferFunction(0, 50)
wLUT = GetColorTransferFunction("Vorticity"); wLUT.ApplyPreset("Blue Orange (divergent)", True)
pd = Show(vort, view); ColorBy(pd, ("POINTS", "Vorticity", "X")); pd.LookupTable = wLUT
wLUT.VectorMode = "Component"; wLUT.VectorComponent = 0
wLUT.AutomaticRescaleRangeMode = "Never"; wLUT.RescaleTransferFunction(-400, 400)
pd.SetScalarBarVisibility(view, True)
wb = GetScalarBar(wLUT, view); wb.Title = "omega_x [1/s]"; wb.ComponentTitle = ""
wb.TitleColor = wb.LabelColor = [0.1] * 3; wb.WindowLocation = "Any Location"; wb.Position = [0.03, 0.3]
wb.ScalarBarLength = 0.4
HideScalarBarIfNotNeeded(cpLUT, view)
GetScalarBar(cpLUT, view).Visibility = 0
cam([1.0, 0.35, 0.45], [0, 1, 0], focal=[0.45, 0.08, 0.52], zoom=4.0)
save("3_tip_vortex.png")

# 4 -- mid-span section on the symmetry plane
hide_all()
sym = CellDatatoPointData(registrationName="symmetry_pts", Input=block(["patch/symmetry"], "symmetry"))
sc = Show(sym, view); ColorBy(sc, ("POINTS", "U", "Magnitude")); sc.LookupTable = uLUT
uLUT.AutomaticRescaleRangeMode = "Never"; uLUT.RescaleTransferFunction(0, 50)
ub = GetScalarBar(uLUT, view); ub.Title = "|U| [m/s]"; ub.ComponentTitle = ""; ub.TitleColor = ub.LabelColor = [0.9] * 3
ub.WindowLocation = "Any Location"; ub.ScalarBarLength = 0.4
sc.SetScalarBarVisibility(view, True)
ub.Position = [0.9, 0.3]
c4 = GetActiveCamera(); c4.SetParallelProjection(1)
c4.SetFocalPoint(0.25, 0.13, 0.0); c4.SetPosition(0.25, 0.13, 2.0); c4.SetViewUp(0, 1, 0); c4.SetParallelScale(0.2)
save("4_midspan_U.png")
GetActiveCamera().SetParallelProjection(0)
