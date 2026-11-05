"""Locate the endplate relative to the wing from Zerihan (2001) Fig. 1c (to-scale schematic).

The thesis gives the endplate size (400 x 170 x 4 mm, rectangular, parallel to the ground at the
reference incidence) but not its position relative to the wing. We fit the true wing outline
(alpha = 1 deg, datum flap) to the wing drawn in Fig. 1c with an isotropic scale + translation,
then read the rectangle in wing coordinates. Uncertainty: line width (~1 px = ~0.6 mm).

Usage (from this directory):
    python fit_endplate_from_fig1c.py ../../dataset fig1c_crop.png endplate_fig1c.json
fig1c_crop.png: thesis PDF (course/_instructor/refs/00192634.pdf) page 40 rendered at 300 dpi
(pdftoppm -f 40 -l 40 -r 300), rotated by -90 deg, cropped to the box (780, 1290, 1520, 1830) px.
Result on 03/11/2026: rms 0.87 px; endplate x = -3.1 ... 396.9 mm, y = -29.9 ... 140.4 mm (main LE at the origin,
alpha 1 deg); fitted vertical scale gives an endplate height of 170.3 mm (text: 170 mm).
"""
import sys, json
import numpy as np
from PIL import Image
from scipy.optimize import minimize

sys.path.insert(0, sys.argv[1])            # dataset pipeline dir (frontwing package)
from frontwing.geometry import zerihan_wing

img = np.asarray(Image.open(sys.argv[2]).convert("L")) < 128     # crop of Fig. 1c, upright
H, W = img.shape
gray = np.asarray(Image.open(sys.argv[2]).convert("L")) < 200   # rectangle is drawn in a light scanned line

def edge(profile, lo, hi):
    """Weighted position of the darkest line inside [lo, hi) (rows or columns)."""
    seg = profile[lo:hi].astype(float)
    k = np.nonzero(seg >= 0.8 * seg.max())[0]
    return lo + float(np.average(k, weights=seg[k]))

cols = gray.sum(0)
front, rear = edge(cols, 0, W // 4), edge(cols, 3 * W // 4, W)
rows = gray[:, int(front) + 3:int(rear) - 3].sum(1)
top, bot = edge(rows, 0, H // 4), edge(rows, H // 4, int(0.7 * H))
print(f"rect px: front {front:.1f} rear {rear:.1f} top {top:.1f} bottom {bot:.1f}")
L_px, H_px = rear - front, bot - top
print(f"aspect px {L_px / H_px:.3f}  (400/170 = {400 / 170:.3f})")

# wing pixels: inside the rectangle, away from its edges, left of nothing else
ys, xs = np.nonzero(img)
m = (xs > front + 6) & (xs < rear - 6) & (ys > top + 6) & (ys < bot - 6)
# drop the axes arrows / 'x' label near the LE: keep pixels right of front+12 or the LE nose itself
pts = np.column_stack([xs[m], ys[m]]).astype(float)
c = 380.0
w = zerihan_wing(0.0, 1.0)
outline = []
for el in w.elements:
    s, p = el.dense(1500)
    outline += [s, p]
outline = np.vstack(outline) * c            # mm, y up
from scipy.ndimage import distance_transform_edt, map_coordinates
mask = np.ones_like(img, dtype=bool); mask[pts[:, 1].astype(int), pts[:, 0].astype(int)] = False
dist = distance_transform_edt(mask)          # px to the nearest drawn wing pixel
outline = outline[::10]

def cost(q, sx):
    sy, tx, ty = q
    X = np.column_stack([tx + sx * outline[:, 0], ty - sy * outline[:, 1]])
    d = map_coordinates(dist, [X[:, 1], X[:, 0]], order=1, mode="nearest")
    return np.mean(np.minimum(d, 15.0) ** 2)

# horizontal scale fixed by the endplate length (400 mm); vertical scale free within +-6 %
# (the scanned rectangle has aspect 2.31 instead of 2.35, i.e. ~2 % vertical stretch)
s0 = L_px / 400.0
best = None
for tx in np.linspace(front - 10, front + 30, 5):
    for ty in np.linspace(top, bot, 8):
        r = minimize(cost, [s0, tx, ty], args=(s0,), method="Powell",
                     bounds=[(0.94 * s0, 1.06 * s0), (front - 40, front + 80), (top, bot)])
        if best is None or r.fun < best.fun:
            best = r
sy, tx, ty = best.x
s = s0
print(f"fit: sx {s0:.4f} px/mm (endplate length), sy {sy:.4f} px/mm ({sy / s0:.3f} sx), rms {np.sqrt(best.fun):.2f} px")
to_mm = lambda px, py: ((px - tx) / s, (ty - py) / sy)
fx, ty_ = to_mm(front, top); rx, by = to_mm(rear, bot)
res = {"scale_px_per_mm_y_fit": sy, "scale_px_per_mm_endplate": s0, "rms_px": float(np.sqrt(best.fun)),
       "endplate_x_front_mm": fx, "endplate_x_rear_mm": rx, "endplate_y_top_mm": ty_, "endplate_y_bottom_mm": by,
       "endplate_len_mm_from_fig": rx - fx, "endplate_h_mm_from_fig": ty_ - by,
       "wing_lowest_y_mm": float(w.lowest_point()[1] * c), "main_le_mm": (w.main.le * c).tolist(),
       "flap_te_mm": (w.flap.te_mid * c).tolist()}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[3], "w"), indent=1)
# overlay for checking
from PIL import ImageDraw
ov = Image.open(sys.argv[2]).convert("RGB"); d = ImageDraw.Draw(ov)
X = np.column_stack([tx + s * outline[:, 0], ty - sy * outline[:, 1]])
for x, y in X[::3]:
    d.point((x, y), fill=(255, 0, 0))
ov.save(sys.argv[3].replace(".json", "_overlay.png"))
