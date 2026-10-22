#!/usr/bin/env python3
"""
Lesson 3 live demo (was demo_l2_slope.py): the physics of Lesson 1, written as a DERIVATIVE.

Same made-up lift curve, same network and same data as course/L1/demo_l1_network.py.
In Lesson 1 the physics loss compared the network's VALUE with thin-airfoil theory,
Cl = 2*pi*(alpha - alpha0). Here we can also say the same law with derivatives of the
network's output with respect to its INPUT (autograd, exactly as in a PINN):

    --constraint none      network alone (as Lesson 1 without --physics)
    --constraint value     Cl = 2*pi*(alpha - alpha0)       (as Lesson 1 with --physics)
    --constraint slope     dCl/dalpha = 2*pi per radian     (alpha0 not needed)
    --constraint linear    d2Cl/dalpha2 = 0                 (only "a straight line": no 2*pi)

The network is fed x = alpha_deg / 10. Autograd gives dCl/dx; the law is per radian:
    dCl/dalpha_rad = dCl/dx * dx/dalpha_rad = dCl/dx * 18/pi   (about 5.73)
--no-chain-rule forgets that factor, to see what happens.

    python course/L3/demo_l3_slope.py --points 6                              network alone, seeds 0-4
    python course/L3/demo_l3_slope.py --points 6 --constraint slope           with the slope
    python course/L3/demo_l3_slope.py --points 6 --constraint slope --no-chain-rule
    python course/L3/demo_l3_slope.py --points 6 --constraint slope --compare  table: does it help, seed by seed?
    python course/L3/demo_l3_slope.py --points 6 --constraint linear --activation relu   why ReLU makes it empty

Each run is one seed (training points, noise, initial weights); the script loops over --seeds
and prints mean and standard deviation, because one run proves nothing. Laptop CPU, no GPU.
"""
import argparse
import math
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--points", type=int, default=40, help="number of training points")
ap.add_argument("--constraint", choices=["none", "value", "slope", "linear"], default="none")
ap.add_argument("--no-chain-rule", action="store_true", help="forget the factor 18/pi (slope only)")
ap.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
ap.add_argument("--lam", type=float, default=1.0, help="weight of the physics loss")
ap.add_argument("--hidden", type=int, default=32, help="neurons per hidden layer")
ap.add_argument("--epochs", type=int, default=3000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--compare", action="store_true", help="also run the network alone and say where the constraint helps")
args = ap.parse_args()
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

ALPHA0 = -2.0                 # zero-lift angle of this (cambered) section, degrees
CHAIN = 18 / math.pi          # dx/dalpha_rad for x = alpha_deg / 10


def thin_airfoil(alpha):
    """Thin-airfoil theory, valid only at small angles: Cl = 2*pi*(alpha - alpha0), alpha in radians."""
    return 2 * math.pi * np.radians(alpha - ALPHA0)


def true_cl(alpha):
    """The 'real' curve (made up, the same as Lesson 1): the theory line up to about 12 degrees, then stall."""
    s = 1 / (1 + np.exp(-(alpha - 13) / 1.2))
    return thin_airfoil(alpha) * (1 - s) + s * (0.75 * thin_airfoil(13.0) - 0.03 * (alpha - 13))


def run(seed, constraint):
    """One training run. Data, noise and initial weights are those of demo_l1_network.py with the same seed."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a_train = rng.uniform(-5, 20, args.points)
    cl_train = true_cl(a_train) + rng.normal(0, 0.03, args.points)
    a_val = np.linspace(-5, 20, 201)
    cl_val = torch.tensor(true_cl(a_val), dtype=torch.float32)

    act = torch.nn.Tanh if args.activation == "tanh" else torch.nn.ReLU
    H = args.hidden
    net = torch.nn.Sequential(torch.nn.Linear(1, H), act(), torch.nn.Linear(H, H), act(), torch.nn.Linear(H, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    def f(alpha):                       # the network, fed x = alpha/10 (degrees) so the input is of order 1
        return net(torch.as_tensor(alpha, dtype=torch.float32).reshape(-1, 1) / 10).flatten()

    def derivative(out, x):             # derivative of the OUTPUT with respect to the INPUT, as in a PINN
        return torch.autograd.grad(out, x, torch.ones_like(out), create_graph=True)[0]

    y = torch.tensor(cl_train, dtype=torch.float32)
    for _ in range(args.epochs + 1):
        loss = (f(a_train) - y).pow(2).mean()                        # does it match the data?
        a_phys = rng.uniform(-4, 6, 50)                              # small angles only: where the theory holds
        if constraint == "value":
            target = torch.tensor(thin_airfoil(a_phys), dtype=torch.float32)
            loss = loss + args.lam * (f(a_phys) - target).pow(2).mean()
        elif constraint in ("slope", "linear"):
            x = torch.tensor(a_phys / 10, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
            cl = net(x)
            dcl_dx = derivative(cl, x)
            if constraint == "slope":
                factor = 1.0 if args.no_chain_rule else CHAIN       # dCl/dalpha_rad = dCl/dx * 18/pi
                loss = loss + args.lam * (dcl_dx * factor - 2 * math.pi).pow(2).mean()
            else:
                d2cl_dx2 = derivative(dcl_dx, x)                     # zero for ReLU almost everywhere
                loss = loss + args.lam * d2cl_dx2.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        mse = (f(a_val) - cl_val).pow(2).mean().item()
    # slope of the trained network at alpha = 2 deg, per radian, by autograd (with the chain rule, always)
    x = torch.tensor([[0.2]], requires_grad=True)
    slope = (derivative(net(x), x) * CHAIN).item()
    return mse, slope


def label(c):
    if c == "slope" and args.no_chain_rule:
        return "slope, factor forgotten"
    return {"none": "network alone", "value": "value (Lesson 1)", "slope": "slope", "linear": "linearity"}[c]


name = label(args.constraint)
print(f"DATA  {args.points} training points (alpha between -5 and 20 deg), 201 validation points")
print(f"MODEL 1 -> {args.hidden} -> {args.hidden} -> 1, {args.activation}, input x = alpha/10, {args.epochs} steps")
print(f"PHYSICS  {name}" + (f", lambda = {args.lam:g}" if args.constraint != "none" else ""))
if args.constraint == "slope":
    print(f"  target: dCl/dalpha = 2*pi = 6.28 per radian; in network units dCl/dx = "
          + ("6.28  <- the factor 18/pi is forgotten" if args.no_chain_rule else "2*pi*pi/18 = 1.10"))
print()

t0 = time.time()
rows = []
head = f"{'seed':>4s} {'validation MSE':>15s} {'slope at 2 deg':>15s}"
if args.compare:
    head += f" {'alone':>9s} {'helps':>6s}"
print(head)
for s in args.seeds:
    mse, slope = run(s, args.constraint)
    line = f"{s:4d} {mse:15.4f} {slope:15.2f}"
    base = None
    if args.compare and args.constraint != "none":
        base, _ = run(s, "none")
        verdict = "same" if abs(mse - base) < 1e-9 else ("yes" if mse < base else "no")
        line += f" {base:9.4f} {verdict:>6s}"
    rows.append((mse, slope, base))
    print(line, flush=True)

m = np.array([r[0] for r in rows])
sl = np.array([r[1] for r in rows])
print(f"\nSUMMARY  {name}: validation MSE {m.mean():.4f} +- {m.std(ddof=1) if len(m) > 1 else 0:.4f} over {len(m)} seeds"
      f"   (slope at 2 deg {sl.mean():.2f} per rad, theory 6.28)")
if args.compare and args.constraint != "none":
    b = np.array([r[2] for r in rows])
    print(f"         network alone:  validation MSE {b.mean():.4f} +- {b.std(ddof=1) if len(b) > 1 else 0:.4f}"
          f"   -> the constraint helps in {int((m < b - 1e-9).sum())} of {len(m)} seeds")
    if np.allclose(m, b, atol=1e-9):
        print("         identical to the network alone: the physics term gives no gradient (ReLU: second derivative = 0)")
print(f"({time.time() - t0:.0f} s)")
