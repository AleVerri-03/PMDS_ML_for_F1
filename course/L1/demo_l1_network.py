#!/usr/bin/env python3
"""
Lesson 1 live demo: what a neural network is, on a lift curve.

The data is a MADE-UP but realistic lift curve of a wing section: lift coefficient
Cl against angle of attack alpha (degrees). It rises in a straight line at small
angles, then the wing stalls and Cl drops. We add a little measurement noise.

    python course/L1/demo_l1_network.py                      linear regression vs network, 40 points
    python course/L1/demo_l1_network.py --points 6           only 6 points: what happens?
    python course/L1/demo_l1_network.py --points 6 --physics add our physics loss
    python course/L1/demo_l1_network.py --plot               also save a PNG

About 10 seconds on a laptop CPU, no GPU.
"""
import argparse
import math
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--points", type=int, default=40, help="number of training points")
ap.add_argument("--physics", action="store_true", help="add the thin-airfoil physics loss")
ap.add_argument("--lam", type=float, default=1.0, help="weight of the physics loss")
ap.add_argument("--hidden", type=int, default=32, help="neurons per hidden layer")
ap.add_argument("--epochs", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--plot", action="store_true")
ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)
args = ap.parse_args()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

ALPHA0 = -2.0   # zero-lift angle of this (cambered) section, degrees


def thin_airfoil(alpha):
    """Thin-airfoil theory, valid only at small angles: Cl = 2*pi*(alpha - alpha0), alpha in radians."""
    return 2 * math.pi * np.radians(alpha - ALPHA0)


def true_cl(alpha):
    """The 'real' curve (made up): the theory line up to about 12 degrees, then stall."""
    s = 1 / (1 + np.exp(-(alpha - 13) / 1.2))
    return thin_airfoil(alpha) * (1 - s) + s * (0.75 * thin_airfoil(13.0) - 0.03 * (alpha - 13))


# ------------------------------------------------------------------ data
a_train = rng.uniform(-5, 20, args.points)
cl_train = true_cl(a_train) + rng.normal(0, 0.03, args.points)      # "measurements"
a_val = np.linspace(-5, 20, 201)                                       # validation: never used to train
cl_val = true_cl(a_val)
print(f"DATA  {args.points} training points (alpha between -5 and 20 deg), 201 validation points")

# ------------------------------------------------------------------ 1. linear regression
# Cl = w * alpha + b : two parameters. The same as a network with no hidden layer.
w, b = np.polyfit(a_train, cl_train, 1)
mse_lin = np.mean((w * a_val + b - cl_val) ** 2)
print(f"\nLINEAR REGRESSION  Cl = {w:.4f} * alpha + {b:.3f}      (2 parameters)")
print(f"  validation error (MSE): {mse_lin:.4f}")

# ------------------------------------------------------------------ 2. neural network
# alpha -> [32 neurons, tanh] -> [32 neurons, tanh] -> Cl
H = args.hidden
net = torch.nn.Sequential(
    torch.nn.Linear(1, H), torch.nn.Tanh(),
    torch.nn.Linear(H, H), torch.nn.Tanh(),
    torch.nn.Linear(H, 1),
)
n_par = sum(p.numel() for p in net.parameters())
opt = torch.optim.Adam(net.parameters(), lr=3e-3)


def f(alpha):            # the network, fed alpha/10 so the input is of order 1
    return net(torch.as_tensor(alpha, dtype=torch.float32).reshape(-1, 1) / 10).flatten()


x, y = a_train, torch.tensor(cl_train, dtype=torch.float32)
label = "NEURAL NETWORK + PHYSICS LOSS" if args.physics else "NEURAL NETWORK"
print(f"\n{label}  ({n_par:,} parameters, {args.epochs} training steps)")
print(f"  {'step':>5s} {'data loss':>11s} {'physics loss':>13s} {'validation':>11s}")
t0 = time.time()
for step in range(args.epochs + 1):
    loss_data = (f(x) - y).pow(2).mean()                       # does it match the data?
    a_phys = rng.uniform(-4, 6, 50)                             # small angles only: where the theory holds
    loss_phys = (f(a_phys) - torch.tensor(thin_airfoil(a_phys), dtype=torch.float32)).pow(2).mean()
    loss = loss_data + args.lam * loss_phys if args.physics else loss_data

    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        with torch.no_grad():
            val = (f(a_val) - torch.tensor(cl_val)).pow(2).mean().item()
        print(f"  {step:5d} {loss_data.item():11.4f} {loss_phys.item():13.4f} {val:11.4f}")

with torch.no_grad():
    cl_net = f(a_val).numpy()
mse_net = np.mean((cl_net - cl_val) ** 2)
print(f"  validation error (MSE): {mse_net:.4f}      ({time.time() - t0:.1f} s)")

print(f"\nSUMMARY  linear {mse_lin:.4f}   network{' + physics' if args.physics else ''} {mse_net:.4f}")

if args.dump:
    with open(args.dump, "w") as fh:
        fh.write("alpha,cl_true,cl_net,cl_lin\n")
        for a, c, n in zip(a_val, cl_val, cl_net):
            fh.write(f"{a:.3f},{c:.5f},{n:.5f},{w * a + b:.5f}\n")
        fh.write("#train\n")
        for a, c in zip(a_train, cl_train):
            fh.write(f"{a:.3f},{c:.5f}\n")
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(a_val, cl_val, "k-", lw=2, label="true curve")
    plt.plot(a_val, w * a_val + b, ":", label="linear regression")
    plt.plot(a_val, cl_net, "--", lw=2, label="network" + (" + physics" if args.physics else ""))
    plt.plot(a_train, cl_train, "o", label="training points")
    plt.xlabel("angle of attack (deg)"); plt.ylabel("Cl"); plt.legend()
    plt.savefig("lift_curve.png", dpi=140)
    print("  saved lift_curve.png")
