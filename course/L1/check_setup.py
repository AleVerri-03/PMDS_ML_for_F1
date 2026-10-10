#!/usr/bin/env python3
"""
Setup check for the PMDS F1 PINN course  (run it BEFORE the 20 October lesson).

    python course/L1/check_setup.py

It takes about 10 seconds on a laptop CPU, needs no GPU and no internet, and
checks the three things the live demo in the 20 October lesson depends on:

  1. the libraries import (and their versions are printed);
  2. torch.autograd can differentiate a function with respect to its INPUT,
     twice -- the one mechanism a PINN adds to ordinary training;
  3. a tiny network trains with a fixed seed.

If you see "ALL CHECKS PASSED" you are ready. Otherwise send me the full output.
"""
import importlib
import math
import platform
import sys

print(f"python    {platform.python_version()}  ({platform.system()} {platform.machine()})")

failed = []
for name in ["numpy", "pandas", "sklearn", "matplotlib", "joblib", "torch"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name:<9} {mod.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:<9} MISSING  ({exc.__class__.__name__}: {exc})")
        failed.append(name)

if failed:
    sys.exit(f"\nInstall the missing packages first: {', '.join(failed)}")

import torch  # noqa: E402

# ---- 1. derivatives with respect to the INPUT -------------------------------
x = torch.linspace(0, math.pi, 5).reshape(-1, 1).requires_grad_(True)
u = torch.sin(x)
u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
err1 = (u_x - torch.cos(x)).abs().max().item()
err2 = (u_xx + torch.sin(x)).abs().max().item()
print(f"\nd/dx   sin(x) vs cos(x)  : max error {err1:.1e}")
print(f"d2/dx2 sin(x) vs -sin(x) : max error {err2:.1e}")
assert err1 < 1e-6 and err2 < 1e-6, "autograd derivative check failed"

# ---- 2. a small network, fixed seed -----------------------------------------
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
xs = torch.linspace(-1, 1, 64).reshape(-1, 1)
ys = xs ** 2
for _ in range(300):
    loss = (net(xs) - ys).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"\nfinal loss after 300 steps (seed 0): {loss.item():.6e}")
print("(tiny differences from other machines are normal; the order of magnitude "
      "should be 1e-4 or lower)")
assert loss.item() < 1e-2, "tiny network failed to train"

print("\nALL CHECKS PASSED")
