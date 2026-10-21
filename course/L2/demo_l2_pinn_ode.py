#!/usr/bin/env python3
"""
Lesson 2 live demo: a physics-informed neural network for an ODE.

Problem (one variable, t):
    dy/dt = -k * y        for t in [0, 2]
    y(0)  = 1             (the starting point)

Exact solution, used ONLY at the end to measure the error:
    y(t) = exp(-k * t)

NO TRAINING DATA IS USED. The network never sees a value of the solution.
It learns only from the equation and the starting point.

Run:  python course/L2/demo_l2_pinn_ode.py            (about 10 s on a laptop CPU)
      python course/L2/demo_l2_pinn_ode.py --k 3      (try another k)
      python course/L2/demo_l2_pinn_ode.py --plot     (also save a PNG)
      python course/L2/demo_l2_pinn_ode.py --no-start (what if we forget y(0) = 1?)
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=float, default=1.0, help="decay rate k")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--plot", action="store_true", help="save pinn_ode.png")
ap.add_argument("--no-start", action="store_true", help="drop the starting-point term (homework)")
ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)
args = ap.parse_args()
k, T = args.k, 2.0
torch.manual_seed(0)


def derivative(y, t):
    """dy/dt, computed exactly by PyTorch (autograd). Copy these arguments as they are."""
    return torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]


# ---------------------------------------------------------------- part 1
# The residual R = dy/dt + k*y: what is left over when a guess goes into the equation.
print("PART 1  The residual R = dy/dt + k*y of three guesses (no network yet)")
t = torch.linspace(0, T, 201).reshape(-1, 1).requires_grad_(True)
guesses = {
    "y = exp(-k t)  (exact)": torch.exp(-k * t),
    "y = 1          (never moves)": torch.ones_like(t) + 0 * t,
    "y = 1 - k t    (straight line)": 1 - k * t,
}
for name, y in guesses.items():
    R = derivative(y, t) + k * y
    print(f"  {name:32s} largest |R| = {R.abs().max().item():.4f}")

# ---------------------------------------------------------------- part 2
# The network: t in, y out. 1,153 parameters.
net = torch.nn.Sequential(
    torch.nn.Linear(1, 32), torch.nn.Tanh(),
    torch.nn.Linear(32, 32), torch.nn.Tanh(),
    torch.nn.Linear(32, 1),
)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
print(f"\nPART 2  Training the network ({sum(p.numel() for p in net.parameters()):,} parameters)")
print(f"  {'step':>5s} {'equation':>12s} {'start':>12s}")

t0 = time.time()
for step in range(args.steps + 1):
    t = (T * torch.rand(200, 1)).requires_grad_(True)   # 200 random times, new every step
    y = net(t)
    loss_equation = (derivative(y, t) + k * y).pow(2).mean()   # does it obey dy/dt = -k y?
    loss_start = (net(torch.zeros(1, 1)) - 1.0).pow(2).mean()  # does it start at 1?
    loss = loss_equation if args.no_start else loss_equation + loss_start

    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"  {step:5d} {loss_equation.item():12.2e} {loss_start.item():12.2e}")

# ---------------------------------------------------------------- part 3
# Only now we look at the exact solution, to measure the error.
t = torch.linspace(0, T, 201).reshape(-1, 1)
with torch.no_grad():
    y_net, y_exact = net(t), torch.exp(-k * t)
err = (torch.linalg.norm(y_net - y_exact) / torch.linalg.norm(y_exact)).item()
print(f"\nPART 3  Relative error against exp(-k t): {err:.3%}   ({time.time() - t0:.1f} s)")

if args.dump:
    with open(args.dump, "w") as f:
        f.write("t,y_net,y_exact\n")
        for a, b, c in zip(t.flatten(), y_net.flatten(), y_exact.flatten()):
            f.write(f"{a:.4f},{b:.6f},{c:.6f}\n")
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(t, y_exact, "k-", lw=2, label="exact  exp(-k t)")
    plt.plot(t, y_net, "--", lw=2, label="network")
    plt.xlabel("t"); plt.ylabel("y"); plt.legend(); plt.title(f"dy/dt = -{k:g} y, y(0) = 1")
    plt.savefig("pinn_ode.png", dpi=140)
    print("  saved pinn_ode.png")
