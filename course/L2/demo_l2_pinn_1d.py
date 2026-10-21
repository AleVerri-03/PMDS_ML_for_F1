#!/usr/bin/env python3
"""
Lesson 2 live demo — a physics-informed neural network in ~40 lines.

Problem: 1D heat (diffusion) equation
    u_t = nu * u_xx        on  x in [0,1],  t in [0,1]
    u(x,0) = sin(pi x)                       (initial condition)
    u(0,t) = u(1,t) = 0                      (boundary conditions)

Exact solution:
    u(x,t) = sin(pi x) * exp(-nu * pi^2 * t)

NO TRAINING DATA IS USED. The network never sees a single value of the exact
solution. It learns only by being penalised for violating the equation.

Run:  python course/L2/demo_l2_pinn_1d.py [--activation tanh|relu]
"""
import argparse
import math
import time

import torch
import torch.nn as nn

NU = 0.1            # diffusivity
torch.manual_seed(0)


def exact(x, t):
    return torch.sin(math.pi * x) * torch.exp(-NU * math.pi**2 * t)


class PINN(nn.Module):
    """Maps (x, t) -> u. Note the INPUT is coordinates, not physical data."""

    def __init__(self, width=32, depth=3, act="tanh"):
        super().__init__()
        A = {"tanh": nn.Tanh, "relu": nn.ReLU}[act]
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), A()]
            d_in = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def residual(model, x, t):
    """R[u] = u_t - nu * u_xx, computed exactly by automatic differentiation.

    This function IS the physics. Everything else is boilerplate.
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)

    # First derivatives. create_graph=True keeps the graph alive so we can
    # differentiate u_x again below -- this is the line that makes it a PINN.
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    # Second derivative: differentiate the graph that already contains a derivative.
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return u_t - NU * u_xx


def train(act="tanh", n_collocation=2000, epochs=4000, verbose=True):
    model = PINN(act=act)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t0 = time.time()

    for ep in range(epochs):
        # --- interior collocation points: enforce the PDE -------------------
        xr = torch.rand(n_collocation, 1)
        tr = torch.rand(n_collocation, 1)
        loss_pde = residual(model, xr, tr).pow(2).mean()

        # --- initial condition t = 0: u = sin(pi x) -------------------------
        x0 = torch.rand(200, 1)
        t_zero = torch.zeros_like(x0)
        loss_ic = (model(x0, t_zero) - torch.sin(math.pi * x0)).pow(2).mean()

        # --- boundary conditions x = 0 and x = 1: u = 0 ---------------------
        tb = torch.rand(200, 1)
        loss_bc = (model(torch.zeros_like(tb), tb).pow(2).mean()
                   + model(torch.ones_like(tb), tb).pow(2).mean())

        loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (ep % 1000 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:5d}  total {loss.item():.3e}   "
                  f"pde {loss_pde.item():.3e}  ic {loss_ic.item():.3e}  "
                  f"bc {loss_bc.item():.3e}")

    return model, time.time() - t0


def evaluate(model):
    """Relative L2 error against the exact solution on a 101x101 grid."""
    g = torch.linspace(0, 1, 101)
    X, T = torch.meshgrid(g, g, indexing="ij")
    x, t = X.reshape(-1, 1), T.reshape(-1, 1)
    with torch.no_grad():
        pred = model(x, t)
    ref = exact(x, t)
    rel_l2 = (torch.linalg.norm(pred - ref) / torch.linalg.norm(ref)).item()
    return rel_l2, pred.reshape(101, 101), ref.reshape(101, 101)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="tanh", choices=["tanh", "relu"])
    ap.add_argument("--epochs", type=int, default=4000)
    args = ap.parse_args()

    print(f"Training a PINN with {args.activation} activation, "
          f"{args.epochs} epochs, no training data.")
    model, secs = train(act=args.activation, epochs=args.epochs)
    rel, pred, ref = evaluate(model)

    print(f"\nwall clock            : {secs:.1f} s (CPU)")
    print(f"parameters            : {sum(p.numel() for p in model.parameters())}")
    print(f"relative L2 error     : {rel:.4%}")
    print(f"u(0.5, 0.0)  pred {pred[50, 0]:+.5f}   exact {ref[50, 0]:+.5f}")
    print(f"u(0.5, 0.5)  pred {pred[50, 50]:+.5f}   exact {ref[50, 50]:+.5f}")
    print(f"u(0.5, 1.0)  pred {pred[50, 100]:+.5f}  exact {ref[50, 100]:+.5f}")

    if args.activation == "relu":
        print("\nNOTE: ReLU is piecewise linear, so u_xx is zero almost everywhere.")
        print("The PDE residual therefore collapses to u_t = 0 and the network")
        print("cannot represent diffusion at all. This is not a tuning problem.")
