#!/usr/bin/env python3
"""
Lesson 2 mini-lab -- what a residual is, and why automatic differentiation is exact.

We use the diffusion problem of the live demo, because we know its exact answer:

    u_t = nu * u_xx,   nu = 0.1,   u(x,0) = sin(pi x),   u(0,t) = u(1,t) = 0
    exact:  u(x,t) = sin(pi x) * exp(-nu * pi^2 * t)

The residual of a candidate function u is   R[u] = u_t - nu * u_xx.

    python course/L2/lab_l2_residual.py       (about 2 seconds, no training, no data)

Three things to see:
  PART 1  the exact solution has residual zero -- to machine precision;
  PART 2  the residual alone is not enough: u = 0 also has residual zero;
  PART 3  autograd is exact, finite differences are not.
"""
import math

import torch

NU = 0.1


def exact(x, t):
    return torch.sin(math.pi * x) * torch.exp(-NU * math.pi ** 2 * t)


def zero(x, t):
    return 0.0 * exact(x, t)          # u = 0 everywhere (written so autograd still has a graph)


def too_fast(x, t):                   # right shape, wrong decay rate (twice as fast)
    return torch.sin(math.pi * x) * torch.exp(-2 * NU * math.pi ** 2 * t)


def grad(u, x):
    """Derivative of u with respect to x, kept differentiable (create_graph=True)."""
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def residual(u_fn, x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_fn(x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    return u_t - NU * u_xx


torch.manual_seed(0)

# --------------------------------------------------------------------------
print("PART 1 -- residual of the exact solution (1000 random points)")
for dtype in (torch.float32, torch.float64):
    x = torch.rand(1000, 1, dtype=dtype)
    t = torch.rand(1000, 1, dtype=dtype)
    r = residual(exact, x, t)
    print(f"  {str(dtype):<14} max |R[u_exact]| = {r.abs().max().item():.2e}")
print("  -> zero up to rounding: the equation holds EVERYWHERE, not on average.\n")

# --------------------------------------------------------------------------
print("PART 2 -- three candidates: residual term and initial-condition term")
x = torch.rand(1000, 1)
t = torch.rand(1000, 1)
x0 = torch.rand(1000, 1)
ic_target = torch.sin(math.pi * x0)
print(f"  {'candidate':<28}{'mean R^2':>12}{'IC error':>12}")
for name, fn in [("exact solution", exact),
                 ("u = 0", zero),
                 ("sin(pi x) exp(-2 nu pi^2 t)", too_fast)]:
    r2 = residual(fn, x, t).pow(2).mean().item()
    ic = (fn(x0, torch.zeros_like(x0)) - ic_target).pow(2).mean().item()
    print(f"  {name:<28}{r2:>12.2e}{ic:>12.2e}")
print("  -> u = 0 satisfies the PDE perfectly and is still wrong: that is why the")
print("     loss also contains the initial and boundary conditions.\n")

# --------------------------------------------------------------------------
print("PART 3 -- u_xx at one point: autograd vs finite differences")
x_pt, t_pt = 0.3, 0.5
analytic = -(math.pi ** 2) * math.sin(math.pi * x_pt) * math.exp(-NU * math.pi ** 2 * t_pt)
print(f"  analytic u_xx(0.3, 0.5) = {analytic:.8f}\n")
print(f"  {'method':<22}{'float32 error':>16}{'float64 error':>16}")

row = {}
for dtype in (torch.float32, torch.float64):
    xx = torch.tensor([[x_pt]], dtype=dtype, requires_grad=True)
    tt = torch.tensor([[t_pt]], dtype=dtype)
    u_xx = grad(grad(exact(xx, tt), xx), xx).item()
    row[dtype] = abs(u_xx - analytic)
print(f"  {'autograd':<22}{row[torch.float32]:>16.1e}{row[torch.float64]:>16.1e}")

for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    errs = []
    for dtype in (torch.float32, torch.float64):
        xx = torch.tensor(x_pt, dtype=dtype)
        tt = torch.tensor(t_pt, dtype=dtype)
        hh = torch.tensor(h, dtype=dtype)
        fd = (exact(xx + hh, tt) - 2 * exact(xx, tt) + exact(xx - hh, tt)) / hh ** 2
        errs.append(abs(fd.item() - analytic))
    print(f"  {'finite diff, h=%g' % h:<22}{errs[0]:>16.1e}{errs[1]:>16.1e}")
print("  -> finite differences: large h = truncation error, small h = cancellation")
print("     error. Shrinking h makes float32 WORSE, and the best h is different in")
print("     each precision. Autograd has no h to tune and no truncation error.")
