# Lesson 1 · From PDE to residual (Tuesday 20 October 2026, 18:15)

We run code live in this lesson, and you run it on your own laptop at the same time.
Please arrive with the environment already working.

## Setup (about 20 minutes, do it before the lesson)

You need Python 3.10 or newer, about 2 GB of free disk space, and an internet connection for the
download. No GPU is needed. From the root of this repository:

**1. Create and activate a virtual environment**

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**2. Install torch first (this order matters)**

```
# Linux: CPU-only build, about 200 MB
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Windows / macOS
pip install torch
```

On Linux do not run a plain `pip install torch`: it downloads the CUDA build (several GB of
`nvidia-*` packages), which is useless without an NVIDIA GPU and can fail with
`No space left on device`.

**3. Install the other libraries and check**

```
pip install -r course/requirements.txt
python course/L1/check_setup.py
```

You are ready when the last line says `ALL CHECKS PASSED`. It takes about 10 seconds and needs
no GPU and no internet. If it fails, send the full output to the organisers.

### Troubleshooting

- **`No space left on device` during `pip install`, but `df -h` shows free space.** Pip unpacks
  downloads into the temporary folder, which on some Linux systems (Fedora and others) is a small
  RAM-backed `/tmp`. Point it at your disk and retry:
  ```
  mkdir -p ~/pip-tmp
  TMPDIR=~/pip-tmp pip install -r course/requirements.txt
  rm -rf ~/pip-tmp
  ```
  Also check that you installed the CPU build of torch (step 2).
- **A previous install failed halfway.** Delete the environment and start again from step 1:
  `rm -rf .venv` (Windows: delete the `.venv` folder).
- **`pip install` is slow or repeats downloads.** Add `--no-cache-dir` only if your disk is nearly
  full; otherwise pip reuses its cache.
- **Command `python` not found.** Try `python3` instead.

## Before the lesson (recommended)

- Barba, CFD Python, steps 1 to 4: https://github.com/barbagroup/CFDPython
- PyTorch autograd tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Read `main.py` and open the CSVs in this repository. Write down anything that looks odd. No pressure, nothing to hand in.

Please read before you run, and run only inside your virtual environment.

## During the lesson

The other scripts are added to this folder at the start of the lesson: run `git pull`.

| Script | What it does | Time |
|---|---|---|
| `check_setup.py` | environment check | about 10 s |
| `audit_replication.py` | five checks on the paper's data (the CSVs in the repository root); run `python course/L1/audit_replication.py` | a few s |
| `lab_l2_residual.py` | residual of an exact solution, autograd vs finite differences | about 2 s |
| `demo_l2_pinn_1d.py` | a PINN for 1D diffusion, no data | about 30 s on CPU |
