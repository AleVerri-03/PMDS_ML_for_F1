# Lesson 1 · Neural networks and the audit (Tuesday 20 October 2026, 18:15)

We run code live in this lesson, and you run it on your own laptop at the same time.
Please arrive with the environment already working.

## Setup (about 20 minutes, do it before the lesson)

From the root of this repository:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r course/requirements.txt
# Linux, CPU only (much smaller than the default wheel):
# pip install torch --index-url https://download.pytorch.org/whl/cpu

python course/L1/check_setup.py
```

You are ready when the last line says `ALL CHECKS PASSED`. It takes about 10 seconds and needs
no GPU and no internet. If it fails, send the full output to the organisers.

## During the lesson

The other scripts are added to this folder at the start of the lesson: run `git pull`.

| Script | What it does | Time |
|---|---|---|
| `check_setup.py` | environment check | about 10 s |
| `demo_l1_network.py` | a straight line against a small neural network on a lift curve; `--points 6 --physics` adds our physics loss | about 10 s on CPU |
| `audit_replication.py` | four short checks on the paper's data (the CSVs in the repository root); `--full` for the long version | a few s |

## Homework (about 1.5 hours)

- `python course/L1/demo_l1_network.py --hidden 4`, then `--hidden 32` and `--hidden 128`: compare the validation errors. When is a bigger network worse?
- `python course/L1/demo_l1_network.py --points 6`, with and without `--physics`, for `--seed 0` to `--seed 4`. In how many runs does the physics term help? Bring the table to Lesson 2.
- 3Blue1Brown, "But what is a neural network?", chapters 1 and 2: https://www.youtube.com/watch?v=aircAruvnKk
- Recommended: `python course/L1/audit_replication.py --full`. Do not trust the lesson.
- Optional, a preview of Lesson 2: Moseley, "So, what is a physics-informed neural network?": https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/

## Before the lesson (recommended)

- Read `main.py` and open the CSVs in this repository. Write down anything that looks odd. No pressure, nothing to hand in.

Please read before you run, and run only inside your virtual environment.
