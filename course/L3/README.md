# Lesson 3 · The machine learning of the project (Tuesday 27 October 2026)

Same environment as Lessons 1 and 2: activate your virtual environment and run `git pull` at the start of the lesson.
No new packages are needed.

## During the lesson

| Script | What it does | Time |
|---|---|---|
| `demo_l3_slope.py` | the physics of Lesson 1 written as a derivative of the network (autograd, as in a PINN): `--constraint none / value / slope / linear`; `--no-chain-rule` forgets the factor 18/π; `--compare` says, seed by seed, whether the constraint helps; `--lam` sets the weight of the physics term | about 30 s on CPU (60 s with `--compare`) |
| `../L1/audit_replication.py --full` | the defects of the paper, including the leakage check (check 5) | a few s |

```
python course/L3/demo_l3_slope.py --points 6
python course/L3/demo_l3_slope.py --points 6 --constraint slope
python course/L3/demo_l3_slope.py --points 6 --constraint slope --no-chain-rule
python course/L1/audit_replication.py --full
```

The folder `expected_output/` has the output of these commands, for comparison.

## Homework (about 1 hour, before the kickoff)

1. **Value, slope or line? (20 min).** Run `python course/L3/demo_l3_slope.py --points 6 --compare` with `--constraint value`, `--constraint slope` and `--constraint linear`, then `--constraint linear --activation relu`. Fill a table: in how many seeds out of 5 does the constraint help? Why does the linearity constraint do nothing with ReLU?
2. **Read `course/Project_Objectives.md` (30 min)**, especially T4 and T5, and come to the kickoff with a preferred role.
3. Optional: `--constraint slope --lam 0.01` and `--lam 100`. What changes, and why?

Please read before you run, and run only inside your virtual environment.
