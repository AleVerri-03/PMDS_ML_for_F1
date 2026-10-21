# Lesson 2 · The context of the network: equations, the dataset, the SDF (Thursday 22 October 2026)

Same environment as Lesson 1: activate your virtual environment and run `git pull` at the start of the lesson.
No new packages are needed.

## During the lesson

| Script | What it does | Time |
|---|---|---|
| `demo_l2_pinn_ode.py` | a PINN for d*y*/d*t* = −*k y*, *y*(0) = 1, with no data; `--no-start` drops the starting point, `--k 3` changes *k*, `--plot` saves a PNG | about 10 s on CPU |

In the second half of the lesson we run one CFD simulation of our front wing live (OpenFOAM, on the instructor's laptop).
You do not need OpenFOAM.

## After the lesson: think about the problem

No homework: Lesson 3 is about machine learning. Four questions to think about; we will discuss them together in the last lesson.

1. A new wing arrives. What would you give the network as input, and why?
2. We will have about 1,200 runs. Which ones would you keep away from training, to test the network honestly?
3. Very close to the ground a run does not converge. What should the network do there?
4. What can a physics law tell the network that the data cannot?

If you want to play, nothing to hand in:

- `demo_l2_pinn_ode.py --k 3`, `--k 10`, `--no-start --plot`.
- `demo_l2_pinn_1d.py`: the same recipe with two variables, heat in a rod (*u<sub>t</sub>* = ν *u<sub>xx</sub>*). Read the top of the file, then run it (about 30 s).
- `lab_l2_residual.py`: the residual of the heat equation, and why autograd is exact.

Please read before you run, and run only inside your virtual environment.
