# IFW one-at-a-time sensitivity test on the Polimi cluster: instructions for the SSH agent

You are working in an SSH shell on a cluster of Politecnico di Milano for the user (Verri).
Goal: run the **one-at-a-time (OAT) sensitivity test** of the Imperial Front Wing (IFW). That is 37 RANS CFD
cases (OpenFOAM v2606 in an Apptainer container): the baseline, then each of 18 parameters alone at its
low and its high end. Then collect the ranking of the parameters.

You do NOT need to understand the CFD. Follow the steps in order. Every step has a check. If a check fails,
stop, collect what the "When something fails" table asks for, and report to the user. Do not improvise fixes
in the Python or OpenFOAM files.

## Rules

1. **Login node = light work only**: git, editing `fw3d.env`, `sbatch`, `squeue`, reading logs, and setup
   steps 2-3 (download/build). Never run `run_case_hpc.sh` or `oat_local.sh` on the login node.
2. Do not edit any file of the repository except `hpc/fw3d.env`, which is local and not in git. Do not commit or push.
3. Write only inside the clone and inside `$FW3D_HOME` (step 1). Never delete anything outside `$FW3D_HOME`.
4. The partition, account and QoS to use are a decision of the user. Propose them from `probe.txt` and ask
   before the first `sbatch`.
5. Do not submit more than the steps say. Use the whole array only after the baseline gate (step 7) passes.
6. Report to the user at the end of steps 1, 6, 7 and 9, and on any failure. Keep reports short: what ran,
   the numbers, and what is next.

## Step 0: clone the branch

The user gives you the repository URL. If `git clone` asks for credentials, stop and ask the user. They set
up the SSH key or token themselves; never handle tokens.

```bash
cd ~                                   # or the directory the user indicates
git clone -b hpc-ifw --single-branch <REPO_URL> F1-Front-Wing-Physics-Informed-NN
cd F1-Front-Wing-Physics-Informed-NN/course/_instructor/dataset3d/hpc
ls                                     # expect: AGENT_INSTRUCTIONS.md lib.sh setup_cluster.sh run_case_hpc.sh oat_*.{py,sh,sbatch} ifw_v2606.def oat_plan.csv
ls ../../../../imperial/ifw.stp        # the CAD (4.5 MB) must be there
```
From now on, **every command runs from this `hpc/` directory**.

## Step 1: probe the machine (changes nothing)

```bash
bash setup_cluster.sh probe
```
Read `probe.txt` and report to the user:

- scheduler: SLURM, PBS, or none;
- partitions with cores per node, memory and time limit; your accounts/QoS;
- apptainer or singularity: present? (also check `module avail`);
- python version;
- whether the login node reaches `registry-1.docker.io` and `pypi.org` (HTTP code 200/401/404 = reachable;
  000 = blocked);
- storage: where the big scratch area is and the quotas.

Decide `FW3D_HOME`, the directory for the container (about 1-2 GB), STL (about 100 MB) and runs (about 1-3 GB
while a case runs; about 20 MB per finished case after cleanup; 37 cases fit in about 5 GB if 8 run at once).
Use the scratch/work area, not a small `$HOME`. If modules are needed (e.g. `module load apptainer`), set
them too:
```bash
export FW3D_HOME=/path/to/scratch/$USER/fw3d      # physical path with enough quota
export FW3D_MODULES="apptainer"                     # only if apptainer comes from a module; else leave unset
```
These are saved into `fw3d.env` by the next step.

## Step 2: container

```bash
bash setup_cluster.sh container
```
- Route A tries `apptainer build --fakeroot` of `ifw_v2606.def` (OpenFOAM + Python with gmsh). This takes
  10-30 min; its log is `$FW3D_HOME/build_sif.log`.
- If fakeroot is not allowed, route B pulls the bare OpenFOAM image (`apptainer pull`), about 5-10 min.
- If the login node has no internet, both fail. Report this to the user. The .sif can be built elsewhere
  and copied to `$FW3D_HOME/ifw_v2606.sif`. The script then picks it up.

Check: the last line prints the sha256 of the .sif, and `cat fw3d.env` shows `FW3D_SIF=...` and
`FW3D_SIF_HAS_PY=1` (route A) or `0` (route B).

## Step 3: python side

```bash
bash setup_cluster.sh python
```
Route A uses `/opt/venv` inside the .sif. Route B creates `$FW3D_HOME/venv` with numpy/scipy/matplotlib
(needs python >= 3.10, from a module if necessary: add it to `FW3D_MODULES` and re-run). It also tries gmsh.
Check: it prints `python in .sif OK` or `host venv OK`, and `fw3d.env` has `FW3D_HAS_GMSH=0|1`.

## Step 4: geometry (STEP -> STL)

```bash
bash setup_cluster.sh geometry
```
- With gmsh (`FW3D_HAS_GMSH=1`): it tessellates `imperial/ifw.stp` (about 5 min, 2-3 GB RAM), then classifies
  the wing elements. If the login node kills it (out of memory or CPU limit), run the same step on a
  compute node:
  `srun -p <partition> -c 2 --mem=8G -t 00:30:00 bash setup_cluster.sh geometry`.
- Without gmsh: the script stops and prints an `rsync` command. Ask the USER to run it **from the laptop**
  (the STL are already in the laptop repo, `imperial/stl/`), then re-run this step.

Check: it prints the element face counts `{'main': 12, 'flap1': 20, 'flap2': 20, 'endplate': 118}`. A line
`wing.stl identical to the laptop one` is ideal. A WARNING about a different sha256 is acceptable: tell the
user.

Then check that all 37 variants are geometrically valid (about 5 min of one core: use srun if the login node
is strict):
```bash
. lib.sh && py oat_plan.py oat_plan.csv --check "$FW3D_STL" | tail -2    # expect: valid 37/37
```

## Step 5: quick test of the tools

```bash
bash setup_cluster.sh test
```
Check: it prints `OpenFOAM v2606`, two hostnames from `mpirun -np 2`, `python OK`, `37 cases -> ...`, and
the content of `fw3d.env`.

## Step 6: smoke test on a compute node (about 15-30 min)

Ask the user for partition/account (and QoS), then submit 2 cases at the coarse "smoke" level: the baseline
(row 1) and one deformed variant (row 3, `rot_flap2_hi`):
```bash
mkdir -p logs
sbatch --array=1,3 -p <partition> [-A <account>] [--qos <qos>] --time=01:00:00 \
       --export=ALL,LEVEL=smoke,CAMPAIGN=smoke oat_array.sbatch oat_plan.csv
squeue -u $USER                                 # wait until both disappear
cat logs/ifw-oat_*_1.out logs/ifw-oat_*_3.out | tail -30
. lib.sh; for d in $FW3D_RUNS/smoke/*_smoke_np6; do echo "$d: $(cat $d/STATUS)"; done
```
Check, for both cases:

- `STATUS` = `done`;
- `log.Allrun` lists `surfaceFeatureExtract ... simpleFoam` all with `rc 0`;
- `report3d.json` exists and `python3 -c "import json; print(json.load(open('<case>/report3d.json'))['coeffs']['coeffs_total'])"`
  prints finite numbers.

After 300 smoke iterations CL is not converged. Any finite positive value is fine here.

Report to the user: smoke cells (`grep -m1 cells <case>/log.checkMesh`), wall time (`timings.txt`), peak memory
(`report3d.json` peak_rss_GB).

## Step 7: baseline gate at the real level (L0), about 1-3 h

```bash
sbatch --array=1 -p <partition> [-A <account>] [--qos <qos>] oat_array.sbatch oat_plan.csv
```
When it ends (`squeue -u $USER` empty), check `$FW3D_RUNS/oat_v1/base_L0_np6/`:

| quantity | laptop reference (04/11/2026, same image) | acceptable |
|---|---|---|
| STATUS | done | done |
| cells (`report3d.json` mesh.cells) | 2,170,474 | within 2 % |
| CL (coeffs_total Cl mean) | 5.389 | 5.30 - 5.48 |
| CD (coeffs_total Cd mean) | 0.594 | 0.58 - 0.61 |
| autostop.json flag | converged / oscillating | either (missing = ran to 3000 iterations: report it) |

Report these numbers and the wall time to the user, then wait for their go-ahead before step 8.

## Step 8: the whole campaign

```bash
sbatch --array=2-37%8 -p <partition> [-A <account>] [--qos <qos>] oat_array.sbatch oat_plan.csv
```
`%8` = at most 8 cases at once. Use the value the user agrees (it depends on the fair share). Each task asks
for 6 cores, 16 GB and 8 h. Rows already `done` are skipped, so re-submitting the same array after a problem
is safe.

Monitoring (light, every 30-60 min is enough):
```bash
squeue -u $USER
. lib.sh; cat $FW3D_RUNS/oat_v1/*_L0_np6/STATUS | sort | uniq -c
tail -3 $FW3D_RUNS/oat_v1/<case_id>_L0_np6/log.Allrun           # current step of one case
```
STATUS meanings: `running`; `done`; `invalid` (geometry rejected, not run: expected 0 here); `failed:<step>`.

## Step 9: collect and hand over

```bash
. lib.sh
py oat_collect.py oat_plan.csv "$FW3D_RUNS/oat_v1"            # table + ranking; prints the ranking
tar czf "$FW3D_HOME/oat_v1_summary.tgz" -C "$FW3D_RUNS" \
    --exclude='processor*' --exclude='constant' --exclude='*.old_*' oat_v1
ls -lh "$FW3D_HOME/oat_v1_summary.tgz"
```
Report to the user:

- the printed ranking table (`oat_ranking.md`);
- the count of done/failed cases;
- the full path of the tarball, so they can copy it to the laptop with
  `scp <cluster>:<path>/oat_v1_summary.tgz .`

The case `base_L0_np6` keeps its fields for ParaView (`processor*` directories). Do not delete it.

## When something fails

| symptom | collect and report | likely cause / allowed action |
|---|---|---|
| `setup_cluster.sh container`: build fails | last 30 lines of `$FW3D_HOME/build_sif.log` | no fakeroot: route B runs automatically; no internet: ask the user |
| `apptainer: command not found` | `module avail 2>&1 \| grep -i -E "apptainer\|singularity"` | set `FW3D_MODULES`, re-run the step |
| test: `mpirun` fails | full output | try `. lib.sh; setv FW3D_MPIRUN_OPTS "--oversubscribe"` then re-test; else report |
| `STATUS = failed:snappyHexMesh` | `tail -40 <case>/log.snappyHexMesh` | report; do not change meshing parameters |
| `STATUS = failed:simpleFoam` | `tail -60 <case>/log.simpleFoam`, `tail mem.log` | if `Killed`/OOM: re-submit that row with `--mem=24G`; if `nan`/`Floating point exception`: report |
| job killed at the time limit (STATUS stays `running`) | `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | re-submit that row with `--time=12:00:00` |
| `invalid` | `<case>/deform.json` checks | report (should not happen: 37/37 were checked) |
| anything else | the error text, the command, `cat fw3d.env` | stop and report |

Re-running a single row: `sbatch --array=<idx> ... oat_array.sbatch oat_plan.csv` (a non-`done` case is
moved aside to `<case>.old_<time>` and redone).

## No SLURM?

If `probe.txt` shows **PBS** (`qsub`), tell the user. The sbatch file must be translated (`#PBS -l
select=1:ncpus=6:mem=16gb`, `$PBS_ARRAY_INDEX` instead of `$SLURM_ARRAY_TASK_ID`). Do this only with their OK.

If there is **no scheduler** and the user confirms that you may run on a compute server directly:
```bash
PAR=2 nohup bash oat_local.sh oat_plan.csv 1 37 > oat_local.log 2>&1 &     # PAR = cases at once (x 6 cores)
```

## What the files do (for reference)

- `lib.sh`: paths, `fw3d.env`, and the helpers `py` (Python in the .sif or the venv) and `foam` (OpenFOAM in
  the .sif).
- `setup_cluster.sh`: steps 1-5.
- `oat_plan.py`: writes `oat_plan.csv` (37 rows; `--check` = geometric validity).
- `run_case_hpc.sh <plan> <idx>`: one case. It deforms the geometry (`../ifw/deform_ifw.py`), writes the
  case (`../ifw/case_ifw.py`), meshes and solves with `../zerihan3d/Allrun` in the container next to the
  autostop watcher (`../zerihan3d/autostop3d.py`, stops at the regime), then reports
  (`../zerihan3d/report3d.py`).
- `oat_array.sbatch`: SLURM array, one task = one row.
- `oat_local.sh`: the same without a scheduler.
- `oat_collect.py`: results table + ranking.
