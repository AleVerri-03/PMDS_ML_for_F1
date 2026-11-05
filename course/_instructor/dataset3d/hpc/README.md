# dataset3d/hpc: casi IFW su un cluster (Apptainer + SLURM)

È il gemello per cluster di `../run_ifw.sh` (che usa podman sul portatile). Serve per il test di sensibilità
"uno alla volta" (OAT) sui 18 parametri dell'IFW, cioè 37 casi L0. La procedura passo per passo per chi lo
esegue sul cluster è in **AGENT_INSTRUCTIONS.md** (in inglese, scritta per l'agent SSH).

| file | ruolo |
|---|---|
| `setup_cluster.sh` | Esegue, in ordine: sondaggio della macchina; .sif (build con python, oppure pull dell'immagine nuda); python; STEP -> STL; test. |
| `ifw_v2606.def` | Ricetta Apptainer: `opencfd/openfoam-run` v2606 fissato (stesso digest del portatile) più gmsh, numpy, scipy e matplotlib. |
| `lib.sh` | Percorsi, `fw3d.env`, funzioni di appoggio `py` e `foam`. |
| `oat_plan.py`, `oat_plan.csv` | Il piano: baseline più 18 parametri × {basso, alto}. Con `--check` verifica la validità geometrica (37/37 il 05/11/2026). |
| `run_case_hpc.sh` | Un caso: deformazione → caso → Allrun nel .sif con il watcher di autostop → report. Si può rilanciare: i casi già finiti vengono saltati. |
| `oat_array.sbatch` | Job array SLURM (6 core, 16 GB, 8 h per caso). |
| `oat_local.sh` | Lo stesso senza scheduler. |
| `oat_collect.py` | Tabella dei risultati e classifica (`oat_ranking.md/.csv/.png`). |

Differenze rispetto alla catena del portatile:

- OpenFOAM gira con `apptainer exec --cleanenv`, così l'OpenMPI del container non vede le variabili SLURM dell'host;
- si usa `mpirun --oversubscribe --bind-to none`, con solo trasporto in memoria condivisa (`FW3D_MPIRUN_OPTS` e `FW3D_OMPI_ENV` in `lib.sh`);
- dopo il report i campi vengono cancellati, tranne quelli della baseline;
- il monitor di memoria di `Allrun` conta solo i processi dell'utente.

Collaudo del 05/11/2026, nel cloud con OpenFOAM v1912 nativo (`FW3D_FOAM=native`):

- caso smoke della variante `rot_flap2_hi` completo in 12 minuti su 2 core;
- funzionano anche il salto dei casi già fatti, la raccolta dei risultati, gli step `python` (venv) e `geometry` (STL identico al portatile, `ae3d2579`).

**Non è ancora collaudato** il percorso Apptainer: nel sandbox non c'è Apptainer né accesso al registry Docker.

## Branch `hpc-ifw` (come il kit arriva sul cluster)

Su `main`, `course/_instructor/` è ignorata da git, quindi il kit viaggia su un branch dedicato con i file
aggiunti con `-f`. Dalla radice della repo sul portatile:
```bash
git switch -c hpc-ifw
git add -f course/_instructor/dataset3d ':(exclude)course/_instructor/dataset3d/**/__pycache__/**' imperial/ifw.stp
git commit -m "IFW 3D kit + cluster scripts for the OAT sensitivity test"
git push -u origin hpc-ifw
git switch main
```
Restano fuori `imperial/stl/` (100 MB) e `imperial/ifw_mesh.xml` (418 MB): gli STL si rigenerano sul cluster
da `ifw.stp`, oppure si copiano con rsync.
