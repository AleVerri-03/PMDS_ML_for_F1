"""Summarise a finished (or running) 3D trial: mesh, timings, memory, coefficients, convergence, cost projection.

Usage: python report3d.py <case_dir> [--np N]
Writes <case>/report3d.json and <case>/report3d.png (CL/CD history + residuals). Only numpy and matplotlib.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

# experimental reference (Zerihan 2001, Fig. 36b, datum flap, alpha 1 deg), digitised from the figure
# (fig36b_datum_digitised.csv, +-0.01): a comparison, NOT a validation
_EXP = np.array([[float(x) for x in l.split(",")] for l in
                 (Path(__file__).resolve().parent / "fig36b_datum_digitised.csv").read_text().splitlines()
                 if l.strip() and not l.startswith("#") and not l.startswith("h_over_c")])
EXP_H, EXP_CL_VALUES = _EXP[:, 0], _EXP[:, 1]


def exp_cl(h: float) -> float | None:
    """Measured CL at h/c: exact point if digitised there (+-0.003 in h/c), else linear interpolation."""
    if h < EXP_H.min() or h > EXP_H.max():
        return None
    j = np.argmin(abs(EXP_H - h))
    return float(EXP_CL_VALUES[j]) if abs(EXP_H[j] - h) <= 0.003 else float(np.interp(h, EXP_H, EXP_CL_VALUES))
CL_2D_REF = {0.211: 4.279126}           # 2D reference case of the dataset (laptop/desktop, level 1)


def _coeff_file(case: Path, name: str) -> Path | None:
    for pat in (f"postProcessing/{name}/*/coefficient.dat", f"postProcessing/{name}/*/forceCoeffs.dat"):
        fs = sorted(case.glob(pat))
        if fs:
            return fs[-1]
    return None


def read_coeffs(case: Path, name: str) -> dict | None:
    f = _coeff_file(case, name)
    if f is None:
        return None
    header = None
    rows = []
    for line in f.read_text().splitlines():
        if line.startswith("#"):
            toks = line[1:].split()
            if toks and toks[0] == "Time":
                header = toks
            continue
        if line.strip():
            rows.append([float(x) for x in line.split()])
    a = np.array(rows)
    cols = {h: i for i, h in enumerate(header or [])}
    get = lambda *names: next((a[:, cols[n]] for n in names if n in cols), None)
    return {"it": a[:, 0], "Cl": get("Cl"), "Cd": get("Cd")}


def window_stats(y: np.ndarray, n: int = 500) -> dict:
    w = y[-n:] if len(y) >= n else y
    half = len(w) // 2
    return {"mean": float(w.mean()), "amp_pct": float(100 * (w.max() - w.min()) / 2 / abs(w.mean())),
            "drift_pct": float(100 * (w[half:].mean() - w[:half].mean()) / abs(w.mean()))}


def parse_checkmesh(case: Path) -> dict:
    t = (case / "log.checkMesh").read_text() if (case / "log.checkMesh").exists() else ""
    g = lambda pat, conv=float: (conv(m.group(1)) if (m := re.search(pat, t)) else None)
    return {"cells": g(r"cells:\s+(\d+)", int), "max_nonortho_deg": g(r"Mesh non-orthogonality Max: ([\d.eE+-]+)"),
            "max_skewness": g(r"Max skewness = ([\d.eE+-]+)"), "mesh_ok": "Mesh OK" in t,
            "failed_checks": g(r"Failed (\d+) mesh checks", int) or 0}


def parse_layers(case: Path) -> dict:
    t = (case / "log.snappyHexMesh").read_text() if (case / "log.snappyHexMesh").exists() else ""
    # table printed at the end of layer addition: patch faces layers overall thickness [m] [%]
    # v1912: name faces layers thick% ; v2606 adds a "target" layers column after faces
    rows = re.findall(r"^(\S+)\s+(\d+)\s+(?:\d+\s+)?([\d.]+)\s+([\d.eE+-]+)\s+([\d.]+)\s*$", t, re.M)
    return {r[0]: {"faces": int(r[1]), "layers_avg": float(r[2]), "thickness_m": float(r[3]), "coverage_pct": float(r[4])}
            for r in rows if not r[0].startswith("-")}


def parse_solver(case: Path) -> dict:
    t = (case / "log.simpleFoam").read_text() if (case / "log.simpleFoam").exists() else ""
    times = [float(x) for x in re.findall(r"^ExecutionTime = ([\d.]+) s", t, re.M)]
    res = {}
    for f in ("Ux", "Uy", "Uz", "p", "k", "omega"):
        r = re.findall(rf"Solving for {f}, Initial residual = ([\d.eE+-]+)", t)
        if r:
            res[f] = np.array([float(x) for x in r])
    out = {"iterations": len(times), "converged_residualControl": "SIMPLE solution converged" in t}
    if len(times) > 20:
        dt = np.diff(times)
        out["s_per_iter_median"] = float(np.median(dt[10:]))
    for f, v in res.items():   # p is solved once per non-orthogonal corrector: keep the first solve of each iteration
        if len(times) and len(v) >= 2 * len(times) and len(v) % len(times) == 0:
            res[f] = v[::len(v) // len(times)]
    out["final_initial_residuals"] = {k: float(v[-1]) for k, v in res.items()}
    out["_res"] = res
    return out


def main(case: Path):
    meta = json.loads((case / "case3d.json").read_text())
    p = meta["params"]
    rep = {"case": str(case), "level": p["level"], "h_over_c": p["h_over_c"], "n_procs": p["n_procs"]}
    tm = {}
    if (case / "timings.txt").exists():
        for line in (case / "timings.txt").read_text().splitlines():
            name, dt, rc = line.split()
            tm[name] = float(dt)
    rep["timings_s"] = tm
    if (case / "mem.log").exists():
        rss = [int(t[1]) for t in (l.split() for l in (case / "mem.log").read_text().splitlines())
               if len(t) == 2 and t[1].isdigit()]
        rep["peak_rss_GB"] = max(rss) / 1024 if rss else None   # sampled every 5 s, whole container
    rep["mesh"] = parse_checkmesh(case)
    rep["layers"] = parse_layers(case)
    sol = parse_solver(case)
    res = sol.pop("_res")
    rep["solver"] = sol
    cells = rep["mesh"]["cells"]
    if cells and sol.get("s_per_iter_median"):
        rep["solver"]["us_per_cell_iter_per_rank"] = sol["s_per_iter_median"] * p["n_procs"] / cells * 1e6

    coeffs = {}
    for name in ("coeffs_total", "coeffs_wing", "coeffs_main", "coeffs_flap", "coeffs_endplate", "coeffs_nose",
                 "coeffs_hanger"):
        c = read_coeffs(case, name)
        if c is not None and c["Cl"] is not None:
            coeffs[name] = {"Cl": window_stats(c["Cl"]), "Cd": window_stats(c["Cd"]), "_hist": c}
    rep["coeffs"] = {k: {kk: vv for kk, vv in v.items() if kk != "_hist"} for k, v in coeffs.items()}
    if "coeffs_total" in coeffs:
        cl = rep["coeffs"]["coeffs_total"]["Cl"]
        rep["coeffs_ok"] = cl["amp_pct"] < 0.5 and abs(cl["drift_pct"]) < 0.5
        h = p["h_over_c"]
        ref = meta.get("reference")
        e = None if (ref or meta.get("case_type") == "ifw") else exp_cl(h)   # Fig. 36b is the Zerihan wing only
        if ref:
            cd = rep["coeffs"]["coeffs_total"]["Cd"]["mean"]
            rep["compare"] = {"CL_3D": cl["mean"], "CD_3D": cd, "CL_ref": ref["CL"], "CD_ref": ref["CD"],
                              "CL_rel_diff_pct": 100 * (cl["mean"] / ref["CL"] - 1),
                              "CD_rel_diff_pct": 100 * (cd / ref["CD"] - 1), "ref_source": ref["source"]}
        if e is not None:
            rep["compare"] = {"CL_3D": cl["mean"], "CL_exp_fig36b": e, "rel_diff_pct": 100 * (cl["mean"] / e - 1),
                              "Cl_2D_dataset": CL_2D_REF.get(h)}
    if (case / "autostop.json").exists():
        rep["autostop"] = json.loads((case / "autostop.json").read_text())

    # cost projection: a case = mesh + solve, at this level and rank count
    if tm.get("total"):
        per_case_h = tm["total"] / 3600
        rep["projection"] = {"hours_per_case": per_case_h, "cases_per_day": 24 / per_case_h,
                             **{f"days_for_{n}_cases": n * per_case_h / 24 for n in (20, 80, 160, 400)}}
    (case / "report3d.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for name, c in coeffs.items():
            ax[0].plot(c["_hist"]["it"], c["_hist"]["Cl"], label=name.replace("coeffs_", ""))
            ax[1].plot(c["_hist"]["it"], c["_hist"]["Cd"], label=name.replace("coeffs_", ""))
        if "compare" in rep:
            ref_cl = rep["compare"].get("CL_exp_fig36b", rep["compare"].get("CL_ref"))
            ax[0].axhline(ref_cl, ls="--", c="k", lw=1,
                          label="exp. (Fig. 36b, digitised)" if "CL_exp_fig36b" in rep["compare"] else "reference (Nektar++)")
        ax[0].set(xlabel="iteration", ylabel="CL (downforce +)"); ax[0].legend(fontsize=7)
        ax[1].set(xlabel="iteration", ylabel="CD"); ax[1].legend(fontsize=7)
        for f, r in res.items():
            ax[2].semilogy(r, label=f, lw=0.8)
        ax[2].set(xlabel="iteration", ylabel="initial residual"); ax[2].legend(fontsize=7)
        fig.suptitle(f"{'Imperial Front Wing' if meta.get('case_type') == 'ifw' else '3D Zerihan wing'}, "
                     f"h/c {p['h_over_c']}, level {p['level']}, {cells} cells")
        fig.tight_layout(); fig.savefig(case / "report3d.png", dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main(Path(sys.argv[1]))
