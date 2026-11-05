"""Collect the OAT campaign: one table of all cases and the sensitivity ranking of the parameters.

    python oat_collect.py <plan.csv> <campaign_dir> [--level L0] [--np 6]

campaign_dir = $FW3D_RUNS/<CAMPAIGN> (cases <case_id>_<level>_np<np>/ written by run_case_hpc.sh).
Writes into campaign_dir: oat_results.csv (one row per case), oat_ranking.csv, oat_ranking.md, oat_ranking.png,
and prints the ranking.

Effect of a parameter = coefficient at the high end minus the low end, in % of the baseline (CL = downforce
coefficient, CD, L/D = CL/CD). Noise floor = 2 x the median half-amplitude of CL left in the finished cases
(autostop 'oscillating' leaves up to ~0.3 %): effects below it are marked as not resolved. Cases whose own CL
half-amplitude exceeds 1 % are listed as unsteady (their effect is uncertain).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_case(d: Path) -> dict:
    r = {"status": (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "missing"}
    if (d / "report3d.json").exists():
        rep = json.loads((d / "report3d.json").read_text())
        c = rep.get("coeffs", {}).get("coeffs_total", {})
        if c:
            r.update(CL=c["Cl"]["mean"], CD=c["Cd"]["mean"], CL_amp_pct=c["Cl"]["amp_pct"], CD_amp_pct=c["Cd"]["amp_pct"])
            r["LD"] = r["CL"] / r["CD"]
        for part in ("wing", "nose", "hanger"):
            cp = rep.get("coeffs", {}).get(f"coeffs_{part}")
            if cp:
                r[f"CL_{part}"] = cp["Cl"]["mean"]
        r["cells"] = rep.get("mesh", {}).get("cells")
        r["layers_wing_pct"] = rep.get("layers", {}).get("wing", {}).get("coverage_pct")
        r["iterations"] = rep.get("solver", {}).get("iterations")
        t = rep.get("timings_s", {})
        r["wall_h"] = round(sum(v for k, v in t.items() if k != "total") / 3600, 2) if t else None
        r["peak_rss_GB"] = rep.get("peak_rss_GB")
    if (d / "autostop.json").exists():
        a = json.loads((d / "autostop.json").read_text())
        r["autostop"] = a.get("flag")
    if (d / "deform.json").exists():
        g = json.loads((d / "deform.json").read_text())["geometry"]
        r["h_over_c"], r["pitch_deg"] = g["h_te_over_c"], g["pitch_deg_lower_edge"] if "pitch_deg_lower_edge" in g else g["pitch_deg"]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--level", default="L0")
    ap.add_argument("--np", type=int, default=6)
    a = ap.parse_args()
    plan = list(csv.DictReader(open(a.plan)))
    rows = []
    for p in plan:
        d = a.campaign / f"{p['case_id']}_{a.level}_np{a.np}"
        rows.append({"idx": int(p["idx"]), "case_id": p["case_id"], "param": p["param"], "value": float(p["value"]),
                     **load_case(d)})
    cols = ["idx", "case_id", "param", "value", "status", "CL", "CD", "LD", "CL_amp_pct", "CD_amp_pct", "CL_wing",
            "CL_nose", "CL_hanger", "autostop", "iterations", "cells", "layers_wing_pct", "wall_h", "peak_rss_GB",
            "h_over_c", "pitch_deg"]
    with open(a.campaign / "oat_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_done = sum(r["status"] == "done" for r in rows)
    print(f"cases done {n_done}/{len(rows)}; status: " +
          ", ".join(f"{s} {sum(r['status'] == s for r in rows)}" for s in sorted({r['status'] for r in rows})))

    ok = lambda r: r.get("status") == "done" and "CL" in r       # effects only from finished cases
    base = next((r for r in rows if r["param"] == "baseline" and ok(r)), None)
    if base is None:
        print("no baseline result yet: ranking not possible")
        return
    amps = [r["CL_amp_pct"] for r in rows if ok(r)]
    amps.sort()
    noise = 2 * amps[len(amps) // 2]           # 2 x median half-amplitude: robust to one badly converged case
    by = {}
    for r in rows:
        if r["param"] != "baseline":
            by.setdefault(r["param"], {})["lo" if r["case_id"].endswith("_lo") else "hi"] = r
    rank = []
    for p, e in by.items():
        lo, hi = e.get("lo", {}), e.get("hi", {})
        rec = {"param": p, "lo": lo.get("value"), "hi": hi.get("value"),
               "status": f"{lo.get('status', '-')}/{hi.get('status', '-')}"}
        if ok(lo) and ok(hi):
            for q in ("CL", "CD", "LD"):
                rec[f"d{q}_pct"] = 100 * (hi[q] - lo[q]) / base[q]
                rec[f"d{q}_lo_pct"] = 100 * (lo[q] / base[q] - 1)
                rec[f"d{q}_hi_pct"] = 100 * (hi[q] / base[q] - 1)
            # curvature check: the two half-effects of a linear response are equal
            rec["nonlinear"] = abs(rec["dCL_hi_pct"] + rec["dCL_lo_pct"]) > 0.5 * max(abs(rec["dCL_pct"]), noise)
            rec["resolved"] = abs(rec["dCL_pct"]) > noise or abs(rec["dCD_pct"]) > 2 * noise
        rank.append(rec)
    rank.sort(key=lambda r: -abs(r["dCL_pct"]) if "dCL_pct" in r else float("inf"))
    keys = ["param", "lo", "hi", "status", "dCL_pct", "dCD_pct", "dLD_pct", "dCL_lo_pct", "dCL_hi_pct", "dCD_lo_pct",
            "dCD_hi_pct", "nonlinear", "resolved"]
    with open(a.campaign / "oat_ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rank)
    fmt = lambda v: "-" if v is None else (f"{v:+.2f}" if isinstance(v, float) else str(v))
    md = [f"# IFW one-at-a-time sensitivity ({a.campaign.name}, {a.level}, np {a.np})", "",
          f"Baseline: CL {base['CL']:.4f}, CD {base['CD']:.4f}, L/D {base['LD']:.3f}. Cases done {n_done}/{len(rows)}. "
          f"Noise floor on CL (2 x median half-amplitude): {noise:.2f} %. Unsteady cases (CL half-amplitude > 1 %): "
          f"{', '.join(r['case_id'] for r in rows if ok(r) and r['CL_amp_pct'] > 1) or 'none'}.", "",
          "Effects = high end minus low end, % of the baseline. Sorted by |dCL|.", "",
          "| # | parameter | range | dCL % | dCD % | d(L/D) % | lo / hi dCL % | resolved | nonlinear | status |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rank, 1):
        md.append(f"| {i} | {r['param']} | {r['lo']} .. {r['hi']} | {fmt(r.get('dCL_pct'))} | {fmt(r.get('dCD_pct'))} | "
                  f"{fmt(r.get('dLD_pct'))} | {fmt(r.get('dCL_lo_pct'))} / {fmt(r.get('dCL_hi_pct'))} | "
                  f"{fmt(r.get('resolved'))} | {fmt(r.get('nonlinear'))} | {r['status']} |")
    (a.campaign / "oat_ranking.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R = [r for r in rank if "dCL_pct" in r][::-1]
        if R:
            fig, ax = plt.subplots(1, 2, figsize=(11, 0.35 * len(R) + 1.5), sharey=True)
            y = range(len(R))
            for k, (q, lab) in enumerate((("CL", "CL (downforce)"), ("CD", "CD"))):
                ax[k].barh(y, [r[f"d{q}_lo_pct"] for r in R], color="#4C78A8", label="low end")
                ax[k].barh(y, [r[f"d{q}_hi_pct"] for r in R], color="#F58518", alpha=0.8, label="high end")
                ax[k].axvline(0, color="k", lw=0.8)
                ax[k].set_xlabel(f"change of {lab} vs baseline (%)")
                ax[k].grid(axis="x", alpha=0.3)
            if noise:
                ax[0].axvspan(-noise, noise, color="0.85", zorder=0, label="noise floor")
            ax[0].set_yticks(list(y), [r["param"] for r in R])
            ax[0].legend(loc="lower right", fontsize=8)
            fig.suptitle(f"IFW one-at-a-time sensitivity, {a.level} (baseline CL {base['CL']:.3f}, CD {base['CD']:.3f})")
            fig.tight_layout()
            fig.savefig(a.campaign / "oat_ranking.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
