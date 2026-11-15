"""Collect a Sobol campaign of the IFW dataset (pilot or full): one table of inputs and outputs, and the
numbers that decide the full run (stable fraction, iterations, cost).

    python ds_collect.py <ds_plan.csv> <campaign_dir> [--rows 1-32] [--level L0] [--np 6]

Writes into campaign_dir: ds_results.csv (8 inputs + CL, CD, quality flags, cost per case) and ds_summary.md.
Quality class of a finished case:
  steady      autostop 'converged' or 'oscillating' (CL half-amplitude <= 0.25 %): a clean label;
  slow        reached the iteration cap with CL half-amplitude <= 1 % (slow wander, usable with an error bar);
  unsteady    CL half-amplitude > 1 % (separation; the mean CL is not a regime value).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oat_collect import load_case  # noqa: E402

INPUTS = ["dz_mm", "dpitch_deg", "rot_flap1", "rot_flap2", "twist_flap1", "twist_flap2", "camber_main", "camber2_main"]


def quality(r: dict) -> str:
    if r.get("status") != "done" or "CL" not in r:
        return "-"
    if r["CL_amp_pct"] > 1:
        return "unsteady"
    return "steady" if r.get("autostop") in ("converged", "oscillating") else "slow"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--level", default="L0")
    ap.add_argument("--np", type=int, default=6)
    a = ap.parse_args()
    plan = list(csv.DictReader(open(a.plan)))
    if a.rows:
        i0, i1 = map(int, a.rows.split("-"))
        plan = [p for p in plan if i0 <= int(p["idx"]) <= i1]
    rows = []
    for p in plan:
        d = a.campaign / f"{p['case_id']}_{a.level}_np{a.np}"
        r = {"idx": int(p["idx"]), "case_id": p["case_id"], **{k: float(p[k]) for k in INPUTS}, **load_case(d)}
        r["quality"] = quality(r)
        rep = d / "report3d.json"
        if rep.exists():
            r["s_per_iter"] = json.loads(rep.read_text()).get("solver", {}).get("s_per_iter_median")
        r["core_h"] = round(r["wall_h"] * a.np, 2) if r.get("wall_h") else None
        rows.append(r)
    cols = ["idx", "case_id", *INPUTS, "status", "quality", "CL", "CD", "LD", "CL_amp_pct", "CD_amp_pct", "autostop",
            "iterations", "s_per_iter", "wall_h", "core_h", "cells", "layers_wing_pct", "h_over_c", "pitch_deg"]
    with open(a.campaign / "ds_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    count = lambda key, v: sum(r.get(key) == v for r in rows)
    done = [r for r in rows if r["status"] == "done"]
    q = {k: count("quality", k) for k in ("steady", "slow", "unsteady")}
    lines = [f"# IFW dataset campaign {a.campaign.name} ({a.level}, np {a.np}), rows {plan[0]['idx']}-{plan[-1]['idx']}", "",
             f"- cases: {n}; done {len(done)}; status: " +
             ", ".join(f"{s} {count('status', s)}" for s in sorted({r['status'] for r in rows})),
             f"- quality of the done cases: steady {q['steady']}, slow {q['slow']}, unsteady {q['unsteady']}"]
    if done:
        its = [r["iterations"] for r in done if r.get("iterations")]
        ch = [r["core_h"] for r in done if r.get("core_h")]
        spi = [r["s_per_iter"] for r in done if r.get("s_per_iter")]
        cl = [r["CL"] for r in done if "CL" in r]
        lines += [f"- iterations: median {st.median(its):.0f}, max {max(its)}; reached the cap: {sum(i >= 3000 for i in its)}",
                  f"- s/iteration: median {st.median(spi):.2f}, min {min(spi):.2f}, max {max(spi):.2f}",
                  f"- cost per case: median {st.median(ch):.1f} core-h, mean {st.mean(ch):.1f}, total {sum(ch):.0f} core-h",
                  f"- projection to 1024 cases: {1024 * st.mean(ch):.0f} core-h",
                  f"- CL range {min(cl):.3f} .. {max(cl):.3f}, CD range "
                  f"{min(r['CD'] for r in done if 'CD' in r):.3f} .. {max(r['CD'] for r in done if 'CD' in r):.3f}"]
        bad = [r for r in rows if r["quality"] in ("unsteady", "slow") or r["status"] != "done"]
        if bad:
            lines += ["", "## Cases not steady or not done", "",
                      "| case | status | quality | CL | amp % | " + " | ".join(INPUTS) + " |",
                      "|---" * (5 + len(INPUTS)) + "|"]
            for r in bad:
                cl_s = f"{r['CL']:.3f}" if "CL" in r else "-"
                amp_s = f"{r['CL_amp_pct']:.2f}" if "CL_amp_pct" in r else "-"
                lines.append(f"| {r['case_id']} | {r['status']} | {r['quality']} | {cl_s} | {amp_s} | " +
                             " | ".join(f"{r[k]:g}" for k in INPUTS) + " |")
    (a.campaign / "ds_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
