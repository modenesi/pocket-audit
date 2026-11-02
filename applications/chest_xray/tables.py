"""Supplement tables for the chest X-ray application (NIH ChestX-ray14), from the saved run.

    python applications/chest_xray/tables.py --run-dir results/chest_xray --out results/tables

Writes s9_cxr_gate.tex (validation gate) and s9_cxr_audit.tex (mu, delta, kappa of both primary maps, with pocket
profiles from discovery images). Intervals are omitted when the gate failed (descriptive-only branch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pp(x, scale=100.0, d=1):
    return "--" if pd.isna(x) else f"{scale * x:+.{d}f}".replace("+-", "-")


def gate_table(run: Path) -> str:
    s = pd.read_csv(run / "validation_summary.csv")
    names = {"independent": "Null", "effect": "AP over-prediction (up to 5 points)", "weak_sbm": "Weak null",
             "weak_ridge_kmeans": "Weak null"}
    s = s.assign(map=s["map"].map({"sbm": "forest", "ridge_kmeans": "ridge", "combined": "combined"}),
                 gen_model=s.gen_model.map({"chex": "CheXpert", "mimic": "MIMIC"}))
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Scenario & Generating model & Map & False flags, \% [Wilson] & Coverage, \% & Reps \\", r"\midrule"]
    for r in s.itertuples():
        ff = ("-- (no true-null coordinates)" if r.null_coords_per_rep == 0 else
              f"{100 * r.fwer:.1f} [{100 * r.fwer_lo:.1f}, {100 * r.fwer_hi:.1f}]")
        L.append(f"{names.get(r.scenario, r.scenario)} & {r.gen_model} & {r.map.replace('_', ' ')} & "
                 f"{ff} & {100 * r.simultaneous_coverage:.1f} & {r.replicates} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def audit_table(run: Path, descriptive: bool) -> str:
    res = pd.read_csv(run / "audit_results.csv", keep_default_na=False, na_values=[""])
    prof = pd.read_csv(run / "pocket_profiles_discovery.csv")
    L = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
         r"Map & Pocket & Images & Female & Age & AP & $\mu$ CheXpert & $\mu$ MIMIC & $\delta$ CheXpert & $\delta$ MIMIC"
         r" & $\kappa$ \\",
         r"\midrule"]
    for mp, mname in (("sbm", "Forest"), ("ridge_kmeans", "Ridge")):
        r = res[res.partition == mp]
        pockets = ["all"] + sorted({p for p in r.pocket.astype(str) if p != "all"}, key=int)
        for p in pockets:
            def cell(kind, model):
                x = r[(r.kind == kind) & (r.model == model) & (r.pocket.astype(str) == p)]
                if x.empty or x.status.iloc[0] != "tested":
                    return "untested"
                x = x.iloc[0]
                dd = 2 if p == "all" else 1
                s = pp(x.estimate, d=dd)
                if not descriptive:
                    s += f" [{pp(x.ci_low, d=dd)}, {pp(x.ci_high, d=dd)}]" + ("$^*$" if bool(x.flag) else "")
                return s
            pr = prof[(prof.partition == mp) & (prof.pocket.astype(str) == p)]
            mu = r[(r.kind == "mu") & (r.model == "chex") & (r.pocket.astype(str) == p)]
            n = int(mu.n_focal.iloc[0]) if len(mu) else 0
            prof_cells = ("--", "--", "--") if pr.empty else (f"{100 * pr.share_female.iloc[0]:.0f}\\%",
                                                              f"{pr.mean_age.iloc[0]:.0f}",
                                                              f"{100 * pr.share_ap.iloc[0]:.0f}\\%")
            L.append(f"{mname if p == 'all' else ''} & {p} & {n:,} & " + " & ".join(prof_cells) + " & "
                     + " & ".join(cell(k, m) for k in ("mu", "delta") for m in ("chex", "mimic")) + " & "
                     + cell("kappa", "chex-mimic") + r" \\")
        L.append(r"\midrule" if mp == "sbm" else "")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def support_tables(run: Path) -> tuple[str, str, str]:
    """Untested and abstained shares by sex; combined directional shares; per-pocket counts and discovery prevalence."""
    ro = json.loads((run / "readout.json").read_text())
    L = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule", r"Map & Share of images & Female, \% & Male, \% \\", r"\midrule"]
    for mp, name in (("sbm", "Forest"), ("ridge_kmeans", "Ridge")):
        rows = [("Abstained", ro[f"abstained_{mp}"]), (r"Untested $\mu$, CheXpert", ro[f"untested_mu_{mp}_chex"]),
                (r"Untested $\mu$, MIMIC", ro[f"untested_mu_{mp}_mimic"]),
                (r"Untested $\delta$ and $\kappa$", ro[f"untested_delta_{mp}_chex"])]
        for i, (lab, v) in enumerate(rows):
            L.append(f"{name if i == 0 else ''} & {lab} & {100 * v['female']:.1f} & {100 * v['male']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    C = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule", r"Model & Flag inherited & Female, \% & Male, \% \\", r"\midrule"]
    for m, name in (("chex", "CheXpert"), ("mimic", "MIMIC")):
        for i, k in enumerate(("over", "under", "conflict")):
            v = ro[f"combined_mu_{m}"][k]
            d = 2 if k == "conflict" else 1
            C.append(f"{name if i == 0 else ''} & {k} & {100 * v['female']:.{d}f} & {100 * v['male']:.{d}f} \\\\")
    C += [r"\bottomrule", r"\end{tabular}"]
    res = pd.read_csv(run / "audit_results.csv", keep_default_na=False, na_values=[""])
    prof = pd.read_csv(run / "pocket_profiles_discovery.csv")
    P = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Forest} & \multicolumn{3}{c}{Ridge} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Pocket & Female & Male & Prev., \% & Female & Male & Prev., \% \\", r"\midrule"]

    def counts(mp, p):
        x = res[(res.partition == mp) & (res.kind == "delta") & (res.model == "chex") & (res.pocket.astype(str) == p)]
        return (f"{int(x.n_focal.iloc[0]):,}", f"{int(x.n_reference.iloc[0]):,}") if len(x) else ("--", "--")

    def prev(mp, p):
        x = prof[(prof.partition == mp) & (prof.pocket.astype(str) == p)]
        return f"{100 * x.prevalence_discovery.iloc[0]:.1f}" if len(x) else "--"
    for p in ["all"] + [str(k) for k in range(17)]:
        cf, cr = counts("sbm", p), counts("ridge_kmeans", p)
        P.append(f"{p} & {cf[0]} & {cf[1]} & {prev('sbm', p) if p != 'all' else '--'} & {cr[0]} & {cr[1]} & "
                 f"{prev('ridge_kmeans', p) if p != 'all' else '--'} \\\\")
    P += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L), "\n".join(C), "\n".join(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    man = json.loads((args.run_dir / "audit_manifest.json").read_text())
    (args.out / "s9_cxr_gate.tex").write_text(gate_table(args.run_dir) + "\n")
    (args.out / "s9_cxr_audit.tex").write_text(audit_table(args.run_dir, man["descriptive_only"]) + "\n")
    sup, comb, cnt = support_tables(args.run_dir)
    (args.out / "s9_cxr_support.tex").write_text(sup + "\n")
    (args.out / "s9_cxr_combined.tex").write_text(comb + "\n")
    (args.out / "s9_cxr_counts.tex").write_text(cnt + "\n")
    print("wrote", args.out / "s9_cxr_gate.tex", args.out / "s9_cxr_audit.tex")


if __name__ == "__main__":
    main()
