"""LaTeX table for the combined forest + ridge audit (paper v2, 2026-10-07).

Inputs (results/tables): union_audit_fresh.csv and union_audit_nulls_fresh.csv (continuous ladder,
fresh seeds, rule fixed beforehand), union_audit_binary.csv (saved binary replicates), union_audit_semi.csv
(saved real-covariate replicates). Writes tab_union.tex there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables"

CONT = [("1", "Linear risk"), ("2", "+ raw feature scales"), ("3", "+ categorical features"),
        ("4", "+ nonlinear risk"), ("4c", "Nonlinear, region on strongest linear feature"),
        ("4z", "Nonlinear, standardized features"), ("5", "+ missing values"),
        ("1m", "Masking, linear risk"), ("4zm", "Masking, nonlinear risk")]
BIN = [("confirm_20261004", "D1", "Region on the risk feature"), ("confirm_20261004", "DH1", "Same, 150 hospitals"),
       ("night_binary_20261006", "DX1", "Focal group concentrated at high risk"),
       ("confirm_20261004", "DN2", "Nonlinear one-tail region"), ("night_binary_20261006", "DM1", "Masking")]
SEMI = [("semi_uci", "inpt10", "UCI diabetes"), ("semi_wids_enc", "vent10", "WiDS, encounter-level roles"),
        ("semi_wids", "vent10", "WiDS, hospital-disjoint roles")]


def f(x):
    if not pd.isna(x) and abs(x) < .005:
        x = 0.0
    return "--" if pd.isna(x) else (f"{x:.2f}".replace("0.", ".", 1) if x >= 0 else f"{x:.2f}".replace("-0.", "$-$.", 1))


def row(label, forest, ridge, comb, d, lo, hi, reps):
    best = max(forest, ridge)
    cells = []
    for v in (forest, ridge, comb):
        s = f(v)
        cells.append(r"\textbf{" + s + "}" if abs(v - max(forest, ridge, comb)) < 5e-3 else s)
    return f"\\quad {label} & {reps} & " + " & ".join(cells) + f" & {f(d)} [{f(lo)}, {f(hi)}] \\\\"


def main():
    L = [r"\begin{tabular}{@{}lrcccl@{}}", r"\toprule",
         r"Scenario & Reps & Forest & Ridge & Combined & Combined $-$ better map [95\% CI] \\", r"\midrule"]
    p = OUT / "union_audit_fresh.csv"
    if p.exists():
        c = pd.read_csv(p)
        L.append(r"\multicolumn{6}{@{}l}{\emph{Continuous outcome (fresh seeds; $n = 10{,}000$, masking $n = 20{,}000$)}} \\")
        for sp, lab in CONT:
            g = c[c.spec.astype(str) == sp].set_index("method")
            if g.empty:
                continue
            u = g.loc["forest + ridge (.025 each)"]
            L.append(row(lab, g.loc["forest (.05)", "f1"], g.loc["ridge (.05)", "f1"], u.f1, u.minus_best_single,
                         u.mbs_lo, u.mbs_hi, int(u.reps)))
    p = OUT / "union_audit_binary.csv"
    if p.exists():
        b = pd.read_csv(p)
        L.append(r"\addlinespace\multicolumn{6}{@{}l}{\emph{Binary outcome, $n = 50{,}000$ (saved replicates; forest = SBM)}} \\")
        for batch, cell, lab in BIN:
            g = b[(b.batch == batch) & (b.cell == cell)].set_index("method")
            if g.empty:
                continue
            u = g.loc["sbm+ridge_half"]
            L.append(row(lab, g.loc["sbm_full", "f1"], g.loc["ridge_kmeans_full", "f1"], u.f1, u.minus_best_sbm_ridge,
                         u.mb_lo, u.mb_hi, int(u.reps)))
    p = OUT / "union_audit_semi.csv"
    if p.exists():
        s = pd.read_csv(p)
        L.append(r"\addlinespace\multicolumn{6}{@{}l}{\emph{Real covariates, simulated outcomes (saved replicates; forest = SBM)}} \\")
        for batch, cell, lab in SEMI:
            g = s[(s.batch == batch) & (s.cell == cell)].set_index("method")
            if g.empty:
                continue
            u = g.loc["sbm+ridge_half"]
            L.append(row(lab, g.loc["sbm_full", "f1"], g.loc["ridge_kmeans_full", "f1"], u.f1, u.minus_best_sbm_ridge,
                         u.mb_lo, u.mb_hi, int(u.reps)))
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_union.tex").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
