"""Supplement tables for the where-localization (mu) simulations, from mu_localization_sims.py summaries.

    python simulations/mu_sims_tables.py --run results/mu_sims_20261101 --out results/tables/s8_mu_sims_tables.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROWS = [("1", "Continuous, linear risk"), ("2", "\\quad + raw scales"), ("3", "\\quad + categorical"),
        ("4", "\\quad + nonlinear (U-shaped region)"), ("4c", "Nonlinear; region on top linear feature"),
        ("4z", "Nonlinear; standardized"), ("5", "\\quad + missing values"),
        ("1n", "Linear; region on an irrelevant feature"),
        ("1m", "Masking, linear ($n = 20{,}000$)"), ("4zm", "Masking, nonlinear ($n = 20{,}000$)"),
        ("D1", "Binary, region on risk feature"), ("D1n", "Binary, region on an irrelevant feature"),
        ("DH1", "Binary, within-unit dependence"), ("DN2", "Binary, nonlinear one-tail region$^\\ddagger$"),
        ("DM1", "Binary, masking$^\\ddagger$")]
COLS = [("forest", "Forest"), ("combined", "Comb."), ("ridge_kmeans", "Ridge"), ("leiden", "Leiden"),
        ("risk_bins", "Risk bins"), ("kmeans", "K-m."), ("mahalanobis_kmeans", "Mahal."), ("cem", "CEM"),
        ("cem_top", "CEM-3"), ("error_bins", "Error-t."), ("oracle", "Oracle")]
ALTS = ["risk_bins", "kmeans", "mahalanobis_kmeans", "cem", "cem_top", "error_bins"]


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}".lstrip("0").replace("-0.", "-.")


def f1_table(F: pd.DataFrame, cond: str) -> str:
    F = F[(F.condition == cond) & (F.family == "mu")]
    L = [r"\begin{tabular}{@{}l" + "r" * (len(COLS) + 1) + "@{}}", r"\toprule",
         "Scenario & Reps & " + " & ".join(h for _, h in COLS) + r" \\", r"\midrule"]
    for sid, lab in ROWS:
        g = F[F.scenario == sid].set_index("method")
        if g.empty:
            continue
        vals = {m: g.f1.get(m, np.nan) for m, _ in COLS}
        best = max(v for m, v in vals.items() if m != "oracle" and pd.notna(v))
        cells = []
        for m, _ in COLS:
            s = f2(vals[m])
            if m != "oracle" and pd.notna(vals[m]) and vals[m] == best:     # the largest unrounded score
                s = r"\textbf{" + s + "}"
            cells.append(s)
        L.append(f"{lab} & {int(g.reps.max())} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def error_table(E: pd.DataFrame, F: pd.DataFrame) -> str:
    """Exact-null and weak-null false-flag rates (any flag in the mu family), and coverage under the matched shift."""
    def pct(x):
        return "--" if pd.isna(x) else f"{100 * x:.1f}"
    L = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Exact null: false flags, \%} & \multicolumn{2}{c}{Weak null, \%} & "
         r"\multicolumn{2}{c}{Coverage, \%} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Scenario & Reps & Forest & Comb. & Others (max) & Forest & Ridge & Forest & Comb. \\", r"\midrule"]
    for sid, lab in ROWS:
        e = E[(E.scenario == sid) & (E.family == "mu")]
        if e.empty and F[F.scenario == sid].empty:
            continue
        ex = e[e.condition == "exact_null"].set_index("method")
        wk = e[e.condition == "weak"].set_index("method")
        others = ex.drop(index=[m for m in ("forest", "combined", "oracle") if m in ex.index]).false_flag
        fm = F[(F.scenario == sid) & (F.condition == "matched") & (F.family == "mu")].set_index("method")
        reps = int(ex.reps.max()) if len(ex) else (int(fm.reps.max()) if len(fm) else 0)
        L.append(f"{lab} & {reps} & {pct(ex.false_flag.get('forest'))} & {pct(ex.false_flag.get('combined'))} & "
                 f"{pct(others.max() if len(others) else np.nan)} & {pct(wk.false_flag.get('forest'))} & "
                 f"{pct(wk.false_flag.get('ridge_kmeans'))} & {pct(fm.covered_fp.get('forest'))} & "
                 f"{pct(fm.covered_fp.get('combined'))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


MAIN_ROWS = [("cont", "Continuous outcome, $n = 10{,}000$"), ("1", "Linear risk"),
             ("4", "\\quad + scales, categories, U-shaped risk"),
             ("4c", "Nonlinear; region on top linear feature"), ("4z", "Nonlinear; standardized"),
             ("5", "\\quad + missing values"), ("1n", "Linear; region on an irrelevant feature"),
             ("mask", "Masking: overall error zero, $n = 20{,}000$"), ("1m", "Linear risk"), ("4zm", "Nonlinear risk"),
             ("bin", "Binary, 8\\% events, $n = 50{,}000$"), ("D1", "Region on risk feature"),
             ("D1n", "Region on an irrelevant feature"), ("DH1", "Within-unit dependence"),
             ("DN2", "Nonlinear one-tail region"), ("DM1", "Masking")]
OTHER = ["risk_bins", "kmeans", "mahalanobis_kmeans", "cem", "cem_top"]
SHORT = {"risk_bins": "risk bins", "kmeans": "K-means", "mahalanobis_kmeans": "Mahal.", "cem": "CEM", "cem_top": "CEM-3"}


def main_table(F: pd.DataFrame, E: pd.DataFrame, D: pd.DataFrame) -> str:
    """Main-text table: F1 at both shifts for the combined audit, the error-targeted comparator and the best other
    alternative, with the combined audit's exact-null false-flag rate; bold = row winner among the three."""
    F = F[F.family == "mu"]
    E = E[(E.family == "mu") & (E.condition == "exact_null")]
    L = [r"\begin{tabular}{@{}lrrlrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Smaller shift} & \multicolumn{3}{c}{Larger shift ($\delta$ designs)} & Null \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Scenario & Comb. & Error-t. & Best other & Comb. & Error-t. & Best other & flags, \% \\", r"\midrule"]
    for sid, lab in MAIN_ROWS:
        if sid in ("cont", "mask", "bin"):
            L.append(r"\multicolumn{8}{@{}l}{\emph{" + lab + r"}} \\")
            continue
        cells = []
        for cond in ("matched", "same"):
            g = F[(F.scenario == sid) & (F.condition == cond)].set_index("method").f1
            oth = {m: g[m] for m in OTHER if m in g}
            bo = max(oth, key=oth.get)
            vals = [g.get("combined"), g.get("error_bins"), oth[bo]]
            top = max(vals)
            txt = [(r"\textbf{" + f2(v) + "}") if v == top else f2(v) for v in vals]
            txt[2] = f"{txt[2]} ({SHORT[bo]})" if cond == "matched" else txt[2]
            cells += txt
        e = E[(E.scenario == ("1" if sid == "1n" else "D1" if sid == "D1n" else sid)) & (E.method == "combined")]
        ff = f"{100 * e.false_flag.iloc[0]:.1f}" if len(e) and sid not in ("1n", "D1n") else "--"
        L.append(f"{lab} & " + " & ".join(cells) + f" & {ff} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    F = pd.read_csv(args.run / "f1.csv", keep_default_na=False, na_values=[""], dtype={"scenario": str})
    E = pd.read_csv(args.run / "false_flags.csv", keep_default_na=False, na_values=[""], dtype={"scenario": str})
    D = pd.read_csv(args.run / "paired_diffs.csv", keep_default_na=False, na_values=[""], dtype={"scenario": str})
    parts = {"matched": f1_table(F, "matched"), "same": f1_table(F, "same"), "errors": error_table(E, F),
             "main": main_table(F, E, D)}
    for k, v in parts.items():
        (args.out.parent / f"{args.out.stem}_{k}.tex").write_text(v + "\n")
    # counts for the text (matched shift, mu family; pointwise paired intervals). Predeclared comparator: the
    # error-targeted bins. Also: better than EVERY alternative (all intervals above 0) / worse than at least one.
    d = D[(D.condition == "matched") & (D.family == "mu")]
    rows = []
    for (ours, sid), g in d.groupby(["ours", "scenario"], sort=False):
        e = g[g.versus == "error_bins"]
        rows.append(dict(ours=ours, scenario=sid, n_alts=len(g), beats_all=bool((g.lo > 0).all()),
                         loses_any=bool((g.hi < 0).any()), losers=";".join(g[g.hi < 0].versus),
                         vs_error=e["diff"].squeeze() if len(e) else np.nan, vs_error_lo=e.lo.squeeze() if len(e) else np.nan,
                         vs_error_hi=e.hi.squeeze() if len(e) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(args.run / "text_counts.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string())
    for ours in ("combined", "forest"):
        s_ = S[S.ours == ours]
        print(f"{ours}: better than every alternative in {int(s_.beats_all.sum())}, worse than at least one in "
              f"{int(s_.loses_any.sum())}, of {len(s_)}")
        win, loss = int((s_.vs_error_lo > 0).sum()), int((s_.vs_error_hi < 0).sum())
        print(f"{ours} vs error-targeted: better {win}, worse {loss}, not distinguishable "
              f"{int(s_.vs_error.notna().sum()) - win - loss}")


if __name__ == "__main__":
    main()
