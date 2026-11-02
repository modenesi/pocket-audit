"""Summarize the max_features sensitivity runs of 2026-10-07 (development seeds; sensitivity, not tuning).

Continuous ladder: results/mfsweep_20261007/cont_mf{sqrt,0.33,0.5,1.0} share base seed 2026100700, so
replicate r has the same data in every setting (paired). Binary: bin_mf1.0 pairs with night_binary_20261006
(sqrt; base seed 2026100620). Real covariates: semi_wids_enc_mf{1.0,0.33} pair with semi_wids_enc_x (sqrt;
base seed 2026100560, reps 0-19).

Pooled person-level F1 with 95% replicate-bootstrap intervals, paired differences against the default
setting and against ridge-weighted K-means in the same replicates, and null false-flag rates (continuous).

Usage: python simulations/mf_sweep_summary.py --out results/tables
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
TP, WS, FU, FN = "person_tp", "person_wrong_sign", "person_false_unaffected", "person_fn"
B = 2000


def read(pattern):
    fs = sorted(glob.glob(str(R / pattern)))
    return pd.concat((pd.read_csv(f) for f in fs), ignore_index=True) if fs else pd.DataFrame()


def f1(t):
    den = 2 * t[..., 0] + t[..., 1] + t[..., 2] + t[..., 3]
    return np.where(den > 0, 2 * t[..., 0] / np.where(den > 0, den, 1), np.nan)


def arr(d, reps, key):
    a = np.zeros((len(reps), 4))
    g = d.set_index("rep")
    for i, r in enumerate(reps):
        if r in g.index:
            row = g.loc[[r]].iloc[0]
            a[i] = [row[TP], row[WS], row[FU], row[FN]]
    return a


def paired(series: dict, reps, rng, base: str):
    """series: name -> (reps x 4) counts; returns rows with F1, CI, and diff vs base with CI."""
    draws = rng.integers(0, len(reps), (B, len(reps)))
    cnt = np.stack([np.bincount(x, minlength=len(reps)) for x in draws])
    boot = {k: f1(np.einsum("br,rk->bk", cnt, v)) for k, v in series.items()}
    out = []
    for k, v in series.items():
        est = float(f1(v.sum(0)))
        dd = boot[k] - boot[base]
        out.append(dict(series=k, reps=len(reps), f1=est, lo=np.nanpercentile(boot[k], 2.5),
                        hi=np.nanpercentile(boot[k], 97.5), diff_vs=base,
                        diff=est - float(f1(series[base].sum(0))), diff_lo=np.nanpercentile(dd, 2.5),
                        diff_hi=np.nanpercentile(dd, 97.5)))
    return out


def continuous(rng):
    rows, nulls = [], []
    settings = {"sqrt": "sqrt", "0.33": "1/3", "0.5": "1/2 (default)", "1.0": "all"}
    data = {}
    for mf in settings:
        d = read(f"mfsweep_20261007/cont_mf{mf}/learned/*/rep_*/metrics.csv")
        if d.empty:
            continue
        d = d[(d.alpha == .05) & (d.floor == 0)]
        if d.rep.nunique() < 50:          # an incomplete (stopped) setting would shrink the paired replicate set
            continue
        data[mf] = d
    if not data:
        return pd.DataFrame(), pd.DataFrame()
    specs = sorted(set.intersection(*[set(d.spec.astype(str)) for d in data.values()]))
    for sp in specs:
        alt = {mf: d[(d.spec.astype(str) == sp) & (d.delta == .4)] for mf, d in data.items()}
        reps = sorted(set.intersection(*[set(a[a.method == "rf_spectral"].rep) for a in alt.values()]))
        if not reps:
            continue
        series = {f"forest mf={mf}": arr(a[a.method == "rf_spectral"], reps, mf) for mf, a in alt.items()}
        first = next(iter(alt.values()))
        series["ridge"] = arr(first[first.method == "ridge_kmeans"], reps, "ridge")
        base = "forest mf=0.5" if "forest mf=0.5" in series else next(iter(series))
        for r in paired(series, reps, rng, base):
            rows.append(dict(design="continuous", cell=sp, **r))
        for mf, d in data.items():
            n = d[(d.spec.astype(str) == sp) & (d.delta == 0) & (d.method == "rf_spectral")]
            oob = d[(d.spec.astype(str) == sp) & (d.method == "rf_spectral")].oob_r2.mean()
            nulls.append(dict(cell=sp, mf=mf, reps=len(n), false_flags=int(n.any_flag.sum()),
                              rate=n.any_flag.mean() if len(n) else np.nan, oob_r2=oob))
    return pd.DataFrame(rows), pd.DataFrame(nulls)


def binary(rng):
    rows = []
    for cell in ["D1", "DN2", "DM1"]:
        new = read(f"mfsweep_20261007/bin_mf1.0/learned/{cell}/rep_*/metrics.csv")
        old = read(f"night_binary_20261006/learned/{cell}/rep_*/metrics.csv")
        if new.empty:
            continue
        new, old = new[new.tails == "symmetric"], old[old.tails == "symmetric"]
        reps = sorted(set(new[new.partition == "sbm"].rep) & set(old[old.partition == "sbm"].rep))
        series = {}
        for part in ["sbm", "leiden"]:
            series[f"{part} mf=sqrt"] = arr(old[old.partition == part], reps, part)
            series[f"{part} mf=all"] = arr(new[new.partition == part], reps, part)
        series["ridge"] = arr(old[old.partition == "ridge_kmeans"], reps, "ridge")
        for r in paired(series, reps, rng, "sbm mf=sqrt"):
            rows.append(dict(design="binary", cell=cell, **r))
    return pd.DataFrame(rows)


def semi(rng):
    rows = []
    old = read("app_20261005/semi_wids_enc_x/vent10/rep_*/metrics.csv")
    news = {mf: read(f"mfsweep_20261007/semi_wids_enc_mf{mf}/vent10/rep_*/metrics.csv") for mf in ["1.0", "0.33"]}
    news = {k: v for k, v in news.items() if not v.empty}
    if not news:
        return pd.DataFrame()
    reps = sorted(set(old[old.partition == "sbm"].rep).intersection(*[set(v[v.partition == "sbm"].rep) for v in news.values()]))
    series = {}
    for part in ["sbm", "leiden"]:
        series[f"{part} mf=sqrt"] = arr(old[old.partition == part], reps, part)
        for mf, v in news.items():
            series[f"{part} mf={mf}"] = arr(v[v.partition == part], reps, part)
    series["ridge"] = arr(old[old.partition == "ridge_kmeans"], reps, "ridge")
    for r in paired(series, reps, rng, "sbm mf=sqrt"):
        rows.append(dict(design="real covariates (WiDS, encounter roles)", cell="vent10", **r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261007)
    c, n = continuous(rng)
    res = pd.concat([c, binary(rng), semi(rng)], ignore_index=True)
    res.to_csv(args.out / "mf_sweep.csv", index=False)
    latex_table(res, args.out)
    n.to_csv(args.out / "mf_sweep_nulls.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string())
    print(n.round(3).to_string())



def latex_table(res: pd.DataFrame, out: Path):
    """Compact appendix table: F1 by max_features setting (paired replicates within each row)."""
    def f(x):
        return "--" if pd.isna(x) else f"{x:.2f}".lstrip("0")
    L = [r"\begin{tabular}{@{}llrcccc@{}}", r"\toprule",
         r"Design & Scenario & Reps & Default & 1/3 of $p$ & All $p$ & Ridge \\", r"\midrule"]
    lab = {"1": "Linear risk", "4": "+ nonlinear risk", "4c": "Region on strongest linear feature",
           "4z": "Nonlinear, standardized", "5": "+ missing values", "D1": "Region on the risk feature",
           "DN2": "Nonlinear one-tail region", "DM1": "Masking", "vent10": "WiDS ventilated (encounter roles)"}
    c = res[res.design == "continuous"]
    for cell, g in c.groupby("cell", sort=False):
        v = g.set_index("series").f1
        L.append(f"Continuous (forest) & {lab.get(cell, cell)} & {int(g.reps.iloc[0])} & {f(v.get('forest mf=0.5'))} "
                 f"& {f(v.get('forest mf=0.33'))} & {f(v.get('forest mf=1.0'))} & {f(v.get('ridge'))} \\\\")
    b = res[res.design == "binary"]
    for cell, g in b.groupby("cell", sort=False):
        v = g.set_index("series").f1
        L.append(f"Binary (SBM / Leiden) & {lab.get(cell, cell)} & {int(g.reps.iloc[0])} & "
                 f"{f(v.get('sbm mf=sqrt'))} / {f(v.get('leiden mf=sqrt'))} & -- & "
                 f"{f(v.get('sbm mf=all'))} / {f(v.get('leiden mf=all'))} & {f(v.get('ridge'))} \\\\")
    s = res[res.design.str.startswith("real")]
    for cell, g in s.groupby("cell", sort=False):
        v = g.set_index("series").f1
        L.append(f"Real covariates (SBM / Leiden) & {lab.get(cell, cell)} & {int(g.reps.iloc[0])} & "
                 f"{f(v.get('sbm mf=sqrt'))} / {f(v.get('leiden mf=sqrt'))} & "
                 f"{f(v.get('sbm mf=0.33'))} / {f(v.get('leiden mf=0.33'))} & "
                 f"{f(v.get('sbm mf=1.0'))} / {f(v.get('leiden mf=1.0'))} & {f(v.get('ridge'))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_mtry.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
