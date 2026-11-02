"""Paper v2 (2026-10-07): scoreboard, paired differences and error-control tables from saved runs.

Reads per-replicate ``metrics.csv`` files of the completed simulation batches (local ``results/``),
keeps the locked inference settings of each batch, and writes tidy CSVs, LaTeX tables and figures:

- pooled person-level F1 = 2 sum TP / (2 sum TP + sum FP + sum FN) per scenario x method, with
  95% intervals from a replicate bootstrap (replicate indices resampled jointly across the methods
  of one batch, so method differences are paired);
- ours minus the strongest deployable competitor, paired within one batch;
- one batch per scenario in every display (rule fixed on 2026-10-10): when a scenario ran in
  several batches, report the batch in which the strongest non-forest competitor scored highest (the least
  favourable comparison for the forest); methods that batch did not run are shown from another batch with a
  dagger and never enter the paired differences;
- family-wise false-flag rates under all-null designs with Wilson intervals, and simultaneous
  coverage under alternatives.

Usage: python simulations/paper_v2_tables.py --out results/tables --fig-dir results/figures
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
B_DRAWS = 5000
TP, WS, FU, FN = "person_tp", "person_wrong_sign", "person_false_unaffected", "person_fn"

# display columns; "ours" is forest-spectral (continuous) or forest-graph SBM (binary, real covariates)
METHOD_KEYS = {
    "ours": ["rf_spectral", "sbm"],
    "leiden": ["leiden"],
    "ridge": ["ridge_kmeans"],
    "riskbins": ["score_bins", "risk_bins"],
    "kmeans": ["kmeans"],
    "mahal": ["mahalanobis_kmeans"],
    "cemtop": ["cem_top"],
    "disp": ["disparity_bins"],
    "oracle": ["oracle"],
    "combined": ["__combined__"],      # forest + ridge families at half the level each (union_audit*.py)
}
LABELS = {"ours": "Forest pockets (ours)", "leiden": "Forest graph + Leiden", "ridge": "Ridge-weighted K-means",
          "riskbins": "Risk bins", "kmeans": "K-means", "mahal": "Mahalanobis K-means", "cemtop": "CEM (top 3)",
          "disp": "Disparity-targeted", "oracle": "Oracle (truth-aware)", "combined": "Forest + ridge (combined)"}
COMPETITORS = ["ridge", "riskbins", "kmeans", "mahal", "cemtop", "disp"]     # deployable, not forest-based
SINGLES = ["ours", "leiden"] + COMPETITORS                                     # every single deployable grouping
# reporting rule (2026-10-11): headline comparisons are against the alternative groupings, i.e. every
# single grouping that is not one of ours. COMPETITORS still drives the batch rule and the table bolding.
OWN = ["ours", "leiden", "ridge"]                                              # our groupings (forest, Leiden, ridge)
ALTERNATIVES = ["riskbins", "kmeans", "mahal", "cemtop", "disp"]               # alternative groupings
REP_OUT = ROOT / "results/tables"


def combined_rows(kind: str, batch: str, cell: str) -> pd.DataFrame:
    """Per-replicate person counts of the combined forest + ridge audit, from the union_audit* outputs."""
    try:
        if kind == "cont":
            u = pd.read_csv(REP_OUT / "union_audit_reps.csv")
            u = u[(u.batch == batch) & (u.spec.astype(str) == cell) & (u.method == "forest + ridge (.025 each)")]
            return u[["rep", TP, WS, FU, FN]].copy()
        src = REP_OUT / ("union_audit_binary_reps.pkl" if kind == "bin" else "union_audit_semi_reps.pkl")
        u = pd.read_pickle(src)
        u = u[(u.batch == batch) & (u.cell.astype(str) == cell)]
        if u.empty:
            return pd.DataFrame()
        a = np.stack(u["sbm+ridge_half"].to_numpy())
        return pd.DataFrame({"rep": u.rep.to_numpy(), TP: a[:, 0], WS: a[:, 1], FU: a[:, 2], FN: a[:, 3]})
    except FileNotFoundError:
        return pd.DataFrame()

CONT = {"conf": "contladder_20261005/conf_v1_P_*", "comp": "contladder_20261005/night_comp",
        "mask": "contladder_20261005/night_mask", "fresh10k": "union_conf_20261007/n10k",
        "fresh20k": "union_conf_20261007/n20k_mask", "null20k": "nullcheck_20261007/n20k"}
N20K = ("mask", "fresh20k", "null20k")
SCENARIOS = [  # (id, group, label, batches); first batch is the main one
    ("1", "Continuous outcome", "Linear risk", [("cont", "conf", "1"), ("cont", "comp", "1")]),
    ("2", "Continuous outcome", "+ raw feature scales", [("cont", "conf", "2")]),
    ("3", "Continuous outcome", "+ categorical features", [("cont", "conf", "3")]),
    ("4", "Continuous outcome", "+ nonlinear risk (region on U-shaped feature)", [("cont", "conf", "4")]),
    ("4c", "Continuous outcome", "Nonlinear risk, region on strongest linear feature", [("cont", "conf", "4c")]),
    ("4z", "Continuous outcome", "Nonlinear risk, standardized features", [("cont", "conf", "4z"), ("cont", "comp", "4z")]),
    ("5", "Continuous outcome", "+ missing values", [("cont", "conf", "5")]),
    ("1m", "Masking (global gap zero)", "Linear risk", [("cont", "mask", "1m")]),
    ("4zm", "Masking (global gap zero)", "Nonlinear risk", [("cont", "mask", "4zm")]),
    ("D1", "Binary outcome, n = 50,000", "Region on the risk feature", [("bin", "confirm_20261004", "D1"), ("bin", "night_binary_20261006", "D1")]),
    ("DH1", "Binary outcome, n = 50,000", "Same, 150 hospitals", [("bin", "confirm_20261004", "DH1")]),
    ("DX1", "Binary outcome, n = 50,000", "Focal group concentrated at high risk", [("bin", "night_binary_20261006", "DX1")]),
    ("DN2", "Binary outcome, n = 50,000", "Nonlinear one-tail region", [("bin", "confirm_20261004", "DN2"), ("bin", "night_binary_20261006", "DN2")]),
    ("DM1", "Binary outcome, n = 50,000", "Masking", [("bin", "night_binary_20261006", "DM1")]),
    ("WH", "Real covariates, simulated outcomes", "WiDS, hospital-disjoint roles", [("semi", "semi_wids", "vent10")]),
    ("WE", "Real covariates, simulated outcomes", "WiDS, encounter-level roles", [("semi", "semi_wids_enc", "vent10"), ("semi", "semi_wids_enc_x", "vent10")]),
    ("UCI", "Real covariates, simulated outcomes", "UCI diabetes", [("semi", "semi_uci", "inpt10"), ("semi", "semi_uci_x", "inpt10")]),
]


def _read(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(R / pattern)))
    if not fs:
        raise FileNotFoundError(pattern)
    return pd.concat((pd.read_csv(f) for f in fs), ignore_index=True)


_CACHE: dict = {}


def batch_rows(kind: str, batch: str, cell: str, alt: bool = True) -> pd.DataFrame:
    """Rows of one batch/cell under its locked settings, with a 'method' column (display key) and 'rep'."""
    key = (kind, batch, cell, alt)
    if key in _CACHE:
        return _CACHE[key]
    if kind == "cont":
        n = "20000" if batch in N20K else "10000"
        d = _read(f"{CONT[batch]}/learned/spec{cell}_armP_n{n}/rep_*/metrics.csv")
        d = d[(d.alpha == .05) & (d.floor == 0) & (d.arm == "P")]
        d = d[d.delta == (.4 if alt else 0.0)]
        d = d.rename(columns={"method": "raw"})
    elif kind == "bin":
        d = _read(f"{batch}/learned/{cell}/rep_*/metrics.csv")
        d = d[d.tails == "symmetric"].rename(columns={"partition": "raw"})
    else:
        d = _read(f"app_20261005/{batch}/{cell}/rep_*/metrics.csv").rename(columns={"partition": "raw"})
    inv = {raw: k for k, raws in METHOD_KEYS.items() for raw in raws}
    d = d[d.raw.isin(inv)].copy()
    d["method"] = d.raw.map(inv)
    d["batch"] = batch
    if alt:
        c = combined_rows(kind, batch, cell)
        if not c.empty:
            c = c[c.rep.isin(d.rep.unique())]
            c["method"], c["raw"], c["batch"] = "combined", "__combined__", batch
            d = pd.concat([d, c], ignore_index=True)
    _CACHE[key] = d
    return d


def pooled_f1(tp, fs, fu, fn):
    den = 2 * tp + fs + fu + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)


def boot_batch(d: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    """Pooled F1 per method and a joint replicate bootstrap (B_DRAWS x methods)."""
    reps = np.sort(d.rep.unique())
    methods = sorted(d.method.unique())
    arr = np.zeros((len(methods), len(reps), 4))
    idx = {r: i for i, r in enumerate(reps)}
    for mi, m in enumerate(methods):
        g = d[d.method == m]
        for _, row in g.iterrows():
            arr[mi, idx[row.rep]] = [row[TP], row[WS], row[FU], row[FN]]
    tot = arr.sum(1)
    est = pooled_f1(*tot.T)
    draws = rng.integers(0, len(reps), (B_DRAWS, len(reps)))
    counts = np.stack([np.bincount(dr, minlength=len(reps)) for dr in draws])        # B x reps
    bt = np.einsum("br,mrk->bmk", counts, arr)                                       # B x methods x 4
    bf = pooled_f1(bt[..., 0], bt[..., 1], bt[..., 2], bt[..., 3])                   # B x methods
    rows = []
    for mi, m in enumerate(methods):
        aff = arr[mi, :, 0].sum() + arr[mi, :, 3].sum()
        g = d[d.method == m]
        untested = (g.aff_untested.sum() / g.aff_n.sum()) if "aff_untested" in g and g.aff_n.sum() else (
            g.aff_unknown_count.sum() / g.person_n_affected.sum() if "aff_unknown_count" in g and g.person_n_affected.sum() else np.nan)
        rows.append(dict(method=m, reps=len(reps), f1=est[mi], lo=np.nanpercentile(bf[:, mi], 2.5),
                         hi=np.nanpercentile(bf[:, mi], 97.5),
                         precision=tot[mi, 0] / (tot[mi, 0] + tot[mi, 1] + tot[mi, 2]) if tot[mi, :3].sum() else np.nan,
                         recall=tot[mi, 0] / aff if aff else np.nan, untested=untested))
    return pd.DataFrame(rows), {"methods": methods, "boot": bf, "est": est}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


MAIN_BATCH: dict = {}      # scenario id -> (kind, batch, cell) reported in every display


def paired(bt: dict, a: str, pool: list, tag: str) -> dict:
    """Method a minus the best method of pool in one batch, with a paired replicate-bootstrap 95% interval."""
    ms = bt["methods"]
    ci = [ms.index(m) for m in pool if m in ms]
    if a not in ms or not ci:
        return {}
    j = ci[int(np.nanargmax(bt["est"][ci]))]
    i = ms.index(a)
    dd = bt["boot"][:, i] - bt["boot"][:, j]
    return {f"{tag}_vs": ms[j], f"{tag}_f1": bt["est"][j], f"{tag}_diff": bt["est"][i] - bt["est"][j],
            f"{tag}_lo": np.nanpercentile(dd, 2.5), f"{tag}_hi": np.nanpercentile(dd, 97.5)}


def scoreboard(rng):
    cells, diffs = [], []
    for sid, group, label, batches in SCENARIOS:
        best = None
        tabs = []
        for bi, (kind, batch, cell) in enumerate(batches):
            d = batch_rows(kind, batch, cell)
            tab, bt = boot_batch(d, rng)
            tab["scenario"], tab["group"], tab["label"], tab["batch"] = sid, group, label, batch
            tabs.append(tab)
            ms = bt["methods"]
            if "ours" not in ms:
                continue
            comp = [m for m in COMPETITORS if m in ms]
            if not comp:
                continue
            ci = [ms.index(m) for m in comp]
            j = ci[int(np.nanargmax(bt["est"][ci]))]
            o = ms.index("ours")
            dd = bt["boot"][:, o] - bt["boot"][:, j]
            cand = dict(scenario=sid, group=group, label=label, batch=batch, competitor=ms[j],
                        f1_ours=bt["est"][o], f1_comp=bt["est"][j], diff=bt["est"][o] - bt["est"][j],
                        lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5), reps=int(tab.reps.iloc[0]))
            cand.update(paired(bt, "ours", ALTERNATIVES, "alt"))
            if "combined" in ms:
                si = [ms.index(m) for m in SINGLES if m in ms]
                k = si[int(np.nanargmax(bt["est"][si]))]
                c = ms.index("combined")
                dc = bt["boot"][:, c] - bt["boot"][:, k]
                cand.update(comb_vs=ms[k], f1_comb=bt["est"][c], f1_best_single=bt["est"][k],
                            comb_diff=bt["est"][c] - bt["est"][k], comb_lo=np.nanpercentile(dc, 2.5),
                            comb_hi=np.nanpercentile(dc, 97.5))
                cand.update(paired(bt, "combined", ALTERNATIVES, "comb_alt"))
                cand.update(paired(bt, "combined", OWN, "comb_own"))
                cand.update(paired(bt, "combined", ["ours", "ridge"], "comb_parts"))   # its two component maps
            if best is None or cand["f1_comp"] > best["f1_comp"]:
                best = cand
        main = best["batch"] if best is not None else batches[0][1]
        MAIN_BATCH[sid] = next(b for b in batches if b[1] == main)
        for tab in tabs:
            tab["main"] = tab["batch"] == main
            cells.append(tab)
        if best is not None:
            diffs.append(best)
    return pd.concat(cells, ignore_index=True), pd.DataFrame(diffs)


def error_control():
    rows = []

    def counts(kind, batch, cell, flagcol, testcol):
        try:
            d = batch_rows(kind, batch, cell, alt=False)
        except FileNotFoundError:
            return {}
        out = {}
        for m, g in d.groupby("method"):
            g = g.drop_duplicates("rep")
            out[m] = (int(g[flagcol].fillna(False).astype(bool).sum()), len(g),
                      int((g[testcol] > 0).sum()) if testcol else len(g))
        return out

    def add(design, outcome, alpha, parts):
        tot = {}
        for c in parts:
            for m, (k, n, nt) in c.items():
                a = tot.setdefault(m, [0, 0, 0])
                a[0] += k; a[1] += n; a[2] += nt
        for m, (k, n, nt) in tot.items():
            lo, hi = wilson(k, n)
            rows.append(dict(design=design, outcome=outcome, alpha=alpha, method=m, reps=n, tested_reps=nt,
                             false_flags=k, rate=k / n, lo=lo, hi=hi))

    for sp in ["1", "2", "3", "4", "4c", "4z", "5"]:      # every null batch of the design ("comp": specs 1, 4z)
        add(f"Continuous, spec {sp}", "continuous", .05,
            [counts("cont", b, sp, "any_flag", "n_local_tested") for b in ("conf", "fresh10k", "comp")])
    add("Continuous, $n = 20{,}000$, linear risk", "continuous", .05,
        [counts("cont", "mask", "1m", "any_flag", "n_local_tested"),
         counts("cont", "fresh20k", "1m", "any_flag", "n_local_tested"),
         counts("cont", "null20k", "1", "any_flag", "n_local_tested")])
    add("Continuous, $n = 20{,}000$, nonlinear risk", "continuous", .05,
        [counts("cont", "mask", "4zm", "any_flag", "n_local_tested"),
         counts("cont", "fresh20k", "4zm", "any_flag", "n_local_tested"),
         counts("cont", "null20k", "4z", "any_flag", "n_local_tested")])
    add("Binary, $n = 50{,}000$, independent observations", "binary", .025,
        [counts("bin", "confirm_20261004", "D0", "any_null_flag", "n_tested")])
    add("Binary, $n = 50{,}000$, 150 dependent units", "binary", .0125,
        [counts("bin", "confirm_20261004", "DH0", "any_null_flag", "n_tested")])
    add("Binary, focal group at high risk (small check)", "binary", .025,
        [counts("bin", "night_binary_20261006", "DX0", "any_null_flag", "n_tested")])
    # combined audit (forest + ridge, each family at half the level): any flag in either family
    U2, U3 = ROOT / "results/tables", ROOT / "results/tables"
    un = pd.concat([pd.read_csv(U2 / "union_audit_nulls.csv"), pd.read_csv(U2 / "union_audit_nulls_fresh.csv"),
                    pd.read_csv(U3 / "union_audit_nulls_null20k.csv")])
    un = un[np.isclose(un.alpha_each, .025)]
    def comb(design, outcome, alpha, pairs):
        k = n = 0
        for b, sp in pairs:
            g = un[(un.batch == b) & (un.spec.astype(str) == sp)]
            if not g.empty:
                k += int(round(g.union.iloc[0] * g.reps.iloc[0])); n += int(g.reps.iloc[0])
        lo, hi = wilson(k, n)
        rows.append(dict(design=design, outcome=outcome, alpha=alpha, method="combined", reps=n, tested_reps=n,
                         false_flags=k, rate=k / n, lo=lo, hi=hi))
    for sp in ["1", "2", "3", "4", "4c", "4z", "5"]:
        comb(f"Continuous, spec {sp}", "continuous", .05, [("conf", sp), ("fresh10k", sp), ("comp", sp)])
    comb("Continuous, $n = 20{,}000$, linear risk", "continuous", .05, [("mask", "1m"), ("fresh20k", "1m"), ("null20k", "1")])
    comb("Continuous, $n = 20{,}000$, nonlinear risk", "continuous", .05,
         [("mask", "4zm"), ("fresh20k", "4zm"), ("null20k", "4z")])
    ub = pd.concat([pd.read_csv(U2 / "union_audit_binary_nulls450.csv"), pd.read_csv(U2 / "union_audit_binary.csv")])
    ub = ub[ub.method == "sbm+ridge_half"]
    for design, alpha, b, cell in [("Binary, $n = 50{,}000$, independent observations", .025, "confirm_20261004", "D0"),
                                   ("Binary, $n = 50{,}000$, 150 dependent units", .0125, "confirm_20261004", "DH0"),
                                   ("Binary, focal group at high risk (small check)", .025,
                                    "night_binary_20261006", "DX0")]:
        g = ub[(ub.batch == b) & (ub.cell == cell)].sort_values("reps").iloc[-1]       # the largest null batch
        k, n = int(round(g.any_flag_rate * g.reps)), int(g.reps)
        lo, hi = wilson(k, n)
        rows.append(dict(design=design, outcome="binary", alpha=alpha, method="combined", reps=n, tested_reps=n,
                         false_flags=k, rate=k / n, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def coverage(sc_cells):
    """Simultaneous coverage of all tested coordinates under alternatives (ours)."""
    rows = []
    for sid, group, label, batches in SCENARIOS:
        kind, batch, cell = MAIN_BATCH.get(sid, batches[0])
        d = batch_rows(kind, batch, cell)
        g = d[d.method == "ours"].drop_duplicates("rep")
        col = "covered_all" if "covered_all" in g else "all_covered"
        tested = g[(g.get("n_local_tested", g.get("n_tested", 1)) > 0)]
        k = int(tested[col].fillna(True).astype(bool).sum())
        n = len(tested)
        lo, hi = wilson(k, n)
        rows.append(dict(scenario=sid, label=label, reps=len(g), tested_reps=n, covered=k, rate=k / n if n else np.nan,
                         lo=lo, hi=hi))
    return pd.DataFrame(rows)


def fmt(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}".lstrip("0").replace("-0.", "-.") if abs(x) < 1 else f"{x:.{nd}f}"


def write_tables(cells, diffs, ec, cov, out: Path):
    cols = ["ours", "combined", "leiden", "ridge", "riskbins", "kmeans", "mahal", "cemtop", "disp", "oracle"]
    head = ["Forest", "Combined", "Leiden", "Ridge", "Risk bins", "K-means", "Mahal.", "CEM-3", "Disparity", "Oracle"]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
             "Scenario & " + " & ".join(head) + r" \\", r"\midrule"]
    grp = None
    for sid, group, label, batches in SCENARIOS:
        if group != grp:
            if grp is not None:
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{%d}{@{}l}{\emph{%s}} \\" % (len(cols) + 1, group))
            grp = group
        c = cells[cells.scenario == sid]
        main = c[c.main]
        top = max(main[main.method.isin(["ours", "combined", "leiden"] + COMPETITORS)].f1.max(),
                  c[(~c.main) & c.method.isin(COMPETITORS) & ~c.method.isin(main.method)].f1.max() if (~c.main).any() else -1)
        vals = []
        for m in cols:
            r = main[main.method == m]
            dag = ""
            if r.empty:
                r = c[(~c.main) & (c.method == m)]
                dag = r"$^\dagger$" if not r.empty else ""
            if r.empty:
                vals.append("--")
                continue
            v = fmt(r.f1.iloc[0])
            if m != "oracle" and abs(r.f1.iloc[0] - top) < 5e-3:
                v = r"\textbf{" + v + "}"
            if m == "oracle":
                v = r"\textcolor{gray}{" + v + "}"
            vals.append(v + dag)
        lines.append(f"\\quad {label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_scoreboard.tex").write_text("\n".join(lines) + "\n")

    write_error_table(ec, out)


# saved error-control CSVs written before the generic renaming (2026-11-01) use the old design names
OLD_DESIGNS = {"Binary, $n = 50{,}000$, independent patients": "Binary, $n = 50{,}000$, independent observations",
               "Binary, $n = 50{,}000$, 150 hospitals": "Binary, $n = 50{,}000$, 150 dependent units"}
# mu localization nulls (results/mu_sims_*/false_flags.csv): design label, scenarios, single-map alpha
MU_DESIGNS = [("Continuous, $n = 10{,}000$, 7 specifications", ["1", "2", "3", "4", "4c", "4z", "5"], .05),
              ("Continuous, $n = 20{,}000$, linear risk", ["1m"], .05),
              ("Continuous, $n = 20{,}000$, nonlinear risk", ["4zm"], .05),
              ("Binary, $n = 50{,}000$, independent observations", ["D1"], .025),
              ("Binary, $n = 50{,}000$, 150 dependent units", ["DH1"], .0125)]


def write_error_table(ec, out: Path, mu_ff: pd.DataFrame | None = None):
    """Error control: forest alone at the full level and the combined audit (each map at half the level); with
    ``mu_ff``, a second block for the mu-only family of the localization simulations (exact nulls)."""
    def cell(g):
        if len(g) == 1:
            r = g.iloc[0]
            return f"{fmt(r.rate, 3)} [{fmt(r.lo, 3)}, {fmt(r.hi, 3)}]"
        return f"{fmt(g.rate.min(), 3)}--{fmt(g.rate.max(), 3)} [{fmt(g.lo.min(), 3)}, {fmt(g.hi.max(), 3)}]"

    def reps(g):
        return f"{g.reps.min()}--{g.reps.max()}" if g.reps.nunique() > 1 else f"{g.reps.iloc[0]}"
    ec = ec.assign(design=ec.design.replace(OLD_DESIGNS))
    L = [r"\begin{tabular}{@{}llrll@{}}", r"\toprule",
         r"Null design & Single-map $\alpha$ & Replicates & Forest & Combined \\", r"\midrule"]
    if mu_ff is not None:
        L.append(r"\multicolumn{5}{@{}l}{\emph{Between-group gap $\delta$}} \\")
    e, c = ec[ec.method == "ours"], ec[ec.method == "combined"]
    cs = e[e.design.str.startswith("Continuous, spec")]
    L.append(f"Continuous, $n = 10{{,}}000$, 7 specifications & .05 & {reps(cs)} & {cell(cs)} & "
             f"{cell(c[c.design.str.startswith('Continuous, spec')])} \\\\")
    for _, r in e[~e.design.str.startswith("Continuous, spec")].iterrows():
        L.append(f"{r.design} & {f"{r.alpha:g}".lstrip("0")} & {r.reps} & {cell(e[e.design == r.design])} & "
                 f"{cell(c[c.design == r.design]) if (c.design == r.design).any() else '--'} \\\\")
    if mu_ff is not None:
        m = mu_ff[(mu_ff.family == "mu") & (mu_ff.condition == "exact_null")].rename(columns={"false_flag": "rate"})
        L += [r"\addlinespace", r"\multicolumn{5}{@{}l}{\emph{Local error $\mu$}} \\"]
        for design, scen, alpha in MU_DESIGNS:
            f, cb = (m[(m.method == k) & m.scenario.astype(str).isin(scen)] for k in ("forest", "combined"))
            assert len(f) == len(cb) == len(scen), design
            L.append(f"{design} & {f'{alpha:g}'.lstrip('0')} & {reps(f)} & {cell(f)} & {cell(cb)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_error_control.tex").write_text("\n".join(L) + "\n")


def figure(cells, diffs, out: Path, fig_dir: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3 (ACM requirement)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cols = ["ours", "combined", "leiden", "ridge", "riskbins", "kmeans", "mahal", "cemtop", "disp", "oracle"]
    head = ["Forest", "Combined (recommended)", "Leiden", "Ridge", "Risk bins",
            "K-means", "Mahal. K-means", "CEM top-3", "Disparity-targeted", "Oracle"]
    GROUP_TITLES = {"Continuous outcome": "Continuous, n = 10,000",
                    "Masking (global gap zero)": "Masking: global gap zero (n = 20,000)",
                    "Binary outcome, n = 50,000": "Binary, 8% events, n = 50,000",
                    "Real covariates, simulated outcomes": "Real covariates, simulated outcomes"}
    SHORT = {"1": "Linear risk", "2": "+ raw scales", "3": "+ categorical", "4": "+ nonlinear (U-shaped)",
             "4c": "Nonlinear; region on top linear feat.", "4z": "Nonlinear; standardized",
             "5": "+ missing values", "1m": "Linear risk", "4zm": "Nonlinear risk", "D1": "Region on risk feature",
             "DH1": "Same, within-unit dependence", "DX1": "Focal group at high risk", "DN2": "Nonlinear one-tail region",
             "DM1": "Masking", "WH": "WiDS, hospital-disjoint", "WE": "WiDS, encounter roles", "UCI": "UCI diabetes"}
    disp_rows = []          # (kind, payload): ("group", title) or ("scen", index into SCENARIOS)
    grp = None
    for i, s in enumerate(SCENARIOS):
        if s[1] != grp:
            disp_rows.append(("group", GROUP_TITLES[s[1]]))
            grp = s[1]
        disp_rows.append(("scen", i))
    nR = len(disp_rows)
    M = np.full((nR, len(cols)), np.nan)
    dag = np.zeros((nR, len(cols)), bool)
    ylab = []
    for r, (kind, v) in enumerate(disp_rows):
        if kind == "group":
            ylab.append("")
            continue
        sid, group, label, _ = SCENARIOS[v]
        c = cells[cells.scenario == sid]
        ylab.append(f"{SHORT.get(sid, label)} ({int(c[c.main].reps.iloc[0])})")
        for j, m in enumerate(cols):
            x = c[c.main & (c.method == m)]
            if x.empty:
                x = c[(~c.main) & (c.method == m)]
                dag[r, j] = not x.empty
            if not x.empty:
                M[r, j] = x.f1.iloc[0]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 6.6), gridspec_kw=dict(width_ratios=[2.7, 1.2]))
    cmap = LinearSegmentedColormap.from_list("f1", ["#f7f7f7", "#9ecae1", "#08519c"])
    C = np.where(np.isnan(M), 0, M)
    C[:, cols.index("oracle")] = 0
    ax.imshow(C, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    oj = cols.index("oracle")
    ax.add_patch(plt.Rectangle((oj - .5, -.5), 1, nR, color="#ececec", zorder=1))
    for r, (kind, v) in enumerate(disp_rows):
        if kind == "group":
            ax.add_patch(plt.Rectangle((-.5, r - .5), len(cols), 1, color="white", zorder=5))   # above the box
            ax.text(-.45, r, v, ha="left", va="center", fontsize=7.5, fontweight="bold", zorder=6)
            continue
        dep = [M[r, j] for j, m in enumerate(cols) if m != "oracle" and not np.isnan(M[r, j])]
        top = max(dep) if dep else np.nan
        for j in range(len(cols)):
            if np.isnan(M[r, j]):
                ax.text(j, r, "–", ha="center", va="center", fontsize=7, color="#999999", zorder=3)
                continue
            bold = cols[j] != "oracle" and abs(M[r, j] - top) < 5e-3
            txt = f"{M[r, j]:.2f}".lstrip("0") + ("†" if dag[r, j] else "")
            color = "#777777" if cols[j] == "oracle" else ("white" if M[r, j] > .6 else "black")
            ax.text(j, r, txt, ha="center", va="center", fontsize=7.2, fontweight="bold" if bold else "normal",
                    color=color, zorder=3)
    ax.set_xticks(range(len(cols)), head, fontsize=7, rotation=40, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(nR), ylab, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.tick_top()
    n_own = len(["ours", "combined"] + [m for m in OWN if m != "ours"])        # our groupings lead the columns
    assert cols[:n_own] == ["ours", "combined", "leiden", "ridge"]
    ax.add_patch(plt.Rectangle((-.5, -.5), n_own, nR, fill=False, lw=2, ec="#d95f02", zorder=4))
    ax.text((n_own - 1) / 2, nR - .35, "our groupings", ha="center", va="top", fontsize=7, color="#d95f02",
            fontweight="bold")
    ax.text(n_own + (len(cols) - 2 - n_own) / 2, nR - .35, "alternative groupings", ha="center", va="top",
            fontsize=7, color="#555555", fontweight="bold")
    ax.text(-.02, 1.01, "(a) observation-level F1 (replicates)\n(bold = best non-oracle)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7.5, fontweight="bold")

    dd = diffs.set_index("scenario")
    names = {"ridge": "ridge", "riskbins": "risk bins", "kmeans": "K-means", "mahal": "Mahal.", "cemtop": "CEM-3",
             "disp": "dispar.-targeted", "ours": "forest", "leiden": "Leiden"}
    for r, (kind, v) in enumerate(disp_rows):
        if kind == "group":
            continue
        sid = SCENARIOS[v][0]
        if sid not in dd.index:
            continue
        x = dd.loc[sid]
        # 2026-10-11: both markers compare with the best ALTERNATIVE grouping (not one of ours)
        for tag, dy, fmt_, ms_, lw_, mfc in [("alt", -.17, "o", 5, 1.4, None), ("comb_alt", .17, "D", 4.5, 1.1, "white")]:
            if pd.isna(x.get(f"{tag}_diff")):
                continue
            d0, lo, hi = x[f"{tag}_diff"], x[f"{tag}_lo"], x[f"{tag}_hi"]
            colr = "#1b7837" if lo > 0 else ("#b2182b" if hi < 0 else "#555555")
            bx.errorbar([d0], [r + dy], xerr=[[d0 - lo], [hi - d0]], fmt=fmt_, color=colr, ms=ms_, lw=lw_,
                        **({"mfc": mfc} if mfc else {}))
    for r, (kind, v) in enumerate(disp_rows):
        if kind == "group":
            bx.axhspan(r - .5, r + .5, color="#f3f3f3", lw=0)
    bx.set_ylim(nR - .5, -.5)
    bx.set_yticks([])
    bx.axvline(0, color="black", lw=.8)
    bx.set_xlim(-.45, .45)
    bx.set_xlabel("difference in F1 (paired 95% CI)", fontsize=7.5)
    bx.tick_params(labelsize=7)
    bx.set_title("(b) vs best alternative grouping\n     ● forest   ◇ combined", fontsize=7,
                 loc="left", pad=4)
    fig.tight_layout()
    for d in [out] + ([fig_dir] if fig_dir else []):
        fig.savefig(d / "fig_scoreboard.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_scoreboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--figure-from", type=Path, default=None,
                    help="re-render the figure only, from a saved run's scoreboard_{cells,diffs}.csv (no recomputation)")
    ap.add_argument("--table-from", type=Path, default=None,
                    help="rewrite the error-control table only, from a saved run's error_control.csv")
    ap.add_argument("--mu-flags", type=Path, default=None,
                    help="false_flags.csv of the mu localization simulations: adds the mu block to that table")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.table_from is not None:
        mu_ff = pd.read_csv(args.mu_flags, keep_default_na=False, na_values=[""]) if args.mu_flags else None
        write_error_table(pd.read_csv(args.table_from / "error_control.csv"), args.out, mu_ff)
        return
    if args.figure_from is not None:
        cells = pd.read_csv(args.figure_from / "scoreboard_cells.csv", dtype={"scenario": str})
        diffs = pd.read_csv(args.figure_from / "scoreboard_diffs.csv", dtype={"scenario": str})
        figure(cells, diffs, args.out, args.fig_dir)
        return
    rng = np.random.default_rng(args.seed)
    cells, diffs = scoreboard(rng)
    ec = error_control()
    cov = coverage(cells)
    cells.to_csv(args.out / "scoreboard_cells.csv", index=False)
    diffs.to_csv(args.out / "scoreboard_diffs.csv", index=False)
    ec.to_csv(args.out / "error_control.csv", index=False)
    cov.to_csv(args.out / "coverage_alternatives.csv", index=False)
    write_tables(cells, diffs, ec, cov, args.out)
    figure(cells, diffs, args.out, args.fig_dir)
    (args.out / "provenance.json").write_text(json.dumps(dict(seed=args.seed, b_draws=B_DRAWS, scenarios=SCENARIOS),
                                                         indent=1, default=str))
    print(diffs.round(3).to_string())


if __name__ == "__main__":
    main()
