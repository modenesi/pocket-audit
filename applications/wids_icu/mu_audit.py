"""WiDS 2020 ICU application: the mu family (pocket-average signed prediction error) on the frozen v1 maps.

Secondary, post-protocol, exploratory (the v1 audit outcomes were read for delta/kappa in October 2026).
Two stages, run in this order:

    python applications/wids_icu/mu_audit.py validate --out results/wids_mu
    python applications/wids_icu/mu_audit.py audit    --out results/wids_mu

``validate`` draws synthetic outcomes on the real audit rows, hospitals and frozen pocket assignments and never
reads an audit outcome; ``audit`` reads them once. See applications/wids_icu/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pocket_audit.audit import AuditConfig, run_audit  # noqa: E402
from pocket_audit.data import load_dataset, to_numeric_matrix  # noqa: E402
from pocket_audit.pipeline import assign_partition  # noqa: E402

RUN_V1 = ROOT / "results/wids_v1"
MAPS = ("sbm_cut", "sbm")
MODELS = ("lightgbm", "apache")
ALPHA = 0.0125


def load_inputs() -> dict:
    with open(RUN_V1 / "frozen.pkl", "rb") as f:
        saved = pickle.load(f)
    frozen = saved["frozen"]
    data = load_dataset(saved["config_path"], root=ROOT, n_units=saved["n_units"], seed=saved["seed"])
    role = frozen["role"]
    rows = np.flatnonzero(role == "audit")
    Xnum = to_numeric_matrix(data.X)
    pockets = {m: assign_partition(frozen, m, rows, Xnum) for m in MAPS}
    # row alignment with the saved v1 assignments
    saved_assign = pd.read_csv(RUN_V1 / "audit_assignments.csv.gz")
    if not np.array_equal(saved_assign.row_id.astype(str).to_numpy(), np.asarray(data.row_id).astype(str)[rows]):
        raise RuntimeError("audit row order differs from the saved v1 assignments")
    for m in MAPS:
        if not np.array_equal(saved_assign[m].to_numpy(), pockets[m]):
            raise RuntimeError(f"recomputed {m} assignments differ from the saved v1 assignments")
    preds = {m: np.asarray(frozen["preds"][m], float) for m in MODELS}
    common = rows[~np.isnan(preds["apache"][rows])]               # stays with a valid APACHE probability
    keep = ~np.isnan(preds["apache"][rows])
    out = {"data": data, "frozen": frozen, "role": role, "rows": common,
           "pockets": {m: pockets[m][keep] for m in MAPS},
           "preds": {m: preds[m][common] for m in MODELS},
           "unit": np.asarray(data.unit)[common], "y": np.asarray(data.y, float)[common],
           "protected": data.protected.iloc[common].reset_index(drop=True),
           "contrasts": data.contrasts, "support": frozen["config"].support,
           "preds_all": preds, "y_all": np.asarray(data.y, float), "unit_all": np.asarray(data.unit)}
    out["ventilated"] = data.X["ventilated_apache"].to_numpy()[common] == 1
    hr = pd.to_numeric(data.X["d1_heartrate_max"], errors="coerce").to_numpy()[common]
    out["heartrate"] = np.where(np.isnan(hr), np.nanmedian(hr), hr)
    return out


def audit_config(n_boot: int, seed: int, support) -> AuditConfig:
    return AuditConfig(support=support, primary_models=list(MODELS), n_boot=n_boot, alpha=ALPHA, seed=seed,
                       method="wild_restricted_mammen", se_floor=True, include_mu=True, include_disparity=False,
                       include_losses=False)


def shock_scale(inp: dict, model: str) -> float:
    """Method-of-moments scale s of v_h ~ s*U(-1,1) from non-audit hospitals' observed/expected dispersion."""
    p, y, unit = inp["preds_all"][model], inp["y_all"], inp["unit_all"]
    ok = (inp["role"] != "audit") & ~np.isnan(p)
    df = pd.DataFrame({"h": unit[ok], "y": y[ok], "p": p[ok], "v": p[ok] * (1 - p[ok])})
    g = df.groupby("h").agg(O=("y", "sum"), E=("p", "sum"), V=("v", "sum"), n=("y", "size"))
    g = g[g.E >= 5]
    oe = g.O / g.E
    oe = oe / np.average(oe, weights=g.E)                           # remove overall miscalibration
    # E[O/E | v_h] = 1 + v_h a_h with a_h = sum p(1-p) / sum p; binomial noise adds V/E^2; Var(v) = s^2 / 3
    a = g.V / g.E
    var_v = max(float((np.var(oe, ddof=1) - np.mean(g.V / g.E ** 2)) / np.mean(a ** 2)), 0.0)
    return float(min(np.sqrt(3 * var_v), 1.0))


def scenario_pgen(inp: dict, scenario: str, model: str) -> np.ndarray:
    p = inp["preds"][model]
    if scenario == "null":
        return p.copy()
    if scenario == "effect":                    # true risk 5 pp lower if ventilated: the model over-predicts, mu > 0
        return np.where(inp["ventilated"], np.maximum(p - 0.05, 0.0), p)
    if scenario == "weak_null":                 # opposite errors that cancel within each tree-cut pocket
        q = p.copy()
        lab = inp["pockets"]["sbm_cut"]
        for c in np.unique(lab):
            m = lab == c
            z = np.where(inp["heartrate"][m] > np.median(inp["heartrate"][m]), 1.0, -1.0)
            r = p[m] * (1 + 0.3 * z * (1 - p[m]))                   # stays inside (0, 1) without clipping
            q[m] = r * (p[m].mean() / r.mean())                     # rescale so the pocket mean equals p's
        if q.min() < 0 or q.max() > 1:
            raise RuntimeError("weak-null probabilities left (0, 1)")
        return q
    raise ValueError(scenario)


def one_replicate(inp: dict, scenario: str, model: str, s: float, rep: int, n_boot: int) -> list[dict]:
    code = 10 * ("null", "effect", "weak_null").index(scenario) + MODELS.index(model)
    rng = np.random.default_rng([20261031, code, rep])
    pgen = scenario_pgen(inp, scenario, model)
    hosp, codes = np.unique(inp["unit"], return_inverse=True)
    v = s * rng.uniform(-1, 1, len(hosp))[codes]
    y = (rng.random(len(pgen)) < pgen * (1 + v * (1 - pgen))).astype(float)
    out = []
    for mp in MAPS:
        lab = inp["pockets"][mp]
        table, _ = run_audit(y, inp["preds"], lab, inp["unit"], inp["protected"], inp["contrasts"],
                             audit_config(n_boot, 7 + rep, inp["support"]))
        for r in table.itertuples():
            cell = np.ones(len(y), bool) if r.pocket == "all" else lab == r.pocket
            cell &= ~np.isnan(inp["preds"][r.model])
            truth = float(np.mean(inp["preds"][r.model][cell] - pgen[cell])) if cell.any() else np.nan
            out.append(dict(scenario=scenario, gen_model=model, rep=rep, map=mp, model=r.model, pocket=r.pocket,
                            status=r.status, estimate=r.estimate, ci_low=r.ci_low, ci_high=r.ci_high,
                            truth=truth, flag=bool(r.flag)))
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.proportion import proportion_confint
    status = df.groupby(["scenario", "gen_model", "map"]).status.value_counts().unstack(fill_value=0)
    df = df[df.status == "tested"].copy()
    df["null"] = df.truth.abs() < 1e-12
    df["covered"] = (df.ci_low <= df.truth) & (df.truth <= df.ci_high)
    rows = []
    for (sc, gm, mp), g in df.groupby(["scenario", "gen_model", "map"]):
        per = g.groupby("rep")
        any_false = per.apply(lambda d: bool((d.flag & d.null).any()), include_groups=False)
        all_cov = per.covered.all()
        n = len(any_false)
        lo, hi = proportion_confint(int(any_false.sum()), n, method="wilson")
        clo, chi = proportion_confint(int(all_cov.sum()), n, method="wilson")
        rows.append(dict(scenario=sc, gen_model=gm, map=mp, replicates=n, fwer=any_false.mean(), fwer_lo=lo,
                         fwer_hi=hi, simultaneous_coverage=all_cov.mean(), coverage_lo=clo, coverage_hi=chi,
                         coordinate_coverage=g.covered.mean(), tested_per_rep=len(g) / n,
                         null_coords_per_rep=g.null.sum() / n,
                         degenerate_total=int(status.loc[(sc, gm, mp)].get("degenerate", 0))))
    return pd.DataFrame(rows)


def cmd_validate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inp = load_inputs()
    scales = {m: shock_scale(inp, m) for m in MODELS}
    jobs = [(sc, m, r) for sc in ("null", "effect", "weak_null") for m in MODELS for r in range(args.reps)]
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, batch_size=4)(
        delayed(one_replicate)(inp, sc, m, scales[m], r, args.n_boot) for sc, m, r in jobs)
    df = pd.DataFrame([row for rep in res for row in rep])
    df.to_csv(out / "validation_coordinates.csv.gz", index=False)
    summ = summarize(df)
    summ.to_csv(out / "validation_summary.csv", index=False)
    meta = {"stage": "validate", "argv": sys.argv, "shock_scale": scales, "n_rows": int(len(inp["rows"])),
            "n_hospitals": int(len(np.unique(inp["unit"]))), "seconds": round(time.time() - t0, 1),
            "alpha": ALPHA, "maps": MAPS, "models": MODELS}
    (out / "validation_manifest.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))
    print(summ.to_string(index=False))


def cmd_audit(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit_mu.csv").exists():
        raise SystemExit("audit already run; the protocol allows one run")
    inp = load_inputs()
    tables = []
    for mp in MAPS:
        table, summ = run_audit(inp["y"], inp["preds"], inp["pockets"][mp], inp["unit"], inp["protected"],
                                inp["contrasts"], audit_config(args.n_boot, 20261031, inp["support"]))
        table.insert(0, "map", mp)
        table["q"] = summ["primary"]["q"]
        tables.append(table)
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out / "audit_mu.csv", index=False)
    meta = {"stage": "audit", "argv": sys.argv, "n_rows": int(len(inp["rows"])), "alpha": ALPHA,
            "n_boot": args.n_boot, "maps": MAPS, "models": MODELS,
            "observed_mean": {m: float(np.mean(inp["preds"][m] - inp["y"])) for m in MODELS}}
    (out / "audit_manifest.json").write_text(json.dumps(meta, indent=2, default=str))
    cols = ["map", "model", "pocket", "n_focal", "status", "estimate", "ci_low", "ci_high", "flag"]
    print(res[cols].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--out", required=True)
    v.add_argument("--reps", type=int, default=500)
    v.add_argument("--n-boot", type=int, default=1999)
    v.add_argument("--n-jobs", type=int, default=30)
    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)
    a.add_argument("--n-boot", type=int, default=4999)
    args = ap.parse_args(argv)
    {"validate": cmd_validate, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
