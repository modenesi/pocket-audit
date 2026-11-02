"""Care-management application: the Obermeyer et al. (2019) synthetic data, audited for health need.

    python applications/obermeyer/download_data.py
    python applications/obermeyer/audit_obermeyer.py discover --run-dir results/obermeyer
    python applications/obermeyer/audit_obermeyer.py describe --run-dir results/obermeyer
    python applications/obermeyer/audit_obermeyer.py validate --run-dir results/obermeyer
    python applications/obermeyer/audit_obermeyer.py audit    --run-dir results/obermeyer

``discover`` fits the assessed models on development rows and freezes the pocket maps on discovery rows;
``describe`` profiles the pockets on discovery rows; ``validate`` draws synthetic outcomes on the audit rows'
features and frozen assignments; none of these reads an audit outcome. ``audit`` reads them once.
Settings: applications/obermeyer/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "code"))

from pocket_audit.assignment import ABSTAIN  # noqa: E402
from pocket_audit.audit import AuditConfig, SupportRule, run_audit  # noqa: E402
from pocket_audit.data import AuditData, Contrast, to_numeric_matrix  # noqa: E402
from pocket_audit.pipeline import PipelineConfig, assign_partition, audit, config_dict, discover  # noqa: E402
from pocket_audit.splits import allocate_units  # noqa: E402

SOURCE = ROOT / "data/obermeyer2019/data_new.csv"
SEED, DEV_SEED = 20261101, 20261102
MODELS = ("score", "cost_label", "health_label")
PAIRS = [("cost_label", "health_label"), ("score", "health_label")]
PRIMARY_MAPS = ("sbm", "ridge_kmeans")
ALPHA_PER_MAP = 0.025                 # combined audit: each primary map at alpha/2, 0.05 in total
TOLERANCES = (0.25, 0.5)              # active chronic conditions


def load() -> tuple[AuditData, pd.DataFrame]:
    from download_obermeyer2019 import SHA256, sha256
    if sha256(SOURCE) != SHA256:
        raise RuntimeError("source checksum differs from the pinned download")
    df = pd.read_csv(SOURCE)
    pocket_features = [c for c in df.columns if c.endswith("_tm1")]
    assert len(pocket_features) == 148 and "dem_female" not in pocket_features
    row_id = np.arange(len(df))
    protected = pd.DataFrame({"race": df.race.to_numpy(),
                              "sex": np.where(df.dem_female.to_numpy() == 1, "female", "male")})
    data = AuditData(name="obermeyer2019", X=df[pocket_features].astype(float), y=df.gagne_sum_t.to_numpy(float),
                     unit=row_id.astype(str), row_id=row_id, protected=protected,
                     contrasts=[Contrast("black_vs_white", "race", "black", "white"),
                                Contrast("female_vs_male", "sex", "female", "male")],
                     primary_contrast="black_vs_white", outcome="continuous",
                     log={"source": str(SOURCE.relative_to(ROOT)), "sha256": SHA256, "n_rows": len(df),
                          "pocket_features": pocket_features})
    return data, df


def pipeline_config(n_jobs: int) -> PipelineConfig:
    return PipelineConfig(
        seed=SEED, n_jobs=n_jobs, alpha=ALPHA_PER_MAP, n_trees=200, min_samples_leaf=20, max_features="sqrt", k=30,
        sbm_n_init=3, min_pocket_size=50, min_projected_per_group=150, min_graph_profiled_share=0.8,
        abstain_quantile=0.01, sbm_tree_cut=True, run_sbm=True, run_leiden=True, run_baselines=True,
        support=SupportRule(min_n=100, min_units=10, min_effective_units=10, max_unit_share=None, min_expected=0),
        pairs=PAIRS, primary_models=list(MODELS), n_boot=4999, inference="wild_restricted_mammen",
        tails="symmetric", se_floor=False, outcome="continuous", include_mu=True)


def fit_models(df: pd.DataFrame, y: np.ndarray, dev: np.ndarray, unit: np.ndarray) -> tuple[dict, dict]:
    """Assessed models on development rows only; outputs mapped to the need scale by isotonic fits on dev-calib."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LassoCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dev_idx = np.flatnonzero(dev)
    part = allocate_units(unit[dev_idx], (0.75, 0.25), roles=("train", "calib"), seed=DEV_SEED)
    train, calib = dev_idx[part == "train"], dev_idx[part == "calib"]
    feats = [c for c in df.columns if c.endswith("_tm1")] + ["dem_female"]
    X = df[feats].to_numpy(float)
    raw = {"score": df.risk_score_t.to_numpy(float)}
    labels = {"cost_label": np.log(df.cost_t.to_numpy(float) + 1.0), "health_label": y}
    diag = {"n_train": int(len(train)), "n_calib": int(len(calib)), "assessed_features": feats}
    for name, target in labels.items():
        lasso = make_pipeline(StandardScaler(), LassoCV(alphas=100, cv=5, max_iter=10000, random_state=0))
        lasso.fit(X[train], target[train])
        raw[name] = lasso.predict(X)
        diag[f"{name}_alpha"] = float(lasso[-1].alpha_)
        diag[f"{name}_nonzero"] = int((lasso[-1].coef_ != 0).sum())
    preds = {}
    for name in MODELS:
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(raw[name][calib], y[calib])
        preds[name] = iso.predict(raw[name])
        r = y[calib] - preds[name][calib]
        diag[f"{name}_calib_r2"] = float(1 - np.mean(r ** 2) / np.var(y[calib]))
        diag[f"{name}_calib_mean_error"] = float(np.mean(-r))
    return preds, {"diag": diag, "train": train, "calib": calib, "raw": raw}


def _git() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "src", "code"))}


def cmd_discover(args):
    run = Path(args.run_dir)
    if (run / "frozen.pkl").exists():
        raise SystemExit(f"{run} already holds a frozen discovery")
    run.mkdir(parents=True, exist_ok=True)
    data, df = load()
    cfg = pipeline_config(args.n_jobs)
    role = allocate_units(data.unit, cfg.fractions, seed=cfg.seed)          # the same call discover() makes
    t0 = time.time()
    preds, extra = fit_models(df, data.y, role == "development", data.unit)
    frozen = discover(data, cfg, preds=preds)
    if not np.array_equal(frozen["role"], role):
        raise RuntimeError("role allocation differs from discover()")
    with open(run / "frozen.pkl", "wb") as f:
        pickle.dump({"frozen": frozen, "extra": extra}, f)
    pd.DataFrame({"row_id": data.row_id, "role": role}).to_csv(run / "roles.csv", index=False)
    rep = frozen["reports"]
    rep["sbm"]["levels"].to_csv(run / "sbm_levels.csv", index=False)
    import graph_tool, lightgbm, sklearn  # noqa: F401
    manifest = {"stage": "discover", "argv": sys.argv, "git": _git(), "dataset": data.log | {"pocket_features": len(
        data.log["pocket_features"])}, "config": config_dict(cfg), "models": extra["diag"],
        "roles": pd.Series(role).value_counts().to_dict(), "n_pockets": frozen["n_pockets"],
        "chosen_level": rep["sbm"]["chosen_level"], "sbm_status": rep["sbm"]["status"],
        "forest": rep["forest"], "graph": rep["graph"], "timings": rep["timings"], "seconds": round(time.time() - t0),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "graph_tool": graph_tool.__version__.split()[0]}}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("models", "n_pockets", "chosen_level", "sbm_status", "forest",
                                               "timings")}, indent=2, default=str))
    print(rep["sbm"]["levels"].to_string(index=False))


def _load_frozen(run: Path):
    with open(run / "frozen.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved["frozen"], saved["extra"]


def cmd_describe(args):
    """Pocket profiles on DISCOVERY rows only (graph + calibration rows); audit rows are never read."""
    run = Path(args.run_dir)
    data, df = load()
    frozen, _ = _load_frozen(run)
    disc = np.flatnonzero(frozen["role"] == "discovery")
    Xnum = to_numeric_matrix(data.X)
    cols = data.X.columns
    mu, sd = data.X.iloc[disc].mean(), data.X.iloc[disc].std().replace(0, np.nan)
    out = []
    for name in ("sbm", "ridge_kmeans", "sbm_cut"):
        lab = assign_partition(frozen, name, disc, Xnum)
        for c in sorted(set(lab)):
            m = lab == c
            smd = ((data.X.iloc[disc][m].mean() - mu) / sd).dropna()
            top = smd.abs().sort_values(ascending=False).index[:5]
            out.append({"partition": name, "pocket": int(c), "n_discovery": int(m.sum()),
                        "share_black": float((data.protected.race.iloc[disc][m] == "black").mean()),
                        "mean_need_t": float(data.y[disc][m].mean()),
                        "mean_gagne_tm1": float(df.gagne_sum_tm1.iloc[disc][m].mean()),
                        "mean_cost_tm1": float(df[[c_ for c_ in cols if c_.startswith("cost_")]]
                                               .iloc[disc][m].sum(1).mean()),
                        "top_features": "; ".join(f"{t} ({smd[t]:+.2f})" for t in top)})
    res = pd.DataFrame(out)
    res.to_csv(run / "pocket_profiles_discovery.csv", index=False)
    print(res.to_string(index=False, max_colwidth=120))


def _truths(preds: dict, mean_y: np.ndarray, pocket: np.ndarray, prot: pd.DataFrame, contrasts) -> dict:
    """Finite-population truths of mu and delta given E[Y | x] = mean_y on the audit rows."""
    t = {}
    for c in ["all"] + sorted(int(k) for k in np.unique(pocket) if k != ABSTAIN):
        cell = np.ones(len(mean_y), bool) if c == "all" else pocket == c
        for m in MODELS:
            e = preds[m] - mean_y
            t[("mu", m, "", c)] = float(e[cell].mean())
            for con in contrasts:
                a = prot[con.attribute].to_numpy()
                f, r = cell & (a == con.focal), cell & (a == con.reference)
                if f.any() and r.any():
                    t[("delta", m, con.name, c)] = float(e[f].mean() - e[r].mean())
    return t


def _replicate(args_tuple):
    scenario, gen, rep, ctx = args_tuple
    rng = np.random.default_rng([SEED, ("null", "effect").index(scenario), MODELS.index(gen), rep])
    f = ctx["preds"][gen]
    dec = np.clip(np.searchsorted(ctx["edges"][gen], f, side="right"), 0, 9)
    e = np.empty(len(f))
    for d in range(10):
        m = dec == d
        pool = ctx["resid"][gen][d]
        e[m] = pool[rng.integers(0, len(pool), m.sum())]
    mean_y = f + (0.5 * ctx["region"] if scenario == "effect" else 0.0)
    y = mean_y + e
    rows = []
    for mp in PRIMARY_MAPS:
        lab = ctx["pockets"][mp]
        acfg = AuditConfig(support=ctx["support"], pairs=PAIRS, primary_models=list(MODELS), n_boot=ctx["n_boot"],
                           alpha=ALPHA_PER_MAP, seed=11 + rep, method="wild_restricted_mammen", outcome="continuous",
                           include_losses=False, include_mu=True, engine=ctx.get("engine", "loop"))
        table, _ = run_audit(y, ctx["preds"], lab, ctx["unit"], ctx["prot"], ctx["contrasts"], acfg)
        truth = _truths(ctx["preds"], mean_y, lab, ctx["prot"], ctx["contrasts"])
        for r in table.itertuples():
            if r.kind == "kappa":                       # outcome-free: its truth equals its estimate
                continue
            tv = truth.get((r.kind, r.model, r.contrast, r.pocket), np.nan)
            rows.append(dict(scenario=scenario, gen_model=gen, rep=rep, map=mp, kind=r.kind, model=r.model,
                             contrast=r.contrast, pocket=r.pocket, status=r.status, estimate=r.estimate,
                             ci_low=r.ci_low, ci_high=r.ci_high, truth=tv, flag=bool(r.flag)))
    return rows


def cmd_validate(args):
    from joblib import Parallel, delayed
    from statsmodels.stats.proportion import proportion_confint
    run = Path(args.run_dir)
    data, df = load()
    frozen, extra = _load_frozen(run)
    rows = np.flatnonzero(frozen["role"] == "audit")
    Xnum = to_numeric_matrix(data.X)
    calib = extra["calib"]
    preds_all = frozen["preds"]
    ctx = {"preds": {m: np.asarray(preds_all[m], float)[rows] for m in MODELS},
           "pockets": {mp: assign_partition(frozen, mp, rows, Xnum) for mp in PRIMARY_MAPS},
           "unit": np.asarray(data.unit)[rows], "prot": data.protected.iloc[rows].reset_index(drop=True),
           "contrasts": data.contrasts, "support": frozen["config"].support, "n_boot": args.n_boot,
           "region": (df.gagne_sum_tm1.to_numpy()[rows] >= 3).astype(float), "edges": {}, "resid": {},
           "engine": args.engine}
    for m in MODELS:          # dev-calib residuals, centred within deciles of the fitted value (no audit outcome)
        fc = np.asarray(preds_all[m], float)[calib]
        rc = data.y[calib] - fc
        edges = np.unique(np.quantile(fc, np.linspace(0, 1, 11)[1:-1]))
        dec = np.clip(np.searchsorted(edges, fc, side="right"), 0, 9)
        ctx["edges"][m] = edges
        pools = []
        for d in range(10):
            pool = rc[dec == d] if (dec == d).any() else rc
            pools.append(pool - pool.mean())
        ctx["resid"][m] = pools
    jobs = [("null", g, r, ctx) for g in MODELS for r in range(args.reps)]
    jobs += [("effect", "health_label", r, ctx) for r in range(args.reps)]
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, batch_size=2, verbose=5)(delayed(_replicate)(j) for j in jobs)
    out = pd.DataFrame([x for r in res for x in r])
    out.to_csv(run / "validation_coordinates.csv.gz", index=False)
    tested = out[out.status == "tested"].copy()
    tested["null"] = tested.truth.abs() < 1e-9
    tested["covered"] = (tested.ci_low <= tested.truth) & (tested.truth <= tested.ci_high)
    summ = []
    for (sc, gm), g in tested.groupby(["scenario", "gen_model"]):
        for mp in list(PRIMARY_MAPS) + ["combined"]:
            gg = g if mp == "combined" else g[g["map"] == mp]
            per = gg.groupby("rep")
            anyf = per.apply(lambda d: bool((d.flag & d.null).any()), include_groups=False)
            cov = per.covered.all()
            n = len(anyf)
            lo, hi = proportion_confint(int(anyf.sum()), n, method="wilson")
            summ.append(dict(scenario=sc, gen_model=gm, map=mp, replicates=n, fwer=anyf.mean(), fwer_lo=lo,
                             fwer_hi=hi, simultaneous_coverage=cov.mean(), coordinate_coverage=gg.covered.mean(),
                             null_coords_per_rep=gg.null.sum() / n, tested_per_rep=len(gg) / n))
    summ = pd.DataFrame(summ)
    summ.to_csv(run / "validation_summary.csv", index=False)
    status = out.status.value_counts().to_dict()
    (run / "validation_manifest.json").write_text(json.dumps(
        {"stage": "validate", "argv": sys.argv, "git": _git(), "reps": args.reps, "n_boot": args.n_boot,
         "engine": args.engine,
         "status_counts": status, "seconds": round(time.time() - t0)}, indent=2, default=str))
    print(summ.to_string(index=False))
    print(status)


def cmd_audit(args):
    run = Path(args.run_dir)
    if (run / "audit_results.csv").exists():
        raise SystemExit("audit already run; the protocol allows one run")
    data, df = load()
    frozen, extra = _load_frozen(run)
    t0 = time.time()
    out = audit(data, frozen)
    res = out["results"]
    res.to_csv(run / "audit_results.csv", index=False)
    out["coverage"].to_csv(run / "assignment_coverage.csv", index=False)
    out["assignments"].to_csv(run / "audit_assignments.csv.gz", index=False)
    summ = {p: {f: {k: v for k, v in s.items() if k != "M"} for f, s in fam.items()}
            for p, fam in out["summary"].items()} if isinstance(next(iter(out["summary"].values())), dict) else {}
    (run / "audit_manifest.json").write_text(json.dumps(
        {"stage": "audit", "argv": sys.argv, "git": _git(), "seconds": round(time.time() - t0),
         "summary": summ}, indent=2, default=str))
    show = res[res.partition.isin(PRIMARY_MAPS) & (res.status == "tested")]
    cols = ["partition", "kind", "model", "contrast", "pocket", "n_focal", "n_reference", "estimate", "ci_low",
            "ci_high", "flag"]
    print(show[cols].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "describe", "validate", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        if name == "discover":
            p.add_argument("--n-jobs", type=int, default=16)
        if name == "validate":
            p.add_argument("--reps", type=int, default=500)
            p.add_argument("--n-boot", type=int, default=1999)
            p.add_argument("--n-jobs", type=int, default=30)
            p.add_argument("--engine", choices=("loop", "gemm"), default="gemm",
                           help="bootstrap arithmetic; gemm gives the same draws and results, faster")
    args = ap.parse_args(argv)
    {"discover": cmd_discover, "describe": cmd_describe, "validate": cmd_validate, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
