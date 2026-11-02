"""Command line: every run writes a new directory with a manifest; nothing is overwritten.

    python -m pocket_audit discover --config configs/features_uci.yaml --run-dir results/<new>
    python -m pocket_audit audit    --run-dir results/<same>          # only after discovery is reviewed
    python -m pocket_audit simulate --reps 20 --run-dir results/<new> [--tau 0.8] [--clustered] [--dgp graded]
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import METHODS, SupportRule
from .data import load_config, load_dataset
from .pipeline import PipelineConfig, audit, config_dict, discover


def _versions() -> dict:
    import importlib
    out = {"python": platform.python_version()}
    for m in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "graph_tool", "igraph", "leidenalg"]:
        try:
            out[m] = str(getattr(importlib.import_module(m), "__version__", "?"))
        except Exception as e:  # recorded, not hidden
            out[m] = f"unavailable: {type(e).__name__}"
    return out


def _git() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _new_run_dir(path: str) -> Path:
    p = Path(path)
    if p.exists():
        sys.exit(f"refusing to overwrite existing run directory: {p}")
    p.mkdir(parents=True)
    return p


def _pipeline_config(ds_cfg: dict, args) -> PipelineConfig:
    s = ds_cfg.get("support", {})
    return PipelineConfig(
        seed=args.seed, alpha=args.alpha, n_jobs=args.n_jobs,
        support=SupportRule(**s) if s else SupportRule(),
        pairs=[tuple(p) for p in ds_cfg.get("pairs", [])],
        primary_models=ds_cfg.get("primary_models"),
        n_boot=args.n_boot,
        inference=args.inference,
        sbm_n_init=args.sbm_n_init,
        run_sbm=not args.no_sbm,
        min_projected_per_group=args.min_projected_per_group,
        min_graph_profiled_share=args.min_graph_profiled_share,
        abstain_quantile=args.abstain_quantile,
        project_expected_events=args.project_expected_events,
        se_floor=args.se_floor,
        restricted_loss=args.restricted_loss,
        tails=args.tails,
        sbm_tree_cut=args.sbm_tree_cut,
        graph_method=args.graph_method,
        select_on_audit_design=args.select_on_audit_design,
    )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def cmd_discover(args):
    run = _new_run_dir(args.run_dir)
    ds_cfg = load_config(args.config)
    data = load_dataset(args.config, n_units=args.n_units, seed=args.seed)
    cfg = _pipeline_config(ds_cfg, args)
    frozen = discover(data, cfg)
    role = frozen["role"]
    pd.DataFrame({"row_id": data.row_id, "unit": data.unit, "role": role}).to_csv(run / "roles.csv", index=False)
    # Development diagnostics use DISCOVERY rows (outcomes allowed there), never audit rows.
    disc = role == "discovery"
    from sklearn.metrics import brier_score_loss, roc_auc_score
    diag = []
    for m, p in frozen["preds"].items():
        p = np.asarray(p)
        ok = disc & ~np.isnan(p)
        diag.append({"model": m, "rows": "discovery", "n": int(ok.sum()),
                     "auroc": roc_auc_score(data.y[ok], p[ok]), "brier": brier_score_loss(data.y[ok], p[ok]),
                     "mean_pred": float(p[ok].mean()), "mean_outcome": float(data.y[ok].mean())})
    pd.DataFrame(diag).to_csv(run / "model_diagnostics_discovery.csv", index=False)
    rep = frozen["reports"]
    if "sbm" in rep:
        rep["sbm"]["levels"].to_csv(run / "sbm_levels.csv", index=False)
    counts = pd.crosstab(role, data.protected[next(c for c in data.contrasts
                                                   if c.name == data.primary_contrast).attribute].fillna("missing"))
    counts.to_csv(run / "role_group_counts.csv")
    with open(run / "frozen.pkl", "wb") as f:
        pickle.dump({"frozen": frozen, "config_path": args.config, "n_units": args.n_units,
                     "seed": args.seed}, f)
    manifest = {"stage": "discover", "argv": sys.argv, "dataset_log": data.log,
                "pipeline_config": config_dict(cfg), "reports": rep, "versions": _versions(),
                "git": _git()}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    print(json.dumps({"graph": rep["graph"], "timings": rep["timings"],
                      "sbm_status": rep.get("sbm", {}).get("status"),
                      "n_pockets": frozen["n_pockets"], "peak_rss_mb": rep["peak_rss_mb"]},
                     indent=2, default=_json_default))
    print(f"\nDiscovery frozen in {run}. Review sbm_levels.csv and manifest.json before auditing.")


def cmd_audit(args):
    run = Path(args.run_dir)
    if (run / "audit_results.csv").exists():
        sys.exit("audit already run for this discovery; start a new run directory")
    with open(run / "frozen.pkl", "rb") as f:
        saved = pickle.load(f)
    data = load_dataset(saved["config_path"], n_units=saved["n_units"], seed=saved["seed"])
    out = audit(data, saved["frozen"])
    out["results"].to_csv(run / "audit_results.csv", index=False)
    out["q_curves"].to_csv(run / "q_curves.csv", index=False)
    out["coverage"].to_csv(run / "assignment_coverage.csv", index=False)
    out["assignments"].to_csv(run / "audit_assignments.csv.gz", index=False)   # pocket of every audit row, per partition
    maxima = {f"{part}__{fam}": np.asarray(v.pop("M")) for part, fams in out["summary"].items()
              for fam, v in fams.items() if v.get("M") is not None}
    np.savez_compressed(run / "bootstrap_maxima.npz", **maxima)               # CIs at any alpha, A(t), etc.
    (run / "audit_summary.json").write_text(json.dumps(
        {"summary": out["summary"], "argv": sys.argv, "git": _git()}, indent=2, default=_json_default))
    print(json.dumps(out["summary"], indent=2, default=_json_default))


def cmd_simulate(args):
    from .simulation import GRADED, TOY, SimConfig, run_simulation, summarize
    run = _new_run_dir(args.run_dir)
    dgp = {"original": {}, "toy": TOY, "graded": GRADED}[args.dgp]
    if args.p_noise is not None:
        dgp = {**dgp, "p_noise": args.p_noise}
    sc = SimConfig(n=args.n, tau=args.tau, region_feature=args.region_feature,
                   s_x_assoc=args.s_x_assoc,
                   n_units=args.n_clusters if args.clustered else None,
                   shared_frac=args.shared_frac if args.clustered else 0.0, **dgp)
    support = SupportRule(min_n=args.min_n, min_units=10 if args.clustered else 1,
                          min_effective_units=10 if args.clustered else 1)
    pc = PipelineConfig(alpha=args.alpha, n_jobs=args.n_jobs, n_boot=args.n_boot, inference=args.inference, sbm_n_init=args.sbm_n_init,
                        support=support, min_projected_per_group=args.min_n, run_sbm=not args.no_sbm)
    df = run_simulation(sc, pc, args.reps, base_seed=args.seed)
    df.to_csv(run / "replicates.csv", index=False)
    summ = summarize(df)
    summ.to_csv(run / "summary.csv", index=False)
    (run / "manifest.json").write_text(json.dumps(
        {"stage": "simulate", "argv": sys.argv, "sim_config": asdict(sc),
         "pipeline_config": config_dict(pc), "versions": _versions(), "git": _git()},
        indent=2, default=_json_default))
    print(summ.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pocket_audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", required=True)
    common.add_argument("--alpha", type=float, default=.05)
    common.add_argument("--n-jobs", type=int, default=1)
    common.add_argument("--seed", type=int, default=20260928)
    common.add_argument("--n-boot", type=int, default=4999)
    common.add_argument("--sbm-n-init", type=int, default=3)
    common.add_argument("--no-sbm", action="store_true", help="skip graph-tool (Leiden + baselines only)")
    # frozen with the discovery config, so it cannot be changed after audit results are seen
    frozen_inference = argparse.ArgumentParser(add_help=False)
    frozen_inference.add_argument("--inference", choices=METHODS, default="wild_restricted")
    d = sub.add_parser("discover", parents=[common, frozen_inference])
    d.add_argument("--config", required=True)
    d.add_argument("--n-units", type=int, default=None, help="seeded subsample of units for fast runs")
    d.add_argument("--min-projected-per-group", type=int, default=150,
                   help="resolution rule: projected audit rows per primary group for a supported pocket")
    d.add_argument("--min-graph-profiled-share", type=float, default=None,
                   help="proposal: a level also needs this share of graph rows in profiled blocks")
    # settings used in the paper; the defaults keep the earlier behavior
    d.add_argument("--abstain-quantile", type=float, default=.05, help="PipelineConfig.abstain_quantile")
    d.add_argument("--project-expected-events", action="store_true",
                   help="resolution rule also projects the audit's expected-event support (predictions only)")
    d.add_argument("--se-floor", action="store_true", help="AuditConfig.se_floor (null-model SE floor)")
    d.add_argument("--restricted-loss", action="store_true", help="AuditConfig.restricted_loss (theta/psi)")
    d.add_argument("--tails", choices=["symmetric", "equal"], default="symmetric")
    d.add_argument("--sbm-tree-cut", action="store_true", help="also freeze the SBM tree-cut partition (sensitivity)")
    d.add_argument("--graph-method", choices=["approx", "exact_cpu", "exact_gpu"], default="approx")
    d.add_argument("--select-on-audit-design", action="store_true",
                   help="v2 proposal: choose the resolution from the audit rows' exact support (no outcomes)")
    d.set_defaults(func=cmd_discover)
    a = sub.add_parser("audit", parents=[common])
    a.set_defaults(func=cmd_audit)
    s = sub.add_parser("simulate", parents=[common, frozen_inference])
    s.add_argument("--reps", type=int, default=20)
    s.add_argument("--n", type=int, default=20000)
    s.add_argument("--tau", type=float, default=0.0)
    s.add_argument("--region-feature", default="x1")
    s.add_argument("--dgp", choices=["original", "toy", "graded"], default="original",
                   help="toy: one relevant, one irrelevant feature; graded: strong, moderate, weak + noise")
    s.add_argument("--p-noise", type=int, default=None, help="irrelevant features (default: the DGP's)")
    s.add_argument("--s-x-assoc", type=float, default=0.0)
    s.add_argument("--clustered", action="store_true")
    s.add_argument("--n-clusters", type=int, default=150)
    s.add_argument("--shared-frac", type=float, default=0.2)
    s.add_argument("--min-n", type=int, default=100)
    s.set_defaults(func=cmd_simulate)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
