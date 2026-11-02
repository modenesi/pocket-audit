"""Semi-synthetic validation on the real applications: real patients, features, hospitals and protected
attributes; simulated outcome and a planted, known calibration bias.

    p0(x)  = calibrated LightGBM fit to the real outcome on the REAL analysis's development + discovery rows
             (pocket_audit.semisynthetic.base; real audit rows never shape the truth)
    Y      ~ Bernoulli(p0), independently per replicate
    pf     = min(p0 + delta * S * R, .995)     additive, as in the 2026-10-04 simulations
    R      = a region defined on real features, fixed in advance (REGIONS below); S = primary focal group

Each replicate redraws Y and the unit roles (new seed), reruns discovery with the frozen application
application settings (B = 1999), audits the primary contrast, and scores every partition
against the finite-population truth. delta = 0 replicates are the null (the region is irrelevant there).

    python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/semi_wids \
        --cells null vent05 vent10 --reps 0 30 --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = {
    "wids": dict(config="configs/features_wids.yaml", alpha=.0125),
    "uci": dict(config="configs/features_uci.yaml", alpha=.025),
}
REGIONS = {   # fixed before any semi-synthetic run (2026-10-05)
    "wids": {"vent": "ventilated_apache == 1", "sepsis": "apache_3j_bodysystem == 'Sepsis'"},
    "uci": {"inpt": "number_inpatient >= 2"},
}
CELLS = {"null": (None, 0.0), "vent05": ("vent", .05), "vent10": ("vent", .10), "sepsis10": ("sepsis", .10),
         "inpt05": ("inpt", .05), "inpt10": ("inpt", .10)}
_BASE: dict = {}


def get_base(dataset: str, region_key: str | None):
    from pocket_audit.semisynthetic import SemiConfig, base
    key = (dataset, region_key)
    if key not in _BASE:
        q = REGIONS[dataset][region_key] if region_key else next(iter(REGIONS[dataset].values()))
        _BASE[key] = base(SemiConfig(config=DATASETS[dataset]["config"], region=q))
    return _BASE[key]


def pipeline_config(dataset: str, seed: int, n_boot: int, support: dict, design: bool = False, extra: bool = False,
                    max_features=None, leiden_auto=False, run_sbm=True):
    from pocket_audit.audit import SupportRule
    from pocket_audit.pipeline import PipelineConfig
    return PipelineConfig(seed=seed, n_jobs=1, alpha=DATASETS[dataset]["alpha"], n_boot=n_boot,
                          inference="wild_restricted_mammen", tails="symmetric", se_floor=True, restricted_loss=True,
                          abstain_quantile=.01, project_expected_events=True, min_projected_per_group=150,
                          min_graph_profiled_share=.8, sbm_tree_cut=run_sbm, support=SupportRule(**support),
                          select_on_audit_design=design, extra_baselines=extra, leiden_auto=leiden_auto,
                          run_sbm=run_sbm,
                          **({} if max_features is None else {"max_features": max_features}))


def run_one(task):
    opts, cell, rep = task
    dataset = opts["dataset"]
    dest = Path(opts["out_dir"]) / cell / f"rep_{rep:03d}"
    if (dest / "done.json").exists():
        old = json.loads((dest / "done.json").read_text()).get("opts_key")
        if old is not None and old != opts_key(opts):
            raise RuntimeError(f"{dest} was produced with different settings; use a new --out-dir")
        return cell, rep, "cached", 0.0
    tmp = dest.with_name(dest.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    try:
        import graph_tool.all as gt
        gt.openmp_set_num_threads(1)
        import smoke_ladder as S
        from pocket_audit.data import AuditData
        from pocket_audit.pipeline import assign_partition, audit, discover
        from pocket_audit.data import to_numeric_matrix
        from pocket_audit.semisynthetic import true_values_fixed
        region_key, delta = CELLS[cell]
        b = get_base(dataset, region_key)
        # replicate seed shared across cells: the same outcome draws and roles for every delta and region
        seed = int(np.random.SeedSequence([opts["base_seed"], rep]).generate_state(1)[0] % (2 ** 31))
        rng = np.random.default_rng(seed)
        p0 = b["p0"]
        rho = opts.get("hosp_rho", 0.0)
        if rho > 0:   # shared-uniform mixture within the dataset unit (hospital): keeps Pr(Y=1 | x) = p0 exactly
            codes = np.unique(np.asarray(b["data"].unit).astype(str), return_inverse=True)[1]
            shared_u = rng.random(codes.max() + 1)[codes]
            u = np.where(rng.random(len(p0)) < rho, shared_u, rng.random(len(p0)))
        else:
            u = rng.random(len(p0))
        y = (u < p0).astype(int)
        R = b["region"] if region_key else np.zeros(len(p0), bool)
        shift = delta * b["focal"] * R
        pf = np.where(shift == 0, p0, np.minimum(p0 + shift, np.maximum(p0, .995)))   # zero shifts preserved exactly
        real = b["data"]
        data = AuditData(name=f"{real.name}_semi", X=real.X, y=y, unit=real.unit, row_id=real.row_id,
                         protected=real.protected, contrasts=[b["contrast"]], primary_contrast=b["contrast"].name,
                         role_unit=real.row_id if opts.get("role_unit") == "row" else None)
        pc = pipeline_config(dataset, seed, opts["n_boot"], b["support"], opts.get("design", False),
                             opts.get("extra", False), opts.get("max_features"), opts.get("leiden_auto", False),
                             not opts.get("no_sbm", False))
        if opts.get("family"):
            return run_family(opts, cell, rep, seed, b, y, p0, data, pc, dest, tmp, t0)
        frozen = discover(data, pc, preds={"model": pf})
        t_disc = time.perf_counter() - t0
        res = audit(data, frozen)
        truth = true_values_fixed(frozen, b, pf)
        truth = truth[truth.kind == "delta"][["partition", "pocket", "truth", "exact_null"]].copy()
        truth["pocket"] = truth.pocket.astype(str)
        aud = np.flatnonzero(frozen["role"] == "audit")
        Xnum = to_numeric_matrix(real.X)
        labels = {n: assign_partition(frozen, n, aud, Xnum) for n in frozen["partitions"]}
        bias = (pf - p0)[aud]
        d_aud = {"s": b["focal"][aud], "bias": bias, "affected": b["focal"][aud] & (bias != 0), "unit": real.unit[aud]}
        coords = res["results"].copy()
        coords["pocket"] = coords.pocket.astype(str)
        rows, pockets = [], []
        for name, lab in labels.items():
            r, p = S.evaluate(name, lab, d_aud, coords, truth)
            rows.append(r)
            pockets.append(p)
        sbm = frozen["reports"].get("sbm", {})
        meta = dict(dataset=dataset, cell=cell, rep=rep, seed=seed, region=REGIONS[dataset].get(region_key),
                    delta=delta, sbm_level=sbm.get("chosen_level"), sbm_status=sbm.get("status"),
                    n_pockets=frozen["n_pockets"], timings=frozen["reports"]["timings"],
                    seconds_discovery=round(t_disc, 1), seconds=round(time.perf_counter() - t0, 1),
                    region_share=float(R.mean()), focal_region_share=float(R[b["focal"]].mean()))
        m = pd.DataFrame(rows)
        for k in ("rep", "cell", "dataset"):
            m.insert(0, k, meta[k])
        m.insert(3, "delta", delta)
        m.insert(4, "k_sbm", meta["n_pockets"])
        m.to_csv(tmp / "metrics.csv", index=False)
        pd.concat(pockets).to_csv(tmp / "pockets.csv", index=False)
        coords.to_csv(tmp / "coordinates.csv", index=False)
        res["q_curves"].to_csv(tmp / "q_curves.csv", index=False)
        truth.to_csv(tmp / "truth.csv", index=False)
        if "levels" in sbm:
            sbm["levels"].to_csv(tmp / "sbm_levels.csv", index=False)
        np.savez_compressed(tmp / "audit_inputs.npz", rows=aud, y=y[aud], pf=pf[aud], p0=p0[aud],
                            **{f"labels_{k}": v for k, v in labels.items()})
        meta["opts_key"] = opts_key(opts)
        meta["forest_oob"] = frozen["reports"].get("forest")    # discovery-only out-of-bag fit
        meta["leiden_auto"] = {k: v for k, v in frozen["reports"].get("leiden_auto", {}).items() if k != "levels"}
        (tmp / "done.json").write_text(json.dumps(meta, indent=1, default=str))
        tmp.rename(dest)
        return cell, rep, "completed", meta["seconds"]
    except Exception:
        (tmp / "failure.txt").write_text(traceback.format_exc())
        raise


def run_family(opts, cell, rep, seed, b, y, p0, data, pc, dest, tmp, t0):
    """Application-family null check (application-family null check): fitted LightGBM and logistic models (development
    rows, simulated outcomes), every real contrast, paired kappa, the secondary loss family, and hospital-shared
    outcome shocks. Each primary coordinate is scored against its exact audit-design truth:
    delta: mean(pred - p0 | pocket, focal) - mean(pred - p0 | pocket, reference) over valid audit rows;
    kappa: the same with pred1 - pred2. These need no outcomes because E[Y | x] = p0."""
    import dataclasses
    from pocket_audit.data import AuditData, to_numeric_matrix
    from pocket_audit.pipeline import assign_partition, audit, discover
    real = b["data"]
    data = AuditData(name=data.name, X=real.X, y=y, unit=real.unit, row_id=real.row_id, protected=real.protected,
                     contrasts=real.contrasts, primary_contrast=real.primary_contrast, role_unit=data.role_unit)
    pc = dataclasses.replace(pc, primary_models=["lightgbm", "logistic"], pairs=[("lightgbm", "logistic")])
    frozen = discover(data, pc)
    res = audit(data, frozen)
    aud = np.flatnonzero(frozen["role"] == "audit")
    Xnum = to_numeric_matrix(real.X)
    labels = {n: assign_partition(frozen, n, aud, Xnum) for n in frozen["partitions"]}
    preds = {k: np.asarray(v, float)[aud] for k, v in frozen["preds"].items()}
    prot = real.protected.iloc[aud].reset_index(drop=True)
    coords = res["results"].copy()
    coords["pocket"] = coords.pocket.astype(str)
    truth = np.full(len(coords), np.nan)
    for i, r in enumerate(coords.itertuples()):
        if r.family != "primary" or r.kind not in ("delta", "kappa"):
            continue
        con = next(c for c in real.contrasts if c.name == r.contrast)
        attr = prot[con.attribute].to_numpy()
        lab = labels[r.partition]
        inp = np.ones(len(aud), bool) if r.pocket == "all" else lab == int(r.pocket)
        if r.kind == "delta":
            x = preds[r.model] - p0[aud]
        else:
            m1, m2 = r.model.split("-", 1)
            x = preds[m1] - preds[m2]
        ok = inp & np.isfinite(x)
        f, g = ok & (attr == con.focal), ok & (attr == con.reference)
        if f.any() and g.any():
            truth[i] = x[f].mean() - x[g].mean()
    coords["truth"] = truth
    rows = []
    for part, g in coords[(coords.family == "primary") & coords.kind.isin(["delta", "kappa"])].groupby("partition"):
        t = g[(g.status == "tested") & g.truth.notna()]
        cov = (t.ci_low <= t.truth) & (t.truth <= t.ci_high)
        near0 = t.truth.abs() < .005
        rows.append(dict(dataset=opts["dataset"], cell=cell, rep=rep, partition=part, n_tested=len(t),
                         n_local_tested=int((t.pocket != "all").sum()),
                         all_covered=bool(cov.all()) if len(t) else np.nan, n_miss=int((~cov).sum()),
                         miss_up=int((t.ci_low > t.truth).sum()), miss_down=int((t.ci_high < t.truth).sum()),
                         any_flag=bool(t.flag.any()), any_flag_near_null=bool((t.flag & near0).any()),
                         n_near_null=int(near0.sum()), max_abs_truth=float(t.truth.abs().max()) if len(t) else np.nan,
                         K=frozen["n_pockets"], q=res["summary"][part]["primary"].get("q")))
    pd.DataFrame(rows).to_csv(tmp / "metrics.csv", index=False)
    coords.to_csv(tmp / "coordinates.csv", index=False)
    np.savez_compressed(tmp / "audit_inputs.npz", rows=aud, y=y[aud], p0=p0[aud],
                        **{f"pred_{k}": v for k, v in preds.items()}, **{f"labels_{k}": v for k, v in labels.items()})
    meta = dict(dataset=opts["dataset"], cell=cell, rep=rep, seed=seed, mode="family", hosp_rho=opts.get("hosp_rho", 0.0),
                role_unit=opts.get("role_unit"), timings=frozen["reports"]["timings"],
                seconds=round(time.perf_counter() - t0, 1), n_pockets=frozen["n_pockets"],
                sbm_level=frozen["reports"].get("sbm", {}).get("chosen_level"), opts_key=opts_key(opts))
    (tmp / "done.json").write_text(json.dumps(meta, indent=1, default=str))
    tmp.rename(dest)
    return cell, rep, "completed", meta["seconds"]


def opts_key(opts):
    """Identity of a run's scientific settings, checked on resume."""
    import hashlib
    keep = {k: opts.get(k) for k in ("dataset", "n_boot", "base_seed", "design", "role_unit", "extra", "family", "hosp_rho")}
    if opts.get("max_features") is not None:     # sensitivity runs only; default runs keep their old keys
        keep["max_features"] = opts["max_features"]
    if opts.get("leiden_auto"):
        keep["leiden_auto"] = True
    if opts.get("no_sbm"):
        keep["no_sbm"] = True
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", choices=list(CELLS), required=True)
    ap.add_argument("--reps", type=int, nargs=2, default=[0, 10])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1999)
    ap.add_argument("--base-seed", type=int, default=2026100560)
    ap.add_argument("--extra-baselines", action="store_true", help="also cem_top and disparity_bins")
    ap.add_argument("--max-features", type=lambda v: v if v in ("sqrt", "log2") else float(v), default=None,
                    help="sensitivity override of the similarity forest's max_features (default: PipelineConfig)")
    ap.add_argument("--leiden-auto", action="store_true",
                    help="proposal: also freeze leiden_auto (Leiden at the support-rule resolution of a fixed grid)")
    ap.add_argument("--no-sbm", action="store_true",
                    help="skip the nested SBM (and its tree cut); comparators then use K = 4. For leiden_auto smoke runs")
    ap.add_argument("--hosp-rho", type=float, default=0.0, help="shared-uniform outcome dependence within unit")
    ap.add_argument("--family", action="store_true", help="fitted application family (null check, audit-design truth)")
    ap.add_argument("--role-unit", choices=["unit", "row"], default="unit",
                    help="row: roles by encounter, inference still clustered by the dataset unit (hospital)")
    ap.add_argument("--select-on-audit-design", action="store_true",
                    help="v2 proposal: resolution from the audit rows' exact support (outcome-blind)")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    prov = S_prov(args)
    (out / f"provenance_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(prov, indent=1, default=str))
    opts = dict(dataset=args.dataset, out_dir=str(out), n_boot=args.n_boot, base_seed=args.base_seed,
                design=args.select_on_audit_design, role_unit=args.role_unit, extra=args.extra_baselines,
                hosp_rho=args.hosp_rho, family=args.family, max_features=args.max_features,
                leiden_auto=args.leiden_auto, no_sbm=args.no_sbm)
    tasks = [(opts, c, r) for r in range(*args.reps) for c in args.cells]
    failures = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                print(*fut.result(), flush=True)
            except Exception as e:
                failures.append(dict(task=[str(x) for x in futs[fut][1:]], error=repr(e)))
                print("FAILED", futs[fut][1:], repr(e), flush=True)
    (out / "failures.json").write_text(json.dumps(failures, indent=1))
    sys.exit(1 if failures else 0)


def S_prov(args):
    import hashlib
    import subprocess
    h = hashlib.sha256()
    for p in sorted((ROOT / "src/pocket_audit").glob("*.py")) + [Path(__file__).resolve()]:
        h.update(p.read_bytes())
    return dict(command=sys.argv, args={k: str(v) for k, v in vars(args).items()}, regions=REGIONS, cells=CELLS,
                datasets=DATASETS, source_sha256=h.hexdigest(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())


if __name__ == "__main__":
    main()
