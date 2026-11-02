"""Chest X-ray application: a chest X-ray AI model on NIH ChestX-ray14, audited for Effusion (supplement application).

    python applications/chest_xray/inference.py ...    (imaging env; outputs and features, .npz)
    python applications/chest_xray/audit_cxr.py discover --run-dir results/chest_xray
    python applications/chest_xray/audit_cxr.py describe --run-dir results/chest_xray
    python applications/chest_xray/audit_cxr.py validate --run-dir results/chest_xray
    python applications/chest_xray/audit_cxr.py audit    --run-dir results/chest_xray

Images: each patient's first image (lowest follow-up number), so the 30,805 patients are independent units.
Label boundary: before ``audit``, the Effusion label is parsed only for development and discovery images; audit images
carry a placeholder 0 that no stage uses (``discover`` fits on development and discovery rows; ``validate`` draws
synthetic labels for the audit images). ``audit`` reserves its single run with an exclusive lock before parsing any
audit label, reads the gate result, and reports descriptive estimates only if the gate failed.
Settings: applications/chest_xray/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
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

from pocket_audit.assignment import ABSTAIN  # noqa: E402
from pocket_audit.audit import AuditConfig, SupportRule, run_audit  # noqa: E402
from pocket_audit.data import AuditData, Contrast, to_numeric_matrix  # noqa: E402
from pocket_audit.pipeline import PipelineConfig, assign_partition, audit, config_dict, discover  # noqa: E402
from pocket_audit.splits import allocate_units  # noqa: E402

DATA = Path(os.environ.get("NIH_CXR14_DIR", ROOT / "data" / "nih_cxr14"))
META = DATA / "raw/Data_Entry_2017_v2020.csv"
OUTPUTS = DATA / "derived/cxr_outputs.npz"
FEATURES = DATA / "derived/cxr_pc_features.npz"
WEIGHTS = Path.home() / ".torchxrayvision/models_data"
SEED = 20261111
MODELS = ("chex", "mimic")
PAIRS = [("chex", "mimic")]
PRIMARY_MAPS = ("sbm", "ridge_kmeans")
ALPHA_PER_MAP = 0.025
N_PC = 32
CLIP = (0.001, 0.999)
SUPPORT = SupportRule(min_n=100, min_units=30, min_effective_units=20, max_unit_share=0.10, min_expected=10.0)
CONTRAST = Contrast("female_vs_male", "sex", "female", "male")
WEAK_A = 0.5                  # gate: weak-null amplitude, p + a p(1-p)(z - w_c)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _roles(unit: np.ndarray) -> np.ndarray:
    return allocate_units(unit, (0.4, 0.3, 0.3), seed=SEED)          # the call discover() makes


def _npz(path: Path, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if not np.array_equal(z["image"], names):
        raise RuntimeError(f"{path}: image order differs from the metadata")
    return z["columns"], z["values"]


def load(stage: str, transform: dict | None = None) -> tuple[AuditData, pd.DataFrame, dict]:
    """Data for one stage: each patient's first image (lowest follow-up number), so patients are independent units.
    Labels are parsed for audit images only when stage == 'audit'."""
    full = pd.read_csv(META)
    all_names = full["Image Index"].to_numpy().astype(str)
    assert len(full) == 112120 and len(np.unique(all_names)) == len(all_names)
    keep = full.groupby("Patient ID")["Follow-up #"].transform("min") == full["Follow-up #"]
    keep &= ~full.duplicated(["Patient ID", "Follow-up #"])          # ties (none expected) keep the first row
    meta = full[keep].reset_index(drop=True)
    names = meta["Image Index"].to_numpy().astype(str)
    assert len(meta) == full["Patient ID"].nunique()
    unit = meta["Patient ID"].astype(str).to_numpy()
    role = _roles(unit)
    cols, vals = _npz(OUTPUTS, all_names)
    vals = vals[keep.to_numpy()]
    logits = {m: vals[:, list(cols).index(f"logit_{m}")].astype(float) for m in MODELS}
    assert all(np.isfinite(v).all() for v in logits.values())
    disc = role == "discovery"
    if transform is None:                                     # discover: fit on discovery images only
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        _, F = _npz(FEATURES, all_names)
        F = F[keep.to_numpy()]
        assert F.shape[1] == 1024 and np.isfinite(F).all()
        F = F.astype(np.float64)
        sc = StandardScaler().fit(F[disc])
        pca = PCA(N_PC, random_state=SEED).fit(sc.transform(F[disc]))
        transform = {"scaler": sc, "pca": pca}
    else:
        _, F = _npz(FEATURES, all_names)
        F = F[keep.to_numpy()].astype(np.float64)
    Z = transform["pca"].transform(transform["scaler"].transform(F))
    X = pd.DataFrame(Z, columns=[f"img_pc{j:02d}" for j in range(N_PC)])
    X.insert(0, "age", meta["Patient Age"].astype(float).to_numpy())
    X.insert(1, "view_ap", (meta["View Position"] == "AP").astype(float).to_numpy())
    readable = np.ones(len(meta), bool) if stage == "audit" else role != "audit"
    y = np.zeros(len(meta))                                   # placeholder for unreadable (audit) labels
    y[readable] = meta.loc[readable, "Finding Labels"].str.split("|").map(lambda s: "Effusion" in s).astype(float)
    sex = np.where(meta["Patient Gender"].to_numpy() == "F", "female", "male")
    data = AuditData(name="nih_cxr14", X=X, y=y, unit=unit, row_id=np.arange(len(meta)),
                     protected=pd.DataFrame({"sex": sex}), contrasts=[CONTRAST], primary_contrast=CONTRAST.name,
                     outcome="binary",
                     log={"n_images": len(meta), "n_patients": int(len(np.unique(unit))), "image_rule": "first per patient",
                          "labels_read": stage == "audit" and "all" or "development and discovery only",
                          "image_order_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                          "pca_explained": float(transform["pca"].explained_variance_ratio_.sum())})
    df = pd.DataFrame({"image": names, "followup": meta["Follow-up #"].to_numpy(), "role": role,
                       **{f"logit_{m}": v for m, v in logits.items()}})
    return data, df, transform


def pipeline_config(n_jobs: int) -> PipelineConfig:
    return PipelineConfig(
        seed=SEED, n_jobs=n_jobs, alpha=ALPHA_PER_MAP, n_trees=200, min_samples_leaf=20, max_features="sqrt", k=30,
        sbm_n_init=3, min_pocket_size=50, min_projected_per_group=150, project_expected_events=True,
        min_graph_profiled_share=0.8, abstain_quantile=0.01, sbm_tree_cut=True, run_sbm=True, run_leiden=True,
        run_baselines=True, support=SUPPORT, pairs=PAIRS, primary_models=list(MODELS), n_boot=4999,
        inference="wild_restricted_mammen", tails="symmetric", se_floor=True, outcome="binary", include_mu=True)


def fit_models(df: pd.DataFrame, y: np.ndarray, dev: np.ndarray) -> tuple[dict, dict, dict]:
    """Isotonic maps of each logit to the Effusion label on development images only."""
    from scipy.special import expit
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    preds, isos, diag = {}, {}, {"n_dev_images": int(dev.sum()), "dev_prevalence": float(y[dev].mean())}
    for m in MODELS:
        z = df[f"logit_{m}"].to_numpy(float)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=CLIP[0], y_max=CLIP[1]).fit(z[dev], y[dev])
        preds[m], isos[m] = np.clip(iso.predict(z), *CLIP), iso
        diag[m] = {"raw_mean_dev": float(expit(z[dev]).mean()), "remapped_mean_dev": float(preds[m][dev].mean()),
                   "auc_raw_dev": float(roc_auc_score(y[dev], z[dev])),
                   "auc_remapped_dev": float(roc_auc_score(y[dev], preds[m][dev])),
                   "isotonic_steps": int(len(np.unique(preds[m][dev]))),
                   "share_at_clip": float(np.isin(preds[m][dev], CLIP).mean())}
    return preds, isos, diag


def _git() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "src", "code"))}


def _provenance() -> dict:
    out = {"metadata": sha256(META), "outputs_npz": sha256(OUTPUTS), "features_npz": sha256(FEATURES)}
    for f in sorted(WEIGHTS.glob("*.pt")):
        out[f"weights/{f.name}"] = sha256(f)
    for js in (OUTPUTS.with_suffix(".json"), FEATURES.with_suffix(".json")):
        if js.exists():
            out[f"inference/{js.name}"] = json.loads(js.read_text())
    return out


def cmd_discover(args):
    run = Path(args.run_dir)
    if (run / "frozen.pkl").exists():
        raise SystemExit(f"{run} already holds a frozen discovery")
    run.mkdir(parents=True, exist_ok=True)
    (run / ".gitignore").write_text("frozen.pkl\n*.npz\nroles.csv.gz\naudit_assignments.csv.gz\n")
    data, df, transform = load("discover")
    cfg = pipeline_config(args.n_jobs)
    role = df.role.to_numpy()
    t0 = time.time()
    preds, isos, diag = fit_models(df, data.y, role == "development")
    frozen = discover(data, cfg, preds=preds)
    if not np.array_equal(frozen["role"], role):
        raise RuntimeError("role allocation differs from discover()")
    with open(run / "frozen.pkl", "wb") as f:
        pickle.dump({"frozen": frozen, "diag": diag, "transform": transform, "isotonic": isos,
                     "image_order_sha256": data.log["image_order_sha256"]}, f)
    pd.DataFrame({"row_id": data.row_id, "image": df.image, "patient": data.unit, "role": role}).to_csv(
        run / "roles.csv.gz", index=False)
    rep = frozen["reports"]
    rep["sbm"]["levels"].to_csv(run / "sbm_levels.csv", index=False)
    import graph_tool, sklearn  # noqa: F401
    manifest = {"stage": "discover", "argv": sys.argv, "git": _git(), "dataset": data.log, "inputs": _provenance(),
                "config": config_dict(cfg), "models": diag, "roles_images": pd.Series(role).value_counts().to_dict(),
                "roles_patients": pd.Series(role).groupby(data.unit).first().value_counts().to_dict(),
                "n_pockets": frozen["n_pockets"], "chosen_level": rep["sbm"]["chosen_level"],
                "sbm_status": rep["sbm"]["status"], "forest": rep["forest"], "graph": rep["graph"],
                "timings": rep["timings"], "seconds": round(time.time() - t0),
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                             "sklearn": sklearn.__version__, "graph_tool": graph_tool.__version__.split()[0]}}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("models", "n_pockets", "chosen_level", "sbm_status", "forest",
                                               "timings", "roles_patients")}, indent=2, default=str))
    print(rep["sbm"]["levels"].to_string(index=False))


def _load_frozen(run: Path, stage: str):
    frozen_inputs = json.loads((run / "manifest.json").read_text())["inputs"]
    now = _provenance()                                       # checked before any data are loaded
    changed = [k for k, v in frozen_inputs.items() if not k.startswith("inference/") and now.get(k) != v]
    if changed:
        raise RuntimeError(f"inputs differ from the frozen run: {changed}")
    with open(run / "frozen.pkl", "rb") as f:
        saved = pickle.load(f)
    data, df, _ = load(stage, transform=saved["transform"])
    if data.log["image_order_sha256"] != saved["image_order_sha256"]:
        raise RuntimeError("image order differs from the frozen run")
    if not np.array_equal(saved["frozen"]["role"], df.role.to_numpy()):
        raise RuntimeError("roles differ from the frozen run")
    return saved, data, df


def cmd_describe(args):
    """Pocket profiles on DISCOVERY images only."""
    run = Path(args.run_dir)
    saved, data, df = _load_frozen(run, "describe")
    frozen = saved["frozen"]
    disc = np.flatnonzero(frozen["role"] == "discovery")
    Xnum = to_numeric_matrix(data.X)
    out = []
    for name in ("sbm", "ridge_kmeans", "sbm_cut"):
        lab = assign_partition(frozen, name, disc, Xnum)
        for c in sorted(set(lab)):
            rows = disc[lab == c]
            out.append({"partition": name, "pocket": int(c), "n_images": len(rows),
                        "n_patients": int(len(np.unique(data.unit[rows]))),
                        "share_female": float((data.protected.sex.iloc[rows] == "female").mean()),
                        "mean_age": float(data.X.age.iloc[rows].mean()),
                        "share_ap": float(data.X.view_ap.iloc[rows].mean()),

                        "prevalence_discovery": float(data.y[rows].mean()),
                        **{f"pred_{k}": float(np.asarray(frozen['preds'][k])[rows].mean()) for k in MODELS}})
    res = pd.DataFrame(out)
    res.to_csv(run / "pocket_profiles_discovery.csv", index=False)
    print(res.round(3).to_string(index=False))


def _truths(preds: dict, mean_y: np.ndarray, pocket: np.ndarray, sex: np.ndarray) -> dict:
    t = {}
    for c in ["all"] + sorted(int(k) for k in np.unique(pocket) if k != ABSTAIN):
        cell = np.ones(len(mean_y), bool) if c == "all" else pocket == c
        for m in MODELS:
            e = preds[m] - mean_y
            t[("mu", m, "", c)] = float(e[cell].mean())
            f, r = cell & (sex == "female"), cell & (sex == "male")
            if f.any() and r.any():
                t[("delta", m, CONTRAST.name, c)] = float(e[f].mean() - e[r].mean())
    return t


SCENARIOS = ("independent", "effect", "weak_sbm", "weak_ridge_kmeans")


def _replicate(args_tuple):
    scenario, gen, rep, ctx = args_tuple
    rng = np.random.default_rng([SEED, SCENARIOS.index(scenario), MODELS.index(gen), rep])
    p = ctx["preds"][gen].copy()
    maps = PRIMARY_MAPS
    if scenario == "effect":
        p = np.where(ctx["ap"], np.maximum(p - 0.05, 0.0), p)
    if scenario.startswith("weak_"):                  # errors vary within pockets, every pocket mean of p - p_gen = 0
        mp_ = scenario[5:]
        maps = (mp_,)
        lab, z = ctx["pockets"][mp_], ctx["z"]
        v = p * (1 - p)
        for c in np.unique(lab):
            m = lab == c
            w = (v[m] * z[m]).sum() / v[m].sum()
            p[m] = p[m] + WEAK_A * v[m] * (z[m] - w)
        assert max(abs((ctx["preds"][gen] - p)[lab == c].mean()) for c in np.unique(lab)) < 1e-12
    y = (rng.random(len(p)) < p).astype(float)       # one image per patient: independent units
    rows = []
    for mp in maps:
        lab = ctx["pockets"][mp]
        acfg = AuditConfig(support=SUPPORT, pairs=PAIRS, primary_models=list(MODELS), n_boot=ctx["n_boot"],
                           alpha=ALPHA_PER_MAP, seed=11 + rep, method="wild_restricted_mammen", outcome="binary",
                           se_floor=True, include_losses=False, include_mu=True, engine="gemm")
        table, _ = run_audit(y, ctx["preds"], lab, ctx["unit"], ctx["prot"], [CONTRAST], acfg)
        truth = _truths(ctx["preds"], p, lab, ctx["prot"].sex.to_numpy())
        for r in table.itertuples():
            if r.kind == "kappa":                       # outcome-free under conditional resampling
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
    saved, data, df = _load_frozen(run, "validate")
    frozen = saved["frozen"]
    rows = np.flatnonzero(frozen["role"] == "audit")
    dev = np.flatnonzero(frozen["role"] == "development")
    Xnum = to_numeric_matrix(data.X)
    preds_all = {m: np.asarray(frozen["preds"][m], float) for m in MODELS}
    upat, inv = np.unique(np.asarray(data.unit)[rows], return_inverse=True)
    zpat = np.random.default_rng([SEED, 99]).choice([-1.0, 1.0], len(upat))[inv]   # a patient-level coin
    ctx = {"preds": {m: preds_all[m][rows] for m in MODELS},
           "pockets": {mp: assign_partition(frozen, mp, rows, Xnum) for mp in PRIMARY_MAPS},
           "unit": np.asarray(data.unit)[rows], "prot": data.protected.iloc[rows].reset_index(drop=True),
           "ap": data.X.view_ap.to_numpy()[rows] == 1, "z": zpat,
           "n_boot": args.n_boot}
    jobs = [("independent", g, r, ctx) for g in MODELS for r in range(args.reps)]
    jobs += [(sc, "chex", r, ctx) for sc in ("effect", "weak_sbm", "weak_ridge_kmeans") for r in range(args.reps)]
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, batch_size=2, verbose=5)(delayed(_replicate)(j) for j in jobs)
    out = pd.DataFrame([x for r in res for x in r])
    out.to_csv(run / "validation_coordinates.csv.gz", index=False)
    out["tested"] = out.status == "tested"
    out["null"] = out.truth.abs() < 1e-12
    out["covered"] = ~out.tested | ((out.ci_low <= out.truth) & (out.truth <= out.ci_high))
    out["local_mu"] = out.tested & (out.kind == "mu") & (out.pocket.astype(str) != "all")
    pockets_tested = out.assign(lp=out.pocket.where(out.local_mu)).groupby(
        ["scenario", "gen_model", "rep", "map"]).lp.nunique()          # distinct tested local pockets; 0 kept
    out["halfwidth"] = (out.ci_high - out.ci_low) / 2
    summ = []
    for (sc, gm), g in out.groupby(["scenario", "gen_model"]):
        maps = sorted(g["map"].unique())
        for mp in maps + (["combined"] if len(maps) == 2 else []):
            gg = g if mp == "combined" else g[g["map"] == mp]
            per = gg.groupby("rep")
            anyf = per.apply(lambda d: bool((d.flag & d.null & d.tested).any()), include_groups=False)
            n = len(anyf)
            lo, hi = proportion_confint(int(anyf.sum()), n, method="wilson")
            nn = gg[gg.tested & ~gg.null & (gg.kind == "mu") & (gg.model == gm)]
            summ.append(dict(scenario=sc, gen_model=gm, map=mp, replicates=n, fwer=anyf.mean(), fwer_lo=lo,
                             fwer_hi=hi, simultaneous_coverage=per.covered.all().mean(),
                             coordinate_coverage=gg[gg.tested].covered.mean(),
                             min_local_pockets_tested=int(pockets_tested.loc[sc, gm].groupby("rep").min().min()
                                                          if mp == "combined" else
                                                          pockets_tested.loc[sc, gm].xs(mp, level="map").min()),
                             null_coords_per_rep=(gg.null & gg.tested).sum() / n, tested_per_rep=gg.tested.sum() / n,
                             gen_mu_detection=nn.flag.mean() if len(nn) else np.nan,
                             gen_mu_abs_truth=nn.truth.abs().mean() if len(nn) else np.nan,
                             gen_mu_halfwidth=nn.halfwidth.mean() if len(nn) else np.nan))
    summ = pd.DataFrame(summ)
    summ.to_csv(run / "validation_summary.csv", index=False)
    planned = {"independent": 2, "effect": 1, "weak_sbm": 1, "weak_ridge_kmeans": 1}
    complete = all(len(out[(out.scenario == sc)].groupby(["gen_model", "rep"])) == k * args.reps
                   for sc, k in planned.items())
    nulls = summ[summ.scenario == "independent"]
    gate = {"complete": bool(complete),
            "combined_fwer_max": float(nulls[nulls["map"] == "combined"].fwer.max()),
            "weak_null_per_map_fwer_max": float(summ[summ.scenario.str.startswith("weak")].fwer.max()),
            "per_map_fwer_max (diagnostic)": float(nulls[nulls["map"] != "combined"].fwer.max()),
            "coverage_min": float(summ[summ["map"] == "combined"].simultaneous_coverage.min()),
            "coverage_min_weak": float(summ[summ.scenario.str.startswith("weak")].simultaneous_coverage.min()),
            "min_local_pockets_tested_per_map": int(summ[summ["map"] != "combined"].min_local_pockets_tested.min()),
            "protocol_settings": bool(args.reps >= 300 and args.n_boot == 4999)}
    gate["passed"] = bool(gate["complete"] and gate["combined_fwer_max"] <= 0.05
                          and gate["weak_null_per_map_fwer_max"] <= 0.05 and gate["coverage_min"] >= 0.93
                          and gate["coverage_min_weak"] >= 0.93 and gate["min_local_pockets_tested_per_map"] >= 2
                          and gate["protocol_settings"])
    (run / "validation_manifest.json").write_text(json.dumps(
        {"stage": "validate", "argv": sys.argv, "git": _git(), "reps": args.reps, "n_boot": args.n_boot,
         "weak_a": WEAK_A, "gate": gate, "status_counts": out.status.value_counts().to_dict(),
         "seconds": round(time.time() - t0)}, indent=2, default=str))
    print(summ.round(3).to_string(index=False))
    print(json.dumps({"gate": gate}, indent=1))


def _cell_units(unit: np.ndarray) -> tuple[int, float]:
    _, n = np.unique(unit, return_counts=True)
    return len(n), float(n.sum() ** 2 / (n ** 2).sum()) if len(n) else 0.0


def readout(run: Path, data: AuditData, frozen: dict, res: pd.DataFrame, assign: pd.DataFrame, descriptive: bool):
    rows = np.flatnonzero(frozen["role"] == "audit")
    sex = data.protected.sex.to_numpy()[rows]
    unit = np.asarray(data.unit)[rows]
    groups = ("female", "male")
    out = {"descriptive_only": descriptive, "n_audit_images": len(rows),
           "n_audit_patients": int(len(np.unique(unit))),
           "prevalence": float(data.y[rows].mean()),
           "mean_pred": {m: float(np.asarray(frozen["preds"][m])[rows].mean()) for m in MODELS}}
    cells = []
    for mp in PRIMARY_MAPS:
        lab = assign[mp].to_numpy()
        out[f"abstained_{mp}"] = {g: float((lab[sex == g] == ABSTAIN).mean()) for g in groups}
        for (kind, model), t in res[(res.partition == mp) & res.kind.isin(["mu", "delta", "kappa"])
                                    & (res.pocket.astype(str) != "all")].groupby(["kind", "model"]):
            tested = t[t.status == "tested"].pocket.astype(int).to_numpy()
            out[f"untested_{kind}_{mp}_{model}"] = {g: float((~np.isin(lab[sex == g], tested)).mean()) for g in groups}
        for c in ["all"] + sorted(set(lab) - {ABSTAIN}):
            for g in groups + ("all",):
                msk = (np.ones(len(lab), bool) if c == "all" else lab == c) & ((sex == g) if g != "all" else True)
                nu, ne = _cell_units(unit[msk])
                cells.append(dict(map=mp, pocket=c, group=g, images=int(msk.sum()), patients=nu,
                                  effective_patients=round(ne, 1), max_patient_share=float(
                                      np.unique(unit[msk], return_counts=True)[1].max() / msk.sum()) if msk.any() else np.nan))
    pd.DataFrame(cells).to_csv(run / "audit_cells.csv", index=False)
    if not descriptive:                                # combined audit: directional shares by sex, conflicts by sex
        for m in MODELS:
            dirs = []
            for mp in PRIMARY_MAPS:
                t = res[(res.partition == mp) & (res.kind == "mu") & (res.model == m) & (res.status == "tested")
                        & (res.pocket.astype(str) != "all")]
                d = np.zeros(len(rows), int)
                lab = assign[mp].to_numpy()
                for r in t.itertuples():
                    d[lab == int(r.pocket)] = int(r.ci_low > 0) - int(r.ci_high < 0)
                dirs.append(d)
            pos = (dirs[0] > 0) | (dirs[1] > 0)
            neg = (dirs[0] < 0) | (dirs[1] < 0)
            out[f"combined_mu_{m}"] = {k: {g: float(v[sex == g].mean()) for g in groups}
                                       for k, v in (("over", pos & ~neg), ("under", neg & ~pos), ("conflict", pos & neg))}
    (run / "readout.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


def cmd_audit(args):
    run = Path(args.run_dir)
    lock = run / "AUDIT_RESERVED"
    vm = run / "validation_manifest.json"
    if not vm.exists():
        raise SystemExit("run the validation gate first")
    vman = json.loads(vm.read_text())
    gate = vman["gate"]
    if vman["reps"] < 300 or vman["n_boot"] != 4999:
        raise SystemExit("the gate did not run with the protocol settings (300 replicates, B = 4,999)")
    descriptive = not gate["passed"]
    try:                                               # reserve the single audit run before any audit label is parsed
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{time.strftime('%Y%m%dT%H%M%S')} pid {os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit("audit already reserved or run; the protocol allows one run")
    saved, data, df = _load_frozen(run, "audit")
    frozen = saved["frozen"]
    t0 = time.time()
    out = audit(data, frozen)
    res = out["results"]
    readout(run, data, frozen, res, out["assignments"], descriptive)
    if descriptive:                                    # failed gate: estimates only, no inferential output kept
        res = res.drop(columns=[c for c in ("se", "t", "ci_low", "ci_high", "p_adj", "flag", "flag_positive",
                                            "flag_negative") if c in res.columns])
    res.to_csv(run / "audit_results.csv", index=False)
    out["coverage"].to_csv(run / "assignment_coverage.csv", index=False)
    out["assignments"].to_csv(run / "audit_assignments.csv.gz", index=False)
    (run / "audit_manifest.json").write_text(json.dumps(
        {"stage": "audit", "argv": sys.argv, "git": _git(), "gate_passed": gate["passed"],
         "descriptive_only": descriptive, "seconds": round(time.time() - t0)}, indent=2, default=str))
    show = res[res.partition.isin(PRIMARY_MAPS) & res.kind.isin(["mu", "delta", "kappa"])]
    cols = ["partition", "kind", "model", "contrast", "pocket", "status", "n_focal", "n_reference", "estimate"]
    print(show[cols + ([] if descriptive else ["ci_low", "ci_high", "flag"])].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "describe", "validate", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        if name == "discover":
            p.add_argument("--n-jobs", type=int, default=16)
        if name == "validate":
            p.add_argument("--reps", type=int, default=300)
            p.add_argument("--n-boot", type=int, default=4999)
            p.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args(argv)
    {"discover": cmd_discover, "describe": cmd_describe, "validate": cmd_validate, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
