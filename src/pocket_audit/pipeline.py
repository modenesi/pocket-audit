"""End-to-end workflow in two stages that keep audit outcomes out of every choice.

``discover``: roles -> assessed models (development) -> similarity forest, sparse graph,
SBM hierarchy, resolution choice, assignment thresholds, Leiden and baseline
partitions (discovery only). Its outputs are frozen before ``audit`` runs.

``audit``: assign audit rows through each frozen map and run simultaneous inference.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, log_loss, roc_auc_score

from . import graph as G
from .assignment import ABSTAIN, build_pocket_map, calibrate_thresholds, self_excluded_agreement
from .audit import AuditConfig, SupportRule, coverage_by_group, run_audit
from .baselines import (CoarsenedExactPockets, KMeansPockets, MahalanobisKMeansPockets, RidgeKMeansPockets,
                        RidgeRegressionKMeansPockets, RiskBins)
from .data import AuditData, to_numeric_matrix
from .models import fit_assessed_models, predict
from .pockets import leiden_partition, sbm_hierarchy
from .splits import allocate_units, check_roles


@dataclass
class PipelineConfig:
    seed: int = 20260928
    fractions: tuple = (0.4, 0.3, 0.3)
    graph_fraction: float = 0.8          # discovery units used for the graph; rest calibrate assignment
    n_jobs: int = 1
    alpha: float = 0.05
    n_trees: int = 200
    min_samples_leaf: int = 20
    max_features: str | float = "sqrt"   # compared with 0.5 and 1.0 in the graded-relevance simulation
    k: int = 30
    b: int = 8
    q: int = 256
    graph_method: str = "approx"         # "approx" (CPU candidate search), "exact_cpu" (n <= 5000), "exact_gpu"
    sbm_n_init: int = 3
    sbm_weighting: str = "none"        # see pockets.py for why counts are not multiplicities
    min_pocket_size: int = 50            # graph rows; smaller pockets get no profile
    min_projected_per_group: int = 150   # resolution rule, projected audit count per primary group
    min_group_coverage: float = 0.0      # optional coverage floor; report support gaps by default
    min_graph_profiled_share: float | None = None  # proposal (off by default): graph rows in profiled blocks
    sbm_tree_cut: bool = False            # proposal (off by default): also freeze "sbm_cut", the cut of the SBM
                                           # tree with the most projected-supported pockets (see tree_cut)
    project_expected_events: bool = False  # proposal (off by default): projected support also needs the audit's
                                           # expected events/non-events per group, from predictions (never outcomes)
    abstain_quantile: float = 0.05
    run_sbm: bool = True
    run_leiden: bool = True
    run_baselines: bool = True
    cem_bins: int = 3                    # quantile bins per feature for coarsened exact matching
    support: SupportRule = field(default_factory=SupportRule)
    pairs: list = field(default_factory=list)
    primary_models: list | None = None
    n_boot: int = 4999
    inference: str = "wild_restricted"   # audit.METHODS; "gauss" is the original multiplier max-t
    tails: str = "symmetric"             # audit.AuditConfig.tails: "symmetric" (default) or "equal"
    restricted_loss: bool = False        # audit.AuditConfig.restricted_loss
    se_floor: bool = False               # audit.AuditConfig.se_floor
    extra_baselines: bool = False        # also freeze cem_top (CEM on the 3 most relevant features) and
                                           # disparity_bins (T-learner on residuals; uses the protected attribute)
    select_on_audit_design: bool = False  # proposal (v2): resolution from the audit rows' exact SupportRule
                                           # (features, groups, units, predictions; never outcomes)
    leiden_auto: bool = False            # proposal (2026-10-07, off by default): also freeze "leiden_auto", Leiden
                                           # at the grid resolution chosen by the SBM's discovery-only support rule
    leiden_auto_n_init: int = 3
    outcome: str = "binary"              # "continuous": regression similarity forest, ridge-regression map,
                                           # no expected-events rule, no Brier loss family (v2, 2026-10-31)
    include_mu: bool = False             # audit mu_c (pocket-average signed error) in the primary family


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.t[name] = round(time.perf_counter() - self.s, 3)
        return _C()


def peak_rss_mb() -> float:
    try:
        import resource              # Unix only; Windows can still import the package
    except ImportError:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _projected_support(lab_calib, groups_calib, focal, ref, K, scale, min_count, probs=None, min_expected=0.0):
    """Projected audit counts per pocket for the primary contrast groups.

    With ``probs`` (prediction arrays on the calibration rows, one per model), a pocket also needs
    ``min_expected`` projected expected events and non-events per group under every model, mirroring
    the audit's ``SupportRule.min_expected``. Predictions only; outcomes are never used."""
    rows = []
    for c in range(K):
        in_c = lab_calib == c
        g1, g0 = in_c & (groups_calib == focal), in_c & (groups_calib == ref)
        n1, n0 = scale * np.sum(g1), scale * np.sum(g0)
        e = np.inf
        for p in probs or []:
            for g in (g1, g0):
                e = min(e, scale * np.nansum(p[g]), scale * np.nansum(1 - p[g]))
        rows.append((c, n1, n0, e, (n1 >= min_count) and (n0 >= min_count) and e >= min_expected))
    return pd.DataFrame(rows, columns=["pocket", "proj_focal", "proj_reference", "proj_min_expected", "supported"])


def _design_support(lab_aud, groups_aud, unit_aud, focal, ref, K, rule, probs=None):
    """Exact audit-design support per pocket for the primary contrast: the audit's own SupportRule applied to
    the audit rows' feature-only pocket assignments, protected groups, units, and model predictions.
    Audit outcomes are never used, so selecting a resolution with it is an outcome-blind design step."""
    from .audit import _expected_ok, _unit_support
    codes = np.unique(np.asarray(unit_aud).astype(str), return_inverse=True)[1]
    rows = []
    masks = [~np.isnan(p) for p in probs] if probs else [np.ones(len(lab_aud), bool)]
    for c in range(K):
        in_c = lab_aud == c
        I1, I0 = in_c & (groups_aud == focal), in_c & (groups_aud == ref)
        ok, s1, s0 = True, None, None
        for j, valid in enumerate(masks):       # each model's coordinate uses only its valid-prediction rows
            ok1, _, a1 = _unit_support(codes, I1 & valid, rule)
            ok0, _, a0 = _unit_support(codes, I0 & valid, rule)
            oke = _expected_ok([probs[j]], [I1 & valid, I0 & valid], rule)[0] if probs else True
            ok = ok and ok1 and ok0 and oke
            s1, s0 = (a1, a0) if s1 is None else (min(s1, a1, key=lambda d: d["n"]), min(s0, a0, key=lambda d: d["n"]))
        rows.append((c, s1["n"], s0["n"], s1["eff_units"], s0["eff_units"], bool(ok)))
    return pd.DataFrame(rows, columns=["pocket", "proj_focal", "proj_reference", "eff_units_focal",
                                       "eff_units_reference", "supported"])


def select_level(levels: pd.DataFrame, cfg: PipelineConfig) -> tuple[int, str]:
    """Discovery-only resolution: most supported pockets, then coarser level."""
    if levels.empty:
        raise ValueError("SBM returned no nontrivial hierarchy levels")
    eligible = levels.n_supported >= 2
    if cfg.min_group_coverage > 0:
        for col in ("coverage_focal", "coverage_reference"):
            eligible &= np.isfinite(levels[col]) & (levels[col] >= cfg.min_group_coverage)
    if cfg.min_graph_profiled_share is not None:
        eligible &= np.isfinite(levels.graph_profiled_share) & (levels.graph_profiled_share >= cfg.min_graph_profiled_share)
    qualified = levels.loc[eligible]
    if len(qualified):
        selected = qualified.sort_values(["n_supported", "level"], ascending=[False, False]).iloc[0]
        return int(selected.level), "qualified"
    candidates = levels.loc[levels.n_pockets_profiled >= 2]
    return int((candidates if len(candidates) else levels).level.max()), "no_level_met_support_rule"


def tree_cut(levels: list, supported: list) -> tuple[np.ndarray, int]:
    """Cut a nested hierarchy at varying depth to maximize the number of supported pockets.

    ``levels``: per-node block labels, fine -> coarse, nested. ``supported[l]``: set of level-l block
    ids that meet the projected support rule. The best count for a block is 0 if it is unsupported,
    else max(1, sum over its children). A block is split only when its children yield >= 2 supported
    pockets (ties stay coarse, as in ``select_level``); unsupported children, including single-group
    ones, stay in the cut as visible support gaps. Support is monotone down the tree (children have
    fewer rows and expected events), so an unsupported block is never split.
    Returns (per-node cut labels, number of supported pockets in the cut)."""
    top = len(levels) - 1
    children = [None] + [{b: np.unique(levels[l - 1][levels[l] == b]) for b in np.unique(levels[l])}
                         for l in range(1, top + 1)]
    memo = {}

    def best(l, b):
        if (l, b) not in memo:
            if b not in supported[l]:
                memo[l, b] = 0
            else:
                below = sum(best(l - 1, c) for c in children[l][b]) if l > 0 else 0
                memo[l, b] = max(1, below)
        return memo[l, b]

    cut = []

    def descend(l, b):
        if l > 0 and b in supported[l] and sum(best(l - 1, c) for c in children[l][b]) >= 2:
            for c in children[l][b]:
                descend(l - 1, c)
        else:
            cut.append((l, b))
    for b in np.unique(levels[top]):
        descend(top, b)
    labels = np.full(len(levels[0]), -1, np.int32)
    for k, (l, b) in enumerate(cut):
        labels[levels[l] == b] = k
    return labels, sum(best(top, b) for b in np.unique(levels[top]))


def _oob_report(forest, y) -> dict:
    """Out-of-bag fit of the similarity forest on graph rows (discovery outcomes only)."""
    if not hasattr(forest, "oob_decision_function_"):   # regression forest
        p = forest.oob_prediction_
        ok = ~np.isnan(p)
        mse = float(np.mean((y[ok] - p[ok]) ** 2))
        return {"max_features": forest.max_features, "min_samples_leaf": forest.min_samples_leaf,
                "oob_mse": mse, "oob_r2": float(1 - mse / np.var(y[ok])),
                "mean_leaves": float(np.mean([e.get_n_leaves() for e in forest.estimators_]))}
    p = forest.oob_decision_function_[:, 1]
    ok = ~np.isnan(p)
    return {"max_features": forest.max_features, "min_samples_leaf": forest.min_samples_leaf,
            "oob_log_loss": float(log_loss(y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6), labels=[0, 1])),
            "oob_auroc": float(roc_auc_score(y[ok], p[ok])) if len(np.unique(y[ok])) == 2 else np.nan,
            "mean_leaves": float(np.mean([e.get_n_leaves() for e in forest.estimators_]))}


def discover(data: AuditData, cfg: PipelineConfig, preds: dict | None = None) -> dict:
    """Stage 1. Uses development and discovery rows only (audit rows get predictions and
    leaf signatures, but their outcomes are never touched)."""
    T = Timer()
    role_unit = data.unit if getattr(data, "role_unit", None) is None else data.role_unit
    role = allocate_units(role_unit, cfg.fractions, seed=cfg.seed)
    check_roles(role_unit, role)
    dev, disc, aud = role == "development", role == "discovery", role == "audit"

    with T("assessed_models"):
        if preds is None:
            models = fit_assessed_models(data.X.loc[dev], data.y[dev], data.unit[dev], seed=cfg.seed)
            preds = predict(models, data.X)
        else:
            models = None
        preds = {**preds, **data.comparators}

    sub = allocate_units(role_unit[disc], (cfg.graph_fraction, 1 - cfg.graph_fraction),
                         roles=("graph", "calib"), seed=cfg.seed + 2)
    disc_idx = np.flatnonzero(disc)
    g_idx, c_idx = disc_idx[sub == "graph"], disc_idx[sub == "calib"]

    Xnum = to_numeric_matrix(data.X)
    with T("forest"):
        forest = G.fit_similarity_forest(Xnum[g_idx], data.y[g_idx], cfg.n_trees,
                                         cfg.min_samples_leaf, cfg.seed, max_features=cfg.max_features,
                                         oob=True, n_jobs=cfg.n_jobs, outcome=cfg.outcome)
    with T("signatures"):
        sig = G.leaf_signatures(forest, Xnum)
    n_nodes = np.array([e.tree_.node_count for e in forest.estimators_])
    with T("graph"):
        if cfg.graph_method == "exact_gpu":   # optional: exact neighbor lists on a GPU (graph_gpu.py)
            from .graph_gpu import build_graph_exact_gpu
            A, gstats = build_graph_exact_gpu(sig[g_idx], k=cfg.k)
        elif cfg.graph_method == "exact_cpu":  # optional: CPU brute force, small graphs (n <= 5000)
            A = G.mutual_graph(*G.exact_topk(sig[g_idx], k=cfg.k), len(g_idx))
            deg = np.diff(A.indptr)
            gstats = {"n_nodes": int(A.shape[0]), "n_edges": int(A.nnz // 2), "k": cfg.k, "exact": True,
                      "isolated_nodes": int((deg == 0).sum()), "mean_degree": float(deg.mean())}
        else:
            A, gstats = G.build_graph(sig[g_idx], k=cfg.k, b=cfg.b, q=cfg.q, seed=cfg.seed)

    con = next(c for c in data.contrasts if c.name == data.primary_contrast)
    groups = data.protected[con.attribute].to_numpy()
    scale = aud.sum() / len(c_idx)
    partitions, reports = {}, {"graph": gstats, "forest": _oob_report(forest, data.y[g_idx])}

    a_idx = np.flatnonzero(aud)
    design_models = [m for m in preds if m in (cfg.primary_models or list(preds))]

    def freeze(labels_graph, name):
        pmap, lab = build_pocket_map(sig[g_idx], labels_graph, n_nodes, cfg.min_pocket_size)
        calibrate_thresholds(pmap, sig[c_idx], cfg.abstain_quantile)
        lab_calib = pmap.assign(sig[c_idx])
        if cfg.select_on_audit_design:     # exact audit support from the outcome-free audit design
            lab_aud = pmap.assign(sig[a_idx])
            sup = _design_support(lab_aud, groups[a_idx], data.unit[a_idx], con.focal, con.reference,
                                  pmap.n_pockets, cfg.support,
                                  probs=[np.asarray(preds[m], float)[a_idx] for m in design_models])
            ok_pockets = sup.loc[sup.supported, "pocket"].to_numpy()
            cov = {g: float(np.isin(lab_aud[groups[a_idx] == g], ok_pockets).mean())
                   for g in (con.focal, con.reference)}
            return pmap, lab, lab_calib, sup, cov
        probs = [np.asarray(p, float)[c_idx] for p in preds.values()] if cfg.project_expected_events else None
        sup = _projected_support(lab_calib, groups[c_idx], con.focal, con.reference,
                                 pmap.n_pockets, scale, cfg.min_projected_per_group,
                                 probs=probs, min_expected=cfg.support.min_expected)
        ok_pockets = sup.loc[sup.supported, "pocket"].to_numpy()
        cov = {g: float(np.isin(lab_calib[groups[c_idx] == g], ok_pockets).mean())
               for g in (con.focal, con.reference)}
        return pmap, lab, lab_calib, sup, cov

    if cfg.run_sbm:
        with T("sbm"):
            hier = sbm_hierarchy(A, seed=cfg.seed, n_init=cfg.sbm_n_init, weighting=cfg.sbm_weighting)
        level_rows, frozen, supported_blocks = [], {}, []
        for li, labels in enumerate(hier["levels"]):
            pmap, lab, lab_calib, sup, cov = freeze(labels, f"sbm_level{li}")
            n_sup = int(sup.supported.sum())
            raw = np.asarray(labels)
            supported_blocks.append({int(raw[lab == k][0]) for k in sup.loc[sup.supported, "pocket"]})
            qualifies = n_sup >= 2
            if cfg.min_group_coverage > 0:
                qualifies = qualifies and all(np.isfinite(v) and v >= cfg.min_group_coverage for v in cov.values())
            if cfg.min_graph_profiled_share is not None:
                qualifies = qualifies and (lab != ABSTAIN).mean() >= cfg.min_graph_profiled_share
            level_rows.append({"level": li, "n_blocks": hier["n_blocks"][li],
                               "n_pockets_profiled": pmap.n_pockets, "n_supported": n_sup,
                               # calibration rows are always assigned to profiled pockets, so
                               # coverage alone cannot see a level whose profiled blocks are few
                               "graph_profiled_share": float((lab != ABSTAIN).mean()),
                               "coverage_focal": cov[con.focal],
                               "coverage_reference": cov[con.reference],
                               "calib_abstain": float((lab_calib == ABSTAIN).mean()),
                               "qualifies": qualifies})
            frozen[li] = (pmap, lab)
        levels = pd.DataFrame(level_rows)
        chosen, status = select_level(levels, cfg)
        pmap, lab = frozen[chosen]
        partitions["sbm"] = pmap
        reports["sbm"] = {"levels": levels, "chosen_level": chosen, "status": status,
                          "entropies": hier["entropies"], "seconds": hier["seconds"],
                          "self_excluded_agreement": self_excluded_agreement(pmap, sig[g_idx], lab)}
        K = pmap.n_pockets
        sbm_graph_labels = lab
        if cfg.sbm_tree_cut:
            cut_labels, n_cut = tree_cut([np.asarray(x) for x in hier["levels"]], supported_blocks)
            pmap_c, lab_c, _, sup_c, cov_c = freeze(cut_labels, "sbm_cut")
            partitions["sbm_cut"] = pmap_c
            reports["sbm_cut"] = {"n_cut_blocks": int(cut_labels.max()) + 1, "n_supported_tree": int(n_cut),
                                  "n_pockets": pmap_c.n_pockets, "n_supported": int(sup_c.supported.sum()),
                                  "graph_profiled_share": float((lab_c != ABSTAIN).mean()), "coverage": cov_c}
    else:
        K, sbm_graph_labels = 4, None

    if cfg.run_leiden:
        with T("leiden"):
            lei = leiden_partition(A, n_pockets=K, seed=cfg.seed)
        pmap, lab, lab_calib, sup, cov = freeze(lei["labels"], "leiden")
        partitions["leiden"] = pmap
        rep = {k: v for k, v in lei.items() if k != "labels"}
        rep.update(n_pockets=pmap.n_pockets, coverage=cov, n_supported=int(sup.supported.sum()))
        if sbm_graph_labels is not None:
            both = (lab >= 0) & (sbm_graph_labels >= 0)
            rep["ari_vs_sbm_graph"] = float(adjusted_rand_score(lab[both], sbm_graph_labels[both]))
            rep["ari_vs_sbm_calib"] = float(adjusted_rand_score(
                lab_calib, partitions["sbm"].assign(sig[c_idx])))
        reports["leiden"] = rep

    if cfg.leiden_auto:
        from .pockets import leiden_grid
        with T("leiden_auto"):
            grid = leiden_grid(A, seed=cfg.seed, n_init=cfg.leiden_auto_n_init)
        rows_la, frozen_la = [], {}
        for li, labels in enumerate(grid["levels"]):
            pmap_l, lab_l, lab_calib_l, sup_l, cov_l = freeze(labels, f"leiden_res{li}")
            share = float((lab_l != ABSTAIN).mean())
            rows_la.append({"level": li, "resolution": grid["resolutions"][li], "n_blocks": grid["n_communities"][li],
                            "n_pockets_profiled": pmap_l.n_pockets, "n_supported": int(sup_l.supported.sum()),
                            "graph_profiled_share": share, "coverage_focal": cov_l[con.focal],
                            "coverage_reference": cov_l[con.reference]})
            frozen_la[li] = pmap_l
        levels_la = pd.DataFrame(rows_la)
        chosen_la, status_la = select_level(levels_la, cfg)
        partitions["leiden_auto"] = frozen_la[chosen_la]
        reports["leiden_auto"] = {"levels": levels_la, "chosen_level": chosen_la, "status": status_la,
                                  "resolution": grid["resolutions"][chosen_la], "seconds": grid["seconds"]}

    if cfg.run_baselines:
        with T("baselines"):
            Xg = Xnum[g_idx]
            partitions["kmeans"] = KMeansPockets(K, seed=cfg.seed).fit(Xg)
            partitions["mahalanobis_kmeans"] = MahalanobisKMeansPockets(K, seed=cfg.seed).fit(Xg)
            Ridge = RidgeRegressionKMeansPockets if cfg.outcome == "continuous" else RidgeKMeansPockets
            partitions["ridge_kmeans"] = Ridge(K, seed=cfg.seed).fit(Xg, data.y[g_idx])
            cem = CoarsenedExactPockets(cfg.cem_bins, cfg.min_pocket_size).fit(Xg)
            partitions["cem"] = cem
            reports["cem"] = {"n_strata": cem.n_strata, "n_pockets": cem.n_pockets,
                              "graph_profiled_share": float((cem.assign(Xg) != ABSTAIN).mean())}
            risk_calib = _forest_risk(forest, Xnum[c_idx])
            partitions["risk_bins"] = RiskBins(K).fit(risk_calib)
            if cfg.extra_baselines:
                from .baselines import DisparityBins, TopFeatureCEM
                partitions["cem_top"] = TopFeatureCEM(forest.feature_importances_, 3, 3, cfg.min_pocket_size).fit(Xg)
                reports["cem_top"] = {"n_pockets": partitions["cem_top"].n_pockets,
                                      "graph_profiled_share": float((partitions["cem_top"].assign(Xg) != ABSTAIN).mean())}
                m0 = (cfg.primary_models or list(preds))[0]
                p_g = np.asarray(preds[m0], float)[g_idx]
                ok = ~np.isnan(p_g)
                gg = groups[g_idx]
                partitions["disparity_bins"] = DisparityBins(K, seed=cfg.seed).fit(
                    Xg, p_g - data.y[g_idx], (gg == con.focal) & ok, (gg == con.reference) & ok, Xnum[c_idx])

    reports["timings"] = T.t
    reports["peak_rss_mb"] = peak_rss_mb()
    return {"config": cfg, "role": role, "preds": preds, "forest": forest, "sig": sig,
            "partitions": partitions, "reports": reports, "models": models,
            "graph_rows": g_idx, "calib_rows": c_idx, "n_pockets": K}


def _forest_risk(forest, X: np.ndarray) -> np.ndarray:
    """The similarity forest's prediction: event probability, or conditional mean for a regression forest."""
    return forest.predict_proba(X)[:, 1] if hasattr(forest, "predict_proba") else forest.predict(X)


def assign_partition(frozen: dict, name: str, rows: np.ndarray, Xnum: np.ndarray) -> np.ndarray:
    part = frozen["partitions"][name]
    space = getattr(part, "space", "leaves")      # pocket maps read leaf signatures
    if space == "features":
        return part.assign(Xnum[rows])
    if space == "risk":
        return part.assign(_forest_risk(frozen["forest"], Xnum[rows]))
    return part.assign(frozen["sig"][rows])


def audit(data: AuditData, frozen: dict) -> dict:
    """Stage 2. The only step that combines audit outcomes, predictions, and protected attributes."""
    cfg: PipelineConfig = frozen["config"]
    rows = np.flatnonzero(frozen["role"] == "audit")
    Xnum = to_numeric_matrix(data.X)
    acfg = AuditConfig(support=cfg.support, pairs=cfg.pairs, primary_models=cfg.primary_models,
                       n_boot=cfg.n_boot, alpha=cfg.alpha, seed=cfg.seed, method=cfg.inference, tails=cfg.tails,
                       restricted_loss=cfg.restricted_loss, se_floor=cfg.se_floor,
                       outcome=getattr(cfg, "outcome", "binary"),
                       include_losses=getattr(cfg, "outcome", "binary") == "binary",
                       include_mu=getattr(cfg, "include_mu", False))
    preds = {m: np.asarray(p)[rows] for m, p in frozen["preds"].items()}
    prot = data.protected.iloc[rows].reset_index(drop=True)
    tables, summaries, coverage, curves, assignments = [], {}, [], [], {}
    for name in frozen["partitions"]:
        pocket = assign_partition(frozen, name, rows, Xnum)
        assignments[name] = pocket
        table, summ = run_audit(data.y[rows], preds, pocket, data.unit[rows], prot,
                                data.contrasts, acfg)
        table.insert(0, "partition", name)
        tables.append(table)
        from .evaluation import q_curve
        for (model, contrast), coord in table.loc[table.kind == "delta"].groupby(["model", "contrast"]):
            con = next(c for c in data.contrasts if c.name == contrast)
            focal = prot[con.attribute].to_numpy() == con.focal
            q = q_curve(pocket, focal, coord, np.linspace(0, .3, 61))
            q.insert(0, "contrast", contrast); q.insert(0, "model", model); q.insert(0, "partition", name)
            curves.append(q)
        summaries[name] = summ
        cov = coverage_by_group(pocket, prot, data.contrasts)
        cov.insert(0, "partition", name)
        coverage.append(cov)
    return {"results": pd.concat(tables, ignore_index=True), "summary": summaries,
            "q_curves": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
            "coverage": pd.concat(coverage, ignore_index=True),
            "assignments": pd.DataFrame({"row_id": np.asarray(data.row_id)[rows], **assignments})}


def config_dict(cfg: PipelineConfig) -> dict:
    d = asdict(cfg)
    d["fractions"] = list(cfg.fractions)
    return d
