"""Continuous-outcome simulation ladder: small samples, proximity-matrix pockets, independent rows.

Every frozen setting is in CONFIG / SPECS below; their hash is
stored in each replicate's done.json and checked on resume.

DGP. Every spec and arm of a replicate shares the same latent draws (a paired ladder):
    Z ~ N(0, I_12); eps ~ N(0, variance .6); S ~ Bern(.3), independent of everything else
    c1 in {0..3} w.p. (.4, .3, .2, .1); c2 ~ U{0..5}; c3 ~ U{0..9}; missingness uniforms; unit roles
    m = c_spec * m0(Z, c1, c2), with c_spec the exact analytic normalizer so Var(m) = .4 (R^2 = .4, null SD(Y) = 1)
    linear m0 = z1 + .5 z2 + .5 z3 + .2 (z4 + z5 + z6)                         (z7..z12 noise)
    nonlinear m0 = z1 + .8 (z2^2 - 1)/sqrt2 + .5 z3 1{c1 >= 2} + .3 1{z4 > 1} + .2 (z5 + z6)
    categorical specs add c1 effects (0, .25, .5, .75) and c2 effects (.4, .4, 0, 0, 0, 0); c3 is noise
    Arm P: Y = m + eps,            f = m + delta S R   (f is the truth plus a planted distortion; not fit on Y)
    Arm T: Y = m - delta S R + eps, f = HistGradientBoostingRegressor fit on development rows (features only, no S)
With S independent of the features and missingness, and the models and maps frozen, the pocket truth in both
arms is delta_c = delta * Pr(R | C = c) (estimated on a 50k integration sample) and the global truth is .2 delta.

Specs (cumulative):  1 vanilla N(0,1) | 2 raw scales (monotone transforms) | 3 + categoricals |
    4 + nonlinear, region moved to the upper tail of the U-shaped z2 | 4c = 4 with the region on z1 |
    4z = 4 with untransformed Z (a linear method cannot see the U shape) | 5 = 4 + missing values.

Pockets (all fit on discovery rows; no protected attribute anywhere):
    rf_spectral (primary)  regression-forest proximity -> normalized spectral embedding -> K-means; K = 2..16
                           chosen on calibration rows (most pockets projected to have >= 75 audit rows per group)
    rf_rows                K-means on the rows of the proximity matrix (the earlier approach)
    ridge_kmeans, kmeans (standardized), kmeans_raw (unstandardized), score_bins (forest prediction),
    oracle (R + K-1 bins of the true m outside R; a localization benchmark, not a method)
All comparators use the primary's profiled K. Forest-map methods abstain at the 1% calibration quantiles.

Audit: delta coordinates (pockets + "all"), wild bootstrap max-t, Mammen weights, null imposed, symmetric;
critical values at alpha .05 and .025 from the same draws; SE floor off (primary) and on (sensitivity).

    python simulations/continuous_ladder.py learned --out-dir results/<run> --specs 1 --arms P --reps 0 200
    python simulations/continuous_ladder.py oracle  --out-dir results/<run> --reps 0 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import norm  # noqa: E402

CONFIG = dict(
    version="continuous_ladder_v1",
    p=12, var_m=.4, var_eps=.6, prev_s=.3, region_quantile=.8,
    fractions=(.4, .3, .3), graph_fraction=.8,
    forest=dict(n_estimators=500, min_samples_leaf=20, max_features=.5),
    k_min=2, k_max=16, min_pocket_size=50, abstain_quantile=.01, min_projected=75, min_graph_profiled_share=.8,
    support=dict(min_n=50, min_units=1, min_effective_units=1.0, min_expected=0.0),
    alphas=(.05, .025), n_int=50000, int_chunk=25000,
    c1_probs=(.4, .3, .2, .1), c1_effects=(0, .25, .5, .75), c2_effects=(.4, .4, 0, 0, 0, 0), c3_levels=10,
    missing=dict(x1=.10, x2_mar_rate=.20, x2_mar_slope_on_z1=1.0, x4=.30, c1=.10),
    transforms=["65+15z", "exp(.8z)", "1000*Phi(z)", "10*sinh(1.5z)", ".01z", "100+40z"],   # x1..x6, repeated x7..x12
)
SPECS = {
    "1":  dict(transform=False, cats=False, nonlinear=False, region="z1", missing=False),
    "2":  dict(transform=True, cats=False, nonlinear=False, region="z1", missing=False),
    "3":  dict(transform=True, cats=True, nonlinear=False, region="z1", missing=False),
    "4":  dict(transform=True, cats=True, nonlinear=True, region="z2", missing=False),
    "4c": dict(transform=True, cats=True, nonlinear=True, region="z1", missing=False),
    "4z": dict(transform=False, cats=True, nonlinear=True, region="z2", missing=False),
    "5":  dict(transform=True, cats=True, nonlinear=True, region="z2", missing=True),
    # masking (2026-10-06): +delta in one half of the region, -delta in the other, so the global gap is ~0
    "1m":  dict(transform=False, cats=False, nonlinear=False, region="mask_z1z2", missing=False),
    "4zm": dict(transform=False, cats=True, nonlinear=True, region="mask_z2z1", missing=False),
}
METHODS = ["rf_spectral", "rf_rows", "ridge_kmeans", "kmeans", "kmeans_raw", "score_bins", "cem_top", "oracle"]
PER_DELTA = ["disparity_bins"]     # comparators refit for each delta (they use the audited model's residuals)
Q_R = float(norm.ppf(CONFIG["region_quantile"]))
C1_P, C1_EFF, C2_EFF = (np.array(CONFIG[k], float) for k in ("c1_probs", "c1_effects", "c2_effects"))
TRANSFORMS = [lambda z: 65 + 15 * z, lambda z: np.exp(.8 * z), lambda z: 1000 * norm.cdf(z),
              lambda z: 10 * np.sinh(1.5 * z), lambda z: .01 * z, lambda z: 100 + 40 * z]


def _cat_var(p, eff):
    p, eff = np.asarray(p, float), np.asarray(eff, float)
    return float(p @ eff ** 2 - (p @ eff) ** 2)


def m_variance(spec: dict) -> float:
    """Exact Var(m0) for the spec (all terms independent or uncorrelated)."""
    cats = _cat_var(C1_P, C1_EFF) + _cat_var(np.full(6, 1 / 6), C2_EFF) if spec["cats"] else 0.0
    if not spec["nonlinear"]:
        return 1 + .25 + .25 + 3 * .04 + cats
    p4 = float(norm.sf(1.0))
    return 1 + .64 + .25 * float(C1_P[2:].sum()) + .09 * p4 * (1 - p4) + 2 * .04 + cats


def mar_intercept(rate: float, slope: float) -> float:
    """a with E[expit(a + slope * Z)] = rate for Z ~ N(0,1) (numerical integration + bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(100):
        a = (lo + hi) / 2
        val = quad(lambda z: expit(a + slope * z) * norm.pdf(z), -12, 12)[0]
        lo, hi = (a, hi) if val < rate else (lo, a)
    return (lo + hi) / 2


MAR_A = mar_intercept(CONFIG["missing"]["x2_mar_rate"], CONFIG["missing"]["x2_mar_slope_on_z1"])


def config_hash(extra: dict) -> str:
    blob = json.dumps(dict(config=CONFIG, specs=SPECS, methods=METHODS, **extra), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- data
def seed_of(base: int, n: int, rep: int, stream: int) -> int:
    """Streams: 0 latent data, 1 roles, 2 integration data, 10 forest, 11 k-means, 12 bootstrap, 13 assessed model.
    Specs and arms are deliberately not in the key: the ladder is paired within a replicate."""
    return int(np.random.SeedSequence([base, n, rep, stream]).generate_state(1)[0])


def draw_latent(n: int, rng: np.random.Generator) -> dict:
    return dict(Z=rng.standard_normal((n, CONFIG["p"])), eps=rng.normal(0, np.sqrt(CONFIG["var_eps"]), n),
                s=rng.random(n) < CONFIG["prev_s"], c1=rng.choice(4, n, p=C1_P), c2=rng.integers(0, 6, n),
                c3=rng.integers(0, CONFIG["c3_levels"], n), u_miss=rng.random((n, 4)))


def mean_fn(lat: dict, spec: dict) -> np.ndarray:
    Z = lat["Z"]
    if spec["nonlinear"]:
        m = (Z[:, 0] + .8 * (Z[:, 1] ** 2 - 1) / np.sqrt(2) + .5 * Z[:, 2] * (lat["c1"] >= 2)
             + .3 * (Z[:, 3] > 1) + .2 * (Z[:, 4] + Z[:, 5]))
    else:
        m = Z[:, 0] + .5 * Z[:, 1] + .5 * Z[:, 2] + .2 * (Z[:, 3] + Z[:, 4] + Z[:, 5])
    if spec["cats"]:
        m = m + C1_EFF[lat["c1"]] + C2_EFF[lat["c2"]]
    return np.sqrt(CONFIG["var_m"] / m_variance(spec)) * m


def region(lat: dict, spec: dict) -> np.ndarray:
    """Boolean region, or for masking specs a signed array (+1 / -1 halves of the region, 0 elsewhere)."""
    Z = lat["Z"]
    if spec["region"] == "mask_z1z2":
        return np.where(Z[:, 0] > Q_R, np.where(Z[:, 1] > 0, 1, -1), 0)
    if spec["region"] == "mask_z2z1":
        return np.where(Z[:, 1] > Q_R, np.where(Z[:, 0] > 0, 1, -1), 0)
    return Z[:, 0 if spec["region"] == "z1" else 1] > Q_R


def _onehot(codes: np.ndarray, k: int) -> np.ndarray:
    return (codes[:, None] == np.arange(k)[None, :]).astype(float)


def observe(lat: dict, spec: dict) -> dict:
    """Observed features: 'forest' (ordinal c1 with NaN, one-hot c2/c3, numeric NaN) and 'baseline' (one-hot
    categoricals, missing c1 as its own level; numeric NaN imputed later from discovery medians)."""
    Z = lat["Z"]
    num = (np.column_stack([TRANSFORMS[j % 6](Z[:, j]) for j in range(Z.shape[1])]) if spec["transform"]
           else Z.copy())
    c1 = lat["c1"].astype(float)
    if spec["missing"]:
        u, mc = lat["u_miss"], CONFIG["missing"]
        num[u[:, 0] < mc["x1"], 0] = np.nan
        num[u[:, 1] < expit(MAR_A + mc["x2_mar_slope_on_z1"] * Z[:, 0]), 1] = np.nan
        num[u[:, 2] < mc["x4"], 3] = np.nan
        c1[u[:, 3] < mc["c1"]] = np.nan
    names = [f"x{j + 1}" for j in range(num.shape[1])]
    if not spec["cats"]:
        return dict(forest=num, baseline=num, forest_names=names)
    c2, c3 = _onehot(lat["c2"], 6), _onehot(lat["c3"], CONFIG["c3_levels"])
    c1_codes = np.where(np.isnan(c1), 4, c1).astype(int)
    c1_oh = _onehot(c1_codes, 5 if spec["missing"] else 4)
    return dict(forest=np.column_stack([num, c1, c2, c3]), baseline=np.column_stack([num, c1_oh, c2, c3]),
                forest_names=names + ["c1"] + ["c2"] * 6 + ["c3"] * CONFIG["c3_levels"])


# ----------------------------------------------------------------------------- discovery
def projected_support(lab_calib, s_calib, K, scale, min_count):
    """Counts-only projection of audit support per pocket (mirrors pipeline._projected_support)."""
    rows = []
    for c in range(K):
        in_c = lab_calib == c
        n1, n0 = scale * np.sum(in_c & s_calib), scale * np.sum(in_c & ~s_calib)
        rows.append((c, n1, n0, bool(n1 >= min_count and n0 >= min_count)))
    return pd.DataFrame(rows, columns=["pocket", "proj_focal", "proj_reference", "supported"])


class RawKMeans:
    """K-means on median-imputed, unstandardized features (the 'forgot to standardize' pitfall)."""

    def __init__(self, k, seed):
        self.k, self.seed = k, seed

    def fit(self, X):
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(self.imp.transform(X))
        return self

    def assign(self, X):
        return self.km.predict(self.imp.transform(X)).astype(np.int32)


class OracleBins:
    """R is one pocket; the rest is cut into K-1 bins of the true m at discovery quantiles."""

    def __init__(self, k):
        self.k = k

    def fit(self, m, R):
        R = np.asarray(R)
        self.signed = bool((R < 0).any())          # masking: the two signed halves are separate pockets
        nb = self.k - (2 if self.signed else 1)
        q = np.linspace(0, 1, max(nb, 1) + 1)[1:-1]
        self.edges = np.unique(np.quantile(m[R == 0], q)) if nb > 1 else np.array([])
        return self

    def assign(self, m, R):
        R = np.asarray(R)
        rest = (2 if self.signed else 1) + np.searchsorted(self.edges, m, side="right")
        return np.where(R > 0, 0, np.where(R < 0, 1, rest)).astype(np.int32)


def discover(Xf, Xb, y, s, m, R, g, c, seeds: dict, cfg_forest: dict) -> tuple[dict, dict]:
    """Fit every partition on discovery rows (g = graph rows, c = calibration rows). Returns (models, info)."""
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from pocket_audit.assignment import build_pocket_map, calibrate_thresholds, self_excluded_agreement
    from pocket_audit.baselines import KMeansPockets, RidgeRegressionKMeansPockets, RiskBins, TopFeatureCEM
    from pocket_audit.proximity_clustering import proximity_matrix, spectral_embedding
    T0 = time.perf_counter()
    tm = {}
    forest = RandomForestRegressor(oob_score=True, n_jobs=1, random_state=seeds["forest"], **cfg_forest)
    forest.fit(Xf[g], y[g])
    tm["forest"] = time.perf_counter() - T0
    sig_g, sig_c = forest.apply(Xf[g]).astype(np.int32), forest.apply(Xf[c]).astype(np.int32)
    n_nodes = np.array([e.tree_.node_count for e in forest.estimators_])
    t = time.perf_counter()
    P = proximity_matrix(sig_g)
    E, einfo = spectral_embedding(P, CONFIG["k_max"])
    tm["proximity_spectral"] = time.perf_counter() - t
    t = time.perf_counter()
    scale = seeds["n_audit"] / len(c)
    curve, maps = [], {}
    for K in range(CONFIG["k_min"], CONFIG["k_max"] + 1):
        lab_g = KMeans(K, n_init=10, random_state=seeds["kmeans"]).fit_predict(E[:, :K])
        pmap, lab = build_pocket_map(sig_g, lab_g, n_nodes, CONFIG["min_pocket_size"])
        calibrate_thresholds(pmap, sig_c, CONFIG["abstain_quantile"])
        lab_c = pmap.assign(sig_c)
        sup = projected_support(lab_c, s[c], pmap.n_pockets, scale, CONFIG["min_projected"])
        share = float((lab >= 0).mean())
        n_sup = int(sup.supported.sum())
        curve.append(dict(K=K, K_profiled=pmap.n_pockets, n_supported=n_sup, graph_profiled_share=share,
                          calib_abstain=float((lab_c < 0).mean()),
                          qualifies=bool(n_sup >= 2 and share >= CONFIG["min_graph_profiled_share"])))
        maps[K] = (pmap, lab)
    cv = pd.DataFrame(curve)
    q = cv.loc[cv.qualifies]
    if len(q):
        K_sel, status = int(q.sort_values(["n_supported", "K"], ascending=[False, True]).iloc[0].K), "qualified"
    else:
        K_sel, status = CONFIG["k_min"], "fallback_k2"
    tm["k_selection"] = time.perf_counter() - t
    pmap, lab_sel = maps[K_sel]
    Kp = max(pmap.n_pockets, 2)
    t = time.perf_counter()
    rows_lab = KMeans(Kp, n_init=10, random_state=seeds["kmeans"]).fit_predict(P)
    pmap_rows, lab_rows = build_pocket_map(sig_g, rows_lab, n_nodes, CONFIG["min_pocket_size"])
    calibrate_thresholds(pmap_rows, sig_c, CONFIG["abstain_quantile"])
    tm["rows_kmeans"] = time.perf_counter() - t
    leaf_values = [e.tree_.value[:, 0, 0] for e in forest.estimators_]
    pred_c = np.mean([v[sig_c[:, j]] for j, v in enumerate(leaf_values)], axis=0)
    t = time.perf_counter()
    models = dict(
        forest=forest, leaf_values=leaf_values, rf_spectral=pmap, rf_rows=pmap_rows,
        ridge_kmeans=RidgeRegressionKMeansPockets(Kp, seed=seeds["kmeans"]).fit(Xb[g], y[g]),
        kmeans=KMeansPockets(Kp, seed=seeds["kmeans"]).fit(Xb[g]),
        kmeans_raw=RawKMeans(Kp, seeds["kmeans"]).fit(Xb[g]),
        score_bins=RiskBins(Kp).fit(pred_c),
        cem_top=TopFeatureCEM(forest.feature_importances_, 3, 3, CONFIG["min_pocket_size"]).fit(Xf[g]),
        oracle=OracleBins(Kp).fit(m[g], R[g]))
    tm["baselines"] = time.perf_counter() - t
    oob = forest.oob_prediction_
    ok = np.isfinite(oob)
    info = dict(K_requested=K_sel, K_profiled=int(pmap.n_pockets), K_status=status, k_curve=curve,
                rows_K_profiled=int(pmap_rows.n_pockets),
                self_agreement=self_excluded_agreement(pmap, sig_g, lab_sel),
                oob_r2=float(forest.oob_score_), oob_rmse=float(np.sqrt(np.mean((oob[ok] - y[g][ok]) ** 2))),
                importances=dict(zip(range(len(forest.feature_importances_)), map(float, forest.feature_importances_))),
                spectral=einfo, ridge_alpha=float(models["ridge_kmeans"].lr.alpha_),
                ridge_weights=[float(w) for w in models["ridge_kmeans"].w],
                timings={k: round(v, 2) for k, v in tm.items()})
    models["graph_labels"] = lab_sel
    return models, info


def assign_all(models: dict, Xf, Xb, m, R) -> dict:
    """Labels for every method on new rows (chunked forest application)."""
    out = {k: [] for k in METHODS}
    step = CONFIG["int_chunk"]
    for a in range(0, len(Xf), step):
        sl = slice(a, a + step)
        sig = models["forest"].apply(Xf[sl]).astype(np.int32)
        pred = np.mean([v[sig[:, j]] for j, v in enumerate(models["leaf_values"])], axis=0)
        out["rf_spectral"].append(models["rf_spectral"].assign(sig))
        out["rf_rows"].append(models["rf_rows"].assign(sig))
        for k in ("ridge_kmeans", "kmeans", "kmeans_raw"):
            out[k].append(models[k].assign(Xb[sl]))
        out["score_bins"].append(models["score_bins"].assign(pred))
        out["cem_top"].append(models["cem_top"].assign(Xf[sl]))
        out["oracle"].append(models["oracle"].assign(m[sl], R[sl]))
    return {k: np.concatenate(v).astype(np.int32) for k, v in out.items()}


def region_share(lab_int: np.ndarray, R_int: np.ndarray) -> tuple[dict, dict]:
    """Pr(R | C = c) and integration counts per pocket (abstained rows excluded)."""
    pr, nn = {}, {}
    for c in np.unique(lab_int[lab_int >= 0]):
        msk = lab_int == c
        nn[int(c)] = int(msk.sum())
        pr[int(c)] = float(R_int[msk].mean())
    return pr, nn


# ----------------------------------------------------------------------------- audit + evaluation
def audit_eval(y_a, f_a, lab, s_a, R_a, delta, pr, nint, boot_seed, n_boot, weights="mammen",
               glob: float = .2) -> tuple[list, pd.DataFrame]:
    """Audit one partition at one delta: floor off/on x alpha .05/.025. Returns metric rows and pocket rows."""
    from pocket_audit.audit import AuditConfig, SupportRule, _quantile, build_coordinates, wild_maxt
    from pocket_audit.data import Contrast
    from pocket_audit.evaluation import person_metrics
    cfg = AuditConfig(support=SupportRule(**CONFIG["support"]), outcome="continuous", include_losses=False)
    prot = pd.DataFrame({"S": np.where(s_a, "1", "0")})
    unit = np.arange(len(y_a))
    table, U, _, cells = build_coordinates(y_a, {"model": f_a}, lab, unit, prot, [Contrast("S", "S", "1", "0")], cfg)
    table = table.loc[table.kind == "delta"].copy()
    table["pocket"] = table.pocket.astype(str)
    table["truth"] = [0.0 if delta == 0 else glob * delta if p == "all" else delta * pr.get(int(p), np.nan)
                      for p in table.pocket]                      # delta = 0: every truth is exactly zero
    table["p_region"] = [glob if p == "all" else pr.get(int(p), np.nan) for p in table.pocket]
    table["n_int"] = [np.nan if p == "all" else nint.get(int(p), 0) for p in table.pocket]
    idx = np.flatnonzero(table.supported.to_numpy())          # positions within the delta table
    status = np.where(table.supported, "tested", "not_eligible").astype(object)
    if len(idx):
        orig = table.index.to_numpy()[idx]                      # positions in build_coordinates' U / cells
        Uc = U[:, orig] - U[:, orig].mean(0)
        est0 = table.estimate.to_numpy()[idx]
        zero = (Uc ** 2).sum(0) <= (32 * np.finfo(float).eps * np.maximum(1, np.abs(est0))) ** 2
        status[idx[zero]] = "degenerate"
        idx = idx[~zero]
    table["status"] = status
    est = table.estimate.to_numpy()[idx]
    truth = table.truth.to_numpy()[idx]
    is_all = (table.pocket.to_numpy()[idx] == "all")
    p_region = table.p_region.to_numpy()[idx]
    aff = s_a & (R_a != 0) & (delta > 0)
    rows, pockets = [], None
    se_raw = None
    for floor in (False, True):
        if len(idx):
            fl = table.se_model0.to_numpy()[idx] if floor else None
            orig = table.index.to_numpy()[idx]
            res = wild_maxt([cells[i] for i in orig], U[:, orig], est, np.ones(len(idx), bool), n_boot=n_boot,
                            seed=boot_seed, weights=weights, se_floor=fl)
            se, M = res["se"], np.sort(res["M"])
            if not floor:
                se_raw = se.copy()
        for alpha in CONFIG["alphas"]:
            co = table.copy()
            co["se"], co["ci_low"], co["ci_high"] = np.nan, np.nan, np.nan
            if len(idx):
                qa = _quantile(M, alpha)
                lo, hi = est - qa * se, est + qa * se
                co.iloc[idx, co.columns.get_loc("se")] = se
                co.iloc[idx, co.columns.get_loc("ci_low")] = lo
                co.iloc[idx, co.columns.get_loc("ci_high")] = hi
                up, dn = lo > 0, hi < 0
                cov = (lo <= truth) & (truth <= hi)
                null_pk = ~is_all & (p_region == 0)
            else:
                qa, up, dn, cov, null_pk = np.nan, np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool)
            loc = co.loc[co.pocket != "all"]
            pm = person_metrics(lab, s_a, delta * s_a * R_a, loc)
            tested_local = loc.loc[loc.status == "tested", "pocket"].astype(int).to_numpy()
            known = np.isin(lab, tested_local)
            rows.append(dict(
                delta=delta, floor=int(floor), alpha=alpha, q=qa, J=int(len(idx)),
                n_pockets=int((co.pocket != "all").sum()), n_local_tested=int(len(tested_local)),
                any_flag=bool((up | dn).any()), any_flag_up=bool(up.any()), any_flag_down=bool(dn.any()),
                any_local_flag=bool(((up | dn) & ~is_all).any()), global_flag=bool(((up | dn) & is_all).any()),
                covered_all=bool(cov.all()) if len(idx) else np.nan,
                lower_miss=bool((lo > truth).any()) if len(idx) else np.nan,
                upper_miss=bool((hi < truth).any()) if len(idx) else np.nan,
                n_null_pockets_tested=int(null_pk.sum()), null_pocket_flag=bool(((up | dn) & null_pk).any()),
                floor_binds=int((table.se_model0.to_numpy()[idx] > se_raw).sum()) if (floor and len(idx)) else 0,
                mean_halfwidth=float(np.mean(qa * se)) if len(idx) else np.nan,
                est_all=float(table.loc[table.pocket == "all", "estimate"].iloc[0]),
                aff_n=int(aff.sum()), aff_untested=int((aff & ~known).sum()), aff_abstain=int((aff & (lab < 0)).sum()),
                focal_abstain_share=float((s_a & (lab < 0)).mean() / max(s_a.mean(), 1e-12)),
                ref_abstain_share=float((~s_a & (lab < 0)).mean() / max((~s_a).mean(), 1e-12)),
                **pm))
            if not floor and alpha == CONFIG["alphas"][0]:
                se0 = table.se_model0.to_numpy()
                pockets = co.assign(se_floor_value=se0)[["pocket", "n_focal", "n_reference", "status", "reason",
                                                          "p_region", "n_int", "truth", "estimate", "se",
                                                          "se_floor_value", "ci_low", "ci_high"]]
    return rows, pockets


# ----------------------------------------------------------------------------- one replicate
def run_task(task) -> tuple:
    opts, spec_name, arm, n, rep = task
    spec = SPECS[spec_name]
    dest = Path(opts["out_dir"]) / "learned" / f"spec{spec_name}_arm{arm}_n{n}" / f"rep_{rep:03d}"
    if (dest / "done.json").exists():
        meta = json.loads((dest / "done.json").read_text())
        if meta.get("config_hash") != opts["config_hash"]:
            raise RuntimeError(f"{dest}: existing replicate has config hash {meta.get('config_hash')}, "
                               f"this run {opts['config_hash']}; use a new --out-dir")
        return spec_name, arm, n, rep, "cached", 0.0
    tmp = dest.with_name(dest.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    from pocket_audit.splits import allocate_units
    t0 = time.perf_counter()
    try:
        base = opts["base_seed"]
        sd = {k: seed_of(base, n, rep, s) for k, s in
              dict(data=0, roles=1, integration=2, forest=10, kmeans=11, bootstrap=12, model=13).items()}
        lat = draw_latent(n, np.random.default_rng(sd["data"]))
        obs = observe(lat, spec)
        m, R, s = mean_fn(lat, spec), region(lat, spec), lat["s"]
        uid = np.arange(n).astype(str)
        role = allocate_units(uid, CONFIG["fractions"], seed=sd["roles"])
        disc = np.flatnonzero(role == "discovery")
        sub = allocate_units(uid[disc], (CONFIG["graph_fraction"], 1 - CONFIG["graph_fraction"]),
                             roles=("graph", "calib"), seed=sd["roles"] + 2)
        g, c = disc[sub == "graph"], disc[sub == "calib"]
        dev, aud = np.flatnonzero(role == "development"), np.flatnonzero(role == "audit")
        lat_i = draw_latent(CONFIG["n_int"], np.random.default_rng(sd["integration"]))
        obs_i = observe(lat_i, spec)
        m_i, R_i = mean_fn(lat_i, spec), region(lat_i, spec)
        seeds = dict(forest=sd["forest"], kmeans=sd["kmeans"] % (2 ** 31), n_audit=len(aud))
        deltas = opts["deltas"] if rep < opts["alt_reps"] else [d for d in opts["deltas"] if d == 0]
        rows, pockets, discoveries, saved = [], [], {}, {"aud_rows": aud, "graph_rows": g, "s_aud": s[aud], "R_aud": R[aud]}
        cache = None
        for di, delta in enumerate(deltas):
            y = m - (delta * s * R if arm == "T" else 0.0) + lat["eps"]
            if arm == "P":
                f = m + delta * s * R
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                hgb = HistGradientBoostingRegressor(random_state=sd["model"] % (2 ** 31)).fit(obs["forest"][dev], y[dev])
                f = hgb.predict(obs["forest"])
            if cache is None or arm == "T":       # arm P: discovery Y does not contain the distortion
                td = time.perf_counter()
                models, info = discover(obs["forest"], obs["baseline"], y, s, m, R, g, c, seeds, opts["forest"])
                lab_aud = assign_all(models, obs["forest"][aud], obs["baseline"][aud], m[aud], R[aud])
                ti = time.perf_counter()
                if any(d > 0 for d in deltas):     # null-only replicates need no integration (truth is exactly 0)
                    lab_int = assign_all(models, obs_i["forest"], obs_i["baseline"], m_i, R_i)
                else:
                    lab_int = {k: np.zeros(0, np.int32) for k in METHODS}
                shares = {k: region_share(v, R_i[:len(v)]) for k, v in lab_int.items()}
                info["timings"]["integration"] = round(time.perf_counter() - ti, 2)
                info["timings"]["discovery_total"] = round(time.perf_counter() - td, 2)
                info["int_abstain"] = {k: float((v < 0).mean()) if len(v) else None for k, v in lab_int.items()}
                info["importance_by_feature"] = pd.Series(models["forest"].feature_importances_,
                                                          index=obs["forest_names"]).groupby(level=0).sum().to_dict()
                del info["importances"]
                cache = (models, info, lab_aud, shares)
                saved[f"graph_labels_d{di}"] = models["graph_labels"]
                for k, v in lab_aud.items():
                    saved[f"lab_{k}_d{di}"] = v
            models, info, lab_aud, shares = cache
            from pocket_audit.baselines import DisparityBins
            dbin = DisparityBins(max(info["K_profiled"], 2), seed=seeds["kmeans"]).fit(
                obs["forest"][g], (f - y)[g], s[g], ~s[g], obs["forest"][c])
            lab_aud = dict(lab_aud, disparity_bins=dbin.assign(obs["forest"][aud]))
            lab_int_d = dbin.assign(obs_i["forest"]) if delta > 0 else np.zeros(0, np.int32)
            shares = dict(shares, disparity_bins=region_share(lab_int_d, R_i[:len(lab_int_d)]))
            saved[f"lab_disparity_bins_d{di}"] = lab_aud["disparity_bins"]
            if arm == "T":
                info = dict(info, model_r2_audit=float(1 - np.mean((f[aud] - y[aud]) ** 2) / np.var(y[aud])),
                            model_params={k: str(v) for k, v in hgb.get_params().items()})
            discoveries[str(delta)] = info
            saved[f"y_d{di}"], saved[f"f_d{di}"] = y[aud], f[aud]
            ta = time.perf_counter()
            for meth in METHODS + PER_DELTA:
                pr, nint = shares[meth]
                r, pk = audit_eval(y[aud], f[aud], lab_aud[meth], s[aud], R[aud], delta, pr, nint,
                                   sd["bootstrap"] % (2 ** 31), opts["n_boot"],
                                   glob=0.0 if spec["region"].startswith("mask") else .2)   # analytic global truth
                for x in r:
                    x.update(method=meth, K_profiled=info["K_profiled"], K_requested=info["K_requested"],
                             K_status=info["K_status"], oob_r2=info["oob_r2"])
                rows += r
                if pk is not None:
                    pockets.append(pk.assign(method=meth, delta=delta))
            info["timings"][f"audit_d{di}"] = round(time.perf_counter() - ta, 2)
        metrics = pd.DataFrame(rows)
        for k, v in reversed(list(dict(spec=spec_name, arm=arm, n=n, rep=rep).items())):
            metrics.insert(0, k, v)
        metrics.to_csv(tmp / "metrics.csv", index=False)
        pk = pd.concat(pockets) if pockets else pd.DataFrame()
        pk.insert(0, "rep", rep)
        pk.insert(0, "spec", spec_name)
        pk.insert(1, "arm", arm)
        pk.to_csv(tmp / "pockets.csv", index=False)
        np.savez_compressed(tmp / "audit_inputs.npz", **saved)
        (tmp / "discovery.json").write_text(json.dumps(discoveries, indent=1, default=float))
        meta = dict(spec=spec_name, arm=arm, n=n, rep=rep, deltas=deltas, seeds=sd, config_hash=opts["config_hash"],
                    role_counts=dict(zip(*np.unique(role, return_counts=True))) | {"graph": len(g), "calib": len(c)},
                    seconds=round(time.perf_counter() - t0, 1))
        (tmp / "done.json").write_text(json.dumps(meta, indent=1, default=int))
        if dest.exists():
            raise RuntimeError(f"{dest} appeared during the run")
        tmp.rename(dest)
        return spec_name, arm, n, rep, "completed", meta["seconds"]
    except Exception:
        (tmp / "failure.txt").write_text(traceback.format_exc())
        raise


# ----------------------------------------------------------------------------- oracle (fixed pockets)
def run_oracle(task) -> list:
    """Stage 1: fixed known pockets (quintiles of z1; R = the top one) on Spec 1 audit rows, arm P. No discovery."""
    opts, rep = task
    from pocket_audit.splits import allocate_units
    n = opts["n"]
    sd = {k: seed_of(opts["base_seed"] + 7, n, rep, s) for k, s in dict(data=0, roles=1, bootstrap=12).items()}
    lat = draw_latent(n, np.random.default_rng(sd["data"]))
    spec = SPECS["1"]
    m, R, s = mean_fn(lat, spec), region(lat, spec), lat["s"]
    aud = np.flatnonzero(allocate_units(np.arange(n).astype(str), CONFIG["fractions"], seed=sd["roles"]) == "audit")
    edges = norm.ppf([.2, .4, .6, .8])
    lab = np.searchsorted(edges, lat["Z"][aud, 0]).astype(np.int32)
    pr, nint = {c: float(c == 4) for c in range(5)}, {c: 10 ** 9 for c in range(5)}
    rows = []
    for delta in opts["deltas"]:
        y, f = m + lat["eps"], m + delta * s * R
        r, _ = audit_eval(y[aud], f[aud], lab, s[aud], R[aud], delta, pr, nint, sd["bootstrap"] % (2 ** 31), opts["n_boot"],
                          weights=opts["weights"])
        rows += [dict(x, rep=rep, method="fixed_quintiles", weights=opts["weights"]) for x in r]
    return rows


# ----------------------------------------------------------------------------- driver
def provenance(args, chash) -> dict:
    files = sorted((ROOT / "src/pocket_audit").glob("*.py")) + [Path(__file__).resolve()]
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
    import importlib.metadata as md
    return dict(command=sys.argv, args={k: str(v) for k, v in vars(args).items()}, config_hash=chash, config=CONFIG,
                specs=SPECS, mar_intercept=MAR_A, normalizers={k: m_variance(v) for k, v in SPECS.items()},
                source_sha256=h.hexdigest(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                   cwd=ROOT, text=True),
                versions={m: md.version(m) for m in ("numpy", "pandas", "scipy", "scikit-learn")},
                python=sys.version, conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["learned", "oracle"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--specs", nargs="+", choices=list(SPECS), default=["1"])
    ap.add_argument("--arms", nargs="+", choices=["P", "T"], default=["P"])
    ap.add_argument("--n-total", type=int, nargs="+", default=[10000])
    ap.add_argument("--reps", type=int, nargs=2, default=[0, 5], metavar=("START", "STOP"))
    ap.add_argument("--alt-reps", type=int, default=10 ** 9, help="replicates >= this run delta = 0 only")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, .2, .4])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1999)
    ap.add_argument("--base-seed", type=int, default=2026100500)
    ap.add_argument("--max-features", default=None, help="sensitivity override of CONFIG forest max_features")
    ap.add_argument("--weights", choices=["mammen", "rademacher"], default="mammen",
                    help="oracle stage only: bootstrap weights (diagnostic; learned runs use Mammen)")
    args = ap.parse_args()
    forest = dict(CONFIG["forest"])
    if args.max_features is not None:
        forest["max_features"] = args.max_features if args.max_features == "sqrt" else float(args.max_features)
    chash = config_hash(dict(forest=forest, n_boot=args.n_boot, base_seed=args.base_seed))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    lock = out / f"RUNNING_{args.stage}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        sys.exit(f"{lock} exists: another writer is running here (remove it only if that process is gone)")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    prov = provenance(args, chash)
    (out / f"manifest_{args.stage}_{stamp}.json").write_text(json.dumps(prov, indent=1, default=str))
    t0 = time.time()
    failures = []
    try:
        if args.stage == "oracle":
            opts = dict(base_seed=args.base_seed, n=args.n_total[0], deltas=args.deltas, n_boot=args.n_boot,
                        weights=args.weights)
            rows = []
            with ProcessPoolExecutor(args.workers) as pool:
                for fut in as_completed([pool.submit(run_oracle, (opts, r)) for r in range(*args.reps)]):
                    rows += fut.result()
            (out / "oracle").mkdir(exist_ok=True)
            tmpf = out / "oracle" / f"metrics_{args.reps[0]}_{args.reps[1]}.csv.tmp"
            pd.DataFrame(rows).to_csv(tmpf, index=False)
            tmpf.rename(tmpf.with_suffix(""))
            print(f"oracle: {len(rows)} rows", flush=True)
        else:
            opts = dict(out_dir=str(out), base_seed=args.base_seed, n_boot=args.n_boot, deltas=args.deltas,
                        alt_reps=args.alt_reps, forest=forest, config_hash=chash)
            tasks = [(opts, sp, arm, n, r) for r in range(*args.reps) for n in args.n_total
                     for arm in args.arms for sp in args.specs]
            with ProcessPoolExecutor(args.workers) as pool:
                futs = {pool.submit(run_task, t): t for t in tasks}
                for fut in as_completed(futs):
                    try:
                        print(*fut.result(), flush=True)
                    except Exception as e:
                        failures.append(dict(task=[str(x) for x in futs[fut][1:]], error=repr(e)))
                        print("FAILED", futs[fut][1:], repr(e), flush=True)
    finally:
        prov.update(status="failed" if failures else "completed", failures=failures,
                    runtime_seconds=round(time.time() - t0, 1), finished=time.strftime("%Y%m%dT%H%M%S"))
        (out / f"manifest_{args.stage}_{stamp}.json").write_text(json.dumps(prov, indent=1, default=str))
        lock.unlink(missing_ok=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
