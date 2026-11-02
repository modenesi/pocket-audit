"""Comparison partitions, the graded-relevance DGP, and localization metrics."""
import numpy as np
import pandas as pd
from scipy.special import expit

from pocket_audit.assignment import ABSTAIN
from pocket_audit.baselines import CoarsenedExactPockets, MahalanobisKMeansPockets, RidgeKMeansPockets
from pocket_audit.simulation import GRADED, SimConfig, _intercept, localization, simulate


def test_ridge_kmeans_ignores_irrelevant_feature():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 2))
    y = (rng.random(4000) < expit(-2 + 1.5 * X[:, 0])).astype(int)
    part = RidgeKMeansPockets(4, seed=0).fit(X, y)
    assert part.w[1] < 0.1 * part.w[0]
    lab = part.assign(X)
    # pockets follow the relevant feature: x1 explains nearly all between-pocket variation
    r2 = [pd.Series(X[:, j]).groupby(lab).transform("mean").var() / X[:, j].var() for j in (0, 1)]
    assert r2[0] > 0.8 and r2[1] < 0.05


def test_mahalanobis_kmeans_whitens():
    rng = np.random.default_rng(1)
    X = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 3000)
    part = MahalanobisKMeansPockets(3, seed=0).fit(X)
    Z = part._embed(X)
    assert np.allclose(np.cov(Z, rowvar=False), np.eye(2), atol=0.02)
    assert set(np.unique(part.assign(X))) == {0, 1, 2}


def test_cem_strata_and_abstention():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2000, 2))
    X[:5, 1] = np.nan
    part = CoarsenedExactPockets(n_bins=3, min_size=50).fit(X)
    assert part.n_pockets == 9            # 3 x 3 strata, ~222 rows each
    lab = part.assign(X)
    assert (lab[:5] == ABSTAIN).all()     # the 5-row "missing" strata are too small
    assert (lab[5:] >= 0).all()
    many = CoarsenedExactPockets(n_bins=3, min_size=50).fit(rng.standard_normal((2000, 20)))
    assert many.n_pockets == 0            # exact matching on many features: every stratum is tiny


def test_default_dgp_unchanged_and_graded_prevalence():
    rng = np.random.default_rng(12345)
    x1, x2 = rng.standard_normal((2, 400000))
    lo, hi = -10.0, 5.0
    for _ in range(60):                   # the original intercept search
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(mid + x1 + 0.8 * (x2 ** 2 - 1)).mean() < 0.08 else (lo, mid)
    assert _intercept(0.08) == (lo + hi) / 2
    data, info = simulate(SimConfig(n=20000, **GRADED), seed=0)
    assert data.X.shape[1] == 3 + GRADED["p_noise"]
    assert abs(info["p0"].mean() - 0.08) < 0.005


def test_localization_scores():
    region = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    pocket = np.array([0, 0, 1, 1, 2, 2, ABSTAIN, 2])
    delta = pd.DataFrame({"pocket": ["all", 0, 1, 2], "supported": [True, True, True, True],
                          "flag": [False, True, False, False]})
    loc = localization(pocket, region, delta)
    assert loc["flag_pocket"] and not loc["flag_global"]
    assert loc["loc_precision"] == 1.0 and np.isclose(loc["loc_recall"], 2 / 3)
    assert loc["best_supported_purity"] == 1.0
    assert loc["n_pockets"] == 3 and loc["abstain_share"] == 1 / 8
