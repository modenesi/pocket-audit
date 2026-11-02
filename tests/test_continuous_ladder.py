"""Continuous-outcome audit options, proximity clustering, and the continuous ladder DGP."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))

from pocket_audit.audit import AuditConfig, SupportRule, build_coordinates  # noqa: E402
from pocket_audit.data import Contrast  # noqa: E402
from pocket_audit.proximity_clustering import leaf_incidence, proximity_matrix, spectral_embedding  # noqa: E402


def _toy(n=400, seed=0, binary=False):
    rng = np.random.default_rng(seed)
    s = rng.random(n) < .3
    pocket = rng.integers(0, 2, n)
    if binary:
        p = rng.uniform(.05, .6, n)
        y = (rng.random(n) < p).astype(float)
    else:
        p = 5 + rng.normal(0, 1, n)          # far outside [0, 1]: must not be clipped
        y = p + rng.normal(0, 1, n)
    prot = pd.DataFrame({"S": np.where(s, "1", "0")})
    return y, p, pocket, prot


def _coords(y, p, pocket, prot, **kw):
    sup = SupportRule(min_n=10, min_units=1, min_effective_units=1, min_expected=kw.pop("min_expected", 0.0))
    return build_coordinates(y, {"m": p}, pocket, np.arange(len(y)), prot, [Contrast("S", "S", "1", "0")],
                             AuditConfig(support=sup, **kw))


def test_continuous_floor_is_pooled_variance_without_clipping():
    y, p, pocket, prot = _toy()
    table, *_ = _coords(y, p, pocket, prot, outcome="continuous", include_losses=False)
    s = prot.S.to_numpy() == "1"
    for row in table.itertuples():
        m = np.ones(len(y), bool) if row.pocket == "all" else pocket == row.pocket
        r = (p - y)[m]
        n1, n0 = (m & s).sum(), (m & ~s).sum()
        assert row.se_model0 == pytest.approx(np.sqrt(r.var(ddof=1) * (1 / n1 + 1 / n0)))


def test_include_losses_false_omits_loss_family():
    y, p, pocket, prot = _toy()
    with_l, *_ = _coords(y, p, pocket, prot, outcome="continuous")
    without, *_ = _coords(y, p, pocket, prot, outcome="continuous", include_losses=False)
    assert "theta" in set(with_l.kind) and set(without.kind) == {"delta"}


def test_continuous_rejects_expected_event_rule_and_unknown_outcome():
    y, p, pocket, prot = _toy()
    with pytest.raises(ValueError, match="binary-only"):
        _coords(y, p, pocket, prot, outcome="continuous", min_expected=10.0)
    with pytest.raises(ValueError, match="outcome"):
        _coords(y, p, pocket, prot, outcome="count")


def test_binary_default_floor_unchanged():
    y, p, pocket, prot = _toy(binary=True)
    table, *_ = _coords(y, p, pocket, prot)
    s = prot.S.to_numpy() == "1"
    row = table.loc[(table.kind == "delta") & (table.pocket == "all")].iloc[0]
    r = p - y
    mu = np.clip(p - r.mean(), 1e-6, 1 - 1e-6)
    v = mu * (1 - mu)
    assert row.se_model0 == pytest.approx(np.sqrt(v[s].sum() / s.sum() ** 2 + v[~s].sum() / (~s).sum() ** 2))


def test_leaf_incidence_proximity_matches_bruteforce():
    rng = np.random.default_rng(1)
    sig = rng.integers(0, 7, (60, 25)) * 3 + 1        # arbitrary node ids, with gaps
    brute = (sig[:, None, :] == sig[None, :, :]).mean(axis=2)
    assert leaf_incidence(sig).sum() == 60 * 25
    np.testing.assert_allclose(proximity_matrix(sig), brute)


def test_spectral_embedding_separates_blocks():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    truth = np.repeat([0, 1, 2], 40)
    rng = np.random.default_rng(2)
    P = np.where(truth[:, None] == truth[None, :], .6, .05) + rng.uniform(0, .02, (120, 120))
    P = (P + P.T) / 2
    np.fill_diagonal(P, 1.0)
    E, info = spectral_embedding(P, kmax=4)
    assert info["n_unit_eigenvalues"] == 1 and E.shape == (120, 4)
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(E[:, :3])
    assert adjusted_rand_score(truth, lab) == 1.0


def test_ladder_dgp_normalization_pairing_and_missingness():
    import continuous_ladder as L
    rng = np.random.default_rng(3)
    lat = L.draw_latent(400000, rng)
    for name, spec in L.SPECS.items():
        assert np.var(L.mean_fn(lat, spec)) == pytest.approx(.4, abs=.006), name
    # Specs 1 and 2 share the latent draws: same truth, same region, features differ only by monotone maps
    o1, o2 = L.observe(lat, L.SPECS["1"]), L.observe(lat, L.SPECS["2"])
    np.testing.assert_array_equal(L.mean_fn(lat, L.SPECS["1"]), L.mean_fn(lat, L.SPECS["2"]))
    for j in range(12):
        assert np.all(np.diff(o2["forest"][np.argsort(o1["forest"][:, j]), j]) >= 0)
    # Spec 5: no row dropped; missing rates at their targets; truth uses complete Z
    o5 = L.observe(lat, L.SPECS["5"])
    assert o5["forest"].shape[0] == 400000 and o5["baseline"].shape[0] == 400000
    miss = np.isnan(o5["forest"])
    assert miss[:, 0].mean() == pytest.approx(.10, abs=.003)
    assert miss[:, 1].mean() == pytest.approx(.20, abs=.003)
    assert miss[:, 3].mean() == pytest.approx(.30, abs=.003)
    assert miss[:, 12].mean() == pytest.approx(.10, abs=.003)          # c1 (ordinal column)
    assert not np.isnan(o5["baseline"][:, 12:]).any()                    # categoricals one-hot with a missing level
    np.testing.assert_array_equal(L.mean_fn(lat, L.SPECS["5"]), L.mean_fn(lat, L.SPECS["4"]))
    assert L.region(lat, L.SPECS["4"]).mean() == pytest.approx(.2, abs=.003)


def test_ridge_regression_kmeans_weights_relevant_features():
    from pocket_audit.baselines import RidgeRegressionKMeansPockets
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2000, 4)) * [1, 10, 100, 1000]
    y = X[:, 0] + rng.normal(0, .5, 2000)
    rk = RidgeRegressionKMeansPockets(3, seed=0).fit(X, y)
    assert np.argmax(rk.w) == 0 and len(np.unique(rk.assign(X))) == 3
