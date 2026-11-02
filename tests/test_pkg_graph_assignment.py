"""Sparse graph, assignment profiles, and role allocation."""
import numpy as np
import pandas as pd

from pocket_audit import graph as G
from pocket_audit.assignment import ABSTAIN, build_pocket_map, calibrate_thresholds
from pocket_audit.splits import allocate_units, check_roles


def _forest_and_sig(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] * 2)))).astype(int)
    rf = G.fit_similarity_forest(X, y, n_trees=50, min_samples_leaf=5, seed=seed)
    return rf, G.leaf_signatures(rf, X), X


def test_candidate_search_recovers_exact_topk():
    _, sig, _ = _forest_and_sig()
    approx = G.candidate_topk(sig, k=10, b=64, q=600, seed=1)
    exact = G.exact_topk(sig, k=10)
    assert G.neighbor_recall(approx, exact) > 0.95


def test_exact_scores_are_coleaf_counts():
    _, sig, _ = _forest_and_sig(n=50)
    src, dst, score = G.exact_topk(sig, k=5)
    for s, d, w in zip(src[:20], dst[:20], score[:20]):
        assert w == int((sig[s] == sig[d]).sum())


def test_mutual_graph_symmetric_and_bounded():
    _, sig, _ = _forest_and_sig()
    A, stats = G.build_graph(sig, k=10, b=16, q=128, seed=0)
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0
    assert stats["n_edges"] <= sig.shape[0] * 10 / 2


def test_affinity_matches_brute_force_mean_proximity():
    rf, sig, X = _forest_and_sig(n=300)
    labels = np.repeat([0, 1, 2], 100)
    n_nodes = np.array([e.tree_.node_count for e in rf.estimators_])
    pmap, lab = build_pocket_map(sig, labels, n_nodes)
    new = sig[:7]
    A = pmap.affinity(new)
    for c in range(3):
        members = sig[labels == c]
        brute = (new[:, None, :] == members[None, :, :]).mean(axis=2).mean(axis=1)
        np.testing.assert_allclose(A[:, c], brute, atol=1e-6)


def test_thresholds_cause_abstention_for_low_affinity():
    rf, sig, X = _forest_and_sig(n=300)
    n_nodes = np.array([e.tree_.node_count for e in rf.estimators_])
    pmap, _ = build_pocket_map(sig, np.repeat([0, 1], 150), n_nodes)
    calibrate_thresholds(pmap, sig, quantile=0.5)
    lab = pmap.assign(sig)
    assert 0.3 < (lab == ABSTAIN).mean() < 0.9


def test_level_without_profiled_pockets_abstains_everywhere():
    """A fine SBM level can have no block of min_pocket_size rows; discover() used to crash."""
    rf, sig, X = _forest_and_sig(n=300)
    n_nodes = np.array([e.tree_.node_count for e in rf.estimators_])
    pmap, lab = build_pocket_map(sig, np.arange(300) % 30, n_nodes, min_pocket_size=50)
    assert pmap.n_pockets == 0 and (lab == ABSTAIN).all()
    calibrate_thresholds(pmap, sig)
    assert (pmap.assign(sig) == ABSTAIN).all()


def test_units_never_cross_roles_and_allocation_is_outcome_blind():
    unit = np.repeat(np.arange(200), 5).astype(str)
    r1 = allocate_units(unit, seed=3)
    check_roles(unit, r1)
    r2 = allocate_units(unit[::-1].copy(), seed=3)[::-1]
    assert (r1 == r2).all()          # depends only on unit ids and seed, not row order
    counts = pd.Series(r1).groupby(unit).first().value_counts()
    assert counts["development"] == 80 and counts["discovery"] == 60 and counts["audit"] == 60
