"""Audit estimands and the multiplier max-t procedure."""
import numpy as np
import pandas as pd

from pocket_audit.audit import AuditConfig, SupportRule, multiplier_maxt, run_audit
from pocket_audit.data import Contrast


def test_delta_estimate_is_difference_of_mean_residuals():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    p = np.clip(rng.random(n), 0.01, 0.99)
    g = np.where(rng.random(n) < 0.5, "a", "b")
    pocket = rng.integers(0, 2, n)
    cfg = AuditConfig(support=SupportRule(min_n=10, min_units=1, min_effective_units=1), n_boot=199)
    table, _ = run_audit(y, {"m": p}, pocket, np.arange(n).astype(str), pd.DataFrame({"g": g}),
                         [Contrast("a_vs_b", "g", "a", "b")], cfg)
    row = table[(table.kind == "delta") & (table.pocket == 1)].iloc[0]
    m = pocket == 1
    expected = (p - y)[m & (g == "a")].mean() - (p - y)[m & (g == "b")].mean()
    assert np.isclose(row.estimate, expected)


def test_maxt_simultaneous_coverage_under_null():
    """Independent rows, 20 true-zero mean coordinates: coverage should be near 95%."""
    rng = np.random.default_rng(1)
    reps, cover = 200, 0
    for r in range(reps):
        H, J = 400, 20
        X = rng.standard_normal((H, J))
        est = X.mean(0)
        U = (X - est) / H
        res = multiplier_maxt(U, est, n_boot=499, seed=r)
        cover += np.all(np.abs(est) <= res["q"] * res["se"])
    assert 0.90 <= cover / reps <= 0.99


def test_unsupported_coordinates_are_reported_not_tested():
    n = 300
    y = np.zeros(n, int); y[::3] = 1
    p = np.full(n, 0.3)
    g = np.array(["a"] * 20 + ["b"] * 280)
    cfg = AuditConfig(support=SupportRule(min_n=100, min_units=1, min_effective_units=1), n_boot=99)
    table, _ = run_audit(y, {"m": p}, np.zeros(n, int), np.arange(n).astype(str),
                         pd.DataFrame({"g": g}), [Contrast("a_vs_b", "g", "a", "b")], cfg)
    d = table[table.kind == "delta"]
    assert not d.supported.any()
    assert d.ci_low.isna().all() and not d.flag.any()
