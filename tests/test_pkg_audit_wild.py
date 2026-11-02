"""Wild cluster bootstrap-t max statistic and the expected-events support rule."""
import numpy as np
import pandas as pd

from pocket_audit.audit import (AuditConfig, SupportRule, _quantile, build_coordinates, run_audit, wild_maxt,
                                wild_weights)
from pocket_audit.data import Contrast

CON = [Contrast("a_vs_b", "g", "a", "b")]


def _toy(seed=3, n=600, H=12):
    rng = np.random.default_rng(seed)
    unit = rng.integers(0, H, n).astype(str)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.random(n), 0.05, 0.95)
    g = np.where(rng.random(n) < 0.4, "a", "b")
    pocket = rng.integers(0, 2, n)
    return y, p, g, pocket, unit


def _direct_t(parts, codes, H, w, restricted):
    """Build the bootstrap pseudo-data explicitly, re-estimate, and re-studentize."""
    null = np.concatenate([x[m] for _, m, x in parts]).mean() if restricted else None
    est_star, target, v = 0.0, 0.0, np.zeros(len(codes))
    for s, m, x in parts:
        c = null if restricted else x[m].mean()
        xs = c + w[m] * (x[m] - c)
        est_star += s * xs.mean()
        target += s * c
        v[m] += s * (xs - xs.mean()) / m.sum()
    Uh = np.bincount(codes, weights=v, minlength=H)
    se = np.sqrt(H / (H - 1)) * np.sqrt(((Uh - Uh.mean()) ** 2).sum())
    return abs(est_star - target) / se


def test_wild_maxt_matches_direct_recomputation():
    y, p, g, pocket, unit = _toy()
    cfg = AuditConfig(support=SupportRule(min_n=10, min_units=1, min_effective_units=1, min_expected=0))
    table, U, unit_ids, cells = build_coordinates(y, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    idx = np.flatnonzero(table.supported.to_numpy())
    assert set(table.kind[idx]) == {"delta", "theta"}
    est = table.estimate.to_numpy()[idx]
    H = len(unit_ids)
    codes = np.searchsorted(unit_ids, unit)
    x, B, n = p - y, (p - y) ** 2, len(y)
    everyone = np.ones(n, bool)

    def parts(row):
        m = everyone if row.pocket == "all" else pocket == row.pocket
        if row.kind == "delta":
            return [(1, m & (g == "a"), x), (-1, m & (g == "b"), x)]
        return [(1, m, B), (-1, everyone, B)]

    n_boot = 60
    for kind, restrict_delta in (("rademacher", False), ("rademacher", True), ("mammen", True)):
        draws = wild_weights(np.random.default_rng(5), (n_boot, H), kind)
        restricted = (table.kind.to_numpy()[idx] == "delta") & restrict_delta
        res = wild_maxt([cells[i] for i in idx], U[:, idx], est, restricted, n_boot=n_boot, seed=5, batch=n_boot,
                        weights=kind)
        M = np.array([max(_direct_t(parts(table.iloc[i]), codes, H, xi[codes], r)
                          for i, r in zip(idx, restricted)) for xi in draws])
        assert np.isclose(res["q"], _quantile(M, 0.05))
        M.sort()
        # A draw can tie the observed t to rounding (summation order differs by BLAS); accept either side.
        eps = 1e-9 * np.maximum(1, np.abs(res["t"]))
        p_hi = (1 + (n_boot - np.searchsorted(M, res["t"] - eps, side="left"))) / (n_boot + 1)
        p_lo = (1 + (n_boot - np.searchsorted(M, res["t"] + eps, side="left"))) / (n_boot + 1)
        assert np.all((res["p_adj"] >= p_lo - 1e-12) & (res["p_adj"] <= p_hi + 1e-12))


def test_methods_share_estimates_and_se():
    y, p, g, pocket, unit = _toy(seed=4)
    out = {}
    for method in ("gauss", "wild", "wild_restricted", "wild_restricted_mammen"):
        cfg = AuditConfig(support=SupportRule(min_n=10, min_units=5, min_effective_units=1), n_boot=99,
                          method=method)
        out[method], summ = run_audit(y, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
        assert summ["primary"]["method"] == method
    for method in ("wild", "wild_restricted", "wild_restricted_mammen"):
        assert np.allclose(out[method].estimate, out["gauss"].estimate, equal_nan=True)
        assert np.allclose(out[method].se, out["gauss"].se, equal_nan=True)


def test_expected_events_rule_uses_predictions_not_outcomes():
    rng = np.random.default_rng(0)
    n = 4000
    g = np.where(rng.random(n) < 0.5, "a", "b")
    p = np.where(g == "a", 0.002, 0.2)           # ~4 expected events among ~2000 focal rows
    y = np.zeros(n, int)
    y[:400] = 1                                  # outcomes plentiful in both groups
    cfg = AuditConfig(support=SupportRule(min_n=10, min_units=1, min_effective_units=1), n_boot=99)
    table, _ = run_audit(y, {"m": p}, np.zeros(n, int), np.arange(n).astype(str), pd.DataFrame({"g": g}),
                         CON, cfg)
    d = table[table.kind == "delta"]
    assert not d.supported.any()
    assert d.reason.str.contains("expected<10").all()
    cfg.support.min_expected = 0
    table, _ = run_audit(y, {"m": p}, np.zeros(n, int), np.arange(n).astype(str), pd.DataFrame({"g": g}),
                         CON, cfg)
    assert table[table.kind == "delta"].supported.all()


def test_mammen_weights_moments():
    xi = wild_weights(np.random.default_rng(0), 400000, "mammen")
    assert abs(xi.mean()) < 0.01 and abs(xi.var() - 1) < 0.01 and abs((xi ** 3).mean() - 1) < 0.03
