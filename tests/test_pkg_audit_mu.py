"""One-sample coordinate mu_c = E[p - Y | pocket]: estimate, SEs, null-restricted bootstrap, floor, families."""
import numpy as np
import pandas as pd

from pocket_audit.audit import AuditConfig, SupportRule, build_coordinates, run_audit, wild_maxt, wild_weights
from pocket_audit.data import Contrast

CON = [Contrast("a_vs_b", "g", "a", "b")]
LOOSE = SupportRule(min_n=10, min_units=1, min_effective_units=1, min_expected=0)


def _toy(seed=3, n=600, H=12):
    rng = np.random.default_rng(seed)
    unit = rng.integers(0, H, n).astype(str)
    p = np.clip(rng.random(n), 0.05, 0.95)
    y = (rng.random(n) < p).astype(float)
    g = np.where(rng.random(n) < 0.4, "a", "b")
    pocket = rng.integers(0, 3, n)
    return y, p, g, pocket, unit


def test_mu_estimate_cluster_se_and_floor():
    y, p, g, pocket, unit = _toy()
    cfg = AuditConfig(support=LOOSE, include_mu=True, include_disparity=False, include_losses=False)
    table, U, unit_ids, _ = build_coordinates(y, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    assert set(table.kind) == {"mu"}
    assert list(table.pocket) == ["all", 0, 1, 2]
    H = len(unit_ids)
    codes = np.searchsorted(unit_ids, unit)
    for j, row in table.iterrows():
        m = np.ones(len(y), bool) if row.pocket == "all" else pocket == row.pocket
        x = p[m] - y[m]
        assert np.isclose(row.estimate, x.mean())
        # cluster-robust SE from the per-unit sums of centered residuals
        Uh = np.bincount(codes[m], weights=(x - x.mean()) / m.sum(), minlength=H)
        se = np.sqrt(H / (H - 1)) * np.sqrt(((Uh - Uh.mean()) ** 2).sum())
        se_u = np.sqrt(H / (H - 1)) * np.sqrt(((U[:, j] - U[:, j].mean()) ** 2).sum())
        assert np.isclose(se, se_u)
        assert np.isclose(row.se_model0, np.sqrt((p[m] * (1 - p[m])).sum()) / m.sum())


def test_mu_bootstrap_imposes_zero_null():
    """Restricted draws use x* = w_h * x (null centre 0), then re-estimate the mean and its SE."""
    y, p, g, pocket, unit = _toy()
    cfg = AuditConfig(support=LOOSE, include_mu=True, include_disparity=False, include_losses=False)
    table, U, unit_ids, cells = build_coordinates(y, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    idx = np.flatnonzero(table.supported.to_numpy())
    est = table.estimate.to_numpy()[idx]
    H = len(unit_ids)
    codes = np.searchsorted(unit_ids, unit)
    n_boot = 50
    w_all = wild_weights(np.random.default_rng(7), (n_boot, H), "mammen")
    res = wild_maxt([cells[i] for i in idx], U[:, idx], est, np.ones(len(idx), bool), n_boot=n_boot, seed=7,
                    batch=n_boot, weights="mammen")
    direct = np.zeros(n_boot)
    for b in range(n_boot):
        w = w_all[b][codes]
        for j in idx:
            row = table.loc[j]
            m = np.ones(len(y), bool) if row.pocket == "all" else pocket == row.pocket
            xs = w[m] * (p[m] - y[m])
            Uh = np.bincount(codes[m], weights=(xs - xs.mean()) / m.sum(), minlength=H)
            se = np.sqrt(H / (H - 1)) * np.sqrt(((Uh - Uh.mean()) ** 2).sum())
            direct[b] = max(direct[b], abs(xs.mean()) / se)
    assert np.allclose(np.sort(direct), res["M"])


def test_mu_joins_primary_family_and_floor_studentizes():
    y, p, g, pocket, unit = _toy()
    p2 = np.clip(p + 0.02, 0.01, 0.99)
    p2[:30] = np.nan                                  # missing predictions (e.g. invalid APACHE rows)
    cfg = AuditConfig(support=LOOSE, include_mu=True, include_losses=False, se_floor=True, n_boot=199,
                      primary_models=["m"], method="wild_restricted_mammen")
    table, summ = run_audit(y, {"m": p, "m2": p2}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    prim = table[table.family == "primary"]
    assert set(prim.kind) == {"delta", "mu"}           # mu and delta of primary models share one family
    assert set(table[table.kind == "mu"].groupby("model").family.first()) == {"primary", "secondary"}
    mu2 = table[(table.kind == "mu") & (table.model == "m2") & (table.pocket == "all")].iloc[0]
    assert mu2.n_focal == len(y) - 30                  # NaN predictions are excluded from the cell
    tested = table[(table.status == "tested") & (table.kind == "mu")]
    assert (tested.se >= tested.se_model0 - 1e-12).all()


def test_mu_support_uses_pooled_pocket_and_expected_events():
    y, p, g, pocket, unit = _toy()
    p_rare = np.full(len(y), 0.03)
    cfg = AuditConfig(support=SupportRule(min_n=10, min_units=1, min_effective_units=1, min_expected=10),
                      include_mu=True, include_disparity=False, include_losses=False)
    table, *_ = build_coordinates(y, {"m": p_rare}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    assert not table.loc[table.pocket != "all", "supported"].any()   # about 6 expected events per pocket
    assert table.loc[table.pocket == "all", "supported"].all()       # 18 expected events overall


def test_run_audit_floored_mu_matches_explicit_pseudo_data():
    """End to end: run_audit's restricted, floored mu critical value equals the max over explicit draws."""
    from pocket_audit.audit import _quantile
    y, p, g, pocket, unit = _toy(n=400, H=10)
    n_boot, seed = 80, 11
    cfg = AuditConfig(support=LOOSE, include_mu=True, include_disparity=False, include_losses=False, se_floor=True,
                      n_boot=n_boot, seed=seed, method="wild_restricted_mammen", alpha=0.1)
    table, summ = run_audit(y, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    unit_ids = np.unique(unit)
    H, codes = len(unit_ids), np.searchsorted(unit_ids, unit)
    w_all = wild_weights(np.random.default_rng(seed), (n_boot, H), "mammen")   # family 0 uses cfg.seed
    floor = table.se_model0.to_numpy()
    direct = np.zeros(n_boot)
    for b in range(n_boot):
        w = w_all[b][codes]
        for j, row in table.iterrows():
            m = np.ones(len(y), bool) if row.pocket == "all" else pocket == row.pocket
            xs = w[m] * (p[m] - y[m])
            Uh = np.bincount(codes[m], weights=(xs - xs.mean()) / m.sum(), minlength=H)
            se = np.sqrt(H / (H - 1)) * np.sqrt(((Uh - Uh.mean()) ** 2).sum())
            direct[b] = max(direct[b], abs(xs.mean()) / max(se, floor[j]))
    assert np.isclose(summ["primary"]["q"], _quantile(direct, 0.1))
    assert (table.se >= table.se_model0 - 1e-12).all()


def test_continuous_mu_has_no_floor_and_nonfinite_is_not_rescued():
    y, p, g, pocket, unit = _toy()
    yc = y + np.random.default_rng(1).normal(size=len(y))
    cfg = AuditConfig(support=LOOSE, include_mu=True, include_disparity=False, include_losses=False, se_floor=True,
                      outcome="continuous", n_boot=99, method="wild_restricted_mammen")
    table, _ = run_audit(yc, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg)
    assert "se_model0" not in table or table.se_model0.isna().all()
    assert (table.status == "tested").all()
    ybad = y.copy()
    ybad[0] = np.nan                                       # an invalid outcome must not yield a "tested" NaN row
    cfg_b = AuditConfig(support=LOOSE, include_mu=True, include_disparity=False, include_losses=False, se_floor=True,
                        n_boot=99, method="wild_restricted_mammen")
    tb, _ = run_audit(ybad, {"m": p}, pocket, unit, pd.DataFrame({"g": g}), CON, cfg_b)
    bad = tb[(tb.status == "tested") & ~np.isfinite(tb.ci_low.astype(float))]
    assert bad.empty
