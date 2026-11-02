"""Equal-tailed max-t: same draws as the symmetric version; tails differ only when t* is skewed."""
import numpy as np
import pandas as pd

from pocket_audit.audit import AuditConfig, SupportRule, run_audit
from pocket_audit.data import Contrast


def _data(seed=0, n=4000, p=0.3):
    rng = np.random.default_rng(seed)
    s = rng.random(n) < .3
    pocket = rng.integers(0, 3, n)
    prob = np.full(n, p)
    y = (rng.random(n) < prob).astype(int)
    return y, {"m": prob}, pocket, np.arange(n).astype(str), pd.DataFrame({"S": np.where(s, "1", "0")})


def _audit(tails, **kw):
    y, preds, pocket, unit, prot = _data(**kw)
    cfg = AuditConfig(support=SupportRule(min_n=50, min_units=1, min_effective_units=1), n_boot=1999,
                      alpha=.05, seed=3, method="wild_restricted_mammen", tails=tails)
    return run_audit(y, preds, pocket, unit, prot, [Contrast("S", "S", "1", "0")], cfg)


def test_symmetric_default_unchanged_and_tails_share_draws():
    t_sym, s_sym = _audit("symmetric")
    t_eq, s_eq = _audit("equal")
    d = t_sym.kind == "delta"
    assert np.allclose(t_sym.loc[d, "estimate"], t_eq.loc[d, "estimate"])
    assert np.allclose(t_sym.loc[d, "se"], t_eq.loc[d, "se"])
    assert s_sym["primary"]["q"] == s_eq["primary"]["q"]           # same bootstrap draws
    # roughly symmetric residuals (p = .3): both one-sided quantiles sit near the symmetric one
    assert abs(s_eq["primary"]["q_up"] - s_eq["primary"]["q"]) < .35
    assert abs(s_eq["primary"]["q_dn"] - s_eq["primary"]["q"]) < .35


def test_equal_tail_intervals_and_flags_are_consistent():
    t, s = _audit("equal", p=.03)
    tested = t.loc[t.status == "tested"]
    q_up, q_dn = s["primary"]["q_up"], s["primary"]["q_dn"]
    prim = tested.family == "primary"
    assert np.allclose(tested.loc[prim, "ci_low"], tested.loc[prim, "estimate"] - q_up * tested.loc[prim, "se"])
    assert np.allclose(tested.loc[prim, "ci_high"], tested.loc[prim, "estimate"] + q_dn * tested.loc[prim, "se"])
    assert ((tested.flag_positive) == (tested.ci_low > 0)).all()


def test_restricted_loss_is_opt_in_and_only_changes_the_loss_family():
    y, preds, pocket, unit, prot = _data(p=.08)
    base = dict(support=SupportRule(min_n=50, min_units=1, min_effective_units=1), n_boot=999, alpha=.05, seed=3,
                method="wild_restricted_mammen")
    t0, s0 = run_audit(y, preds, pocket, unit, prot, [Contrast("S", "S", "1", "0")], AuditConfig(**base))
    t1, s1 = run_audit(y, preds, pocket, unit, prot, [Contrast("S", "S", "1", "0")],
                       AuditConfig(**base, restricted_loss=True))
    assert s0["primary"]["q"] == s1["primary"]["q"]                 # disparity family untouched
    assert s0["secondary"]["q"] != s1["secondary"]["q"]
    th = t0.kind == "theta"
    assert np.allclose(t0.loc[th, "estimate"], t1.loc[th, "estimate"]) and np.allclose(t0.loc[th, "se"], t1.loc[th, "se"])


def test_se_floor_is_opt_in_and_never_shrinks_se():
    y, preds, pocket, unit, prot = _data(p=.05)
    base = dict(support=SupportRule(min_n=50, min_units=1, min_effective_units=1, min_expected=5), n_boot=499, alpha=.05,
                seed=3, method="wild_restricted_mammen")
    t0, _ = run_audit(y, preds, pocket, unit, prot, [Contrast("S", "S", "1", "0")], AuditConfig(**base))
    t1, _ = run_audit(y, preds, pocket, unit, prot, [Contrast("S", "S", "1", "0")], AuditConfig(**base, se_floor=True))
    d = (t0.kind == "delta") & (t0.status == "tested")
    assert (t1.loc[d, "se"] >= t0.loc[d, "se"] - 1e-15).all()
    assert np.allclose(t1.loc[d, "se"], np.maximum(t0.loc[d, "se"], t0.loc[d, "se_model0"]))
    assert np.allclose(t0.loc[d, "estimate"], t1.loc[d, "estimate"])
