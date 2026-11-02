"""wild_maxt_gemm reproduces wild_maxt (same draws, matrix-product algebra) across outcome types and families."""
import numpy as np
import pandas as pd
import pytest

from pocket_audit.audit import AuditConfig, SupportRule, run_audit
from pocket_audit.data import Contrast

CON = [Contrast("a_vs_b", "g", "a", "b")]
LOOSE = SupportRule(min_n=10, min_units=3, min_effective_units=3, min_expected=0)


def _data(seed, n, H, outcome):
    rng = np.random.default_rng(seed)
    unit = rng.integers(0, H, n).astype(str)
    p1 = np.clip(rng.random(n), 0.02, 0.98)
    p2 = np.clip(p1 + rng.normal(0, 0.05, n), 0.01, 0.99)
    y = (rng.random(n) < p1).astype(float) if outcome == "binary" else 3 * p1 + rng.normal(0, 1, n)
    if outcome == "continuous":
        p1, p2 = 3 * p1, 3 * p2 + 0.1
    g = np.where(rng.random(n) < 0.3, "a", "b")
    pocket = rng.integers(-1, 4, n)                    # -1: abstained rows (only in "all")
    return y, {"m1": p1, "m2": p2}, pocket, unit, pd.DataFrame({"g": g})


@pytest.mark.parametrize("outcome,H,method,se_floor,tails", [
    ("binary", 40, "wild_restricted_mammen", True, "symmetric"),
    ("binary", 400, "wild_restricted", False, "equal"),
    ("continuous", 2000, "wild_restricted_mammen", False, "symmetric"),
    ("continuous", 60, "wild", False, "symmetric"),
])
def test_gemm_matches_loop(outcome, H, method, se_floor, tails):
    y, preds, pocket, unit, prot = _data(5, 3000, H, outcome)
    out = {}
    for engine in ("loop", "gemm"):
        cfg = AuditConfig(support=LOOSE, pairs=[("m1", "m2")], primary_models=["m1"], n_boot=299, alpha=0.05,
                          seed=9, method=method, se_floor=se_floor, outcome=outcome, tails=tails,
                          include_losses=outcome == "binary", include_mu=True, engine=engine)
        out[engine] = run_audit(y, preds, pocket, unit, prot, CON, cfg)
    a, b = out["loop"][0], out["gemm"][0]
    assert (a.status == "tested").sum() > 10
    for col in ["estimate", "se", "ci_low", "ci_high", "p_adj"]:
        np.testing.assert_allclose(b[col].to_numpy(float), a[col].to_numpy(float), rtol=1e-9, atol=1e-12)
    assert (a.flag == b.flag).all()
    for fam in out["loop"][1]:
        np.testing.assert_allclose(out["gemm"][1][fam]["M"], out["loop"][1][fam]["M"], rtol=1e-9)


def test_unknown_engine_rejected():
    y, preds, pocket, unit, prot = _data(1, 500, 20, "binary")
    with pytest.raises(ValueError, match="engine"):
        run_audit(y, preds, pocket, unit, prot, CON, AuditConfig(support=LOOSE, engine="fast"))
