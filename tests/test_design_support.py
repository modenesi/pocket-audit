"""Outcome-blind resolution selection from the audit design (PipelineConfig.select_on_audit_design)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))

from pocket_audit.audit import SupportRule  # noqa: E402
from pocket_audit.pipeline import PipelineConfig, _design_support, discover  # noqa: E402


def test_design_support_applies_the_audit_rule():
    lab = np.array([0] * 6 + [1] * 6)
    groups = np.array(["A", "B"] * 6)
    unit = np.array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 7])
    rule = SupportRule(min_n=3, min_units=3, min_effective_units=2.5, min_expected=0)
    sup = _design_support(lab, groups, unit, "A", "B", 2, rule)
    assert sup.supported.tolist() == [True, False]          # pocket 1's groups sit in 2 units
    assert sup.proj_focal.tolist() == [3, 3]


def test_selection_never_reads_audit_outcomes():
    import smoke_ladder as S
    d = S.draw(dict(risk="toy", delta=0.0, region="x1>q80"), 4000, np.random.default_rng(5), units=60)
    data = S.to_auditdata(d)
    cfg = PipelineConfig(seed=3, run_sbm=False, run_baselines=False, select_on_audit_design=True, n_boot=99,
                         support=SupportRule(min_n=30, min_units=5, min_effective_units=3, min_expected=0))
    f1 = discover(data, cfg, preds={"model": d["pf"]})
    aud = f1["role"] == "audit"
    y2 = data.y.copy()
    y2[aud] = 1 - y2[aud]                                    # scramble every audit outcome
    data2 = S.to_auditdata({**d, "y": y2})
    f2 = discover(data2, cfg, preds={"model": d["pf"]})
    a1 = f1["partitions"]["leiden"].assign(f1["sig"][aud])
    a2 = f2["partitions"]["leiden"].assign(f2["sig"][aud])
    np.testing.assert_array_equal(a1, a2)
    assert f1["reports"]["leiden"]["n_supported"] == f2["reports"]["leiden"]["n_supported"]


def test_design_support_uses_valid_predictions_only():
    """Counts must use the rows where the model has a prediction (APACHE-style gaps)."""
    lab = np.zeros(8, int)
    groups = np.array(["A"] * 4 + ["B"] * 4)
    unit = np.arange(8)
    rule = SupportRule(min_n=3, min_units=3, min_effective_units=1, min_expected=0)
    p = np.array([.1, .2, np.nan, np.nan, .1, .2, np.nan, np.nan])
    assert _design_support(lab, groups, unit, "A", "B", 1, rule).supported.tolist() == [True]
    assert _design_support(lab, groups, unit, "A", "B", 1, rule, probs=[p]).supported.tolist() == [False]
