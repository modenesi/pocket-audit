"""End-to-end smoke test of the graph pipeline with a continuous outcome (regression similarity forest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocket_audit.audit import SupportRule  # noqa: E402
from pocket_audit.data import AuditData, Contrast  # noqa: E402
from pocket_audit.pipeline import PipelineConfig, audit, discover  # noqa: E402


def _data(n=4000, seed=11, shift=0.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{j}" for j in range(5)])
    g = np.where(rng.random(n) < 0.3, "f", "r")
    mean = 1.5 + X.x0 + 0.8 * X.x1 ** 2
    y = (mean + rng.normal(size=n)).to_numpy()
    pred = mean.to_numpy() + shift * ((g == "f") & (X.x0.to_numpy() > 0.5))   # planted focal over-prediction
    data = AuditData(name="toy", X=X, y=y, unit=np.arange(n).astype(str), row_id=np.arange(n),
                     protected=pd.DataFrame({"g": g}), contrasts=[Contrast("f_vs_r", "g", "f", "r")],
                     primary_contrast="f_vs_r", outcome="continuous")
    return data, pred


def test_continuous_pipeline_runs_and_audits_mu_and_delta():
    data, pred = _data(shift=1.0)
    cfg = PipelineConfig(seed=4, n_trees=40, k=15, sbm_n_init=1, outcome="continuous", include_mu=True,
                         min_projected_per_group=60, min_pocket_size=30, n_boot=199,
                         inference="wild_restricted_mammen", run_leiden=False,
                         support=SupportRule(min_n=50, min_units=10, min_effective_units=10, min_expected=0))
    frozen = discover(data, cfg, preds={"m": pred})
    assert "oob_r2" in frozen["reports"]["forest"]
    assert {"sbm", "ridge_kmeans", "risk_bins"} <= set(frozen["partitions"])
    out = audit(data, frozen)
    res = out["results"] if isinstance(out, dict) and "results" in out else out[0]
    sbm = res[res.partition == "sbm"]
    assert {"mu", "delta"} <= set(sbm.kind)
    assert not {"theta", "psi"} & set(sbm.kind)                      # no Brier family for continuous outcomes
    prim = sbm[sbm.family == "primary"]
    assert {"mu", "delta"} <= set(prim.kind)                          # one joint family
    glob = sbm[(sbm.kind == "delta") & (sbm.pocket == "all")].iloc[0]
    assert glob.estimate > 0.1                                        # the planted focal over-prediction


def test_continuous_auditdata_rejects_unknown_outcome():
    import pytest
    with pytest.raises(ValueError):
        _data()[0].__class__(name="t", X=pd.DataFrame({"a": [1.0]}), y=np.array([0.5]), unit=np.array(["1"]),
                             row_id=np.array([0]), protected=pd.DataFrame({"g": ["f"]}), contrasts=[],
                             primary_contrast="", outcome="count")
