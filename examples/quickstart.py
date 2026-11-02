"""Quickstart: find where a model is miscalibrated, and for whom, on simulated data (about a minute).

The true risk depends on x1 and x2. Among patients with high x1, the assessed model over-predicts risk for
everyone, and by more for group B; elsewhere it is calibrated. The audit should flag the pocket that holds those
patients twice: its pocket-average error (mu, "where") and its B-minus-A gap (delta, "for whom") are both positive,
while the other pockets look fine.

    python examples/quickstart.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from pocket_audit.audit import SupportRule
from pocket_audit.data import AuditData, Contrast
from pocket_audit.pipeline import PipelineConfig, audit, discover

rng = np.random.default_rng(0)
n = 20000
X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"x{i}" for i in range(1, 7)])  # x3..x6 are noise
group = np.where(rng.random(n) < 0.3, "B", "A")              # protected attribute: never a feature
p_true = expit(-2.5 + X.x1 + 0.8 * (X.x2 ** 2 - 1))
y = (rng.random(n) < p_true).astype(int)
region = (X.x1 > 1.28).to_numpy()                            # top 10% of x1
p_model = expit(logit(p_true) + region * (0.6 + 1.2 * (group == "B")))  # over-predicts in the region, more for B

data = AuditData(
    name="quickstart", X=X, y=y,
    unit=np.arange(n).astype(str),                           # independent observations: each row is its own unit
    row_id=np.arange(n),
    protected=pd.DataFrame({"group": group}),
    contrasts=[Contrast("B_vs_A", "group", focal="B", reference="A")],
    primary_contrast="B_vs_A",
)

has_sbm = importlib.util.find_spec("graph_tool") is not None  # nested SBM needs graph-tool (conda-forge)
cfg = PipelineConfig(
    seed=1, n_jobs=4, n_boot=999, alpha=0.05,
    include_mu=True,                    # audit "where" (mu) alongside "for whom" (delta)
    run_sbm=has_sbm, run_leiden=not has_sbm, run_baselines=False,
    abstain_quantile=0.01, se_floor=True,
    support=SupportRule(min_n=100, min_units=10, min_effective_units=10, min_expected=10),
)

# Stage 1: discovery. Uses development and discovery rows only; never looks at the audit rows' outcomes.
frozen = discover(data, cfg, preds={"model": p_model})
# Stage 2: audit. Assigns the held-out audit rows to the frozen pockets and tests every pocket at once.
res = audit(data, frozen)

partition = "sbm" if has_sbm else "leiden"
r = res["results"]
r = r[(r.partition == partition) & r.kind.isin(["mu", "delta"]) & (r.status == "tested")]
cols = ["kind", "pocket", "estimate", "ci_low", "ci_high", "flag"]
cols = [c for c in cols if c in r.columns]
print(f"\nPartition: {partition}. Simultaneous 95% intervals; flag = interval excludes 0.\n")
print(r[cols].round(3).to_string(index=False))

aud = np.flatnonzero(frozen["role"] == "audit")
pockets = res["assignments"][partition].to_numpy()
share = pd.Series(region[aud]).groupby(pockets).mean().rename("share in planted region")
print("\nShare of each pocket's audit rows inside the planted region (-1 = abstained):\n")
print(share.round(2).to_string())
