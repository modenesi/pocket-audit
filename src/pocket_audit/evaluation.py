"""Person-weighted localization and interval projections for frozen pocket audits.

A person inherits a subgroup contrast; this is not a causal individual effect estimate.
Unknown people stay in recall and population-share denominators.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def person_metrics(pocket, focal, bias, coordinates):
    pocket, focal, bias = np.asarray(pocket), np.asarray(focal, bool), np.asarray(bias)
    local = coordinates.loc[coordinates.pocket != "all"].copy()
    if local.pocket.duplicated().any():
        raise ValueError("select one model and contrast before person evaluation")
    estimates = np.full(len(pocket), np.nan)
    direction = np.zeros(len(pocket), int)
    known = np.zeros(len(pocket), bool)
    for row in local.itertuples():
        m = pocket == int(row.pocket)
        tested = getattr(row, 'status', 'tested' if row.supported else 'not_eligible') == 'tested'
        if tested and np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            known[m] = True
            estimates[m] = row.estimate
            direction[m] = int(row.ci_low > 0) - int(row.ci_high < 0)
    affected = focal & (bias != 0)
    correct = affected & (direction == np.sign(bias))
    wrong = affected & (direction != 0) & ~correct
    false = focal & ~affected & (direction != 0)
    tp, fs, fu = map(lambda x: int(x.sum()), (correct, wrong, false))
    fn = int(affected.sum()) - tp
    flagged = tp + fs + fu
    evaluable = focal & known
    n = int(focal.sum())
    total_bias = np.abs(bias[affected]).sum()
    errors = estimates[evaluable] - bias[evaluable]
    between, within = 0., 0.
    for c in np.unique(pocket[evaluable]):
        m = evaluable & (pocket == c)
        mean_bias = bias[m].mean()
        between += m.sum() * (estimates[m][0] - mean_bias)**2
        within += ((bias[m] - mean_bias)**2).sum()
    denom = int(evaluable.sum())
    return dict(person_tp=tp, person_wrong_sign=fs, person_false_unaffected=fu, person_fn=fn,
                person_n_focal=n, person_n_affected=int(affected.sum()), person_n_flagged=flagged,
                person_precision=tp/flagged if flagged else np.nan,
                person_recall=tp/affected.sum() if affected.any() else np.nan,
                person_f1=2*tp/(2*tp+fs+fu+fn) if 2*tp+fs+fu+fn else np.nan,
                person_bias_captured=np.abs(bias[correct]).sum()/total_bias if total_bias else np.nan,
                person_unknown_share=(focal & ~known).sum()/n if n else np.nan,
                person_n_evaluable=denom,
                person_mae=np.abs(errors).mean() if denom else np.nan,
                person_rmse=np.sqrt(np.mean(errors**2)) if denom else np.nan,
                person_between_mse=between/denom if denom else np.nan,
                person_within_mse=within/denom if denom else np.nan)


def q_curve(pocket, focal, coordinates, thresholds):
    """Unconditional focal-population Q curves; unknown mass is shown separately."""
    pocket, focal = np.asarray(pocket), np.asarray(focal, bool)
    local = coordinates.loc[coordinates.pocket != "all"].copy()
    if local.pocket.duplicated().any():
        raise ValueError("select one model and contrast before Q projection")
    n = focal.sum()
    rows = []
    for direction in (1, -1):
        eligible = []
        for r in local.itertuples():
            if getattr(r, 'status', 'tested' if r.supported else 'not_eligible') != 'tested':
                continue
            if not np.isfinite([r.estimate, r.ci_low, r.ci_high]).all():
                continue
            weight = (focal & (pocket == int(r.pocket))).sum()/n if n else 0.
            lo, hi = (r.ci_low, r.ci_high) if direction == 1 else (-r.ci_high, -r.ci_low)
            eligible.append((weight, direction*r.estimate, lo, hi))
        unknown = 1-sum(r[0] for r in eligible) if n else np.nan
        for t in thresholds:
            if t < 0:
                raise ValueError("Q thresholds must be nonnegative")
            rows.append(dict(direction='positive' if direction == 1 else 'negative', threshold=float(t),
                             q_estimate=sum(w for w,e,l,h in eligible if e > t),
                             q_lower=sum(w for w,e,l,h in eligible if l > t),
                             q_upper=sum(w for w,e,l,h in eligible if h > t), unknown_share=unknown))
    return pd.DataFrame(rows)
