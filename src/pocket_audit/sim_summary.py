"""Summaries of simulation replicates (light imports, so results can be read without the pipeline)."""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCALIZATION = ["flag_global", "flag_pocket", "loc_precision", "loc_recall", "best_supported_purity",
                "n_pockets", "n_supported_pockets", "median_pocket_size", "abstain_share"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(df: pd.DataFrame, by: tuple = ()) -> pd.DataFrame:
    """Rates with 95% Wilson intervals, by partition and estimand (and any extra ``by`` columns)."""
    keys = [*by, "partition", "kind"]
    out = []
    for key, g in df.groupby(keys):
        row = dict(zip(keys, key))
        row["reps"] = len(g)
        for col in ["all_covered", "any_null_flag", "any_flag"]:
            v = g[col].dropna().astype(bool)
            lo, hi = wilson(v.sum(), len(v))
            row.update({col: v.mean() if len(v) else np.nan, f"{col}_lo": lo, f"{col}_hi": hi})
        row["global_reject"] = float((g.p_global < 0.05).mean())
        row["mean_supported"] = g.n_supported.mean()
        row["mean_ci_width"] = g.mean_ci_width.mean()
        for col in LOCALIZATION:
            if col in g:
                row[col] = pd.to_numeric(g[col], errors="coerce").mean()
        out.append(row)
    return pd.DataFrame(out)
