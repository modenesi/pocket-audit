"""Summarize semisynthetic_app.py --family runs: simultaneous coverage of the primary family against the exact
audit-design truth, miss directions, and flag rates, per partition, with Wilson intervals.

    python simulations/semisynthetic_family_summary.py --runs results/app_20261005/dep_wids_enc_rho20 ... --out <dir>
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def wilson(k, n, z=norm.ppf(.975)):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rows = []
for run in a.runs:
    fs = glob.glob(f"{run}/*/rep_[0-9][0-9][0-9]/metrics.csv")
    if not fs:
        continue
    m = pd.concat([pd.read_csv(f) for f in fs])
    for part, g in m.groupby("partition"):
        g = g[g.n_tested > 0]
        miss = int((~g.all_covered.astype(bool)).sum())
        lo, hi = wilson(miss, len(g))
        rows.append(dict(run=Path(run).name, partition=part, reps=len(g), noncoverage=f"{miss}/{len(g)}",
                         noncoverage_rate=miss / len(g) if len(g) else np.nan, nc_lo=lo, nc_hi=hi,
                         miss_up=int(g.miss_up.sum()), miss_down=int(g.miss_down.sum()),
                         any_flag_rate=g.any_flag.mean(), tested_median=g.n_tested.median(),
                         tested_local_median=g.n_local_tested.median(), K_median=g.K.median()))
s = pd.DataFrame(rows)
s.to_csv(a.out / "family_summary.csv", index=False)
(a.out / "SUMMARY.md").write_text("# Application-family check: simultaneous non-coverage of the exact audit-design truth\n\n"
                                  + s.round(3).to_markdown(index=False) + "\n")
print(s.round(3).to_string(index=False))
