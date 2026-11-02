"""Combined audit (exploratory, 2026-10-07): forest pockets and ridge-weighted K-means as two pre-declared
families, each at half the error budget (Bonferroni: alpha = .025 each, so the family-wise false-flag rate of
the union is at most the sum). A patient is flagged if either family flags their pocket; conflicting signs
count as a wrong-sign flag.

Recomputed exactly from saved continuous-ladder replicates (no new runs): per-pocket estimates and standard
errors (pockets.csv), per-alpha bootstrap critical values (metrics.csv; q at alpha .025), per-patient pocket
labels and the planted bias (audit_inputs.npz). Reconstruction of the stored alpha-.05 intervals and person
F1 is exact (checked on spec 4c, replicate 0).

Usage: python simulations/union_audit.py --out results/tables
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
R = ROOT / "results"
SPECS7 = ["1", "2", "3", "4", "4c", "4z", "5"]
BATCH_SETS = {   # "design": the confirmation seeds on which the rule was designed; "fresh": its fresh-seed check
    "design": {"conf": ("contladder_20261005/conf_v1_P_*", "10000", SPECS7),
               "mask": ("contladder_20261005/night_mask", "20000", ["1m", "4zm"]),
               "comp": ("contladder_20261005/night_comp", "10000", ["1", "4z"])},
    "fresh": {"fresh10k": ("union_conf_20261007/n10k", "10000", SPECS7),
              "fresh20k": ("union_conf_20261007/n20k_mask", "20000", ["1m", "4zm"])},
    "null20k": {"null20k": ("nullcheck_20261007/n20k", "20000", ["1", "4z"])},   # null-only check (2026-10-10)
}
METHODS = ("rf_spectral", "ridge_kmeans")
DELTA = .4
B = 2000


def directions(pk: pd.DataFrame, q: float, labels: np.ndarray) -> np.ndarray:
    """Per-patient flag direction (+1, -1, 0) for one method's local pockets at critical value q."""
    out = np.zeros(len(labels), int)
    loc = pk[(pk.pocket.astype(str) != "all") & (pk.status == "tested")]
    for r in loc.itertuples():
        lo, hi = r.estimate - q * r.se, r.estimate + q * r.se
        dirn = int(lo > 0) - int(hi < 0)
        if dirn:
            out[labels == int(r.pocket)] = dirn
    return out


def score(dirn: np.ndarray, focal: np.ndarray, bias: np.ndarray, conflict: np.ndarray | None = None):
    affected = focal & (bias != 0)
    sign = np.sign(bias)
    flagged = dirn != 0
    if conflict is not None:
        flagged = flagged | conflict
    correct = affected & (dirn == sign) & ~(conflict if conflict is not None else False)
    wrong = affected & flagged & ~correct
    false = focal & ~affected & flagged
    tp = int(correct.sum())
    return np.array([tp, int(wrong.sum()), int(false.sum()), int(affected.sum()) - tp])


def replicate(rep_dir: Path):
    pk = pd.read_csv(rep_dir / "pockets.csv")
    m = pd.read_csv(rep_dir / "metrics.csv")
    m = m[m.floor == 0]
    z = np.load(rep_dir / "audit_inputs.npz")
    focal = z["s_aud"].astype(bool)
    Rr = z["R_aud"].astype(float)
    bias = DELTA * focal * Rr
    out = {}
    d = {}
    for meth in METHODS:
        p = pk[(pk.method == meth) & np.isclose(pk.delta, DELTA)]
        lab = z[f"lab_{meth}_d0"]
        for a in (.05, .025):
            row = m[(m.method == meth) & np.isclose(m.delta, DELTA) & np.isclose(m.alpha, a)]
            if row.empty or p.empty:
                return None
            d[(meth, a)] = directions(p, float(row.q.iloc[0]), lab)
    out["forest (.05)"] = score(d[("rf_spectral", .05)], focal, bias)
    out["ridge (.05)"] = score(d[("ridge_kmeans", .05)], focal, bias)
    f, r = d[("rf_spectral", .025)], d[("ridge_kmeans", .025)]
    conflict = (f != 0) & (r != 0) & (f != r)
    u = np.where(f != 0, f, r)
    u[conflict] = 0
    out["forest + ridge (.025 each)"] = score(u, focal, bias, conflict)
    return out


def f1(t):
    den = 2 * t[..., 0] + t[..., 1] + t[..., 2] + t[..., 3]
    return np.where(den > 0, 2 * t[..., 0] / np.where(den > 0, den, 1), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--set", choices=list(BATCH_SETS), default="design")
    args = ap.parse_args()
    BATCHES = BATCH_SETS[args.set]
    tag = "" if args.set == "design" else f"_{args.set}"
    rng = np.random.default_rng(20261007)
    rows, nulls, per_rep = [], [], []
    for batch, (pat, n, specs) in BATCHES.items():
        for sp in specs:
            dirs = sorted(d for d in glob.glob(str(R / pat / "learned" / f"spec{sp}_armP_n{n}" / "rep_*"))
                          if not d.endswith(".partial"))
            if not dirs:
                continue
            # nulls: union false flag = any flag in either family at .025 (global coordinate included)
            mm = pd.concat(pd.read_csv(Path(dd) / "metrics.csv") for dd in dirs)
            mm = mm[(mm.floor == 0) & (mm.delta == 0)]
            piv = mm[mm.method.isin(METHODS)].pivot_table(index=["rep", "alpha"], columns="method", values="any_flag",
                                                          aggfunc="first").reset_index()
            for a in (.05, .025):
                g = piv[np.isclose(piv.alpha, a)]
                nulls.append(dict(batch=batch, spec=sp, alpha_each=a, reps=len(g),
                                  forest=g.rf_spectral.astype(bool).mean(), ridge=g.ridge_kmeans.astype(bool).mean(),
                                  union=(g.rf_spectral.astype(bool) | g.ridge_kmeans.astype(bool)).mean()))
            res = {}
            for dd in dirs:
                rep = int(dd.rsplit("_", 1)[1])
                out = replicate(Path(dd))
                if out is not None:
                    res[rep] = out
            reps = sorted(res)
            if not reps:
                continue
            for r in reps:
                for k, v in res[r].items():
                    per_rep.append(dict(batch=batch, spec=sp, rep=r, method=k, person_tp=int(v[0]),
                                        person_wrong_sign=int(v[1]), person_false_unaffected=int(v[2]),
                                        person_fn=int(v[3])))
            names = list(next(iter(res.values())))
            arr = {k: np.stack([res[r][k] for r in reps]) for k in names}
            draws = rng.integers(0, len(reps), (B, len(reps)))
            cnt = np.stack([np.bincount(x, minlength=len(reps)) for x in draws])
            boot = {k: f1(np.einsum("br,rk->bk", cnt, v)) for k, v in arr.items()}
            best_single = np.maximum(boot["forest (.05)"], boot["ridge (.05)"])
            for k, v in arr.items():
                est = float(f1(v.sum(0)))
                gap = boot[k] - best_single
                rows.append(dict(batch=batch, spec=sp, method=k, reps=len(reps), f1=est,
                                 lo=np.nanpercentile(boot[k], 2.5), hi=np.nanpercentile(boot[k], 97.5),
                                 minus_best_single=est - max(float(f1(arr["forest (.05)"].sum(0))),
                                                             float(f1(arr["ridge (.05)"].sum(0)))),
                                 mbs_lo=np.nanpercentile(gap, 2.5), mbs_hi=np.nanpercentile(gap, 97.5)))
    res = pd.DataFrame(rows)
    nl = pd.DataFrame(nulls)
    res.to_csv(args.out / f"union_audit{tag}.csv", index=False)
    pd.DataFrame(per_rep).to_csv(args.out / f"union_audit{tag}_reps.csv", index=False)
    nl.to_csv(args.out / f"union_audit_nulls{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    if not res.empty:
        print(res.pivot_table(index=["batch", "spec"], columns="method", values="f1").round(3).to_string())
        print(res[res.method.str.startswith("forest +")][["spec", "f1", "minus_best_single", "mbs_lo", "mbs_hi"]].round(3).to_string())
    print(nl.round(3).to_string())


if __name__ == "__main__":
    main()
