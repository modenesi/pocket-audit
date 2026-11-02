"""Combined forest + ridge audit for the binary (WiDS-scale) simulation cells (exploratory, 2026-10-07).

Re-runs only the audit step from each replicate's saved audit inputs (outcomes, predictions, units, protected
attribute, frozen pocket labels) with the replicate's own seed and the locked settings (wild restricted Mammen
bootstrap, symmetric max-t, SE floor, restricted loss, B = 999 as in the saved runs). Each partition is its own
family. The combined audit runs the forest family (nested SBM; Leiden as a variant) and the ridge-weighted
K-means family each at half the cell's level, and flags a patient if either family flags their pocket
(conflicting signs count as wrong-sign flags). Reproduction of the stored single-family person F1 at the full
level is checked and reported.

Usage: python simulations/union_audit_binary.py --out results/tables [--workers 6]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))
R = ROOT / "results"
SETS = [  # (batch, cell, alpha of the saved run, hospital units?)
    ("confirm_20261004", "D1", .025, False), ("confirm_20261004", "DH1", .0125, True),
    ("confirm_20261004", "DN2", .025, False), ("confirm_20261004", "D0", .025, False),
    ("confirm_20261004", "DH0", .0125, True),
    ("night_binary_20261006", "DX1", .025, False), ("night_binary_20261006", "DM1", .025, False),
    ("night_binary_20261006", "DN2", .025, False), ("night_binary_20261006", "D1", .025, False),
    ("night_binary_20261006", "DX0", .025, False),
]
PARTS = ("sbm", "leiden", "ridge_kmeans")


def audit_dirs(y, pf, lab, unit, s, seed, alpha, hosp):
    from pocket_audit.audit import AuditConfig, run_audit
    from pocket_audit.data import Contrast
    from smoke_ladder import support_rule
    cell = {"units": 150} if hosp else {}
    acfg = AuditConfig(support=support_rule(cell), n_boot=999, alpha=alpha, seed=seed,
                       method="wild_restricted_mammen", tails="symmetric", restricted_loss=True, se_floor=True)
    prot = pd.DataFrame({"S": np.where(s, "1", "0")})
    table, summ = run_audit(y, {"model": pf}, lab, unit, prot, [Contrast("S1_vs_S0", "S", "1", "0")], acfg)
    return table, summ


def directions(table, lab, M, alpha, n_boot=999):
    """Per-patient flag direction at level alpha, from the family's sorted bootstrap maxima M."""
    from pocket_audit.audit import _quantile
    q = _quantile(np.sort(np.asarray(M)), alpha)
    d = table[(table.kind == "delta") & (table.family == "primary") & (table.pocket.astype(str) != "all")
              & (table.status == "tested")]
    out = np.zeros(len(lab), int)
    for r in d.itertuples():
        lo, hi = r.estimate - q * r.se, r.estimate + q * r.se
        dirn = int(lo > 0) - int(hi < 0)
        if dirn:
            out[lab == int(r.pocket)] = dirn
    g = table[(table.kind == "delta") & (table.family == "primary") & (table.status == "tested")]
    any_flag = bool(((g.estimate - q * g.se > 0) | (g.estimate + q * g.se < 0)).any())
    return out, any_flag


def score(dirn, focal, bias, conflict=None):
    affected = focal & (bias != 0)
    flagged = dirn != 0 if conflict is None else (dirn != 0) | conflict
    correct = affected & (dirn == np.sign(bias)) & (True if conflict is None else ~conflict)
    wrong = affected & flagged & ~correct
    false = focal & ~affected & flagged
    tp = int(correct.sum())
    return [tp, int(wrong.sum()), int(false.sum()), int(affected.sum()) - tp]


def one(task):
    batch, cell, alpha, hosp, rep_dir = task
    import warnings
    warnings.filterwarnings("ignore")
    z = np.load(Path(rep_dir) / "audit_inputs.npz")
    seed = json.loads((Path(rep_dir) / "done.json").read_text())["seed"]
    focal, bias = z["s"].astype(bool), z["bias"].astype(float)
    out = dict(batch=batch, cell=cell, rep=int(Path(rep_dir).name.split("_")[1]))
    dirs = {}
    for p in PARTS:
        lab = z[f"labels_{p}"]
        table, summ = audit_dirs(z["y"], z["pf"], lab, z["unit"], focal, seed, alpha, hosp)
        M = summ["primary"]["M"]
        if M is None or len(np.atleast_1d(M)) == 0:
            dirs[(p, alpha)], dirs[(p, alpha / 2)] = (np.zeros(len(lab), int), False), (np.zeros(len(lab), int), False)
        else:
            dirs[(p, alpha)] = directions(table, lab, M, alpha)
            dirs[(p, alpha / 2)] = directions(table, lab, M, alpha / 2)
    for p in PARTS:
        out[f"{p}_full"] = score(dirs[(p, alpha)][0], focal, bias)
        out[f"{p}_full_any"] = dirs[(p, alpha)][1]
    for f in ("sbm", "leiden"):
        a, b = dirs[(f, alpha / 2)][0], dirs[("ridge_kmeans", alpha / 2)][0]
        conflict = (a != 0) & (b != 0) & (a != b)
        u = np.where(a != 0, a, b)
        u[conflict] = 0
        out[f"{f}+ridge_half"] = score(u, focal, bias, conflict)
        out[f"{f}+ridge_half_any"] = dirs[(f, alpha / 2)][1] or dirs[("ridge_kmeans", alpha / 2)][1]
    return out


def f1(t):
    t = np.asarray(t, float)
    den = 2 * t[..., 0] + t[..., 1] + t[..., 2] + t[..., 3]
    return np.where(den > 0, 2 * t[..., 0] / np.where(den > 0, den, 1), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-reps", type=int, default=10 ** 6)
    ap.add_argument("--cells", nargs="*", default=None, help="restrict to these cells (e.g. D0 DH0)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    tasks = []
    for batch, cell, alpha, hosp in SETS:
        if args.cells and cell not in args.cells:
            continue
        dirs = sorted(glob.glob(str(R / batch / "learned" / cell / "rep_[0-9][0-9][0-9]")))[:args.max_reps]
        tasks += [(batch, cell, alpha, hosp, d) for d in dirs]
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(one, tasks, chunksize=4))
    df = pd.DataFrame(res)
    df.to_pickle(args.out / f"union_audit_binary{args.tag}_reps.pkl")
    # stored single-family F1 for the reproduction check
    rows = []
    rng = np.random.default_rng(20261007)
    keys = [f"{p}_full" for p in PARTS] + ["sbm+ridge_half", "leiden+ridge_half"]
    for (batch, cell), g in df.groupby(["batch", "cell"], sort=False):
        stored = pd.concat(pd.read_csv(f) for f in glob.glob(str(R / batch / "learned" / cell / "rep_*/metrics.csv")))
        stored = stored[stored.tails == "symmetric"]
        arr = {k: np.stack(g[k].to_numpy()) for k in keys}
        draws = rng.integers(0, len(g), (2000, len(g)))
        cnt = np.stack([np.bincount(x, minlength=len(g)) for x in draws])
        boot = {k: f1(np.einsum("br,rk->bk", cnt, v)) for k, v in arr.items()}
        best = np.maximum(boot["sbm_full"], boot["ridge_kmeans_full"])
        for k in keys:
            est = float(f1(arr[k].sum(0)))
            st = stored[stored.partition == k.replace("_full", "")] if k.endswith("_full") else None
            stored_f1 = float(f1(st[["person_tp", "person_wrong_sign", "person_false_unaffected", "person_fn"]].sum().to_numpy())) if st is not None and len(st) else np.nan
            null_rate = g[k + "_any"].mean()
            rows.append(dict(batch=batch, cell=cell, method=k, reps=len(g), f1=est, lo=np.nanpercentile(boot[k], 2.5),
                             hi=np.nanpercentile(boot[k], 97.5), stored_f1=stored_f1, any_flag_rate=null_rate,
                             minus_best_sbm_ridge=est - max(float(f1(arr["sbm_full"].sum(0))), float(f1(arr["ridge_kmeans_full"].sum(0)))),
                             mb_lo=np.nanpercentile(boot[k] - best, 2.5), mb_hi=np.nanpercentile(boot[k] - best, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv(args.out / f"union_audit_binary{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
