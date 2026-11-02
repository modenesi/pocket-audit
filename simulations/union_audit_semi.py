"""Combined forest + ridge audit on the real-covariate (semi-synthetic) replicates (exploratory, 2026-10-07).

Same logic as union_audit_binary.py: re-run only the audit step from each replicate's saved inputs (rows,
outcomes, predictions, true risk, frozen pocket labels) with the replicate's seed and the run's locked settings
(dataset level, SupportRule from the feature config, B = 1999, wild restricted Mammen, symmetric, SE floor,
restricted loss). Units and the protected attribute come from the real data (same loader and seed as the runs).
The planted bias is pf - p0.

Usage: python simulations/union_audit_semi.py --out results/tables [--workers 4]
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))
from union_audit_binary import directions, f1, score  # noqa: E402

R = ROOT / "results/app_20261005"
DATA = {"wids": ("configs/features_wids.yaml", .0125), "uci": ("configs/features_uci.yaml", .025)}
SETS = [("semi_uci", "uci", "inpt10"), ("semi_uci", "uci", "null"), ("semi_wids_enc", "wids", "vent10"),
        ("semi_wids_enc", "wids", "null"), ("semi_wids", "wids", "vent10")]
PARTS = ("sbm", "leiden", "ridge_kmeans")
_REAL: dict = {}


def real(ds):
    if ds not in _REAL:
        from pocket_audit.data import load_config, load_dataset
        from pocket_audit.semisynthetic import SemiConfig
        cfg = DATA[ds][0]
        sc = SemiConfig(config=cfg)
        data = load_dataset(ROOT / cfg, root=ROOT, n_units=sc.n_units, seed=sc.real_seed)
        con = next(c for c in data.contrasts if c.name == data.primary_contrast)
        _REAL[ds] = (data, con, load_config(ROOT / cfg).get("support") or {})
    return _REAL[ds]


def one(task):
    import warnings
    warnings.filterwarnings("ignore")
    from pocket_audit.audit import AuditConfig, SupportRule, run_audit
    batch, ds, cell, rep_dir = task
    data, con, support = real(ds)
    alpha = DATA[ds][1]
    z = np.load(Path(rep_dir) / "audit_inputs.npz")
    seed = json.loads((Path(rep_dir) / "done.json").read_text())["seed"]
    rows = z["rows"]
    prot = data.protected.iloc[rows].reset_index(drop=True)
    unit = np.asarray(data.unit)[rows]
    focal = prot[con.attribute].to_numpy() == con.focal
    bias = z["pf"] - z["p0"]
    acfg = AuditConfig(support=SupportRule(**support), n_boot=1999, alpha=alpha, seed=seed,
                       method="wild_restricted_mammen", tails="symmetric", restricted_loss=True, se_floor=True)
    out = dict(batch=batch, cell=cell, rep=int(Path(rep_dir).name.split("_")[1]))
    dirs = {}
    for p in PARTS:
        lab = z[f"labels_{p}"]
        table, summ = run_audit(z["y"], {"model": z["pf"]}, lab, unit, prot, [con], acfg)
        M = summ.get("primary", {}).get("M")
        if M is None or len(np.atleast_1d(M)) == 0:
            dirs[(p, 1)] = dirs[(p, 2)] = (np.zeros(len(lab), int), False)
        else:
            dirs[(p, 1)] = directions(table, lab, M, alpha)
            dirs[(p, 2)] = directions(table, lab, M, alpha / 2)
    for p in PARTS:
        out[f"{p}_full"] = score(dirs[(p, 1)][0], focal, bias)
        out[f"{p}_full_any"] = dirs[(p, 1)][1]
    for fm in ("sbm", "leiden"):
        a, b = dirs[(fm, 2)][0], dirs[("ridge_kmeans", 2)][0]
        conflict = (a != 0) & (b != 0) & (a != b)
        u = np.where(a != 0, a, b)
        u[conflict] = 0
        out[f"{fm}+ridge_half"] = score(u, focal, bias, conflict)
        out[f"{fm}+ridge_half_any"] = dirs[(fm, 2)][1] or dirs[("ridge_kmeans", 2)][1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-reps", type=int, default=10 ** 6)
    args = ap.parse_args()
    tasks = []
    for batch, ds, cell in SETS:
        dirs = sorted(glob.glob(str(R / batch / cell / "rep_[0-9][0-9][0-9]")))[:args.max_reps]
        tasks += [(batch, ds, cell, d) for d in dirs]
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(one, tasks, chunksize=2))
    df = pd.DataFrame(res)
    df.to_pickle(args.out / "union_audit_semi_reps.pkl")
    rng = np.random.default_rng(20261007)
    keys = [f"{p}_full" for p in PARTS] + ["sbm+ridge_half", "leiden+ridge_half"]
    rows = []
    for (batch, cell), g in df.groupby(["batch", "cell"], sort=False):
        arr = {k: np.stack(g[k].to_numpy()) for k in keys}
        draws = rng.integers(0, len(g), (2000, len(g)))
        cnt = np.stack([np.bincount(x, minlength=len(g)) for x in draws])
        boot = {k: f1(np.einsum("br,rk->bk", cnt, v)) for k, v in arr.items()}
        best = np.maximum(boot["sbm_full"], boot["ridge_kmeans_full"])
        stored = pd.concat(pd.read_csv(f) for f in glob.glob(str(R / batch / cell / "rep_*/metrics.csv")))
        for k in keys:
            est = float(f1(arr[k].sum(0)))
            st = stored[stored.partition == k.replace("_full", "")] if k.endswith("_full") else None
            stored_f1 = float(f1(st[st.rep.isin(g.rep)][["person_tp", "person_wrong_sign", "person_false_unaffected",
                                                        "person_fn"]].sum().to_numpy())) if st is not None else np.nan
            rows.append(dict(batch=batch, cell=cell, method=k, reps=len(g), f1=est,
                             lo=np.nanpercentile(boot[k], 2.5), hi=np.nanpercentile(boot[k], 97.5), stored_f1=stored_f1,
                             any_flag_rate=g[k + "_any"].mean(),
                             minus_best_sbm_ridge=est - max(float(f1(arr["sbm_full"].sum(0))),
                                                            float(f1(arr["ridge_kmeans_full"].sum(0)))),
                             mb_lo=np.nanpercentile(boot[k] - best, 2.5), mb_hi=np.nanpercentile(boot[k] - best, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "union_audit_semi.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
