"""Smoke ladder: small, inspectable simulations of discover -> audit, from simplest upward.

Every cell uses an ADDITIVE probability distortion, so each affected person's planted
expected residual is exactly ``delta`` and effects are matched across designs by construction:

    x1..xp ~ N(0,1);  logit p0 = a + r(x)   (a sets 8% prevalence)
    S ~ Bernoulli(.2), independent of x and hospital (cell T2: S = 1 whenever x1 > q.90)
    R = feature region;  pf = p0 + delta * S * R;  Y ~ Bernoulli(p0)
    b_i = pf_i - p0_i (= delta for affected people, 0 otherwise); affected = S & R & delta > 0

Risk functions: toy r = x1 (+ 1 noise feature); graded r = x1 + .5 x2 + .25 x3 (+ 20 noise);
nonlinear r = x1 + .8 (x2^2 - 1) (+ 20 noise) with the region R = top 20% of |x2|, which a
linear relevance score cannot see. Hospitals (H cells): 150 units with lognormal sizes and the
package's shared-uniform outcome dependence (shared fraction .2), groups balanced across units.

Stages: ``oracle`` audits fixed, known pockets (no discovery) to check truth, support policy and
person metrics; ``learned`` runs the full pipeline. Every learned replicate also freezes "sbm_cut",
the tree cut of the same SBM fit (most projected-supported pockets over cuts at varying depth),
so "sbm" (one level) and "sbm_cut" are exactly paired. Paired ablations reuse T1's data: T1L is the
legacy selector (projected 100 per group = the audit floor, counts only), T1C adds the 150 margin,
and T1 (default) also projects the audit's expected-event rule. Run from the repository root:

    python simulations/smoke_ladder.py oracle  --out-dir results/<run> --reps 200
    python simulations/smoke_ladder.py learned --out-dir results/<run> --cells T0 T1 --reps 0 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import norm  # noqa: E402

N, N_INT, PREV_S, PREV_Y = 20000, 200000, 0.2, 0.08
Q = {p: float(norm.ppf(p)) for p in (.2, .4, .6, .7, .8, .9)}

# cell -> design. "data" names the cell whose simulated data (same seeds) it reuses.
CELLS = {
    "T0":  dict(risk="toy", delta=0.0, region="x1>q80"),
    "T1":  dict(risk="toy", delta=0.10, region="x1>q80"),
    "T1L": dict(risk="toy", delta=0.10, region="x1>q80", data="T1", min_projected=100, project_events=False),
    "T1C": dict(risk="toy", delta=0.10, region="x1>q80", data="T1", min_projected=150, project_events=False),
    "T1N": dict(risk="toy", delta=0.10, region="x1>q80", n=50000),      # same design, 2.5x the rows
    "T2":  dict(risk="toy", delta=0.10, region="x1>q70", focal_only="x1>q90"),
    "H0":  dict(risk="toy", delta=0.0, region="x1>q80", units=150),
    "H1":  dict(risk="toy", delta=0.10, region="x1>q80", units=150),
    "G1":  dict(risk="graded", delta=0.10, region="x1>q80"),
    "N1":  dict(risk="nonlinear", delta=0.10, region="|x2|>top20"),   # two disjoint tails of 10% each
    "N2":  dict(risk="nonlinear", delta=0.10, region="x2>q80"),       # one tail: invisible to a linear score
    # single-group island: binary x3 (10% of rows, +1.5 logit risk) where everyone is focal; bias in the
    # island (unidentifiable within it) and in the top 20% of x1 elsewhere (identifiable)
    "T3":  dict(risk="island", delta=0.10, region="x1>q80|x3", focal_only="x3"),
    # development cells at application scale (n = 50,000: ~3,000 focal audit people, like WiDS)
    "D0":  dict(risk="toy", delta=0.0, region="x1>q80", n=50000),
    "D1":  dict(risk="toy", delta=0.10, region="x1>q80", n=50000),
    "D1E": dict(risk="toy", delta=0.10, region="x1>q80", n=50000, data="D1", project_events=True),
    "D1m": dict(risk="toy", delta=0.05, region="x1>q80", n=50000),
    "D3":  dict(risk="island", delta=0.10, region="x1>q80|x3", focal_only="x3", n=50000),
    "DH0": dict(risk="toy", delta=0.0, region="x1>q80", units=150, n=50000),
    "DH1": dict(risk="toy", delta=0.10, region="x1>q80", units=150, n=50000),
    "DN2": dict(risk="nonlinear", delta=0.10, region="x2>q80", n=50000),
    # 2026-10-06 additions. Masking: +delta for focal people with x1 > q80 and x2 > 0, -delta with x1 > q80 and
    # x2 <= 0, so the global disparity is about zero while two pockets carry opposite gaps (graded risk, so x2 is
    # outcome-relevant). S-X: focal membership rises with x1 (logit slope .8, prevalence kept at .2).
    "DM0": dict(risk="graded", delta=0.0, region="mask", n=50000),
    "DM1": dict(risk="graded", delta=0.10, region="mask", n=50000),
    "DX0": dict(risk="toy", delta=0.0, region="x1>q80", n=50000, s_x=0.8),
    "DX1": dict(risk="toy", delta=0.10, region="x1>q80", n=50000, s_x=0.8),
}
N_FEATURES = {"toy": 2, "graded": 23, "nonlinear": 22, "island": 3}


def features(kind: str, Z: np.ndarray) -> np.ndarray:
    """Map standard normal draws to the cell's features (island: x3 = 1{z3 > q.90}, a 10% binary)."""
    if kind == "island":
        Z = Z.copy()
        Z[:, 2] = (Z[:, 2] > Q[.9]).astype(float)
    return Z


def risk_score(X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "toy":
        return X[:, 0]
    if kind == "graded":
        return X[:, 0] + .5 * X[:, 1] + .25 * X[:, 2]
    if kind == "nonlinear":
        return X[:, 0] + .8 * (X[:, 1] ** 2 - 1)
    if kind == "island":
        return X[:, 0] + 1.5 * X[:, 2]
    raise ValueError(kind)


def region_mask(X: np.ndarray, spec: str) -> np.ndarray:
    if spec == "mask":            # signed: +1 / -1 halves of the top-20% x1 region, split by the sign of x2
        return np.where(X[:, 0] > Q[.8], np.where(X[:, 1] > 0, 1, -1), 0)
    if spec == "x3":
        return X[:, 2] == 1
    if spec == "x1>q80|x3":
        return (X[:, 0] > Q[.8]) | (X[:, 2] == 1)
    return {"x1>q80": X[:, 0] > Q[.8], "x1>q70": X[:, 0] > Q[.7], "x1>q90": X[:, 0] > Q[.9],
            "|x2|>top20": np.abs(X[:, 1]) > Q[.9], "x2>q80": X[:, 1] > Q[.8]}[spec]   # P(|Z| > q.90) = .20


_INTERCEPTS: dict = {}


def intercept(kind: str) -> float:
    if kind not in _INTERCEPTS:
        Z = features(kind, np.random.default_rng(12345).standard_normal((400000, N_FEATURES[kind])))
        s = risk_score(Z, kind)
        lo, hi = -10., 5.
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if expit(mid + s).mean() < PREV_Y else (lo, mid)
        _INTERCEPTS[kind] = (lo + hi) / 2
    return _INTERCEPTS[kind]


def distort(p0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Additive distortion, capped at .995 (rare extreme-risk rows get a gap slightly below delta). Negative shifts
    (masking cell only) stop at min(p0, .005), so a prediction never goes below zero or above its true risk."""
    up = np.minimum(p0 + shift, np.maximum(p0, .995))         # never moves a row that received no shift
    out = np.where(shift < 0, np.maximum(p0 + shift, np.minimum(p0, .005)), up)
    return np.where(shift == 0, p0, out)


def draw(cell: dict, n: int, rng: np.random.Generator, units: int | None = None) -> dict:
    """One sample from the cell's DGP; ``units`` adds hospitals and within-unit dependence."""
    X = features(cell["risk"], rng.standard_normal((n, N_FEATURES[cell["risk"]])))
    p0 = expit(intercept(cell["risk"]) + risk_score(X, cell["risk"]))
    if cell.get("s_x"):           # focal membership depends on x1, overall prevalence kept at PREV_S
        lo, hi = -20., 20.
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if expit(mid + cell["s_x"] * X[:, 0]).mean() < PREV_S else (lo, mid)
        s = rng.random(n) < expit((lo + hi) / 2 + cell["s_x"] * X[:, 0])
    else:
        s = rng.random(n) < PREV_S
    if cell.get("focal_only"):
        s |= region_mask(X, cell["focal_only"])
    R = region_mask(X, cell["region"])
    pf = distort(p0, cell["delta"] * s * R)
    if units is None:
        unit, u = np.arange(n).astype(str), rng.random(n)
    else:
        w = rng.lognormal(0, 1, units)
        uid = rng.choice(units, size=n, p=w / w.sum())
        unit = uid.astype(str)
        shared = rng.random(units)[uid]
        u = np.where(rng.random(n) < .2, shared, rng.random(n))
        if cell.get("s_unit_sd"):     # concentrate the focal group in some hospitals (overall prevalence kept)
            eta = rng.normal(0, cell["s_unit_sd"], units)[uid]
            lo, hi = -40., 40.
            for _ in range(80):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if expit(mid + eta).mean() < PREV_S else (lo, mid)
            s = rng.random(n) < expit((lo + hi) / 2 + eta)
            if cell.get("focal_only"):
                s |= region_mask(X, cell["focal_only"])
            pf = distort(p0, cell["delta"] * s * R)
    y = (u < p0).astype(int)
    bias = pf - p0
    return dict(X=X, p0=p0, pf=pf, s=s, R=R, y=y, unit=unit, bias=bias, affected=s & (bias != 0))


def to_auditdata(d: dict):
    from pocket_audit.data import AuditData, Contrast
    X = pd.DataFrame(d["X"], columns=[f"x{j + 1}" for j in range(d["X"].shape[1])])
    return AuditData(name="smoke", X=X, y=d["y"], unit=d["unit"], row_id=np.arange(len(d["y"])),
                     protected=pd.DataFrame({"S": np.where(d["s"], "1", "0")}),
                     contrasts=[Contrast("S1_vs_S0", "S", "1", "0")], primary_contrast="S1_vs_S0")


def seeds(base: int, cell: str, rep: int) -> tuple[int, int]:
    """(data/pipeline seed, integration seed). Paired arms share their data cell's seeds."""
    key = list(CELLS).index(CELLS[cell].get("data", cell))
    ss = np.random.SeedSequence([base, key, rep]).generate_state(2)
    return int(ss[0]), int(ss[1])


def support_rule(cell: dict):
    from pocket_audit.audit import SupportRule
    if cell.get("units"):
        return SupportRule(min_n=100, min_units=15, min_effective_units=10, max_unit_share=.2)
    return SupportRule(min_n=100, min_units=1, min_effective_units=1)


def truth_table(labels_int: dict, d: dict) -> pd.DataFrame:
    """Population pocket truths from a large independent sample assigned through the frozen maps."""
    out = []
    for name, lab in labels_int.items():
        for c in ["all"] + sorted(int(v) for v in np.unique(lab) if v >= 0):
            m = np.ones(len(lab), bool) if c == "all" else lab == c
            f, r = m & d["s"], m & ~d["s"]
            nf, nr, na = int(f.sum()), int(r.sum()), int((m & d["affected"]).sum())
            t = d["bias"][f].mean() - d["bias"][r].mean() if nf and nr else np.nan
            se = np.sqrt(d["bias"][f].var(ddof=1) / nf + d["bias"][r].var(ddof=1) / nr) if nf > 1 and nr > 1 else np.nan
            out.append(dict(partition=name, pocket=str(c), truth=t, truth_mcse=se, n_int_focal=nf,
                            n_int_reference=nr, n_int_affected=na, exact_null=bool(na == 0 and nf and nr)))
    return pd.DataFrame(out)


def with_tails(results: pd.DataFrame, summary: dict, tails: str) -> pd.DataFrame:
    """Rebuild intervals and flags for one tails rule from a single audit's shared bootstrap draws."""
    out = results.copy()
    for (part, fam), g in out.groupby(["partition", "family"]):
        s = summary[part][fam]
        idx = g.index[g.status == "tested"]
        if not len(idx) or s.get("q") is None or not np.isfinite(s["q"]):
            continue
        lo_q, hi_q = (s["q"], s["q"]) if tails == "symmetric" else (s["q_up"], s["q_dn"])
        out.loc[idx, "ci_low"] = out.loc[idx, "estimate"] - lo_q * out.loc[idx, "se"]
        out.loc[idx, "ci_high"] = out.loc[idx, "estimate"] + hi_q * out.loc[idx, "se"]
    out["flag_positive"] = (out.status == "tested") & (out.ci_low > 0)
    out["flag_negative"] = (out.status == "tested") & (out.ci_high < 0)
    out["flag"] = out.flag_positive | out.flag_negative
    return out


def evaluate(name: str, lab: np.ndarray, d_aud: dict, coords: pd.DataFrame, truth: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Person-level, inference and per-pocket summaries for one partition on the audit rows."""
    from pocket_audit.evaluation import person_metrics
    delta = coords.loc[(coords.partition == name) & (coords.kind == "delta")].copy()
    delta["pocket"] = delta.pocket.astype(str)
    delta = delta.merge(truth.loc[truth.partition == name].drop(columns="partition"), on="pocket",
                        how="left", validate="one_to_one")
    focal, bias, aff = d_aud["s"], d_aud["bias"], d_aud["affected"]
    row = dict(partition=name, **person_metrics(lab, focal, bias, delta))
    loc = delta.loc[delta.pocket != "all"]
    tested = loc.loc[loc.status == "tested"]
    known = np.isin(lab, tested.pocket.astype(int).to_numpy())
    status_of = dict(zip(loc.pocket.astype(int), loc.status + ":" + loc.reason.fillna("").astype(str)))
    reason_person = np.array(["abstain" if c < 0 else status_of.get(int(c), "absent").split(":", 1)[1] or "tested"
                              for c in lab], dtype=str)
    missed = aff & ~known

    def n_missed(*words):   # affected people without a local test whose pocket failed for one of these reasons
        hit = np.zeros(len(lab), bool)
        for w in words:
            hit |= np.char.find(reason_person, w) >= 0
        return int((missed & hit).sum())
    row.update(
        n_pockets=int(len(loc)), n_local_tested=int(len(tested)),
        n_single_group=int(((loc.n_focal == 0) | (loc.n_reference == 0)).sum()),
        aff_known=int((aff & known).sum()),
        recall_supported=row["person_tp"] / (aff & known).sum() if (aff & known).any() else np.nan,
        aff_unknown_abstain=int((aff & (lab < 0)).sum()),
        aff_unknown_count=n_missed("n<"), aff_unknown_expected=n_missed("expected"),
        aff_unknown_units=n_missed("units", "share"), aff_unknown_singlegroup=n_missed("empty"),
        focal_unknown_share=float((focal & ~known).sum() / focal.sum()),
    )
    # inference counters over tested delta coordinates with a defined truth (local and global)
    t = delta.loc[(delta.status == "tested") & delta.truth.notna()]
    cov = (t.ci_low <= t.truth) & (t.truth <= t.ci_high)
    null = t.exact_null.astype(bool)
    row.update(n_tested=int(len(t)), n_tested_null=int(null.sum()),
               all_covered=bool(cov.all()) if len(t) else np.nan,
               lower_miss=bool((t.ci_low > t.truth).any()) if len(t) else np.nan,
               upper_miss=bool((t.ci_high < t.truth).any()) if len(t) else np.nan,
               any_null_flag=bool(t.loc[null, "flag"].any()) if null.any() else np.nan,
               any_local_flag=bool(tested.flag.any()) if len(tested) else False,
               global_flag=bool(delta.loc[delta.pocket == "all", "flag"].any()))
    # pocket table (abstention as pocket -1)
    pk = []
    for c in sorted(np.unique(lab)):
        m = lab == c
        info = delta.loc[delta.pocket == str(int(c))]
        r = info.iloc[0] if len(info) else None
        direction = 0 if r is None or r.status != "tested" else int(r.ci_low > 0) - int(r.ci_high < 0)
        pk.append(dict(partition=name, pocket=int(c), n=int(m.sum()), n_focal=int((m & focal).sum()),
                       n_reference=int((m & ~focal).sum()), n_affected=int((m & aff).sum()),
                       n_correct=int((m & aff & (direction == np.sign(bias)) & (direction != 0)).sum()),
                       status="abstain" if c < 0 else (r.status if r is not None else "absent"),
                       reason="" if r is None or pd.isna(r.reason) else r.reason,
                       truth=np.nan if r is None else r.truth, estimate=np.nan if r is None else r.estimate,
                       ci_low=np.nan if r is None else r.ci_low, ci_high=np.nan if r is None else r.ci_high,
                       direction=direction))
    return row, pd.DataFrame(pk)


# ----------------------------------------------------------------------------- learned maps
def run_learned(task) -> tuple:
    opts, cell_name, rep = task
    import graph_tool.all as gt
    from pocket_audit.data import to_numeric_matrix
    from pocket_audit.pipeline import PipelineConfig, assign_partition, audit, discover
    gt.openmp_set_num_threads(1)
    cell = CELLS[cell_name]
    dest = Path(opts["out_dir"]) / "learned" / cell_name / f"rep_{rep:03d}"
    if (dest / "done.json").exists():
        return cell_name, rep, "cached", 0.0
    tmp = dest.with_name(dest.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.perf_counter()
    try:
        seed, int_seed = seeds(opts["base_seed"], cell_name, rep)
        d = draw(cell, cell.get("n", N), np.random.default_rng(seed), units=cell.get("units"))
        data = to_auditdata(d)
        pc = PipelineConfig(seed=seed, n_jobs=1, alpha=opts["alpha"], n_boot=opts["n_boot"],
                            inference="wild_restricted_mammen", tails="equal",
                            min_projected_per_group=cell.get("min_projected", opts["min_projected"]),
                            project_expected_events=cell.get("project_events", opts["project_events"]),
                            sbm_tree_cut=True, min_graph_profiled_share=.8, sbm_n_init=3, graph_method=opts["graph"],
                            abstain_quantile=opts.get("abstain_quantile", .05),
                            restricted_loss=opts.get("restricted_loss", False), se_floor=opts.get("se_floor", False),
                            extra_baselines=opts.get("extra_baselines", False), support=support_rule(cell),
                            leiden_auto=opts.get("leiden_auto", False), run_sbm=not opts.get("no_sbm", False),
                            **({} if opts.get("max_features") is None else {"max_features": opts["max_features"]}))
        frozen = discover(data, pc, preds={"model": d["pf"]})
        t_disc = time.perf_counter() - t0
        # abstention arms: the same frozen maps re-thresholded on discovery calibration rows (outcome-free)
        for q in opts.get("abstain_arms") or []:
            import copy
            from pocket_audit.assignment import calibrate_thresholds
            for name in ("sbm", "sbm_cut", "leiden"):
                if name in frozen["partitions"]:
                    pm = copy.deepcopy(frozen["partitions"][name])
                    calibrate_thresholds(pm, frozen["sig"][frozen["calib_rows"]], q)
                    frozen["partitions"][f"{name}@a{q:g}"] = pm
        res = audit(data, frozen)
        aud = np.flatnonzero(frozen["role"] == "audit")
        Xnum = to_numeric_matrix(data.X)
        labels = {n: assign_partition(frozen, n, aud, Xnum) for n in frozen["partitions"]}
        # integration truth through the same frozen maps (independent rows; dependence does not move means)
        di = draw(cell, N_INT, np.random.default_rng(int_seed))
        Xi = to_numeric_matrix(to_auditdata(di).X)
        fi = dict(frozen)
        fi["sig"] = frozen["forest"].apply(Xi).astype(np.int32)
        truth = truth_table({n: assign_partition(fi, n, np.arange(N_INT), Xi) for n in frozen["partitions"]}, di)
        d_aud = {k: d[k][aud] for k in ("s", "bias", "affected", "unit")}
        rows, pockets, coords = [], [], []
        for tails in ("symmetric", "equal"):
            co = with_tails(res["results"], res["summary"], tails)
            coords.append(co.assign(tails=tails))
            for n, lab in labels.items():
                r, p = evaluate(n, lab, d_aud, co, truth)
                rows.append({**r, "tails": tails})
                pockets.append(p.assign(tails=tails))
        sbm = frozen["reports"].get("sbm", {})
        meta = dict(cell=cell_name, rep=rep, seed=seed, integration_seed=int(int_seed), design=cell,
                    pipeline={k: v for k, v in asdict(pc).items() if k != "support"} | {"support": asdict(pc.support)},
                    sbm_level=sbm.get("chosen_level"), sbm_status=sbm.get("status"),
                    n_pockets=frozen["n_pockets"], sbm_cut=frozen["reports"].get("sbm_cut"),
                    forest_oob=frozen["reports"].get("forest"),
                    leiden_auto={k: v for k, v in frozen["reports"].get("leiden_auto", {}).items() if k != "levels"},
                    timings=frozen["reports"]["timings"],
                    seconds_discovery=round(t_disc, 1), seconds=round(time.perf_counter() - t0, 1))
        metrics = pd.DataFrame(rows)
        for k in ("cell", "rep", "seed", "sbm_level", "sbm_status"):
            metrics.insert(0, k, meta[k])
        metrics.insert(5, "k_matched", meta["n_pockets"])     # SBM's profiled pocket count, reused by the baselines
        metrics.to_csv(tmp / "metrics.csv", index=False)
        pd.concat(pockets).to_csv(tmp / "pockets.csv", index=False)
        pd.concat(coords).to_csv(tmp / "coordinates.csv", index=False)
        (tmp / "audit_summary.json").write_text(json.dumps(
            {k: {f: {kk: vv for kk, vv in v.items() if kk not in ("M", "Mp", "Mn")} for f, v in fam.items()}
             for k, fam in res["summary"].items()}, default=float))
        truth.to_csv(tmp / "truth.csv", index=False)
        if "levels" in sbm:
            sbm["levels"].to_csv(tmp / "sbm_levels.csv", index=False)
        np.savez_compressed(tmp / "audit_inputs.npz", y=data.y[aud], pf=d["pf"][aud], **d_aud,
                            x=d["X"][aud], **{f"labels_{k}": v for k, v in labels.items()})
        (tmp / "done.json").write_text(json.dumps(meta, indent=1, default=str))
        tmp.rename(dest)
        return cell_name, rep, "completed", meta["seconds"]
    except Exception:
        (tmp / "failure.txt").write_text(traceback.format_exc())
        raise


# ----------------------------------------------------------------------------- oracle maps
ORACLE_EDGES = {"T0": [.2, .4, .6, .8], "T1": [.2, .4, .6, .8], "T2": [.2, .4, .7, .9],
                "H0": [.2, .4, .6, .8], "H1": [.2, .4, .6, .8]}


def run_oracle(task) -> list[dict]:
    """Fixed known pockets (quantile bins of x1) on the audit split; no discovery."""
    opts, cell_name, rep = task
    from pocket_audit.audit import AuditConfig, run_audit
    from pocket_audit.splits import allocate_units
    cell = CELLS[cell_name]
    seed, int_seed = seeds(opts["base_seed"] + 1, cell_name, rep)       # disjoint from learned seeds
    d = draw(cell, N, np.random.default_rng(seed), units=cell.get("units"))
    aud = allocate_units(d["unit"], (.4, .3, .3), seed=seed) == "audit"
    edges = [Q[p] for p in ORACLE_EDGES[cell_name]]
    lab = np.searchsorted(edges, d["X"][aud, 0]).astype(np.int32)
    acfg = AuditConfig(support=support_rule(cell), n_boot=opts["n_boot"], alpha=opts["alpha"], seed=seed,
                       method="wild_restricted_mammen")
    table, _ = run_audit(d["y"][aud], {"model": d["pf"][aud]}, lab, d["unit"][aud],
                         pd.DataFrame({"S": np.where(d["s"][aud], "1", "0")}),
                         to_auditdata(d).contrasts, acfg)
    table.insert(0, "partition", "oracle")
    di = draw(cell, N_INT, np.random.default_rng(int_seed))
    truth = truth_table({"oracle": np.searchsorted(edges, di["X"][:, 0]).astype(np.int32)}, di)
    r, pk = evaluate("oracle", lab, {k: d[k][aud] for k in ("s", "bias", "affected", "unit")}, table, truth)
    r.update(cell=cell_name, rep=rep, seed=seed)
    pk.insert(0, "rep", rep)
    pk.insert(0, "cell", cell_name)
    return [r], pk


# ----------------------------------------------------------------------------- driver
def provenance(args) -> dict:
    files = sorted((ROOT / "src/pocket_audit").glob("*.py")) + [Path(__file__).resolve()]
    h = hashlib.sha256()
    for p in files:
        h.update(p.read_bytes())
    import importlib.metadata as md
    return dict(command=sys.argv, args={k: str(v) for k, v in vars(args).items()}, source_sha256=h.hexdigest(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                   cwd=ROOT, text=True),
                versions={m: md.version(m) for m in ("numpy", "pandas", "scipy", "scikit-learn", "igraph", "leidenalg")},
                python=sys.version, cells=CELLS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["oracle", "learned"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", choices=CELLS, default=None)
    ap.add_argument("--reps", type=int, nargs="+", default=[0, 5], help="START STOP (learned) or COUNT (oracle)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=999)
    ap.add_argument("--alpha", type=float, default=.025)
    ap.add_argument("--min-projected", type=int, default=150)
    ap.add_argument("--no-project-events", dest="project_events", action="store_false",
                    help="selector projects counts only (legacy); default also projects expected events")
    ap.add_argument("--graph", choices=["approx", "exact_cpu", "exact_gpu"], default="approx")
    ap.add_argument("--base-seed", type=int, default=2026100410)
    ap.add_argument("--abstain-quantile", type=float, default=.05, help="PipelineConfig.abstain_quantile")
    ap.add_argument("--restricted-loss", action="store_true", help="AuditConfig.restricted_loss for theta/psi")
    ap.add_argument("--se-floor", action="store_true", help="AuditConfig.se_floor for disparity coordinates")
    ap.add_argument("--extra-baselines", action="store_true", help="PipelineConfig.extra_baselines (cem_top, disparity_bins)")
    ap.add_argument("--max-features", type=lambda v: v if v in ("sqrt", "log2") else float(v), default=None,
                    help="sensitivity override of the similarity forest's max_features (default: PipelineConfig)")
    ap.add_argument("--leiden-auto", action="store_true",
                    help="proposal: also freeze leiden_auto (Leiden at the support-rule resolution of a fixed grid)")
    ap.add_argument("--no-sbm", action="store_true",
                    help="skip the nested SBM; comparators then use K = 4 (for leiden_auto smoke runs)")
    ap.add_argument("--abstain-arms", type=float, nargs="*", default=[],
                    help="also audit sbm/sbm_cut/leiden re-thresholded at these abstention quantiles (paired)")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n!.gitignore\n")
    opts = dict(out_dir=str(out), base_seed=args.base_seed, n_boot=args.n_boot, alpha=args.alpha,
                min_projected=args.min_projected, project_events=args.project_events, graph=args.graph,
                abstain_arms=args.abstain_arms, abstain_quantile=args.abstain_quantile,
                restricted_loss=args.restricted_loss, se_floor=args.se_floor, extra_baselines=args.extra_baselines,
                max_features=args.max_features, leiden_auto=args.leiden_auto, no_sbm=args.no_sbm)
    (out / f"provenance_{args.stage}_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(provenance(args), indent=1, default=str))
    if args.stage == "oracle":
        cells = args.cells or list(ORACLE_EDGES)
        tasks = [(opts, c, r) for c in cells for r in range(args.reps[-1])]
        rows, pockets = [], []
        with ProcessPoolExecutor(args.workers) as pool:
            for fut in as_completed([pool.submit(run_oracle, t) for t in tasks]):
                r, p = fut.result()
                rows += r
                pockets.append(p)
        (out / "oracle").mkdir(exist_ok=True)
        pd.DataFrame(rows).sort_values(["cell", "rep"]).to_csv(out / "oracle/metrics.csv", index=False)
        pd.concat(pockets).sort_values(["cell", "rep", "pocket"]).to_csv(out / "oracle/pockets.csv", index=False)
        print(f"oracle: {len(rows)} replicate rows written to {out / 'oracle'}")
        return
    cells = args.cells or list(CELLS)
    start, stop = args.reps if len(args.reps) == 2 else (0, args.reps[0])
    tasks = [(opts, c, r) for r in range(start, stop) for c in cells]      # interleave cells
    failures = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_learned, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                print(*fut.result(), flush=True)
            except Exception as e:  # recorded in failure.txt inside the partial directory
                failures.append(dict(cell=futs[fut][1], rep=futs[fut][2], error=repr(e)))
                print("FAILED", futs[fut][1:], repr(e), flush=True)
    (out / "learned_failures.json").write_text(json.dumps(failures, indent=1))


if __name__ == "__main__":
    main()
