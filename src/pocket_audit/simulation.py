"""Simulation: known local calibration disparity, full-pipeline replicates.

DGP. x1, x2 relevant, ``p_noise`` irrelevant features, all N(0,1). Protected S ~ Bernoulli
with logit(P(S=1)) = logit(prevalence_s) + s_x_assoc * x1. True risk
logit p0 = a + x1 + 0.8 (x2^2 - 1), intercept a chosen for ~8% prevalence.
Y ~ Bernoulli(p0), optionally dependent within units via a shared uniform
(Y = 1{U < p0}, U = unit uniform w.p. ``shared_frac``), which keeps E[Y|X,S] = p0.
The assessed prediction is p_f = expit(logit p0 + tau * S * R(X)), where R is a
region with ~10% prevalence on ``region_feature``. At tau = 0 every mean-residual
disparity is exactly zero; Brier contrasts are not, so they are judged against their
own true values.

Graded relevance: with ``relevance``
= (c1, ..., cm), logit p0 = a + sum_j c_j x_j instead, followed by ``p_noise`` irrelevant
features. ``TOY`` is one relevant and one irrelevant feature; ``GRADED`` is strong,
moderate, and weak features. The default (empty ``relevance``) reproduces the original DGP
draw for draw.

Each replicate regenerates data, refits discovery, selects the resolution, and
audits. Truths for the learned map come from a large independent integration sample.
Besides coverage and flags, each partition is scored on localization: the share of audit
rows in flagged pockets that lie in the biased region (precision), the share of the
region's audit rows that lie in flagged pockets (recall), and the number and size of pockets.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .assignment import ABSTAIN
from .data import AuditData, Contrast, to_numeric_matrix
from .pipeline import PipelineConfig, assign_partition, audit, discover
from .sim_summary import LOCALIZATION, summarize  # noqa: F401  (re-exported for the CLI)

Z90 = 1.2815515655446004
TOY = dict(relevance=(1.0,), p_noise=1)
GRADED = dict(relevance=(1.0, 0.5, 0.25), p_noise=20)


@dataclass
class SimConfig:
    n: int = 20000
    p_noise: int = 20
    prevalence_s: float = 0.2
    s_unit_sd: float = 0.0         # protected-group concentration across independent units
    s_x_assoc: float = 0.0
    tau: float = 0.0
    region_feature: str = "x1"      # "x1" (task-relevant) or "x3" (irrelevant: a known hard case)
    n_units: int | None = None      # None: each row is its own unit
    unit_size_sigma: float = 1.0
    shared_frac: float = 0.0
    target_prevalence: float = 0.08
    n_integration: int = 200000
    relevance: tuple = ()           # () = original DGP; else linear coefficients on x1..xm


def _n_relevant(sc: SimConfig) -> int:
    return len(sc.relevance) or 2


def _risk_score(X: pd.DataFrame, relevance: tuple) -> np.ndarray:
    if relevance:
        return sum(c * X[f"x{j + 1}"].to_numpy() for j, c in enumerate(relevance))
    return X["x1"].to_numpy() + 0.8 * (X["x2"].to_numpy() ** 2 - 1)


@lru_cache(maxsize=None)
def _intercept(target: float, relevance: tuple = ()) -> float:
    rng = np.random.default_rng(12345)
    m = len(relevance) or 2
    Z = rng.standard_normal((m, 400000))
    score = _risk_score(pd.DataFrame(Z.T, columns=[f"x{j + 1}" for j in range(m)]), relevance)
    lo, hi = -10.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mid + score).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _features(n, sc: SimConfig, rng):
    X = rng.standard_normal((n, _n_relevant(sc) + sc.p_noise))
    return pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(X.shape[1])])


def _truth_inputs(X: pd.DataFrame, sc: SimConfig, a: float, rng):
    s = rng.random(len(X)) < expit(logit(sc.prevalence_s) + sc.s_x_assoc * X["x1"].to_numpy())
    lp0 = a + _risk_score(X, sc.relevance)
    region = X[sc.region_feature].to_numpy() > Z90
    p0 = expit(lp0)
    pf = expit(lp0 + sc.tau * s * region)
    return s, p0, pf, region


def simulate(sc: SimConfig, seed: int) -> tuple[AuditData, dict]:
    rng = np.random.default_rng(seed)
    a = _intercept(sc.target_prevalence, tuple(sc.relevance))
    X = _features(sc.n, sc, rng)
    s, p0, pf, region = _truth_inputs(X, sc, a, rng)
    if sc.n_units is None:
        unit = np.arange(sc.n).astype(str)
        u = rng.random(sc.n)
    else:
        w = rng.lognormal(0, sc.unit_size_sigma, sc.n_units)
        uid = rng.choice(sc.n_units, size=sc.n, p=w / w.sum())
        unit = uid.astype(str)
        shared = rng.random(sc.n_units)[uid]
        u = np.where(rng.random(sc.n) < sc.shared_frac, shared, rng.random(sc.n))
    if sc.s_unit_sd:
        if sc.n_units is None:
            raise ValueError("s_unit_sd requires clustered units")
        eta = rng.normal(0, sc.s_unit_sd, sc.n_units)
        offset = eta[uid] + sc.s_x_assoc * X.x1.to_numpy()
        lo, hi = -40., 40.
        for _ in range(80):
            mid = (lo + hi)/2
            if expit(mid + offset).mean() < sc.prevalence_s:
                lo = mid
            else:
                hi = mid
        s = rng.random(sc.n) < expit((lo+hi)/2 + offset)
        pf = expit(a + _risk_score(X, sc.relevance) + sc.tau*s*region)
    y = (u < p0).astype(int)
    data = AuditData(
        name="simulation", X=X, y=y, unit=unit, row_id=np.arange(sc.n),
        protected=pd.DataFrame({"S": np.where(s, "1", "0")}),
        contrasts=[Contrast("S1_vs_S0", "S", "1", "0")], primary_contrast="S1_vs_S0",
    )
    return data, {"pf": pf, "p0": p0, "region": region, "intercept": a, "bias": pf-p0, "affected": s & region & (sc.tau != 0)}


def true_values(frozen: dict, sc: SimConfig, a: float, seed: int) -> pd.DataFrame:
    """Population delta (model), theta for each partition's frozen map."""
    rng = np.random.default_rng(seed)
    X = _features(sc.n_integration, sc, rng)
    s, p0, pf, _ = _truth_inputs(X, sc, a, rng)
    Xnum = to_numeric_matrix(X)
    frozen = dict(frozen)
    frozen["sig"] = frozen["forest"].apply(Xnum).astype(np.int32)
    rows = np.arange(len(X))
    resid = pf - p0                       # E[p - Y | X, S]
    brier = p0 * (1 - p0) + resid ** 2    # E[(p - Y)^2 | X, S]
    out = []
    for name in frozen["partitions"]:
        pocket = assign_partition(frozen, name, rows, Xnum)
        for c in ["all"] + sorted(int(v) for v in np.unique(pocket) if v >= 0):
            m = np.ones(len(X), bool) if c == "all" else pocket == c
            d = resid[m & s].mean() - resid[m & ~s].mean() if (m & s).any() and (m & ~s).any() else np.nan
            a1, a0 = resid[m & s], resid[m & ~s]
            mcse = np.sqrt(a1.var(ddof=1)/len(a1) + a0.var(ddof=1)/len(a0)) if len(a1)>1 and len(a0)>1 else np.nan
            out.append({"partition": name, "truth_mcse": mcse, "kind": "delta", "model": "model", "contrast": "S1_vs_S0", "pocket": c, "truth": d, "exact_null": sc.tau == 0})
            if c != "all":
                out.append({"partition": name, "kind": "theta", "model": "model", "contrast": "", "pocket": c,
                            "truth": brier[m].mean() - brier.mean()})
    return pd.DataFrame(out)


def localization(pocket: np.ndarray, region: np.ndarray, delta: pd.DataFrame) -> dict:
    """Pocket-level scores for one partition on audit rows. ``delta``: that partition's delta
    rows for the assessed model (pocket, supported, flag)."""
    pk = delta[delta.pocket != "all"]
    flagged = pk.loc[pk.flag.astype(bool), "pocket"].astype(int).to_numpy()
    supported = pk.loc[pk.supported.astype(bool), "pocket"].astype(int).to_numpy()
    in_flagged = np.isin(pocket, flagged)
    assigned = pocket != ABSTAIN
    sizes = np.bincount(pocket[assigned]) if assigned.any() else np.array([], int)
    sizes = sizes[sizes > 0]
    purity = [region[pocket == c].mean() for c in supported if (pocket == c).any()]
    return {
        "flag_global": bool(delta.loc[delta.pocket == "all", "flag"].astype(bool).any()),
        "flag_pocket": bool(len(flagged)),
        "loc_precision": float(region[in_flagged].mean()) if in_flagged.any() else np.nan,
        "loc_recall": float(in_flagged[region].mean()) if region.any() else np.nan,
        "best_supported_purity": float(max(purity)) if purity else np.nan,
        "n_pockets": int(len(sizes)),
        "n_supported_pockets": int(len(supported)),
        "median_pocket_size": float(np.median(sizes)) if len(sizes) else np.nan,
        "abstain_share": float((~assigned).mean()),
    }


def replicate_metrics(data: AuditData, info: dict, frozen: dict, res: dict, truth: pd.DataFrame,
                      sc: SimConfig) -> list[dict]:
    """One row per partition and estimand: coverage, flags, and (delta) localization."""
    r = res["results"].merge(truth, on=["partition", "kind", "model", "contrast", "pocket"], how="left", validate="many_to_one")
    aud = np.flatnonzero(frozen["role"] == "audit")
    Xnum = to_numeric_matrix(data.X)
    region = np.asarray(info["region"])[aud]
    rows = []
    for name, grp in r.groupby("partition"):
        loc = localization(assign_partition(frozen, name, aud, Xnum), region, grp[grp.kind == "delta"])
        for kind in ["delta", "theta"]:
            g = grp[(grp.kind == kind) & (grp.status == "tested")]
            covered = ((g.ci_low <= g.truth) & (g.truth <= g.ci_high)).to_numpy()
            null = g.exact_null.eq(True).to_numpy(dtype=bool) if kind == "delta" else np.zeros(len(g), bool)
            row = {
                "partition": name, "kind": kind, "tau": sc.tau,
                "n_supported": len(g), "n_local_tested": int((g.pocket != "all").sum()), "all_covered": bool(covered.all()) if len(g) else np.nan,
                "any_null_flag": bool((g.flag.to_numpy() & null).any()) if null.any() else np.nan,
                "any_flag": bool(g.flag.any()) if len(g) else False,
                "lower_endpoint_miss": bool((g.ci_low > g.truth).any()) if len(g) else np.nan,
                "upper_endpoint_miss": bool((g.ci_high < g.truth).any()) if len(g) else np.nan,
                "n_degenerate": int(((grp.kind == kind) & (grp.status == "degenerate")).sum()),
                "p_global": res["summary"][name]["primary" if kind == "delta" else "secondary"]["p_global"],
                "mean_ci_width": float((g.ci_high - g.ci_low).mean()) if len(g) else np.nan,
                "sbm_status": frozen["reports"].get("sbm", {}).get("status"),
            }
            if kind == "delta":
                row.update(loc)
                from .evaluation import person_metrics
                labels = assign_partition(frozen, name, aud, Xnum)
                con = next(c for c in data.contrasts if c.name == data.primary_contrast)
                focal = data.protected[con.attribute].to_numpy()[aud] == con.focal
                row.update(person_metrics(labels, focal, np.asarray(info.get("bias", info["pf"]-info["p0"]))[aud], grp[grp.kind == "delta"]))
            rows.append(row)
    return rows


def run_replicate(sc: SimConfig, pc: PipelineConfig, rep: int, base_seed: int = 0) -> list[dict]:
    t0 = time.perf_counter()
    seed = base_seed + 1000 * rep
    data, truth_info = simulate(sc, seed)
    pc = PipelineConfig(**{**asdict(pc), "seed": seed, "support": pc.support})
    frozen = discover(data, pc, preds={"model": truth_info["pf"]})
    res = audit(data, frozen)
    truth = true_values(frozen, sc, truth_info["intercept"], seed + 7)
    rows = replicate_metrics(data, truth_info, frozen, res, truth, sc)
    for row in rows:
        row.update(rep=rep, seconds=round(time.perf_counter() - t0, 1))
    return rows


def run_simulation(sc: SimConfig, pc: PipelineConfig, n_reps: int, base_seed: int = 0,
                   progress: bool = True) -> pd.DataFrame:
    rows = []
    for rep in range(n_reps):
        rows += run_replicate(sc, pc, rep, base_seed)
        if progress:
            print(f"replicate {rep + 1}/{n_reps} done ({rows[-1]['seconds']}s)", flush=True)
    return pd.DataFrame(rows)


