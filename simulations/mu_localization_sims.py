"""Where-localization simulations (2026-11-01): plant a local miscalibration mu, audit it on saved frozen maps.

Maps. Every map is a saved, frozen map from an earlier simulation run; nothing is rediscovered.
- Continuous ladder (arm P, the confirmation batch): discovery saw features and Y only; Y is drawn from the truth,
  so the maps never saw any assessed model's predictions.
- Binary, independent rows (D1, D1n) and 150 units (DH1): the saved maps of the null designs D0 and DH0 (450
  replicates each), whose discovery projected expected events from the TRUE risk (their assessed model was the truth).
- Binary DN2 (one-tail nonlinear) and DM1 (masking): the maps of the delta designs, whose level rule projected
  expected events from the delta design's predictions. These are audits of maps frozen under the original design.

Assessed predictions, for EVERY observation (no group involved); R is a feature region:
    continuous   f = m + s R              (R in {0, 1}; {-1, 0, +1} for masking)
    binary       f = distort(p0, s R)     (the delta designs' capping; the realized, clipped bias is used throughout)
    null         f = truth
    weak null    per map, the error varies within each pocket but its audit-sample pocket mean is exactly zero:
                 continuous f = m + .4 (z - zbar_c); binary f = p0 + .5 p0 (1 - p0) (z - w_c), w_c the
                 p0(1 - p0)-weighted mean of z; z = +-1 by the sign of an outcome-irrelevant feature, centred within
                 the map's pockets (and its abstained rows); single maps only (each map gets its own f)
Shifts s: "matched" (continuous .2, binary .05; approximately matched to the delta designs' precision) and "same"
(.4, .10), both reported. Regions: the original design region, plus an outcome-irrelevant one (continuous 1n:
z7 > q80 with spec-1 maps; binary D1n: x2 > q80 with D0 maps), where outcome-learned maps should do poorly.

Audit: the mu-only family (each pocket plus "all"; include_disparity=False, include_losses=False), wild restricted
bootstrap, Mammen weights, symmetric tails, through run_audit (engine "gemm", tested equal to the loop). Secondary:
the joint mu + delta family (the group S, independent of everything, so every delta is null) for the forest and
ridge maps under the null and the matched shift. Single-map levels as in the delta designs (continuous .05;
binary .025, and .0125 with 150 units); the combined audit runs forest and ridge at half. Support as in the delta
designs, on the pooled pocket cell, with the new predictions (expected-event rule for binary). B = 1,999
(continuous and independent binary rows) and 4,999 (binary with 150 units).

Error-targeted comparator ("error_bins", the mu analogue of the disparity-targeted method): on graph rows, a
gradient-boosting regression of the assessed model's residual f - y on the features (DisparityBins' settings),
cut into K quantile bins with edges from calibration rows, K = the forest map's pocket count; refit for every
condition (not under the weak null, which is defined per map on audit rows). Discovery rows are regenerated from the
saved seeds, and the regenerated audit rows (outcomes, original predictions, units, regions) are checked against the
saved arrays.

Truth. "fp": the audit-sample mean of the planted bias per pocket (finite-population, conditional on the audit
covariates; the same as the applications' validation). "pop" (continuous original regions only): s E[R | C = c] from
the saved integration shares (a signed mean under masking). Scoring is over all observations: affected = realized
bias != 0; right-direction flags are true positives; flagged unaffected observations are false positives;
wrong-direction flags count as a false positive and a miss; untested or abstained affected observations are misses;
the combined audit counts conflicting flags as errors. The "all" coordinate counts for false flags, never for
localization.

    python simulations/mu_localization_sims.py run --out results/mu_sims_20261101 --workers 28
    python simulations/mu_localization_sims.py summarize --out results/mu_sims_20261101
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = ROOT / "results"
CONF = "contladder_20261005/conf_v1_P_*/learned/spec{}_armP_n10000"
MASK = "contladder_20261005/night_mask/learned/spec{}_armP_n20000"
# id, family, saved-replicate pattern, spec/cell of the saved maps, cell of the DGP, region override, label
SCENARIOS = [dict(sid=sp, fam="cont", pat=CONF.format(sp), spec=sp, region=None) for sp in ("1", "2", "3", "4", "4c", "4z", "5")]
SCENARIOS += [dict(sid="1n", fam="cont", pat=CONF.format("1"), spec="1", region="z7", null=False, max_reps=200)]
SCENARIOS += [dict(sid=sp, fam="cont", pat=MASK.format(sp), spec=sp, region=None) for sp in ("1m", "4zm")]
SCENARIOS += [dict(sid="D1", fam="bin", pat="confirm_20261004/learned/D0", cell="D0", region="x1>q80"),
              dict(sid="D1n", fam="bin", pat="confirm_20261004/learned/D0", cell="D0", region="x2>q80", null=False),
              dict(sid="DH1", fam="bin", pat="confirm_20261004/learned/DH0", cell="DH0", region="x1>q80"),
              dict(sid="DN2", fam="bin", pat="confirm_20261004/learned/DN2", cell="DN2", region="x2>q80"),
              dict(sid="DM1", fam="bin", pat="night_binary_20261006/learned/DM1", cell="DM1", region="mask")]
SHIFTS = {"cont": {"matched": 0.2, "same": 0.4}, "bin": {"matched": 0.05, "same": 0.10}}
WEAK = {"cont": 0.4, "bin": 0.5}         # binary: f = p + .5 p(1-p)(z - w_c), |z - w_c| <= 2
CONT_METHODS = ["rf_spectral", "ridge_kmeans", "kmeans", "score_bins", "cem_top", "oracle"]
BIN_METHODS = ["sbm", "ridge_kmeans", "leiden", "risk_bins", "kmeans", "mahalanobis_kmeans", "cem", "cem_top"]
B_CONT, B_BIN, B_BIN_UNITS = 1999, 1999, 4999   # 4,999 where the combined per-map level is .00625 (150 units)


# ----------------------------------------------------------------------------- shared pieces
class ErrorBins:
    """mu analogue of DisparityBins: bins of a boosted regression of the assessed model's residual on the features."""

    def __init__(self, k: int, seed: int = 0):
        self.k, self.seed = k, seed

    def fit(self, X, resid, X_calib):
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = dict(max_iter=200, learning_rate=.05, min_samples_leaf=50, random_state=self.seed)   # DisparityBins'
        self.m = HistGradientBoostingRegressor(**kw).fit(X, resid)
        self.edges = np.unique(np.quantile(self.m.predict(X_calib), np.linspace(0, 1, self.k + 1)[1:-1]))
        return self

    def assign(self, X):
        return np.searchsorted(self.edges, self.m.predict(X), side="right").astype(np.int32)


def audit_family(y, f, lab, unit, s, cfg_kw: dict, alphas, joint: bool) -> dict:
    """One run_audit (mu-only, or mu + delta for S) and intervals at each alpha from the SAME bootstrap draws."""
    from pocket_audit.audit import AuditConfig, _quantile, run_audit
    from pocket_audit.data import Contrast
    cfg = AuditConfig(include_mu=True, include_disparity=joint, include_losses=False, engine="gemm",
                      method="wild_restricted_mammen", tails="symmetric", alpha=alphas[0], **cfg_kw)
    prot = pd.DataFrame({"S": np.where(s, "1", "0")})
    table, summ = run_audit(y, {"model": f}, lab, unit, prot, [Contrast("S", "S", "1", "0")], cfg)
    table = table[table.kind.isin(["mu", "delta"])].copy()
    table["pocket"] = table.pocket.astype(str)
    M = summ.get("primary", {}).get("M")
    out = {}
    for a in alphas:
        t = table.copy()
        tested = (t.status == "tested").to_numpy()
        q = _quantile(np.sort(np.asarray(M)), a) if (M is not None and tested.any()) else np.nan
        if tested.any():
            t.loc[tested, "ci_low"] = t.loc[tested, "estimate"] - q * t.loc[tested, "se"]
            t.loc[tested, "ci_high"] = t.loc[tested, "estimate"] + q * t.loc[tested, "se"]
        t["q"] = q
        t["dir"] = np.where(tested, (t.ci_low > 0).astype(int) - (t.ci_high < 0).astype(int), 0)
        out[a] = t
    return out


def directions(t: pd.DataFrame, lab: np.ndarray) -> np.ndarray:
    """Per-observation flag direction (+1, -1, 0) from a map's local mu pockets."""
    d = np.zeros(len(lab), int)
    for r in t[(t.kind == "mu") & (t.pocket != "all") & (t.dir != 0)].itertuples():
        d[lab == int(r.pocket)] = r.dir
    return d


def score(dirn: np.ndarray, bias: np.ndarray, conflict: np.ndarray | None = None) -> dict:
    """tp / wrong-sign / false-unaffected / fn over all observations (union_audit.score with focal = everyone)."""
    affected = bias != 0
    flagged = (dirn != 0) if conflict is None else ((dirn != 0) | conflict)
    correct = affected & (dirn == np.sign(bias))
    if conflict is not None:
        correct &= ~conflict
    tp = int(correct.sum())
    return dict(tp=tp, ws=int((affected & flagged & ~correct).sum()), fu=int((~affected & flagged).sum()),
                fn=int(affected.sum()) - tp)


def fp_truth(bias, lab, s, kind, pocket) -> float:
    cell = np.ones(len(bias), bool) if pocket == "all" else lab == int(pocket)
    if kind == "mu":
        return float(bias[cell].mean())
    return float(bias[cell & s].mean() - bias[cell & ~s].mean())


def flags_row(t: pd.DataFrame, lab, bias, s, pop=None) -> dict:
    """False-flag, coverage and support summaries of one map's table at one alpha."""
    tested = t[t.status == "tested"]
    fp = np.array([fp_truth(bias, lab, s, r.kind, r.pocket) for r in tested.itertuples()])
    cov = ((tested.ci_low <= fp) & (fp <= tested.ci_high)).to_numpy()
    out = dict(J=len(tested), n_pockets=int(((t.kind == "mu") & (t.pocket != "all")).sum()),
               n_local_tested=int(((tested.kind == "mu") & (tested.pocket != "all")).sum()),
               any_flag=bool((t.dir != 0).any()), any_mu_flag=bool(((t.kind == "mu") & (t.dir != 0)).any()),
               any_delta_flag=bool(((t.kind == "delta") & (t.dir != 0)).any()),
               global_flag=bool(((t.kind == "mu") & (t.pocket == "all") & (t.dir != 0)).any()),
               any_null_flag=bool(((t.dir != 0).to_numpy() & (np.abs(np.array(
                   [fp_truth(bias, lab, s, r.kind, r.pocket) if r.status == "tested" else 0.0
                    for r in t.itertuples()])) < 1e-12)).any()),
               covered_fp=bool(cov.all()) if len(tested) else np.nan)
    if pop is not None:
        mu_t = tested[tested.kind == "mu"]
        pv = np.array([pop(r.pocket) for r in mu_t.itertuples()])
        ok = np.isfinite(pv)
        out["covered_pop"] = bool(((mu_t.ci_low.to_numpy() <= pv) & (pv <= mu_t.ci_high.to_numpy()))[ok].all()) \
            if ok.any() else np.nan
    tested_local = tested[(tested.kind == "mu") & (tested.pocket != "all")].pocket.astype(int).to_numpy()
    aff = bias != 0
    out.update(n_affected=int(aff.sum()), untested_affected=int((aff & ~np.isin(lab, tested_local)).sum()),
               abstained_affected=int((aff & (lab < 0)).sum()), abstained=int((lab < 0).sum()))
    return out


def evaluate(cond, y, f, bias, labs: dict, unit, s, cfg_kw, alpha, forest, joint_maps=(), pop=None) -> list[dict]:
    """Audit every map (mu-only) at alpha, the forest/ridge pair at alpha/2 (combined), and optionally the joint
    mu + delta family for the primary maps."""
    rows = []
    for family in ("mu",) + (("joint",) if joint_maps else ()):
        half = {}
        for meth, lab in labs.items():
            if family == "joint" and meth not in joint_maps:
                continue
            res = audit_family(y, f, lab, unit, s, cfg_kw, [alpha, alpha / 2], joint=family == "joint")
            t = res[alpha]
            rows.append(dict(condition=cond, family=family, method=meth, alpha=alpha,
                             **flags_row(t, lab, bias, s, pop=(lambda p, m=meth: pop(m, p)) if pop else None),
                             **score(directions(t, lab), bias)))
            if meth in (forest, "ridge_kmeans"):
                th = res[alpha / 2]
                loc = th[(th.kind == "mu") & (th.pocket != "all") & (th.status == "tested")]
                known = np.isin(lab, loc.pocket.astype(int).to_numpy())
                half[meth] = (directions(th, lab), flags_row(th, lab, bias, s), known)
        if len(half) == 2:
            (fd, ff, fk), (rd, rf, rk) = half[forest], half["ridge_kmeans"]
            conflict = (fd != 0) & (rd != 0) & (fd != rd)
            u = np.where(fd != 0, fd, rd)
            u[conflict] = 0
            both = lambda k: bool(ff[k] or rf[k])
            cov = (ff["covered_fp"] and rf["covered_fp"]) if not (pd.isna(ff["covered_fp"]) or pd.isna(rf["covered_fp"])) else np.nan
            rows.append(dict(condition=cond, family=family, method="combined", alpha=alpha,
                             any_flag=both("any_flag"), any_mu_flag=both("any_mu_flag"),
                             any_delta_flag=both("any_delta_flag"), global_flag=both("global_flag"),
                             any_null_flag=both("any_null_flag"), covered_fp=cov, conflicts=int(conflict.sum()),
                             n_affected=ff["n_affected"],
                             untested_affected=int(((bias != 0) & ~fk & ~rk).sum()),
                             **score(u, bias, conflict)))
    return rows


def weak_rows(fam, y, truth, z, labs, unit, s, cfg_kw, alpha) -> list[dict]:
    """Per-map weak null: within-pocket errors that cancel exactly in every audit-sample pocket mean."""
    rows = []
    for meth, lab in labs.items():
        f = truth.copy()
        for c in np.unique(lab):                     # abstained rows (-1) are centred as their own cell
            m = lab == c
            if fam == "cont":
                f[m] = truth[m] + WEAK["cont"] * (z[m] - z[m].mean())
            else:                                    # weights p(1-p): f stays in (0, 1), no clipping
                v = truth[m] * (1 - truth[m])
                w = (v * z[m]).sum() / v.sum()
                f[m] = truth[m] + WEAK["bin"] * v * (z[m] - w)
        bias = f - truth
        assert max(abs(bias[lab == c].mean()) for c in np.unique(lab)) < 1e-12, "weak null is not mean-zero"
        assert fam == "cont" or ((f > 0) & (f < 1)).all()
        res = audit_family(y, f, lab, unit, s, cfg_kw, [alpha], joint=False)
        r = flags_row(res[alpha], lab, bias, s)
        r["max_abs_pocket_mean"] = max(abs(bias[lab == c].mean()) for c in np.unique(lab))
        rows.append(dict(condition="weak", family="mu", method=meth, alpha=alpha, **r))
    return rows


# ----------------------------------------------------------------------------- continuous ladder replicates
def run_cont(rep_dir: str, sc: dict) -> list[dict]:
    import continuous_ladder as CL
    from pocket_audit.audit import SupportRule
    from pocket_audit.splits import allocate_units
    rd = Path(rep_dir)
    done = json.loads((rd / "done.json").read_text())
    disc = json.loads((rd / "discovery.json").read_text())
    z = np.load(rd / "audit_inputs.npz")
    pk = pd.read_csv(rd / "pockets.csv")
    spec, sd, n = CL.SPECS[sc["spec"]], done["seeds"], done["n"]
    lat = CL.draw_latent(n, np.random.default_rng(sd["data"]))
    obs = CL.observe(lat, spec)
    m, R0 = CL.mean_fn(lat, spec), CL.region(lat, spec).astype(float)
    y = m + lat["eps"]
    uid = np.arange(n).astype(str)
    role = allocate_units(uid, CL.CONFIG["fractions"], seed=sd["roles"])
    dsc = np.flatnonzero(role == "discovery")
    sub = allocate_units(uid[dsc], (CL.CONFIG["graph_fraction"], 1 - CL.CONFIG["graph_fraction"]),
                         roles=("graph", "calib"), seed=sd["roles"] + 2)
    g, c, aud = dsc[sub == "graph"], dsc[sub == "calib"], np.flatnonzero(role == "audit")
    assert np.array_equal(aud, z["aud_rows"]) and np.array_equal(g, z["graph_rows"]), "roles differ"
    assert np.allclose(y[aud], z["y_d0"], rtol=0, atol=1e-10) and np.allclose(m[aud], z["f_d0"], rtol=0, atol=1e-10)
    assert np.array_equal(R0[aud], z["R_aud"].astype(float)) and np.array_equal(lat["s"][aud], z["s_aud"]), "R/S differ"
    if sc["region"] == "z7":                                   # outcome-irrelevant region
        R = (lat["Z"][:, 6] > CL.Q_R).astype(float)
        share = None
    else:
        R = R0
        pk0 = pk[np.isclose(pk.delta, 0.0)]
        share = {meth: dict(zip(pk0[pk0.method == meth].pocket.astype(str), pk0[pk0.method == meth].p_region))
                 for meth in pk0.method.unique()}
    has_pop = share is not None and all(np.isfinite(v) for mm in ("rf_spectral", "ridge_kmeans")
                                        for v in share.get(mm, {}).values())
    alt_ok = has_pop or (sc["region"] == "z7" and done["rep"] < 200)    # alternatives on the 200 alternative reps
    glob_share = 0.0 if spec["region"].startswith("mask") else .2            # E[R] (signed under masking)
    labs = {k: z[f"lab_{k}_d0"] for k in CONT_METHODS if f"lab_{k}_d0" in z.files
            and not (k == "oracle" and sc["region"] == "z7")}       # the saved oracle knows only the original region
    K = max(int(disc["0.0"]["K_profiled"]), 2)
    s_a = lat["s"][aud]
    unit = np.arange(len(aud))
    cfg_kw = dict(support=SupportRule(**CL.CONFIG["support"]), outcome="continuous", se_floor=False, n_boot=B_CONT,
                  seed=int(sd["bootstrap"] % (2 ** 31)))
    alpha = .05
    meta = dict(scenario=sc["sid"], family_kind="cont", rep=int(done["rep"]), maps="frozen, outcome-only")
    rows = []
    if sc.get("null", True):                              # null (truth) on every replicate of a map set
        eb0 = ErrorBins(K, seed=int(sd["kmeans"] % (2 ** 31))).fit(obs["forest"][g], (m - y)[g], obs["forest"][c])
        labs0 = dict(labs, error_bins=eb0.assign(obs["forest"][aud]))
        rows += [dict(meta, shift=0.0, **r) for r in evaluate("exact_null", y[aud], m[aud], np.zeros(len(aud)), labs0,
                                                               unit, s_a, cfg_kw, alpha, "rf_spectral",
                                                               joint_maps=("rf_spectral", "ridge_kmeans"))]
    if sc["region"] is None:                              # one weak-null check per replicate set
        zf = np.sign(lat["Z"][aud, 11]).astype(float)       # z12: outcome-irrelevant in every spec
        rows += [dict(meta, shift=WEAK["cont"], **r) for r in weak_rows("cont", y[aud], m[aud], zf, labs, unit, s_a,
                                                                         cfg_kw, alpha)]
    if not alt_ok:
        return rows
    lat_i = obs_i = R_i = None
    for cond, s in SHIFTS["cont"].items():
        f = m + s * R
        eb = ErrorBins(K, seed=int(sd["kmeans"] % (2 ** 31))).fit(obs["forest"][g], (f - y)[g], obs["forest"][c])
        labs_c = dict(labs, error_bins=eb.assign(obs["forest"][aud]))
        pop = None
        if has_pop:
            if lat_i is None:
                lat_i = CL.draw_latent(CL.CONFIG["n_int"], np.random.default_rng(sd["integration"]))
                obs_i, R_i = CL.observe(lat_i, spec), CL.region(lat_i, spec).astype(float)
            pr, _ = CL.region_share(eb.assign(obs_i["forest"]), R_i)
            sh = dict(share, error_bins={str(k): v for k, v in pr.items()})
            pop = lambda meth, p, s=s, sh=sh: s * (glob_share if p == "all" else sh[meth].get(p, np.nan))
        rows += [dict(meta, shift=s, **r) for r in evaluate(cond, y[aud], f[aud], s * R[aud], labs_c, unit, s_a,
                                                             cfg_kw, alpha, "rf_spectral",
                                                             joint_maps=("rf_spectral", "ridge_kmeans") if cond == "matched" else (),
                                                             pop=pop)]
    return rows


# ----------------------------------------------------------------------------- binary smoke-ladder replicates
def run_bin(rep_dir: str, sc: dict) -> list[dict]:
    import smoke_ladder as SL
    from pocket_audit.splits import allocate_units
    rd = Path(rep_dir)
    done = json.loads((rd / "done.json").read_text())
    z = np.load(rd / "audit_inputs.npz")
    cell, seed = SL.CELLS[sc["cell"]], int(done["seed"])
    pipe = done["pipeline"]
    d = SL.draw(cell, cell.get("n", SL.N), np.random.default_rng(seed), units=cell.get("units"))
    role = allocate_units(d["unit"], tuple(pipe["fractions"]), seed=seed)
    dsc = np.flatnonzero(role == "discovery")
    sub = allocate_units(d["unit"][dsc], (pipe["graph_fraction"], 1 - pipe["graph_fraction"]),
                         roles=("graph", "calib"), seed=seed + 2)
    g, c, aud = dsc[sub == "graph"], dsc[sub == "calib"], np.flatnonzero(role == "audit")
    # the saved pf must match one of the two capping rules (runs before 2026-10-06 capped every row at .995)
    pf_old = np.minimum(d["p0"] + cell["delta"] * d["s"] * d["R"], .995)
    assert np.array_equal(d["y"][aud], z["y"]), "outcomes differ"
    assert (np.allclose(d["pf"][aud], z["pf"], rtol=0, atol=1e-12)
            or np.allclose(pf_old[aud], z["pf"], rtol=0, atol=1e-12)), "predictions differ"
    assert np.array_equal(d["unit"][aud].astype(str), z["unit"].astype(str)), "units differ"
    assert np.array_equal(d["s"][aud], z["s"].astype(bool)), "groups differ"
    assert np.allclose(d["X"][aud][:, :z["x"].shape[1]], z["x"], rtol=0, atol=1e-12), "covariates differ"
    X, y, p0 = d["X"], d["y"].astype(float), d["p0"]
    R = SL.region_mask(X, sc["region"]).astype(float)
    labs = {k: z[f"labels_{k}"] for k in BIN_METHODS if f"labels_{k}" in z.files}
    K = max(int(done["n_pockets"]), 2)
    alpha = .0125 if cell.get("units") else .025                     # the delta designs' single-map levels
    sr = SL.support_rule(cell)
    sr.min_expected = 10.0                                           # the delta runs' expected-event rule
    cfg_kw = dict(support=sr, outcome="binary", se_floor=True, n_boot=B_BIN_UNITS if cell.get("units") else B_BIN,
                  seed=seed % (2 ** 31))
    meta = dict(scenario=sc["sid"], family_kind="bin", rep=int(done["rep"]),
                maps="null-design" if sc["cell"] in ("D0", "DH0") else "delta-design")
    s_a, unit = d["s"][aud], d["unit"][aud]
    rows = []
    primary = ("sbm", "ridge_kmeans")
    if sc.get("null", True):
        eb0 = ErrorBins(K, seed=seed % (2 ** 31)).fit(X[g], (p0 - y)[g], X[c])
        labs0 = dict(labs, error_bins=eb0.assign(X[aud]))
        rows += [dict(meta, shift=0.0, **r) for r in evaluate("exact_null", y[aud], p0[aud], np.zeros(len(aud)), labs0,
                                                               unit, s_a, cfg_kw, alpha, "sbm", joint_maps=primary)]
    if sc["sid"] in ("D1", "DH1"):                    # weak null once per map set (D1n shares D1's maps)
        zf = np.sign(X[aud, -1]).astype(float)        # last feature: outcome-irrelevant in the toy design
        rows += [dict(meta, shift=WEAK["bin"], **r) for r in weak_rows("bin", y[aud], p0[aud], zf, labs, unit, s_a,
                                                                        cfg_kw, alpha)]
    for cond, s in SHIFTS["bin"].items():
        f = SL.distort(p0, s * R)
        eb = ErrorBins(K, seed=seed % (2 ** 31)).fit(X[g], (f - y)[g], X[c])
        labs_c = dict(labs, error_bins=eb.assign(X[aud]))
        rows += [dict(meta, shift=s, **r) for r in evaluate(cond, y[aud], f[aud], (f - p0)[aud], labs_c, unit, s_a,
                                                             cfg_kw, alpha, "sbm",
                                                             joint_maps=primary if cond == "matched" else ())]
    return rows


def calibrate_bin(args) -> list[dict]:
    """Exact null, mu-only family: each primary map's smallest adjusted p-value (p_global) per replicate, for level
    calibration (2026-11-01: the 150-unit design flagged ~9% at the delta-calibrated levels)."""
    rep_dir, sc = args
    import smoke_ladder as SL
    from pocket_audit.audit import AuditConfig, run_audit
    from pocket_audit.data import Contrast
    from pocket_audit.splits import allocate_units
    done = json.loads((Path(rep_dir) / "done.json").read_text())
    z = np.load(Path(rep_dir) / "audit_inputs.npz")
    cell, seed = SL.CELLS[sc["cell"]], int(done["seed"])
    d = SL.draw(cell, cell.get("n", SL.N), np.random.default_rng(seed), units=cell.get("units"))
    aud = np.flatnonzero(allocate_units(d["unit"], tuple(done["pipeline"]["fractions"]), seed=seed) == "audit")
    assert np.array_equal(d["y"][aud], z["y"])
    sr = SL.support_rule(cell)
    sr.min_expected = 10.0
    out = []
    for mp in ("sbm", "ridge_kmeans"):
        cfg = AuditConfig(include_mu=True, include_disparity=False, include_losses=False, engine="gemm",
                          method="wild_restricted_mammen", tails="symmetric", alpha=.05, support=sr, outcome="binary",
                          se_floor=True, n_boot=B_BIN_UNITS if cell.get("units") else B_BIN, seed=seed % (2 ** 31))
        prot = pd.DataFrame({"S": np.where(d["s"][aud], "1", "0")})
        _, summ = run_audit(d["y"][aud].astype(float), {"model": d["p0"][aud]}, z[f"labels_{mp}"], d["unit"][aud], prot,
                            [Contrast("S", "S", "1", "0")], cfg)
        out.append(dict(scenario=sc["sid"], rep=int(done["rep"]), map=mp, p_global=summ.get("primary", {}).get("p_global")))
    return out


def task(args):
    sc, rep_dir = args
    try:
        return sc["sid"], rep_dir, (run_cont if sc["fam"] == "cont" else run_bin)(rep_dir, sc), None
    except Exception:
        return sc["sid"], rep_dir, [], traceback.format_exc()


# ----------------------------------------------------------------------------- summary
def f1(t):
    den = 2 * t[..., 0] + t[..., 1] + t[..., 2] + t[..., 3]
    return np.where(den > 0, 2 * t[..., 0] / np.where(den > 0, den, 1), np.nan)


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


ALTS = ["risk_bins", "kmeans", "mahalanobis_kmeans", "cem", "cem_top", "error_bins"]
OURS = ["forest", "combined", "ridge_kmeans", "leiden"]


def summarize(out: Path, n_draws: int = 5000):
    d = pd.read_csv(out / "reps.csv.gz")
    d["method"] = d.method.replace({"rf_spectral": "forest", "sbm": "forest", "score_bins": "risk_bins"})
    rng = np.random.default_rng(20261101)
    errs, f1rows, diffs = [], [], []
    for sid, g in d.groupby("scenario", sort=False):
        for (cond, fam, meth), h in g[g.condition.isin(["exact_null", "weak"])].groupby(["condition", "family", "method"]):
            k, n = int(h.any_flag.sum()), len(h)
            lo, hi = wilson(k, n)
            errs.append(dict(scenario=sid, condition=cond, family=fam, method=meth, reps=n, false_flag=k / n,
                             lo=lo, hi=hi, global_flag=h.global_flag.mean(), covered_fp=h.covered_fp.mean()))
        for cond in ("matched", "same"):
            for fam in ("mu", "joint"):
                a = g[(g.condition == cond) & (g.family == fam)]
                if a.empty:
                    continue
                meths = sorted(a.method.unique())
                reps = sorted(set.intersection(*(set(a[a.method == mm].rep) for mm in meths)))   # complete sets
                arr = {}
                for meth in meths:
                    h = a[a.method == meth].set_index("rep").loc[reps]
                    arr[meth] = h[["tp", "ws", "fu", "fn"]].to_numpy(float)
                    tot = arr[meth].sum(0)
                    f1rows.append(dict(scenario=sid, condition=cond, family=fam, method=meth, reps=len(reps),
                                       f1=float(f1(tot)), precision=tot[0] / max(tot[0] + tot[1] + tot[2], 1),
                                       recall=tot[0] / max(tot[0] + tot[3], 1),
                                       untested_affected=h.untested_affected.sum() / max(h.n_affected.sum(), 1),
                                       covered_fp=h.covered_fp.mean(),
                                       covered_pop=h.covered_pop.mean() if "covered_pop" in h else np.nan,
                                       any_flag=h.any_flag.mean(), global_flag=h.global_flag.mean(),
                                       any_delta_flag=h.any_delta_flag.mean()))
                draws = rng.integers(0, len(reps), (n_draws, len(reps)))
                cnt = np.stack([np.bincount(x, minlength=len(reps)) for x in draws])
                boot = {k: f1(np.einsum("br,rk->bk", cnt, v)) for k, v in arr.items()}
                est = {k: float(f1(v.sum(0))) for k, v in arr.items()}
                for ours in [o for o in OURS if o in arr]:
                    for comp in [c for c in ALTS if c in arr]:
                        gap = boot[ours] - boot[comp]
                        diffs.append(dict(scenario=sid, condition=cond, family=fam, ours=ours, versus=comp,
                                          diff=est[ours] - est[comp], lo=float(np.nanpercentile(gap, 2.5)),
                                          hi=float(np.nanpercentile(gap, 97.5))))
    E, F, D = pd.DataFrame(errs), pd.DataFrame(f1rows), pd.DataFrame(diffs)
    E.to_csv(out / "false_flags.csv", index=False)
    F.to_csv(out / "f1.csv", index=False)
    D.to_csv(out / "paired_diffs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(E.pivot_table(index=["scenario", "condition", "family"], columns="method", values="false_flag",
                        sort=False).round(3).to_string())
    for cond in ("matched", "same"):
        print(cond)
        print(F[(F.condition == cond) & (F.family == "mu")].pivot_table(index="scenario", columns="method",
                                                                         values="f1", sort=False).round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "summarize", "check", "calibrate"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--max-reps", type=int, default=None, help="cap per scenario (smoke tests)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        summarize(args.out)
        return
    if args.stage == "calibrate":
        jobs = []
        for sc in SCENARIOS:
            if sc["sid"] in ("D1", "DH1"):
                dirs = sorted(d for d in glob.glob(str(RES / sc["pat"] / "rep_*")) if not d.endswith(".partial"))
                jobs += [(d, sc) for d in dirs[:args.max_reps]]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [r for res in pool.map(calibrate_bin, jobs) for r in res]
        cal = pd.DataFrame(rows)
        cal.to_csv(args.out / "calibration_pglobal.csv", index=False)
        for sid, g in cal.groupby("scenario"):
            w = g.pivot(index="rep", columns="map", values="p_global")
            single = w["sbm"].quantile(.05)
            comb = (2 * w.min(axis=1)).quantile(.05)          # combined: each map at a/2, any flag if min p <= a/2
            print(f"{sid}: reps {len(w)}; single-map level for 5% = {single:.4f}; combined level for 5% = {comb:.4f};"
                  f" FWER at .025/.0125: forest {np.mean(w.sbm <= (.0125 if sid == 'DH1' else .025)):.3f}")
        return
    tasks = []
    for sc in SCENARIOS:
        if args.scenarios and sc["sid"] not in args.scenarios:
            continue
        dirs = sorted(d for d in glob.glob(str(RES / sc["pat"] / "rep_*")) if not d.endswith(".partial")
                      and (Path(d) / "done.json").exists())
        tasks += [(sc, d) for d in dirs[:min(args.max_reps or 10 ** 9, sc.get("max_reps", 10 ** 9))]]
    tasks.sort(key=lambda t: t[0]["fam"] != "bin")          # long binary tasks first
    prov = dict(command=sys.argv, started=time.strftime("%Y%m%dT%H%M%S"), n_tasks=len(tasks), shifts=SHIFTS, weak=WEAK,
                scenarios=SCENARIOS, b=dict(cont=B_CONT, bin=B_BIN, bin_units=B_BIN_UNITS),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                                   text=True), conda_env=os.environ.get("CONDA_DEFAULT_ENV"))
    rows, fails = [], []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(task, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            sid, rd, r, err = fu.result()
            rows += r
            if err:
                fails.append(dict(rep_dir=rd, error=err))
                print("FAILED", rd, err.strip().splitlines()[-1], flush=True)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    name = "reps_check.csv.gz" if args.stage == "check" else "reps.csv.gz"
    pd.DataFrame(rows).to_csv(args.out / name, index=False)
    prov.update(finished=time.strftime("%Y%m%dT%H%M%S"), seconds=round(time.time() - t0, 1), failures=fails)
    (args.out / f"manifest_{args.stage}.json").write_text(json.dumps(prov, indent=1, default=str))
    print(f"done: {len(rows)} rows, {len(fails)} failures, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
