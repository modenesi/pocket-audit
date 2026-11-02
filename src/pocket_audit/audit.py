"""Local audit estimands and simultaneous inference (studentized cluster max-t).

Estimands (conditional on the frozen models and pocket map):

- ``delta``: calibration disparity  E[p-y | c, focal] - E[p-y | c, reference]
- ``kappa``: paired calibration disparity with p replaced by p_model1 - p_model2
- ``theta``: excess Brier loss      E[(p-y)^2 | c] - E[(p-y)^2 | eligible cohort]
- ``psi``:   paired Brier difference E[B_model1 - B_model2 | c]

Pocket "all" is the global (no-pocket) audit; it includes abstained rows.

Each coordinate is a signed sum of cell means, est = sum_k s_k mean(x | cell_k), so its
estimator is linear in per-row contributions v_i. Contributions are summed within
independent units (hospital or patient), U_hj, and

    se_j^2 = H/(H-1) * sum_h (U_hj - mean_h U_hj)^2

Two ways to calibrate the maximum of |est_j / se_j| over a family (``AuditConfig.method``):

``gauss``
    Gaussian multiplier bootstrap with the SEs held fixed:
    Z*_bj = sqrt(H/(H-1)) * sum_h xi_hb (U_hj - mean U_j) / se_j, xi ~ N(0,1).
    Too liberal with few clusters and rare outcomes.
``wild`` / ``wild_restricted`` (default)
    Wild cluster bootstrap-t: every draw multiplies each cluster's residuals by a
    Rademacher weight, re-estimates the cell means, and recomputes the cluster-robust
    SE. With a_hk = sum_{i in h, cell k} (x_i - m_k) / n_k and c_hk = n_hk / n_k,
    est* - est = sum_h xi_h U_h and U*_h = xi_h U_h - sum_k s_k (sum_g xi_g a_gk) c_hk.
    ``wild_restricted`` centers disparity coordinates (delta, kappa) at their null,
    the pooled mean of both cells (m_k = pooled), which is better calibrated with few
    clusters; loss coordinates (theta, psi) have no zero null and use their cell means.
``wild_restricted_mammen``
    The same with Mammen's two-point weights (third moment 1), which keep the skewness of
    rare-outcome residuals; in simulation it controlled the false-flag rate for independent
    rows slightly better.

In every case q is the conservative (1-alpha) quantile of the bootstrap maximum,
CI_j = est_j +/- q se_j, and the same draws are shared by every coordinate in a family,
preserving dependence between overlapping groups, pockets, and models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assignment import ABSTAIN
from .data import Contrast

METHODS = ("gauss", "wild", "wild_restricted", "wild_restricted_mammen")


@dataclass
class SupportRule:
    min_n: int = 100                  # per protected group (delta/kappa) or per pocket (theta/psi)
    min_units: int = 10               # distinct independent units per group
    min_effective_units: float = 10.0  # (sum n_h)^2 / sum n_h^2 per group
    max_unit_share: float | None = None  # e.g. 0.2 for hospitals
    min_expected: float = 10.0        # expected events AND non-events per cell, from predictions (never outcomes)


@dataclass
class AuditConfig:
    support: SupportRule = field(default_factory=SupportRule)
    pairs: list[tuple[str, str]] = field(default_factory=list)   # e.g. [("lightgbm", "apache")]
    primary_models: list[str] | None = None                      # models in the primary family
    n_boot: int = 4999
    alpha: float = 0.05
    seed: int = 20260928
    method: str = "wild_restricted"
    restricted_loss: bool = False  # also impose the null (theta = 0, psi = 0) when bootstrapping the loss family;
                                   # by default loss coordinates are centered at their own means (unrestricted)
    se_floor: bool = False     # studentize disparity coordinates with max(cluster SE, null-model SE) (opt-in)
    outcome: str = "binary"    # "continuous": null-model SE from the pocket's pooled residual variance (no [0,1] clip)
    include_losses: bool = True  # False: skip the theta/psi loss family (e.g. continuous outcomes, where it is not Brier)
    include_mu: bool = False     # also audit mu_c = E[p - Y | pocket], the pocket-average signed prediction error
                                 # ("where"); same family as delta/kappa for primary models (v2, 2026-10-31)
    include_disparity: bool = True  # False: skip delta/kappa (e.g. a mu-only exploratory family)
    tails: str = "symmetric"   # "symmetric": one critical value from max |t*| (default);
                               # "equal": upper/lower critical values from max t* and max -t*, each at
                               # alpha/2, so a skewed statistic keeps both flag directions near alpha/2
    engine: str = "loop"       # "gemm": wild_maxt_gemm, the same draws as matrix products (faster with many units)


@dataclass
class Cells:
    """Cluster-level pieces of one coordinate, restricted to the clusters it touches."""
    idx: np.ndarray      # touched cluster codes
    A: np.ndarray        # |idx| x K contributions centered at each cell's mean
    AR: np.ndarray       # |idx| x K contributions centered at the null (pooled) mean
    C: np.ndarray        # |idx| x K cell count shares n_hk / n_k
    S: np.ndarray        # K signs


def _unit_support(unit_codes: np.ndarray, mask: np.ndarray, rule: SupportRule) -> tuple[bool, str, dict]:
    n = int(mask.sum())
    counts = np.bincount(unit_codes[mask]) if n else np.array([0])
    counts = counts[counts > 0]
    info = {"n": n, "units": int(len(counts)),
            "eff_units": float(counts.sum() ** 2 / (counts ** 2).sum()) if n else 0.0,
            "max_unit_share": float(counts.max() / n) if n else 1.0}
    if n < rule.min_n:
        return False, f"n<{rule.min_n}", info
    if info["units"] < rule.min_units:
        return False, f"units<{rule.min_units}", info
    if info["eff_units"] < rule.min_effective_units:
        return False, f"eff_units<{rule.min_effective_units}", info
    if rule.max_unit_share is not None and info["max_unit_share"] > rule.max_unit_share:
        return False, f"unit_share>{rule.max_unit_share}", info
    return True, "", info


def _expected_ok(probs: list[np.ndarray], masks: list[np.ndarray], rule: SupportRule) -> tuple[bool, str]:
    """Each cell needs min_expected expected events and non-events under every model's predictions."""
    if rule.min_expected <= 0:
        return True, ""
    for p in probs:
        for m in masks:
            q = p[m]
            if min(np.nansum(q), np.nansum(1 - q)) < rule.min_expected:
                return False, f"expected<{rule.min_expected:g}"
    return True, ""


def _mean_contrib(x: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    n = mask.sum()
    mu = float(x[mask].mean())
    v = np.zeros(len(x))
    v[mask] = (x[mask] - mu) / n
    return mu, v


def _cells(unit_codes, H, parts, null_center) -> Cells:
    """parts: [(sign, mask, x)]; null_center: pooled null mean for disparity coordinates, else None."""
    touched = np.zeros(H, bool)
    for _, m, _ in parts:
        touched[np.unique(unit_codes[m])] = True
    idx = np.flatnonzero(touched)
    pos = np.full(H, -1)
    pos[idx] = np.arange(len(idx))
    A, AR, C, S = [], [], [], []
    for sign, m, x in parts:
        n = m.sum()
        codes = pos[unit_codes[m]]
        mu = x[m].mean()
        A.append(np.bincount(codes, weights=(x[m] - mu) / n, minlength=len(idx)))
        ref = mu if null_center is None else null_center
        AR.append(np.bincount(codes, weights=(x[m] - ref) / n, minlength=len(idx)))
        C.append(np.bincount(codes, minlength=len(idx)) / n)
        S.append(sign)
    return Cells(idx, np.column_stack(A), np.column_stack(AR), np.column_stack(C), np.array(S, float))


def build_coordinates(y, preds: dict, pocket, unit, protected: pd.DataFrame,
                      contrasts: list[Contrast], cfg: AuditConfig):
    """Return (coordinate table, contribution matrix U [H x J], unit ids, cells per coordinate)."""
    if cfg.outcome not in ("binary", "continuous"):
        raise ValueError("outcome must be 'binary' or 'continuous'")
    if cfg.outcome == "continuous" and cfg.support.min_expected > 0:
        raise ValueError("expected-event support is binary-only; set SupportRule(min_expected=0) for continuous outcomes")
    y = np.asarray(y, float)
    pocket = np.asarray(pocket)
    unit_ids, unit_codes = np.unique(np.asarray(unit).astype(str), return_inverse=True)
    H = len(unit_ids)
    pockets = ["all"] + sorted(int(c) for c in np.unique(pocket) if c != ABSTAIN)
    primary_models = cfg.primary_models or list(preds)
    rows, cols, cells = [], [], []

    def add(meta, est, v, ok, reason, parts=None, null_center=None):
        meta.update(estimate=est, supported=ok, reason=reason)
        rows.append(meta)
        cols.append(np.bincount(unit_codes, weights=v, minlength=H) if ok else np.zeros(H))
        cells.append(_cells(unit_codes, H, parts, null_center) if ok else None)

    def in_pocket(c):
        return np.ones(len(y), bool) if c == "all" else pocket == c

    # disparity-type coordinates: delta (per model) and kappa (per model pair)
    probs = {m: np.asarray(p, float) for m, p in preds.items()}
    scores = {m: (p - y, ~np.isnan(p), [p]) for m, p in probs.items()}
    paired = {}
    for m1, m2 in cfg.pairs:
        p1, p2 = probs[m1], probs[m2]
        paired[f"{m1}-{m2}"] = (p1 - p2, ~np.isnan(p1) & ~np.isnan(p2), [p1, p2])
    for kind, source in ([("delta", scores), ("kappa", paired)] if cfg.include_disparity else []):
        for model, (x, valid, ps) in source.items():
            family = "primary" if (kind == "kappa" or model in primary_models) else "secondary"
            for con in contrasts:
                attr = protected[con.attribute].to_numpy()
                for c in pockets:
                    base = valid & in_pocket(c)
                    I1, I0 = base & (attr == con.focal), base & (attr == con.reference)
                    ok1, r1, s1 = _unit_support(unit_codes, I1, cfg.support)
                    ok0, r0, s0 = _unit_support(unit_codes, I0, cfg.support)
                    meta = dict(kind=kind, family=family, model=model, contrast=con.name, pocket=c,
                                n_focal=s1["n"], n_reference=s0["n"], units_focal=s1["units"],
                                units_reference=s0["units"])
                    if s1["n"] == 0 or s0["n"] == 0:
                        add(meta, np.nan, None, False, "empty_group")
                        continue
                    mu1, v1 = _mean_contrib(x, I1)
                    mu0, v0 = _mean_contrib(x, I0)
                    meta.update(mean_focal=mu1, mean_reference=mu0)
                    if kind == "delta" and cfg.outcome == "continuous":   # pooled residual variance (the null)
                        n_pool = int((I1 | I0).sum())
                        s2 = float(x[I1 | I0].var(ddof=1)) if n_pool > 1 else np.nan
                        meta["se_model0"] = float(np.sqrt(s2 * (1 / I1.sum() + 1 / I0.sum())))
                    elif kind == "delta":   # null-model SE: predictions recalibrated to the pooled mean residual
                        m0 = np.clip(ps[0] - x[I1 | I0].mean(), 1e-6, 1 - 1e-6)
                        vv = m0 * (1 - m0)
                        meta["se_model0"] = float(np.sqrt(vv[I1].sum() / I1.sum() ** 2 + vv[I0].sum() / I0.sum() ** 2))
                    oke, re_ = _expected_ok(ps, [I1, I0], cfg.support)
                    ok = ok1 and ok0 and oke
                    reason = "; ".join(r for r in (r1 and f"focal {r1}", r0 and f"reference {r0}", re_) if r)
                    add(meta, mu1 - mu0, v1 - v0, ok, reason,
                        parts=[(1, I1, x), (-1, I0, x)], null_center=float(x[I1 | I0].mean()))

    # one-sample coordinates: mu (per model), the pocket-average signed prediction error, null mu = 0.
    # Support is the rule applied to the pooled pocket cell. Binary floor: the independent-Bernoulli SE
    # under strong calibration, sqrt(sum p(1-p))/n, used only as a stabilizer (max with the cluster SE).
    for model, (x, valid, ps) in (scores.items() if cfg.include_mu else ()):
        family = "primary" if model in primary_models else "secondary"
        for c in pockets:
            Ic = valid & in_pocket(c)
            ok, r, s = _unit_support(unit_codes, Ic, cfg.support)
            meta = dict(kind="mu", family=family, model=model, contrast="", pocket=c,
                        n_focal=s["n"], units_focal=s["units"])
            if s["n"] == 0:
                add(meta, np.nan, None, False, "empty_pocket")
                continue
            mu, v = _mean_contrib(x, Ic)
            meta["mean_focal"] = mu
            if cfg.outcome == "binary":
                pc = ps[0][Ic]
                meta["se_model0"] = float(np.sqrt((pc * (1 - pc)).sum()) / Ic.sum())
            oke, re_ = _expected_ok(ps, [Ic], cfg.support)
            add(meta, mu, v, ok and oke, "; ".join(z for z in (r, re_) if z),
                parts=[(1, Ic, x)], null_center=0.0)

    # loss-type coordinates: theta (per model, relative to eligible cohort) and psi (paired)
    for model, p in (probs.items() if cfg.include_losses else ()):
        valid = ~np.isnan(p)
        B = np.where(valid, (p - y) ** 2, 0.0)
        muV, vV = _mean_contrib(B, valid)
        for c in pockets[1:]:
            Ic = valid & in_pocket(c)
            ok, r, s = _unit_support(unit_codes, Ic, cfg.support)
            meta = dict(kind="theta", family="secondary", model=model, contrast="", pocket=c,
                        n_focal=s["n"], units_focal=s["units"], mean_reference=muV)
            if s["n"] == 0:
                add(meta, np.nan, None, False, "empty_pocket")
                continue
            muc, vc = _mean_contrib(B, Ic)
            meta.update(mean_focal=muc)
            oke, re_ = _expected_ok([p], [Ic], cfg.support)
            add(meta, muc - muV, vc - vV, ok and oke, "; ".join(x for x in (r, re_) if x),
                parts=[(1, Ic, B), (-1, valid, B)], null_center=muV)   # null theta = 0: cohort mean
    for name, (d, valid, ps) in (paired.items() if cfg.include_losses else ()):
        m1, m2 = name.split("-", 1)
        D = np.where(valid, (probs[m1] - y) ** 2 - (probs[m2] - y) ** 2, 0.0)
        for c in pockets:
            Ic = valid & in_pocket(c)
            ok, r, s = _unit_support(unit_codes, Ic, cfg.support)
            meta = dict(kind="psi", family="secondary", model=name, contrast="", pocket=c,
                        n_focal=s["n"], units_focal=s["units"])
            if s["n"] == 0:
                add(meta, np.nan, None, False, "empty_pocket")
                continue
            mu, v = _mean_contrib(D, Ic)
            oke, re_ = _expected_ok(ps, [Ic], cfg.support)
            add(meta, mu, v, ok and oke, "; ".join(x for x in (r, re_) if x), parts=[(1, Ic, D)],
                null_center=0.0)                                       # null psi = 0

    table = pd.DataFrame(rows)
    U = np.column_stack(cols) if cols else np.zeros((H, 0))
    return table, U, unit_ids, cells


def _quantile(M: np.ndarray, alpha: float) -> float:
    M = np.sort(M)
    return float(M[min(len(M) - 1, int(np.ceil((len(M) + 1) * (1 - alpha))) - 1)])


def _exceed(M_sorted: np.ndarray, t: np.ndarray, n_boot: int) -> np.ndarray:
    """Bootstrap p-value (1 + #{M >= t}) / (B + 1) for each t."""
    return (1 + (n_boot - np.searchsorted(M_sorted, t, side="left"))) / (n_boot + 1)


def _result(M, est, se, n_boot, alpha, H, J, Mp=None, Mn=None) -> dict:
    M = np.sort(M)
    t = np.abs(est / se)
    p_adj = _exceed(M, t, n_boot)
    out = {"q": _quantile(M, alpha), "se": se, "t": t, "p_adj": p_adj, "p_global": float(p_adj.min()),
           "H": H, "J": J, "n_boot": n_boot, "M": M, "Mp": Mp, "Mn": Mn}
    if Mp is not None:       # equal-tailed: max over the family of t* and of -t*, each at alpha / 2
        Mp, Mn, ts = np.sort(Mp), np.sort(Mn), est / se
        p_eq = np.minimum(1.0, 2 * np.minimum(_exceed(Mp, ts, n_boot), _exceed(Mn, -ts, n_boot)))
        out.update(q_up=_quantile(Mp, alpha / 2), q_dn=_quantile(Mn, alpha / 2), p_adj_equal=p_eq,
                   p_global_equal=float(p_eq.min()))
    return out


def multiplier_maxt(U: np.ndarray, est: np.ndarray, n_boot: int = 4999, alpha: float = 0.05,
                    seed: int = 0, batch: int = 250) -> dict:
    """Studentized cluster Gaussian multiplier max-t over the columns of U (SEs fixed)."""
    H, J = U.shape
    if J == 0:
        return {"q": np.nan, "se": np.array([]), "p_adj": np.array([]), "p_global": np.nan}
    Uc = U - U.mean(axis=0, keepdims=True)
    scale = np.sqrt(H / (H - 1))
    se = scale * np.sqrt((Uc ** 2).sum(axis=0))
    if np.any(se <= 0):
        raise ValueError("zero standard error; screen coordinates before calling")
    W = Uc / se
    rng = np.random.default_rng(seed)
    M, Mp, Mn = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for s in range(0, n_boot, batch):
        m = min(batch, n_boot - s)
        Z = scale * (rng.standard_normal((m, H)) @ W)
        M[s:s + m], Mp[s:s + m], Mn[s:s + m] = np.abs(Z).max(axis=1), Z.max(axis=1), (-Z).max(axis=1)
    return _result(M, est, se, n_boot, alpha, H, J, Mp, Mn)


def wild_weights(rng, shape, kind: str = "rademacher") -> np.ndarray:
    """Bootstrap weights with mean 0 and variance 1. Mammen's two-point weights also have third
    moment 1, so the bootstrap keeps the skewness of rare-outcome residuals."""
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], shape)
    if kind == "mammen":
        r5 = np.sqrt(5.0)
        return np.where(rng.random(shape) < (r5 + 1) / (2 * r5), -(r5 - 1) / 2, (r5 + 1) / 2)
    raise ValueError(f"unknown weights {kind!r}")


def wild_maxt(cells: list[Cells], U: np.ndarray, est: np.ndarray, restricted: np.ndarray, n_boot: int = 4999,
              alpha: float = 0.05, seed: int = 0, batch: int = 100, weights: str = "rademacher",
              se_floor: np.ndarray | None = None) -> dict:
    """Wild cluster bootstrap-t max statistic (see module docstring).

    ``restricted[j]`` centers coordinate j's bootstrap residuals at its null (pooled) mean.
    ``weights``: "rademacher" or "mammen" (see ``wild_weights``).
    ``se_floor[j]`` (optional): studentize coordinate j with max(SE, floor) in the sample and in every
    bootstrap draw, so a cluster SE that collapses with sparse events cannot produce an extreme t.
    """
    H, J = U.shape
    if J == 0:
        return {"q": np.nan, "se": np.array([]), "p_adj": np.array([]), "p_global": np.nan}
    scale = np.sqrt(H / (H - 1))
    se = scale * np.sqrt(((U - U.mean(axis=0, keepdims=True)) ** 2).sum(axis=0))
    floor = np.zeros(J) if se_floor is None else np.asarray(se_floor, float)
    if np.any((se <= 0) & (floor <= 0)):
        raise ValueError("zero standard error; screen coordinates before calling")
    rng = np.random.default_rng(seed)
    M, Mp, Mn = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for s in range(0, n_boot, batch):
        b = min(batch, n_boot - s)
        xi = wild_weights(rng, (b, H), weights)
        mx, mxp, mxn = np.zeros(b), np.full(b, -np.inf), np.full(b, -np.inf)
        for j, cell in enumerate(cells):
            A = cell.AR if restricted[j] else cell.A
            u = A @ cell.S
            x = xi[:, cell.idx]
            num = x @ u
            ustar = x * u[None, :] - ((x @ A) * cell.S[None, :]) @ cell.C.T
            # clusters outside cell.idx contribute zero; center over all H clusters
            mean = ustar.sum(axis=1, keepdims=True) / H
            ss = ((ustar - mean) ** 2).sum(axis=1) + (H - len(cell.idx)) * mean[:, 0] ** 2
            tstar = num / np.maximum(np.maximum(scale * np.sqrt(ss), floor[j]), 1e-300)
            mx, mxp, mxn = np.maximum(mx, np.abs(tstar)), np.maximum(mxp, tstar), np.maximum(mxn, -tstar)
        M[s:s + b], Mp[s:s + b], Mn[s:s + b] = mx, mxp, mxn
    return _result(M, est, np.maximum(se, floor), n_boot, alpha, H, J, Mp, Mn)


def wild_maxt_gemm(cells: list[Cells], U: np.ndarray, est: np.ndarray, restricted: np.ndarray,
                   n_boot: int = 4999, alpha: float = 0.05, seed: int = 0, batch: int = 100,
                   weights: str = "rademacher", se_floor: np.ndarray | None = None) -> dict:
    """``wild_maxt`` rewritten as matrix products, for many draws over many clusters (same weights, same result
    up to rounding). With T = sum_h U*_h, the per-draw sum of squares is sum_h U*_h^2 - T^2 / H, and

        sum_h U*_h^2 = (xi^2) u^2 - 2 sum_k v_k xi (u c_k) + v' (C'C) v,   v_k = s_k xi a_k,

    so every term is one product of the weight matrix with a stacked column of cluster values."""
    H, J = U.shape
    if J == 0:
        return {"q": np.nan, "se": np.array([]), "p_adj": np.array([]), "p_global": np.nan}
    scale = np.sqrt(H / (H - 1))
    se = scale * np.sqrt(((U - U.mean(axis=0, keepdims=True)) ** 2).sum(axis=0))
    floor = np.zeros(J) if se_floor is None else np.asarray(se_floor, float)
    if np.any((se <= 0) & (floor <= 0)):
        raise ValueError("zero standard error; screen coordinates before calling")
    cols, sq, layout = [], np.zeros((H, J)), []
    for j, cell in enumerate(cells):
        A = cell.AR if restricted[j] else cell.A
        u = A @ cell.S
        K = A.shape[1]
        block = np.zeros((H, 1 + 2 * K))
        block[cell.idx, 0] = u
        block[cell.idx, 1:1 + K] = A
        block[cell.idx, 1 + K:] = u[:, None] * cell.C
        sq[cell.idx, j] = u ** 2
        layout.append((sum(c.shape[1] for c in cols), K, cell.S, cell.C.sum(axis=0), cell.C.T @ cell.C))
        cols.append(block)
    V = np.hstack(cols)
    rng = np.random.default_rng(seed)
    M, Mp, Mn = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for s in range(0, n_boot, batch):
        b = min(batch, n_boot - s)
        xi = wild_weights(rng, (b, H), weights)
        P, R = xi @ V, (xi ** 2) @ sq
        tstar = np.empty((b, J))
        for j, (o, K, S, csum, G) in enumerate(layout):
            num = P[:, o]
            v = P[:, o + 1:o + 1 + K] * S[None, :]
            T = num - v @ csum
            ss = R[:, j] - 2 * (v * P[:, o + 1 + K:o + 1 + 2 * K]).sum(axis=1) + np.einsum("bk,kl,bl->b", v, G, v)
            ss = np.maximum(ss - T ** 2 / H, 0.0)
            tstar[:, j] = num / np.maximum(np.maximum(scale * np.sqrt(ss), floor[j]), 1e-300)
        M[s:s + b], Mp[s:s + b], Mn[s:s + b] = np.abs(tstar).max(axis=1), tstar.max(axis=1), (-tstar).max(axis=1)
    return _result(M, est, np.maximum(se, floor), n_boot, alpha, H, J, Mp, Mn)


def run_audit(y, preds: dict, pocket, unit, protected, contrasts, cfg: AuditConfig) -> tuple[pd.DataFrame, dict]:
    """Build all coordinates, then run the max-t procedure separately per family."""
    if cfg.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if cfg.tails not in ("symmetric", "equal"):
        raise ValueError("tails must be 'symmetric' or 'equal'")
    if cfg.engine not in ("loop", "gemm"):
        raise ValueError("engine must be 'loop' or 'gemm'")
    table, U, _, cells = build_coordinates(y, preds, pocket, unit, protected, contrasts, cfg)
    for col in ["se", "t", "ci_low", "ci_high", "p_adj"]:
        table[col] = np.nan
    table["design_eligible"] = table["supported"].astype(bool)
    table["status"] = np.where(table["supported"], "tested", "not_eligible")
    summary = {}
    for fam_i, family in enumerate(["primary", "secondary"]):
        idx = np.flatnonzero((table["family"] == family) & table["supported"])
        if len(idx):
            Uc = U[:, idx] - U[:, idx].mean(0)
            tolerance = (32*np.finfo(float).eps*np.maximum(1, np.abs(table.loc[idx, "estimate"].to_numpy())))**2
            variance = (Uc ** 2).sum(0)
            zero = ~np.isfinite(variance) | (variance <= tolerance)
            if cfg.se_floor and "se_model0" in table and cfg.method.startswith("wild_restricted"):
                # a finite mu with zero empirical variance is studentized by its positive floor, not dropped
                # (restricted wild only; nonfinite estimates or variances are never rescued)
                s0 = table.loc[idx, "se_model0"].to_numpy(float)
                floored_mu = ((table.loc[idx, "kind"] == "mu").to_numpy() & np.isfinite(variance)
                              & np.isfinite(table.loc[idx, "estimate"].to_numpy(float))
                              & np.isfinite(s0) & (s0 > 0))
                zero &= ~floored_mu
            table.loc[idx[zero], ["status", "reason"]] = ["degenerate", "zero_variance"]
            idx = idx[~zero]
        est = table.loc[idx, "estimate"].to_numpy()
        seed = cfg.seed + fam_i
        if cfg.method == "gauss":
            res = multiplier_maxt(U[:, idx], est, n_boot=cfg.n_boot, alpha=cfg.alpha, seed=seed)
        else:
            loss_kinds = ("theta", "psi") if cfg.restricted_loss else ()
            restricted = np.array([cfg.method.startswith("wild_restricted")
                                   and table.loc[i, "kind"] in ("delta", "kappa", "mu", *loss_kinds) for i in idx], bool)
            floor = (table.loc[idx, "se_model0"].fillna(0).to_numpy()
                     if cfg.se_floor and "se_model0" in table and len(idx) else None)
            maxt = wild_maxt_gemm if cfg.engine == "gemm" else wild_maxt
            res = maxt([cells[i] for i in idx], U[:, idx], est, restricted, n_boot=cfg.n_boot,
                            alpha=cfg.alpha, seed=seed,
                            weights="mammen" if cfg.method.endswith("mammen") else "rademacher", se_floor=floor)
        if len(idx):
            table.loc[idx, "se"] = res["se"]
            table.loc[idx, "t"] = res["t"]
            if cfg.tails == "equal":
                table.loc[idx, "ci_low"] = est - res["q_up"] * res["se"]
                table.loc[idx, "ci_high"] = est + res["q_dn"] * res["se"]
                table.loc[idx, "p_adj"] = res["p_adj_equal"]
            else:
                table.loc[idx, "ci_low"] = est - res["q"] * res["se"]
                table.loc[idx, "ci_high"] = est + res["q"] * res["se"]
                table.loc[idx, "p_adj"] = res["p_adj"]
        summary[family] = {k: res.get(k) for k in ("q", "q_up", "q_dn", "H", "J", "n_boot", "M")}  # M: sorted bootstrap maxima
        summary[family]["p_global"] = res.get("p_global_equal" if cfg.tails == "equal" else "p_global")
        summary[family]["tails"] = cfg.tails
        summary[family]["method"] = cfg.method
        summary[family]["alpha"] = cfg.alpha
        summary[family]["n_design_eligible"] = int(((table.family == family) & table.design_eligible).sum())
        summary[family]["n_degenerate"] = int(((table.family == family) & (table.status == "degenerate")).sum())
    table["flag_positive"] = (table.status == "tested") & (table.ci_low > 0)
    table["flag_negative"] = (table.status == "tested") & (table.ci_high < 0)
    table["flag"] = table.flag_positive | table.flag_negative
    return table, summary


def coverage_by_group(pocket, protected: pd.DataFrame, contrasts: list[Contrast]) -> pd.DataFrame:
    """Share of rows assigned to a pocket (not abstained), for each protected group."""
    out = []
    assigned = np.asarray(pocket) != ABSTAIN
    for attr in sorted({c.attribute for c in contrasts}):
        col = protected[attr]
        for level in col.dropna().unique():
            m = (col == level).to_numpy()
            out.append({"attribute": attr, "group": level, "n": int(m.sum()),
                        "assigned_share": float(assigned[m].mean())})
    return pd.DataFrame(out)
