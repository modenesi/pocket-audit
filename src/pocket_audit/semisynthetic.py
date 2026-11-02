"""Semi-synthetic simulation: real features, patients, and protected attributes; simulated
outcome and injected bias.

The true risk p0(x) is a calibrated LightGBM fit to the real outcome on the real analysis's
development and discovery rows only (roles from the real seed), so real audit outcomes never
shape the simulation. Each replicate draws Y ~ Bernoulli(p0) independently, sets
p_f = expit(logit p0 + tau * S * R(x)), where S marks the primary contrast's focal group and R is
a region given as a query on the real features (for example ``number_inpatient >= 2``), and
reruns discovery and audit with new roles. Only the primary contrast is audited. Because the
features are fixed, truths are finite-population values over all rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .data import AuditData, load_config, load_dataset, to_numeric_matrix
from .models import fit_assessed_models, predict
from .pipeline import PipelineConfig, assign_partition, audit, discover
from .splits import allocate_units


@dataclass
class SemiConfig:
    config: str = "configs/features_uci.yaml"
    tau: float = 0.0
    region: str = "number_inpatient >= 2"
    real_seed: int = 20260928      # the real analysis's role seed; its audit rows never train p0
    n_units: int | None = None     # optional seeded subsample of units


def base(sc: SemiConfig) -> dict:
    """Load the real features once and fix the truth: p0, the region, and the focal group."""
    data = load_dataset(sc.config, n_units=sc.n_units, seed=sc.real_seed)
    role = allocate_units(data.unit, PipelineConfig().fractions, seed=sc.real_seed)
    keep = role != "audit"
    models = fit_assessed_models(data.X.loc[keep].reset_index(drop=True), data.y[keep], data.unit[keep],
                                 seed=sc.real_seed)
    p0 = np.clip(predict(models, data.X)["lightgbm"], 0.002, 0.998)
    region = data.X.eval(sc.region, engine="python").to_numpy(bool)
    con = next(c for c in data.contrasts if c.name == data.primary_contrast)
    attr = data.protected[con.attribute].to_numpy()
    support = load_config(sc.config).get("support") or {}
    return {"data": data, "p0": p0, "region": region, "contrast": con,
            "focal": attr == con.focal, "reference": attr == con.reference, "support": support,
            "info": {"n": len(p0), "p0_mean": float(p0.mean()), "region_share": float(region.mean()),
                     "focal_share": float((attr == con.focal).mean())}}


def simulate_semi(b: dict, tau: float, seed: int) -> tuple[AuditData, dict]:
    rng = np.random.default_rng(seed)
    real = b["data"]
    p0 = b["p0"]
    y = (rng.random(len(p0)) < p0).astype(int)
    pf = expit(logit(p0) + tau * b["focal"] * b["region"])
    data = AuditData(name=f"{real.name}_semisynthetic", X=real.X, y=y, unit=real.unit, row_id=real.row_id,
                     protected=real.protected, contrasts=[b["contrast"]], primary_contrast=b["contrast"].name)
    return data, {"pf": pf, "p0": p0, "region": b["region"], "bias": pf-p0}


def true_values_fixed(frozen: dict, b: dict, pf: np.ndarray) -> pd.DataFrame:
    """Finite-population delta and theta for each partition, over all rows."""
    p0 = b["p0"]
    Xnum = to_numeric_matrix(b["data"].X)
    rows = np.arange(len(p0))
    resid = pf - p0                       # E[p - Y | X, S]
    brier = p0 * (1 - p0) + resid ** 2    # E[(p - Y)^2 | X, S]
    focal, ref = b["focal"], b["reference"]
    out = []
    for name in frozen["partitions"]:
        pocket = assign_partition(frozen, name, rows, Xnum)
        for c in ["all"] + sorted(int(v) for v in np.unique(pocket) if v >= 0):
            m = np.ones(len(p0), bool) if c == "all" else pocket == c
            d = resid[m & focal].mean() - resid[m & ref].mean() if (m & focal).any() and (m & ref).any() else np.nan
            out.append({"partition": name, "kind": "delta", "model": "model", "contrast": b["contrast"].name, "pocket": c, "truth": d, "exact_null": bool(np.all(resid[m & focal] == 0))})
            if c != "all":
                out.append({"partition": name, "kind": "theta", "model": "model", "contrast": "", "pocket": c, "truth": brier[m].mean() - brier.mean()})
    return pd.DataFrame(out)


def replicate_semi(b: dict, sc: SemiConfig, pc: PipelineConfig, rep: int, base_seed: int = 3):
    """Returns (data, info, frozen, truth) for one replicate; audit with ``pipeline.audit``."""
    seed = base_seed + 1000 * rep
    data, info = simulate_semi(b, sc.tau, seed)
    pc = PipelineConfig(**{**asdict(pc), "seed": seed, "support": pc.support})
    frozen = discover(data, pc, preds={"model": info["pf"]})
    return data, info, frozen, true_values_fixed(frozen, b, info["pf"])
