"""Dataset adapters: raw file + YAML feature contract -> a standard ``AuditData``.

Features come only from the explicit allowlist in the config. Protected
attributes, identifiers, outcomes, and comparator scores never enter ``X``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Contrast:
    name: str
    attribute: str
    focal: str
    reference: str


@dataclass
class AuditData:
    name: str
    X: pd.DataFrame                    # allowlisted features; categoricals as pandas category
    y: np.ndarray                      # 0/1 outcome
    unit: np.ndarray                   # independent unit (hospital or patient)
    row_id: np.ndarray                 # stable encounter identifier
    protected: pd.DataFrame            # recorded attributes (NaN = missing)
    contrasts: list[Contrast]
    primary_contrast: str
    comparators: dict[str, np.ndarray] = field(default_factory=dict)  # NaN = invalid
    log: dict = field(default_factory=dict)
    role_unit: np.ndarray | None = None  # optional: allocate roles by this unit (e.g. encounter) while
                                         # inference still clusters by ``unit`` (e.g. hospital)
    outcome: str = "binary"              # "continuous": y is numeric (regression similarity forest)

    def __post_init__(self):
        n = len(self.y)
        for name, arr in [("X", self.X), ("unit", self.unit), ("row_id", self.row_id),
                          ("protected", self.protected)] + ([("role_unit", self.role_unit)] if self.role_unit is not None else []):
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows, expected {n}")
        if len(np.unique(self.row_id)) != n:
            raise ValueError("row_id is not unique")
        if self.outcome not in ("binary", "continuous"):
            raise ValueError("outcome must be 'binary' or 'continuous'")
        if self.outcome == "binary" and not set(np.unique(self.y)) <= {0, 1}:
            raise ValueError("y must be binary 0/1")
        leaked = set(self.X.columns) & set(self.protected.columns)
        if leaked:
            raise ValueError(f"protected attributes in features: {sorted(leaked)}")

    def subset(self, mask: np.ndarray) -> "AuditData":
        mask = np.asarray(mask, bool)
        return AuditData(
            name=self.name, X=self.X.loc[mask].reset_index(drop=True), y=self.y[mask],
            unit=self.unit[mask], row_id=self.row_id[mask],
            protected=self.protected.loc[mask].reset_index(drop=True),
            contrasts=self.contrasts, primary_contrast=self.primary_contrast,
            comparators={k: v[mask] for k, v in self.comparators.items()}, log=dict(self.log),
            role_unit=None if self.role_unit is None else self.role_unit[mask], outcome=self.outcome,
        )


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Diagnosis groups from Strack et al. (2014), Table 2.
def icd9_group(code) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)) or code == "?":
        return "missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "other"
    x = float(s)
    if 390 <= x <= 459 or int(x) == 785:
        return "circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "digestive"
    if int(x) == 250:
        return "diabetes"
    if 800 <= x <= 999:
        return "injury"
    if 710 <= x <= 739:
        return "musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "genitourinary"
    if 140 <= x <= 239:
        return "neoplasms"
    return "other"


def load_dataset(config_path: str | Path, root: str | Path = ".", verify_hash: bool = True,
                 n_units: int | None = None, seed: int = 0) -> AuditData:
    """Load a dataset from its YAML contract.

    ``n_units`` keeps a seeded random subset of independent units (for fast
    development runs); the subset is recorded in ``log``.
    """
    cfg = load_config(config_path)
    src = Path(root) / cfg["source"]
    log = {"config": str(config_path), "source": str(src)}
    if verify_hash:
        digest = sha256(src)
        if digest != cfg["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {src}: {digest}")
        log["sha256"] = digest
    raw = pd.read_csv(src, low_memory=False, dtype={cfg["row_id"]: str})
    log["n_raw"] = len(raw)

    keep = np.ones(len(raw), bool)
    for col, codes in (cfg.get("exclude_rows") or {}).items():
        drop = raw[col].isin(codes).to_numpy()
        log[f"excluded_{col}"] = int(drop.sum())
        keep &= ~drop
    raw = raw.loc[keep].reset_index(drop=True)

    if n_units is not None:
        units = raw[cfg["unit"]].unique()
        rng = np.random.default_rng(seed)
        chosen = rng.choice(units, size=min(n_units, len(units)), replace=False)
        raw = raw.loc[raw[cfg["unit"]].isin(chosen)].reset_index(drop=True)
        log["subsample_units"] = int(len(chosen))
    log["n_analysis"] = len(raw)

    out = cfg["outcome"]
    y = (raw[out["column"]] == out["positive"]).astype(int).to_numpy()

    feats = {}
    numeric = list(cfg.get("numeric", []))
    for prefix in cfg.get("numeric_prefix_expanded", []) or []:
        numeric += [c for c in raw.columns if c.startswith(prefix)]
    log["numeric_features"] = numeric
    for c in numeric:
        feats[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
    for c, levels in (cfg.get("ordinal") or {}).items():
        feats[c] = raw[c].map({lvl: i for i, lvl in enumerate(levels)}).astype(float)
    for c in cfg.get("categorical", []) or []:
        # fillna first: under pandas 3, astype(str) keeps NaN (e.g. max_glu_serum "None" read as
        # NaN), so the "nan" replacement never fired and "not measured" became NaN, not a category
        feats[c] = (raw[c].astype(object).where(raw[c].notna(), "missing").astype(str)
                    .replace({"nan": "missing", "?": "missing"}).astype("category"))
    for c in cfg.get("icd9_grouped", []) or []:
        feats[c] = raw[c].map(icd9_group).astype("category")
    X = pd.DataFrame(feats)

    protected = pd.DataFrame({
        attr: raw[attr].where(~raw[attr].isin(spec.get("missing", []) or []))
        for attr, spec in cfg["protected"].items()
    })
    contrasts = [Contrast(k, *v) for k, v in cfg["contrasts"].items()]

    comparators = {}
    for name, col in (cfg.get("comparators") or {}).items():
        # copy=True: under pandas 3 copy-on-write, to_numpy() can return a read-only view
        p = pd.to_numeric(raw[col], errors="coerce").to_numpy(float, copy=True)
        invalid = ~((p >= 0) & (p <= 1))
        log[f"{name}_missing"] = int(np.isnan(p).sum())
        log[f"{name}_out_of_range"] = int((invalid & ~np.isnan(p)).sum())
        p[invalid] = np.nan
        comparators[name] = p

    return AuditData(
        name=cfg["dataset"], X=X, y=y, unit=raw[cfg["unit"]].astype(str).to_numpy(),
        row_id=raw[cfg["row_id"]].to_numpy(), protected=protected, contrasts=contrasts,
        primary_contrast=cfg["primary_contrast"], comparators=comparators, log=log,
    )


def to_numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """Encode features for scikit-learn trees: categories -> integer codes, missing -> NaN."""
    cols = []
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            codes = s.cat.codes.to_numpy().astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(s.to_numpy(float))
    return np.column_stack(cols) if cols else np.empty((len(X), 0))
