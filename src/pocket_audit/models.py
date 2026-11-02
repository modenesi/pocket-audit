"""Assessed models, fit and calibrated on development rows only."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .splits import allocate_units


def _logistic(X: pd.DataFrame):
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                            keep_empty_features=True), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=2000))


def _lightgbm(seed: int):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                              n_jobs=-1, verbose=-1)


def fit_assessed_models(X: pd.DataFrame, y: np.ndarray, unit: np.ndarray,
                        seed: int = 20260928) -> dict:
    """Fit LightGBM and logistic regression on development rows, then Platt-calibrate
    each on a held-out 20% of development units. Returns {name: fitted calibrated model}."""
    part = allocate_units(unit, fractions=(0.8, 0.2), roles=("fit", "calib"), seed=seed + 1)
    fit, cal = part == "fit", part == "calib"
    models = {}
    for name, est in [("lightgbm", _lightgbm(seed)), ("logistic", _logistic(X))]:
        est.fit(X.loc[fit], y[fit])
        cal_model = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid")
        cal_model.fit(X.loc[cal], y[cal])
        models[name] = cal_model
    return models


def predict(models: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
