"""Comparison partitions, all fit on discovery rows and applied unchanged to audit rows.

A ladder from "ignores relevance" to "learns relevance":

- global: one pocket (the ordinary group-level audit)
- kmeans: standardized, imputed legitimate features, all weighted equally
- mahalanobis_kmeans: K-means under the Mahalanobis metric (whitened features); equal
  weight for every direction of the feature distribution, still blind to the outcome
- cem: coarsened exact matching strata (every feature cut into quantile bins); strata
  too small to profile abstain, which is what happens with many features
- ridge_kmeans: K-means after scaling each standardized feature by the absolute
  coefficient of an L2-penalized logistic regression of the outcome (linear relevance)
- risk bins: quantiles of the discovery forest's predicted risk
  (answers "are the pockets just risk strata?")

``space`` tells ``pipeline.assign_partition`` what each partition reads: raw features,
the forest's predicted risk, or (for pocket maps, which lack the attribute) leaf signatures.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .assignment import ABSTAIN


def _standardizer():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())


class KMeansPockets:
    space = "features"

    def __init__(self, k: int, seed: int = 0):
        self.k, self.seed = k, seed

    def _embed(self, X: np.ndarray) -> np.ndarray:
        return self.pre.transform(X)

    def fit(self, X: np.ndarray):
        self.pre = _standardizer().fit(X)
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed)
        self.km.fit(self._embed(X))
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        return self.km.predict(self._embed(X)).astype(np.int32)


class MahalanobisKMeansPockets(KMeansPockets):
    """K-means on features whitened with the discovery covariance (Euclidean = Mahalanobis)."""

    def __init__(self, k: int, seed: int = 0, ridge: float = 1e-3):
        super().__init__(k, seed)
        self.ridge = ridge

    def _embed(self, X):
        return self.pre.transform(X) @ self.W

    def fit(self, X):
        self.pre = _standardizer().fit(X)
        Z = self.pre.transform(X)
        vals, vecs = np.linalg.eigh(np.cov(Z, rowvar=False).reshape(Z.shape[1], Z.shape[1]))
        self.W = vecs @ np.diag(1 / np.sqrt(np.clip(vals, 0, None) + self.ridge)) @ vecs.T
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(Z @ self.W)
        return self


class RidgeKMeansPockets(KMeansPockets):
    """K-means after weighting standardized features by |ridge logistic coefficient|.

    The penalty is chosen by 5-fold cross-validated log loss on the rows passed to ``fit``
    (discovery graph rows). If every coefficient is zero the weights fall back to 1.
    """

    def _embed(self, X):
        return self.pre.transform(X) * self.w

    def fit(self, X, y):
        self.pre = _standardizer().fit(X)
        Z = self.pre.transform(X)
        self.lr = LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=5000,
                                       l1_ratios=(0.0,)).fit(Z, y)   # pure L2 (explicit for sklearn >= 1.8)
        w = np.abs(self.lr.coef_.ravel())
        self.w = w if w.max() > 0 else np.ones_like(w)
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(Z * self.w)
        return self


class RidgeRegressionKMeansPockets(KMeansPockets):
    """Continuous-outcome version of ``RidgeKMeansPockets``: weights are |coefficients| of a ridge
    regression of the outcome on standardized features, penalty by efficient leave-one-out CV
    (``RidgeCV``) on the rows passed to ``fit``. If every coefficient is zero the weights fall back to 1."""

    def __init__(self, k: int, seed: int = 0, alphas=tuple(np.logspace(-3, 3, 13))):
        super().__init__(k, seed)
        self.alphas = alphas

    def _embed(self, X):
        return self.pre.transform(X) * self.w

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        self.pre = _standardizer().fit(X)
        Z = self.pre.transform(X)
        self.lr = RidgeCV(alphas=self.alphas).fit(Z, y)
        w = np.abs(self.lr.coef_.ravel())
        self.w = w if w.max() > 0 else np.ones_like(w)
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(Z * self.w)
        return self


class CoarsenedExactPockets:
    """Strata of features cut at discovery quantiles into ``n_bins`` bins (missing is its own
    bin). Strata with at least ``min_size`` discovery rows are pockets; other rows abstain."""
    space = "features"

    def __init__(self, n_bins: int = 3, min_size: int = 50):
        self.n_bins, self.min_size = n_bins, min_size

    def _codes(self, X):
        cols = []
        for j, e in enumerate(self.edges):
            c = np.searchsorted(e, X[:, j], side="right")
            c[np.isnan(X[:, j])] = len(e) + 1
            cols.append(c)
        return np.column_stack(cols).astype(np.int64) if cols else np.zeros((len(X), 0), np.int64)

    def fit(self, X):
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.nanquantile(X[:, j], q)) if (~np.isnan(X[:, j])).any() else np.array([])
                      for j in range(X.shape[1])]
        keys, counts = np.unique(self._codes(X), axis=0, return_counts=True)
        big = keys[counts >= self.min_size]
        self.lookup = {k.tobytes(): i for i, k in enumerate(big)}
        self.n_pockets, self.n_strata = len(big), len(keys)
        return self

    def assign(self, X):
        codes = np.ascontiguousarray(self._codes(X))
        return np.array([self.lookup.get(r.tobytes(), ABSTAIN) for r in codes], np.int32)


class RiskBins:
    space = "risk"

    def __init__(self, k: int):
        self.k = k

    def fit(self, risk: np.ndarray):
        self.edges = np.unique(np.quantile(risk, np.linspace(0, 1, self.k + 1)[1:-1]))
        return self

    def assign(self, risk: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, risk, side="right").astype(np.int32)


class TopFeatureCEM(CoarsenedExactPockets):
    """Coarsened exact matching restricted to the ``n_features`` most outcome-relevant features (ranked by the
    discovery forest's importances, which use outcomes but never protected attributes), so the strata stay
    populated. Plain CEM on every feature leaves almost every row in its own stratum."""

    def __init__(self, importances, n_features: int = 3, n_bins: int = 3, min_size: int = 50):
        super().__init__(n_bins, min_size)
        self.idx = np.argsort(np.asarray(importances))[::-1][:n_features]

    def fit(self, X):
        return super().fit(X[:, self.idx])

    def assign(self, X):
        return super().assign(X[:, self.idx])


class DisparityBins:
    """Comparator that targets the disparity directly (a T-learner). On discovery rows, the residual of the
    assessed model (prediction - outcome) is regressed on the features separately in the focal and reference
    groups; tau(x) is the difference of the two fits, and pockets are ``k`` quantile bins of tau with edges
    from held-out calibration rows. Unlike the similarity forest, it uses the protected attribute and one
    model's predictions in discovery, so its pockets are model- and contrast-specific."""
    space = "features"

    def __init__(self, k: int, seed: int = 0):
        self.k, self.seed = k, seed

    def tau(self, X):
        return self.m1.predict(X) - self.m0.predict(X)

    def fit(self, X, resid, focal, reference, X_calib):
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = dict(max_iter=200, learning_rate=.05, min_samples_leaf=50, random_state=self.seed)
        self.m1 = HistGradientBoostingRegressor(**kw).fit(X[focal], resid[focal])
        self.m0 = HistGradientBoostingRegressor(**kw).fit(X[reference], resid[reference])
        self.edges = np.unique(np.quantile(self.tau(X_calib), np.linspace(0, 1, self.k + 1)[1:-1]))
        return self

    def assign(self, X):
        return np.searchsorted(self.edges, self.tau(X), side="right").astype(np.int32)
