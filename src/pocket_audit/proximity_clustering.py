"""Simple pockets from an exact random-forest proximity matrix (small samples; no SBM).

Proximity P[i, j] = share of trees in which rows i and j fall in the same leaf. With the sparse
leaf-incidence matrix B (one column per tree/leaf, one entry per row and tree), P = B B' / T exactly.

The primary clusterer is a normalized spectral embedding:

    W = P with a zero diagonal,  d = W 1,  N = D^-1/2 W D^-1/2
    U = eigenvectors of N for its Kmax + 1 largest eigenvalues (descending)
    E_K = D^-1/2 U[:, 1:K+1]     (the leading, trivial vector is dropped; no row normalization)

followed by seeded K-means on E_K. A proximity graph from a forest is connected in practice
(every pair of rows that ever shares a leaf is linked); ``spectral_embedding`` reports the number
of eigenvalues numerically equal to 1, which counts connected components, so a disconnected graph is
visible rather than silently mishandled (its extra unit eigenvectors are component indicators, which
K-means separates, so the same rule still applies).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import eigh


def leaf_incidence(sig: np.ndarray) -> sparse.csr_matrix:
    """Sparse n x (total leaves) 0/1 matrix from leaf signatures ``sig`` (n x T node ids)."""
    n, T = sig.shape
    width = 0
    cols = np.empty((n, T), np.int64)
    for t in range(T):
        u, inv = np.unique(sig[:, t], return_inverse=True)
        cols[:, t] = inv + width
        width += len(u)
    rows = np.repeat(np.arange(n), T)
    return sparse.csr_matrix((np.ones(n * T, np.float64), (rows, cols.ravel())), shape=(n, width))


def proximity_matrix(sig: np.ndarray) -> np.ndarray:
    """Dense n x n co-leaf proximity (float64; diagonal = 1). Intended for n up to ~5,000."""
    B = leaf_incidence(sig)
    return (B @ B.T).toarray() / sig.shape[1]


def spectral_embedding(P: np.ndarray, kmax: int) -> tuple[np.ndarray, dict]:
    """Columns 0..kmax-1 of E (see module docstring) and diagnostics."""
    n = len(P)
    W = P.copy()
    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    if np.any(d <= 0):
        raise ValueError(f"{int((d <= 0).sum())} rows share no leaf with any other row (zero degree)")
    r = 1.0 / np.sqrt(d)
    N = W * r[:, None] * r[None, :]
    vals, vecs = eigh(N, subset_by_index=[n - kmax - 1, n - 1])
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    E = vecs[:, 1:kmax + 1] * r[:, None]
    info = {"eigenvalues": [float(v) for v in vals], "n_unit_eigenvalues": int((vals > 1 - 1e-9).sum()),
            "min_degree": float(d.min()), "max_degree": float(d.max()), "solver": "scipy.linalg.eigh(subset_by_index)"}
    return E, info
