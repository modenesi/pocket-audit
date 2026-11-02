"""Task-relevant similarity: random-forest leaf signatures -> sparse mutual top-k graph.

Two observations are similar when they share terminal leaves across many trees
(co-leaf count, 0..T). The dense n x n proximity matrix is never formed:

1. For each query row and tree, sample up to ``b`` rows from its leaf (bounded
   candidate search; approximate).
2. Keep the ``q`` most frequently sampled candidates and re-score them exactly
   across all T trees.
3. Keep the top-``k`` exact scores per row and retain an undirected edge only
   when both rows list each other (mutual top-k). Edge weight = exact co-leaf count.

Work is about O(n*T*b + n*q*T) and the graph has at most n*k/2 edges.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def fit_similarity_forest(X: np.ndarray, y: np.ndarray, n_trees: int = 200,
                          min_samples_leaf: int = 20, seed: int = 0, max_features="sqrt",
                          oob: bool = False, n_jobs: int = 1, outcome: str = "binary"):
    """Outcome forest on discovery rows (legitimate features only, no protected attributes).

    ``oob=True`` keeps out-of-bag predictions (``oob_decision_function_``), a discovery-only
    criterion for choosing ``max_features`` and ``min_samples_leaf``; the trees are unchanged.
    """
    Forest = RandomForestRegressor if outcome == "continuous" else RandomForestClassifier
    rf = Forest(n_estimators=n_trees, min_samples_leaf=min_samples_leaf,
                max_features=max_features, oob_score=oob, n_jobs=n_jobs, random_state=seed)
    return rf.fit(X, y)


def leaf_signatures(forest: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """(n, T) int32 matrix of terminal-node ids; column t is only comparable within tree t."""
    sig = forest.apply(X)
    if sig.max() >= np.iinfo(np.int32).max:
        raise OverflowError("leaf ids exceed int32")
    return sig.astype(np.int32)


def _group_rank(keys: np.ndarray) -> np.ndarray:
    """Rank of each element within its run of equal (sorted) keys."""
    first = np.searchsorted(keys, keys, side="left")
    return np.arange(len(keys)) - first


def _exact_scores(sig: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = np.empty(len(a), np.int32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = (sig[a[s:s + chunk]] == sig[b[s:s + chunk]]).sum(1)
    return out


def _topk_directed(qq, cc, score, k):
    o = np.lexsort((cc, -score, qq))           # per query: score desc, then id asc (deterministic)
    qq, cc, score = qq[o], cc[o], score[o]
    keep = (_group_rank(qq) < k) & (score > 0)
    return qq[keep], cc[keep], score[keep]


def candidate_topk(sig: np.ndarray, k: int = 30, b: int = 8, q: int = 256,
                   seed: int = 0, batch: int = 2048):
    """Approximate directed top-k neighbor lists. Returns (src, dst, score) arrays."""
    n, T = sig.shape
    rng = np.random.default_rng(seed)
    orders = np.empty((T, n), np.int32)
    starts = np.empty((n, T), np.int32)
    sizes = np.empty((n, T), np.int32)
    for t in range(T):
        order = np.argsort(sig[:, t], kind="stable")
        uniq, first, counts = np.unique(sig[order, t], return_index=True, return_counts=True)
        idx = np.searchsorted(uniq, sig[:, t])
        orders[t], starts[:, t], sizes[:, t] = order, first[idx], counts[idx]
    tree_idx = np.arange(T)[None, :, None]
    src, dst, sc = [], [], []
    for s in range(0, n, batch):
        Q = np.arange(s, min(s + batch, n))
        m = len(Q)
        offs = (rng.random((m, T, b)) * sizes[Q][:, :, None]).astype(np.int32)
        cand = orders[tree_idx, starts[Q][:, :, None] + offs].reshape(m, T * b)
        qi = np.repeat(np.arange(m), T * b)
        c = cand.ravel()
        keep = c != Q[qi]
        key = qi[keep].astype(np.int64) * n + c[keep]
        uk, cnt = np.unique(key, return_counts=True)
        qq, cc = (uk // n).astype(np.int64), (uk % n).astype(np.int64)
        o = np.lexsort((cc, -cnt, qq))
        qq, cc = qq[o], cc[o]
        top = _group_rank(qq) < q
        qq, cc = Q[qq[top]], cc[top]
        score = _exact_scores(sig, qq, cc)
        a, bb, w = _topk_directed(qq, cc, score, k)
        src.append(a); dst.append(bb); sc.append(w)
    return np.concatenate(src), np.concatenate(dst), np.concatenate(sc)


def exact_topk(sig: np.ndarray, k: int = 30):
    """Brute-force directed top-k (O(n^2 T)); for tests and recall checks on small n only."""
    n = sig.shape[0]
    if n > 5000:
        raise ValueError("exact_topk is for small reference problems (n <= 5000)")
    qq = np.repeat(np.arange(n), n)
    cc = np.tile(np.arange(n), n)
    keep = qq != cc
    qq, cc = qq[keep], cc[keep]
    return _topk_directed(qq, cc, _exact_scores(sig, qq, cc), k)


def mutual_graph(src, dst, score, n: int) -> sparse.csr_matrix:
    """Undirected graph keeping (i, j) only if i lists j and j lists i. Weights = co-leaf counts."""
    fwd = src.astype(np.int64) * n + dst
    rev = dst.astype(np.int64) * n + src
    fwd_sorted = np.sort(fwd)
    pos = np.searchsorted(fwd_sorted, rev)
    pos[pos == len(fwd_sorted)] = 0
    mutual = (fwd_sorted[pos] == rev) & (src < dst)
    i, j, w = src[mutual], dst[mutual], score[mutual]
    A = sparse.coo_matrix((w, (i, j)), shape=(n, n))
    return (A + A.T).tocsr()


def neighbor_recall(approx, exact) -> float:
    """Mean over rows of |approx top-k ∩ exact top-k| / |exact top-k|."""
    a = set(zip(approx[0].tolist(), approx[1].tolist()))
    e_src, e_dst = exact[0], exact[1]
    hits = np.array([(s, d) in a for s, d in zip(e_src.tolist(), e_dst.tolist())])
    per_row = np.bincount(e_src, weights=hits) / np.maximum(np.bincount(e_src), 1)
    rows = np.unique(e_src)
    return float(per_row[rows].mean())


def build_graph(sig: np.ndarray, k: int = 30, b: int = 8, q: int = 256, seed: int = 0):
    """Sparse mutual top-k co-leaf graph plus construction statistics."""
    t0 = time.perf_counter()
    src, dst, score = candidate_topk(sig, k=k, b=b, q=q, seed=seed)
    A = mutual_graph(src, dst, score, sig.shape[0])
    deg = np.diff(A.indptr)
    stats = {"n_nodes": int(A.shape[0]), "n_edges": int(A.nnz // 2), "k": k, "b": b, "q": q,
             "isolated_nodes": int((deg == 0).sum()), "mean_degree": float(deg.mean()),
             "seconds": round(time.perf_counter() - t0, 3)}
    return A, stats
