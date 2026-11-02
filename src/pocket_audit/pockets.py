"""Pocket discovery on the sparse co-leaf graph.

Primary: degree-corrected nested SBM (graph-tool) on the mutual top-k graph; the
hierarchy gives candidate resolutions. Robustness comparison: Leiden (flat
partitions; resolution tuned to match a target number of pockets).

Weighting (verified 2026-09-28): by default both methods use the unweighted graph.
Co-leaf counts already decide which edges exist (top-k by exact count). Passing the
counts as SBM edge multiplicities (``weighting="multiplicity"``) treats a count of 40
as 40 independent edges; on a 4,800-node simulation graph this produced 4,796 level-0
blocks (versus 80 -> 22 -> 7 -> 3 unweighted) and ran 2.6x slower, so it is kept only
as an option for comparison.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse


def _relabel(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1].astype(np.int32)


def _edges(A: sparse.csr_matrix):
    U = sparse.triu(A, k=1).tocoo()
    return U.row.astype(np.int64), U.col.astype(np.int64), U.data


def sbm_hierarchy(A: sparse.csr_matrix, seed: int = 0, n_init: int = 3,
                  weighting: str = "none") -> dict:
    """Fit ``n_init`` seeded nested SBMs; keep the lowest description length.

    Returns {"levels": [labels per level, fine -> coarse, deduplicated],
             "n_blocks": [...], "entropy": best, "entropies": all, "seconds": t}.
    """
    import graph_tool.all as gt                    # graph_tool.inference alone hits a circular import
    minimize_nested_blockmodel_dl = gt.minimize_nested_blockmodel_dl

    t0 = time.perf_counter()
    i, j, w = _edges(A)
    if weighting not in ("none", "multiplicity"):
        raise ValueError("weighting must be 'none' or 'multiplicity'")
    if weighting == "multiplicity" and (np.any(w != np.round(w)) or np.any(w <= 0)):
        raise ValueError("SBM edge multiplicities must be positive integers")
    fits = []
    for r in range(n_init):
        g = gt.Graph(directed=False)
        g.add_vertex(A.shape[0])
        g.add_edge_list(np.column_stack([i, j]))
        state_args = dict(deg_corr=True)
        if weighting == "multiplicity":
            ew = g.new_edge_property("int")
            ew.a[:] = w.astype(np.int64)
            state_args["eweight"] = ew
        gt.seed_rng(seed + r)
        np.random.seed(seed + r)
        state = minimize_nested_blockmodel_dl(g, state_args=state_args)
        fits.append((state.entropy(), state))
    entropies = [float(e) for e, _ in fits]
    best = fits[int(np.argmin(entropies))][1]
    levels, n_blocks = [], []
    # Node partitions come from composing the per-level block maps (bs[0]: node -> level-0
    # block; bs[l]: level-(l-1) block -> level-l block). With graph-tool 3.9 on CHPC,
    # project_level(l).get_blocks() returned labels that were not the nested partition
    # (e.g. 14 distinct labels at a level with 11 blocks), so it is not used.
    bs = [np.asarray(b, dtype=np.int64) for b in best.get_bs()]
    current = bs[0]
    for level in range(len(bs)):
        if level > 0:
            current = bs[level][current]
        lab = _relabel(current)
        k = int(lab.max()) + 1
        if k == 1:
            break
        if not n_blocks or k != n_blocks[-1]:
            levels.append(lab)
            n_blocks.append(k)
    return {"levels": levels, "n_blocks": n_blocks, "entropy": min(entropies),
            "entropies": entropies, "seconds": round(time.perf_counter() - t0, 2)}


def leiden_partition(A: sparse.csr_matrix, n_pockets: int, seed: int = 0,
                     max_iter: int = 30, weighted: bool = False) -> dict:
    """Leiden (RB configuration, weighted) with resolution bisected to hit ``n_pockets``
    communities of size >= 1% of nodes (smaller communities are counted separately)."""
    import igraph as ig
    import leidenalg as la

    t0 = time.perf_counter()
    i, j, w = _edges(A)
    g = ig.Graph(n=A.shape[0], edges=np.column_stack([i, j]).tolist())
    g.es["weight"] = (w.astype(float) if weighted else np.ones(len(w))).tolist()
    min_size = max(1, int(0.01 * A.shape[0]))

    def run(res):
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=res, seed=seed)
        lab = np.asarray(part.membership)
        sizes = np.bincount(lab)
        return lab, int((sizes >= min_size).sum())

    lo, hi = 1e-4, 10.0
    best = None
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        lab, k = run(mid)
        if best is None or abs(k - n_pockets) < abs(best[1] - n_pockets):
            best = (lab, k, mid)
        if k == n_pockets:
            break
        lo, hi = (mid, hi) if k < n_pockets else (lo, mid)
    lab, k, res = best
    return {"labels": _relabel(lab), "n_large": k, "resolution": float(res),
            "seconds": round(time.perf_counter() - t0, 2)}


LEIDEN_GRID = (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)   # fine -> coarse 


def leiden_grid(A: sparse.csr_matrix, resolutions=LEIDEN_GRID, seed: int = 0, n_init: int = 3) -> dict:
    """Leiden (RB configuration, unweighted) at each resolution of a fixed grid, fine -> coarse.

    Proposal (2026-10-07; not the validated primary): gives Leiden its own candidate resolutions, to be
    chosen by the same discovery-only support rule as the SBM levels (``pipeline.select_level``). At each
    resolution, ``n_init`` seeded fits are run and the one with the best objective *at that resolution*
    is kept (objectives are not compared across resolutions); ties go to the lower seed.
    Returns {"levels": [labels per resolution], "resolutions": [...], "n_communities": [...], "seconds": t}.
    """
    import igraph as ig
    import leidenalg as la

    t0 = time.perf_counter()
    i, j, _ = _edges(A)
    g = ig.Graph(n=A.shape[0], edges=np.column_stack([i, j]).tolist())
    levels, ks = [], []
    for res in resolutions:
        best = None
        for r in range(n_init):
            part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=res,
                                     seed=int(seed) + r)
            q = part.quality()
            if best is None or q > best[0] + 1e-12:
                best = (q, np.asarray(part.membership))
        lab = _relabel(best[1])
        levels.append(lab)
        ks.append(int(lab.max()) + 1)
    return {"levels": levels, "resolutions": list(resolutions), "n_communities": ks,
            "seconds": round(time.perf_counter() - t0, 2)}
