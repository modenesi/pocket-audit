"""Optional GPU path: EXACT directed top-k co-leaf neighbors (needs PyTorch with CUDA).

``graph.candidate_topk`` is approximate (bounded candidate search). On a GPU the exact
co-leaf counts for every pair are cheap enough to compute block by block:

    count[i, j] = sum_t 1{sig[i, t] == sig[j, t]}

For a block of m query rows the kernel loops over the T trees, comparing an (m, 1) column
with a (1, n) row and accumulating into an (m, n) int16 matrix, so memory is O(m n) and work
is O(n^2 T). Ties are broken exactly as on the CPU (score descending, then neighbor id
ascending) by ranking on the composite key score * 2^31 + (2^31 - 1 - id). The output has the
same (src, dst, score) form as ``candidate_topk``, so ``graph.mutual_graph`` builds the
undirected graph unchanged. Nothing in the default pipeline imports this module.
"""

from __future__ import annotations

import time

import numpy as np

from . import graph as G


def exact_topk_gpu(sig: np.ndarray, k: int = 30, block: int = 1024, device: str = "cuda"):
    """Exact directed top-k by co-leaf count. Returns (src, dst, score) numpy arrays."""
    import torch

    n, T = sig.shape
    if T > np.iinfo(np.int16).max:
        raise ValueError("tree count exceeds GPU int16 co-leaf accumulator")
    if block < 1 or n < 2 or k < 1:
        raise ValueError("positive block/k and at least two rows required")
    if n >= 2 ** 31:
        raise ValueError("too many rows for the tie-break key")
    S = torch.as_tensor(np.ascontiguousarray(sig.T), dtype=torch.int32, device=device)   # (T, n)
    ids = torch.arange(n, device=device, dtype=torch.int64)
    tiebreak = (2 ** 31 - 1) - ids                                                     # smaller id wins
    src, dst, sc = [], [], []
    for s in range(0, n, block):
        q = ids[s:s + block]
        m = len(q)
        count = torch.zeros((m, n), dtype=torch.int16, device=device)
        for t in range(T):
            count += (S[t, q][:, None] == S[t][None, :])
        count[torch.arange(m, device=device), q] = -1                                  # no self edges
        key = count.to(torch.int64) * (2 ** 31) + tiebreak[None, :]
        top = torch.topk(key, k=min(k, n - 1), dim=1).indices
        score = torch.gather(count, 1, top)
        keep = score > 0
        src.append(q[:, None].expand(-1, top.shape[1])[keep].cpu().numpy())
        dst.append(top[keep].cpu().numpy())
        sc.append(score[keep].to(torch.int32).cpu().numpy())
    return np.concatenate(src), np.concatenate(dst), np.concatenate(sc)


def build_graph_exact_gpu(sig: np.ndarray, k: int = 30, block: int = 1024, device: str = "cuda"):
    """Mutual top-k co-leaf graph from exact GPU neighbor lists; same stats as ``graph.build_graph``."""
    import torch

    t0 = time.perf_counter()
    src, dst, score = exact_topk_gpu(sig, k=k, block=block, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    A = G.mutual_graph(src, dst, score, sig.shape[0])
    deg = np.diff(A.indptr)
    stats = {"n_nodes": int(A.shape[0]), "n_edges": int(A.nnz // 2), "k": k, "exact": True,
             "isolated_nodes": int((deg == 0).sum()), "mean_degree": float(deg.mean()),
             "seconds": round(time.perf_counter() - t0, 3)}
    return A, stats
