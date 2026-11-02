"""Frozen out-of-sample pocket assignment with abstention.

Affinity of a new row x to pocket c (T trees, n_c graph rows in c):

    a_c(x) = (1/T) * sum_t n[c, t, leaf_t(x)] / n_c

which equals the mean co-leaf proximity from x to the graph rows of pocket c.
Rows abstain (label -1) when the best affinity or the best-vs-second margin is
below thresholds calibrated on held-out discovery rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ABSTAIN = -1


@dataclass
class PocketMap:
    profiles: list[np.ndarray]     # per tree: (n_nodes_t, K) float32 counts / n_c
    n_pockets: int
    pocket_sizes: np.ndarray
    score_threshold: float = 0.0
    margin_threshold: float = 0.0

    def affinity(self, sig: np.ndarray, batch: int = 20000) -> np.ndarray:
        n, T = sig.shape
        out = np.zeros((n, self.n_pockets), np.float64)
        for s in range(0, n, batch):
            blk = sig[s:s + batch]
            acc = np.zeros((len(blk), self.n_pockets), np.float64)
            for t in range(T):
                acc += self.profiles[t][blk[:, t]]
            out[s:s + batch] = acc / T
        return out

    def assign(self, sig: np.ndarray, return_scores: bool = False):
        A = self.affinity(sig)
        best, top, margin = _best(A)
        abstain = (top <= 0) | (margin <= 0) | (top < self.score_threshold) | (margin < self.margin_threshold)
        lab = np.where(abstain, ABSTAIN, best).astype(np.int32)
        return (lab, top, margin) if return_scores else lab


def _best(A: np.ndarray):
    """Deterministic argmax (lowest pocket id wins exact ties), top score, top-two margin."""
    if A.shape[1] == 0:            # no block was large enough to profile: every row abstains
        zero = np.zeros(len(A))
        return np.zeros(len(A), np.int32), zero, zero
    if A.shape[1] == 1:
        return np.zeros(len(A), np.int32), A[:, 0], A[:, 0]
    best = np.argmax(A, axis=1)
    top = A[np.arange(len(A)), best]
    part = np.partition(A, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    return best, top, margin


def build_pocket_map(sig_graph: np.ndarray, labels: np.ndarray, n_nodes_per_tree: np.ndarray,
                     min_pocket_size: int = 1) -> tuple[PocketMap, np.ndarray]:
    """Profiles from graph rows. Pockets smaller than ``min_pocket_size`` are dropped
    (their rows get no profile). Returns the map and the pocket-id relabeling used."""
    labels = np.asarray(labels)
    sizes = np.bincount(labels[labels >= 0])
    keep = np.flatnonzero(sizes >= min_pocket_size)
    remap = np.full(len(sizes), ABSTAIN, np.int32)
    remap[keep] = np.arange(len(keep))
    lab = np.where(labels >= 0, remap[np.clip(labels, 0, None)], ABSTAIN)
    K = len(keep)
    n_c = np.bincount(lab[lab >= 0], minlength=K).astype(np.float64)
    profiles = []
    for t in range(sig_graph.shape[1]):
        P = np.zeros((int(n_nodes_per_tree[t]), K), np.float32)
        ok = lab >= 0
        np.add.at(P, (sig_graph[ok, t], lab[ok]), 1.0)
        profiles.append((P / n_c[None, :]).astype(np.float32))
    return PocketMap(profiles=profiles, n_pockets=K, pocket_sizes=n_c.astype(int)), lab


def calibrate_thresholds(pmap: PocketMap, sig_calib: np.ndarray, quantile: float = 0.05) -> PocketMap:
    """Set score/margin thresholds at the ``quantile`` of held-out discovery rows."""
    _, top, margin = _best(pmap.affinity(sig_calib))
    pmap.score_threshold = float(np.quantile(top, quantile))
    pmap.margin_threshold = float(np.quantile(margin, quantile))
    return pmap


def self_excluded_agreement(pmap: PocketMap, sig_graph: np.ndarray, lab_graph: np.ndarray) -> float:
    """Share of graph rows whose affinity assignment (excluding their own contribution)
    matches their discovery label."""
    A = pmap.affinity(sig_graph)
    ok = lab_graph >= 0
    n_c = pmap.pocket_sizes.astype(float)
    A[np.flatnonzero(ok), lab_graph[ok]] -= 1.0 / n_c[lab_graph[ok]]   # remove self (1 per tree)
    best, _, _ = _best(A)
    return float((best[ok] == lab_graph[ok]).mean())
