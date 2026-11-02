"""Leiden with its own discovery-only resolution choice (proposal, 2026-10-07)."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score


def _planted_graph(seed=1, n_blocks=8, size=50, p_in=0.3, p_out=0.01):
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(n_blocks), size)
    same = blocks[:, None] == blocks[None, :]
    upper = np.triu(rng.random((len(blocks), len(blocks))) < np.where(same, p_in, p_out), 1)
    return sparse.csr_matrix((upper | upper.T).astype(float)), blocks


def test_leiden_grid_is_deterministic_and_spans_resolutions():
    pytest.importorskip("leidenalg")
    from pocket_audit.pockets import leiden_grid
    A, planted = _planted_graph()
    a = leiden_grid(A, seed=3, n_init=2)
    b = leiden_grid(A, seed=3, n_init=2)
    assert all(np.array_equal(x, y) for x, y in zip(a["levels"], b["levels"]))
    assert len(a["levels"]) == len(a["resolutions"]) == len(a["n_communities"])
    # some resolution on the grid recovers the planted communities
    assert max(adjusted_rand_score(planted, lab) for lab in a["levels"]) > 0.95
    # fine -> coarse: the community count does not increase along the grid
    assert a["n_communities"] == sorted(a["n_communities"], reverse=True)


def test_selection_rule_prefers_most_supported_then_coarser():
    from pocket_audit.pipeline import PipelineConfig, select_level
    levels = pd.DataFrame({"level": [0, 1, 2, 3], "n_supported": [1, 3, 3, 2],
                           "n_pockets_profiled": [9, 6, 4, 2], "graph_profiled_share": [1.0, 1.0, 1.0, 1.0],
                           "coverage_focal": [1.0] * 4, "coverage_reference": [1.0] * 4})
    chosen, status = select_level(levels, PipelineConfig(min_graph_profiled_share=.8))
    assert (chosen, status) == (2, "qualified")
    none = levels.assign(n_supported=[1, 1, 0, 0])
    chosen, status = select_level(none, PipelineConfig())
    assert status == "no_level_met_support_rule"
