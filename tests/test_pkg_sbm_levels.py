"""Nested SBM levels must be the fitted nested partition (needs graph-tool).

With graph-tool 3.9, ``project_level(l).get_blocks()`` did not return the nested node
partition: on this planted 8-community graph it gave 11 labels at level 0 and 2 at the
top, while composing ``get_bs()`` recovers the 8 communities.
"""
import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from pocket_audit.pockets import sbm_hierarchy


def test_default_hierarchy_allows_low_coverage_and_single_group_pockets():
    from pocket_audit.pipeline import PipelineConfig, _projected_support

    cfg = PipelineConfig()
    # Two comparable pockets, plus large pockets containing only one group each.
    labels = np.repeat(np.arange(4), [300, 300, 2000, 2000])
    groups = np.concatenate([
        np.tile(["focal", "reference"], 150),
        np.tile(["focal", "reference"], 150),
        np.repeat("focal", 2000), np.repeat("reference", 2000),
    ])
    support = _projected_support(labels, groups, "focal", "reference", 4,
                                 scale=1, min_count=cfg.min_projected_per_group)
    assert support.supported.tolist() == [True, True, False, False]
    supported_labels = support.loc[support.supported, "pocket"].to_numpy()
    coverage = [np.isin(labels[groups == g], supported_labels).mean()
                for g in ("focal", "reference")]
    assert min(coverage) < 0.8
    assert cfg.min_group_coverage == 0.0
    assert support.supported.sum() >= 2 and min(coverage) >= cfg.min_group_coverage


def _planted_graph(seed=1, n_blocks=8, size=50, p_in=0.3, p_out=0.01):
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(n_blocks), size)
    same = blocks[:, None] == blocks[None, :]
    upper = np.triu(rng.random((len(blocks), len(blocks))) < np.where(same, p_in, p_out), 1)
    return sparse.csr_matrix((upper | upper.T).astype(float)), blocks


def test_sbm_levels_recover_planted_blocks_and_nest():
    pytest.importorskip("graph_tool")
    A, planted = _planted_graph()
    hier = sbm_hierarchy(A, seed=0, n_init=1)
    assert hier["n_blocks"][0] == 8
    assert adjusted_rand_score(planted, hier["levels"][0]) > 0.95
    sizes = [int(lab.max()) + 1 for lab in hier["levels"]]
    assert sizes == hier["n_blocks"] and sizes == sorted(sizes, reverse=True)
    for fine, coarse in zip(hier["levels"][:-1], hier["levels"][1:]):
        pairs = np.unique(np.column_stack([fine, coarse]), axis=0)
        assert len(pairs) == fine.max() + 1   # each finer pocket sits inside one coarser pocket
