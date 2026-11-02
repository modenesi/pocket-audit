"""Tree cut of a nested hierarchy: most supported pockets, coarse on ties, gaps kept visible."""
import numpy as np

from pocket_audit.pipeline import tree_cut


def _levels():
    # 7 level-0 blocks (one node each) -> 3 level-1 blocks -> 2 top blocks
    l0 = np.arange(7)
    l1 = np.array([0, 0, 0, 1, 1, 2, 2])
    l2 = np.array([0, 0, 0, 0, 0, 1, 1])
    return [l0, l1, l2]


def test_splits_only_where_two_children_are_supported_and_keeps_gaps():
    # level-1 block 0 has two supported children (0, 1) and an unsupported one (2: e.g. single group);
    # level-1 block 1 has one supported child (3) -> no gain, stays whole; top block 1 cannot split.
    supported = [{0, 1, 3}, {0, 1, 2}, {0, 1}]
    labels, n = tree_cut(_levels(), supported)
    groups = [set(np.flatnonzero(labels == k)) for k in np.unique(labels)]
    assert {0} in groups and {1} in groups and {2} in groups      # split, gap child kept as its own pocket
    assert {3, 4} in groups                                       # one supported child only: not split
    assert {5, 6} in groups                                       # top block 1: children unsupported/unsplit
    assert n == 4                                                 # 0, 1, level-1 block 1, top block 1


def test_unsupported_root_is_a_gap_and_never_split():
    labels, n = tree_cut(_levels(), [{0, 1, 3}, {0, 1}, {1}])
    assert set(np.flatnonzero(labels == labels[0])) == {0, 1, 2, 3, 4}
    assert n == 1


def test_ties_stay_coarse():
    # splitting top block 0 gives level-1 blocks 0 (supported) + 1 (unsupported): 1 supported -> keep whole
    labels, n = tree_cut(_levels(), [set(), {0}, {0, 1}])
    assert len(np.unique(labels[:5])) == 1 and n == 2
