"""Outcome-blind allocation of independent units to analysis roles.

Whole units (hospitals for WiDS, patients for UCI) are assigned to roles, so no
unit ever crosses roles. Allocation uses only unit identifiers and a seed; it
never looks at outcomes or protected attributes.
"""

from __future__ import annotations

import numpy as np

ROLES = ("development", "discovery", "audit")


def allocate_units(unit: np.ndarray, fractions=(0.4, 0.3, 0.3), roles=ROLES,
                   seed: int = 20260928) -> np.ndarray:
    """Return a role label per row. Fractions apply to the number of units."""
    fractions = np.asarray(fractions, float)
    if len(fractions) != len(roles) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("fractions must be positive, one per role, and sum to 1")
    uniq, inverse = np.unique(np.asarray(unit).astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    # largest-remainder allocation of unit counts
    raw = fractions * len(uniq)
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts))[: len(uniq) - counts.sum()]] += 1
    unit_role = np.empty(len(uniq), dtype=object)
    start = 0
    for role, k in zip(roles, counts):
        unit_role[order[start:start + k]] = role
        start += k
    return unit_role[inverse].astype(str)


def check_roles(unit: np.ndarray, role: np.ndarray) -> None:
    """Raise if any unit appears in more than one role."""
    unit = np.asarray(unit).astype(str)
    pairs = np.unique(np.char.add(np.char.add(unit, "\x00"), np.asarray(role).astype(str)))
    units_of_pairs = np.array([p.split("\x00")[0] for p in pairs])
    if len(units_of_pairs) != len(np.unique(units_of_pairs)):
        raise AssertionError("a unit crosses roles")
