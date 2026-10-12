"""Planar 2-link arm geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number, got {value!r}.")
    return value


@dataclass(frozen=True)
class PlanarArm:
    """Two rigid links in a plane, units in metres.

    Defaults (l1 = 1.0, l2 = 0.8) give a reachable annulus 0.2 <= r <= 1.8.
    """

    l1: float = 1.0
    l2: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "l1", _positive_finite(self.l1, "l1"))
        object.__setattr__(self, "l2", _positive_finite(self.l2, "l2"))

    @property
    def max_reach(self) -> float:
        """Outer workspace radius, l1 + l2."""
        return self.l1 + self.l2

    @property
    def min_reach(self) -> float:
        """Inner workspace radius, |l1 - l2| (0 when the links are equal)."""
        return abs(self.l1 - self.l2)
