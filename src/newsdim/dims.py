from __future__ import annotations

from dataclasses import dataclass

DIMS = ("mom", "stab", "horz", "eng", "hype", "sent", "sec", "pol")
"""Canonical dimension keys in fixed order."""

DIM_COUNT = len(DIMS)
"""Number of scoring dimensions (8)."""

DIM_LABELS = {
    "mom": "动势",
    "stab": "稳定性",
    "horz": "时间尺度",
    "eng": "事件活跃度",
    "hype": "关注度",
    "sent": "情绪",
    "sec": "范围",
    "pol": "政策",
}
"""Chinese labels for each dimension key."""


@dataclass
class DimScores:
    """Integer scores (-3 to +3) on 8 investment-behavior dimensions.

    Most dimensions default to 0. ``eng`` and ``hype`` are non-negative
    by design. ``sec`` is bidirectional: positive = broad/macro scope,
    negative = narrow/micro scope (not good/bad).

    Attributes:
        mom: Momentum — price trend direction.
        stab: Stability — business predictability.
        horz: Horizon — investment time scale.
        eng: Event activity — how notable the event is (>= 0).
        hype: Hype — market attention level (>= 0).
        sent: Sentiment — positive/negative tone.
        sec: Scope — impact breadth. +3 = macro, -3 = single company.
        pol: Policy — policy relevance.
    """

    mom: int = 0
    stab: int = 0
    horz: int = 0
    eng: int = 0
    hype: int = 0
    sent: int = 0
    sec: int = 0
    pol: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return scores as a dict keyed by :data:`DIMS`."""
        return {d: getattr(self, d) for d in DIMS}

    def to_array(self) -> list[int]:
        """Return scores as a list in :data:`DIMS` order."""
        return [getattr(self, d) for d in DIMS]

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> DimScores:
        """Construct from a dict, ignoring unknown keys."""
        return cls(**{k: int(d[k]) for k in DIMS if k in d})

    @classmethod
    def from_array(cls, arr: list[int] | tuple[int, ...]) -> DimScores:
        """Construct from a sequence in :data:`DIMS` order."""
        return cls(**{d: int(v) for d, v in zip(DIMS, arr)})
