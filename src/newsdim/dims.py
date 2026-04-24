from __future__ import annotations

from dataclasses import dataclass

DIMS = ("mom", "stab", "horz", "eng", "hype", "sent", "sec", "pol")
DIM_COUNT = len(DIMS)

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


@dataclass
class DimScores:
    mom: int = 0
    stab: int = 0
    horz: int = 0
    eng: int = 0
    hype: int = 0
    sent: int = 0
    sec: int = 0
    pol: int = 0

    def to_dict(self) -> dict[str, int]:
        return {d: getattr(self, d) for d in DIMS}

    def to_array(self) -> list[int]:
        return [getattr(self, d) for d in DIMS]

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> DimScores:
        return cls(**{k: int(d[k]) for k in DIMS if k in d})

    @classmethod
    def from_array(cls, arr: list[int] | tuple[int, ...]) -> DimScores:
        return cls(**{d: int(v) for d, v in zip(DIMS, arr)})
