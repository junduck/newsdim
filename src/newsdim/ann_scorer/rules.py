"""Announcement scoring rules.

Each rule maps an announcement to a dict of {dimension: score}.
Dimensions: mom, stab, horz, eng, hype, sent, sec, pol.
Score range: -3 to +3, integer. Missing dimensions default to 0.

Strategy:
  - 3 of 4 types are type-only (资产收购, 股份质押, 对外担保)
  - 资产重组 needs content keywords (it's a grab-bag)
  - 对外担保 differentiates by size feel: 亿 feels bigger than 万
"""

from __future__ import annotations

DIMS = ("mom", "stab", "horz", "eng", "hype", "sent", "sec", "pol")


def _full(d: dict[str, int]) -> dict[str, int]:
    return {dim: d.get(dim, 0) for dim in DIMS}


def _classify_restructure(content: str) -> dict[str, int]:
    if any(kw in content for kw in ("处置", "出售", "剥离")):
        return _full({"eng": 2, "stab": -2, "sent": -2, "sec": -3})
    if any(kw in content for kw in ("收购", "购买")):
        return _full({"eng": 2, "stab": -1, "horz": 1, "sent": 1, "sec": -3})
    if "增资" in content:
        return _full({"eng": 1, "horz": 1, "sent": 1, "sec": -3})
    if any(kw in content for kw in ("设立", "成立")):
        return _full({"eng": 1, "horz": 1, "sent": 1, "sec": -3})
    return _full({"eng": 1, "sec": -3})


def _classify_guarantee(content: str) -> dict[str, int]:
    if "亿" in content:
        return _full({"eng": 2, "stab": -1, "sec": -3})
    return _full({"eng": 1, "stab": -1, "sec": -3})


_RULES = {
    "资产收购": lambda _c: _full({"eng": 2, "stab": -1, "horz": 1, "sent": 1, "sec": -3}),
    "股份质押": lambda _c: _full({"eng": 1, "stab": -1, "sent": -1, "sec": -3}),
    "对外担保": _classify_guarantee,
    "资产重组": _classify_restructure,
}


def score_announcement(event_type: str, content: str) -> dict[str, int]:
    rule = _RULES.get(event_type)
    if rule is None:
        return _full({})
    return rule(content)
