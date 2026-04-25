from __future__ import annotations

from pathlib import Path

import jieba

_DICT_DIR = Path(__file__).resolve().parent / "dict"
_BUNDLED_DICT = _DICT_DIR / "finance.txt"
_SUPPLEMENT_DICT = _DICT_DIR / "finance_supplement.txt"

_jieba_initialized = False
_user_dicts_loaded: set[str] = set()


def _ensure_jieba(user_dict: str | None = None) -> None:
    global _jieba_initialized

    if not _jieba_initialized:
        if _BUNDLED_DICT.exists():
            jieba.load_userdict(str(_BUNDLED_DICT))
        if _SUPPLEMENT_DICT.exists():
            jieba.load_userdict(str(_SUPPLEMENT_DICT))
        _jieba_initialized = True

    if user_dict and user_dict not in _user_dicts_loaded:
        jieba.load_userdict(user_dict)
        _user_dicts_loaded.add(user_dict)


def tokenize(text: str, user_dict: str | None = None) -> list[str]:
    _ensure_jieba(user_dict)
    return [t for t in jieba.cut(text) if len(t) >= 2]


def tokenize_batch(texts: list[str], user_dict: str | None = None) -> list[list[str]]:
    _ensure_jieba(user_dict)
    return [[t for t in jieba.cut(text) if len(t) >= 2] for text in texts]
