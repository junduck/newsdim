from __future__ import annotations

import bm25s

from newsdim.retrieval.tokenizer import tokenize, tokenize_batch


class Corpus:
    def __init__(self, user_dict: str | None = None) -> None:
        self._user_dict = user_dict
        self._texts: list[str] = []
        self._tokens: list[list[str]] = []
        self._index: bm25s.BM25 | None = None
        self._dirty: bool = True

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, text: str) -> None:
        self._texts.append(text)
        self._tokens.append(tokenize(text, user_dict=self._user_dict))
        self._dirty = True

    def add_batch(self, texts: list[str]) -> None:
        if not texts:
            return
        batch_tokens = tokenize_batch(texts, user_dict=self._user_dict)
        self._texts.extend(texts)
        self._tokens.extend(batch_tokens)
        self._dirty = True

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        if not self._tokens:
            return
        self._index = bm25s.BM25()
        self._index.index(self._tokens)
        self._dirty = False

    def get(self, keywords: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        if not self._texts or not keywords:
            return []
        self._ensure_index()
        if self._index is None:
            return []

        results, scores = self._index.retrieve(
            [keywords], k=min(top_k, len(self._texts))
        )
        return [
            (int(idx), float(scores[0][j]))
            for j, idx in enumerate(results[0])
            if float(scores[0][j]) > 0
        ]

    def score_all(self, keywords: list[str]) -> list[float]:
        if not self._texts or not keywords:
            return [0.0] * len(self._texts)
        self._ensure_index()
        if self._index is None:
            return [0.0] * len(self._texts)

        scores_matrix = self._index.get_scores(keywords)
        return [float(s) for s in scores_matrix]

    def to_records(self) -> list[dict]:
        return [
            {"text": self._texts[i], "tokens": list(self._tokens[i])}
            for i in range(len(self._texts))
        ]

    @classmethod
    def from_records(
        cls, records: list[dict], user_dict: str | None = None
    ) -> Corpus:
        corpus = cls(user_dict=user_dict)
        for rec in records:
            corpus._texts.append(rec["text"])
            corpus._tokens.append(list(rec["tokens"]))
        return corpus
