from __future__ import annotations

import bm25s

from newsdim.retrieval.tokenizer import tokenize, tokenize_batch


class Corpus:
    """A BM25-searchable collection of Chinese financial news articles.

    Tokenizes documents with jieba + bundled financial dictionary at add time.
    Builds BM25 index lazily on first query and invalidates it on new adds.

    Args:
        user_dict: Optional path to a user jieba dictionary file, appended
            to the bundled financial dictionary.

    Example::

        from newsdim.retrieval import Corpus

        corpus = Corpus()
        corpus.add("央行宣布降准50个基点")
        corpus.add_batch(["煤炭板块盘初走强", "财政部公布普惠金融名单"])

        hits = corpus.get(["降准", "货币政策"], top_k=5)
        # [(0, 2.34)]

        scores = corpus.score_all(["降准"])
        # [2.34, 0.0, 0.0]

        # Serialize for persistence
        records = corpus.to_records()
        corpus2 = Corpus.from_records(records)
    """

    def __init__(self, user_dict: str | None = None) -> None:
        self._user_dict = user_dict
        self._texts: list[str] = []
        self._tokens: list[list[str]] = []
        self._index: bm25s.BM25 | None = None
        self._dirty: bool = True

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, text: str) -> None:
        """Add a single document to the corpus. Tokenizes immediately."""
        self._texts.append(text)
        self._tokens.append(tokenize(text, user_dict=self._user_dict))
        self._dirty = True

    def add_batch(self, texts: list[str]) -> None:
        """Add multiple documents. More efficient than calling :meth:`add` in a loop."""
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
        """Query the corpus for documents matching given keywords.

        Keywords are used as-is for BM25 matching (no re-tokenization).
        Returns results sorted by BM25 score descending, excluding zero-score
        documents.

        Args:
            keywords: List of keyword strings to match.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(index, score)`` tuples, sorted by score descending.
            Indices correspond to insertion order (0-based).
        """
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
        """Score every document against given keywords.

        Useful for materializing BM25 scores into a database column for
        SQL-level sorting.

        Args:
            keywords: List of keyword strings to match.

        Returns:
            List of float scores, one per document in insertion order.
        """
        if not self._texts or not keywords:
            return [0.0] * len(self._texts)
        self._ensure_index()
        if self._index is None:
            return [0.0] * len(self._texts)

        scores_matrix = self._index.get_scores(keywords)
        return [float(s) for s in scores_matrix]

    def to_records(self) -> list[dict]:
        """Serialize corpus to a list of dicts for user-controlled persistence.

        Each record contains ``"text"`` (original text) and ``"tokens"``
        (pre-tokenized list of strings). Use :meth:`from_records` to
        reconstruct without re-tokenizing.

        Returns:
            List of ``{"text": str, "tokens": list[str]}`` dicts.
        """
        return [
            {"text": self._texts[i], "tokens": list(self._tokens[i])}
            for i in range(len(self._texts))
        ]

    @classmethod
    def from_records(
        cls, records: list[dict], user_dict: str | None = None
    ) -> Corpus:
        """Reconstruct a Corpus from serialized records.

        Skips tokenization — tokens are taken directly from records.
        BM25 index is rebuilt lazily on first query.

        Args:
            records: Output of :meth:`to_records`, or a concatenated list
                for merging multiple corpora.
            user_dict: Optional path to a user jieba dictionary.

        Returns:
            A new :class:`Corpus` instance.
        """
        corpus = cls(user_dict=user_dict)
        for rec in records:
            corpus._texts.append(rec["text"])
            corpus._tokens.append(list(rec["tokens"]))
        return corpus
