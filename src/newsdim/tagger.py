from __future__ import annotations

from pathlib import Path


from newsdim.dims import DIMS, DimScores
from newsdim.embed.encoder import BGEEncoder
from newsdim.train.trainer import LinearHead

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_DEFAULT_WEIGHTS = _ASSETS_DIR / "head_v2_ridge1.0.npz"


class Tagger:
    """Score Chinese financial news on 8 investment-behavior dimensions.

    Architecture: frozen BAAI/bge-base-zh-v1.5 (768-dim) → trained linear
    head (ridge regression) → 8 integer scores in [-3, +3].

    Example::

        from newsdim import Tagger

        tagger = Tagger()
        scores = tagger.score("煤炭板块盘初走强，大有能源涨停")
        print(scores.to_dict())
        # {'mom': 2, 'stab': 0, 'horz': -1, 'eng': 1, 'hype': 1, 'sent': 1, 'sec': 0, 'pol': 0}

    Args:
        weights_path: Path to trained linear head weights (``.npz``).
            Defaults to the bundled weights trained with ridge=1.0.
        device: Torch device for the embedding model (e.g. ``"cuda"``,
            ``"mps"``). Defaults to auto-detection.
    """

    def __init__(self, weights_path: str | Path | None = None, device: str | None = None):
        self._encoder = BGEEncoder(device=device)
        weights = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self._head = LinearHead.load(weights)

    @property
    def encoder(self) -> BGEEncoder:
        """The underlying BGE encoder instance."""
        return self._encoder

    @property
    def head(self) -> LinearHead:
        """The trained linear head."""
        return self._head

    def score(self, text: str) -> DimScores:
        """Score a single article. Returns integer scores (-3 to +3).

        Args:
            text: Chinese financial news text.

        Returns:
            :class:`~newsdim.DimScores` with 8 integer dimension scores.
        """
        emb = self._encoder.encode([text])
        pred = self._head.predict(emb)[0]
        return DimScores.from_array(pred.tolist())

    def score_batch(self, texts: list[str], batch_size: int = 64) -> list[DimScores]:
        """Score multiple articles. More efficient than calling :meth:`score` in a loop.

        Args:
            texts: List of article texts.
            batch_size: Encoding batch size.

        Returns:
            List of :class:`~newsdim.DimScores`, one per input text.
        """
        if not texts:
            return []
        embeddings = self._encoder.encode(texts, batch_size=batch_size)
        preds = self._head.predict(embeddings)
        return [DimScores.from_array(row.tolist()) for row in preds]

    def score_raw(self, text: str) -> dict[str, float]:
        """Score a single article returning raw floats (before rounding).

        Useful for ranking where finer granularity matters. A raw score
        near 0 indicates genuine uncertainty — the model has no strong signal.

        Args:
            text: Chinese financial news text.

        Returns:
            Dict mapping dimension keys to raw float scores.
        """
        emb = self._encoder.encode([text])
        raw = self._head.predict_raw(emb)[0]
        return {d: float(v) for d, v in zip(DIMS, raw)}

    def score_batch_raw(self, texts: list[str], batch_size: int = 64) -> list[dict[str, float]]:
        """Batch version of :meth:`score_raw`."""
        if not texts:
            return []
        embeddings = self._encoder.encode(texts, batch_size=batch_size)
        raws = self._head.predict_raw(embeddings)
        return [{d: float(v) for d, v in zip(DIMS, row)} for row in raws]
