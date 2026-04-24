from __future__ import annotations

from pathlib import Path


from newsdim.dims import DIMS, DimScores
from newsdim.embed.encoder import BGEEncoder
from newsdim.train.trainer import LinearHead

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_DEFAULT_WEIGHTS = _ASSETS_DIR / "head_v2_ridge1.0.npz"


class Tagger:
    def __init__(self, weights_path: str | Path | None = None, device: str | None = None):
        self._encoder = BGEEncoder(device=device)
        weights = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self._head = LinearHead.load(weights)

    @property
    def encoder(self) -> BGEEncoder:
        return self._encoder

    @property
    def head(self) -> LinearHead:
        return self._head

    def score(self, text: str) -> DimScores:
        emb = self._encoder.encode([text])
        pred = self._head.predict(emb)[0]
        return DimScores.from_array(pred.tolist())

    def score_batch(self, texts: list[str], batch_size: int = 64) -> list[DimScores]:
        if not texts:
            return []
        embeddings = self._encoder.encode(texts, batch_size=batch_size)
        preds = self._head.predict(embeddings)
        return [DimScores.from_array(row.tolist()) for row in preds]

    def score_raw(self, text: str) -> dict[str, float]:
        emb = self._encoder.encode([text])
        raw = self._head.predict_raw(emb)[0]
        return {d: float(v) for d, v in zip(DIMS, raw)}

    def score_batch_raw(self, texts: list[str], batch_size: int = 64) -> list[dict[str, float]]:
        if not texts:
            return []
        embeddings = self._encoder.encode(texts, batch_size=batch_size)
        raws = self._head.predict_raw(embeddings)
        return [{d: float(v) for d, v in zip(DIMS, row)} for row in raws]
