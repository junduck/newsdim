from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from newsdim.dims import DIMS


@dataclass
class LinearHead:
    weight: np.ndarray  # (768, 8)
    bias: np.ndarray  # (8,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = X @ self.weight + self.bias
        return np.clip(np.round(raw), -3, 3).astype(int)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weight + self.bias

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self.weight, bias=self.bias)

    @classmethod
    def load(cls, path: str | Path) -> LinearHead:
        data = np.load(path)
        return cls(weight=data["weight"], bias=data["bias"])


@dataclass
class TrainResult:
    head: LinearHead
    metrics: dict
    condition_number: float


def train_analytical(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float = 0.0,
) -> TrainResult:
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    X_b = np.hstack([X, ones])

    if ridge > 0:
        A = X_b.T @ X_b + ridge * np.eye(X_b.shape[1])
        b = X_b.T @ y
        W, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    else:
        W, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

    cond = np.linalg.cond(X_b.T @ X_b)

    weight = W[:-1, :]
    bias = W[-1, :]

    head = LinearHead(weight=weight, bias=bias)

    pred_raw = head.predict_raw(X)
    pred_clamped = head.predict(X)

    metrics = _compute_metrics(y, pred_raw, pred_clamped)

    return TrainResult(head=head, metrics=metrics, condition_number=float(cond))


def _compute_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray, y_pred_clamped: np.ndarray) -> dict:
    mae_per_dim = np.mean(np.abs(y_pred_raw - y_true), axis=0)
    exact_match_per_dim = np.mean(y_pred_clamped == y_true, axis=0)

    nonzero_mask = y_true != 0
    sign_agree_per_dim = np.zeros(len(DIMS))
    for i in range(len(DIMS)):
        nz = nonzero_mask[:, i]
        if nz.sum() > 0:
            sign_agree_per_dim[i] = np.mean(np.sign(y_pred_raw[nz, i]) == np.sign(y_true[nz, i]))
        else:
            sign_agree_per_dim[i] = 1.0

    dim_metrics = {}
    for i, d in enumerate(DIMS):
        dim_metrics[d] = {
            "mae": float(mae_per_dim[i]),
            "sign_agreement": float(sign_agree_per_dim[i]),
            "exact_match": float(exact_match_per_dim[i]),
        }

    return {
        "per_dim": dim_metrics,
        "overall": {
            "mae": float(np.mean(mae_per_dim)),
            "sign_agreement": float(np.mean(sign_agree_per_dim)),
            "exact_match": float(np.mean(exact_match_per_dim)),
        },
    }
