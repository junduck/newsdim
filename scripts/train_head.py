"""Train linear head on pre-computed embeddings.

Usage:
  uv run python scripts/train_head.py
  uv run python scripts/train_head.py --ridge 1.0
"""

import argparse
import json
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from newsdim.train.trainer import DIMS, train_analytical

DB = "local_data/data.db"
MODELS_DIR = Path("models")


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


def load_data(conn: sqlite3.Connection):
    rows = conn.execute("""
        SELECT e.source, e.date, e.seq_id, e.embedding,
               s.mom, s.stab, s.horz, s.eng, s.hype, s.sent, s.sec, s.pol
        FROM news_embedding e
        INNER JOIN news_score s
            ON e.source = s.source AND e.date = s.date AND e.seq_id = s.seq_id
    """).fetchall()

    X = np.stack([_blob_to_vec(r[3]) for r in rows])
    y = np.array([list(r[4:12]) for r in rows], dtype=np.float32)
    keys = [(r[0], r[1], r[2]) for r in rows]
    return X, y, keys


def main():
    parser = argparse.ArgumentParser(description="Train linear head")
    parser.add_argument("--ridge", type=float, default=0.0, help="Ridge regularization (0=pure lstsq)")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    X, y, keys = load_data(conn)
    conn.close()

    print(f"dataset: {X.shape[0]} articles, {X.shape[1]}-dim embeddings")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=args.seed,
    )
    print(f"split:   {X_train.shape[0]} train, {X_val.shape[0]} val")

    result = train_analytical(X_train, y_train, ridge=args.ridge)
    print(f"condition number: {result.condition_number:.1f}")

    print("\n=== Train metrics ===")
    print_metrics(result.metrics)

    val_pred_raw = result.head.predict_raw(X_val)
    val_pred_clamped = result.head.predict(X_val)
    from newsdim.train.trainer import _compute_metrics

    val_metrics = _compute_metrics(y_val, val_pred_raw, val_pred_clamped)

    print("\n=== Val metrics ===")
    print_metrics(val_metrics)

    version = f"v2_ridge{args.ridge}"
    model_path = MODELS_DIR / f"head_{version}.npz"
    result.head.save(model_path)
    print(f"\nweights saved to {model_path}")

    meta = {
        "version": version,
        "ridge": args.ridge,
        "seed": args.seed,
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "condition_number": result.condition_number,
        "train_metrics": result.metrics,
        "val_metrics": val_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = MODELS_DIR / f"head_{version}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"meta saved to {meta_path}")


def print_metrics(metrics: dict) -> None:
    print(
        f"  overall: MAE={metrics['overall']['mae']:.3f}  "
        f"sign={metrics['overall']['sign_agreement']:.1%}  "
        f"exact={metrics['overall']['exact_match']:.1%}"
    )
    for d in DIMS:
        m = metrics["per_dim"][d]
        print(f"    {d:>4}: MAE={m['mae']:.3f}  sign={m['sign_agreement']:.1%}  exact={m['exact_match']:.1%}")


if __name__ == "__main__":
    main()
