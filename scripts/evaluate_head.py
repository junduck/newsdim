"""Evaluate trained head: learning curve, per-dim metrics, plots.

Usage:
  uv run python scripts/evaluate_head.py
"""

import json
import sqlite3
import struct
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split

from newsdim.dims import DIMS
from newsdim.train.trainer import _compute_metrics, train_analytical

DB = "local_data/data.db"
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
matplotlib.use("Agg")


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
    return X, y


def learning_curve(X_train, y_train, X_val, y_val, sizes: list[int]) -> list[dict]:
    results = []
    for n in sizes:
        if n > len(X_train):
            continue
        idx = np.random.RandomState(42).choice(len(X_train), n, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]

        result = train_analytical(X_sub, y_sub)
        val_pred_raw = result.head.predict_raw(X_val)
        val_pred_clamped = result.head.predict(X_val)
        val_metrics = _compute_metrics(y_val, val_pred_raw, val_pred_clamped)

        kt_per_dim = []
        for i in range(len(DIMS)):
            tau, _ = kendalltau(y_val[:, i], val_pred_raw[:, i])
            kt_per_dim.append(float(tau) if not np.isnan(tau) else 0.0)

        entry = {
            "n": n,
            "train_metrics": result.metrics,
            "val_metrics": val_metrics,
            "val_kendall_tau": {d: kt_per_dim[i] for i, d in enumerate(DIMS)},
        }
        results.append(entry)
        overall = val_metrics["overall"]
        print(
            f"  N={n:>5}: sign={overall['sign_agreement']:.1%}  MAE={overall['mae']:.3f}  exact={overall['exact_match']:.1%}"
        )
    return results


def plot_learning_curve(curve_data: list[dict], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ns = [d["n"] for d in curve_data]

    ax = axes[0]
    for d in DIMS:
        vals = [d_["val_metrics"]["per_dim"][d]["sign_agreement"]
                for d_ in curve_data]
        ax.plot(ns, vals, marker="o", label=d)
    ax.set_xlabel("Training articles")
    ax.set_ylabel("Sign agreement")
    ax.set_title("Sign Agreement (val)")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for d in DIMS:
        vals = [d_["val_metrics"]["per_dim"][d]["mae"] for d_ in curve_data]
        ax.plot(ns, vals, marker="o", label=d)
    ax.set_xlabel("Training articles")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error (val)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for d in DIMS:
        vals = [d_["val_kendall_tau"][d] for d_ in curve_data]
        ax.plot(ns, vals, marker="o", label=d)
    ax.set_xlabel("Training articles")
    ax.set_ylabel("Kendall's tau")
    ax.set_title("Kendall's Tau (val)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"plot saved to {save_path}")


def main():
    conn = sqlite3.connect(DB)
    X, y = load_data(conn)
    conn.close()

    print(f"dataset: {X.shape[0]} articles")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)
    print(f"split:   {X_train.shape[0]} train, {X_val.shape[0]} val")

    print("\n=== Learning curve ===")
    sizes = [100, 300, 500, 1000, 1500, 2000, len(X_train)]
    curve = learning_curve(X_train, y_train, X_val, y_val, sizes)

    PLOTS_DIR.mkdir(exist_ok=True)
    plot_learning_curve(curve, PLOTS_DIR / "learning_curve.png")

    curve_path = PLOTS_DIR / "learning_curve.json"
    curve_path.write_text(json.dumps(curve, indent=2, ensure_ascii=False))
    print(f"data saved to {curve_path}")

    print("\n=== Final model (full train set) ===")
    final = train_analytical(X_train, y_train)
    val_pred_raw = final.head.predict_raw(X_val)
    val_pred_clamped = final.head.predict(X_val)
    val_metrics = _compute_metrics(y_val, val_pred_raw, val_pred_clamped)

    print(f"condition number: {final.condition_number:.1f}")
    print(
        f"overall: sign={val_metrics['overall']['sign_agreement']:.1%}  "
        f"MAE={val_metrics['overall']['mae']:.3f}  "
        f"exact={val_metrics['overall']['exact_match']:.1%}"
    )
    for d in DIMS:
        m = val_metrics["per_dim"][d]
        kt, _ = kendalltau(y_val[:, list(DIMS).index(d)],
                           val_pred_raw[:, list(DIMS).index(d)])
        print(
            f"  {d:>4}: sign={m['sign_agreement']:.1%}  MAE={m['mae']:.3f}  exact={m['exact_match']:.1%}  tau={kt:.3f}"
        )


if __name__ == "__main__":
    main()
