"""LoRA fine-tuning of BGE + linear head for news dimension scoring.

End-to-end: unfreezes BGE via LoRA adapters, trains with gradient descent,
then merges LoRA back into BGE and retrains the linear head analytically.

Usage:
  uv run python scripts/train_lora.py --epochs 10 --rank 16 --merge
  uv run python scripts/train_lora.py --rank 8 --lr 2e-4 --device cuda
  uv run python scripts/train_lora.py --resume models/lora_ckpt_ep5.pt
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

from newsdim.dims import DIMS
from newsdim.train.trainer import _compute_metrics
from newsdim.embed.encoder import MODEL_NAME

DB = "local_data/data.db"
MODELS_DIR = Path("models")
CKPT_PREFIX = "lora_ckpt_ep"


def load_data(conn: sqlite3.Connection):
    rows = conn.execute("""
        SELECT n.source, n.date, n.seq_id, n.content,
               s.mom, s.stab, s.horz, s.eng, s.hype, s.sent, s.sec, s.pol
        FROM news n
        INNER JOIN news_score s
            ON n.source = s.source AND n.date = s.date AND n.seq_id = s.seq_id
        ORDER BY n.source, n.date, n.seq_id
    """).fetchall()

    texts = [r[3] for r in rows]
    y = np.array([list(r[4:12]) for r in rows], dtype=np.float32)
    return texts, y


class SignWeightedMSELoss(nn.Module):
    def __init__(self, sign_weight: float = 2.0):
        super().__init__()
        self.sign_weight = sign_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        sign_mismatch = (torch.sign(pred) != torch.sign(target)) & (target != 0)
        weights = torch.where(sign_mismatch, self.sign_weight, 1.0)
        return (weights * mse).mean()


def encode_texts(model, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    model.eval()
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embs, dtype=np.float32)


def save_checkpoint(peft_model, head, optimizer, epoch, best_val_sign, best_head_state, path: Path):
    torch.save({
        "epoch": epoch,
        "peft_state_dict": peft_model.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_sign": best_val_sign,
        "best_head_state": best_head_state,
    }, path)
    print(f"  checkpoint saved to {path}")


def load_checkpoint(path: str | Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def train_lora(
    texts_train: list[str],
    y_train: np.ndarray,
    texts_val: list[str],
    y_val: np.ndarray,
    rank: int = 16,
    lr: float = 1e-4,
    head_lr: float = 1e-3,
    epochs: int = 15,
    batch_size: int = 32,
    sign_weight: float = 2.0,
    device: str = "auto",
    seed: int = 42,
    resume_path: str | None = None,
    checkpoint_every: int = 1,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(seed)

    base_model = SentenceTransformer(MODEL_NAME, device=device)
    dim = base_model.get_embedding_dimension()

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    head = nn.Linear(dim, len(DIMS), device=device)
    loss_fn = SignWeightedMSELoss(sign_weight=sign_weight)

    optimizer = torch.optim.AdamW([
        {"params": peft_model.parameters(), "lr": lr},
        {"params": head.parameters(), "lr": head_lr},
    ], weight_decay=0.01)

    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    start_epoch = 1
    best_val_sign = 0.0
    best_head_state = None
    patience_counter = 0

    if resume_path:
        ckpt = load_checkpoint(resume_path, device)
        peft_model.load_state_dict(ckpt["peft_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_sign = ckpt["best_val_sign"]
        best_head_state = ckpt["best_head_state"]
        print(f"resumed from epoch {ckpt['epoch']}, best_val_sign={best_val_sign:.1%}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- epoch {epoch}/{epochs} ---")

        X_train_emb = encode_texts(peft_model, texts_train, batch_size, device)
        X_val_emb = encode_texts(peft_model, texts_val, batch_size, device)

        X_train_t = torch.tensor(X_train_emb, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_emb, dtype=torch.float32, device=device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        peft_model.train()
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = head(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        head.eval()
        with torch.no_grad():
            val_pred = head(X_val_t).cpu().numpy()
            val_clamped = np.clip(np.round(val_pred), -3, 3).astype(int)
            val_metrics = _compute_metrics(y_val, val_pred, val_clamped)

        val_sign = val_metrics["overall"]["sign_agreement"]
        avg_loss = epoch_loss / n_batches
        print(
            f"  loss={avg_loss:.4f}  val_sign={val_sign:.1%}  "
            f"val_mae={val_metrics['overall']['mae']:.3f}  "
            f"val_exact={val_metrics['overall']['exact_match']:.1%}"
        )

        if val_sign > best_val_sign:
            best_val_sign = val_sign
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
            print("  ** new best **")
        else:
            patience_counter += 1

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = MODELS_DIR / f"{CKPT_PREFIX}{epoch}.pt"
            MODELS_DIR.mkdir(exist_ok=True)
            save_checkpoint(peft_model, head, optimizer, epoch, best_val_sign, best_head_state, ckpt_path)

        if patience_counter >= 5:
            print(f"  early stopping at epoch {epoch}")
            break

    print(f"\nbest val sign agreement: {best_val_sign:.1%}")

    head.load_state_dict(best_head_state)
    return peft_model, head, best_val_sign, start_epoch


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune BGE + linear head")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sign-weight", type=float, default=2.0, help="Sign mismatch penalty weight")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA and save full model")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs (0 to disable)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    texts, y = load_data(conn)
    conn.close()

    print(f"dataset: {len(texts)} articles")

    indices = np.arange(len(texts))
    train_idx, val_idx = train_test_split(indices, test_size=args.val_size, random_state=args.seed)

    texts_train = [texts[i] for i in train_idx]
    texts_val = [texts[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"split:   {len(texts_train)} train, {len(texts_val)} val")

    MODELS_DIR.mkdir(exist_ok=True)

    peft_model, head, best_sign, start_epoch = train_lora(
        texts_train, y_train, texts_val, y_val,
        rank=args.rank, lr=args.lr, head_lr=args.head_lr,
        epochs=args.epochs, batch_size=args.batch_size,
        sign_weight=args.sign_weight, device=args.device, seed=args.seed,
        resume_path=args.resume, checkpoint_every=args.checkpoint_every,
    )

    if args.merge:
        print("\nmerging LoRA into base model...")
        merged_model = peft_model.merge_and_unload()
        save_path = MODELS_DIR / "bge_lora_merged"
        merged_model.save(str(save_path))
        print(f"merged model saved to {save_path}")

        print("\nretraining linear head on LoRA-finetuned embeddings...")
        from newsdim.train.trainer import train_analytical

        merged_enc = SentenceTransformer(str(save_path), device=args.device)
        X_train_new = encode_texts(merged_enc, texts_train, args.batch_size, args.device)
        X_val_new = encode_texts(merged_enc, texts_val, args.batch_size, args.device)

        result = train_analytical(X_train_new, y_train, ridge=1.0)

        val_pred_raw = result.head.predict_raw(X_val_new)
        val_pred_clamped = result.head.predict(X_val_new)
        val_metrics = _compute_metrics(y_val, val_pred_raw, val_pred_clamped)

        print("\n=== Retrained head on LoRA embeddings ===")
        print(f"  overall: sign={val_metrics['overall']['sign_agreement']:.1%}  "
              f"MAE={val_metrics['overall']['mae']:.3f}")
        for d in DIMS:
            m = val_metrics["per_dim"][d]
            print(f"    {d:>4}: sign={m['sign_agreement']:.1%}  MAE={m['mae']:.3f}")

        head_path = MODELS_DIR / "head_lora_ridge1.0.npz"
        result.head.save(head_path)
        print(f"\nhead weights saved to {head_path}")

    meta = {
        "method": "lora",
        "rank": args.rank,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "sign_weight": args.sign_weight,
        "epochs": args.epochs,
        "train_size": len(texts_train),
        "val_size": len(texts_val),
        "best_val_sign_agreement": float(best_sign),
        "seed": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = MODELS_DIR / "lora_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"meta saved to {meta_path}")


if __name__ == "__main__":
    main()
