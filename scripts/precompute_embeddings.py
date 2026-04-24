"""Pre-compute BGE embeddings for all scored articles and store in news_embedding table.

Usage:
  uv run python scripts/precompute_embeddings.py
  uv run python scripts/precompute_embeddings.py --device cpu
  uv run python scripts/precompute_embeddings.py --force
"""

import argparse
import sqlite3
import struct

import numpy as np
from tqdm import tqdm

from newsdim.embed.encoder import BGEEncoder

DB = "local_data/data.db"

EMBEDDING_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_embedding (
    source   TEXT NOT NULL,
    date     TEXT NOT NULL,
    seq_id   INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    model    TEXT NOT NULL,
    UNIQUE(source, date, seq_id)
);

CREATE INDEX IF NOT EXISTS idx_news_emb_source_date ON news_embedding(source, date);
CREATE INDEX IF NOT EXISTS idx_news_emb_date ON news_embedding(date);
"""

BATCH_SIZE = 64


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def fetch_unembedded(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute("""
        SELECT n.source, n.date, n.seq_id, n.content
        FROM news n
        INNER JOIN news_score s
            ON n.source = s.source AND n.date = s.date AND n.seq_id = s.seq_id
        LEFT JOIN news_embedding e
            ON n.source = e.source AND n.date = e.date AND n.seq_id = e.seq_id
        WHERE e.source IS NULL
        ORDER BY n.source, n.date, n.seq_id
    """).fetchall()


def fetch_all_scored(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute("""
        SELECT n.source, n.date, n.seq_id, n.content
        FROM news n
        INNER JOIN news_score s
            ON n.source = s.source AND n.date = s.date AND n.seq_id = s.seq_id
        ORDER BY n.source, n.date, n.seq_id
    """).fetchall()


def insert_embeddings(
    conn: sqlite3.Connection,
    keys: list[tuple],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO news_embedding (source, date, seq_id, embedding, model)
           VALUES (?, ?, ?, ?, ?)""",
        [(src, date, seq, _vec_to_blob(emb), model_name) for (src, date, seq, _txt), emb in zip(keys, embeddings)],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute BGE embeddings for scored articles")
    parser.add_argument("--device", default=None, help="Device: cpu, mps, cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Re-encode even if already embedded")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    conn.executescript(EMBEDDING_TABLE_SCHEMA)

    encoder = BGEEncoder(device=args.device)
    print(f"model:  {encoder.model_name}")
    print(f"device: {encoder.model.device}")
    print(f"dim:    {encoder.dim}")

    fetch_fn = fetch_all_scored if args.force else fetch_unembedded
    rows = fetch_fn(conn)
    print(f"articles to encode: {len(rows)}")

    if not rows:
        print("nothing to do")
        conn.close()
        return

    existing = conn.execute("SELECT COUNT(*) FROM news_embedding").fetchone()[0]

    for i in tqdm(range(0, len(rows), args.batch_size), desc="encoding", unit="batch"):
        batch = rows[i : i + args.batch_size]
        texts = [r[3] for r in batch]
        embeddings = encoder.encode(texts, batch_size=len(texts))
        insert_embeddings(conn, batch, embeddings, encoder.model_name)

    total = conn.execute("SELECT COUNT(*) FROM news_embedding").fetchone()[0]
    conn.close()
    print(f"done. {total} embeddings in DB ({total - existing} new)")


if __name__ == "__main__":
    main()
