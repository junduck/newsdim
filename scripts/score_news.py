"""Score news articles using LLM and store results in news_score table.

Usage:
  uv run python scripts/score_news.py --limit 300
  uv run python scripts/score_news.py --limit 1000 --source sina
  uv run python scripts/score_news.py --force --limit 50
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

from tqdm import tqdm

from newsdim.news_scorer import DimScores, NewsScorer

DB = "local_data/data.db"

SCORE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_score (
    source         TEXT NOT NULL,
    date           TEXT NOT NULL,
    seq_id         INTEGER NOT NULL,
    mom            INTEGER NOT NULL,
    stab           INTEGER NOT NULL,
    horz           INTEGER NOT NULL,
    eng            INTEGER NOT NULL,
    hype           INTEGER NOT NULL,
    sent           INTEGER NOT NULL,
    sec            INTEGER NOT NULL,
    pol            INTEGER NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    updated_at     INTEGER NOT NULL,
    UNIQUE(source, date, seq_id)
);

CREATE INDEX IF NOT EXISTS idx_news_score_date ON news_score(date);
CREATE INDEX IF NOT EXISTS idx_news_score_source_date ON news_score(source, date);
"""


def init_score_table(conn: sqlite3.Connection) -> None:
    conn.executescript(SCORE_TABLE_SCHEMA)
    conn.commit()


def sample_unscored(
    conn: sqlite3.Connection,
    limit: int,
    source: str | None = None,
) -> list[tuple]:
    where_parts = ["n.content IS NOT NULL"]
    params: list = []
    if source:
        where_parts.append("n.source = ?")
        params.append(source)

    where = " AND ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT n.source, n.date, n.seq_id, n.content
        FROM news n
        LEFT JOIN news_score s
            ON n.source = s.source AND n.date = s.date AND n.seq_id = s.seq_id
        WHERE s.source IS NULL AND {where}
        ORDER BY RANDOM()
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    return rows


def sample_force(
    conn: sqlite3.Connection,
    limit: int,
    source: str | None = None,
) -> list[tuple]:
    where_parts = ["n.content IS NOT NULL"]
    params: list = []
    if source:
        where_parts.append("n.source = ?")
        params.append(source)

    where = " AND ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT n.source, n.date, n.seq_id, n.content
        FROM news n
        WHERE {where}
        ORDER BY RANDOM()
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    return rows


def insert_scores(
    conn: sqlite3.Connection,
    articles: list[tuple],
    scores: list[DimScores],
    model: str,
    prompt_version: str,
) -> None:
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    conn.executemany(
        """INSERT OR REPLACE INTO news_score
           (source, date, seq_id, mom, stab, horz, eng, hype, sent, sec, pol, model, prompt_version, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                src,
                date,
                seq,
                s.mom,
                s.stab,
                s.horz,
                s.eng,
                s.hype,
                s.sent,
                s.sec,
                s.pol,
                model,
                prompt_version,
                now,
            )
            for (src, date, seq, _text), s in zip(articles, scores)
        ],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Score news articles via LLM")
    parser.add_argument("--limit", type=int, default=300, help="Number of articles to score")
    parser.add_argument("--source", default=None, help="Only score this source (sina, 10jqk)")
    parser.add_argument("--force", action="store_true", help="Re-score even if already scored")
    parser.add_argument("--batch-size", type=int, default=50, help="Articles per API batch")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel API calls per batch")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    init_score_table(conn)

    sampler = sample_force if args.force else sample_unscored

    scorer = NewsScorer()
    print(f"model:   {scorer.config.model}")
    print(f"prompt:  {scorer.prompt_version}")
    print(f"limit:   {args.limit}")
    print(f"source:  {args.source or 'all'}")
    print(f"force:   {args.force}")
    print()

    total_scored = 0
    total_errors = 0
    remaining = args.limit
    t0 = time.time()

    pbar = tqdm(total=args.limit, desc="scoring", unit="article")

    while remaining > 0:
        batch_articles = sampler(conn, remaining, args.source)
        if not batch_articles:
            print("no more un-scored articles available")
            break

        texts = [a[3] for a in batch_articles]
        batch_scores: list[DimScores] = []
        batch_errors = 0

        for i in tqdm(range(0, len(texts), args.batch_size), desc="batch", leave=False):
            chunk = texts[i : i + args.batch_size]
            try:
                chunk_scores = scorer.score_batch(chunk, concurrency=args.concurrency)
                batch_scores.extend(chunk_scores)
            except Exception as e:
                print(f"\n  batch error: {e}")
                batch_errors += len(chunk)
                for _ in chunk:
                    batch_scores.append(DimScores())

        scored_in_batch = len(batch_articles) - batch_errors
        insert_scores(conn, batch_articles, batch_scores, scorer.config.model, scorer.prompt_version)

        total_scored += scored_in_batch
        total_errors += batch_errors
        remaining -= len(batch_articles)
        pbar.update(len(batch_articles))

    pbar.close()
    elapsed = time.time() - t0

    existing = conn.execute("SELECT COUNT(*) FROM news_score").fetchone()[0]
    conn.close()

    print(f"\ndone. {total_scored} scored, {total_errors} errors in {elapsed:.0f}s")
    print(f"total rows in news_score: {existing}")


if __name__ == "__main__":
    main()
