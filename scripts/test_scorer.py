"""Test run: score 10 random news articles from the DB."""

import json
import random
import sqlite3
import time

from newsdim.news_scorer import NewsScorer

DB = "local_data/data.db"


def load_random_news(n: int = 10) -> list[dict]:
    random.seed(42)
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT source, date, seq_id, content FROM news ORDER BY RANDOM() LIMIT ?",
        (n * 5,),
    ).fetchall()
    conn.close()
    picked = random.sample(rows, min(n, len(rows)))
    return [{"source": r[0], "date": r[1], "seq_id": r[2], "text": r[3]} for r in picked]


def main():
    scorer = NewsScorer()
    print(f"model:  {scorer.config.model}")
    print(f"prompt: {scorer.prompt_version}")
    print(f"thinking: {scorer.config.thinking}")
    print()

    samples = load_random_news(10)
    t0 = time.time()
    for i, s in enumerate(samples):
        label = f"[{s['source']} {s['date']} #{s['seq_id']}]"
        print(f"{i + 1:>2}. {label} {s['text'][:80]}...")
        try:
            scores = scorer.score(s["text"])
            print(f"    → {json.dumps(scores.to_dict(), ensure_ascii=False)}")
        except Exception as e:
            print(f"    ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n{len(samples)} articles in {elapsed:.1f}s ({elapsed / len(samples):.2f}s/article)")


if __name__ == "__main__":
    main()
