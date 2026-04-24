"""Data quality report for news_score + news_embedding tables.

Usage:
  uv run python scripts/data_quality.py
"""

import sqlite3
from pathlib import Path

import numpy as np

from newsdim.dims import DIMS

DB = Path("local_data/data.db")


def report(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    dims = list(DIMS)
    cols = ", ".join(dims)

    cur.execute("SELECT COUNT(*) FROM news_score")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM news_embedding")
    emb_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM news")
    news_total = cur.fetchone()[0]

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"news:       {news_total:>8,}")
    print(f"news_score: {total:>8,}")
    print(f"news_emb:   {emb_total:>8,}")

    if total == 0:
        print("\nNo scores to analyze.")
        return

    cur.execute(f"SELECT {cols} FROM news_score")
    rows = np.array(cur.fetchall(), dtype=np.float32)

    # --- per-dim stats ---
    print(f"\n{'Dim':>6} | {'mean':>6} | {'std':>6} | {'min':>4} | {'max':>4} | {'nz%':>7} | {'zeros':>6}")
    print("-" * 60)
    for i, d in enumerate(dims):
        col = rows[:, i]
        nz = np.count_nonzero(col)
        print(
            f"{d:>6} | {col.mean():6.2f} | {col.std():6.2f} | {int(col.min()):4d} | {int(col.max()):4d} "
            f"| {100 * nz / len(col):6.1f}% | {len(col) - nz:>6d}"
        )

    # --- distribution ---
    print("\nScore distribution:")
    for d in dims:
        cur.execute(f"SELECT {d}, COUNT(*) FROM news_score GROUP BY {d} ORDER BY {d}")
        dist = cur.fetchall()
        print(f"  {d:>6}: {' '.join(f'{v}:{c}' for v, c in dist)}")

    # --- correlation matrix ---
    corr = np.corrcoef(rows.T)
    print("\nCorrelation matrix:")
    header = " ".join(f"{d:>6}" for d in dims)
    print(f"{'':>6} {header}")
    for i, d in enumerate(dims):
        vals = " ".join(f"{corr[i, j]:6.2f}" for j in range(len(dims)))
        print(f"{d:>6} {vals}")

    # --- join integrity ---
    print("\nJoin integrity:")
    for other in ["news", "news_embedding"]:
        cur.execute(
            f"SELECT COUNT(*) FROM news_score ns "
            f"LEFT JOIN {other} o ON ns.source=o.source AND ns.date=o.date AND ns.seq_id=o.seq_id "
            f"WHERE o.source IS NULL"
        )
        orphans = cur.fetchone()[0]
        status = "OK" if orphans == 0 else f"MISSING {orphans}"
        print(f"  score → {other:>16}: {status}")

    # --- all-zero rows ---
    where = " AND ".join(f"{d}=0" for d in dims)
    cur.execute(f"SELECT COUNT(*) FROM news_score WHERE {where}")
    all_zero = cur.fetchone()[0]
    print(f"  all-zero rows:          {all_zero} ({100 * all_zero / total:.1f}%)")

    # --- source / prompt breakdown ---
    print("\nSource breakdown:")
    cur.execute("SELECT source, COUNT(*) FROM news_score GROUP BY source ORDER BY COUNT(*) DESC")
    for src, cnt in cur.fetchall():
        print(f"  {src}: {cnt:,}")

    cur.execute("SELECT prompt_version, COUNT(*) FROM news_score GROUP BY prompt_version")
    versions = cur.fetchall()
    if len(versions) > 1:
        print("\nWARNING: multiple prompt versions:")
        for pv, cnt in versions:
            print(f"  {pv}: {cnt:,}")
    elif versions:
        print(f"\nPrompt version: {versions[0][0]}")

    print("=" * 60)


def main():
    conn = sqlite3.connect(str(DB))
    try:
        report(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
