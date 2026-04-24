"""Ingest local_data CSVs into local_data/data.db (SQLite).

Tables:
  - news: sina, 10jqk, and future news sources
  - ann:  company announcements

Idempotent: re-running replaces data for each (source, date) / date.
"""

import argparse
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

SHANGHAI = timezone(timedelta(hours=8))
UTC = timezone.utc

LOCAL_DATA = Path(__file__).resolve().parent.parent / "local_data"
DB_PATH = LOCAL_DATA / "data.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    source   TEXT NOT NULL,
    date     TEXT NOT NULL,
    seq_id   INTEGER NOT NULL,
    ts       INTEGER NOT NULL,
    content  TEXT NOT NULL,
    UNIQUE(source, date, seq_id)
);

CREATE TABLE IF NOT EXISTS ann (
    date     TEXT NOT NULL,
    seq_id   INTEGER NOT NULL,
    code     TEXT NOT NULL,
    name     TEXT NOT NULL,
    type     TEXT NOT NULL,
    content  TEXT NOT NULL,
    UNIQUE(date, seq_id)
);

CREATE INDEX IF NOT EXISTS idx_news_source_date ON news(source, date);
CREATE INDEX IF NOT EXISTS idx_news_date        ON news(date);
CREATE INDEX IF NOT EXISTS idx_ann_code         ON ann(code);
CREATE INDEX IF NOT EXISTS idx_ann_type         ON ann(type);
CREATE INDEX IF NOT EXISTS idx_ann_date         ON ann(date);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def discover_csvs(source: str) -> list[Path]:
    base = LOCAL_DATA / source
    if not base.is_dir():
        return []
    csvs = []
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.glob("*.csv")):
            csvs.append(f)
    return csvs


def parse_date_from_path(source: str, path: Path) -> str | None:
    name = path.stem
    name = re.sub(r"\(\d+\)$", "", name)
    if source == "ann":
        try:
            datetime.strptime(name, "%Y%m%d")
            return f"{name[:4]}-{name[4:6]}-{name[6:8]}"
        except ValueError:
            return None
    else:
        try:
            datetime.strptime(name, "%Y-%m-%d")
            return name
        except ValueError:
            return None


def ingest_news(conn: sqlite3.Connection, source: str, csvs: list[Path]) -> int:
    total_rows = 0
    for path in tqdm(csvs, desc=source, unit="file"):
        date_str = parse_date_from_path(source, path)
        if date_str is None:
            continue

        try:
            df = pl.read_csv(path, encoding="utf-8", infer_schema_length=0)
        except Exception:
            continue

        if "datetime" not in df.columns or "content" not in df.columns:
            continue

        df = (
            df.select(
                pl.col("datetime").alias("ts_raw"),
                pl.col("content"),
            )
            .with_columns(
                pl.col("ts_raw")
                .str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="Asia/Shanghai")
                .dt.convert_time_zone("UTC")
                .dt.timestamp("ms")
                .alias("ts")
            )
            .sort("ts")
            .unique(subset=["content"], keep="last")
            .sort("ts")
            .with_row_index("seq_id")
            .select("seq_id", "ts", "content")
        )

        conn.execute(
            "DELETE FROM news WHERE source = ? AND date = ?", (source, date_str))

        rows = df.iter_rows()
        conn.executemany(
            "INSERT INTO news (source, date, seq_id, ts, content) VALUES (?, ?, ?, ?, ?)",
            [(source, date_str, int(seq), int(ts), content)
             for seq, ts, content in rows],
        )
        conn.commit()
        total_rows += df.height

    return total_rows


def ingest_ann(conn: sqlite3.Connection, csvs: list[Path]) -> int:
    total_rows = 0
    for path in tqdm(csvs, desc="ann", unit="file"):
        date_str = parse_date_from_path("ann", path)
        if date_str is None:
            continue

        try:
            df = pl.read_csv(path, encoding="utf-8", infer_schema_length=0)
        except Exception:
            continue

        expected = {"序号", "代码", "简称", "事件类型", "具体事项", "交易日"}
        if not expected.issubset(set(df.columns)):
            continue

        df = (
            df.rename(
                {
                    "序号": "seq_id",
                    "代码": "code",
                    "简称": "name",
                    "事件类型": "type",
                    "具体事项": "content",
                    "交易日": "trade_date",
                }
            )
            .with_columns(
                pl.col("code").str.replace("code.", ""),
                pl.col("trade_date").str.to_datetime(
                    "%Y-%m-%d").dt.strftime("%Y-%m-%d").alias("date"),
            )
            .select("date", "seq_id", "code", "name", "type", "content")
        )

        conn.execute("DELETE FROM ann WHERE date = ?", (date_str,))

        rows = df.iter_rows()
        conn.executemany(
            "INSERT INTO ann (date, seq_id, code, name, type, content) VALUES (?, ?, ?, ?, ?, ?)",
            [(d, int(seq), code, name, typ, content)
             for d, seq, code, name, typ, content in rows],
        )
        conn.commit()
        total_rows += df.height

    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Ingest local_data CSVs into SQLite")
    parser.add_argument(
        "--source", help="Only ingest specific source (sina, 10jqk, ann)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    news_sources = ["sina", "10jqk"]
    if args.source:
        if args.source == "ann":
            news_sources = []
        else:
            news_sources = [args.source]

    total = 0
    for source in news_sources:
        csvs = discover_csvs(source)
        if csvs:
            n = ingest_news(conn, source, csvs)
            print(f"  {source}: {n:,} rows")
            total += n

    if not args.source or args.source == "ann":
        csvs = discover_csvs("ann")
        if csvs:
            n = ingest_ann(conn, csvs)
            print(f"  ann: {n:,} rows")
            total += n

    conn.close()
    print(f"\ndone. {total:,} total rows ingested into {DB_PATH}")


if __name__ == "__main__":
    main()
