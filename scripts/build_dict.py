"""Build jieba financial dictionary from tushare + glossary.

Requires TUSHARE_TOKEN in environment (or .env.local).

Usage:
  uv run python scripts/build_dict.py
  uv run python scripts/build_dict.py --glossary local_data/glossary_clean.csv
  uv run python scripts/build_dict.py --output src/newsdim/retrieval/dict/finance.txt
"""

import argparse
import csv
import os
from pathlib import Path

DEFAULT_GLOSSARY = "local_data/glossary_clean.csv"
DEFAULT_SUPPLEMENT = "src/newsdim/retrieval/dict/finance_supplement.txt"
DEFAULT_OUTPUT = "src/newsdim/retrieval/dict/finance.txt"


def load_env():
    env_path = Path(".env.local")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def fetch_tushare_names() -> set[str]:
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("TUSHARE_TOKEN not set, skipping tushare")
        return set()

    ts.set_token(token)
    pro = ts.pro_api()
    names: set[str] = set()

    # A-share stocks
    print("Fetching stock_basic...")
    df = pro.stock_basic(list_status="L", fields="name,fullname,industry")
    names.update(df["name"].dropna().tolist())
    names.update(df["fullname"].dropna().tolist())
    names.update(df["industry"].dropna().unique().tolist())
    print(f"  stocks: {len(names)} names")

    # Funds (exchange-traded)
    print("Fetching fund_basic...")
    for market in ("E", "O"):
        try:
            df = pro.fund_basic(market=market, fields="name,management")
            names.update(df["name"].dropna().tolist())
            names.update(df["management"].dropna().tolist())
        except Exception as e:
            print(f"  fund market={market} skipped: {e}")
    print(f"  after funds: {len(names)} names")

    # Indices (CSI, SW industry classification is gold)
    print("Fetching index_basic...")
    for market in ("SSE", "SZSE", "CSI", "SW"):
        try:
            df = pro.index_basic(market=market, fields="name,fullname")
            names.update(df["name"].dropna().tolist())
            names.update(df["fullname"].dropna().tolist())
        except Exception as e:
            print(f"  index market={market} skipped: {e}")
    print(f"  after indices: {len(names)} names")

    return names


def load_glossary(path: str) -> set[str]:
    terms: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 1:
                continue
            term = row[0].strip()
            if len(term) >= 2:
                terms.add(term)
    return terms


def write_jieba_dict(terms: set[str], output: str) -> None:
    sorted_terms = sorted(terms)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for term in sorted_terms:
            f.write(f"{term}\n")
    print(f"Wrote {len(sorted_terms)} terms to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build jieba financial dictionary")
    parser.add_argument("--glossary", default=DEFAULT_GLOSSARY, help="Glossary CSV path")
    parser.add_argument("--supplement", default=DEFAULT_SUPPLEMENT, help="Manual supplement dict path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output dict path")
    parser.add_argument("--no-tushare", action="store_true", help="Skip tushare fetch")
    args = parser.parse_args()

    load_env()

    all_terms: set[str] = set()

    if not args.no_tushare:
        all_terms.update(fetch_tushare_names())

    glossary_path = Path(args.glossary)
    if glossary_path.exists():
        glossary_terms = load_glossary(str(glossary_path))
        print(f"Glossary: {len(glossary_terms)} terms from {glossary_path}")
        all_terms.update(glossary_terms)
    else:
        print(f"Glossary not found: {glossary_path}")

    supplement_path = Path(args.supplement)
    if supplement_path.exists():
        supp_terms = set()
        for line in supplement_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                supp_terms.add(line)
        print(f"Supplement: {len(supp_terms)} terms from {supplement_path}")
        all_terms.update(supp_terms)

    print(f"Total unique terms: {len(all_terms)}")
    write_jieba_dict(all_terms, args.output)


if __name__ == "__main__":
    main()
