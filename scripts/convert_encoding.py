"""Convert all GB2312/GBK CSVs in local_data to UTF-8 (in-place)."""

from pathlib import Path

LOCAL_DATA = Path(__file__).resolve().parent.parent / "local_data"


def convert_file(path: Path) -> None:
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
        return
    except UnicodeDecodeError:
        pass

    text = raw.decode("gbk")
    path.write_text(text, encoding="utf-8")
    print(f"  converted: {path}")


def main() -> None:
    sources = sorted(p for p in LOCAL_DATA.iterdir() if p.is_dir() and not p.name.startswith("."))
    converted = 0

    for src in sources:
        for year_dir in sorted(src.iterdir()):
            if not year_dir.is_dir():
                continue
            csvs = sorted(year_dir.glob("*.csv"))
            if not csvs:
                continue
            print(f"{src.name}/{year_dir.name}/ ({len(csvs)} files)")
            for f in csvs:
                size_before = f.stat().st_size
                convert_file(f)
                if f.stat().st_size != size_before:
                    converted += 1

    print(f"\ndone. {converted} files converted.")


if __name__ == "__main__":
    main()
