from pathlib import Path
import sqlite3
import polars as pl


root = Path(__file__).resolve().parents[1]
db_path = root / "data" / "processed" / "docs.db"
connection = sqlite3.connect(db_path)
print("tables", connection.execute("select name from sqlite_master where type='table'").fetchall())
print("documents", connection.execute("select count(*) from documents").fetchone()[0])
print("schema", connection.execute("pragma table_info(documents)").fetchall())
print("sample", connection.execute("select * from documents limit 3").fetchall())
connection.close()
print("results files")
for item in sorted((root / "results").rglob("*")):
    if item.is_file():
        print(item.relative_to(root).as_posix())

print("dataset statistics")
for split in ("train", "val", "test"):
    frame = pl.read_parquet(root / "data" / "processed" / f"viquad_{split}_clean.parquet")
    print(split, "rows", len(frame), "columns", list(frame.columns))
    for column in ("is_impossible", "answers"):
        if column in frame.columns:
            if column == "is_impossible":
                impossible = int(frame[column].fill_null(False).cast(pl.Boolean).sum())
                print("  impossible", impossible)
                print("  answerable", len(frame) - impossible)
            else:
                print("  answer sample", frame[column][0])
    if "answer_text" in frame.columns:
        answerable = int(frame.select(pl.col("answer_text").fill_null("").str.strip_chars().ne("").sum()).item())
        print("  answerable", answerable)
        print("  unanswerable", len(frame) - answerable)

corpus = pl.read_parquet(root / "data" / "processed" / "corpus_clean.parquet")
print("corpus", len(corpus), "columns", list(corpus.columns))
