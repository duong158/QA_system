from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "learning_rate",
    "epochs",
    "max_length",
    "stride",
    "overall_em",
    "overall_f1",
    "answerable_em",
    "answerable_f1",
    "unanswerable_accuracy",
    "best_threshold",
    "answerable_predicted_empty_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed Reader baseline artifacts")
    parser.add_argument("--results-root", type=Path, default=Path("results/reader"))
    parser.add_argument("--output", type=Path, default=Path("results/reader_baseline_comparison.csv"))
    args = parser.parse_args()
    rows = []
    for run_dir in sorted(path for path in args.results_root.glob("*") if path.is_dir()):
        training_path = run_dir / "training_args.json"
        metrics_path = run_dir / "validation_metrics.json"
        threshold_path = run_dir / "best_threshold.json"
        if not (training_path.is_file() and metrics_path.is_file() and threshold_path.is_file()):
            continue
        training = json.loads(training_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run_dir.name,
                "learning_rate": training.get("lr"),
                "epochs": training.get("epochs"),
                "max_length": training.get("max_seq_len"),
                "stride": training.get("doc_stride"),
                "overall_em": metrics["overall"]["em"],
                "overall_f1": metrics["overall"]["f1"],
                "answerable_em": metrics["answerable"]["em"],
                "answerable_f1": metrics["answerable"]["f1"],
                "unanswerable_accuracy": metrics["unanswerable"]["accuracy"],
                "best_threshold": threshold["threshold"],
                "answerable_predicted_empty_rate": metrics["answerable"]["predicted_empty_rate"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} completed runs to {args.output}")


if __name__ == "__main__":
    main()
