from __future__ import annotations

import argparse
from pathlib import Path

from backend.feedback import FeedbackStore, export_approved_feedback


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export human-approved span corrections for future offline training."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "feedback" / "feedback.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "approved_feedback_dataset.jsonl",
    )
    args = parser.parse_args()

    # Importing the corpus lookup does not invoke the QA Reader or change corpus data.
    from backend.viqa_api import _lookup_socratic_passage

    count = export_approved_feedback(
        FeedbackStore(args.db),
        _lookup_socratic_passage,
        args.output,
    )
    print(f"Exported {count} approved correction(s) to {args.output}")


if __name__ == "__main__":
    main()
