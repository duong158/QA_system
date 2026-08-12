from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.config import DEFAULT_DOC_STRIDE, DEFAULT_MAX_LENGTH, validate_window_config  # noqa: E402
from reader.data_utils import _map_char_span_to_feature, build_text_preprocessor, get_tokenizer  # noqa: E402
from reader.qa_tokenizer import encode_qa_batch  # noqa: E402
from reader.text_preprocessing import normalize_span_text, restore_text_offsets, uses_compact_offsets  # noqa: E402


def _read_rows(path: Path, subset_size: int) -> list[dict[str, Any]]:
    try:
        import polars as pl

        frame = pl.read_parquet(path).select(
            ["id", "question", "context", "answer_text", "answer_start"]
        )
        if subset_size > 0:
            frame = frame.head(subset_size)
        return frame.to_dicts()
    except ImportError:  # pragma: no cover - polars is present in the audit environment
        import pandas as pd

        frame = pd.read_parquet(path)
        if subset_size > 0:
            frame = frame.head(subset_size)
        return frame[["id", "question", "context", "answer_text", "answer_start"]].to_dict("records")


def _legacy_first_match(context: str, answer: str) -> tuple[int, int]:
    context_characters = [(char, index) for index, char in enumerate(context) if char not in (" ", "_")]
    answer_characters = [char for char in answer if char not in (" ", "_")]
    compact_context = "".join(char for char, _ in context_characters)
    compact_answer = "".join(answer_characters)
    index = compact_context.find(compact_answer)
    if index < 0 or not compact_answer:
        return -1, -1
    return context_characters[index][1], context_characters[index + len(compact_answer) - 1][1] + 1


def _write_errors(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "id", "reason", "question", "answer_text", "answer_start"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def validate_split(
    split: str,
    rows: list[dict[str, Any]],
    tokenizer,
    preprocessor,
    max_length: int,
    stride: int,
    batch_size: int,
    model_vocab_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answerable_rows = [
        row
        for row in rows
        if int(row.get("answer_start", -1) if row.get("answer_start") is not None else -1) >= 0
        and bool(str(row.get("answer_text") or ""))
    ]
    valid_examples: set[str] = set()
    invalid: list[dict[str, Any]] = []
    reasons: defaultdict[str, int] = defaultdict(int)
    legacy_exact_offset = 0
    legacy_mapped = 0
    feature_count = 0
    answer_features = 0
    tokenizer_boundary_ids: set[str] = set()
    answer_exceeds_window_ids: set[str] = set()
    answer_longer_than_stride_ids: set[str] = set()
    max_input_id = -1
    out_of_range_features = 0
    compact_offsets = uses_compact_offsets(tokenizer)

    for batch_start in range(0, len(answerable_rows), batch_size):
        batch = answerable_rows[batch_start : batch_start + batch_size]
        prepared = []
        active_rows = []
        for row in batch:
            answer_start = int(row["answer_start"])
            answer_text = str(row["answer_text"])
            legacy_start, legacy_end = _legacy_first_match(str(row["context"]), answer_text)
            if legacy_start >= 0:
                legacy_mapped += 1
                if legacy_start == answer_start and legacy_end == answer_start + len(answer_text):
                    legacy_exact_offset += 1
            try:
                prepared.append(
                    preprocessor.prepare(
                        str(row["question"]),
                        str(row["context"]),
                        answer_text,
                        answer_start,
                    )
                )
                active_rows.append(row)
            except ValueError as error:
                reason = str(error).split(":", 1)[0]
                reasons[reason] += 1
                invalid.append({**row, "split": split, "reason": str(error)})

        if not prepared:
            continue
        for item, row in zip(prepared, active_rows):
            question_tokens = len(tokenizer.tokenize(item.model_question))
            available_context = max_length - question_tokens - 4
            answer_tokens = len(
                tokenizer.tokenize(item.model_context[item.model_answer_start : item.model_answer_end])
            )
            if answer_tokens > available_context:
                answer_exceeds_window_ids.add(str(row["id"]))
            elif answer_tokens > stride:
                # An answer may fit in a context window but still be skipped when it
                # straddles two windows and is longer than their overlap.
                answer_longer_than_stride_ids.add(str(row["id"]))
        encoded = encode_qa_batch(
            tokenizer,
            [item.model_question for item in prepared],
            [item.model_context for item in prepared],
            max_length=max_length,
            stride=stride,
            padding="max_length",
        )
        sample_mapping = encoded["overflow_to_sample_mapping"]
        for feature_index, tokenizer_offsets in enumerate(encoded["offset_mapping"]):
            feature_count += 1
            feature_max_id = max(encoded["input_ids"][feature_index])
            max_input_id = max(max_input_id, feature_max_id)
            if feature_max_id >= model_vocab_size:
                out_of_range_features += 1
            sample_index = int(sample_mapping[feature_index])
            item = prepared[sample_index]
            offsets = restore_text_offsets(item.model_context, tokenizer_offsets, compact_offsets)
            row = active_rows[sample_index]
            input_ids = encoded["input_ids"][feature_index]
            try:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            except ValueError:
                cls_index = 0
            sequence_ids = encoded.sequence_ids(feature_index)
            start_token, end_token = _map_char_span_to_feature(
                offsets,
                sequence_ids,
                item.model_answer_start,
                item.model_answer_end,
                cls_index,
            )
            if start_token == cls_index and end_token == cls_index:
                continue
            answer_features += 1
            decoded = item.model_context[offsets[start_token][0] : offsets[end_token][1]]
            if normalize_span_text(decoded) == normalize_span_text(item.answer_text):
                valid_examples.add(str(row["id"]))
            else:
                tokenizer_boundary_ids.add(str(row["id"]))

        print(
            f"{split}: checked {min(batch_start + batch_size, len(answerable_rows))}/"
            f"{len(answerable_rows)} answerable examples",
            flush=True,
        )

    invalid_ids = {str(row["id"]) for row in invalid}
    for row in answerable_rows:
        example_id = str(row["id"])
        if example_id not in valid_examples and example_id not in invalid_ids:
            if example_id in answer_exceeds_window_ids:
                reason_code = "ANSWER_EXCEEDS_WINDOW"
            elif example_id in answer_longer_than_stride_ids:
                reason_code = "ANSWER_LONGER_THAN_STRIDE"
            elif example_id in tokenizer_boundary_ids:
                reason_code = "TOKENIZER_BOUNDARY"
            else:
                reason_code = "CHUNK_OR_TOKENIZER_MAPPING"
            reasons[reason_code] += 1
            invalid.append(
                {
                    **row,
                    "split": split,
                    "reason": f"{reason_code}: no overflow feature decoded exactly to the annotated answer",
                }
            )

    total = len(answerable_rows)
    valid = len(valid_examples)
    report = {
        "split": split,
        "total_rows": len(rows),
        "total_answerable_samples": total,
        "valid_span_mappings": valid,
        "invalid_span_mappings": total - valid,
        "span_mapping_accuracy": 100.0 * valid / total if total else 0.0,
        "overflow_features": feature_count,
        "features_containing_answer": answer_features,
        "maximum_input_token_id": max_input_id,
        "model_vocab_size": model_vocab_size,
        "features_with_out_of_range_token_id": out_of_range_features,
        "legacy_first_match_mapped": legacy_mapped,
        "legacy_exact_annotated_offset": legacy_exact_offset,
        "legacy_exact_annotated_offset_accuracy": 100.0 * legacy_exact_offset / total if total else 0.0,
        "invalid_reason_counts": dict(sorted(reasons.items())),
    }
    return report, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every gold answer_start through PhoBERT overflow features")
    parser.add_argument("--model", default="models/reader/vinai_phobert-base-v2")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--subset_size", type=int, default=-1)
    parser.add_argument("--output", default="results/span_integrity_report.json")
    parser.add_argument("--errors", default="results/span_integrity_errors.csv")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    config = AutoConfig.from_pretrained(args.model)
    validate_window_config(args.max_length, args.stride, config, tokenizer)
    preprocessor = build_text_preprocessor(args.model, tokenizer=tokenizer, model_config=config)

    reports = []
    all_errors: list[dict[str, Any]] = []
    names = {"train": "viquad_train_clean.parquet", "validation": "viquad_val_clean.parquet"}
    for split in args.splits:
        rows = _read_rows(ROOT / "data" / "processed" / names[split], args.subset_size)
        report, errors = validate_split(
            split,
            rows,
            tokenizer,
            preprocessor,
            args.max_length,
            args.stride,
            args.batch_size,
            int(config.vocab_size),
        )
        reports.append(report)
        all_errors.extend(errors)

    total = sum(report["total_answerable_samples"] for report in reports)
    valid = sum(report["valid_span_mappings"] for report in reports)
    payload = {
        "model": args.model,
        "max_length": args.max_length,
        "stride": args.stride,
        "preprocessing": "raw text -> PyVi once -> model-compatible PhoBERT BPE with explicit offsets",
        "gold_mapping": "answer_start + len(answer_text) -> offset_mapping context tokens",
        "splits": reports,
        "total_answerable_samples": total,
        "valid_span_mappings": valid,
        "invalid_span_mappings": total - valid,
        "span_mapping_accuracy": 100.0 * valid / total if total else 0.0,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_errors(ROOT / args.errors, all_errors)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Invalid examples: {len(all_errors)} -> {ROOT / args.errors}")
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
