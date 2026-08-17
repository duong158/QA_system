from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.config import DEFAULT_DOC_STRIDE, DEFAULT_MAX_ANSWER_LENGTH, DEFAULT_MAX_LENGTH  # noqa: E402
from reader.metrics import (  # noqa: E402
    compute_exact,
    compute_f1,
    evaluate_predictions,
    exact_match,
    f1_score,
    normalize_answer,
)
from reader.postprocessing import has_clean_word_boundaries, select_span_candidates  # noqa: E402


class QaRecordDataset:
    """Dependency-light validation view used for inference-only evaluation."""

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            selected = self.rows[index]
            return {
                key: [row.get(key) for row in selected]
                for key in ("id", "question", "context", "answer_text", "answer_start")
            }
        return self.rows[index]


def load_validation_records(path: Path, subset_size: int = -1) -> QaRecordDataset:
    try:
        import polars as pl

        frame = pl.read_parquet(path).select(
            ["id", "question", "context", "answer_text", "answer_start"]
        )
        if subset_size > 0:
            frame = frame.head(subset_size)
        return QaRecordDataset(frame.to_dicts())
    except ImportError:  # pragma: no cover
        import pandas as pd

        frame = pd.read_parquet(path)
        if subset_size > 0:
            frame = frame.head(subset_size)
        return QaRecordDataset(frame.to_dict("records"))


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _metric_row(threshold: float, metrics: dict[str, Any]) -> dict[str, float]:
    answerable_f1 = float(metrics["answerable"]["f1"])
    no_answer_accuracy = float(metrics["unanswerable"]["accuracy"])
    return {
        "threshold": float(threshold),
        "overall_em": float(metrics["overall"]["em"]),
        "overall_f1": float(metrics["overall"]["f1"]),
        "answerable_em": float(metrics["answerable"]["em"]),
        "answerable_f1": answerable_f1,
        "unanswerable_accuracy": no_answer_accuracy,
        "predicted_no_answer_rate": float(metrics["predicted_no_answer"]["rate"]),
        "answerable_predicted_empty_rate": float(metrics["answerable"]["predicted_empty_rate"]),
        "false_positive_rate": 100.0 - no_answer_accuracy,
        "false_negative_rate": float(metrics["answerable"]["predicted_empty_rate"]),
        # Retained for analysis only.
        "balanced_score": 0.5 * answerable_f1 + 0.5 * no_answer_accuracy,
        # Answerable F1 is the primary Reader objective. The no-answer term
        # prevents the calibrated threshold from collapsing that class entirely.
        "reader_priority_score": 0.7 * answerable_f1 + 0.3 * no_answer_accuracy,
    }


def apply_score_margin_threshold(
    raw_predictions: Sequence[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_predictions:
        margin = float(raw.get("score_margin", float("-inf")))
        prediction = str(raw.get("best_span_answer") or "") if margin >= threshold else ""
        rows.append({**raw, "predicted_answer": prediction, "prediction": prediction})
    return rows


def default_thresholds(raw_predictions: Sequence[dict[str, Any]]) -> list[float]:
    finite_margins = sorted(
        float(row["score_margin"])
        for row in raw_predictions
        if math.isfinite(float(row.get("score_margin", float("nan"))))
    )
    coarse = [value / 2 for value in range(-20, 21)]  # -10 .. 10, step 0.5
    focused = [value / 10 for value in range(-20, 21)]  # -2 .. 2, step 0.1
    quantiles: list[float] = []
    if finite_margins:
        quantiles = [
            float(np.quantile(finite_margins, quantile))
            for quantile in np.linspace(0.02, 0.98, 49)
        ]
    return sorted({round(value, 6) for value in [*coarse, *focused, *quantiles]})


def sweep_thresholds(
    raw_predictions: Sequence[dict[str, Any]],
    thresholds: Sequence[float] | None = None,
    selection_metric: str = "reader_priority_score",
) -> tuple[list[dict[str, float]], dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    thresholds = list(thresholds) if thresholds is not None else default_thresholds(raw_predictions)
    if not thresholds:
        raise ValueError("Threshold sweep requires at least one threshold")
    sweep_rows: list[dict[str, float]] = []
    evaluated: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for threshold in thresholds:
        predictions = apply_score_margin_threshold(raw_predictions, float(threshold))
        metrics = evaluate_predictions(predictions)
        sweep_rows.append(_metric_row(float(threshold), metrics))
        evaluated.append((predictions, metrics))
    if selection_metric not in sweep_rows[0]:
        raise ValueError(f"Unknown threshold selection metric: {selection_metric}")
    best_index = max(
        range(len(sweep_rows)),
        key=lambda index: (
            sweep_rows[index][selection_metric],
            sweep_rows[index]["answerable_f1"],
            sweep_rows[index]["overall_f1"],
        ),
    )
    predictions, metrics = evaluated[best_index]
    return sweep_rows, sweep_rows[best_index], predictions, metrics


def flatten_metrics(metrics: dict[str, Any], best_threshold: float | None = None) -> dict[str, float]:
    flattened = {
        "overall_em": float(metrics["overall"]["em"]),
        "overall_f1": float(metrics["overall"]["f1"]),
        "answerable_em": float(metrics["answerable"]["em"]),
        "answerable_f1": float(metrics["answerable"]["f1"]),
        "unanswerable_accuracy": float(metrics["unanswerable"]["accuracy"]),
        "predicted_no_answer_rate": float(metrics["predicted_no_answer"]["rate"]),
        "answerable_predicted_empty_rate": float(metrics["answerable"]["predicted_empty_rate"]),
        "false_positive_rate": 100.0 - float(metrics["unanswerable"]["accuracy"]),
        "false_negative_rate": float(metrics["answerable"]["predicted_empty_rate"]),
    }
    if best_threshold is not None:
        flattened["best_threshold"] = float(best_threshold)
    return flattened


def _classify_error(gold: str, prediction: str, is_answerable: bool) -> str:
    if not is_answerable:
        return "CORRECT_NO_ANSWER" if not prediction.strip() else "UNANSWERABLE_FALSE_POSITIVE"
    if not prediction.strip():
        return "ANSWERABLE_PREDICTED_EMPTY"
    if exact_match(gold, prediction):
        return "CORRECT"
    overlap = f1_score(gold, prediction)
    if overlap <= 0:
        return "WRONG_SPAN"
    gold_length = len(normalize_answer(gold).split())
    prediction_length = len(normalize_answer(prediction).split())
    if prediction_length < gold_length:
        return "SPAN_TOO_SHORT"
    if prediction_length > gold_length:
        return "SPAN_TOO_LONG"
    return "PARTIAL_SPAN"


def build_error_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in predictions:
        gold = str(row.get("gold_answer") or "")
        prediction = str(row.get("predicted_answer") or "")
        answerable = bool(row.get("is_answerable"))
        rows.append(
            {
                "question_id": row.get("id", ""),
                "question": row.get("question", ""),
                "gold_answer": gold,
                "predicted_answer": prediction,
                "has_gold_answer": answerable,
                "predicted_no_answer": not bool(prediction.strip()),
                "em": exact_match(gold, prediction),
                "f1": f1_score(gold, prediction),
                "reader_confidence": row.get("confidence", ""),
                "no_answer_score": row.get("no_answer_score", ""),
                "best_span_score": row.get("best_span_score", ""),
                "score_margin": row.get("score_margin", ""),
                "error_type": _classify_error(gold, prediction, answerable),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distribution_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
    }


def build_score_distribution(
    raw_predictions: Sequence[dict[str, Any]],
    threshold: float = 0.0,
    margin_scale: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    margin_scale = max(0.01, float(margin_scale))

    def calibrated_display_confidence(row: dict[str, Any]) -> float | None:
        margin = _json_number(row.get("score_margin"))
        if margin is None:
            return None
        shifted = max(-60.0, min(60.0, (margin - threshold) / margin_scale))
        return 1.0 / (1.0 + math.exp(-shifted))

    rows = [
        {
            "question_id": str(row.get("id", "")),
            "is_answerable": bool(row.get("is_answerable")),
            "gold_answer": str(row.get("gold_answer") or ""),
            "predicted_span": str(row.get("best_span_answer") or ""),
            "null_score": _json_number(row.get("null_score")),
            "best_span_score": _json_number(row.get("best_span_score")),
            "score_margin": _json_number(row.get("score_margin")),
            # The raw pass deliberately decodes every best span with a -inf
            # threshold. Its confidence is therefore saturated and cannot be
            # reused. Map the raw margin only after validation selects a
            # checkpoint-specific threshold.
            "reader_confidence_display": calibrated_display_confidence(row),
        }
        for row in raw_predictions
    ]
    summary: dict[str, Any] = {}
    for label, is_answerable in (("answerable", True), ("unanswerable", False)):
        selected = [row for row in rows if row["is_answerable"] is is_answerable]
        summary[label] = {
            "count": len(selected),
            "null_score": _distribution_statistics(
                [row["null_score"] for row in selected if row["null_score"] is not None]
            ),
            "best_span_score": _distribution_statistics(
                [row["best_span_score"] for row in selected if row["best_span_score"] is not None]
            ),
            "score_margin": _distribution_statistics(
                [row["score_margin"] for row in selected if row["score_margin"] is not None]
            ),
            "reader_confidence_display": _distribution_statistics(
                [
                    row["reader_confidence_display"]
                    for row in selected
                    if row["reader_confidence_display"] is not None
                ]
            ),
        }
    return rows, summary


def select_threshold_objectives(sweep_rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    objectives = {
        "maximize_overall_f1": "overall_f1",
        "maximize_answerable_f1": "answerable_f1",
        "reader_priority_0.7_answerable_f1_0.3_unanswerable_accuracy": "reader_priority_score",
    }
    selected: dict[str, dict[str, float]] = {}
    for objective, metric in objectives.items():
        best = max(
            sweep_rows,
            key=lambda row: (
                row[metric],
                row["answerable_f1"],
                row["unanswerable_accuracy"],
                row["overall_f1"],
            ),
        )
        selected[objective] = dict(best)
    return selected


def save_evaluation_artifacts(
    output_dir: str | Path,
    checkpoint: str,
    raw_predictions: Sequence[dict[str, Any]],
    sweep_rows: Sequence[dict[str, float]],
    best_row: dict[str, float],
    predictions: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    selection_metric: str,
    max_seq_len: int = DEFAULT_MAX_LENGTH,
    doc_stride: int = DEFAULT_DOC_STRIDE,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    error_rows = build_error_rows(predictions)
    answerable_empty = [
        row for row in error_rows if row["error_type"] == "ANSWERABLE_PREDICTED_EMPTY"
    ][:100]
    config = {
        "checkpoint": checkpoint,
        "threshold": best_row["threshold"],
        "score_type": "best_span_score_minus_null_score",
        "selection_metric": selection_metric,
        "selection_score": best_row[selection_metric],
        "overall_f1": metrics["overall"]["f1"],
        "answerable_f1": metrics["answerable"]["f1"],
        "unanswerable_accuracy": metrics["unanswerable"]["accuracy"],
        "calibrated": True,
    }
    result = {
        "dataset": "validation",
        "checkpoint": checkpoint,
        "threshold_semantics": "return answer when best_span_score - null_score >= threshold",
        "metrics": metrics,
        "predictions": _json_safe(list(predictions)),
    }
    objectives = select_threshold_objectives(sweep_rows)
    answerable_margins = [
        margin
        for row in raw_predictions
        if bool(row.get("is_answerable"))
        for margin in [_json_number(row.get("score_margin"))]
        if margin is not None
    ]
    margin_stats = _distribution_statistics(answerable_margins)
    p25 = margin_stats.get("p25")
    p75 = margin_stats.get("p75")
    margin_scale = max(0.01, float(p75) - float(p25)) if p25 is not None and p75 is not None else 10.0
    distribution_rows, distribution_summary = build_score_distribution(
        raw_predictions,
        threshold=float(best_row["threshold"]),
        margin_scale=margin_scale,
    )
    base_model = None
    checkpoint_config = Path(checkpoint) / "config.json"
    if checkpoint_config.is_file():
        try:
            base_model = json.loads(checkpoint_config.read_text(encoding="utf-8")).get(
                "_name_or_path"
            )
        except (OSError, json.JSONDecodeError):
            base_model = None
    profile = {
        "checkpoint": checkpoint,
        "base_model": base_model,
        "max_length": int(max_seq_len),
        "stride": int(doc_stride),
        "threshold": float(best_row["threshold"]),
        "score_type": "best_span_score_minus_null_score",
        "margin_scale": margin_scale,
        "temperature": None,
        "calibrated": len(raw_predictions) == 3814,
        "validation": {
            "examples": len(raw_predictions),
            "overall_em": metrics["overall"]["em"],
            "overall_f1": metrics["overall"]["f1"],
            "answerable_em": metrics["answerable"]["em"],
            "answerable_f1": metrics["answerable"]["f1"],
            "unanswerable_accuracy": metrics["unanswerable"]["accuracy"],
            "answerable_predicted_empty_rate": metrics["answerable"]["predicted_empty_rate"],
        },
        "threshold_objectives": objectives,
    }
    paths = {
        "metrics": output / "validation_metrics.json",
        "predictions": output / "predictions_validation.json",
        "sweep": output / "threshold_sweep.csv",
        "threshold": output / "best_threshold.json",
        "errors": output / "error_analysis.csv",
        "empty_analysis": output / "answerable_predicted_empty_sample.csv",
        "distribution_csv": output / "score_distribution.csv",
        "distribution_json": output / "score_distribution.json",
        "objectives": output / "threshold_objectives.json",
        "profile": output / "reader_profile.json",
    }
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["predictions"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    paths["threshold"].write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["distribution_json"].write_text(
        json.dumps(
            {"checkpoint": checkpoint, "summary": distribution_summary, "rows": distribution_rows},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    paths["objectives"].write_text(
        json.dumps(objectives, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paths["profile"].write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(paths["sweep"], list(sweep_rows))
    _write_csv(paths["errors"], error_rows)
    _write_csv(paths["empty_analysis"], answerable_empty)
    _write_csv(paths["distribution_csv"], distribution_rows)
    return paths


def raw_predictions_from_predictor(
    predictor,
    dataset,
    batch_size: int = 32,
    max_seq_len: int = DEFAULT_MAX_LENGTH,
    doc_stride: int = DEFAULT_DOC_STRIDE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        outputs = predictor.predict_batch(
            batch["question"],
            batch["context"],
            max_seq_len=max_seq_len,
            doc_stride=doc_stride,
            no_answer_threshold=float("-inf"),
        )
        for index, output in enumerate(outputs):
            gold = str(batch["answer_text"][index] or "")
            answer_start = int(batch["answer_start"][index])
            rows.append(
                {
                    "id": str(batch["id"][index]),
                    "question": str(batch["question"][index]),
                    "gold_answer": gold if answer_start >= 0 else "",
                    "is_answerable": answer_start >= 0 and bool(gold),
                    "best_span_answer": str(output.get("best_span_answer") or ""),
                    "best_span_score": float(output["best_span_score"]),
                    "null_score": float(output["null_score"]),
                    "no_answer_score": float(output["no_answer_score"]),
                    "score_margin": float(output["score_margin"]),
                    "confidence": float(output["confidence"]),
                }
            )
        print(f"Evaluated {min(start + batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return rows


def postprocess_validation_logits(
    examples,
    validation_features,
    predictions,
    preprocessor,
    max_answer_length: int = DEFAULT_MAX_ANSWER_LENGTH,
    top_n_start: int = 20,
    top_n_end: int = 20,
) -> list[dict[str, Any]]:
    """Decode Trainer logits into one raw candidate prediction per example."""

    start_logits, end_logits = predictions
    example_rows = {str(row["id"]): row for row in examples}
    prepared = {
        example_id: preprocessor.prepare(row["question"], row["context"])
        for example_id, row in example_rows.items()
    }
    best: dict[str, tuple[Any, int, int] | None] = {example_id: None for example_id in example_rows}
    null_scores: dict[str, list[float]] = {example_id: [] for example_id in example_rows}

    for feature_index in range(len(validation_features)):
        feature = validation_features[feature_index]
        example_id = str(feature["example_id"])
        cls_index = int(feature["cls_index"])
        null_scores[example_id].append(
            float(start_logits[feature_index][cls_index] + end_logits[feature_index][cls_index])
        )
        offsets = feature["offset_mapping"]
        candidates = select_span_candidates(
            start_logits[feature_index],
            end_logits[feature_index],
            offsets,
            sequence_ids=None,
            top_n_start=top_n_start,
            top_n_end=top_n_end,
            max_answer_length=max_answer_length,
        )
        item = prepared[example_id]
        for candidate in candidates:
            raw_start, raw_end = item.context_alignment.model_span_to_raw(
                candidate.start_char,
                candidate.end_char,
            )
            if not has_clean_word_boundaries(item.raw_context, raw_start, raw_end):
                continue
            current = best[example_id]
            if current is None or candidate.score > current[0].score:
                best[example_id] = (candidate, raw_start, raw_end)
            break

    rows: list[dict[str, Any]] = []
    for example_id, example in example_rows.items():
        selected = best[example_id]
        null_score = min(null_scores[example_id], default=float("inf"))
        if selected is None:
            span_answer = ""
            best_score = float("-inf")
        else:
            candidate, raw_start, raw_end = selected
            span_answer = str(example["context"])[raw_start:raw_end].strip()
            best_score = float(candidate.score)
        margin = best_score - null_score
        gold = str(example.get("answer_text") or "")
        answer_start = int(example.get("answer_start", -1))
        rows.append(
            {
                "id": example_id,
                "question": str(example["question"]),
                "gold_answer": gold if answer_start >= 0 else "",
                "is_answerable": answer_start >= 0 and bool(gold),
                "best_span_answer": span_answer,
                "best_span_score": best_score,
                "null_score": null_score,
                "no_answer_score": null_score - best_score,
                "score_margin": margin,
                "confidence": "",
            }
        )
    return rows


def evaluate(
    model_path: str,
    data_variant: str = "clean",
    subset_size: int = -1,
    use_cpu: bool = False,
    output_file: str | None = None,
    output_dir: str | None = None,
    thresholds: Sequence[float] | None = None,
    selection_metric: str = "reader_priority_score",
    batch_size: int = 32,
    max_seq_len: int = DEFAULT_MAX_LENGTH,
    doc_stride: int = DEFAULT_DOC_STRIDE,
    write_profile_to_checkpoint: bool = False,
):
    from reader.predict import ReaderPredictor

    # Validation has gold labels. The 7,301-row test split intentionally is not
    # accepted here because its processed answer labels are empty.
    validation_file = ROOT / "data" / "processed" / "viquad_val_clean.parquet"
    dataset = load_validation_records(validation_file, subset_size=subset_size)
    if subset_size <= 0 and len(dataset) != 3814:
        raise ValueError(f"Expected the full 3,814-question validation split, found {len(dataset)}")
    predictor = ReaderPredictor(model_path, use_cpu=use_cpu)
    raw = raw_predictions_from_predictor(
        predictor,
        dataset,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        doc_stride=doc_stride,
    )
    sweep, best, final_predictions, metrics = sweep_thresholds(
        raw,
        thresholds=thresholds,
        selection_metric=selection_metric,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        "Answerable questions predicted as empty: "
        f"{metrics['answerable']['predicted_empty']} / {metrics['answerable']['count']} "
        f"({metrics['answerable']['predicted_empty_rate']:.2f}%)"
    )

    artifact_dir = output_dir or str(ROOT / "results" / "reader" / Path(model_path).name)
    paths = save_evaluation_artifacts(
        artifact_dir,
        model_path,
        raw,
        sweep,
        best,
        final_predictions,
        metrics,
        selection_metric,
        max_seq_len=max_seq_len,
        doc_stride=doc_stride,
    )
    if write_profile_to_checkpoint:
        if len(dataset) != 3814:
            raise ValueError("A checkpoint profile may only be installed after full 3,814-example validation")
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise ValueError("--write_profile_to_checkpoint requires a local checkpoint directory")
        shutil.copyfile(paths["profile"], model_dir / "reader_profile.json")
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(
            json.dumps({"metrics": metrics, "predictions": final_predictions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Best threshold: {best['threshold']:.6f} ({selection_metric}={best[selection_metric]:.2f})")
    print(f"Artifacts: {paths}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Reader on the full UIT-ViQuAD2.0 validation split")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_variant", default="clean", choices=["clean", "segmented", "auto"])
    parser.add_argument("--subset_size", type=int, default=-1, help="Smoke tests only; never use for final selection")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--output_file", default=None, help="Legacy combined JSON path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--doc_stride", type=int, default=DEFAULT_DOC_STRIDE)
    parser.add_argument(
        "--write_profile_to_checkpoint",
        action="store_true",
        help="Install reader_profile.json only after full validation",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["reader_priority_score", "balanced_score", "overall_f1", "answerable_f1"],
        default="reader_priority_score",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()


__all__ = [
    "apply_score_margin_threshold",
    "build_error_rows",
    "compute_exact",
    "compute_f1",
    "evaluate",
    "flatten_metrics",
    "build_score_distribution",
    "select_threshold_objectives",
    "normalize_answer",
    "postprocess_validation_logits",
    "save_evaluation_artifacts",
    "sweep_thresholds",
]
