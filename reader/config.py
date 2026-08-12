from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_LENGTH = 256
DEFAULT_DOC_STRIDE = 80
DEFAULT_MAX_ANSWER_LENGTH = 40
DEFAULT_SCORE_MARGIN_THRESHOLD = 0.0
DEFAULT_CONFIDENCE_TEMPERATURE = 10.0


@dataclass(frozen=True)
class ReaderDecisionConfig:
    checkpoint: str | None = None
    threshold: float = DEFAULT_SCORE_MARGIN_THRESHOLD
    score_type: str = "best_span_score_minus_null_score"
    confidence_temperature: float = DEFAULT_CONFIDENCE_TEMPERATURE
    calibrated: bool = False
    source: str | None = None


def _candidate_config_paths(model_path: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if model_path:
        model_dir = Path(model_path)
        if model_dir.is_dir():
            paths.append(model_dir / "best_reader_config.json")
    paths.append(ROOT / "models" / "reader" / "best_reader_config.json")
    return paths


def load_reader_decision_config(model_path: str | Path | None = None) -> ReaderDecisionConfig:
    """Load the validation-calibrated Reader decision threshold when available.

    The threshold is applied to ``best_span_score - null_score``. It is not a
    probability threshold. Environment overrides are intended for controlled
    experiments and take precedence over the persisted calibration artifact.
    """

    payload: dict[str, Any] = {}
    source: str | None = None
    explicit_path = os.getenv("QA_READER_CONFIG")
    candidates = [Path(explicit_path)] if explicit_path else _candidate_config_paths(model_path)
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            source = str(path)
            break

    threshold = float(
        os.getenv(
            "QA_READER_SCORE_MARGIN_THRESHOLD",
            payload.get("threshold", DEFAULT_SCORE_MARGIN_THRESHOLD),
        )
    )
    temperature = max(
        0.01,
        float(
            os.getenv(
                "QA_READER_CONFIDENCE_TEMPERATURE",
                payload.get("confidence_temperature", DEFAULT_CONFIDENCE_TEMPERATURE),
            )
        ),
    )
    return ReaderDecisionConfig(
        checkpoint=payload.get("checkpoint"),
        threshold=threshold,
        score_type=str(payload.get("score_type", "best_span_score_minus_null_score")),
        confidence_temperature=temperature,
        calibrated=bool(payload.get("calibrated", False)),
        source=source,
    )


def effective_model_max_length(model_or_config: Any, tokenizer: Any) -> int:
    """Return a safe sequence length for the model/tokenizer pair.

    PhoBERT has 258 position embeddings, of which two are consumed by the
    RoBERTa position-id convention, leaving 256 usable input tokens.
    """

    config = getattr(model_or_config, "config", model_or_config)
    model_limit = int(getattr(config, "max_position_embeddings", DEFAULT_MAX_LENGTH))
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type == "roberta":
        model_limit = max(8, model_limit - 2)

    tokenizer_limit = int(getattr(tokenizer, "model_max_length", model_limit))
    # Hugging Face uses very large sentinels for tokenizers with no declared limit.
    if tokenizer_limit > 1_000_000:
        tokenizer_limit = model_limit
    return min(model_limit, tokenizer_limit)


def validate_window_config(max_length: int, doc_stride: int, model_or_config: Any, tokenizer: Any) -> None:
    limit = effective_model_max_length(model_or_config, tokenizer)
    if max_length > limit:
        raise ValueError(
            f"max_length={max_length} exceeds the safe model/tokenizer limit {limit}. "
            "PhoBERT baseline must use max_length=256."
        )
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if doc_stride <= 0 or doc_stride >= max_length:
        raise ValueError("doc_stride must be positive and strictly smaller than max_length")
