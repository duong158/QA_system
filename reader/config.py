from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_LENGTH = 256
DEFAULT_DOC_STRIDE = 80
DEFAULT_MAX_ANSWER_LENGTH = 40
DEFAULT_MAX_ANSWER_LENGTH_BY_TYPE = {
    "TIME": 12,
    "NUMBER": 10,
    "PERSON": 16,
    "LOCATION": 20,
    "ENTITY": 24,
    "DEFINITION": 48,
    "GENERAL": 64,
}
DEFAULT_SCORE_MARGIN_THRESHOLD = 0.0
DEFAULT_CONFIDENCE_TEMPERATURE = 10.0
DEFAULT_MARGIN_SCALE = 10.0


@dataclass(frozen=True)
class ReaderDecisionConfig:
    checkpoint: str | None = None
    threshold: float = DEFAULT_SCORE_MARGIN_THRESHOLD
    score_type: str = "best_span_score_minus_null_score"
    confidence_temperature: float = DEFAULT_CONFIDENCE_TEMPERATURE
    margin_scale: float = DEFAULT_MARGIN_SCALE
    calibrated: bool = False
    source: str | None = None


def max_answer_length_for_type(question_type: Any) -> int:
    """Return the initial, benchmarkable span-length cap for a question type."""

    name = str(getattr(question_type, "value", question_type)).upper()
    env_name = f"QA_MAX_ANSWER_LENGTH_{name}"
    configured = os.getenv(env_name)
    if configured is not None:
        value = int(configured)
    else:
        value = int(
            DEFAULT_MAX_ANSWER_LENGTH_BY_TYPE.get(name, DEFAULT_MAX_ANSWER_LENGTH)
        )
    if value <= 0:
        raise ValueError(f"{env_name} must be positive")
    return value


def _candidate_config_paths(model_path: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if model_path:
        model_dir = Path(model_path)
        if model_dir.is_dir():
            paths.append(model_dir / "reader_profile.json")
            paths.append(model_dir / "best_reader_config.json")
    else:
        # Legacy global configuration is only considered when no checkpoint was
        # supplied. A checkpoint must never inherit another checkpoint's margin.
        paths.append(ROOT / "models" / "reader" / "best_reader_config.json")
    return paths


def _profile_matches_checkpoint(payload: dict[str, Any], model_path: str | Path | None) -> bool:
    """Reject a profile that names a different local checkpoint."""

    declared = payload.get("checkpoint")
    if not declared or not model_path:
        return True
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return str(declared) == str(model_path)
    declared_path = Path(str(declared))
    if not declared_path.is_absolute():
        declared_path = ROOT / declared_path
    try:
        return declared_path.resolve() == model_dir.resolve()
    except OSError:
        return False


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
                candidate = json.load(handle)
            if not _profile_matches_checkpoint(candidate, model_path):
                warnings.warn(
                    f"Ignoring Reader profile {path}: checkpoint does not match {model_path}",
                    RuntimeWarning,
                )
                continue
            payload = candidate
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
    margin_scale = max(
        0.01,
        float(
            os.getenv(
                "QA_READER_MARGIN_SCALE",
                payload.get("margin_scale", payload.get("confidence_temperature", DEFAULT_MARGIN_SCALE))
                or DEFAULT_MARGIN_SCALE,
            )
        ),
    )
    return ReaderDecisionConfig(
        checkpoint=payload.get("checkpoint"),
        threshold=threshold,
        score_type=str(payload.get("score_type", "best_span_score_minus_null_score")),
        confidence_temperature=temperature,
        margin_scale=margin_scale,
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
