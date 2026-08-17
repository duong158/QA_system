from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reader.config import load_reader_decision_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "qa_pipeline.json"


def _env(name: str, value: Any, cast):
    raw = os.getenv(name)
    return cast(raw) if raw is not None else cast(value)


@dataclass(frozen=True)
class PipelineConfig:
    default_retriever: str
    default_top_k: int
    max_top_k: int
    candidate_multiplier: int
    minimum_candidate_count: int
    maximum_candidate_count: int
    retriever_weight: float
    reader_weight: float
    answer_type_weight: float
    relation_weight: float
    reader_checkpoint: Path
    reader_score_margin_threshold: float
    reader_margin_scale: float
    reader_profile_calibrated: bool
    require_calibrated_reader_profile: bool
    minimum_reader_score: float
    minimum_answer_type_score: float
    minimum_fallback_answer_type_score: float
    minimum_ranking_score: float
    fallback_penalty: float
    phrase_fallback_penalty: float
    reader_fallback_threshold: float
    sentence_fallback_threshold: float
    reader_max_length: int
    reader_stride: int
    chunk_max_tokens: int
    chunk_overlap_sentences: int

    def candidate_count(self, top_k: int) -> int:
        return min(
            self.maximum_candidate_count,
            max(top_k, self.minimum_candidate_count, top_k * max(1, self.candidate_multiplier)),
        )


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    config_path = Path(path or os.getenv("QA_PIPELINE_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    reader_checkpoint = Path(
        os.getenv("QA_READER_CHECKPOINT", payload["reader_checkpoint"])
    )
    if not reader_checkpoint.is_absolute():
        reader_checkpoint = ROOT / reader_checkpoint
    decision = load_reader_decision_config(reader_checkpoint)
    configured_margin = payload.get("reader_score_margin_threshold")
    margin_threshold = (
        decision.threshold if configured_margin is None else float(configured_margin)
    )
    margin_threshold = _env("QA_READER_SCORE_MARGIN_THRESHOLD", margin_threshold, float)

    retriever_weight = _env("QA_RETRIEVER_WEIGHT", payload["retriever_weight"], float)
    reader_weight = _env("QA_READER_WEIGHT", payload["reader_weight"], float)
    answer_type_weight = _env("QA_ANSWER_TYPE_WEIGHT", payload["answer_type_weight"], float)
    relation_weight = _env("QA_RELATION_WEIGHT", payload.get("relation_weight", 0.0), float)
    weights = (retriever_weight, reader_weight, answer_type_weight, relation_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("Ranking weights must be non-negative")
    total = sum(weights)
    if not 0.999999 <= total <= 1.000001:
        raise ValueError(
            "retriever_weight + reader_weight + answer_type_weight + relation_weight must equal 1"
        )

    config = PipelineConfig(
        default_retriever=str(payload.get("default_retriever", "bm25")),
        default_top_k=_env("QA_TOP_K", payload["default_top_k"], int),
        max_top_k=int(payload.get("max_top_k", 20)),
        candidate_multiplier=_env(
            "QA_RETRIEVER_CANDIDATE_MULTIPLIER", payload["candidate_multiplier"], int
        ),
        minimum_candidate_count=_env(
            "QA_RETRIEVER_MIN_CANDIDATES", payload["minimum_candidate_count"], int
        ),
        maximum_candidate_count=_env(
            "QA_RETRIEVER_MAX_CANDIDATES", payload["maximum_candidate_count"], int
        ),
        retriever_weight=retriever_weight,
        reader_weight=reader_weight,
        answer_type_weight=answer_type_weight,
        relation_weight=relation_weight,
        reader_checkpoint=reader_checkpoint,
        reader_score_margin_threshold=margin_threshold,
        reader_margin_scale=decision.margin_scale,
        reader_profile_calibrated=decision.calibrated,
        require_calibrated_reader_profile=bool(
            payload.get("require_calibrated_reader_profile", False)
        ),
        minimum_reader_score=_env(
            "QA_MIN_READER_SCORE", payload["minimum_reader_score"], float
        ),
        minimum_answer_type_score=_env(
            "QA_MIN_ANSWER_TYPE_SCORE", payload["minimum_answer_type_score"], float
        ),
        minimum_fallback_answer_type_score=_env(
            "QA_MIN_FALLBACK_ANSWER_TYPE_SCORE",
            payload["minimum_fallback_answer_type_score"],
            float,
        ),
        minimum_ranking_score=_env(
            "QA_MIN_RANKING_SCORE", payload["minimum_ranking_score"], float
        ),
        fallback_penalty=_env(
            "QA_FALLBACK_PENALTY", payload["fallback_penalty"], float
        ),
        phrase_fallback_penalty=_env(
            "QA_PHRASE_FALLBACK_PENALTY",
            payload.get("phrase_fallback_penalty", 1.0),
            float,
        ),
        reader_fallback_threshold=_env(
            "QA_READER_FALLBACK_THRESHOLD", payload["reader_fallback_threshold"], float
        ),
        sentence_fallback_threshold=_env(
            "QA_SENTENCE_FALLBACK_THRESHOLD", payload["sentence_fallback_threshold"], float
        ),
        reader_max_length=_env("QA_READER_MAX_LENGTH", payload["reader_max_length"], int),
        reader_stride=_env("QA_READER_STRIDE", payload["reader_stride"], int),
        chunk_max_tokens=_env("QA_CHUNK_MAX_TOKENS", payload["chunk_max_tokens"], int),
        chunk_overlap_sentences=_env(
            "QA_CHUNK_OVERLAP_SENTENCES", payload["chunk_overlap_sentences"], int
        ),
    )
    if config.default_top_k <= 0 or config.default_top_k > config.max_top_k:
        raise ValueError("default_top_k must be within 1..max_top_k")
    if config.minimum_candidate_count < config.default_top_k:
        raise ValueError("minimum_candidate_count must be at least default_top_k")
    if config.require_calibrated_reader_profile and not config.reader_profile_calibrated:
        raise ValueError(
            f"Reader checkpoint {config.reader_checkpoint} has no matching full-validation "
            "reader_profile.json. Refusing calibrated production mode."
        )
    for name in (
        "minimum_reader_score",
        "minimum_answer_type_score",
        "minimum_fallback_answer_type_score",
        "minimum_ranking_score",
        "fallback_penalty",
        "phrase_fallback_penalty",
    ):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within 0..1")
    return config
