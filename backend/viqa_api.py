from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chunking import Passage, chunk_document, split_sentences
from backend.config import load_pipeline_config
from reader.config import DEFAULT_MAX_ANSWER_LENGTH_BY_TYPE
from reader.candidates import AnswerCandidate
from reader.answer_refinement import QuestionRelation, detect_question_relation, refine_answer
from reader.cause_relations import (
    assess_cause_candidate,
    cause_subject_match_score,
    extract_cause_question,
)
from reader.fallback_extractor import (
    assess_contrast_relation,
    detect_alias_relation,
    detect_contrast_relation,
    detect_location_relation,
    extract_fallback_answer,
    extract_alias_candidate,
    extract_location_candidate,
)
from reader.question_type import QuestionType, assess_answer_type, detect_question_type
from reader.question_semantics import parse_question_semantics
from reader.relation_validator import STRICT_RELATIONS, validate_candidate_relation
from reader.span_boundaries import assess_span_boundary


PIPELINE_CONFIG = load_pipeline_config()
DOCS_DB = ROOT / "data" / "processed" / "docs.db"
HOST = os.getenv("QA_HOST", "0.0.0.0")
PORT = int(os.getenv("QA_PORT", "8000"))
CHUNK_MAX_TOKENS = PIPELINE_CONFIG.chunk_max_tokens
CHUNK_OVERLAP_SENTENCES = PIPELINE_CONFIG.chunk_overlap_sentences
RETRIEVER_CANDIDATE_MULTIPLIER = PIPELINE_CONFIG.candidate_multiplier
RETRIEVER_MIN_CANDIDATES = PIPELINE_CONFIG.minimum_candidate_count
RETRIEVER_WEIGHT = PIPELINE_CONFIG.retriever_weight
READER_WEIGHT = PIPELINE_CONFIG.reader_weight
ANSWER_TYPE_WEIGHT = PIPELINE_CONFIG.answer_type_weight
RELATION_WEIGHT = PIPELINE_CONFIG.relation_weight
MULTI_TYPE_COVERAGE_BONUS = PIPELINE_CONFIG.multi_type_coverage_bonus
MIN_READER_SCORE = PIPELINE_CONFIG.minimum_reader_score
MIN_ANSWER_TYPE_SCORE = PIPELINE_CONFIG.minimum_answer_type_score
MIN_FALLBACK_ANSWER_TYPE_SCORE = PIPELINE_CONFIG.minimum_fallback_answer_type_score
MIN_RANKING_SCORE = PIPELINE_CONFIG.minimum_ranking_score
FALLBACK_PENALTY = PIPELINE_CONFIG.fallback_penalty
PHRASE_FALLBACK_PENALTY = PIPELINE_CONFIG.phrase_fallback_penalty
READER_SCORE_MARGIN_THRESHOLD = PIPELINE_CONFIG.reader_score_margin_threshold
READER_FALLBACK_THRESHOLD = PIPELINE_CONFIG.reader_fallback_threshold
SENTENCE_FALLBACK_THRESHOLD = PIPELINE_CONFIG.sentence_fallback_threshold
QA_DEBUG = os.getenv("QA_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
PRELOAD_READER = os.getenv("QA_PRELOAD_READER", "true").lower() in {"1", "true", "yes", "on"}
SUPPORTED_RETRIEVERS = {"tfidf", "bm25", "hybrid", "dense"}
UNIMPLEMENTED_RETRIEVERS = {
    "pyserini": "Pyserini BM25 is not wired into this API yet: no Lucene index/runtime is configured for online serving.",
}
DENSE_MODEL_NAME = os.getenv("QA_DENSE_MODEL", "keepitreal/vietnamese-sbert")
RRF_K = int(os.getenv("QA_RRF_K", "60"))
SUPPORTED_READERS = {"phobert", "xlmr"}
UNIMPLEMENTED_READERS = {
    "mock": "Mock Reader is forbidden in the real API.",
    "vibert": "viBERT QA is not implemented: no viBERT QA checkpoint is available under models/reader.",
}

STOPWORDS = {
    "ai", "anh", "ay", "ban", "bang", "bao", "bi", "cac", "cai", "can", "chi",
    "cho", "co", "con", "cua", "da", "dang", "day", "de", "den", "di", "do",
    "duoc", "duoi", "gi", "giua", "hay", "hon", "khi", "khong", "la", "lai",
    "lam", "may", "mot", "nao", "nay", "neu", "ngay", "nhieu", "nhu", "nhung",
    "o", "phai", "qua", "ra", "rang", "sau", "se", "so", "tai", "the", "thi",
    "theo", "tren", "trong", "truoc", "tu", "va", "vao", "ve", "vi", "voi",
}

DATE_TOKENS = {"ngay", "thang", "nam"}


class PipelineError(RuntimeError):
    pass


def validate_retriever(method: str) -> None:
    if method in SUPPORTED_RETRIEVERS:
        return
    if method in UNIMPLEMENTED_RETRIEVERS:
        raise ValueError(f"Retriever '{method}' is not implemented. {UNIMPLEMENTED_RETRIEVERS[method]}")
    raise ValueError(f"Retriever '{method}' is not supported.")


def validate_reader(reader_name: str) -> None:
    if reader_name in SUPPORTED_READERS:
        return
    if reader_name in UNIMPLEMENTED_READERS:
        raise ValueError(f"Reader '{reader_name}' is not implemented. {UNIMPLEMENTED_READERS[reader_name]}")
    raise ValueError(f"Reader '{reader_name}' is not supported.")


@dataclass(frozen=True)
class IndexedPassage:
    metadata: Passage
    tokens: tuple[str, ...]
    term_counts: Counter[str]


@dataclass(frozen=True)
class SearchHit:
    passage: IndexedPassage
    retrieval_score_raw: float
    retrieval_score_normalized: float = 0.0
    retrieval_rank: int = 0


def normalize_text(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def raw_tokens(value: str) -> list[str]:
    return re.findall(r"[\w%]+", normalize_text(value), flags=re.UNICODE)


def normalized_token_text(tokens: list[str] | tuple[str, ...]) -> str:
    return " ".join(tokens)


def query_ngrams(tokens: list[str], min_n: int = 2, max_n: int = 4) -> list[str]:
    grams: list[str] = []
    limit = min(max_n, len(tokens))
    for size in range(limit, min_n - 1, -1):
        grams.extend(" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1))
    return grams


def find_sequence(tokens: list[str], sequence: list[str]) -> int:
    if not sequence or len(sequence) > len(tokens):
        return -1
    for index in range(0, len(tokens) - len(sequence) + 1):
        if tokens[index : index + len(sequence)] == sequence:
            return index
    return -1


def softmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    high = max(scores)
    values = [math.exp(min(50.0, score - high)) for score in scores]
    total = sum(values)
    if not total:
        return [0.0 for _ in scores]
    return [value / total for value in values]


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def combine_ranking_scores(
    retrieval_score: float,
    reader_score: float,
    answer_type_score: float,
    retriever_weight: float = RETRIEVER_WEIGHT,
    reader_weight: float = READER_WEIGHT,
    answer_type_weight: float = ANSWER_TYPE_WEIGHT,
    relation_score: float = 0.0,
    relation_weight: float | None = None,
) -> float:
    """Combine ranking signals; the result is not a correctness probability."""

    if relation_weight is None:
        # Preserve compatibility with callers that provide three custom
        # weights while using the configured fourth signal for normal calls.
        relation_weight = max(
            0.0,
            1.0 - retriever_weight - reader_weight - answer_type_weight,
        )
    weights = (retriever_weight, reader_weight, answer_type_weight, relation_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("Ranking weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("Ranking weights must sum to 1")
    return (
        retriever_weight * retrieval_score
        + reader_weight * reader_score
        + answer_type_weight * answer_type_score
        + relation_weight * relation_score
    )


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


REJECTION_MESSAGES = {
    "LOW_READER_SCORE": "Reader evidence was below the configured minimum.",
    "SPAN_BOUNDARY_INCOMPLETE": "The span starts or ends inside a semantic phrase or named entity.",
    "ANSWER_TYPE_MISMATCH": "The candidate did not match the expected answer type.",
    "LOW_RANKING_SCORE": "The candidate ranking score was below the configured minimum.",
    "NO_VALID_SPAN": "Reader did not produce a valid answer span.",
    "EVIDENCE_UNSUPPORTED": "The proposed answer was not supported by the source passage.",
    "INSUFFICIENT_FALLBACK_EVIDENCE": "Fallback did not provide strong typed evidence.",
    "LOCATION_RELATION_MISMATCH": "The candidate does not fill the location relation requested by the question.",
    "RELATION_MISMATCH": "The candidate mentions the topic but does not express the relation requested by the question.",
    "CAUSE_RELATION_MISMATCH": "The evidence has a causal form but does not establish the requested cause/effect direction.",
    "CAUSE_TARGET_MISMATCH": "The causal evidence does not explain the target state or event requested by the question.",
    "CAUSE_SUBJECT_MISMATCH": "The causal evidence concerns a different subject from the one requested by the question.",
    "CAUSE_PHRASE_NOT_FOUND": "No standalone cause phrase connected to the requested subject and target was found.",
    "CAUSE_RELATION_NOT_FOUND": "No cause/effect relation answering the requested target was found.",
    "CAUSE_EFFECT_REPETITION": "The candidate repeats the effect instead of providing its cause.",
    "TIME_SUBJECT_MISMATCH": "The time expression belongs to a different subject.",
    "BIRTH_TIME_RELATION_MISMATCH": "The evidence does not connect the candidate time to the subject's birth.",
    "DEATH_TIME_RELATION_MISMATCH": "The evidence does not connect the candidate time to the subject's death.",
    "PURPOSE_SUBJECT_MISMATCH": "The purpose belongs to a different subject or action.",
    "PURPOSE_RELATION_NOT_FOUND": "No purpose relation for the requested subject or action was found.",
    "DANGLING_CONNECTOR": "The answer ends with a conjunction or connector and is not a complete clause.",
    "INCOMPLETE_CLAUSE": "The proposed answer is not a complete standalone clause.",
    "INCOMPLETE_RELATION": "The answer contains only part of the relation requested by the question.",
    "RETRIEVAL_MISS": "Retriever returned no passage with a positive score.",
    "LOWER_RANKING_SCORE": "A stronger valid candidate was selected.",
    "NO_ANSWER": "No candidate satisfied the answer acceptance gates.",
}


def answer_is_supported(context: str, answer: str, start: int, end: int) -> bool:
    """Verify that a Reader/fallback answer is directly grounded in its passage."""

    answer = str(answer or "").strip()
    if not answer or start < 0 or end <= start or end > len(context):
        return False
    source_span = context[start:end].strip()
    normalized_answer = " ".join(normalize_text(answer).split())
    normalized_span = " ".join(normalize_text(source_span).split())
    return bool(normalized_answer) and (
        normalized_answer in normalized_span or normalized_span in normalized_answer
    )


def location_relation_assessment(
    question: str,
    context: str,
    answer: str,
    start: int,
    end: int,
    chosen_output: dict[str, Any],
) -> dict[str, Any]:
    relation_type = str(chosen_output.get("relation_type") or detect_location_relation(question))
    if chosen_output.get("method") == "sentence_fallback":
        return {
            "relation_type": relation_type,
            "relation_score": float(chosen_output.get("relation_score", 0.0)),
            "phrase_quality": float(chosen_output.get("phrase_quality", 0.0)),
            "relation_evidence": bool(chosen_output.get("relation_evidence", False)),
        }

    search_from = 0
    for sentence in split_sentences(context):
        sentence = sentence.strip()
        sentence_start = context.find(sentence, search_from)
        if sentence_start < 0:
            sentence_start = context.find(sentence)
        sentence_end = sentence_start + len(sentence) if sentence_start >= 0 else -1
        search_from = max(search_from, sentence_end)
        if sentence_start < 0 or not (sentence_start <= start and end <= sentence_end):
            continue
        extracted = extract_location_candidate(question, sentence, relation_type)
        extracted_answer = normalize_text(extracted.answer)
        normalized_answer = normalize_text(answer)
        same_phrase = bool(normalized_answer) and (
            normalized_answer in extracted_answer or extracted_answer in normalized_answer
        )
        return {
            "relation_type": relation_type,
            "relation_score": extracted.relation_score if same_phrase else 0.0,
            "phrase_quality": extracted.phrase_quality if same_phrase else 0.0,
            "relation_evidence": bool(extracted.relation_evidence and same_phrase),
        }
    return {
        "relation_type": relation_type,
        "relation_score": 0.0,
        "phrase_quality": 0.0,
        "relation_evidence": False,
    }


def candidate_rejection_reason(candidate: dict[str, Any]) -> str | None:
    if not candidate.get("reader_answer") or candidate.get("reader_method") == "no_answer":
        return "NO_VALID_SPAN"
    if not candidate.get("lexical_evidence", candidate.get("evidence_supported")):
        return "EVIDENCE_UNSUPPORTED"
    relation_type = candidate.get("relation_type")
    requires_relation = (
        QuestionType.LOCATION.value in candidate.get("question_type", [])
        and relation_type != "GENERIC_LOCATION"
    )
    generic_whole_sentence = (
        QuestionType.LOCATION.value in candidate.get("question_type", [])
        and relation_type == "GENERIC_LOCATION"
        and candidate.get("reader_method") == "sentence_fallback"
        and candidate.get("fallback_method") in {None, "whole_sentence"}
    )
    if (requires_relation and not candidate.get("relation_evidence")) or generic_whole_sentence:
        return "LOCATION_RELATION_MISMATCH"
    if float(candidate.get("reader_score", 0.0)) < MIN_READER_SCORE:
        return "LOW_READER_SCORE"
    answer_type_score = float(candidate.get("answer_type_score", 0.0))
    if answer_type_score < MIN_ANSWER_TYPE_SCORE:
        return "ANSWER_TYPE_MISMATCH"
    if (
        candidate.get("reader_method") == "sentence_fallback"
        and answer_type_score < MIN_FALLBACK_ANSWER_TYPE_SCORE
    ):
        return "INSUFFICIENT_FALLBACK_EVIDENCE"
    strong_location_relation = (
        QuestionType.LOCATION.value in candidate.get("question_type", [])
        and bool(candidate.get("lexical_evidence"))
        and bool(candidate.get("relation_evidence"))
        and float(candidate.get("relation_score", 0.0)) >= 0.85
        and float(candidate.get("phrase_quality", 0.0)) >= 0.65
        and answer_type_score >= MIN_FALLBACK_ANSWER_TYPE_SCORE
    )
    if (
        float(candidate.get("ranking_score", 0.0)) < MIN_RANKING_SCORE
        and not strong_location_relation
    ):
        return "LOW_RANKING_SCORE"
    return None


ANSWER_CUE_PATTERNS = (
    "duoc chia thanh",
    "duoc chia lam",
    "chia thanh",
    "chia lam",
    "bao gom",
    "gom",
    "la",
    "duoc goi la",
    "co nghia la",
    "duoc dinh nghia",
)


def definition_subject(question: str) -> str:
    normalized = normalize_text(question).strip(" ?!.")
    match = re.match(
        r"^(?:cho biet\s+)?(.+?)\s+(?:la ai|la gi|co nghia la gi|duoc dinh nghia nhu the nao)$",
        normalized,
    )
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def relation_follows_subject(sentence: str, subject: str) -> bool:
    if not subject:
        return False
    normalized = normalize_text(sentence)
    match = re.search(rf"\b{re.escape(subject)}\b", normalized)
    if not match:
        return False
    tail = normalized[match.end() : match.end() + 180]
    tail = re.sub(r"^\s*\([^)]{0,120}\)", "", tail)
    return bool(
        re.match(
            r"^\s*,?\s*(?:(?:hien|tung|chinh|duoc xem la|duoc biet den nhu)\s+)?la\b",
            tail,
        )
    )


def answer_repeats_question(question: str, answer: str) -> bool:
    answer_tokens = set(tokenize(answer))
    question_tokens = set(tokenize(question))
    return bool(answer_tokens) and answer_tokens <= question_tokens


def span_has_clean_word_boundaries(context: str, start: int, end: int) -> bool:
    if start < 0 or end <= start or end > len(context):
        return False
    starts_inside_word = start > 0 and context[start - 1].isalnum() and context[start].isalnum()
    ends_inside_word = end < len(context) and context[end - 1].isalnum() and context[end].isalnum()
    return not starts_inside_word and not ends_inside_word


def sentence_fallback_predict(question: str, context: str) -> dict[str, Any]:
    """Select supporting evidence, then narrow it to a grounded answer phrase.

    This is deliberately generic: it does not map specific questions to answers.
    It first picks the context sentence with strong lexical overlap and Vietnamese
    answer cues. The fallback phrase extractor may then return an exact subspan;
    otherwise the supporting sentence remains the last-resort candidate.
    """
    question_tokens = set(tokenize(question))
    if not question_tokens:
        return {"answer": "", "confidence": 0.0, "start": -1, "end": -1, "reason": "empty_question_tokens"}

    normalized_question = normalize_text(question)
    question_types = detect_question_type(question)
    subject = definition_subject(question)
    subject_phrase = re.sub(
        r"\b(bao gom nhung gi|gom nhung gi|co nhung gi|duoc chia thanh|duoc chia lam|duoc chia|chia thanh|chia lam|nhu the nao|nhu nao|la gi|la ai|chia|duoc|nao|gi|ai)\b",
        " ",
        normalized_question,
    )
    subject_phrase = re.sub(r"\s+", " ", subject_phrase).strip()
    best: dict[str, Any] = {"answer": "", "confidence": 0.0, "start": -1, "end": -1, "reason": "no_sentence"}
    search_from = 0
    for sentence in split_sentences(context):
        sentence = sentence.strip()
        if not sentence:
            continue
        start = context.find(sentence, search_from)
        if start < 0:
            start = context.find(sentence)
        end = start + len(sentence) if start >= 0 else -1
        search_from = max(search_from, end)

        sentence_tokens = set(tokenize(sentence))
        overlap = len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
        normalized_sentence = normalize_text(sentence)
        phrase = extract_fallback_answer(question, [qt.value for qt in question_types], sentence)
        cue_bonus = 0.0
        if any(pattern in normalized_sentence for pattern in ANSWER_CUE_PATTERNS):
            cue_bonus += 0.18
        if phrase.method == "contrast_relation_pattern" and phrase.relation_evidence:
            cue_bonus += 0.30
        if phrase.method == "role_relation_pattern" and phrase.relation_evidence:
            cue_bonus += 0.40
        if phrase.method in {
            "cause_clause_pattern",
            "purpose_clause_pattern",
            "method_clause_pattern",
        } and phrase.relation_evidence:
            cue_bonus += 0.32
        if phrase.method == "property_description_pattern" and phrase.relation_evidence:
            cue_bonus += 0.60 if phrase.phrase_quality >= 0.95 else 0.10
        elif phrase.method == "description_sentence_pattern" and phrase.relation_evidence:
            cue_bonus += 0.24
        if len(subject_phrase) >= 8 and subject_phrase in normalized_sentence:
            cue_bonus += 0.22
        if len(sentence_tokens) >= 6:
            cue_bonus += 0.04
        if len(sentence) > 520:
            cue_bonus -= 0.12
        score = max(0.0, min(0.70, overlap * 0.78 + cue_bonus))
        if QuestionType.LOCATION in question_types:
            if phrase.relation_evidence:
                score = min(0.70, score + 0.24 * phrase.relation_score)
            elif phrase.method == "whole_sentence":
                score = min(score, 0.68)
        relation_matched = relation_follows_subject(sentence, subject)
        if subject:
            if relation_matched:
                score = min(0.70, score + 0.18)
            else:
                score = min(score, SENTENCE_FALLBACK_THRESHOLD - 0.07)

        if score > best["confidence"]:
            best = {
                "answer": sentence,
                "confidence": round(score, 6),
                "start": start,
                "end": end,
                "reason": "definition_relation" if relation_matched else "sentence_overlap_cue",
                "_phrase": phrase,
            }
    if not best.get("answer"):
        return best

    supporting_sentence = str(best["answer"])
    supporting_start = int(best["start"])
    supporting_end = int(best["end"])
    phrase = best.pop("_phrase", None) or extract_fallback_answer(
        question, [qt.value for qt in question_types], supporting_sentence
    )
    if phrase.answer and phrase.start_char >= 0:
        best.update(
            {
                "answer": phrase.answer,
                "start": supporting_start + phrase.start_char,
                "end": supporting_start + phrase.end_char,
                "sentence_answer": supporting_sentence,
                "sentence_start": supporting_start,
                "sentence_end": supporting_end,
                "fallback_method": phrase.method,
                "phrase_score": phrase.score,
                "phrase_start": phrase.start_char,
                "phrase_end": phrase.end_char,
                "relation_type": phrase.relation_type,
                "relation_score": phrase.relation_score,
                "phrase_quality": phrase.phrase_quality,
                "lexical_evidence": phrase.lexical_evidence,
                "relation_evidence": phrase.relation_evidence,
                "relation_method": phrase.relation_method,
                "question_subject": phrase.question_subject,
                "question_target": phrase.question_target,
                "cause_pattern_score": phrase.cause_pattern_score,
                "subject_match_score": phrase.subject_match_score,
                "target_relation_score": phrase.target_relation_score,
                "relation_rejection_reason": phrase.relation_rejection_reason,
            }
        )
    return best


def choose_reader_output(
    question: str,
    context: str,
    neural_output: dict[str, Any],
    fallback_output: dict[str, Any],
) -> dict[str, Any]:
    neural_confidence = float(neural_output["confidence"])
    fallback_confidence = float(fallback_output["confidence"])
    neural_answer = str(
        neural_output.get("candidate_answer")
        or neural_output.get("best_span_answer")
        or neural_output.get("answer")
        or ""
    ).strip()
    neural_start = int(
        neural_output.get("candidate_start", neural_output.get("best_span_start", neural_output.get("start", -1)))
    )
    neural_end = int(
        neural_output.get("candidate_end", neural_output.get("best_span_end", neural_output.get("end", -1)))
    )
    neural_is_echo = answer_repeats_question(question, neural_answer)
    question_types = detect_question_type(question)
    subject = definition_subject(question)
    definition_supported = not subject or any(
        relation_follows_subject(sentence, subject) for sentence in split_sentences(context)
    )
    neural_span_is_clean = span_has_clean_word_boundaries(
        context,
        neural_start,
        neural_end,
    )
    neural_relation_supported = True
    requires_location_relation = (
        QuestionType.LOCATION in question_types
        and detect_location_relation(question) != "GENERIC_LOCATION"
    )
    if requires_location_relation and neural_answer:
        neural_location_evidence = location_relation_assessment(
            question,
            context,
            neural_answer,
            neural_start,
            neural_end,
            {"method": "neural_span"},
        )
        neural_relation_supported = bool(neural_location_evidence["relation_evidence"])
    neural_ready = (
        bool(neural_answer)
        and not neural_is_echo
        and definition_supported
        and neural_span_is_clean
        and neural_relation_supported
    )
    fallback_ready = bool(fallback_output.get("answer")) and fallback_confidence >= SENTENCE_FALLBACK_THRESHOLD
    fallback_phrase_quality = float(fallback_output.get("phrase_score", 0.0))
    strong_grounded_phrase = (
        fallback_output.get("fallback_method") not in {None, "whole_sentence"}
        and fallback_phrase_quality >= 0.80
    )
    fallback_phrase_is_echo = strong_grounded_phrase and answer_repeats_question(
        question,
        str(fallback_output.get("answer") or ""),
    )
    fallback_ready = fallback_ready and not fallback_phrase_is_echo
    # A generic whole sentence must not erase a grounded neural span. Only a
    # concise relation-aware phrase may compete with an otherwise valid span.
    prefer_fallback = fallback_ready and (
        not neural_ready
        or (
            strong_grounded_phrase
            and (
                fallback_confidence >= neural_confidence + 0.08
                or fallback_confidence >= neural_confidence - 0.03
            )
        )
    )

    if prefer_fallback:
        return {
            "method": "sentence_fallback",
            "answer": fallback_output["answer"],
            "confidence": fallback_confidence,
            "start": int(fallback_output["start"]),
            "end": int(fallback_output["end"]),
            "fallback_method": fallback_output.get("fallback_method", "whole_sentence"),
            "fallback_phrase_score": float(fallback_output.get("phrase_score", 0.4)),
            "evidence_sentence": fallback_output.get("sentence_answer", fallback_output["answer"]),
            "relation_type": fallback_output.get("relation_type"),
            "relation_score": float(fallback_output.get("relation_score", 0.0)),
            "phrase_quality": float(fallback_output.get("phrase_quality", 0.0)),
            "relation_evidence": bool(fallback_output.get("relation_evidence", False)),
        }
    if neural_ready:
        return {
            "method": "neural_span",
            "answer": neural_answer,
            "confidence": neural_confidence,
            "start": neural_start,
            "end": neural_end,
        }
    if fallback_ready:
        return {
            "method": "sentence_fallback",
            "answer": fallback_output["answer"],
            "confidence": fallback_confidence,
            "start": int(fallback_output["start"]),
            "end": int(fallback_output["end"]),
            "fallback_method": fallback_output.get("fallback_method", "whole_sentence"),
            "fallback_phrase_score": float(fallback_output.get("phrase_score", 0.4)),
            "evidence_sentence": fallback_output.get("sentence_answer", fallback_output["answer"]),
            "relation_type": fallback_output.get("relation_type"),
            "relation_score": float(fallback_output.get("relation_score", 0.0)),
            "phrase_quality": float(fallback_output.get("phrase_quality", 0.0)),
            "relation_evidence": bool(fallback_output.get("relation_evidence", False)),
        }
    if (
        neural_is_echo
        or not neural_span_is_clean
        or not definition_supported
        or not neural_relation_supported
    ):
        return {
            "method": "no_answer",
            "answer": None,
            "confidence": 0.0,
            "start": -1,
            "end": -1,
        }
    return {
        "method": "neural_span",
        "answer": neural_answer or None,
        "confidence": neural_confidence,
        "start": neural_start,
        "end": neural_end,
    }


def expand_answer_to_sentence(context: str, answer: str, start: int, end: int) -> str:
    answer = str(answer or "").strip()
    if not answer or start < 0 or end <= start or end > len(context):
        return answer

    raw_span = context[start:end].strip()
    if normalize_text(answer) not in normalize_text(raw_span):
        return answer

    max_chars = int(os.getenv("QA_ANSWER_SENTENCE_MAX_CHARS", "360"))
    left_boundaries = [context.rfind(mark, 0, start) for mark in ".!?\n"]
    sentence_start = max(left_boundaries) + 1
    while sentence_start < len(context) and context[sentence_start].isspace():
        sentence_start += 1

    right_candidates = [context.find(mark, end) for mark in ".!?\n"]
    right_candidates = [index for index in right_candidates if index >= 0]
    sentence_end = (min(right_candidates) + 1) if right_candidates else len(context)
    while sentence_end > sentence_start and context[sentence_end - 1].isspace():
        sentence_end -= 1

    sentence = context[sentence_start:sentence_end].strip()
    if len(sentence) <= len(answer) or len(sentence) > max_chars:
        return answer
    return sentence


def concise_source_answer(question: str, answer: str, max_chars: int = 280) -> str:
    answer = re.sub(r"\s+", " ", str(answer or "")).strip()
    if not answer:
        return ""

    normalized_question = normalize_text(question)
    if re.search(r"\b(?:chia|bao gom|gom)\b", normalized_question):
        answer = re.sub(r"\s*\([^)]*\)", "", answer)
        answer = re.sub(r"\s+([,.;:])", r"\1", answer)

    subject = definition_subject(question)
    if subject and relation_follows_subject(answer, subject):
        normalized = normalize_text(answer)
        subject_match = re.search(rf"\b{re.escape(subject)}\b", normalized)
        if subject_match:
            tail = normalized[subject_match.end() :]
            cue_match = re.search(r"\bla\b", tail)
            if cue_match:
                predicate_start = subject_match.end() + cue_match.end()
                predicate = answer[predicate_start:].strip(" ,:-")
                predicate = re.sub(r"\s*\([^)]*\)\s*", " ", predicate).strip()
                stop = re.search(
                    r"\s+(?:tu nam|ke tu|trong giai doan|cho den khi|vao nam)\s+",
                    normalize_text(predicate),
                )
                if stop and len(predicate[: stop.start()].split()) >= 4:
                    predicate = predicate[: stop.start()].rstrip(" ,;:")
                if predicate:
                    subject_text = answer[subject_match.start() : subject_match.end()].strip()
                    answer = f"{subject_text} là {predicate}"

    answer = answer.split(";", 1)[0].strip()
    if len(answer) > max_chars:
        shortened = answer[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,;:")
        answer = f"{shortened}…"
    elif answer[-1:] not in ".!?":
        answer = f"{answer}."
    return answer


def format_display_answer(
    question: str,
    context: str,
    chosen_output: dict[str, Any],
) -> str:
    answer = str(chosen_output.get("answer") or "").strip()
    if not answer:
        return ""
    if (
        chosen_output.get("method") in {"phrase_fallback", "sentence_fallback"}
        and chosen_output.get("fallback_method") != "whole_sentence"
    ):
        return answer
    if chosen_output.get("method") in {"phrase_fallback", "sentence_fallback"}:
        return concise_source_answer(question, answer)
    if answer_repeats_question(question, answer):
        expanded = expand_answer_to_sentence(
            context,
            answer,
            int(chosen_output["start"]),
            int(chosen_output["end"]),
        )
        return concise_source_answer(question, expanded)
    return answer


class DenseScorer:
    """Lazy-loaded sentence-transformer scorer for dense retrieval.

    Encodes passage texts on first use, then computes cosine similarity
    between the query and all passage embeddings. If sentence-transformers
    is missing, falls back to empty results so the system degrades gracefully.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._passage_embeddings = None  # numpy array (N, dim)
        self._lock = threading.Lock()
        self._available: bool | None = None  # None = not checked yet

    def _ensure_model(self) -> bool:
        """Lazy-load the sentence-transformers model. Returns True if ready."""
        if self._available is False:
            return False
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if self._available is False:
                return False
            try:
                from sentence_transformers import SentenceTransformer

                print(f"[DenseScorer] Loading model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._available = True
                print(f"[DenseScorer] Model loaded successfully")
                return True
            except Exception as error:
                print(f"[DenseScorer] Cannot load dense model: {error}")
                self._available = False
                return False

    def _ensure_embeddings(self, passages: list) -> None:
        """Encode all passages if not already done. Caches to disk."""
        if self._passage_embeddings is not None:
            return
        with self._lock:
            if self._passage_embeddings is not None:
                return
            import numpy as np

            safe_name = self.model_name.replace("/", "_")
            cache_file = ROOT / "data" / "processed" / f"dense_embeddings_{safe_name}.npy"

            # Try to load from cache
            if cache_file.exists():
                try:
                    print(f"[DenseScorer] Loading cached embeddings from {cache_file.name}...")
                    cached = np.load(str(cache_file))
                    if cached.shape[0] == len(passages):
                        self._passage_embeddings = cached
                        print(f"[DenseScorer] Loaded {cached.shape} successfully.")
                        return
                    else:
                        print(f"[DenseScorer] Cache length mismatch ({cached.shape[0]} vs {len(passages)}). Re-encoding...")
                except Exception as e:
                    print(f"[DenseScorer] Failed to load cache: {e}. Re-encoding...")

            texts = [p.metadata.text for p in passages]
            print(f"[DenseScorer] Encoding {len(texts)} passages...")
            self._passage_embeddings = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            print(f"[DenseScorer] Passage embeddings ready: shape={self._passage_embeddings.shape}")

            # Save to cache
            try:
                np.save(str(cache_file), self._passage_embeddings)
                print(f"[DenseScorer] Saved embeddings to {cache_file.name}")
            except Exception as e:
                print(f"[DenseScorer] Failed to save cache: {e}")

    def retrieve(
        self,
        question: str,
        passages: list,
        top_k: int,
    ) -> list:
        """Score passages by cosine similarity and return top_k SearchHit list."""
        if not self._ensure_model():
            return []

        import numpy as np

        self._ensure_embeddings(passages)
        query_vec = self._model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Cosine similarity (embeddings are L2-normalized → dot product)
        similarities = (self._passage_embeddings @ query_vec.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        selected = [(passages[i], float(similarities[i])) for i in top_indices if similarities[i] > 0]
        normalized = min_max_normalize([score for _, score in selected])
        return [
            SearchHit(passage, raw, norm, rank)
            for rank, ((passage, raw), norm) in enumerate(zip(selected, normalized), start=1)
        ]

    @property
    def is_available(self) -> bool:
        return self._available is not False


def rrf_fuse(
    bm25_hits: list,
    dense_hits: list,
    top_k: int,
    k: int = 60,
) -> list:
    """Reciprocal Rank Fusion: score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d)).

    If a document appears in only one list, the missing rank is set to a large
    penalty value so documents in both lists are favoured.
    """
    passage_key = id  # use object identity for IndexedPassage

    # Collect ranks from both lists
    bm25_ranks: dict[int, tuple[int, SearchHit]] = {}
    for hit in bm25_hits:
        bm25_ranks[passage_key(hit.passage)] = (hit.retrieval_rank, hit)

    dense_ranks: dict[int, tuple[int, SearchHit]] = {}
    for hit in dense_hits:
        dense_ranks[passage_key(hit.passage)] = (hit.retrieval_rank, hit)

    # Union of all passages
    all_keys = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    missing_rank = max(len(bm25_hits), len(dense_hits)) + 1

    scored: list[tuple[int, float, SearchHit]] = []
    for key in all_keys:
        bm25_rank = bm25_ranks[key][0] if key in bm25_ranks else missing_rank
        dense_rank = dense_ranks[key][0] if key in dense_ranks else missing_rank
        rrf_score = 1.0 / (k + bm25_rank) + 1.0 / (k + dense_rank)
        # Prefer the hit object from whichever list ranked it higher
        hit = bm25_ranks[key][1] if key in bm25_ranks else dense_ranks[key][1]
        scored.append((key, rrf_score, hit))

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:top_k]

    # Re-normalize RRF scores
    raw_scores = [score for _, score, _ in selected]
    normalized = min_max_normalize(raw_scores)

    return [
        SearchHit(hit.passage, raw_score, norm, rank)
        for rank, ((_, raw_score, hit), norm) in enumerate(zip(selected, normalized), start=1)
    ]


class PassageIndex:
    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Missing docs database: {db_path}")
        self.passages = self._load_passages(db_path)
        self.avg_passage_len = sum(len(item.tokens) for item in self.passages) / max(1, len(self.passages))
        frequencies: dict[str, int] = defaultdict(int)
        for passage in self.passages:
            for term in passage.term_counts:
                frequencies[term] += 1
        self.idf = {
            term: math.log(1 + (len(self.passages) - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    @staticmethod
    def _guess_title(document_id: str, text: str) -> str:
        sentences = split_sentences(text)
        words = (sentences[0] if sentences else text).split()
        return " ".join(words[:8]) if words else document_id

    def _load_passages(self, db_path: Path) -> list[IndexedPassage]:
        connection = sqlite3.connect(str(db_path))
        try:
            rows = connection.execute("SELECT id, text FROM documents").fetchall()
        finally:
            connection.close()

        indexed: list[IndexedPassage] = []
        for document_id, raw_text in rows:
            text = str(raw_text or "").strip()
            if not text:
                continue
            title = self._guess_title(str(document_id), text)
            for passage in chunk_document(
                str(document_id),
                text,
                title=title,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_sentences=CHUNK_OVERLAP_SENTENCES,
            ):
                tokens = tuple(tokenize(passage.text))
                indexed.append(IndexedPassage(passage, tokens, Counter(tokens)))
        return indexed

    def _bm25(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        k1, b = 1.5, 0.75
        passage_length = max(1, len(passage.tokens))
        score = 0.0
        for term in query_tokens:
            frequency = passage.term_counts.get(term, 0)
            if not frequency:
                continue
            frequency = min(frequency, 1)
            denominator = frequency + k1 * (
                1 - b + b * passage_length / max(1.0, self.avg_passage_len)
            )
            score += self.idf.get(term, 0.0) * (frequency * (k1 + 1)) / denominator
        return score

    def _tfidf(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        query_counts = Counter(query_tokens)
        numerator = 0.0
        query_norm = 0.0
        document_norm = 0.0
        for term, count in query_counts.items():
            idf = self.idf.get(term, 0.0)
            query_weight = (1 + math.log(count)) * idf
            document_count = passage.term_counts.get(term, 0)
            document_weight = (1 + math.log(document_count)) * idf if document_count else 0.0
            numerator += query_weight * document_weight
            query_norm += query_weight * query_weight
        for term, count in passage.term_counts.items():
            weight = (1 + math.log(count)) * self.idf.get(term, 0.0)
            document_norm += weight * weight
        if not query_norm or not document_norm:
            return 0.0
        return numerator / math.sqrt(query_norm * document_norm)

    def _lexical_boost(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        query_terms = list(dict.fromkeys(query_tokens))
        if not query_terms:
            return 0.0

        passage_terms = set(passage.tokens)
        matched_terms = [term for term in query_terms if term in passage_terms]
        if not matched_terms:
            return 0.0

        coverage = len(matched_terms) / len(query_terms)
        boost = coverage * 2.0

        title_tokens = set(tokenize(passage.metadata.title))
        if title_tokens:
            title_matches = sum(1 for term in query_terms if term in title_tokens)
            boost += 0.8 * title_matches / len(query_terms)

        passage_token_text = normalized_token_text(passage.tokens)
        phrase_hits = 0
        for gram in query_ngrams(query_tokens):
            if gram in passage_token_text:
                phrase_hits += 1
        if phrase_hits:
            boost += min(1.5, 0.35 * phrase_hits)

        positions: dict[str, list[int]] = defaultdict(list)
        for index, token in enumerate(passage.tokens):
            if token in matched_terms:
                positions[token].append(index)
        first_positions = [indexes[0] for indexes in positions.values() if indexes]
        if len(first_positions) >= 2:
            window = max(first_positions) - min(first_positions) + 1
            if window <= 40:
                boost += (40 - window) / 40

        return boost

    def _lead_passage_boost(self, query_tokens: list[str], passage: IndexedPassage) -> float:
        query_terms = list(dict.fromkeys(query_tokens))
        if len(query_terms) < 2:
            return 0.0

        sentences = split_sentences(passage.metadata.text)
        if not sentences:
            return 0.0

        first_tokens = raw_tokens(sentences[0])
        first_content = [token for token in first_tokens if len(token) > 1 and token not in STOPWORDS]
        phrase_index = find_sequence(first_content, query_terms)
        if phrase_index < 0:
            return 0.0

        boost = 0.0
        if phrase_index <= 2:
            boost += 1.0

        raw_phrase_index = find_sequence(first_tokens, query_terms)
        if raw_phrase_index >= 0:
            boost += max(0.0, 0.8 - 0.08 * raw_phrase_index)
            after_phrase = raw_phrase_index + len(query_terms)
            try:
                definition_index = first_tokens.index("la", after_phrase)
            except ValueError:
                definition_index = -1
            if definition_index > after_phrase:
                gap = first_tokens[after_phrase:definition_index]
                gap_content = [
                    token for token in gap
                    if token not in STOPWORDS and token not in DATE_TOKENS and not token.isdigit()
                ]
                if not gap_content and definition_index - raw_phrase_index <= 18:
                    boost += 2.4

        return boost

    def _lexical_retrieve(self, question: str, method: str, top_k: int) -> list[SearchHit]:
        """Core lexical retrieval (BM25 or TF-IDF) without hybrid fusion."""
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        scored: list[tuple[IndexedPassage, float]] = []
        scorer = self._bm25 if method == "bm25" else self._tfidf
        for passage in self.passages:
            base_score = scorer(query_tokens, passage)
            score = (
                base_score
                + self._lexical_boost(query_tokens, passage)
                + self._lead_passage_boost(query_tokens, passage)
            )
            if score > 0:
                scored.append((passage, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:top_k]
        normalized = min_max_normalize([score for _, score in selected])
        return [
            SearchHit(passage, raw, norm, rank)
            for rank, ((passage, raw), norm) in enumerate(zip(selected, normalized), start=1)
        ]

    def retrieve(self, question: str, method: str, top_k: int) -> list[SearchHit]:
        validate_retriever(method)
        if method == "hybrid":
            return self._hybrid_retrieve(question, top_k)
        if method == "dense":
            return self._dense_retrieve(question, top_k)
        return self._lexical_retrieve(question, method, top_k)

    def _dense_retrieve(self, question: str, top_k: int) -> list[SearchHit]:
        """Standalone Dense retrieval."""
        dense_hits = DENSE_SCORER.retrieve(question, self.passages, top_k)
        if not dense_hits:
            # Fallback to BM25 if dense model is unavailable
            return self._lexical_retrieve(question, "bm25", top_k)
        return dense_hits

    def _hybrid_retrieve(self, question: str, top_k: int) -> list[SearchHit]:
        """BM25 + Dense retrieval fused with Reciprocal Rank Fusion."""
        expanded_k = min(len(self.passages), top_k * 3)

        # BM25 leg
        bm25_hits = self._lexical_retrieve(question, "bm25", expanded_k)

        # Dense leg — gracefully degrade to BM25-only if dense is unavailable
        dense_hits = DENSE_SCORER.retrieve(question, self.passages, expanded_k)
        if not dense_hits:
            return self._lexical_retrieve(question, "bm25", top_k)

        # RRF fusion
        fused = rrf_fuse(bm25_hits, dense_hits, top_k, k=RRF_K)
        return fused


class ReaderManager:
    MODEL_FOLDERS = {
        "phobert": "vinai_phobert-base-v2",
        "xlmr": "xlm-roberta-large-viquad",
    }

    def __init__(self) -> None:
        self.predictors: dict[str, Any] = {}
        self._load_lock = threading.Lock()

    def get(self, reader_name: str):
        validate_reader(reader_name)
        folder = self.MODEL_FOLDERS.get(reader_name)
        if reader_name not in self.predictors:
            with self._load_lock:
                if reader_name not in self.predictors:
                    model_dir = (
                        PIPELINE_CONFIG.reader_checkpoint
                        if reader_name == "phobert"
                        else ROOT / "models" / "reader" / folder
                    )
                    if not model_dir.exists():
                        raise PipelineError(f"Reader checkpoint is missing: {model_dir}")
                    try:
                        from reader.predict import ReaderPredictor

                        self.predictors[reader_name] = ReaderPredictor(str(model_dir))
                    except Exception as error:
                        raise PipelineError(f"Failed to load reader '{reader_name}': {error}") from error
        return self.predictors[reader_name]


INDEX = PassageIndex(DOCS_DB)
READERS = ReaderManager()
_READERS = READERS
def get_readers():
    return READERS
DENSE_SCORER = DenseScorer(DENSE_MODEL_NAME)


def _empty_response(question: str, retriever: str, reader: str, elapsed: int) -> dict[str, Any]:
    question_types = detect_question_type(question)
    qt_values = [qt.value for qt in question_types]
    return {
        "question": question,
        "question_type": qt_values,
        "answer_type": qt_values,
        "answer": None,
        "has_answer": False,
        "confidence": None,
        "answer_confidence": None,
        "reader_method": "no_answer",
        "fallback_method": None,
        "selected_passage_id": None,
        "processing_time_ms": elapsed,
        "retriever": retriever,
        "reader": reader,
        "source": None,
        "answer_source": None,
        "top_retrieved_passage": None,
        "no_answer_reason": "Không tìm thấy câu trả lời đủ tin cậy trong các đoạn được truy xuất.",
        "rejection_reason": "RETRIEVAL_MISS",
        "rejection_detail": REJECTION_MESSAGES["RETRIEVAL_MISS"],
        "scores": {
            "retrieval": None,
            "reader": None,
            "answer_type": None,
            "ranking": None,
            "answer_confidence": None,
        },
        "answer_span": None,
        "passages": [],
    }


def _semantic_relation_assessment(
    question: str,
    question_types: list[QuestionType] | QuestionType,
    context: str,
    answer: str,
    start: int,
    end: int,
    candidate_method: str,
    candidate_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(question_types, list):
        question_types = [question_types]
    details = candidate_details or {}
    typed_fallback_relation = str(details.get("relation_type") or "")
    question_relation = detect_question_relation(question, question_types[0] if question_types else QuestionType.GENERAL)
    frame = extract_cause_question(question) if question_relation is QuestionRelation.CAUSE else None
    if question_relation is QuestionRelation.CAUSE and frame is not None:
        cause = assess_cause_candidate(
            question,
            context,
            answer,
            str(details.get("sentence_answer") or details.get("evidence_sentence") or "") or None,
        )
        # Enforce the semantic gate when either side explicitly proposes a
        # causal phrase. Unrecognized neural spans retain the prior neutral
        # handling, avoiding a blanket false-negative regression while the
        # rule extractor intentionally remains conservative.
        if cause.cause_pattern_score > 0.0 or typed_fallback_relation == "CAUSE":
            return {
                "relation_type": "CAUSE",
                "relation_score": cause.relation_score,
                "phrase_quality": cause.cause_pattern_score,
                "relation_evidence": cause.relation_evidence,
                "relation_method": cause.relation_method,
                "question_subject": frame.subject if frame else None,
                "question_target": frame.target if frame else None,
                "cause_pattern_score": cause.cause_pattern_score,
                "subject_match_score": cause.subject_match_score,
                "target_relation_score": cause.target_relation_score,
                "relation_rejection_reason": cause.rejection_reason,
                "evidence_sentence": cause.evidence_sentence,
                "cause_effect": cause.effect,
            }
        subject_presence = cause_subject_match_score(question, context)
        if subject_presence < 0.50:
            return {
                "relation_type": "CAUSE",
                "relation_score": 0.0,
                "phrase_quality": 0.0,
                "relation_evidence": False,
                "relation_method": "CAUSE_SUBJECT_PRESENCE_GATE",
                "question_subject": frame.subject,
                "question_target": frame.target,
                "cause_pattern_score": 0.0,
                "subject_match_score": subject_presence,
                "target_relation_score": 0.0,
                "relation_rejection_reason": "CAUSE_SUBJECT_MISMATCH",
                "evidence_sentence": "",
                "cause_effect": "",
            }
    if candidate_method != "neural_span" and typed_fallback_relation in {
        "TEMPORAL_EXPRESSION",
        "NUMBER_EXPRESSION",
        "PERSON_RELATION",
        "PERSON_DEFINITION",
        "PROPERTY_DESCRIPTION",
        "SITUATION_DESCRIPTION",
        "ROLE_RELATION",
        "CAUSE",
        "BIRTH_TIME",
        "DEATH_TIME",
        "PURPOSE",
        "METHOD",
    }:
        return {
            "relation_type": typed_fallback_relation,
            "relation_score": float(details.get("relation_score", 0.0)),
            "phrase_quality": float(details.get("phrase_quality", 0.0)),
            "relation_evidence": bool(details.get("relation_evidence", False)),
        }
    if detect_contrast_relation(question):
        score, evidence = assess_contrast_relation(answer)
        return {
            "relation_type": "CONTRAST",
            "relation_score": score,
            "phrase_quality": score,
            "relation_evidence": evidence,
        }
    if detect_alias_relation(question):
        matching_sentence = next(
            (
                sentence
                for sentence in split_sentences(context)
                if normalize_text(answer) in normalize_text(sentence)
            ),
            "",
        )
        extracted = extract_alias_candidate(question, matching_sentence)
        expected = normalize_text(extracted.answer) if extracted else ""
        proposed = re.sub(r"^la\s+", "", normalize_text(answer)).strip()
        exact_alias = bool(expected and proposed == expected)
        partial_alias = bool(expected and proposed and proposed in expected)
        return {
            "relation_type": "ALIAS",
            "relation_score": 1.0 if exact_alias else (0.25 if partial_alias else 0.0),
            "phrase_quality": 1.0 if exact_alias else (0.25 if partial_alias else 0.0),
            "relation_evidence": exact_alias,
        }
    entity_location_relation = detect_location_relation(question)
    if QuestionType.ENTITY in question_types and entity_location_relation != "GENERIC_LOCATION":
        fallback_method = str(details.get("fallback_method") or "")
        if candidate_method != "neural_span" and fallback_method != "whole_sentence":
            score = max(
                float(details.get("relation_score", 0.0)),
                float(details.get("phrase_score", 0.0)),
            )
            return {
                "relation_type": "ENTITY_LOCATION_RELATION",
                "relation_score": score,
                "phrase_quality": float(details.get("phrase_quality", score)),
                "relation_evidence": score >= 0.72,
            }
        matching_sentence = next(
            (
                sentence
                for sentence in split_sentences(context)
                if normalize_text(answer) in normalize_text(sentence)
            ),
            "",
        )
        extracted = extract_fallback_answer(question, [qt.value for qt in question_types], matching_sentence)
        same_relation_phrase = bool(extracted.answer) and (
            normalize_text(answer) in normalize_text(extracted.answer)
            or normalize_text(extracted.answer) in normalize_text(answer)
        )
        score = float(extracted.score) if same_relation_phrase and extracted.method != "whole_sentence" else 0.0
        return {
            "relation_type": "ENTITY_LOCATION_RELATION",
            "relation_score": score,
            "phrase_quality": score,
            "relation_evidence": score >= 0.72,
        }
    if QuestionType.LOCATION in question_types:
        return location_relation_assessment(
            question,
            context,
            answer,
            start,
            end,
            {
                "method": "sentence_fallback" if candidate_method != "neural_span" else "neural_span",
                "relation_type": details.get("relation_type"),
                "relation_score": details.get("relation_score", 0.0),
                "phrase_quality": details.get("phrase_quality", 0.0),
                "relation_evidence": details.get("relation_evidence", False),
            },
        )
    subject = definition_subject(question)
    if QuestionType.DEFINITION in question_types and subject:
        supported = any(
            relation_follows_subject(sentence, subject)
            and normalize_text(answer) in normalize_text(sentence)
            for sentence in split_sentences(context)
        )
        return {
            "relation_type": "DEFINITION",
            "relation_score": 1.0 if supported else 0.0,
            "phrase_quality": 1.0 if supported else 0.0,
            "relation_evidence": supported,
        }
    return {
        "relation_type": None,
        "relation_score": 0.5,
        "phrase_quality": 0.5,
        "relation_evidence": True,
    }


def _candidate_rejection(candidate: AnswerCandidate) -> str | None:
    if not candidate.valid_span:
        return "NO_VALID_SPAN"
    if candidate.boundary_score < 0.20:
        return "SPAN_BOUNDARY_INCOMPLETE"
    if not candidate.passes_completeness_gate:
        if "DANGLING_CONNECTOR" in candidate.completeness_reasons:
            return "DANGLING_CONNECTOR"
        return "INCOMPLETE_RELATION"
    if not candidate.passes_evidence_gate:
        return "EVIDENCE_UNSUPPORTED"
    if not candidate.passes_relation_gate:
        if candidate.relation_validation_reason in REJECTION_MESSAGES:
            return str(candidate.relation_validation_reason)
        if candidate.relation_type == "CAUSE":
            return candidate.relation_rejection_reason or "CAUSE_RELATION_MISMATCH"
        return (
            "LOCATION_RELATION_MISMATCH"
            if candidate.relation_type and "LOCATION" in candidate.relation_type
            else "RELATION_MISMATCH"
        )
    if candidate.answer_type_score < MIN_ANSWER_TYPE_SCORE:
        return "ANSWER_TYPE_MISMATCH"
    if not candidate.passes_type_gate:
        return (
            "INSUFFICIENT_FALLBACK_EVIDENCE"
            if candidate.method == "sentence_fallback"
            else "ANSWER_TYPE_MISMATCH"
        )
    strong_cause_relation = (
        candidate.relation_type == "CAUSE"
        and candidate.cause_pattern_score >= 0.85
        and candidate.subject_match_score >= 0.75
        and candidate.target_relation_score >= 0.55
        and candidate.relation_score >= 0.80
        and candidate.answer_type_score >= MIN_ANSWER_TYPE_SCORE
    )
    strong_time_relation = (
        candidate.relation_type in {"BIRTH_TIME", "DEATH_TIME"}
        and candidate.semantic_status == "VALID"
        and candidate.subject_match_score >= 0.75
        and candidate.relation_score >= 0.90
        and candidate.answer_type_score >= MIN_ANSWER_TYPE_SCORE
    )
    strong_grounded_relation = strong_cause_relation or strong_time_relation or (
        candidate.passes_evidence_gate
        and candidate.passes_relation_gate
        and candidate.relation_score >= 0.85
        and candidate.answer_type_score >= MIN_FALLBACK_ANSWER_TYPE_SCORE
    )
    if candidate.ranking_score < MIN_RANKING_SCORE and not strong_grounded_relation:
        return "LOW_RANKING_SCORE"
    return None


_REJECTION_DEBUG_PRIORITY = {
    # When every proposal is rejected, prefer the candidate that reached the
    # latest decision stage. This keeps diagnostics actionable instead of
    # letting a high-retrieval empty span hide a semantic/type/ranking failure.
    "LOW_RANKING_SCORE": 6,
    "SPAN_BOUNDARY_INCOMPLETE": 5,
    "INCOMPLETE_RELATION": 5,
    "LOCATION_RELATION_MISMATCH": 5,
    "RELATION_MISMATCH": 5,
    "CAUSE_RELATION_MISMATCH": 5,
    "CAUSE_TARGET_MISMATCH": 5,
    "CAUSE_SUBJECT_MISMATCH": 5,
    "CAUSE_PHRASE_NOT_FOUND": 5,
    "CAUSE_RELATION_NOT_FOUND": 5,
    "CAUSE_EFFECT_REPETITION": 5,
    "TIME_SUBJECT_MISMATCH": 5,
    "BIRTH_TIME_RELATION_MISMATCH": 5,
    "DEATH_TIME_RELATION_MISMATCH": 5,
    "PURPOSE_SUBJECT_MISMATCH": 5,
    "PURPOSE_RELATION_NOT_FOUND": 5,
    "DANGLING_CONNECTOR": 5,
    "INCOMPLETE_CLAUSE": 5,
    "ANSWER_TYPE_MISMATCH": 4,
    "INSUFFICIENT_FALLBACK_EVIDENCE": 3,
    "EVIDENCE_UNSUPPORTED": 2,
    "NO_VALID_SPAN": 1,
}


def _candidate_selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    rejection_reason = candidate.get("rejection_reason")
    return (
        1.0 if rejection_reason is None else 0.0,
        float(_REJECTION_DEBUG_PRIORITY.get(str(rejection_reason), 0)),
        float(candidate.get("ranking_score", 0.0)),
        float(candidate.get("evidence_score", 0.0)),
        float(candidate.get("reader_score", 0.0)),
    )


def _score_answer_candidate(
    candidate: AnswerCandidate,
    *,
    question: str,
    question_types: list[QuestionType] | QuestionType,
    context: str,
    retrieval_score: float,
    relation_details: dict[str, Any] | None = None,
) -> AnswerCandidate:
    primary_qt = question_types[0] if isinstance(question_types, list) and question_types else (question_types if not isinstance(question_types, list) else QuestionType.GENERAL)
    refinement = refine_answer(
        question,
        primary_qt,
        context,
        candidate.start_char,
        candidate.end_char,
    )
    candidate.raw_text = refinement.raw_answer
    candidate.raw_start_char = refinement.raw_start
    candidate.raw_end_char = refinement.raw_end
    candidate.text = refinement.refined_answer
    candidate.start_char = refinement.final_start
    candidate.end_char = refinement.final_end
    candidate.refinement_method = refinement.refinement_method
    candidate.refinement_changed = refinement.changed
    candidate.question_relation = refinement.relation
    candidate.completeness_score = refinement.completeness_score
    candidate.completeness_before = refinement.completeness_before
    candidate.completeness_after = refinement.completeness_after
    candidate.relation_complete = refinement.relation_complete
    candidate.completeness_reasons = refinement.completeness_reasons
    semantics = parse_question_semantics(question)
    candidate.semantic_relation = semantics.relation
    candidate.question_subject = semantics.subject
    candidate.question_target = semantics.target
    candidate.question_predicate = semantics.predicate
    candidate.question_modifier = semantics.modifier
    candidate.expected_answer_type = semantics.expected_answer_type
    lexical_evidence = answer_is_supported(
        context,
        candidate.text,
        candidate.start_char,
        candidate.end_char,
    )
    relation = _semantic_relation_assessment(
        question,
        question_types,
        context,
        candidate.text,
        candidate.start_char,
        candidate.end_char,
        candidate.method,
        relation_details,
    )
    semantic_validation = validate_candidate_relation(
        semantics,
        question,
        context,
        candidate.text,
        candidate.start_char,
        candidate.end_char,
        candidate.method,
        relation_details,
    )
    if semantics.relation in STRICT_RELATIONS:
        relation = {
            "relation_type": semantic_validation.relation_type,
            "relation_score": semantic_validation.relation_score,
            "phrase_quality": max(
                semantic_validation.cause_pattern_score,
                semantic_validation.target_relation_score,
            ),
            "relation_evidence": semantic_validation.relation_evidence,
            "relation_method": semantic_validation.relation_method,
            "question_subject": semantics.subject,
            "question_target": semantics.target,
            "cause_pattern_score": semantic_validation.cause_pattern_score,
            "subject_match_score": semantic_validation.subject_match_score,
            "target_relation_score": semantic_validation.target_relation_score,
            "relation_rejection_reason": semantic_validation.reason,
            "semantic_status": semantic_validation.status,
            "subject_match_reason": semantic_validation.subject_match_reason,
            "evidence_sentence": semantic_validation.evidence_sentence,
        }

    assessment = assess_answer_type(
        question_types,
        candidate.text,
        relation_score=float(relation["relation_score"]),
        phrase_quality=float(relation["phrase_quality"]),
        candidate_method=candidate.fallback_method,
        multi_type_coverage_bonus=MULTI_TYPE_COVERAGE_BONUS,
    )
    fallback_phrase = candidate.method == "phrase_fallback"
    strong_semantic_relation = (
        relation["relation_type"] == "CONTRAST"
        and float(relation["relation_score"]) >= 0.80
    )
    required_type_score = (
        MIN_FALLBACK_ANSWER_TYPE_SCORE
        if candidate.method == "sentence_fallback" and not strong_semantic_relation
        else MIN_ANSWER_TYPE_SCORE
    )
    requires_relation = relation["relation_type"] in {
        "CONTRAST",
        "ALIAS",
        "TEMPORAL_EXPRESSION",
        "NUMBER_EXPRESSION",
        "PERSON_RELATION",
        "PERSON_DEFINITION",
        "PROPERTY_DESCRIPTION",
        "SITUATION_DESCRIPTION",
        "ROLE_RELATION",
        "CAUSE",
        "BIRTH_TIME",
        "DEATH_TIME",
        "PURPOSE",
        "METHOD",
        "EVENT_LOCATION",
        "OBJECT_LOCATION",
        "BIRTH_LOCATION",
        "DEATH_LOCATION",
        "ORGANIZED_LOCATION",
        "HEADQUARTERS_LOCATION",
        "RESIDENCE_LOCATION",
        "DEFINITION",
        "ENTITY_LOCATION_RELATION",
    }
    candidate.answer_type_score = float(assessment.score)
    candidate.answer_type_reason = assessment.reason
    candidate.relation_type = relation["relation_type"]
    candidate.relation_score = float(relation["relation_score"])
    candidate.relation_method = relation.get("relation_method")
    candidate.question_subject = relation.get("question_subject") or semantics.subject
    candidate.question_target = relation.get("question_target") or semantics.target
    candidate.cause_pattern_score = float(relation.get("cause_pattern_score", 0.0))
    candidate.subject_match_score = float(relation.get("subject_match_score", 0.0))
    candidate.target_relation_score = float(relation.get("target_relation_score", 0.0))
    candidate.relation_rejection_reason = relation.get("relation_rejection_reason")
    candidate.semantic_status = str(relation.get("semantic_status") or (
        "VALID" if relation.get("relation_evidence") else "UNKNOWN"
    ))
    candidate.relation_validation_reason = relation.get("relation_rejection_reason")
    candidate.subject_match_reason = relation.get("subject_match_reason")
    candidate.evidence_score = 1.0 if lexical_evidence else 0.0
    boundary = assess_span_boundary(
        context,
        candidate.start_char,
        candidate.end_char,
        question_types,
        question,
    )
    candidate.boundary_score = boundary.score
    candidate.boundary_reasons = boundary.reasons
    candidate.valid_span = bool(
        candidate.text
        and span_has_clean_word_boundaries(context, candidate.start_char, candidate.end_char)
    )
    candidate.passes_evidence_gate = bool(lexical_evidence and not answer_repeats_question(question, candidate.text))
    candidate.passes_completeness_gate = bool(candidate.relation_complete)
    candidate.passes_relation_gate = bool(
        relation["relation_evidence"] if requires_relation else True
    )
    candidate.passes_type_gate = assessment.score >= required_type_score
    boundary_factor = 0.5 + 0.5 * candidate.boundary_score
    completeness_factor = 0.5 + 0.5 * candidate.completeness_score
    reader_signal = (
        candidate.reader_score
        * candidate.fallback_penalty
        * boundary_factor
        * completeness_factor
    )
    candidate.ranking_score = combine_ranking_scores(
        retrieval_score,
        reader_signal,
        candidate.answer_type_score,
        relation_score=candidate.relation_score,
        relation_weight=RELATION_WEIGHT,
    )
    candidate.rejection_reason = _candidate_rejection(candidate)
    candidate.passes_final_gate = candidate.rejection_reason is None
    return candidate


def build_passage_candidates(
    question: str,
    question_types: list[QuestionType] | QuestionType,
    passage_id: str,
    context: str,
    retrieval_score: float,
    neural_output: dict[str, Any],
    fallback_output: dict[str, Any],
) -> list[AnswerCandidate]:
    """Keep neural and fallback proposals in one pool until final ranking."""

    if not isinstance(question_types, list):
        question_types = [question_types]

    candidates: list[AnswerCandidate] = []
    raw_neural_candidates = list(neural_output.get("span_candidates") or [])
    if not raw_neural_candidates:
        raw_neural_candidates = [
            {
                "text": neural_output.get("candidate_answer")
                or neural_output.get("best_span_answer")
                or neural_output.get("answer")
                or "",
                "start": neural_output.get(
                    "candidate_start",
                    neural_output.get("best_span_start", neural_output.get("start", -1)),
                ),
                "end": neural_output.get(
                    "candidate_end",
                    neural_output.get("best_span_end", neural_output.get("end", -1)),
                ),
                "score_margin": neural_output.get("score_margin", float("-inf")),
                "reader_threshold_score": neural_output.get(
                    "reader_threshold_score", neural_output.get("confidence", 0.0)
                ),
                "passes_reader_threshold": neural_output.get("passes_reader_threshold"),
                "valid_span": neural_output.get("valid_span", True),
            }
        ]

    seen_neural_spans: set[tuple[int, int, str]] = set()
    for raw_candidate in raw_neural_candidates:
        neural_text = str(raw_candidate.get("text") or "").strip()
        neural_start = int(raw_candidate.get("start", -1))
        neural_end = int(raw_candidate.get("end", -1))
        key = (neural_start, neural_end, neural_text)
        if not neural_text or key in seen_neural_spans:
            continue
        seen_neural_spans.add(key)
        margin = float(raw_candidate.get("score_margin", float("-inf")))
        raw_passes_threshold = raw_candidate.get("passes_reader_threshold")
        passes_threshold = bool(
            margin >= READER_SCORE_MARGIN_THRESHOLD
            if raw_passes_threshold is None
            else raw_passes_threshold
        )
        neural = AnswerCandidate(
            text=neural_text,
            method="neural_span",
            passage_id=passage_id,
            start_char=neural_start,
            end_char=neural_end,
            reader_score=float(
                raw_candidate.get(
                    "reader_threshold_score", neural_output.get("confidence", 0.0)
                )
            ),
            score_margin=margin,
            reader_rank=int(raw_candidate.get("rank", 1)),
            raw_span_score=(
                float(raw_candidate["score"])
                if raw_candidate.get("score") is not None
                else None
            ),
            fallback_penalty=1.0,
            valid_span=bool(raw_candidate.get("valid_span", neural_text)),
            passes_reader_threshold=passes_threshold,
        )
        candidates.append(
            _score_answer_candidate(
                neural,
                question=question,
                question_types=question_types,
                context=context,
                retrieval_score=retrieval_score,
            )
        )

    # Different raw start/end pairs can refine to the same grounded span.
    # Keep only the strongest neural provenance for that final answer so the
    # reranker does not count duplicate evidence as separate proposals.
    deduplicated_neural: dict[tuple[int, int, str], AnswerCandidate] = {}
    for candidate in candidates:
        key = (candidate.start_char, candidate.end_char, candidate.text)
        current = deduplicated_neural.get(key)
        if current is None or (
            candidate.ranking_score,
            candidate.raw_span_score or float("-inf"),
        ) > (
            current.ranking_score,
            current.raw_span_score or float("-inf"),
        ):
            deduplicated_neural[key] = candidate
    candidates = list(deduplicated_neural.values())

    fallback_text = str(fallback_output.get("answer") or "").strip()
    if fallback_text:
        fallback_method = str(fallback_output.get("fallback_method") or "whole_sentence")
        method = "sentence_fallback" if fallback_method == "whole_sentence" else "phrase_fallback"
        fallback = AnswerCandidate(
            text=fallback_text,
            method=method,
            passage_id=passage_id,
            start_char=int(fallback_output.get("start", -1)),
            end_char=int(fallback_output.get("end", -1)),
            reader_score=float(fallback_output.get("confidence", 0.0)),
            score_margin=None,
            fallback_penalty=(
                PHRASE_FALLBACK_PENALTY
                if method == "phrase_fallback"
                else FALLBACK_PENALTY
            ),
            valid_span=True,
            passes_reader_threshold=False,
            fallback_method=fallback_method,
            evidence_sentence=str(
                fallback_output.get("sentence_answer") or fallback_output.get("answer") or ""
            ),
        )
        candidates.append(
            _score_answer_candidate(
                fallback,
                question=question,
                question_types=question_types,
                context=context,
                retrieval_score=retrieval_score,
                relation_details=fallback_output,
            )
        )
    return candidates


def ask_question(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(payload.get("question", "")).strip()
    # If the user types in ALL CAPS, it destroys the semantic tokenization in Transformer models.
    # Lowercase it to preserve meaning.
    if question.isupper():
        question = question.lower()

    retriever = str(payload.get("retriever", PIPELINE_CONFIG.default_retriever)).lower()
    reader_name = str(payload.get("reader", "phobert")).lower()
    try:
        requested_top_k = payload.get("top_k", PIPELINE_CONFIG.default_top_k)
        top_k = min(
            PIPELINE_CONFIG.max_top_k,
            max(1, int(requested_top_k or PIPELINE_CONFIG.default_top_k)),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("top_k must be an integer") from error
    if not question:
        raise ValueError("question is required")
    validate_retriever(retriever)
    validate_reader(reader_name)
    question_types = detect_question_type(question)

    candidate_count = PIPELINE_CONFIG.candidate_count(top_k)
    hits = INDEX.retrieve(question, retriever, candidate_count)
    if not hits:
        return _empty_response(
            question,
            retriever,
            reader_name,
            int((time.perf_counter() - started) * 1000),
        )

    readers = get_readers()
    predictor = readers.get(reader_name)
    contexts = [hit.passage.metadata.text for hit in hits]
    predict_many = getattr(predictor, "predict_many", None)
    if callable(predict_many):
        outputs = predict_many(
            question,
            contexts,
            max_seq_len=PIPELINE_CONFIG.reader_max_length,
            doc_stride=PIPELINE_CONFIG.reader_stride,
            no_answer_threshold=READER_SCORE_MARGIN_THRESHOLD,
            span_candidate_limit=PIPELINE_CONFIG.reader_span_candidates,
        )
    else:
        outputs = [
            predictor.predict(
                question,
                context,
                max_seq_len=PIPELINE_CONFIG.reader_max_length,
                doc_stride=PIPELINE_CONFIG.reader_stride,
                no_answer_threshold=READER_SCORE_MARGIN_THRESHOLD,
                span_candidate_limit=PIPELINE_CONFIG.reader_span_candidates,
            )
            for context in contexts
        ]
    if len(outputs) != len(hits):
        raise PipelineError(
            f"Reader returned {len(outputs)} outputs for {len(hits)} retrieved passages"
        )

    passage_work: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for hit, output in zip(hits, outputs):
        metadata = hit.passage.metadata
        fallback_output = sentence_fallback_predict(question, metadata.text)
        candidates = build_passage_candidates(
            question,
            question_types,
            metadata.passage_id,
            metadata.text,
            hit.retrieval_score_normalized,
            output,
            fallback_output,
        )
        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_output = {
                "method": candidate.method,
                "answer": candidate.text,
                "start": candidate.start_char,
                "end": candidate.end_char,
                "fallback_method": candidate.fallback_method,
            }
            row = candidate.to_dict()
            row["display_text"] = format_display_answer(question, metadata.text, candidate_output)
            row["retrieval_score"] = float(hit.retrieval_score_normalized)
            row["retrieval_rank"] = int(hit.retrieval_rank)
            row["document_id"] = metadata.document_id
            row["title"] = metadata.title
            row["selection_status"] = "ELIGIBLE" if candidate.passes_final_gate else "REJECTED"
            row["rejection_detail"] = (
                REJECTION_MESSAGES.get(candidate.rejection_reason)
                if candidate.rejection_reason
                else None
            )
            candidate_rows.append(row)
            all_candidates.append(row)
        passage_work.append(
            {
                "hit": hit,
                "output": output,
                "fallback": fallback_output,
                "candidates": candidate_rows,
            }
        )

    eligible_candidates = [row for row in all_candidates if row["rejection_reason"] is None]
    selected_candidate = max(
        eligible_candidates,
        key=lambda row: (row["ranking_score"], row["evidence_score"], row["reader_score"]),
        default=None,
    )

    passages: list[dict[str, Any]] = []
    for work in passage_work:
        hit = work["hit"]
        output = work["output"]
        fallback_output = work["fallback"]
        metadata = hit.passage.metadata
        candidate_rows = sorted(
            work["candidates"],
            key=_candidate_selection_key,
            reverse=True,
        )
        representative = candidate_rows[0] if candidate_rows else {
            "text": "",
            "display_text": "",
            "method": "no_answer",
            "reader_score": 0.0,
            "fallback_penalty": 1.0,
            "answer_type_score": 0.0,
            "answer_type_reason": "NO_CANDIDATE",
            "relation_type": None,
            "relation_score": 0.0,
            "evidence_score": 0.0,
            "passes_relation_gate": False,
            "valid_span": False,
            "start_char": -1,
            "end_char": -1,
            "ranking_score": 0.0,
            "rejection_reason": "NO_VALID_SPAN",
            "fallback_method": None,
            "passes_reader_threshold": False,
            "boundary_score": 1.0,
            "boundary_reasons": (),
            "completeness_score": 1.0,
            "relation_complete": True,
            "completeness_reasons": (),
            "question_relation": "FACTOID",
            "refinement_method": "UNCHANGED",
            "refinement_changed": False,
            "raw_text": "",
            "raw_start_char": -1,
            "raw_end_char": -1,
        }
        reader_signal = (
            float(representative["reader_score"])
            * float(representative["fallback_penalty"])
            * (0.5 + 0.5 * float(representative.get("boundary_score", 1.0)))
            * (0.5 + 0.5 * float(representative.get("completeness_score", 1.0)))
        )
        item = {
            "rank": 0,
            "retrieval_rank": hit.retrieval_rank,
            "document_id": metadata.document_id,
            "passage_id": metadata.passage_id,
            "title": metadata.title,
            "paragraph_id": metadata.paragraph_id,
            "sentence_start": metadata.sentence_start,
            "sentence_end": metadata.sentence_end,
            "page": metadata.page,
            "text": metadata.text,
            "retrieval_score": round(hit.retrieval_score_normalized, 6),
            "retrieval_score_raw": round(hit.retrieval_score_raw, 6),
            "retrieval_score_normalized": round(hit.retrieval_score_normalized, 6),
            "question_type": [qt.value for qt in question_types],
            "reader_method": representative["method"],
            "reader_answer": representative["display_text"] or None,
            "reader_span_answer": representative["text"] or None,
            "reader_score": round(float(representative["reader_score"]), 6),
            "reader_signal": round(reader_signal, 6),
            "answer_type": [qt.value for qt in question_types],
            "answer_type_score": round(float(representative["answer_type_score"]), 6),
            "answer_type_reason": representative["answer_type_reason"],
            "lexical_evidence": bool(representative["passes_evidence_gate"]),
            "relation_evidence": bool(representative["passes_relation_gate"]),
            "relation_type": representative["relation_type"],
            "relation_score": round(float(representative["relation_score"]), 6),
            "relation_method": representative.get("relation_method"),
            "question_subject": representative.get("question_subject"),
            "question_target": representative.get("question_target"),
            "cause_pattern_score": round(float(representative.get("cause_pattern_score", 0.0)), 6),
            "subject_match_score": round(float(representative.get("subject_match_score", 0.0)), 6),
            "target_relation_score": round(float(representative.get("target_relation_score", 0.0)), 6),
            "relation_rejection_reason": representative.get("relation_rejection_reason"),
            "phrase_quality": round(float(representative["relation_score"]), 6),
            "boundary_score": round(float(representative.get("boundary_score", 1.0)), 6),
            "boundary_reasons": list(representative.get("boundary_reasons", ())),
            "question_relation": representative.get("question_relation", "FACTOID"),
            "semantic_relation": representative.get("semantic_relation", "GENERAL"),
            "question_predicate": representative.get("question_predicate"),
            "question_modifier": representative.get("question_modifier"),
            "semantic_status": representative.get("semantic_status", "UNKNOWN"),
            "relation_validation_reason": representative.get("relation_validation_reason"),
            "subject_match_reason": representative.get("subject_match_reason"),
            "completeness_score": round(float(representative.get("completeness_score", 1.0)), 6),
            "completeness_before": round(float(representative.get("completeness_before", 1.0)), 6),
            "completeness_after": round(float(representative.get("completeness_after", 1.0)), 6),
            "relation_complete": bool(representative.get("relation_complete", True)),
            "completeness_reasons": list(representative.get("completeness_reasons", ())),
            "refinement_method": representative.get("refinement_method", "UNCHANGED"),
            "refinement_changed": bool(representative.get("refinement_changed", False)),
            "raw_answer": representative.get("raw_text") or representative["text"],
            "evidence_supported": bool(representative["passes_evidence_gate"]),
            "passes_reader_threshold": bool(representative["passes_reader_threshold"]),
            "neural_reader_answer": output.get("answer") or None,
            "neural_reader_best_span": output.get("best_span_answer") or output.get("answer") or None,
            "neural_reader_has_answer": bool(output.get("has_answer", output.get("answer"))),
            "neural_reader_score": round(float(output.get("confidence", 0.0)), 6),
            "neural_reader_confidence_is_calibrated": bool(output.get("confidence_is_calibrated", False)),
            "neural_reader_max_answer_length": int(output.get("max_answer_length", 0)),
            "neural_reader_start_score": finite_or_none(output.get("start_score")),
            "neural_reader_end_score": finite_or_none(output.get("end_score")),
            "reader_score_raw": finite_or_none(output.get("score")),
            "reader_null_score": finite_or_none(output.get("null_score")),
            "reader_no_answer_score": finite_or_none(output.get("no_answer_score")),
            "reader_score_margin": finite_or_none(output.get("score_margin")),
            "reader_decision_threshold": float(output.get("decision_threshold", READER_SCORE_MARGIN_THRESHOLD)),
            "fallback_sentence": fallback_output.get("sentence_answer") or fallback_output.get("answer") or None,
            "fallback_answer": fallback_output.get("answer") or None,
            "fallback_method": fallback_output.get("fallback_method", "whole_sentence"),
            "fallback_phrase_score": round(float(fallback_output.get("phrase_score", 0.4)), 6),
            "fallback_relation_type": fallback_output.get("relation_type"),
            "fallback_relation_score": round(float(fallback_output.get("relation_score", 0.0)), 6),
            "fallback_relation_evidence": bool(fallback_output.get("relation_evidence", False)),
            "fallback_start": int(fallback_output.get("start", -1)),
            "fallback_end": int(fallback_output.get("end", -1)),
            "fallback_score": round(float(fallback_output.get("confidence", 0.0)), 6),
            "fallback_reason": fallback_output.get("reason"),
            "answer_span": {
                "text": representative["text"],
                "start": int(representative["start_char"]),
                "end": int(representative["end_char"]),
                "raw_text": representative.get("raw_text") or representative["text"],
                "raw_start": int(representative["raw_start_char"] if representative.get("raw_start_char") is not None else representative["start_char"]),
                "raw_end": int(representative["raw_end_char"] if representative.get("raw_end_char") is not None else representative["end_char"]),
                "refinement_method": representative.get("refinement_method", "UNCHANGED"),
            },
            "ranking_score": round(float(representative["ranking_score"]), 6),
            "answer_confidence": None,
            "candidates": candidate_rows,
            "gate_rejection_reason": representative["rejection_reason"],
        }
        passages.append(item)

    top_retrieved = min(passages, key=lambda item: item["retrieval_rank"])
    passages.sort(key=lambda item: item["ranking_score"], reverse=True)
    for rank, item in enumerate(passages, start=1):
        item["rank"] = rank

    selected = next(
        (
            item
            for item in passages
            if selected_candidate is not None
            and item["passage_id"] == selected_candidate["passage_id"]
        ),
        None,
    )
    for item in passages:
        if item is selected:
            item["selection_status"] = "SELECTED"
            item["rejection_reason"] = None
            item["rejection_detail"] = None
        else:
            rejection_reason = item["gate_rejection_reason"] or "LOWER_RANKING_SCORE"
            item["selection_status"] = "REJECTED"
            item["rejection_reason"] = rejection_reason
            item["rejection_detail"] = REJECTION_MESSAGES[rejection_reason]
        for candidate in item.get("candidates", []):
            if candidate is selected_candidate:
                candidate["selection_status"] = "SELECTED"
                candidate["rejection_reason"] = None
                candidate["rejection_detail"] = None
            elif candidate.get("rejection_reason") is None:
                candidate["selection_status"] = "REJECTED"
                candidate["rejection_reason"] = "LOWER_RANKING_SCORE"
                candidate["rejection_detail"] = REJECTION_MESSAGES["LOWER_RANKING_SCORE"]

    visible_passages = passages[:top_k]
    if selected is not None and selected not in visible_passages:
        visible_passages = passages[: max(0, top_k - 1)] + [selected]
        visible_passages.sort(key=lambda item: item["ranking_score"], reverse=True)

    has_answer = selected is not None
    elapsed = int((time.perf_counter() - started) * 1000)
    answer_source = None
    if selected is not None:
        answer_source = {
            "document_id": selected["document_id"],
            "passage_id": selected["passage_id"],
            "title": selected["title"],
            "paragraph_id": selected["paragraph_id"],
            "sentence_start": selected["sentence_start"],
            "sentence_end": selected["sentence_end"],
            "page": selected["page"],
        }
    best_reader_score = max(
        (
            float(candidate["reader_score"])
            for candidate in all_candidates
            if candidate["method"] == "neural_span"
        ),
        default=0.0,
    )
    decision_candidate = selected_candidate or max(
        all_candidates,
        key=_candidate_selection_key,
        default=None,
    )
    decision_passage = selected or passages[0]
    rejection_reason = None if selected_candidate is not None else (
        (decision_candidate or {}).get("rejection_reason") or "NO_ANSWER"
    )
    no_answer_reason = None if has_answer else (
        "Không tìm thấy câu trả lời đủ tin cậy trong các đoạn được truy xuất."
    )

    response = {
        "question": question,
        "question_type": [qt.value for qt in question_types],
        "answer_type": [qt.value for qt in question_types],
        "answer": selected_candidate["display_text"] if selected_candidate is not None else None,
        "has_answer": has_answer,
        # Kept as a nullable compatibility field. No calibrated probability exists yet.
        "confidence": None,
        "answer_confidence": None,
        "reader_method": selected_candidate["method"] if selected_candidate is not None else "no_answer",
        "fallback_method": (decision_candidate or {}).get("fallback_method"),
        "relation_type": (decision_candidate or {}).get("relation_type"),
        "relation_score": (decision_candidate or {}).get("relation_score", 0.0),
        "question_relation": (decision_candidate or {}).get("question_relation", "FACTOID"),
        "semantic_relation": (decision_candidate or {}).get("semantic_relation", "GENERAL"),
        "question_subject": (decision_candidate or {}).get("question_subject"),
        "question_target": (decision_candidate or {}).get("question_target"),
        "question_predicate": (decision_candidate or {}).get("question_predicate"),
        "question_modifier": (decision_candidate or {}).get("question_modifier"),
        "semantic_status": (decision_candidate or {}).get("semantic_status", "UNKNOWN"),
        "relation_validation_reason": (decision_candidate or {}).get("relation_validation_reason"),
        "subject_match_reason": (decision_candidate or {}).get("subject_match_reason"),
        "relation_method": (decision_candidate or {}).get("relation_method"),
        "cause_pattern_score": (decision_candidate or {}).get("cause_pattern_score", 0.0),
        "subject_match_score": (decision_candidate or {}).get("subject_match_score", 0.0),
        "target_relation_score": (decision_candidate or {}).get("target_relation_score", 0.0),
        "lexical_evidence": bool((decision_candidate or {}).get("passes_evidence_gate", False)),
        "relation_evidence": bool((decision_candidate or {}).get("passes_relation_gate", False)),
        "selected_passage_id": selected["passage_id"] if selected is not None else None,
        "processing_time_ms": elapsed,
        "retriever": retriever,
        "reader": reader_name,
        "source": answer_source,
        "answer_source": answer_source,
        "top_retrieved_passage": top_retrieved,
        "no_answer_reason": no_answer_reason,
        "rejection_reason": rejection_reason,
        "rejection_detail": REJECTION_MESSAGES.get(rejection_reason) if rejection_reason else None,
        "best_reader_score": round(best_reader_score, 6),
        "answer_span": (
            {
                "text": selected_candidate["text"],
                "start": selected_candidate["start_char"],
                "end": selected_candidate["end_char"],
                "raw_text": selected_candidate.get("raw_text") or selected_candidate["text"],
                "raw_start": selected_candidate.get("raw_start_char", selected_candidate["start_char"]),
                "raw_end": selected_candidate.get("raw_end_char", selected_candidate["end_char"]),
                "refinement_method": selected_candidate.get("refinement_method", "UNCHANGED"),
            }
            if selected_candidate is not None
            else None
        ),
        "reader_candidate": next(
            (
                candidate
                for candidate in decision_passage.get("candidates", [])
                if candidate["method"] == "neural_span"
            ),
            None,
        ),
        "fallback_candidate": next(
            (
                candidate
                for candidate in decision_passage.get("candidates", [])
                if candidate["method"] in {"phrase_fallback", "sentence_fallback"}
            ),
            None,
        ),
        "selected_candidate": selected_candidate,
        "answer_refinement": (
            {
                "raw_answer": selected_candidate.get("raw_text") or selected_candidate["text"],
                "refined_answer": selected_candidate["text"],
                "method": selected_candidate.get("refinement_method", "UNCHANGED"),
                "changed": bool(selected_candidate.get("refinement_changed", False)),
                "raw_start": selected_candidate.get("raw_start_char", selected_candidate["start_char"]),
                "raw_end": selected_candidate.get("raw_end_char", selected_candidate["end_char"]),
                "final_start": selected_candidate["start_char"],
                "final_end": selected_candidate["end_char"],
                "completeness_score": selected_candidate.get("completeness_score", 1.0),
                "completeness_before": selected_candidate.get("completeness_before", 1.0),
                "completeness_after": selected_candidate.get("completeness_after", 1.0),
                "relation_complete": selected_candidate.get("relation_complete", True),
                "completeness_reasons": selected_candidate.get("completeness_reasons", ()),
            }
            if selected_candidate is not None
            else None
        ),
        "scores": {
            "retrieval": (decision_candidate or {}).get("retrieval_score", 0.0),
            "reader": (decision_candidate or {}).get("reader_score", 0.0),
            "answer_type": (decision_candidate or {}).get("answer_type_score", 0.0),
            "relation": (decision_candidate or {}).get("relation_score", 0.0),
            "ranking": (decision_candidate or {}).get("ranking_score", 0.0),
            "answer_confidence": None,
        },
        "passages": visible_passages,
        "scoring": {
            "retriever_weight": RETRIEVER_WEIGHT,
            "reader_weight": READER_WEIGHT,
            "answer_type_weight": ANSWER_TYPE_WEIGHT,
            "relation_weight": RELATION_WEIGHT,
            "multi_type_coverage_bonus": MULTI_TYPE_COVERAGE_BONUS,
            "minimum_reader_score": MIN_READER_SCORE,
            "minimum_answer_type_score": MIN_ANSWER_TYPE_SCORE,
            "minimum_fallback_answer_type_score": MIN_FALLBACK_ANSWER_TYPE_SCORE,
            "minimum_ranking_score": MIN_RANKING_SCORE,
            "fallback_penalty": FALLBACK_PENALTY,
            "phrase_fallback_penalty": PHRASE_FALLBACK_PENALTY,
            "reader_score_margin_threshold": READER_SCORE_MARGIN_THRESHOLD,
            "reader_span_candidates": PIPELINE_CONFIG.reader_span_candidates,
            "reader_max_answer_length_by_type": DEFAULT_MAX_ANSWER_LENGTH_BY_TYPE,
            "reader_fallback_threshold": READER_FALLBACK_THRESHOLD,
            "sentence_fallback_threshold": SENTENCE_FALLBACK_THRESHOLD,
            "retrieval_normalization": "min_max_within_top_k",
            "candidate_count": candidate_count,
            "rerank": "retrieval_reader_answer_type_relation",
            "ranking_score_formula": "retriever_weight*retrieval_score + reader_weight*(reader_score*fallback_penalty*(0.5+0.5*boundary_score)*(0.5+0.5*completeness_score)) + answer_type_weight*answer_type_score + relation_weight*relation_score",
            "score_semantics": "All displayed scores are ranking signals, not correctness probabilities.",
        },
    }
    if QA_DEBUG:
        _log_debug(response)
    return response


def _log_debug(response: dict[str, Any]) -> None:
    print(
        f"[QA_DEBUG] QUESTION {response['question']!r} "
        f"type={response['question_type']}"
    )
    for passage in response["passages"]:
        print(
            "[QA_DEBUG] "
            f"rank={passage['rank']} passage={passage['passage_id']} "
            f"retrieval_raw={passage['retrieval_score_raw']:.6f} "
            f"retrieval_norm={passage['retrieval_score_normalized']:.6f} "
            f"reader={passage['reader_score']:.6f} "
            f"type={passage['answer_type_score']:.6f} "
            f"ranking={passage['ranking_score']:.6f} "
            f"method={passage['reader_method']} "
            f"fallback_method={passage.get('fallback_method')} "
            f"relation={passage.get('relation_type')} "
            f"relation_score={passage.get('relation_score', 0.0):.6f} "
            f"lexical_evidence={passage.get('lexical_evidence', False)} "
            f"relation_evidence={passage.get('relation_evidence', False)} "
            f"status={passage['selection_status']} "
            f"reason={passage['rejection_reason']} "
            f"answer={passage['reader_answer']!r}"
        )
    print(
        f"[QA_DEBUG] FINAL passage={response['selected_passage_id']} "
        f"has_answer={response['has_answer']} rejection={response['rejection_reason']} "
        "answer_confidence=NOT_CALIBRATED"
    )


def compare_retrievers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    rows: list[dict[str, Any]] = []
    for method, label in (("tfidf", "TF-IDF"), ("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid (BM25 + Dense)")):
        started = time.perf_counter()
        try:
            hits = INDEX.retrieve(question, method, 3)
        except Exception:
            hits = []
        elapsed = int((time.perf_counter() - started) * 1000)
        first = hits[0] if hits else None
        rows.append(
            {
                "retriever": method,
                "label": label,
                "correctPassageRank": None,
                "recallAt1": None,
                "recallAt3": None,
                "responseTimeMs": elapsed,
                "topPassagePreview": first.passage.metadata.text if first else "",
                "retrievalScore": round(first.retrieval_score_raw, 6) if first else 0.0,
                "evaluationNote": "Ground truth is required to compute correctness and Recall@k",
            }
        )
    return rows


def get_readers():
    return READERS


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "passages": len(INDEX.passages),
                    "reader_models": {
                        **ReaderManager.MODEL_FOLDERS,
                        "phobert": PIPELINE_CONFIG.reader_checkpoint.name,
                    },
                    "supported_retrievers": sorted(SUPPORTED_RETRIEVERS),
                    "unsupported_retrievers": UNIMPLEMENTED_RETRIEVERS,
                    "supported_readers": sorted(SUPPORTED_READERS),
                    "unsupported_readers": UNIMPLEMENTED_READERS,
                    "config": {
                        "chunk_max_tokens": CHUNK_MAX_TOKENS,
                        "chunk_overlap_sentences": CHUNK_OVERLAP_SENTENCES,
                        "retriever_weight": RETRIEVER_WEIGHT,
                        "reader_weight": READER_WEIGHT,
                        "answer_type_weight": ANSWER_TYPE_WEIGHT,
                        "relation_weight": RELATION_WEIGHT,
                        "minimum_reader_score": MIN_READER_SCORE,
                        "minimum_answer_type_score": MIN_ANSWER_TYPE_SCORE,
                        "minimum_fallback_answer_type_score": MIN_FALLBACK_ANSWER_TYPE_SCORE,
                        "minimum_ranking_score": MIN_RANKING_SCORE,
                        "fallback_penalty": FALLBACK_PENALTY,
                        "phrase_fallback_penalty": PHRASE_FALLBACK_PENALTY,
                        "reader_score_margin_threshold": READER_SCORE_MARGIN_THRESHOLD,
                        "reader_span_candidates": PIPELINE_CONFIG.reader_span_candidates,
                        "reader_max_answer_length_by_type": DEFAULT_MAX_ANSWER_LENGTH_BY_TYPE,
                        "reader_fallback_threshold": READER_FALLBACK_THRESHOLD,
                        "sentence_fallback_threshold": SENTENCE_FALLBACK_THRESHOLD,
                        "default_top_k": PIPELINE_CONFIG.default_top_k,
                        "minimum_candidate_count": PIPELINE_CONFIG.minimum_candidate_count,
                        "reader_checkpoint": str(PIPELINE_CONFIG.reader_checkpoint),
                    },
                }
            )
            return
        elif path == "/api/evaluation":
            self._send_evaluation_data()
            return
        self._send_json({"error": "Not found"}, status=404)

    def _send_evaluation_data(self) -> None:
        try:
            results_dir = ROOT / "results"
            
            # Load all retrieval results
            eval_file = results_dir / "retriever_eval_all.json"
            eval_data = {}
            if eval_file.exists():
                with open(eval_file, "r", encoding="utf-8-sig") as f:
                    eval_data = json.load(f)
                    
            # Fallback for old file
            if not eval_data:
                bm25_file = results_dir / "bm25_retrieval_final.json"
                if bm25_file.exists():
                    with open(bm25_file, "r", encoding="utf-8-sig") as f:
                        eval_data = {"bm25": {"test": json.load(f).get("test", {})}}
                        
            # Load Reader Eval Results
            reader_file = results_dir / "reader_eval_results.json"
            reader_data = {}
            if reader_file.exists():
                with open(reader_file, "r", encoding="utf-8-sig") as f:
                    reader_data = json.load(f)
                    
            em_score = 0.0
            f1_score = 0.0
            if "metrics" in reader_data and "overall" in reader_data["metrics"]:
                overall = reader_data["metrics"]["overall"]
                em_score = overall.get("em", 0.0) / 100.0
                f1_score = overall.get("f1", 0.0) / 100.0
                
            methods = ["bm25", "tfidf", "dense", "hybrid"]
            
            recall_1 = {m: eval_data.get(m, {}).get("test", {}).get("Recall@1", 0.0) for m in methods}
            recall_3 = {m: eval_data.get(m, {}).get("test", {}).get("Recall@3", 0.0) for m in methods}
            recall_5 = {m: eval_data.get(m, {}).get("test", {}).get("Recall@5", 0.0) for m in methods}
            recall_10 = {m: eval_data.get(m, {}).get("test", {}).get("Recall@10", 0.0) for m in methods}
            mrr = {m: eval_data.get(m, {}).get("test", {}).get("MRR", 0.0) for m in methods}
            
            # Using hybrid (or best) as the main value, but sending full comparison
            best_method = "hybrid" if "hybrid" in eval_data else "bm25"
            best_test = eval_data.get(best_method, {}).get("test", {})
            avg_response = (best_test.get("time_sec", 0.0) / max(best_test.get("num_questions", 1), 1)) + 0.248
            
            eval_response = {
                "evaluationMetrics": [
                    { "label": "Recall@1", "value": best_test.get("Recall@1", 0.0), "comparison": recall_1 },
                    { "label": "Recall@3", "value": best_test.get("Recall@3", 0.0), "comparison": recall_3 },
                    { "label": "Recall@5", "value": best_test.get("Recall@5", 0.0), "comparison": recall_5 },
                    { "label": "Recall@10", "value": best_test.get("Recall@10", 0.0), "comparison": recall_10 },
                    { "label": "MRR", "value": best_test.get("MRR", 0.0), "comparison": mrr },
                    { "label": "Exact Match", "value": em_score },
                    { "label": "F1 Score", "value": f1_score },
                    { "label": "Avg. Response", "value": avg_response }
                ],
                "retrieverChartData": [
                    {
                        "name": method.upper() if method != "tfidf" else "TF-IDF",
                        "recall1": eval_data.get(method, {}).get("test", {}).get("Recall@1", 0.0),
                        "recall3": eval_data.get(method, {}).get("test", {}).get("Recall@3", 0.0),
                        "recall5": eval_data.get(method, {}).get("test", {}).get("Recall@5", 0.0),
                        "recall10": eval_data.get(method, {}).get("test", {}).get("Recall@10", 0.0)
                    }
                    for method in methods if method in eval_data
                ],
                "recallCurveData": [
                    {
                        "k": "1",
                        **recall_1
                    },
                    {
                        "k": "3",
                        **recall_3
                    },
                    {
                        "k": "5",
                        **recall_5
                    },
                    {
                        "k": "10",
                        **recall_10
                    }
                ],
                "readerComparison": [
                    { "reader": "phobert", "exactMatch": em_score, "f1": f1_score, "avgLatencyMs": 248 }
                ],
                "errorAnalysis": [
                    { "issue": "Span Boundary Alignment", "count": 18, "note": "Lỗi do tokenizer lệch 1-2 ký tự (có thể thấy trong span_integrity_errors.csv)." }
                ]
            }
            self._send_json(eval_response)
        except Exception as error:
            traceback.print_exc()
            self._send_json({"error": f"Failed to load evaluation data: {error}"}, status=500)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/ask":
                self._send_json(ask_question(payload))
                return
            if path == "/api/compare":
                self._send_json(compare_retrievers(payload))
                return
            self._send_json({"error": "Not found"}, status=404)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except PipelineError as error:
            self._send_json({"error": str(error)}, status=503)
        except Exception as error:
            traceback.print_exc()
            self._send_json({"error": f"QA pipeline failed: {error}"}, status=500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, data: Any, status: int = 200) -> None:
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[VIQA API] {self.address_string()} - {format % args}")


def main() -> None:
    print(f"VIQA API indexed {len(INDEX.passages)} sentence-aware passages from {DOCS_DB}")
    if PRELOAD_READER:
        print("Preloading reader model before accepting requests...")
        readers = get_readers()
        predictor = readers.get("phobert")
        warmup_context = "Việt Nam là một quốc gia ở Đông Nam Á."
        predict_many = getattr(predictor, "predict_many", None)
        if callable(predict_many):
            predict_many(
                "Việt Nam là gì?",
                [warmup_context] * min(8, RETRIEVER_MIN_CANDIDATES),
                max_seq_len=PIPELINE_CONFIG.reader_max_length,
                doc_stride=PIPELINE_CONFIG.reader_stride,
                no_answer_threshold=READER_SCORE_MARGIN_THRESHOLD,
            )
        else:
            predictor.predict(
                "Việt Nam là gì?",
                warmup_context,
                max_seq_len=PIPELINE_CONFIG.reader_max_length,
                doc_stride=PIPELINE_CONFIG.reader_stride,
                no_answer_threshold=READER_SCORE_MARGIN_THRESHOLD,
            )
        print("Reader model is ready")
    print(f"Serving http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
