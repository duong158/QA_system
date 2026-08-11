from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pyvi import ViTokenizer

    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False


@dataclass(frozen=True)
class SpanCandidate:
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    score: float


def select_best_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None],
    top_n_start: int = 20,
    top_n_end: int = 20,
    max_answer_length: int = 40,
) -> SpanCandidate | None:
    """Select the best valid start/end pair restricted to context tokens."""
    candidates = select_span_candidates(
        start_logits,
        end_logits,
        offsets,
        sequence_ids,
        top_n_start=top_n_start,
        top_n_end=top_n_end,
        max_answer_length=max_answer_length,
        limit=1,
    )
    return candidates[0] if candidates else None


def select_span_candidates(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None],
    top_n_start: int = 20,
    top_n_end: int = 20,
    max_answer_length: int = 40,
    limit: int = 20,
) -> list[SpanCandidate]:
    """Return ranked context spans so callers can reject malformed boundaries."""
    start_indexes = np.argsort(np.asarray(start_logits))[::-1][:top_n_start]
    end_indexes = np.argsort(np.asarray(end_logits))[::-1][:top_n_end]
    candidates: dict[tuple[int, int], SpanCandidate] = {}

    for start_index in start_indexes:
        if sequence_ids[int(start_index)] != 1:
            continue
        for end_index in end_indexes:
            start_index = int(start_index)
            end_index = int(end_index)
            if sequence_ids[end_index] != 1 or end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue

            start_char = int(offsets[start_index][0])
            end_char = int(offsets[end_index][1])
            if end_char <= start_char:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            key = (start_char, end_char)
            previous = candidates.get(key)
            if previous is None or score > previous.score:
                candidates[key] = SpanCandidate(start_index, end_index, start_char, end_char, score)
    return sorted(candidates.values(), key=lambda candidate: candidate.score, reverse=True)[:limit]


def map_segmented_span_to_raw(
    raw_context: str,
    segmented_context: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    """Map PhoBERT/PyVi offsets back to the exact original-context characters."""
    mapping: list[int | None] = [None] * len(segmented_context)
    raw_index = 0
    for segmented_index, char in enumerate(segmented_context):
        if char.isspace() or char == "_":
            continue
        while raw_index < len(raw_context) and (raw_context[raw_index].isspace() or raw_context[raw_index] == "_"):
            raw_index += 1
        if raw_index >= len(raw_context):
            break
        if raw_context[raw_index].casefold() != char.casefold():
            while raw_index < len(raw_context) and raw_context[raw_index].casefold() != char.casefold():
                raw_index += 1
            if raw_index >= len(raw_context):
                break
        mapping[segmented_index] = raw_index
        raw_index += 1

    mapped = [position for position in mapping[start_char:end_char] if position is not None]
    if not mapped:
        return -1, -1
    return mapped[0], mapped[-1] + 1


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _has_clean_word_boundaries(context: str, start: int, end: int) -> bool:
    if start < 0 or end <= start or end > len(context):
        return False
    starts_inside_word = start > 0 and context[start - 1].isalnum() and context[start].isalnum()
    ends_inside_word = end < len(context) and context[end - 1].isalnum() and context[end].isalnum()
    return not starts_inside_word and not ends_inside_word


class ReaderPredictor:
    def __init__(self, model_path_or_name: str, use_cpu: bool = False):
        self.model_path_or_name = model_path_or_name
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
        print(f"Loading reader model from {model_path_or_name} to {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("Reader requires a fast tokenizer for offset_mapping and sequence_ids")

        self.model, loading_info = AutoModelForQuestionAnswering.from_pretrained(
            model_path_or_name,
            output_loading_info=True,
        )
        missing_qa_head = [key for key in loading_info.get("missing_keys", []) if "qa_outputs" in key]
        if missing_qa_head:
            raise RuntimeError(
                "Checkpoint does not contain a trained QA head; refusing to use randomly initialized qa_outputs"
            )
        self.model.to(self.device)
        self.model.eval()
        self.is_phobert = "phobert" in model_path_or_name.lower() or self.model.config.model_type == "roberta" and "Phobert" in self.tokenizer.__class__.__name__
        self.checkpoint_is_qa = not missing_qa_head

    def predict(
        self,
        question: str,
        context: str,
        max_seq_len: int | None = None,
        doc_stride: int | None = None,
        max_answer_len: int | None = None,
        no_answer_threshold: float | None = None,
        top_n_start: int | None = None,
        top_n_end: int | None = None,
    ) -> dict[str, float | int | str | bool]:
        return self.predict_many(
            question,
            [context],
            max_seq_len=max_seq_len,
            doc_stride=doc_stride,
            max_answer_len=max_answer_len,
            no_answer_threshold=no_answer_threshold,
            top_n_start=top_n_start,
            top_n_end=top_n_end,
        )[0]

    def predict_many(
        self,
        question: str,
        contexts: Sequence[str],
        max_seq_len: int | None = None,
        doc_stride: int | None = None,
        max_answer_len: int | None = None,
        no_answer_threshold: float | None = None,
        top_n_start: int | None = None,
        top_n_end: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Run one question against multiple passages with batched model forwards."""
        if not contexts:
            return []
        max_seq_len = max_seq_len or int(os.getenv("QA_READER_MAX_LENGTH", "384"))
        doc_stride = doc_stride or int(os.getenv("QA_READER_STRIDE", "128"))
        max_answer_len = max_answer_len or int(os.getenv("QA_MAX_ANSWER_LENGTH", "40"))
        top_n_start = top_n_start or int(os.getenv("QA_TOP_N_START", "20"))
        top_n_end = top_n_end or int(os.getenv("QA_TOP_N_END", "20"))
        no_answer_threshold = (
            float(no_answer_threshold)
            if no_answer_threshold is not None
            else float(os.getenv("QA_ANSWER_THRESHOLD", "0.30"))
        )

        model_limit = int(getattr(self.model.config, "max_position_embeddings", max_seq_len))
        if self.is_phobert:
            max_seq_len = min(max_seq_len, max(8, model_limit - 2))
        doc_stride = min(doc_stride, max(1, max_seq_len // 3))

        # Truncate exceptionally long questions to prevent HuggingFace Tokenizer PanicException
        # (stride must be strictly less than max_len)
        question = question[:120]

        raw_contexts = list(contexts)
        if self.is_phobert:
            if not HAS_PYVI:
                raise ImportError("pyvi is required for PhoBERT inference")
            model_question = ViTokenizer.tokenize(question)
            model_contexts = [ViTokenizer.tokenize(context) for context in raw_contexts]
        else:
            model_question = question
            model_contexts = raw_contexts

        try:
            inputs = self.tokenizer(
                [model_question] * len(model_contexts),
                model_contexts,
                truncation="only_second",
                max_length=max_seq_len,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )
        except BaseException as e:
            print("--- TOKENIZATION FAILURE DIAGNOSTICS ---")
            print(f"question (truncated to 120 chars): {question!r}")
            print(f"model_question: {model_question!r}")
            print(f"max_seq_len: {max_seq_len}")
            print(f"doc_stride: {doc_stride}")
            try:
                q_tokens = self.tokenizer.tokenize(model_question)
                print(f"Question tokens count: {len(q_tokens)}")
                print(f"Question tokens: {q_tokens}")
            except Exception as tokenize_err:
                print(f"Failed to tokenize question alone: {tokenize_err}")
            print("----------------------------------------")
            raise e
        batch_size = max(1, int(os.getenv("QA_READER_BATCH_SIZE", "8")))
        start_batches: list[torch.Tensor] = []
        end_batches: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch_start in range(0, len(inputs["input_ids"]), batch_size):
                batch_end = batch_start + batch_size
                model_inputs = {
                    key: value[batch_start:batch_end].to(self.device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                outputs = self.model(**model_inputs)
                start_batches.append(outputs.start_logits.detach().cpu())
                end_batches.append(outputs.end_logits.detach().cpu())

        start_logits = torch.cat(start_batches).numpy()
        end_logits = torch.cat(end_batches).numpy()
        offsets = inputs["offset_mapping"].tolist()
        sample_mapping = inputs["overflow_to_sample_mapping"].tolist()
        best_by_sample: list[tuple[SpanCandidate, int, int] | None] = [None] * len(raw_contexts)
        null_scores_by_sample: list[list[float]] = [[] for _ in raw_contexts]

        for feature_index in range(len(start_logits)):
            sample_index = int(sample_mapping[feature_index])
            sequence_ids = inputs.sequence_ids(feature_index)
            input_ids = inputs["input_ids"][feature_index].tolist()
            try:
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
            except ValueError:
                cls_index = 0
            null_scores_by_sample[sample_index].append(
                float(start_logits[feature_index][cls_index] + end_logits[feature_index][cls_index])
            )

            candidates = select_span_candidates(
                start_logits[feature_index],
                end_logits[feature_index],
                offsets[feature_index],
                sequence_ids,
                top_n_start=top_n_start,
                top_n_end=top_n_end,
                max_answer_length=max_answer_len,
            )
            for candidate in candidates:
                if self.is_phobert:
                    raw_start, raw_end = map_segmented_span_to_raw(
                        raw_contexts[sample_index],
                        model_contexts[sample_index],
                        candidate.start_char,
                        candidate.end_char,
                    )
                else:
                    raw_start, raw_end = candidate.start_char, candidate.end_char
                if not _has_clean_word_boundaries(raw_contexts[sample_index], raw_start, raw_end):
                    continue
                current = best_by_sample[sample_index]
                if current is None or candidate.score > current[0].score:
                    best_by_sample[sample_index] = (candidate, raw_start, raw_end)
                break

        temperature = max(0.01, float(os.getenv("QA_READER_CONFIDENCE_TEMPERATURE", "10.0")))
        results: list[dict[str, float | int | str | bool]] = []
        for sample_index, raw_context in enumerate(raw_contexts):
            selected = best_by_sample[sample_index]
            null_scores = null_scores_by_sample[sample_index]
            if selected is None:
                results.append(
                    {
                        "answer": "",
                        "score": float("-inf"),
                        "null_score": min(null_scores, default=float("inf")),
                        "score_margin": float("-inf"),
                        "start": -1,
                        "end": -1,
                        "confidence": 0.0,
                        "has_answer": False,
                    }
                )
                continue

            best, raw_start, raw_end = selected
            while raw_start < raw_end and raw_context[raw_start].isspace():
                raw_start += 1
            while raw_end > raw_start and raw_context[raw_end - 1].isspace():
                raw_end -= 1
            answer = raw_context[raw_start:raw_end]
            null_score = min(null_scores, default=best.score)
            margin = best.score - null_score
            confidence = _sigmoid(margin / temperature)
            results.append(
                {
                    "answer": answer,
                    "score": float(best.score),
                    "null_score": float(null_score),
                    "score_margin": float(margin),
                    "start": raw_start,
                    "end": raw_end,
                    "confidence": round(confidence, 6),
                    "has_answer": bool(answer) and confidence >= no_answer_threshold,
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an extractive answer span")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    predictor = ReaderPredictor(args.model_path, use_cpu=args.use_cpu)
    print(predictor.predict(args.question, args.context))


if __name__ == "__main__":
    main()
