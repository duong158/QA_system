from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.config import (  # noqa: E402
    DEFAULT_DOC_STRIDE,
    DEFAULT_MAX_ANSWER_LENGTH,
    DEFAULT_MAX_LENGTH,
    effective_model_max_length,
    load_reader_decision_config,
    validate_window_config,
)
from reader.postprocessing import (  # noqa: E402
    SpanCandidate,
    has_clean_word_boundaries,
    score_margin_to_confidence,
    select_best_span,
    select_span_candidates,
    should_return_answer,
)
from reader.qa_tokenizer import encode_qa_batch, load_phobert_tokenizer  # noqa: E402
from reader.text_preprocessing import (  # noqa: E402
    ReaderTextPreprocessor,
    align_raw_and_model_text,
    restore_text_offsets,
    uses_compact_offsets,
    uses_word_segmentation,
)


def map_segmented_span_to_raw(
    raw_context: str,
    segmented_context: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    """Backwards-compatible wrapper around the shared deterministic alignment."""

    alignment = align_raw_and_model_text(raw_context, segmented_context)
    return alignment.model_span_to_raw(start_char, end_char)


class ReaderPredictor:
    """PhoBERT-compatible extractive QA inference with joint span search.

    No-answer formula::

        null_score = min_feature(start_logit[CLS] + end_logit[CLS])
        best_span_score = max_valid_span(start_logit + end_logit)
        score_margin = best_span_score - null_score
        return answer iff score_margin >= validation_calibrated_threshold

    ``confidence`` is a checkpoint-relative display/ranking signal and is not a
    calibrated probability. The best valid span is preserved even when its
    margin is below the calibrated threshold; callers receive
    ``passes_reader_threshold`` as a signal for final candidate selection.
    """

    def __init__(self, model_path_or_name: str, use_cpu: bool = False):
        self.model_path_or_name = model_path_or_name
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
        print(f"Loading reader model from {model_path_or_name} to {self.device}...")

        if "phobert" in model_path_or_name.lower():
            self.tokenizer = load_phobert_tokenizer(model_path_or_name)
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
            if not self.tokenizer.is_fast:
                raise RuntimeError("Non-PhoBERT Reader requires a fast tokenizer for offset mappings")
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
        self.is_phobert = uses_word_segmentation(
            model_path_or_name,
            tokenizer=self.tokenizer,
            model_config=self.model.config,
        )
        self.preprocessor = ReaderTextPreprocessor(word_segment=self.is_phobert)
        self.decision_config = load_reader_decision_config(model_path_or_name)
        self.checkpoint_is_qa = True

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
    ) -> dict[str, float | int | str | bool | None]:
        return self.predict_batch(
            [question],
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
        **kwargs,
    ) -> list[dict[str, float | int | str | bool | None]]:
        return self.predict_batch([question] * len(contexts), contexts, **kwargs)

    def predict_batch(
        self,
        questions: Sequence[str],
        contexts: Sequence[str],
        max_seq_len: int | None = None,
        doc_stride: int | None = None,
        max_answer_len: int | None = None,
        no_answer_threshold: float | None = None,
        top_n_start: int | None = None,
        top_n_end: int | None = None,
    ) -> list[dict[str, float | int | str | bool | None]]:
        """Run arbitrary question/context pairs in shared model-forward batches."""

        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length")
        if not contexts:
            return []

        max_seq_len = int(max_seq_len or os.getenv("QA_READER_MAX_LENGTH", DEFAULT_MAX_LENGTH))
        doc_stride = int(doc_stride or os.getenv("QA_READER_STRIDE", DEFAULT_DOC_STRIDE))
        max_answer_len = int(
            max_answer_len or os.getenv("QA_MAX_ANSWER_LENGTH", DEFAULT_MAX_ANSWER_LENGTH)
        )
        top_n_start = int(top_n_start or os.getenv("QA_TOP_N_START", "20"))
        top_n_end = int(top_n_end or os.getenv("QA_TOP_N_END", "20"))
        threshold = (
            float(no_answer_threshold)
            if no_answer_threshold is not None
            else self.decision_config.threshold
        )
        validate_window_config(max_seq_len, doc_stride, self.model, self.tokenizer)

        prepared = [
            self.preprocessor.prepare(question, context)
            for question, context in zip(questions, contexts)
        ]
        inputs = encode_qa_batch(
            self.tokenizer,
            [item.model_question for item in prepared],
            [item.model_context for item in prepared],
            max_length=max_seq_len,
            stride=doc_stride,
            padding="max_length",
            return_tensors="pt",
        )

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
        compact_offsets = uses_compact_offsets(self.tokenizer)
        sample_mapping = inputs["overflow_to_sample_mapping"].tolist()
        best_by_sample: list[tuple[SpanCandidate, int, int] | None] = [None] * len(prepared)
        null_scores_by_sample: list[list[float]] = [[] for _ in prepared]

        for feature_index in range(len(start_logits)):
            sample_index = int(sample_mapping[feature_index])
            feature_offsets = restore_text_offsets(
                prepared[sample_index].model_context,
                offsets[feature_index],
                compact_offsets,
            )
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
                feature_offsets,
                sequence_ids,
                top_n_start=top_n_start,
                top_n_end=top_n_end,
                max_answer_length=max_answer_len,
            )
            item = prepared[sample_index]
            for candidate in candidates:
                raw_start, raw_end = item.context_alignment.model_span_to_raw(
                    candidate.start_char,
                    candidate.end_char,
                )
                if not has_clean_word_boundaries(item.raw_context, raw_start, raw_end):
                    continue
                current = best_by_sample[sample_index]
                if current is None or candidate.score > current[0].score:
                    best_by_sample[sample_index] = (candidate, raw_start, raw_end)
                break

        results: list[dict[str, float | int | str | bool | None]] = []
        for sample_index, item in enumerate(prepared):
            selected = best_by_sample[sample_index]
            null_score = min(null_scores_by_sample[sample_index], default=float("inf"))
            if selected is None:
                results.append(
                    {
                        "answer": "",
                        "best_span_answer": "",
                        "score": float("-inf"),
                        "best_span_score": float("-inf"),
                        "start_score": None,
                        "end_score": None,
                        "null_score": null_score,
                        "no_answer_score": float("inf"),
                        "score_margin": float("-inf"),
                        "start": -1,
                        "end": -1,
                        "confidence": 0.0,
                        "confidence_is_calibrated": False,
                        "decision_threshold": threshold,
                        "reader_threshold_score": 0.0,
                        "valid_span": False,
                        "passes_reader_threshold": False,
                        "has_answer": False,
                    }
                )
                continue

            best, raw_start, raw_end = selected
            while raw_start < raw_end and item.raw_context[raw_start].isspace():
                raw_start += 1
            while raw_end > raw_start and item.raw_context[raw_end - 1].isspace():
                raw_end -= 1
            best_span_answer = item.raw_context[raw_start:raw_end]
            margin = float(best.score - null_score)
            valid_span = bool(best_span_answer)
            passes_threshold = valid_span and should_return_answer(margin, threshold)
            confidence = score_margin_to_confidence(
                margin - threshold,
                temperature=self.decision_config.margin_scale,
            )
            results.append(
                {
                    # Backwards-compatible final Reader answer. The candidate is
                    # independently exposed below and is never deleted here.
                    "answer": best_span_answer if passes_threshold else "",
                    "candidate_answer": best_span_answer,
                    "best_span_answer": best_span_answer,
                    "score": float(best.score),
                    "best_span_score": float(best.score),
                    "start_score": best.start_score,
                    "end_score": best.end_score,
                    "null_score": float(null_score),
                    "no_answer_score": float(null_score - best.score),
                    "score_margin": margin,
                    "start": raw_start if passes_threshold else -1,
                    "end": raw_end if passes_threshold else -1,
                    "candidate_start": raw_start,
                    "candidate_end": raw_end,
                    "best_span_start": raw_start,
                    "best_span_end": raw_end,
                    "confidence": round(confidence, 6),
                    "confidence_is_calibrated": False,
                    "decision_threshold": threshold,
                    "reader_threshold_score": round(confidence, 6),
                    "valid_span": valid_span,
                    "passes_reader_threshold": passes_threshold,
                    "has_answer": passes_threshold,
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict an extractive answer span")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Raw best-span-minus-null margin")
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()
    predictor = ReaderPredictor(args.model_path, use_cpu=args.use_cpu)
    print(
        predictor.predict(
            args.question,
            args.context,
            no_answer_threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ReaderPredictor",
    "SpanCandidate",
    "map_segmented_span_to_raw",
    "select_best_span",
    "select_span_candidates",
]
