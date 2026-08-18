from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers.models.phobert.tokenization_phobert import PhobertTokenizer


class QaBatchEncoding(dict):
    """Small BatchEncoding-compatible container with explicit sequence ids."""

    def __init__(self, data, sequence_ids_by_feature):
        super().__init__(data)
        self._sequence_ids_by_feature = sequence_ids_by_feature

    def sequence_ids(self, feature_index: int = 0):
        return self._sequence_ids_by_feature[feature_index]


def load_phobert_tokenizer(model_name_or_path: str):
    """Load the model-compatible 64k PhoBERT tokenizer.

    The generated fast tokenizer previously committed in this project had no
    BPE merges. The upstream fast conversion also contains 2,118 token ids past
    this checkpoint's 64,001-row embedding table. The original PhoBERT
    ``vocab.txt`` + ``bpe.codes`` implementation is therefore the authoritative
    tokenizer; offsets are generated explicitly below.
    """

    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        vocab_file = model_path / "vocab.txt"
        merges_file = model_path / "bpe.codes"
        if not vocab_file.is_file() or not merges_file.is_file():
            raise FileNotFoundError(
                f"PhoBERT tokenizer requires {vocab_file} and {merges_file}. "
                "Do not train with a tokenizer.json that has an empty BPE merges list."
            )
        return PhobertTokenizer(
            str(vocab_file),
            str(merges_file),
            model_max_length=256,
        )
    return PhobertTokenizer.from_pretrained(model_name_or_path, model_max_length=256)


def _phobert_tokens_with_offsets(tokenizer, text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for word_match in re.finditer(r"\S+", text):
        word = word_match.group(0)
        pieces = tokenizer.tokenize(word)
        cursor = 0
        reconstructed = ""
        for piece in pieces:
            surface = piece[:-2] if piece.endswith("@@") else piece
            start = word_match.start() + cursor
            cursor += len(surface)
            end = word_match.start() + cursor
            tokens.append(piece)
            offsets.append((start, end))
            reconstructed += surface
        if reconstructed != word:
            raise ValueError(
                "PhoBERT BPE pieces could not be aligned to their segmented word: "
                f"{reconstructed!r} != {word!r}"
            )
    return tokens, offsets


def _slow_phobert_qa_batch(
    tokenizer,
    questions: Sequence[str],
    contexts: Sequence[str],
    max_length: int,
    stride: int,
    padding: str | bool,
    return_tensors: str | None,
) -> QaBatchEncoding:
    data = {
        "input_ids": [],
        "attention_mask": [],
        "offset_mapping": [],
        "overflow_to_sample_mapping": [],
    }
    sequence_ids_by_feature = []
    pad_to_max = padding == "max_length" or padding is True
    for sample_index, (question, context) in enumerate(zip(questions, contexts)):
        question_tokens = tokenizer.tokenize(question)
        question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        context_tokens, context_offsets = _phobert_tokens_with_offsets(tokenizer, context)
        context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        available_context = max_length - len(question_ids) - 4
        if available_context <= 0:
            raise ValueError(
                f"Question consumes {len(question_ids)} tokens; no room remains for context at max_length={max_length}"
            )

        window_start = 0
        while True:
            window_end = min(len(context_ids), window_start + available_context)
            window_ids = context_ids[window_start:window_end]
            input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, window_ids)
            question_special_count = 1 + len(question_ids) + 2
            sequence_ids = (
                [None]
                + [0] * len(question_ids)
                + [None, None]
                + [1] * len(window_ids)
                + [None]
            )
            offsets = (
                [(0, 0)] * question_special_count
                + context_offsets[window_start:window_end]
                + [(0, 0)]
            )
            attention_mask = [1] * len(input_ids)
            if pad_to_max:
                pad_count = max_length - len(input_ids)
                if pad_count < 0:
                    raise AssertionError("QA window exceeded max_length")
                input_ids.extend([tokenizer.pad_token_id] * pad_count)
                attention_mask.extend([0] * pad_count)
                offsets.extend([(0, 0)] * pad_count)
                sequence_ids.extend([None] * pad_count)

            data["input_ids"].append(input_ids)
            data["attention_mask"].append(attention_mask)
            data["offset_mapping"].append(offsets)
            data["overflow_to_sample_mapping"].append(sample_index)
            sequence_ids_by_feature.append(sequence_ids)
            if window_end >= len(context_ids):
                break
            next_start = window_end - stride
            if next_start <= window_start:
                raise ValueError("stride leaves no forward progress between context windows")
            window_start = next_start

    if return_tensors is not None:
        if return_tensors != "pt":
            raise ValueError("The QA tokenizer currently supports return_tensors='pt' only")
        data = {key: torch.tensor(value, dtype=torch.long) for key, value in data.items()}
    return QaBatchEncoding(data, sequence_ids_by_feature)


def encode_qa_batch(
    tokenizer,
    questions: Sequence[str],
    contexts: Sequence[str],
    *,
    max_length: int,
    stride: int,
    padding: str | bool = "max_length",
    return_tensors: str | None = None,
):
    if len(questions) != len(contexts):
        raise ValueError("questions and contexts must have the same length")
    if isinstance(tokenizer, PhobertTokenizer):
        return _slow_phobert_qa_batch(
            tokenizer,
            questions,
            contexts,
            max_length,
            stride,
            padding,
            return_tensors,
        )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Non-PhoBERT QA tokenizers must be fast to provide offset mappings")
    return tokenizer(
        list(questions),
        list(contexts),
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
        return_tensors=return_tensors,
    )
