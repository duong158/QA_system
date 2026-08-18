from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from reader.config import DEFAULT_DOC_STRIDE, DEFAULT_MAX_LENGTH
from reader.qa_tokenizer import encode_qa_batch, load_phobert_tokenizer
from reader.text_preprocessing import (
    ReaderTextPreprocessor,
    restore_text_offsets,
    uses_compact_offsets,
    uses_word_segmentation,
)


def load_qa_dataset(file_path: str, data_variant: str = "clean", subset_size: int = -1):
    """Load raw QA examples without replacing annotated offsets.

    Even when a legacy ``*_segmented.parquet`` file is supplied, the raw
    ``question/context/answer_text`` columns are retained. Word segmentation is
    performed exactly once later by :class:`ReaderTextPreprocessor`, where the
    raw ``answer_start`` can be mapped deterministically.
    """

    try:
        import pandas as pd
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - depends on training environment
        raise ImportError(
            "Training/evaluation data loading requires pandas, pyarrow, and datasets. "
            "Install requirements-reader.txt."
        ) from error

    frame = pd.read_parquet(file_path)
    required = ["id", "title", "context", "question", "answer_text", "answer_start"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"QA dataset is missing required columns: {missing}")
    frame = frame[required].copy()
    frame["id"] = frame["id"].astype(str)
    frame["answer_text"] = frame["answer_text"].fillna("").astype(str)
    frame["answer_start"] = frame["answer_start"].fillna(-1).astype(int)
    if subset_size > 0:
        frame = frame.head(subset_size)
    return Dataset.from_pandas(frame, preserve_index=False)


def get_tokenizer(model_name: str):
    if "phobert" in str(model_name).lower():
        return load_phobert_tokenizer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for non-PhoBERT QA offset mappings")
    return tokenizer


def build_text_preprocessor(model_name_or_path: str, tokenizer=None, model_config=None) -> ReaderTextPreprocessor:
    return ReaderTextPreprocessor(
        uses_word_segmentation(model_name_or_path, tokenizer=tokenizer, model_config=model_config)
    )


def _context_bounds(sequence_ids: list[int | None]) -> tuple[int, int]:
    context_indexes = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
    if not context_indexes:
        raise ValueError("Tokenizer produced a feature with no context tokens")
    return context_indexes[0], context_indexes[-1]


def _map_char_span_to_feature(
    offsets: list[tuple[int, int]],
    sequence_ids: list[int | None],
    start_char: int,
    end_char: int,
    cls_index: int,
) -> tuple[int, int]:
    """Map an annotated context character span into one overflow feature."""

    if start_char < 0 or end_char <= start_char:
        return cls_index, cls_index
    context_start, context_end = _context_bounds(sequence_ids)
    if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
        return cls_index, cls_index

    token_start = context_start
    while token_start <= context_end and offsets[token_start][1] <= start_char:
        token_start += 1
    token_end = context_end
    while token_end >= context_start and offsets[token_end][0] >= end_char:
        token_end -= 1
    if token_start > token_end:
        return cls_index, cls_index
    return token_start, token_end


def _prepare_features(
    examples: dict[str, list[Any]],
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
    preprocessor: ReaderTextPreprocessor,
    keep_evaluation_metadata: bool,
):
    prepared = []
    for index, (question, context, answer_text, answer_start) in enumerate(
        zip(
            examples["question"],
            examples["context"],
            examples["answer_text"],
            examples["answer_start"],
        )
    ):
        try:
            prepared.append(preprocessor.prepare(question, context, answer_text, int(answer_start)))
        except ValueError as error:
            example_id = examples.get("id", [str(index)] * len(examples["question"]))[index]
            raise ValueError(f"Invalid gold span for example {example_id}: {error}") from error

    tokenized = encode_qa_batch(
        tokenizer,
        [item.model_question for item in prepared],
        [item.model_context for item in prepared],
        max_length=max_seq_length,
        stride=doc_stride,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    raw_offsets = tokenized["offset_mapping"]
    compact_offsets = uses_compact_offsets(tokenizer)
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    if keep_evaluation_metadata:
        tokenized["example_id"] = []
        tokenized["cls_index"] = []

    evaluation_offsets: list[list[tuple[int, int] | None]] = []
    for feature_index, tokenizer_offsets in enumerate(raw_offsets):
        input_ids = tokenized["input_ids"][feature_index]
        try:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        except ValueError:
            cls_index = 0
        sequence_ids = tokenized.sequence_ids(feature_index)
        sample_index = int(sample_mapping[feature_index])
        item = prepared[sample_index]
        offsets = restore_text_offsets(item.model_context, tokenizer_offsets, compact_offsets)
        start_position, end_position = _map_char_span_to_feature(
            offsets,
            sequence_ids,
            item.model_answer_start,
            item.model_answer_end,
            cls_index,
        )
        tokenized["start_positions"].append(start_position)
        tokenized["end_positions"].append(end_position)

        if keep_evaluation_metadata:
            tokenized["example_id"].append(str(examples["id"][sample_index]))
            tokenized["cls_index"].append(cls_index)
            evaluation_offsets.append(
                [tuple(offset) if sequence_id == 1 else None for offset, sequence_id in zip(offsets, sequence_ids)]
            )

    if keep_evaluation_metadata:
        tokenized["offset_mapping"] = evaluation_offsets
    else:
        tokenized.pop("offset_mapping")
    return tokenized


def prepare_train_features(
    examples,
    tokenizer,
    max_seq_length: int = DEFAULT_MAX_LENGTH,
    doc_stride: int = DEFAULT_DOC_STRIDE,
    preprocessor: ReaderTextPreprocessor | None = None,
):
    preprocessor = preprocessor or build_text_preprocessor(
        str(getattr(tokenizer, "name_or_path", "")), tokenizer=tokenizer
    )
    return _prepare_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        preprocessor,
        keep_evaluation_metadata=False,
    )


def prepare_validation_features(
    examples,
    tokenizer,
    max_seq_length: int = DEFAULT_MAX_LENGTH,
    doc_stride: int = DEFAULT_DOC_STRIDE,
    preprocessor: ReaderTextPreprocessor | None = None,
):
    preprocessor = preprocessor or build_text_preprocessor(
        str(getattr(tokenizer, "name_or_path", "")), tokenizer=tokenizer
    )
    return _prepare_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        preprocessor,
        keep_evaluation_metadata=True,
    )


def model_input_features(validation_features):
    """Remove post-processing metadata before passing features to the model."""

    removable = [
        column
        for column in ("example_id", "offset_mapping", "cls_index")
        if column in validation_features.column_names
    ]
    return validation_features.remove_columns(removable)
