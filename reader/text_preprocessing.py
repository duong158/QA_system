from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache


try:
    from pyvi import ViTokenizer

    HAS_PYVI = True
except ImportError:  # pragma: no cover - exercised only in minimal production installs
    ViTokenizer = None
    HAS_PYVI = False


def _is_alignment_separator(char: str) -> bool:
    return char.isspace() or char == "_"


def _comparable_char(char: str) -> str:
    return unicodedata.normalize("NFC", char).casefold()


def normalize_span_text(text: str) -> str:
    """Normalization used only for validating raw/segmented span equivalence."""

    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return "".join(char for char in text if not _is_alignment_separator(char))


@lru_cache(maxsize=32)
def tokenizer_uses_compact_offsets(tokenizer_class: str, tokenizer_name: str, tokenizer_json: str) -> bool:
    """Cacheable implementation detail for :func:`uses_compact_offsets`.

    ``tokenizer_json`` is the serialized backend tokenizer. It is included in
    the cache key because two instances of the same class may have different
    pre-tokenizers.
    """

    del tokenizer_class, tokenizer_name
    # A missing pre-tokenizer in the converted PhoBERT tokenizer causes spaces
    # to be discarded from the offset coordinate system.
    return '"pre_tokenizer":null' in tokenizer_json


def uses_compact_offsets(tokenizer) -> bool:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialized = backend.to_str() if backend is not None else ""
    return tokenizer_uses_compact_offsets(
        type(tokenizer).__name__,
        str(getattr(tokenizer, "name_or_path", "")),
        serialized,
    )


def restore_text_offsets(
    text: str,
    offsets: list[tuple[int, int]] | list[list[int]],
    compact_offsets: bool,
) -> list[tuple[int, int]]:
    """Convert tokenizer offsets back to character positions in ``text``.

    Official tokenizer configurations with a normal pre-tokenizer already
    return source-text offsets and pass through unchanged. The converted local
    PhoBERT tokenizer has no pre-tokenizer and reports coordinates after
    whitespace removal; those coordinates are expanded deterministically.
    """

    if not compact_offsets:
        return [tuple(map(int, offset)) for offset in offsets]
    compact_positions = [index for index, char in enumerate(text) if not char.isspace()]
    restored: list[tuple[int, int]] = []
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        if end <= start or start >= len(compact_positions):
            restored.append((0, 0))
            continue
        end_index = min(end - 1, len(compact_positions) - 1)
        restored.append((compact_positions[start], compact_positions[end_index] + 1))
    return restored


@dataclass(frozen=True)
class TextAlignment:
    raw_text: str
    model_text: str
    model_to_raw: tuple[int | None, ...]
    raw_to_model: tuple[int | None, ...]

    def raw_span_to_model(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end <= start or end > len(self.raw_text):
            return -1, -1
        mapped = [position for position in self.raw_to_model[start:end] if position is not None]
        if not mapped:
            return -1, -1
        return mapped[0], mapped[-1] + 1

    def model_span_to_raw(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end <= start or end > len(self.model_text):
            return -1, -1
        mapped = [position for position in self.model_to_raw[start:end] if position is not None]
        if not mapped:
            return -1, -1
        return mapped[0], mapped[-1] + 1


def _identity_alignment(text: str) -> TextAlignment:
    positions = tuple(range(len(text)))
    return TextAlignment(text, text, positions, positions)


def align_raw_and_model_text(raw_text: str, model_text: str) -> TextAlignment:
    """Align texts whose only differences are spaces and PyVi underscores.

    The alignment is global and deterministic, so a repeated answer is mapped
    using its annotated raw offset rather than by searching for the first copy.
    """

    raw_positions = [i for i, char in enumerate(raw_text) if not _is_alignment_separator(char)]
    model_positions = [i for i, char in enumerate(model_text) if not _is_alignment_separator(char)]
    raw_chars = [_comparable_char(raw_text[i]) for i in raw_positions]
    model_chars = [_comparable_char(model_text[i]) for i in model_positions]
    if raw_chars != model_chars:
        mismatch = next(
            (
                index
                for index, (raw_char, model_char) in enumerate(zip(raw_chars, model_chars))
                if raw_char != model_char
            ),
            min(len(raw_chars), len(model_chars)),
        )
        raise ValueError(
            "Word segmentation changed non-formatting characters; "
            f"cannot align raw/model text at compact character {mismatch}."
        )

    raw_to_model: list[int | None] = [None] * len(raw_text)
    model_to_raw: list[int | None] = [None] * len(model_text)
    for raw_position, model_position in zip(raw_positions, model_positions):
        raw_to_model[raw_position] = model_position
        model_to_raw[model_position] = raw_position
    return TextAlignment(raw_text, model_text, tuple(model_to_raw), tuple(raw_to_model))


@lru_cache(maxsize=16_384)
def segment_with_alignment(text: str) -> TextAlignment:
    raw_text = unicodedata.normalize("NFC", str(text or ""))
    if not HAS_PYVI:
        raise ImportError("pyvi is required for PhoBERT preprocessing")
    assert ViTokenizer is not None
    segmented = unicodedata.normalize("NFC", ViTokenizer.tokenize(raw_text))
    return align_raw_and_model_text(raw_text, segmented)


def uses_word_segmentation(model_name_or_path: str, tokenizer=None, model_config=None) -> bool:
    name = str(model_name_or_path or "").lower()
    tokenizer_name = type(tokenizer).__name__.lower() if tokenizer is not None else ""
    model_type = str(getattr(model_config, "model_type", "")).lower()
    return "phobert" in name or "phobert" in tokenizer_name or (
        model_type == "roberta" and "phobert" in str(getattr(model_config, "_name_or_path", "")).lower()
    )


@dataclass(frozen=True)
class PreparedQaText:
    raw_question: str
    raw_context: str
    model_question: str
    model_context: str
    context_alignment: TextAlignment
    answer_start: int = -1
    answer_end: int = -1
    model_answer_start: int = -1
    model_answer_end: int = -1
    answer_text: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.answer_start >= 0 and bool(self.answer_text)


class ReaderTextPreprocessor:
    """Single source of truth for train, validation, and inference text handling."""

    def __init__(self, word_segment: bool):
        self.word_segment = bool(word_segment)

    def prepare(
        self,
        question: str,
        context: str,
        answer_text: str = "",
        answer_start: int = -1,
    ) -> PreparedQaText:
        raw_question = unicodedata.normalize("NFC", str(question or "")).strip()
        raw_context = unicodedata.normalize("NFC", str(context or ""))
        answer_text = unicodedata.normalize("NFC", str(answer_text or ""))
        answer_start = int(answer_start if answer_start is not None else -1)

        if self.word_segment:
            question_alignment = segment_with_alignment(raw_question)
            context_alignment = segment_with_alignment(raw_context)
        else:
            question_alignment = _identity_alignment(raw_question)
            context_alignment = _identity_alignment(raw_context)

        answer_end = -1
        model_start = -1
        model_end = -1
        if answer_start >= 0 and answer_text:
            answer_end = answer_start + len(answer_text)
            if answer_end > len(raw_context):
                raise ValueError("BAD_ANSWER_START: annotated answer extends past the context")
            annotated_text = raw_context[answer_start:answer_end]
            if normalize_span_text(annotated_text) != normalize_span_text(answer_text):
                raise ValueError(
                    "BAD_ANSWER_START: context slice at answer_start does not equal answer_text "
                    f"({annotated_text!r} != {answer_text!r})"
                )
            model_start, model_end = context_alignment.raw_span_to_model(answer_start, answer_end)
            if model_start < 0:
                raise ValueError("TOKENIZER_ALIGNMENT: answer contains no alignable characters")
            model_answer = context_alignment.model_text[model_start:model_end]
            if normalize_span_text(model_answer) != normalize_span_text(answer_text):
                raise ValueError(
                    "TOKENIZER_ALIGNMENT: segmented answer does not match the annotated answer "
                    f"({model_answer!r} != {answer_text!r})"
                )

        return PreparedQaText(
            raw_question=raw_question,
            raw_context=raw_context,
            model_question=question_alignment.model_text,
            model_context=context_alignment.model_text,
            context_alignment=context_alignment,
            answer_start=answer_start,
            answer_end=answer_end,
            model_answer_start=model_start,
            model_answer_end=model_end,
            answer_text=answer_text,
        )
