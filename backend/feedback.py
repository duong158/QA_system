from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


FEEDBACK_TYPES = {
    "CORRECT",
    "INCORRECT",
    "SPAN_CORRECTION",
    "NO_ANSWER_BUT_SHOULD_HAVE",
    "ANSWERED_BUT_SHOULD_NOT",
    "DOCUMENT_SUBMISSION",
}
FEEDBACK_STATUSES = {"PENDING", "REVIEWED", "APPROVED", "REJECTED"}
REVIEW_DECISIONS = {"REVIEWED", "APPROVED", "REJECTED"}
GAP_TYPES = {
    "CORPUS_GAP",
    "RETRIEVAL_GAP",
    "READER_SEMANTIC_GAP",
    "UNKNOWN_GAP",
}
DOCUMENT_STATUSES = {"PENDING_REVIEW", "APPROVED", "REJECTED"}
DOCUMENT_SOURCE_TYPES = {"PLAIN_TEXT", "TXT"}


class FeedbackValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_feedback_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _optional_text(value: Any, *, max_length: int = 10_000) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_length:
        raise FeedbackValidationError(f"text field exceeds {max_length} characters")
    return text


def _passage_value(passage: Any, key: str, default: Any = None) -> Any:
    if isinstance(passage, Mapping):
        return passage.get(key, default)
    return getattr(passage, key, default)


def classify_gap(
    feedback_type: str,
    *,
    corrected_passage_id: str | None = None,
    retrieved_passage_ids: Iterable[str] | None = None,
    corpus_support_found: bool | None = None,
    selected_passage_id: str | None = None,
    rejection_reason: str | None = None,
) -> str | None:
    """Classify only when the submitted metadata provides enough evidence."""
    kind = str(feedback_type or "").upper()
    if kind == "CORRECT":
        return None
    if corpus_support_found is False:
        return "CORPUS_GAP"

    if corrected_passage_id:
        if retrieved_passage_ids is None:
            return "UNKNOWN_GAP"
        retrieved = {str(item) for item in retrieved_passage_ids if item}
        return (
            "READER_SEMANTIC_GAP"
            if corrected_passage_id in retrieved
            else "RETRIEVAL_GAP"
        )

    if selected_passage_id and (
        kind in {"INCORRECT", "ANSWERED_BUT_SHOULD_NOT"} or rejection_reason
    ):
        return "READER_SEMANTIC_GAP"
    return "UNKNOWN_GAP"


@dataclass(frozen=True)
class QAFeedback:
    feedback_id: str
    timestamp: str
    question: str
    predicted_answer: str | None
    feedback_type: str
    question_type: str | None
    semantic_relation: str | None
    subject: str | None
    selected_passage_id: str | None
    corrected_passage_id: str | None
    corrected_answer: str | None
    corrected_start_char: int | None
    corrected_end_char: int | None
    rejection_reason: str | None
    user_note: str | None
    model_version: str | None
    corpus_version: str | None
    semantic_policy_version: str | None
    status: str
    gap_type: str | None
    source: str
    synthetic: bool
    conflict: bool
    duplicate_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentSubmission:
    submission_id: str
    title: str
    content: str
    timestamp: str
    status: str
    review_note: str | None
    source_type: str
    synthetic: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    """SQLite review store. It is intentionally independent from runtime QA lookup."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if schema_version > 1:
                raise RuntimeError(
                    f"feedback database schema v{schema_version} is newer than supported v1"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    predicted_answer TEXT,
                    feedback_type TEXT NOT NULL,
                    question_type TEXT,
                    semantic_relation TEXT,
                    subject TEXT,
                    selected_passage_id TEXT,
                    corrected_passage_id TEXT,
                    corrected_answer TEXT,
                    corrected_start_char INTEGER,
                    corrected_end_char INTEGER,
                    retrieved_passage_ids TEXT NOT NULL DEFAULT '[]',
                    rejection_reason TEXT,
                    user_note TEXT,
                    model_version TEXT,
                    corpus_version TEXT,
                    semantic_policy_version TEXT,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    gap_type TEXT,
                    source TEXT NOT NULL DEFAULT 'DIRECT_QUERY',
                    synthetic INTEGER NOT NULL DEFAULT 0,
                    conflict INTEGER NOT NULL DEFAULT 0,
                    fingerprint TEXT NOT NULL UNIQUE,
                    duplicate_count INTEGER NOT NULL DEFAULT 1,
                    review_note TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);
                CREATE INDEX IF NOT EXISTS idx_feedback_relation ON feedback(semantic_relation);
                CREATE INDEX IF NOT EXISTS idx_feedback_question_type ON feedback(question_type);
                CREATE INDEX IF NOT EXISTS idx_feedback_gap ON feedback(gap_type);
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);

                CREATE TABLE IF NOT EXISTS document_submissions (
                    submission_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
                    review_note TEXT,
                    source_type TEXT NOT NULL,
                    synthetic INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_document_status
                    ON document_submissions(status);
                """
            )
            connection.execute("PRAGMA user_version = 1")

    @staticmethod
    def _fingerprint(values: Iterable[Any]) -> str:
        normalized = "\u241f".join(normalize_feedback_text(value) for value in values)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _feedback_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["synthetic"] = bool(result.get("synthetic"))
        result["conflict"] = bool(result.get("conflict"))
        try:
            result["retrieved_passage_ids"] = json.loads(
                result.get("retrieved_passage_ids") or "[]"
            )
        except json.JSONDecodeError:
            result["retrieved_passage_ids"] = []
        result.pop("normalized_question", None)
        result.pop("fingerprint", None)
        return result

    @staticmethod
    def _document_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["synthetic"] = bool(result.get("synthetic"))
        return result

    def submit_feedback(
        self,
        payload: Mapping[str, Any],
        *,
        passage_lookup: Callable[[str], Any | None],
        system_version: Mapping[str, str] | None = None,
        synthetic: bool = False,
    ) -> dict[str, Any]:
        question = _optional_text(payload.get("question"), max_length=2_000)
        if not question:
            raise FeedbackValidationError("question is required")
        feedback_type = str(payload.get("feedback_type") or "").strip().upper()
        if feedback_type not in FEEDBACK_TYPES or feedback_type == "DOCUMENT_SUBMISSION":
            raise FeedbackValidationError("unsupported feedback_type")

        predicted_answer = _optional_text(payload.get("predicted_answer"))
        selected_passage_id = _optional_text(payload.get("selected_passage_id"), max_length=300)
        corrected_passage_id = _optional_text(
            payload.get("corrected_passage_id"), max_length=300
        )
        corrected_answer = _optional_text(payload.get("corrected_answer"))
        start = payload.get("corrected_start_char")
        end = payload.get("corrected_end_char")
        retrieved_known = "retrieved_passage_ids" in payload
        retrieved_raw = payload.get("retrieved_passage_ids") or []
        if not isinstance(retrieved_raw, list):
            raise FeedbackValidationError("retrieved_passage_ids must be an array")
        retrieved_ids = list(dict.fromkeys(str(item) for item in retrieved_raw if item))

        for passage_id in [selected_passage_id, *retrieved_ids]:
            if passage_id and passage_lookup(passage_id) is None:
                raise FeedbackValidationError(f"unknown passage_id: {passage_id}")

        has_any_correction = any(value is not None for value in (corrected_answer, start, end))
        correction_required = feedback_type == "SPAN_CORRECTION"
        if correction_required and not has_any_correction:
            raise FeedbackValidationError("span correction fields are required")
        if has_any_correction:
            if not corrected_passage_id:
                corrected_passage_id = selected_passage_id
            if not corrected_passage_id or corrected_answer is None or start is None or end is None:
                raise FeedbackValidationError(
                    "corrected_passage_id, corrected_answer and offsets are required together"
                )
            if isinstance(start, bool) or isinstance(end, bool):
                raise FeedbackValidationError("span offsets must be integers")
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError) as error:
                raise FeedbackValidationError("span offsets must be integers") from error
            passage = passage_lookup(corrected_passage_id)
            if passage is None:
                raise FeedbackValidationError(f"unknown passage_id: {corrected_passage_id}")
            passage_text = str(_passage_value(passage, "text", ""))
            if start < 0 or end <= start or end > len(passage_text):
                raise FeedbackValidationError("corrected span offsets are outside the passage")
            corpus_span = passage_text[start:end]
            if normalize_feedback_text(corpus_span) != normalize_feedback_text(corrected_answer):
                raise FeedbackValidationError("corrected_answer does not match the corpus span")
            corrected_answer = corpus_span
        elif corrected_passage_id:
            raise FeedbackValidationError("corrected_passage_id requires a corrected span")
        else:
            start = end = None

        corpus_support_raw = payload.get("corpus_support_found")
        corpus_support_found = (
            corpus_support_raw if isinstance(corpus_support_raw, bool) else None
        )
        gap_type = classify_gap(
            feedback_type,
            corrected_passage_id=corrected_passage_id,
            retrieved_passage_ids=retrieved_ids if retrieved_known else None,
            corpus_support_found=corpus_support_found,
            selected_passage_id=selected_passage_id,
            rejection_reason=_optional_text(payload.get("rejection_reason"), max_length=300),
        )
        versions = dict(system_version or {})
        timestamp = utc_now()
        fingerprint = self._fingerprint(
            (
                question,
                predicted_answer,
                feedback_type,
                corrected_passage_id,
                corrected_answer,
                start,
                end,
            )
        )
        record = {
            "feedback_id": f"fb_{uuid.uuid4().hex}",
            "timestamp": timestamp,
            "last_seen": timestamp,
            "question": question,
            "normalized_question": normalize_feedback_text(question),
            "predicted_answer": predicted_answer,
            "feedback_type": feedback_type,
            "question_type": _optional_text(payload.get("question_type"), max_length=100),
            "semantic_relation": _optional_text(payload.get("semantic_relation"), max_length=100),
            "subject": _optional_text(payload.get("subject"), max_length=500),
            "selected_passage_id": selected_passage_id,
            "corrected_passage_id": corrected_passage_id,
            "corrected_answer": corrected_answer,
            "corrected_start_char": start,
            "corrected_end_char": end,
            "retrieved_passage_ids": json.dumps(retrieved_ids, ensure_ascii=False),
            "rejection_reason": _optional_text(payload.get("rejection_reason"), max_length=300),
            "user_note": _optional_text(payload.get("user_note"), max_length=4_000),
            "model_version": versions.get("reader"),
            "corpus_version": versions.get("corpus"),
            "semantic_policy_version": versions.get("semantic_policy"),
            "status": "PENDING",
            "gap_type": gap_type,
            "source": str(payload.get("source") or "DIRECT_QUERY").strip().upper(),
            "synthetic": int(bool(synthetic)),
            "conflict": 0,
            "fingerprint": fingerprint,
            "duplicate_count": 1,
            "review_note": None,
        }

        with self._write_lock, self._connection() as connection:
            duplicate = connection.execute(
                "SELECT feedback_id FROM feedback WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if duplicate:
                connection.execute(
                    """UPDATE feedback
                       SET duplicate_count = duplicate_count + 1, last_seen = ?
                       WHERE feedback_id = ?""",
                    (timestamp, duplicate["feedback_id"]),
                )
                row = connection.execute(
                    "SELECT * FROM feedback WHERE feedback_id = ?",
                    (duplicate["feedback_id"],),
                ).fetchone()
                return {**self._feedback_dict(row), "deduplicated": True}

            columns = ", ".join(record)
            placeholders = ", ".join("?" for _ in record)
            connection.execute(
                f"INSERT INTO feedback ({columns}) VALUES ({placeholders})",
                tuple(record.values()),
            )

            if corrected_answer and corrected_passage_id:
                conflicts = connection.execute(
                    """SELECT feedback_id FROM feedback
                       WHERE normalized_question = ?
                         AND corrected_passage_id = ?
                         AND feedback_id != ?
                         AND corrected_answer IS NOT NULL
                         AND (corrected_start_char != ? OR corrected_end_char != ?
                              OR corrected_answer != ?)
                         AND status != 'REJECTED'""",
                    (
                        record["normalized_question"],
                        corrected_passage_id,
                        record["feedback_id"],
                        start,
                        end,
                        corrected_answer,
                    ),
                ).fetchall()
                if conflicts:
                    conflicting_ids = [row["feedback_id"] for row in conflicts]
                    connection.execute(
                        "UPDATE feedback SET conflict = 1 WHERE feedback_id = ?",
                        (record["feedback_id"],),
                    )
                    connection.executemany(
                        "UPDATE feedback SET conflict = 1 WHERE feedback_id = ?",
                        ((item,) for item in conflicting_ids),
                    )

            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?", (record["feedback_id"],)
            ).fetchone()
            return {**self._feedback_dict(row), "deduplicated": False}

    def list_feedback(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        limit = min(1_000, max(1, int(limit)))
        parameters: list[Any] = []
        where = ""
        if status:
            normalized_status = status.upper()
            if normalized_status not in FEEDBACK_STATUSES:
                raise FeedbackValidationError("invalid feedback status")
            where = "WHERE status = ?"
            parameters.append(normalized_status)
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM feedback {where} ORDER BY timestamp DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._feedback_dict(row) for row in rows]

    def get_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?", (feedback_id,)
            ).fetchone()
        return self._feedback_dict(row) if row else None

    def review_feedback(self, feedback_id: str, decision: str, note: Any = None) -> dict[str, Any]:
        normalized = str(decision or "").upper()
        if normalized not in REVIEW_DECISIONS:
            raise FeedbackValidationError("decision must be REVIEWED, APPROVED or REJECTED")
        review_note = _optional_text(note, max_length=4_000)
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE feedback SET status = ?, review_note = ? WHERE feedback_id = ?",
                (normalized, review_note, feedback_id),
            )
            if cursor.rowcount != 1:
                raise FeedbackValidationError("feedback not found")
            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?", (feedback_id,)
            ).fetchone()
        return self._feedback_dict(row)

    def submit_document(self, payload: Mapping[str, Any], *, synthetic: bool = False) -> dict[str, Any]:
        title = _optional_text(payload.get("title"), max_length=500)
        content = _optional_text(payload.get("content"), max_length=1_000_000)
        if not title:
            raise FeedbackValidationError("document title is required")
        if not content or len(content) < 20:
            raise FeedbackValidationError("document content must contain at least 20 characters")
        source_type = str(payload.get("source_type") or "PLAIN_TEXT").upper()
        if source_type not in DOCUMENT_SOURCE_TYPES:
            raise FeedbackValidationError("V1 supports PLAIN_TEXT and TXT only")
        record = {
            "submission_id": f"docsub_{uuid.uuid4().hex}",
            "title": title,
            "content": content,
            "timestamp": utc_now(),
            "status": "PENDING_REVIEW",
            "review_note": None,
            "source_type": source_type,
            "synthetic": int(bool(synthetic)),
        }
        with self._write_lock, self._connection() as connection:
            columns = ", ".join(record)
            placeholders = ", ".join("?" for _ in record)
            connection.execute(
                f"INSERT INTO document_submissions ({columns}) VALUES ({placeholders})",
                tuple(record.values()),
            )
            row = connection.execute(
                "SELECT * FROM document_submissions WHERE submission_id = ?",
                (record["submission_id"],),
            ).fetchone()
        return self._document_dict(row)

    def list_documents(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        limit = min(1_000, max(1, int(limit)))
        where = ""
        parameters: list[Any] = []
        if status:
            normalized = status.upper()
            if normalized not in DOCUMENT_STATUSES:
                raise FeedbackValidationError("invalid document status")
            where = "WHERE status = ?"
            parameters.append(normalized)
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM document_submissions {where} ORDER BY timestamp DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._document_dict(row) for row in rows]

    def review_document(self, submission_id: str, decision: str, note: Any = None) -> dict[str, Any]:
        normalized = str(decision or "").upper()
        if normalized not in {"APPROVED", "REJECTED"}:
            raise FeedbackValidationError("decision must be APPROVED or REJECTED")
        review_note = _optional_text(note, max_length=4_000)
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """UPDATE document_submissions SET status = ?, review_note = ?
                   WHERE submission_id = ?""",
                (normalized, review_note, submission_id),
            )
            if cursor.rowcount != 1:
                raise FeedbackValidationError("document submission not found")
            row = connection.execute(
                "SELECT * FROM document_submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        return self._document_dict(row)


def export_approved_feedback(
    store: FeedbackStore,
    passage_lookup: Callable[[str], Any | None],
    output_path: str | Path,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in reversed(store.list_feedback(status="APPROVED", limit=1_000)):
            passage_id = record.get("corrected_passage_id")
            answer = record.get("corrected_answer")
            start = record.get("corrected_start_char")
            if not passage_id or answer is None or start is None:
                continue
            passage = passage_lookup(str(passage_id))
            if passage is None:
                continue
            row = {
                "question": record["question"],
                "context": str(_passage_value(passage, "text", "")),
                "answer_text": answer,
                "answer_start": int(start),
                "source": "human_feedback",
                "feedback_id": record["feedback_id"],
                "system_version": {
                    "reader": record.get("model_version"),
                    "corpus": record.get("corpus_version"),
                    "semantic_policy": record.get("semantic_policy_version"),
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = [
    "DOCUMENT_SOURCE_TYPES",
    "DOCUMENT_STATUSES",
    "FEEDBACK_STATUSES",
    "FEEDBACK_TYPES",
    "GAP_TYPES",
    "DocumentSubmission",
    "FeedbackStore",
    "FeedbackValidationError",
    "QAFeedback",
    "classify_gap",
    "export_approved_feedback",
    "normalize_feedback_text",
]
