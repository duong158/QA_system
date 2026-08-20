import ast
import tempfile
import unittest
from pathlib import Path

from backend.feedback import (
    FeedbackStore,
    FeedbackValidationError,
    classify_gap,
    export_approved_feedback,
)
from backend.feedback_analytics import build_feedback_analytics


ROOT = Path(__file__).resolve().parents[1]


class FeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "feedback.db"
        self.passages = {
            "fixture_P0001": {
                "passage_id": "fixture_P0001",
                "text": "Đối tượng A sinh năm 1945 và giữ một vai trò quan trọng.",
            },
            "fixture_P0002": {
                "passage_id": "fixture_P0002",
                "text": "Đối tượng B hoạt động tại một địa điểm khác.",
            },
        }
        self.store = FeedbackStore(self.db_path)
        self.version = {
            "reader": "reader-test",
            "corpus": "corpus-test",
            "semantic_policy": "semantic-test",
        }

    def tearDown(self):
        self.temp.cleanup()

    def lookup(self, passage_id):
        return self.passages.get(passage_id)

    def submit(self, **overrides):
        payload = {
            "question": "Đối tượng A là gì?",
            "predicted_answer": "Một đối tượng.",
            "feedback_type": "CORRECT",
            "selected_passage_id": "fixture_P0001",
            "retrieved_passage_ids": ["fixture_P0001"],
            "question_type": "DEFINITION",
            "semantic_relation": "IDENTITY",
            "subject": "Đối tượng A",
        }
        payload.update(overrides)
        return self.store.submit_feedback(
            payload,
            passage_lookup=self.lookup,
            system_version=self.version,
        )

    def test_valid_positive_feedback_is_pending_and_versioned(self):
        result = self.submit()
        self.assertEqual(result["feedback_type"], "CORRECT")
        self.assertEqual(result["status"], "PENDING")
        self.assertEqual(result["model_version"], "reader-test")
        self.assertIsNone(result["gap_type"])

    def test_valid_span_correction_is_verified_against_corpus(self):
        text = self.passages["fixture_P0001"]["text"]
        answer = "năm 1945"
        start = text.index(answer)
        result = self.submit(
            feedback_type="SPAN_CORRECTION",
            corrected_passage_id="fixture_P0001",
            corrected_answer=answer,
            corrected_start_char=start,
            corrected_end_char=start + len(answer),
        )
        self.assertEqual(result["corrected_answer"], answer)
        self.assertEqual(result["gap_type"], "READER_SEMANTIC_GAP")

    def test_invalid_span_offset_is_rejected(self):
        with self.assertRaisesRegex(FeedbackValidationError, "outside the passage"):
            self.submit(
                feedback_type="SPAN_CORRECTION",
                corrected_passage_id="fixture_P0001",
                corrected_answer="năm 1945",
                corrected_start_char=999,
                corrected_end_char=1008,
            )

    def test_passage_text_mismatch_is_rejected(self):
        text = self.passages["fixture_P0001"]["text"]
        start = text.index("năm 1945")
        with self.assertRaisesRegex(FeedbackValidationError, "does not match"):
            self.submit(
                feedback_type="SPAN_CORRECTION",
                corrected_passage_id="fixture_P0001",
                corrected_answer="năm 1970",
                corrected_start_char=start,
                corrected_end_char=start + len("năm 1945"),
            )

    def test_unknown_passage_is_rejected(self):
        with self.assertRaisesRegex(FeedbackValidationError, "unknown passage_id"):
            self.submit(selected_passage_id="missing_P0001")

    def test_feedback_persists_across_store_instances(self):
        created = self.submit()
        reopened = FeedbackStore(self.db_path)
        loaded = reopened.get_feedback(created["feedback_id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["question"], created["question"])

    def test_duplicate_feedback_is_aggregated(self):
        first = self.submit()
        second = self.submit()
        self.assertEqual(first["feedback_id"], second["feedback_id"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(second["duplicate_count"], 2)
        self.assertEqual(len(self.store.list_feedback()), 1)

    def test_conflicting_spans_are_flagged_for_review(self):
        text = self.passages["fixture_P0001"]["text"]
        first_text = "Đối tượng A"
        second_text = "năm 1945"
        first = self.submit(
            feedback_type="SPAN_CORRECTION",
            corrected_passage_id="fixture_P0001",
            corrected_answer=first_text,
            corrected_start_char=text.index(first_text),
            corrected_end_char=text.index(first_text) + len(first_text),
        )
        second = self.submit(
            feedback_type="SPAN_CORRECTION",
            corrected_passage_id="fixture_P0001",
            corrected_answer=second_text,
            corrected_start_char=text.index(second_text),
            corrected_end_char=text.index(second_text) + len(second_text),
        )
        self.assertTrue(self.store.get_feedback(first["feedback_id"])["conflict"])
        self.assertTrue(self.store.get_feedback(second["feedback_id"])["conflict"])

    def test_review_approve_and_reject(self):
        approved = self.submit(question="Câu hỏi duyệt?")
        rejected = self.submit(question="Câu hỏi loại?")
        approved_row = self.store.review_feedback(approved["feedback_id"], "APPROVED", "verified")
        rejected_row = self.store.review_feedback(rejected["feedback_id"], "REJECTED", "invalid")
        self.assertEqual(approved_row["status"], "APPROVED")
        self.assertEqual(rejected_row["status"], "REJECTED")

    def test_document_submission_stays_out_of_production(self):
        document = self.store.submit_document(
            {
                "title": "Tài liệu đóng góp",
                "content": "Đây là nội dung tài liệu đủ dài để đưa vào hàng chờ xem xét.",
                "source_type": "PLAIN_TEXT",
            }
        )
        self.assertEqual(document["status"], "PENDING_REVIEW")
        self.assertEqual(len(self.store.list_documents(status="PENDING_REVIEW")), 1)
        reviewed = self.store.review_document(document["submission_id"], "APPROVED")
        self.assertEqual(reviewed["status"], "APPROVED")

    def test_gap_classification_covers_all_evidence_states(self):
        self.assertEqual(
            classify_gap("NO_ANSWER_BUT_SHOULD_HAVE", corpus_support_found=False),
            "CORPUS_GAP",
        )
        self.assertEqual(
            classify_gap(
                "SPAN_CORRECTION",
                corrected_passage_id="p2",
                retrieved_passage_ids=["p1"],
            ),
            "RETRIEVAL_GAP",
        )
        self.assertEqual(
            classify_gap(
                "SPAN_CORRECTION",
                corrected_passage_id="p1",
                retrieved_passage_ids=["p1"],
            ),
            "READER_SEMANTIC_GAP",
        )
        self.assertEqual(
            classify_gap("NO_ANSWER_BUT_SHOULD_HAVE"),
            "UNKNOWN_GAP",
        )

    def test_analytics_aggregates_ten_cause_records_with_four_failures(self):
        for index in range(6):
            self.submit(
                question=f"Câu đúng {index}?",
                semantic_relation="CAUSE",
                question_type="GENERAL",
            )
        for index in range(4):
            self.submit(
                question=f"Câu sai {index}?",
                feedback_type="INCORRECT",
                semantic_relation="CAUSE",
                question_type="GENERAL",
                rejection_reason="CAUSE_RELATION_NOT_FOUND",
            )
        analytics = build_feedback_analytics(self.store.list_feedback())
        cause = next(row for row in analytics["relations"] if row["semantic_relation"] == "CAUSE")
        self.assertEqual(cause["total"], 10)
        self.assertEqual(cause["correct"], 6)
        self.assertEqual(cause["incorrect"], 4)
        self.assertAlmostEqual(cause["failure_rate"], 0.4)
        self.assertGreater(cause["blind_spot_score"], 0.4)

    def test_only_approved_span_corrections_are_exported(self):
        text = self.passages["fixture_P0001"]["text"]
        answer = "năm 1945"
        start = text.index(answer)
        approved = self.submit(
            question="Câu đã duyệt?",
            feedback_type="SPAN_CORRECTION",
            corrected_passage_id="fixture_P0001",
            corrected_answer=answer,
            corrected_start_char=start,
            corrected_end_char=start + len(answer),
        )
        self.submit(
            question="Câu chưa duyệt?",
            feedback_type="SPAN_CORRECTION",
            corrected_passage_id="fixture_P0001",
            corrected_answer=answer,
            corrected_start_char=start,
            corrected_end_char=start + len(answer),
        )
        self.store.review_feedback(approved["feedback_id"], "APPROVED")
        output = Path(self.temp.name) / "approved.jsonl"
        count = export_approved_feedback(self.store, self.lookup, output)
        self.assertEqual(count, 1)
        self.assertIn('"source": "human_feedback"', output.read_text(encoding="utf-8"))

    def test_ask_pipeline_does_not_read_feedback_store(self):
        source = (ROOT / "backend" / "viqa_api.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        ask = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "ask_question"
        )
        segment = ast.get_source_segment(source, ask) or ""
        self.assertNotIn("feedback", segment.casefold())
        self.assertNotIn("FEEDBACK_DB", segment)

    def test_required_feedback_api_routes_are_registered(self):
        source = (ROOT / "backend" / "viqa_api.py").read_text(encoding="utf-8")
        for route in (
            "/api/feedback",
            "/api/feedback/analytics",
            "/api/feedback/review",
            "/api/documents/submissions",
        ):
            self.assertIn(route, source)

    def test_production_feedback_modules_have_no_fixture_literals(self):
        production = "\n".join(
            (ROOT / "backend" / name).read_text(encoding="utf-8")
            for name in ("feedback.py", "feedback_analytics.py")
        )
        for forbidden in ("fixture_P0001", "Đối tượng A", "năm 1945"):
            self.assertNotIn(forbidden, production)


if __name__ == "__main__":
    unittest.main()
