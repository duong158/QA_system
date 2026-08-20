import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FeedbackUiContractTests(unittest.TestCase):
    def test_positive_and_negative_feedback_actions_exist(self):
        source = (ROOT / "src" / "components" / "feedback" / "AnswerFeedback.tsx").read_text(encoding="utf-8")
        self.assertIn("Chính xác", source)
        self.assertIn("Chưa chính xác", source)
        self.assertIn("feedback_type: 'CORRECT'", source)
        self.assertIn("feedback_type: 'ANSWERED_BUT_SHOULD_NOT'", source)

    def test_incorrect_action_opens_span_and_note_options(self):
        source = (ROOT / "src" / "components" / "feedback" / "AnswerFeedback.tsx").read_text(encoding="utf-8")
        self.assertIn("Chọn đoạn đúng trong nguồn", source)
        self.assertIn("setMode('span')", source)
        self.assertIn("setMode('note')", source)

    def test_span_selection_sends_offsets_and_is_disabled_without_selection(self):
        source = (ROOT / "src" / "components" / "feedback" / "AnswerFeedback.tsx").read_text(encoding="utf-8")
        helper = (ROOT / "src" / "utils" / "textSelection.ts").read_text(encoding="utf-8")
        self.assertIn("getSelectionOffsets", source)
        self.assertIn("corrected_start_char: span.start", source)
        self.assertIn("corrected_end_char: span.end", source)
        self.assertIn("disabled={!span || submitting}", source)
        self.assertIn("prefix.selectNodeContents(container)", helper)
        self.assertIn("Array.from(prefix.toString()).length", helper)

    def test_dashboard_loads_analytics_and_supports_review(self):
        page = (ROOT / "src" / "pages" / "KnowledgeBlindSpotsPage.tsx").read_text(encoding="utf-8")
        app = (ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
        self.assertIn("fetchFeedbackAnalytics()", page)
        self.assertIn("fetchPendingFeedback()", page)
        self.assertIn("reviewFeedback", page)
        self.assertIn("Heatmap: Question Type × Relation", page)
        self.assertIn('path="/knowledge-blind-spots"', app)

    def test_document_submission_is_explicitly_review_only(self):
        page = (ROOT / "src" / "pages" / "KnowledgeBlindSpotsPage.tsx").read_text(encoding="utf-8")
        service = (ROOT / "src" / "services" / "feedbackService.ts").read_text(encoding="utf-8")
        self.assertIn("không chunk/index production tự động", page)
        self.assertIn("/api/documents/submissions", service)
        self.assertIn("reviewDocument", service)

    def test_ui_does_not_claim_immediate_learning(self):
        feedback = (ROOT / "src" / "components" / "feedback" / "AnswerFeedback.tsx").read_text(encoding="utf-8")
        dashboard = (ROOT / "src" / "pages" / "KnowledgeBlindSpotsPage.tsx").read_text(encoding="utf-8")
        self.assertIn("không tự cập nhật trọng số", feedback)
        self.assertIn("không tự cập nhật tại runtime", dashboard)
        self.assertNotIn("đã học ngay", feedback.casefold())


if __name__ == "__main__":
    unittest.main()
