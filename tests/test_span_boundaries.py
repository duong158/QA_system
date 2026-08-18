import unittest

from backend.viqa_api import build_passage_candidates, sentence_fallback_predict
from reader.fallback_extractor import extract_fallback_answer
from reader.question_type import QuestionType, detect_question_type
from reader.span_boundaries import assess_span_boundary


class SpanBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.context = (
            "Ông còn có tên gọi là Lâm Bá Kiệt khi làm Phó chủ nhiệm cơ quan "
            "Biện sự xứ tại Quế Lâm."
        )

    def test_complete_name_has_clean_boundary(self):
        answer = "Lâm Bá Kiệt"
        start = self.context.index(answer)

        assessment = assess_span_boundary(
            self.context,
            start,
            start + len(answer),
            QuestionType.ENTITY,
        )

        self.assertEqual(assessment.score, 1.0)
        self.assertTrue(assessment.complete)

    def test_missing_right_name_token_is_detected(self):
        answer = "Lâm Bá"
        start = self.context.index(answer)

        assessment = assess_span_boundary(
            self.context,
            start,
            start + len(answer),
            QuestionType.ENTITY,
        )

        self.assertEqual(assessment.score, 0.15)
        self.assertIn("TRUNCATED_NAMED_ENTITY_RIGHT", assessment.reasons)

    def test_same_length_shifted_span_loses_to_complete_candidate(self):
        question = (
            "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm "
            "cơ quan Biện sự xứ tại Quế Lâm?"
        )
        shifted = "là Lâm Bá"
        complete = "Lâm Bá Kiệt"
        shifted_start = self.context.index(shifted)
        complete_start = self.context.index(complete)
        neural_output = {
            "span_candidates": [
                {
                    "rank": 1,
                    "text": shifted,
                    "start": shifted_start,
                    "end": shifted_start + len(shifted),
                    "score": 8.8,
                    "score_margin": 1.2,
                    "reader_threshold_score": 0.73,
                    "passes_reader_threshold": True,
                    "valid_span": True,
                },
                {
                    "rank": 2,
                    "text": complete,
                    "start": complete_start,
                    "end": complete_start + len(complete),
                    "score": 8.5,
                    "score_margin": 0.9,
                    "reader_threshold_score": 0.71,
                    "passes_reader_threshold": True,
                    "valid_span": True,
                },
            ]
        }

        candidates = build_passage_candidates(
            question,
            QuestionType.ENTITY,
            "doc_00001_P0001",
            self.context,
            1.0,
            neural_output,
            {},
        )
        self.assertEqual(len(candidates), 1)
        refined = candidates[0]
        self.assertEqual(refined.raw_text, shifted)
        self.assertEqual(refined.text, complete)
        self.assertTrue(refined.refinement_changed)
        self.assertIn("leading_scaffold_compression", refined.refinement_method)
        self.assertIn("phrase_expand_right", refined.refinement_method)
        self.assertIsNone(refined.rejection_reason)

    def test_overlong_time_span_is_incomplete(self):
        context = (
            "Tháng 1 năm 1947, ông gia nhập nội các với chức vụ Bộ trưởng "
            "cựu chiến binh chiến tranh."
        )

        assessment = assess_span_boundary(
            context,
            0,
            len(context),
            QuestionType.TIME,
            "Ông gia nhập nội các khi nào?",
        )

        self.assertEqual(assessment.score, 0.15)
        self.assertIn("OVERLONG_TIME_SPAN", assessment.reasons)

    def test_temporal_fallback_returns_expression_not_whole_sentence(self):
        sentence = (
            "Tháng 1 năm 1947, ông gia nhập nội các với chức vụ Bộ trưởng "
            "cựu chiến binh chiến tranh."
        )

        candidate = extract_fallback_answer(
            "Ông gia nhập nội các khi nào?",
            "TIME",
            sentence,
        )

        self.assertEqual(candidate.answer, "Tháng 1 năm 1947")
        self.assertEqual(candidate.method, "temporal_expression_pattern")

    def test_person_fallback_returns_complete_titled_name(self):
        sentence = "Tổ chức an ninh do Thiếu tướng Fahd Ahmed Al-Fahd đứng đầu."

        candidate = extract_fallback_answer(
            "Người đứng đầu tổ chức an ninh là ai?",
            "PERSON",
            sentence,
        )

        self.assertEqual(candidate.answer, "Thiếu tướng Fahd Ahmed Al-Fahd")
        self.assertEqual(candidate.method, "person_relation_pattern")

    def test_named_person_definition_returns_complete_predicate(self):
        sentence = (
            "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là "
            "Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam "
            "từ năm 1976 cho đến khi nghỉ hưu năm 1987."
        )

        candidate = extract_fallback_answer(
            "phạm văn đồng là ai",
            "PERSON",
            sentence,
        )

        self.assertEqual(
            candidate.answer,
            "Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam "
            "từ năm 1976 cho đến khi nghỉ hưu năm 1987",
        )
        self.assertEqual(candidate.method, "person_definition_pattern")
        self.assertEqual(candidate.relation_type, "PERSON_DEFINITION")

    def test_split_dau_tien_phrase_is_incomplete(self):
        context = "Phạm Văn Đồng là Thủ tướng đầu tiên của Việt Nam."
        answer = "Thủ tướng đầu"
        start = context.index(answer)

        assessment = assess_span_boundary(
            context,
            start,
            start + len(answer),
            QuestionType.PERSON,
            "Phạm Văn Đồng là ai?",
        )

        self.assertEqual(assessment.score, 0.15)
        self.assertIn("TRUNCATED_FIXED_PHRASE_RIGHT", assessment.reasons)

    def test_property_question_selects_complete_adjective_phrase(self):
        context = (
            "Năm 1954, ông được giao nhiệm vụ Trưởng phái đoàn Chính phủ dự "
            "Hội nghị Genève về Đông Dương. Những đóng góp của đoàn Việt Nam "
            "do ông đứng đầu là vô cùng quan trọng, tạo ra những đột phá đưa "
            "Hội nghị tới thành công. Trải qua 8 phiên họp toàn thể và 23 "
            "phiên họp rất căng thẳng và phức tạp, với tinh thần chủ động của "
            "phái đoàn Việt Nam."
        )

        output = sentence_fallback_predict(
            "Hội nghị Genève về Đông Dương có tính chất như thế nào?",
            context,
        )

        self.assertEqual(output["answer"], "rất căng thẳng và phức tạp")
        self.assertEqual(output["fallback_method"], "property_description_pattern")
        self.assertEqual(output["relation_type"], "PROPERTY_DESCRIPTION")

    def test_role_question_removes_scaffold_and_keeps_complete_title(self):
        sentence = (
            "Năm 1954, ông được giao nhiệm vụ Trưởng phái đoàn Chính phủ "
            "dự Hội nghị Genève về Đông Dương."
        )

        candidate = extract_fallback_answer(
            "Chức vụ mà Phạm Văn Đồng đảm nhiệm tại Hội nghị Genève về Đông Dương?",
            "GENERAL",
            sentence,
        )

        self.assertEqual(candidate.answer, "Trưởng phái đoàn Chính phủ")
        self.assertEqual(candidate.method, "role_relation_pattern")
        self.assertEqual(candidate.relation_type, "ROLE_RELATION")

    def test_descriptive_quantity_question_is_not_forced_to_number(self):
        self.assertEqual(
            detect_question_type("Số lượng cá thể kiến biến đổi như thế nào?"),
            QuestionType.GENERAL,
        )

    def test_khu_vuc_question_is_location(self):
        self.assertEqual(
            detect_question_type("Người di cư tập trung ở khu vực nào?"),
            QuestionType.LOCATION,
        )


if __name__ == "__main__":
    unittest.main()
