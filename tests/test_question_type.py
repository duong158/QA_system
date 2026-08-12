import unittest

from reader.question_type import QuestionType, assess_answer_type, detect_question_type


class QuestionTypeTests(unittest.TestCase):
    def test_detects_required_vietnamese_question_types(self):
        cases = {
            "Hồ Chí Minh sinh năm nào?": QuestionType.TIME,
            "Việt Nam có bao nhiêu tỉnh thành?": QuestionType.NUMBER,
            "Ai là tác giả của tác phẩm này?": QuestionType.PERSON,
            "Tháp Eiffel nằm ở đâu?": QuestionType.LOCATION,
            "Thực vật hạt kín là gì?": QuestionType.DEFINITION,
            "Công trình nào nằm trên đỉnh Montmartre?": QuestionType.ENTITY,
            "Hãy trình bày sự kiện này.": QuestionType.GENERAL,
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(detect_question_type(question), expected)

    def test_time_scoring_supports_centuries_years_and_roman_numerals(self):
        self.assertEqual(assess_answer_type(QuestionType.TIME, "Vào thế kỷ 10").score, 1.0)
        self.assertEqual(assess_answer_type(QuestionType.TIME, "thế kỷ XIX").score, 1.0)
        self.assertEqual(assess_answer_type(QuestionType.TIME, "cuối thế kỷ XIX").score, 1.0)
        self.assertEqual(assess_answer_type(QuestionType.TIME, "thế kỷ thứ mười").score, 1.0)
        self.assertGreaterEqual(
            assess_answer_type(QuestionType.TIME, "Ngày 2 tháng 9 năm 1945").score,
            0.9,
        )

    def test_time_question_penalizes_non_temporal_parisien_candidate(self):
        assessment = assess_answer_type(
            QuestionType.TIME,
            'Từ "parisien" trong tiếng Pháp là tính từ của Paris.',
        )
        self.assertEqual(assessment.score, 0.0)
        self.assertFalse(assessment.matched)
        self.assertEqual(assessment.reason, "NO_TEMPORAL_EXPRESSION")

    def test_number_person_and_location_scoring_are_soft(self):
        self.assertEqual(assess_answer_type(QuestionType.NUMBER, "63 tỉnh thành").score, 1.0)
        self.assertGreaterEqual(assess_answer_type(QuestionType.PERSON, "Nguyễn Du").score, 0.8)
        self.assertGreaterEqual(assess_answer_type(QuestionType.LOCATION, "Paris").score, 0.8)
        self.assertGreater(
            assess_answer_type(QuestionType.LOCATION, "một khu vực ở phía bắc").score,
            0.0,
        )


    def test_event_names_score_below_their_location_arguments(self):
        pairs = (
            ("C\u00e1ch m\u1ea1ng Ph\u00e1p", "Ph\u00e1p"),
            ("Chi\u1ebfn tranh Vi\u1ec7t Nam", "Vi\u1ec7t Nam"),
            ("H\u1ed9i ngh\u1ecb Gen\u00e8ve", "Gen\u00e8ve"),
        )
        for event, location in pairs:
            with self.subTest(event=event, location=location):
                event_score = assess_answer_type(
                    QuestionType.LOCATION,
                    event,
                    relation_score=0.0,
                    phrase_quality=0.05,
                ).score
                location_score = assess_answer_type(
                    QuestionType.LOCATION,
                    location,
                    relation_score=1.0,
                    phrase_quality=0.75,
                    candidate_method="location_relation_pattern",
                ).score
                self.assertLess(event_score, 0.50)
                self.assertGreaterEqual(location_score, 0.75)

    def test_whole_sentence_location_candidate_is_penalized(self):
        assessment = assess_answer_type(
            QuestionType.LOCATION,
            "M\u1ed9t c\u00f4ng tr\u00ecnh b\u1ecb ph\u00e1 khi n\u1ed5 ra C\u00e1ch m\u1ea1ng Ph\u00e1p.",
            relation_score=0.0,
            phrase_quality=0.0,
            candidate_method="whole_sentence",
        )

        self.assertLess(assessment.score, 0.50)
        self.assertEqual(assessment.reason, "WHOLE_SENTENCE_LOCATION_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
