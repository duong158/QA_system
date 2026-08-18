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
                self.assertEqual(detect_question_type(question), [expected])

    def test_time_scoring_supports_centuries_years_and_roman_numerals(self):
        self.assertEqual(assess_answer_type(QuestionType.TIME, "Vào thế kỷ 10").score, 1.0)
        self.assertEqual(assess_answer_type(QuestionType.TIME, "thế kỷ XIX").score, 1.0)
        self.assertEqual(assess_answer_type(QuestionType.TIME, "cuối thế kỷ XIX").score, 1.0)
        self.assertEqual(assess_answer_type(QuestionType.TIME, "thế kỷ thứ mười").score, 1.0)
        self.assertEqual(assess_answer_type([QuestionType.TIME], "Vào thế kỷ 10").score, 1.0)
        self.assertEqual(assess_answer_type([QuestionType.TIME], "thế kỷ XIX").score, 1.0)
        self.assertEqual(assess_answer_type([QuestionType.TIME], "cuối thế kỷ XIX").score, 1.0)
        self.assertEqual(assess_answer_type([QuestionType.TIME], "thế kỷ thứ mười").score, 1.0)
        self.assertGreaterEqual(
            assess_answer_type([QuestionType.TIME], "Ngày 2 tháng 9 năm 1945").score,
            0.9,
        )

    def test_time_question_penalizes_non_temporal_parisien_candidate(self):
        temporal = assess_answer_type([QuestionType.TIME], "1945")
        wrong_type = assess_answer_type([QuestionType.TIME], "nhà toán học")
        self.assertGreater(temporal.score, wrong_type.score)
        self.assertEqual(wrong_type.score, 0.0)
        self.assertFalse(wrong_type.matched)
        self.assertEqual(wrong_type.reason, "NO_TEMPORAL_EXPRESSION")

    def test_number_person_and_location_scoring_are_soft(self):
        self.assertEqual(assess_answer_type([QuestionType.NUMBER], "63 tỉnh thành").score, 1.0)
        self.assertGreaterEqual(assess_answer_type([QuestionType.PERSON], "Nguyễn Du").score, 0.8)
        self.assertGreaterEqual(assess_answer_type([QuestionType.LOCATION], "Paris").score, 0.8)
        self.assertGreater(
            assess_answer_type([QuestionType.LOCATION], "một khu vực ở phía bắc").score,
            0.0,
        )


    def test_event_names_score_below_their_location_arguments(self):
        cases = [
            ("Cách mạng Pháp", "Pháp"),
            ("Chiến tranh Việt Nam", "Việt Nam"),
            ("Hội nghị Genève", "Genève"),
        ]
        for event, location in cases:
            with self.subTest(event=event, location=location):
                event_assessment = assess_answer_type([QuestionType.LOCATION], event)
                location_assessment = assess_answer_type([QuestionType.LOCATION], location)
                self.assertLess(event_assessment.score, location_assessment.score)

    def test_whole_sentence_location_candidate_is_penalized(self):
        phrase = assess_answer_type([QuestionType.LOCATION], "Hà Nội")
        sentence = assess_answer_type(
            [QuestionType.LOCATION],
            "Hà Nội là thủ đô của Việt Nam.",
            candidate_method="whole_sentence",
        )
        self.assertGreater(phrase.score, sentence.score)
        self.assertLessEqual(sentence.score, 0.45)
        self.assertEqual(sentence.reason, "WHOLE_SENTENCE_LOCATION_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
