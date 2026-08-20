import unittest
from unittest.mock import patch

from reader.question_type import (
    QuestionType,
    _detect_question_type_cached,
    detect_question_type,
)


class QuestionTypeLatencyTests(unittest.TestCase):
    def setUp(self):
        _detect_question_type_cached.cache_clear()

    def tearDown(self):
        _detect_question_type_cached.cache_clear()

    def test_explicit_person_question_never_loads_zero_shot_model(self):
        with patch(
            "reader.question_type._get_classifier",
            side_effect=AssertionError("classifier should not load"),
        ):
            self.assertEqual(
                detect_question_type("Phạm Văn Đồng là ai?"),
                [QuestionType.PERSON],
            )

    def test_explicit_time_and_location_questions_use_fast_rules(self):
        with patch(
            "reader.question_type._get_classifier",
            side_effect=AssertionError("classifier should not load"),
        ):
            self.assertIn(QuestionType.TIME, detect_question_type("Nhân vật sinh năm nào?"))
            self.assertIn(QuestionType.LOCATION, detect_question_type("Sự kiện diễn ra ở đâu?"))
            self.assertEqual(
                detect_question_type("Paris nằm bên bờ con sông nào?"),
                [QuestionType.LOCATION],
            )

    def test_ambiguous_question_classification_is_cached(self):
        calls = []

        class FakeClassifier:
            def __call__(self, question, labels, multi_label):
                calls.append(question)
                return {"labels": labels, "scores": [0.9, *([0.0] * (len(labels) - 1))]}

        with patch("reader.question_type._get_classifier", return_value=FakeClassifier()):
            first = detect_question_type("Hãy trình bày nội dung này.")
            second = detect_question_type("Hãy trình bày nội dung này.")

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
