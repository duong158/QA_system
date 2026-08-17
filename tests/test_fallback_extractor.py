import unittest

from reader.fallback_extractor import extract_fallback_answer, extract_location_candidate
from reader.question_type import QuestionType, assess_answer_type


class FallbackExtractorTests(unittest.TestCase):
    def assert_grounded_entity(
        self,
        question: str,
        sentence: str,
        expected: str,
        expected_method: str = "entity_relation_pattern",
    ) -> None:
        candidate = extract_fallback_answer(question, QuestionType.ENTITY, sentence)

        self.assertEqual(candidate.answer, expected)
        self.assertEqual(candidate.method, expected_method)
        self.assertNotEqual(candidate.answer, sentence)
        self.assertEqual(sentence[candidate.start_char : candidate.end_char], expected)
        self.assertEqual(candidate.evidence_sentence, sentence)
        self.assertGreaterEqual(
            assess_answer_type(QuestionType.ENTITY, candidate.answer).score,
            0.75,
        )

    def test_extracts_montmartre_entity_from_supporting_sentence(self):
        self.assert_grounded_entity(
            "Công trình nào nằm trên đỉnh Montmartre?",
            "Ở hữu ngạn: đồi Montmartre có độ cao là 131 mét, đỉnh là vị trí nhà thờ Saint-Pierre; Belleville cao 128,5 m.",
            "nhà thờ Saint-Pierre",
        )

    def test_extracts_capital_from_copular_relation(self):
        self.assert_grounded_entity(
            "Công trình nào là thủ đô của Việt Nam?",
            "Thủ đô của Việt Nam là Hà Nội.",
            "Hà Nội",
        )

    def test_extracts_named_building_from_copular_relation(self):
        self.assert_grounded_entity(
            "Công trình nổi bật nhất là công trình nào?",
            "Công trình nổi bật nhất là Nhà hát Lớn Hà Nội.",
            "Nhà hát Lớn Hà Nội",
        )

    def test_extracts_called_name_relation(self):
        self.assert_grounded_entity(
            "Tác phẩm nào được nhắc đến?",
            "Tác phẩm được gọi là Tấn trò đời.",
            "Tấn trò đời",
        )

    def test_extracts_location_relation_without_returning_whole_sentence(self):
        self.assert_grounded_entity(
            "Địa điểm này thuộc công trình nào?",
            "Địa điểm này nằm tại quận Hoàn Kiếm.",
            "quận Hoàn Kiếm",
        )

    def test_whole_sentence_is_last_resort_when_no_relation_phrase_exists(self):
        sentence = "Nhiều công trình lịch sử được bảo tồn qua nhiều thế kỷ."
        candidate = extract_fallback_answer(
            "Công trình nào được bảo tồn?",
            QuestionType.ENTITY,
            sentence,
        )

        self.assertEqual(candidate.answer, sentence)
        self.assertEqual(candidate.method, "whole_sentence")
        self.assertEqual((candidate.start_char, candidate.end_char), (0, len(sentence)))

    def test_comparison_hon_la_is_not_treated_as_entity_copula(self):
        sentence = (
            'Ông cho rằng tình trạng "vô chính phủ có tổ chức" '
            'nguy hiểm hơn là vô chính phủ thật sự.'
        )
        candidate = extract_fallback_answer(
            "Tình trạng vô chính phủ thật sự nguy hiểm hơn điều gì?",
            QuestionType.ENTITY,
            sentence,
        )

        self.assertEqual(candidate.answer, sentence)
        self.assertEqual(candidate.method, "whole_sentence")

    def test_time_question_extracts_exact_temporal_expression(self):
        sentence = "Vào thế kỷ 10, Paris đã là một thành phố chính của Pháp."
        candidate = extract_fallback_answer(
            "Paris trở thành thành phố quan trọng từ thế kỷ nào?",
            QuestionType.TIME,
            sentence,
        )

        self.assertEqual(candidate.answer, "thế kỷ 10")
        self.assertEqual(candidate.method, "temporal_expression_pattern")


    def test_extracts_event_location_with_relation_evidence(self):
        candidate = extract_location_candidate(
            "S\u1ef1 ki\u1ec7n di\u1ec5n ra \u1edf \u0111\u00e2u?",
            "S\u1ef1 ki\u1ec7n di\u1ec5n ra t\u1ea1i Paris.",
        )

        self.assertEqual(candidate.answer, "Paris")
        self.assertEqual(candidate.method, "location_relation_pattern")
        self.assertEqual(candidate.relation_type, "EVENT_LOCATION")
        self.assertTrue(candidate.relation_evidence)

    def test_event_mention_without_location_argument_is_not_a_location_phrase(self):
        sentence = (
            "M\u1ed9t c\u00f4ng tr\u00ecnh b\u1ecb ph\u00e1 khi n\u1ed5 ra "
            "C\u00e1ch m\u1ea1ng Ph\u00e1p."
        )
        candidate = extract_location_candidate(
            "C\u00e1ch m\u1ea1ng Ph\u00e1p di\u1ec5n ra \u1edf \u0111\u00e2u?",
            sentence,
        )

        self.assertEqual(candidate.method, "whole_sentence")
        self.assertFalse(candidate.relation_evidence)
        self.assertEqual(candidate.relation_score, 0.0)

    def test_extracts_object_location_and_keeps_compound_place(self):
        candidate = extract_location_candidate(
            "Th\u00e1p Eiffel n\u1eb1m \u1edf \u0111\u00e2u?",
            "Th\u00e1p Eiffel n\u1eb1m t\u1ea1i Paris, Ph\u00e1p.",
        )

        self.assertEqual(candidate.answer, "Paris, Ph\u00e1p")
        self.assertEqual(candidate.relation_type, "OBJECT_LOCATION")
        self.assertTrue(candidate.relation_evidence)

    def test_extracts_birth_location_and_trims_temporal_tail(self):
        candidate = extract_location_candidate(
            "\u00d4ng sinh \u1edf \u0111\u00e2u?",
            "\u00d4ng sinh t\u1ea1i H\u00e0 N\u1ed9i n\u0103m 1940.",
        )

        self.assertEqual(candidate.answer, "H\u00e0 N\u1ed9i")
        self.assertEqual(candidate.relation_type, "BIRTH_LOCATION")
        self.assertTrue(candidate.relation_evidence)

    def test_extracts_plain_preposition_location(self):
        candidate = extract_location_candidate(
            "H\u1ed3 Ho\u00e0n Ki\u1ebfm \u1edf \u0111\u00e2u?",
            "H\u1ed3 Ho\u00e0n Ki\u1ebfm \u1edf qu\u1eadn Ho\u00e0n Ki\u1ebfm, H\u00e0 N\u1ed9i.",
        )

        self.assertEqual(candidate.answer, "qu\u1eadn Ho\u00e0n Ki\u1ebfm, H\u00e0 N\u1ed9i")
        self.assertEqual(candidate.relation_type, "OBJECT_LOCATION")
        self.assertTrue(candidate.relation_evidence)

    def test_extracts_thuoc_location_relation(self):
        candidate = extract_location_candidate(
            "Khu di t\u00edch thu\u1ed9c t\u1ec9nh n\u00e0o?",
            "Khu di t\u00edch thu\u1ed9c t\u1ec9nh Ninh B\u00ecnh.",
        )

        self.assertEqual(candidate.answer, "t\u1ec9nh Ninh B\u00ecnh")
        self.assertEqual(candidate.relation_type, "OBJECT_LOCATION")
        self.assertTrue(candidate.relation_evidence)

    def test_proper_nouns_are_not_extracted_without_location_relation(self):
        sentence = "Victor Hugo vi\u1ebft v\u1ec1 C\u00e1ch m\u1ea1ng Ph\u00e1p."
        candidate = extract_location_candidate(
            "C\u00e1ch m\u1ea1ng Ph\u00e1p di\u1ec5n ra \u1edf \u0111\u00e2u?",
            sentence,
        )

        self.assertEqual(candidate.method, "whole_sentence")
        self.assertFalse(candidate.relation_evidence)
        self.assertNotIn(candidate.answer, {"Victor Hugo", "C\u00e1ch m\u1ea1ng Ph\u00e1p"})


if __name__ == "__main__":
    unittest.main()
