import unittest
from unittest.mock import patch

from backend.chunking import Passage, split_sentences
from backend.viqa_api import (
    SENTENCE_FALLBACK_THRESHOLD,
    IndexedPassage,
    SearchHit,
    ask_question,
    choose_reader_output,
    concise_source_answer,
    expand_answer_to_sentence,
    format_display_answer,
    sentence_fallback_predict,
)


def make_hit(passage_id: str, text: str, raw: float, normalized: float) -> SearchHit:
    metadata = Passage("DOC", passage_id, "Title", "DOC_PAR0001", 0, 0, text)
    return SearchHit(IndexedPassage(metadata, tuple(), {}), raw, normalized)


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, question, context, no_answer_threshold):
        self.calls.append(context)
        if context == "retriever favorite":
            return {"answer": "weak", "confidence": 0.2, "score": 1.0, "null_score": 2.0, "score_margin": -1.0, "start": 0, "end": 4}
        return {"answer": "reader wins", "confidence": 0.9, "score": 5.0, "null_score": 1.0, "score_margin": 4.0, "start": 0, "end": 11}


class LowConfidencePredictor:
    def predict(self, question, context, no_answer_threshold):
        return {"answer": "weak span", "confidence": 0.04, "score": 1.0, "null_score": 4.0, "score_margin": -3.0, "start": 0, "end": 9}


class PipelineTests(unittest.TestCase):
    def test_stronger_definition_fallback_beats_a_barely_accepted_neural_span(self):
        question = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 ai?"
        neural = {
            "answer": "Ph\u1ea1m V\u0103n \u0110\u1ed3ng (1 th\u00e1ng 3 n\u0103m 1906 - 29",
            "confidence": 0.346,
            "start": 0,
            "end": 40,
        }
        fallback = {
            "answer": "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 Th\u1ee7 t\u01b0\u1edbng Vi\u1ec7t Nam.",
            "confidence": 0.70,
            "start": 0,
            "end": 45,
        }

        chosen = choose_reader_output(question, neural, fallback)

        self.assertEqual(chosen["method"], "sentence_fallback")
        self.assertEqual(chosen["confidence"], 0.70)

    def test_vietnamese_sentence_split_does_not_merge_the_next_sentence(self):
        text = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 Th\u1ee7 t\u01b0\u1edbng Vi\u1ec7t Nam. Tr\u01b0\u1edbc \u0111\u00f3 \u00f4ng gi\u1eef m\u1ed9t ch\u1ee9c v\u1ee5 kh\u00e1c."

        self.assertEqual(
            split_sentences(text),
            [
                "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 Th\u1ee7 t\u01b0\u1edbng Vi\u1ec7t Nam.",
                "Tr\u01b0\u1edbc \u0111\u00f3 \u00f4ng gi\u1eef m\u1ed9t ch\u1ee9c v\u1ee5 kh\u00e1c.",
            ],
        )

    def test_expands_reader_span_to_containing_sentence(self):
        context = "Paris is the capital of France. The city sits on the Seine river."

        answer = expand_answer_to_sentence(
            context,
            "Seine river",
            context.index("Seine"),
            context.index("river") + len("river"),
        )

        self.assertEqual(answer, "The city sits on the Seine river.")

    def test_definition_fallback_requires_relation_to_the_question_subject(self):
        question = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 ai?"
        biography = (
            "Ph\u1ea1m V\u0103n \u0110\u1ed3ng (1906-2000) l\u00e0 Th\u1ee7 t\u01b0\u1edbng \u0111\u1ea7u ti\u00ean c\u1ee7a "
            "n\u01b0\u1edbc C\u1ed9ng h\u00f2a X\u00e3 h\u1ed9i ch\u1ee7 ngh\u0129a Vi\u1ec7t Nam t\u1eeb n\u0103m 1976."
        )
        unrelated = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng c\u00f3 v\u1ee3 l\u00e0 b\u00e0 Ph\u1ea1m Th\u1ecb C\u00fac v\u00e0 c\u00f3 m\u1ed9t ng\u01b0\u1eddi con trai."

        good = sentence_fallback_predict(question, biography)
        bad = sentence_fallback_predict(question, unrelated)

        self.assertGreaterEqual(good["confidence"], SENTENCE_FALLBACK_THRESHOLD)
        self.assertLess(bad["confidence"], SENTENCE_FALLBACK_THRESHOLD)

    def test_definition_answer_is_complete_but_concise(self):
        question = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 ai?"
        source = (
            "Ph\u1ea1m V\u0103n \u0110\u1ed3ng (1 th\u00e1ng 3 n\u0103m 1906 - 29 th\u00e1ng 4 n\u0103m 2000) l\u00e0 Th\u1ee7 t\u01b0\u1edbng "
            "\u0111\u1ea7u ti\u00ean c\u1ee7a n\u01b0\u1edbc C\u1ed9ng h\u00f2a X\u00e3 h\u1ed9i ch\u1ee7 ngh\u0129a Vi\u1ec7t Nam t\u1eeb n\u0103m 1976."
        )

        answer = concise_source_answer(question, source)

        self.assertEqual(
            answer,
            "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 Th\u1ee7 t\u01b0\u1edbng \u0111\u1ea7u ti\u00ean c\u1ee7a n\u01b0\u1edbc C\u1ed9ng h\u00f2a X\u00e3 h\u1ed9i ch\u1ee7 ngh\u0129a Vi\u1ec7t Nam.",
        )

    def test_factoid_reader_answer_keeps_the_exact_span(self):
        output = {
            "method": "phobert",
            "answer": "29 th\u00e1ng 4 n\u0103m 2000",
            "start": 26,
            "end": 46,
        }

        answer = format_display_answer(
            "Ph\u1ea1m V\u0103n \u0110\u1ed3ng m\u1ea5t khi n\u00e0o?",
            "Ph\u1ea1m V\u0103n \u0110\u1ed3ng m\u1ea5t ng\u00e0y 29 th\u00e1ng 4 n\u0103m 2000 t\u1ea1i H\u00e0 N\u1ed9i.",
            output,
        )

        self.assertEqual(answer, "29 th\u00e1ng 4 n\u0103m 2000")

    def test_list_answer_removes_parenthetical_details(self):
        question = "Th\u1ef1c v\u1eadt c\u00f3 hoa \u0111\u01b0\u1ee3c chia nh\u01b0 n\u00e0o?"
        source = (
            "Th\u1ef1c v\u1eadt c\u00f3 hoa \u0111\u01b0\u1ee3c chia th\u00e0nh hai nh\u00f3m ch\u00ednh l\u00e0 Magnoliopsida "
            "(\u1edf c\u1ea5p \u0111\u1ed9 l\u1edbp) v\u00e0 Liliopsida (d\u1ef1a tr\u00ean t\u00ean g\u1ecdi Lilium)."
        )

        answer = concise_source_answer(question, source)

        self.assertEqual(
            answer,
            "Th\u1ef1c v\u1eadt c\u00f3 hoa \u0111\u01b0\u1ee3c chia th\u00e0nh hai nh\u00f3m ch\u00ednh l\u00e0 Magnoliopsida v\u00e0 Liliopsida.",
        )

    def test_reader_runs_on_every_top_k_and_reranks(self):
        hits = [
            make_hit("DOC_P0001", "retriever favorite", 12.0, 1.0),
            make_hit("DOC_P0002", "reader favorite", 8.0, 0.0),
        ]
        predictor = FakePredictor()
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=predictor):
            result = ask_question({"question": "test question", "retriever": "bm25", "reader": "phobert", "top_k": 2})

        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(result["selected_passage_id"], "DOC_P0002")
        self.assertEqual(result["answer"], "reader wins")
        self.assertEqual(result["answer_source"]["passage_id"], "DOC_P0002")
        self.assertEqual(result["scoring"]["retriever_weight"], 0.15)
        self.assertEqual(result["scoring"]["reader_weight"], 0.85)
        self.assertEqual(result["passages"][0]["passage_id"], "DOC_P0002")
        self.assertGreater(result["passages"][0]["final_score"], result["passages"][1]["final_score"])

    def test_low_reader_scores_return_no_answer_without_answer_source(self):
        hits = [
            make_hit("DOC_P0001", "top retrieved without useful overlap", 12.0, 1.0),
            make_hit("DOC_P0002", "second retrieved without useful overlap", 8.0, 0.7),
        ]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=LowConfidencePredictor()):
            result = ask_question({"question": "test question", "retriever": "bm25", "reader": "phobert", "top_k": 2})

        self.assertFalse(result["has_answer"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertIsNone(result["source"])
        self.assertIsNone(result["answer_source"])
        self.assertIsNone(result["selected_passage_id"])
        self.assertEqual(result["top_retrieved_passage"]["passage_id"], "DOC_P0001")
        self.assertEqual(result["best_reader_score"], 0.04)
        self.assertEqual(result["no_answer_reason"], "Reader confidence below threshold.")

    def test_sentence_fallback_recovers_answer_when_phobert_is_low_confidence(self):
        text = (
            "Thông tin mở đầu không liên quan. "
            "Theo truyền thống, thực vật có hoa được chia thành hai nhóm chính là thực vật hai lá mầm và thực vật một lá mầm."
        )
        hits = [make_hit("DOC_P0001", text, 12.0, 1.0)]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=LowConfidencePredictor()):
            result = ask_question({"question": "Thực vật có hoa được chia như nào?", "retriever": "bm25", "reader": "phobert", "top_k": 1})

        self.assertTrue(result["has_answer"])
        self.assertIn("thực vật có hoa được chia thành hai nhóm chính", result["answer"])
        self.assertEqual(result["answer_source"]["passage_id"], "DOC_P0001")
        self.assertEqual(result["passages"][0]["reader_method"], "sentence_fallback")
        self.assertGreaterEqual(result["passages"][0]["fallback_score"], 0.42)

    def test_unsupported_retriever_and_reader_raise_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "Retriever 'pyserini' is not implemented"):
            ask_question({"question": "test question", "retriever": "pyserini", "reader": "phobert", "top_k": 2})

        with self.assertRaisesRegex(ValueError, "Reader 'xlmr' is not implemented"):
            ask_question({"question": "test question", "retriever": "bm25", "reader": "xlmr", "top_k": 2})

    def test_production_backend_contains_no_question_answer_mapping(self):
        source = __import__("pathlib").Path("backend/viqa_api.py").read_text(encoding="utf-8")
        self.assertNotIn("Phạm Văn Đồng", source)
        self.assertNotIn("Falling back to heuristic", source)
        self.assertNotIn("reader_bias", source)


if __name__ == "__main__":
    unittest.main()
