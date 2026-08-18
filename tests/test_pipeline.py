import unittest
from unittest.mock import patch

from backend.chunking import Passage, split_sentences
from backend.viqa_api import (
    MIN_FALLBACK_ANSWER_TYPE_SCORE,
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

    def predict(self, question, context, no_answer_threshold, **kwargs):
        self.calls.append(context)
        if context == "retriever favorite":
            return {"answer": "retriever", "confidence": 0.2, "score": 1.0, "null_score": 2.0, "score_margin": -1.0, "start": 0, "end": 9}
        return {"answer": "reader favorite", "confidence": 0.9, "score": 5.0, "null_score": 1.0, "score_margin": 4.0, "start": 0, "end": 15}


class LowConfidencePredictor:
    def predict(self, question, context, no_answer_threshold, **kwargs):
        answer = context.split()[0]
        return {"answer": answer, "confidence": 0.04, "score": 1.0, "null_score": 4.0, "score_margin": -3.0, "start": 0, "end": len(answer)}


class BatchPredictor:
    def __init__(self):
        self.calls = []

    def predict_many(self, question, contexts, no_answer_threshold, **kwargs):
        self.calls.append((question, list(contexts)))
        return [
            {
                "answer": context,
                "confidence": 0.8,
                "score": 4.0,
                "null_score": 1.0,
                "score_margin": 3.0,
                "start": 0,
                "end": len(context),
            }
            for context in contexts
        ]


class NoAnswerPredictor:
    def predict_many(self, question, contexts, no_answer_threshold, **kwargs):
        return [
            {
                "answer": "",
                "best_span_answer": "không liên quan",
                "confidence": 0.30,
                "score": 1.0,
                "null_score": 5.0,
                "score_margin": -4.0,
                "start": -1,
                "end": -1,
                "has_answer": False,
            }
            for _ in contexts
        ]


class MontmartreWrongSpanPredictor:
    def predict_many(self, question, contexts, no_answer_threshold, **kwargs):
        results = []
        for context in contexts:
            answer = "đồi Montmartre"
            start = context.index(answer)
            results.append(
                {
                    "answer": answer,
                    "best_span_answer": answer,
                    "confidence": 0.466266,
                    "score": 7.325768,
                    "best_span_score": 7.325768,
                    "start_score": 3.4,
                    "end_score": 3.925768,
                    "null_score": 8.677185,
                    "no_answer_score": 1.351417,
                    "score_margin": -1.351417,
                    "start": start,
                    "end": start + len(answer),
                    "has_answer": True,
                }
            )
        return results


class BelowThresholdContrastPredictor:
    def predict_many(self, question, contexts, no_answer_threshold, **kwargs):
        answer = (
            "sự khác biệt giữa các chức năng học thực sự (như ở đây) và "
            "phần mềm thiên về mặt giáo dục (được trình bay ở sau)"
        )
        results = []
        for context in contexts:
            start = context.index(answer)
            results.append(
                {
                    # This mirrors the new checkpoint failure: the best span is
                    # useful even though its null-score margin misses threshold.
                    "answer": "",
                    "candidate_answer": answer,
                    "candidate_start": start,
                    "candidate_end": start + len(answer),
                    "confidence": 0.468,
                    "reader_threshold_score": 0.468,
                    "score_margin": -1.276,
                    "passes_reader_threshold": False,
                    "valid_span": True,
                    "start": -1,
                    "end": -1,
                }
            )
        return results


class TruncatedAliasPredictor:
    def predict_many(self, question, contexts, no_answer_threshold, **kwargs):
        results = []
        for context in contexts:
            answer = "là Lâm Bá"
            start = context.index(answer)
            results.append(
                {
                    "answer": answer,
                    "candidate_answer": answer,
                    "candidate_start": start,
                    "candidate_end": start + len(answer),
                    "confidence": 0.73,
                    "reader_threshold_score": 0.73,
                    "score_margin": 1.22,
                    "passes_reader_threshold": True,
                    "valid_span": True,
                    "start": start,
                    "end": start + len(answer),
                }
            )
        return results


class PipelineTests(unittest.TestCase):
    def test_complete_alias_phrase_beats_truncated_neural_name(self):
        question = (
            "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm "
            "cơ quan Biện sự xứ tại Quế Lâm?"
        )
        context = (
            "Ông còn có tên gọi là Lâm Bá Kiệt khi làm Phó chủ nhiệm cơ quan "
            "Biện sự xứ tại Quế Lâm (Chủ nhiệm là Hồ Học Lãm)."
        )
        with patch(
            "backend.viqa_api.INDEX.retrieve",
            return_value=[make_hit("doc_00001_P0001", context, 12.0, 1.0)],
        ), patch(
            "backend.viqa_api.READERS.get",
            return_value=TruncatedAliasPredictor(),
        ):
            result = ask_question(
                {
                    "question": question,
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 1,
                }
            )

        self.assertEqual(result["question_type"], "ENTITY")
        self.assertEqual(result["answer"], "Lâm Bá Kiệt")
        self.assertEqual(result["reader_method"], "neural_span")
        self.assertEqual(result["answer_refinement"]["raw_answer"], "là Lâm Bá")
        self.assertEqual(result["answer_refinement"]["refined_answer"], "Lâm Bá Kiệt")
        self.assertTrue(result["answer_refinement"]["changed"])
        self.assertIsNone(result["fallback_method"])
        self.assertEqual(result["reader_candidate"]["raw_text"], "là Lâm Bá")
        self.assertEqual(result["reader_candidate"]["text"], "Lâm Bá Kiệt")
        self.assertIsNone(result["reader_candidate"]["rejection_reason"])
        self.assertEqual(result["fallback_candidate"]["relation_type"], "ALIAS")

    def test_generic_sentence_fallback_cannot_overwrite_valid_neural_span(self):
        question = "Địa danh nào được nhắc đến?"
        context = "Hà Nội là thủ đô của Việt Nam."
        neural = {
            "answer": "Hà Nội",
            "confidence": 0.45,
            "start": 0,
            "end": len("Hà Nội"),
        }
        fallback = {
            "answer": context,
            "confidence": 0.80,
            "start": 0,
            "end": len(context),
            "fallback_method": "whole_sentence",
            "phrase_score": 0.4,
        }

        chosen = choose_reader_output(question, context, neural, fallback)

        self.assertEqual(chosen["method"], "neural_span")
        self.assertEqual(chosen["answer"], "Hà Nội")

    def test_below_threshold_contrast_span_survives_until_final_ranking(self):
        question = (
            "Các bậc phụ huynh cần biết rõ sự khác biệt nào để làm căn cứ "
            "lựa chọn phần mềm giáo dục cho trẻ?"
        )
        context = (
            "Việc thiết kế các phần mềm giáo dục tại nhà đã bị ảnh hưởng mạnh mẽ bởi "
            "khái niệm trò chơi trên máy tính. Tuy nhiên, ở mức độ nhất định thì cần "
            "thấy rõ sự khác biệt giữa các chức năng học thực sự (như ở đây) và phần "
            "mềm thiên về mặt giáo dục (được trình bay ở sau). Các bậc phụ huynh cần "
            "phải biết rõ sự khác biệt này để làm căn cứ lựa chọn."
        )
        hits = [make_hit("doc_00031_P0001", context, 12.0, 1.0)]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=BelowThresholdContrastPredictor()
        ):
            result = ask_question(
                {
                    "question": question,
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 1,
                }
            )

        self.assertTrue(result["has_answer"])
        self.assertIn("chức năng học thực sự", result["answer"])
        self.assertIn("phần mềm thiên về mặt giáo dục", result["answer"])
        neural_candidates = [
            candidate
            for candidate in result["passages"][0]["candidates"]
            if candidate["method"] == "neural_span"
        ]
        self.assertEqual(len(neural_candidates), 1)
        self.assertFalse(neural_candidates[0]["passes_reader_threshold"])
        self.assertTrue(neural_candidates[0]["valid_span"])

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

        context = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng (1 th\u00e1ng 3 n\u0103m 1906 - 29 th\u00e1ng 4 n\u0103m 2000) l\u00e0 Th\u1ee7 t\u01b0\u1edbng Vi\u1ec7t Nam."
        chosen = choose_reader_output(question, context, neural, fallback)

        self.assertEqual(chosen["method"], "sentence_fallback")
        self.assertEqual(chosen["confidence"], 0.70)

    def test_neural_span_cut_inside_a_word_is_rejected(self):
        question = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng l\u00e0 ai?"
        context = "Ph\u1ea1m V\u0103n \u0110\u1ed3ng c\u00f3 v\u1ee3 l\u00e0 b\u00e0 Ph\u1ea1m Th\u1ecb C\u00fac."
        start = context.index("v\u1ee3") + 1
        neural = {
            "answer": context[start : start + 15],
            "confidence": 0.63,
            "start": start,
            "end": start + 15,
        }
        fallback = {"answer": "", "confidence": 0.35, "start": -1, "end": -1}

        chosen = choose_reader_output(question, context, neural, fallback)

        self.assertEqual(chosen["method"], "no_answer")
        self.assertEqual(chosen["confidence"], 0.0)
        self.assertIsNone(chosen["answer"])

    def test_definition_question_rejects_an_unrelated_neural_relation(self):
        question = "Phạm Văn Đồng là ai?"
        context = "Phạm Văn Đồng có vợ là bà Phạm Thị Cúc."
        start = context.index("vợ")
        answer = "vợ là bà Phạm Thị Cúc"
        neural = {
            "answer": answer,
            "confidence": 0.95,
            "start": start,
            "end": start + len(answer),
        }
        fallback = {"answer": "", "confidence": 0.35, "start": -1, "end": -1}

        chosen = choose_reader_output(question, context, neural, fallback)

        self.assertEqual(chosen["method"], "no_answer")
        self.assertIsNone(chosen["answer"])

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
            make_hit("DOC_P0002", "reader favorite", 8.0, 0.75),
        ]
        predictor = FakePredictor()
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=predictor):
            result = ask_question({"question": "test question", "retriever": "bm25", "reader": "phobert", "top_k": 2})

        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(result["selected_passage_id"], "DOC_P0002")
        self.assertEqual(result["answer"], "reader favorite")
        self.assertEqual(result["answer_source"]["passage_id"], "DOC_P0002")
        self.assertEqual(result["scoring"]["retriever_weight"], 0.4)
        self.assertEqual(result["scoring"]["reader_weight"], 0.4)
        self.assertEqual(result["scoring"]["answer_type_weight"], 0.2)
        self.assertEqual(result["scoring"]["relation_weight"], 0.0)
        self.assertEqual(result["passages"][0]["passage_id"], "DOC_P0002")
        self.assertGreater(result["passages"][0]["ranking_score"], result["passages"][1]["ranking_score"])
        self.assertIsNone(result["answer_confidence"])

    def test_reader_passages_are_sent_in_one_batch(self):
        hits = [
            make_hit("DOC_P0001", "first useful answer", 12.0, 1.0),
            make_hit("DOC_P0002", "second useful answer", 8.0, 0.5),
        ]
        predictor = BatchPredictor()
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=predictor
        ):
            result = ask_question(
                {"question": "test question", "retriever": "bm25", "reader": "phobert", "top_k": 2}
            )

        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(predictor.calls[0][1], ["first useful answer", "second useful answer"])
        self.assertEqual(len(result["passages"]), 2)

    def test_low_reader_scores_return_no_answer_without_answer_source(self):
        hits = [
            make_hit("DOC_P0001", "top retrieved without useful overlap", 12.0, 1.0),
            make_hit("DOC_P0002", "second retrieved without useful overlap", 8.0, 0.7),
        ]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=LowConfidencePredictor()):
            result = ask_question({"question": "test question", "retriever": "bm25", "reader": "phobert", "top_k": 2})

        self.assertFalse(result["has_answer"])
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["source"])
        self.assertIsNone(result["answer_source"])
        self.assertIsNone(result["selected_passage_id"])
        self.assertEqual(result["top_retrieved_passage"]["passage_id"], "DOC_P0001")
        self.assertEqual(result["best_reader_score"], 0.04)
        self.assertEqual(result["rejection_reason"], "LOW_RANKING_SCORE")

    def test_sentence_fallback_recovers_answer_when_phobert_is_low_confidence(self):
        text = (
            "Thông tin mở đầu không liên quan. "
            "Theo truyền thống, thực vật có hoa được chia thành hai nhóm chính là thực vật hai lá mầm và thực vật một lá mầm."
        )
        hits = [make_hit("DOC_P0001", text, 12.0, 1.0)]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch("backend.viqa_api.READERS.get", return_value=LowConfidencePredictor()):
            result = ask_question({"question": "Thực vật có hoa được chia như nào?", "retriever": "bm25", "reader": "phobert", "top_k": 1})

        self.assertTrue(result["has_answer"])
        self.assertEqual(result["answer"], "thực vật hai lá mầm và thực vật một lá mầm")
        self.assertEqual(result["answer_source"]["passage_id"], "DOC_P0001")
        self.assertEqual(result["passages"][0]["reader_method"], "phrase_fallback")
        self.assertEqual(result["passages"][0]["fallback_method"], "entity_relation_pattern")
        self.assertGreaterEqual(result["passages"][0]["fallback_score"], 0.42)

    def test_paris_temporal_regression_rejects_parisien_and_selects_temporal_evidence(self):
        wrong = (
            'Cùng với Venezia, Paris còn được ví là "Thành phố của tình yêu". '
            'Từ "parisien" trong tiếng Pháp là tính từ của Paris, cũng là danh từ để chỉ '
            'những người dân của thành phố này.'
        )
        correct = (
            "Paris nằm ở điểm gặp nhau của các hành trình thương mại. "
            "Vào thế kỷ 10, Paris đã là một trong những thành phố chính của Pháp."
        )
        hits = [
            make_hit("DOC_P0001", wrong, 15.1, 1.0),
            make_hit("DOC_P0002", correct, 14.8, 0.93),
        ]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=NoAnswerPredictor()
        ):
            result = ask_question(
                {
                    "question": "Paris trở thành thành phố quan trọng của Pháp từ thế kỷ nào?",
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 10,
                }
            )

        by_id = {passage["passage_id"]: passage for passage in result["passages"]}
        self.assertEqual(result["question_type"], "TIME")
        self.assertTrue(result["has_answer"])
        self.assertEqual(result["selected_passage_id"], "DOC_P0002")
        self.assertIn("thế kỷ 10", result["answer"])
        self.assertEqual(by_id["DOC_P0001"]["answer_type_score"], 0.0)
        self.assertEqual(by_id["DOC_P0001"]["rejection_reason"], "SPAN_BOUNDARY_INCOMPLETE")
        self.assertEqual(by_id["DOC_P0002"]["answer_type_score"], 1.0)
        self.assertIsNone(result["scores"]["answer_confidence"])

    def test_montmartre_fallback_extracts_grounded_entity_and_passes_existing_gate(self):
        text = (
            "Tại trung tâm của bồn Paris, Paris nằm hai bên bờ sông Seine. "
            "Ở hữu ngạn: đồi Montmartre có độ cao là 131 mét, đỉnh là vị trí "
            "nhà thờ Saint-Pierre; Belleville cao 128,5 m."
        )
        hits = [make_hit("DOC_P0001", text, 12.0, 1.0)]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=MontmartreWrongSpanPredictor()
        ):
            result = ask_question(
                {
                    "question": "Công trình nào nằm trên đỉnh Montmartre?",
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 10,
                }
            )

        candidate = result["passages"][0]
        self.assertEqual(MIN_FALLBACK_ANSWER_TYPE_SCORE, 0.75)
        self.assertEqual(result["question_type"], "ENTITY")
        self.assertTrue(result["has_answer"])
        self.assertEqual(result["answer"], "nhà thờ Saint-Pierre")
        self.assertEqual(result["fallback_method"], "entity_relation_pattern")
        self.assertEqual(candidate["reader_method"], "phrase_fallback")
        self.assertEqual(candidate["fallback_answer"], "nhà thờ Saint-Pierre")
        self.assertNotEqual(candidate["fallback_answer"], candidate["fallback_sentence"])
        self.assertGreaterEqual(candidate["answer_type_score"], MIN_FALLBACK_ANSWER_TYPE_SCORE)
        self.assertTrue(candidate["evidence_supported"])
        span = candidate["answer_span"]
        self.assertEqual(text[span["start"] : span["end"]], "nhà thờ Saint-Pierre")
        self.assertIsNone(candidate["rejection_reason"])

    def test_general_false_premise_does_not_accept_sentence_fallback(self):
        text = (
            "Paris có nhà hàng Jules Verne nằm trên tầng hai của tháp Eiffel. "
            "Mặt Trăng là vệ tinh tự nhiên của Trái Đất."
        )
        with patch(
            "backend.viqa_api.INDEX.retrieve",
            return_value=[make_hit("DOC_P0001", text, 12.0, 1.0)],
        ), patch("backend.viqa_api.READERS.get", return_value=NoAnswerPredictor()):
            result = ask_question(
                {
                    "question": "Tháp Eiffel nằm trên Mặt Trăng phải không?",
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 10,
                }
            )

        self.assertFalse(result["has_answer"])
        self.assertIsNone(result["answer"])
        self.assertIsNone(result["answer_source"])
        self.assertEqual(result["rejection_reason"], "INSUFFICIENT_FALLBACK_EVIDENCE")

    def test_location_relation_rejects_topic_overlap_and_selects_lower_grounded_phrase(self):
        wrong = (
            "M\u1ed9t ph\u1ea7n c\u1ee7a c\u00f4ng tr\u00ecnh \u0111\u00e3 t\u1eebng l\u00e0 nh\u00e0 t\u00f9 "
            "khi n\u1ed5 ra C\u00e1ch m\u1ea1ng Ph\u00e1p."
        )
        correct = (
            "Th\u1ebf k\u1ef7 14, Paris l\u00e0 th\u00e0nh ph\u1ed1 quan tr\u1ecdng v\u00e0 "
            "\u0111\u00e2y l\u00e0 n\u01a1i di\u1ec5n ra C\u00e1ch m\u1ea1ng Ph\u00e1p."
        )
        hits = [
            make_hit("DOC_P0001", wrong, 12.0, 1.0),
            make_hit("DOC_P0002", correct, 8.0, 0.35),
        ]
        with patch("backend.viqa_api.INDEX.retrieve", return_value=hits), patch(
            "backend.viqa_api.READERS.get", return_value=NoAnswerPredictor()
        ):
            result = ask_question(
                {
                    "question": "C\u00e1ch m\u1ea1ng Ph\u00e1p di\u1ec5n ra \u1edf \u0111\u00e2u?",
                    "retriever": "bm25",
                    "reader": "phobert",
                    "top_k": 10,
                }
            )

        by_id = {passage["passage_id"]: passage for passage in result["passages"]}
        self.assertTrue(result["has_answer"])
        self.assertEqual(result["answer"], "Paris")
        self.assertEqual(result["selected_passage_id"], "DOC_P0002")
        self.assertEqual(by_id["DOC_P0001"]["rejection_reason"], "LOCATION_RELATION_MISMATCH")
        self.assertFalse(by_id["DOC_P0001"]["relation_evidence"])
        self.assertEqual(by_id["DOC_P0001"]["relation_score"], 0.0)
        self.assertEqual(by_id["DOC_P0002"]["fallback_method"], "location_relation_pattern")
        self.assertTrue(by_id["DOC_P0002"]["relation_evidence"])
        self.assertGreaterEqual(by_id["DOC_P0002"]["answer_type_score"], 0.75)

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
