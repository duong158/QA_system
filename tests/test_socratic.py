import json
import ast
import unittest
from dataclasses import replace
from pathlib import Path

from backend.socratic import (
    SOCRATIC_CONFIG,
    SocraticConfig,
    generate_followup_response,
    generate_followups,
    normalize_question,
    question_similarity,
)


ROOT = Path(__file__).resolve().parents[1]


def passage(passage_id: str, text: str, score: float = 1.0) -> dict:
    return {
        "passage_id": passage_id,
        "text": text,
        "retrieval_score_normalized": score,
    }


NO_PROBE_CONFIG = replace(SOCRATIC_CONFIG, allow_bm25_probe=False)


class SocraticGeneratorTests(unittest.TestCase):
    def test_same_relation_is_not_suggested(self):
        source = passage(
            "DOC_P0001",
            "Cây lúa phát triển nhanh do nguồn nước dồi dào. "
            "Cây lúa được trồng tại vùng đồng bằng.",
        )
        followups = generate_followups(
            "Vì sao cây lúa phát triển nhanh?",
            "Do nguồn nước dồi dào.",
            {"subject": "Cây lúa", "relation": "CAUSE"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertNotIn("CAUSE", {item.relation for item in followups})

    def test_semantic_duplicates_are_collapsed(self):
        sources = [
            passage("DOC_P0001", "Nhà khoa học Nguyễn Văn An sinh năm 1945."),
            passage("DOC_P0002", "Nguyễn Văn An chào đời vào năm 1945.", 0.9),
        ]
        followups = generate_followups(
            "Nguyễn Văn An là ai?",
            "Một nhà khoa học.",
            {"subject": "Nguyễn Văn An", "relation": "IDENTITY"},
            sources[0],
            sources[1:],
            config=NO_PROBE_CONFIG,
        )
        birth_questions = [item for item in followups if item.relation == "BIRTH_TIME"]
        self.assertEqual(len(birth_questions), 1)

    def test_grounded_unregistered_fact_uses_general_evidence_fallback(self):
        source = passage(
            "DOC_P0001",
            "Nguyễn Văn An yêu thích việc đọc sách và thường xuyên ghi chép.",
        )
        followups = generate_followups(
            "Nguyễn Văn An là ai?",
            "Một người được nhắc đến trong tài liệu.",
            {"subject": "Nguyễn Văn An", "relation": "IDENTITY"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual([item.relation for item in followups], ["EVIDENCE_DETAIL"])
        self.assertEqual(followups[0].source_passage_id, "DOC_P0001")

    def test_tier_one_candidate_does_not_depend_on_probe(self):
        source = passage("DOC_P0001", "Nguyễn Văn An sinh năm 1945.")
        config = replace(NO_PROBE_CONFIG, allow_bm25_probe=True)
        followups = generate_followups(
            "Nguyễn Văn An là ai?",
            "Một nhà giáo.",
            {"subject": "Nguyễn Văn An", "relation": "IDENTITY"},
            None,
            [{**source, "retrieval_score_normalized": 0.5}],
            probe=lambda _question, _top_k: [],
            config=config,
        )
        self.assertEqual([item.relation for item in followups], ["BIRTH_TIME"])

    def test_relation_rich_passage_discovers_time_cause_and_purpose(self):
        source = passage(
            "DOC_P0001",
            "Sự kiện X diễn ra tại Y vào năm 1950 do Z và nhằm mục đích Q.",
        )
        followups = generate_followups(
            "Sự kiện X diễn ra ở đâu?",
            "Tại Y.",
            {"subject": "Sự kiện X", "relation": "EVENT_LOCATION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual(
            {item.relation for item in followups},
            {"EVENT_TIME", "CAUSE", "PURPOSE"},
        )

    def test_relation_sparse_passage_returns_grounded_review_followup(self):
        source = passage("DOC_P0001", "Sự kiện X diễn ra tại Y.")
        followups = generate_followups(
            "Sự kiện X diễn ra ở đâu?",
            "Tại Y.",
            {"subject": "Sự kiện X", "relation": "EVENT_LOCATION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual(len(followups), 1)
        self.assertEqual(followups[0].relation, "EVIDENCE_DETAIL")
        self.assertEqual(followups[0].source_passage_id, "DOC_P0001")
        self.assertTrue(followups[0].evidence_sentence)

    def test_nominal_question_focus_is_reduced_to_the_real_entity(self):
        source = passage(
            "DOC_P0001",
            "Chức năng của hoa là hỗ trợ sinh sản. Hoa có thể sinh ra ở đầu ngọn hoặc ở nách lá.",
        )
        followups = generate_followups(
            "Chức năng của hoa là gì?",
            "Hỗ trợ sinh sản.",
            {"subject": "Chức năng của hoa", "relation": "DEFINITION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        location = next(item for item in followups if item.relation == "PROCESS_LOCATION")
        self.assertEqual(location.subject.casefold(), "hoa")
        self.assertEqual(location.question, "Hoa sinh ra ở đâu?")

    def test_spatial_process_rule_generalizes_across_domains(self):
        cases = (
            ("Cây lúa", "Cây lúa được trồng ở đồng bằng.", "Cây lúa được trồng ở đâu?"),
            ("Loài chim này", "Loài chim này phân bố ở Đông Nam Á.", "Loài chim này phân bố ở đâu?"),
            ("Khoáng vật X", "Khoáng vật X được tìm thấy ở vùng núi.", "Khoáng vật X được tìm thấy ở đâu?"),
        )
        for subject, evidence, expected_question in cases:
            with self.subTest(subject=subject):
                source = passage("DOC_P0001", evidence)
                followups = generate_followups(
                    f"{subject} là gì?",
                    "Một đối tượng được mô tả trong tài liệu.",
                    {"subject": subject, "relation": "IDENTITY"},
                    source,
                    [],
                    config=NO_PROBE_CONFIG,
                )
                candidate = next(item for item in followups if item.relation == "PROCESS_LOCATION")
                self.assertEqual(candidate.question, expected_question)

    def test_one_token_subject_rejects_diacritic_collisions_and_topic_drift(self):
        selected = passage(
            "FLOWER_P0001",
            "Chức năng của hoa là hỗ trợ sinh sản. Hoa sinh ra ở đầu ngọn.",
        )
        unrelated = [
            passage("GENOME_P0001", "Dự án được toàn cầu hóa nhằm tích hợp kiến thức sinh học.", 1.0),
            passage("WATER_P0001", "Hoa Kỳ tham gia thành lập một cơ quan quản lý nước.", 0.9),
        ]
        followups = generate_followups(
            "Chức năng của hoa là gì?",
            "Hỗ trợ sinh sản.",
            {"subject": "Chức năng của hoa", "relation": "DEFINITION"},
            selected,
            unrelated,
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual({item.relation for item in followups}, {"PROCESS_LOCATION"})
        self.assertTrue(all(item.source_passage_id == "FLOWER_P0001" for item in followups))

    def test_non_person_subject_does_not_generate_activity_or_role_prompt(self):
        source = passage(
            "DOC_P0001",
            "Paris tham gia vào mạng lưới giao thông và giữ vai trò trung tâm của khu vực.",
        )
        followups = generate_followups(
            "Paris nằm ở đâu?",
            "Tại Pháp.",
            {"subject": "Paris", "relation": "OBJECT_LOCATION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertFalse({"EVENT", "ROLE"} & {item.relation for item in followups})

    def test_additional_grounded_fact_prevents_empty_followups_for_unknown_relations(self):
        source = passage(
            "DOC_P0001",
            "Năng lượng mặt trời có nguồn cung dồi dào. "
            "Năng lượng mặt trời giúp giảm nhu cầu sử dụng nhiên liệu hóa thạch.",
        )
        followups = generate_followups(
            "Đặc điểm của năng lượng mặt trời là gì?",
            "Có nguồn cung dồi dào.",
            {"subject": "Đặc điểm của năng lượng mặt trời", "relation": "DEFINITION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual([item.relation for item in followups], ["EVIDENCE_DETAIL"])
        self.assertIn("năng lượng mặt trời", followups[0].question.casefold())

    def test_previous_sentence_coreference_discovers_birth_time(self):
        source = passage(
            "DOC_P0001",
            "Nguyễn Văn A là một nhà khoa học. Ông sinh năm 1920.",
        )
        followups = generate_followups(
            "Nguyễn Văn A là ai?",
            "Một nhà khoa học.",
            {"subject": "Nguyễn Văn A", "relation": "IDENTITY"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertIn("BIRTH_TIME", {item.relation for item in followups})

    def test_retrieved_passage_can_supply_opportunity_missing_from_selected(self):
        selected = passage("DOC_P0001", "Nguyễn Văn A là một nhà khoa học.")
        retrieved = passage("DOC_P0002", "Nguyễn Văn A sinh năm 1920.", 0.55)
        followups = generate_followups(
            "Nguyễn Văn A là ai?",
            "Một nhà khoa học.",
            {"subject": "Nguyễn Văn A", "relation": "IDENTITY"},
            selected,
            [retrieved],
            config=NO_PROBE_CONFIG,
        )
        birth = next(item for item in followups if item.relation == "BIRTH_TIME")
        self.assertEqual(birth.source_passage_id, "DOC_P0002")

    def test_relation_marker_must_bind_to_the_subject(self):
        source = passage(
            "DOC_P0001",
            "Chủ nghĩa nô lệ được nhắc đến trong cuộc tranh luận. "
            "Các lực lượng đối lập đến Kansas để biểu quyết và thành lập một tổ chức mới.",
        )
        followups = generate_followups(
            "Chủ nghĩa nô lệ là gì?",
            "Một chế độ xã hội.",
            {"subject": "Chủ nghĩa nô lệ", "relation": "IDENTITY"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual({item.relation for item in followups}, {"EVIDENCE_DETAIL"})

    def test_place_is_not_treated_as_person_or_role(self):
        source = passage(
            "DOC_P0001",
            "Trong thế kỷ 18, Paris là nguồn cảm hứng cho các nhà văn và là nơi sản sinh "
            "những tư tưởng mới.",
        )
        followups = generate_followups(
            "Paris trở thành trung tâm từ thế kỷ nào?",
            "Thế kỷ 18.",
            {"subject": "Paris", "relation": "EVENT_TIME"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertFalse({"BIRTH_TIME", "BIRTH_LOCATION", "ROLE"} & {item.relation for item in followups})

    def test_church_does_not_fold_into_poet_role(self):
        source = passage(
            "DOC_P0001",
            "Đồi Montmartre có độ cao 131 mét, đỉnh là vị trí nhà thờ Saint-Pierre.",
        )
        followups = generate_followups(
            "Montmartre nằm ở đâu?",
            "Paris.",
            {"subject": "Montmartre", "relation": "OBJECT_LOCATION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertNotIn("ROLE", {item.relation for item in followups})

    def test_adverb_triet_de_is_not_a_purpose_marker(self):
        source = passage("DOC_P0001", "Paris có những thay đổi triệt để.")
        followups = generate_followups(
            "Paris là gì?",
            "Một thành phố.",
            {"subject": "Paris", "relation": "IDENTITY"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertNotIn("PURPOSE", {item.relation for item in followups})

    def test_bm25_tier_two_can_rescue_low_topic_candidate(self):
        source = passage("DOC_P0002", "Nguyễn Văn A sinh năm 1920.", 0.05)
        strict = replace(
            NO_PROBE_CONFIG,
            allow_bm25_probe=True,
            min_topic_relevance=0.95,
            max_bm25_probes=1,
        )
        followups = generate_followups(
            "Nguyễn Văn A là ai?",
            "Một nhà khoa học.",
            {"subject": "Nguyễn Văn A", "relation": "IDENTITY"},
            None,
            [source],
            probe=lambda _question, _top_k: [source],
            config=strict,
        )
        self.assertEqual([item.relation for item in followups], ["BIRTH_TIME"])

    def test_bm25_probe_attempts_are_capped(self):
        source = passage(
            "DOC_P0002",
            "Nguyễn Văn A sinh năm 1920. Nguyễn Văn A giữ chức giáo sư. "
            "Nguyễn Văn A tham gia thành lập một trường học.",
            0.01,
        )
        probe_questions: list[str] = []
        strict = replace(
            NO_PROBE_CONFIG,
            allow_bm25_probe=True,
            min_topic_relevance=0.99,
            max_bm25_probes=2,
        )

        def probe(question: str, _top_k: int) -> list[dict]:
            probe_questions.append(question)
            return []

        generate_followups(
            "Nguyễn Văn A là ai?",
            "Một nhà giáo.",
            {"subject": "Nguyễn Văn A", "relation": "IDENTITY"},
            None,
            [source],
            probe=probe,
            config=strict,
        )
        self.assertLessEqual(len(probe_questions), 2)

    def test_debug_distinguishes_no_opportunity_from_all_rejected(self):
        sparse = passage("DOC_P0001", "Sự kiện X diễn ra tại Y.")
        corpus = {"DOC_P0001": sparse}
        response = generate_followup_response(
            {
                "question": "Sự kiện X diễn ra ở đâu?",
                "answer": "Tại Y.",
                "subject": "Sự kiện X",
                "relation": "EVENT_LOCATION",
                "selected_passage_id": "DOC_P0001",
                "retrieved_passage_ids": [],
                "debug": True,
            },
            passage_lookup=corpus.get,
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual(response["debug"]["status"], "EVIDENCE_REVIEW_FALLBACK")
        self.assertEqual(response["debug"]["rejection_distribution"]["SAME_RELATION"], 1)
        trace = response["debug"]["candidates"][0]
        for field in (
            "question",
            "relation",
            "subject",
            "source_passage_id",
            "generated_by",
            "evidence_sentence",
            "subject_match",
            "topic_relevance_score",
            "rejection_reason",
        ):
            self.assertIn(field, trace)

        no_fact = passage("DOC_P0002", "Sự kiện X được nhắc đến trong tài liệu.")
        response = generate_followup_response(
            {
                "question": "Sự kiện X là gì?",
                "answer": "Một sự kiện.",
                "subject": "Sự kiện X",
                "relation": "IDENTITY",
                "selected_passage_id": "DOC_P0002",
                "retrieved_passage_ids": [],
                "debug": True,
            },
            passage_lookup={"DOC_P0002": no_fact}.get,
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual(response["debug"]["status"], "EVIDENCE_REVIEW_FALLBACK")
        self.assertEqual(len(response["followups"]), 1)

    def test_maximum_count_and_source_trace(self):
        source = passage(
            "DOC_P0001",
            "Nguyễn Văn An sinh năm 1945 tại Huế. "
            "Nguyễn Văn An giữ chức hiệu trưởng và tham gia thành lập một trường học. "
            "Hoạt động của Nguyễn Văn An dẫn đến kết quả giáo dục tích cực.",
        )
        followups = generate_followups(
            "Nguyễn Văn An là ai?",
            "Một nhà giáo.",
            {"subject": "Nguyễn Văn An", "relation": "IDENTITY"},
            source,
            [],
            limit=9,
            config=NO_PROBE_CONFIG,
        )
        self.assertLessEqual(len(followups), 3)
        self.assertTrue(followups)
        self.assertTrue(all(item.source_passage_id == "DOC_P0001" for item in followups))

    def test_empty_inputs_do_not_crash(self):
        self.assertEqual(
            generate_followups("", None, {}, None, [], config=NO_PROBE_CONFIG),
            [],
        )
        self.assertEqual(
            generate_followups("Câu hỏi?", "Câu trả lời", {}, None, [], config=NO_PROBE_CONFIG),
            [],
        )

    def test_visited_relation_allows_a_distinct_predicate_question(self):
        source = passage(
            "DOC_P0001",
            "Hoa có thể sinh ra ở đầu ngọn hay ở nách lá. "
            "Thỉnh thoảng, chẳng hạn như ở hoa vi ô let, hoa mọc ra ở nách của lá.",
        )
        followups = generate_followups(
            "Hoa sinh ra ở đâu?",
            "Ở đầu ngọn hay ở nách lá.",
            {
                "subject": "Hoa",
                "relation": "PROCESS_LOCATION",
                "predicate": "sinh ra",
            },
            source,
            [],
            visited_relations=["PROCESS_LOCATION"],
            asked_questions=["Hoa sinh ra ở đâu?"],
            config=NO_PROBE_CONFIG,
        )
        self.assertIn("Hoa mọc ra ở đâu?", {item.question for item in followups})
        self.assertNotIn("Hoa hoa mọc ra ở đâu?", {item.question for item in followups})
        self.assertNotIn("Hoa sinh ra ở đâu?", {item.question for item in followups})

    def test_distinct_predicates_can_produce_multiple_initial_suggestions(self):
        source = passage(
            "DOC_P0001",
            "Hoa có thể sinh ra ở đầu ngọn hay ở nách lá. "
            "Thỉnh thoảng, hoa mọc ra ở nách của lá.",
        )
        followups = generate_followups(
            "Chức năng của hoa là gì?",
            "Hoa hỗ trợ sinh sản.",
            {"subject": "Chức năng của hoa", "relation": "DEFINITION"},
            source,
            [],
            config=NO_PROBE_CONFIG,
        )

        questions = {item.question for item in followups}
        self.assertIn("Hoa sinh ra ở đâu?", questions)
        self.assertIn("Hoa mọc ra ở đâu?", questions)

    def test_repeated_question_tail_is_removed_from_semantic_subject(self):
        source = passage(
            "DOC_P0001",
            "Phạm Văn Đồng từng giữ chức Thủ tướng và tham gia Hội nghị Genève.",
        )
        followups = generate_followups(
            "phạm văn đồng là ai phạm văn đồng là ai",
            "Phạm Văn Đồng là Thủ tướng Việt Nam.",
            {
                "subject": "phạm văn đồng là ai phạm văn đồng",
                "relation": "DEFINITION",
            },
            source,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertTrue(followups)
        self.assertTrue(all(item.subject == "Phạm Văn Đồng" for item in followups))
        self.assertTrue(all("là ai" not in item.question.casefold() for item in followups))

    def test_dispersed_subject_words_do_not_ground_a_followup(self):
        unrelated = passage(
            "DOC_P0002",
            "Họ không xâm phạm thỏa thuận. Trần Văn Giàu tham gia một hoạt động "
            "cùng quân Đồng Minh.",
        )
        followups = generate_followups(
            "Phạm Văn Đồng là ai?",
            "Một chính khách Việt Nam.",
            {"subject": "Phạm Văn Đồng", "relation": "DEFINITION"},
            unrelated,
            [],
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual(followups, [])

    def test_question_normalization_is_accent_and_punctuation_insensitive(self):
        self.assertEqual(
            normalize_question("  Nguyễn Văn An sinh năm nào? "),
            normalize_question("nguyen van an sinh nam nao"),
        )
        self.assertGreater(
            question_similarity(
                "Nguyễn Văn An sinh năm nào?",
                "Nguyễn Văn An sinh vào năm bao nhiêu?",
            ),
            0.70,
        )

    def test_endpoint_response_uses_only_lookup_passages(self):
        corpus = {
            "DOC_P0001": passage("DOC_P0001", "Nguyễn Văn An sinh năm 1945."),
        }
        response = generate_followup_response(
            {
                "question": "Nguyễn Văn An là ai?",
                "answer": "Một nhà giáo.",
                "subject": "Nguyễn Văn An",
                "relation": "IDENTITY",
                "selected_passage_id": "DOC_P0001",
                "retrieved_passage_ids": ["OUTSIDE_CORPUS"],
                "limit": 3,
            },
            passage_lookup=corpus.get,
            config=NO_PROBE_CONFIG,
        )
        self.assertEqual(response["grounding"], "selected_and_retrieved_corpus_passages")
        self.assertTrue(response["followups"])
        self.assertEqual(response["followups"][0]["source_passage_id"], "DOC_P0001")

    def test_qa_answerability_failure_falls_back_to_grounded_review(self):
        source = passage("DOC_P0001", "Nguyễn Văn An sinh năm 1945.")
        response = generate_followup_response(
            {
                "question": "Nguyễn Văn An là ai?",
                "answer": "Một nhà giáo.",
                "subject": "Nguyễn Văn An",
                "relation": "IDENTITY",
                "selected_passage_id": "DOC_P0001",
                "retrieved_passage_ids": [],
                "debug": True,
            },
            passage_lookup={"DOC_P0001": source}.get,
            answerability_validator=lambda _question, passage_id: {
                "verified": False,
                "source_passage_id": passage_id,
                "rejection_reason": "ANSWER_TYPE_MISMATCH",
            },
            config=NO_PROBE_CONFIG,
        )

        self.assertEqual(len(response["followups"]), 1)
        self.assertEqual(response["followups"][0]["relation"], "EVIDENCE_DETAIL")
        self.assertFalse(response["followups"][0]["qa_verified"])
        self.assertTrue(response["followups"][0]["evidence_sentence"])
        self.assertEqual(response["answerability_gate"], "qa_pipeline")
        self.assertGreaterEqual(
            response["debug"]["rejection_distribution"]["QA_ANSWERABILITY_FAILED"],
            1,
        )

    def test_qa_answerability_gate_marks_returned_suggestions_verified(self):
        source = passage("DOC_P0001", "Nguyễn Văn An sinh năm 1945.")
        response = generate_followup_response(
            {
                "question": "Nguyễn Văn An là ai?",
                "answer": "Một nhà giáo.",
                "subject": "Nguyễn Văn An",
                "relation": "IDENTITY",
                "selected_passage_id": "DOC_P0001",
                "retrieved_passage_ids": [],
            },
            passage_lookup={"DOC_P0001": source}.get,
            answerability_validator=lambda _question, passage_id: {
                "verified": True,
                "has_answer": True,
                "source_passage_id": passage_id,
                "method": "phrase_fallback",
            },
            config=NO_PROBE_CONFIG,
        )

        self.assertTrue(response["followups"])
        self.assertTrue(all(item["qa_verified"] for item in response["followups"]))
        self.assertTrue(
            all(item["verification_method"] == "phrase_fallback" for item in response["followups"])
        )

    def test_config_is_valid_and_contains_no_entity_specific_rule(self):
        config_payload = json.loads((ROOT / "config" / "socratic.json").read_text(encoding="utf-8"))
        self.assertLessEqual(config_payload["max_followups"], 3)
        production = (ROOT / "backend" / "socratic.py").read_text(encoding="utf-8")
        for forbidden in ("Voltaire", "Phạm Văn Đồng", "Roosevelt", "Paris", "Saint-Pierre", "Baibars"):
            self.assertNotIn(forbidden, production)

    def test_main_ask_pipeline_never_calls_socratic_generation(self):
        source = (ROOT / "backend" / "viqa_api.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        ask = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "ask_question"
        )
        segment = ast.get_source_segment(source, ask) or ""
        self.assertNotIn("socratic", segment.casefold())
        self.assertNotIn("generate_followup", segment)


if __name__ == "__main__":
    unittest.main()
