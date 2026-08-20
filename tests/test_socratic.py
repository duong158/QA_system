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

    def test_ungrounded_candidate_is_rejected(self):
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
        self.assertEqual(followups, [])

    def test_probe_can_reject_tier_one_candidate(self):
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
        self.assertEqual(followups, [])

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

    def test_visited_relation_is_not_repeated(self):
        source = passage(
            "DOC_P0001",
            "Nguyễn Văn An sinh năm 1945 và giữ chức hiệu trưởng.",
        )
        followups = generate_followups(
            "Nguyễn Văn An là ai?",
            "Một nhà giáo.",
            {"subject": "Nguyễn Văn An", "relation": "IDENTITY"},
            source,
            [],
            visited_relations=["BIRTH_TIME"],
            config=NO_PROBE_CONFIG,
        )
        self.assertNotIn("BIRTH_TIME", {item.relation for item in followups})

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
