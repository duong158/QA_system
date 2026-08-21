from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.socratic import (
    KnowledgeOpportunity,
    SocraticConfig,
    discover_followup_opportunities,
    generate_followup_response,
    generate_followups,
)


ROOT = Path(__file__).resolve().parents[1]
NO_PROBE = SocraticConfig(allow_bm25_probe=False)


def passage(passage_id: str, title: str, text: str, score: float = 1.0) -> dict:
    return {
        "passage_id": passage_id,
        "title": title,
        "text": text,
        "relevance_score": score,
    }


class TestSocraticGeneralization:
    def test_discovery_returns_knowledge_opportunities_before_questions(self):
        source = passage(
            "P1",
            "Hệ thống Delta",
            "Hệ thống Delta suy giảm nhanh chóng bởi vì nguồn năng lượng cạn kiệt.",
        )
        opportunities = discover_followup_opportunities(
            {"subject": "Hệ thống Delta", "relation": "DEFINITION"},
            source,
            [],
            question="Hệ thống Delta là gì?",
            answer="Một hệ thống thử nghiệm.",
            config=NO_PROBE,
        )
        cause = next(item for item in opportunities if item.relation == "CAUSE")
        assert isinstance(cause, KnowledgeOpportunity)
        assert cause.evidence_sentence in source["text"]
        assert cause.source_passage_id == "P1"
        assert cause.relation_score and cause.relation_score >= 0.7
        assert cause.origin and cause.origin.startswith("selected:")

    def test_cause_is_structural_across_unseen_effect_predicates(self):
        contexts = (
            "Mạng lưới Lambda suy giảm đột ngột do nguồn cung bị gián đoạn.",
            "Phong trào Orion trở nên phổ biến bởi vì cộng đồng địa phương ủng hộ.",
            "Dự án Kappa thất bại vì kinh phí bị cắt giảm.",
        )
        for index, text in enumerate(contexts):
            subject = text.split(" ", 2)[0] + " " + text.split(" ", 2)[1]
            source = passage(f"P{index}", subject, text)
            opportunities = discover_followup_opportunities(
                {"subject": subject, "relation": "DEFINITION"},
                source,
                [],
                question=f"{subject} là gì?",
                answer="Một đối tượng trong tài liệu.",
                config=NO_PROBE,
            )
            assert "CAUSE" in {item.relation for item in opportunities}

    def test_process_location_extracts_an_unseen_predicate_from_evidence(self):
        source = passage(
            "P1",
            "Dòng hải lưu Zephyr",
            "Dòng hải lưu Zephyr lan rộng nhanh chóng tại vùng biển phía bắc.",
        )
        followups = generate_followups(
            "Dòng hải lưu Zephyr là gì?",
            "Một dòng hải lưu.",
            {"subject": "Dòng hải lưu Zephyr", "relation": "DEFINITION"},
            source,
            [],
            config=NO_PROBE,
        )
        location = next(item for item in followups if item.relation == "PROCESS_LOCATION")
        assert location.question == "Dòng hải lưu Zephyr lan rộng nhanh chóng ở đâu?"
        assert location.relation_evidence is True

    def test_generic_event_time_does_not_require_a_known_action_phrase(self):
        source = passage(
            "P1",
            "Liên hiệp Khoa học Aurora",
            "Liên hiệp Khoa học Aurora được chính thức đăng ký vào năm 2001.",
        )
        opportunities = discover_followup_opportunities(
            {"subject": "Liên hiệp Khoa học Aurora", "relation": "DEFINITION"},
            source,
            [],
            question="Liên hiệp Khoa học Aurora là gì?",
            answer="Một liên hiệp khoa học.",
            config=NO_PROBE,
        )
        assert "EVENT_TIME" in {item.relation for item in opportunities}

    def test_role_uses_appointment_grammar_not_a_title_whitelist(self):
        source = passage(
            "P1",
            "Nguyễn Minh Quang",
            "Nguyễn Minh Quang được bổ nhiệm làm Tổng quản lý hệ sinh thái.",
        )
        opportunities = discover_followup_opportunities(
            {"subject": "Nguyễn Minh Quang", "relation": "DEFINITION"},
            source,
            [],
            question="Nguyễn Minh Quang là ai?",
            answer="Một nhà quản lý.",
            config=NO_PROBE,
        )
        assert "ROLE" in {item.relation for item in opportunities}

    def test_non_biography_organization_coreference_yields_multiple_facts(self):
        source = passage(
            "P1",
            "Viện Hải dương Lam Sơn",
            "Viện Hải dương Lam Sơn là một tổ chức nghiên cứu. "
            "Tổ chức này được thành lập vào năm 1998 nhằm kết nối các phòng thí nghiệm ven biển.",
        )
        opportunities = discover_followup_opportunities(
            {"subject": "Viện Hải dương Lam Sơn", "relation": "DEFINITION"},
            source,
            [],
            question="Viện Hải dương Lam Sơn là gì?",
            answer="Một tổ chức nghiên cứu.",
            config=NO_PROBE,
        )
        relations = {item.relation for item in opportunities}
        assert {"EVENT_TIME", "PURPOSE"}.issubset(relations)
        assert any(item.subject_match == "COREFERENCE_SUBJECT" for item in opportunities)

    def test_generic_numeric_attribute_is_grounded(self):
        source = passage(
            "P1",
            "Cầu Lam Giang",
            "Cầu Lam Giang có chiều dài 1.245 mét.",
        )
        followups = generate_followups(
            "Cầu Lam Giang là gì?",
            "Một cây cầu.",
            {"subject": "Cầu Lam Giang", "relation": "DEFINITION"},
            source,
            [],
            config=NO_PROBE,
        )
        attribute = next(item for item in followups if item.relation == "ATTRIBUTE")
        assert attribute.question == "Cầu Lam Giang có chiều dài bao nhiêu?"
        assert attribute.evidence_sentence == source["text"]

    def test_sparse_context_returns_a_grounded_evidence_review(self):
        source = passage("P1", "Thuật ngữ Lam", "Thuật ngữ Lam là một khái niệm.")
        response = generate_followup_response(
            {
                "question": "Thuật ngữ Lam là gì?",
                "answer": "Một khái niệm.",
                "subject": "Thuật ngữ Lam",
                "relation": "DEFINITION",
                "selected_passage_id": "P1",
                "retrieved_passage_ids": [],
                "debug": True,
            },
            passage_lookup={"P1": source}.get,
            config=NO_PROBE,
        )
        assert len(response["followups"]) == 1
        assert response["followups"][0]["relation"] == "EVIDENCE_DETAIL"
        assert response["followups"][0]["source_passage_id"] == "P1"
        assert response["followups"][0]["evidence_sentence"] == source["text"]
        assert response["debug"]["status"] == "EVIDENCE_REVIEW_FALLBACK"

    def test_no_main_answer_still_receives_a_grounded_direction(self):
        source = passage(
            "P1",
            "Quần đảo Lam Hải",
            "Quần đảo Lam Hải nằm tại vùng biển phía đông.",
        )
        response = generate_followup_response(
            {
                "question": "Quần đảo Lam Hải được thành lập năm nào?",
                "answer": None,
                "subject": "Quần đảo Lam Hải",
                "relation": "EVENT_TIME",
                "selected_passage_id": None,
                "retrieved_passage_ids": ["P1"],
                "debug": True,
            },
            passage_lookup={"P1": source}.get,
            config=NO_PROBE,
        )
        assert response["followups"]
        assert all(item["source_passage_id"] == "P1" for item in response["followups"])
        assert all(item["evidence_sentence"] for item in response["followups"])

    def test_locked_holdout_has_80_unseen_real_corpus_cases(self):
        data_path = ROOT / "tests" / "data" / "socratic_generalization_holdout_v1.json"
        lock_path = ROOT / "tests" / "data" / "socratic_generalization_holdout_v1.lock.json"
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        assert payload["status"] == "LOCKED_DO_NOT_TUNE"
        assert payload["rows"] == 80
        assert len(payload["cases"]) == 80
        assert digest == lock["dataset_sha256"]
        assert len({case["stratum"] for case in payload["cases"]}) == 8
        assert len({case["title"] for case in payload["cases"]}) >= 10
        production = (ROOT / "backend" / "socratic.py").read_text(encoding="utf-8")
        assert "tests/data" not in production
        assert "socratic_generalization_holdout" not in production
        for forbidden in payload["excluded_benchmark_literals"]:
            assert forbidden not in production

    def test_every_socratic_threshold_and_limit_is_configured(self):
        config = json.loads((ROOT / "config" / "socratic.json").read_text(encoding="utf-8"))
        required = {
            "max_followups",
            "max_internal_candidates",
            "max_passages_for_followup_discovery",
            "min_subject_score",
            "min_relation_score",
            "min_evidence_score",
            "min_topic_relevance",
            "min_answerability_score",
            "min_novelty_score",
            "min_ranking_score",
            "duplicate_similarity_threshold",
        }
        assert required.issubset(config)
