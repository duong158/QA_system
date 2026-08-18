import unittest

from evaluate_reranking import (
    WEIGHT_CONFIGS,
    candidate_passes_hard_gates,
    candidate_score,
    select_candidate,
)


class RerankingEvaluationTests(unittest.TestCase):
    def test_required_weight_configs_sum_to_one(self):
        self.assertEqual(len(WEIGHT_CONFIGS), 4)
        for config in WEIGHT_CONFIGS.values():
            self.assertAlmostEqual(sum(config.values()), 1.0)

    def test_reader_threshold_is_not_a_hard_candidate_gate(self):
        candidate = {
            "text": "sự khác biệt giữa A và B",
            "valid_span": True,
            "passes_evidence_gate": True,
            "passes_type_gate": True,
            "passes_relation_gate": True,
            "passes_reader_threshold": False,
        }

        self.assertTrue(candidate_passes_hard_gates(candidate))

    def test_relation_signal_changes_candidate_score(self):
        candidate = {
            "retrieval_score": 0.8,
            "reader_score": 0.6,
            "fallback_penalty": 1.0,
            "answer_type_score": 0.7,
            "relation_score": 1.0,
        }
        without_relation = candidate_score(candidate, WEIGHT_CONFIGS["B_R40_Reader40_Type20"])
        with_relation = candidate_score(
            candidate,
            WEIGHT_CONFIGS["C_R40_Reader30_Type20_Relation10"],
        )

        self.assertAlmostEqual(without_relation, 0.70)
        self.assertAlmostEqual(with_relation, 0.74)

    def test_strong_cause_evidence_can_cross_gate_without_lowering_threshold(self):
        candidate = {
            "text": "xuất thân thấp kém",
            "valid_span": True,
            "passes_evidence_gate": True,
            "passes_type_gate": True,
            "passes_relation_gate": True,
            "passes_completeness_gate": True,
            "retrieval_score": 0.10,
            "reader_score": 0.40,
            "fallback_penalty": 1.0,
            "answer_type_score": 0.80,
            "relation_type": "CAUSE",
            "relation_score": 0.95,
            "cause_pattern_score": 0.96,
            "subject_match_score": 0.95,
            "target_relation_score": 0.95,
        }
        selected, score = select_candidate(
            [candidate],
            WEIGHT_CONFIGS["B_R40_Reader40_Type20"],
            final_threshold=0.60,
        )

        self.assertLess(score, 0.60)
        self.assertIs(selected, candidate)


if __name__ == "__main__":
    unittest.main()
