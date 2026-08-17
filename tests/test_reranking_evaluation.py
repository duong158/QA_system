import unittest

from evaluate_reranking import WEIGHT_CONFIGS, candidate_passes_hard_gates, candidate_score


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


if __name__ == "__main__":
    unittest.main()
