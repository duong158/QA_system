import unittest

from backend.config import load_pipeline_config
from backend.viqa_api import combine_ranking_scores, min_max_normalize


class RerankingConfigTests(unittest.TestCase):
    def test_temporary_production_weights_and_candidate_pool(self):
        config = load_pipeline_config()
        self.assertAlmostEqual(config.retriever_weight, 0.4)
        self.assertAlmostEqual(config.reader_weight, 0.3)
        self.assertAlmostEqual(config.answer_type_weight, 0.2)
        self.assertAlmostEqual(config.relation_weight, 0.1)
        self.assertAlmostEqual(config.reader_score_margin_threshold, -3.5)
        self.assertTrue(config.reader_profile_calibrated)
        self.assertTrue(config.require_calibrated_reader_profile)
        self.assertAlmostEqual(config.phrase_fallback_penalty, 1.0)
        self.assertAlmostEqual(config.fallback_penalty, 0.6)
        self.assertEqual(config.reader_span_candidates, 5)
        self.assertEqual(config.default_top_k, 10)
        self.assertEqual(config.candidate_count(10), 20)
        self.assertAlmostEqual(
            config.retriever_weight
            + config.reader_weight
            + config.answer_type_weight
            + config.relation_weight,
            1.0,
        )

    def test_reranking_formula(self):
        passage_a = combine_ranking_scores(0.95, 0.10, 0.0, 0.5, 0.3, 0.2)
        passage_b = combine_ranking_scores(0.75, 0.80, 1.0, 0.5, 0.3, 0.2)
        self.assertAlmostEqual(passage_a, 0.505)
        self.assertAlmostEqual(passage_b, 0.815)
        self.assertGreater(passage_b, passage_a)

    def test_min_max_normalization(self):
        self.assertEqual(min_max_normalize([5.0, 10.0, 15.0]), [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
