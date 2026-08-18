import unittest

from reader.evaluate import apply_score_margin_threshold, sweep_thresholds
from reader.metrics import evaluate_predictions, exact_match, f1_score, normalize_answer
from reader.postprocessing import should_return_answer


class ReaderMetricsAndThresholdTests(unittest.TestCase):
    def test_vietnamese_metrics_normalize_underscores_and_punctuation(self):
        self.assertEqual(normalize_answer("Hà_Nội,"), "hà nội")
        self.assertEqual(exact_match("Hà Nội", "Hà_Nội."), 1)
        self.assertAlmostEqual(f1_score("thủ đô Hà Nội", "Hà Nội"), 2 / 3)

    def test_score_margin_comparison_direction(self):
        self.assertTrue(should_return_answer(2.0, 0.5))
        self.assertFalse(should_return_answer(-2.0, 0.5))
        raw = [
            {"gold_answer": "Hà Nội", "is_answerable": True, "best_span_answer": "Hà Nội", "score_margin": 1.0},
            {"gold_answer": "", "is_answerable": False, "best_span_answer": "Paris", "score_margin": -1.0},
        ]
        predictions = apply_score_margin_threshold(raw, 0.0)
        self.assertEqual([row["predicted_answer"] for row in predictions], ["Hà Nội", ""])

    def test_metrics_separate_answerable_and_no_answer(self):
        metrics = evaluate_predictions(
            [
                {"gold_answer": "Paris", "predicted_answer": "", "is_answerable": True},
                {"gold_answer": "", "predicted_answer": "", "is_answerable": False},
            ]
        )
        self.assertEqual(metrics["answerable"]["f1"], 0.0)
        self.assertEqual(metrics["answerable"]["predicted_empty_rate"], 100.0)
        self.assertEqual(metrics["unanswerable"]["accuracy"], 100.0)

    def test_threshold_sweep_uses_balanced_qa_score(self):
        raw = [
            {"gold_answer": "A", "is_answerable": True, "best_span_answer": "A", "score_margin": 1.0},
            {"gold_answer": "B", "is_answerable": True, "best_span_answer": "B", "score_margin": 0.8},
            {"gold_answer": "", "is_answerable": False, "best_span_answer": "wrong", "score_margin": -0.5},
        ]
        _, best, predictions, metrics = sweep_thresholds(raw, thresholds=[-1.0, 0.0, 2.0])
        self.assertEqual(best["threshold"], 0.0)
        self.assertEqual(metrics["answerable"]["f1"], 100.0)
        self.assertEqual(metrics["unanswerable"]["accuracy"], 100.0)
        self.assertEqual([row["predicted_answer"] for row in predictions], ["A", "B", ""])


if __name__ == "__main__":
    unittest.main()
