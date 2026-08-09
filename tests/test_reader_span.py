import unittest

from reader.predict import map_segmented_span_to_raw, select_best_span


class ReaderSpanTests(unittest.TestCase):
    def test_selects_complete_start_end_span_from_context(self):
        context = "Phạm Văn Đồng giữ chức Thủ tướng từ năm 1976 đến năm 1987."
        context_tokens = ["từ", "năm", "1976", "đến", "năm", "1987"]
        token_offsets = []
        cursor = 0
        for token in context_tokens:
            start = context.index(token, cursor)
            token_offsets.append((start, start + len(token)))
            cursor = start + len(token)
        offsets = [(0, 0), (0, 4), (0, 0), *token_offsets, (0, 0)]
        sequence_ids = [None, 0, None, 1, 1, 1, 1, 1, 1, None]
        start_logits = [0.0, 100.0, 0.0, 10.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]
        end_logits = [0.0, 100.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 10.0, 0.0]

        candidate = select_best_span(start_logits, end_logits, offsets, sequence_ids)

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(context[candidate.start_char:candidate.end_char], "từ năm 1976 đến năm 1987")
        self.assertGreater(candidate.end_token, candidate.start_token)

    def test_maps_phobert_segmented_offsets_to_original_text(self):
        raw = "Thủ tướng Chính phủ Việt Nam"
        segmented = "Thủ_tướng Chính_phủ Việt_Nam"
        start = segmented.index("Chính")
        end = len(segmented)

        raw_start, raw_end = map_segmented_span_to_raw(raw, segmented, start, end)

        self.assertEqual(raw[raw_start:raw_end], "Chính phủ Việt Nam")


if __name__ == "__main__":
    unittest.main()
