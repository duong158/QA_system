import unittest

from transformers import AutoConfig

from reader.config import DEFAULT_DOC_STRIDE, DEFAULT_MAX_LENGTH, effective_model_max_length
from reader.data_utils import _map_char_span_to_feature, build_text_preprocessor, prepare_train_features
from reader.data_utils import get_tokenizer
from reader.qa_tokenizer import encode_qa_batch
from reader.text_preprocessing import normalize_span_text, restore_text_offsets, uses_compact_offsets


MODEL_PATH = "models/reader/vinai_phobert-base-v2"


class ReaderPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer(MODEL_PATH)
        cls.config = AutoConfig.from_pretrained(MODEL_PATH)
        cls.preprocessor = build_text_preprocessor(
            MODEL_PATH,
            tokenizer=cls.tokenizer,
            model_config=cls.config,
        )

    def assert_span_round_trips(self, context, answer, answer_start, question="Câu trả lời là gì?"):
        prepared = self.preprocessor.prepare(question, context, answer, answer_start)
        encoded = encode_qa_batch(
            self.tokenizer,
            [prepared.model_question],
            [prepared.model_context],
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_DOC_STRIDE,
            padding="max_length",
        )
        decoded = []
        for feature_index, tokenizer_offsets in enumerate(encoded["offset_mapping"]):
            offsets = restore_text_offsets(
                prepared.model_context,
                tokenizer_offsets,
                uses_compact_offsets(self.tokenizer),
            )
            input_ids = encoded["input_ids"][feature_index]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            start, end = _map_char_span_to_feature(
                offsets,
                encoded.sequence_ids(feature_index),
                prepared.model_answer_start,
                prepared.model_answer_end,
                cls_index,
            )
            if start != cls_index or end != cls_index:
                decoded.append(prepared.model_context[offsets[start][0] : offsets[end][1]])
        self.assertTrue(decoded)
        self.assertTrue(all(normalize_span_text(value) == normalize_span_text(answer) for value in decoded))

    def test_answer_at_context_start(self):
        context = "Phạm Văn Đồng sinh ngày 1 tháng 3 năm 1906."
        self.assert_span_round_trips(context, "Phạm Văn Đồng", 0)

    def test_repeated_answer_uses_annotated_occurrence(self):
        context = "Paris là một thành phố. Năm 1789, cách mạng diễn ra tại Paris."
        second = context.rindex("Paris")
        prepared = self.preprocessor.prepare("Cách mạng diễn ra ở đâu?", context, "Paris", second)
        raw_start, raw_end = prepared.context_alignment.model_span_to_raw(
            prepared.model_answer_start,
            prepared.model_answer_end,
        )
        self.assertEqual(raw_start, second)
        self.assertEqual(context[raw_start:raw_end], "Paris")
        self.assert_span_round_trips(context, "Paris", second, "Cách mạng diễn ra ở đâu?")

    def test_unicode_and_punctuation(self):
        context = "Thủ đô của Việt Nam là Hà Nội, một thành phố lâu đời."
        answer = "Hà Nội,"
        self.assert_span_round_trips(context, answer, context.index(answer))

    def test_answer_at_context_end(self):
        context = "Thành phố được nhắc tới là Thành phố Hồ Chí Minh"
        answer = "Thành phố Hồ Chí Minh"
        self.assert_span_round_trips(context, answer, context.index(answer))

    def test_answer_near_sliding_window_boundary(self):
        prefix = " ".join(f"từ{i}" for i in range(220))
        answer = "đáp án biên cửa sổ"
        suffix = " ".join(f"cuối{i}" for i in range(120))
        context = f"{prefix} {answer} {suffix}"
        self.assert_span_round_trips(context, answer, context.index(answer))

    def test_unanswerable_features_are_labeled_cls(self):
        examples = {
            "id": ["no-answer"],
            "question": ["Ai là người được nhắc đến?"],
            "context": ["Đoạn văn không chứa câu trả lời."],
            "answer_text": [""],
            "answer_start": [-1],
        }
        features = prepare_train_features(
            examples,
            self.tokenizer,
            DEFAULT_MAX_LENGTH,
            DEFAULT_DOC_STRIDE,
            self.preprocessor,
        )
        for input_ids, start, end in zip(
            features["input_ids"], features["start_positions"], features["end_positions"]
        ):
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            self.assertEqual((start, end), (cls_index, cls_index))

    def test_train_and_inference_share_preprocessing_and_window_config(self):
        question = "Phạm Văn Đồng là ai?"
        context = "Phạm Văn Đồng là Thủ tướng Việt Nam."
        first = self.preprocessor.prepare(question, context)
        second = build_text_preprocessor(
            MODEL_PATH,
            tokenizer=self.tokenizer,
            model_config=self.config,
        ).prepare(question, context)
        self.assertEqual(first.model_question, second.model_question)
        self.assertEqual(first.model_context, second.model_context)
        self.assertEqual(effective_model_max_length(self.config, self.tokenizer), 256)
        self.assertEqual((DEFAULT_MAX_LENGTH, DEFAULT_DOC_STRIDE), (256, 80))
        self.assertEqual(self.tokenizer.vocab_size, 64000)
        encoded = self.tokenizer(first.model_context)
        self.assertLess(max(encoded["input_ids"]), self.config.vocab_size)


if __name__ == "__main__":
    unittest.main()
