import unittest

from backend.chunking import chunk_document, split_sentences


class ChunkingTests(unittest.TestCase):
    def test_chunks_end_on_sentence_boundaries(self):
        text = "Sentence A. Sentence B. Sentence C. Sentence D."
        passages = chunk_document(
            "DOC001",
            text,
            max_tokens=4,
            overlap_sentences=1,
        )

        self.assertGreater(len(passages), 1)
        self.assertTrue(all(passage.text.endswith(".") for passage in passages))
        self.assertNotIn("Sent", [passage.text[-4:] for passage in passages])

    def test_overlap_reuses_whole_sentences(self):
        text = "One alpha. Two beta. Three gamma. Four delta."
        passages = chunk_document("DOC002", text, max_tokens=4, overlap_sentences=1)

        self.assertEqual(passages[0].text, "One alpha. Two beta.")
        self.assertTrue(passages[1].text.startswith("Two beta."))
        self.assertEqual(split_sentences(passages[0].text)[-1], split_sentences(passages[1].text)[0])

    def test_metadata_is_preserved(self):
        passage = chunk_document("DOC003", "First. Second.", title="Title", page=12)[0]
        self.assertEqual(passage.document_id, "DOC003")
        self.assertEqual(passage.passage_id, "DOC003_P0001")
        self.assertEqual(passage.paragraph_id, "DOC003_PAR0001")
        self.assertEqual((passage.sentence_start, passage.sentence_end), (0, 1))
        self.assertEqual(passage.page, 12)


if __name__ == "__main__":
    unittest.main()
