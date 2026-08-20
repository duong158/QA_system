import unittest

from backend.source_titles import clean_source_title, derive_source_title


class SourceTitleTests(unittest.TestCase):
    def test_strips_vietnamese_birth_death_date_from_person_title(self):
        self.assertEqual(
            clean_source_title("Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000)"),
            "Phạm Văn Đồng",
        )

    def test_strips_compact_year_range(self):
        self.assertEqual(clean_source_title("Nguyễn Văn A (1906-2000)"), "Nguyễn Văn A")

    def test_preserves_meaningful_single_year_parenthetical(self):
        self.assertEqual(clean_source_title("Hội nghị Genève (1954)"), "Hội nghị Genève (1954)")

    def test_metadata_title_has_priority_over_passage_heuristics(self):
        self.assertEqual(
            derive_source_title(
                "doc_1",
                "Một đoạn nội dung không liên quan.",
                document_title="Biên niên sử Việt Nam (1954)",
                heading="Heading phụ",
            ),
            "Biên niên sử Việt Nam (1954)",
        )

    def test_heading_is_used_when_document_title_is_missing(self):
        self.assertEqual(
            derive_source_title("doc_1", "Nội dung.", heading="  Lịch sử hiện đại  "),
            "Lịch sử hiện đại",
        )

    def test_extracts_entity_before_lifespan_from_passage_start(self):
        text = (
            "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) "
            "là Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam."
        )
        self.assertEqual(derive_source_title("doc_00001", text), "Phạm Văn Đồng")

    def test_extracts_entity_before_predicate_without_parenthetical(self):
        self.assertEqual(
            derive_source_title("doc_00003", "Phạm Văn Đồng có vợ là bà Phạm Thị Cúc."),
            "Phạm Văn Đồng",
        )

    def test_fallback_is_bounded_and_does_not_leave_open_parenthesis(self):
        title = derive_source_title(
            "doc_x",
            "Năm 1954, một phái đoàn rất quan trọng (với nhiều đại biểu và thông tin bổ sung kéo dài) tham dự cuộc họp lịch sử.",
        )
        self.assertLessEqual(len(title), 89)
        self.assertEqual(title.count("("), title.count(")"))


if __name__ == "__main__":
    unittest.main()
