import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HeaderStatusTests(unittest.TestCase):
    def test_header_renders_one_system_status_component(self):
        header = (ROOT / "src" / "components" / "layout" / "Header.tsx").read_text(
            encoding="utf-8"
        )
        status = (ROOT / "src" / "components" / "layout" / "SystemStatus.tsx").read_text(
            encoding="utf-8"
        )

        self.assertEqual(header.count("<SystemStatus />"), 1)
        self.assertNotIn("SYSTEM ONLINE", header)
        self.assertEqual(status.count('data-testid="system-status"'), 1)
        self.assertIn("{status}", status)


if __name__ == "__main__":
    unittest.main()
