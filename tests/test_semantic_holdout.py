import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "evaluation" / "semantic_holdout_v1.jsonl"
LOCK = ROOT / "data" / "evaluation" / "semantic_holdout_v1.lock.json"


class LockedSemanticHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))
        cls.rows = [
            json.loads(line)
            for line in HOLDOUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_checksum_and_row_count_match_lock(self):
        digest = hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
        self.assertEqual(self.lock["status"], "LOCKED_DO_NOT_TUNE")
        self.assertEqual(digest, self.lock["dataset_sha256"])
        self.assertEqual(len(self.rows), self.lock["rows"])
        self.assertEqual(len({row["id"] for row in self.rows}), len(self.rows))

    def test_locked_quotas_are_preserved(self):
        counts = Counter(row["holdout_stratum"] for row in self.rows)
        self.assertEqual(counts, Counter(self.lock["actual_quotas"]))
        self.assertEqual(counts, Counter(self.lock["requested_quotas"]))


if __name__ == "__main__":
    unittest.main()
