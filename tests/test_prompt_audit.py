import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from prompt_audit import audit_file, build_manifest


class PromptAuditTests(unittest.TestCase):
    def test_valid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps([{
                "id": "1", "description": "voice", "trait": "status",
                "keywords": "high", "prompt_text": "hello",
            }]), encoding="utf-8")
            self.assertEqual(audit_file(path)["errors"], [])

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            item = {"id": "1", "description": "voice", "trait": "status", "keywords": "high", "prompt_text": "hello"}
            path.write_text(json.dumps([item, item]), encoding="utf-8")
            self.assertIn("duplicate ids", audit_file(path)["errors"])

    def test_overcount_is_not_exact_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.json"
            item = {"id": "1", "description": "voice", "trait": "status", "keywords": "high", "prompt_text": "hello"}
            path.write_text(json.dumps([item, dict(item, id="2")]), encoding="utf-8")
            with patch("prompt_audit.PAPER_EXPECTED", {"expected.json": 1}):
                manifest = build_manifest(Path(directory))
            self.assertFalse(manifest["paper_count_alignment"])
            self.assertEqual(manifest["count_mismatches"]["expected.json"], {"expected": 1, "actual": 2})
            self.assertFalse(manifest["exact_paper_reproduction_available"])


if __name__ == "__main__":
    unittest.main()
