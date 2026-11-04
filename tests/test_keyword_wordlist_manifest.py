from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from filterguard.keyword import compile_rules, keyword_score, load_wordlist_manifest


class KeywordWordlistManifestTest(unittest.TestCase):
    def test_manifest_loads_enabled_wordlists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wordlist = root / "wordlist.txt"
            wordlist.write_text("alpha\nbeta phrase\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "wordlists": [
                            {
                                "id": "unit",
                                "enabled": True,
                                "local_path": "wordlist.txt",
                                "weight": 3,
                                "match": "auto",
                            },
                            {
                                "id": "disabled",
                                "enabled": False,
                                "local_path": "missing.txt",
                                "weight": 99,
                                "match": "substring",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rules = load_wordlist_manifest(manifest, repo_root=root)
            compiled = compile_rules(rules)
            self.assertEqual(len(rules), 2)
            self.assertEqual(keyword_score("alpha appears", compiled), 3)
            self.assertEqual(keyword_score("beta phrase appears", compiled), 3)

    def test_manifest_reports_missing_enabled_wordlists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"wordlists": [{"enabled": True, "local_path": "missing.txt"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_wordlist_manifest(manifest, repo_root=root)


if __name__ == "__main__":
    unittest.main()
