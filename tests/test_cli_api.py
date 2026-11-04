from __future__ import annotations

import contextlib
import io
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from filterguard.cli import main
from filterguard.keyword import KeywordRule
from filterguard.server import make_handler
from filterguard.service import KeywordAnalyzer


class CliApiTest(unittest.TestCase):
    def test_keyword_cli_predicts_json(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(
                [
                    "predict",
                    "--methods",
                    "keyword",
                    "--keyword-no-base-rules",
                    "--keyword-extra-wordlist",
                    "configs/example_wordlist.txt",
                    "--keyword-threshold",
                    "1",
                    "--text",
                    "ordinary educational text",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "keyword")
        self.assertEqual(payload["count"], 1)
        self.assertIn("prediction", payload["results"][0])

    def test_http_predict_keyword(self) -> None:
        analyzer = KeywordAnalyzer(rules=[KeywordRule(pattern="blocked", weight=2, match="word")], threshold=1)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler({"keyword": analyzer}))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/predict/keyword"
            request = urllib.request.Request(
                url,
                data=json.dumps({"text": "this is blocked"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(payload["method"], "keyword")
        self.assertEqual(payload["results"][0]["prediction"], 1)
        self.assertEqual(payload["results"][0]["details"]["hits"], ["blocked"])


if __name__ == "__main__":
    unittest.main()

