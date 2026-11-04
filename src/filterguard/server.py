from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from filterguard.service import Analyzer, result_to_dict


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length_raw = handler.headers.get("Content-Length") or "0"
    try:
        length = int(length_raw)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length <= 0:
        raise ValueError("request body must be non-empty JSON")
    raw = handler.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("request JSON body must be an object")
    return payload


def _texts_from_payload(payload: dict[str, Any]) -> list[str]:
    has_text = "text" in payload
    has_texts = "texts" in payload
    if has_text and has_texts:
        raise ValueError("provide only one of 'text' or 'texts'")
    if has_text:
        text = payload["text"]
        if not isinstance(text, str):
            raise ValueError("'text' must be a string")
        return [text]
    if has_texts:
        texts = payload["texts"]
        if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
            raise ValueError("'texts' must be a list of strings")
        return [str(x) for x in texts]
    raise ValueError("request must include 'text' or 'texts'")


def _method_from_path(path: str) -> str:
    parsed = urlparse(path)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 1 and parts[0] == "predict":
        return ""
    if len(parts) == 2 and parts[0] == "predict":
        return parts[1]
    return "__not_found__"


def make_handler(analyzers: dict[str, Analyzer]) -> type[BaseHTTPRequestHandler]:
    class FilterGuardHandler(BaseHTTPRequestHandler):
        server_version = "FilterGuardHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - - %s" % (self.address_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                _json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "methods": sorted(analyzers),
                    },
                )
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            method = _method_from_path(self.path)
            if method == "__not_found__":
                _json_response(self, 404, {"error": "not found"})
                return
            try:
                payload = _read_json(self)
                requested = str(payload.get("method") or method or "")
                if not requested:
                    if len(analyzers) == 1:
                        requested = next(iter(analyzers))
                    else:
                        raise ValueError("method is required when multiple methods are loaded")
                if requested not in analyzers:
                    raise ValueError(f"method {requested!r} is not loaded")
                texts = _texts_from_payload(payload)
                results = analyzers[requested].predict(texts)
            except ValueError as exc:
                _json_response(self, 400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                _json_response(self, 500, {"error": str(exc)})
                return

            _json_response(
                self,
                200,
                {
                    "method": requested,
                    "count": len(results),
                    "results": [{"index": i, **result_to_dict(result)} for i, result in enumerate(results)],
                },
            )

    return FilterGuardHandler


def serve(*, analyzers: dict[str, Analyzer], host: str, port: int) -> None:
    if not analyzers:
        raise ValueError("at least one analyzer must be loaded")
    server = ThreadingHTTPServer((str(host), int(port)), make_handler(analyzers))
    print(f"FilterGuard API listening on http://{host}:{port} methods={','.join(sorted(analyzers))}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for analyzer in analyzers.values():
            analyzer.close()

