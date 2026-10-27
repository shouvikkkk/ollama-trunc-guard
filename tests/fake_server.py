"""In-process fake Ollama HTTP server for the test suite.

Serves /api/tags, /api/ps and /api/show from a scenario dict so probe
tests can exercise every field-provenance branch without a real Ollama
server. Nothing here talks to the network beyond localhost.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(scenario):
    class FakeOllamaHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep test output clean
            pass

        def _send_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send_json(scenario.get("tags", {"models": []}))
            elif self.path == "/api/ps":
                self._send_json(scenario.get("ps", {"models": []}))
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                payload = {}
            if self.path == "/api/show":
                shows = scenario.get("shows", {})
                model = payload.get("name", "")
                self._send_json(shows.get(model, shows.get("__default__", {})))
            else:
                self.send_error(404)

    return FakeOllamaHandler


class FakeOllamaServer:
    """A ThreadingHTTPServer bound to an ephemeral localhost port."""

    def __init__(self, scenario):
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(scenario)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        return False
