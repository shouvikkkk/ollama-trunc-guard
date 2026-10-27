"""Tests for ollama_trunc_guard.probe using the fake Ollama server."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_server import FakeOllamaServer
from ollama_trunc_guard.probe import OllamaClient


def scenario_with_context():
    return {
        "tags": {"models": [{"name": "llama3.1:8b"}]},
        "ps": {"models": [{"name": "llama3.1:8b", "context_length": 4096}]},
        "shows": {
            "llama3.1:8b": {
                "model_info": {"llama.context_length": 131072},
                "parameters": {"num_ctx": "16384"},
                "modelfile": "# Modelfile\nPARAMETER num_ctx 16384\n",
            }
        },
    }


def scenario_without_ps_context():
    return {
        "tags": {"models": [{"name": "qwen2.5:7b"}]},
        "ps": {"models": [{"name": "qwen2.5:7b", "size": 12345}]},
        "shows": {
            "qwen2.5:7b": {
                "model_info": {"qwen2.context_length": 32768},
                "parameters": {"num_ctx": 8192},
            }
        },
    }


def scenario_no_window_anywhere():
    return {
        "tags": {"models": [{"name": "mystery:1b"}]},
        "ps": {"models": [{"name": "mystery:1b"}]},
        "shows": {"mystery:1b": {"details": {}}},
    }


class TestProbeWithContext(unittest.TestCase):
    def test_loaded_window_preferred_over_configured_and_ceiling(self):
        with FakeOllamaServer(scenario_with_context()) as server:
            result = OllamaClient(base_url=server.base_url).probe("llama3.1:8b")
        self.assertTrue(result.reachable)
        self.assertEqual(result.loaded.value, 4096)
        self.assertEqual(result.loaded.source, "loaded (ps)")
        self.assertEqual(result.configured_num_ctx.value, 16384)
        self.assertEqual(result.ceiling.value, 131072)
        effective = result.effective_window()
        self.assertEqual(effective.value, 4096)
        self.assertEqual(effective.source, "loaded (ps)")

    def test_installed_models_listed(self):
        with FakeOllamaServer(scenario_with_context()) as server:
            result = OllamaClient(base_url=server.base_url).probe("llama3.1:8b")
        self.assertEqual(result.installed_models, ["llama3.1:8b"])


class TestProbeAbsentContextField(unittest.TestCase):
    def test_absent_ps_context_field_yields_unknown_loaded(self):
        # The core fail-closed behaviour: a missing field must NOT be
        # replaced by a made-up number.
        with FakeOllamaServer(scenario_without_ps_context()) as server:
            result = OllamaClient(base_url=server.base_url).probe("qwen2.5:7b")
        self.assertIsNone(result.loaded.value)
        self.assertEqual(result.loaded.source, "unknown")
        # Effective window falls back to configured num_ctx, not a guess.
        effective = result.effective_window()
        self.assertEqual(effective.value, 8192)
        self.assertEqual(effective.source, "configured num_ctx")

    def test_no_window_anywhere_yields_unknown_effective(self):
        with FakeOllamaServer(scenario_no_window_anywhere()) as server:
            result = OllamaClient(base_url=server.base_url).probe("mystery:1b")
        self.assertTrue(result.reachable)
        effective = result.effective_window()
        self.assertIsNone(effective.value)
        self.assertEqual(effective.source, "unknown")


class TestProbeUnreachable(unittest.TestCase):
    def test_unreachable_server_is_not_reachable(self):
        # Port 1 on localhost is (essentially always) closed.
        result = OllamaClient(base_url="http://127.0.0.1:1", timeout=2).probe("x:1b")
        self.assertFalse(result.reachable)
        self.assertTrue(result.error)
        self.assertEqual(result.effective_window().source, "unknown")

    def test_show_failure_does_not_invalidate_tags_ps(self):
        scenario = {
            "tags": {"models": [{"name": "a:1b"}]},
            "ps": {"models": [{"name": "a:1b", "context_length": 2048}]},
            "shows": {},  # /api/show returns {} for every model
        }
        with FakeOllamaServer(scenario) as server:
            result = OllamaClient(base_url=server.base_url).probe("a:1b")
        self.assertTrue(result.reachable)
        self.assertEqual(result.loaded.value, 2048)
        self.assertIsNone(result.ceiling.value)


if __name__ == "__main__":
    unittest.main()
