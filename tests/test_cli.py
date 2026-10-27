"""End-to-end CLI tests against the fake Ollama server."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_server import FakeOllamaServer
from ollama_trunc_guard.cli import main


def run_cli(argv):
    """Run the CLI, capturing stdout; returns (exit_code, stdout)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(argv)
    return code, buffer.getvalue()


def write_temp(content, suffix=".txt"):
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    handle.write(content)
    handle.close()
    return handle.name


SCENARIO_LOADED_4096 = {
    "tags": {"models": [{"name": "llama3.1:8b"}]},
    "ps": {"models": [{"name": "llama3.1:8b", "context_length": 4096}]},
    "shows": {
        "llama3.1:8b": {
            "model_info": {"llama.context_length": 131072},
            "parameters": {"num_ctx": "16384"},
        }
    },
}

SCENARIO_UNKNOWN = {
    "tags": {"models": [{"name": "mystery:1b"}]},
    "ps": {"models": [{"name": "mystery:1b"}]},
    "shows": {"mystery:1b": {}},
}


class TestCheckCLI(unittest.TestCase):
    def test_check_fail_on_overflow_names_lost_elements(self):
        big_prompt = "You are a coding agent. " * 800  # ~5200 tokens
        prompt_path = write_temp(big_prompt)
        tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": "x" * 200}} for i in range(10)]
        tools_path = write_temp(json.dumps(tools), suffix=".json")
        try:
            with FakeOllamaServer(SCENARIO_LOADED_4096) as server:
                code, out = run_cli([
                    "check", "--base-url", server.base_url, "--model", "llama3.1:8b",
                    "--prompt-file", prompt_path, "--tools-file", tools_path,
                    "--endpoint", "openai",
                ])
        finally:
            os.unlink(prompt_path)
            os.unlink(tools_path)
        self.assertEqual(code, 2)  # FAIL
        self.assertIn("VERDICT: FAIL", out)
        self.assertIn("LOST", out)

    def test_check_ok_on_small_prompt(self):
        with FakeOllamaServer(SCENARIO_LOADED_4096) as server:
            code, out = run_cli([
                "check", "--base-url", server.base_url, "--model", "llama3.1:8b",
                "--prompt", "Be helpful.",
            ])
        self.assertEqual(code, 0)
        self.assertIn("VERDICT: OK", out)

    def test_check_unknown_window_exits_3(self):
        with FakeOllamaServer(SCENARIO_UNKNOWN) as server:
            code, out = run_cli([
                "check", "--base-url", server.base_url, "--model", "mystery:1b",
                "--prompt", "Be helpful.",
            ])
        self.assertEqual(code, 3)
        self.assertIn("VERDICT: UNKNOWN", out)

    def test_check_unreachable_server_exits_3(self):
        code, out = run_cli([
            "check", "--base-url", "http://127.0.0.1:1", "--model", "x:1b",
            "--prompt", "Be helpful.",
        ])
        self.assertEqual(code, 3)
        self.assertIn("VERDICT: UNKNOWN", out)

    def test_check_json_output_parses(self):
        with FakeOllamaServer(SCENARIO_LOADED_4096) as server:
            code, out = run_cli([
                "check", "--base-url", server.base_url, "--model", "llama3.1:8b",
                "--prompt", "Be helpful.", "--json",
            ])
        data = json.loads(out)
        self.assertEqual(data["verdict"]["verdict"], "OK")
        self.assertEqual(data["probe"]["effective_window"], 4096)
        self.assertEqual(data["probe"]["effective_window_source"], "loaded (ps)")

    def test_check_log_hits_force_fail(self):
        log_path = write_temp("2026-10-22 10:15:32 truncating input from the beginning\n")
        try:
            with FakeOllamaServer(SCENARIO_LOADED_4096) as server:
                code, out = run_cli([
                    "check", "--base-url", server.base_url, "--model", "llama3.1:8b",
                    "--prompt", "Be helpful.", "--log-file", log_path,
                ])
        finally:
            os.unlink(log_path)
        self.assertEqual(code, 2)
        self.assertIn("VERDICT: FAIL", out)


class TestSimulateCLI(unittest.TestCase):
    def test_simulate_reports_fall_out_turn(self):
        messages = [{"role": "user", "content": "x" * 400}]  # ~100 tokens
        messages_path = write_temp(json.dumps(messages), suffix=".json")
        prompt_path = write_temp("System instructions. " * 20)  # ~105 tokens
        try:
            code, out = run_cli([
                "simulate", "--messages-file", messages_path,
                "--prompt-file", prompt_path, "--window", "1000",
                "--growth", "200", "--max-turns", "20",
            ])
        finally:
            os.unlink(messages_path)
            os.unlink(prompt_path)
        self.assertEqual(code, 0)
        self.assertIn("FALLS OUT on turn", out)

    def test_simulate_without_window_or_model_exits_3(self):
        code, _ = run_cli(["simulate", "--prompt", "hi"])
        self.assertEqual(code, 3)


class TestScanLogsCLI(unittest.TestCase):
    def test_scan_logs_hit_exits_2(self):
        log_path = write_temp("WARN prompt too long, truncating\n")
        try:
            code, out = run_cli(["scan-logs", "--log-file", log_path])
        finally:
            os.unlink(log_path)
        self.assertEqual(code, 2)
        self.assertIn("1 hit(s)", out)

    def test_scan_logs_clean_exits_0(self):
        log_path = write_temp("all good\n")
        try:
            code, _ = run_cli(["scan-logs", "--log-file", log_path])
        finally:
            os.unlink(log_path)
        self.assertEqual(code, 0)

    def test_scan_logs_json(self):
        log_path = write_temp("truncating input here\n")
        try:
            code, out = run_cli(["scan-logs", "--log-file", log_path, "--json"])
        finally:
            os.unlink(log_path)
        data = json.loads(out)
        self.assertEqual(data["scan"]["hit_count"], 1)


class TestAuditConfigCLI(unittest.TestCase):
    def test_audit_mismatch_exits_2(self):
        modelfile_path = write_temp("FROM x\nPARAMETER num_ctx 16384\n", suffix="Modelfile")
        try:
            code, out = run_cli([
                "audit-config", "--modelfile", modelfile_path,
                "--loaded-window", "4096",
            ])
        finally:
            os.unlink(modelfile_path)
        self.assertEqual(code, 2)
        self.assertIn("MISMATCH", out)

    def test_audit_match_exits_0(self):
        modelfile_path = write_temp("FROM x\nPARAMETER num_ctx 4096\n", suffix="Modelfile")
        try:
            code, _ = run_cli([
                "audit-config", "--modelfile", modelfile_path,
                "--loaded-window", "4096",
            ])
        finally:
            os.unlink(modelfile_path)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
