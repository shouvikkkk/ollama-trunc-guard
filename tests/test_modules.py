"""Tests for endpoints, logscan, config, and verdict modules."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_trunc_guard.config import (
    audit_config,
    parse_modelfile_num_ctx,
    read_env_num_ctx,
)
from ollama_trunc_guard.endpoints import ENDPOINTS, endpoint_verdict
from ollama_trunc_guard.logscan import scan_log_text
from ollama_trunc_guard.verdict import Verdict, combine, judge


class TestEndpoints(unittest.TestCase):
    def test_openai_small_loaded_window_is_fail(self):
        verdict, reason = endpoint_verdict("openai", needed_window=16384, loaded_window=4096)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("4096", reason)

    def test_openai_large_loaded_window_is_ok(self):
        verdict, _ = endpoint_verdict("openai", needed_window=4096, loaded_window=16384)
        self.assertEqual(verdict, "OK")

    def test_native_api_chat_honors_num_ctx(self):
        verdict, _ = endpoint_verdict("api-chat", needed_window=16384, loaded_window=4096)
        self.assertEqual(verdict, "OK")

    def test_native_generate_honors_num_ctx(self):
        verdict, _ = endpoint_verdict("generate", needed_window=32768, loaded_window=4096)
        self.assertEqual(verdict, "OK")

    def test_unknown_loaded_window_is_unknown(self):
        verdict, _ = endpoint_verdict("openai", needed_window=16384, loaded_window=None)
        self.assertEqual(verdict, "UNKNOWN")

    def test_unknown_endpoint_is_unknown(self):
        verdict, _ = endpoint_verdict("nope", needed_window=100, loaded_window=100)
        self.assertEqual(verdict, "UNKNOWN")

    def test_table_marks_openai_as_not_honoring_num_ctx(self):
        self.assertFalse(ENDPOINTS["openai"].honors_per_request_num_ctx)
        self.assertTrue(ENDPOINTS["api-chat"].honors_per_request_num_ctx)


class TestLogscan(unittest.TestCase):
    SAMPLE = """2026-10-22 10:15:32 [GIN] truncating input from the beginning model=llama3.1:8b
2026-10-22 10:16:01 some unrelated info line
2026-10-22 10:17:44 WARN prompt too long, truncating
10:18:02 ERROR input exceeds context length for model: qwen2.5:7b
a line about TRUNCATED history
"""

    def test_patterns_matched_and_counted(self):
        result = scan_log_text(self.SAMPLE)
        self.assertEqual(result.hit_count, 4)
        self.assertEqual(result.counts_by_pattern["truncating input"], 1)
        self.assertEqual(result.counts_by_pattern["prompt too long"], 1)
        self.assertEqual(result.counts_by_pattern["exceeds context"], 1)
        self.assertEqual(result.counts_by_pattern["truncat"], 1)

    def test_most_specific_pattern_wins(self):
        # "truncating input" line also contains "truncat" but must be
        # labelled with the specific pattern.
        result = scan_log_text("truncating input from the beginning\n")
        self.assertEqual(result.hits[0]["pattern"], "truncating input")

    def test_case_insensitive(self):
        result = scan_log_text("TRUNCATING INPUT detected\n")
        self.assertEqual(result.hit_count, 1)

    def test_timestamps_and_line_numbers(self):
        result = scan_log_text(self.SAMPLE)
        first = result.hits[0]
        self.assertEqual(first["line_number"], 1)
        self.assertEqual(first["timestamp"], "2026-10-22 10:15:32")
        self.assertEqual(first["model"], "llama3.1:8b")

    def test_counts_by_model(self):
        result = scan_log_text(self.SAMPLE)
        self.assertEqual(result.counts_by_model, {"llama3.1:8b": 1, "qwen2.5:7b": 1})

    def test_clean_log_has_no_hits(self):
        result = scan_log_text("everything is fine\nno problems here\n")
        self.assertEqual(result.hit_count, 0)
        self.assertEqual(result.total_lines, 2)

    def test_empty_text(self):
        result = scan_log_text("")
        self.assertEqual(result.hit_count, 0)


class TestConfig(unittest.TestCase):
    def test_parse_modelfile_num_ctx(self):
        text = "FROM llama3.1:8b\nPARAMETER num_ctx 16384\nPARAMETER temperature 0.7\n"
        self.assertEqual(parse_modelfile_num_ctx(text), 16384)

    def test_last_parameter_wins(self):
        text = "PARAMETER num_ctx 4096\nPARAMETER num_ctx 8192\n"
        self.assertEqual(parse_modelfile_num_ctx(text), 8192)

    def test_absent_parameter_is_none(self):
        self.assertIsNone(parse_modelfile_num_ctx("FROM llama3.1:8b\n"))

    def test_env_var_parsed(self):
        self.assertEqual(read_env_num_ctx({"OLLAMA_CONTEXT_LENGTH": "32768"}), 32768)
        self.assertIsNone(read_env_num_ctx({}))
        self.assertIsNone(read_env_num_ctx({"OLLAMA_CONTEXT_LENGTH": "not-a-number"}))

    def test_declared_vs_loaded_mismatch(self):
        audit = audit_config(loaded_window=4096, ceiling=131072, environ={})
        # No modelfile: no mismatch, just notes.
        self.assertEqual(audit.mismatches, [])

    def test_modelfile_mismatch_reported(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix="Modelfile", delete=False) as fh:
            fh.write("FROM llama3.1:8b\nPARAMETER num_ctx 16384\n")
            path = fh.name
        try:
            audit = audit_config(modelfile_path=path, loaded_window=4096, environ={})
        finally:
            os.unlink(path)
        self.assertEqual(audit.modelfile_num_ctx, 16384)
        self.assertEqual(len(audit.mismatches), 1)
        self.assertIn("16384", audit.mismatches[0])
        self.assertIn("4096", audit.mismatches[0])

    def test_declared_above_ceiling_flagged(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix="Modelfile", delete=False) as fh:
            fh.write("PARAMETER num_ctx 999999\n")
            path = fh.name
        try:
            audit = audit_config(modelfile_path=path, loaded_window=999999, ceiling=131072, environ={})
        finally:
            os.unlink(path)
        self.assertTrue(any("ceiling" in m for m in audit.mismatches))


class TestVerdict(unittest.TestCase):
    def test_fitting_prompt_is_ok(self):
        result = judge(
            effective_window=4096, total_prompt_tokens=2000, tool_tokens=200,
            endpoint_verdict="OK",
        )
        self.assertEqual(result.verdict, Verdict.OK)
        self.assertEqual(result.exit_code, 0)

    def test_overflow_is_fail(self):
        result = judge(
            effective_window=4096, total_prompt_tokens=5000, tool_tokens=200,
            endpoint_verdict="OK",
        )
        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertEqual(result.exit_code, 2)

    def test_headroom_below_20_percent_is_warn(self):
        # 4096 window, 3500 tokens used -> headroom 596 = 14.5%.
        result = judge(
            effective_window=4096, total_prompt_tokens=3500, tool_tokens=100,
            endpoint_verdict="OK",
        )
        self.assertEqual(result.verdict, Verdict.WARN)
        self.assertEqual(result.exit_code, 1)

    def test_tool_share_above_30_percent_is_warn(self):
        result = judge(
            effective_window=4096, total_prompt_tokens=2000, tool_tokens=1500,
            endpoint_verdict="OK",
        )
        self.assertEqual(result.verdict, Verdict.WARN)

    def test_unknown_window_is_unknown_exit_3(self):
        result = judge(
            effective_window=None, total_prompt_tokens=2000, tool_tokens=100,
            endpoint_verdict="UNKNOWN",
        )
        self.assertEqual(result.verdict, Verdict.UNKNOWN)
        self.assertEqual(result.exit_code, 3)

    def test_unreachable_server_is_unknown(self):
        result = judge(
            effective_window=None, total_prompt_tokens=0, tool_tokens=0,
            endpoint_verdict="UNKNOWN", server_reachable=False,
        )
        self.assertEqual(result.verdict, Verdict.UNKNOWN)

    def test_log_hits_force_fail(self):
        result = judge(
            effective_window=4096, total_prompt_tokens=1000, tool_tokens=100,
            log_hits=3, endpoint_verdict="OK",
        )
        self.assertEqual(result.verdict, Verdict.FAIL)

    def test_endpoint_fail_forces_fail(self):
        result = judge(
            effective_window=4096, total_prompt_tokens=1000, tool_tokens=100,
            endpoint_verdict="FAIL",
        )
        self.assertEqual(result.verdict, Verdict.FAIL)

    def test_combine_severity_order(self):
        self.assertEqual(combine(Verdict.OK, Verdict.WARN), Verdict.WARN)
        self.assertEqual(combine(Verdict.WARN, Verdict.UNKNOWN), Verdict.UNKNOWN)
        self.assertEqual(combine(Verdict.UNKNOWN, Verdict.FAIL), Verdict.FAIL)
        self.assertEqual(combine(), Verdict.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
