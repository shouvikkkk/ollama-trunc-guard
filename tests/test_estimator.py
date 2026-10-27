"""Tests for ollama_trunc_guard.estimator."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_trunc_guard.estimator import (
    chars_per_token_for,
    estimate_messages_tokens,
    estimate_model_tokens,
    estimate_tokens,
    estimate_tool_tokens,
)


class TestEstimateTokens(unittest.TestCase):
    def test_empty_text_costs_zero(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("   \n\t"), 0)

    def test_chars_rule(self):
        # 400 chars / 4 chars-per-token = 100 tokens.
        self.assertEqual(estimate_tokens("a" * 400), 100)

    def test_word_term_can_dominate(self):
        # 100 single-letter words: chars rule gives 50, words rule gives
        # ceil(100 * 1.3) = 130, so the max wins.
        text = " ".join(["a"] * 100)
        self.assertEqual(estimate_tokens(text), 130)

    def test_long_text_scales(self):
        text = "hello world " * 1000  # 12000 chars, 2000 words
        self.assertEqual(estimate_tokens(text), 3000)  # chars rule: 3000

    def test_invalid_ratio_rejected(self):
        with self.assertRaises(ValueError):
            estimate_tokens("hello", chars_per_token=0)

    def test_model_override_qwen(self):
        self.assertEqual(chars_per_token_for("qwen2.5:7b"), 3.8)
        self.assertEqual(chars_per_token_for("QWEN3"), 3.8)

    def test_model_override_unknown_family_falls_back(self):
        self.assertEqual(chars_per_token_for("some-new-model:1b"), 4.0)
        self.assertEqual(chars_per_token_for(None), 4.0)

    def test_estimate_model_tokens_uses_override(self):
        text = "x" * 380
        # qwen ratio 3.8 -> 100 tokens; default 4.0 -> 95 tokens.
        self.assertEqual(estimate_model_tokens(text, "qwen2.5"), 100)
        self.assertEqual(estimate_model_tokens(text, None), 95)


class TestToolTokens(unittest.TestCase):
    def test_tool_json_estimated_compactly(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        total, per_tool = estimate_tool_tokens(tools)
        self.assertEqual(len(per_tool), 1)
        self.assertEqual(per_tool[0]["name"], "read_file")
        self.assertEqual(total, per_tool[0]["tokens"])
        self.assertGreater(total, 0)

    def test_tool_without_name_gets_index_name(self):
        total, per_tool = estimate_tool_tokens([{"type": "function"}])
        self.assertEqual(per_tool[0]["name"], "tool[0]")
        self.assertGreater(total, 0)

    def test_many_tools_sum(self):
        tools = [
            {"type": "function", "function": {"name": f"tool_{i}", "description": "d" * 40}}
            for i in range(29)
        ]
        total, per_tool = estimate_tool_tokens(tools)
        self.assertEqual(len(per_tool), 29)
        self.assertEqual(total, sum(entry["tokens"] for entry in per_tool))

    def test_empty_tool_list(self):
        total, per_tool = estimate_tool_tokens([])
        self.assertEqual(total, 0)
        self.assertEqual(per_tool, [])


class TestMessageTokens(unittest.TestCase):
    def test_roles_and_order_preserved(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there friend"},
        ]
        out = estimate_messages_tokens(messages)
        self.assertEqual([entry["role"] for entry in out], ["user", "assistant"])
        self.assertEqual([entry["index"] for entry in out], [0, 1])
        self.assertTrue(all(entry["content_tokens"] > 0 for entry in out))

    def test_non_string_content_serialized(self):
        messages = [{"role": "user", "content": {"parts": ["a", "b"]}}]
        out = estimate_messages_tokens(messages)
        self.assertGreater(out[0]["content_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
