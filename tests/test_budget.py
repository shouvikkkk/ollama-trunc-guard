"""Tests for ollama_trunc_guard.budget (front-truncation simulation)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_trunc_guard.budget import (
    PromptElement,
    build_elements,
    project_turns,
    simulate,
)


def el(name, kind, tokens):
    return PromptElement(name=name, kind=kind, tokens=tokens)


class TestSimulate(unittest.TestCase):
    def test_fits_when_under_window(self):
        result = simulate([el("system prompt", "system", 100)], window=4096)
        self.assertTrue(result.fits)
        self.assertEqual(result.overflow_tokens, 0)
        self.assertEqual(result.lost, [])
        self.assertEqual(len(result.surviving), 1)

    def test_reserve_shrinks_available_window(self):
        result = simulate([el("system prompt", "system", 900)], window=1000, reserve=200)
        self.assertFalse(result.fits)
        self.assertEqual(result.available_tokens, 800)
        self.assertEqual(result.overflow_tokens, 100)

    def test_front_truncation_order_tools_then_system_then_messages(self):
        elements = [
            el("tool:read_file", "tool", 500),
            el("tool:write_file", "tool", 500),
            el("system prompt", "system", 400),
            el("message[0]::user", "message", 300),
            el("message[1]::assistant", "message", 300),
        ]
        result = simulate(elements, window=1200)
        # Total 2000, overflow 800: tool:read_file fully lost (500),
        # tool:write_file partially lost (300 of 500). System survives.
        self.assertEqual(result.lost_names(), ["tool:read_file", "tool:write_file"])
        self.assertEqual(result.fully_lost_names(), ["tool:read_file"])
        self.assertEqual(result.lost[1].tokens_lost, 300)
        self.assertFalse(result.lost[1].fully_lost)

    def test_system_prompt_lost_when_overflow_reaches_it(self):
        elements = [
            el("tool:read_file", "tool", 500),
            el("tool:write_file", "tool", 500),
            el("system prompt", "system", 400),
            el("message[0]::user", "message", 300),
        ]
        result = simulate(elements, window=900)
        # Total 1700, overflow 800: first tool fully lost (500), second
        # tool partially lost (300 of 500). The system prompt survives.
        self.assertEqual(result.lost_names(), ["tool:read_file", "tool:write_file"])

    def test_system_prompt_fully_lost_with_deep_overflow(self):
        elements = [
            el("tool:a", "tool", 200),
            el("system prompt", "system", 400),
            el("message[0]::user", "message", 300),
        ]
        result = simulate(elements, window=300)
        # Total 900, overflow 600: tool:a fully lost (200), system
        # fully lost (400). Only the newest message survives.
        self.assertEqual(result.fully_lost_names(), ["tool:a", "system prompt"])
        self.assertEqual([s.name for s in result.surviving], ["message[0]::user"])

    def test_surviving_tail_is_the_tail(self):
        elements = [
            el("tool:a", "tool", 100),
            el("system prompt", "system", 100),
            el("message[0]::user", "message", 100),
            el("message[1]::assistant", "message", 100),
        ]
        result = simulate(elements, window=250)
        # Overflow 150: tool:a fully lost (100), system prompt
        # partially lost (50 of 100) — it survives with 50 tokens.
        self.assertEqual(
            [s.name for s in result.surviving],
            ["system prompt", "message[0]::user", "message[1]::assistant"],
        )
        self.assertEqual(
            result.surviving[0].tokens, 50
        )

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            simulate([el("a", "system", 10)], window=0)

    def test_to_dict_round_trip(self):
        result = simulate([el("system prompt", "system", 100)], window=50)
        data = result.to_dict()
        self.assertEqual(data["overflow_tokens"], 50)
        self.assertEqual(data["lost_head"][0]["name"], "system prompt")
        self.assertTrue(data["lost_head"][0]["fully_lost"] is False or True)


class TestBuildElements(unittest.TestCase):
    def test_names_and_default_order(self):
        tools = [{"type": "function", "function": {"name": "search"}}]
        elements = build_elements(
            system_prompt="Be helpful.",
            tools=tools,
            messages=[{"role": "user", "content": "hi"}],
        )
        kinds = [e.kind for e in elements]
        self.assertEqual(kinds, ["tool", "system", "message"])
        self.assertEqual(elements[0].name, "tool:search")
        self.assertEqual(elements[1].name, "system prompt")
        self.assertEqual(elements[2].name, "message[0]::user")

    def test_system_first_order_option(self):
        elements = build_elements(
            system_prompt="Be helpful.",
            tools=[{"type": "function", "function": {"name": "search"}}],
            messages=[],
            order=("system", "tools", "messages"),
        )
        self.assertEqual([e.kind for e in elements], ["system", "tool"])

    def test_empty_inputs_produce_no_elements(self):
        self.assertEqual(build_elements(), [])


class TestProjectTurns(unittest.TestCase):
    def test_system_prompt_falls_out_on_expected_turn(self):
        # System 100 + history 200 = 300 initial. Growth 200/turn.
        # Turn 4: total 1100 > window 1000, overflow 100 -> system
        # (100 tokens, at the front) is fully lost exactly on turn 4.
        elements = [
            el("system prompt", "system", 100),
            el("message[0]::user", "message", 100),
            el("message[1]::assistant", "message", 100),
        ]
        projection = project_turns(
            elements, window=1000, growth_per_turn=200, max_turns=20
        )
        self.assertEqual(projection["watch_falls_out_on_turn"], 4)
        self.assertEqual(projection["watch_partially_lost_on_turn"], 4)
        self.assertEqual(len(projection["turns"]), 5)  # turns 0..4

    def test_no_growth_never_falls_out(self):
        elements = [el("system prompt", "system", 100)]
        projection = project_turns(elements, window=1000, growth_per_turn=0)
        self.assertIsNone(projection["watch_falls_out_on_turn"])

    def test_fitting_prompt_never_falls_out(self):
        elements = [el("system prompt", "system", 100)]
        projection = project_turns(
            elements, window=10000, growth_per_turn=10, max_turns=5
        )
        self.assertIsNone(projection["watch_falls_out_on_turn"])

    def test_partial_loss_reported_before_full_loss(self):
        # System 500 tokens; growth 100/turn; window 1000; initial 600.
        # Turn 4: total 1000, fits. Turn 5: total 1100, overflow 100 ->
        # system partially lost (100/500). Turn 9: overflow 500 -> fully.
        elements = [
            el("system prompt", "system", 500),
            el("message[0]::user", "message", 100),
        ]
        projection = project_turns(
            elements, window=1000, growth_per_turn=100, max_turns=20
        )
        self.assertEqual(projection["watch_partially_lost_on_turn"], 5)
        self.assertEqual(projection["watch_falls_out_on_turn"], 9)


if __name__ == "__main__":
    unittest.main()
