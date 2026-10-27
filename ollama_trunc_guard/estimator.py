"""Token-count estimation for ollama-trunc-guard.

This is a HEURISTIC, not a tokenizer. Ollama models ship with real
tokenizers (SentencePiece / BPE / TikToken-style), and reproducing any of
them exactly would require model-specific vocabularies. What this module
does instead is a transparent, conservative approximation that is good
enough for a pre-flight gate:

    tokens ~= max(ceil(chars / chars_per_token), ceil(words * 1.3))

The ``chars / 4`` rule is the widely used rough English-text estimate
(~4 characters per token). The word-count term guards against text that
is dense in short words or punctuation, where chars/4 would undercount.
Taking the MAX of the two is deliberate: for a fail-closed gate,
overestimating the prompt is safer than underestimating it.

Measured accuracy: on ordinary English prose this heuristic typically
lands within roughly +/-25% of a real tokenizer's count. It is worse on
code with long identifiers, non-English text, and heavily formatted JSON
(tool definitions). The README states this limitation plainly, and the
verdict logic treats headroom < 20% as WARN for exactly this reason.

Per-model overrides let callers tune the chars-per-token ratio for a
model family. These are documented defaults, not measured facts about
any specific Ollama release; the caller can override them at runtime.
"""

from __future__ import annotations

import json
import math

DEFAULT_CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN_FACTOR = 1.3

# Documented default chars-per-token by model-family substring.
# Values are starting points for the heuristic, tuned so that families
# known for tokenizing code/English slightly differently get a nudge.
# Matching is case-insensitive substring search, first match wins.
MODEL_CHARS_PER_TOKEN = {
    "qwen": 3.8,      # Qwen tokenizers are slightly more compact on code.
    "llama": 4.0,     # Llama-family SentencePiece, close to the default.
    "mistral": 4.0,
    "gemma": 3.9,
    "phi": 4.0,
    "deepseek": 3.8,
    "codellama": 3.6,  # Code-heavy text tokenizes denser per char.
    "starcoder": 3.6,
}


def chars_per_token_for(model: str | None) -> float:
    """Return the chars-per-token ratio for a model name, or the default."""
    if not model:
        return DEFAULT_CHARS_PER_TOKEN
    lowered = model.lower()
    for family, ratio in MODEL_CHARS_PER_TOKEN.items():
        if family in lowered:
            return ratio
    return DEFAULT_CHARS_PER_TOKEN


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate tokens for *text* using the max(chars, words) heuristic.

    Empty/whitespace-only text costs 0 tokens. The result is always an
    integer >= 0.
    """
    if not text or not text.strip():
        return 0
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    by_chars = math.ceil(len(text) / chars_per_token)
    word_count = len(text.split())
    by_words = math.ceil(word_count * WORDS_PER_TOKEN_FACTOR)
    return max(by_chars, by_words)


def estimate_model_tokens(text: str, model: str | None) -> int:
    """Estimate tokens using the per-model chars-per-token override."""
    return estimate_tokens(text, chars_per_token_for(model))


def estimate_tool_tokens(tools: list[dict]) -> tuple[int, list[dict]]:
    """Estimate the token cost of a list of OpenAI-style tool definitions.

    Each tool is serialized to compact JSON (no insignificant whitespace)
    and estimated separately, so the caller can report per-tool cost and
    the tool share of the window. Returns ``(total, per_tool)`` where
    ``per_tool`` is a list of ``{"name": ..., "tokens": ...}`` dicts.
    A tool without a ``function.name`` falls back to its index as name.
    """
    per_tool: list[dict] = []
    total = 0
    for index, tool in enumerate(tools):
        compact = json.dumps(tool, separators=(",", ":"), ensure_ascii=False)
        tokens = estimate_tokens(compact)
        name = None
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
        per_tool.append({"name": name or f"tool[{index}]", "tokens": tokens})
        total += tokens
    return total, per_tool


def estimate_messages_tokens(messages: list[dict], model: str | None = None) -> list[dict]:
    """Estimate tokens per message dict, preserving order and roles.

    Each message dict is expected to carry ``role`` and ``content`` keys.
    Returns a list of ``{"role": ..., "content_tokens": ..., "index": ...}``
    entries in the original order.
    """
    out = []
    for index, message in enumerate(messages):
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(content, str):
            content = json.dumps(content, separators=(",", ":"), ensure_ascii=False)
        out.append(
            {
                "index": index,
                "role": message.get("role", f"message[{index}]") if isinstance(message, dict) else f"message[{index}]",
                "content_tokens": estimate_model_tokens(content, model),
            }
        )
    return out
