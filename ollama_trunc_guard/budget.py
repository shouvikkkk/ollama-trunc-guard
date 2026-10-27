"""Front-truncation simulation for ollama-trunc-guard.

MODEL ASSUMPTION (read this before trusting a report)
-----------------------------------------------------
Ollama truncates an over-long prompt from the BEGINNING of the fully
constructed prompt and keeps the tail. It does not raise an error and,
on the paths this tool studies, it still returns HTTP 200. This module
models the constructed prompt as, in order:

    1. tool definitions (in definition order)
    2. system prompt
    3. conversation history, oldest message first

Tool definitions sit at the front of the constructed prompt in the
agent setups this tool was built for, so under front-truncation they are
the FIRST elements lost — the exact failure reported in the orca_forge
case, where an agent's tool definitions vanished silently. The system
prompt follows, then the oldest messages. The newest messages and the
reserved output budget are never truncated (the reserve is output
space, not input, so it shrinks the available input window instead).

Caveat: the exact placement of tool definitions relative to the system
prompt depends on the client and chat template in use. If your client
places the system prompt before the tools, pass
``order=("system", "tools", "messages")`` and the loss order follows.
The report always names the order it assumed, so a mismatch is visible
rather than silent.

What this module returns
------------------------
``simulate()`` reports, for one prompt at one effective window:

* total prompt tokens, available input tokens (window minus reserve),
* overflow tokens (0 when everything fits),
* the lost head: each element lost, fully or partially, by name,
* the surviving tail: each element kept, by name.

``project_turns()`` extends this across turns: given a per-turn growth
rate (new tokens appended per turn, e.g. a fresh user/assistant exchange),
it finds the turn on which the original system prompt (or any named
element) is pushed out of the window entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .estimator import estimate_messages_tokens, estimate_model_tokens, estimate_tool_tokens

DEFAULT_ORDER = ("tools", "system", "messages")


@dataclass
class PromptElement:
    """One named chunk of the constructed prompt, in prompt order."""

    name: str
    kind: str  # "tool" | "system" | "message"
    tokens: int


@dataclass
class ElementLoss:
    name: str
    kind: str
    tokens: int
    tokens_lost: int

    @property
    def fully_lost(self) -> bool:
        return self.tokens_lost >= self.tokens


@dataclass
class TruncationResult:
    elements: list[PromptElement]
    window: int
    reserve: int
    total_tokens: int
    available_tokens: int
    overflow_tokens: int
    lost: list[ElementLoss] = field(default_factory=list)
    surviving: list[PromptElement] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.overflow_tokens == 0

    def lost_names(self) -> list[str]:
        return [loss.name for loss in self.lost]

    def fully_lost_names(self) -> list[str]:
        return [loss.name for loss in self.lost if loss.fully_lost]

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "reserve_output_tokens": self.reserve,
            "total_prompt_tokens": self.total_tokens,
            "available_input_tokens": self.available_tokens,
            "overflow_tokens": self.overflow_tokens,
            "fits": self.fits,
            "lost_head": [
                {
                    "name": loss.name,
                    "kind": loss.kind,
                    "tokens": loss.tokens,
                    "tokens_lost": loss.tokens_lost,
                    "fully_lost": loss.fully_lost,
                }
                for loss in self.lost
            ],
            "surviving_tail": [
                {"name": el.name, "kind": el.kind, "tokens": el.tokens}
                for el in self.surviving
            ],
        }


def build_elements(
    system_prompt: str = "",
    tools: list[dict] | None = None,
    messages: list[dict] | None = None,
    model: str | None = None,
    order: tuple[str, ...] = DEFAULT_ORDER,
) -> list[PromptElement]:
    """Build the constructed prompt's elements in prompt order.

    Element names are stable and human-readable: tool elements are named
    ``tool:<function-name>``, the system prompt is ``system prompt``, and
    messages are ``message[<index>]:<role>``.
    """
    tools = tools or []
    messages = messages or []
    _, per_tool = estimate_tool_tokens(tools)
    tool_elements = [
        PromptElement(name=f"tool:{entry['name']}", kind="tool", tokens=entry["tokens"])
        for entry in per_tool
    ]
    system_elements = []
    if system_prompt:
        system_elements = [
            PromptElement(
                name="system prompt",
                kind="system",
                tokens=estimate_model_tokens(system_prompt, model),
            )
        ]
    per_message = estimate_messages_tokens(messages, model)
    message_elements = [
        PromptElement(
            name=f"message[{entry['index']}]::{entry['role']}",
            kind="message",
            tokens=entry["content_tokens"],
        )
        for entry in per_message
    ]
    blocks = {
        "tools": tool_elements,
        "system": system_elements,
        "messages": message_elements,
    }
    elements: list[PromptElement] = []
    for block_name in order:
        elements.extend(blocks.get(block_name, []))
    return elements


def simulate(
    elements: list[PromptElement],
    window: int,
    reserve: int = 0,
) -> TruncationResult:
    """Simulate front-truncation of *elements* at *window* tokens.

    ``reserve`` output tokens are subtracted from the window first; the
    remaining budget is the available input window. Overflow is consumed
    from the front of the element list, element by element.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if reserve < 0:
        raise ValueError("reserve must be non-negative")
    available = max(window - reserve, 0)
    total = sum(el.tokens for el in elements)
    overflow = max(total - available, 0)
    lost: list[ElementLoss] = []
    surviving: list[PromptElement] = []
    remaining_overflow = overflow
    for element in elements:
        if remaining_overflow <= 0:
            surviving.append(element)
            continue
        tokens_lost = min(element.tokens, remaining_overflow)
        lost.append(
            ElementLoss(
                name=element.name,
                kind=element.kind,
                tokens=element.tokens,
                tokens_lost=tokens_lost,
            )
        )
        remaining_overflow -= tokens_lost
        kept = element.tokens - tokens_lost
        if kept > 0:
            surviving.append(
                PromptElement(name=element.name, kind=element.kind, tokens=kept)
            )
    return TruncationResult(
        elements=list(elements),
        window=window,
        reserve=reserve,
        total_tokens=total,
        available_tokens=available,
        overflow_tokens=overflow,
        lost=lost,
        surviving=surviving,
    )


def project_turns(
    elements: list[PromptElement],
    window: int,
    reserve: int = 0,
    growth_per_turn: int = 0,
    max_turns: int = 200,
    watch_name: str = "system prompt",
) -> dict:
    """Project, turn by turn, when *watch_name* falls out of the window.

    Each turn appends ``growth_per_turn`` tokens to the tail (a new
    exchange) and re-runs front-truncation. Returns a dict with:

    * ``turns``: per-turn snapshots (turn, total, overflow, lost names),
    * ``watch_falls_out_on_turn``: the first turn where the watched
      element is fully lost, or None if it survives all ``max_turns``,
    * ``watch_partially_lost_on_turn``: first turn it loses any token.

    Turn 0 is the initial prompt before any growth.
    """
    if growth_per_turn < 0:
        raise ValueError("growth_per_turn must be non-negative")
    turns: list[dict] = []
    falls_out = None
    partially_lost = None
    for turn in range(max_turns + 1):
        grown = list(elements)
        if turn > 0 and growth_per_turn:
            # Growth accumulates: every past turn's exchange stays in
            # history, so rebuild with all turns so far.
            grown = list(elements) + [
                PromptElement(
                    name=f"turn[{t}] exchange", kind="message", tokens=growth_per_turn
                )
                for t in range(1, turn + 1)
            ]
        result = simulate(grown, window, reserve)
        lost_map = {loss.name: loss for loss in result.lost}
        watch = lost_map.get(watch_name)
        if watch is not None and partially_lost is None:
            partially_lost = turn
        if watch is not None and watch.fully_lost and falls_out is None:
            falls_out = turn
        turns.append(
            {
                "turn": turn,
                "total_prompt_tokens": result.total_tokens,
                "overflow_tokens": result.overflow_tokens,
                "lost_names": result.lost_names(),
                "fits": result.fits,
            }
        )
        if falls_out is not None:
            break
    return {
        "window": window,
        "reserve_output_tokens": reserve,
        "growth_per_turn": growth_per_turn,
        "watch_name": watch_name,
        "watch_partially_lost_on_turn": partially_lost,
        "watch_falls_out_on_turn": falls_out,
        "turns": turns,
    }
