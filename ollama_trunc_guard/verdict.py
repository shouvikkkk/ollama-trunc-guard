"""Fail-closed verdicts and severity-gated exit codes.

Verdict levels, worst-first:

* ``FAIL``    — the run must not proceed. Exit code 2.
* ``WARN``    — the run may proceed, but headroom is thin. Exit code 1.
* ``OK``      — the prompt fits with comfortable headroom. Exit code 0.
* ``UNKNOWN`` — the effective window could not be determined. The tool
  FAILS CLOSED: it never claims OK on unknown data. Exit code 3.

FAIL conditions (any one suffices):

* the budget exceeds the effective window (something is truncated);
* truncation evidence was found in the server logs;
* the chosen endpoint cannot honor the needed window and the loaded
  window is smaller (see :mod:`ollama_trunc_guard.endpoints`).

WARN conditions (when nothing FAILs):

* headroom below 20% of the effective window;
* tool definitions consume more than 30% of the effective window
  (the October 2026 finding was 10-36% for tool definitions alone
  against Ollama's 4,096 default — above 30% leaves little room for
  actual conversation).

UNKNOWN conditions:

* no effective window could be determined from any probe source;
* the server was unreachable (no evidence at all is also UNKNOWN, not OK).

Severity ordering for combining verdicts: FAIL > UNKNOWN > WARN > OK.
UNKNOWN outranks WARN because "we don't know" is worse than "tight but
measured" — but FAIL still outranks UNKNOWN, since a known failure is
the strongest signal of all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

HEADROOM_WARN_FRACTION = 0.20
TOOL_SHARE_WARN_FRACTION = 0.30


class Verdict(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"

    @property
    def exit_code(self) -> int:
        return {"OK": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": 3}[self.value]


_SEVERITY = {"FAIL": 4, "UNKNOWN": 3, "WARN": 2, "OK": 1}


@dataclass
class VerdictResult:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return self.verdict.exit_code

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "exit_code": self.exit_code,
            "reasons": self.reasons,
        }


def combine(*verdicts: Verdict) -> Verdict:
    """Return the most severe verdict (FAIL > UNKNOWN > WARN > OK)."""
    if not verdicts:
        return Verdict.UNKNOWN
    return max(verdicts, key=lambda v: _SEVERITY[v.value])


def judge(
    *,
    effective_window: int | None,
    total_prompt_tokens: int,
    tool_tokens: int,
    reserve: int = 0,
    log_hits: int = 0,
    endpoint_verdict: str = "UNKNOWN",
    server_reachable: bool = True,
) -> VerdictResult:
    """Compute the overall verdict from all evidence.

    ``endpoint_verdict`` is the string verdict from
    :func:`ollama_trunc_guard.endpoints.endpoint_verdict` ("OK" / "FAIL"
    / "UNKNOWN"). Every parameter is evidence; nothing is defaulted to
    a reassuring value.
    """
    reasons: list[str] = []
    parts: list[Verdict] = []

    if not server_reachable:
        reasons.append("Ollama server unreachable; no window evidence available.")
        return VerdictResult(verdict=Verdict.UNKNOWN, reasons=reasons)

    if effective_window is None:
        reasons.append(
            "effective window unknown: no loaded, configured, or ceiling "
            "window could be determined from the server."
        )
        parts.append(Verdict.UNKNOWN)
    else:
        available = max(effective_window - reserve, 0)
        if total_prompt_tokens > available:
            overflow = total_prompt_tokens - available
            reasons.append(
                f"budget exceeds effective window: prompt {total_prompt_tokens} "
                f"tokens > available {available} tokens (window "
                f"{effective_window} minus reserve {reserve}); overflow "
                f"{overflow} tokens would be truncated from the front."
            )
            parts.append(Verdict.FAIL)
        else:
            headroom = available - total_prompt_tokens
            headroom_fraction = headroom / effective_window if effective_window else 0
            if headroom_fraction < HEADROOM_WARN_FRACTION:
                reasons.append(
                    f"headroom below 20%: only {headroom} tokens free of "
                    f"{effective_window} (estimator error is ~+/-25%, so "
                    "this margin is not safe)."
                )
                parts.append(Verdict.WARN)
            tool_share = tool_tokens / effective_window if effective_window else 0
            if tool_share > TOOL_SHARE_WARN_FRACTION:
                reasons.append(
                    f"tool definitions consume {tool_share:.0%} of the window "
                    f"({tool_tokens} of {effective_window} tokens), above the "
                    "30% warning threshold."
                )
                parts.append(Verdict.WARN)
            if not reasons:
                reasons.append(
                    f"prompt fits: {total_prompt_tokens} tokens within "
                    f"available {available} (window {effective_window}, "
                    f"reserve {reserve})."
                )
                parts.append(Verdict.OK)

    if log_hits > 0:
        reasons.append(
            f"truncation evidence found in server logs: {log_hits} hit(s)."
        )
        parts.append(Verdict.FAIL)

    endpoint_map = {"OK": Verdict.OK, "FAIL": Verdict.FAIL, "UNKNOWN": Verdict.UNKNOWN}
    endpoint = endpoint_map.get(endpoint_verdict, Verdict.UNKNOWN)
    if endpoint is Verdict.FAIL:
        reasons.append("endpoint cannot honor the needed window (see endpoint verdict).")
    elif endpoint is Verdict.UNKNOWN:
        reasons.append("endpoint verdict unknown (loaded window unknown).")
    parts.append(endpoint)

    overall = combine(*parts)
    return VerdictResult(verdict=overall, reasons=reasons)
