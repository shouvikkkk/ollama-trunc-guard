"""Per-endpoint num_ctx capability table for ollama-trunc-guard.

This table is DOCUMENTED FROM EVIDENCE, not guessed, and it is
version-dependent — Ollama's behaviour here has changed across releases,
so every entry carries the evidence it rests on and the date it was
recorded. Re-verify against your installed Ollama version before
treating a verdict as gospel.

Evidence base (October 2026):

* Native ``/api/chat`` and ``/api/generate`` accept an ``options``
  object per request, and ``options.num_ctx`` is the documented way to
  set the context window for that request. A per-request window set here
  is honored (subject to the model ceiling and available VRAM).
* The OpenAI-compatible ``/v1/chat/completions`` path is where the
  orca_forge failure happened: the author expected a 16,384 window, the
  model was actually loaded at 4,096, and the OpenAI path did not reflect
  the expected window — the prompt was silently truncated from the
  beginning while the endpoint still returned HTTP 200. Historically
  this path has NOT applied a per-request ``num_ctx`` the way the
  native paths do, so a caller that needs a window larger than the
  loaded one cannot fix it per-request on this endpoint; the fix is to
  restart/reload the model with the right window (Modelfile
  ``PARAMETER num_ctx`` or ``OLLAMA_CONTEXT_LENGTH``), not a request
  parameter.

Verdict rule used by :mod:`ollama_trunc_guard.verdict`: when the user
needs a window the endpoint cannot set per-request AND the loaded
window is smaller than needed, the endpoint verdict is FAIL — the run
cannot be saved by a request parameter, only by reloading the model.
"""

from __future__ import annotations

from dataclasses import dataclass

TABLE_DATE = "2026-10-27"
TABLE_NOTE = (
    "Version-dependent and dated: Ollama endpoint behaviour has changed "
    "across releases. Re-verify against your installed version."
)


@dataclass(frozen=True)
class EndpointCapability:
    """What one endpoint can and cannot do about context windows."""

    key: str  # CLI value: "api-chat" | "generate" | "openai"
    path: str
    honors_per_request_num_ctx: bool
    evidence: str


ENDPOINTS: dict[str, EndpointCapability] = {
    "api-chat": EndpointCapability(
        key="api-chat",
        path="/api/chat",
        honors_per_request_num_ctx=True,
        evidence=(
            "Native Ollama API: options.num_ctx is honored per request "
            "(documented Ollama API behaviour)."
        ),
    ),
    "generate": EndpointCapability(
        key="generate",
        path="/api/generate",
        honors_per_request_num_ctx=True,
        evidence=(
            "Native Ollama API: options.num_ctx is honored per request "
            "(documented Ollama API behaviour)."
        ),
    ),
    "openai": EndpointCapability(
        key="openai",
        path="/v1/chat/completions",
        honors_per_request_num_ctx=False,
        evidence=(
            "orca_forge, dev.to, 2026-10-22: app expected 16,384, model "
            "loaded at 4,096, OpenAI-compatible path did not reflect the "
            "expected window; input silently truncated from the beginning "
            "with HTTP 200 returned."
        ),
    ),
}

ENDPOINT_CHOICES = tuple(ENDPOINTS.keys())


def endpoint_verdict(
    endpoint_key: str,
    needed_window: int,
    loaded_window: int | None,
) -> tuple[str, str]:
    """Return ``(verdict, reason)`` for one endpoint at one loaded window.

    * ``loaded_window is None`` → UNKNOWN: without a loaded window we
      cannot say whether the endpoint can serve the need.
    * Endpoint honors per-request num_ctx → OK (the caller can request
      the window; the model ceiling / VRAM still bounds it, which the
      budget verdict covers separately).
    * Endpoint cannot set per-request num_ctx and loaded < needed →
      FAIL: no request parameter can rescue this run.
    * Endpoint cannot set per-request num_ctx and loaded >= needed →
      OK: the loaded window already covers the need.
    """
    capability = ENDPOINTS.get(endpoint_key)
    if capability is None:
        return "UNKNOWN", f"unknown endpoint '{endpoint_key}'"
    if loaded_window is None:
        return (
            "UNKNOWN",
            f"loaded window unknown; cannot judge {capability.path} against "
            f"needed window {needed_window}",
        )
    if capability.honors_per_request_num_ctx:
        return (
            "OK",
            f"{capability.path} honors per-request num_ctx; caller can request "
            f"{needed_window} (model ceiling / VRAM still apply).",
        )
    if loaded_window < needed_window:
        return (
            "FAIL",
            f"{capability.path} does not apply per-request num_ctx and the "
            f"loaded window ({loaded_window}) is smaller than the needed "
            f"window ({needed_window}); reload the model with a larger "
            f"window before running.",
        )
    return (
        "OK",
        f"{capability.path} does not apply per-request num_ctx, but the "
        f"loaded window ({loaded_window}) already covers the needed window "
        f"({needed_window}).",
    )
