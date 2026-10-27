"""Ollama HTTP probing over urllib (stdlib only).

This module talks to a running Ollama server and reports what the server
actually says — never inventing a field the server did not return. That
discipline is the whole point of this tool: the failure mode it guards
against is a client *assuming* a context window that the server never
confirmed.

Endpoints used (all part of Ollama's documented local API):

* ``GET /api/tags`` — installed models (``models[].name``).
* ``GET /api/ps`` — currently running models. The fields Ollama returns
  here have varied across versions (``size``, ``size_vram``, and, in some
  releases, ``context_length`` / ``context``-related fields). This module
  reads whatever fields are present and records their provenance; it
  NEVER fabricates a context field that is absent.
* ``POST /api/show`` — model details: ``model_info`` (which may contain
  an architecture context ceiling, e.g. ``llama.context_length``),
  ``parameters`` (which may contain ``num_ctx``), and the raw
  ``modelfile`` text.

Every window number that leaves this module carries a ``source`` label
drawn from: ``"model ceiling"``, ``"configured num_ctx"``,
``"loaded (ps)"``, or ``"unknown"``. Downstream verdict logic treats
``"unknown"`` as fail-closed (verdict UNKNOWN, exit code 3) — it never
substitutes a guessed number.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Field names that have been observed (across Ollama versions) carrying a
# running model's loaded context length inside a /api/ps entry. We check
# them in order; the first one present wins, and its name is recorded as
# provenance. If none are present, the loaded window is UNKNOWN.
PS_CONTEXT_FIELDS = (
    "context_length",
    "context",
    "num_ctx",
    "ctx_len",
)

# Keys inside /api/show "model_info" that carry an architecture ceiling.
# The key is namespaced by architecture, e.g. "llama.context_length" or
# "qwen2.context_length". We accept any key ending in ".context_length"
# or exactly "context_length".
MODEL_INFO_CEILING_SUFFIX = ".context_length"


@dataclass
class WindowInfo:
    """One candidate context-window number, with its provenance."""

    value: int | None
    source: str  # "model ceiling" | "configured num_ctx" | "loaded (ps)" | "unknown"
    detail: str = ""  # e.g. which field the value came from


@dataclass
class ProbeResult:
    """Everything a probe run learned, with gaps left visible."""

    base_url: str
    reachable: bool
    installed_models: list[str] = field(default_factory=list)
    running_models: list[dict] = field(default_factory=list)
    show: dict | None = None
    error: str = ""
    # Window candidates, best-effort, each labelled by source.
    ceiling: WindowInfo = field(default_factory=lambda: WindowInfo(None, "unknown"))
    configured_num_ctx: WindowInfo = field(default_factory=lambda: WindowInfo(None, "unknown"))
    loaded: WindowInfo = field(default_factory=lambda: WindowInfo(None, "unknown"))

    def effective_window(self) -> WindowInfo:
        """The window the model REALLY loaded with, best evidence first.

        Priority: loaded (ps) > configured num_ctx > model ceiling.
        If none is known, returns UNKNOWN — callers must fail closed.
        """
        for candidate in (self.loaded, self.configured_num_ctx, self.ceiling):
            if candidate.value is not None:
                return candidate
        return WindowInfo(None, "unknown", detail="no window field returned by server")


def _to_int(value) -> int | None:
    """Coerce a JSON value to a positive int, or None if not usable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


class OllamaClient:
    """Minimal Ollama client over urllib with a configurable timeout."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(self.base_url + path, method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def tags(self) -> list[str]:
        data = self._get("/api/tags")
        models = data.get("models", []) if isinstance(data, dict) else []
        return [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]

    def ps(self) -> list[dict]:
        data = self._get("/api/ps")
        models = data.get("models", []) if isinstance(data, dict) else []
        return [m for m in models if isinstance(m, dict)]

    def show(self, model: str) -> dict:
        return self._post("/api/show", {"name": model})

    def probe(self, model: str) -> ProbeResult:
        """Probe tags + ps + show for *model*, recording field provenance."""
        result = ProbeResult(base_url=self.base_url, reachable=False)
        try:
            result.installed_models = self.tags()
            result.running_models = self.ps()
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            result.error = f"could not reach Ollama at {self.base_url}: {exc}"
            return result
        result.reachable = True

        # Loaded window: scan the running entry for this model.
        for entry in result.running_models:
            name = entry.get("name", "")
            if name == model or name.split(":")[0] == model.split(":")[0]:
                for field_name in PS_CONTEXT_FIELDS:
                    value = _to_int(entry.get(field_name))
                    if value is not None:
                        result.loaded = WindowInfo(
                            value, "loaded (ps)", detail=f"/api/ps field '{field_name}'"
                        )
                        break
                break

        # Model details: ceiling + configured num_ctx.
        try:
            show = self.show(model)
            result.show = show
            model_info = show.get("model_info", {}) if isinstance(show, dict) else {}
            if isinstance(model_info, dict):
                for key, value in model_info.items():
                    if key == "context_length" or key.endswith(MODEL_INFO_CEILING_SUFFIX):
                        parsed = _to_int(value)
                        if parsed is not None:
                            result.ceiling = WindowInfo(
                                parsed, "model ceiling", detail=f"model_info '{key}'"
                            )
                            break
            parameters = show.get("parameters", {}) if isinstance(show, dict) else {}
            if isinstance(parameters, dict):
                parsed = _to_int(parameters.get("num_ctx"))
                if parsed is not None:
                    result.configured_num_ctx = WindowInfo(
                        parsed, "configured num_ctx", detail="parameters.num_ctx"
                    )
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            # /api/show failing does not invalidate tags/ps evidence.
            if not result.error:
                result.error = f"/api/show failed for {model}: {exc}"
        return result
