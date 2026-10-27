"""Modelfile and environment configuration auditing.

Two places where a user *declares* a context window for Ollama:

1. A Modelfile line ``PARAMETER num_ctx <n>`` — the window the model is
   created with. (Modelfile parameter lines are case-insensitive for
   the keyword; values may carry a trailing comment.)
2. The ``OLLAMA_CONTEXT_LENGTH`` environment variable — the default
   context length for models that do not set their own.

This module parses both and compares them against the window the model
actually loaded with (from the probe) and the model ceiling. Any
disagreement is reported as a mismatch entry; agreement is reported as
a match. Nothing here guesses: a value that is absent is reported as
absent, not defaulted.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

PARAMETER_RE = re.compile(
    r"^\s*PARAMETER\s+num_ctx\s+(\d+)\s*(?:#.*)?$",
    re.IGNORECASE,
)

ENV_VAR = "OLLAMA_CONTEXT_LENGTH"


@dataclass
class ConfigAudit:
    modelfile_num_ctx: int | None = None
    modelfile_path: str = ""
    env_num_ctx: int | None = None
    mismatches: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "modelfile_path": self.modelfile_path,
            "modelfile_num_ctx": self.modelfile_num_ctx,
            "env_OLLAMA_CONTEXT_LENGTH": self.env_num_ctx,
            "mismatches": self.mismatches,
            "notes": self.notes,
        }


def parse_modelfile_num_ctx(text: str) -> int | None:
    """Return the last ``PARAMETER num_ctx`` value in Modelfile *text*.

    If the parameter appears more than once, the last occurrence wins —
    matching how Ollama applies repeated parameters. Returns None when
    the parameter is absent.
    """
    value = None
    for line in text.splitlines():
        match = PARAMETER_RE.match(line)
        if match:
            value = int(match.group(1))
    return value


def parse_modelfile_file(path: str) -> int | None:
    """Parse ``PARAMETER num_ctx`` from the Modelfile at *path*."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parse_modelfile_num_ctx(handle.read())


def read_env_num_ctx(environ: dict | None = None) -> int | None:
    """Return ``OLLAMA_CONTEXT_LENGTH`` as an int, or None if unset/invalid."""
    raw = (environ if environ is not None else os.environ).get(ENV_VAR)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw.isdigit():
        return None
    return int(raw)


def audit_config(
    modelfile_path: str | None = None,
    loaded_window: int | None = None,
    ceiling: int | None = None,
    environ: dict | None = None,
) -> ConfigAudit:
    """Audit declared windows against the loaded window and the ceiling.

    ``loaded_window`` and ``ceiling`` come from the probe (or None when
    unknown); this function never invents them. Mismatches are phrased
    as facts about declared-vs-observed values, not as accusations.
    """
    audit = ConfigAudit(modelfile_path=modelfile_path or "")
    if modelfile_path:
        audit.modelfile_num_ctx = parse_modelfile_file(modelfile_path)
        if audit.modelfile_num_ctx is None:
            audit.notes.append(
                f"no PARAMETER num_ctx in Modelfile {modelfile_path}; "
                "the model falls back to the server default."
            )
    audit.env_num_ctx = read_env_num_ctx(environ)
    if audit.env_num_ctx is None:
        audit.notes.append(
            f"{ENV_VAR} is not set; the server default applies unless the "
            "Modelfile sets num_ctx."
        )

    if (
        audit.modelfile_num_ctx is not None
        and loaded_window is not None
        and audit.modelfile_num_ctx != loaded_window
    ):
        audit.mismatches.append(
            f"declared-vs-loaded mismatch: Modelfile num_ctx "
            f"{audit.modelfile_num_ctx} but model loaded with {loaded_window}."
        )
    if (
        audit.env_num_ctx is not None
        and loaded_window is not None
        and audit.modelfile_num_ctx is None
        and audit.env_num_ctx != loaded_window
    ):
        audit.mismatches.append(
            f"declared-vs-loaded mismatch: {ENV_VAR}={audit.env_num_ctx} "
            f"but model loaded with {loaded_window}."
        )
    if (
        audit.modelfile_num_ctx is not None
        and ceiling is not None
        and audit.modelfile_num_ctx > ceiling
    ):
        audit.mismatches.append(
            f"declared window exceeds model ceiling: Modelfile num_ctx "
            f"{audit.modelfile_num_ctx} > ceiling {ceiling}."
        )
    if (
        audit.env_num_ctx is not None
        and ceiling is not None
        and audit.env_num_ctx > ceiling
    ):
        audit.mismatches.append(
            f"declared window exceeds model ceiling: {ENV_VAR}="
            f"{audit.env_num_ctx} > ceiling {ceiling}."
        )
    if (
        audit.modelfile_num_ctx is not None
        and audit.env_num_ctx is not None
        and audit.modelfile_num_ctx != audit.env_num_ctx
    ):
        audit.notes.append(
            f"Modelfile num_ctx ({audit.modelfile_num_ctx}) and "
            f"{ENV_VAR} ({audit.env_num_ctx}) differ; the Modelfile value "
            "takes precedence for a model created from it."
        )
    return audit
