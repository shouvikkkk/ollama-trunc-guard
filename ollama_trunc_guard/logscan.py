"""Server-log scanning for truncation signals.

Ollama's server log is one of the few places where a silent truncation
leaves a trace. This module scans log text (from a file path or from
stdin text) for truncation signals and returns timestamped hits, with
per-model counts when a model name is parseable from the hit line.

Patterns (case-insensitive substring search, in priority order):

* ``truncating input`` — the most direct signal.
* ``prompt too long`` — an over-long prompt warning.
* ``exceeds context`` — prompt/context overflow wording.
* ``truncat`` — the catch-all; it also matches the two above, so hits
  are de-duplicated per line and labelled with the most specific pattern
  that matched.

A hit is a dict with ``line_number``, ``timestamp`` (ISO-ish prefix if
the line starts with one, else ""), ``pattern``, ``model`` (parsed from
``model=<name>`` / ``model: <name>`` / quoted model names when present,
else ""), and ``line`` (the raw line, stripped). Counts are returned
both overall and per model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Most specific first: a line is labelled with the first pattern that
# matches, so the catch-all "truncat" never shadows a specific signal.
PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("truncating input", re.compile(r"truncating input", re.IGNORECASE)),
    ("prompt too long", re.compile(r"prompt too long", re.IGNORECASE)),
    ("exceeds context", re.compile(r"exceeds context", re.IGNORECASE)),
    ("truncat", re.compile(r"truncat", re.IGNORECASE)),
)

# A leading timestamp: ISO dates, "2026/10/27 08:44:02", or bare times.
TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)

# Model-name hints inside a log line, tried in order.
MODEL_RES = (
    re.compile(r"""model\s*=\s*["']?([\w.:/-]+)""", re.IGNORECASE),
    re.compile(r"""model\s*:\s*["']?([\w.:/-]+)""", re.IGNORECASE),
    re.compile(r"""["'](model|name)["']\s*:\s*["']([\w.:/-]+)["']""", re.IGNORECASE),
)


def _parse_model(line: str) -> str:
    for pattern in MODEL_RES:
        match = pattern.search(line)
        if match:
            # The third pattern captures the name in group 2.
            return match.group(match.lastindex or 1).strip("\"'")  # type: ignore[arg-type]
    return ""


@dataclass
class LogScanResult:
    hits: list[dict] = field(default_factory=list)
    total_lines: int = 0

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def counts_by_pattern(self) -> dict[str, int]:
        return dict(Counter(hit["pattern"] for hit in self.hits))

    @property
    def counts_by_model(self) -> dict[str, int]:
        return dict(
            Counter(hit["model"] for hit in self.hits if hit["model"])
        )

    def to_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "hit_count": self.hit_count,
            "counts_by_pattern": self.counts_by_pattern,
            "counts_by_model": self.counts_by_model,
            "hits": self.hits,
        }


def scan_log_text(text: str) -> LogScanResult:
    """Scan log *text* line by line for truncation signals."""
    result = LogScanResult()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        result.total_lines += 1
        line = raw_line.strip()
        if not line:
            continue
        matched = None
        for name, pattern in PATTERNS:
            if pattern.search(line):
                matched = name
                break
        if matched is None:
            continue
        timestamp_match = TIMESTAMP_RE.match(line)
        result.hits.append(
            {
                "line_number": line_number,
                "timestamp": timestamp_match.group("ts") if timestamp_match else "",
                "pattern": matched,
                "model": _parse_model(line),
                "line": line,
            }
        )
    return result


def scan_log_file(path: str) -> LogScanResult:
    """Scan the log file at *path* (read as UTF-8, errors replaced)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return scan_log_text(handle.read())
