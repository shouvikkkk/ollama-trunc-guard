"""Command-line interface for ollama-trunc-guard.

Subcommands:

* ``check`` — probe the server, budget the prompt, judge the endpoint,
  and print a fail-closed verdict with a severity-gated exit code.
* ``simulate`` — turn-by-turn front-truncation projection: the turn on
  which a watched element (default: the system prompt) falls out.
* ``scan-logs`` — scan an Ollama server log (file or stdin) for
  truncation signals.
* ``audit-config`` — audit a Modelfile and OLLAMA_CONTEXT_LENGTH
  against the loaded window and the model ceiling.

Every subcommand supports ``--json`` for machine-readable output.
Exit codes: 0 OK, 1 WARN, 2 FAIL, 3 UNKNOWN (fail-closed).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .budget import build_elements, project_turns, simulate
from .config import audit_config
from .endpoints import ENDPOINT_CHOICES, ENDPOINTS, endpoint_verdict
from .estimator import estimate_tool_tokens
from .logscan import scan_log_file, scan_log_text
from .probe import OllamaClient
from .verdict import Verdict, judge


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _read_json_file(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_report(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(report["_text"])


def _build_check_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("check", help="probe + budget + endpoint verdict")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", required=True, help="model name, e.g. llama3.1:8b")
    parser.add_argument("--prompt-file", help="file containing the system prompt")
    parser.add_argument("--prompt", default="", help="system prompt text")
    parser.add_argument("--tools-file", help="JSON file: array of tool definitions")
    parser.add_argument("--messages-file", help="JSON file: list of message dicts")
    parser.add_argument("--reserve", type=int, default=0, help="output tokens to reserve")
    parser.add_argument("--endpoint", choices=ENDPOINT_CHOICES, default="api-chat")
    parser.add_argument("--log-file", help="Ollama server log to scan")
    parser.add_argument("--modelfile", help="Modelfile to audit")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--timeout", type=float, default=10.0)


def _build_simulate_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("simulate", help="turn-by-turn truncation projection")
    parser.add_argument("--messages-file", help="JSON file: list of message dicts")
    parser.add_argument("--prompt-file", help="file containing the system prompt")
    parser.add_argument("--prompt", default="", help="system prompt text")
    parser.add_argument("--tools-file", help="JSON file: array of tool definitions")
    parser.add_argument("--window", type=int, help="effective window in tokens")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", help="model to probe for the window if --window omitted")
    parser.add_argument("--reserve", type=int, default=0)
    parser.add_argument("--growth", type=int, default=0, help="tokens added per turn")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--watch", default="system prompt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)


def _build_scan_logs_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("scan-logs", help="scan a server log for truncation signals")
    parser.add_argument("--log-file", help="log file path (default: read stdin)")
    parser.add_argument("--json", action="store_true")


def _build_audit_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("audit-config", help="audit Modelfile / env window declarations")
    parser.add_argument("--modelfile", help="Modelfile path")
    parser.add_argument("--loaded-window", type=int, help="window the model actually loaded with")
    parser.add_argument("--ceiling", type=int, help="model architecture ceiling")
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-trunc-guard",
        description="Fail-closed pre-flight gate for Ollama agent runs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_check_parser(sub)
    _build_simulate_parser(sub)
    _build_scan_logs_parser(sub)
    _build_audit_parser(sub)
    return parser


def _load_prompt_inputs(args) -> tuple[str, list[dict], list[dict]]:
    system_prompt = _read_text_file(args.prompt_file) if args.prompt_file else args.prompt
    tools = _read_json_file(args.tools_file) if args.tools_file else []
    messages = _read_json_file(args.messages_file) if args.messages_file else []
    if not isinstance(tools, list):
        raise ValueError("--tools-file must contain a JSON array")
    if not isinstance(messages, list):
        raise ValueError("--messages-file must contain a JSON list")
    return system_prompt, tools, messages


def cmd_check(args) -> int:
    system_prompt, tools, messages = _load_prompt_inputs(args)
    client = OllamaClient(base_url=args.base_url, timeout=args.timeout)
    probe = client.probe(args.model)
    effective = probe.effective_window()

    elements = build_elements(system_prompt, tools, messages, model=args.model)
    tool_tokens, _ = estimate_tool_tokens(tools)

    log_result = None
    if args.log_file:
        log_result = scan_log_file(args.log_file)

    audit = None
    if args.modelfile:
        audit = audit_config(
            modelfile_path=args.modelfile,
            loaded_window=effective.value,
            ceiling=probe.ceiling.value,
        )

    if effective.value is None:
        budget = None
        endpoint_v, endpoint_reason = endpoint_verdict(
            args.endpoint, needed_window=0, loaded_window=None
        )
        total_tokens = sum(el.tokens for el in elements)
        verdict = judge(
            effective_window=None,
            total_prompt_tokens=total_tokens,
            tool_tokens=tool_tokens,
            reserve=args.reserve,
            log_hits=log_result.hit_count if log_result else 0,
            endpoint_verdict=endpoint_v,
            server_reachable=probe.reachable,
        )
    else:
        budget = simulate(elements, effective.value, reserve=args.reserve)
        needed = budget.total_tokens + args.reserve
        endpoint_v, endpoint_reason = endpoint_verdict(
            args.endpoint, needed_window=needed, loaded_window=effective.value
        )
        verdict = judge(
            effective_window=effective.value,
            total_prompt_tokens=budget.total_tokens,
            tool_tokens=tool_tokens,
            reserve=args.reserve,
            log_hits=log_result.hit_count if log_result else 0,
            endpoint_verdict=endpoint_v,
            server_reachable=probe.reachable,
        )

    report: dict = {
        "command": "check",
        "model": args.model,
        "base_url": args.base_url,
        "endpoint": {"key": args.endpoint, "path": ENDPOINTS[args.endpoint].path},
        "probe": {
            "reachable": probe.reachable,
            "error": probe.error,
            "installed_models": probe.installed_models,
            "effective_window": effective.value,
            "effective_window_source": effective.source,
            "effective_window_detail": effective.detail,
            "ceiling": probe.ceiling.value,
            "configured_num_ctx": probe.configured_num_ctx.value,
            "loaded": probe.loaded.value,
        },
        "budget": budget.to_dict() if budget else None,
        "tool_tokens": tool_tokens,
        "endpoint_verdict": {"verdict": endpoint_v, "reason": endpoint_reason},
        "log_scan": log_result.to_dict() if log_result else None,
        "config_audit": audit.to_dict() if audit else None,
        "verdict": verdict.to_dict(),
    }

    lines = [
        f"ollama-trunc-guard check — model {args.model} via {args.base_url}",
        f"  probe reachable: {probe.reachable}"
        + (f" (error: {probe.error})" if probe.error else ""),
        f"  effective window: {effective.value if effective.value is not None else 'UNKNOWN'} "
        f"(source: {effective.source}"
        + (f", {effective.detail}" if effective.detail else "") + ")",
        f"  ceiling: {probe.ceiling.value}, configured num_ctx: "
        f"{probe.configured_num_ctx.value}, loaded (ps): {probe.loaded.value}",
    ]
    if budget is not None:
        lines.append(
            f"  budget: prompt {budget.total_tokens} tokens, available "
            f"{budget.available_tokens} (window {budget.window} - reserve "
            f"{budget.reserve}), overflow {budget.overflow_tokens}"
        )
        lines.append(f"  tool tokens: {tool_tokens}")
        if budget.lost:
            lines.append("  LOST (front-truncated head):")
            for loss in budget.lost:
                lines.append(
                    f"    - {loss.name} ({loss.kind}): {loss.tokens_lost}/"
                    f"{loss.tokens} tokens lost"
                    + (" — FULLY LOST" if loss.fully_lost else " — partially lost")
                )
        else:
            lines.append("  lost: none — prompt fits the window")
        lines.append("  surviving tail: " + ", ".join(
            el.name for el in budget.surviving) if budget.surviving else "  surviving tail: (empty)")
    else:
        lines.append(
            "  budget: cannot simulate — effective window unknown (fail-closed)"
        )
        lines.append(f"  tool tokens: {tool_tokens}")
    lines.append(f"  endpoint {ENDPOINTS[args.endpoint].path}: {endpoint_v} — {endpoint_reason}")
    if log_result is not None:
        lines.append(
            f"  log scan: {log_result.hit_count} hit(s) in "
            f"{log_result.total_lines} lines; by pattern "
            f"{log_result.counts_by_pattern}; by model {log_result.counts_by_model}"
        )
    if audit is not None:
        for mismatch in audit.mismatches:
            lines.append(f"  config mismatch: {mismatch}")
        for note in audit.notes:
            lines.append(f"  config note: {note}")
    lines.append(f"  VERDICT: {verdict.verdict.value} (exit {verdict.exit_code})")
    for reason in verdict.reasons:
        lines.append(f"    - {reason}")
    report["_text"] = "\n".join(lines)
    _print_report(report, args.json)
    return verdict.exit_code


def cmd_simulate(args) -> int:
    system_prompt, tools, messages = _load_prompt_inputs(args)
    window = args.window
    if window is None and args.model:
        probe = OllamaClient(base_url=args.base_url, timeout=args.timeout).probe(args.model)
        window = probe.effective_window().value
    if window is None:
        print(
            "error: no window available — pass --window or --model with a "
            "reachable Ollama server",
            file=sys.stderr,
        )
        return Verdict.UNKNOWN.exit_code
    elements = build_elements(system_prompt, tools, messages, model=args.model)
    projection = project_turns(
        elements,
        window,
        reserve=args.reserve,
        growth_per_turn=args.growth,
        max_turns=args.max_turns,
        watch_name=args.watch,
    )
    report: dict = {"command": "simulate", "window": window, "projection": projection}
    lines = [
        f"ollama-trunc-guard simulate — window {window}, reserve {args.reserve}, "
        f"growth {args.growth} tokens/turn, watching '{args.watch}'",
    ]
    for turn in projection["turns"]:
        lines.append(
            f"  turn {turn['turn']}: total {turn['total_prompt_tokens']} tokens, "
            f"overflow {turn['overflow_tokens']}"
            + (f", lost: {', '.join(turn['lost_names'])}" if turn["lost_names"] else ", fits")
        )
    falls = projection["watch_falls_out_on_turn"]
    if falls is None:
        lines.append(
            f"  '{args.watch}' survives all {args.max_turns} projected turns."
        )
    else:
        lines.append(f"  '{args.watch}' FALLS OUT on turn {falls}.")
    report["_text"] = "\n".join(lines)
    _print_report(report, args.json)
    return 0


def cmd_scan_logs(args) -> int:
    if args.log_file:
        result = scan_log_file(args.log_file)
        source = args.log_file
    else:
        result = scan_log_text(sys.stdin.read())
        source = "<stdin>"
    report: dict = {"command": "scan-logs", "source": source, "scan": result.to_dict()}
    lines = [
        f"ollama-trunc-guard scan-logs — {source}: {result.hit_count} hit(s) "
        f"in {result.total_lines} lines",
        f"  by pattern: {result.counts_by_pattern}",
        f"  by model: {result.counts_by_model}",
    ]
    for hit in result.hits:
        lines.append(
            f"  line {hit['line_number']}"
            + (f" [{hit['timestamp']}]" if hit["timestamp"] else "")
            + (f" model={hit['model']}" if hit["model"] else "")
            + f" ({hit['pattern']}): {hit['line']}"
        )
    report["_text"] = "\n".join(lines)
    _print_report(report, args.json)
    return 2 if result.hit_count > 0 else 0


def cmd_audit_config(args) -> int:
    audit = audit_config(
        modelfile_path=args.modelfile,
        loaded_window=args.loaded_window,
        ceiling=args.ceiling,
    )
    report: dict = {"command": "audit-config", "audit": audit.to_dict()}
    lines = ["ollama-trunc-guard audit-config"]
    lines.append(f"  modelfile num_ctx: {audit.modelfile_num_ctx}")
    lines.append(f"  env OLLAMA_CONTEXT_LENGTH: {audit.env_num_ctx}")
    for mismatch in audit.mismatches:
        lines.append(f"  MISMATCH: {mismatch}")
    for note in audit.notes:
        lines.append(f"  note: {note}")
    if not audit.mismatches:
        lines.append("  no mismatches found")
    report["_text"] = "\n".join(lines)
    _print_report(report, args.json)
    return 2 if audit.mismatches else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return cmd_check(args)
        if args.command == "simulate":
            return cmd_simulate(args)
        if args.command == "scan-logs":
            return cmd_scan_logs(args)
        if args.command == "audit-config":
            return cmd_audit_config(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    parser.error(f"unknown command {args.command}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
