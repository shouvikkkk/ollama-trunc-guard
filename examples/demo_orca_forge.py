"""Demo: the orca_forge scenario (dev.to, 2026-10-22).

An agent app expected a 16,384-token window, but the model was actually
loaded at 4,096 — and the OpenAI-compatible /v1/chat/completions path
did not reflect the expected window. The input was silently truncated
from the beginning while the endpoint still returned HTTP 200.

This script rebuilds that shape with 29 tool definitions, a long agent
system prompt, and a short conversation, then runs the budget simulation
and the verdict logic at the LOADED window (4,096). It asserts:

* verdict is FAIL,
* every tool definition is named as lost,
* the system prompt is named as lost,
* the OpenAI-endpoint verdict is FAIL.

Run:  python3 examples/demo_orca_forge.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_trunc_guard.budget import build_elements, simulate
from ollama_trunc_guard.endpoints import endpoint_verdict
from ollama_trunc_guard.estimator import estimate_tool_tokens
from ollama_trunc_guard.verdict import judge

EXPECTED_WINDOW = 16384
LOADED_WINDOW = 4096

TOOL_NAMES = [
    "read_file", "write_file", "edit_file", "list_dir", "search_code",
    "run_command", "run_tests", "get_git_status", "get_git_diff",
    "create_branch", "commit_changes", "fetch_url", "parse_json",
    "format_code", "lint_file", "typecheck", "build_project",
    "install_package", "read_logs", "query_database", "send_request",
    "create_issue", "comment_issue", "review_pr", "merge_pr",
    "deploy_preview", "check_coverage", "generate_docs", "summarize",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Use the {name} capability on the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
            },
        },
    }
    for name in TOOL_NAMES
]

SYSTEM_PARAGRAPH = (
    "You are an expert coding agent working inside a software repository. "
    "Follow the user's instructions precisely, explain your plan before "
    "editing files, verify every change by running the relevant tests, "
    "and never invent file contents you have not read. "
)
SYSTEM_PROMPT = SYSTEM_PARAGRAPH * 60  # ~12,000 chars -> ~3,000 tokens

MESSAGES = [
    {"role": "user", "content": ("Please fix the failing tests. " * 40)},
    {"role": "assistant", "content": ("I will investigate the failures. " * 40)},
    {"role": "user", "content": ("Also update the documentation. " * 40)},
    {"role": "assistant", "content": ("Documentation updated. " * 40)},
    {"role": "user", "content": ("Now run the full suite again. " * 40)},
]


def main() -> int:
    elements = build_elements(SYSTEM_PROMPT, TOOLS, MESSAGES)
    tool_tokens, _ = estimate_tool_tokens(TOOLS)
    result = simulate(elements, window=LOADED_WINDOW, reserve=0)
    endpoint_v, endpoint_reason = endpoint_verdict(
        "openai", needed_window=result.total_tokens, loaded_window=LOADED_WINDOW
    )
    verdict = judge(
        effective_window=LOADED_WINDOW,
        total_prompt_tokens=result.total_tokens,
        tool_tokens=tool_tokens,
        endpoint_verdict=endpoint_v,
    )

    print("orca_forge scenario reproduction")
    print(f"  expected window: {EXPECTED_WINDOW} tokens")
    print(f"  loaded window:   {LOADED_WINDOW} tokens (model actually loaded)")
    print(f"  endpoint:        /v1/chat/completions (OpenAI-compatible)")
    print(f"  tool definitions: {len(TOOLS)} ({tool_tokens} tokens)")
    print(f"  total prompt:    {result.total_tokens} tokens")
    print(f"  overflow:        {result.overflow_tokens} tokens truncated from the front")
    print("  LOST (front-truncated head):")
    for loss in result.lost:
        flag = "FULLY LOST" if loss.fully_lost else "partially lost"
        print(f"    - {loss.name}: {loss.tokens_lost}/{loss.tokens} tokens ({flag})")
    print(f"  endpoint verdict: {endpoint_v} — {endpoint_reason}")
    print(f"  OVERALL VERDICT: {verdict.verdict.value} (exit {verdict.exit_code})")
    for reason in verdict.reasons:
        print(f"    - {reason}")

    # The demo's claims, checked rather than asserted on faith.
    assert verdict.verdict.value == "FAIL", "expected FAIL verdict"
    assert endpoint_v == "FAIL", "expected FAIL endpoint verdict"
    tool_losses = [loss for loss in result.lost if loss.kind == "tool"]
    assert len(tool_losses) == len(TOOLS), "expected every tool definition lost"
    assert all(loss.fully_lost for loss in tool_losses), "expected tools fully lost"
    assert "system prompt" in result.lost_names(), "expected system prompt lost"
    print("\nDemo checks passed: FAIL verdict, all 29 tools + system prompt named as lost.")
    return verdict.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
