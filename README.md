# ollama-trunc-guard

**Fail-closed pre-flight gate for Ollama agent runs: proves your prompt + tool definitions actually fit the window the model REALLY loaded with, before a silent front-truncation eats your system prompt.**

No third-party dependencies. Python 3.10+ standard library only — there is no `requirements.txt` because there is nothing to install.

## The problem

Ollama silently truncates over-long prompts **from the front** and still returns HTTP 200. A local coding agent can lose its system prompt and its tool definitions with no error at all — and then fail in confusing ways downstream, because the model is answering without the instructions and tools it was supposed to have.

Real evidence (October 2026):

- **dev.to/orca_forge, "Why Ollama Tool Calls Fail at a 4,096 Context Length" (2026-10-22):** the app expected a 16,384 window, the model was actually loaded at 4,096, and *"it silently truncated the input from the beginning"* — while the OpenAI-compatible `/v1/chat/completions` path did not reflect the expected window.
- **dev.to/ji_ai:** 287 of 400 prompts cut, with `num_ctx` 2048 against a 128K model card.
- **Decoding AI (Oct 10, 2026):** tool definitions alone take **10–36% of Ollama's 4,096 default** window; of the four runtimes they read, only Ollama silently cuts history by default.

The failure has three compounding causes, and this tool checks all three: the model may be loaded with a smaller window than you think, your prompt may exceed it, and the endpoint you are calling may not let you fix the window per-request.

## Quickstart

```bash
# From the repo root — no install step, stdlib only.
python3 -m ollama_trunc_guard check \
  --model llama3.1:8b \
  --prompt-file system-prompt.txt \
  --tools-file tools.json \
  --messages-file history.json \
  --reserve 512 \
  --endpoint openai

# Just the turn projection: when does the system prompt fall out?
python3 -m ollama_trunc_guard simulate \
  --messages-file history.json --prompt-file system-prompt.txt \
  --window 4096 --growth 400

# Scan a server log for truncation evidence
python3 -m ollama_trunc_guard scan-logs --log-file /path/to/ollama.log
# ... or pipe it in:
ollama serve 2>&1 | python3 -m ollama_trunc_guard scan-logs

# Audit a Modelfile / env declaration against the loaded window
python3 -m ollama_trunc_guard audit-config \
  --modelfile ./Modelfile --loaded-window 4096
```

`check` exits with a severity-gated code your CI or agent launcher can gate on:

| Exit | Verdict | Meaning |
|------|---------|---------|
| 0 | OK | Prompt fits with comfortable headroom. |
| 1 | WARN | Headroom < 20% of the window, or tools > 30% of it. |
| 2 | FAIL | Budget exceeds the window, log evidence found, or the endpoint can't honor the needed window. |
| 3 | UNKNOWN | Effective window could not be determined — **fail closed, never OK on unknown data.** |

Add `--json` to any subcommand for machine-readable output.

## Example output

```
$ python3 -m ollama_trunc_guard check --model llama3.1:8b \
    --prompt-file system-prompt.txt --tools-file tools.json --endpoint openai
ollama-trunc-guard check — model llama3.1:8b via http://localhost:11434
  probe reachable: True
  effective window: 4096 (source: loaded (ps), /api/ps field 'context_length')
  ceiling: 131072, configured num_ctx: 16384, loaded (ps): 4096
  budget: prompt 6720 tokens, available 4096 (window 4096 - reserve 0), overflow 2624
  tool tokens: 1458
  LOST (front-truncated head):
    - tool:read_file (tool): 49/49 tokens lost — FULLY LOST
    ... (all 29 tool definitions)
    - system prompt (system): 1166/3780 tokens lost — partially lost
  endpoint /v1/chat/completions: FAIL — does not apply per-request num_ctx
    and the loaded window (4096) is smaller than the needed window (6720)
  VERDICT: FAIL (exit 2)
```

The demo in `examples/demo_orca_forge.py` reproduces the orca_forge scenario end to end (29 tools, 16,384 expected vs 4,096 loaded) and asserts the FAIL verdict with the tools and system prompt named as lost.

## How the budget model works

Ollama truncates from the **beginning** of the fully constructed prompt and keeps the tail. This tool models the constructed prompt as, in order:

1. tool definitions (in definition order),
2. system prompt,
3. conversation history, oldest message first.

Tool definitions sit at the front, so under front-truncation they are lost **first** — the exact failure in the orca_forge case. The reserved output budget is never truncated; it shrinks the available input window instead.

Caveat: the exact placement of tool definitions relative to the system prompt depends on the client and chat template. If your client places the system prompt first, pass `order=("system", "tools", "messages")` to `build_elements()` — the report always names the order it assumed, so a mismatch is visible rather than silent.

## Per-endpoint behaviour (dated, version-dependent)

| Endpoint | Per-request `num_ctx` honored? | What happens when the loaded window is too small |
|----------|-------------------------------|--------------------------------------------------|
| `/api/chat` (native) | Yes | Caller can request a larger window (model ceiling / VRAM still apply). |
| `/api/generate` (native) | Yes | Same as `/api/chat`. |
| `/v1/chat/completions` (OpenAI-compatible) | **No (historically)** | **FAIL** — no request parameter rescues the run; reload the model with a larger window (Modelfile `PARAMETER num_ctx` or `OLLAMA_CONTEXT_LENGTH`) before running. |

This table is dated **2026-10-27** and version-dependent: Ollama's endpoint behaviour has changed across releases. Re-verify against your installed version. The native-path entries rest on Ollama's documented `options.num_ctx` behaviour; the OpenAI-path entry rests on the orca_forge failure (2026-10-22).

## How window numbers are sourced (never guessed)

Every window number in a report carries a source label:

- **"model ceiling"** — architecture limit from `/api/show` `model_info` (e.g. `llama.context_length`).
- **"configured num_ctx"** — `parameters.num_ctx` from `/api/show`.
- **"loaded (ps)"** — a context field actually returned by `/api/ps` for the running model.
- **"unknown"** — no source returned a window. The verdict is UNKNOWN (exit 3), never a substituted guess.

The probe reads whatever fields the server returns and records their provenance. It never fabricates a context field that is absent — a missing `/api/ps` context field yields UNKNOWN, not a made-up number.

## Differentiation

- **vs llm-doctor (dev.to/arthur031221):** llm-doctor is a broad local-LLM health scanner. This tool is a narrow, deep pre-flight gate for one specific failure: silent front-truncation on agent tool-call runs, with budget simulation, endpoint verdicts, log evidence, and config auditing for that failure alone.
- **vs manual Modelfile / env fixes:** editing `PARAMETER num_ctx` or `OLLAMA_CONTEXT_LENGTH` is the *fix*; this tool is the *check that the fix actually took* — declared-vs-loaded-vs-ceiling comparison catches the case where you edited the config but the running model still loaded at 4,096.

## Honest limitations

- **The token estimator is a heuristic, not a tokenizer.** Counts use `max(chars/4, words×1.3)` with per-family chars-per-token overrides. On ordinary English prose it lands within roughly **±25%** of a real tokenizer; it is worse on code with long identifiers, non-English text, and dense JSON (tool definitions). That is why headroom < 20% is WARN, not OK.
- **Loaded-window discovery depends on what your local Ollama version exposes** via `/api/ps` and logs. If your version returns no context field, the verdict is UNKNOWN — correctly, but less useful.
- **Live-Ollama run NOT verified.** The test suite uses an in-process fake Ollama server (real HTTP, fake data). The tool has not yet been run against a real `ollama serve` instance.
- **OpenAI-endpoint behaviour is version-dependent** (see the table above, dated 2026-10-27).
- **The truncation-order model is an assumption**, documented above: tools, then system prompt, then oldest messages. Real clients vary in template layout; the order is configurable for that reason.

## Development

```bash
python3 -m unittest discover -s tests -v   # 78 tests, stdlib unittest
python3 examples/demo_orca_forge.py        # orca_forge scenario reproduction
```

## License

MIT — see [LICENSE](LICENSE).
