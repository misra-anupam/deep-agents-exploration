# Equity Research Desk

A LangChain **Deep Agents** harness: a lead analyst delegates to scoped subagents, writes a cited
earnings note, and proposes **paper** trades behind human approval gates. Research aid only; not
investment advice. See `PLAN.md` for the phased build.

```
make test        # 109 offline tests, no API key needed
make eval        # offline: checks the scorer separates good from bad notes
make eval-live   # real model on the golden event   (needs OPENROUTER_API_KEY)
make demo        # interactive run on synthetic ACME (needs OPENROUTER_API_KEY)
```

## What is verified, and what is not
- **Verified offline (scripted models):** wiring, tool scoping, permissions, interrupts/resume,
  memory persistence, guardrails, sandbox, audit chain, scorers.
- **Not verified:** how a *real* model behaves (planning quality, whether it follows the
  prompts, cost). No API key was available while building, and OpenRouter tool-calling has not been exercised live. `make eval-live` is the first thing to run.
- Live SEC access (`EQUITY_DESK_DATA=edgar`) is tested only against a mock transport.

## Models (OpenRouter)
Set `OPENROUTER_API_KEY` and, optionally, `EQUITY_DESK_MODEL` (default `openrouter:anthropic/claude-sonnet-5.5`).
Use a different family for the critic and a cheaper model for the readers via
`EQUITY_DESK_REDTEAM_MODEL` / `EQUITY_DESK_SUBAGENT_MODEL`. See `.env.example` and https://openrouter.ai/models.

## Capability map

| Capability | Where |
|---|---|
| Planning | `TodoListMiddleware` in `agent.py` (opt-in in deepagents 0.7+) |
| Filesystem, skills | `backends.py`, `skills/lead/`, `skills/quant/` (per-role) |
| Backends | Composite: `/filings/` disk, `/skills/` disk, `/memories/` + `/reports/` Store, rest in state |
| Permissions | `/filings`, `/skills`, thesis paths read-only; red-team and news agents write-denied everywhere |
| Subagents | `subagents.py`: filings-reader, quant-analyst, news-sentiment, red-team |
| Parallelism | Independent `task` calls in one turn (proved concurrent with a barrier test) |
| Sandbox | `sandbox.py`: `run_python` tool for quant-analyst only (macOS `sandbox-exec`) |
| Human-in-the-loop | `approvals.py`: trade, thesis, publish gated; subagents inherit the gates |
| Memory | `AGENTS.md` house memory + per-ticker thesis/history in the Store (`memory.py`) |
| Guardrails | `guardrails.py`, `middleware.py`: compliance language, citations, red-team-before-publish |
| Audit | `audit.py`: hash-chained JSONL of tool calls and human decisions |
| Resilience | Built-in `ToolRetryMiddleware` (transient HTTP only), `ModelCallLimitMiddleware` |
| Evals / tracing | `evals/`, `tracing.py`: grounding, citations, flag recall, plan, usage; LangSmith via env |

## Design decisions worth knowing
- **Deterministic work is not agentic.** Ratios, order validation, and compliance checks are plain
  code; the agent decides *what to investigate*.
- **The lead has no data or code tools.** It delegates; only reports reach its context.
- **Defence in depth, not prompts.** A trade needs (1) the agent to propose it, (2) a human to approve,
  (3) the broker's limits to pass. A note needs red-team review, a compliance pass, and approval.
- **News is untrusted.** Only the news subagent sees raw headlines; the fixture includes a poisoned one.

## Known limits
- The sandbox blocks network and confines writes, but still allows most file reads and depends on
  Apple's deprecated `sandbox-exec`. Use a container/VM backend before running untrusted workloads.
- The audit chain is tamper-*evident*, not tamper-proof.
- `execute` is still listed in the model's tool menu (it returns an error). Hiding it needs a harness profile.
- Fixtures are one synthetic company; the golden set has one event. Add more before trusting scores.
- Prices are fixtures; there is no live quote source.
