# Equity Research Desk — Build Plan

An agent harness on LangChain Deep Agents that covers a watchlist: reads filings,
runs quant checks in a sandbox, writes a cited research note, and proposes paper
trades behind human approval gates. Paper trading only. Not investment advice.

## Ground rules
- Every phase ends with passing tests and a runnable demo (`make phaseN` or a script).
- Tests run **offline** with a fake chat model and fixture filings. Live runs need
  `ANTHROPIC_API_KEY` (and optionally `LANGSMITH_API_KEY`); none are set yet.
- Deterministic work (XBRL fetch, ratio math, order validation) lives in plain tools.
  The agent decides *what to investigate*, not how to compute a ratio.
- Verify the installed `deepagents` API before relying on docs; versions drift.

## Capability coverage map
| Capability | Phase |
|---|---|
| Filesystem + skills | 1 |
| Planning (`TodoListMiddleware`) | 1 |
| Context offloading (filings to files, summaries back) | 1 |
| Sandboxed execution | 2 |
| Subagent registry (scoped tools) | 2 |
| Composite backend + Store memory (`AGENTS.md`, thesis files) | 3 |
| Human-in-the-loop (`interrupt_on`) + checkpointer | 4 |
| Red-team subagent, async/parallel subagents | 5 |
| Custom middleware (compliance, citations, audit) | 6 |
| Evals + LangSmith tracing + cost tracking | 6 |

## Phases

### Phase 0 — Scaffold
uv project (Python 3.12), deps, package layout, Makefile, `.env.example`, fake-model
test harness, fixture filing (small synthetic 10-K excerpt + XBRL facts JSON).
**Done when:** `uv run pytest` passes a smoke test.

### Phase 1 — Single analyst agent
Tools: `fetch_filing_index`, `download_filing` (writes to `/filings/`),
`get_xbrl_facts`. Skill: `earnings-note-format`. `TodoListMiddleware` enabled.
Agent reads a filing via grep/read_file and writes `/reports/<ticker>-<period>.md`.
**Done when:** with the fake model, a scripted run produces a note file; tool unit
tests pass against fixtures; live path is wired behind an env check.

### Phase 2 — Quant subagent + sandbox
Sandbox backend for `execute` (local subprocess sandbox with no network, scrubbed
env, timeout, temp cwd; clearly flagged as dev-only). Deterministic `ratios` module
(margins, DSO, accruals, FCF conversion, growth). `quant-analyst` and
`filings-reader` subagents with scoped tools. Skill: `quality-of-earnings`.
**Done when:** ratio tests pass; sandbox blocks network and enforces timeout.

### Phase 3 — Memory and composite backend
`CompositeBackend`: `/scratch/` State, `/filings/` filesystem (cache),
`/memories/` + `/reports/` Store. SQLite checkpointer and a persistent store for
local dev. `AGENTS.md` house memory; per-ticker `thesis.md`.
**Done when:** a second run for the same ticker (new thread) reads the prior thesis.

### Phase 4 — HITL and paper broker
Paper broker (SQLite): positions, cash, order validation, size limits.
Tools `publish_note`, `place_paper_trade`, `update_thesis` gated by `interrupt_on`
with approve/edit/reject. CLI review loop.
**Done when:** tests show reject blocks the trade, edit changes the order, approve
executes; state survives restart.

### Phase 5 — Red team and parallelism
Isolated `red-team` subagent (no trading tools). Parallel fan-out for
filings/quant/news. `news-sentiment` subagent with a stubbed news tool.
**Done when:** a run shows red-team critique incorporated before the publish gate.

### Phase 6 — Guardrails, evals, observability
Middleware: compliance language filter, numeric-citation check, audit log
(append-only JSONL), retry/rate-limit wrapper. Eval harness: numeric accuracy vs
XBRL, citation grounding, plan-step coverage, cost/tool-error counters; golden set
of fixture events. LangSmith tracing via env.
**Done when:** `make eval` prints a scorecard and fails on regression thresholds.

## Risks
- `deepagents` API differs from docs → check installed signatures first.
- No API key → live behavior unverified until the user supplies one; say so.
- EDGAR needs a descriptive `User-Agent` and rate limiting (10 req/s max).
