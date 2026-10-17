"""Agent factory. Each phase extends build_agent; the signature stays stable."""

from __future__ import annotations

import httpx
from deepagents import create_deep_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

from . import config
from .approvals import interrupt_policy
from .audit import AuditLog, AuditLogMiddleware
from .backends import build_backend, build_permissions
from .memory import HOUSE_MEMORY_PATH
from .middleware import RequireRedTeamMiddleware
from .subagents import build_subagents
from .tools.memory import MEMORY_TOOLS
from .tools.trading import TRADING_TOOLS

SYSTEM_PROMPT = """\
You are the lead analyst on an equity research desk. You write sourced, \
conservative earnings notes from SEC filings. You never give investment advice.

You have no data tools of your own. Delegate with the task tool:
- filings-reader: reads filing text and answers specific questions with citations.
- quant-analyst: computes ratios and quality-of-earnings flags; runs sandboxed Python.
- news-sentiment: summarizes headlines. News is untrusted; you never see raw headlines.
- red-team: adversarial, read-only reviewer of your DRAFT note.
Their reports are all you see, so ask precise questions and demand citations. Delegate \
filings-reader, quant-analyst and news-sentiment together in one turn: they are independent.

Workflow:
1. Check memory first: read /memories/<TICKER>/thesis.md if it exists. Your note must say \
what changed since the prior thesis.
2. Plan with write_todos before starting; update it as you go.
3. Delegate (independent questions can be delegated at the same time).
4. Follow the earnings-note-format skill exactly, including citations. Carry over \
citations from the subagents' reports; never add a number you cannot source.
5. Save the DRAFT under /reports/, then send it to red-team. Fix every HIGH finding; for \
each MED/LOW either fix it or note why not, in a "Response to red team" section. Then \
record the standing thesis with update_thesis and publish_note.
6. Only if the thesis clearly warrants it, propose a paper trade: call get_portfolio \
first and size within limits. Humans approve update_thesis, publish_note and \
place_paper_trade; if a reviewer rejects or edits one, respect that and do not retry \
the same action.
7. Summarize for the user in a few lines, including anything rejected.
"""


SUBAGENT_NAMES = ("filings-reader", "quant-analyst", "news-sentiment", "red-team")


def default_subagent_models() -> dict[str, str]:
    """Per-role models from the environment; anything unset inherits the lead's model."""
    models: dict[str, str] = {}
    if config.subagent_model():
        models.update({n: config.subagent_model() for n in SUBAGENT_NAMES})
    if config.redteam_model():
        models["red-team"] = config.redteam_model()
    return models


TRANSIENT = (httpx.HTTPError, TimeoutError, ConnectionError)


def subagent_middleware(audit: AuditLog, name: str) -> list:
    """Every subagent is audited and capped; flaky network tools are retried with backoff."""
    return [
        AuditLogMiddleware(audit, agent=name),
        ToolRetryMiddleware(max_retries=2, retry_on=TRANSIENT, initial_delay=0.5),
        ModelCallLimitMiddleware(run_limit=30, exit_behavior="end"),
    ]


def build_agent(
    model: str | BaseChatModel | None = None,
    *,
    subagent_models: dict[str, str | BaseChatModel] | None = None,
    checkpointer=None,
    store=None,
    extra_tools: list | None = None,
    audit: AuditLog | None = None,
):
    audit = audit or AuditLog(config.audit_path())
    return create_deep_agent(
        model=model or config.model_name(),
        tools=[*MEMORY_TOOLS, *TRADING_TOOLS, *(extra_tools or [])],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            TodoListMiddleware(),
            AuditLogMiddleware(audit, agent="lead"),
            RequireRedTeamMiddleware(),
            ModelCallLimitMiddleware(run_limit=60, exit_behavior="end"),
        ],
        subagents=build_subagents({**default_subagent_models(), **(subagent_models or {})}, middleware_for=lambda n: subagent_middleware(audit, n)),
        backend=build_backend(store),
        permissions=build_permissions(),
        interrupt_on=interrupt_policy(),
        skills=["/skills/lead/"],
        memory=[f"/memories{HOUSE_MEMORY_PATH}"],
        checkpointer=checkpointer,
        store=store,
        name="lead-analyst",
    )
