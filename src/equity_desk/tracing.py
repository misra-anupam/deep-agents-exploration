"""Observability: LangSmith tagging (via env) and a callback that counts usage across all agents."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def tracing_enabled() -> bool:
    on = os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", "")).lower() == "true"
    return on and bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


def tracing_status() -> str:
    if tracing_enabled():
        project = os.environ.get("LANGSMITH_PROJECT", "default")
        return f"LangSmith tracing ON (project: {project})"
    return "LangSmith tracing off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)"


class UsageTracker(BaseCallbackHandler):
    """Counts model calls, tokens, and tool use/errors, including inside subagents.

    Subagent messages never appear in the lead's state, so token accounting has to live in a
    callback attached to the run config.
    """

    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls: Counter[str] = Counter()
        self.tool_errors = 0

    def on_llm_end(self, response, **kwargs: Any) -> None:
        self.llm_calls += 1
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def on_tool_start(self, serialized, input_str, **kwargs: Any) -> None:
        self.tool_calls[(serialized or {}).get("name") or kwargs.get("name", "?")] += 1

    def on_tool_error(self, error, **kwargs: Any) -> None:
        self.tool_errors += 1

    def summary(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": sum(self.tool_calls.values()),
            "tool_errors": self.tool_errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def run_config(thread_id: str, ticker: str, period: str, *, callbacks: list | None = None) -> dict:
    """Config for agent.invoke: thread id for persistence, tags/metadata for LangSmith filtering."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"equity-desk:{ticker}:{period}",
        "tags": ["equity-desk", f"ticker:{ticker}", f"period:{period}"],
        "metadata": {"ticker": ticker, "period": period, "app": "equity-desk"},
        "callbacks": callbacks or [],
    }
