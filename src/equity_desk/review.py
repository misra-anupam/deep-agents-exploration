"""Drive an agent through human-in-the-loop interrupts until it finishes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.types import Command

from .audit import AuditLog

Decide = Callable[[list[dict], list[dict]], list[dict]]
"""(action_requests, review_configs) -> one decision per action request, in order."""


def drive(
    agent, first_input: Any, config: dict, decide: Decide, max_rounds: int = 20,
    audit: AuditLog | None = None,
) -> dict:
    """Invoke, and while the graph is paused for approval, ask `decide` and resume."""
    payload = first_input
    for _ in range(max_rounds):
        result = agent.invoke(payload, config)
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return result
        answers = {}
        for i in interrupts:
            requests = i.value["action_requests"]
            decisions = decide(requests, i.value["review_configs"])
            answers[i.id] = {"decisions": decisions}
            for req, dec in zip(requests, decisions):
                if audit:
                    audit.append({"event": "human_decision", "tool": req["name"], "args": req["args"],
                                  "decision": dec["type"], "detail": dec.get("message") or dec.get("edited_action")})
        # A single pending interrupt resumes positionally; several resume by interrupt id.
        payload = Command(resume=next(iter(answers.values())) if len(answers) == 1 else answers)
    raise RuntimeError(f"Agent still paused for approval after {max_rounds} rounds")


def approve_all(requests: list[dict], configs: list[dict]) -> list[dict]:
    return [{"type": "approve"} for _ in requests]


def reject_all(message: str = "Rejected by reviewer") -> Decide:
    return lambda requests, configs: [{"type": "reject", "message": message} for _ in requests]
