"""Custom policy middleware (retry/limits come from LangChain's built-ins)."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .guardrails import red_team_done


class RequireRedTeamMiddleware(AgentMiddleware):
    """Refuse publish_note unless the red team reviewed the draft after it was written."""

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] != "publish_note":
            return handler(request)
        report = call["args"].get("report_name", "")
        if not red_team_done(request.state["messages"], report):
            return ToolMessage(
                content=(
                    f"Error: cannot publish {report}: the red-team has not reviewed this draft. "
                    "Send the draft to the red-team subagent, address its findings, then retry."
                ),
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
        return handler(request)
