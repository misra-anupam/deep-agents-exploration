"""Offline test helpers: a scripted chat model that supports tool calling."""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ids = itertools.count(1)


def call(name: str, **args: Any) -> dict:
    """Build a tool-call dict with a unique id."""
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def turn(*calls: dict, text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per model invocation.

    Records every prompt it receives in `seen` so tests can assert on context.
    """

    script: list[AIMessage]
    cursor: int = 0
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # tool schemas are irrelevant to a script
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if self.cursor >= len(self.script):
            raise AssertionError("ScriptedChatModel ran out of scripted turns")
        msg = self.script[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])
