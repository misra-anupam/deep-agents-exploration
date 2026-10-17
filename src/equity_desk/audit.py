"""Append-only, hash-chained audit log of every tool call and human decision.

Each line embeds the SHA-256 of the previous line, so editing or deleting a past entry
breaks the chain and `verify()` reports where. This is tamper-*evident*, not tamper-proof:
someone with write access can rewrite the whole file; ship the log off-box for real use.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

GENESIS = "0" * 64
_SECRET_KEY = re.compile(r"key|token|secret|password|authorization", re.IGNORECASE)
_MAX_FIELD = 500


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "[REDACTED]" if _SECRET_KEY.search(str(k)) else _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, str) and len(value) > _MAX_FIELD:
        return value[:_MAX_FIELD] + f"…[+{len(value) - _MAX_FIELD} chars]"
    return value


class AuditLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        last = self.path.read_text().rstrip("\n").splitlines()[-1]
        return hashlib.sha256(last.encode()).hexdigest()

    def append(self, event: dict) -> None:
        with self._lock:  # parallel subagents log concurrently
            record = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                **_clean(event),
                "prev": self._last_hash(),
            }
            with self.path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]

    def verify(self) -> int | None:
        """Return None if the chain is intact, else the 1-based line number of the first break."""
        prev = GENESIS
        if not self.path.exists():
            return None
        for n, line in enumerate(self.path.read_text().splitlines(), start=1):
            if json.loads(line).get("prev") != prev:
                return n
            prev = hashlib.sha256(line.encode()).hexdigest()
        return None


def _outcome(result: Any) -> tuple[str, str]:
    text = result.text if isinstance(result, ToolMessage) else str(result)[:200]
    if isinstance(result, ToolMessage) and result.status == "error":
        return "error", text[:200]
    if text.startswith(("Error", "REJECTED")) or "is not a valid tool" in text:
        return "refused", text[:200]
    return "ok", text[:200]


class AuditLogMiddleware(AgentMiddleware):
    """Logs every tool call this agent executes (post-approval, so edited args are what's logged)."""

    def __init__(self, log: AuditLog, agent: str = "lead"):
        super().__init__()
        self.log, self.agent = log, agent

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        started = time.monotonic()
        try:
            result = handler(request)
        except Exception as exc:
            self.log.append({"event": "tool_call", "agent": self.agent, "tool": call["name"],
                             "args": call["args"], "outcome": "exception", "summary": repr(exc)[:200]})
            raise
        outcome, summary = _outcome(result)
        self.log.append({"event": "tool_call", "agent": self.agent, "tool": call["name"], "args": call["args"],
                         "outcome": outcome, "summary": summary,
                         "ms": round((time.monotonic() - started) * 1000)})
        return result
