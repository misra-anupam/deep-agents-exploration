"""Filesystem routing for the agent."""

from __future__ import annotations

from deepagents import FilesystemPermission
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langgraph.store.base import BaseStore

from . import config
from .memory import memory_backend, report_backend


def build_backend(store: BaseStore | None = None) -> CompositeBackend:
    """Path-routed virtual filesystem.

    /filings/   disk cache written by download_filing (read-only to the agent)
    /skills/    skill directories shipped with the project (read-only to the agent)
    /memories/  durable Store: house AGENTS.md, per-ticker thesis + history
    /reports/   durable Store: published notes
    anything else (e.g. /scratch/) lives in per-thread agent state

    `store` may be omitted when the agent graph is compiled with one; the backends then
    pick it up from the LangGraph runtime.
    """
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/filings/": FilesystemBackend(root_dir=config.cache_dir(), virtual_mode=True),
            "/skills/": FilesystemBackend(root_dir=config.skills_dir(), virtual_mode=True),
            "/memories/": memory_backend(store),
            "/reports/": report_backend(store),
        },
    )


def build_permissions() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=["write"], paths=["/filings/**", "/skills/**"], mode="deny"
        ),
        # Thesis changes must go through update_thesis (validated, archived, and gated in Phase 4).
        FilesystemPermission(
            operations=["write"], paths=["/memories/*/thesis.md", "/memories/*/history/**"], mode="deny"
        ),
    ]
