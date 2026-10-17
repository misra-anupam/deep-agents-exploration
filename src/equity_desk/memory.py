"""Persistent storage: Store-backed memory/report namespaces and SQLite persistence."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from deepagents.backends import StoreBackend
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from . import config

MEMORY_NS = ("equity-desk", "memories")
REPORT_NS = ("equity-desk", "reports")
HOUSE_MEMORY_PATH = "/AGENTS.md"  # inside the /memories/ route


def memory_backend(store: BaseStore | None = None) -> StoreBackend:
    return StoreBackend(store=store, namespace=lambda _rt: MEMORY_NS)


def report_backend(store: BaseStore | None = None) -> StoreBackend:
    return StoreBackend(store=store, namespace=lambda _rt: REPORT_NS)


def seed_house_memory(store: BaseStore) -> bool:
    """Install memory/AGENTS.md on first run. Never overwrites edits. Returns True if seeded."""
    backend = memory_backend(store)
    if backend.read(HOUSE_MEMORY_PATH).file_data is not None:
        return False
    backend.write(HOUSE_MEMORY_PATH, (config.ROOT / "memory" / "AGENTS.md").read_text())
    return True


@contextmanager
def open_persistence(directory: Path | str | None = None):
    """Yield (checkpointer, store) backed by SQLite files, so state survives restarts."""
    directory = Path(directory or config.ROOT / "data")
    directory.mkdir(parents=True, exist_ok=True)
    conns = [
        sqlite3.connect(directory / "checkpoints.sqlite", check_same_thread=False),
        # SqliteStore manages its own transactions, so its connection must autocommit.
        sqlite3.connect(directory / "store.sqlite", check_same_thread=False, isolation_level=None),
    ]
    try:
        checkpointer = SqliteSaver(conns[0])
        checkpointer.setup()
        store = SqliteStore(conns[1])
        store.setup()
        seed_house_memory(store)
        yield checkpointer, store
    finally:
        for conn in conns:
            conn.close()
