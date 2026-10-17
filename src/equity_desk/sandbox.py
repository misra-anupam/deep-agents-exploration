"""Run model-written Python in a restricted subprocess.

macOS only (uses sandbox-exec): network denied, writes confined to a temp working
directory, environment scrubbed (no API keys), hard timeout, capped output.

Limits to be honest about: file *reads* are mostly allowed (a few credential
directories are blocked), and sandbox-exec is deprecated by Apple. Treat this as a
development-grade guard. For production, use a container or VM sandbox backend.
Fails closed: without sandbox-exec it refuses to run unless explicitly overridden.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 20_000
DEFAULT_TIMEOUT_S = 20
_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class RunResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

    def render(self) -> str:
        parts = []
        if self.timed_out:
            parts.append("TIMEOUT: killed")
        parts.append(f"exit_code={self.exit_code}")
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        if self.truncated:
            parts.append("[output truncated]")
        return "\n".join(parts)


def _profile(workdir: Path) -> str:
    home = Path.home()
    blocked = " ".join(
        f'(subpath "{home / d}")' for d in (".ssh", ".aws", ".gnupg", ".config", ".kube")
    )
    return (
        "(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n"
        f'(allow file-write* (subpath "{workdir.resolve()}") '
        '(literal "/dev/null") (literal "/dev/dtracehelper"))\n'
        f"(deny file-read* {blocked})\n"
    )


def _clip(text: str) -> tuple[str, bool]:
    raw = text.encode()
    if len(raw) <= MAX_OUTPUT_BYTES:
        return text, False
    return raw[:MAX_OUTPUT_BYTES].decode(errors="ignore"), True


def run_python(
    code: str,
    files: dict[str, str] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> RunResult:
    """Execute `code` with `files` (name -> text) available in the working directory."""
    if not os.path.exists(_SANDBOX_EXEC) and os.environ.get("EQUITY_DESK_ALLOW_UNSANDBOXED") != "1":
        raise SandboxUnavailable(
            "sandbox-exec not found; refusing to run model-written code unsandboxed. "
            "Set EQUITY_DESK_ALLOW_UNSANDBOXED=1 to override (unsafe)."
        )
    workdir = Path(tempfile.mkdtemp(prefix="eqdesk-sbx-"))
    try:
        for name, text in (files or {}).items():
            target = (workdir / name).resolve()
            if workdir.resolve() not in target.parents:
                raise ValueError(f"Illegal file name: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        (workdir / "main.py").write_text(code)

        env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = [sys.executable, "-I", "main.py"]
        if os.path.exists(_SANDBOX_EXEC):
            cmd = [_SANDBOX_EXEC, "-p", _profile(workdir), *cmd]
        try:
            proc = subprocess.run(
                cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            return RunResult(None, _clip(out)[0], "", timed_out=True)
        out, t1 = _clip(proc.stdout)
        err, t2 = _clip(proc.stderr)
        return RunResult(proc.returncode, out, err, truncated=t1 or t2)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
