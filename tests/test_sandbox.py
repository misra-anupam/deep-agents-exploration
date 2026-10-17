import os
import sys
from pathlib import Path

import pytest

from equity_desk.sandbox import SandboxUnavailable, run_python

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/bin/sandbox-exec"), reason="needs macOS sandbox-exec"
)


def test_runs_code_and_reads_provided_files():
    r = run_python("print(open('facts.json').read().upper())", files={"facts.json": "abc"})
    assert (r.exit_code, r.stdout.strip()) == (0, "ABC")


def test_network_is_blocked():
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "s.connect(('93.184.216.34', 80))\n"
    )
    r = run_python(code)
    assert r.exit_code != 0
    assert "PermissionError" in r.stderr or "Operation not permitted" in r.stderr


def test_cannot_write_outside_workdir(tmp_path):
    target = tmp_path / "escape.txt"
    r = run_python(f"open({str(target)!r}, 'w').write('x')")
    assert r.exit_code != 0
    assert not target.exists()


def test_can_write_inside_workdir():
    r = run_python("open('out.txt','w').write('ok'); print(open('out.txt').read())")
    assert r.stdout.strip() == "ok"


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    r = run_python("import os; print(sorted(os.environ))")
    assert "ANTHROPIC_API_KEY" not in r.stdout


def test_credential_dirs_are_unreadable():
    ssh = Path.home() / ".ssh"
    if not ssh.exists():
        pytest.skip("no ~/.ssh to probe")
    r = run_python(f"import os; print(os.listdir({str(ssh)!r}))")
    assert r.exit_code != 0


def test_timeout_kills_runaway_code():
    r = run_python("while True: pass", timeout_s=1)
    assert r.timed_out and r.exit_code is None


def test_output_is_capped():
    r = run_python("print('x' * 100000)")
    assert r.truncated and len(r.stdout) <= 20_000


def test_path_traversal_in_file_names_rejected():
    with pytest.raises(ValueError):
        run_python("pass", files={"../evil.py": "x"})


def test_fails_closed_without_sandbox(monkeypatch):
    monkeypatch.setattr("equity_desk.sandbox._SANDBOX_EXEC", "/nonexistent/sandbox-exec")
    monkeypatch.delenv("EQUITY_DESK_ALLOW_UNSANDBOXED", raising=False)
    with pytest.raises(SandboxUnavailable):
        run_python("print(1)")
