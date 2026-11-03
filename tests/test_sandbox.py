# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest

from smolagents.sandbox import (
    Sandbox,
    SandboxCommandError,
    SandboxConfig,
    SandboxFileError,
    SandboxLifecycleError,
    SandboxPathViolation,
    SandboxResourceError,
    SandboxSecurityError,
    create_sandbox_tools,
)


def _sandbox(tmp_path, **config_kwargs):
    config = SandboxConfig(
        root_dir=tmp_path / "sandbox",
        use_os_isolation=False,
        **config_kwargs,
    )
    sandbox = Sandbox.create(config)
    sandbox.start()
    return sandbox


def test_lifecycle_create_and_destroy():
    sandbox = Sandbox.create(SandboxConfig(use_os_isolation=False))

    assert sandbox.is_active
    assert (sandbox.root / "workspace").is_dir()

    sandbox.destroy()

    assert not sandbox.is_active
    assert not sandbox.root.exists()
    with pytest.raises(SandboxLifecycleError):
        sandbox.read_text("missing.txt")


def test_read_is_confined_to_readable_roots(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.write_text("data/hello.txt", "hello")

    assert sandbox.read_text("data/hello.txt") == "hello"
    assert sandbox.read_text(sandbox.workspace / "data" / "hello.txt") == "hello"

    with pytest.raises(SandboxPathViolation):
        sandbox.read_text("../outside.txt")
    with pytest.raises(SandboxPathViolation):
        sandbox.read_text("/etc/passwd")


def test_symlink_escape_is_blocked(tmp_path):
    sandbox = _sandbox(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = sandbox.workspace / "escape.txt"
    os.symlink(outside, link)

    with pytest.raises(SandboxPathViolation):
        sandbox.read_text("escape.txt")


def test_write_is_confined_and_validates_file_type(tmp_path):
    sandbox = _sandbox(tmp_path)

    target = sandbox.write_text("reports/result.txt", "content")
    assert target == sandbox.workspace / "reports" / "result.txt"
    assert target.read_text(encoding="utf-8") == "content"

    with pytest.raises(SandboxPathViolation):
        sandbox.write_text("../escape.txt", "content")
    with pytest.raises(SandboxSecurityError):
        sandbox.write_text("script.sh", "echo unsafe")


def test_write_size_limit(tmp_path):
    sandbox = _sandbox(tmp_path, max_file_size_bytes=8)

    with pytest.raises(SandboxResourceError):
        sandbox.write_text("too-large.txt", "x" * 9)


def test_allowed_write_extensions(tmp_path):
    sandbox = _sandbox(tmp_path, allowed_write_extensions=frozenset({".txt"}))

    assert sandbox.write_text("ok.txt", "ok").exists()
    with pytest.raises(SandboxSecurityError):
        sandbox.write_text("blocked.json", "{}")


def test_list_files(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.write_text("a.txt", "a")
    sandbox.write_text("nested/b.txt", "b")

    assert sandbox.list_files() == ["a.txt", "nested/b.txt"]


def test_command_whitelist_and_path_validation(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.write_text("hello.txt", "hello")

    result = sandbox.run_command("echo hello")
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"

    result = sandbox.run_command(["cat", "hello.txt"])
    assert result.returncode == 0
    assert result.stdout == "hello"

    with pytest.raises(SandboxCommandError):
        sandbox.run_command(["rm", "-rf", "hello.txt"])
    with pytest.raises(SandboxPathViolation):
        sandbox.run_command(["cat", "/etc/passwd"])
    with pytest.raises(SandboxSecurityError):
        sandbox.run_command("echo hello; rm -rf hello.txt")


def test_command_timeout_and_output_limit(tmp_path):
    sandbox = _sandbox(tmp_path, command_timeout_seconds=0.1, max_output_bytes=16)

    result = sandbox.run_command("sleep 1")
    assert result.timed_out is True
    assert result.returncode != 0

    result = sandbox.run_command("echo " + "x" * 100)
    assert result.truncated is True
    assert len(result.stdout) <= 16


def test_command_stdin_and_environment_validation(tmp_path):
    sandbox = _sandbox(tmp_path)

    result = sandbox.run_command(["cat"], stdin="from stdin")
    assert result.returncode == 0
    assert result.stdout == "from stdin"

    with pytest.raises(SandboxSecurityError):
        sandbox.run_command(["echo"], env={"LD_PRELOAD": "evil.so"})


def test_default_sandbox_tools_route_through_sandbox(tmp_path):
    sandbox = _sandbox(tmp_path)
    read_tool, write_tool, command_tool = create_sandbox_tools(sandbox)

    assert write_tool.forward("tool.txt", "tool content") == f"Wrote {sandbox.workspace / 'tool.txt'}"
    assert read_tool.forward("tool.txt") == "tool content"
    output = command_tool.forward("echo hello")
    assert "hello" in output

    with pytest.raises(SandboxPathViolation):
        read_tool.forward("/etc/passwd")


def test_os_isolation_backend_is_selected(tmp_path):
    from smolagents.sandbox import ISOLATION_BACKEND_NONE, _detect_isolation_backend

    backend = _detect_isolation_backend()
    if backend == ISOLATION_BACKEND_NONE:
        pytest.skip("No OS-level sandbox backend available on this platform.")

    sandbox = Sandbox.create(SandboxConfig(root_dir=tmp_path / "isolated"))
    try:
        assert sandbox.backend == backend
        result = sandbox.run_command("echo isolated")
        assert result.returncode == 0
        assert result.stdout.strip() == "isolated"
    finally:
        sandbox.destroy()


def test_file_read_tool_rejects_directory_reads(tmp_path):
    sandbox = _sandbox(tmp_path)
    sandbox.write_text("a.txt", "a")

    with pytest.raises(SandboxFileError):
        sandbox.read_text(".")
