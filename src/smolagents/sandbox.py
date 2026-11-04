#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Isolated sandbox for tool file and command operations.

[`Sandbox`] owns a dedicated temporary directory and exposes the only supported file read/write and command
execution paths. Path resolution is confined to explicit readable/writable roots, commands run from a whitelist with
sanitized environment and resource limits, and an OS-level sandbox backend (macOS `sandbox-exec` or Linux `bwrap`)
further restricts file and network access.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import AuditEventType, AuditLog
from .tools import Tool


ISOLATION_BACKEND_SANDBOX_EXEC = "sandbox-exec"
ISOLATION_BACKEND_BWRAP = "bwrap"
ISOLATION_BACKEND_NONE = "none"

DEFAULT_ALLOWED_COMMANDS = frozenset(
    {
        "cat",
        "cut",
        "date",
        "df",
        "du",
        "echo",
        "false",
        "file",
        "grep",
        "head",
        "ls",
        "printf",
        "pwd",
        "sleep",
        "sort",
        "stat",
        "tail",
        "tr",
        "true",
        "uniq",
        "wc",
    }
)

DEFAULT_DENIED_WRITE_EXTENSIONS = frozenset(
    {
        ".app",
        ".bat",
        ".bash",
        ".cmd",
        ".com",
        ".command",
        ".dll",
        ".dylib",
        ".exe",
        ".fish",
        ".jar",
        ".msi",
        ".pyc",
        ".pyo",
        ".scr",
        ".sh",
        ".so",
        ".zsh",
    }
)

DENIED_ENVIRONMENT_KEYS = frozenset(
    {
        "BASH_ENV",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "GIT_EXEC_PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "SHELLOPTS",
    }
)


class SandboxError(Exception):
    """Base class for sandbox errors."""


class SandboxLifecycleError(SandboxError):
    """Raised when an operation is attempted on a closed or unstarted sandbox."""


class SandboxSecurityError(SandboxError):
    """Base class for security interception errors."""


class SandboxPathViolation(SandboxSecurityError):
    """Raised when a path escapes the configured readable or writable roots."""


class SandboxCommandError(SandboxError):
    """Raised when a command is not allowed or cannot be started."""


class SandboxTimeoutError(SandboxError):
    """Raised when a command exceeds its execution deadline."""


class SandboxOutputLimitError(SandboxError):
    """Raised when command output exceeds the configured limit."""


class SandboxResourceError(SandboxError):
    """Raised when resource limits cannot be applied or are exceeded."""


class SandboxIsolationError(SandboxError):
    """Raised when OS-level isolation is required but no backend is available."""


class SandboxFileError(SandboxError):
    """Raised for invalid file operations."""


@dataclass
class SandboxConfig:
    """Configuration for a [`Sandbox`].

    Args:
        root_dir (`str` or `Path`, *optional*): Existing sandbox root. When omitted a temporary directory is created.
        readable_dirs (`tuple[str | Path, ...]`, *optional*): Extra read-only roots outside the sandbox root.
        writable_dir (`str` or `Path`, *optional*): Writable root inside the sandbox root. Defaults to `root/workspace`.
        allowed_commands (`frozenset[str]`, *optional*): Command whitelist by basename.
        denied_write_extensions (`frozenset[str]`, *optional*): File extensions rejected for writes.
        allowed_write_extensions (`frozenset[str] | None`, *optional*): If set, only these extensions may be written.
        max_file_size_bytes (`int`): Maximum size for reads and writes.
        max_output_bytes (`int`): Maximum captured bytes per command stream.
        command_timeout_seconds (`float`): Default command timeout.
        cpu_time_limit_seconds (`int`): Child CPU time limit.
        memory_limit_bytes (`int`): Child address-space limit.
        max_open_files (`int`): Child open-file limit.
        use_os_isolation (`bool`): Whether an OS sandbox backend is required.
        cleanup_on_destroy (`bool`): Whether `destroy()` removes sandbox-owned directories.
        env (`dict[str, str]`, *optional*): Extra environment variables passed to commands.
    """

    root_dir: str | Path | None = None
    readable_dirs: tuple[str | Path, ...] = ()
    writable_dir: str | Path | None = None
    allowed_commands: frozenset[str] = DEFAULT_ALLOWED_COMMANDS
    denied_write_extensions: frozenset[str] = DEFAULT_DENIED_WRITE_EXTENSIONS
    allowed_write_extensions: frozenset[str] | None = None
    max_file_size_bytes: int = 10 * 1024 * 1024
    max_output_bytes: int = 200_000
    command_timeout_seconds: float = 30.0
    cpu_time_limit_seconds: int = 30
    memory_limit_bytes: int = 512 * 1024 * 1024
    max_open_files: int = 64
    use_os_isolation: bool = True
    cleanup_on_destroy: bool = True
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class CommandResult:
    """Result of a sandboxed command execution."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False


def _detect_isolation_backend() -> str:
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
        return ISOLATION_BACKEND_SANDBOX_EXEC
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return ISOLATION_BACKEND_BWRAP
    return ISOLATION_BACKEND_NONE


def _quote_seatbelt_path(path: str | Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _make_preexec_fn(config: SandboxConfig):
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:
        return None

    def _apply_limits():
        limits = [
            (resource.RLIMIT_CPU, (config.cpu_time_limit_seconds, config.cpu_time_limit_seconds + 1)),
            (resource.RLIMIT_AS, (config.memory_limit_bytes, config.memory_limit_bytes)),
            (resource.RLIMIT_FSIZE, (config.max_file_size_bytes, config.max_file_size_bytes)),
            (resource.RLIMIT_NOFILE, (config.max_open_files, config.max_open_files)),
        ]
        for resource_id, value in limits:
            try:
                resource.setrlimit(resource_id, value)
            except (OSError, ValueError):
                continue

    return _apply_limits


class Sandbox:
    """Lifecycle-managed isolated workspace for file and command operations.

    Args:
        config ([`SandboxConfig`], *optional*): Sandbox configuration.
        audit_log ([`AuditLog`], *optional*): Append-only log receiving sandbox denials.
        run_id (`str`, *optional*): Run identifier attached to audit events.
        actor (`str`): Actor recorded for audit events produced by the sandbox.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        audit_log: AuditLog | None = None,
        run_id: str | None = None,
        actor: str = "sandbox",
    ):
        self.audit_log = audit_log
        self.run_id = run_id
        self.actor = actor
        self.config = config or SandboxConfig()
        if self.config.max_file_size_bytes <= 0 or self.config.max_output_bytes <= 0:
            raise SandboxError("File and output limits must be positive.")
        self._owns_root = self.config.root_dir is None
        if self._owns_root:
            self.root = Path(tempfile.mkdtemp(prefix="smolagents-sandbox-")).resolve()
        else:
            self.root = Path(self.config.root_dir).expanduser().resolve()
            self.root.mkdir(parents=True, exist_ok=True)

        self.workspace = self.root / "workspace"
        self.tmp_dir = self.root / "tmp"
        self.writable_root = (
            Path(self.config.writable_dir).expanduser().resolve()
            if self.config.writable_dir is not None
            else self.workspace
        )
        self._assert_path_within(self.writable_root, [self.root], "writable_dir")
        self.readable_roots = [
            self.workspace,
            self.writable_root,
            *[Path(path).expanduser().resolve() for path in self.config.readable_dirs],
        ]
        self.backend = ISOLATION_BACKEND_NONE
        self._active = False

    @classmethod
    def create(
        cls,
        config: SandboxConfig | None = None,
        *,
        audit_log: AuditLog | None = None,
        run_id: str | None = None,
        actor: str = "sandbox",
    ) -> "Sandbox":
        sandbox = cls(config, audit_log=audit_log, run_id=run_id, actor=actor)
        sandbox.start()
        return sandbox

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> "Sandbox":
        """Create sandbox directories, select an isolation backend, and mark the sandbox active."""
        if self._active:
            return self
        if self.config.use_os_isolation:
            self.backend = _detect_isolation_backend()
            if self.backend == ISOLATION_BACKEND_NONE:
                raise SandboxIsolationError(
                    "OS-level sandbox isolation is unavailable. Install `bwrap` on Linux or set "
                    "`use_os_isolation=False` only if you accept reduced isolation."
                )
        for directory in (self.workspace, self.tmp_dir, self.writable_root):
            directory.mkdir(parents=True, exist_ok=True)
        self._active = True
        return self

    def destroy(self) -> None:
        """Remove sandbox-owned directories and mark the sandbox closed."""
        if not self._active:
            return
        self._active = False
        if self.config.cleanup_on_destroy:
            shutil.rmtree(self.workspace, ignore_errors=True)
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            if self._owns_root:
                shutil.rmtree(self.root, ignore_errors=True)
            elif self.writable_root != self.workspace:
                shutil.rmtree(self.writable_root, ignore_errors=True)

    close = destroy

    def __enter__(self) -> "Sandbox":
        if not self._active:
            self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()

    def _ensure_active(self) -> None:
        if not self._active:
            raise SandboxLifecycleError("Sandbox is not active; call start() before using it.")

    def audit_denial(self, *, rule: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a sandbox denial in the audit log when one is configured."""
        if self.audit_log is None:
            return
        self.audit_log.append(
            event_type=AuditEventType.SANDBOX_DENY,
            actor=self.actor,
            run_id=self.run_id,
            tool_name="sandbox",
            rule=rule,
            decision="denied",
            message=message,
            metadata=metadata,
        )

    def _assert_path_within(self, target: Path, roots: list[Path], label: str) -> None:
        for root in roots:
            try:
                if os.path.commonpath([str(target), str(root)]) == str(root):
                    return
            except ValueError:
                continue
        message = f"{label} '{target}' is outside the allowed sandbox boundary."
        self.audit_denial(rule="path_escape", message=message, metadata={"path": str(target), "label": label})
        raise SandboxPathViolation(message)

    def _resolve_read_path(self, path: str | Path) -> Path:
        self._ensure_active()
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.workspace / target
        target = target.resolve(strict=False)
        self._assert_path_within(target, self.readable_roots, "read path")
        if not target.exists():
            raise SandboxFileError(f"Read path '{target}' does not exist.")
        if target.is_dir():
            raise SandboxFileError(f"Read path '{target}' is a directory.")
        try:
            size = target.stat().st_size
        except OSError as e:
            raise SandboxFileError(f"Could not stat '{target}': {e}") from e
        if size > self.config.max_file_size_bytes:
            raise SandboxResourceError(
                f"Read path '{target}' exceeds max_file_size_bytes ({self.config.max_file_size_bytes})."
            )
        return target

    def _resolve_write_path(self, path: str | Path) -> Path:
        self._ensure_active()
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.writable_root / target
        target = target.resolve(strict=False)
        self._assert_path_within(target, [self.writable_root], "write path")
        if target.exists() and target.is_symlink():
            raise SandboxPathViolation(f"Refusing to write through symlink '{target}'.")
        extension = target.suffix.lower()
        denied = self.config.denied_write_extensions
        if extension and extension in denied:
            raise SandboxSecurityError(f"Writing files with extension '{extension}' is not allowed.")
        allowed = self.config.allowed_write_extensions
        if allowed is not None and extension not in allowed:
            raise SandboxSecurityError(f"Writing files with extension '{extension}' is not allowed.")
        return target

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        """Read a UTF-8 text file inside the sandbox read boundary."""
        self._ensure_active()
        target = self._resolve_read_path(path)
        try:
            return target.read_text(encoding=encoding)
        except (OSError, UnicodeError) as e:
            raise SandboxFileError(f"Could not read '{target}': {e}") from e

    def read_bytes(self, path: str | Path) -> bytes:
        """Read a file inside the sandbox read boundary."""
        self._ensure_active()
        target = self._resolve_read_path(path)
        try:
            return target.read_bytes()
        except OSError as e:
            raise SandboxFileError(f"Could not read '{target}': {e}") from e

    def write_text(self, path: str | Path, content: str, *, encoding: str = "utf-8", overwrite: bool = True) -> Path:
        """Atomically write a text file inside the sandbox writable root."""
        self._ensure_active()
        data = content.encode(encoding)
        return self.write_bytes(path, data, overwrite=overwrite)

    def write_bytes(self, path: str | Path, content: bytes, *, overwrite: bool = True) -> Path:
        """Atomically write bytes inside the sandbox writable root."""
        self._ensure_active()
        if not isinstance(content, (bytes, bytearray)):
            raise SandboxFileError("'content' must be bytes for write_bytes.")
        if len(content) > self.config.max_file_size_bytes:
            raise SandboxResourceError(
                f"Write content exceeds max_file_size_bytes ({self.config.max_file_size_bytes})."
            )
        target = self._resolve_write_path(path)
        if target.exists() and not overwrite:
            raise SandboxFileError(f"Write path '{target}' already exists.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._assert_path_within(target.parent.resolve(strict=False), [self.writable_root], "write parent")
            fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=".sandbox-write-", suffix=".tmp")
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(content)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, target)
            return target
        except OSError as e:
            raise SandboxFileError(f"Could not write '{target}': {e}") from e

    def list_files(self, path: str | Path = ".") -> list[str]:
        """List files under a readable directory, relative to that directory."""
        self._ensure_active()
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.workspace / target
        target = target.resolve(strict=False)
        self._assert_path_within(target, self.readable_roots, "list path")
        if not target.is_dir():
            raise SandboxFileError(f"List path '{target}' is not a directory.")
        return sorted(item.relative_to(target).as_posix() for item in target.rglob("*") if item.is_file())

    def run_command(
        self,
        command: str | list[str] | tuple[str, ...],
        *,
        timeout: float | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> CommandResult:
        """Run a whitelisted command inside the sandbox with limits and output capture."""
        self._ensure_active()
        argv = self._parse_command(command)
        executable = self._resolve_command(argv[0])
        argv[0] = executable
        work_dir = self._resolve_command_cwd(cwd)
        self._validate_command_arguments(argv[1:], work_dir)
        effective_env = self._build_environment(env)
        effective_timeout = self.config.command_timeout_seconds if timeout is None else timeout
        if effective_timeout <= 0:
            raise SandboxCommandError("Command timeout must be positive.")

        full_argv = self._isolation_prefix(argv, work_dir)
        stdout_file = tempfile.TemporaryFile(dir=self.tmp_dir)
        stderr_file = tempfile.TemporaryFile(dir=self.tmp_dir)
        start_time = time.monotonic()
        timed_out = False
        try:
            process = subprocess.Popen(
                full_argv,
                cwd=str(work_dir),
                env=effective_env,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                preexec_fn=_make_preexec_fn(self.config),
            )
            try:
                process.communicate(
                    input=stdin.encode("utf-8") if stdin is not None else None,
                    timeout=effective_timeout,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_process_group(process)
                process.communicate()
            duration = time.monotonic() - start_time
            stdout, stdout_truncated = self._read_captured_output(stdout_file)
            stderr, stderr_truncated = self._read_captured_output(stderr_file)
            return CommandResult(
                command=argv,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=timed_out,
                truncated=stdout_truncated or stderr_truncated,
            )
        except FileNotFoundError as e:
            raise SandboxCommandError(f"Command '{argv[0]}' could not be executed: {e}") from e
        except OSError as e:
            raise SandboxCommandError(f"Could not start command '{argv[0]}': {e}") from e
        finally:
            stdout_file.close()
            stderr_file.close()

    def _parse_command(self, command: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(command, str):
            if any(character in command for character in ";&|><`$\n"):
                message = "Shell operators and command substitution are not allowed; pass an argument list instead."
                self.audit_denial(rule="shell_operator", message=message, metadata={"command": command})
                raise SandboxSecurityError(message)
            try:
                argv = shlex.split(command)
            except ValueError as e:
                raise SandboxCommandError(f"Could not parse command: {e}") from e
        elif isinstance(command, (list, tuple)):
            argv = list(command)
        else:
            raise SandboxCommandError("Command must be a string or a list of strings.")
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise SandboxCommandError("Command must contain at least one non-empty argument.")
        return argv

    def _resolve_command(self, command: str) -> str:
        resolved = shutil.which(command)
        if resolved is None:
            raise SandboxCommandError(f"Command '{command}' was not found on PATH.")
        if Path(resolved).name not in self.config.allowed_commands:
            message = f"Command '{Path(resolved).name}' is not in the allowed command whitelist."
            self.audit_denial(rule="command_not_allowed", message=message, metadata={"command": command})
            raise SandboxCommandError(message)
        return resolved

    def _resolve_command_cwd(self, cwd: str | Path | None) -> Path:
        target = Path(cwd).expanduser() if cwd is not None else self.writable_root
        if not target.is_absolute():
            target = self.writable_root / target
        target = target.resolve(strict=False)
        self._assert_path_within(target, [self.root, self.writable_root], "command cwd")
        if not target.is_dir():
            raise SandboxFileError(f"Command cwd '{target}' is not a directory.")
        return target

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return value.startswith(("/", "./", "../", "~")) or "/" in value or "\\" in value or Path(value).exists()

    def _validate_command_arguments(self, arguments: list[str], cwd: Path) -> None:
        output_flags = {"-o", "--output", "--out"}
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument in output_flags and index + 1 < len(arguments):
                self._validate_command_path(arguments[index + 1], cwd, writable=True)
                index += 2
                continue
            value = argument.split("=", 1)[1] if argument.startswith("--") and "=" in argument else argument
            if value in {"-", "--"} or value.startswith("-"):
                index += 1
                continue
            if self._looks_like_path(value):
                self._validate_command_path(value, cwd, writable=False)
            index += 1

    def _validate_command_path(self, value: str, cwd: Path, *, writable: bool) -> None:
        target = Path(value).expanduser()
        if not target.is_absolute():
            target = cwd / target
        target = target.resolve(strict=False)
        roots = [self.writable_root] if writable else [*self.readable_roots, self.writable_root]
        self._assert_path_within(target, roots, "command argument path")

    def _build_environment(self, extra_env: dict[str, str] | None) -> dict[str, str]:
        environment = {
            "HOME": str(self.workspace),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(self.tmp_dir),
        }
        merged = {**self.config.env, **(extra_env or {})}
        for key, value in merged.items():
            if key in DENIED_ENVIRONMENT_KEYS:
                message = f"Environment variable '{key}' is not allowed in the sandbox."
                self.audit_denial(rule="environment_denied", message=message, metadata={"key": key})
                raise SandboxSecurityError(message)
            if not isinstance(key, str) or not isinstance(value, str):
                raise SandboxSecurityError("Environment keys and values must be strings.")
            environment[key] = value
        return environment

    def _isolation_prefix(self, argv: list[str], cwd: Path) -> list[str]:
        if self.backend == ISOLATION_BACKEND_SANDBOX_EXEC:
            return ["/usr/bin/sandbox-exec", "-p", self._seatbelt_profile(), *argv]
        if self.backend == ISOLATION_BACKEND_BWRAP:
            return self._bwrap_prefix(argv, cwd)
        return argv

    def _seatbelt_profile(self) -> str:
        read_allowed = [
            self.root,
            self.workspace,
            self.writable_root,
            *self.readable_roots,
        ]
        write_allowed = [self.writable_root, self.tmp_dir]
        deny_read = [
            Path.home(),
            "/Users",
            "/home",
            "/root",
            "/Volumes",
            "/private/etc",
            "/etc",
        ]
        rules = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow process-exec*)",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            '(allow file-read* (subpath "/"))',
            "(deny file-read* "
            + " ".join(f"(subpath {_quote_seatbelt_path(path)})" for path in _unique(deny_read))
            + ")",
            "(allow file-read* "
            + " ".join(f"(subpath {_quote_seatbelt_path(path)})" for path in _unique(read_allowed))
            + ")",
            "(allow file-write* "
            + " ".join(f"(subpath {_quote_seatbelt_path(path)})" for path in _unique(write_allowed))
            + ' (literal "/dev/null"))',
        ]
        return "\n".join(rules)

    def _bwrap_prefix(self, argv: list[str], cwd: Path) -> list[str]:
        prefix = ["bwrap", "--die-with-parent", "--unshare-all", "--new-session", "--dev", "/dev", "--proc", "/proc"]
        for system_path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
            if Path(system_path).exists():
                prefix.extend(["--ro-bind", system_path, system_path])
        prefix.extend(["--bind", str(self.root), str(self.root)])
        prefix.extend(["--bind", str(self.tmp_dir), str(self.tmp_dir)])
        for readable in self.config.readable_dirs:
            resolved = str(Path(readable).expanduser().resolve())
            if Path(resolved).exists():
                prefix.extend(["--ro-bind", resolved, resolved])
        prefix.extend(["--chdir", str(cwd), *argv])
        return prefix

    @staticmethod
    def _kill_process_group(process: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass

    def _read_captured_output(self, stream) -> tuple[str, bool]:
        stream.seek(0)
        data = stream.read()
        truncated = len(data) > self.config.max_output_bytes
        if truncated:
            data = data[: self.config.max_output_bytes]
        return data.decode("utf-8", errors="replace"), truncated


def _unique(paths: list[Any]) -> list[str]:
    seen = set()
    result = []
    for path in paths:
        value = str(path)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class SandboxFileReadTool(Tool):
    name = "sandbox_file_read"
    description = "Read a UTF-8 text file from the sandbox workspace."
    inputs = {"path": {"type": "string", "description": "Path relative to the sandbox workspace."}}
    output_type = "string"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        super().__init__()

    def forward(self, path: str) -> str:
        return self.sandbox.read_text(path)


class SandboxFileWriteTool(Tool):
    name = "sandbox_file_write"
    description = "Write UTF-8 text to a file inside the sandbox workspace."
    inputs = {
        "path": {"type": "string", "description": "Path relative to the sandbox workspace."},
        "content": {"type": "string", "description": "Text content to write."},
    }
    output_type = "string"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        super().__init__()

    def forward(self, path: str, content: str) -> str:
        target = self.sandbox.write_text(path, content)
        return f"Wrote {target}"


class SandboxCommandTool(Tool):
    name = "sandbox_command"
    description = "Run a whitelisted command inside the sandbox."
    inputs = {"command": {"type": "string", "description": "Command line to execute."}}
    output_type = "string"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        super().__init__()

    def forward(self, command: str) -> str:
        result = self.sandbox.run_command(command)
        return (
            f"Return code: {result.returncode}\n"
            f"Timed out: {result.timed_out}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )


def create_sandbox_tools(sandbox: Sandbox) -> list[Tool]:
    """Create the default sandbox-backed file and command tools."""
    return [SandboxFileReadTool(sandbox), SandboxFileWriteTool(sandbox), SandboxCommandTool(sandbox)]


__all__ = [
    "CommandResult",
    "Sandbox",
    "SandboxCommandError",
    "SandboxCommandTool",
    "SandboxConfig",
    "SandboxError",
    "SandboxFileError",
    "SandboxFileReadTool",
    "SandboxFileWriteTool",
    "SandboxIsolationError",
    "SandboxLifecycleError",
    "SandboxOutputLimitError",
    "SandboxPathViolation",
    "SandboxResourceError",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "create_sandbox_tools",
]
