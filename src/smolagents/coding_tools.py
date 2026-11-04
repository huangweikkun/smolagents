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
"""Workspace-scoped coding tools for [`ToolCallingAgent`].

The tools operate on the real workspace exposed by a [`Sandbox`] configured with `workspace_root`, so every read,
write and command still goes through the sandbox boundary. They are meant to be registered in a
[`ToolExecutionGate`](gate) together with `ToolRiskPolicy`: write and edit tools are high risk and need approval.

Tools:
    - `list_files`: navigate the workspace
    - `read_file`: read a file with line numbers
    - `search_code`: find symbols without reading whole files
    - `write_file`: create or overwrite an allowed path
    - `edit_file`: replace an exact string, with an expected-match guard
    - `run_test`: run the task-configured test command only
    - `git_diff`: show the changed lines for the final report
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .gate import ToolRiskLevel, ToolRiskPolicy
from .sandbox import Sandbox, SandboxFileError, SandboxSecurityError
from .tools import Tool


IGNORED_DIRS = frozenset({".git", ".venv", "__pycache__", "build", "dist", "node_modules"})
FORBIDDEN_READ_PARTS = frozenset({".git", ".ssh", "node_modules", "__pycache__", ".venv"})
FORBIDDEN_READ_NAMES = frozenset({".env", ".env.local", "id_rsa", "id_dsa", "credentials", "secrets.json"})
TEXT_EXTENSIONS = frozenset(
    {
        ".cfg",
        ".ini",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)
DEFAULT_READ_LINE_LIMIT = 200
DEFAULT_READ_CHAR_LIMIT = 20_000
DEFAULT_LIST_LIMIT = 300
DEFAULT_SEARCH_LIMIT = 30

CODING_WRITE_TOOLS = frozenset({"write_file", "edit_file"})


def default_coding_risk_policy() -> ToolRiskPolicy:
    """Risk policy that requires approval before workspace writes and edits."""
    return ToolRiskPolicy(tool_levels={name: ToolRiskLevel.HIGH for name in CODING_WRITE_TOOLS})


class WorkspaceTool(Tool):
    """Base class for tools that operate on a sandbox workspace."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        super().__init__()

    @staticmethod
    def normalize_path(path: Any) -> str:
        """Normalize a model-supplied path and reject absolute or escaping paths."""
        if path is None:
            return "."
        value = str(path).strip().replace("\\", "/")
        if value in ("", "."):
            return "."
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SandboxSecurityError(f"Path '{value}' must stay inside the workspace.")
        return candidate.as_posix()

    @staticmethod
    def is_forbidden_read(path: str) -> bool:
        """Whether `path` targets credentials, VCS internals or dependency caches."""
        path_obj = Path(path)
        if any(part in FORBIDDEN_READ_PARTS for part in path_obj.parts):
            return True
        return path_obj.name in FORBIDDEN_READ_NAMES

    @staticmethod
    def is_ignored(path: str) -> bool:
        return any(part in IGNORED_DIRS for part in Path(path).parts)

    @staticmethod
    def is_text_file(path: str) -> bool:
        return Path(path).suffix.lower() in TEXT_EXTENSIONS


class ReadableWorkspaceTool(WorkspaceTool):
    """Workspace tool that refuses credential and VCS paths before touching the sandbox."""

    def __init__(self, sandbox: Sandbox):
        super().__init__(sandbox)

    def _check_read_path(self, path: str) -> str:
        if self.is_forbidden_read(path):
            raise SandboxSecurityError(f"Reading '{path}' is not allowed.")
        return path


class WritableWorkspaceTool(WorkspaceTool):
    """Workspace tool with an allow/deny policy for write targets.

    Args:
        sandbox ([`Sandbox`]): Sandbox exposing the workspace root.
        allowed_paths (`Iterable[str]`, *optional*): fnmatch patterns explicitly allowed. `None` allows everything
            not denied.
        denied_paths (`Iterable[str]`, *optional*): fnmatch patterns denied unless an explicit allow pattern matches
            first, which supports "deny tests/** but allow tests/generated/**" setups.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        allowed_paths: Iterable[str] | None = None,
        denied_paths: Iterable[str] = (),
    ):
        self.allowed_paths: tuple[str, ...] | None = tuple(allowed_paths) if allowed_paths is not None else None
        self.denied_paths: tuple[str, ...] = tuple(denied_paths)
        super().__init__(sandbox)

    def check_write_path(self, path: str) -> str:
        """Validate `path` against the write policy and return the normalized path."""
        normalized = self.normalize_path(path)
        if normalized == ".":
            raise SandboxSecurityError("A file path is required for writes.")
        if self.allowed_paths is not None and any(
            fnmatch.fnmatchcase(normalized, pattern) for pattern in self.allowed_paths
        ):
            return normalized
        if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in self.denied_paths):
            raise SandboxSecurityError(f"Writing '{normalized}' is not allowed by the task policy.")
        if self.allowed_paths is not None:
            raise SandboxSecurityError(
                f"Writing '{normalized}' is outside the allowed paths: {', '.join(self.allowed_paths)}."
            )
        return normalized


class ListFilesTool(ReadableWorkspaceTool):
    name = "list_files"
    description = "List files in the workspace, optionally recursively."
    inputs = {
        "path": {"type": "string", "description": "Directory relative to the workspace.", "nullable": True},
        "recursive": {"type": "boolean", "description": "List nested files too.", "nullable": True},
    }
    output_type = "string"

    def forward(self, path: str | None = None, recursive: bool | None = None) -> str:
        directory = self._check_read_path(self.normalize_path(path))
        files = self.sandbox.list_files(directory)
        if not recursive:
            files = [item for item in files if "/" not in item]
        files = [item for item in files if not self.is_ignored(item)]
        if len(files) > DEFAULT_LIST_LIMIT:
            files = files[:DEFAULT_LIST_LIMIT]
            suffix = f"\n... truncated at {DEFAULT_LIST_LIMIT} entries"
        else:
            suffix = ""
        if not files:
            return "No files found." + suffix
        return "\n".join(files) + suffix


class ReadFileTool(ReadableWorkspaceTool):
    name = "read_file"
    description = "Read a UTF-8 text file with line numbers. Use start_line/end_line to page through large files."
    inputs = {
        "path": {"type": "string", "description": "File path relative to the workspace."},
        "start_line": {"type": "integer", "description": "First line to return (1-based).", "nullable": True},
        "end_line": {"type": "integer", "description": "Last line to return (inclusive).", "nullable": True},
    }
    output_type = "string"

    def forward(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        target = self._check_read_path(self.normalize_path(path))
        if not self.is_text_file(target):
            raise SandboxFileError(f"'{target}' is not a readable text file.")
        text = self.sandbox.read_text(target)
        lines = text.splitlines()
        start = max(int(start_line or 1), 1)
        end = int(end_line) if end_line is not None else len(lines)
        end = min(end, start + DEFAULT_READ_LINE_LIMIT - 1, len(lines))
        if start > len(lines):
            return f"File has {len(lines)} lines; start_line {start} is past the end."
        rendered = "\n".join(f"{number:>4} | {lines[number - 1]}" for number in range(start, end + 1))
        header = f"{target} (lines {start}-{end} of {len(lines)})"
        if len(rendered) > DEFAULT_READ_CHAR_LIMIT:
            rendered = rendered[:DEFAULT_READ_CHAR_LIMIT] + "\n... truncated"
        return f"{header}\n{rendered}"


class SearchCodeTool(ReadableWorkspaceTool):
    name = "search_code"
    description = "Search text files for a keyword or regular expression and return path:line matches."
    inputs = {
        "pattern": {"type": "string", "description": "Keyword or regular expression to search for."},
        "path": {"type": "string", "description": "Directory to search under.", "nullable": True},
        "regex": {"type": "boolean", "description": "Treat the pattern as a regular expression.", "nullable": True},
        "max_results": {"type": "integer", "description": "Maximum matches to return.", "nullable": True},
    }
    output_type = "string"

    def forward(
        self,
        pattern: str,
        path: str | None = None,
        regex: bool | None = None,
        max_results: int | None = None,
    ) -> str:
        directory = self._check_read_path(self.normalize_path(path))
        limit = max(int(max_results or DEFAULT_SEARCH_LIMIT), 1)
        expression = re.compile(pattern, re.IGNORECASE) if regex else None
        needle = pattern.lower()
        matches: list[str] = []
        for relative in self.sandbox.list_files(directory):
            if self.is_ignored(relative) or not self.is_text_file(relative):
                continue
            full_path = f"{directory}/{relative}" if directory != "." else relative
            try:
                text = self.sandbox.read_text(full_path)
            except SandboxFileError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                hit = bool(expression.search(line)) if expression else needle in line.lower()
                if hit:
                    matches.append(f"{full_path}:{number}: {line.strip()}")
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        if not matches:
            return "No matches."
        suffix = f"\n... truncated at {limit} matches" if len(matches) >= limit else ""
        return "\n".join(matches) + suffix


class WriteFileTool(WritableWorkspaceTool):
    name = "write_file"
    description = "Create or overwrite a text file inside the allowed workspace paths."
    inputs = {
        "path": {"type": "string", "description": "File path relative to the workspace."},
        "content": {"type": "string", "description": "Full text content of the file."},
        "overwrite": {"type": "boolean", "description": "Whether an existing file may be replaced.", "nullable": True},
    }
    output_type = "string"

    def forward(self, path: str, content: str, overwrite: bool | None = None) -> str:
        target = self.check_write_path(path)
        written = self.sandbox.write_text(target, content, overwrite=overwrite is not False)
        return f"Wrote {written.relative_to(self.sandbox.writable_root).as_posix()}"


class EditFileTool(WritableWorkspaceTool):
    name = "edit_file"
    description = "Replace an exact string in a text file, refusing when the match count differs."
    inputs = {
        "path": {"type": "string", "description": "File path relative to the workspace."},
        "old_string": {"type": "string", "description": "Exact text to replace."},
        "new_string": {"type": "string", "description": "Replacement text."},
        "expected_replacements": {
            "type": "integer",
            "description": "Expected number of occurrences of old_string.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        path: str,
        old_string: str,
        new_string: str,
        expected_replacements: int | None = None,
    ) -> str:
        target = self.check_write_path(path)
        text = self.sandbox.read_text(target)
        occurrences = text.count(old_string)
        expected = int(expected_replacements or 1)
        if occurrences != expected:
            raise SandboxFileError(
                f"'{target}' contains {occurrences} occurrences of old_string, expected {expected}."
            )
        updated = text.replace(old_string, new_string, expected)
        self.sandbox.write_text(target, updated)
        return f"Updated {target} ({expected} replacement(s))."


class RunTestTool(WorkspaceTool):
    name = "run_test"
    description = "Run the task-configured test command in the workspace and return its exit code and output."
    inputs = {}
    output_type = "string"

    def __init__(self, sandbox: Sandbox, command: Sequence[str]):
        if not command:
            raise ValueError("'command' must contain at least one argument.")
        self.command = tuple(command)
        super().__init__(sandbox)

    def forward(self) -> str:
        result = self.sandbox.run_command(list(self.command))
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if len(output) > DEFAULT_READ_CHAR_LIMIT:
            output = "...\n" + output[-DEFAULT_READ_CHAR_LIMIT:]
        return (
            f"Command: {' '.join(self.command)}\n"
            f"Exit code: {result.returncode}\n"
            f"Timed out: {result.timed_out}\n"
            f"Output:\n{output}"
        )


class GitDiffTool(WorkspaceTool):
    name = "git_diff"
    description = "Show the workspace diff using git, for summarizing changed lines."
    inputs = {
        "paths": {"type": "string", "description": "Optional path to restrict the diff to.", "nullable": True},
        "staged": {"type": "boolean", "description": "Show staged changes instead of the worktree.", "nullable": True},
    }
    output_type = "string"

    def forward(self, paths: str | None = None, staged: bool | None = None) -> str:
        command = ["git", "diff"]
        if staged:
            command.append("--staged")
        if paths:
            command.extend(["--", self.normalize_path(paths)])
        result = self.sandbox.run_command(command)
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if result.returncode != 0 and not output:
            return f"git diff failed with exit code {result.returncode}."
        return output or "No changes."


def create_coding_tools(
    sandbox: Sandbox,
    *,
    allowed_write_paths: Iterable[str] | None = None,
    denied_write_paths: Iterable[str] = (),
    test_command: Sequence[str] | None = None,
    include_git: bool = True,
) -> list[Tool]:
    """Create the workspace-scoped coding toolset.

    Args:
        sandbox ([`Sandbox`]): Sandbox configured with a real `workspace_root`.
        allowed_write_paths (`Iterable[str]`, *optional*): fnmatch allowlist for writes; `None` allows all non-denied.
        denied_write_paths (`Iterable[str]`, *optional*): fnmatch denylist for writes. Explicit allows win.
        test_command (`Sequence[str]`, *optional*): Fixed test command exposed through `run_test`.
        include_git (`bool`): Whether to expose `git_diff`.
    """
    tools: list[Tool] = [
        ListFilesTool(sandbox),
        ReadFileTool(sandbox),
        SearchCodeTool(sandbox),
        WriteFileTool(sandbox, allowed_paths=allowed_write_paths, denied_paths=denied_write_paths),
        EditFileTool(sandbox, allowed_paths=allowed_write_paths, denied_paths=denied_write_paths),
    ]
    if test_command:
        tools.append(RunTestTool(sandbox, test_command))
    if include_git:
        tools.append(GitDiffTool(sandbox))
    return tools


__all__ = [
    "CODING_WRITE_TOOLS",
    "EditFileTool",
    "GitDiffTool",
    "ListFilesTool",
    "ReadFileTool",
    "RunTestTool",
    "SearchCodeTool",
    "WorkspaceTool",
    "WriteFileTool",
    "create_coding_tools",
    "default_coding_risk_policy",
]
