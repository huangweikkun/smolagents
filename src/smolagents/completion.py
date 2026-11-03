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
"""Post-run completion verification for business tasks.

[`CompletionVerifier`] maps a task name to a [`TaskCompletionValidator`]. Once an agent has finished, the validator
inspects the workspace and reports whether the task was actually completed, as a [`CompletionReport`].

Two validators are provided:

- [`LoginFlowDocumentValidator`] (`understand`): the agent must write `artifacts/login-flow.md` whose conclusions
  carry file/line level evidence and whose "不能照搬到生产" facts are listed separately, without writing anything
  else.
- [`SessionBoundaryFixValidator`] (`fix_tests`): the project test command must exit with code 0, the fixed source
  file must be present, and the tests that already existed must stay untouched outside `tests/generated/`.

Both validators are purely local: they read the workspace, hash files, and optionally run the test command.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_DIRS = frozenset({".git", ".venv", "__pycache__", "build", "dist", "node_modules"})

LINE_REFERENCE_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:js|jsx|md|py|ts|tsx)):(?P<start>\d+)(?:-(?P<end>\d+))?"
)

LOGIN_FLOW_REQUIRED_REFERENCES = (
    "src/routes/login.ts",
    "src/services/auth-service.ts",
    "src/repositories/user-repository.ts",
    "src/db/database.ts",
    "src/middleware/auth-middleware.ts",
)

DEFAULT_FIX_TEST_COMMAND = ("npm", "test", "--")
DEFAULT_FIX_DELIVERABLE = "src/auth/session-policy.ts"
DEFAULT_PROTECTED_TEST_DIR = "tests"
DEFAULT_GENERATED_TEST_DIR = "tests/generated"

DETAIL_OUTPUT_LIMIT = 400


class CompletionError(Exception):
    """Base class for completion verification errors."""


class UnknownCompletionTaskError(CompletionError):
    """Raised when a task name has no registered completion validator."""


@dataclass(frozen=True)
class CompletionCheck:
    """Outcome of a single completion condition.

    Args:
        name (`str`): Stable identifier of the condition.
        passed (`bool`): Whether the condition holds.
        detail (`str`): Human-readable explanation, including the observed evidence.
        skipped (`bool`): Whether the condition could not be evaluated, e.g. because no baseline was provided.
    """

    name: str
    passed: bool
    detail: str
    skipped: bool = False


@dataclass(frozen=True)
class CommandResult:
    """Normalized result of a test command run inside the workspace."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


CommandRunner = Callable[[Sequence[str], Path], CommandResult]


@dataclass(frozen=True)
class CompletionReport:
    """Aggregated result of every completion condition checked for one task."""

    task: str
    checks: tuple[CompletionCheck, ...]

    @property
    def passed(self) -> bool:
        """Whether the task is completed, i.e. every evaluated condition passed."""
        return all(check.passed for check in self.checks if not check.skipped)

    @property
    def failed_checks(self) -> tuple[CompletionCheck, ...]:
        """Conditions that were evaluated and did not hold."""
        return tuple(check for check in self.checks if not check.passed and not check.skipped)

    def summary(self) -> str:
        """Render a compact multi-line summary suitable for logs or agent output."""
        lines = [f"task={self.task} completed={self.passed}"]
        for check in self.checks:
            status = "skip" if check.skipped else ("pass" if check.passed else "fail")
            lines.append(f"  [{status}] {check.name}: {check.detail}")
        return "\n".join(lines)


def snapshot_files(
    root: str | Path,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> dict[str, str]:
    """Hash every regular file under `root`, keyed by its POSIX path relative to `root`.

    Used to build a baseline before the agent runs and to detect writes made by the agent afterwards.

    Args:
        root (`str` or `Path`): Directory to walk.
        excluded_dirs (`Iterable[str]`, *optional*): Directory names skipped at any depth.

    Returns:
        `dict[str, str]`: Relative path to SHA-256 digest.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise CompletionError(f"Workspace {root} is not a directory.")
    excluded = frozenset(excluded_dirs)
    snapshot: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.is_symlink() or not path.is_file():
                continue
            snapshot[path.relative_to(root).as_posix()] = _hash_file(path)
    return snapshot


def create_default_completion_verifier() -> "CompletionVerifier":
    """Build a verifier wired with the supported business task validators."""
    return CompletionVerifier([LoginFlowDocumentValidator(), SessionBoundaryFixValidator()])


class TaskCompletionValidator(ABC):
    """Base class for task-specific completion checks.

    Subclasses implement [`run_checks`] and are evaluated by [`validate`], which resolves the workspace, takes a
    file snapshot and wraps every condition into a [`CompletionReport`].

    Args:
        excluded_dirs (`tuple[str, ...]`, *optional*): Directory names ignored while snapshotting the workspace.
    """

    task_name: str = ""
    excluded_dirs: tuple[str, ...] = tuple(sorted(DEFAULT_EXCLUDED_DIRS))

    def validate(
        self,
        workspace: str | Path,
        baseline: Mapping[str, str] | None = None,
    ) -> CompletionReport:
        """Check whether the task was completed in `workspace`.

        Args:
            workspace (`str` or `Path`): Root of the repository the agent worked in.
            baseline (`Mapping[str, str]`, *optional*): Snapshot taken before the agent ran, as returned by
                [`snapshot_files`]. When omitted, write-scope checks are skipped.

        Returns:
            `CompletionReport`: Per-condition outcomes and the aggregated verdict.
        """
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise CompletionError(f"Workspace {root} is not a directory.")
        current = snapshot_files(root, self.excluded_dirs)
        checks = tuple(self.run_checks(root, current, dict(baseline) if baseline is not None else None))
        return CompletionReport(task=self.task_name, checks=checks)

    @abstractmethod
    def run_checks(
        self,
        workspace: Path,
        current: dict[str, str],
        baseline: dict[str, str] | None,
    ) -> Iterable[CompletionCheck]:
        """Return the conditions evaluated for this task."""


class LoginFlowDocumentValidator(TaskCompletionValidator):
    """Validator for the `understand` task: a login call-chain document with cited evidence.

    Conditions:

    1. `deliverable_present`: the deliverable exists and is not empty.
    2. `references_valid`: every `path:line` citation resolves to an existing file and line range, and each
       required pipeline stage is cited.
    3. `production_notes`: a dedicated "不能照搬到生产" section exists.
    4. `write_scope`: with a baseline, nothing but the deliverable was written.

    Args:
        deliverable (`str`, *optional*): Repository-relative path of the expected document.
        required_references (`Iterable[str]`, *optional*): Source files that must be cited with line numbers.
        production_section_keyword (`str`, *optional*): Keyword that marks the "do not ship to production" section.
    """

    task_name = "understand"

    def __init__(
        self,
        deliverable: str = "artifacts/login-flow.md",
        required_references: Iterable[str] = LOGIN_FLOW_REQUIRED_REFERENCES,
        production_section_keyword: str = "不能照搬到生产",
    ):
        self.deliverable = Path(deliverable).as_posix()
        self.required_references = tuple(required_references)
        self.production_section_keyword = production_section_keyword

    def run_checks(
        self,
        workspace: Path,
        current: dict[str, str],
        baseline: dict[str, str] | None,
    ) -> tuple[CompletionCheck, ...]:
        document = workspace / self.deliverable
        text = document.read_text(encoding="utf-8", errors="replace") if document.is_file() else None
        checks = [
            self._check_deliverable(text),
            self._check_references(workspace, text),
            self._check_production_notes(text),
            self._check_write_scope(current, baseline),
        ]
        return tuple(checks)

    def _check_deliverable(self, text: str | None) -> CompletionCheck:
        if text is None:
            return CompletionCheck("deliverable_present", False, f"{self.deliverable} is missing")
        if not text.strip():
            return CompletionCheck("deliverable_present", False, f"{self.deliverable} is empty")
        return CompletionCheck("deliverable_present", True, f"{self.deliverable} exists and is non-empty")

    def _check_references(self, workspace: Path, text: str | None) -> CompletionCheck:
        if text is None:
            return CompletionCheck("references_valid", False, f"{self.deliverable} is missing, no evidence found")
        scan = _scan_line_references(text, workspace)
        missing = [path for path in self.required_references if path not in scan.by_path]
        problems = list(scan.problems)
        if missing:
            problems.insert(0, f"no line-level citation for: {', '.join(missing)}")
        if problems:
            return CompletionCheck("references_valid", False, _format_problems(problems))
        cited = sum(len(ranges) for ranges in scan.by_path.values())
        return CompletionCheck(
            "references_valid", True, f"{cited} citations across {len(scan.by_path)} files all resolve"
        )

    def _check_production_notes(self, text: str | None) -> CompletionCheck:
        if text is None:
            return CompletionCheck("production_notes", False, f"{self.deliverable} is missing")
        if self.production_section_keyword not in text:
            return CompletionCheck(
                "production_notes", False, f"no section mentioning '{self.production_section_keyword}'"
            )
        return CompletionCheck("production_notes", True, f"found '{self.production_section_keyword}'")

    def _check_write_scope(self, current: dict[str, str], baseline: dict[str, str] | None) -> CompletionCheck:
        if baseline is None:
            return CompletionCheck("write_scope", True, "no baseline provided; skipped", skipped=True)
        changed, added = _touched_paths(current, baseline)
        unexpected = sorted((changed | added) - {self.deliverable})
        if unexpected:
            return CompletionCheck("write_scope", False, _format_paths("unexpected writes", unexpected))
        if self.deliverable in (changed | added):
            return CompletionCheck("write_scope", True, f"only {self.deliverable} was written")
        return CompletionCheck("write_scope", True, f"no writes outside {self.deliverable}")


class SessionBoundaryFixValidator(TaskCompletionValidator):
    """Validator for the `fix_tests` task: tests green, source fixed, existing tests untouched.

    Conditions:

    1. `tests_pass`: the configured test command exits with code 0.
    2. `deliverable_present`: the fixed source file exists and is not empty.
    3. `deliverable_changed`: with a baseline, the fixed source file content differs from it.
    4. `existing_tests_untouched`: with a baseline, nothing under the protected test directory changed, except
       files under the generated test directory.

    Args:
        test_command (`Sequence[str]`, *optional*): Command run inside the workspace to validate the fix.
        deliverable (`str`, *optional*): Repository-relative path of the file that must be fixed.
        protected_tests_dir (`str`, *optional*): Directory holding pre-existing tests, relative to the workspace.
        generated_tests_dir (`str`, *optional*): Directory where the agent may add new tests.
        runner (`Callable`, *optional*): Inject a command runner for tests or non-`npm` projects.
        timeout_seconds (`float`, *optional*): Timeout applied to the default command runner.
    """

    task_name = "fix_tests"

    def __init__(
        self,
        test_command: Sequence[str] = DEFAULT_FIX_TEST_COMMAND,
        deliverable: str = DEFAULT_FIX_DELIVERABLE,
        protected_tests_dir: str = DEFAULT_PROTECTED_TEST_DIR,
        generated_tests_dir: str = DEFAULT_GENERATED_TEST_DIR,
        runner: CommandRunner | None = None,
        timeout_seconds: float = 300.0,
    ):
        self.test_command = tuple(test_command)
        self.deliverable = Path(deliverable).as_posix()
        self.protected_tests_dir = Path(protected_tests_dir).as_posix().rstrip("/")
        self.generated_tests_dir = Path(generated_tests_dir).as_posix().rstrip("/")
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def run_checks(
        self,
        workspace: Path,
        current: dict[str, str],
        baseline: dict[str, str] | None,
    ) -> tuple[CompletionCheck, ...]:
        document = workspace / self.deliverable
        checks = [
            self._check_tests_pass(workspace),
            self._check_deliverable(document),
            self._check_deliverable_changed(document, current, baseline),
            self._check_existing_tests(current, baseline),
        ]
        return tuple(checks)

    def _check_tests_pass(self, workspace: Path) -> CompletionCheck:
        result = self.run_test_command(workspace)
        command = " ".join(self.test_command)
        if result.timed_out:
            return CompletionCheck("tests_pass", False, f"`{command}` timed out after {self.timeout_seconds:.0f}s")
        if result.returncode != 0:
            output = _truncate("\n".join(part for part in (result.stdout, result.stderr) if part.strip()))
            return CompletionCheck(
                "tests_pass",
                False,
                f"`{command}` exited with code {result.returncode}" + (f"; output: {output}" if output else ""),
            )
        return CompletionCheck("tests_pass", True, f"`{command}` exited with code 0")

    def run_test_command(self, workspace: Path) -> CommandResult:
        """Run the configured test command, delegating to `runner` when one was injected."""
        if self.runner is not None:
            return self.runner(self.test_command, workspace)
        try:
            completed = subprocess.run(
                list(self.test_command),
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return CommandResult(
                returncode=-1,
                stdout=_to_text(error.stdout),
                stderr=_to_text(error.stderr),
                timed_out=True,
            )
        except OSError as error:
            return CommandResult(returncode=-1, stderr=str(error))
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def _check_deliverable(self, document: Path) -> CompletionCheck:
        if not document.is_file():
            return CompletionCheck("deliverable_present", False, f"{self.deliverable} is missing")
        if not document.read_text(encoding="utf-8", errors="replace").strip():
            return CompletionCheck("deliverable_present", False, f"{self.deliverable} is empty")
        return CompletionCheck("deliverable_present", True, f"{self.deliverable} exists and is non-empty")

    def _check_deliverable_changed(
        self,
        document: Path,
        current: dict[str, str],
        baseline: dict[str, str] | None,
    ) -> CompletionCheck:
        if baseline is None:
            return CompletionCheck("deliverable_changed", True, "no baseline provided; skipped", skipped=True)
        if not document.is_file():
            return CompletionCheck("deliverable_changed", False, f"{self.deliverable} is missing")
        if current.get(self.deliverable) == baseline.get(self.deliverable):
            return CompletionCheck("deliverable_changed", False, f"{self.deliverable} was not modified")
        return CompletionCheck("deliverable_changed", True, f"{self.deliverable} was modified")

    def _check_existing_tests(self, current: dict[str, str], baseline: dict[str, str] | None) -> CompletionCheck:
        if baseline is None:
            return CompletionCheck("existing_tests_untouched", True, "no baseline provided; skipped", skipped=True)
        touched = {path for path in baseline if self._is_protected_test(path) and current.get(path) != baseline[path]}
        added = {path for path in current if path not in baseline and self._is_protected_test(path)}
        offenders = sorted(touched | added)
        if offenders:
            return CompletionCheck(
                "existing_tests_untouched",
                False,
                _format_paths("existing tests were added, changed or deleted", offenders),
            )
        return CompletionCheck(
            "existing_tests_untouched",
            True,
            f"nothing under {self.protected_tests_dir}/ outside {self.generated_tests_dir}/ changed",
        )

    def _is_protected_test(self, path: str) -> bool:
        if not path.startswith(f"{self.protected_tests_dir}/"):
            return False
        return not path.startswith(f"{self.generated_tests_dir}/")


class CompletionVerifier:
    """Registry that runs the validator matching a task name.

    Args:
        validators (`Iterable[TaskCompletionValidator]`, *optional*): Validators registered on construction.
    """

    def __init__(self, validators: Iterable[TaskCompletionValidator] = ()):
        self._validators: dict[str, TaskCompletionValidator] = {}
        for validator in validators:
            self.register(validator)

    def register(self, validator: TaskCompletionValidator) -> None:
        """Register `validator` under its `task_name`, replacing any previous one."""
        if not validator.task_name:
            raise CompletionError("A completion validator must declare a non-empty task_name.")
        self._validators[validator.task_name] = validator

    @property
    def task_names(self) -> tuple[str, ...]:
        """Registered task names, sorted."""
        return tuple(sorted(self._validators))

    def verify(
        self,
        task_name: str,
        workspace: str | Path,
        baseline: Mapping[str, str] | None = None,
    ) -> CompletionReport:
        """Run the validator registered for `task_name` against `workspace`.

        Raises:
            UnknownCompletionTaskError: If no validator is registered for `task_name`.
        """
        if task_name not in self._validators:
            raise UnknownCompletionTaskError(
                f"Unknown completion task {task_name!r}, expected one of: {', '.join(self.task_names) or '<none>'}"
            )
        return self._validators[task_name].validate(workspace, baseline)


@dataclass
class _ReferenceScan:
    """Line-level citations found in a document, plus the ones that do not resolve."""

    by_path: dict[str, list[tuple[int, int]]]
    problems: list[str]


def _scan_line_references(text: str, workspace: Path) -> _ReferenceScan:
    scan = _ReferenceScan(by_path={}, problems=[])
    line_counts: dict[str, int | None] = {}
    for match in LINE_REFERENCE_PATTERN.finditer(text):
        raw_path = match.group("path")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        resolved = Path(raw_path).resolve() if Path(raw_path).is_absolute() else (workspace / raw_path).resolve()
        try:
            relative = resolved.relative_to(workspace).as_posix()
        except ValueError:
            scan.problems.append(f"{raw_path}:{start} points outside the workspace")
            continue
        if relative not in line_counts:
            line_counts[relative] = _count_lines(resolved)
        line_count = line_counts[relative]
        if line_count is None:
            scan.problems.append(f"{raw_path}:{start} refers to a missing file")
            continue
        if start < 1 or end < start or end > line_count:
            scan.problems.append(f"{raw_path}:{start}-{end} is outside the {line_count} lines of the file")
            continue
        scan.by_path.setdefault(relative, []).append((start, end))
    return scan


def _count_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _touched_paths(current: dict[str, str], baseline: dict[str, str]) -> tuple[set[str], set[str]]:
    changed = {path for path, digest in baseline.items() if current.get(path) != digest}
    added = set(current) - set(baseline)
    return changed, added


def _format_paths(prefix: str, paths: Iterable[str], limit: int = 5) -> str:
    paths = list(paths)
    shown = ", ".join(paths[:limit])
    suffix = f" (+{len(paths) - limit} more)" if len(paths) > limit else ""
    return f"{prefix}: {shown}{suffix}"


def _format_problems(problems: Iterable[str], limit: int = 5) -> str:
    problems = list(problems)
    suffix = f" (+{len(problems) - limit} more)" if len(problems) > limit else ""
    return "; ".join(problems[:limit]) + suffix


def _truncate(text: str, limit: int = DETAIL_OUTPUT_LIMIT) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _to_text(stream: Any) -> str:
    if stream is None:
        return ""
    return stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else str(stream)


__all__ = [
    "CommandResult",
    "CompletionCheck",
    "CompletionError",
    "CompletionReport",
    "CompletionVerifier",
    "LoginFlowDocumentValidator",
    "SessionBoundaryFixValidator",
    "TaskCompletionValidator",
    "UnknownCompletionTaskError",
    "create_default_completion_verifier",
    "snapshot_files",
]
