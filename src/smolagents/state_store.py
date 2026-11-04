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
"""Execution state persistence for [`ToolCallingAgent`] runs.

[`StateStore`] writes a single JSON snapshot containing the task information, run status and exit code, planning
steps, per-step completion evidence, workspace information, and the full list of messages exchanged with the model.
The snapshot is overwritten atomically after every recorded step so it can be inspected while a run is in progress.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

from .memory import ActionStep, PlanningStep
from .models import ChatMessage
from .utils import AgentError, make_json_serializable


SCHEMA_VERSION = 1


class ExecutionStatus(str, Enum):
    """Lifecycle status of an agent run."""

    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class ExitCode(IntEnum):
    """Process-style exit codes associated with a finished run."""

    SUCCESS = 0
    FAILED = 1
    MAX_STEPS = 2
    BUDGET_EXCEEDED = 3
    INTERRUPTED = 130


class StopReason(str, Enum):
    """Explicit reason why a run stopped."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    COMPLETION_FAILED = "completion_failed"


STATUS_EXIT_CODES: dict[ExecutionStatus, int | None] = {
    ExecutionStatus.RUNNING: None,
    ExecutionStatus.COMPLETED: int(ExitCode.SUCCESS),
    ExecutionStatus.MAX_STEPS: int(ExitCode.MAX_STEPS),
    ExecutionStatus.BUDGET_EXCEEDED: int(ExitCode.BUDGET_EXCEEDED),
    ExecutionStatus.INTERRUPTED: int(ExitCode.INTERRUPTED),
    ExecutionStatus.FAILED: int(ExitCode.FAILED),
}

STATUS_STOP_REASONS: dict[ExecutionStatus, StopReason] = {
    ExecutionStatus.COMPLETED: StopReason.COMPLETED,
    ExecutionStatus.MAX_STEPS: StopReason.MAX_STEPS,
    ExecutionStatus.BUDGET_EXCEEDED: StopReason.BUDGET_EXCEEDED,
    ExecutionStatus.INTERRUPTED: StopReason.INTERRUPTED,
    ExecutionStatus.FAILED: StopReason.FAILED,
}


class StateStoreError(Exception):
    """Raised when execution state cannot be initialized, read, or written."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exit_code_for(status: ExecutionStatus | str) -> int | None:
    """Return the exit code associated with `status`."""
    try:
        status = ExecutionStatus(status)
    except ValueError as e:
        raise StateStoreError(f"Unknown execution status: {status}") from e
    return STATUS_EXIT_CODES[status]


def stop_reason_for(status: ExecutionStatus | str) -> str:
    """Return the default stop reason associated with `status`."""
    try:
        status = ExecutionStatus(status)
    except ValueError as e:
        raise StateStoreError(f"Unknown execution status: {status}") from e
    return STATUS_STOP_REASONS[status].value


def _error_to_dict(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    if isinstance(error, AgentError):
        return error.dict()
    return {"type": type(error).__name__, "message": str(error)}


def serialize_messages(messages: list[ChatMessage] | None) -> list[dict[str, Any]]:
    """Serialize chat messages into JSON-safe dictionaries."""
    return [make_json_serializable(message.dict()) for message in (messages or [])]


@dataclass
class TaskInfo:
    """Information required to describe the task being executed."""

    task: str
    agent_class: str
    max_steps: int
    agent_name: str | None = None
    model_id: str | None = None
    planning_interval: int | None = None
    additional_args: Any = None

    def dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "agent_class": self.agent_class,
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
            "planning_interval": self.planning_interval,
            "additional_args": make_json_serializable(self.additional_args),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInfo":
        return cls(
            task=data["task"],
            agent_class=data["agent_class"],
            max_steps=data["max_steps"],
            agent_name=data.get("agent_name"),
            model_id=data.get("model_id"),
            planning_interval=data.get("planning_interval"),
            additional_args=data.get("additional_args"),
        )


@dataclass
class PlanRecord:
    """A plan produced during a planning step."""

    step_number: int
    plan: str
    recorded_at: str
    timing: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None

    def dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "plan": self.plan,
            "recorded_at": self.recorded_at,
            "timing": make_json_serializable(self.timing),
            "token_usage": make_json_serializable(self.token_usage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanRecord":
        return cls(
            step_number=data["step_number"],
            plan=data["plan"],
            recorded_at=data["recorded_at"],
            timing=data.get("timing"),
            token_usage=data.get("token_usage"),
        )


@dataclass
class StepEvidence:
    """Evidence that an action step completed, including its tool calls and observations."""

    step_number: int
    completed: bool
    recorded_at: str
    model_output: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: str | None = None
    action_output: Any = None
    is_final_answer: bool = False
    error: dict[str, str] | None = None
    timing: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None

    def dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "completed": self.completed,
            "recorded_at": self.recorded_at,
            "model_output": make_json_serializable(self.model_output),
            "tool_calls": make_json_serializable(self.tool_calls),
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
            "is_final_answer": self.is_final_answer,
            "error": self.error,
            "timing": make_json_serializable(self.timing),
            "token_usage": make_json_serializable(self.token_usage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepEvidence":
        return cls(
            step_number=data["step_number"],
            completed=data["completed"],
            recorded_at=data["recorded_at"],
            model_output=data.get("model_output"),
            tool_calls=data.get("tool_calls") or [],
            observations=data.get("observations"),
            action_output=data.get("action_output"),
            is_final_answer=data.get("is_final_answer", False),
            error=data.get("error"),
            timing=data.get("timing"),
            token_usage=data.get("token_usage"),
        )


@dataclass
class WorkspaceInfo:
    """Information about the environment in which the agent runs."""

    cwd: str
    platform: dict[str, str]
    python: dict[str, str]
    git: dict[str, str] | None = None
    workspace_dir: str | None = None
    files: list[dict[str, Any]] | None = None

    def dict(self) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "platform": dict(self.platform),
            "python": dict(self.python),
            "git": dict(self.git) if self.git is not None else None,
            "workspace_dir": self.workspace_dir,
            "files": [dict(file) for file in self.files] if self.files is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceInfo":
        return cls(
            cwd=data["cwd"],
            platform=data.get("platform") or {},
            python=data.get("python") or {},
            git=data.get("git"),
            workspace_dir=data.get("workspace_dir"),
            files=data.get("files"),
        )


@dataclass
class ExecutionState:
    """Complete persisted state of an agent run."""

    task: TaskInfo
    workspace: WorkspaceInfo
    status: str = ExecutionStatus.RUNNING.value
    exit_code: int | None = None
    run_id: str | None = None
    stop_reason: str | None = None
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    error: dict[str, str] | None = None
    final_output: Any = None
    budget: dict[str, Any] | None = None
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    plans: list[PlanRecord] = field(default_factory=list)
    evidence: list[StepEvidence] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "task": self.task.dict(),
            "status": self.status,
            "exit_code": self.exit_code,
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "final_output": self.final_output,
            "budget": self.budget,
            "pending_approvals": self.pending_approvals,
            "plans": [plan.dict() for plan in self.plans],
            "evidence": [item.dict() for item in self.evidence],
            "workspace": self.workspace.dict(),
            "messages": self.messages,
        }
        return make_json_serializable(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionState":
        return cls(
            task=TaskInfo.from_dict(data["task"]),
            workspace=WorkspaceInfo.from_dict(data["workspace"]),
            status=data.get("status", ExecutionStatus.RUNNING.value),
            exit_code=data.get("exit_code"),
            run_id=data.get("run_id"),
            stop_reason=data.get("stop_reason"),
            started_at=data.get("started_at", _utc_now()),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
            final_output=data.get("final_output"),
            budget=data.get("budget"),
            pending_approvals=data.get("pending_approvals") or [],
            plans=[PlanRecord.from_dict(item) for item in data.get("plans", [])],
            evidence=[StepEvidence.from_dict(item) for item in data.get("evidence", [])],
            messages=data.get("messages", []),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


class StateStore:
    """Persist an [`ExecutionState`] snapshot to a JSON file.

    Args:
        path (`str` or `Path`):
            Destination JSON file. Parent directories are created automatically.
        workspace_dir (`str` or `Path`, *optional*):
            Directory whose files should be inventoried as workspace information. When omitted, only runtime
            environment information is captured and no directory is scanned.
        max_workspace_files (`int`, *optional*, defaults to 1000):
            Maximum number of files recorded from `workspace_dir`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        workspace_dir: str | Path | None = None,
        max_workspace_files: int = 1000,
    ):
        self.path = Path(path)
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self.max_workspace_files = max_workspace_files
        self._state: ExecutionState | None = None

    @property
    def state(self) -> ExecutionState:
        """The current execution state, or raise if `start()` has not been called."""
        if self._state is None:
            raise StateStoreError("Execution state has not been initialized: call start() first.")
        return self._state

    @property
    def is_started(self) -> bool:
        """Whether a run has already initialized this store."""
        return self._state is not None

    def start(
        self,
        *,
        task: str,
        agent_class: str,
        max_steps: int,
        agent_name: str | None = None,
        model_id: str | None = None,
        planning_interval: int | None = None,
        additional_args: Any = None,
        messages: list[ChatMessage] | None = None,
        run_id: str | None = None,
        budget: dict[str, Any] | None = None,
        pending_approvals: list[dict[str, Any]] | None = None,
    ) -> ExecutionState:
        """Initialize a running execution state and write the first snapshot."""
        self._state = ExecutionState(
            task=TaskInfo(
                task=task,
                agent_class=agent_class,
                max_steps=max_steps,
                agent_name=agent_name,
                model_id=model_id,
                planning_interval=planning_interval,
                additional_args=additional_args,
            ),
            workspace=collect_workspace_info(self.workspace_dir, max_files=self.max_workspace_files),
            run_id=run_id,
            budget=make_json_serializable(budget) if budget is not None else None,
            pending_approvals=[make_json_serializable(item) for item in (pending_approvals or [])],
            messages=serialize_messages(messages),
        )
        self.save()
        return self._state

    def record_budget(self, budget: dict[str, Any]) -> None:
        """Persist the current execution budget and write a snapshot."""
        if not isinstance(budget, dict):
            raise StateStoreError("'budget' must be a dictionary.")
        self.state.budget = make_json_serializable(budget)
        self.save()

    def record_pending_approvals(self, records: list[dict[str, Any]]) -> None:
        """Persist the pending approval records and write a snapshot."""
        if not isinstance(records, list):
            raise StateStoreError("'records' must be a list.")
        self.state.pending_approvals = [make_json_serializable(record) for record in records]
        self.save()

    def record_plan(
        self,
        *,
        step_number: int,
        planning_step: PlanningStep,
        messages: list[ChatMessage] | None = None,
    ) -> None:
        """Append a planning step record and write a snapshot."""
        self.state.plans.append(
            PlanRecord(
                step_number=step_number,
                plan=planning_step.plan,
                recorded_at=_utc_now(),
                timing=planning_step.timing.dict() if planning_step.timing else None,
                token_usage=asdict(planning_step.token_usage) if planning_step.token_usage else None,
            )
        )
        if messages is not None:
            self.state.messages = serialize_messages(messages)
        self.save()

    def record_action(
        self,
        *,
        action_step: ActionStep,
        messages: list[ChatMessage] | None = None,
    ) -> None:
        """Append action-step completion evidence and write a snapshot."""
        self.state.evidence.append(
            StepEvidence(
                step_number=action_step.step_number,
                completed=action_step.error is None,
                recorded_at=_utc_now(),
                model_output=make_json_serializable(action_step.model_output),
                tool_calls=[tool_call.dict() for tool_call in action_step.tool_calls]
                if action_step.tool_calls
                else [],
                observations=action_step.observations,
                action_output=make_json_serializable(action_step.action_output),
                is_final_answer=action_step.is_final_answer,
                error=_error_to_dict(action_step.error),
                timing=action_step.timing.dict() if action_step.timing else None,
                token_usage=asdict(action_step.token_usage) if action_step.token_usage else None,
            )
        )
        if messages is not None:
            self.state.messages = serialize_messages(messages)
        self.save()

    def finish(
        self,
        *,
        status: ExecutionStatus | str,
        output: Any = None,
        error: BaseException | None = None,
        messages: list[ChatMessage] | None = None,
        stop_reason: StopReason | str | None = None,
        budget: dict[str, Any] | None = None,
        pending_approvals: list[dict[str, Any]] | None = None,
    ) -> None:
        """Mark the run as finished, set exit code and stop reason, and write a snapshot."""
        status = ExecutionStatus(status)
        self.state.status = status.value
        self.state.exit_code = exit_code_for(status)
        self.state.stop_reason = stop_reason_for(status) if stop_reason is None else StopReason(stop_reason).value
        self.state.finished_at = _utc_now()
        self.state.final_output = make_json_serializable(output)
        self.state.error = _error_to_dict(error)
        if budget is not None:
            self.state.budget = make_json_serializable(budget)
        if pending_approvals is not None:
            self.state.pending_approvals = [make_json_serializable(record) for record in pending_approvals]
        if messages is not None:
            self.state.messages = serialize_messages(messages)
        self.save()

    def save(self) -> None:
        """Atomically write the current state to `path`."""
        payload = self.state.dict()
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StateStoreError(f"Could not create state directory '{parent}': {e}") from e

        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=f".{self.path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.path)
            tmp_path = None
        except (OSError, TypeError, ValueError) as e:
            raise StateStoreError(f"Could not write execution state to '{self.path}': {e}") from e
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> ExecutionState:
        """Load and validate an execution state snapshot from `path`."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise StateStoreError(f"Could not read execution state from '{path}': {e}") from e

        if not isinstance(data, dict):
            raise StateStoreError(f"Execution state in '{path}' must be a JSON object.")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise StateStoreError(
                f"Unsupported execution state schema version {data.get('schema_version')!r} "
                f"in '{path}', expected {SCHEMA_VERSION}."
            )
        try:
            return ExecutionState.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise StateStoreError(f"Malformed execution state in '{path}': {e}") from e


def _run_git(args: list[str], cwd: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _collect_git_info(cwd: str) -> dict[str, str] | None:
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    if branch is None and commit is None:
        return None
    return {"branch": branch, "commit": commit}


def _collect_files(workspace_dir: Path, max_files: int) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not workspace_dir.is_dir():
        return files
    try:
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            files.append(
                {
                    "relative_path": path.relative_to(workspace_dir).as_posix(),
                    "size_bytes": size_bytes,
                }
            )
            if len(files) >= max_files:
                break
    except OSError:
        pass
    return files


def collect_workspace_info(
    workspace_dir: str | Path | None = None,
    max_files: int = 1000,
) -> WorkspaceInfo:
    """Capture runtime environment information and, optionally, a directory file inventory."""
    cwd = str(Path.cwd())
    resolved_workspace_dir: str | None = None
    files: list[dict[str, Any]] | None = None

    if workspace_dir is not None:
        resolved = Path(workspace_dir).expanduser().resolve()
        resolved_workspace_dir = str(resolved)
        files = _collect_files(resolved, max_files)

    return WorkspaceInfo(
        cwd=cwd,
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        python={
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        git=_collect_git_info(cwd),
        workspace_dir=resolved_workspace_dir,
        files=files,
    )


__all__ = [
    "ExecutionState",
    "ExecutionStatus",
    "ExitCode",
    "PlanRecord",
    "StateStore",
    "StateStoreError",
    "StepEvidence",
    "StopReason",
    "TaskInfo",
    "WorkspaceInfo",
    "collect_workspace_info",
    "exit_code_for",
    "serialize_messages",
    "stop_reason_for",
]
