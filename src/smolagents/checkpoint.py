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
"""Checkpoint and resume support for [`ToolCallingAgent`] runs.

Checkpoints contain the full agent memory, state variables, budget counters, plans, step evidence, pending approvals,
and an in-flight marker. They are written atomically to a checkpoint directory and can be restored after a process
interruption.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .memory import ActionStep, FinalAnswerStep, PlanningStep, TaskStep, ToolCall
from .models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
from .monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from .serialization import SafeSerializer, SerializationError
from .state_store import ExecutionStatus
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    make_json_serializable,
)


SCHEMA_VERSION = 1
UNKNOWN_RESULT = "unknown_result"


class CheckpointError(Exception):
    """Base class for checkpoint errors."""


class CheckpointPersistenceError(CheckpointError):
    """Raised when checkpoint files cannot be read or written."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint is corrupted or fails integrity validation."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when no checkpoint can be found."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the current agent configuration."""


class CheckpointValidationError(CheckpointError):
    """Raised when resume parameters are invalid."""


class CheckpointRestoreError(CheckpointError):
    """Raised when checkpoint contents cannot be deserialized."""


class CheckpointInFlightError(CheckpointError):
    """Raised when an in-flight tool call has an unknown result and the policy is abort."""


class StepResultStatus(str, Enum):
    """Outcome recorded for a completed action step."""

    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN_RESULT = "unknown_result"


class UnknownResultError(AgentToolExecutionError):
    """Error stored when a tool call may have run but its result was never persisted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex


def serialize_value(value: Any) -> Any:
    """Best-effort JSON-safe serialization for arbitrary checkpoint values."""
    try:
        return SafeSerializer.to_json_safe(value)
    except SerializationError:
        return make_json_serializable(value)


def deserialize_value(value: Any) -> Any:
    return SafeSerializer.from_json_safe(value)


def _timing_to_dict(timing: Timing | None) -> dict[str, float] | None:
    if timing is None:
        return None
    return {"start_time": timing.start_time, "end_time": timing.end_time}


def _timing_from_dict(data: dict[str, Any] | None) -> Timing | None:
    if not data:
        return None
    return Timing(start_time=data["start_time"], end_time=data.get("end_time"))


def _token_usage_to_dict(token_usage: TokenUsage | None) -> dict[str, int] | None:
    if token_usage is None:
        return None
    return {"input_tokens": token_usage.input_tokens, "output_tokens": token_usage.output_tokens}


def _token_usage_from_dict(data: dict[str, Any] | None) -> TokenUsage | None:
    if not data:
        return None
    return TokenUsage(input_tokens=data["input_tokens"], output_tokens=data["output_tokens"])


def serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": serialize_value(tool_call.arguments),
    }


def deserialize_tool_call(data: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=deserialize_value(data.get("arguments")),
    )


def serialize_chat_message(message: ChatMessage) -> dict[str, Any]:
    tool_calls = None
    if message.tool_calls:
        tool_calls = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": serialize_value(tool_call.function.arguments),
                    "description": tool_call.function.description,
                },
            }
            for tool_call in message.tool_calls
        ]
    return {
        "role": message.role.value,
        "content": serialize_value(message.content),
        "tool_calls": tool_calls,
        "token_usage": _token_usage_to_dict(message.token_usage),
    }


def deserialize_chat_message(data: dict[str, Any]) -> ChatMessage:
    tool_calls = None
    if data.get("tool_calls"):
        tool_calls = []
        for item in data["tool_calls"]:
            function = ChatMessageToolCallFunction(
                name=item["function"]["name"],
                arguments=deserialize_value(item["function"].get("arguments")),
                description=item["function"].get("description"),
            )
            tool_calls.append(ChatMessageToolCall(function=function, id=item["id"], type=item["type"]))
    return ChatMessage(
        role=MessageRole(data["role"]),
        content=deserialize_value(data.get("content")),
        tool_calls=tool_calls,
        token_usage=_token_usage_from_dict(data.get("token_usage")),
    )


def _serialize_messages(messages: list[ChatMessage] | None) -> list[dict[str, Any]] | None:
    if messages is None:
        return None
    return [serialize_chat_message(message) for message in messages]


def _deserialize_messages(messages: list[dict[str, Any]] | None) -> list[ChatMessage] | None:
    if messages is None:
        return None
    return [deserialize_chat_message(message) for message in messages]


_ERROR_CLASSES = {
    "AgentError": AgentError,
    "AgentParsingError": AgentParsingError,
    "AgentExecutionError": AgentExecutionError,
    "AgentMaxStepsError": AgentMaxStepsError,
    "AgentToolCallError": AgentToolCallError,
    "AgentToolExecutionError": AgentToolExecutionError,
    "AgentGenerationError": AgentGenerationError,
    "UnknownResultError": UnknownResultError,
}

_SILENT_LOGGER = AgentLogger(level=LogLevel.OFF)


def serialize_error(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def deserialize_error(data: dict[str, Any] | None) -> AgentError | None:
    if not data:
        return None
    error_class = _ERROR_CLASSES.get(data.get("type"), AgentError)
    return error_class(data.get("message", ""), _SILENT_LOGGER)


def serialize_memory_step(step: Any) -> dict[str, Any]:
    if isinstance(step, TaskStep):
        return {
            "step_type": "task",
            "task": step.task,
            "task_images": serialize_value(step.task_images),
        }
    if isinstance(step, PlanningStep):
        return {
            "step_type": "planning",
            "model_input_messages": _serialize_messages(step.model_input_messages),
            "model_output_message": serialize_chat_message(step.model_output_message),
            "plan": step.plan,
            "timing": _timing_to_dict(step.timing),
            "token_usage": _token_usage_to_dict(step.token_usage),
        }
    if isinstance(step, ActionStep):
        return {
            "step_type": "action",
            "step_number": step.step_number,
            "timing": _timing_to_dict(step.timing),
            "model_input_messages": _serialize_messages(step.model_input_messages),
            "tool_calls": [serialize_tool_call(item) for item in step.tool_calls] if step.tool_calls else [],
            "error": serialize_error(step.error),
            "model_output_message": serialize_chat_message(step.model_output_message)
            if step.model_output_message
            else None,
            "model_output": serialize_value(step.model_output),
            "code_action": step.code_action,
            "observations": step.observations,
            "observations_images": serialize_value(step.observations_images),
            "action_output": serialize_value(step.action_output),
            "token_usage": _token_usage_to_dict(step.token_usage),
            "is_final_answer": step.is_final_answer,
        }
    if isinstance(step, FinalAnswerStep):
        return {"step_type": "final_answer", "output": serialize_value(step.output)}
    raise CheckpointRestoreError(f"Unsupported memory step type: {type(step).__name__}")


def deserialize_memory_step(data: dict[str, Any]) -> Any:
    step_type = data.get("step_type")
    if step_type == "task":
        return TaskStep(task=data["task"], task_images=deserialize_value(data.get("task_images")))
    if step_type == "planning":
        return PlanningStep(
            model_input_messages=_deserialize_messages(data.get("model_input_messages")) or [],
            model_output_message=deserialize_chat_message(data["model_output_message"]),
            plan=data["plan"],
            timing=_timing_from_dict(data.get("timing")) or Timing(start_time=0.0, end_time=0.0),
            token_usage=_token_usage_from_dict(data.get("token_usage")),
        )
    if step_type == "action":
        return ActionStep(
            step_number=data["step_number"],
            timing=_timing_from_dict(data.get("timing")) or Timing(start_time=0.0, end_time=0.0),
            model_input_messages=_deserialize_messages(data.get("model_input_messages")),
            tool_calls=[deserialize_tool_call(item) for item in data.get("tool_calls") or []],
            error=deserialize_error(data.get("error")),
            model_output_message=deserialize_chat_message(data["model_output_message"])
            if data.get("model_output_message")
            else None,
            model_output=deserialize_value(data.get("model_output")),
            code_action=data.get("code_action"),
            observations=data.get("observations"),
            observations_images=deserialize_value(data.get("observations_images")),
            action_output=deserialize_value(data.get("action_output")),
            token_usage=_token_usage_from_dict(data.get("token_usage")),
            is_final_answer=data.get("is_final_answer", False),
        )
    if step_type == "final_answer":
        return FinalAnswerStep(output=deserialize_value(data.get("output")))
    raise CheckpointRestoreError(f"Unsupported serialized memory step type: {step_type}")


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: serialize_value(value) for key, value in state.items()}


def deserialize_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: deserialize_value(value) for key, value in state.items()}


def build_plans(steps: list[Any]) -> list[dict[str, Any]]:
    plans = []
    for index, step in enumerate(steps):
        if isinstance(step, PlanningStep):
            plans.append(
                {
                    "index": index,
                    "plan": step.plan,
                    "timing": _timing_to_dict(step.timing),
                    "token_usage": _token_usage_to_dict(step.token_usage),
                }
            )
    return plans


def build_evidence(steps: list[Any]) -> list[dict[str, Any]]:
    evidence = []
    for step in steps:
        if not isinstance(step, ActionStep):
            continue
        if isinstance(step.error, UnknownResultError):
            result_status = StepResultStatus.UNKNOWN_RESULT.value
        elif step.error is not None:
            result_status = StepResultStatus.FAILED.value
        else:
            result_status = StepResultStatus.COMPLETED.value
        evidence.append(
            {
                "step_number": step.step_number,
                "result_status": result_status,
                "tool_calls": [serialize_tool_call(item) for item in step.tool_calls] if step.tool_calls else [],
                "observations": step.observations,
                "action_output": serialize_value(step.action_output),
                "error": serialize_error(step.error),
                "is_final_answer": step.is_final_answer,
                "timing": _timing_to_dict(step.timing),
                "token_usage": _token_usage_to_dict(step.token_usage),
            }
        )
    return evidence


@dataclass(frozen=True)
class ExecutionBudget:
    """Budget counters persisted with a checkpoint."""

    max_steps: int
    steps_used: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    started_at: str

    def dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "steps_used": self.steps_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionBudget":
        return cls(
            max_steps=data["max_steps"],
            steps_used=data.get("steps_used", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            started_at=data.get("started_at", utc_now()),
        )


def build_budget(steps: list[Any], *, max_steps: int, started_at: str) -> ExecutionBudget:
    input_tokens = 0
    output_tokens = 0
    steps_used = 0
    for step in steps:
        if isinstance(step, ActionStep):
            steps_used += 1
        if isinstance(step, (ActionStep, PlanningStep)) and step.token_usage is not None:
            input_tokens += step.token_usage.input_tokens
            output_tokens += step.token_usage.output_tokens
    return ExecutionBudget(
        max_steps=max_steps,
        steps_used=steps_used,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        started_at=started_at,
    )


@dataclass(frozen=True)
class CheckpointState:
    """Complete persisted checkpoint for an agent run."""

    run_id: str
    task: str
    agent_class: str
    max_steps: int
    budget: ExecutionBudget
    memory_system_prompt: str
    memory_steps: list[dict[str, Any]]
    state: dict[str, Any]
    checkpoint_id: str = ""
    sequence: int = 0
    created_at: str = ""
    updated_at: str = ""
    task_images: Any = None
    agent_name: str | None = None
    model_id: str | None = None
    tool_names: list[str] = field(default_factory=list)
    managed_agent_names: list[str] = field(default_factory=list)
    planning_interval: int | None = None
    step_number: int = 0
    next_step_number: int = 1
    status: str = ExecutionStatus.RUNNING.value
    exit_code: int | None = None
    plans: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    final_output: Any = None
    error: dict[str, str] | None = None
    in_flight_step: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_id": self.run_id,
            "task": self.task,
            "task_images": self.task_images,
            "agent_class": self.agent_class,
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "tool_names": self.tool_names,
            "managed_agent_names": self.managed_agent_names,
            "max_steps": self.max_steps,
            "planning_interval": self.planning_interval,
            "step_number": self.step_number,
            "next_step_number": self.next_step_number,
            "status": self.status,
            "exit_code": self.exit_code,
            "budget": self.budget.dict(),
            "plans": self.plans,
            "evidence": self.evidence,
            "pending_approvals": self.pending_approvals,
            "memory_system_prompt": self.memory_system_prompt,
            "memory_steps": self.memory_steps,
            "state": self.state,
            "final_output": self.final_output,
            "error": self.error,
            "in_flight_step": self.in_flight_step,
        }
        return make_json_serializable(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        return cls(
            run_id=data["run_id"],
            task=data["task"],
            agent_class=data["agent_class"],
            max_steps=data["max_steps"],
            budget=ExecutionBudget.from_dict(data["budget"]),
            memory_system_prompt=data.get("memory_system_prompt", ""),
            memory_steps=data.get("memory_steps", []),
            state=data.get("state", {}),
            checkpoint_id=data.get("checkpoint_id", ""),
            sequence=data.get("sequence", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            task_images=data.get("task_images"),
            agent_name=data.get("agent_name"),
            model_id=data.get("model_id"),
            tool_names=data.get("tool_names", []),
            managed_agent_names=data.get("managed_agent_names", []),
            planning_interval=data.get("planning_interval"),
            step_number=data.get("step_number", 0),
            next_step_number=data.get("next_step_number", 1),
            status=data.get("status", ExecutionStatus.RUNNING.value),
            exit_code=data.get("exit_code"),
            plans=data.get("plans", []),
            evidence=data.get("evidence", []),
            pending_approvals=data.get("pending_approvals", []),
            final_output=data.get("final_output"),
            error=data.get("error"),
            in_flight_step=data.get("in_flight_step"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def validate_checkpoint_compatibility(
    checkpoint: CheckpointState,
    *,
    agent_class: str,
    model_id: str | None,
    tool_names: Collection[str],
    managed_agent_names: Collection[str],
) -> None:
    """Validate that a checkpoint can be restored into the current agent."""
    if checkpoint.agent_class != agent_class:
        raise CheckpointCompatibilityError(
            f"Checkpoint was created by agent class '{checkpoint.agent_class}', but current agent is '{agent_class}'."
        )
    if checkpoint.model_id and model_id and checkpoint.model_id != model_id:
        raise CheckpointCompatibilityError(
            f"Checkpoint model id '{checkpoint.model_id}' does not match current model id '{model_id}'."
        )
    missing_tools = sorted(set(checkpoint.tool_names) - set(tool_names))
    if missing_tools:
        raise CheckpointCompatibilityError(
            f"Checkpoint references tools that are not available: {', '.join(missing_tools)}."
        )
    missing_agents = sorted(set(checkpoint.managed_agent_names) - set(managed_agent_names))
    if missing_agents:
        raise CheckpointCompatibilityError(
            f"Checkpoint references managed agents that are not available: {', '.join(missing_agents)}."
        )


class CheckpointManager:
    """Persist checkpoints as per-step JSON snapshots with a latest pointer.

    Args:
        directory (`str` or `Path`): Directory where `checkpoint-*.json` files and `latest.json` are stored.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if self.directory.exists() and not self.directory.is_dir():
            raise CheckpointError(f"Checkpoint path '{self.directory}' must be a directory.")
        self._lock = threading.RLock()
        self._sequence = self._find_latest_sequence()
        self._checkpoint_id: str | None = None

    def save(self, state: CheckpointState) -> Path:
        """Assign metadata to `state`, persist it atomically, and update the latest pointer."""
        with self._lock:
            sequence = self._sequence + 1
            now = utc_now()
            checkpoint_id = state.checkpoint_id or uuid.uuid4().hex
            state = replace(
                state,
                sequence=sequence,
                checkpoint_id=checkpoint_id,
                created_at=state.created_at or now,
                updated_at=now,
            )
            checkpoint_path = self.directory / f"checkpoint-{sequence:04d}.json"
            document = {
                "schema_version": SCHEMA_VERSION,
                "checksum": _compute_checksum(state.dict()),
                "checkpoint": state.dict(),
            }
            self._atomic_write_json(checkpoint_path, document)
            self._atomic_write_json(
                self.directory / "latest.json",
                {"file": checkpoint_path.name, "checkpoint_id": checkpoint_id},
            )
            self._sequence = sequence
            self._checkpoint_id = checkpoint_id
            return checkpoint_path

    def load(self, path: str | Path | None = None) -> CheckpointState:
        """Load a checkpoint file, or the latest checkpoint when `path` is None."""
        checkpoint_path = Path(path) if path is not None else self.latest_path()
        if checkpoint_path is None or not checkpoint_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint file '{checkpoint_path}' was not found.")

        try:
            document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise CheckpointIntegrityError(f"Could not read checkpoint '{checkpoint_path}': {e}") from e

        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointIntegrityError(
                f"Unsupported checkpoint schema in '{checkpoint_path}': {document.get('schema_version')!r}."
            )
        checkpoint_data = document.get("checkpoint")
        if not isinstance(checkpoint_data, dict):
            raise CheckpointIntegrityError(f"Checkpoint '{checkpoint_path}' has no checkpoint object.")
        if document.get("checksum") != _compute_checksum(checkpoint_data):
            raise CheckpointIntegrityError(f"Checkpoint '{checkpoint_path}' failed checksum validation.")
        try:
            return CheckpointState.from_dict(checkpoint_data)
        except (KeyError, TypeError, ValueError) as e:
            raise CheckpointIntegrityError(f"Malformed checkpoint '{checkpoint_path}': {e}") from e

    def latest_path(self) -> Path | None:
        """Return the path of the latest checkpoint, falling back to the highest sequence."""
        pointer_path = self.directory / "latest.json"
        if pointer_path.exists():
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                candidate = self.directory / pointer["file"]
                if candidate.exists():
                    return candidate
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass
        candidates = sorted(self.directory.glob("checkpoint-*.json"))
        return candidates[-1] if candidates else None

    def _find_latest_sequence(self) -> int:
        max_sequence = 0
        for path in self.directory.glob("checkpoint-*.json"):
            try:
                max_sequence = max(max_sequence, int(path.stem.split("-")[-1]))
            except ValueError:
                continue
        return max_sequence

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CheckpointPersistenceError(f"Could not create checkpoint directory '{path.parent}': {e}") from e

        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            tmp_path = None
        except (OSError, TypeError, ValueError) as e:
            raise CheckpointPersistenceError(f"Could not write checkpoint '{path}': {e}") from e
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)


def _compute_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointInFlightError",
    "CheckpointIntegrityError",
    "CheckpointManager",
    "CheckpointNotFoundError",
    "CheckpointPersistenceError",
    "CheckpointRestoreError",
    "CheckpointState",
    "CheckpointValidationError",
    "ExecutionBudget",
    "SCHEMA_VERSION",
    "StepResultStatus",
    "UNKNOWN_RESULT",
    "UnknownResultError",
    "build_budget",
    "build_evidence",
    "build_plans",
    "deserialize_chat_message",
    "deserialize_error",
    "deserialize_memory_step",
    "deserialize_state",
    "deserialize_tool_call",
    "deserialize_value",
    "new_run_id",
    "serialize_chat_message",
    "serialize_error",
    "serialize_memory_step",
    "serialize_state",
    "serialize_tool_call",
    "serialize_value",
    "utc_now",
    "validate_checkpoint_compatibility",
]
