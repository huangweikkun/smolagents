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
"""Execution runtime for [`ToolCall`] objects emitted by [`ToolCallingAgent`].

[`ToolExecutionGate`] is the single entry point used to run a tool call. Every call goes through the same pipeline:

1. Tool existence check
2. Argument validation
3. Permission check
4. Risk check
5. Approval check
6. Duplicate-call check
7. Tool execution and result normalization

Rejected calls never reach the tool handler. [`try_execute`] returns a structured [`ToolExecutionOutcome`] carrying
the original `tool_call_id`; [`execute`] raises the matching error type for callers that rely on exceptions. Every
decision and execution stage can be written to an append-only [`AuditLog`].
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .agent_types import AgentAudio, AgentImage
from .audit import ApprovalManager, ApprovalRecord, ApprovalStatus, AuditEventType, AuditLog
from .memory import ToolCall
from .monitoring import AgentLogger, LogLevel
from .tools import BaseTool, validate_tool_arguments
from .utils import AgentToolCallError, AgentToolExecutionError, make_json_serializable


logger = logging.getLogger(__name__)

# Canonical memory keys used to store media returned by tools.
MEDIA_OBSERVATION_NAMES: dict[type, str] = {AgentImage: "image.png", AgentAudio: "audio.mp3"}


class ToolPermissionError(AgentToolExecutionError):
    """Raised when a tool call is rejected by the permission policy."""


class GateDecisionStatus(str, Enum):
    """Outcome of the pre-execution checks applied to a tool call."""

    ALLOWED = "allowed"
    DENIED = "denied"
    WAITING_APPROVAL = "waiting_approval"
    DUPLICATE = "duplicate"


class ToolRiskLevel(str, Enum):
    """Risk classification used to decide whether a tool call needs approval."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ToolRiskPolicy:
    """Risk levels assigned to tools, with high-risk tools gated by approval.

    Args:
        default_level ([`ToolRiskLevel`]): Level applied to tools without an explicit entry.
        tool_levels (`Mapping[str, ToolRiskLevel]`, *optional*): Per-tool overrides.
    """

    default_level: ToolRiskLevel = ToolRiskLevel.LOW
    tool_levels: Mapping[str, ToolRiskLevel] = field(default_factory=dict)

    def __post_init__(self):
        self.default_level = ToolRiskLevel(self.default_level)
        self.tool_levels = {name: ToolRiskLevel(level) for name, level in self.tool_levels.items()}

    def level_for(self, tool_name: str) -> ToolRiskLevel:
        """Return the risk level assigned to `tool_name`."""
        return self.tool_levels.get(tool_name, self.default_level)

    def requires_approval(self, tool_name: str) -> bool:
        """Whether `tool_name` is risky enough to require an explicit approval."""
        return self.level_for(tool_name) is ToolRiskLevel.HIGH


@dataclass(frozen=True)
class GateDecision:
    """Structured decision taken by the gate before running a tool call.

    Args:
        tool_call_id (`str`): Identifier of the originating tool call.
        tool_name (`str`): Name of the requested tool.
        status ([`GateDecisionStatus`]): Decision outcome.
        rule (`str`, *optional*): Rule that produced the decision.
        message (`str`): Human-readable explanation.
        approval_id (`str`, *optional*): Pending or consumed approval record, when relevant.
    """

    tool_call_id: str
    tool_name: str
    status: GateDecisionStatus
    rule: str | None = None
    message: str = ""
    approval_id: str | None = None

    @property
    def allowed(self) -> bool:
        """Whether the decision lets the call reach the tool handler."""
        return self.status is GateDecisionStatus.ALLOWED


@dataclass(frozen=True)
class ToolExecutionOutcome:
    """Structured result of [`ToolExecutionGate.try_execute`].

    Args:
        decision ([`GateDecision`]): Decision taken before execution.
        result ([`ToolExecutionResult`], *optional*): Normalized result, only present when the call was allowed.
    """

    decision: GateDecision
    result: "ToolExecutionResult | None" = None


class ToolGateRejection(AgentToolExecutionError):
    """Base class for gate rejections, carrying the structured decision."""

    def __init__(self, decision: GateDecision, logger: AgentLogger):
        super().__init__(
            f"Tool call '{decision.tool_call_id}' rejected by rule '{decision.rule}': {decision.message}", logger
        )
        self.decision = decision


class ToolDuplicateCallError(ToolGateRejection):
    """Raised when an identical tool call exceeds the duplicate-call limit."""


class ToolApprovalRequiredError(ToolGateRejection):
    """Raised when a tool call is waiting for approval or approval is unavailable."""


class ToolApprovalConsumedError(ToolDuplicateCallError):
    """Raised when an already approved and executed call is replayed."""


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized result of a single tool call.

    Args:
        tool_call_id (`str`): Identifier of the originating tool call.
        tool_name (`str`): Name of the executed tool or managed agent.
        output (`Any`): Raw output returned by the tool.
        observation (`str`): String representation used as the agent observation.
        is_final_answer (`bool`): Whether this call produced the agent's final answer.
    """

    tool_call_id: str
    tool_name: str
    output: Any
    observation: str
    is_final_answer: bool


class ToolPermissionPolicy:
    """Allow/deny rules applied before a tool call is executed.

    Args:
        allowed_tools (`Collection[str]`, *optional*):
            If provided, only these tools may be executed. If `None`, every tool not explicitly denied is allowed.
        denied_tools (`Collection[str]`, *optional*):
            Tool names that are always rejected. Deny rules take precedence over allow rules.
    """

    def __init__(
        self,
        allowed_tools: Collection[str] | None = None,
        denied_tools: Collection[str] | None = None,
    ):
        self.allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else None
        self.denied_tools = frozenset(denied_tools or ())

    def is_allowed(self, tool_name: str) -> bool:
        """Return whether `tool_name` may be executed under this policy."""
        if tool_name in self.denied_tools:
            return False
        return self.allowed_tools is None or tool_name in self.allowed_tools


class ToolExecutionGate:
    """Execution runtime used by [`ToolCallingAgent`] to run [`ToolCall`] objects.

    The gate performs, in order:

    1. tool existence check,
    2. argument validation,
    3. permission check,
    4. risk check,
    5. approval check,
    6. duplicate-call check,
    7. execution and normalization.

    Rejections never reach the tool handler. [`try_execute`] returns a structured [`ToolExecutionOutcome`] that keeps
    the original `tool_call_id`; [`execute`] raises the matching error type for exception-based callers.

    Args:
        tools (`dict[str, BaseTool]` or `Iterable[BaseTool]`):
            Available tools, either keyed by name or as an iterable of [`BaseTool`] instances.
        managed_agents (`dict[str, Any]`, *optional*):
            Managed agents callable by name, which are validated and executed like tools.
        permissions (`ToolPermissionPolicy`, *optional*):
            Permission policy applied to every call. Defaults to allowing every tool.
        state (`MutableMapping[str, Any]`, *optional*):
            Agent state used to substitute state variables in arguments and to store media outputs.
        logger (`AgentLogger`, *optional*):
            Logger used to report execution errors.
        risk_policy ([`ToolRiskPolicy`], *optional*):
            Risk levels applied before execution; high-risk tools require approval. Defaults to all-low risk.
        approval_manager ([`ApprovalManager`], *optional*):
            Approval backend used for risky calls. Without it, risky calls are denied instead of executed.
        audit_log ([`AuditLog`], *optional*):
            Append-only audit log receiving allow, deny, approval and execution events.
        run_id (`str`, *optional*): Run identifier attached to audit events.
        actor (`str`): Actor recorded for audit events produced by the gate.
        duplicate_call_limit (`int`): Number of identical calls that may execute before duplicates are denied.
    """

    def __init__(
        self,
        tools: dict[str, BaseTool] | Iterable[BaseTool],
        managed_agents: dict[str, Any] | None = None,
        permissions: ToolPermissionPolicy | None = None,
        state: MutableMapping[str, Any] | None = None,
        logger: AgentLogger | None = None,
        *,
        risk_policy: ToolRiskPolicy | None = None,
        approval_manager: ApprovalManager | None = None,
        audit_log: AuditLog | None = None,
        run_id: str | None = None,
        actor: str = "gate",
        duplicate_call_limit: int = 2,
    ):
        if duplicate_call_limit < 1:
            raise ValueError("'duplicate_call_limit' must be at least 1.")
        self.tools: dict[str, BaseTool] = (
            dict(tools) if isinstance(tools, dict) else {tool.name: tool for tool in tools}
        )
        self.managed_agents: dict[str, Any] = dict(managed_agents or {})
        self.permissions = permissions or ToolPermissionPolicy()
        self.state = state if state is not None else {}
        self.logger = logger or AgentLogger(level=LogLevel.OFF)
        self.risk_policy = risk_policy or ToolRiskPolicy()
        self.approval_manager = approval_manager
        self.audit_log = audit_log
        self.run_id = run_id
        self.actor = actor
        self.duplicate_call_limit = duplicate_call_limit
        self._call_counts: dict[str, int] = {}
        self._consumed_approvals: set[str] = set()
        self._consumed_call_ids: set[str] = set()

    @property
    def available_tools(self) -> dict[str, Any]:
        """All callable entries, with tools and managed agents merged."""
        return {**self.tools, **self.managed_agents}

    def bind_run(self, run_id: str) -> None:
        """Attach a run identifier used by subsequent audit events."""
        self.run_id = run_id

    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        """Run a tool call through the full execution pipeline.

        Args:
            tool_call (`ToolCall`): The call emitted by the model.

        Returns:
            `ToolExecutionResult`: The normalized execution result.

        Raises:
            AgentToolExecutionError: If the tool does not exist or fails during execution.
            AgentToolCallError: If the arguments do not match the tool's input schema.
            ToolPermissionError: If the permission policy rejects the tool.
            ToolApprovalRequiredError: If the call waits for approval or approval is unavailable.
            ToolDuplicateCallError: If the call repeats too often or replays a consumed approval.
        """
        outcome = self.try_execute(tool_call)
        if outcome.result is not None:
            return outcome.result
        self._raise_rejection(outcome.decision)

    def try_execute(self, tool_call: ToolCall) -> ToolExecutionOutcome:
        """Evaluate every pre-execution rule and run the call only when allowed.

        Args:
            tool_call (`ToolCall`): The call emitted by the model.

        Returns:
            `ToolExecutionOutcome`: A structured decision, plus the normalized result when allowed.
        """
        tool_name = tool_call.name
        try:
            tool, is_managed_agent = self.resolve_tool(tool_name)
        except AgentToolExecutionError as error:
            return self._reject(tool_call, GateDecisionStatus.DENIED, "unknown_tool", str(error))

        # State variables are resolved before validation so the effective values are checked.
        arguments = self.substitute_state_variables(tool_call.arguments or {})
        try:
            self.validate_arguments(tool, arguments, tool_name)
        except AgentToolCallError as error:
            return self._reject(tool_call, GateDecisionStatus.DENIED, "invalid_arguments", str(error))
        except AgentToolExecutionError as error:
            return self._reject(tool_call, GateDecisionStatus.DENIED, "validation_error", str(error))

        if not self.permissions.is_allowed(tool_name):
            return self._reject(
                tool_call,
                GateDecisionStatus.DENIED,
                "permission",
                f"Tool '{tool_name}' is not permitted by the current permission policy.",
            )

        approval_outcome, approved_record = self.apply_risk_and_approval(tool_call, tool_name, arguments)
        if approval_outcome is not None:
            return approval_outcome

        if self._call_counts.get(self.fingerprint(tool_name, arguments), 0) >= self.duplicate_call_limit:
            return self._reject(
                tool_call,
                GateDecisionStatus.DUPLICATE,
                "duplicate_call",
                f"Tool '{tool_name}' was already called {self.duplicate_call_limit} times with the same arguments.",
            )

        self.audit(
            AuditEventType.GATE_ALLOW,
            tool_call=tool_call,
            decision=GateDecisionStatus.ALLOWED.value,
            message=f"Tool '{tool_name}' allowed.",
        )
        self.audit(
            AuditEventType.EXECUTION_STARTED,
            tool_call=tool_call,
            decision="started",
            message=f"Tool '{tool_name}' started.",
        )
        try:
            output = self.invoke_tool(tool, arguments, tool_name, is_managed_agent)
        except Exception as error:
            self.audit(
                AuditEventType.EXECUTION_FAILED,
                tool_call=tool_call,
                decision="failed",
                message=str(error),
            )
            raise
        result = self.normalize_result(tool_call, output)
        fingerprint = self.fingerprint(tool_name, arguments)
        self._call_counts[fingerprint] = self._call_counts.get(fingerprint, 0) + 1
        if approved_record is not None:
            self._consumed_approvals.add(approved_record.record_id)
            self._consumed_call_ids.add(tool_call.id)
        self.audit(
            AuditEventType.EXECUTION_COMPLETED,
            tool_call=tool_call,
            decision="completed",
            message=f"Tool '{tool_name}' completed.",
        )
        return ToolExecutionOutcome(
            decision=GateDecision(tool_call.id, tool_name, GateDecisionStatus.ALLOWED, message="Tool call allowed."),
            result=result,
        )

    def apply_risk_and_approval(
        self,
        tool_call: ToolCall,
        tool_name: str,
        arguments: dict[str, Any] | Any,
    ) -> tuple[ToolExecutionOutcome | None, ApprovalRecord | None]:
        """Return an approval rejection, or the approved record that unlocks the call."""
        if not self.risk_policy.requires_approval(tool_name):
            return None, None
        if self.approval_manager is None:
            return (
                self._reject(
                    tool_call,
                    GateDecisionStatus.DENIED,
                    "approval_unavailable",
                    f"Tool '{tool_name}' is high risk but no approval manager is configured.",
                ),
                None,
            )
        if tool_call.id in self._consumed_call_ids:
            return (
                self._reject(
                    tool_call,
                    GateDecisionStatus.DUPLICATE,
                    "approval_consumed",
                    f"Tool call '{tool_call.id}' was already approved and executed.",
                ),
                None,
            )

        approved = self.find_approved_record(tool_call, tool_name)
        if approved is not None:
            return None, approved

        pending = self.find_pending_record(tool_call, tool_name)
        if pending is None:
            pending = self.approval_manager.request_approval(
                subject="tool",
                operation_type=tool_name,
                content=make_json_serializable(arguments),
                applicant=self.actor,
                metadata={
                    "tool_call_id": tool_call.id,
                    "run_id": self.run_id,
                    "risk": self.risk_policy.level_for(tool_name).value,
                },
            )
        return (
            self._reject(
                tool_call,
                GateDecisionStatus.WAITING_APPROVAL,
                "approval_required",
                f"Tool '{tool_name}' requires approval; record '{pending.record_id}' is {pending.status.value}.",
                approval_id=pending.record_id,
            ),
            None,
        )

    def find_approved_record(self, tool_call: ToolCall, tool_name: str) -> ApprovalRecord | None:
        """Return an unused approved record bound to this tool call, if any."""
        if self.approval_manager is None:
            return None
        for record in reversed(self.approval_manager.list_records(status=ApprovalStatus.APPROVED)):
            if record.operation_type != tool_name or record.metadata.get("tool_call_id") != tool_call.id:
                continue
            if record.record_id in self._consumed_approvals:
                continue
            return record
        return None

    def find_pending_record(self, tool_call: ToolCall, tool_name: str) -> ApprovalRecord | None:
        """Return an existing pending approval for this tool call, avoiding duplicate requests."""
        if self.approval_manager is None:
            return None
        for record in self.approval_manager.list_records(status=ApprovalStatus.PENDING):
            if record.operation_type == tool_name and record.metadata.get("tool_call_id") == tool_call.id:
                return record
        return None

    @staticmethod
    def fingerprint(tool_name: str, arguments: dict[str, Any] | Any) -> str:
        """Build a stable fingerprint used for duplicate-call detection."""
        payload = {"tool": tool_name, "arguments": make_json_serializable(arguments)}
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def _reject(
        self,
        tool_call: ToolCall,
        status: GateDecisionStatus,
        rule: str,
        message: str,
        approval_id: str | None = None,
    ) -> ToolExecutionOutcome:
        decision = GateDecision(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            status=status,
            rule=rule,
            message=message,
            approval_id=approval_id,
        )
        event_type = (
            AuditEventType.GATE_WAITING_APPROVAL
            if status is GateDecisionStatus.WAITING_APPROVAL
            else AuditEventType.GATE_DENY
        )
        metadata = {"approval_id": approval_id} if approval_id is not None else None
        self.audit(
            event_type, tool_call=tool_call, decision=status.value, rule=rule, message=message, metadata=metadata
        )
        return ToolExecutionOutcome(decision=decision)

    def _raise_rejection(self, decision: GateDecision) -> None:
        message = f"Tool call '{decision.tool_call_id}' rejected by rule '{decision.rule}': {decision.message}"
        error: AgentToolExecutionError
        if decision.rule == "unknown_tool":
            error = AgentToolExecutionError(message, self.logger)
        elif decision.rule == "permission":
            error = ToolPermissionError(message, self.logger)
        elif decision.rule in {"approval_required", "approval_unavailable"}:
            error = ToolApprovalRequiredError(decision, self.logger)
        elif decision.rule == "approval_consumed":
            error = ToolApprovalConsumedError(decision, self.logger)
        elif decision.status is GateDecisionStatus.DUPLICATE:
            error = ToolDuplicateCallError(decision, self.logger)
        else:
            error = AgentToolCallError(message, self.logger)
        error.decision = decision
        raise error

    def audit(
        self,
        event_type: AuditEventType,
        *,
        tool_call: ToolCall,
        decision: str,
        rule: str | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append an audit event when an audit log is configured."""
        if self.audit_log is None:
            return
        self.audit_log.append(
            event_type=event_type,
            actor=self.actor,
            run_id=self.run_id,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            rule=rule,
            decision=decision,
            message=message,
            metadata=metadata,
        )

    def resolve_tool(self, tool_name: str) -> tuple[Any, bool]:
        """Return the callable registered under `tool_name` and whether it is a managed agent."""
        available_tools = self.available_tools
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )
        return available_tools[tool_name], tool_name in self.managed_agents

    def substitute_state_variables(self, arguments: dict[str, Any] | Any) -> dict[str, Any] | Any:
        """Replace string argument values with their corresponding state values when available."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def validate_arguments(self, tool: Any, arguments: dict[str, Any] | Any, tool_name: str) -> None:
        """Validate `arguments` against the tool's input schema."""
        try:
            validate_tool_arguments(tool, arguments)
        except (ValueError, TypeError) as e:
            raise AgentToolCallError(str(e), self.logger) from e
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}"
            raise AgentToolExecutionError(error_msg, self.logger) from e

    def check_permission(self, tool_name: str) -> None:
        """Reject `tool_name` when the permission policy does not allow it."""
        if not self.permissions.is_allowed(tool_name):
            raise ToolPermissionError(
                f"Tool '{tool_name}' is not permitted by the current permission policy.", self.logger
            )

    def invoke_tool(self, tool: Any, arguments: dict[str, Any] | Any, tool_name: str, is_managed_agent: bool) -> Any:
        """Call the tool or managed agent and translate failures into agent errors."""
        try:
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)
        except Exception as e:
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e

    def normalize_result(self, tool_call: ToolCall, output: Any) -> ToolExecutionResult:
        """Normalize a raw tool output into a [`ToolExecutionResult`]."""
        observation = self.build_observation(output)
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output=output,
            observation=observation,
            is_final_answer=tool_call.name == "final_answer",
        )

    def build_observation(self, output: Any) -> str:
        """Build the observation string, storing media outputs in state under canonical keys."""
        observation_name = MEDIA_OBSERVATION_NAMES.get(type(output))
        if observation_name is not None:
            self.state[observation_name] = output
            return f"Stored '{observation_name}' in memory."
        return str(output).strip()


__all__ = [
    "GateDecision",
    "GateDecisionStatus",
    "ToolApprovalConsumedError",
    "ToolApprovalRequiredError",
    "ToolDuplicateCallError",
    "ToolExecutionGate",
    "ToolExecutionOutcome",
    "ToolExecutionResult",
    "ToolGateRejection",
    "ToolPermissionError",
    "ToolPermissionPolicy",
    "ToolRiskLevel",
    "ToolRiskPolicy",
]
