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
4. Tool execution and result normalization

Keeping those concerns in one place makes tool execution predictable and lets callers rely on a single, normalized
result type: [`ToolExecutionResult`].
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, MutableMapping
from dataclasses import dataclass
from typing import Any

from .agent_types import AgentAudio, AgentImage
from .memory import ToolCall
from .monitoring import AgentLogger, LogLevel
from .tools import BaseTool, validate_tool_arguments
from .utils import AgentToolCallError, AgentToolExecutionError


logger = logging.getLogger(__name__)

# Canonical memory keys used to store media returned by tools.
MEDIA_OBSERVATION_NAMES: dict[type, str] = {AgentImage: "image.png", AgentAudio: "audio.mp3"}


class ToolPermissionError(AgentToolExecutionError):
    """Raised when a tool call is rejected by the permission policy."""


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

    1. tool existence check: raises [`AgentToolExecutionError`] for unknown tools,
    2. argument validation: raises [`AgentToolCallError`] for invalid arguments,
    3. permission check: raises [`ToolPermissionError`] for rejected tools,
    4. execution and normalization: invokes the tool (or managed agent) and returns a
       [`ToolExecutionResult`].

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
    """

    def __init__(
        self,
        tools: dict[str, BaseTool] | Iterable[BaseTool],
        managed_agents: dict[str, Any] | None = None,
        permissions: ToolPermissionPolicy | None = None,
        state: MutableMapping[str, Any] | None = None,
        logger: AgentLogger | None = None,
    ):
        self.tools: dict[str, BaseTool] = (
            dict(tools) if isinstance(tools, dict) else {tool.name: tool for tool in tools}
        )
        self.managed_agents: dict[str, Any] = dict(managed_agents or {})
        self.permissions = permissions or ToolPermissionPolicy()
        self.state = state if state is not None else {}
        self.logger = logger or AgentLogger(level=LogLevel.OFF)

    @property
    def available_tools(self) -> dict[str, Any]:
        """All callable entries, with tools and managed agents merged."""
        return {**self.tools, **self.managed_agents}

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
        """
        tool_name = tool_call.name
        arguments = tool_call.arguments or {}

        # 1. The tool must exist.
        tool, is_managed_agent = self.resolve_tool(tool_name)
        # State variables are resolved before validation so the effective values are checked.
        arguments = self.substitute_state_variables(arguments)
        # 2. Arguments must match the tool's input schema.
        self.validate_arguments(tool, arguments, tool_name)
        # 3. The tool must be allowed by the permission policy.
        self.check_permission(tool_name)
        # 4. Execute and normalize the result.
        output = self.invoke_tool(tool, arguments, tool_name, is_managed_agent)
        return self.normalize_result(tool_call, output)

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
    "ToolExecutionGate",
    "ToolExecutionResult",
    "ToolPermissionError",
    "ToolPermissionPolicy",
]
