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
import pytest

from smolagents.agents import ToolCallingAgent
from smolagents.default_tools import FinalAnswerTool
from smolagents.gate import (
    ToolExecutionGate,
    ToolExecutionResult,
    ToolPermissionError,
    ToolPermissionPolicy,
)
from smolagents.memory import ToolCall
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.tools import tool
from smolagents.utils import AgentToolCallError, AgentToolExecutionError


class DummyModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        return ChatMessage(role=MessageRole.ASSISTANT, content="")


@pytest.fixture
def echo_tool():
    @tool
    def echo(text: str) -> str:
        """Echo the input text.

        Args:
            text: The text to echo.
        Returns:
            The same text.
        """
        return text

    return echo


def test_execute_returns_normalized_result(echo_tool):
    gate = ToolExecutionGate(tools=[echo_tool])

    result = gate.execute(ToolCall(name=echo_tool.name, arguments={"text": "hello"}, id="call_1"))

    assert isinstance(result, ToolExecutionResult)
    assert result.tool_call_id == "call_1"
    assert result.tool_name == echo_tool.name
    assert str(result.output) == "hello"
    assert result.observation == "hello"
    assert result.is_final_answer is False


def test_execute_raises_for_unknown_tool(echo_tool):
    gate = ToolExecutionGate(tools=[echo_tool])

    with pytest.raises(AgentToolExecutionError, match="Unknown tool"):
        gate.execute(ToolCall(name="missing_tool", arguments={}, id="call_1"))


def test_execute_raises_for_invalid_arguments(echo_tool):
    gate = ToolExecutionGate(tools=[echo_tool])

    with pytest.raises(AgentToolCallError):
        gate.execute(ToolCall(name=echo_tool.name, arguments={"unknown": "value"}, id="call_1"))


def test_execute_respects_denied_tools(echo_tool):
    gate = ToolExecutionGate(
        tools=[echo_tool],
        permissions=ToolPermissionPolicy(denied_tools=[echo_tool.name]),
    )

    with pytest.raises(ToolPermissionError):
        gate.execute(ToolCall(name=echo_tool.name, arguments={"text": "hello"}, id="call_1"))


def test_execute_respects_allowed_tools(echo_tool):
    gate = ToolExecutionGate(
        tools=[echo_tool],
        permissions=ToolPermissionPolicy(allowed_tools=["other_tool"]),
    )

    with pytest.raises(ToolPermissionError):
        gate.execute(ToolCall(name=echo_tool.name, arguments={"text": "hello"}, id="call_1"))


def test_execute_substitutes_state_variables(echo_tool):
    gate = ToolExecutionGate(tools=[echo_tool], state={"payload": "hello"})

    result = gate.execute(ToolCall(name=echo_tool.name, arguments={"text": "payload"}, id="call_1"))

    assert str(result.output) == "hello"


def test_execute_flags_final_answer():
    gate = ToolExecutionGate(tools=[FinalAnswerTool()])

    result = gate.execute(ToolCall(name="final_answer", arguments={"answer": "42"}, id="call_1"))

    assert result.is_final_answer is True
    assert result.observation == "42"


def test_tool_calling_agent_routes_calls_through_gate(echo_tool):
    agent = ToolCallingAgent(model=DummyModel(), tools=[echo_tool])

    assert isinstance(agent.tool_execution_gate, ToolExecutionGate)
    assert str(agent.execute_tool_call(echo_tool.name, {"text": "hello"})) == "hello"


def test_tool_calling_agent_applies_permission_policy(echo_tool):
    agent = ToolCallingAgent(
        model=DummyModel(),
        tools=[echo_tool],
        tool_permissions=ToolPermissionPolicy(denied_tools=[echo_tool.name]),
    )

    with pytest.raises(ToolPermissionError):
        agent.execute_tool_call(echo_tool.name, {"text": "hello"})
