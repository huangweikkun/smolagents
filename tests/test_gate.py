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
from smolagents.audit import ApprovalManager, ApprovalStatus, AuditEventType, AuditLog
from smolagents.default_tools import FinalAnswerTool
from smolagents.gate import (
    GateDecisionStatus,
    ToolApprovalRequiredError,
    ToolDuplicateCallError,
    ToolExecutionGate,
    ToolExecutionResult,
    ToolPermissionError,
    ToolPermissionPolicy,
    ToolRiskLevel,
    ToolRiskPolicy,
)
from smolagents.memory import ToolCall
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.tools import Tool, tool
from smolagents.utils import AgentToolCallError, AgentToolExecutionError


class CountingEchoTool(Tool):
    name = "counting_echo"
    description = "Echo the given text and count how often the handler runs."
    inputs = {"text": {"type": "string", "description": "Text to echo."}}
    output_type = "string"

    def __init__(self):
        self.calls = 0
        super().__init__()

    def forward(self, text: str) -> str:
        self.calls += 1
        return text


class WritingTool(Tool):
    name = "write_file"
    description = "Write text to a file."
    inputs = {
        "path": {"type": "string", "description": "Relative path."},
        "content": {"type": "string", "description": "Text content."},
    }
    output_type = "string"

    def __init__(self, root):
        self.root = root
        self.calls = 0
        super().__init__()

    def forward(self, path: str, content: str) -> str:
        self.calls += 1
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)


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


def test_invalid_arguments_return_structured_decision_without_running_handler():
    tool = CountingEchoTool()
    gate = ToolExecutionGate(tools=[tool])

    outcome = gate.try_execute(ToolCall(name=tool.name, arguments={}, id="call-bad"))

    assert outcome.result is None
    assert outcome.decision.status is GateDecisionStatus.DENIED
    assert outcome.decision.rule == "invalid_arguments"
    assert outcome.decision.tool_call_id == "call-bad"
    assert tool.calls == 0


def test_third_identical_call_is_blocked_before_execution():
    tool = CountingEchoTool()
    gate = ToolExecutionGate(tools=[tool], duplicate_call_limit=2)

    for index in range(2):
        gate.execute(ToolCall(name=tool.name, arguments={"text": "same"}, id=f"call-{index}"))
    assert tool.calls == 2

    outcome = gate.try_execute(ToolCall(name=tool.name, arguments={"text": "same"}, id="call-2"))

    assert outcome.result is None
    assert outcome.decision.status is GateDecisionStatus.DUPLICATE
    assert outcome.decision.rule == "duplicate_call"
    assert outcome.decision.tool_call_id == "call-2"
    assert tool.calls == 2


def test_unapproved_high_risk_call_waits_and_keeps_target_unchanged(tmp_path):
    approvals = ApprovalManager(tmp_path / "approvals.json")
    audit = AuditLog(tmp_path / "audit.jsonl", run_id="run-1")
    tool = WritingTool(tmp_path / "repo")
    gate = ToolExecutionGate(
        tools=[tool],
        risk_policy=ToolRiskPolicy(tool_levels={tool.name: ToolRiskLevel.HIGH}),
        approval_manager=approvals,
        audit_log=audit,
        run_id="run-1",
    )
    call = ToolCall(name=tool.name, arguments={"path": "draft.md", "content": "hello"}, id="call-w1")

    outcome = gate.try_execute(call)

    assert outcome.result is None
    assert outcome.decision.status is GateDecisionStatus.WAITING_APPROVAL
    assert outcome.decision.rule == "approval_required"
    assert outcome.decision.approval_id is not None
    assert tool.calls == 0
    assert not (tmp_path / "repo" / "draft.md").exists()
    records = approvals.list_records()
    assert [record.record_id for record in records] == [outcome.decision.approval_id]
    events = audit.events(run_id="run-1", tool_call_id="call-w1")
    assert events[-1].event_type is AuditEventType.GATE_WAITING_APPROVAL
    assert events[-1].decision == "waiting_approval"


def test_execute_raises_approval_error_for_unapproved_call(tmp_path):
    approvals = ApprovalManager(tmp_path / "approvals.json")
    tool = WritingTool(tmp_path / "repo")
    gate = ToolExecutionGate(
        tools=[tool],
        risk_policy=ToolRiskPolicy(tool_levels={tool.name: ToolRiskLevel.HIGH}),
        approval_manager=approvals,
    )
    call = ToolCall(name=tool.name, arguments={"path": "draft.md", "content": "hello"}, id="call-w2")

    with pytest.raises(ToolApprovalRequiredError) as error:
        gate.execute(call)

    assert error.value.decision.status is GateDecisionStatus.WAITING_APPROVAL
    assert tool.calls == 0


def test_approved_call_executes_once_and_replay_is_rejected(tmp_path):
    approvals = ApprovalManager(tmp_path / "approvals.json")
    tool = WritingTool(tmp_path / "repo")
    gate = ToolExecutionGate(
        tools=[tool],
        risk_policy=ToolRiskPolicy(tool_levels={tool.name: ToolRiskLevel.HIGH}),
        approval_manager=approvals,
    )
    call = ToolCall(name=tool.name, arguments={"path": "draft.md", "content": "hello"}, id="call-w3")
    outcome = gate.try_execute(call)
    approvals.approve(outcome.decision.approval_id, approver="maintainer")

    result = gate.execute(call)
    assert "draft.md" in result.observation
    assert tool.calls == 1

    with pytest.raises(ToolDuplicateCallError) as error:
        gate.execute(call)

    assert error.value.decision.rule == "approval_consumed"
    assert tool.calls == 1
    assert (tmp_path / "repo" / "draft.md").read_text(encoding="utf-8") == "hello"


def test_high_risk_call_without_approval_manager_is_denied():
    tool = CountingEchoTool()
    gate = ToolExecutionGate(
        tools=[tool],
        risk_policy=ToolRiskPolicy(tool_levels={tool.name: ToolRiskLevel.HIGH}),
    )

    outcome = gate.try_execute(ToolCall(name=tool.name, arguments={"text": "hello"}, id="call-risk"))

    assert outcome.decision.status is GateDecisionStatus.DENIED
    assert outcome.decision.rule == "approval_unavailable"
    assert tool.calls == 0


def test_tool_calling_agent_wires_audit_and_approval(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    approvals = ApprovalManager(tmp_path / "approvals.json", audit_log=audit)
    tool = WritingTool(tmp_path / "repo")
    agent = ToolCallingAgent(
        model=DummyModel(),
        tools=[tool],
        risk_policy=ToolRiskPolicy(tool_levels={tool.name: ToolRiskLevel.HIGH}),
        approval_manager=approvals,
        audit_log=audit,
    )
    arguments = {"path": "draft.md", "content": "hello"}

    with pytest.raises(ToolApprovalRequiredError):
        agent.execute_tool_call(tool.name, arguments)

    assert tool.calls == 0
    waiting = [event for event in audit.events() if event.event_type is AuditEventType.GATE_WAITING_APPROVAL]
    assert waiting and waiting[0].tool_call_id == ""

    record = approvals.list_records(status=ApprovalStatus.PENDING)[0]
    approvals.approve(record.record_id, approver="maintainer")

    assert agent.execute_tool_call(tool.name, arguments) is not None
    assert tool.calls == 1
    with pytest.raises(ToolDuplicateCallError):
        agent.execute_tool_call(tool.name, arguments)
    assert tool.calls == 1
