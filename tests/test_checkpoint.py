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
import json

import pytest
from PIL import Image

from smolagents.agents import ToolCallingAgent
from smolagents.audit import ApprovalManager
from smolagents.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointInFlightError,
    CheckpointIntegrityError,
    CheckpointManager,
    CheckpointState,
    ExecutionBudget,
    UnknownResultError,
    deserialize_error,
    deserialize_memory_step,
    serialize_memory_step,
    serialize_tool_call,
    utc_now,
)
from smolagents.memory import ActionStep, TaskStep, ToolCall
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.state_store import ExecutionStatus
from smolagents.tools import Tool
from smolagents.utils import AgentError


class FinalAnswerModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if tools_to_call_from is None:
            return ChatMessage(role=MessageRole.ASSISTANT, content="done")
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_final",
                    type="function",
                    function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "done"}),
                )
            ],
        )


class EchoToolModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if tools_to_call_from is None:
            return ChatMessage(role=MessageRole.ASSISTANT, content="finished")
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_echo",
                    type="function",
                    function=ChatMessageToolCallFunction(name="echo", arguments={"text": "hi"}),
                )
            ],
        )


class CountingEchoTool(Tool):
    name = "echo"
    description = "Echo the input text."
    inputs = {"text": {"type": "string", "description": "Text to echo."}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, text: str) -> str:
        self.calls += 1
        return text


def _make_state(
    *,
    task: str = "task",
    run_id: str = "run-1",
    memory_steps=None,
    tool_names=None,
    step_number: int = 0,
    next_step_number: int = 1,
    status: ExecutionStatus = ExecutionStatus.RUNNING,
    in_flight_step=None,
    agent_class: str = "ToolCallingAgent",
    final_output=None,
    model_id: str | None = None,
) -> CheckpointState:
    started_at = utc_now()
    return CheckpointState(
        run_id=run_id,
        task=task,
        agent_class=agent_class,
        model_id=model_id,
        max_steps=3,
        budget=ExecutionBudget(
            max_steps=3,
            steps_used=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            started_at=started_at,
        ),
        memory_system_prompt="system prompt",
        memory_steps=memory_steps or [serialize_memory_step(TaskStep(task=task))],
        state={},
        tool_names=tool_names or [],
        step_number=step_number,
        next_step_number=next_step_number,
        status=status.value,
        final_output=final_output,
        in_flight_step=in_flight_step,
    )


def test_manager_saves_history_and_loads_latest(tmp_path):
    manager = CheckpointManager(tmp_path / "checkpoints")

    first_path = manager.save(_make_state(task="first"))
    second_path = manager.save(_make_state(task="second"))

    assert first_path.name == "checkpoint-0001.json"
    assert second_path.name == "checkpoint-0002.json"
    assert manager.latest_path() == second_path
    assert manager.load().task == "second"
    assert manager.load(first_path).task == "first"


def test_manager_detects_checksum_mismatch(tmp_path):
    manager = CheckpointManager(tmp_path / "checkpoints")
    manager.save(_make_state(task="original"))
    path = manager.latest_path()

    document = json.loads(path.read_text(encoding="utf-8"))
    document["checkpoint"]["task"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError):
        manager.load()


def test_memory_step_serialization_roundtrip():
    tool_call = ToolCall(id="call_1", name="echo", arguments={"text": "hi"})
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=[{"type": "text", "text": "calling"}],
        tool_calls=[
            ChatMessageToolCall(
                id="call_2",
                type="function",
                function=ChatMessageToolCallFunction(name="echo", arguments={"text": "hi"}),
            )
        ],
    )
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        model_input_messages=[ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "hello"}])],
        tool_calls=[tool_call],
        error=UnknownResultError("UNKNOWN_RESULT", AgentLogger(level=LogLevel.OFF)),
        model_output_message=message,
        model_output="calling",
        observations="hi",
        observations_images=[Image.new("RGB", (2, 2))],
        action_output={"ok": True},
        token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        is_final_answer=True,
    )

    restored = deserialize_memory_step(serialize_memory_step(step))

    assert isinstance(restored.tool_calls[0], ToolCall)
    assert restored.tool_calls[0].name == "echo"
    assert isinstance(restored.model_output_message.tool_calls[0], ChatMessageToolCall)
    assert restored.model_output_message.tool_calls[0].function.name == "echo"
    assert restored.observations_images[0].size == (2, 2)
    assert isinstance(restored.error, UnknownResultError)
    assert restored.token_usage.total_tokens == 8


def test_checkpoint_contains_run_step_plan_evidence_budget_and_pending_approvals(tmp_path):
    approvals = ApprovalManager(tmp_path / "approvals.json")
    pending = approvals.request_approval(
        subject="workspace-a",
        operation_type="delete_workspace",
        content={"workspace_id": 42},
        applicant="alice",
    )

    agent = ToolCallingAgent(
        model=FinalAnswerModel(),
        tools=[],
        planning_interval=1,
        approval_manager=approvals,
        checkpoint_store=tmp_path / "checkpoints",
    )
    agent.run("do the task")

    state = CheckpointManager(tmp_path / "checkpoints").load()
    assert state.status == ExecutionStatus.COMPLETED.value
    assert state.run_id
    assert state.step_number >= 1
    assert state.plans
    assert state.evidence
    assert state.budget.steps_used >= 1
    assert state.pending_approvals
    assert state.pending_approvals[0]["record_id"] == pending.record_id


def test_tools_pending_marks_unknown_result_without_replay(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    in_flight = {
        "step_number": 1,
        "started_at": utc_now(),
        "phase": "tools_pending",
        "tool_calls": [serialize_tool_call(ToolCall(id="call_echo", name="echo", arguments={"text": "hi"}))],
    }
    CheckpointManager(checkpoint_dir).save(
        _make_state(
            task="do the task",
            tool_names=["echo", "final_answer"],
            step_number=1,
            next_step_number=1,
            in_flight_step=in_flight,
        )
    )

    echo = CountingEchoTool()
    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[echo], checkpoint_store=checkpoint_dir)

    result = agent.resume_from_checkpoint()

    assert str(result) == "done"
    assert echo.calls == 0

    state = CheckpointManager(checkpoint_dir).load()
    assert state.status == ExecutionStatus.COMPLETED.value
    assert any(
        evidence["step_number"] == 1 and evidence["result_status"] == "unknown_result" for evidence in state.evidence
    )
    unknown_steps = [
        item
        for item in state.memory_steps
        if item["step_type"] == "action" and item["error"] and item["error"]["type"] == "UnknownResultError"
    ]
    assert unknown_steps
    assert isinstance(deserialize_error(unknown_steps[0]["error"]), UnknownResultError)


def test_tools_pending_abort_policy_does_not_execute_tools(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    in_flight = {
        "step_number": 1,
        "started_at": utc_now(),
        "phase": "tools_pending",
        "tool_calls": [serialize_tool_call(ToolCall(id="call_echo", name="echo", arguments={"text": "hi"}))],
    }
    CheckpointManager(checkpoint_dir).save(
        _make_state(
            task="do the task",
            tool_names=["echo", "final_answer"],
            step_number=1,
            next_step_number=1,
            in_flight_step=in_flight,
        )
    )

    echo = CountingEchoTool()
    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[echo], checkpoint_store=checkpoint_dir)

    with pytest.raises(CheckpointInFlightError):
        agent.resume_from_checkpoint(in_flight_policy="abort")

    assert echo.calls == 0


def test_action_started_in_flight_is_retried_safely(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    in_flight = {
        "step_number": 1,
        "started_at": utc_now(),
        "phase": "action_started",
        "tool_calls": [],
    }
    CheckpointManager(checkpoint_dir).save(
        _make_state(
            task="do the task",
            tool_names=["final_answer"],
            step_number=1,
            next_step_number=1,
            in_flight_step=in_flight,
        )
    )

    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[], checkpoint_store=checkpoint_dir)
    result = agent.resume_from_checkpoint()

    assert str(result) == "done"
    state = CheckpointManager(checkpoint_dir).load()
    assert all(evidence["result_status"] != "unknown_result" for evidence in state.evidence)


def test_process_restart_resumes_without_duplicate_steps_or_reset_counters(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"

    def interrupt_callback(memory_step, agent):
        agent.interrupt()

    first_echo = CountingEchoTool()
    first_agent = ToolCallingAgent(
        model=EchoToolModel(),
        tools=[first_echo],
        max_steps=2,
        step_callbacks=[interrupt_callback],
        checkpoint_store=checkpoint_dir,
    )
    with pytest.raises(AgentError, match="Agent interrupted"):
        first_agent.run("do the task")

    interrupted_state = CheckpointManager(checkpoint_dir).load()
    assert interrupted_state.status == ExecutionStatus.INTERRUPTED.value
    assert interrupted_state.next_step_number == 2
    assert interrupted_state.budget.steps_used == 1
    assert interrupted_state.in_flight_step is None

    second_agent = ToolCallingAgent(
        model=FinalAnswerModel(),
        tools=[CountingEchoTool()],
        checkpoint_store=checkpoint_dir,
    )
    result = second_agent.resume_from_checkpoint()

    assert str(result) == "done"
    resumed_state = CheckpointManager(checkpoint_dir).load()
    assert resumed_state.status == ExecutionStatus.COMPLETED.value
    assert resumed_state.run_id == interrupted_state.run_id
    assert resumed_state.budget.steps_used >= 2
    assert resumed_state.step_number > interrupted_state.step_number
    assert sum(1 for item in resumed_state.memory_steps if item["step_type"] == "task") == 1


def test_terminal_checkpoint_returns_cached_output_without_calling_model(tmp_path):
    class FailingModel(Model):
        def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
            raise AssertionError("The model must not be called for a terminal checkpoint.")

    checkpoint_dir = tmp_path / "checkpoints"
    CheckpointManager(checkpoint_dir).save(
        _make_state(
            task="do the task",
            status=ExecutionStatus.COMPLETED,
            final_output="cached answer",
            step_number=1,
            next_step_number=2,
        )
    )

    agent = ToolCallingAgent(model=FailingModel(), tools=[], checkpoint_store=checkpoint_dir)
    assert agent.resume_from_checkpoint() == "cached answer"


def test_compatibility_mismatch_is_rejected(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    CheckpointManager(checkpoint_dir).save(_make_state(task="do the task", tool_names=["missing_tool"]))

    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[], checkpoint_store=checkpoint_dir)

    with pytest.raises(CheckpointCompatibilityError):
        agent.resume_from_checkpoint()


def test_no_checkpoint_store_writes_nothing(tmp_path):
    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[])
    agent.run("do the task")

    assert not (tmp_path / "checkpoints").exists()
