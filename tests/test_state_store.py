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

from smolagents.agents import ToolCallingAgent
from smolagents.memory import ActionStep, PlanningStep, ToolCall
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.state_store import (
    ExecutionState,
    ExecutionStatus,
    PlanRecord,
    StateStore,
    StateStoreError,
    StepEvidence,
    collect_workspace_info,
    exit_code_for,
    serialize_messages,
)
from smolagents.tools import tool
from smolagents.utils import AgentError, AgentMaxStepsError


def _read_state(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _message(text: str, role: MessageRole = MessageRole.USER) -> ChatMessage:
    return ChatMessage(role=role, content=[{"type": "text", "text": text}])


class FinalAnswerModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if tools_to_call_from is None:
            return ChatMessage(role=MessageRole.ASSISTANT, content="done")
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_1",
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
                    id="call_1",
                    type="function",
                    function=ChatMessageToolCallFunction(name="echo", arguments={"text": "hi"}),
                )
            ],
        )


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


def test_start_writes_running_state_with_task_info(tmp_path):
    store = StateStore(tmp_path / "state.json")

    state = store.start(
        task="do something",
        agent_class="ToolCallingAgent",
        agent_name="worker",
        model_id="test-model",
        max_steps=3,
        planning_interval=2,
        additional_args={"payload": "value"},
        messages=[_message("task")],
    )

    assert isinstance(state, ExecutionState)
    assert state.schema_version == 1
    assert state.status == ExecutionStatus.RUNNING.value
    assert state.exit_code is None
    assert state.task.task == "do something"
    assert state.task.agent_class == "ToolCallingAgent"
    assert state.task.max_steps == 3
    assert state.task.additional_args == {"payload": "value"}
    assert store.path.exists()

    data = _read_state(store.path)
    assert data["status"] == "running"
    assert data["exit_code"] is None
    assert data["task"]["agent_name"] == "worker"
    assert data["workspace"]["cwd"]
    assert data["messages"]


def test_record_plan_and_action_evidence(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.start(task="task", agent_class="ToolCallingAgent", max_steps=2, messages=[_message("task")])

    planning_step = PlanningStep(
        model_input_messages=[_message("plan?")],
        model_output_message=_message("the plan", MessageRole.ASSISTANT),
        plan="1. do the thing",
        timing=Timing(start_time=0.0, end_time=1.0),
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    store.record_plan(step_number=1, planning_step=planning_step, messages=[_message("task")])

    action_step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})],
        observations="hi",
        action_output="hi",
        token_usage=TokenUsage(input_tokens=20, output_tokens=2),
    )
    store.record_action(action_step=action_step, messages=[_message("task")])

    state = store.state
    assert len(state.plans) == 1
    assert isinstance(state.plans[0], PlanRecord)
    assert state.plans[0].plan == "1. do the thing"
    assert state.plans[0].token_usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    assert len(state.evidence) == 1
    assert isinstance(state.evidence[0], StepEvidence)
    assert state.evidence[0].completed is True
    assert state.evidence[0].tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": {"text": "hi"}}}
    ]
    assert state.evidence[0].observations == "hi"
    assert state.evidence[0].action_output == "hi"

    data = _read_state(store.path)
    assert data["plans"][0]["step_number"] == 1
    assert data["evidence"][0]["completed"] is True
    assert data["evidence"][0]["timing"]["duration"] == 1.0


def test_record_action_marks_failed_step(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.start(task="task", agent_class="ToolCallingAgent", max_steps=1)

    logger = AgentLogger(level=LogLevel.OFF)
    action_step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        error=AgentMaxStepsError("Reached max steps.", logger),
    )
    store.record_action(action_step=action_step)

    evidence = store.state.evidence[0]
    assert evidence.completed is False
    assert evidence.error == {"type": "AgentMaxStepsError", "message": "Reached max steps."}


@pytest.mark.parametrize(
    "status, expected_exit_code",
    [
        (ExecutionStatus.COMPLETED, 0),
        (ExecutionStatus.FAILED, 1),
        (ExecutionStatus.MAX_STEPS, 2),
        (ExecutionStatus.INTERRUPTED, 130),
    ],
)
def test_finish_sets_exit_code(tmp_path, status, expected_exit_code):
    store = StateStore(tmp_path / "state.json")
    store.start(task="task", agent_class="ToolCallingAgent", max_steps=1)

    store.finish(status=status, output="result")

    assert store.state.status == status.value
    assert store.state.exit_code == expected_exit_code
    assert store.state.finished_at is not None
    assert exit_code_for(status) == expected_exit_code

    data = _read_state(store.path)
    assert data["status"] == status.value
    assert data["exit_code"] == expected_exit_code
    assert data["final_output"] == "result"


def test_save_and_load_roundtrip(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.start(task="task", agent_class="ToolCallingAgent", max_steps=1, messages=[_message("task")])
    store.finish(status=ExecutionStatus.COMPLETED, output={"answer": 42})

    loaded = StateStore.load(store.path)

    assert isinstance(loaded, ExecutionState)
    assert loaded.status == "completed"
    assert loaded.exit_code == 0
    assert loaded.task.task == "task"
    assert loaded.final_output == {"answer": 42}
    assert loaded.messages


def test_load_rejects_invalid_state(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(StateStoreError):
        StateStore.load(bad_json)

    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(StateStoreError):
        StateStore.load(unsupported)


def test_workspace_info_records_environment_and_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "b.txt").write_text("bb", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")

    info = collect_workspace_info(workspace)

    assert info.cwd
    assert info.platform["system"]
    assert info.python["version"]
    assert info.workspace_dir == str(workspace.resolve())
    assert info.files == [
        {"relative_path": "a.txt", "size_bytes": 1},
        {"relative_path": "b.txt", "size_bytes": 2},
    ]

    without_directory = collect_workspace_info()
    assert without_directory.workspace_dir is None
    assert without_directory.files is None


def test_serialize_messages():
    messages = [_message("hello"), _message("answer", MessageRole.ASSISTANT)]

    serialized = serialize_messages(messages)

    assert serialized[0]["role"] == "user"
    assert serialized[0]["content"] == [{"type": "text", "text": "hello"}]
    assert serialized[1]["role"] == "assistant"


def test_tool_calling_agent_persists_completed_state(tmp_path):
    state_path = tmp_path / "state.json"
    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[], state_store=state_path)

    result = agent.run("do the task")

    assert str(result) == "done"
    data = _read_state(state_path)
    assert data["status"] == "completed"
    assert data["exit_code"] == 0
    assert data["task"]["agent_class"] == "ToolCallingAgent"
    assert data["evidence"]
    assert data["evidence"][0]["is_final_answer"] is True
    assert data["evidence"][0]["completed"] is True
    assert data["messages"]
    assert data["workspace"]["cwd"]


def test_tool_calling_agent_persists_interrupted_state(tmp_path, echo_tool):
    state_path = tmp_path / "state.json"

    def interrupt_callback(memory_step, agent):
        agent.interrupt()

    agent = ToolCallingAgent(
        model=EchoToolModel(),
        tools=[echo_tool],
        max_steps=2,
        step_callbacks=[interrupt_callback],
        state_store=state_path,
    )

    with pytest.raises(AgentError, match="Agent interrupted"):
        agent.run("do the task")

    data = _read_state(state_path)
    assert data["status"] == "interrupted"
    assert data["exit_code"] == 130
    assert data["evidence"]


def test_tool_calling_agent_persists_max_steps_state(tmp_path, echo_tool):
    state_path = tmp_path / "state.json"
    agent = ToolCallingAgent(
        model=EchoToolModel(),
        tools=[echo_tool],
        max_steps=1,
        state_store=state_path,
    )

    agent.run("do the task")

    data = _read_state(state_path)
    assert data["status"] == "max_steps"
    assert data["exit_code"] == 2
    assert data["evidence"][-1]["error"]["type"] == "AgentMaxStepsError"


def test_tool_calling_agent_without_state_store_writes_nothing(tmp_path):
    agent = ToolCallingAgent(model=FinalAnswerModel(), tools=[])

    agent.run("do the task")

    assert agent.state_store is None
    assert list(tmp_path.glob("state.json")) == []
