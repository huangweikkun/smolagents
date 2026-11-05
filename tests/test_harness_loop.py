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
from pathlib import Path

import pytest

from smolagents.audit import ApprovalStatus, AuditEventType
from smolagents.gate import ToolRiskLevel, ToolRiskPolicy
from smolagents.harness import (
    Harness,
    HarnessConfig,
    HarnessConfigurationError,
    ModelAssembler,
    ToolRegistry,
)
from smolagents.memory import ToolCall
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.state_store import ExecutionStatus, StateStore
from smolagents.tools import Tool


REQUIRED_SOURCES = (
    "src/routes/login.ts",
    "src/services/auth-service.ts",
    "src/repositories/user-repository.ts",
    "src/db/database.ts",
    "src/middleware/auth-middleware.ts",
)

LOGIN_FLOW_DOCUMENT = """# 登录调用链说明

- `src/routes/login.ts:8-11` 读取请求体。
- `src/services/auth-service.ts:7-34` 查用户、校验密码、签发 token。
- `src/repositories/user-repository.ts:4-9` 转调假数据库。
- `src/db/database.ts:5-20` 是内存数组模拟的数据库。
- `src/middleware/auth-middleware.ts:9-14` 校验 Bearer 头。

## 不能照搬到生产的事实

- 密码是明文拼接比较。
"""


class DummyModel(Model):
    """Minimal model so harness tests never reach a real provider."""

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")


class EchoTool(Tool):
    name = "echo_text"
    description = "Echo the given text."
    inputs = {"text": {"type": "string", "description": "Text to echo."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return text


def _source_file(line_count: int = 40) -> str:
    return "".join(f"// line {index}\n" for index in range(1, line_count + 1))


def _write_files(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _understand_repo(tmp_path: Path, document: str | None = LOGIN_FLOW_DOCUMENT) -> Path:
    files = {path: _source_file() for path in REQUIRED_SOURCES}
    if document is not None:
        files["artifacts/login-flow.md"] = document
    return _write_files(tmp_path / "repo", files)


def _harness(
    tmp_path: Path,
    workspace: Path,
    *,
    task_name: str = "understand",
    tools: tuple[Tool, ...] = (),
    **config_kwargs,
) -> Harness:
    config = HarnessConfig(
        workspace=workspace,
        task_name=task_name,
        runtime_dir=tmp_path / "runtime",
        enable_sandbox=False,
        **config_kwargs,
    )
    return Harness(config, model=DummyModel(model_id="dummy-model"), tools=tools)


def test_harness_runs_task_and_verifies_completion(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path))

    outcome = harness.run(lambda context: "done")

    assert outcome.succeeded
    assert outcome.status is ExecutionStatus.COMPLETED
    assert outcome.exit_code == 0
    assert outcome.completion is not None and outcome.completion.passed
    assert outcome.checkpoint_path is not None and outcome.checkpoint_path.is_file()
    state = StateStore.load(harness.state_path)
    assert state.status == ExecutionStatus.COMPLETED.value
    assert state.final_output == "done"


def test_harness_can_skip_its_own_state_and_checkpoint_when_executor_persists(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path), persist_state=False, persist_checkpoint=False)

    outcome = harness.run(lambda context: "done")

    assert outcome.succeeded
    assert outcome.status is ExecutionStatus.COMPLETED
    assert outcome.stop_reason == "completed"
    assert outcome.checkpoint_path is None
    assert not harness.state_path.exists()
    assert list((tmp_path / "runtime" / "checkpoints").glob("*.json")) == []


def test_harness_fails_when_completion_conditions_fail(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path, document=None))

    outcome = harness.run(lambda context: "done")

    assert not outcome.succeeded
    assert outcome.status is ExecutionStatus.FAILED
    assert outcome.exit_code == 1
    assert outcome.completion is not None and not outcome.completion.passed
    assert "deliverable_present" in outcome.error
    state = StateStore.load(harness.state_path)
    assert state.status == ExecutionStatus.FAILED.value
    assert state.stop_reason == "completion_failed"
    assert state.run_id


def test_harness_marks_run_failed_when_executor_raises(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path))

    def executor(context):
        raise RuntimeError("boom")

    outcome = harness.run(executor)

    assert outcome.status is ExecutionStatus.FAILED
    assert outcome.error == "boom"
    assert outcome.completion is None
    assert outcome.checkpoint_path is not None


def test_harness_executes_tool_through_gate(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path), tools=(EchoTool(),))

    def executor(context):
        return context.execute_tool("echo_text", {"text": "hello"}).observation

    outcome = harness.run(executor)

    assert outcome.succeeded
    assert outcome.output == "hello"


def test_harness_gate_denies_tool(tmp_path):
    harness = _harness(
        tmp_path,
        _understand_repo(tmp_path),
        tools=(EchoTool(),),
        denied_tools=frozenset({"echo_text"}),
    )

    outcome = harness.run(lambda context: context.execute_tool("echo_text", {"text": "hello"}))

    assert outcome.status is ExecutionStatus.FAILED
    assert "not permitted" in outcome.error


def test_harness_runs_sandbox_tools(tmp_path):
    config = HarnessConfig(
        workspace=_understand_repo(tmp_path),
        task_name="understand",
        runtime_dir=tmp_path / "runtime",
        enable_sandbox=True,
        use_os_isolation=False,
    )
    harness = Harness(config, model=lambda: DummyModel(model_id="dummy-model"))

    def executor(context):
        assert context.sandbox is not None
        context.execute_tool("sandbox_file_write", {"path": "draft.md", "content": "hello"})
        return context.execute_tool("sandbox_file_read", {"path": "draft.md"}).observation

    outcome = harness.run(executor)

    assert outcome.succeeded
    assert outcome.output == "hello"


def test_harness_skips_completion_for_unknown_task(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path), task_name="custom")

    outcome = harness.run(lambda context: "done")

    assert outcome.succeeded
    assert outcome.completion is None


def test_harness_persists_state_variables_in_checkpoint(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path))

    def executor(context):
        context.state_variables["note"] = "checked"
        return "done"

    harness.run(executor)

    checkpoint = harness.load_latest_checkpoint()
    assert checkpoint.state == {"note": "checked"}
    assert checkpoint.tool_names == []


def test_harness_persists_approval_records(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path))

    record = harness.request_approval(operation_type="write_file", content="draft", metadata={"task": "understand"})
    result = harness.approve(record.record_id, approver="maintainer", comment="ok")

    assert result.approved
    assert [item.record_id for item in harness.list_approvals(ApprovalStatus.APPROVED)] == [record.record_id]
    payload = json.loads((tmp_path / "runtime" / "approvals.json").read_text(encoding="utf-8"))
    assert payload["records"][0]["status"] == "approved"


def test_harness_links_audit_events_to_checkpoint_run_id(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path), tools=(EchoTool(),))

    outcome = harness.run(lambda context: context.execute_tool("echo_text", {"text": "hello"}).observation)

    assert outcome.succeeded
    events = harness.audit_log.events()
    assert [event.event_type for event in events] == [
        AuditEventType.GATE_ALLOW,
        AuditEventType.EXECUTION_STARTED,
        AuditEventType.EXECUTION_COMPLETED,
    ]
    checkpoint = harness.load_latest_checkpoint()
    assert {event.run_id for event in events} == {checkpoint.run_id}
    assert StateStore.load(harness.state_path).run_id == checkpoint.run_id
    assert all(event.tool_name == "echo_text" for event in events)


def test_harness_risk_policy_waits_for_approval_and_audits(tmp_path):
    harness = _harness(
        tmp_path,
        _understand_repo(tmp_path),
        tools=(EchoTool(),),
        risk_policy=ToolRiskPolicy(tool_levels={"echo_text": ToolRiskLevel.HIGH}),
    )

    outcome = harness.run(lambda context: context.execute_tool("echo_text", {"text": "hello"}))

    assert outcome.status is ExecutionStatus.FAILED
    assert "requires approval" in outcome.error
    assert [event.event_type for event in harness.audit_log.events()] == [
        AuditEventType.APPROVAL_REQUESTED,
        AuditEventType.GATE_WAITING_APPROVAL,
    ]
    assert len(harness.list_approvals(ApprovalStatus.PENDING)) == 1


def test_harness_budget_exceeded_sets_stop_reason(tmp_path):
    harness = _harness(
        tmp_path,
        _understand_repo(tmp_path),
        tools=(EchoTool(),),
        max_tool_calls=1,
    )

    def executor(context):
        context.execute_tool("echo_text", {"text": "one"})
        context.execute_tool("echo_text", {"text": "two"})

    outcome = harness.run(executor)

    assert outcome.status is ExecutionStatus.BUDGET_EXCEEDED
    assert outcome.exit_code == 3
    assert "budget exceeded" in outcome.error.lower()
    assert StateStore.load(harness.state_path).stop_reason == "budget_exceeded"


def test_harness_rejects_missing_workspace(tmp_path):
    with pytest.raises(HarnessConfigurationError, match="not a directory"):
        Harness(
            HarnessConfig(workspace=tmp_path / "missing", task_name="understand", runtime_dir=tmp_path / "runtime"),
            model=DummyModel(),
        )


def test_harness_rejects_non_callable_executor(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path))

    with pytest.raises(HarnessConfigurationError, match="callable"):
        harness.run("not-a-callable")


def test_harness_rejects_concurrent_runs(tmp_path):
    harness = _harness(tmp_path, _understand_repo(tmp_path))

    def executor(context):
        harness.run(lambda inner: "nested")

    outcome = harness.run(executor)

    assert outcome.status is ExecutionStatus.FAILED
    assert "already running" in outcome.error


def test_tool_registry_rejects_duplicate_and_unknown_tools():
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(Exception, match="already registered"):
        registry.register(EchoTool())
    with pytest.raises(Exception, match="not registered"):
        registry.get("missing_tool")
    assert registry.names() == ("echo_text",)
    assert (
        registry.build_gate().execute(ToolCall(name="echo_text", arguments={"text": "x"}, id="call-1")).observation
        == "x"
    )


def test_model_assembler_supports_instance_callable_and_invalid_specs():
    model = DummyModel(model_id="dummy-model")

    assert ModelAssembler(model).assemble() is model
    assert isinstance(ModelAssembler(lambda: model).assemble(), DummyModel)
    with pytest.raises(HarnessConfigurationError, match="Model specification"):
        ModelAssembler(123).assemble()
    with pytest.raises(HarnessConfigurationError, match="expected a smolagents Model"):
        ModelAssembler(lambda: "not-a-model").assemble()
