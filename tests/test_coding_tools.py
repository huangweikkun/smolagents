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
from pathlib import Path

import pytest

from smolagents.agents import ToolCallingAgent
from smolagents.audit import ApprovalManager, ApprovalStatus, AuditLog
from smolagents.checkpoint import CheckpointManager
from smolagents.coding_tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    RunTestTool,
    SearchCodeTool,
    WriteFileTool,
    default_coding_risk_policy,
)
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
)
from smolagents.sandbox import Sandbox, SandboxConfig
from smolagents.state_store import ExecutionStatus, StateStore
from smolagents.utils import AgentApprovalRequiredError


class WriteToolModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None, **kwargs):
        if tools_to_call_from is None:
            return ChatMessage(role=MessageRole.ASSISTANT, content="finished")
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_write",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="write_file",
                        arguments={"path": "artifacts/note.md", "content": "approved"},
                    ),
                )
            ],
        )


class FinalAnswerModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None, **kwargs):
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


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "auth" / "session-policy.ts"
    source.parent.mkdir(parents=True)
    source.write_text(
        "export function isSessionExpired(session, now) {\n  const expiresOn = session.expiresAt.slice(0, 10);\n}\n",
        encoding="utf-8",
    )
    return workspace


def _sandbox(workspace: Path, allowed_commands=frozenset({"echo", "git"})) -> Sandbox:
    return Sandbox.create(
        SandboxConfig(
            workspace_root=workspace,
            use_os_isolation=False,
            allowed_commands=allowed_commands,
        )
    )


def test_file_tools_operate_on_the_workspace_with_write_policy(tmp_path):
    workspace = _workspace(tmp_path)
    sandbox = _sandbox(workspace)
    try:
        list_files = ListFilesTool(sandbox)
        read_file = ReadFileTool(sandbox)
        search_code = SearchCodeTool(sandbox)
        write_file = WriteFileTool(sandbox, allowed_paths=("artifacts/**",), denied_paths=("tests/**",))
        edit_file = EditFileTool(sandbox, allowed_paths=("src/**",))

        assert "src/auth/session-policy.ts" in list_files.forward(recursive=True)
        content = read_file.forward("src/auth/session-policy.ts")
        assert "isSessionExpired" in content and "1 |" in content
        assert "src/auth/session-policy.ts:" in search_code.forward("isSessionExpired")

        assert write_file.forward("artifacts/note.md", "hello") == "Wrote artifacts/note.md"
        assert (workspace / "artifacts" / "note.md").read_text(encoding="utf-8") == "hello"

        with pytest.raises(Exception, match="not allowed"):
            write_file.forward("tests/bad.md", "x")
        with pytest.raises(Exception, match="inside the workspace"):
            write_file.forward("../outside.md", "x")

        edit_file.forward("src/auth/session-policy.ts", "const expiresOn", "const expiresAtValue")
        assert "expiresAtValue" in (workspace / "src" / "auth" / "session-policy.ts").read_text(encoding="utf-8")
    finally:
        sandbox.destroy()


def test_run_test_uses_the_configured_command_only(tmp_path):
    workspace = _workspace(tmp_path)
    sandbox = _sandbox(workspace, allowed_commands=frozenset({"echo"}))
    try:
        tool = RunTestTool(sandbox, ("echo", "tests ok"))
        output = tool.forward()
        assert "Exit code: 0" in output
        assert "tests ok" in output
    finally:
        sandbox.destroy()


def test_agent_pauses_for_approval_and_resumes_without_duplicate_write(tmp_path):
    workspace = _workspace(tmp_path)
    sandbox = _sandbox(workspace, allowed_commands=frozenset())
    approvals = ApprovalManager(tmp_path / "approvals.json")
    audit = AuditLog(tmp_path / "audit.jsonl")
    try:
        write_tools = [WriteFileTool(sandbox, allowed_paths=("artifacts/**",))]
        first_agent = ToolCallingAgent(
            model=WriteToolModel(),
            tools=write_tools,
            risk_policy=default_coding_risk_policy(),
            approval_manager=approvals,
            audit_log=audit,
            state_store=tmp_path / "state.json",
            checkpoint_store=tmp_path / "checkpoints",
            workspace_dir=workspace,
        )
        with pytest.raises(AgentApprovalRequiredError, match="Approval required"):
            first_agent.run("write the note")

        state = StateStore.load(tmp_path / "state.json")
        assert state.status == ExecutionStatus.WAITING_APPROVAL.value
        assert state.stop_reason == "approval_required"
        checkpoint = CheckpointManager(tmp_path / "checkpoints").load()
        assert checkpoint.status == ExecutionStatus.WAITING_APPROVAL.value
        assert checkpoint.in_flight_step["tool_calls"]
        assert not (workspace / "artifacts" / "note.md").exists()

        record = approvals.list_records(status=ApprovalStatus.PENDING)[0]
        approvals.approve(record.record_id, approver="maintainer")

        second_agent = ToolCallingAgent(
            model=FinalAnswerModel(),
            tools=[WriteFileTool(sandbox, allowed_paths=("artifacts/**",))],
            risk_policy=default_coding_risk_policy(),
            approval_manager=approvals,
            audit_log=audit,
            state_store=tmp_path / "state.json",
            checkpoint_store=tmp_path / "checkpoints",
            workspace_dir=workspace,
        )
        result = second_agent.resume_from_checkpoint()

        assert str(result) == "done"
        assert (workspace / "artifacts" / "note.md").read_text(encoding="utf-8") == "approved"
        assert StateStore.load(tmp_path / "state.json").status == ExecutionStatus.COMPLETED.value
        assert CheckpointManager(tmp_path / "checkpoints").load().status == ExecutionStatus.COMPLETED.value
    finally:
        sandbox.destroy()
