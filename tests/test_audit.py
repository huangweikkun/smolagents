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
import logging

import pytest

from smolagents.audit import (
    ApprovalHandlerError,
    ApprovalManager,
    ApprovalNotFoundError,
    ApprovalRecord,
    ApprovalStateError,
    ApprovalStatus,
    ApprovalValidationError,
    AuditEventType,
    AuditLog,
    AuditPersistenceError,
)
from smolagents.gate import ToolExecutionGate
from smolagents.memory import ToolCall


def _read_records(path):
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def test_request_approval_creates_and_persists_record(tmp_path):
    path = tmp_path / "approvals.json"
    manager = ApprovalManager(path)

    record = manager.request_approval(
        subject="workspace-a",
        operation_type="delete_workspace",
        content={"workspace_id": 42},
        applicant="alice",
        metadata={"source": "api"},
    )

    assert isinstance(record, ApprovalRecord)
    assert record.record_id
    assert record.subject == "workspace-a"
    assert record.operation_type == "delete_workspace"
    assert record.content == {"workspace_id": 42}
    assert record.applicant == "alice"
    assert record.status == ApprovalStatus.PENDING
    assert record.is_pending
    assert record.requested_at
    assert record.updated_at == record.requested_at
    assert record.metadata == {"source": "api"}
    assert manager.get(record.record_id) == record
    assert path.exists()

    persisted = _read_records(path)[0]
    assert persisted["record_id"] == record.record_id
    assert persisted["status"] == "pending"
    assert persisted["subject"] == "workspace-a"
    assert persisted["content"] == {"workspace_id": 42}

    reloaded_manager = ApprovalManager(path)
    assert reloaded_manager.get(record.record_id) == record


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subject": "", "operation_type": "op", "content": "c", "applicant": "alice"},
        {"subject": "s", "operation_type": "  ", "content": "c", "applicant": "alice"},
        {"subject": "s", "operation_type": "op", "content": "c", "applicant": ""},
        {"subject": "s", "operation_type": "op", "content": None, "applicant": "alice"},
        {"subject": "s", "operation_type": "op", "content": "   ", "applicant": "alice"},
        {"subject": "s", "operation_type": "op", "content": "c", "applicant": "alice", "metadata": []},
    ],
)
def test_request_approval_validates_parameters(tmp_path, kwargs):
    manager = ApprovalManager(tmp_path / "approvals.json")

    with pytest.raises(ApprovalValidationError):
        manager.request_approval(**kwargs)


def test_approve_updates_record_and_runs_handler(tmp_path):
    path = tmp_path / "approvals.json"
    handled_records = []

    def handler(record):
        handled_records.append(record)
        return {"processed": True}

    manager = ApprovalManager(path, handlers={"delete_workspace": handler})
    record = manager.request_approval(
        subject="workspace-a",
        operation_type="delete_workspace",
        content={"workspace_id": 42},
        applicant="alice",
    )

    result = manager.approve(record.record_id, "bob", comment="looks good")

    assert result.approved is True
    assert result.message
    assert result.handler_result == {"processed": True}
    assert len(handled_records) == 1
    assert handled_records[0].status == ApprovalStatus.APPROVED

    updated = manager.get(record.record_id)
    assert updated.status == ApprovalStatus.APPROVED
    assert updated.approver == "bob"
    assert updated.comment == "looks good"
    assert updated.reason is None
    assert updated.decided_at is not None
    assert updated.updated_at == updated.decided_at

    persisted = _read_records(path)[0]
    assert persisted["status"] == "approved"
    assert persisted["approver"] == "bob"
    assert persisted["comment"] == "looks good"


def test_reject_updates_record_and_logs_rejection(tmp_path, caplog):
    path = tmp_path / "approvals.json"
    manager = ApprovalManager(path)
    record = manager.request_approval(
        subject="workspace-a",
        operation_type="delete_workspace",
        content={"workspace_id": 42},
        applicant="alice",
    )

    caplog.set_level(logging.WARNING, logger="smolagents.audit")
    result = manager.reject(record.record_id, "bob", "insufficient permissions")

    assert result.approved is False
    assert record.record_id in result.message
    assert "insufficient permissions" in result.message

    updated = manager.get(record.record_id)
    assert updated.status == ApprovalStatus.REJECTED
    assert updated.approver == "bob"
    assert updated.reason == "insufficient permissions"
    assert updated.comment is None
    assert updated.decided_at is not None

    persisted = _read_records(path)[0]
    assert persisted["status"] == "rejected"
    assert persisted["reason"] == "insufficient permissions"
    assert record.record_id in caplog.text
    assert "insufficient permissions" in caplog.text


def test_reject_requires_reason(tmp_path):
    manager = ApprovalManager(tmp_path / "approvals.json")
    record = manager.request_approval(subject="s", operation_type="op", content="c", applicant="alice")

    with pytest.raises(ApprovalValidationError):
        manager.reject(record.record_id, "bob", "  ")


def test_unknown_record_raises_not_found(tmp_path):
    manager = ApprovalManager(tmp_path / "approvals.json")

    with pytest.raises(ApprovalNotFoundError):
        manager.approve("missing", "bob")
    with pytest.raises(ApprovalNotFoundError):
        manager.reject("missing", "bob", "no")
    with pytest.raises(ApprovalNotFoundError):
        manager.get("missing")


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_decided_record_cannot_be_decided_again(tmp_path, decision):
    manager = ApprovalManager(tmp_path / "approvals.json")
    record = manager.request_approval(subject="s", operation_type="op", content="c", applicant="alice")
    manager.approve(record.record_id, "bob")

    with pytest.raises(ApprovalStateError):
        if decision == "approve":
            manager.approve(record.record_id, "carol")
        else:
            manager.reject(record.record_id, "carol", "no")


def test_handler_failure_keeps_record_approved(tmp_path):
    path = tmp_path / "approvals.json"

    def failing_handler(record):
        raise RuntimeError("downstream unavailable")

    manager = ApprovalManager(path, handlers={"deploy": failing_handler})
    record = manager.request_approval(
        subject="service", operation_type="deploy", content={"version": "1.0"}, applicant="alice"
    )

    with pytest.raises(ApprovalHandlerError):
        manager.approve(record.record_id, "bob")

    assert manager.get(record.record_id).status == ApprovalStatus.APPROVED
    assert _read_records(path)[0]["status"] == "approved"


def test_list_records_filters(tmp_path):
    manager = ApprovalManager(tmp_path / "approvals.json")
    first = manager.request_approval(subject="a", operation_type="deploy", content="c", applicant="alice")
    second = manager.request_approval(subject="b", operation_type="delete", content="c", applicant="bob")
    manager.approve(first.record_id, "carol")
    first = manager.get(first.record_id)

    assert len(manager) == 2
    assert manager.list_records() == [first, second]
    assert manager.list_records(status=ApprovalStatus.PENDING) == [second]
    assert manager.list_records(status="approved") == [first]
    assert manager.list_records(operation_type="delete") == [second]
    with pytest.raises(ApprovalValidationError):
        manager.list_records(status="unknown")


def test_register_handler_validates_parameters(tmp_path):
    manager = ApprovalManager(tmp_path / "approvals.json")

    with pytest.raises(ApprovalValidationError):
        manager.register_handler("", lambda record: None)
    with pytest.raises(ApprovalValidationError):
        manager.register_handler("op", "not callable")


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"schema_version": 99}),
        json.dumps({"schema_version": 1}),
    ],
)
def test_load_rejects_invalid_files(tmp_path, content):
    path = tmp_path / "approvals.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(AuditPersistenceError):
        ApprovalManager(path)


def test_audit_log_appends_and_reloads_events(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, run_id="run-1")

    log.append(
        event_type=AuditEventType.GATE_ALLOW,
        actor="gate",
        tool_call_id="call-1",
        tool_name="echo",
        decision="allowed",
        message="allowed",
    )
    log.append(
        event_type=AuditEventType.GATE_DENY,
        actor="gate",
        tool_call_id="call-2",
        tool_name="echo",
        rule="permission",
        decision="denied",
        message="not permitted",
    )

    assert len(log) == 2
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    reloaded = AuditLog(path)
    assert len(reloaded) == 2
    events = reloaded.events(run_id="run-1", tool_call_id="call-2")
    assert len(events) == 1
    assert events[0].rule == "permission"
    assert events[0].decision == "denied"


def test_audit_log_never_rewrites_existing_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(event_type=AuditEventType.GATE_ALLOW, actor="gate", tool_call_id="call-1", decision="allowed")
    first_content = path.read_text(encoding="utf-8")

    log.append(
        event_type=AuditEventType.EXECUTION_COMPLETED, actor="gate", tool_call_id="call-1", decision="completed"
    )

    assert path.read_text(encoding="utf-8").startswith(first_content)


def test_approval_manager_audits_requests_and_decisions(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl", run_id="run-1")
    manager = ApprovalManager(tmp_path / "approvals.json", audit_log=log)
    record = manager.request_approval(
        subject="tool",
        operation_type="write_file",
        content={"path": "draft.md"},
        applicant="gate",
        metadata={"tool_call_id": "call-1", "run_id": "run-1"},
    )

    manager.approve(record.record_id, approver="maintainer")

    events = log.events(run_id="run-1", tool_call_id="call-1")
    assert [event.event_type for event in events] == [
        AuditEventType.APPROVAL_REQUESTED,
        AuditEventType.APPROVAL_APPROVED,
    ]
    assert events[0].decision == "pending"
    assert events[-1].decision == "approved"
    assert events[-1].actor == "maintainer"


def test_gate_denial_is_audited_with_rule_decision_actor_and_call_id(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl", run_id="run-1")
    gate = ToolExecutionGate(tools=[], audit_log=log, run_id="run-1", actor="gate")

    outcome = gate.try_execute(ToolCall(name="missing_tool", arguments={}, id="call-1"))

    assert not outcome.decision.allowed
    events = log.events(run_id="run-1", tool_call_id="call-1")
    assert len(events) == 1
    assert events[0].event_type is AuditEventType.GATE_DENY
    assert events[0].rule == "unknown_tool"
    assert events[0].decision == "denied"
    assert events[0].actor == "gate"
    assert events[0].tool_call_id == "call-1"


def test_audit_log_rejects_malformed_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("{not json\n", encoding="utf-8")

    with pytest.raises(AuditPersistenceError, match="Malformed audit event"):
        AuditLog(path)
