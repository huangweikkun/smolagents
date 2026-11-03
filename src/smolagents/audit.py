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
"""Approval and audit module.

[`ApprovalManager`] persists approval records for user actions that require review, exposes interfaces to approve or
reject a request, and invokes registered post-approval handlers for the corresponding operation type.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .utils import make_json_serializable


SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)

ApprovalHandler = Callable[["ApprovalRecord"], Any]


class AuditError(Exception):
    """Base class for approval and audit errors."""


class ApprovalValidationError(AuditError):
    """Raised when approval parameters fail validation."""


class ApprovalNotFoundError(AuditError):
    """Raised when an approval record cannot be found."""


class ApprovalStateError(AuditError):
    """Raised when an approval record is not in a state that allows the requested transition."""


class ApprovalHandlerError(AuditError):
    """Raised when the post-approval business handler fails."""


class AuditPersistenceError(AuditError):
    """Raised when approval records cannot be read from or written to disk."""


class ApprovalStatus(str, Enum):
    """Lifecycle status of an approval record."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApprovalValidationError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApprovalValidationError(f"'{field_name}' must be a string when provided.")
    value = value.strip()
    return value or None


def _validate_content(content: Any) -> Any:
    if content is None:
        raise ApprovalValidationError("'content' is required and cannot be None.")
    if isinstance(content, str) and not content.strip():
        raise ApprovalValidationError("'content' must not be empty when provided as a string.")
    return content


@dataclass(frozen=True)
class ApprovalRecord:
    """A persisted approval record.

    Args:
        record_id (`str`): Unique identifier of the approval record.
        subject (`str`): The subject requesting approval.
        operation_type (`str`): Type of the operation requiring approval.
        content (`Any`): The content of the request.
        requested_at (`str`): ISO-8601 timestamp of when approval was requested.
        applicant (`str`): The user who requested approval.
        updated_at (`str`): ISO-8601 timestamp of the last update.
        status ([`ApprovalStatus`]): Current approval status.
        approver (`str`, *optional*): The user who approved or rejected the request.
        comment (`str`, *optional*): Approval comment supplied by the approver.
        reason (`str`, *optional*): Rejection reason supplied by the approver.
        decided_at (`str`, *optional*): ISO-8601 timestamp of the decision.
        metadata (`dict`, *optional*): Additional operation-specific data.
    """

    record_id: str
    subject: str
    operation_type: str
    content: Any
    requested_at: str
    applicant: str
    updated_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str | None = None
    comment: str | None = None
    reason: str | None = None
    decided_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        """Whether the record is still awaiting a decision."""
        return self.status is ApprovalStatus.PENDING

    def dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "subject": self.subject,
            "operation_type": self.operation_type,
            "content": make_json_serializable(self.content),
            "requested_at": self.requested_at,
            "applicant": self.applicant,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "approver": self.approver,
            "comment": self.comment,
            "reason": self.reason,
            "decided_at": self.decided_at,
            "metadata": make_json_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRecord":
        return cls(
            record_id=data["record_id"],
            subject=data["subject"],
            operation_type=data["operation_type"],
            content=data.get("content"),
            requested_at=data["requested_at"],
            applicant=data["applicant"],
            updated_at=data.get("updated_at") or data["requested_at"],
            status=ApprovalStatus(data.get("status", ApprovalStatus.PENDING.value)),
            approver=data.get("approver"),
            comment=data.get("comment"),
            reason=data.get("reason"),
            decided_at=data.get("decided_at"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class ApprovalResult:
    """Result of processing an approval decision.

    Args:
        record ([`ApprovalRecord`]): The updated approval record.
        approved (`bool`): Whether the request was approved.
        message (`str`): Human-readable processing result.
        handler_result (`Any`, *optional*): Return value of the post-approval handler, if any.
    """

    record: ApprovalRecord
    approved: bool
    message: str
    handler_result: Any = None


class ApprovalManager:
    """Manage persisted approval records and their decisions.

    Args:
        path (`str` or `Path`, *optional*, defaults to `"approvals.json"`):
            JSON file used to persist approval records.
        handlers (`Mapping[str, ApprovalHandler]`, *optional*):
            Post-approval handlers keyed by operation type. Handlers are called after an approval is persisted.
    """

    def __init__(
        self,
        path: str | Path = "approvals.json",
        *,
        handlers: Mapping[str, ApprovalHandler] | None = None,
    ):
        self.path = Path(path)
        self._records: dict[str, ApprovalRecord] = {}
        self._handlers: dict[str, ApprovalHandler] = {}
        self._lock = threading.RLock()
        if self.path.exists():
            self._load()
        for operation_type, handler in (handlers or {}).items():
            self.register_handler(operation_type, handler)

    def request_approval(
        self,
        *,
        subject: str,
        operation_type: str,
        content: Any,
        applicant: str,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        """Create and persist a pending approval record for a user action."""
        subject = _require_text(subject, "subject")
        operation_type = _require_text(operation_type, "operation_type")
        applicant = _require_text(applicant, "applicant")
        content = _validate_content(content)
        if metadata is not None and not isinstance(metadata, dict):
            raise ApprovalValidationError("'metadata' must be a dictionary when provided.")

        now = _utc_now()
        record = ApprovalRecord(
            record_id=uuid.uuid4().hex,
            subject=subject,
            operation_type=operation_type,
            content=content,
            requested_at=now,
            applicant=applicant,
            updated_at=now,
            status=ApprovalStatus.PENDING,
            metadata=metadata or {},
        )
        with self._lock:
            self._records[record.record_id] = record
            self._save()
        logger.info(
            "Approval record '%s' created for operation '%s' by '%s'",
            record.record_id,
            operation_type,
            applicant,
        )
        return record

    def approve(self, record_id: str, approver: str, comment: str | None = None) -> ApprovalResult:
        """Approve a pending record, persist the decision, and run the post-approval handler."""
        record_id = _require_text(record_id, "record_id")
        approver = _require_text(approver, "approver")
        comment = _optional_text(comment, "comment")

        with self._lock:
            record = self._require_pending(record_id)
            now = _utc_now()
            updated_record = replace(
                record,
                status=ApprovalStatus.APPROVED,
                approver=approver,
                comment=comment,
                reason=None,
                decided_at=now,
                updated_at=now,
            )
            self._records[record_id] = updated_record
            self._save()

        logger.info("Approval record '%s' approved by '%s'", record_id, approver)
        handler_result = self._run_post_approval_handler(updated_record)
        return ApprovalResult(
            record=updated_record,
            approved=True,
            message=f"Approval record '{record_id}' approved.",
            handler_result=handler_result,
        )

    def reject(self, record_id: str, approver: str, reason: str) -> ApprovalResult:
        """Reject a pending record, persist the decision, and log the rejection."""
        record_id = _require_text(record_id, "record_id")
        approver = _require_text(approver, "approver")
        reason = _require_text(reason, "reason")

        with self._lock:
            record = self._require_pending(record_id)
            now = _utc_now()
            updated_record = replace(
                record,
                status=ApprovalStatus.REJECTED,
                approver=approver,
                comment=None,
                reason=reason,
                decided_at=now,
                updated_at=now,
            )
            self._records[record_id] = updated_record
            self._save()

        logger.warning("Approval record '%s' rejected by '%s': %s", record_id, approver, reason)
        return ApprovalResult(
            record=updated_record,
            approved=False,
            message=f"Approval record '{record_id}' rejected: {reason}",
        )

    def get(self, record_id: str) -> ApprovalRecord:
        """Return an approval record by id."""
        record_id = _require_text(record_id, "record_id")
        with self._lock:
            record = self._records.get(record_id)
        if record is None:
            raise ApprovalNotFoundError(f"Approval record '{record_id}' was not found.")
        return record

    def list_records(
        self,
        status: ApprovalStatus | str | None = None,
        operation_type: str | None = None,
    ) -> list[ApprovalRecord]:
        """List approval records, optionally filtered by status and operation type."""
        normalized_status: ApprovalStatus | None = None
        if status is not None:
            try:
                normalized_status = ApprovalStatus(status)
            except ValueError as e:
                raise ApprovalValidationError(f"Unknown approval status: {status}") from e
        normalized_operation_type = _optional_text(operation_type, "operation_type")

        with self._lock:
            records = list(self._records.values())
        if normalized_status is not None:
            records = [record for record in records if record.status is normalized_status]
        if normalized_operation_type is not None:
            records = [record for record in records if record.operation_type == normalized_operation_type]
        return records

    def register_handler(self, operation_type: str, handler: ApprovalHandler) -> None:
        """Register a post-approval handler for an operation type."""
        operation_type = _require_text(operation_type, "operation_type")
        if not callable(handler):
            raise ApprovalValidationError("'handler' must be callable.")
        with self._lock:
            self._handlers[operation_type] = handler

    def save(self) -> None:
        """Persist the current records to disk."""
        with self._lock:
            self._save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def _require_pending(self, record_id: str) -> ApprovalRecord:
        record = self._records.get(record_id)
        if record is None:
            raise ApprovalNotFoundError(f"Approval record '{record_id}' was not found.")
        if not record.is_pending:
            raise ApprovalStateError(f"Approval record '{record_id}' has already been {record.status.value}.")
        return record

    def _run_post_approval_handler(self, record: ApprovalRecord) -> Any:
        handler = self._handlers.get(record.operation_type)
        if handler is None:
            return None
        try:
            return handler(record)
        except Exception as e:
            logger.exception(
                "Post-approval handler for operation '%s' failed for record '%s'",
                record.operation_type,
                record.record_id,
            )
            raise ApprovalHandlerError(
                f"Post-approval handler for operation '{record.operation_type}' failed: {e}"
            ) from e

    def _save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "records": [record.dict() for record in self._records.values()],
        }
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise AuditPersistenceError(f"Could not create approval directory '{parent}': {e}") from e

        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=f".{self.path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.path)
            tmp_path = None
        except (OSError, TypeError, ValueError) as e:
            raise AuditPersistenceError(f"Could not write approval records to '{self.path}': {e}") from e
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise AuditPersistenceError(f"Could not read approval records from '{self.path}': {e}") from e

        if not isinstance(raw, dict):
            raise AuditPersistenceError(f"Approval file '{self.path}' must contain a JSON object.")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise AuditPersistenceError(
                f"Unsupported approval schema version {raw.get('schema_version')!r} "
                f"in '{self.path}', expected {SCHEMA_VERSION}."
            )
        raw_records = raw.get("records")
        if not isinstance(raw_records, list):
            raise AuditPersistenceError(f"Approval file '{self.path}' must contain a 'records' list.")

        records: dict[str, ApprovalRecord] = {}
        try:
            for item in raw_records:
                record = ApprovalRecord.from_dict(item)
                records[record.record_id] = record
        except (KeyError, TypeError, ValueError) as e:
            raise AuditPersistenceError(f"Malformed approval record in '{self.path}': {e}") from e
        self._records = records


# Alternate name for integrations that describe the module as an audit service.
AuditManager = ApprovalManager


__all__ = [
    "ApprovalHandler",
    "ApprovalManager",
    "ApprovalNotFoundError",
    "ApprovalRecord",
    "ApprovalResult",
    "ApprovalStateError",
    "ApprovalStatus",
    "ApprovalValidationError",
    "ApprovalHandlerError",
    "AuditError",
    "AuditManager",
    "AuditPersistenceError",
]
