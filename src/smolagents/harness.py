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
"""Unified execution harness that assembles the runtime of a business task.

[`Harness`] wires the following modules into a single entry point and keeps their interfaces separate:

1. model assembly: [`ModelAssembler`] turns a model instance, a factory, or a model id into a [`Model`]
2. tool registration: [`ToolRegistry`] validates tool names, rejects duplicates and builds the execution gate
3. gate pipeline: [`ToolExecutionGate`] validates arguments and permissions before calling a tool
4. sandbox isolation: [`Sandbox`] hosts the sandbox-backed file and command tools
5. state management: [`StateStore`] persists the run status, exit code and workspace inventory
6. checkpointing: [`CheckpointManager`] stores per-run JSON snapshots with a checksum and latest pointer
7. audit trail: [`ApprovalManager`] persists approval requests and decisions
8. completion verification: [`CompletionVerifier`] checks the workspace after the executor returns

Example:
    ```py
    from smolagents.harness import Harness, HarnessConfig
    from smolagents.models import InferenceClientModel

    harness = Harness(
        HarnessConfig(workspace="samples/1_login_demo", task_name="understand"),
        model=InferenceClientModel(model_id="Qwen/Qwen2.5-7B-Instruct"),
    )

    def executor(context):
        # Every tool call goes through the gate and is recorded with the sandbox tool names.
        context.execute_tool("sandbox_file_write", {"path": "draft.md", "content": "hello"})
        return "done"

    outcome = harness.run(executor)
    print(outcome.succeeded, outcome.completion.summary() if outcome.completion else "")
    ```

Individual features are usable on their own:

    ```py
    # model assembly: instance, factory, or model id (falling back to InferenceClientModel)
    model = ModelAssembler("Qwen/Qwen2.5-7B-Instruct").assemble()

    # tool registration and gate: duplicates and unknown names are rejected
    registry = ToolRegistry([EchoTool()])
    registry.build_gate().execute(ToolCall(name="echo_text", arguments={"text": "hi"}, id="call-1"))

    # audit trail: requests and decisions are persisted in approvals.json
    record = harness.request_approval(operation_type="write_file", content="draft")
    harness.approve(record.record_id, approver="maintainer")

    # state and checkpoint snapshots
    StateStore.load(harness.state_path)
    harness.load_latest_checkpoint()

    # completion verification alone
    harness.completion_verifier.verify("understand", "samples/1_login_demo")
    ```
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import (
    ApprovalManager,
    ApprovalRecord,
    ApprovalResult,
    ApprovalStatus,
    AuditError,
)
from .checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointState,
    build_budget,
    new_run_id,
    serialize_error,
    serialize_state,
    utc_now,
)
from .completion import (
    CompletionError,
    CompletionReport,
    CompletionVerifier,
    UnknownCompletionTaskError,
    create_default_completion_verifier,
    snapshot_files,
)
from .gate import ToolExecutionGate, ToolExecutionResult, ToolPermissionPolicy
from .memory import ToolCall
from .models import Model
from .sandbox import Sandbox, SandboxConfig, SandboxError, create_sandbox_tools
from .state_store import ExecutionStatus, StateStore, StateStoreError, exit_code_for
from .tools import BaseTool


logger = logging.getLogger(__name__)

ModelSpec = Model | str | Callable[[], Model]
TaskExecutor = Callable[["HarnessContext"], Any]


class HarnessError(Exception):
    """Base class for harness errors."""


class HarnessConfigurationError(HarnessError):
    """Raised when the harness configuration or model specification is invalid."""


class HarnessToolRegistryError(HarnessError):
    """Raised when a tool cannot be registered or looked up."""


class HarnessSandboxError(HarnessError):
    """Raised when the sandbox cannot be started or configured."""


class HarnessStateError(HarnessError):
    """Raised when state, checkpoint, or audit data cannot be persisted."""


class CompletionVerificationError(HarnessError):
    """Raised when the executor returned but the business task is not actually completed."""


class ModelAssembler:
    """Resolve a model specification into a [`Model`] instance.

    Args:
        spec (`Model`, `str` or `Callable`):
            A ready-to-use model, a zero-argument factory returning a [`Model`], or a model id.
        factories (`Mapping[str, Callable[[], Model]]`, *optional*):
            Named factories consulted before falling back to the default inference client factory for string specs.

    Example:
        ```py
        assembler = ModelAssembler(lambda: my_model)
        model = assembler.assemble()
        ```
    """

    def __init__(self, spec: ModelSpec, factories: Mapping[str, Callable[[], Model]] | None = None):
        self.spec = spec
        self.factories: dict[str, Callable[[], Model]] = dict(factories or {})

    def assemble(self) -> Model:
        """Build the model, raising [`HarnessConfigurationError`] when the result is not a [`Model`]."""
        model = self._build()
        if not isinstance(model, Model):
            raise HarnessConfigurationError(
                f"Model specification resolved to {type(model).__name__}, expected a smolagents Model instance."
            )
        return model

    def _build(self) -> Any:
        spec = self.spec
        if isinstance(spec, Model):
            return spec
        if isinstance(spec, str):
            factory = self.factories.get(spec)
            if factory is not None:
                return factory()
            return self._build_default(spec)
        if callable(spec):
            return spec()
        raise HarnessConfigurationError(
            "Model specification must be a Model instance, a callable returning a Model, or a model id string."
        )

    @staticmethod
    def _build_default(model_id: str) -> Model:
        try:
            from .models import InferenceClientModel
        except ImportError as e:
            raise HarnessConfigurationError(
                f"Cannot build model '{model_id}' because InferenceClientModel is unavailable: {e}"
            ) from e
        return InferenceClientModel(model_id=model_id)


class ToolRegistry:
    """Name-keyed tool registry with duplicate protection and gate construction.

    Args:
        tools (`Iterable[BaseTool]`, *optional*): Tools registered on construction.
    """

    def __init__(self, tools: Iterable[BaseTool] = ()):
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register `tool` by its `name`, rejecting non-tools and duplicate names."""
        if not isinstance(tool, BaseTool):
            raise HarnessToolRegistryError(f"Expected a BaseTool instance, got {type(tool).__name__}.")
        name = getattr(tool, "name", "")
        if not isinstance(name, str) or not name:
            raise HarnessToolRegistryError("A tool must declare a non-empty string name.")
        if name in self._tools:
            raise HarnessToolRegistryError(f"A tool named '{name}' is already registered.")
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool, raising [`HarnessToolRegistryError`] when it is unknown."""
        if name not in self._tools:
            raise HarnessToolRegistryError(f"Tool '{name}' is not registered.")
        del self._tools[name]

    def get(self, name: str) -> BaseTool:
        """Return a registered tool by name."""
        tool = self._tools.get(name)
        if tool is None:
            raise HarnessToolRegistryError(
                f"Tool '{name}' is not registered; available tools: {', '.join(self.names()) or '<none>'}."
            )
        return tool

    def names(self) -> tuple[str, ...]:
        """Registered tool names, sorted."""
        return tuple(sorted(self._tools))

    def values(self) -> tuple[BaseTool, ...]:
        """Registered tools in name order."""
        return tuple(self._tools[name] for name in self.names())

    def as_dict(self) -> dict[str, BaseTool]:
        """Copy of the registry as a name-to-tool mapping."""
        return dict(self._tools)

    def build_gate(
        self,
        permissions: ToolPermissionPolicy | None = None,
        state: dict[str, Any] | None = None,
    ) -> ToolExecutionGate:
        """Create a [`ToolExecutionGate`] backed by the registered tools."""
        return ToolExecutionGate(self.as_dict(), permissions=permissions, state=state)


@dataclass
class HarnessConfig:
    """Configuration of a [`Harness`].

    Args:
        workspace (`str` or `Path`): Repository the executor works in; also the completion verification target.
        task_name (`str`): Business task id resolved by [`CompletionVerifier`], e.g. `understand` or `fix_tests`.
        max_steps (`int`): Step budget recorded in state and checkpoints.
        agent_class (`str`): Agent class label recorded in state and checkpoints.
        agent_name (`str`): Agent name recorded in state and checkpoints.
        runtime_dir (`str` or `Path`, *optional*): Directory for state, checkpoints and approvals. Defaults to a
            temporary directory, which keeps harness artifacts out of the verified workspace.
        state_path (`str` or `Path`, *optional*): Explicit execution state file.
        checkpoint_dir (`str` or `Path`, *optional*): Explicit checkpoint directory.
        approvals_path (`str` or `Path`, *optional*): Explicit approval records file.
        approval_applicant (`str`): Applicant recorded on approval requests.
        enable_sandbox (`bool`): Whether a [`Sandbox`] is started for the run.
        register_sandbox_tools (`bool`): Whether the sandbox file and command tools are registered per run.
        use_os_isolation (`bool`): Whether the sandbox requires an OS-level isolation backend.
        sandbox_config ([`SandboxConfig`], *optional*): Explicit sandbox configuration.
        verify_completion (`bool`): Whether the business task is verified after the executor returns.
        allowed_tools (`frozenset[str]`, *optional*): Gate allowlist; `None` allows every non-denied tool.
        denied_tools (`frozenset[str]`): Gate denylist.
    """

    workspace: str | Path
    task_name: str
    max_steps: int = 10
    agent_class: str = "Harness"
    agent_name: str = "harness"
    runtime_dir: str | Path | None = None
    state_path: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    approvals_path: str | Path | None = None
    approval_applicant: str = "harness"
    enable_sandbox: bool = True
    register_sandbox_tools: bool = True
    use_os_isolation: bool = True
    sandbox_config: SandboxConfig | None = None
    verify_completion: bool = True
    allowed_tools: frozenset[str] | None = None
    denied_tools: frozenset[str] = frozenset()


@dataclass
class HarnessContext:
    """Handles passed to the task executor for a single run.

    Args:
        task (`str`): Task description recorded with the run.
        workspace (`Path`): Repository under test.
        model ([`Model`]): Assembled model.
        sandbox ([`Sandbox`], *optional*): Active sandbox, when enabled.
        tools ([`ToolRegistry`]): Tools registered for this run.
        gate ([`ToolExecutionGate`]): Gate used to execute tool calls.
        state ([`StateStore`]): State store of the run.
        checkpoints ([`CheckpointManager`]): Checkpoint manager of the harness.
        approvals ([`ApprovalManager`]): Approval and audit manager of the harness.
        state_variables (`dict[str, Any]`): Variables shared with the gate and persisted in checkpoints.
        run_id (`str`): Identifier of the run.
    """

    task: str
    workspace: Path
    model: Model
    sandbox: Sandbox | None
    tools: ToolRegistry
    gate: ToolExecutionGate
    state: StateStore
    checkpoints: CheckpointManager
    approvals: ApprovalManager
    state_variables: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    def execute_tool(
        self, name: str, arguments: dict[str, Any] | Any, *, call_id: str | None = None
    ) -> ToolExecutionResult:
        """Execute a tool call through the gate, validating existence, arguments and permissions."""
        return self.gate.execute(ToolCall(name=name, arguments=arguments, id=call_id or uuid.uuid4().hex))


@dataclass(frozen=True)
class HarnessOutcome:
    """Result of a single [`Harness`] run.

    Args:
        task (`str`): Task description of the run.
        status ([`ExecutionStatus`]): Final execution status.
        exit_code (`int`, *optional*): Process-style exit code derived from `status`.
        output (`Any`): Value returned by the executor.
        completion ([`CompletionReport`], *optional*): Completion verification report, when available.
        error (`str`, *optional*): Error message when execution or completion verification failed.
        state_path (`Path`): Execution state file written for the run.
        checkpoint_path (`Path`, *optional*): Checkpoint written for the run.
    """

    task: str
    status: ExecutionStatus
    exit_code: int | None
    output: Any
    completion: CompletionReport | None
    error: str | None
    state_path: Path
    checkpoint_path: Path | None

    @property
    def succeeded(self) -> bool:
        """Whether the run finished and, when verified, the business task is completed."""
        if self.status is not ExecutionStatus.COMPLETED:
            return False
        return self.completion.passed if self.completion is not None else True


class Harness:
    """Assemble and run a business task with state, checkpoint, audit and completion verification.

    Args:
        config ([`HarnessConfig`]): Harness configuration.
        model (`Model`, `str` or `Callable`): Model specification resolved by [`ModelAssembler`].
        tools (`Iterable[BaseTool]`, *optional*): Tools registered for every run.
        verifier ([`CompletionVerifier`], *optional*): Completion verifier; defaults to the supported business tasks.
        model_factories (`Mapping[str, Callable[[], Model]]`, *optional*): Named model factories used for string specs.
    """

    def __init__(
        self,
        config: HarnessConfig,
        *,
        model: ModelSpec,
        tools: Iterable[BaseTool] = (),
        verifier: CompletionVerifier | None = None,
        model_factories: Mapping[str, Callable[[], Model]] | None = None,
    ):
        self.config = config
        self._workspace = Path(config.workspace).expanduser().resolve()
        if not self._workspace.is_dir():
            raise HarnessConfigurationError(f"Workspace '{self._workspace}' is not a directory.")
        if config.max_steps <= 0:
            raise HarnessConfigurationError("'max_steps' must be positive.")

        runtime_dir = (
            Path(config.runtime_dir).expanduser().resolve()
            if config.runtime_dir is not None
            else Path(tempfile.mkdtemp(prefix="smolagents-harness-")).resolve()
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = runtime_dir
        self.state_path = (
            Path(config.state_path).expanduser().resolve()
            if config.state_path is not None
            else runtime_dir / "state.json"
        )
        checkpoint_dir = (
            Path(config.checkpoint_dir).expanduser().resolve()
            if config.checkpoint_dir is not None
            else runtime_dir / "checkpoints"
        )
        approvals_path = (
            Path(config.approvals_path).expanduser().resolve()
            if config.approvals_path is not None
            else runtime_dir / "approvals.json"
        )

        self._model_assembler = ModelAssembler(model, model_factories)
        self._tools = ToolRegistry(tools)
        self._verifier = verifier or create_default_completion_verifier()
        self._state_store = StateStore(self.state_path, workspace_dir=self._workspace)
        self._checkpoints = CheckpointManager(checkpoint_dir)
        self._approvals = ApprovalManager(approvals_path)
        self._running = False

    @property
    def workspace(self) -> Path:
        """Resolved repository under test."""
        return self._workspace

    @property
    def tools(self) -> ToolRegistry:
        """Tools registered for every run."""
        return self._tools

    @property
    def state_store(self) -> StateStore:
        """Execution state store used by the harness."""
        return self._state_store

    @property
    def checkpoints(self) -> CheckpointManager:
        """Checkpoint manager used by the harness."""
        return self._checkpoints

    @property
    def approvals(self) -> ApprovalManager:
        """Approval and audit manager used by the harness."""
        return self._approvals

    @property
    def completion_verifier(self) -> CompletionVerifier:
        """Completion verifier used after each run."""
        return self._verifier

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool available to every subsequent run."""
        self._tools.register(tool)

    def run(self, executor: TaskExecutor, *, task: str | None = None) -> HarnessOutcome:
        """Execute a task and close the loop: state, checkpoint, audit and completion verification.

        Args:
            executor (`Callable[[HarnessContext], Any]`): Callback receiving the run context.
            task (`str`, *optional*): Task description; defaults to `config.task_name`.

        Returns:
            `HarnessOutcome`: Final status, executor output, completion report and artifact paths.

        Raises:
            HarnessConfigurationError: If the model cannot be assembled.
            HarnessStateError: If state or checkpoint persistence fails.
            HarnessSandboxError: If the sandbox cannot be started.
            HarnessToolRegistryError: If sandbox tools clash with registered ones.
        """
        if not callable(executor):
            raise HarnessConfigurationError("'executor' must be callable and accept a HarnessContext.")
        if self._running:
            raise HarnessStateError("Harness is already running a task; wait for the current run to finish.")

        task_text = task or self.config.task_name
        self._running = True
        sandbox: Sandbox | None = None
        started_at = utc_now()
        run_id = new_run_id()
        baseline = snapshot_files(self._workspace) if self.config.verify_completion else None
        try:
            model = self._model_assembler.assemble()
            try:
                self._state_store.start(
                    task=task_text,
                    agent_class=self.config.agent_class,
                    max_steps=self.config.max_steps,
                    agent_name=self.config.agent_name,
                    model_id=model.model_id,
                    additional_args={"task_name": self.config.task_name},
                )
            except StateStoreError as exc:
                raise HarnessStateError(f"Could not initialize execution state: {exc}") from exc
            registry, sandbox = self._prepare_sandbox_tools()
            state_variables: dict[str, Any] = {}
            gate = registry.build_gate(
                permissions=ToolPermissionPolicy(
                    allowed_tools=self.config.allowed_tools,
                    denied_tools=self.config.denied_tools,
                ),
                state=state_variables,
            )
            context = HarnessContext(
                task=task_text,
                workspace=self._workspace,
                model=model,
                sandbox=sandbox,
                tools=registry,
                gate=gate,
                state=self._state_store,
                checkpoints=self._checkpoints,
                approvals=self._approvals,
                state_variables=state_variables,
                run_id=run_id,
            )

            output: Any = None
            error: Exception | None = None
            try:
                output = executor(context)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller through the outcome
                logger.exception("Task executor failed")
                error = exc

            report: CompletionReport | None = None
            if error is None:
                try:
                    report = self._verify_completion(self._workspace, baseline)
                except CompletionVerificationError as exc:
                    error = exc
            if report is not None and not report.passed:
                error = CompletionVerificationError(report.summary())

            status = ExecutionStatus.FAILED if error is not None else ExecutionStatus.COMPLETED
            checkpoint_path = self._finalize_run(
                task_text=task_text,
                run_id=run_id,
                started_at=started_at,
                status=status,
                output=output,
                error=error,
                state_variables=state_variables,
                model_id=model.model_id,
                tool_names=registry.names(),
            )
            return HarnessOutcome(
                task=task_text,
                status=status,
                exit_code=exit_code_for(status),
                output=output,
                completion=report,
                error=str(error) if error is not None else None,
                state_path=self.state_path,
                checkpoint_path=checkpoint_path,
            )
        finally:
            self._running = False
            if sandbox is not None:
                sandbox.destroy()

    def _prepare_sandbox_tools(self) -> tuple[ToolRegistry, Sandbox | None]:
        """Build the per-run tool registry, starting and registering sandbox tools when enabled."""
        registry = ToolRegistry(self._tools.values())
        if not self.config.enable_sandbox:
            return registry, None
        sandbox_config = self.config.sandbox_config or SandboxConfig(
            readable_dirs=(self._workspace,),
            use_os_isolation=self.config.use_os_isolation,
        )
        try:
            sandbox = Sandbox.create(sandbox_config)
        except SandboxError as exc:
            raise HarnessSandboxError(f"Could not start the sandbox: {exc}") from exc
        try:
            if self.config.register_sandbox_tools:
                for tool in create_sandbox_tools(sandbox):
                    registry.register(tool)
        except HarnessToolRegistryError:
            sandbox.destroy()
            raise
        return registry, sandbox

    def _verify_completion(self, workspace: Path, baseline: dict[str, str] | None) -> CompletionReport | None:
        """Run the completion verifier, skipping unknown task names instead of failing the run."""
        if not self.config.verify_completion:
            return None
        try:
            return self._verifier.verify(self.config.task_name, workspace, baseline)
        except UnknownCompletionTaskError:
            logger.warning(
                "No completion validator registered for task '%s'; skipping verification", self.config.task_name
            )
            return None
        except CompletionError as exc:
            raise CompletionVerificationError(f"Completion verification could not run: {exc}") from exc

    def _finalize_run(
        self,
        *,
        task_text: str,
        run_id: str,
        started_at: str,
        status: ExecutionStatus,
        output: Any,
        error: Exception | None,
        state_variables: dict[str, Any],
        model_id: str | None,
        tool_names: tuple[str, ...],
    ) -> Path:
        """Persist the final execution state and a checkpoint snapshot."""
        try:
            self._state_store.finish(status=status, output=output, error=error)
            checkpoint_state = CheckpointState(
                run_id=run_id,
                task=task_text,
                agent_class=self.config.agent_class,
                max_steps=self.config.max_steps,
                budget=build_budget([], max_steps=self.config.max_steps, started_at=started_at),
                memory_system_prompt="",
                memory_steps=[],
                state=serialize_state(state_variables),
                agent_name=self.config.agent_name,
                model_id=model_id,
                tool_names=list(tool_names),
                status=status.value,
                exit_code=exit_code_for(status),
                final_output=output,
                error=serialize_error(error),
            )
            return self._checkpoints.save(checkpoint_state)
        except StateStoreError as exc:
            raise HarnessStateError(f"Could not persist execution state: {exc}") from exc
        except CheckpointError as exc:
            raise HarnessStateError(f"Could not persist checkpoint: {exc}") from exc

    def request_approval(
        self,
        *,
        operation_type: str,
        content: Any,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
        """Write a pending approval request to the audit trail."""
        try:
            return self._approvals.request_approval(
                subject=subject or self.config.agent_name,
                operation_type=operation_type,
                content=content,
                applicant=self.config.approval_applicant,
                metadata=metadata,
            )
        except AuditError as exc:
            raise HarnessStateError(f"Could not record approval request: {exc}") from exc

    def approve(self, record_id: str, *, approver: str, comment: str | None = None) -> ApprovalResult:
        """Approve a pending request and run its post-approval handler."""
        try:
            return self._approvals.approve(record_id, approver=approver, comment=comment)
        except AuditError as exc:
            raise HarnessStateError(f"Could not approve record '{record_id}': {exc}") from exc

    def reject(self, record_id: str, *, approver: str, reason: str) -> ApprovalResult:
        """Reject a pending request and record the rejection reason."""
        try:
            return self._approvals.reject(record_id, approver=approver, reason=reason)
        except AuditError as exc:
            raise HarnessStateError(f"Could not reject record '{record_id}': {exc}") from exc

    def list_approvals(self, status: ApprovalStatus | str | None = None) -> list[ApprovalRecord]:
        """List approval records, optionally filtered by status."""
        try:
            return self._approvals.list_records(status=status)
        except AuditError as exc:
            raise HarnessStateError(f"Could not list approval records: {exc}") from exc

    def load_latest_checkpoint(self) -> CheckpointState:
        """Load the most recent checkpoint of this harness."""
        try:
            return self._checkpoints.load()
        except CheckpointError as exc:
            raise HarnessStateError(f"Could not load the latest checkpoint: {exc}") from exc


__all__ = [
    "CompletionVerificationError",
    "Harness",
    "HarnessConfig",
    "HarnessContext",
    "HarnessError",
    "HarnessOutcome",
    "HarnessSandboxError",
    "HarnessStateError",
    "HarnessToolRegistryError",
    "HarnessConfigurationError",
    "ModelAssembler",
    "ModelSpec",
    "TaskExecutor",
    "ToolRegistry",
]
