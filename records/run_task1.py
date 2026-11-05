"""Execute task one (code understanding) through the unified Harness entry point.

The Harness assembles the model, sandbox, coding tools, gate, state, checkpoints, audit and completion
verification; the executor only drives a ToolCallingAgent with those assembled pieces.
"""

import json
import os
import shutil
from pathlib import Path

from smolagents.agents import ToolCallingAgent
from smolagents.audit import ApprovalStatus
from smolagents.coding_tools import create_coding_tools, default_coding_risk_policy
from smolagents.harness import Harness, HarnessConfig
from smolagents.models import OpenAIModel
from smolagents.sandbox import SandboxConfig
from smolagents.state_store import StateStore
from smolagents.utils import AgentApprovalRequiredError


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "records" / "task1"
WORKSPACE = RECORDS / "workspace"
HARNESS_DIR = RECORDS / "harness"
AGENT_DIR = RECORDS / "agent"
SAMPLES = REPO / "samples" / "1_login_demo"

PROMPT = """任务：为这个仓库写一份写给新同事的登录调用链说明，交付物是 artifacts/login-flow.md。

要求：
1. 先定位再阅读：用 list_files / search_code 找到登录相关文件，再用 read_file 按需读取，不要通读整个仓库。
2. 一个文件尽量只读一次，读完路由、中间件、Service、仓储、假数据库这条主链即可开始写。
3. 文档中每条结论都必须给出 文件路径:行号 级别的依据，例如 src/routes/login.ts:8-11。
4. 单列一节 "不能照搬到生产的事实"，逐条给出代码依据。
5. 唯一的写操作是写 artifacts/login-flow.md；不要修改任何源码或测试。
6. 写完后调用 final_answer 给出简短说明。

注意：write_file 是高风险工具，会暂停等待人工审批；这是预期行为，批准后继续。
"""


def prepare_workspace():
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    shutil.copytree(SAMPLES, WORKSPACE)
    (WORKSPACE / "artifacts" / "login-flow.md").unlink(missing_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    for name in ("completion.json", "summary.json", "baseline.json", "state.json", "approvals.json", "audit.jsonl"):
        (RECORDS / name).unlink(missing_ok=True)
    for name in ("harness", "agent", "checkpoints", "sandbox"):
        shutil.rmtree(RECORDS / name, ignore_errors=True)


def resolve_pending(context) -> None:
    """Act as the maintainer: approve artifacts-only writes, reject everything else."""
    for record in context.approvals.list_records(status=ApprovalStatus.PENDING):
        content = record.content if isinstance(record.content, dict) else {}
        path = str(content.get("path", ""))
        if record.operation_type == "write_file" and path.startswith("artifacts/"):
            context.approvals.approve(
                record.record_id,
                approver="maintainer-demo",
                comment=f"approved artifacts-only write: {path}",
            )
            print(f"  approved {record.operation_type} -> {path}")
        else:
            context.approvals.reject(
                record.record_id,
                approver="maintainer-demo",
                reason="only artifacts/** writes are allowed in task one",
            )
            print(f"  rejected {record.operation_type} -> {path}")


def main():
    prepare_workspace()

    model = OpenAIModel(
        model_id="deepseek-chat",
        api_base="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.1,
        max_tokens=4096,
    )
    harness = Harness(
        HarnessConfig(
            workspace=WORKSPACE,
            task_name="understand",
            runtime_dir=HARNESS_DIR,
            enable_sandbox=True,
            register_sandbox_tools=False,
            use_os_isolation=False,
            sandbox_config=SandboxConfig(workspace_root=WORKSPACE, use_os_isolation=False),
            sandbox_tool_factory=lambda sandbox: create_coding_tools(
                sandbox,
                allowed_write_paths=("artifacts/**",),
                denied_write_paths=(),
                include_git=False,
            ),
            risk_policy=default_coding_risk_policy(),
            max_tool_calls=40,
            persist_state=False,
            persist_checkpoint=False,
        ),
        model=model,
    )

    def executor(context):
        agent = ToolCallingAgent(
            model=context.model,
            tools=list(context.tools.values()),
            max_steps=15,
            max_tool_calls=40,
            risk_policy=default_coding_risk_policy(),
            approval_manager=context.approvals,
            audit_log=context.audit_log,
            state_store=AGENT_DIR / "state.json",
            checkpoint_store=AGENT_DIR / "checkpoints",
            workspace_dir=context.workspace,
        )
        # Keep Agent checkpoints, state and audit under the Harness run identity.
        agent._checkpoint_run_id = context.run_id

        result = None
        for round_number in range(1, 6):
            try:
                result = agent.run(PROMPT) if round_number == 1 else agent.resume_from_checkpoint()
                break
            except AgentApprovalRequiredError as error:
                print(f"[round {round_number}] approval required: {error}")
                resolve_pending(context)
        return result

    outcome = harness.run(executor)

    completion = outcome.completion
    completion_payload = {
        "task": outcome.task,
        "passed": bool(completion and completion.passed),
        "checks": [
            {"name": check.name, "passed": check.passed, "skipped": check.skipped, "detail": check.detail}
            for check in (completion.checks if completion else ())
        ],
    }
    (RECORDS / "completion.json").write_text(
        json.dumps(completion_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state = StateStore.load(AGENT_DIR / "state.json")
    summary = {
        "result": str(outcome.output),
        "run_id": state.run_id,
        "status": outcome.status.value,
        "stop_reason": outcome.stop_reason,
        "exit_code": outcome.exit_code,
        "completion_passed": completion_payload["passed"],
        "approvals": [record.status.value for record in harness.approvals.list_records()],
        "audit_events": [event.event_type.value for event in harness.audit_log.events()],
        "deliverable": str((WORKSPACE / "artifacts" / "login-flow.md").relative_to(REPO)),
        "harness_dir": str(HARNESS_DIR.relative_to(REPO)),
        "agent_dir": str(AGENT_DIR.relative_to(REPO)),
    }
    (RECORDS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if completion is not None:
        print(completion.summary())


if __name__ == "__main__":
    main()
