"""Execute task two (test generation and fixing) through the unified Harness entry point."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from smolagents.agents import ToolCallingAgent
from smolagents.audit import ApprovalStatus
from smolagents.coding_tools import create_coding_tools, default_coding_risk_policy
from smolagents.harness import Harness, HarnessConfig
from smolagents.models import OpenAIModel
from smolagents.sandbox import DEFAULT_ALLOWED_COMMANDS, SandboxConfig
from smolagents.state_store import StateStore
from smolagents.utils import AgentApprovalRequiredError


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "records" / "task2"
WORKSPACE = RECORDS / "workspace"
HARNESS_DIR = RECORDS / "harness"
AGENT_DIR = RECORDS / "agent"
SAMPLES = REPO / "samples" / "2_bug_fix" / "demo-app"
TEST_COMMAND = ("npm", "test", "--")
WRITE_PREFIXES = ("src/", "tests/generated/")

PROMPT = """任务：修复这个仓库里失败的测试，让全部用例变绿。

背景：测试用例位于 tests/，被测源码在 src/。现有用例要求按"秒级时刻"判断会话是否过期。

要求：
1. 先调用 run_test 拿到基线的失败信息。
2. 用 list_files / read_file / search_code 阅读失败的测试和被测源码，同文件尽量只读一次。
3. 只修改 src/ 下的源码来修复缺陷；不要修改 tests/ 下已有测试。
4. 如果确实需要补充测试，只能写到 tests/generated/ 下；不强制要求新增测试。
5. 每次修改后重新 run_test，直到全部用例通过。
6. 最后调用 git_diff 查看改动，并用 final_answer 简述改了哪几行。

注意：write_file / edit_file 是高风险工具，会暂停等待审批；这是预期行为，批准后继续。
"""


def _npm_env():
    env = dict(os.environ)
    env.setdefault("PATH", "/usr/bin:/bin")
    return env


def _run_npm_test() -> subprocess.CompletedProcess:
    return subprocess.run(
        list(TEST_COMMAND),
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=_npm_env(),
    )


def _sandbox_path() -> str:
    npm = shutil.which("npm")
    node = shutil.which("node")
    if npm is None or node is None:
        raise RuntimeError("npm and node must be available on PATH.")
    node_dir = str(Path(node).parent)
    return f"{node_dir}:/usr/bin:/bin:/usr/sbin:/sbin"


def prepare_workspace():
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SAMPLES, WORKSPACE)
    for name in ("completion.json", "summary.json", "test-baseline.txt", "test-final.txt", "diff.txt",
                 "baseline.json", "state.json", "approvals.json", "audit.jsonl"):
        (RECORDS / name).unlink(missing_ok=True)
    for name in ("harness", "agent", "checkpoints", "sandbox"):
        shutil.rmtree(RECORDS / name, ignore_errors=True)
    subprocess.run(["git", "init", "-q"], cwd=str(WORKSPACE), check=True)
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=str(WORKSPACE),
        check=True,
        env=_npm_env(),
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-c", "user.name=harness", "-c", "user.email=harness@localhost", "add", "-A"],
        cwd=str(WORKSPACE),
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.name=harness", "-c", "user.email=harness@localhost", "commit", "-q", "-m", "baseline"],
        cwd=str(WORKSPACE),
        check=True,
    )


def approve_pending(context) -> None:
    for record in context.approvals.list_records(status=ApprovalStatus.PENDING):
        content = record.content if isinstance(record.content, dict) else {}
        path = str(content.get("path", ""))
        if record.operation_type in {"write_file", "edit_file"} and path.startswith(WRITE_PREFIXES):
            context.approvals.approve(
                record.record_id,
                approver="maintainer-demo",
                comment=f"approved task-two write: {path}",
            )
            print(f"  approved {record.operation_type} -> {path}")
        else:
            context.approvals.reject(
                record.record_id,
                approver="maintainer-demo",
                reason="only src/** and tests/generated/** writes are allowed",
            )
            print(f"  rejected {record.operation_type} -> {path}")


def main():
    prepare_workspace()
    baseline_result = _run_npm_test()
    (RECORDS / "test-baseline.txt").write_text(
        f"exit={baseline_result.returncode}\n\n{baseline_result.stdout}\n{baseline_result.stderr}",
        encoding="utf-8",
    )
    print(f"baseline npm test exit={baseline_result.returncode}")

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
            task_name="fix_tests",
            runtime_dir=HARNESS_DIR,
            enable_sandbox=True,
            register_sandbox_tools=False,
            use_os_isolation=False,
            sandbox_config=SandboxConfig(
                workspace_root=WORKSPACE,
                use_os_isolation=False,
                allowed_commands=DEFAULT_ALLOWED_COMMANDS | frozenset({"npm", "node", "git"}),
                env={"PATH": _sandbox_path()},
            ),
            sandbox_tool_factory=lambda sandbox: create_coding_tools(
                sandbox,
                allowed_write_paths=("src/**", "tests/generated/**"),
                denied_write_paths=("tests/**",),
                test_command=TEST_COMMAND,
                include_git=True,
            ),
            risk_policy=default_coding_risk_policy(),
            max_tool_calls=50,
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
            max_tool_calls=50,
            risk_policy=default_coding_risk_policy(),
            approval_manager=context.approvals,
            audit_log=context.audit_log,
            state_store=AGENT_DIR / "state.json",
            checkpoint_store=AGENT_DIR / "checkpoints",
            workspace_dir=context.workspace,
        )
        agent._checkpoint_run_id = context.run_id

        result = None
        for round_number in range(1, 6):
            try:
                result = agent.run(PROMPT) if round_number == 1 else agent.resume_from_checkpoint()
                break
            except AgentApprovalRequiredError as error:
                print(f"[round {round_number}] approval required: {error}")
                approve_pending(context)
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

    final_result = _run_npm_test()
    (RECORDS / "test-final.txt").write_text(
        f"exit={final_result.returncode}\n\n{final_result.stdout}\n{final_result.stderr}",
        encoding="utf-8",
    )
    diff = subprocess.run(
        ["git", "-c", "core.pager=cat", "diff"],
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        env=_npm_env(),
    )
    (RECORDS / "diff.txt").write_text(diff.stdout or diff.stderr, encoding="utf-8")

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
        "baseline_test_exit": baseline_result.returncode,
        "final_test_exit": final_result.returncode,
        "harness_dir": str(HARNESS_DIR.relative_to(REPO)),
        "agent_dir": str(AGENT_DIR.relative_to(REPO)),
    }
    (RECORDS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if completion is not None:
        print(completion.summary())
    print("diff:\n" + (diff.stdout or "(none)"))


if __name__ == "__main__":
    main()
