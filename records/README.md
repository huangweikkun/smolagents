# 作业三交付记录（Guarded Coding Harness）

本目录保存作业要求的执行证据：两类仓库任务的真实执行记录、完整运行记录、审计记录、跨进程 Checkpoint 恢复记录，以及 `pytest -q` 结果。

| 记录 | 位置 | 说明 |
| --- | --- | --- |
| 任务一真实执行记录 | `task1/` | DeepSeek 模型通过 coding 工具实际阅读代码并写出交付文档，严格审批 1 次后完成校验通过 |
| 任务二真实执行记录 | `task2/` | DeepSeek 模型先跑失败测试、修复源码、复跑全绿；严格审批 1 次，现有测试未被修改 |
| 完整运行记录 | `run/state.json` | Harness 在 `samples/1_login_demo` 上执行 `understand` 任务，完成校验通过 |
| Checkpoint | `run/checkpoints/checkpoint-0001.json`、`latest.json` | 带 checksum 的运行快照 |
| 审计记录 | `run/audit.jsonl` | append-only JSONL，可用 `run_id` + `tool_call_id` 还原决策链 |
| 审批记录 | `run/approvals.json` | 高风险写操作的审批申请与批准 |
| Checkpoint 恢复记录 | `resume/state.json`、`resume/checkpoints/` | 第一进程中断，第二进程恢复并完成 |
| 恢复审计 | `resume/audit.jsonl` | 恢复后的放行与执行事件，沿用同一 `run_id` |
| 测试结果 | `pytest.txt` | 本次相关测试套件输出 |

## 1. 任务一真实执行记录（task1/）

工作区是 `samples/1_login_demo` 的隔离副本（`task1/workspace`），预先删除已有交付物，强制 Agent 现场产出。

- 模型：`deepseek-chat`（OpenAI 兼容接口）
- 入口：`Harness.run(executor)`，由 Harness 装配模型、Sandbox、coding 工具、Gate、Audit 与完成校验；State/Checkpoint 由内层 Agent 落盘
- 工具：`list_files` / `read_file` / `search_code` / `write_file`
- 严格审批：`write_file` 为 HIGH 风险，进入 `waiting_approval` 后审批再执行
- 产物：`task1/workspace/artifacts/login-flow.md`（由 Agent 通过工具写入）
- 执行状态：`task1/agent/state.json`，`status=completed`、`stop_reason=completed`、`exit_code=0`
- 审计：`task1/harness/audit.jsonl`，含审批申请、等待、批准、放行与执行事件
- Checkpoint：`task1/agent/checkpoints/`（Agent 安全步骤快照，审批暂停后据此恢复）
- 审批：`task1/harness/approvals.json`
- 完成校验：`task1/completion.json`

```
run_id      c35dedc03a7247a5b3a8cdd3512ae827
approvals   approved × 1
completion  deliverable_present ✓ / references_valid ✓ / production_notes ✓ / write_scope ✓
            87 citations across 11 files all resolve
```

## 2. 任务二真实执行记录（task2/）

工作区是 `samples/2_bug_fix/demo-app` 的隔离副本（`task2/workspace`），安装依赖后建立基线。

- 模型：`deepseek-chat`
- 入口：`Harness.run(executor)`，Agent 使用 Harness 装配的工具、审批与审计
- 工具：`list_files` / `read_file` / `search_code` / `edit_file` / `run_test` / `git_diff`
- 写策略：只允许 `src/**` 与 `tests/generated/**`，`tests/**` 其余路径拒绝（Gate 也会拒绝）
- 严格审批：`edit_file` 为 HIGH 风险，进入 `waiting_approval` 后审批再执行
- 基线：`test-baseline.txt`，`npm test` 退出码 1（3 个失败用例）
- 修复后：`test-final.txt`，`npm test` 退出码 0（`tests 7 / pass 7 / fail 0`）
- 改动：`diff.txt`，仅 `src/auth/session-policy.ts` 的 `isSessionExpired` 一处
- 现有测试：`existing_tests_untouched` 通过，`tests/` 未被修改
- 执行状态：`agent/state.json`，`harness/audit.jsonl`，`harness/approvals.json`
- Checkpoint：`agent/checkpoints/`
- 完成校验：`completion.json`

```
run_id      d5b9b20f33364ddb87f0b85a7de4ce70
approvals   approved × 1
baseline    npm test exit=1（3 fail）
final       npm test exit=0（7 pass）
completion  tests_pass ✓ / deliverable_present ✓ / deliverable_changed ✓ / existing_tests_untouched ✓
```

## 3. 完整运行记录（run/）

生成方式：`Harness` + `sandbox_file_write` 高风险工具，审批通过后执行，最后由完成校验器验收 `artifacts/login-flow.md`。

关键结果：

```
run_id       cea0da32733e411e9d991dffc7f7c46e
status       completed / exit_code 0 / stop_reason completed
approvals    approved
audit        approval_requested -> gate_waiting_approval -> approval_approved
             -> gate_allow -> execution_started -> execution_completed
checkpoint   records/run/checkpoints/checkpoint-0001.json
```

审计记录中的每条事件都带 `run_id`、`tool_call_id`、`rule`、`decision`、`actor`，可按 `tool_call_id=call-write-1` 还原"申请 → 等待 → 批准 → 放行 → 执行完成"的完整决策链。

## 4. Checkpoint 跨进程恢复记录（resume/）

两个独立进程共享 `resume/state.json`、`resume/checkpoints/` 与 `resume/audit.jsonl`：

1. 进程一：`ToolCallingAgent` 执行一次 `echo` 后触发中断
   - `status=interrupted`，`stop_reason=interrupted`，`steps_used=1`
   - `run_id=e15992e78c77484ca3df4770026f0a9d`
2. 进程二：`resume_from_checkpoint()` 从第 2 步继续，模型给出最终答案
   - `status=completed`，`stop_reason=completed`，`steps_used=2`
   - `run_id` 与进程一一致，说明恢复没有重置运行身份
   - 已完成的第一步没有被重放

恢复时若 `workspace_dir` 指纹与 checkpoint 不一致，`validate_workspace_unchanged` 会抛出 `CheckpointValidationError`，拒绝使用旧证据和旧审批继续执行。

## 5. pytest -q 结果

`pytest.txt` 记录了本次提交相关测试套件的实际输出：

```
131 passed in 2.16s
```

覆盖 `coding_tools`、`state_store`、`checkpoint`、`gate`、`audit`、`sandbox`、`harness`、`completion`。

`pytest-full.txt` 是安装 `.[test]` 后全量 `pytest -q` 的实际结果：

```
1087 passed, 66 skipped in 71.66s
```

## 6. P0 验收覆盖

| 场景 | 证据 |
| --- | --- |
| G1 参数错误、handler 0 次 | `tests/test_gate.py::test_invalid_arguments_return_structured_decision_without_running_handler` |
| G2 第三次相同调用被拦截 | `tests/test_gate.py::test_third_identical_call_is_blocked_before_execution` |
| G3 未审批写操作 WAITING_APPROVAL、目标不变 | `tests/test_gate.py::test_unapproved_high_risk_call_waits_and_keeps_target_unchanged` |
| G4 审批后重复调用只执行一次 | `tests/test_gate.py::test_approved_call_executes_once_and_replay_is_rejected` |
| S1 路径逃逸 | `tests/test_sandbox.py::test_symlink_escape_is_blocked` |
| S2 软链接逃逸 + 审计留痕 | `tests/test_sandbox.py::test_sandbox_denials_are_audited` |
| S3 白名单外命令 | `tests/test_sandbox.py::test_sandbox_denials_are_audited` |
| S4 超时终止 | `tests/test_sandbox.py::test_command_timeout_and_output_limit` |
| C1 checkpoint 字段完整 | `tests/test_state_checkpoint.py::test_checkpoint_contains_run_step_plan_evidence_budget_and_pending_approvals` |
| C2 UNKNOWN_RESULT 不重放 | `tests/test_state_checkpoint.py::test_tools_pending_marks_unknown_result_without_replay` |
| C3 换进程恢复 | `tests/test_state_checkpoint.py::test_process_restart_resumes_without_duplicate_steps_or_reset_counters` + 本目录 `resume/` |
| C4 工作区变化拒绝恢复 | `tests/test_state_checkpoint.py::test_resume_refuses_when_workspace_changed` |
| A1 拒绝调用的审计字段 | `tests/test_audit.py::test_gate_denial_is_audited_with_rule_decision_actor_and_call_id` |
| A2 完整链路关联 | `tests/test_harness_loop.py::test_harness_links_audit_events_to_checkpoint_run_id` + `run/` 记录 |
| H1 无交付物不得进入完成态 | `tests/test_harness_loop.py::test_harness_fails_when_completion_conditions_fail` |
| H2 测试后修改源码证据失效 | `tests/test_completion.py::test_fix_tests_reruns_when_workspace_changes_during_test_run`、`test_fix_tests_rejects_unstable_workspace_during_test_run` |
| H3 预算耗尽明确停止 | `tests/test_state_store.py::test_agent_stops_with_budget_exceeded_stop_reason`、`tests/test_harness_loop.py::test_harness_budget_exceeded_sets_stop_reason` |
| T1 测试可重复通过 | `pytest.txt` |

## 7. 复现命令

任务一、二的执行脚本已放在本目录（`run_task1.py` / `run_task2.py`），Harness 记录生成脚本为临时文件（`/tmp/generate_harness_records.py`）。核心执行方式如下：

```bash
# 任务一：真实模型执行代码理解（需要 DEEPSEEK_API_KEY）
python records/run_task1.py

# 任务二：真实模型执行测试修复（工作区会先 npm install 再建立基线）
python records/run_task2.py

# 运行记录
python /tmp/generate_harness_records.py run

# 恢复记录：两个独立进程
python /tmp/generate_harness_records.py resume_interrupt
python /tmp/generate_harness_records.py resume_finish

# 测试结果
pytest tests/test_coding_tools.py tests/test_state_store.py tests/test_state_checkpoint.py \
       tests/test_gate.py tests/test_audit.py tests/test_sandbox.py tests/test_harness_loop.py \
       tests/test_completion.py -q
```

`task1/`、`task2/` 的工作区都是 `samples/` 的隔离副本，不会直接修改样例仓库。
`run/` 使用 `samples/1_login_demo` 作为只读工作区；Sandbox 写入落在临时沙箱目录，不会修改样例仓库。
`resume/workspace/note.txt` 用于恢复演示的工作区指纹。
