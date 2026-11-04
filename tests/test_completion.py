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

from smolagents.completion import (
    CommandResult,
    CompletionVerifier,
    LoginFlowDocumentValidator,
    SessionBoundaryFixValidator,
    UnknownCompletionTaskError,
    create_default_completion_verifier,
    snapshot_files,
)


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

- 密码是明文拼接比较（`src/utils/password.ts:6`）。
"""


def _source_file(line_count: int = 40) -> str:
    return "".join(f"// line {index}\n" for index in range(1, line_count + 1))


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _understand_repo(tmp_path: Path, document: str | None = LOGIN_FLOW_DOCUMENT) -> Path:
    files = {path: _source_file() for path in REQUIRED_SOURCES}
    files["src/utils/password.ts"] = _source_file()
    if document is not None:
        files["artifacts/login-flow.md"] = document
    return _make_repo(tmp_path, files)


def _fix_repo(tmp_path: Path) -> Path:
    return _make_repo(
        tmp_path,
        {
            "src/auth/session-policy.ts": "export function isSessionExpired() { return false; }\n",
            "tests/session-boundary.test.ts": 'test("boundary", () => {});\n',
            "tests/session-policy.test.ts": 'test("policy", () => {});\n',
            "package.json": '{ "scripts": { "test": "node --test" } }\n',
        },
    )


def _failed_names(report) -> set[str]:
    return {check.name for check in report.failed_checks}


def test_understand_accepts_document_with_evidence_and_production_notes(tmp_path):
    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path))

    assert report.passed
    assert [check.name for check in report.checks] == [
        "deliverable_present",
        "references_valid",
        "production_notes",
        "write_scope",
    ]


def test_understand_rejects_missing_deliverable(tmp_path):
    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path, document=None))

    assert not report.passed
    assert "deliverable_present" in _failed_names(report)


def test_understand_rejects_unaligned_line_references(tmp_path):
    document = LOGIN_FLOW_DOCUMENT.replace("src/routes/login.ts:8-11", "src/routes/login.ts:1-999")

    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path, document))

    assert not report.passed
    assert "references_valid" in _failed_names(report)


def test_understand_rejects_missing_required_reference(tmp_path):
    document = LOGIN_FLOW_DOCUMENT.replace(
        "- `src/db/database.ts:5-20` 是内存数组模拟的数据库。\n", "数据库没读到。\n"
    )

    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path, document))

    assert not report.passed
    assert "references_valid" in _failed_names(report)


def test_understand_rejects_writes_outside_the_document(tmp_path):
    root = _understand_repo(tmp_path)
    baseline = snapshot_files(root)
    (root / "src/routes/login.ts").write_text(_source_file() + "// edited\n", encoding="utf-8")

    report = LoginFlowDocumentValidator().validate(root, baseline=baseline)

    assert not report.passed
    assert "write_scope" in _failed_names(report)


def test_understand_accepts_document_write_against_baseline(tmp_path):
    root = _understand_repo(tmp_path)
    baseline = snapshot_files(root)
    (root / "artifacts/login-flow.md").write_text(LOGIN_FLOW_DOCUMENT + "\n补充。\n", encoding="utf-8")

    report = LoginFlowDocumentValidator().validate(root, baseline=baseline)

    assert report.passed


def test_understand_skips_write_scope_without_baseline(tmp_path):
    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path))

    write_scope = next(check for check in report.checks if check.name == "write_scope")
    assert write_scope.skipped
    assert write_scope.passed


def test_fix_tests_accepts_green_run_with_generated_tests(tmp_path):
    root = _fix_repo(tmp_path)
    baseline = snapshot_files(root)
    (root / "src/auth/session-policy.ts").write_text(
        "export function isSessionExpired() { return true; }\n", encoding="utf-8"
    )
    generated = root / "tests/generated/boundary.test.ts"
    generated.parent.mkdir(parents=True)
    generated.write_text('test("generated", () => {});\n', encoding="utf-8")

    report = SessionBoundaryFixValidator(runner=lambda command, cwd: CommandResult(0, "7 pass")).validate(
        root, baseline=baseline
    )

    assert report.passed
    assert [check.name for check in report.checks] == [
        "tests_pass",
        "deliverable_present",
        "deliverable_changed",
        "existing_tests_untouched",
    ]


def test_fix_tests_rejects_failing_test_run(tmp_path):
    root = _fix_repo(tmp_path)

    report = SessionBoundaryFixValidator(runner=lambda command, cwd: CommandResult(1, "3 fail")).validate(root)

    assert not report.passed
    assert "tests_pass" in _failed_names(report)


def test_fix_tests_rejects_timeout(tmp_path):
    root = _fix_repo(tmp_path)

    report = SessionBoundaryFixValidator(runner=lambda command, cwd: CommandResult(-1, timed_out=True)).validate(root)

    assert not report.passed
    assert "tests_pass" in _failed_names(report)


def test_fix_tests_rejects_modified_existing_test(tmp_path):
    root = _fix_repo(tmp_path)
    baseline = snapshot_files(root)
    (root / "src/auth/session-policy.ts").write_text("fixed\n", encoding="utf-8")
    (root / "tests/session-boundary.test.ts").write_text('test("weakened", () => {});\n', encoding="utf-8")

    report = SessionBoundaryFixValidator(runner=lambda command, cwd: CommandResult(0)).validate(
        root, baseline=baseline
    )

    assert not report.passed
    assert "existing_tests_untouched" in _failed_names(report)


def test_fix_tests_rejects_new_test_outside_generated_dir(tmp_path):
    root = _fix_repo(tmp_path)
    baseline = snapshot_files(root)
    (root / "src/auth/session-policy.ts").write_text("fixed\n", encoding="utf-8")
    (root / "tests/extra.test.ts").write_text('test("extra", () => {});\n', encoding="utf-8")

    report = SessionBoundaryFixValidator(runner=lambda command, cwd: CommandResult(0)).validate(
        root, baseline=baseline
    )

    assert not report.passed
    assert "existing_tests_untouched" in _failed_names(report)


def test_understand_resolves_unique_bare_filenames(tmp_path):
    document = LOGIN_FLOW_DOCUMENT.replace("`src/routes/login.ts:8-11`", "`login.ts:8-11`")

    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path, document))

    assert report.passed


def test_understand_rejects_ambiguous_bare_filename(tmp_path):
    root = _understand_repo(tmp_path)
    duplicate = root / "src" / "legacy" / "login.ts"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(_source_file(), encoding="utf-8")
    document = LOGIN_FLOW_DOCUMENT.replace("`src/routes/login.ts:8-11`", "`login.ts:8-11`")

    report = LoginFlowDocumentValidator().validate(_understand_repo(tmp_path, document))

    assert not report.passed
    assert "references_valid" in _failed_names(report)


def test_fix_tests_rejects_unmodified_deliverable(tmp_path):
    root = _fix_repo(tmp_path)
    baseline = snapshot_files(root)

    report = SessionBoundaryFixValidator(runner=lambda command, cwd: CommandResult(0)).validate(
        root, baseline=baseline
    )

    assert not report.passed
    assert "deliverable_changed" in _failed_names(report)


def test_fix_tests_reruns_when_workspace_changes_during_test_run(tmp_path):
    root = _fix_repo(tmp_path)
    (root / "src/auth/session-policy.ts").write_text("fixed\n", encoding="utf-8")
    calls = {"count": 0}

    def runner(command, cwd):
        calls["count"] += 1
        if calls["count"] == 1:
            (root / "src/auth/session-policy.ts").write_text("fixed-again\n", encoding="utf-8")
        return CommandResult(0)

    report = SessionBoundaryFixValidator(runner=runner).validate(root)

    assert report.passed
    assert calls["count"] == 2


def test_fix_tests_rejects_unstable_workspace_during_test_run(tmp_path):
    root = _fix_repo(tmp_path)
    calls = {"count": 0}

    def runner(command, cwd):
        calls["count"] += 1
        (root / "src/auth/session-policy.ts").write_text(f"version-{calls['count']}\n", encoding="utf-8")
        return CommandResult(0)

    report = SessionBoundaryFixValidator(runner=runner, test_reruns=1).validate(root)

    assert not report.passed
    assert "tests_pass" in _failed_names(report)
    detail = next(check for check in report.checks if check.name == "tests_pass").detail
    assert "stale" in detail
    assert calls["count"] == 2


def test_snapshot_files_skips_symlinks_and_excluded_dirs(tmp_path):
    root = _make_repo(tmp_path, {"src/app.ts": "app\n", "node_modules/pkg/index.js": "dep\n"})
    (root / "link.ts").symlink_to(root / "src/app.ts")

    snapshot = snapshot_files(root)

    assert snapshot == {"src/app.ts": snapshot["src/app.ts"]}


def test_verifier_dispatches_to_registered_task(tmp_path):
    verifier = create_default_completion_verifier()

    report = verifier.verify("understand", _understand_repo(tmp_path))

    assert report.passed
    assert report.task == "understand"
    assert verifier.task_names == ("fix_tests", "understand")


def test_verifier_rejects_unknown_task(tmp_path):
    with pytest.raises(UnknownCompletionTaskError, match="Unknown completion task"):
        create_default_completion_verifier().verify("missing", tmp_path)


def test_report_summary_lists_each_condition(tmp_path):
    report = CompletionVerifier([LoginFlowDocumentValidator()]).verify("understand", _understand_repo(tmp_path))

    summary = report.summary()
    assert "task=understand completed=True" in summary
    assert "[pass] references_valid" in summary
    assert "[skip] write_scope" in summary
