from __future__ import annotations

import builtins
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed import case_runner  # noqa: E402
from msabp_opt.simulation.distributed import maid  # noqa: E402
from msabp_opt.simulation.distributed.protocol import make_message  # noqa: E402


def _make_config(
    tmp_path: Path,
    *,
    row_count: int = 1,
    dry_run: bool = True,
    max_consecutive_errors: int = 5,
    artifact_timeout_seconds: float = 600.0,
    artifact_commit_deadline_seconds: float = 1800.0,
) -> maid.MaidRuntimeConfig:
    csv_path = tmp_path / "samples.csv"
    csv_lines = ["sample_id,value"]
    csv_lines.extend(f"{index},{index + 0.5}" for index in range(row_count))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    project_path = tmp_path / "worker.cst"
    if not dry_run:
        project_path.write_bytes(b"standalone-cst-template")
    return maid.MaidRuntimeConfig.from_mapping(
        {
            "schema_version": maid.RUNTIME_SCHEMA_VERSION,
            "run_id": "run-1",
            "worker_id": "maid-a",
            "princess_url": "http://princess.invalid:8765",
            "api_token": "test-token",
            "csv_path": str(csv_path),
            "csv_sha256": maid.sha256_file(csv_path),
            "project_path": str(project_path),
            "output_root": str(tmp_path / "runtime-output"),
            "dry_run": dry_run,
            "heartbeat_seconds": 60.0,
            "poll_seconds": 0.001,
            "max_consecutive_errors": max_consecutive_errors,
            "artifact_timeout_seconds": artifact_timeout_seconds,
            "artifact_commit_deadline_seconds": (
                artifact_commit_deadline_seconds
            ),
        }
    )


class FakeSession:
    def __init__(self) -> None:
        self.project = object()
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> object:
        self.open_calls += 1
        return self.project

    def close(self) -> None:
        self.close_calls += 1


class FakePrincessClient:
    def __init__(
        self,
        assignments: list[dict[str, Any]],
        *,
        failure_response_type: str = "failure_ack",
    ) -> None:
        self.assignments = list(assignments)
        self.failure_response_type = failure_response_type
        self.messages: list[dict[str, Any]] = []
        self.uploads: list[tuple[str, str, str, bytes]] = []

    def _reply(self, request: dict[str, Any], response_type: str, payload=None):
        return make_message(
            response_type,
            run_id=request["run_id"],
            worker_id=request.get("worker_id"),
            payload=payload or {},
        )

    def send(self, message: dict[str, Any]) -> dict[str, Any]:
        self.messages.append(message)
        message_type = message["type"]
        if message_type == "hello":
            return self._reply(message, "welcome")
        if message_type == "request_task":
            if self.assignments:
                return self._reply(message, "assignment", self.assignments.pop(0))
            return self._reply(message, "stop")
        if message_type == "complete":
            return self._reply(message, "completed_ack")
        if message_type == "failure":
            return self._reply(message, self.failure_response_type)
        if message_type == "heartbeat":
            return self._reply(message, "heartbeat_ack")
        raise AssertionError(f"unexpected fake Princess request: {message_type}")

    def upload_artifact(
        self,
        attempt_id: str,
        worker_id: str,
        lease_token: str,
        archive_path: str | Path,
    ) -> dict[str, Any]:
        archive = Path(archive_path)
        self.uploads.append(
            (attempt_id, worker_id, lease_token, archive.read_bytes())
        )
        return {
            "sha256": maid.sha256_file(archive),
            "size": archive.stat().st_size,
        }


def _assignment(index: int) -> dict[str, Any]:
    row = {"sample_id": str(index), "value": str(index + 0.5)}
    return {
        "case_id": str(index),
        "row_index": index,
        "attempt_id": f"attempt-{index}",
        "lease_token": f"lease-{index}",
        "row_sha256": maid.canonical_row_sha256(
            ("sample_id", "value"),
            row,
        ),
    }


def test_runtime_config_round_trip_and_csv_hash(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    destination = tmp_path / "runtime" / "maid.json"
    payload = {
        "schema_version": maid.RUNTIME_SCHEMA_VERSION,
        "run_id": config.run_id,
        "worker_id": config.worker_id,
        "princess_url": config.princess_url,
        "api_token": config.api_token,
        "csv_path": str(config.csv_path),
        "csv_sha256": config.csv_sha256.upper(),
        "project_path": str(config.project_path),
        "output_root": str(config.output_root),
        "dry_run": True,
    }

    written = maid.write_runtime_config(destination, payload)
    loaded = maid.MaidRuntimeConfig.load(written)

    assert loaded.csv_sha256 == config.csv_sha256
    assert maid.load_csv_rows(loaded.csv_path, loaded.csv_sha256) == [
        {"sample_id": "0", "value": "0.5"}
    ]
    loaded.csv_path.write_text("sample_id,value\n0,changed\n", encoding="utf-8")
    with pytest.raises(maid.MaidConfigError, match="SHA-256 mismatch"):
        maid.load_csv_rows(loaded.csv_path, loaded.csv_sha256)

    payload["csv_sha256"] = "not-a-digest"
    with pytest.raises(maid.MaidConfigError, match="64 hexadecimal"):
        maid.MaidRuntimeConfig.from_mapping(payload)

    payload["csv_sha256"] = config.csv_sha256
    payload["heartbeat_seconds"] = float("nan")
    with pytest.raises(maid.MaidConfigError, match="finite and positive"):
        maid.MaidRuntimeConfig.from_mapping(payload)


def test_doctor_dry_run_does_not_import_or_require_cst(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, dry_run=True)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cst" or name.startswith("cst."):
            raise AssertionError("dry-run doctor must not import CST")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    report = maid.doctor(config)

    assert report["dry_run"] is True
    assert report["csv_rows"] == 1
    assert "cst.interface" not in report["modules"]
    assert not config.project_path.exists()


def test_fake_princess_and_session_complete_one_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, dry_run=False)
    client = FakePrincessClient([_assignment(0)])
    session = FakeSession()

    def fake_run_csv_row(_row, **kwargs):
        assert kwargs["project"] is session.project
        assert kwargs["case_id"] == "0"
        kwargs["stage_callback"]("building_geometry")
        case_directory = Path(kwargs["output_root"]) / "case_0000"
        case_directory.mkdir(parents=True)
        s11_path = case_directory / "S11.csv"
        farfield_path = case_directory / "Farfield Source [1].ffs"
        manifest_path = case_directory / "manifest.json"
        s11_path.write_bytes(b"1.0 -10.0\n")
        farfield_path.write_bytes(b"fake-ffs")
        manifest_path.write_text('{"status":"completed"}', encoding="utf-8")
        return case_runner.CaseRunResult(
            case_id="0",
            case_directory=case_directory,
            manifest_path=manifest_path,
            s11_path=s11_path,
            farfield_source_path=farfield_path,
            dry_run=False,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(maid.case_runner, "run_csv_row", fake_run_csv_row)
    worker = maid.Maid(config, client=client, session=session)

    assert worker.run() == 0
    assert session.open_calls == 1
    assert session.close_calls == 1
    assert len(client.uploads) == 1
    attempt_id, worker_id, lease_token, archive_bytes = client.uploads[0]
    assert (attempt_id, worker_id, lease_token) == (
        "attempt-0",
        "maid-a",
        "lease-0",
    )
    archive_copy = tmp_path / "uploaded.zip"
    archive_copy.write_bytes(archive_bytes)
    with zipfile.ZipFile(archive_copy) as handle:
        assert set(handle.namelist()) == {
            "Farfield Source [1].ffs",
            "S11.csv",
            "manifest.json",
        }
    message_types = [message["type"] for message in client.messages]
    assert message_types == ["hello", "request_task", "complete", "request_task"]
    assert not (config.output_root / "outbox" / "attempt-0.zip").exists()
    assert not (config.output_root / "attempts" / "attempt-0").exists()


def test_five_consecutive_failures_force_local_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(
        tmp_path,
        row_count=5,
        dry_run=True,
        max_consecutive_errors=5,
    )
    client = FakePrincessClient([_assignment(index) for index in range(5)])
    session = FakeSession()

    def fail_case(*_args, **_kwargs):
        raise RuntimeError("synthetic CST pipeline failure")

    monkeypatch.setattr(maid.case_runner, "run_csv_row", fail_case)
    worker = maid.Maid(config, client=client, session=session)

    assert worker.run() == maid.UNHEALTHY_EXIT_CODE
    failures = [message for message in client.messages if message["type"] == "failure"]
    requests = [
        message for message in client.messages if message["type"] == "request_task"
    ]
    assert len(failures) == 5
    assert all(message["payload"]["retryable"] for message in failures)
    assert worker.failed_case_ids == ["0", "1", "2", "3", "4"]
    assert requests[-1]["payload"]["exclude_case_ids"] == ["0", "1", "2", "3"]
    assert session.open_calls == 0
    assert session.close_calls == 1


def test_archive_is_flat_atomic_and_rejects_unsafe_attempt_id(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    worker = maid.Maid(
        config,
        client=FakePrincessClient([]),
        session=FakeSession(),
    )
    case_directory = tmp_path / "case"
    nested = case_directory / "nested"
    nested.mkdir(parents=True)
    (case_directory / "manifest.json").write_text("{}", encoding="utf-8")
    (nested / "result.bin").write_bytes(b"result")

    archive = worker._archive_result("attempt-safe", case_directory)

    assert archive == config.output_root / "outbox" / "attempt-safe.zip"
    assert not archive.with_suffix(".zip.tmp").exists()
    with zipfile.ZipFile(archive) as handle:
        assert set(handle.namelist()) == {"manifest.json", "nested/result.bin"}
        assert handle.read("nested/result.bin") == b"result"

    with pytest.raises(ValueError, match="attempt_id is unsafe"):
        worker._archive_result("../escape", case_directory)
    assert not (config.output_root / "escape.zip").exists()


def test_rejected_hello_still_closes_owned_session(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()

    class RejectingClient(FakePrincessClient):
        def send(self, message: dict[str, Any]) -> dict[str, Any]:
            return self._reply(message, "rejected")

    worker = maid.Maid(config, client=RejectingClient([]), session=session)
    with pytest.raises(RuntimeError, match="rejected Maid hello"):
        worker.run()
    assert session.close_calls == 1


def test_transient_hello_and_request_failures_are_retried(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    class FlakyClient(FakePrincessClient):
        hello_failures = 1
        request_failures = 1

        def send(self, message: dict[str, Any]) -> dict[str, Any]:
            if message["type"] == "hello" and self.hello_failures:
                self.hello_failures -= 1
                raise ConnectionError("Princess restarting")
            if message["type"] == "request_task" and self.request_failures:
                self.request_failures -= 1
                raise TimeoutError("temporary timeout")
            return super().send(message)

    delays: list[float] = []
    worker = maid.Maid(
        config,
        client=FlakyClient([]),
        session=FakeSession(),
        sleeper=delays.append,
    )

    assert worker.run() == 0
    assert len(delays) == 2


def test_assignment_row_identity_is_verified_before_cst(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    assignment = _assignment(0)
    assignment["row_sha256"] = "0" * 64
    worker = maid.Maid(
        config,
        client=FakePrincessClient([]),
        session=FakeSession(),
    )

    with pytest.raises(RuntimeError, match="row SHA-256 mismatch"):
        worker._run_assignment(assignment)

    assignment = _assignment(0)
    assignment["case_id"] = "different"
    with pytest.raises(RuntimeError, match="does not match CSV sample_id"):
        worker._run_assignment(assignment)


def test_artifact_commit_retries_do_not_report_a_cst_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    class FlakyCommitClient(FakePrincessClient):
        upload_failures = 1
        complete_failures = 1

        def upload_artifact(self, *args, **kwargs):
            if self.upload_failures:
                self.upload_failures -= 1
                raise ConnectionError("upload acknowledgement lost")
            return super().upload_artifact(*args, **kwargs)

        def send(self, message: dict[str, Any]) -> dict[str, Any]:
            if message["type"] == "complete" and self.complete_failures:
                self.complete_failures -= 1
                raise TimeoutError("commit response timeout")
            return super().send(message)

    def fake_run_csv_row(_row, **kwargs):
        case_directory = Path(kwargs["output_root"]) / "case_0000"
        case_directory.mkdir(parents=True)
        manifest_path = case_directory / "manifest.json"
        manifest_path.write_text('{"status":"completed"}', encoding="utf-8")
        return case_runner.CaseRunResult(
            case_id="0",
            case_directory=case_directory,
            manifest_path=manifest_path,
            s11_path=None,
            farfield_source_path=None,
            dry_run=True,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(maid.case_runner, "run_csv_row", fake_run_csv_row)
    client = FlakyCommitClient([_assignment(0)])
    delays: list[float] = []
    worker = maid.Maid(
        config,
        client=client,
        session=FakeSession(),
        sleeper=delays.append,
    )

    assert worker.run() == 0
    assert len(delays) == 2
    assert not [message for message in client.messages if message["type"] == "failure"]


@pytest.mark.parametrize(
    ("failing_phase", "error_pattern"),
    (("upload", "artifact upload"), ("complete", "complete request")),
)
def test_artifact_commit_transient_retries_share_a_finite_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_phase: str,
    error_pattern: str,
) -> None:
    config = _make_config(
        tmp_path,
        artifact_timeout_seconds=0.01,
        artifact_commit_deadline_seconds=0.005,
    )

    class DeadlineClient(FakePrincessClient):
        def __init__(self) -> None:
            super().__init__([_assignment(0)])
            self.upload_attempts = 0
            self.complete_attempts = 0

        def upload_artifact(self, *args, **kwargs):
            self.upload_attempts += 1
            if failing_phase == "upload":
                raise TimeoutError("upload acknowledgement never arrived")
            return super().upload_artifact(*args, **kwargs)

        def send(self, message: dict[str, Any]) -> dict[str, Any]:
            if message["type"] == "complete":
                self.complete_attempts += 1
                if failing_phase == "complete":
                    raise ConnectionError("commit endpoint unavailable")
            return super().send(message)

    def fake_run_csv_row(_row, **kwargs):
        case_directory = Path(kwargs["output_root"]) / "case_0000"
        case_directory.mkdir(parents=True)
        manifest_path = case_directory / "manifest.json"
        manifest_path.write_text('{"status":"completed"}', encoding="utf-8")
        return case_runner.CaseRunResult(
            case_id="0",
            case_directory=case_directory,
            manifest_path=manifest_path,
            s11_path=None,
            farfield_source_path=None,
            dry_run=True,
            elapsed_seconds=0.1,
        )

    clock = {"now": 100.0}
    delays: list[float] = []

    def advance_clock(delay: float) -> None:
        delays.append(delay)
        clock["now"] += delay

    monkeypatch.setattr(maid.case_runner, "run_csv_row", fake_run_csv_row)
    monkeypatch.setattr(maid.time, "monotonic", lambda: clock["now"])
    client = DeadlineClient()
    worker = maid.Maid(
        config,
        client=client,
        session=FakeSession(),
        sleeper=advance_clock,
    )

    with pytest.raises(TimeoutError, match=error_pattern):
        worker.run()

    attempts = (
        client.upload_attempts
        if failing_phase == "upload"
        else client.complete_attempts
    )
    assert 1 < attempts < 10
    assert sum(delays) == pytest.approx(config.artifact_commit_deadline_seconds)
    assert not [message for message in client.messages if message["type"] == "failure"]
    assert (config.output_root / "attempts" / "attempt-0").is_dir()
    assert (config.output_root / "outbox" / "attempt-0.zip").is_file()
