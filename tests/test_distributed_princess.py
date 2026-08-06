from __future__ import annotations

import hashlib
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed import case_runner  # noqa: E402
from msabp_opt.simulation.distributed.http_api import (  # noqa: E402
    ApiError,
    PrincessClient,
    PrincessRequestHandler,
)
from msabp_opt.simulation.distributed.princess import (  # noqa: E402
    PrincessCoordinator,
)
from msabp_opt.simulation.distributed.protocol import make_message  # noqa: E402
from msabp_opt.simulation.distributed.state import (  # noqa: E402
    PrincessState,
    freeze_csv,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _coordinator(
    tmp_path: Path,
    *,
    case_count: int = 1,
) -> tuple[PrincessCoordinator, PrincessState, str]:
    source = tmp_path / "samples.csv"
    rows = ["sample_id,value,geometry_valid"]
    rows.extend(f"{index},{index + 0.25},True" for index in range(case_count))
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    frozen = freeze_csv(source, tmp_path / "run" / "input.csv")
    state = PrincessState(tmp_path / "run" / "princess.sqlite3")
    state.initialize_run("run-1", frozen)
    coordinator = PrincessCoordinator(
        state,
        run_id="run-1",
        csv_sha256=frozen.sha256,
        results_dir=tmp_path / "results",
        incoming_dir=tmp_path / "incoming",
        lease_seconds=90.0,
        max_attempts=3,
    )
    return coordinator, state, frozen.sha256


def _request(
    message_type: str,
    *,
    worker_id: str = "maid-a",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_message(
        message_type,
        run_id="run-1",
        worker_id=worker_id,
        payload=payload or {},
    )


def _hello(
    coordinator: PrincessCoordinator,
    csv_sha256: str,
    worker_id: str = "maid-a",
) -> None:
    response = coordinator.handle_message(
        _request(
            "hello",
            worker_id=worker_id,
            payload={"csv_sha256": csv_sha256, "dry_run": False},
        )
    )
    assert response["type"] == "welcome"


def _claim(
    coordinator: PrincessCoordinator,
    worker_id: str = "maid-a",
    *,
    exclusions: list[str] | None = None,
) -> dict[str, Any]:
    response = coordinator.handle_message(
        _request(
            "request_task",
            worker_id=worker_id,
            payload={"exclude_case_ids": exclusions or []},
        )
    )
    assert response["type"] == "assignment"
    return response["payload"]


def _write_case_archive(
    path: Path,
    *,
    case_id: str,
    corrupt_s11_hash: bool = False,
    include_farfield: bool = True,
) -> tuple[str, str]:
    s11 = b"1.0 -12.0\n2.0 -8.0\n"
    rad_eff = b"1.0 -2.0\n2.0 -1.5\n"
    tot_eff = b"1.0 -3.0\n2.0 -2.5\n"
    farfield = b"farfield-source"
    artifacts: dict[str, dict[str, Any]] = {
        "s11": {
            "path": case_runner.S11_FILENAME,
            "size_bytes": len(s11),
            "sha256": "0" * 64 if corrupt_s11_hash else _sha256_bytes(s11),
        },
        "rad_eff": {
            "path": case_runner.RAD_EFF_FILENAME,
            "size_bytes": len(rad_eff),
            "sha256": _sha256_bytes(rad_eff),
        },
        "tot_eff": {
            "path": case_runner.TOT_EFF_FILENAME,
            "size_bytes": len(tot_eff),
            "sha256": _sha256_bytes(tot_eff),
        },
    }
    if include_farfield:
        artifacts["farfield_source"] = {
            "path": case_runner.FARFIELD_SOURCE_FILENAME,
            "size_bytes": len(farfield),
            "sha256": _sha256_bytes(farfield),
        }
    manifest = {
        "schema_version": case_runner.MANIFEST_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "completed",
        "dry_run": False,
        "parameters": {},
        "geometry": {},
        "artifacts": artifacts,
    }
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(case_runner.MANIFEST_FILENAME, manifest_bytes)
        handle.writestr(case_runner.S11_FILENAME, s11)
        handle.writestr(case_runner.RAD_EFF_FILENAME, rad_eff)
        handle.writestr(case_runner.TOT_EFF_FILENAME, tot_eff)
        if include_farfield:
            handle.writestr(case_runner.FARFIELD_SOURCE_FILENAME, farfield)
    return _sha256_bytes(path.read_bytes()), _sha256_bytes(manifest_bytes)


def _stage_archive(
    coordinator: PrincessCoordinator,
    assignment: dict[str, Any],
    archive: Path,
    archive_sha256: str,
    *,
    worker_id: str = "maid-a",
) -> dict[str, Any]:
    return coordinator.handle_artifact(
        assignment["attempt_id"],
        worker_id,
        assignment["lease_token"],
        archive,
        archive_sha256,
        archive.stat().st_size,
    )


def _complete_payload(
    assignment: dict[str, Any],
    archive_sha256: str,
    manifest_sha256: str,
) -> dict[str, str]:
    return {
        "attempt_id": assignment["attempt_id"],
        "lease_token": assignment["lease_token"],
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256,
    }


def test_valid_upload_and_complete_are_hash_checked_and_complete_is_idempotent(
    tmp_path: Path,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path)
    try:
        _hello(coordinator, csv_sha256)
        assignment = _claim(coordinator)
        archive = tmp_path / "case.zip"
        archive_sha256, manifest_sha256 = _write_case_archive(
            archive,
            case_id=assignment["case_id"],
        )

        upload = _stage_archive(
            coordinator,
            assignment,
            archive,
            archive_sha256,
        )
        assert upload == {
            "sha256": archive_sha256,
            "size": (coordinator.incoming_dir / f"{assignment['attempt_id']}.zip").stat().st_size,
            "status": "staged",
        }
        complete_request = _request(
            "complete",
            payload=_complete_payload(
                assignment,
                archive_sha256,
                manifest_sha256,
            ),
        )
        first = coordinator.handle_message(complete_request)
        second = coordinator.handle_message(complete_request)

        assert first["type"] == second["type"] == "completed_ack"
        assert second["payload"]["duplicate"] is True
        final_directory = Path(first["payload"]["artifact_path"])
        assert final_directory.name == "case_0000"
        assert (final_directory / case_runner.S11_FILENAME).is_file()
        assert state.get_task("run-1", "0")["status"] == "completed"
        assert not (
            coordinator.incoming_dir / f"{assignment['attempt_id']}.zip"
        ).exists()

        mismatched = _request(
            "complete",
            payload=_complete_payload(
                assignment,
                "f" * 64,
                manifest_sha256,
            ),
        )
        with pytest.raises(ApiError, match="does not match receipt") as exc_info:
            coordinator.handle_message(mismatched)
        assert exc_info.value.status == 409
    finally:
        state.close()


def test_concurrent_duplicate_completion_is_serialized_per_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path)
    release_first = threading.Event()
    first_entered = threading.Event()
    unexpected_second_entered = threading.Event()
    call_guard = threading.Lock()
    accept_calls = 0
    try:
        _hello(coordinator, csv_sha256)
        assignment = _claim(coordinator)
        archive = tmp_path / "case.zip"
        archive_sha256, manifest_sha256 = _write_case_archive(
            archive,
            case_id=assignment["case_id"],
        )
        _stage_archive(coordinator, assignment, archive, archive_sha256)

        original_accept = coordinator._accept_archive

        def blocking_accept(*args: Any, **kwargs: Any) -> tuple[Path, str]:
            nonlocal accept_calls
            with call_guard:
                accept_calls += 1
                call_number = accept_calls
            if call_number == 1:
                first_entered.set()
                assert release_first.wait(timeout=3.0)
            else:
                unexpected_second_entered.set()
            return original_accept(*args, **kwargs)

        monkeypatch.setattr(coordinator, "_accept_archive", blocking_accept)
        complete_request = _request(
            "complete",
            payload=_complete_payload(
                assignment,
                archive_sha256,
                manifest_sha256,
            ),
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(coordinator.handle_message, complete_request)
            assert first_entered.wait(timeout=3.0)
            second = executor.submit(coordinator.handle_message, complete_request)
            try:
                assert not unexpected_second_entered.wait(timeout=0.2)
            finally:
                release_first.set()
            responses = [first.result(timeout=3.0), second.result(timeout=3.0)]

        assert accept_calls == 1
        assert all(response["type"] == "completed_ack" for response in responses)
        assert sorted(
            bool(response["payload"].get("duplicate", False))
            for response in responses
        ) == [False, True]
        assert state.get_task("run-1", assignment["case_id"])["status"] == "completed"
    finally:
        release_first.set()
        state.close()


def test_concurrent_identical_uploads_are_serialized_per_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path)
    release_first = threading.Event()
    first_entered = threading.Event()
    second_entered = threading.Event()
    call_guard = threading.Lock()
    mark_calls = 0
    try:
        _hello(coordinator, csv_sha256)
        assignment = _claim(coordinator)
        first_archive = tmp_path / "first.zip"
        archive_sha256, _ = _write_case_archive(
            first_archive,
            case_id=assignment["case_id"],
        )
        second_archive = tmp_path / "second.zip"
        second_archive.write_bytes(first_archive.read_bytes())

        original_mark_ready = state.mark_artifact_ready

        def blocking_mark_ready(*args: Any, **kwargs: Any) -> Any:
            nonlocal mark_calls
            with call_guard:
                mark_calls += 1
                call_number = mark_calls
            if call_number == 1:
                first_entered.set()
                assert release_first.wait(timeout=3.0)
            else:
                second_entered.set()
            return original_mark_ready(*args, **kwargs)

        monkeypatch.setattr(state, "mark_artifact_ready", blocking_mark_ready)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                _stage_archive,
                coordinator,
                assignment,
                first_archive,
                archive_sha256,
            )
            assert first_entered.wait(timeout=3.0)
            second = executor.submit(
                _stage_archive,
                coordinator,
                assignment,
                second_archive,
                archive_sha256,
            )
            try:
                assert not second_entered.wait(timeout=0.2)
            finally:
                release_first.set()
            responses = [first.result(timeout=3.0), second.result(timeout=3.0)]

        assert mark_calls == 2
        assert second_entered.is_set()
        assert all(response["status"] == "staged" for response in responses)
        assert (
            coordinator.incoming_dir / f"{assignment['attempt_id']}.zip"
        ).is_file()
    finally:
        release_first.set()
        state.close()


def test_different_attempt_completions_can_accept_archives_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path, case_count=2)
    try:
        _hello(coordinator, csv_sha256, "maid-a")
        _hello(coordinator, csv_sha256, "maid-b")
        assignments = {
            "maid-a": _claim(coordinator, "maid-a"),
            "maid-b": _claim(coordinator, "maid-b"),
        }
        completion_requests: list[dict[str, Any]] = []
        for worker_id, assignment in assignments.items():
            archive = tmp_path / f"{worker_id}.zip"
            archive_sha256, manifest_sha256 = _write_case_archive(
                archive,
                case_id=assignment["case_id"],
            )
            _stage_archive(
                coordinator,
                assignment,
                archive,
                archive_sha256,
                worker_id=worker_id,
            )
            completion_requests.append(
                _request(
                    "complete",
                    worker_id=worker_id,
                    payload=_complete_payload(
                        assignment,
                        archive_sha256,
                        manifest_sha256,
                    ),
                )
            )

        original_accept = coordinator._accept_archive
        both_attempts_entered = threading.Barrier(2)

        def synchronized_accept(*args: Any, **kwargs: Any) -> tuple[Path, str]:
            both_attempts_entered.wait(timeout=3.0)
            return original_accept(*args, **kwargs)

        monkeypatch.setattr(coordinator, "_accept_archive", synchronized_accept)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(coordinator.handle_message, request)
                for request in completion_requests
            ]
            responses = [future.result(timeout=5.0) for future in futures]

        assert all(response["type"] == "completed_ack" for response in responses)
        assert state.progress("run-1").completed == 2
    finally:
        state.close()


def test_failure_waits_for_same_attempt_completion_and_cannot_overwrite_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path)
    release_completion = threading.Event()
    completion_entered = threading.Event()
    failure_state_call_entered = threading.Event()
    try:
        _hello(coordinator, csv_sha256)
        assignment = _claim(coordinator)
        archive = tmp_path / "case.zip"
        archive_sha256, manifest_sha256 = _write_case_archive(
            archive,
            case_id=assignment["case_id"],
        )
        _stage_archive(coordinator, assignment, archive, archive_sha256)

        original_accept = coordinator._accept_archive
        original_fail_task = state.fail_task

        def blocking_accept(*args: Any, **kwargs: Any) -> tuple[Path, str]:
            completion_entered.set()
            assert release_completion.wait(timeout=3.0)
            return original_accept(*args, **kwargs)

        def recording_fail_task(*args: Any, **kwargs: Any) -> Any:
            failure_state_call_entered.set()
            return original_fail_task(*args, **kwargs)

        monkeypatch.setattr(coordinator, "_accept_archive", blocking_accept)
        monkeypatch.setattr(state, "fail_task", recording_fail_task)
        complete_request = _request(
            "complete",
            payload=_complete_payload(
                assignment,
                archive_sha256,
                manifest_sha256,
            ),
        )
        failure_request = _request(
            "failure",
            payload={
                "attempt_id": assignment["attempt_id"],
                "lease_token": assignment["lease_token"],
                "error_kind": "SyntheticError",
                "error_message": "late failure",
                "retryable": True,
                "counts_toward_streak": True,
            },
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            completion = executor.submit(
                coordinator.handle_message,
                complete_request,
            )
            assert completion_entered.wait(timeout=3.0)
            failure = executor.submit(coordinator.handle_message, failure_request)
            try:
                assert not failure_state_call_entered.wait(timeout=0.2)
            finally:
                release_completion.set()
            assert completion.result(timeout=3.0)["type"] == "completed_ack"
            with pytest.raises(ApiError, match="already completed") as exc_info:
                failure.result(timeout=3.0)

        assert exc_info.value.status == 409
        assert not failure_state_call_entered.is_set()
        assert state.get_task("run-1", assignment["case_id"])["status"] == "completed"
    finally:
        release_completion.set()
        state.close()


def test_wrong_worker_cannot_overwrite_staged_archive_and_valid_retry_can(
    tmp_path: Path,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path)
    try:
        _hello(coordinator, csv_sha256, "maid-a")
        _hello(coordinator, csv_sha256, "maid-b")
        assignment = _claim(coordinator, "maid-a")
        first = tmp_path / "first.zip"
        first_sha256, _ = _write_case_archive(
            first,
            case_id=assignment["case_id"],
        )
        _stage_archive(coordinator, assignment, first, first_sha256)
        staged = coordinator.incoming_dir / f"{assignment['attempt_id']}.zip"
        original_bytes = staged.read_bytes()

        hostile = tmp_path / "hostile.zip"
        hostile.write_bytes(b"not the assigned worker's artifact")
        with pytest.raises(ApiError, match="does not own") as exc_info:
            coordinator.handle_artifact(
                assignment["attempt_id"],
                "maid-b",
                assignment["lease_token"],
                hostile,
                _sha256_bytes(hostile.read_bytes()),
                hostile.stat().st_size,
            )
        assert exc_info.value.status == 409
        assert staged.read_bytes() == original_bytes

        retry = tmp_path / "retry.zip"
        retry.write_bytes(original_bytes)
        response = coordinator.handle_artifact(
            assignment["attempt_id"],
            "maid-a",
            assignment["lease_token"],
            retry,
            first_sha256,
            retry.stat().st_size,
        )
        assert response["status"] == "staged"
        assert staged.read_bytes() == original_bytes

        different = tmp_path / "different.zip"
        different.write_bytes(b"different staged payload")
        with pytest.raises(ApiError, match="different staged artifact"):
            coordinator.handle_artifact(
                assignment["attempt_id"],
                "maid-a",
                assignment["lease_token"],
                different,
                _sha256_bytes(different.read_bytes()),
                different.stat().st_size,
            )
        assert staged.read_bytes() == original_bytes
    finally:
        state.close()


@pytest.mark.parametrize(
    "member_name",
    ["../escape.txt", "..\\escape.txt", "C:/escape.txt", "folder/file. "],
)
def test_zip_member_paths_cannot_escape_or_alias_on_windows(
    tmp_path: Path,
    member_name: str,
) -> None:
    coordinator, state, _ = _coordinator(tmp_path)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member_name, b"hostile")
    try:
        with pytest.raises(ApiError, match="unsafe") as exc_info:
            coordinator._accept_archive(
                "attempt-safe",
                archive,
                "0" * 64,
                expected_case_id="0",
            )
        assert exc_info.value.status == 400
        assert not (tmp_path / "escape.txt").exists()
    finally:
        state.close()


@pytest.mark.parametrize(
    ("case_id", "corrupt_hash", "include_farfield", "message"),
    [
        ("1", False, True, "does not match leased case"),
        ("0", True, True, "SHA-256 mismatch"),
        ("0", False, False, "missing required artifacts"),
    ],
)
def test_manifest_identity_and_declared_artifacts_are_enforced(
    tmp_path: Path,
    case_id: str,
    corrupt_hash: bool,
    include_farfield: bool,
    message: str,
) -> None:
    coordinator, state, _ = _coordinator(tmp_path)
    archive = tmp_path / "case.zip"
    _, manifest_sha256 = _write_case_archive(
        archive,
        case_id=case_id,
        corrupt_s11_hash=corrupt_hash,
        include_farfield=include_farfield,
    )
    try:
        with pytest.raises(ApiError, match=message):
            coordinator._accept_archive(
                "attempt-safe",
                archive,
                manifest_sha256,
                expected_case_id="0",
            )
    finally:
        state.close()


def test_exclusions_fall_back_once_when_they_cover_all_pending_cases(
    tmp_path: Path,
) -> None:
    coordinator, state, csv_sha256 = _coordinator(tmp_path)
    try:
        _hello(coordinator, csv_sha256)
        first = _claim(coordinator)
        failure = coordinator.handle_message(
            _request(
                "failure",
                payload={
                    "attempt_id": first["attempt_id"],
                    "lease_token": first["lease_token"],
                    "error_kind": "SyntheticError",
                    "error_message": "first attempt failed",
                    "retryable": True,
                    "counts_toward_streak": True,
                },
            )
        )
        assert failure["type"] == "failure_ack"

        retried = _claim(coordinator, exclusions=[first["case_id"]])
        assert retried["case_id"] == first["case_id"]
        assert retried["attempt_id"] != first["attempt_id"]
    finally:
        state.close()


def test_http_bearer_authentication_is_constant_time_checked_without_server() -> None:
    handler = object.__new__(PrincessRequestHandler)
    handler.server = SimpleNamespace(api_token="secret-token")
    handler.headers = {"Authorization": "Bearer wrong-token"}
    with pytest.raises(ApiError, match="invalid bearer token") as exc_info:
        handler._require_auth()
    assert exc_info.value.status == 401

    handler.headers = {"Authorization": "Bearer secret-token"}
    handler._require_auth()


def test_http_client_rejects_unsafe_upload_identity_before_network(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "artifact.zip"
    archive.write_bytes(b"payload")
    client = PrincessClient("http://princess.invalid", "cluster-token")

    with pytest.raises(ApiError, match="invalid attempt id"):
        client.upload_artifact("../escape", "maid-a", "lease", archive)
    with pytest.raises(ApiError, match="invalid worker id"):
        client.upload_artifact("attempt-1", "maid/a", "lease", archive)
    with pytest.raises(ValueError, match="line break"):
        client.upload_artifact(
            "attempt-1",
            "maid-a",
            "lease\r\nInjected: yes",
            archive,
        )
