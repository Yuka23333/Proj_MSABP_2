from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.protocol import (  # noqa: E402
    FRAME_MAGIC,
    ProtocolError,
    decode_frame,
    encode_frame,
    iter_frames,
    make_message,
    validate_message,
)
from msabp_opt.simulation.distributed.state import (  # noqa: E402
    LeaseError,
    PrincessState,
    RunMismatchError,
    WorkerUnavailableError,
    freeze_csv,
    validate_frozen_csv,
)


def _write_samples(path: Path, count: int = 10) -> Path:
    rows = ["sample_id,value,geometry_valid"]
    rows.extend(f"{index},{index + 0.25},True" for index in range(count))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _initialized_state(tmp_path: Path, *, count: int = 10) -> PrincessState:
    source = _write_samples(tmp_path / "source.csv", count)
    frozen = freeze_csv(source, tmp_path / "run" / "input.csv")
    state = PrincessState(tmp_path / "run" / "princess.sqlite3")
    assert state.initialize_run("run-1", frozen, metadata={"purpose": "test"})
    assert not state.initialize_run("run-1", frozen)
    return state


def test_protocol_envelope_is_transport_neutral_and_jsonl_safe() -> None:
    message = make_message(
        "heartbeat",
        run_id="run-1",
        worker_id="maid-coconut",
        message_id="message-1",
        sent_at_utc="2026-08-04T00:00:00+00:00",
        payload={"phase": "solve", "note": "仍在运行"},
    )

    assert validate_message(message) == message
    encoded = encode_frame(message)
    assert encoded.startswith(FRAME_MAGIC)
    assert encoded.endswith("\n")
    assert decode_frame(encoded) == message
    assert list(iter_frames(["OpenSSH banner\n", encoded])) == [message]

    with pytest.raises(ProtocolError, match="marker"):
        decode_frame("plain JSON is not a framed message\n")
    with pytest.raises(ProtocolError, match="unsupported"):
        validate_message({**message, "protocol_version": 999})


def test_freeze_csv_is_atomic_idempotent_and_hash_checked(tmp_path: Path) -> None:
    source = _write_samples(tmp_path / "source.csv", 3)
    destination = tmp_path / "frozen" / "input.csv"

    first = freeze_csv(source, destination)
    second = freeze_csv(source, destination)
    checked = validate_frozen_csv(destination, expected_sha256=first.sha256)

    assert first == second == checked
    assert first.row_count == 3
    assert [row.case_id for row in first.rows] == ["0", "1", "2"]
    assert len({row.row_sha256 for row in first.rows}) == 3

    _write_samples(source, 4)
    with pytest.raises(FileExistsError, match="different content"):
        freeze_csv(source, destination)
    with pytest.raises(RunMismatchError, match="SHA-256 mismatch"):
        validate_frozen_csv(destination, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("first_case_id", "second_case_id"),
    (("1", "0001"), ("A", "a"), ("slot", "SLOT")),
)
def test_frozen_csv_rejects_case_ids_with_same_windows_result_directory(
    tmp_path: Path,
    first_case_id: str,
    second_case_id: str,
) -> None:
    source = tmp_path / "conflicting.csv"
    source.write_text(
        "sample_id,value\n"
        f"{first_case_id},1\n"
        f"{second_case_id},2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same Princess result directory"):
        validate_frozen_csv(source)


def test_claim_exclusion_heartbeat_artifact_and_failure_lifecycle(
    tmp_path: Path,
) -> None:
    with _initialized_state(tmp_path, count=6) as state:
        state.register_worker("run-1", "maid-a", host="local")
        state.register_worker("run-1", "maid-b", host="remote")

        first = state.claim_next(
            "run-1",
            "maid-a",
            lease_seconds=10,
            exclude_case_ids=("0",),
            now=100,
        )
        assert first is not None and first.case_id == "1"
        assert first.row_index == 1
        assert first.csv_row_index == 1
        # HTTP request retries call ready before asking for the same assignment.
        state.mark_worker_ready("run-1", "maid-a", now=101)
        assert state.claim_next("run-1", "maid-a", now=101) == first
        assert state.heartbeat(
            "run-1",
            "maid-a",
            first.attempt_id,
            first.lease_token,
            lease_seconds=20,
            phase="solve",
            now=105,
        ) == 125

        second = state.claim_next("run-1", "maid-b", now=105)
        assert second is not None and second.case_id == "0"

        state.mark_artifact_ready(
            "run-1",
            "maid-a",
            first.attempt_id,
            first.lease_token,
            artifact_path="incoming/attempt.zip",
            manifest_sha256="abc123",
            now=106,
        )
        assert state.progress("run-1").running == 2
        state.complete_task(
            "run-1",
            "maid-a",
            first.attempt_id,
            first.lease_token,
            now=107,
        )

        outcome = state.fail_task(
            "run-1",
            "maid-b",
            second.attempt_id,
            second.lease_token,
            error_kind="SolverError",
            error_message="solver stopped",
            retryable=True,
            max_attempts=3,
            now=108,
        )
        assert outcome.task_status == "pending"
        assert outcome.consecutive_errors == 1
        assert not outcome.threshold_reached
        assert state.progress("run-1").as_dict() == {
            "total": 6,
            "pending": 5,
            "running": 0,
            "completed": 1,
            "failed": 0,
            "finished": 1,
            "is_terminal": False,
        }


def test_expired_lease_is_requeued_and_stale_completion_is_rejected(
    tmp_path: Path,
) -> None:
    with _initialized_state(tmp_path, count=2) as state:
        state.register_worker("run-1", "maid-a")
        state.register_worker("run-1", "maid-b")
        claim = state.claim_next("run-1", "maid-a", lease_seconds=10, now=100)
        assert claim is not None

        assert state.release_expired_leases("run-1", now=111) == (claim.case_id,)
        assert state.get_worker("run-1", "maid-a")["status"] == "offline"
        reassigned = state.claim_next("run-1", "maid-b", now=112)
        assert reassigned is not None and reassigned.case_id == claim.case_id
        assert reassigned.attempt_id != claim.attempt_id

        with pytest.raises(LeaseError, match="does not own"):
            state.complete_task(
                "run-1",
                "maid-a",
                claim.attempt_id,
                claim.lease_token,
                now=113,
            )


def test_artifact_ready_attempt_is_requeued_if_maid_drops_before_complete(
    tmp_path: Path,
) -> None:
    with _initialized_state(tmp_path, count=1) as state:
        state.register_worker("run-1", "maid-a")
        claim = state.claim_next("run-1", "maid-a", lease_seconds=10, now=100)
        assert claim is not None
        state.mark_artifact_ready(
            "run-1",
            "maid-a",
            claim.attempt_id,
            claim.lease_token,
            artifact_path="incoming/result.zip",
            now=105,
        )
        assert state.progress("run-1").running == 1

        assert state.release_expired_leases("run-1", now=111) == (claim.case_id,)
        assert state.progress("run-1").as_dict() == {
            "total": 1,
            "pending": 1,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "finished": 0,
            "is_terminal": False,
        }
        assert state.get_worker("run-1", "maid-a")["status"] == "offline"
        with pytest.raises(LeaseError, match="does not own"):
            state.complete_task(
                "run-1",
                "maid-a",
                claim.attempt_id,
                claim.lease_token,
                now=112,
            )


def test_repeated_process_crashes_fail_case_at_lease_attempt_limit(
    tmp_path: Path,
) -> None:
    with _initialized_state(tmp_path, count=1) as state:
        state.register_worker("run-1", "maid-a")

        for attempt_number in range(1, 4):
            claim = state.claim_next(
                "run-1",
                "maid-a",
                lease_seconds=10,
                max_attempts=3,
                now=attempt_number * 100,
            )
            assert claim is not None
            assert claim.attempt_number == attempt_number
            assert state.release_expired_leases(
                "run-1",
                max_attempts=3,
                now=attempt_number * 100 + 11,
            ) == ("0",)
            if attempt_number < 3:
                assert state.get_task("run-1", "0")["status"] == "pending"
                state.wake_worker(
                    "run-1",
                    "maid-a",
                    now=attempt_number * 100 + 12,
                )

        task = state.get_task("run-1", "0")
        assert task["status"] == "failed"
        assert task["last_error_kind"] == "LeaseExpired"
        assert state.progress("run-1").is_terminal
        state.wake_worker("run-1", "maid-a", now=400)
        assert state.claim_next(
            "run-1",
            "maid-a",
            max_attempts=3,
            now=401,
        ) is None


def test_claim_is_atomic_across_concurrent_http_handlers(tmp_path: Path) -> None:
    with _initialized_state(tmp_path, count=2) as state:
        state.register_worker("run-1", "maid-a")
        state.register_worker("run-1", "maid-b")
        barrier = threading.Barrier(2)

        def claim(worker_id: str):
            barrier.wait(timeout=5)
            return state.claim_next("run-1", worker_id, now=100)

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(claim, ("maid-a", "maid-b")))

        assert all(item is not None for item in claims)
        assert {item.case_id for item in claims if item is not None} == {"0", "1"}
        assert state.progress("run-1").running == 2


def test_five_error_bundle_restart_once_then_quarantine(tmp_path: Path) -> None:
    with _initialized_state(tmp_path, count=10) as state:
        state.register_worker("run-1", "maid-a")

        last_outcome = None
        for index in range(5):
            claim = state.claim_next("run-1", "maid-a", now=100 + index * 2)
            assert claim is not None
            last_outcome = state.fail_task(
                "run-1",
                "maid-a",
                claim.attempt_id,
                claim.lease_token,
                error_kind="CstConnectionError",
                error_message=f"failure {index}",
                traceback_text=f"traceback {index}",
                log_path=f"logs/{index}.txt",
                retryable=False,
                streak_threshold=5,
                now=101 + index * 2,
            )

        assert last_outcome is not None and last_outcome.threshold_reached
        assert [item.case_id for item in last_outcome.failure_bundle] == [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]
        with pytest.raises(WorkerUnavailableError):
            state.claim_next("run-1", "maid-a", now=120)

        first_restart = state.prepare_worker_restart("run-1", "maid-a", now=121)
        assert first_restart.should_restart
        assert first_restart.status == "restarting"
        state.mark_worker_ready("run-1", "maid-a", now=122)

        for index in range(5, 10):
            claim = state.claim_next("run-1", "maid-a", now=200 + index * 2)
            assert claim is not None
            last_outcome = state.fail_task(
                "run-1",
                "maid-a",
                claim.attempt_id,
                claim.lease_token,
                error_kind="CstConnectionError",
                error_message=f"failure {index}",
                retryable=False,
                streak_threshold=5,
                now=201 + index * 2,
            )

        assert last_outcome is not None and last_outcome.threshold_reached
        second_restart = state.prepare_worker_restart("run-1", "maid-a", now=250)
        assert not second_restart.should_restart
        assert second_restart.status == "quarantined"
        assert state.progress("run-1").is_terminal
        assert state.get_worker("run-1", "maid-a")["status"] == "quarantined"

        state.wake_worker("run-1", "maid-a", now=251)
        worker = state.get_worker("run-1", "maid-a")
        assert worker["status"] == "idle"
        assert worker["automatic_restarts_without_success"] == 0
