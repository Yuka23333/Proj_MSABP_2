from __future__ import annotations

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.state import (  # noqa: E402
    ControlAuthorityError,
    PrincessState,
    validate_frozen_csv,
)


class FakeAuthority:
    """Synthetic permission issuer; no Shore service, keys, executor, or network."""

    def __init__(self) -> None:
        self.binding = {"job_id": "whole-doe", "epoch": 1, "host_id": "synthetic", "session_id": "s1"}
        self.controller = SimpleNamespace(authority_id="test-authority", fleet_id="test-fleet")
        self.job = {"scope": {"project_id": "synthetic-project", "collector_url": "synthetic-only",
                              "data_root": "synthetic-results", "job_spec": {"approved_units": 128}}}
        self.draining = False
        self.unavailable = False
        self.crash = ""
        self.calls: list[dict] = []
        self.permits: dict[str, dict] = {}
        self.database: Path | None = None

    def require_new_work(self) -> None:
        if self.draining or self.unavailable:
            raise ControlAuthorityError("new work denied")

    def descriptor_for_attempt(self, intent: dict) -> dict:
        return {
            **self.binding,
            **{name: intent[name] for name in (
                "operation_id", "attempt_id", "run_id", "worker_id", "input_sha256"
            )},
            "action": "attempt",
        }

    def issue_attempt(self, intent: dict) -> dict:
        self.calls.append(intent)
        assert self.database is not None
        with sqlite3.connect(self.database, timeout=0.1) as connection:
            # Acquiring a write txn proves the state made no RPC while holding
            # its own transaction. Full descriptor is already crash-durable.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT descriptor_json FROM control_dispatch_intents WHERE operation_id = ?",
                (intent["operation_id"],),
            ).fetchone()
            assert json.loads(row[0]) == intent["descriptor"]
        if self.crash == "before_issue":
            self.crash = ""
            raise TimeoutError("unknown dispatch outcome before issuance")
        operation_id = intent["operation_id"]
        if operation_id not in self.permits:
            if self.draining or self.unavailable:
                raise ControlAuthorityError("new dispatch denied")
            self.permits[operation_id] = {
                "descriptor": intent["descriptor"],
                "permit": {"synthetic_signature": operation_id},
            }
        if self.crash == "after_issue":
            self.crash = ""
            raise TimeoutError("permit issued but RPC reply lost")
        return self.permits[operation_id]


def controlled_state(tmp_path: Path, *, count: int = 3) -> tuple[PrincessState, FakeAuthority]:
    csv = tmp_path / "frozen.csv"
    csv.write_text("sample_id,x\n" + "".join(f"{i},{i + 1}\n" for i in range(count)), encoding="utf-8")
    authority = FakeAuthority()
    authority.database = tmp_path / "state.sqlite3"
    state = PrincessState(authority.database, control_authority=authority)
    state.initialize_run("run", validate_frozen_csv(csv), now=1)
    state.register_worker("run", "maid-a", now=1)
    state.register_worker("run", "maid-b", now=1)
    return state, authority


@pytest.mark.parametrize("window", ["before_issue", "after_issue", "before_commit", "after_commit"])
def test_dispatch_crash_windows_replay_same_operation(tmp_path: Path, monkeypatch, window: str) -> None:
    state, authority = controlled_state(tmp_path)
    if window in {"before_issue", "after_issue"}:
        authority.crash = window
        with pytest.raises(TimeoutError):
            state.claim_next("run", "maid-a", now=10)
    elif window == "before_commit":
        def crash_commit(*args, **kwargs):
            raise RuntimeError("process died after durable permit, before assignment")
        monkeypatch.setattr(state, "_commit_controlled_claim", crash_commit)
        with pytest.raises(RuntimeError):
            state.claim_next("run", "maid-a", now=10)
    else:
        assert state.claim_next("run", "maid-a", now=10) is not None
    intent = authority.calls[0]
    state.close()
    with PrincessState(authority.database, control_authority=authority) as reopened:
        recovered = reopened.claim_next("run", "maid-a", now=1000)
        assert recovered is not None
        assert recovered.attempt_id == intent["attempt_id"]
        assert recovered.control_authority["descriptor"] == intent["descriptor"]
        assert {call["operation_id"] for call in authority.calls} == {intent["operation_id"]}
        assert len(authority.permits) == 1
        assert reopened.get_task("run", "0")["attempt_count"] == 1
        assert reopened.claim_next("run", "maid-a", now=1001).attempt_id == recovered.attempt_id


def test_pending_intent_reserves_worker_and_task_across_other_claims(tmp_path: Path) -> None:
    state, authority = controlled_state(tmp_path)
    authority.crash = "after_issue"
    with pytest.raises(TimeoutError):
        state.claim_next("run", "maid-a", now=10)
    original = authority.calls[0]
    other = state.claim_next("run", "maid-b", now=11)
    assert other is not None and other.case_id == "1"
    first = state.claim_next("run", "maid-a", exclude_case_ids=("0",), now=12)
    assert first is not None and first.case_id == "0"
    assert first.attempt_id == original["attempt_id"]
    state.close()


def test_drain_keeps_lease_ownership_and_late_completion(tmp_path: Path) -> None:
    state, authority = controlled_state(tmp_path)
    first = state.claim_next("run", "maid-a", lease_seconds=2, now=10)
    authority.draining = True
    assert state.release_expired_leases("run", now=1000) == ()
    assert state.claim_next("run", "maid-a", now=1000) == first
    with pytest.raises(ControlAuthorityError, match="denied"):
        state.claim_next("run", "maid-b", now=1000)
    assert state.heartbeat("run", "maid-a", first.attempt_id, first.lease_token, now=1000) == 1090
    state.mark_artifact_ready("run", "maid-a", first.attempt_id, first.lease_token, artifact_path="fake.zip", now=2000)
    with pytest.raises(ControlAuthorityError, match="verified collector evidence"):
        state.complete_task("run", "maid-a", first.attempt_id, first.lease_token, now=2001)
    state.approve_control_completion(first.attempt_id, "a" * 64)
    state.complete_task("run", "maid-a", first.attempt_id, first.lease_token, now=2001)
    state.complete_task("run", "maid-a", first.attempt_id, first.lease_token, now=2002)
    assert state.progress("run").completed == 1
    assert state.control_for_attempt(first.attempt_id) == first.control_authority
    state.close()


def test_reopen_without_authority_fails_closed_but_retains_existing_traffic(tmp_path: Path) -> None:
    state, authority = controlled_state(tmp_path)
    first = state.claim_next("run", "maid-a", lease_seconds=1, now=10)
    state.approve_control_completion(first.attempt_id, "b" * 64)
    state.close()
    with PrincessState(authority.database) as reopened:
        assert reopened.is_controlled_run("run")
        for operation in (
            lambda: reopened.claim_next("run", "maid-a", now=1000),
            lambda: reopened.claim_next("run", "maid-b", now=1000),
            lambda: reopened.request_stop("run", reason="drain", now=1000),
            lambda: reopened.resume_run("run", now=1000),
            lambda: reopened.release_task("run", "maid-a", first.attempt_id, first.lease_token, reason="lost"),
            lambda: reopened.fail_task("run", "maid-a", first.attempt_id, first.lease_token, error_kind="fake", error_message="failed"),
            lambda: reopened.prepare_worker_restart("run", "maid-b"),
            lambda: reopened.wake_worker("run", "maid-b"),
        ):
            with pytest.raises(ControlAuthorityError):
                operation()
        assert reopened.release_expired_leases("run", now=1000) == ()
        reopened.heartbeat("run", "maid-a", first.attempt_id, first.lease_token, now=1000)
        reopened.mark_artifact_ready("run", "maid-a", first.attempt_id, first.lease_token, artifact_path="late.zip", now=2000)
        reopened.complete_task("run", "maid-a", first.attempt_id, first.lease_token, now=2001)
        assert reopened.progress("run").completed == 1


def test_authority_outage_does_not_allocate_new_intent(tmp_path: Path) -> None:
    state, authority = controlled_state(tmp_path)
    authority.unavailable = True
    with pytest.raises(ControlAuthorityError):
        state.claim_next("run", "maid-a", now=10)
    assert not authority.calls
    assert state.get_task("run", "0")["attempt_count"] == 0
    state.close()


def test_rotated_session_replays_pinned_descriptor_but_other_job_rejected(tmp_path: Path) -> None:
    state, authority = controlled_state(tmp_path)
    authority.crash = "after_issue"
    with pytest.raises(TimeoutError):
        state.claim_next("run", "maid-a", now=10)
    original = authority.calls[0]
    authority.binding["session_id"] = "s2"
    authority.draining = True
    recovered = state.claim_next("run", "maid-a", now=20)
    assert recovered.control_authority["descriptor"] == original["descriptor"]
    authority.binding["job_id"] = "different-job"
    with pytest.raises(ControlAuthorityError, match="binding"):
        state.claim_next("run", "maid-a", now=21)
    with pytest.raises(ControlAuthorityError, match="binding"):
        state.bind_control_run("run")
    state.close()


def test_parallel_claims_share_one_durable_intent(tmp_path: Path) -> None:
    state, authority = controlled_state(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: state.claim_next("run", "maid-a", now=10), range(4)))
    assert len({claim.attempt_id for claim in claims}) == 1
    assert len(authority.permits) == 1
    assert state.progress("run").running == 1
    state.close()


@pytest.mark.parametrize("changed", ["authority_id", "fleet_id", "job_spec", "collector_url", "data_root"])
def test_reopen_pins_authority_fleet_and_entire_job_scope(tmp_path: Path, changed: str) -> None:
    state, authority = controlled_state(tmp_path)
    state.close()
    if changed in {"authority_id", "fleet_id"}:
        setattr(authority.controller, changed, "different")
    elif changed == "job_spec":
        authority.job["scope"]["job_spec"]["approved_units"] = 129
    else:
        authority.job["scope"][changed] = "different"
    with PrincessState(authority.database, control_authority=authority) as reopened:
        assert reopened.is_controlled_run("run")
        with pytest.raises(ControlAuthorityError, match="binding"):
            reopened.claim_next("run", "maid-a", now=10)
        with pytest.raises(ControlAuthorityError, match="binding"):
            reopened.bind_control_run("run")
        assert authority.calls == []


@pytest.mark.parametrize("outcome", ["failed", "not_started"])
def test_only_approved_terminal_can_settle_and_replays_are_immutable(tmp_path: Path, outcome: str) -> None:
    state, authority = controlled_state(tmp_path)
    claim = state.claim_next("run", "maid-a", now=10)
    arguments = ("run", "maid-a", claim.attempt_id, claim.lease_token)
    terminal = dict(outcome=outcome, evidence_sha256="d" * 64, artifact_path="verified-terminal")
    with pytest.raises(ControlAuthorityError, match="approved"):
        state.settle_control_terminal(*arguments, **terminal, now=20)
    with pytest.raises(ControlAuthorityError, match="only proven"):
        state.settle_control_terminal(*arguments, **{**terminal, "outcome": "unknown"}, now=20)
    state.approve_control_completion(claim.attempt_id, terminal["evidence_sha256"])
    with pytest.raises(ControlAuthorityError, match="exact approved"):
        state.settle_control_terminal(*arguments, **{**terminal, "evidence_sha256": "e" * 64}, now=20)
    authority.draining = True
    assert state.settle_control_terminal(*arguments, **terminal, now=20) == "0"
    assert state.settle_control_terminal(*arguments, **terminal, now=21) == "0"
    assert state.get_task("run", "0")["status"] == ("failed" if outcome == "failed" else "pending")
    assert state.get_worker("run", "maid-a")["current_attempt_id"] is None
    with pytest.raises(ControlAuthorityError, match="cannot change"):
        state.settle_control_terminal(*arguments, **{**terminal, "artifact_path": "changed"}, now=21)
    with pytest.raises(ControlAuthorityError, match="denied"):
        state.claim_next("run", "maid-a", now=22)
    authority.draining = False
    next_claim = state.claim_next("run", "maid-a", now=23)
    assert next_claim.attempt_id != claim.attempt_id
    assert next_claim.case_id == ("1" if outcome == "failed" else "0")
    # Replaying the old receipt must not erase a newer worker assignment.
    assert state.settle_control_terminal(*arguments, **terminal, now=24) == "0"
    assert state.get_worker("run", "maid-a")["current_attempt_id"] == next_claim.attempt_id
    state.close()
