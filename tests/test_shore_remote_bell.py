"""Offline fake-key wake lifecycle: no sockets, USB, installed keys or process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control", reason="explicit optional Shore checkout required")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.authority import Authority, digest
from shore_control.client import Client, MemorySigner
from shore_control.crypto import canonical_json, public_key_text
from shore_control.crypto import verify_object

from msabp_opt.simulation.distributed.control_authority import ControlAuthority
from msabp_opt.simulation.distributed.control_remote_bell import (
    BellRunner, FixedMaidLauncher, IDLE_DOMAIN, READY_DOMAIN, WakeCoordinator, verify_wake_evidence,
)
from msabp_opt.simulation.distributed.control_worker import ExecutionUncertain


class Transport:
    def __init__(self, authority):
        self.authority, self.drop_method, self.seen = authority, None, []

    def challenge(self, actor):
        return self.authority.challenge(actor)

    def handle(self, request):
        self.seen.append(request)
        response = self.authority.handle(request)
        if request["payload"]["method"] == self.drop_method:
            self.drop_method = None
            raise TimeoutError("synthetic reply loss after authority commit")
        return response


@pytest.fixture
def env(tmp_path):
    keys = {name: Ed25519PrivateKey.generate() for name in ("authority", "controller", "session", "bell", "maid")}
    registry = {role: {"public_key": public_key_text(keys[role].public_key()),
                      "device_id": "test-device", "tailnet_node_id": "test-node", "role": role,
                      "allowed_actions": ["wake" if role == "bell" else "attempt"]}
                for role in ("bell", "maid")}
    authority = Authority(tmp_path / "authority.sqlite", keys["authority"], fleet_id="test-fleet",
                          controllers={"controller": public_key_text(keys["controller"].public_key())},
                          workers=registry, lease_seconds=300)
    transport = Transport(authority)
    clients = {actor: Client(transport, "session:session" if actor == "session" else actor,
                            MemorySigner(keys[actor]), keys["authority"].public_key(), "test-fleet")
               for actor in ("controller", "session", "bell", "maid")}
    gate = ControlAuthority(tmp_path / "control.sqlite", controller_client=clients["controller"],
                            session_client=clients["session"], worker_registrations=registry)
    spec = {"schema": "shore.job-spec", "schema_version": 1, "manifest_sha256": digest({"job": "test"}),
            "approved_units": 1, "budget_unit": "objective_evaluation", "accounting_policy_sha256": digest({"policy": "test"})}
    gate.acquire(job_id="job", host_id="controller-node", session_id="session",
        session_public_key=public_key_text(keys["session"].public_key()), project_id="project",
        collector_url="http://100.64.0.1:8090", data_root=r"D:\Synthetic\Bell", job_spec=spec)
    gate.register_run("run", digest({"plan": "synthetic"}), digest({"input": "synthetic"}))
    runtime = {"schema": "offline-fixture-runtime", "synthetic": True, "worker_id": "maid"}
    binding = {**gate.binding, **gate.job["scope"], "fleet_id": "test-fleet"}

    def coordinator(**changes):
        args = dict(binding=binding, registration=registry["bell"], target_actor_id="maid",
                    target_registration=registry["maid"], runtime=runtime, run_id="run")
        args.update(changes)
        return WakeCoordinator(tmp_path / "bell-state", clients["bell"], keys["authority"].public_key(), **args)

    def assignment(number=1):
        value = gate.issue_operation({"operation_id": f"wake-{number}", "run_id": "run", "worker_id": "bell",
            "attempt_id": f"wake-attempt-{number}", "action": "wake", "input_sha256": digest(runtime),
            "action_context": {"target_actor_id": "maid", "runtime_sha256": digest(runtime)}})
        return {"control_authority": value, "runtime": runtime}

    def ready(coordinator, attempt_id="wake-attempt-1", *, signer=None, change=None, ready_path=None):
        record = coordinator.record(attempt_id)
        permit = coordinator._permit(record)
        payload = {"schema": READY_DOMAIN, "schema_version": 1, "launch_id": record["launch_id"],
                   "permit_id": permit["permit_id"], "runtime_sha256": digest(runtime), "actor_id": "maid",
                   **{k: permit[k] for k in ("device_id", "fleet_id", "job_id", "epoch", "run_id")},
                   "pid": 42, "process_started_utc": "2026-10-16T12:00:00+00:00"}
        payload.update(change or {})
        envelope = (signer or clients["maid"].signer).sign(READY_DOMAIN, payload)
        path = ready_path or tmp_path / "maid-state" / ("ready." + record["launch_id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(envelope))
        return path

    return SimpleNamespace(**locals())


def test_acceptance_reply_loss_replays_exact_request_and_does_not_launch(env):
    c, assignment = env.coordinator(), env.assignment()
    env.transport.drop_method = "accept_permit"
    with pytest.raises(TimeoutError):
        c.accept(assignment)
    original = c.rpc.pending()[0]["request"]
    assert c.record("wake-attempt-1")["state"] == "PREPARED"
    c = env.coordinator()
    c.accept(assignment)
    calls = [r for r in env.transport.seen if r["payload"]["method"] == "accept_permit"]
    assert calls == [original, original]
    assert c.record("wake-attempt-1")["state"] == "ACCEPTED"


def test_only_actual_maid_signed_ready_settles_and_dual_receipts_required(env):
    c = env.coordinator()
    assignment = env.assignment()
    permit = c.accept(assignment)
    launches = []
    c.launch("wake-attempt-1", lambda *paths: launches.append(paths), ready_path=env.tmp_path / "ready.json")
    assert len(launches) == 1
    with pytest.raises(ExecutionUncertain):
        c.checked_result("wake-attempt-1")
    result = c.record_ready("wake-attempt-1", env.ready(c))
    assert result["evidence"]["result"]["state"] == "started"
    assert c.report("wake-attempt-1")["status"] == "AWAITING_PEER_RECEIPT"
    assert c.resolved("wake-attempt-1") is False
    assert verify_wake_evidence(result["evidence"], permit, result["root"],
        authority_public_key=env.keys["authority"].public_key(), bell_registration=env.registry["bell"],
        target_registration=env.registry["maid"]) == result["receipt"]["evidence_sha256"]
    env.gate.ack_evidence(assignment["control_authority"]["permit"], result["evidence"], result["root"])
    assert c.resolved("wake-attempt-1") is True
    assert env.coordinator().record("wake-attempt-1")["state"] == "RESOLVED"
    with pytest.raises(ExecutionUncertain):
        c.launch("wake-attempt-1", lambda *_: pytest.fail("relaunch"), ready_path=env.tmp_path / "ready.json")


def test_crash_after_starting_never_relaunches_or_cancels(env):
    c = env.coordinator()
    c.accept(env.assignment())

    def crash(*_):
        raise OSError("ambiguous process creation boundary")

    with pytest.raises(OSError):
        c.launch("wake-attempt-1", crash, ready_path=env.tmp_path / "ready.json")
    restarted = env.coordinator()
    with pytest.raises(ExecutionUncertain):
        restarted.launch("wake-attempt-1", lambda *_: pytest.fail("relaunch"), ready_path=env.tmp_path / "ready.json")
    with pytest.raises(ValueError, match="Unknown or started"):
        restarted.not_started("wake-attempt-1")
    runner = BellRunner(restarted, None, lambda *_: pytest.fail("relaunch"), ready_root=env.tmp_path / "maid-state")
    assert runner.step()["state"] == "STARTING"


@pytest.mark.parametrize("change", [{"runtime_sha256": "a" * 64}, {"actor_id": "bell"},
                                    {"launch_id": "a" * 32}, {"epoch": True}, {"permit_id": "wrong"}])
def test_signed_ready_wrong_binding_never_success(env, change):
    c = env.coordinator()
    c.accept(env.assignment())
    c.launch("wake-attempt-1", lambda *_: None, ready_path=env.tmp_path / "ready.json")
    with pytest.raises(ValueError):
        c.record_ready("wake-attempt-1", env.ready(c, change=change))
    assert c.record("wake-attempt-1")["state"] == "STARTING"


def test_bell_key_cannot_sign_maid_readiness(env):
    c = env.coordinator()
    c.accept(env.assignment())
    c.launch("wake-attempt-1", lambda *_: None, ready_path=env.tmp_path / "ready.json")
    with pytest.raises(ValueError, match="signature"):
        c.record_ready("wake-attempt-1", env.ready(c, signer=env.clients["bell"].signer))


def test_explicit_not_started_tombstone_prevents_launch(env):
    c = env.coordinator()
    assignment = env.assignment()
    permit = c.accept(assignment)
    result = c.not_started("wake-attempt-1")
    assert result["evidence"]["artifacts"] == []
    assert c.completion_payload("wake-attempt-1")["records"].keys() == {"tombstone"}
    c.report("wake-attempt-1")
    env.gate.ack_evidence(assignment["control_authority"]["permit"], result["evidence"], result["root"])
    assert c.resolved("wake-attempt-1")
    with pytest.raises(ExecutionUncertain):
        c.launch(permit["attempt_id"], lambda *_: pytest.fail("launch after tombstone"), ready_path=env.tmp_path / "ready")


def test_wrong_runtime_actor_device_and_extra_command_are_rejected(env):
    c, assignment = env.coordinator(), env.assignment()
    for invalid in ({**assignment, "command": "arbitrary"},
                    {**assignment, "runtime": {**env.runtime, "command": "arbitrary"}}):
        with pytest.raises(ValueError):
            c.accept(invalid)
    bad = json.loads(json.dumps(assignment))
    bad["control_authority"]["descriptor"]["worker_id"] = "maid"
    with pytest.raises(ValueError):
        c.accept(bad)
    with pytest.raises(ValueError, match="same-device|one-to-one"):
        env.coordinator(target_registration={**env.registry["maid"], "device_id": "other"})


def test_runtime_and_registry_pins_survive_restart(env):
    env.coordinator()
    with pytest.raises(ValueError, match="cannot change"):
        env.coordinator(runtime={**env.runtime, "changed": True})


def test_ready_artifact_tamper_invalidates_outbox(env):
    c = env.coordinator()
    c.accept(env.assignment())
    c.launch("wake-attempt-1", lambda *_: None, ready_path=env.tmp_path / "ready")
    c.record_ready("wake-attempt-1", env.ready(c))
    path = c.launch_paths("wake-attempt-1")["ready"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        c.report("wake-attempt-1")


def test_fixed_launcher_has_no_shell_and_exact_synthetic_entrypoint(tmp_path):
    python = tmp_path / "python.exe"
    python.write_bytes(b"test placeholder never executed")
    script = tmp_path / "scripts" / "simulation" / "shore_synthetic_maid.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"# not executed")
    runtime, startup, ready = (tmp_path / name for name in ("runtime.json", "startup.json", "ready.json"))
    runtime.write_bytes(b"{}")
    startup.write_bytes(b"{}")
    calls = []
    launcher = FixedMaidLauncher(python, tmp_path, popen_factory=lambda *a, **kw: calls.append((a, kw)))
    launcher(runtime, startup, ready)
    args, kwargs = calls[0]
    assert args[0] == [str(python), str(script), "--runtime-config", str(runtime),
                        "--wake-startup", str(startup), "--ready-record", str(ready), "--run"]
    assert kwargs["shell"] is False
    assert kwargs["stdin"] == -3
    if sys.platform == "win32":
        assert kwargs["creationflags"] & 0x08000000  # CREATE_NO_WINDOW


def test_runner_retries_outbox_after_collector_disconnect_without_second_launch(env):
    c, assignment, launches = env.coordinator(), env.assignment(), []

    class Collector:
        lose = True

        def request_json(self, path, body):
            if path == "/v1/wake/poll":
                return {"assignment": assignment}
            assert path == "/v1/wake/complete"
            if self.lose:
                self.lose = False
                raise TimeoutError("synthetic collector disconnect")
            result = c.checked_result(body["attempt_id"])
            return env.gate.ack_evidence(assignment["control_authority"]["permit"], result["evidence"], result["root"])

    runner = BellRunner(c, Collector(), lambda *paths: launches.append(paths), ready_root=env.tmp_path / "maid-state")
    assert runner.step()["state"] == "STARTING"
    env.ready(c)
    with pytest.raises(TimeoutError):
        runner.step()
    assert runner.step()["state"] == "RESOLVED"
    assert len(launches) == 1
    assert hashlib.sha256(c.launch_paths("wake-attempt-1")["runtime"].read_bytes()).hexdigest() == digest(env.runtime)


def _settled_wake(env):
    coordinator = env.coordinator()
    assignment = env.assignment()
    coordinator.accept(assignment)
    result = coordinator.not_started("wake-attempt-1")
    coordinator.report("wake-attempt-1")
    env.gate.ack_evidence(assignment["control_authority"]["permit"], result["evidence"], result["root"])
    assert coordinator.resolved("wake-attempt-1")
    return coordinator


def test_idle_requires_nonempty_fully_resolved_wake_and_no_pending_rpc(env):
    coordinator = env.coordinator()
    runner = BellRunner(coordinator, None, None, ready_root=env.tmp_path / "ready")
    with pytest.raises(ValueError, match="at least one"):
        runner.report_idle()
    coordinator.accept(env.assignment())
    with pytest.raises(ValueError, match="every durable row"):
        runner.report_idle()
    coordinator = _settled_wake(env)
    runner = BellRunner(coordinator, None, None, ready_root=env.tmp_path / "ready")
    env.transport.drop_method = "status"
    with pytest.raises(TimeoutError):
        coordinator.rpc.call(env.clients["bell"], "status", {}, key="idle-pending-status")
    with pytest.raises(ValueError, match="pending signed"):
        runner.report_idle()


def test_idle_signed_exact_binding_and_dropped_reply_can_retry(env):
    coordinator, requests = _settled_wake(env), []

    class Collector:
        lose = True

        def request_json(self, path, body):
            requests.append((path, body))
            if self.lose:
                self.lose = False
                raise TimeoutError("synthetic idle acknowledgement loss")
            return {"status": "RECORDED", "actor_id": "bell"}

    runner = BellRunner(coordinator, Collector(), None, ready_root=env.tmp_path / "ready")
    before = list(env.transport.seen)
    with pytest.raises(TimeoutError):
        runner.report_idle()
    assert runner.report_idle() == {"status": "RECORDED", "actor_id": "bell"}
    assert requests[0] == requests[1]
    assert requests[0][0] == "/v1/control/idle"
    value = verify_object(env.registry["bell"]["public_key"], IDLE_DOMAIN, requests[0][1]["record"])
    assert value == {"schema": IDLE_DOMAIN, "schema_version": 1, "mode": "shore-remote-synthetic-v1",
        "actor_id": "bell", "fleet_id": "test-fleet", "job_id": "job", "epoch": coordinator.binding["epoch"],
        "run_id": "run", "runtime_sha256": digest(env.runtime),
        "permit_ids": [coordinator.record("wake-attempt-1")["permit_id"]], "outboxes_pending": 0}
    assert coordinator.rpc.pending() == []
    assert env.transport.seen == before  # Data-plane idle is not an authority RPC.


def test_empty_poll_only_reports_idle_after_a_resolved_wake(env):
    calls = []

    class Collector:
        def request_json(self, path, body):
            calls.append(path)
            if path == "/v1/wake/poll":
                return {"assignment": None}
            return {"status": "RECORDED", "actor_id": "bell"}

    runner = BellRunner(env.coordinator(), Collector(), None, ready_root=env.tmp_path / "ready")
    assert runner.step() == {"state": "IDLE"}
    assert calls == ["/v1/wake/poll"]
    runner = BellRunner(_settled_wake(env), Collector(), None, ready_root=env.tmp_path / "ready")
    assert runner.step() == {"state": "IDLE_REPORTED"}
    assert calls == ["/v1/wake/poll", "/v1/wake/poll", "/v1/control/idle"]


@pytest.mark.parametrize("reply", [{"status": "RECORDED", "actor_id": "maid"},
                                   {"status": "RESOLVED", "actor_id": "bell"},
                                   {"status": "RECORDED", "actor_id": "bell", "extra": True}])
def test_idle_rejects_wrong_actor_or_unexpected_reply(env, reply):
    runner = BellRunner(_settled_wake(env), SimpleNamespace(request_json=lambda *_: reply), None,
                        ready_root=env.tmp_path / "ready")
    with pytest.raises(ValueError, match="exact idle proof"):
        runner.report_idle()
