"""USB pin integration with the existing controller and session-only drain."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("shore_control")
from shore_control.crypto import fingerprint, public_key_text
from shore_control import usb_identity as shore_usb

from msabp_opt.simulation.distributed import control_remote_controller as controller
from msabp_opt.simulation.distributed import control_remote_session as session
from msabp_opt.simulation.distributed.control_usb import LockedUsbIdentity

from test_shore_remote_controller import project  # noqa: F401
from test_shore_usb import F, GUID, media  # noqa: F401


@pytest.fixture
def setup(project, media, tmp_path, monkeypatch):  # noqa: F811
    media.pin = fingerprint(project.keys[controller.CONTROLLER_ACTOR].public_key())
    media.manifest.update(IdentityId=controller.CONTROLLER_ACTOR.removeprefix("princess-"),
                          Fingerprint=media.pin,
                          PublicKey=public_key_text(project.keys[controller.CONTROLLER_ACTOR].public_key()))
    media.files[F] = json.dumps(media.manifest).encode()
    root = tmp_path / "new-job"
    root.mkdir()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text(json.dumps(project.bootstrap), encoding="utf-8")
    preparation = {"output_root": str(root), "job_id": "new-job", "bootstrap_sha256": "a" * 64,
                   "job_spec": project.spec, "paths": {"bootstrap_path": str(bootstrap_path)}}

    def fake_session(path, **_kwargs):
        # Only generated in-process test keys; never DPAPI or real USB secrets.
        if not path.exists():
            path.write_text("synthetic session stub", encoding="utf-8")
        return "session", project.keys["session:session"]

    monkeypatch.setattr(session, "load_or_create_session", fake_session)
    return preparation, root


def test_new_absent_key_fails_before_creating_session_or_journal(setup, media):  # noqa: F811
    preparation, root = setup
    media.volumes.clear()
    with pytest.raises(ValueError, match="found 0"):
        controller.open_clients(preparation, usb_manifest="auto")
    assert list(root.iterdir()) == []


def test_first_challenge_failure_preserves_pin_for_retry(setup, media, monkeypatch):  # noqa: F811
    preparation, root = setup
    gate, *_ = controller.open_clients(preparation, usb_manifest="auto")
    original = (root / "controller-usb.json").read_bytes()
    def unavailable(*_args):
        raise TimeoutError("mock challenge unavailable")
    monkeypatch.setattr(gate.controller.transport, "challenge", unavailable)
    with pytest.raises(TimeoutError):
        gate.controller.call("status")
    restored, *_ = controller.open_clients(preparation, usb_manifest="auto")
    assert restored.controller.signer.inspect()["manifest_path"] == F
    assert (root / "controller-usb.json").read_bytes() == original


def test_existing_session_can_open_without_any_usb_reads(setup, media):  # noqa: F811
    preparation, root = setup
    controller.open_clients(preparation, usb_manifest="auto")
    media.volumes.clear()
    media.reads.clear()
    restored, *_ = controller.open_clients(preparation, usb_manifest="auto")
    assert media.reads == []
    assert (root / "controller-usb.json").is_file()
    with pytest.raises(OSError):
        restored.controller.signer.inspect()


def test_loss_drains_using_session_and_reinsert_does_not_resume(project, media, monkeypatch):  # noqa: F811
    original_signer = project.gate.controller.signer
    media.pin = fingerprint(project.keys[controller.CONTROLLER_ACTOR].public_key())
    media.manifest.update(IdentityId=controller.CONTROLLER_ACTOR.removeprefix("princess-"),
                          Fingerprint=media.pin,
                          PublicKey=public_key_text(project.keys[controller.CONTROLLER_ACTOR].public_key()))
    media.files[F] = json.dumps(media.manifest).encode()
    monkeypatch.setattr(shore_usb.UsbIdentity, "sign", lambda _self, domain, payload:
                        original_signer.sign(domain, payload))
    project.gate.controller.signer = LockedUsbIdentity(actor=controller.CONTROLLER_ACTOR,
                                                      expected_fingerprint=media.pin)
    assert project.gate.poll_control()["phase"] == "ACTIVE"
    media.volumes.clear()
    with pytest.raises(OSError):
        project.gate.poll_control()
    assert project.gate.job["phase"] == "DRAINING"
    assert project.gate.status()["bound_job"]["phase"] == "DRAINING"
    media.volumes[F] = GUID
    count = len(project.transport.requests)
    assert project.gate.poll_control()["phase"] == "DRAINING"
    assert len(project.transport.requests) == count
