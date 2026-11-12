"""Public USB discovery and continuity guards; never read real USB secrets."""
from __future__ import annotations

import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.crypto import fingerprint, public_key_text
from shore_control import usb_identity as shore_usb

from msabp_opt.simulation.distributed import control_usb as usb


ACTOR = "princess-test"
GUID = "\\\\?\\Volume{11111111-1111-1111-1111-111111111111}\\"
OTHER_GUID = GUID.replace("1", "2")
F = r"F:\PrincessControl\identity.json"
G = r"G:\PrincessControl\identity.json"


@pytest.fixture
def media(monkeypatch):
    key = Ed25519PrivateKey.generate().public_key()
    pin = fingerprint(key)
    manifest = {"Role": "Princess", "IdentityId": "test", "PublicKey": public_key_text(key), "Fingerprint": pin}
    state = SimpleNamespace(volumes={F: GUID}, files={F: json.dumps(manifest).encode()},
                            pin=pin, manifest=manifest, signs=0, after_sign=None, reads=[])

    def forbidden(*_args, **_kwargs):
        pytest.fail("No real sockets, drive probes or key reads in USB unit tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(usb, "removable_roots", lambda: [str(Path(p).anchor) for p in state.volumes])

    def volume(path):
        if str(path) not in state.volumes:
            raise OSError("Mock absent or nonremovable drive")
        return state.volumes[str(path)]

    monkeypatch.setattr(usb, "volume_guid", volume)
    monkeypatch.setattr(shore_usb, "_assert_removable", volume)
    monkeypatch.setattr(shore_usb, "_assert_regular_path", lambda *_: None)

    def read(path, _limit):
        assert str(path).endswith("identity.json"), "Discovery must never read a private key"
        state.reads.append(str(path))
        if str(path) not in state.files:
            raise FileNotFoundError("mock missing manifest")
        return state.files[str(path)]

    monkeypatch.setattr(shore_usb, "_read_bytes", read)

    def sign(_self, domain, payload):
        state.signs += 1
        if state.after_sign:
            state.after_sign()
        return {"domain": domain, "payload": payload, "signature": "synthetic-only"}

    monkeypatch.setattr(shore_usb.UsbIdentity, "sign", sign)
    return state


def signer(media, **kwargs):
    return usb.LockedUsbIdentity(actor=ACTOR, expected_fingerprint=media.pin, **kwargs)


def test_auto_f_without_e_public_only(media):
    locked = signer(media)
    assert locked.inspect()["manifest_path"] == F
    assert media.signs == 0
    assert media.reads and set(media.reads) == {F}


def test_fresh_job_finds_changed_letter(media):
    media.volumes = {G: GUID}
    media.files = {G: media.files[F]}
    assert signer(media).inspect()["manifest_path"] == G


@pytest.mark.parametrize("count", [0, 2])
def test_zero_or_ambiguous_match_rejected(media, count):
    if count == 0:
        media.volumes.clear()
    else:
        media.volumes[G], media.files[G] = OTHER_GUID, media.files[F]
    with pytest.raises(ValueError, match=f"found {count}"):
        signer(media).sign("test", {})
    assert media.signs == 0


@pytest.mark.parametrize("change", [{"Role": "Shore"}, {"IdentityId": "wrong"},
                                   {"Fingerprint": "wrong"}, {"PublicKey": "wrong"}])
def test_public_identity_cannot_self_authorize(media, change):
    media.files[F] = json.dumps({**media.manifest, **change}).encode()
    with pytest.raises(ValueError, match="found 0"):
        signer(media).inspect()


def test_duplicate_json_is_rejected(media):
    media.files[F] = media.files[F][:-1] + b', "Role":"Princess"}'
    with pytest.raises(ValueError, match="found 0"):
        signer(media).inspect()


def test_reparse_check_failure_is_not_ignored_for_explicit_path(media, monkeypatch):
    def redirected(*_):
        raise ValueError("Reparse point")
    monkeypatch.setattr(shore_usb, "_assert_regular_path", redirected)
    with pytest.raises(ValueError, match="Reparse"):
        signer(media, selection=F).inspect()


def test_fixed_or_absent_explicit_drive_never_falls_back(media):
    with pytest.raises(OSError):
        signer(media, selection=G).sign("test", {})
    assert media.signs == 0


def test_volume_swap_rejected_without_rediscovery(media):
    locked = signer(media)
    locked.inspect()
    media.volumes[F] = OTHER_GUID
    media.volumes[G], media.files[G] = GUID, media.files[F]
    with pytest.raises(ValueError, match="volume changed"):
        locked.sign("test", {})
    assert media.signs == 0


def test_unplug_during_signature_never_returns_envelope(media):
    locked = signer(media)
    media.after_sign = media.volumes.clear
    with pytest.raises(OSError):
        locked.sign("test", {})
    assert media.signs == 1


def test_constructor_is_lazy_even_when_absent(media, tmp_path):
    media.volumes.clear()
    locked = signer(media, binding_path=tmp_path / "usb.json", allow_new_binding=False)
    assert media.reads == []
    assert not (tmp_path / "usb.json").exists()
    with pytest.raises(ValueError, match="no Princess USB binding"):
        locked.sign("test", {})


def test_public_binding_survives_restart_and_does_not_relocate(media, tmp_path):
    path = tmp_path / "usb.json"
    locked = signer(media, binding_path=path, context={"job": "test"})
    assert locked.sign("test", {})["signature"] == "synthetic-only"
    data = path.read_bytes()
    media.volumes.clear()
    restored = signer(media, binding_path=path, context={"job": "test"}, allow_new_binding=False)
    assert path.read_bytes() == data
    with pytest.raises(OSError):
        restored.sign("test", {})
    media.volumes[G], media.files[G] = GUID, media.files[F]
    with pytest.raises(OSError):
        restored.sign("test", {})
    assert path.read_bytes() == data
    # Reinsert the original volume at its original letter: signing is possible
    # for release, but the separate gate must remain draining (tested below).
    media.volumes[F] = GUID
    assert restored.sign("test", {})["signature"] == "synthetic-only"


@pytest.mark.parametrize("mutation", ["delete", "corrupt", "context", "path", "volume"])
def test_old_binding_cannot_be_replaced(media, tmp_path, mutation):
    path = tmp_path / "usb.json"
    locked = signer(media, binding_path=path)
    locked.inspect()
    if mutation == "delete":
        path.unlink()
    elif mutation == "corrupt":
        path.write_bytes(b"broken")
    else:
        value = json.loads(path.read_bytes())
        if mutation == "context":
            value["context"] = {"job": "different"}
        elif mutation == "path":
            value["manifest_path"] = G
        else:
            value["volume_guid"] = OTHER_GUID
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        locked.sign("test", {})
    assert media.signs == 0


def test_existing_binding_rejects_changed_explicit_path(media, tmp_path):
    path = tmp_path / "usb.json"
    signer(media, binding_path=path).inspect()
    with pytest.raises(ValueError, match="Explicit USB path"):
        signer(media, selection=G, binding_path=path, allow_new_binding=False).sign("test", {})


def test_wrapper_keeps_original_private_signer_per_signature(media):
    locked = signer(media)
    locked.sign("test", {"n": 1})
    locked.sign("test", {"n": 2})
    assert media.signs == 2


def test_probe_checks_volume_before_and_after_public_read(media, monkeypatch):
    original = usb._public_identity
    def swap(*args):
        original(*args)
        media.volumes[F] = OTHER_GUID
    monkeypatch.setattr(usb, "_public_identity", swap)
    with pytest.raises(ValueError, match="found 0"):
        signer(media).inspect()
