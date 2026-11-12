"""Only temporary random restricted session keys; no USB/device identities."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from shore_control.crypto import canonical_json, public_key_text
from msabp_opt.simulation.distributed import control_remote_session as session


CONTEXT = {"mode": session.MODE, "bootstrap_sha256": "a" * 64, "job_id": "synthetic-job", "host_id": "test-host"}


@pytest.fixture
def fake_dpapi(monkeypatch):
    cipher = AESGCM(AESGCM.generate_key(bit_length=256))

    def protect(data, entropy):
        nonce = os.urandom(12)
        return nonce + cipher.encrypt(nonce, data, entropy)

    def unprotect(data, entropy):
        return cipher.decrypt(data[:12], data[12:], entropy)

    monkeypatch.setattr(session, "_protect", protect)
    monkeypatch.setattr(session, "_unprotect", unprotect)


def test_session_reload_retains_key_and_never_saves_plaintext(tmp_path, fake_dpapi):
    path = tmp_path / "restricted-session.json"
    session_id, key = session.load_or_create_session(path, context=CONTEXT)
    data = path.read_bytes()
    restored_id, restored = session.load_or_create_session(path, context=CONTEXT)
    assert restored_id == session_id
    assert public_key_text(restored.public_key()) == public_key_text(key.public_key())
    assert path.read_bytes() == data
    assert b"private_key_b64" not in data
    assert json.loads(data)["protection"] == "windows-dpapi-current-user"


@pytest.mark.parametrize("change", [{"job_id": "other"}, {"bootstrap_sha256": "b" * 64}, {"host_id": "other"}])
def test_context_mismatch_never_recreates(tmp_path, fake_dpapi, change):
    path = tmp_path / "restricted-session.json"
    session.load_or_create_session(path, context=CONTEXT)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="refusing replacement"):
        session.load_or_create_session(path, context={**CONTEXT, **change})
    assert path.read_bytes() == original


def test_modified_outer_binding_cannot_rebind_encrypted_secret(tmp_path, fake_dpapi):
    path = tmp_path / "restricted-session.json"
    session.load_or_create_session(path, context=CONTEXT)
    outer = json.loads(path.read_bytes())
    outer["context"]["job_id"] = "other"
    path.write_bytes(canonical_json(outer))
    with pytest.raises(Exception):
        session.load_or_create_session(path, context=outer["context"])


def test_failure_has_no_plaintext_or_replacement_fallback(tmp_path, monkeypatch, fake_dpapi):
    path = tmp_path / "restricted-session.json"

    def unavailable(*_):
        raise OSError("mock DPAPI unavailable")

    monkeypatch.setattr(session, "_protect", unavailable)
    with pytest.raises(OSError):
        session.load_or_create_session(path, context=CONTEXT)
    assert not path.exists()
    path.write_bytes(b"corrupted partial old ciphertext")
    with pytest.raises(ValueError):
        session.load_or_create_session(path, context=CONTEXT)
    assert path.read_bytes() == b"corrupted partial old ciphertext"


def test_decryption_failure_never_regenerates(tmp_path, fake_dpapi, monkeypatch):
    path = tmp_path / "restricted-session.json"
    session.load_or_create_session(path, context=CONTEXT)
    original = path.read_bytes()

    def denied(*_):
        raise OSError("mock different Windows user")

    monkeypatch.setattr(session, "_unprotect", denied)
    with pytest.raises(OSError):
        session.load_or_create_session(path, context=CONTEXT)
    assert path.read_bytes() == original


def test_plaintext_and_machine_scope_files_are_rejected(tmp_path, fake_dpapi):
    path = tmp_path / "restricted-session.json"
    session.load_or_create_session(path, context=CONTEXT)
    outer = json.loads(path.read_bytes())
    outer["protection"] = "windows-dpapi-local-machine"
    outer["protected_blob_b64"] = base64.b64encode(b"plaintext fallback forbidden").decode()
    path.write_bytes(canonical_json(outer))
    with pytest.raises(ValueError, match="refusing replacement"):
        session.load_or_create_session(path, context=CONTEXT)


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_actual_current_user_dpapi_round_trip_for_temporary_random_session(tmp_path):
    path = tmp_path / "temporary-session.json"
    identifier, key = session.load_or_create_session(path, context=CONTEXT)
    restored_identifier, restored = session.load_or_create_session(path, context=CONTEXT)
    assert identifier == restored_identifier
    assert public_key_text(key.public_key()) == public_key_text(restored.public_key())
    assert b"private_key_b64" not in path.read_bytes()
