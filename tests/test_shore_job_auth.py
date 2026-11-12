"""Offline job-auth boundary tests: fresh in-memory keys and temporary ledgers."""
from __future__ import annotations

import copy
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("shore_control")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shore_control.client import MemorySigner
from shore_control.crypto import fingerprint, public_key_text, sign_object, verify_object

from msabp_opt.simulation.distributed import control_job_auth as auth
from msabp_opt.simulation.distributed import control_remote_http as run_auth


@pytest.fixture(autouse=True)
def no_network_or_processes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Job-auth tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def lab(tmp_path):
    keys = {name: Ed25519PrivateKey.generate() for name in ("authority", "controller", "bell", "maid")}
    network = {"tailnet_name": "test@example.test", "magic_dns_suffix": "example.ts.net",
        "local_node_id": "nController", "local_address": "100.64.0.1", "nodes": {
            "nController": {"address": "100.64.0.1", "dns_name": "collector.example.ts.net."},
            "nWorker": {"address": "100.64.0.2", "dns_name": "worker.example.ts.net."}}}
    workers = {actor: {"public_key": public_key_text(keys[actor].public_key()),
        "device_id": "test-device", "tailnet_node_id": "nWorker", "role": actor,
        "allowed_actions": ["wake" if actor == "bell" else "attempt"]} for actor in ("bell", "maid")}
    bootstrap = {"schema": "shore.remote-bootstrap", "schema_version": 1, "mode": auth.MODE,
        "fleet_id": "shore-princess-" + "1" * 24,
        "authority_id": fingerprint(keys["authority"].public_key()),
        "authority_public_key": public_key_text(keys["authority"].public_key()),
        "endpoint": "https://100.64.0.1:18766", "tls_sha256": "a" * 64, "network": network,
        "controllers": {"princess-281ea981-06cf-4cfb-ac14-4c86a736b471":
                        public_key_text(keys["controller"].public_key())},
        "workers": workers, "controller_nodes": ["nController", "nWorker"],
        "collector": {"endpoint": "https://100.64.0.1:18767", "tls_sha256": "b" * 64}}
    expected_job = {"fleet_id": bootstrap["fleet_id"], "job_id": "job-1", "epoch": 1}
    context = {**expected_job, "actor_id": "bell", "device_id": "test-device", "node_id": "nWorker"}
    nonce_db = tmp_path / "job-auth.sqlite"
    auth.initialize_job_nonce_db(nonce_db)
    clock = {"now": 1_000_000}
    return SimpleNamespace(keys=keys, bootstrap=bootstrap, expected_job=expected_job,
        context=context, nonce_db=nonce_db, clock=clock, signer=MemorySigner(keys["bell"]))


def signed(lab, *, body=b"{}", path=auth.OFFER_PATH, **kwargs):
    return auth.sign_job_request(lab.signer, lab.context, "POST", path, body,
                                 now_ms=lab.clock["now"], **kwargs)


def verified(lab, envelope, *, body=b"{}", path=auth.OFFER_PATH, **kwargs):
    options = {"method": "POST", "client_node_id": "nWorker", "bootstrap": lab.bootstrap,
        "expected_job": lab.expected_job, "nonce_db": lab.nonce_db,
        "clock_ms": lambda: lab.clock["now"]}
    options.update(kwargs)
    return auth.verify_job_request(envelope, body, path=path, **options)


def rows(lab):
    with sqlite3.connect(lab.nonce_db) as db:
        return db.execute("SELECT actor_id,request_id,issued_at_ms FROM job_http_nonces").fetchall()


@pytest.mark.parametrize("path", sorted(auth.JOB_PATHS))
def test_exact_contract_and_persistent_replay_rejection(lab, path):
    envelope = signed(lab, path=path)
    payload = verify_object(lab.bootstrap["workers"]["bell"]["public_key"], auth.AUTH_DOMAIN, envelope)
    assert set(payload) == auth.AUTH_FIELDS
    assert auth.AUTH_DOMAIN != run_auth.AUTH_DOMAIN
    assert auth.AUTH_HEADER != run_auth.AUTH_HEADER
    assert "run_id" not in payload
    assert verified(lab, envelope, path=path) == "bell"
    assert rows(lab) == [("bell", payload["request_id"], payload["issued_at_ms"])]
    # Every verification opens/closes the database. No in-memory replay cache.
    with pytest.raises(auth.JobAuthError) as error:
        verified(lab, copy.deepcopy(envelope), path=path, nonce_db=Path(str(lab.nonce_db)))
    assert error.value.status == 409
    assert len(rows(lab)) == 1


def test_default_clock_and_random_request_ids(lab, monkeypatch):
    monkeypatch.setattr(auth.time, "time_ns", lambda: lab.clock["now"] * 1_000_000)
    one = auth.sign_job_request(lab.signer, lab.context, "POST", auth.OFFER_PATH, b"{}")
    two = auth.sign_job_request(lab.signer, lab.context, "POST", auth.OFFER_PATH, b"{}")
    assert one["payload"]["issued_at_ms"] == lab.clock["now"]
    assert one["payload"]["request_id"] != two["payload"]["request_id"]
    assert verified(lab, one, clock_ms=None) == "bell"


@pytest.mark.parametrize("field,value", [
    ("fleet_id", "shore-princess-" + "2" * 24), ("job_id", "other-job"), ("epoch", 2),
    ("actor_id", "maid"), ("actor_id", "unknown"), ("device_id", "other-device"),
    ("node_id", "nController"), ("path", auth.MATERIAL_PATH), ("method", "GET"),
    ("body_sha256", "0" * 64), ("body_bytes", 3), ("issued_at_ms", 0),
    ("version", 2), ("mode", "other"), ("content_type", "text/plain"),
    ("epoch", True), ("version", True), ("issued_at_ms", True), ("body_bytes", True),
    ("request_id", "A" * 32), ("request_id", "0" * 31), ("body_sha256", "F" * 64),
    ("job_id", "a" * 129), ("actor_id", ["bell"]), ("epoch", 1.0),
    ("issued_at_ms", 1 << 63),
])
def test_even_valid_bell_signatures_cannot_change_bindings_or_types(lab, field, value):
    payload = signed(lab)["payload"]
    payload[field] = value
    # Shore canonical wire intentionally disallows floats.
    if isinstance(value, float):
        envelope = {"payload": payload, "signature": "A" * 86 + "=="}
    else:
        envelope = sign_object(lab.keys["bell"], auth.AUTH_DOMAIN, payload)
    with pytest.raises(auth.JobAuthError):
        verified(lab, envelope)
    assert rows(lab) == []


@pytest.mark.parametrize("source", [None, "nController", "forged-header", "", 1, ["nWorker"]])
def test_actual_socket_source_is_required(lab, source):
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), client_node_id=source)
    assert rows(lab) == []


def test_registered_same_device_maid_key_does_not_authorize_bell_endpoint(lab):
    context = {**lab.context, "actor_id": "maid"}
    envelope = auth.sign_job_request(MemorySigner(lab.keys["maid"]), context,
        "POST", auth.OFFER_PATH, b"{}", now_ms=lab.clock["now"])
    with pytest.raises(auth.JobAuthError, match="enrolled Bell"):
        verified(lab, envelope)
    # Claiming Bell's actor with Maid's key is also invalid.
    envelope = sign_object(lab.keys["maid"], auth.AUTH_DOMAIN, signed(lab)["payload"])
    with pytest.raises(auth.JobAuthError):
        verified(lab, envelope)
    assert rows(lab) == []


@pytest.mark.parametrize("domain", [run_auth.AUTH_DOMAIN, "permit", "msabp-job-http-request-v2"])
def test_other_signature_domains_are_never_accepted(lab, domain):
    envelope = sign_object(lab.keys["bell"], domain, signed(lab)["payload"])
    with pytest.raises(auth.JobAuthError):
        verified(lab, envelope)
    assert rows(lab) == []


def test_unsigned_body_and_envelope_mutations_are_rejected_without_consumption(lab):
    envelope = signed(lab)
    for body in (b"[]", b'{"extra":true}', b"{", b""):
        with pytest.raises(auth.JobAuthError):
            verified(lab, envelope, body=body)
    changed = copy.deepcopy(envelope)
    changed["payload"]["request_id"] = "a" * 32
    with pytest.raises(auth.JobAuthError):
        verified(lab, changed)
    assert rows(lab) == []
    assert verified(lab, envelope) == "bell"


@pytest.mark.parametrize("method,path", [
    ("GET", auth.OFFER_PATH), ("post", auth.OFFER_PATH), ("POST", "/v1/wake/poll"),
    ("POST", "/v1/message"), ("POST", auth.OFFER_PATH + "?job_id=job-1"),
    ("POST", "/v1/job/%6ffer"), ("POST", auth.OFFER_PATH + "/"), (True, auth.OFFER_PATH),
])
def test_only_exact_post_endpoints_are_allowed(lab, method, path):
    with pytest.raises(auth.JobAuthError):
        auth.sign_job_request(lab.signer, lab.context, method, path, b"{}")
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), method=method, path=path)
    assert rows(lab) == []


def test_request_body_exact_type_and_size_limits(lab):
    largest = b" " * auth.MAX_REQUEST_BYTES
    assert verified(lab, signed(lab, body=largest), body=largest) == "bell"
    for body in (largest + b" ", bytearray(b"{}"), memoryview(b"{}"), "{}", None):
        with pytest.raises(auth.JobAuthError):
            signed(lab, body=body)
        with pytest.raises(auth.JobAuthError):
            verified(lab, signed(lab), body=body)
    assert len(rows(lab)) == 1


@pytest.mark.parametrize("location,operation", [
    ("payload", "add"), ("payload", "remove"), ("envelope", "add"), ("envelope", "remove"),
])
def test_exact_payload_and_envelope_fields(lab, location, operation):
    envelope = signed(lab)
    target = envelope["payload"] if location == "payload" else envelope
    if operation == "add":
        target["run_id"] = "fake-discovery-run"
    else:
        del target[next(iter(target))]
    with pytest.raises(auth.JobAuthError):
        verified(lab, envelope)
    assert rows(lab) == []


@pytest.mark.parametrize("value", [None, [], "", "A" * 10000, "!" * 86 + "=="])
def test_signature_and_envelope_limits(lab, value):
    envelope = signed(lab)
    envelope["signature"] = value
    with pytest.raises(auth.JobAuthError):
        verified(lab, envelope)
    assert rows(lab) == []


def test_total_envelope_limit_is_enforced(lab, monkeypatch):
    envelope = signed(lab)
    monkeypatch.setattr(auth, "MAX_ENVELOPE_BYTES", 16)
    with pytest.raises(auth.JobAuthError) as error:
        verified(lab, envelope)
    assert error.value.status == 413
    assert rows(lab) == []


@pytest.mark.parametrize("field,value", [("epoch", True), ("epoch", 0), ("epoch", 2),
    ("job_id", "another"), ("fleet_id", "shore-princess-" + "2" * 24)])
def test_independent_expected_job_must_match_exactly(lab, field, value):
    expected = {**lab.expected_job, field: value}
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), expected_job=expected)
    assert rows(lab) == []


def test_context_and_expected_job_reject_unknown_fields(lab):
    with pytest.raises(auth.JobAuthError):
        auth.sign_job_request(lab.signer, {**lab.context, "run_id": "pretend"},
                              "POST", auth.OFFER_PATH, b"{}")
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), expected_job={**lab.expected_job, "run_id": "pretend"})


@pytest.mark.parametrize("field,value", [("device_id", "other-device"),
    ("tailnet_node_id", "nController"), ("role", "maid"), ("allowed_actions", ["wake", "attempt"])])
def test_local_registry_binding_cannot_be_bypassed(lab, field, value):
    bootstrap = copy.deepcopy(lab.bootstrap)
    bootstrap["workers"]["bell"][field] = value
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), bootstrap=bootstrap)
    assert rows(lab) == []


@pytest.mark.parametrize("offset,accepted", [(-auth.MAX_AGE_MS, True),
    (-auth.MAX_AGE_MS - 1, False), (auth.MAX_FUTURE_MS, True), (auth.MAX_FUTURE_MS + 1, False)])
def test_millisecond_time_window_boundaries(lab, offset, accepted):
    envelope = auth.sign_job_request(lab.signer, lab.context, "POST", auth.OFFER_PATH,
                                     b"{}", now_ms=lab.clock["now"] + offset)
    if accepted:
        assert verified(lab, envelope) == "bell"
    else:
        with pytest.raises(auth.JobAuthError, match="time window"):
            verified(lab, envelope)
        assert rows(lab) == []


@pytest.mark.parametrize("expire_existing", [False, True])
def test_waiting_for_nonce_write_lock_rechecks_time_before_pruning(lab, monkeypatch, expire_existing):
    envelope = signed(lab)
    if expire_existing:
        assert verified(lab, envelope) == "bell"
    lab.clock["now"] += auth.MAX_AGE_MS - 1
    entered, finished = threading.Event(), threading.Event()
    original_connect = sqlite3.connect
    blocker = original_connect(lab.nonce_db)
    blocker.execute("BEGIN IMMEDIATE")

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE":
                entered.set()
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(auth.sqlite3, "connect",
        lambda *args, **kwargs: original_connect(*args, **kwargs, factory=ObservedConnection))
    outcome = {}

    def request():
        try:
            outcome["actor"] = verified(lab, envelope)
        except BaseException as error:
            outcome["error"] = error
        finally:
            finished.set()

    thread = threading.Thread(target=request, daemon=True)
    thread.start()
    try:
        assert entered.wait(2), "Request never attempted the write lock"
        assert not finished.wait(0.05), "Request must wait for the held lock"
        lab.clock["now"] += 2
    finally:
        blocker.rollback()
        blocker.close()
        thread.join(5)
    assert not thread.is_alive()
    assert "actor" not in outcome
    assert isinstance(outcome.get("error"), auth.JobAuthError)
    assert "time window" in str(outcome["error"])
    assert len(rows(lab)) == int(expire_existing)


def test_clock_moving_back_before_consumption_is_rejected(lab):
    envelope = signed(lab)
    samples = iter((lab.clock["now"], lab.clock["now"] - auth.MAX_FUTURE_MS - 1))
    with pytest.raises(auth.JobAuthError, match="time window"):
        verified(lab, envelope, clock_ms=lambda: next(samples))
    assert rows(lab) == []


def test_concurrent_duplicate_consumed_once(lab):
    envelope = signed(lab)
    barrier = threading.Barrier(2)
    results = []

    def request():
        barrier.wait(timeout=2)
        try:
            results.append(verified(lab, envelope))
        except auth.JobAuthError as error:
            results.append(error.status)

    threads = [threading.Thread(target=request, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert sorted(map(str, results)) == ["409", "bell"]
    assert len(rows(lab)) == 1


def test_nonce_capacity_and_expired_entry_pruning(lab, monkeypatch):
    monkeypatch.setattr(auth, "MAX_NONCES", 1)
    first = signed(lab)
    assert verified(lab, first) == "bell"
    with pytest.raises(auth.JobAuthError) as error:
        verified(lab, signed(lab))
    assert error.value.status == 429
    with pytest.raises(auth.JobAuthError) as replay:
        verified(lab, first)
    assert replay.value.status == 409
    lab.clock["now"] += auth.MAX_AGE_MS + 1
    second = signed(lab)
    assert verified(lab, second) == "bell"
    assert rows(lab)[0][1] == second["payload"]["request_id"]


def test_missing_or_unknown_nonce_ledger_never_reinitializes(lab, tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(auth.JobAuthError) as error:
        verified(lab, signed(lab), nonce_db=missing)
    assert error.value.status == 503
    assert not missing.exists()
    empty = tmp_path / "empty.sqlite"
    with sqlite3.connect(empty):
        pass
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), nonce_db=empty)
    with sqlite3.connect(empty) as db:
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == []
    with sqlite3.connect(lab.nonce_db) as db:
        db.execute("UPDATE job_http_auth_meta SET schema_version=2")
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab))
    assert rows(lab) == []


def test_nonce_initialization_never_overwrites_existing_evidence(lab):
    before = lab.nonce_db.read_bytes()
    with pytest.raises(FileExistsError):
        auth.initialize_job_nonce_db(lab.nonce_db)
    assert lab.nonce_db.read_bytes() == before
    with pytest.raises(auth.JobAuthError):
        auth.initialize_job_nonce_db(":memory:")


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("part", ["parent", "database"])
@pytest.mark.parametrize("operation", ["initialize", "verify"])
def test_nonce_rejects_redirected_database_or_ancestor_before_sqlite(
        lab, monkeypatch, kind, part, operation):
    # Simulate the OS attributes deterministically without requiring symlink
    # privilege or actually redirecting a protected database path.
    envelope = signed(lab)
    before = lab.nonce_db.read_bytes()
    target = lab.nonce_db.parent if part == "parent" else lab.nonce_db
    original_lstat = Path.lstat

    def redirected(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else info.st_mode,
                                   st_file_attributes=0x400 if kind == "junction" else 0)
        return info

    def no_database_open(*_args, **_kwargs):
        pytest.fail("Redirected nonce paths must be rejected before opening SQLite")

    monkeypatch.setattr(Path, "lstat", redirected)
    monkeypatch.setattr(auth.sqlite3, "connect", no_database_open)
    with pytest.raises(auth.JobAuthError) as error:
        if operation == "initialize":
            auth.initialize_job_nonce_db(lab.nonce_db)
        else:
            verified(lab, envelope)
    assert error.value.status == 503
    assert lab.nonce_db.read_bytes() == before


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_nonce_initialization_never_creates_below_redirected_parent(tmp_path, monkeypatch, kind):
    target = tmp_path / "new.sqlite"
    original_lstat = Path.lstat

    def redirected(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == tmp_path:
            return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else info.st_mode,
                                   st_file_attributes=0x400 if kind == "junction" else 0)
        return info

    monkeypatch.setattr(Path, "lstat", redirected)
    with pytest.raises(auth.JobAuthError):
        auth.initialize_job_nonce_db(target)
    assert not target.exists()


def test_hardlinked_nonce_database_and_alias_are_both_rejected(lab, tmp_path):
    alias = tmp_path / "nonce-alias.sqlite"
    alias.hardlink_to(lab.nonce_db)
    before = lab.nonce_db.read_bytes()
    for path in (lab.nonce_db, alias):
        with pytest.raises(auth.JobAuthError, match="non-aliased"):
            verified(lab, signed(lab), nonce_db=path)
        with pytest.raises(auth.JobAuthError, match="non-aliased"):
            auth.initialize_job_nonce_db(path)
    assert lab.nonce_db.read_bytes() == alias.read_bytes() == before


def test_nonce_database_must_be_a_regular_file(lab, tmp_path):
    directory = tmp_path / "not-a-database"
    directory.mkdir()
    with pytest.raises(auth.JobAuthError, match="regular"):
        verified(lab, signed(lab), nonce_db=directory)
    with pytest.raises(auth.JobAuthError, match="regular"):
        auth.initialize_job_nonce_db(directory)
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("kind", ["relative", "traversal", "unc"])
def test_nonce_paths_must_be_absolute_local_and_nontraversing(lab, kind):
    path = {"relative": "nonce.sqlite",
        "traversal": lab.nonce_db.parent / "unused" / ".." / lab.nonce_db.name,
        "unc": r"\\untrusted-host\share\nonce.sqlite"}[kind]
    with pytest.raises(auth.JobAuthError):
        verified(lab, signed(lab), nonce_db=path)
    with pytest.raises(auth.JobAuthError):
        auth.initialize_job_nonce_db(path)
    assert rows(lab) == []
