"""Standalone pre-import boundary; only temporary files and generated public data."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.simulation import supervisor_host as host


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_key(byte):
    wire = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes([byte]) * 32
    return "ssh-ed25519 " + base64.b64encode(wire).decode("ascii")


@pytest.fixture
def policy():
    authority, worker = "n8BsWasrMb11CNTRL", "nJc1aWG7ZA11CNTRL"
    bootstrap = {"schema": "shore.remote-bootstrap", "schema_version": 1, "mode": host.MODE,
        "fleet_id": "shore-princess-" + "1234" * 6,
        "authority_id": host._public_key(public_key(1)), "authority_public_key": public_key(1),
        "endpoint": "https://100.64.0.1:18766", "tls_sha256": "a" * 64,
        "network": {"tailnet_name": "offline-test", "magic_dns_suffix": "example.ts.net",
            "local_node_id": authority, "local_address": "100.64.0.1", "nodes": {
                authority: {"address": "100.64.0.1", "dns_name": "host.example.ts.net."},
                worker: {"address": "100.64.0.2", "dns_name": "worker.example.ts.net."}}},
        "controllers": {"princess-11111111-1111-4111-8111-111111111111": public_key(2)},
        "controller_nodes": [authority, worker],
        "workers": {actor: {"public_key": public_key(index), "device_id": "coconutg2",
            "tailnet_node_id": worker, "role": role, "allowed_actions": [action]}
            for index, actor, role, action in ((3, "coconutg2-bell", "bell", "wake"),
                                              (4, "coconutg2-maid", "maid", "attempt"))},
        "collector": {"endpoint": "https://100.64.0.1:18767", "tls_sha256": "b" * 64}}
    manifest = {"schema": "msabp.supervisor-release", "schema_version": 1, "mode": host.MODE,
        "project_id": "MSABP_2", "source_hashes": {name: sha(name.encode()) for name in host.REQUIRED_PROJECT_FILES},
        "shore_source_hashes": {name: sha(name.encode()) for name in host.SHORE_FILES}}
    profile = {"schema": "msabp.supervisor-release-profile", "schema_version": 1,
        "mode": host.MODE, "project_id": "MSABP_2", "release_id": sha(host.canonical_bytes(manifest)),
        "release_manifest": manifest, "bootstrap_sha256": sha(host.canonical_bytes(bootstrap))}
    base = host.BASE_PATH
    venv = host.VENV_PATH
    interpreter = {"executable": str(host.PYTHON_PATH), "base_executable": str(base / "python.exe"),
        "prefix": str(venv), "base_prefix": str(base), "roots": [str(venv), str(base)],
        "files": {str(path): sha(str(path).encode()) for path in (
            host.PYTHON_PATH, venv / "pyvenv.cfg", base / "python.exe", base / "Lib" / "os.py")},
        "directories": [str(venv / "Scripts"), str(venv / "Include"), str(base / "Lib")]}
    return host.build_policy(profile=profile, bootstrap=bootstrap, host_sha256="c" * 64,
                             interpreter=interpreter, installation_id="d" * 32)


def test_builder_is_exact_public_offline_maintenance_policy(policy, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("No child process expected"))
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("No file/identity reads expected"))
    assert host.validate_policy(policy) == policy
    assert set(policy) == host.POLICY_FIELDS
    assert policy["discovery_enabled"] is False
    assert policy["stop_when_idle"] is False
    assert host.build_active_release(policy)["policy_sha256"] == sha(host.canonical_bytes(policy))
    assert "private_key" not in json.dumps(policy)
    changed = host.validate_policy(policy)
    changed["discovery_enabled"] = True
    assert policy["discovery_enabled"] is False


def test_supervisor_runtime_is_separate_and_has_only_the_two_fixed_roots(policy):
    interpreter = policy["interpreter"]
    venv = r"C:\ProgramData\ShorePrincess\runtime-msabp\venv"
    base = r"C:\ProgramData\ShorePrincess\runtime-msabp\python312"
    assert interpreter["executable"] == venv + r"\Scripts\python.exe"
    assert interpreter["base_executable"] == base + r"\python.exe"
    assert interpreter["prefix"] == venv
    assert interpreter["base_prefix"] == base
    assert set(interpreter["roots"]) == {venv, base}
    assert len(interpreter["roots"]) == 2
    assert str(host.installed_paths(policy)["python"]) == interpreter["executable"]
    interpreter["roots"].reverse()
    assert host.validate_policy(policy) == policy


@pytest.mark.parametrize("field,replacement", [
    ("prefix", r"C:\ProgramData\ShorePrincess\venv"),
    ("prefix", r"C:\Users\David\venv"),
    ("base_prefix", r"C:\Users\David\AppData\Local\Programs\Python\Python312"),
    ("base_prefix", r"C:\Program Files\Python312"),
    ("base_prefix", r"C:\ProgramData\ShorePrincess\other-python312"),
])
def test_consistent_alternative_runtime_inventory_is_never_a_fallback(policy, field, replacement):
    interpreter = policy["interpreter"]
    original = interpreter[field]
    def relocate(path):
        return replacement + path[len(original):] if path == original or path.startswith(original + "\\") else path
    for name in ("executable", "base_executable", "prefix", "base_prefix"):
        interpreter[name] = relocate(interpreter[name])
    interpreter["roots"] = [relocate(path) for path in interpreter["roots"]]
    interpreter["files"] = {relocate(path): digest for path, digest in interpreter["files"].items()}
    interpreter["directories"] = [relocate(path) for path in interpreter["directories"]]
    with pytest.raises(ValueError, match="fixed installation"):
        host.validate_policy(policy)


@pytest.mark.parametrize("field,value", [
    ("base_executable", r"C:\ProgramData\ShorePrincess\runtime-msabp\venv\Scripts\python.exe"),
    ("base_executable", r"C:\Users\David\AppData\Local\Programs\Python\Python312\python.exe"),
    ("base_prefix", r"C:\ProgramData\ShorePrincess\runtime-msabp\venv"),
])
def test_interpreter_fields_cannot_select_another_pinned_or_user_python(policy, field, value):
    policy["interpreter"][field] = value
    with pytest.raises(ValueError, match="fixed installation"):
        host.validate_policy(policy)


@pytest.mark.parametrize("extra", [r"C:\Other\Python312", r"C:\Users\David\Python312",
                                   r"C:\ProgramData\ShorePrincess\other-runtime"])
def test_even_complete_extra_runtime_roots_cannot_expand_the_dedicated_baseline(policy, extra):
    interpreter = policy["interpreter"]
    interpreter["roots"].append(extra)
    interpreter["files"][extra + r"\Lib\os.py"] = "a" * 64
    interpreter["directories"].append(extra + r"\Lib")
    with pytest.raises(ValueError, match="two fixed dedicated runtime roots"):
        host.validate_policy(policy)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("schema_version", 2),
    ("mode", "anything"), ("project_id", "other"), ("discovery_enabled", 1),
    ("installation_id", "../bad"), ("host_sha256", "0" * 64), ("command", "python.exe")])
def test_policy_rejects_unknown_fields_and_type_coercions(policy, field, value):
    policy[field] = value
    with pytest.raises(ValueError):
        host.validate_policy(policy)


@pytest.mark.parametrize("mutation", ["interpreter", "missing-base", "base-root", "outside-root",
    "overlap", "missing-directory", "relative", "case-collision", "empty-root",
    "duplicate-root", "case-root", "old-pyvenv"])
def test_interpreter_inventory_is_explicit_and_closed(policy, mutation):
    interpreter = policy["interpreter"]
    if mutation == "interpreter":
        interpreter["executable"] = r"C:\Users\David\python.exe"
    elif mutation == "missing-base":
        interpreter["files"].pop(interpreter["base_executable"])
    elif mutation == "base-root":
        interpreter["roots"].remove(interpreter["base_prefix"])
    elif mutation == "outside-root":
        interpreter["files"][r"C:\Users\David\surprise.pth"] = "a" * 64
    elif mutation == "overlap":
        interpreter["roots"].append(interpreter["prefix"] + r"\Scripts")
    elif mutation == "missing-directory":
        interpreter["directories"] = []
    elif mutation == "relative":
        interpreter["files"]["python.exe"] = "a" * 64
    elif mutation == "case-collision":
        interpreter["files"][interpreter["base_executable"].upper()] = "a" * 64
    elif mutation == "duplicate-root":
        interpreter["roots"] = [interpreter["prefix"]] * 2
    elif mutation == "case-root":
        interpreter["roots"][0] = interpreter["roots"][0].upper()
    elif mutation == "old-pyvenv":
        digest = interpreter["files"].pop(str(host.VENV_PATH / "pyvenv.cfg"))
        interpreter["files"][str(host.ROOT / "venv" / "pyvenv.cfg")] = digest
    else:
        interpreter["roots"].append(r"C:\Other\Empty")
    with pytest.raises(ValueError):
        host.validate_policy(policy)


@pytest.mark.parametrize("mutation", ["node", "fleet", "authority-key", "role", "actions",
    "port", "duplicate-key", "unknown-worker", "profile-digest", "unknown-bootstrap-field"])
def test_policy_preserves_fixed_identity_and_release_bindings(policy, mutation):
    bootstrap = policy["bootstrap"]
    if mutation == "node":
        bootstrap["network"]["nodes"]["other"] = {"address": "100.64.0.4", "dns_name": "other."}
    elif mutation == "fleet":
        bootstrap["fleet_id"] = "other"
    elif mutation == "authority-key":
        bootstrap["authority_public_key"] = public_key(5)
    elif mutation == "role":
        bootstrap["workers"]["coconutg2-bell"]["role"] = "controller"
    elif mutation == "actions":
        bootstrap["workers"]["coconutg2-bell"]["allowed_actions"] += ["attempt"]
    elif mutation == "port":
        bootstrap["collector"]["endpoint"] = "https://100.64.0.1:443"
    elif mutation == "duplicate-key":
        bootstrap["workers"]["coconutg2-bell"]["public_key"] = bootstrap["authority_public_key"]
    elif mutation == "unknown-worker":
        bootstrap["workers"]["remote-bell"] = bootstrap["workers"]["coconutg2-bell"]
    elif mutation == "profile-digest":
        policy["profile"]["release_id"] = "a" * 64
    else:
        bootstrap["private_key"] = "not accepted"
    with pytest.raises(ValueError):
        host.validate_policy(policy)


def test_active_pointer_matches_all_policy_bytes_and_state_version(policy):
    active = host.build_active_release(policy)
    host.verify_active_release(policy, active)
    for field, value in (("release_id", "f" * 64), ("policy_sha256", "e" * 64),
                         ("installation_id", "b" * 32), ("state_schema_version", True)):
        changed = {**active, field: value}
        with pytest.raises(host.HostMaintenanceRequired, match="installer recovery"):
            host.verify_active_release(policy, changed)
    policy["discovery_enabled"] = True
    with pytest.raises(host.HostMaintenanceRequired):
        host.verify_active_release(policy, active)


def test_normal_maintenance_stop_is_distinct_from_first_activation(policy):
    policy["stop_when_idle"] = True
    assert host.validate_policy(policy)["stop_when_idle"] is True
    policy["discovery_enabled"] = True
    with pytest.raises(ValueError):
        host.validate_policy(policy)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":1.0}', b'{"a":NaN}',
                                  b'\xef\xbb\xbf{}', b'{}' * 65536],
                         ids=["duplicate", "float", "nan", "bom", "oversized"])
def test_duplicate_float_bom_and_oversized_trusted_json_fail_closed(raw):
    with pytest.raises((ValueError, UnicodeError)):
        host._json(raw)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "release"
    names = ["src/pkg/__init__.py", "scripts/worker.py"]
    pins = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ("# fixture " + name).encode()
        path.write_bytes(data)
        pins[name] = sha(data)
    return root, pins


def unchecked_acl(_path):
    # Test staging is deliberately user-owned; no production flag disables ACLs.
    pass


def test_full_source_whitelist_is_checked_before_import(package):
    root, pins = package
    checked = []
    host.verify_tree(root, pins, acl_checker=checked.append)
    assert root in checked
    assert all(root / name in checked for name in pins)


@pytest.mark.parametrize("extra", ["unknown.py", "src/pkg/unknown.pyc", "empty/", "src/__pycache__/"])
def test_unknown_file_and_even_empty_directory_rejected(package, extra):
    root, pins = package
    path = root / extra
    if extra.endswith("/"):
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unexpected")
    with pytest.raises(host.HostMaintenanceRequired, match="Unknown"):
        host.verify_tree(root, pins, acl_checker=unchecked_acl)


def test_runtime_can_pin_an_existing_empty_directory_but_no_other(package):
    root, pins = package
    (root / "Include").mkdir()
    declared = ["src", "src/pkg", "scripts", "Include"]
    host.verify_tree(root, pins, acl_checker=unchecked_acl, allowed_directories=declared)
    (root / "Include").rmdir()
    with pytest.raises(host.HostMaintenanceRequired, match="Missing"):
        host.verify_tree(root, pins, acl_checker=unchecked_acl, allowed_directories=declared)


@pytest.mark.parametrize("kind", ["tamper", "missing", "hardlink", "case-alias", "reparse"])
def test_pin_identity_and_redirects_fail_closed(package, tmp_path, monkeypatch, kind):
    root, pins = package
    target = root / "scripts/worker.py"
    if kind == "tamper":
        target.write_bytes(b"changed")
    elif kind == "missing":
        target.unlink()
    elif kind == "hardlink":
        os.link(target, tmp_path / "alias.py")
    elif kind == "case-alias":
        target.rename(target.with_name("WORKER.PY"))
    else:
        real = Path.lstat
        def redirected(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            if path == target.parent:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        monkeypatch.setattr(Path, "lstat", redirected)
    with pytest.raises(host.HostMaintenanceRequired):
        host.verify_tree(root, pins, acl_checker=unchecked_acl)


@pytest.mark.parametrize("owner,aces", [("S-1-5-21-123", []), ("S-1-5-18", None),
    ("S-1-5-18", [(0, 0, 0x2, "S-1-1-0")]),
    ("S-1-5-18", [(0, 8, 0x40, "S-1-5-32-545")]),
    ("S-1-5-18", [(0, 0, 0x40000, "S-1-5-32-545")]),
    ("S-1-5-18", [(9, 0, 0, "S-1-5-18")])])
def test_untrusted_owner_write_acl_and_unknown_ace_rejected(owner, aces):
    with pytest.raises(host.HostMaintenanceRequired):
        host.assert_acl_records(owner, aces)


def test_trusted_system_admin_owner_rights_and_untrusted_read_are_allowed():
    host.assert_acl_records("S-1-5-32-544", [(0, 0, 0x1F01FF, "S-1-5-18"),
        (0, 0, 0x1F01FF, "S-1-3-4"), (0, 0, 0x120089, "S-1-5-32-545"),
        (1, 0, 0x1F01FF, "S-1-1-0")])


def test_os_ancestors_allow_unrelated_child_creation_but_no_replacement():
    host.assert_acl_records(host.TRUSTED_INSTALLER_SID, [(0, 0, 0x4, "S-1-5-11"),
        (0, 8, 0x1F01FF, "S-1-3-0")], ancestor=True)
    for mask in (0x40, 0x10000, 0x40000, 0x80000, 0x10000000):
        with pytest.raises(host.HostMaintenanceRequired):
            host.assert_acl_records("S-1-5-18", [(0, 0, mask, "S-1-5-11")], ancestor=True)
    with pytest.raises(host.HostMaintenanceRequired):
        host.assert_acl_records("S-1-5-21-user", [], ancestor=True)
    with pytest.raises(host.HostMaintenanceRequired):
        host.assert_acl_records(host.TRUSTED_INSTALLER_SID, [])


@pytest.mark.skipif(os.name != "nt", reason="real read-only Windows ACL call")
def test_win32_acl_reader_can_parse_real_test_file_without_changing_it(tmp_path):
    path = tmp_path / "acl.txt"
    path.write_bytes(b"unchanged")
    try:
        host.assert_private_acl(path)
    except host.HostMaintenanceRequired as exc:
        # A regular user-owned pytest temp tree is correctly untrusted. This
        # still exercises actual GetNamedSecurityInfo/GetAce/SID ctypes calls.
        assert str(exc) == "Untrusted owner or unrestricted NULL DACL"
    assert path.read_bytes() == b"unchanged"


@pytest.mark.skipif(os.name != "nt", reason="real read-only Windows ACL call")
def test_repeated_acl_checks_leave_nothing_behind():
    # One verification makes tens of thousands of these calls, and the worker
    # verifies at every step. On the first live run a ctypes type declared per
    # call cost about 200 MB per step, permanently (2026-11-11, CoconutG2).
    import ctypes
    import gc

    target = os.environ["SystemRoot"]
    host.assert_private_acl(target, ancestor=True)
    api = host._acl_api()
    cache = getattr(ctypes, "_pointer_type_cache", {})
    gc.collect()
    before = (len(cache), len(ctypes.Structure.__subclasses__()))
    for _ in range(300):
        host.assert_private_acl(target, ancestor=True)
    gc.collect()
    assert (len(cache), len(ctypes.Structure.__subclasses__())) == before
    assert host._acl_api() is api


def identity_args(policy):
    interpreter = policy["interpreter"]
    return {name: interpreter[name] for name in ("executable", "base_executable", "prefix", "base_prefix")} | {
        "flags": SimpleNamespace(isolated=1, dont_write_bytecode=1, no_user_site=1, ignore_environment=1),
        "import_paths": interpreter["roots"]}


@pytest.mark.parametrize("mutation", ["executable", "base_executable", "prefix", "base_prefix",
                                      "isolation", "bytecode", "path"])
def test_current_interpreter_and_import_context_are_exact(policy, mutation):
    args = identity_args(policy)
    host.verify_execution_identity(policy, **args)
    if mutation in {"executable", "base_executable", "prefix", "base_prefix"}:
        args[mutation] = r"C:\Other\python.exe"
    elif mutation == "isolation":
        args["flags"].isolated = 0
    elif mutation == "bytecode":
        args["flags"].dont_write_bytecode = 0
    else:
        args["import_paths"] = [r"C:\Users\David\Downloads"]
    with pytest.raises(host.HostMaintenanceRequired):
        host.verify_execution_identity(policy, **args)


def test_host_lock_is_lifelong_distinct_and_never_reinitialized(tmp_path):
    lock, operation = tmp_path / "host.lock", tmp_path / "instance.lock"
    lock.write_bytes(b"0")
    operation.write_bytes(b"0")
    before = lock.stat().st_ino
    with host.lifetime_lock(lock):
        with lock.open("rb", buffering=0) as sentinel:
            assert sentinel.read(1) == b"0"
        with host.lifetime_lock(operation):
            pass
        with pytest.raises(host.HostMaintenanceRequired, match="owns"):
            with host.lifetime_lock(lock):
                pytest.fail("Second host acquired lifetime lock")
    with host.lifetime_lock(lock):
        pass
    assert before == lock.stat().st_ino
    lock.unlink()
    with pytest.raises((OSError, host.HostMaintenanceRequired)):
        with host.lifetime_lock(lock):
            pytest.fail("Missing lock was recreated")
    assert not lock.exists()


def test_host_lock_competes_with_a_second_process(tmp_path):
    lock = tmp_path / "host.lock"
    lock.write_bytes(b"0")
    code = ("import importlib.util; "
            "s=importlib.util.spec_from_file_location('host'," + repr(host.__file__) + "); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "c=m.lifetime_lock(" + repr(str(lock)) + "); c.__enter__()")
    with host.lifetime_lock(lock):
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                capture_output=True, timeout=15, check=False)
    assert result.returncode != 0
    assert b"owns the OS lock" in result.stderr


def test_noninstalled_cli_and_path_override_never_start_or_initialize(tmp_path):
    for extra, expected in (([], 78), (["--policy", str(tmp_path / "fake-policy.json")], 2),
                            (["--root", str(tmp_path)], 2)):
        result = subprocess.run([sys.executable, "-I", "-B", host.__file__, *extra],
                                cwd=tmp_path, capture_output=True, timeout=15, check=False)
        assert result.returncode == expected
    assert list(tmp_path.iterdir()) == []


def test_no_project_import_at_standalone_module_load(monkeypatch):
    tree = ast.parse(Path(host.__file__).read_text(encoding="utf-8"))
    # The only project import is nested inside run_verified_host, reached after
    # full verification. Loading the host itself must remain stdlib-only.
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [node.module if isinstance(node, ast.ImportFrom) else node.names[0].name for node in top_imports]
    assert all(not name.startswith(("msabp_opt", "shore_control")) for name in names)
    spec = importlib.util.spec_from_file_location("offline_host_load", host.__file__)
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("Import cannot read state"))
    spec.loader.exec_module(loaded)
    assert loaded.MODE == host.MODE


def test_managed_launch_is_fixed_and_persists_handle_exit_evidence(tmp_path, policy, monkeypatch):
    paths = {"app": tmp_path / "app", "python": Path(str(host.PYTHON_PATH)),
        "policy": Path(str(host.FIXED_POLICY)), "logs": tmp_path / "logs", "state": tmp_path / "state"}
    for name in ("app", "logs", "state"):
        paths[name].mkdir()
    calls = []
    record = {"pid": 42, "creation_time": 100, "exit_time": 200, "exit_code": 0}
    class Job:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            self.close()
        def spawn(self, argv, cwd, **kwargs):
            calls.append((argv, cwd, kwargs))
            return SimpleNamespace(wait=lambda: 0, exit_record=SimpleNamespace(to_dict=lambda: record))
        def active_processes(self):
            return 0
        def close(self):
            pass
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("No Popen fallback"))
    context = {"paths": paths, "policy": policy, "profile": policy["profile"]}
    assert host.run_verified_host(context, job_factory=Job) == 0
    assert calls[0][0] == [str(host.PYTHON_PATH), "-I", "-B", "-u",
        str(paths["app"] / "scripts/simulation/shore_supervisor.py"), "--policy", str(host.FIXED_POLICY), "--run"]
    saved = json.loads(next(paths["state"].glob("host-exit-*.json")).read_bytes())
    assert saved["worker"] == record
    assert saved["tree_exit_observed"] is saved["normal_tree_exit"] is True
    assert saved["protocol_completion_proven"] is False


def test_managed_assignment_failure_has_no_fallback_or_success_record(tmp_path, policy, monkeypatch):
    paths = {"app": tmp_path, "python": Path(str(host.PYTHON_PATH)),
        "policy": Path(str(host.FIXED_POLICY)), "logs": tmp_path / "logs", "state": tmp_path / "state"}
    paths["logs"].mkdir()
    paths["state"].mkdir()
    class RejectedJob:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def spawn(self, *_args, **_kwargs):
            raise OSError("AssignProcessToJobObject rejected")
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("No Popen fallback"))
    with pytest.raises(OSError, match="AssignProcessToJobObject"):
        host.run_verified_host({"paths": paths, "policy": policy, "profile": policy["profile"]},
                               job_factory=RejectedJob)
    assert list(paths["state"].iterdir()) == []


@pytest.fixture
def installed_snapshot(tmp_path, policy, monkeypatch):
    """Translate fixed paths to a temporary snapshot only within this test.

    No production override flag or alternate installed policy is introduced.
    The pure file checks remain real; Windows ownership is tested separately.
    """
    paths = host.installed_paths(policy)
    def mapped(value):
        value = PureWindowsPath(str(value))
        return tmp_path / "installation" / str(value.relative_to(host.ROOT))
    def put(path, raw):
        path = mapped(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name in policy["interpreter"]["files"]:
        put(name, name.encode())
    for name in policy["interpreter"]["directories"]:
        mapped(name).mkdir(parents=True, exist_ok=True)
    manifest = policy["profile"]["release_manifest"]
    for name in manifest["source_hashes"]:
        put(paths["app"] / name, name.encode())
    for name in manifest["shore_source_hashes"]:
        put(paths["shore"] / name, name.encode())
    host_bytes = b"# pinned fixture host"
    policy["host_sha256"] = sha(host_bytes)
    put(paths["host"], host_bytes)
    put(paths["policy"], host.canonical_bytes(policy))
    put(paths["active_release"], host.canonical_bytes(host.build_active_release(policy)))
    put(paths["bootstrap"], host.canonical_bytes(policy["bootstrap"]))
    put(paths["manifest"], host.canonical_bytes(manifest))
    for key in ("state", "config", "logs"):
        mapped(paths[key]).mkdir(parents=True, exist_ok=True)
    for key in ("host_lock", "maintenance_lock", "worker_config"):
        put(paths[key], b"0")
    marker = {**host.STATE_SCHEMA, "installation_id": policy["installation_id"],
              "config_parent": str(paths["config"]), "profile": policy["profile"]}
    put(paths["state"] / "runtime-installed.json", host.canonical_bytes(marker))
    put(paths["state"] / "runtime.sqlite", b"original ledger; runtime validates SQLite separately")
    original_read, original_trusted, original_tree = host._read, host._trusted, host.verify_tree
    # _read calls the module-global _trusted, so recognize already-mapped paths.
    def trusted(path, **kwargs):
        actual = Path(path)
        if tmp_path != actual and tmp_path not in actual.parents:
            actual = mapped(path)
        return original_trusted(actual, root=tmp_path, acl_checker=unchecked_acl)
    monkeypatch.setattr(host, "_trusted", trusted)
    # verify_tree internally reads mapped paths too; retain real read checks.
    def read(path, **kwargs):
        actual = Path(path)
        if tmp_path != actual and tmp_path not in actual.parents:
            actual = mapped(path)
        return original_read(actual, root=tmp_path, acl_checker=unchecked_acl,
                             maximum=kwargs.get("maximum"))
    monkeypatch.setattr(host, "_read", read)
    monkeypatch.setattr(host, "verify_tree", lambda root, pins, **kwargs:
        original_tree(mapped(root), pins, acl_checker=unchecked_acl,
                      allowed_directories=kwargs.get("allowed_directories")))
    identity = identity_args(policy)
    monkeypatch.setattr(host, "sys", SimpleNamespace(
        flags=identity["flags"], executable=identity["executable"],
        _base_executable=identity["base_executable"], prefix=identity["prefix"],
        base_prefix=identity["base_prefix"], path=identity["import_paths"],
        modules={"supervisor_host": SimpleNamespace(__file__=str(host.HOST_PATH))}))
    monkeypatch.setattr(host, "__file__", str(host.HOST_PATH))
    return SimpleNamespace(policy=policy, paths=paths, mapped=mapped, put=put)


def test_full_preimport_snapshot_accepts_only_consistent_installed_files(installed_snapshot):
    snapshot = installed_snapshot
    before = list(host.sys.path)
    result = host.verify_installation()
    assert result["policy"] == snapshot.policy
    assert result["paths"] == snapshot.paths
    assert result["bootstrap_data"] == host.canonical_bytes(snapshot.policy["bootstrap"])
    assert host.sys.path == before


@pytest.mark.parametrize("failure", ["active", "host", "runtime", "release", "bootstrap",
    "extra-source", "ledger", "marker", "lock", "state", "module"])
def test_preimport_snapshot_rejects_drift_and_missing_original_state(installed_snapshot, failure):
    snapshot = installed_snapshot
    paths = snapshot.paths
    if failure == "extra-source":
        snapshot.put(paths["app"] / "surprise.py", b"unapproved")
    elif failure == "state":
        snapshot.mapped(paths["logs"]).rmdir()
    elif failure == "module":
        host.sys.modules["surprise"] = SimpleNamespace(__file__=r"C:\Users\David\surprise.py")
    elif failure in {"ledger", "marker", "lock"}:
        target = {"ledger": paths["state"] / "runtime.sqlite",
                  "marker": paths["state"] / "runtime-installed.json", "lock": paths["host_lock"]}[failure]
        snapshot.mapped(target).unlink()
    else:
        target = {"active": paths["active_release"], "host": paths["host"],
            "runtime": snapshot.policy["interpreter"]["base_executable"],
            "release": paths["app"] / "scripts/simulation/shore_supervisor.py",
            "bootstrap": paths["bootstrap"]}[failure]
        snapshot.put(target, b"{}")
    with pytest.raises((host.HostMaintenanceRequired, ValueError, OSError)):
        host.verify_installation()


def test_initial_release_marker_is_immutable_history_not_current_pointer(installed_snapshot):
    snapshot = installed_snapshot
    original_profile = json.loads(host.canonical_bytes(snapshot.policy["profile"]))
    original_manifest = original_profile["release_manifest"]
    changed_name = "scripts/simulation/shore_supervisor.py"
    old_bytes = b"# prior approved worker"
    original_manifest["source_hashes"][changed_name] = sha(old_bytes)
    original_profile["release_id"] = sha(host.canonical_bytes(original_manifest))
    original_paths = host.installed_paths({"profile": original_profile})
    for name in original_manifest["source_hashes"]:
        snapshot.put(original_paths["app"] / name, old_bytes if name == changed_name else name.encode())
    snapshot.put(original_paths["manifest"], host.canonical_bytes(original_manifest))
    marker = {**host.STATE_SCHEMA, "installation_id": snapshot.policy["installation_id"],
              "config_parent": str(snapshot.paths["config"]), "profile": original_profile}
    marker_bytes = host.canonical_bytes(marker)
    marker_path = snapshot.paths["state"] / "runtime-installed.json"
    snapshot.put(marker_path, marker_bytes)
    assert host.verify_installation()["profile"] == snapshot.policy["profile"]
    assert snapshot.mapped(marker_path).read_bytes() == marker_bytes
    # Host does not infer a replacement from the active release when historical
    # pinned code disappears; runtime separately gates unreleased old jobs.
    snapshot.mapped(original_paths["app"] / changed_name).unlink()
    with pytest.raises(host.HostMaintenanceRequired, match="Missing"):
        host.verify_installation()
    assert snapshot.mapped(marker_path).read_bytes() == marker_bytes


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("installation_id", "e" * 32),
                                         ("config_parent", r"C:\Other\Config")])
def test_incompatible_original_marker_is_never_rewritten(installed_snapshot, field, value):
    snapshot = installed_snapshot
    path = snapshot.paths["state"] / "runtime-installed.json"
    original = json.loads(snapshot.mapped(path).read_bytes())
    original[field] = value
    changed = host.canonical_bytes(original)
    snapshot.put(path, changed)
    with pytest.raises(host.HostMaintenanceRequired, match="Original runtime marker"):
        host.verify_installation()
    assert snapshot.mapped(path).read_bytes() == changed
