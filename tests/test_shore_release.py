"""Public-only release fixtures; never touch a real source tree or identity."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
from types import SimpleNamespace

import pytest

from msabp_opt.simulation.distributed.control_release import (
    PROFILE_SCHEMA, RELEASE_SCHEMA, RELEASE_SOURCE_FILES, build_release_manifest,
    public_profile, release_id, runtime_paths, safe_job_id, validate_profile, verify_release,
)
from msabp_opt.simulation.distributed.control_remote_prepare import (
    PREPARATION_SOURCE_FILES, remote_paths,
)
from msabp_opt.simulation.distributed.control_remote_worker import MODE, SHORE_SOURCE_FILES


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.socket", lambda *_a, **_k: pytest.fail("No network"))
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_k: pytest.fail("No process"))
    app, shore = tmp_path / "app", tmp_path / "shore"
    for root, names in ((app, RELEASE_SOURCE_FILES), (shore, SHORE_SOURCE_FILES)):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("# dummy public source: " + name + "\n").encode())
    manifest = build_release_manifest(app, shore)
    profile = public_profile(manifest, hashlib.sha256(b"public bootstrap fixture").hexdigest())
    return SimpleNamespace(app=app, shore=shore, manifest=manifest, profile=profile)


def test_fixed_release_closure_retains_existing_and_separately_pinned_host():
    assert set(PREPARATION_SOURCE_FILES) < set(RELEASE_SOURCE_FILES)
    assert len(RELEASE_SOURCE_FILES) == len(set(RELEASE_SOURCE_FILES))
    for name in ("control_usb", "control_winjob", "control_job_auth", "control_release",
                 "control_job_material", "control_supervisor"):
        assert "src/msabp_opt/simulation/distributed/" + name + ".py" in RELEASE_SOURCE_FILES
    # Runtime/Princess are now delivered; the bootstrap host is pinned outside
    # the selected release so no version can choose its own pre-import verifier.
    assert "scripts/simulation/shore_supervisor.py" in RELEASE_SOURCE_FILES
    assert "scripts/simulation/shore_supervisor_princess.py" in RELEASE_SOURCE_FILES
    assert "scripts/simulation/supervisor_host.py" not in RELEASE_SOURCE_FILES
    assert len(SHORE_SOURCE_FILES) == 14


def test_manifest_is_fixed_public_canonical_and_deterministic(package):
    manifest = package.manifest
    assert set(manifest) == {"schema", "schema_version", "mode", "project_id", "source_hashes",
                             "shore_source_hashes"}
    assert manifest["schema"] == RELEASE_SCHEMA
    assert manifest["mode"] == MODE and manifest["project_id"] == "MSABP_2"
    assert manifest == build_release_manifest(package.app, package.shore)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert release_id(manifest) == hashlib.sha256(canonical).hexdigest()
    reversed_manifest = dict(reversed(list(manifest.items())))
    reversed_manifest["source_hashes"] = dict(reversed(list(manifest["source_hashes"].items())))
    assert release_id(reversed_manifest) == release_id(manifest)
    assert "release_id" not in manifest


def test_manifest_build_does_not_promote_unrelated_repo_files(package):
    (package.app / "unreviewed.py").write_bytes(b"# do not approve\n")
    assert build_release_manifest(package.app, package.shore) == package.manifest
    with pytest.raises(ValueError, match="Unknown release file"):
        verify_release(package.profile, package.app, package.shore)


def test_profile_roundtrip_and_defensive_copies(package):
    assert package.profile["schema"] == PROFILE_SCHEMA
    assert package.profile["release_manifest"] == package.manifest
    assert validate_profile(package.profile) == package.profile
    assert verify_release(package.profile, package.app, package.shore) == package.profile
    validated = validate_profile(package.profile)
    validated["release_manifest"]["source_hashes"].clear()
    assert package.profile["release_manifest"]["source_hashes"]
    package.manifest["source_hashes"].clear()
    assert validate_profile(package.profile)


@pytest.mark.parametrize("change", [
    {"schema": "different"}, {"schema_version": True}, {"schema_version": 2},
    {"mode": "shore-fake-v1"}, {"project_id": "another"}, {"release_id": "f" * 64},
    {"bootstrap_sha256": "0" * 64}, {"bootstrap_sha256": "F" * 64},
    {"source_root": r"C:\untrusted"}, {"command": "python untrusted.py"},
])
def test_profile_rejects_malformed_or_redirecting_metadata(package, change):
    with pytest.raises(ValueError):
        validate_profile({**package.profile, **change})


@pytest.mark.parametrize("field", ["schema", "schema_version", "mode", "project_id", "release_id",
                                    "release_manifest", "bootstrap_sha256"])
def test_profile_requires_every_field(package, field):
    value = copy.deepcopy(package.profile)
    del value[field]
    with pytest.raises(ValueError):
        validate_profile(value)


@pytest.mark.parametrize("change", [
    {"schema": PROFILE_SCHEMA}, {"schema_version": True}, {"schema_version": 2},
    {"mode": "different"}, {"project_id": "different"}, {"release_id": "f" * 64},
    {"source_hashes": {}}, {"shore_source_hashes": {}},
])
def test_manifest_rejects_malformed_fields_or_recursive_digest(package, change):
    with pytest.raises(ValueError):
        release_id({**package.manifest, **change})


@pytest.mark.parametrize("kind", ["missing", "extra", "traversal", "case", "invalid_hash"])
@pytest.mark.parametrize("key", ["source_hashes", "shore_source_hashes"])
def test_fixed_pin_sets_cannot_be_redefined(package, kind, key):
    manifest = copy.deepcopy(package.manifest)
    pins = manifest[key]
    name = next(iter(pins))
    if kind == "missing":
        pins.pop(name)
    elif kind == "extra":
        pins["download.py"] = "a" * 64
    elif kind == "traversal":
        pins["../" + name] = pins.pop(name)
    elif kind == "case":
        pins[name.upper()] = pins.pop(name)
    else:
        pins[name] = "G" * 64
    with pytest.raises(ValueError):
        release_id(manifest)


def test_runtime_paths_use_immutable_release_and_dedicated_python_without_migrating_legacy(package):
    expected = remote_paths("synthetic-005")
    assert expected["python_executable"] == r"C:\ProgramData\ShorePrincess\venv\Scripts\python.exe"
    expected["source_root"] = str(PureWindowsPath(r"C:\ProgramData\ShorePrincess")
                                  / "project-msabp" / "releases"
                                  / package.profile["release_id"] / "app")
    expected["python_executable"] = r"C:\ProgramData\ShorePrincess\runtime-msabp\venv\Scripts\python.exe"
    assert runtime_paths(package.profile, "synthetic-005") == expected
    assert runtime_paths(package.profile, "synthetic-006")["source_root"] == expected["source_root"]
    assert remote_paths("synthetic-005")["python_executable"] == r"C:\ProgramData\ShorePrincess\venv\Scripts\python.exe"


def test_runtime_python_matches_the_fixed_standalone_host_for_every_job(package):
    from scripts.simulation import supervisor_host as host

    installed = host.installed_paths({"profile": package.profile})
    for job_id in ("synthetic-005", "synthetic-006"):
        paths = runtime_paths(package.profile, job_id)
        assert paths["python_executable"] == str(host.PYTHON_PATH) == str(installed["python"])


@pytest.mark.parametrize("value", ["job-1", "a", "0", "a" * 80, "job_1.part-2", "com10"])
def test_safe_job_id_accepts_one_unambiguous_component(value):
    assert safe_job_id(value) == value


@pytest.mark.parametrize("value", [
    "", "a" * 81, "Job", "JOB", "测试", "café", "../job", r"..\job", ".", "..",
    "a/../b", "a/b", "a\\b", "C:job", "a:stream", "job.", "job ", " job", "job\n",
    "job\x00", "con", "con.txt", "prn", "aux", "nul", "com1", "com9.csv", "lpt1", "lpt9",
    "conin$", "conout$", "clock$", "com¹", "_job", "-job", None, 1, True,
])
def test_safe_job_id_rejects_windows_aliases_unsafe_paths_and_case_collisions(value):
    with pytest.raises(ValueError, match="Windows-safe job ID"):
        safe_job_id(value)


@pytest.mark.parametrize("root_name, names", [("app", RELEASE_SOURCE_FILES), ("shore", SHORE_SOURCE_FILES)])
def test_missing_and_changed_source_fail_without_repair(package, root_name, names):
    path = getattr(package, root_name) / names[0]
    original = path.read_bytes()
    path.write_bytes(original + b"# changed\n")
    with pytest.raises(ValueError, match="drifted"):
        verify_release(package.profile, package.app, package.shore)
    assert path.read_bytes() == original + b"# changed\n"
    path.unlink()
    with pytest.raises(ValueError, match="Missing release source"):
        verify_release(package.profile, package.app, package.shore)
    with pytest.raises(ValueError, match="Missing release source"):
        build_release_manifest(package.app, package.shore)
    assert not path.exists()


@pytest.mark.parametrize("extra", ["unreviewed.py", "unreviewed.pyc", "__pycache__/entry.pyc",
                                   "src/__pycache__/entry.pyc", "empty-directory/"])
def test_unknown_project_files_and_even_empty_cache_directories_are_refused(package, extra):
    path = package.app / extra
    if extra.endswith("/"):
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="Unknown release"):
        verify_release(package.profile, package.app, package.shore)


@pytest.mark.parametrize("extra", ["deploy/remote/Common-Remote.ps1", "README.md",
                                   "shore_control/other_module.py", "shore_control/__pycache__/other.pyc",
                                   "original-maintenance-directory/"])
def test_shore_maintenance_files_are_preserved_without_claiming_closed_module_tree(package, extra, monkeypatch):
    path = package.shore / extra
    content = b"existing Shore installation resource: not in project release"
    if extra.endswith("/"):
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    real_scandir = os.scandir
    def project_only_scandir(root):
        if Path(root) == package.shore or package.shore in Path(root).parents:
            pytest.fail("Shore maintenance tree must not be scanned")
        return real_scandir(root)
    monkeypatch.setattr(os, "scandir", project_only_scandir)
    assert verify_release(package.profile, package.app, package.shore) == package.profile
    assert path.is_dir() if extra.endswith("/") else path.read_bytes() == content
    assert build_release_manifest(package.app, package.shore) == package.manifest


def test_case_aliases_are_not_installed_project_source_names(package):
    path = package.app / RELEASE_SOURCE_FILES[0]
    path.rename(path.with_name(path.name.upper()))
    with pytest.raises(ValueError, match="Unknown release"):
        verify_release(package.profile, package.app, package.shore)


@pytest.mark.parametrize("kind", ["symlink", "junction"])
@pytest.mark.parametrize("part", ["root", "parent", "file"])
@pytest.mark.parametrize("root_name", ["app", "shore"])
def test_redirected_files_and_ancestors_are_refused_without_following(package, monkeypatch, kind, part, root_name):
    # Deterministic Windows reparse/symlink attributes, without requiring the
    # machine's symlink privilege or creating an actual filesystem redirect.
    root = getattr(package, root_name)
    names = RELEASE_SOURCE_FILES if root_name == "app" else SHORE_SOURCE_FILES
    target = {"root": root, "parent": root / names[0].split("/")[0],
              "file": root / names[0]}[part]
    real_lstat = Path.lstat
    def redirected(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else info.st_mode,
                                   st_file_attributes=0x400 if kind == "junction" else 0)
        return info
    monkeypatch.setattr(Path, "lstat", redirected)
    with pytest.raises(ValueError, match="Redirected runtime paths"):
        verify_release(package.profile, package.app, package.shore)


@pytest.mark.parametrize("root_name", ["app", "shore"])
def test_hardlinked_source_is_not_an_immutable_standalone_file(package, tmp_path, root_name):
    names = RELEASE_SOURCE_FILES if root_name == "app" else SHORE_SOURCE_FILES
    path = getattr(package, root_name) / names[0]
    os.link(path, tmp_path / "mutable-alias.py")
    with pytest.raises(ValueError, match="non-aliased"):
        verify_release(package.profile, package.app, package.shore)


def test_changed_file_identity_during_hashing_is_refused(package, monkeypatch):
    path = package.app / RELEASE_SOURCE_FILES[0]
    real_digest = hashlib.file_digest
    def changed(stream, algorithm):
        result = real_digest(stream, algorithm)
        if Path(stream.name) == path:
            with path.open("ab") as writable:
                writable.write(b"# drift during read\n")
        return result
    monkeypatch.setattr(hashlib, "file_digest", changed)
    with pytest.raises(ValueError, match="changed during verification"):
        verify_release(package.profile, package.app, package.shore)


def test_syntactic_profile_validation_does_not_claim_software_approval(package):
    # A coherent forged profile is valid metadata; policy must still compare it
    # to the administrator-approved profile. It cannot validate installed bytes.
    altered = copy.deepcopy(package.manifest)
    altered["source_hashes"][RELEASE_SOURCE_FILES[0]] = "a" * 64
    forged = public_profile(altered, package.profile["bootstrap_sha256"])
    assert validate_profile(forged) == forged
    assert forged != package.profile
    with pytest.raises(ValueError, match="drifted"):
        verify_release(forged, package.app, package.shore)
