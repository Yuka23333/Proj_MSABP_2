"""Offline project release and pinned Shore dependency/profile validation.

A public profile is deployment metadata, not an authority or trust root. The
supervisor must select its approved profile from protected local policy, never
from downloaded job material. No function here imports Shore, reads identities,
contacts a service, installs code, or changes an existing preparation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat

from .control_remote_worker import (
    MODE, PROJECT_SOURCE_FILES, SHORE_SOURCE_FILES, _hash, _json_bytes, checked_path,
)

RELEASE_SCHEMA = "msabp.supervisor-release"
PROFILE_SCHEMA = "msabp.supervisor-release-profile"
PROJECT_ID = "MSABP_2"
INSTALL_ROOT = PureWindowsPath(r"C:\ProgramData\ShorePrincess")

# This is a reviewed declaration, never a scan of the working tree. The fixed
# host is separately pinned by local policy: including it would make the host's
# pin/release selection unnecessarily self-referential.
RELEASE_SOURCE_FILES = tuple(dict.fromkeys((
    *PROJECT_SOURCE_FILES,
    # Existing preparation closure is listed explicitly to avoid a circular
    # dependency when preparation imports this module for the new profile.
    "scripts/simulation/shore_synthetic_princess.py",
    "scripts/simulation/shore_supervisor_princess.py",
    "scripts/simulation/shore_supervisor.py",
    *("src/msabp_opt/simulation/distributed/" + name + ".py" for name in (
        "control_remote_prepare", "control_remote_controller", "control_remote_session",
        "control_usb", "control_collector", "runtime", "princess", "config", "transport", "bell")),
    *("src/msabp_opt/simulation/distributed/" + name + ".py" for name in (
        "control_usb", "control_winjob", "control_job_auth", "control_release",
        "control_job_material", "control_supervisor", "control_job_http",
        "control_supervisor_controller", "control_supervisor_runtime")),
)))
_MANIFEST_FIELDS = {
    "schema", "schema_version", "mode", "project_id", "source_hashes", "shore_source_hashes",
}
_PROFILE_FIELDS = {
    "schema", "schema_version", "mode", "project_id", "release_id", "release_manifest",
    "bootstrap_sha256",
}
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                    *(f"lpt{i}" for i in range(1, 10))}


def safe_job_id(value: str) -> str:
    """Accept one lowercase ASCII Windows-safe component, at most 80 bytes.

    Reject uppercase instead of normalizing it, so distinct wire identifiers
    cannot silently name the same case-insensitive local directory.
    """
    if (type(value) is not str
            or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", value, flags=re.ASCII) is None
            or value.endswith(".") or value.split(".", 1)[0] in _WINDOWS_DEVICES):
        raise ValueError("Invalid lowercase Windows-safe job ID")
    return value


def _manifest(value: dict) -> dict:
    if (type(value) is not dict or set(value) != _MANIFEST_FIELDS
            or value["schema"] != RELEASE_SCHEMA
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["mode"] != MODE or value["project_id"] != PROJECT_ID):
        raise ValueError("Malformed supervisor release manifest")
    for key, required in (("source_hashes", RELEASE_SOURCE_FILES),
                          ("shore_source_hashes", SHORE_SOURCE_FILES)):
        pins = value[key]
        if type(pins) is not dict or set(pins) != set(required):
            raise ValueError("Release source pins differ from the fixed allowlist: " + key)
        for digest in pins.values():
            _hash(digest, key)
    return json.loads(_json_bytes(value))


def _regular_hash(path: Path) -> str:
    """Hash only a stable regular file, without accepting redirect aliases."""
    path = checked_path(path)
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Missing release source file: " + str(path)) from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Release source must be a regular non-aliased file: " + str(path))
    # The directory ACL/owner gate is an installer/host responsibility. These
    # checks also refuse a changed file identity during offline verification.
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        finished = os.fstat(stream.fileno())
    after = checked_path(path).lstat()
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_nlink)
    # On Windows lstat/fstat can expose different ctime semantics. Compare
    # ctime only within each API, while comparing file identity across APIs.
    if (not identity(before) == identity(opened) == identity(finished) == identity(after)
            or before.st_ctime_ns != after.st_ctime_ns
            or opened.st_ctime_ns != finished.st_ctime_ns):
        raise ValueError("Release source changed during verification: " + str(path))
    return digest


def build_release_manifest(project_root: str | Path, shore_root: str | Path) -> dict:
    """Hash the declared files, allowing unrelated files in a development tree.

    Installed/staged packages must additionally pass ``verify_release``; build
    intentionally does not turn unrelated repository files into approved code.
    """
    project, shore = checked_path(project_root), checked_path(shore_root)
    return _manifest({
        "schema": RELEASE_SCHEMA, "schema_version": 1, "mode": MODE, "project_id": PROJECT_ID,
        "source_hashes": {name: _regular_hash(project / name) for name in RELEASE_SOURCE_FILES},
        "shore_source_hashes": {name: _regular_hash(shore / name) for name in SHORE_SOURCE_FILES},
    })


def release_id(manifest: dict) -> str:
    """SHA256 of canonical manifest bytes, with no self-containing digest field."""
    return hashlib.sha256(_json_bytes(_manifest(manifest))).hexdigest()


def public_profile(manifest: dict, bootstrap_sha256: str) -> dict:
    """Create public metadata for an already locally approved software release."""
    pinned = _manifest(manifest)
    return {
        "schema": PROFILE_SCHEMA, "schema_version": 1, "mode": MODE, "project_id": PROJECT_ID,
        "release_id": release_id(pinned), "release_manifest": pinned,
        "bootstrap_sha256": _hash(bootstrap_sha256, "bootstrap_sha256"),
    }


def validate_profile(profile: dict) -> dict:
    """Validate shape and consistency, not approval; return an independent copy.

    An attacker can construct a consistent profile for arbitrary hashes. Only
    comparison with protected local policy establishes that it is approved.
    """
    if (type(profile) is not dict or set(profile) != _PROFILE_FIELDS
            or profile["schema"] != PROFILE_SCHEMA
            or type(profile["schema_version"]) is not int or profile["schema_version"] != 1
            or profile["mode"] != MODE or profile["project_id"] != PROJECT_ID):
        raise ValueError("Malformed supervisor release profile")
    manifest = _manifest(profile["release_manifest"])
    if _hash(profile["release_id"], "release_id") != release_id(manifest):
        raise ValueError("Profile release ID does not match its canonical manifest")
    return public_profile(manifest, profile["bootstrap_sha256"])


def runtime_paths(profile: dict, job_id: str) -> dict[str, str]:
    """Derive fixed Windows destinations without consulting network path fields."""
    approved = validate_profile(profile)
    job_id = safe_job_id(job_id)
    root = INSTALL_ROOT
    state = root / "state" / "project-msabp" / job_id
    config = root / "project-msabp" / "config" / job_id
    return {
        "source_root": str(root / "project-msabp" / "releases" / approved["release_id"] / "app"),
        "shore_source_root": str(root / "app"),
        "python_executable": str(root / "runtime-msabp" / "venv" / "Scripts" / "python.exe"),
        "worker_config_path": str(root / "state" / "worker" / "worker-config.json"),
        "config_root": str(config),
        "bootstrap_path": str(config / "authority-bootstrap.json"),
        "csv_path": str(config / "samples.csv"),
        "runtime_config_path": str(config / "maid-runtime.json"),
        "bell_config_path": str(config / "bell-config.json"),
        "maid_state_root": str(state / "maid-state"),
        "maid_output_root": str(state / "maid-output"),
        "bell_state_root": str(state / "bell-state"),
    }


def _verify_tree(root: str | Path, pins: dict[str, str], label: str) -> None:
    """Inspect exact package closure, including empty/redirected directories."""
    source = checked_path(root)
    if not source.is_dir():
        raise ValueError("Missing release source directory: " + label)
    directories = {str(parent) for name in pins for parent in PurePosixPath(name).parents
                   if str(parent) != "."}
    found = set()
    pending = [source]
    while pending:
        current = checked_path(pending.pop())
        with os.scandir(current) as entries:
            for entry in entries:
                path = checked_path(Path(entry.path))
                relative = path.relative_to(source).as_posix()
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    if relative not in directories:
                        raise ValueError("Unknown release directory: " + relative)
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    if relative not in pins:
                        raise ValueError("Unknown release file: " + relative)
                    found.add(relative)
                    if _regular_hash(path) != pins[relative]:
                        raise ValueError("Pinned release source has drifted: " + relative)
                else:
                    raise ValueError("Non-regular release entry: " + relative)
    if found != set(pins):
        raise ValueError("Missing release source files: " + ", ".join(sorted(set(pins) - found)))


def _verify_shore_pins(root: str | Path, pins: dict[str, str]) -> None:
    """Check the existing 14-file baseline without inspecting unrelated files.

    Shore's existing app tree also owns maintenance/deployment resources. Those
    are neither project release files nor additions to this fixed dependency
    baseline. This check does not certify a complete Shore import closure.
    """
    source = checked_path(root)
    if not source.is_dir():
        raise ValueError("Missing release source directory: Shore")
    for relative, expected in pins.items():
        if _regular_hash(source / relative) != expected:
            raise ValueError("Pinned release source has drifted: " + relative)


def verify_release(profile: dict, actual_app_root: str | Path, shore_root: str | Path) -> dict:
    """Verify a closed project release and the agreed 14 Shore dependency pins.

    Actual roots are explicit local inputs to support offline package staging.
    The production host must separately enforce the fixed installed destinations
    from local policy, trusted import roots and ACL/owner checks. Project caches,
    unknown files/directories, case aliases, links and missing pins fail closed.
    Shore's original app/maintenance tree is retained: only its 14 declared
    dependencies and their ancestors are inspected for pins and redirects;
    unrelated installed files are not enumerated, removed or rejected. This is
    not proof of a closed whitelist for every Shore module or installed file.
    """
    approved = validate_profile(profile)
    manifest = approved["release_manifest"]
    _verify_tree(actual_app_root, manifest["source_hashes"], "project")
    _verify_shore_pins(shore_root, manifest["shore_source_hashes"])
    return approved
