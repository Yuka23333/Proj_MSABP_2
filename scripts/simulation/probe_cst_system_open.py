"""One-shot CST open/close probe intended for a Windows SYSTEM task.

The probe never opens the source project in place.  It creates a new evidence
directory, copies the standalone CST container into it, starts a fresh CST
DesignEnvironment, opens and closes the copy, and writes one bounded JSON
report.  It does not run a solver and it never terminates a CST process.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "msabp.cst-system-open-probe"
SCHEMA_VERSION = 1
_PROCESS_NAMES = ("cst", "designenvironment", "dbstorage", "solver")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity() -> str:
    try:
        result = subprocess.run(
            ["whoami.exe"], capture_output=True, text=True, timeout=10, check=False,
            encoding="utf-8", errors="replace",
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return getpass.getuser()


def _session_id() -> int | None:
    if os.name != "nt":
        return None
    value = ctypes.c_ulong()
    if not ctypes.windll.kernel32.ProcessIdToSessionId(  # type: ignore[attr-defined]
        os.getpid(), ctypes.byref(value)
    ):
        return None
    return int(value.value)


def _cst_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    try:
        result = subprocess.run(
            ["tasklist.exe", "/FO", "CSV", "/NH"], capture_output=True,
            text=True, timeout=20, check=False, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2 or not any(token in row[0].lower() for token in _PROCESS_NAMES):
            continue
        try:
            pid = int(row[1].replace(",", ""))
        except ValueError:
            continue
        rows.append({"name": row[0], "pid": pid})
    return sorted(rows, key=lambda item: (item["name"].lower(), item["pid"]))


def _bounded_error(error: BaseException) -> tuple[str, str]:
    return type(error).__name__, str(error).replace("\0", "")[:4096]


def run_probe(
    source_project: str | Path,
    workspace: str | Path,
    *,
    expected_sha256: str | None,
    environment_factory: Callable[[], Any],
    module_origin: str | None,
    environment: Mapping[str, str] | None = None,
    control_mode_was_present: bool | None = None,
    identity_fn: Callable[[], str] = _identity,
    session_id_fn: Callable[[], int | None] = _session_id,
    process_snapshot_fn: Callable[[], list[dict[str, Any]]] = _cst_processes,
) -> dict[str, Any]:
    """Run the isolated open/close probe and return its complete evidence."""

    source = Path(source_project).expanduser().resolve()
    root = Path(workspace).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".cst":
        raise FileNotFoundError(f"CST source project does not exist: {source}")
    if root.exists():
        raise FileExistsError(f"Probe workspace already exists: {root}")

    source_hash = _sha256(source)
    if expected_sha256 is not None and source_hash != expected_sha256.lower():
        raise ValueError("CST source project hash differs from the approved probe input")

    root.mkdir(parents=True)
    model_root = root / "model"
    model_root.mkdir()
    project_copy = model_root / source.name
    shutil.copyfile(source, project_copy)
    copy_hash = _sha256(project_copy)
    if copy_hash != source_hash:
        raise OSError("Isolated CST copy differs from its source")

    inherited_control_mode = (
        "MSABP_CONTROL_MODE" in (environment or os.environ)
        if control_mode_was_present is None
        else bool(control_mode_was_present)
    )
    os.environ.pop("MSABP_CONTROL_MODE", None)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "started_utc": _utc_now(),
        "completed_utc": None,
        "success": False,
        "identity": identity_fn(),
        "session_id": session_id_fn(),
        "process_id": os.getpid(),
        "host_name": platform.node(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cst_interface_origin": module_origin,
        "source_project": str(source),
        "source_sha256_before": source_hash,
        "source_sha256_after": None,
        "project_copy": str(project_copy),
        "project_copy_sha256_before": copy_hash,
        "project_copy_sha256_after": None,
        "control_mode_was_present": inherited_control_mode,
        "processes_before": process_snapshot_fn(),
        "processes_after": None,
        "opened": False,
        "project_closed": False,
        "environment_closed": False,
        "error_type": None,
        "error_message": "",
    }
    design_environment: Any | None = None
    project: Any | None = None
    try:
        design_environment = environment_factory()
        project = design_environment.open_project(str(project_copy))
        report["opened"] = True
    except BaseException as error:  # report CST/COM failures before returning nonzero
        report["error_type"], report["error_message"] = _bounded_error(error)
    finally:
        if project is not None:
            try:
                project.close()
                report["project_closed"] = True
            except BaseException as error:
                if report["error_type"] is None:
                    report["error_type"], report["error_message"] = _bounded_error(error)
        if design_environment is not None:
            try:
                design_environment.close()
                report["environment_closed"] = True
            except BaseException as error:
                if report["error_type"] is None:
                    report["error_type"], report["error_message"] = _bounded_error(error)

    report["source_sha256_after"] = _sha256(source)
    report["project_copy_sha256_after"] = _sha256(project_copy)
    report["processes_after"] = process_snapshot_fn()
    report["completed_utc"] = _utc_now()
    report["success"] = bool(
        report["opened"]
        and report["project_closed"]
        and report["environment_closed"]
        and report["error_type"] is None
        and report["source_sha256_after"] == source_hash
    )
    return report


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inherited_control_mode = "MSABP_CONTROL_MODE" in os.environ
    # The CST package must never observe the Shore synthetic/remote mode marker.
    # Remove it before importing cst.interface, not merely before opening CST.
    os.environ.pop("MSABP_CONTROL_MODE", None)
    try:
        import cst.interface

        report = run_probe(
            args.project,
            args.workspace,
            expected_sha256=args.expected_sha256,
            environment_factory=cst.interface.DesignEnvironment.new,
            module_origin=getattr(cst.interface, "__file__", None),
            control_mode_was_present=inherited_control_mode,
        )
    except BaseException as error:
        error_type, message = _bounded_error(error)
        report = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "started_utc": _utc_now(),
            "completed_utc": _utc_now(),
            "success": False,
            "identity": _identity(),
            "session_id": _session_id(),
            "process_id": os.getpid(),
            "host_name": platform.node(),
            "python_executable": sys.executable,
            "control_mode_was_present": inherited_control_mode,
            "error_type": error_type,
            "error_message": message,
        }
    try:
        _write_new_json(args.report.expanduser().resolve(), report)
    except BaseException as error:
        print(f"CST SYSTEM probe could not write its report: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"success": report["success"], "report": str(args.report)}, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
