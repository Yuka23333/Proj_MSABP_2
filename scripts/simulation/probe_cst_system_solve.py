"""One-shot real CST solve probe intended for a Windows SYSTEM task.

The approved template is copied into a fresh workspace. A hash-pinned source
closure builds the default tree antenna, runs one solver pass, and exports the
standard result closure. The probe never retries or terminates a CST process.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile


SCHEMA = "msabp.cst-system-solve-probe"
SCHEMA_VERSION = 1
CASE_ID = "system_solve"
MAX_ARCHIVE_MEMBERS = 256
MAX_ARCHIVE_EXPANDED_BYTES = 32 * 1024 * 1024
_PROCESS_NAMES = ("cst", "designenvironment", "dbstorage", "solver")
REQUIRED_ARTIFACTS = {
    "s11": "S11.csv",
    "rad_eff": "Rad_Eff.csv",
    "tot_eff": "Tot_Eff.csv",
    "farfield_source": "Farfield Source [1].ffs",
}


@dataclass(frozen=True)
class RuntimeBindings:
    environment_factory: Callable[[], Any]
    case_executor: Callable[..., Any]
    default_request: Callable[[], Mapping[str, Any]]
    cst_interface_origin: str | None


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
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
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
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2 or not any(token in row[0].lower() for token in _PROCESS_NAMES):
            continue
        try:
            rows.append({"name": row[0], "pid": int(row[1].replace(",", ""))})
        except ValueError:
            continue
    return sorted(rows, key=lambda item: (item["name"].lower(), item["pid"]))


def _bounded_error(error: BaseException) -> tuple[str, str]:
    return type(error).__name__, str(error).replace("\0", "")[:4096]


def _safe_extract_zip(archive: Path, destination: Path) -> list[dict[str, Any]]:
    records = []
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        files = [member for member in members if not member.is_dir()]
        if not files or len(files) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("source archive member count is outside the approved bound")
        if sum(member.file_size for member in files) > MAX_ARCHIVE_EXPANDED_BYTES:
            raise ValueError("source archive expanded size exceeds the approved bound")
        for member in members:
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if (
                not name or relative.is_absolute() or ".." in relative.parts
                or ":" in relative.parts[0] or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"unsafe source archive member: {member.filename!r}")
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with source.open(member) as reader, target.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != member.file_size:
                raise OSError(f"source archive member size mismatch: {member.filename}")
            records.append({"path": relative.as_posix(), "size_bytes": size, "sha256": digest.hexdigest()})
    return sorted(records, key=lambda item: item["path"])


def _validate_source_manifest(source_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = source_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("source archive does not contain source_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    declared = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(declared, list):
        raise ValueError("source manifest file list is missing")
    actual = [record for record in records if record["path"] != "source_manifest.json"]
    if declared != actual:
        raise ValueError("source manifest does not match the extracted source closure")
    return manifest


def _curve_summary(path: Path) -> dict[str, float | int]:
    numeric = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        fields = line.replace(",", " ").replace(";", " ").split()
        if len(fields) < 2:
            continue
        try:
            pair = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if all(math.isfinite(value) for value in pair):
            numeric.append(pair)
    if len(numeric) < 2 or len({pair[0] for pair in numeric}) < 2:
        raise ValueError(f"result curve has fewer than two distinct numeric frequencies: {path.name}")
    return {
        "point_count": len(numeric),
        "frequency_min": min(pair[0] for pair in numeric),
        "frequency_max": max(pair[0] for pair in numeric),
        "value_min": min(pair[1] for pair in numeric),
        "value_max": max(pair[1] for pair in numeric),
    }


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required solve artifact is missing or empty: {path.name}")
    return {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validate_result(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    case_directory = Path(result.case_directory).resolve()
    manifest_path = Path(result.manifest_path).resolve()
    if manifest_path.parent != case_directory:
        raise ValueError("manifest is outside the case result directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        manifest.get("schema_version") != 1 or manifest.get("simulation_mode") != "antenna_tree"
        or manifest.get("case_id") != CASE_ID or manifest.get("status") != "completed"
        or manifest.get("dry_run") is not False
    ):
        raise ValueError("case manifest does not describe the approved completed solve")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("case manifest artifact map is missing")
    records = {}
    curves = {}
    for key, filename in REQUIRED_ARTIFACTS.items():
        path = case_directory / filename
        record = _artifact_record(path, case_directory)
        expected = declared.get(key)
        if not isinstance(expected, dict) or any(expected.get(field) != record[field] for field in ("path", "size_bytes", "sha256")):
            raise ValueError(f"case manifest artifact mismatch: {key}")
        records[key] = record
        if key != "farfield_source":
            curves[key] = _curve_summary(path)
    records["manifest"] = _artifact_record(manifest_path, case_directory)
    return records, curves


def _load_runtime(source_root: Path) -> RuntimeBindings:
    for entry in (str(source_root / "src"), str(source_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import cst.interface
    from scripts.automation import cst_build_msabp_geometry_tree as builder
    from msabp_opt.simulation.distributed import case_runner_tree
    return RuntimeBindings(
        environment_factory=cst.interface.DesignEnvironment.new,
        case_executor=case_runner_tree.run_csv_row,
        default_request=lambda: builder.normalize_request({}),
        cst_interface_origin=getattr(cst.interface, "__file__", None),
    )


def run_probe(
    source_project: str | Path,
    source_archive: str | Path,
    workspace: str | Path,
    *,
    expected_project_sha256: str | None,
    expected_archive_sha256: str | None,
    runtime_loader: Callable[[Path], RuntimeBindings] = _load_runtime,
    control_mode_was_present: bool | None = None,
    identity_fn: Callable[[], str] = _identity,
    session_id_fn: Callable[[], int | None] = _session_id,
    process_snapshot_fn: Callable[[], list[dict[str, Any]]] = _cst_processes,
) -> dict[str, Any]:
    source = Path(source_project).expanduser().resolve()
    archive = Path(source_archive).expanduser().resolve()
    root = Path(workspace).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".cst":
        raise FileNotFoundError(f"CST source project does not exist: {source}")
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise FileNotFoundError(f"solve source archive does not exist: {archive}")
    if root.exists():
        raise FileExistsError(f"Probe workspace already exists: {root}")
    source_hash, archive_hash = _sha256(source), _sha256(archive)
    if expected_project_sha256 and source_hash != expected_project_sha256.lower():
        raise ValueError("CST source project hash differs from the approved probe input")
    if expected_archive_sha256 and archive_hash != expected_archive_sha256.lower():
        raise ValueError("solve source archive hash differs from the approved probe input")

    root.mkdir(parents=True)
    model_root, source_root, result_root = root / "model", root / "source", root / "results"
    model_root.mkdir()
    source_root.mkdir()
    project_copy = model_root / source.name
    shutil.copyfile(source, project_copy)
    copy_hash = _sha256(project_copy)
    if copy_hash != source_hash:
        raise OSError("isolated CST copy differs from its source")
    bundle_files = _safe_extract_zip(archive, source_root)
    source_manifest = _validate_source_manifest(source_root, bundle_files)
    inherited_control_mode = "MSABP_CONTROL_MODE" in os.environ if control_mode_was_present is None else bool(control_mode_was_present)
    os.environ.pop("MSABP_CONTROL_MODE", None)
    report: dict[str, Any] = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "started_utc": _utc_now(), "completed_utc": None, "success": False,
        "identity": identity_fn(), "session_id": session_id_fn(), "process_id": os.getpid(),
        "host_name": platform.node(), "python_executable": sys.executable,
        "python_version": platform.python_version(), "cst_interface_origin": None,
        "source_project": str(source), "source_sha256_before": source_hash, "source_sha256_after": None,
        "source_archive": str(archive), "source_archive_sha256": archive_hash,
        "source_manifest": source_manifest, "source_bundle_files": bundle_files,
        "project_copy": str(project_copy), "project_copy_sha256_before": copy_hash,
        "project_copy_sha256_after": None, "control_mode_was_present": inherited_control_mode,
        "processes_before": process_snapshot_fn(), "processes_after": None,
        "opened": False, "solver_started": False, "solver_completed": False,
        "project_closed": False, "environment_closed": False, "case_directory": None,
        "artifacts": {}, "curves": {}, "stage_events": [], "elapsed_seconds": None,
        "error_type": None, "error_message": "",
    }
    design_environment = project = None
    started = time.perf_counter()

    def stage(name: str) -> None:
        report["stage_events"].append({"stage": str(name), "elapsed_seconds": time.perf_counter() - started})
        if name == "solving":
            report["solver_started"] = True
        elif name == "exporting_1d_results":
            report["solver_completed"] = True

    try:
        bindings = runtime_loader(source_root)
        report["cst_interface_origin"] = bindings.cst_interface_origin
        request = dict(bindings.default_request())
        row = {"sample_id": CASE_ID, "simulation_mode": "antenna_tree", "tree_json": json.dumps(request, separators=(",", ":"), sort_keys=True)}
        design_environment = bindings.environment_factory()
        project = design_environment.open_project(str(project_copy))
        report["opened"] = True
        result = bindings.case_executor(
            row, project_path=project_copy, output_root=result_root, project=project,
            case_id=CASE_ID, coordinate_quantum_mm=0.01, command_timeout=60,
            overwrite=False, save_project_after_case=True, dry_run=False, stage_callback=stage,
        )
        report["case_directory"] = str(Path(result.case_directory).resolve())
        report["artifacts"], report["curves"] = _validate_result(result)
        report["solver_completed"] = bool(report["solver_started"] and report["solver_completed"])
    except BaseException as error:
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
    report["elapsed_seconds"] = time.perf_counter() - started
    report["completed_utc"] = _utc_now()
    report["success"] = bool(
        report["opened"] and report["solver_started"] and report["solver_completed"]
        and report["project_closed"] and report["environment_closed"]
        and report["error_type"] is None and report["source_sha256_after"] == source_hash
        and set(report["artifacts"]) == {*REQUIRED_ARTIFACTS, "manifest"}
    )
    return report


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-project-sha256", required=True)
    parser.add_argument(
        "--source-archive-sha256", "--expected-archive-sha256",
        dest="expected_archive_sha256", required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inherited_control_mode = "MSABP_CONTROL_MODE" in os.environ
    os.environ.pop("MSABP_CONTROL_MODE", None)
    try:
        report = run_probe(
            args.project, args.source_archive, args.workspace,
            expected_project_sha256=args.expected_project_sha256,
            expected_archive_sha256=args.expected_archive_sha256,
            control_mode_was_present=inherited_control_mode,
        )
    except BaseException as error:
        error_type, message = _bounded_error(error)
        report = {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "started_utc": _utc_now(), "completed_utc": _utc_now(), "success": False,
            "identity": _identity(), "session_id": _session_id(), "process_id": os.getpid(),
            "host_name": platform.node(), "python_executable": sys.executable,
            "control_mode_was_present": inherited_control_mode,
            "error_type": error_type, "error_message": message,
        }
    try:
        _write_new_json(args.report.expanduser().resolve(), report)
    except BaseException as error:
        print(f"CST SYSTEM solve probe could not write its report: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"success": report["success"], "report": str(args.report)}, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
