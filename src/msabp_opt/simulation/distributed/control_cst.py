"""Stdlib-only parent adapter for one Shore-controlled CST case.

The Supervisor runtime imports this module without importing CST, NumPy,
Shapely, or the antenna builders.  One fixed, locally configured solver Python
is started for each case.  The child owns and closes its CST environment, then
the parent independently validates the response and result manifest.

No task row may select an executable, source root, or project path.  Those
values are supplied by :class:`CstProcessPolicy`, which is constructed from the
installed worker policy rather than job material.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


REQUEST_SCHEMA = "msabp.shore-cst-child-request"
RESPONSE_SCHEMA = "msabp.shore-cst-child-response"
SCHEMA_VERSION = 1
CONTROL_MODE_ENV = "MSABP_CONTROL_MODE"
MANIFEST_FILENAME = "manifest.json"
SIMULATION_MODE = "antenna_tree"
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_RESERVED_ROW_KEYS = frozenset(
    {
        "application_root",
        "command",
        "executable",
        "project_path",
        "python_executable",
        "python_path",
        "source_root",
    }
)
_REQUIRED_ARTIFACTS = {
    "s11": "S11.csv",
    "rad_eff": "Rad_Eff.csv",
    "tot_eff": "Tot_Eff.csv",
    "farfield_source": "Farfield Source [1].ffs",
    "geometry_tree": "geometry_tree.json",
}
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "case_id",
        "row",
        "application_root",
        "project_copy",
        "project_copy_sha256",
        "output_root",
        "coordinate_quantum_mm",
        "allow_disconnected_conductor",
        "command_timeout_seconds",
    }
)
_SUCCESS_RESPONSE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "success",
        "case_id",
        "case_directory",
        "manifest_path",
        "project_copy_sha256_before",
        "project_copy_sha256_after",
    }
)
_ERROR_RESPONSE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "success",
        "case_id",
        "error_type",
        "error_message",
    }
)


class CstProcessError(RuntimeError):
    """The isolated CST child failed or returned invalid evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_json(text: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _read_json_object(path: Path, *, expected_fields: frozenset[str]) -> dict[str, Any]:
    payload = _strict_json(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or frozenset(payload) != expected_fields:
        raise CstProcessError(f"Unexpected JSON field set in {path.name}")
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_response(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid4().hex)
    try:
        _write_new_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bounded_tail(path: Path, limit: int = 4096) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace").replace("\0", "")


@dataclass(frozen=True)
class CstProcessPolicy:
    """Worker-local executable, source, and immutable template policy."""

    python_executable: Path
    application_root: Path
    project_template: Path
    project_template_sha256: str

    def validated(self) -> "CstProcessPolicy":
        python = _resolved(self.python_executable)
        application = _resolved(self.application_root)
        project = _resolved(self.project_template)
        expected = str(self.project_template_sha256).lower()
        if not python.is_file():
            raise FileNotFoundError(f"Solver Python does not exist: {python}")
        environment_root = python.parent
        for directory in (
            environment_root,
            environment_root / "Library" / "bin",
            environment_root / "Scripts",
        ):
            if not directory.is_dir():
                raise FileNotFoundError(f"Conda runtime directory does not exist: {directory}")
        if not (application / "src").is_dir() or not (application / "scripts").is_dir():
            raise FileNotFoundError("Application root lacks the fixed src/scripts closure")
        if not project.is_file() or project.suffix.lower() != ".cst":
            raise FileNotFoundError(f"CST template does not exist: {project}")
        if _SHA256.fullmatch(expected) is None or _sha256(project) != expected:
            raise ValueError("CST template SHA-256 differs from worker policy")
        return CstProcessPolicy(python, application, project, expected)


class CstProcessSession:
    """Maid session shim; the isolated child owns every CST object."""

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None


ProcessRunner = Callable[..., subprocess.CompletedProcess[Any]]


class CstCaseExecutor:
    """Callable ``run_csv_row`` replacement backed by one child per case."""

    def __init__(
        self,
        policy: CstProcessPolicy,
        *,
        process_runner: ProcessRunner = subprocess.run,
    ) -> None:
        self.policy = policy.validated()
        self._process_runner = process_runner

    def __call__(self, row: Mapping[str, Any], **kwargs: Any) -> Any:
        return self.run(row, **kwargs)

    def run(self, row: Mapping[str, Any], **kwargs: Any) -> Any:
        case_id = kwargs.get("case_id")
        if type(case_id) is not str or _SAFE_CASE_ID.fullmatch(case_id) is None:
            raise ValueError("CST child requires a bounded safe case ID")
        if not isinstance(row, Mapping) or any(type(key) is not str for key in row):
            raise TypeError("CST child row must be a string-keyed mapping")
        if _RESERVED_ROW_KEYS.intersection(key.lower() for key in row):
            raise ValueError("Task row may not select an executable or filesystem path")
        try:
            frozen_row = json.loads(json.dumps(dict(row), ensure_ascii=True, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise TypeError("CST child row is not JSON serializable") from error

        project_argument = _resolved(kwargs.get("project_path", ""))
        if not _same_path(project_argument, self.policy.project_template):
            raise ValueError("Task runtime project path differs from worker-local template policy")
        if kwargs.get("project") is not None:
            raise ValueError("Parent process may not pass a live CST project to the child")
        if kwargs.get("dry_run") is not False:
            raise ValueError("CST child executor requires a real, explicitly selected run")
        if kwargs.get("overwrite") is not False:
            raise ValueError("CST child executor refuses overwrite or implicit retry semantics")
        if kwargs.get("save_project_after_case") is not False:
            raise ValueError("CST child executor never saves over the frozen template")
        coordinate = kwargs.get("coordinate_quantum_mm")
        timeout = kwargs.get("command_timeout")
        if type(coordinate) not in (int, float) or not math.isfinite(coordinate) or coordinate <= 0:
            raise ValueError("coordinate_quantum_mm must be a positive finite number")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("command_timeout must be a positive finite number")
        if type(kwargs.get("allow_disconnected_conductor")) is not bool:
            raise TypeError("allow_disconnected_conductor must be boolean")
        stage_callback = kwargs.get("stage_callback")
        if stage_callback is not None and not callable(stage_callback):
            raise TypeError("stage_callback must be callable")

        output_root = _resolved(kwargs.get("output_root", ""))
        output_root.mkdir(parents=True, exist_ok=True)
        workspace = output_root / ".cst-child" / (case_id + "-" + uuid4().hex)
        workspace.mkdir(parents=True, exist_ok=False)
        model_root = workspace / "model"
        model_root.mkdir()
        project_copy = model_root / self.policy.project_template.name
        shutil.copyfile(self.policy.project_template, project_copy)
        if _sha256(project_copy) != self.policy.project_template_sha256:
            raise OSError("Isolated CST template copy failed SHA-256 validation")

        request_path = workspace / "request.json"
        response_path = workspace / "response.json"
        stdout_path = workspace / "stdout.log"
        stderr_path = workspace / "stderr.log"
        request = {
            "schema": REQUEST_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "row": frozen_row,
            "application_root": str(self.policy.application_root),
            "project_copy": str(project_copy),
            "project_copy_sha256": self.policy.project_template_sha256,
            "output_root": str(output_root),
            "coordinate_quantum_mm": float(coordinate),
            "allow_disconnected_conductor": kwargs["allow_disconnected_conductor"],
            "command_timeout_seconds": float(timeout),
        }
        _write_new_json(request_path, request)

        environment_root = self.policy.python_executable.parent
        environment = os.environ.copy()
        environment.pop(CONTROL_MODE_ENV, None)
        environment["CONDA_PREFIX"] = str(environment_root)
        environment["PATH"] = os.pathsep.join(
            (
                str(environment_root),
                str(environment_root / "Library" / "bin"),
                str(environment_root / "Scripts"),
                environment.get("PATH", ""),
            )
        )
        command = [
            str(self.policy.python_executable),
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        if stage_callback is not None:
            stage_callback("starting_cst_subprocess")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = self._process_runner(
                command,
                cwd=str(workspace),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if stage_callback is not None:
            stage_callback("cst_subprocess_finished")
        if not response_path.is_file():
            detail = _bounded_tail(stderr_path) or _bounded_tail(stdout_path)
            raise CstProcessError(
                f"CST child exited {completed.returncode} without response evidence: {detail[:4096]}"
            )

        raw_response = _strict_json(response_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw_response, dict):
            raise CstProcessError("CST child response is not a JSON object")
        fields = _SUCCESS_RESPONSE_FIELDS if raw_response.get("success") is True else _ERROR_RESPONSE_FIELDS
        if frozenset(raw_response) != fields:
            raise CstProcessError("CST child response has an unexpected field set")
        if raw_response.get("schema") != RESPONSE_SCHEMA or raw_response.get("schema_version") != SCHEMA_VERSION:
            raise CstProcessError("CST child response schema mismatch")
        if raw_response.get("case_id") != case_id:
            raise CstProcessError("CST child response case identity mismatch")
        if raw_response.get("success") is not True:
            raise CstProcessError(
                f"CST child exited {completed.returncode}: {raw_response.get('error_type')}: "
                f"{str(raw_response.get('error_message', ''))[:4096]}"
            )
        if completed.returncode != 0:
            raise CstProcessError("CST child returned success evidence with a nonzero exit code")
        if _sha256(self.policy.project_template) != self.policy.project_template_sha256:
            raise CstProcessError("Frozen CST template changed during child execution")
        if raw_response["project_copy_sha256_before"] != self.policy.project_template_sha256:
            raise CstProcessError("Child did not start from the approved CST template")

        case_directory = _resolved(raw_response["case_directory"])
        manifest_path = _resolved(raw_response["manifest_path"])
        if not _is_within(case_directory, output_root) or case_directory.parent != output_root:
            raise CstProcessError("CST child result directory escaped the attempt output root")
        if manifest_path != case_directory / MANIFEST_FILENAME:
            raise CstProcessError("CST child manifest path is not canonical")
        return _result_from_manifest(case_id, case_directory, manifest_path)


def _artifact_path(case_directory: Path, record: Any, expected_name: str) -> Path:
    if not isinstance(record, dict) or frozenset(record) != {"path", "size_bytes", "sha256"}:
        raise CstProcessError(f"Malformed artifact record for {expected_name}")
    relative = PurePosixPath(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != expected_name:
        raise CstProcessError(f"Artifact path mismatch for {expected_name}")
    path = case_directory / expected_name
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        raise CstProcessError(f"Artifact size mismatch for {expected_name}")
    digest = str(record["sha256"]).lower()
    if _SHA256.fullmatch(digest) is None or _sha256(path) != digest:
        raise CstProcessError(f"Artifact SHA-256 mismatch for {expected_name}")
    return path


def _result_from_manifest(case_id: str, case_directory: Path, manifest_path: Path) -> Any:
    if not manifest_path.is_file():
        raise CstProcessError("CST child did not produce manifest.json")
    manifest = _strict_json(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise CstProcessError("CST manifest is not a JSON object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("dry_run") is not False
        or manifest.get("case_id") != case_id
        or manifest.get("simulation_mode") != SIMULATION_MODE
    ):
        raise CstProcessError("CST manifest does not describe the approved completed tree case")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or frozenset(artifacts) != frozenset(_REQUIRED_ARTIFACTS):
        raise CstProcessError("CST manifest artifact map is not the exact tree result closure")
    expected_names = {MANIFEST_FILENAME, *_REQUIRED_ARTIFACTS.values()}
    try:
        actual_names = {item.name for item in case_directory.iterdir() if item.is_file()}
        if actual_names != expected_names or any(item.is_dir() for item in case_directory.iterdir()):
            raise CstProcessError("CST result directory is not the exact tree file closure")
    except OSError as error:
        raise CstProcessError("CST result directory cannot be inspected safely") from error
    paths = {
        key: _artifact_path(case_directory, artifacts.get(key), filename)
        for key, filename in _REQUIRED_ARTIFACTS.items()
    }
    elapsed = manifest.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise CstProcessError("CST manifest elapsed_seconds is invalid")
    from msabp_opt.simulation.distributed.case_runner import CaseRunResult

    return CaseRunResult(
        case_id=case_id,
        case_directory=case_directory,
        manifest_path=manifest_path,
        s11_path=paths["s11"],
        rad_eff_path=paths["rad_eff"],
        tot_eff_path=paths["tot_eff"],
        farfield_source_path=paths["farfield_source"],
        dry_run=False,
        elapsed_seconds=float(elapsed),
        simulation_mode=SIMULATION_MODE,
    )


@dataclass(frozen=True)
class _ChildBindings:
    environment_factory: Callable[[], Any]
    case_runner: Callable[..., Any]


def _load_child_bindings(application_root: Path) -> _ChildBindings:
    for entry in (application_root / "src", application_root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import cst.interface
    from msabp_opt.simulation.distributed import case_runner_tree

    return _ChildBindings(cst.interface.DesignEnvironment.new, case_runner_tree.run_csv_row)


def _execute_child_request(
    request_path: Path,
    response_path: Path,
    *,
    bindings_loader: Callable[[Path], _ChildBindings] = _load_child_bindings,
) -> bool:
    os.environ.pop(CONTROL_MODE_ENV, None)
    if response_path.parent != request_path.parent:
        raise ValueError("Child response must remain beside its request")
    request = _read_json_object(request_path, expected_fields=_REQUEST_FIELDS)
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Child request schema mismatch")
    case_id = request.get("case_id")
    if type(case_id) is not str or _SAFE_CASE_ID.fullmatch(case_id) is None:
        raise ValueError("Child request has an invalid case ID")
    row = request.get("row")
    if not isinstance(row, dict) or any(type(key) is not str for key in row):
        raise ValueError("Child request row is not a string-keyed object")
    if _RESERVED_ROW_KEYS.intersection(key.lower() for key in row):
        raise ValueError("Child request row selects a forbidden executable or path")
    application_root = _resolved(request["application_root"])
    project_copy = _resolved(request["project_copy"])
    output_root = _resolved(request["output_root"])
    expected_hash = str(request["project_copy_sha256"]).lower()
    if not _is_within(project_copy, request_path.parent) or not project_copy.is_file():
        raise ValueError("Child project copy escaped its request workspace")
    if _SHA256.fullmatch(expected_hash) is None or _sha256(project_copy) != expected_hash:
        raise ValueError("Child project copy SHA-256 mismatch")
    if not (application_root / "src").is_dir() or not (application_root / "scripts").is_dir():
        raise ValueError("Child application root lacks the fixed src/scripts closure")
    if not output_root.is_dir() or not _is_within(request_path.parent, output_root):
        raise ValueError("Child request workspace escaped the attempt output root")
    bindings = bindings_loader(application_root)
    environment = project = result = None
    copy_before = _sha256(project_copy)
    try:
        environment = bindings.environment_factory()
        project = environment.open_project(str(project_copy))
        result = bindings.case_runner(
            row,
            project_path=project_copy,
            output_root=output_root,
            project=project,
            case_id=case_id,
            coordinate_quantum_mm=request["coordinate_quantum_mm"],
            allow_disconnected_conductor=request["allow_disconnected_conductor"],
            command_timeout=request["command_timeout_seconds"],
            overwrite=False,
            save_project_after_case=False,
            dry_run=False,
            stage_callback=None,
            local_artifact_root=output_root / "local_only",
        )
    finally:
        close_error = None
        if project is not None:
            try:
                project.close()
            except BaseException as error:
                close_error = error
        if environment is not None:
            try:
                environment.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if close_error is not None:
            raise close_error
    case_directory = _resolved(result.case_directory)
    manifest_path = _resolved(result.manifest_path)
    if not _is_within(case_directory, output_root) or case_directory.parent != output_root:
        raise ValueError("Child runner returned a case directory outside output_root")
    if manifest_path != case_directory / MANIFEST_FILENAME:
        raise ValueError("Child runner returned a noncanonical manifest path")
    response = {
        "schema": RESPONSE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "case_id": case_id,
        "case_directory": str(case_directory),
        "manifest_path": str(manifest_path),
        "project_copy_sha256_before": copy_before,
        "project_copy_sha256_after": _sha256(project_copy),
    }
    _write_response(response_path, response)
    return True


def _child_main(request_path: Path, response_path: Path) -> int:
    case_id = "unknown"
    try:
        try:
            untrusted = json.loads(request_path.read_text(encoding="utf-8-sig"))
            if isinstance(untrusted, dict) and type(untrusted.get("case_id")) is str:
                case_id = untrusted["case_id"][:128]
        except Exception:
            pass
        _execute_child_request(request_path, response_path)
        return 0
    except BaseException as error:
        payload = {
            "schema": RESPONSE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "success": False,
            "case_id": case_id,
            "error_type": type(error).__name__,
            "error_message": (str(error) + "\n" + traceback.format_exc()).replace("\0", "")[:16384],
        }
        try:
            _write_response(response_path, payload)
        except BaseException:
            return 2
        return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed Shore CST child request")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args(argv)
    if not args.child or args.request is None or args.response is None:
        parser.error("only --child with --request and --response is supported")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return _child_main(args.request.expanduser().resolve(), args.response.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
