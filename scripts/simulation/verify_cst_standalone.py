r"""Verify that CST projects can be opened without their companion folders.

The script copies each selected ``.cst`` file into its own empty temporary
directory, opens the copied project through CST Studio Suite, and closes it
without saving. It never modifies the source project.

Run this script with the dedicated ``cstpy`` Conda environment:

    C:\Users\David\.conda\envs\cstpy\python.exe \
        scripts\simulation\verify_cst_standalone.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "simulations" / "models"


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of opening one isolated CST project copy."""

    source: Path
    copied_project: Path
    opened: bool
    generated_entries: tuple[str, ...]
    source_unchanged: bool
    error: str | None = None


def file_sha256(path: Path) -> str:
    """Return a stable content hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_cst_files(source_dir: Path) -> list[Path]:
    """Return the CST project containers in deterministic order."""

    return sorted(source_dir.glob("*.cst"), key=lambda path: path.name.casefold())


def prepare_isolated_copy(source: Path, workspace: Path) -> Path:
    """Copy one CST file into a new directory that is empty beforehand."""

    case_dir = workspace / source.stem
    case_dir.mkdir(parents=False, exist_ok=False)
    if any(case_dir.iterdir()):
        raise RuntimeError(f"Test directory is not empty: {case_dir}")

    copied_project = case_dir / source.name
    shutil.copy2(source, copied_project)

    entries = list(case_dir.iterdir())
    if entries != [copied_project]:
        raise RuntimeError(
            f"Expected exactly one copied .cst file in {case_dir}, got {entries}"
        )
    return copied_project


def verify_project(
    design_environment: object,
    source: Path,
    workspace: Path,
) -> VerificationResult:
    """Open one isolated project copy and close it without saving."""

    source = source.resolve()
    original_hash = file_sha256(source)
    copied_project = prepare_isolated_copy(source, workspace)
    project = None

    try:
        project = design_environment.open_project(copied_project)
        if project is None:
            raise RuntimeError("CST returned no Project object")
        opened = True
        error = None
    except Exception as exc:
        opened = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if project is not None:
            try:
                project.close()
            except Exception as close_exc:
                if error is None:
                    opened = False
                    error = (
                        "Project opened, but closing failed: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )

    generated_entries = tuple(
        sorted(
            path.name
            for path in copied_project.parent.iterdir()
            if path != copied_project
        )
    )
    return VerificationResult(
        source=source,
        copied_project=copied_project,
        opened=opened,
        generated_entries=generated_entries,
        source_unchanged=file_sha256(source) == original_hash,
        error=error,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy .cst files into empty temporary folders and verify that CST "
            "can open each copy without its original companion directory."
        )
    )
    parser.add_argument(
        "cst_files",
        nargs="*",
        type=Path,
        help=(
            "CST projects to verify. If omitted, all .cst files directly under "
            "simulations/models/ are used."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory searched when no explicit CST files are supplied.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the isolated test workspace and print its path.",
    )
    return parser.parse_args(argv)


def resolve_sources(args: argparse.Namespace) -> list[Path]:
    sources = (
        [path.resolve() for path in args.cst_files]
        if args.cst_files
        else discover_cst_files(args.source_dir.resolve())
    )
    if not sources:
        raise FileNotFoundError("No .cst files were found")

    invalid = [path for path in sources if not path.is_file() or path.suffix.lower() != ".cst"]
    if invalid:
        raise FileNotFoundError(
            "Invalid CST project path(s): " + ", ".join(str(path) for path in invalid)
        )
    return sources


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = resolve_sources(args)

    try:
        import cst.interface
    except ImportError as exc:
        raise RuntimeError(
            "The CST Python libraries are unavailable. Run this script with "
            r"C:\Users\David\.conda\envs\cstpy\python.exe."
        ) from exc

    workspace = Path(tempfile.mkdtemp(prefix="cst_standalone_check_"))
    print(f"[CST check] workspace: {workspace}")
    print(f"[CST check] source projects: {len(sources)}")

    design_environment = None
    results: list[VerificationResult] = []
    try:
        design_environment = cst.interface.DesignEnvironment()
        for source in sources:
            print(f"[CST check] opening isolated copy: {source.name}")
            result = verify_project(design_environment, source, workspace)
            results.append(result)
            status = "PASS" if result.opened and result.source_unchanged else "FAIL"
            print(f"[CST check] {status}: {source.name}")
            print(
                "[CST check] generated beside copy: "
                + (", ".join(result.generated_entries) or "(none)")
            )
            if result.error:
                print(f"[CST check] error: {result.error}")
    finally:
        if design_environment is not None:
            try:
                design_environment.close()
            except Exception as exc:
                print(
                    f"[CST check] warning: failed to close CST frontend: {exc}",
                    file=sys.stderr,
                )

    all_passed = all(
        result.opened and result.source_unchanged for result in results
    )
    if args.keep_workspace:
        print(f"[CST check] kept workspace: {workspace}")
    else:
        shutil.rmtree(workspace, ignore_errors=True)
        print("[CST check] temporary workspace removed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
