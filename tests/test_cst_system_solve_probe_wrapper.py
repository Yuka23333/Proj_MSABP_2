from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "simulation" / "Invoke-CstSystemSolveProbe.ps1"
REMOTE = ROOT / "scripts" / "simulation" / "Invoke-CstSystemSolveProbeRemote.ps1"
PROBE = ROOT / "scripts" / "simulation" / "probe_cst_system_solve.py"


@pytest.mark.parametrize("path", [WRAPPER, REMOTE])
def test_solve_powershell_scripts_parse(path: Path) -> None:
    executables = [shutil.which("powershell.exe"), shutil.which("pwsh.exe")]
    executables = [value for value in executables if value]
    if not executables:
        pytest.skip("PowerShell is unavailable")
    escaped = str(path).replace("'", "''")
    command = (
        "$tokens=$null; $errors=$null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{escaped}',"
        "[ref]$tokens,[ref]$errors); if($errors.Count){$errors | % ToString; exit 1}"
    )
    for executable in executables:
        result = subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_plan_is_solver_explicit_but_default_execution_is_off() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    guard = source.index("if (-not $Execute -and -not $PreflightOnly)")
    assert "runs_solver = $true" in source
    assert "automatic_retry = $false" in source
    assert "standard_default_tree_case = $true" in source
    assert "PLAN ONLY. -Execute starts one real CST solve as SYSTEM" in source
    assert guard < source.index("$audit = @'")
    assert guard < source.index("New-SolveSourceBundle $repositoryRoot")
    assert guard < source.index("$prepare = @'")
    assert guard < source.index("Invoke-NativeCaptured $pscp")
    assert guard < source.index("Starting one approved SYSTEM CST solve")


def test_preflight_is_read_only_and_precedes_bundle_or_staging() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    audit = source.split("$audit = @'", 1)[1].split("'@", 1)[0]
    assert "New-Item" not in audit
    assert "Copy-Item" not in audit
    assert "RegisterTask" not in audit
    assert "run_solver" not in audit
    assert "if ($PreflightOnly)" in source
    assert source.index("if ($PreflightOnly)") < source.index("New-SolveSourceBundle $repositoryRoot")


def test_transport_is_pinned_and_keeps_full_native_errors() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    for option in (
        "'-batch'", "'-no-antispoof'", "'-T'", "'-ssh'", "'-4'", "'-P'",
        "'-noagent'", "'-a'", "'-x'", "'-noshare'", "'-no-trivial-auth'",
        "'-i'", "'-hostkey'",
    ):
        assert option in source
    assert "'-Command', 'exit 0'" in source
    assert "Get-AuthenticodeSignature -LiteralPath $binary" in source
    assert "$ErrorActionPreference = 'Continue'" in source
    assert "$exitCode = $LASTEXITCODE" in source
    assert "Invoke-ShoreCommand $plink $identity $node $fingerprint $audit" in source
    assert "Invoke-ShoreCommand $plink $identity $node $fingerprint $launch" in source


def test_source_closure_is_fixed_and_hashed() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "scripts\\automation" in source
    assert "scripts\\geometry" in source
    assert "src\\msabp_opt\\__init__.py" in source
    assert "src\\msabp_opt\\simulation\\__init__.py" in source
    assert "src\\msabp_opt\\simulation\\distributed" in source
    assert "Get-FileHash -Algorithm SHA256" in source
    assert "source_manifest.json" in source
    assert "source_closure_sha256" in source
    assert "source_bundle.zip" in source


def test_source_bundle_is_deterministic_and_manifest_matches(tmp_path: Path) -> None:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        pytest.skip("PowerShell is unavailable")
    source = WRAPPER.read_text(encoding="utf-8")
    hash_start = source.index("function Get-StringSha256")
    hash_end = source.index("function ConvertTo-EncodedCommand", hash_start)
    bundle_start = source.index("function Get-SolveSourceFiles")
    bundle_end = source.index("$shoreRoot = Find-ShoreControlRoot", bundle_start)
    functions = source[hash_start:hash_end] + "\n" + source[bundle_start:bundle_end]
    root = str(ROOT).replace("'", "''")
    first = str(tmp_path / "one" / "source_bundle.zip").replace("'", "''")
    second = str(tmp_path / "two" / "source_bundle.zip").replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop'\n" + functions + "\n"
        + f"$root='{root}'; $files=Get-SolveSourceFiles $root; "
        + f"$one=New-SolveSourceBundle $root $files '{first}'; "
        + f"$two=New-SolveSourceBundle $root $files '{second}'; "
        + "[ordered]@{one=$one;two=$two} | ConvertTo-Json -Depth 5 -Compress"
    )
    result = subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["one"]["sha256"] == payload["two"]["sha256"]
    assert payload["one"]["file_count"] >= 10

    with zipfile.ZipFile(tmp_path / "one" / "source_bundle.zip") as archive:
        names = archive.namelist()
        assert names[-1] == "source_manifest.json"
        manifest = json.loads(archive.read("source_manifest.json"))
        assert len(manifest["files"]) == payload["one"]["file_count"]
        assert "src/msabp_opt/simulation/distributed/case_runner_tree.py" in {
            item["path"] for item in manifest["files"]
        }
        for item in manifest["files"]:
            local = ROOT / item["path"]
            assert local.is_file()
            payload_bytes = archive.read(item["path"])
            assert len(payload_bytes) == item["size_bytes"]
            assert hashlib.sha256(payload_bytes).hexdigest() == item["sha256"]


def test_uploaded_files_are_verified_before_system_helper() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    launch = source.split("$launch = @'", 1)[1].split("'@", 1)[0]
    for name in (
        "probe_cst_system_solve.py",
        "Invoke-CstSystemSolveProbeRemote.ps1",
        "source_bundle.zip",
    ):
        assert name in launch
    assert launch.count("Get-FileHash -Algorithm SHA256") >= 2
    assert launch.index("Get-FileHash -Algorithm SHA256") < launch.index("& $helper")
    assert "-SourceArchiveSha256" in launch


def test_remote_task_is_system_and_never_force_terminated_or_retried() -> None:
    source = REMOTE.read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("taskkill", "stop-process", "terminateprocess", ".stop()"):
        assert forbidden not in lowered
    assert "$settings.ExecutionTimeLimit = 'PT0S'" in source
    assert "$settings.AllowHardTerminate = $false" in source
    assert "Principal.UserId = 'SYSTEM'" in source
    assert source.index("LastTaskResult") < source.index("DeleteTask")
    assert "timed_out = $true" in source
    assert "residual_process_grace_seconds" in source
    assert "$residualWatch.Elapsed.TotalSeconds -lt 30" in source
    assert "no process was terminated" in source


def test_solve_probe_removes_control_mode_before_imports() -> None:
    source = PROBE.read_text(encoding="utf-8")
    main = source.index("def main(")
    main_pop = source.index('os.environ.pop("MSABP_CONTROL_MODE", None)', main)
    assert main_pop < source.index("report = run_probe(", main)
    run_probe = source.index("def run_probe(")
    run_pop = source.index('os.environ.pop("MSABP_CONTROL_MODE", None)', run_probe)
    assert run_pop < source.index("bindings = runtime_loader(source_root)", run_probe)
    assert "def _load_runtime" in source
    assert "import cst.interface" in source
    assert "from msabp_opt.simulation.distributed import case_runner_tree" in source
