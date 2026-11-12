from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "simulation" / "Invoke-CstSystemOpenProbe.ps1"
REMOTE = ROOT / "scripts" / "simulation" / "Invoke-CstSystemOpenProbeRemote.ps1"
PROBE = ROOT / "scripts" / "simulation" / "probe_cst_system_open.py"


@pytest.mark.parametrize("path", [WRAPPER, REMOTE])
def test_powershell_probe_scripts_parse(path: Path) -> None:
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


def test_wrapper_uploads_sources_instead_of_embedding_python() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "__PROBE_BASE64__" not in source
    assert "ReadAllBytes" not in source
    assert "PSCP" in source
    for option in ("'-batch'", "'-sftp'", "'-4'", "'-P'", "'-noagent'", "'-i'", "'-hostkey'"):
        assert option in source
    assert "Get-AuthenticodeSignature" in source
    assert "Get-FileHash -Algorithm SHA256" in source
    assert "Refusing an oversized remote command" in source


def test_plan_guard_precedes_every_remote_write_or_cst_start() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    guard = source.index("if (-not $Execute -and -not $PreflightOnly)")
    assert guard < source.index("$audit = @'")
    assert guard < source.index("$prepare = @'")
    assert guard < source.index("Invoke-NativeCaptured $pscp")
    assert guard < source.index("Starting the approved SYSTEM CST")


def test_read_only_preflight_is_separate_from_remote_staging() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    audit = source.split("$audit = @'", 1)[1].split("'@", 1)[0]
    prepare = source.split("$prepare = @'", 1)[1].split("'@", 1)[0]
    assert "New-Item" not in audit
    assert "RegisterTask" not in audit
    assert "New-Item" in prepare
    assert "if ($PreflightOnly)" in source
    assert "READ-ONLY PREFLIGHT COMPLETE" in source


def test_actual_remote_commands_use_direct_pinned_plink_after_control_gate() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "'-Command', 'exit 0'" in source
    assert "Invoke-ShoreCommand $plink $identity $node $fingerprint $audit" in source
    assert "Invoke-ShoreCommand $plink $identity $node $fingerprint $launch" in source
    assert "'-no-trivial-auth'" in source
    assert "Get-AuthenticodeSignature -LiteralPath $binary" in source


def test_only_exact_cst_infrastructure_names_and_paths_are_allowed() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    remote = REMOTE.read_text(encoding="utf-8")
    for source in (wrapper, remote):
        assert "CSTDCSolverServer_AMD64.exe" in source
        assert "CSTDCMainController_AMD64.exe" in source
        assert "License Manager\\cstd.exe" in source
        assert "$knownInfrastructure.ContainsKey([string]$_.Name)" in source
        assert "[string]$_.ExecutablePath -ieq $knownInfrastructure[[string]$_.Name]" in source


def test_native_capture_preserves_multiline_stderr_and_exit_code() -> None:
    executable = shutil.which("powershell.exe")
    if not executable:
        pytest.skip("Windows PowerShell is unavailable")
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("function Invoke-NativeCaptured")
    end = source.index("function Assert-ShoreOnline", start)
    function_source = source[start:end]
    child = (
        "[Console]::Error.WriteLine('REMOTE_DETAIL_1');"
        "[Console]::Error.WriteLine('REMOTE_DETAIL_2'); exit 23"
    )
    script = (
        "$ErrorActionPreference='Stop'\n"
        + function_source
        + "\ntry { Invoke-NativeCaptured 'powershell.exe' "
        + "@('-NoProfile','-NonInteractive','-Command',\""
        + child
        + "\") 'Synthetic remote' } catch { "
        + "Write-Output ('CAUGHT:' + $_.Exception.Message) }"
    )
    result = subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REMOTE_DETAIL_1" in result.stdout
    assert "REMOTE_DETAIL_2" in result.stdout
    assert "CAUGHT:Synthetic remote failed with exit code 23." in result.stdout


def test_remote_task_never_forces_termination_and_captures_result_first() -> None:
    source = REMOTE.read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("taskkill", "stop-process", "terminateprocess", ".stop()"):
        assert forbidden not in lowered
    assert "$settings.ExecutionTimeLimit = 'PT0S'" in source
    assert "$settings.AllowHardTerminate = $false" in source
    assert "Principal.UserId = 'SYSTEM'" in source
    assert source.index("LastTaskResult") < source.index("DeleteTask")
    assert "if ($instanceCount -eq 0)" in source
    assert "$taskState -notin @(2, 4) -and $hasRun" in source
    assert "timed_out = $true" in source
    assert "The running task and all evidence were preserved" in source


def test_control_mode_is_removed_before_cst_import() -> None:
    source = PROBE.read_text(encoding="utf-8")
    main = source.index("def main(")
    pop = source.index('os.environ.pop("MSABP_CONTROL_MODE", None)', main)
    cst_import = source.index("import cst.interface", main)
    assert pop < cst_import
