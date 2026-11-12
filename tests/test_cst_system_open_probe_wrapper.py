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
    guard = source.index("if (-not $Execute)")
    assert guard < source.index("$preflight = @'")
    assert guard < source.index("& $pscp @pscpArguments")
    assert guard < source.index("Starting the approved SYSTEM CST")


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
