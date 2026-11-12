"""Remote synthetic is explicit and cannot reopen legacy execution paths."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from msabp_opt.simulation.distributed import control_policy, transport
from msabp_opt.simulation.distributed.maid import Maid, MaidRuntimeConfig


@pytest.mark.parametrize("mode", [control_policy.LAB_MODE, control_policy.REMOTE_MODE])
def test_controlled_modes_keep_legacy_execution_denied(monkeypatch, mode):
    monkeypatch.setenv(control_policy.MODE_ENV, mode)
    assert control_policy.controlled_mode()
    assert control_policy.lab_mode() == (mode == control_policy.LAB_MODE)
    runner = Mock(side_effect=AssertionError("Must never launch"))
    with pytest.raises(RuntimeError, match="refuses"):
        transport._run_checked(["ssh", "unused"], runner=runner, cwd=None, timeout=1, action="probe")
    with pytest.raises(RuntimeError, match="refuses"):
        transport.push_file_atomic(None, "unused", "unused")
    with pytest.raises(RuntimeError, match="refuses"):
        transport.launch_maid(None)
    runner.assert_not_called()


def test_remote_mode_still_requires_injected_fake_gate_session_and_executor(monkeypatch, tmp_path):
    monkeypatch.setenv(control_policy.MODE_ENV, control_policy.REMOTE_MODE)
    config = MaidRuntimeConfig.from_mapping({
        "schema_version": 1, "protocol_mode": control_policy.REMOTE_MODE,
        "run_id": "synthetic-one", "worker_id": "coconutg2-maid", "princess_url": "https://100.99.182.30:18767",
        "api_token": "unused", "csv_path": str(tmp_path / "cases.csv"), "csv_sha256": "a" * 64,
        "project_path": str(tmp_path / "never-open.cst"), "output_root": str(tmp_path / "output"), "dry_run": True,
    })
    with pytest.raises(ValueError, match="explicitly injected"):
        Maid(config)
    monkeypatch.setenv(control_policy.MODE_ENV, control_policy.LAB_MODE)
    with pytest.raises(ValueError, match="different Shore mode"):
        Maid(config)


def test_transport_only_imports_do_not_load_solver_or_geometry_packages():
    root = Path(__file__).resolve().parents[1]
    code = """
import builtins
original = builtins.__import__
blocked = {'cst','numpy','pandas','shapely','scipy','matplotlib','torch'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked or name.startswith('scripts.automation'):
        raise AssertionError('unexpected scientific/solver dependency: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from msabp_opt.simulation.distributed.maid import Maid
from msabp_opt.simulation.distributed.runtime import PrincessRuntime
from msabp_opt.simulation.distributed import case_runner
assert callable(case_runner.run_csv_row)
assert case_runner.RAD_EFF_FILENAME == 'Rad_Eff.csv'
assert case_runner.TOT_EFF_FILENAME == 'Tot_Eff.csv'
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            env={**os.environ, "PYTHONPATH": str(root / "src")},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
