"""Real-CST Princess CLI authorization and routing without live operations."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def entry(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("CST CLI validation must not contact devices or launch processes")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    path = Path(__file__).resolve().parents[1] / "scripts/simulation/shore_supervisor_cst_princess.py"
    spec = spec_from_file_location("shore_supervisor_cst_princess_cli_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", ["bind", "serve", "close", "serve-receipts", "status"])
def test_live_requires_explicit_approval_before_files_or_clients(entry, command, capsys):
    args = [command]
    if command != "status":
        args += ["--preparation", "NEVER-READ.json"]
    if command == "close":
        args += ["--reason", "explicit cancellation"]
    with pytest.raises(SystemExit) as exc:
        entry.main(args)
    assert exc.value.code == 2
    assert "prior review" in capsys.readouterr().err


def test_no_action_does_not_default_to_live(entry):
    with pytest.raises(SystemExit) as exc:
        entry.main([])
    assert exc.value.code == 2


def test_modes_cannot_silently_replace_another_controlled_process(entry, monkeypatch, tmp_path):
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-remote-synthetic-v1")
    with pytest.raises(ValueError, match="existing controlled"):
        entry.main(["--shore-root", str(tmp_path), "preflight",
                    "--preparation", str(tmp_path / "NEVER-READ.json")])


def test_close_requires_additional_budget_abandonment_flag(entry, capsys):
    with pytest.raises(SystemExit):
        entry.main(["close", "--live-approved", "--preparation", "NEVER-READ.json",
                    "--reason", "explicit reason"])
    assert "--abandon-remaining-budget" in capsys.readouterr().err


@pytest.mark.parametrize("seconds", ["0", "3601"])
def test_receipt_window_is_bounded_before_loading_files(entry, seconds, capsys):
    with pytest.raises(SystemExit):
        entry.main(["serve-receipts", "--live-approved", "--seconds", seconds,
                    "--preparation", "NEVER-READ.json"])
    assert "Receipt window" in capsys.readouterr().err


def test_bind_routes_to_original_controller_lifecycle(entry, monkeypatch, tmp_path):
    from msabp_opt.simulation.distributed import control_cst_prepare as preparation
    from msabp_opt.simulation.distributed import control_remote_controller as controller

    monkeypatch.setenv("MSABP_CONTROL_MODE", "")
    frozen = {
        "source_root": str(entry.ROOT),
        "output_root": str(tmp_path / "job"),
        "job_id": "job",
        "release_profile": {},
    }
    monkeypatch.setattr(preparation, "validate_cst_preparation", lambda *_: frozen)
    gate = SimpleNamespace(job={"phase": "ACTIVE"})
    monkeypatch.setattr(
        controller,
        "open_clients",
        lambda *_, **__: (gate, None, "session", "public", {}),
    )
    calls = []
    monkeypatch.setattr(controller, "bind_job", lambda *_: calls.append("bind"))
    monkeypatch.setattr(controller, "serve_job", lambda *_, **__: calls.append("serve") or {"phase": "ACTIVE"})
    monkeypatch.setattr(
        controller,
        "serve_receipts",
        lambda *_, **__: pytest.fail("Receipt window is release-only"),
    )
    assert entry.main([
        "--shore-root", str(tmp_path), "bind", "--live-approved",
        "--preparation", str(tmp_path / "mock.json"),
    ]) == 0
    assert calls == ["bind", "serve"]
