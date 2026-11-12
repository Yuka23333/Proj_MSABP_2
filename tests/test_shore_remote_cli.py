"""CLI authorization boundaries; no live operation, secret or network access."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import socket
import subprocess

import pytest


@pytest.fixture
def entry(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("CLI validation must not contact devices or launch processes")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    path = Path(__file__).resolve().parents[1] / "scripts/simulation/shore_synthetic_princess.py"
    spec = spec_from_file_location("shore_synthetic_princess_cli_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", ["bind", "serve", "close", "serve-receipts", "status"])
def test_live_requires_explicit_approval_before_files_or_clients(entry, command, capsys):
    with pytest.raises(SystemExit) as exc:
        entry.main([command] + ([] if command == "status" else ["--preparation", "NEVER-READ.json"])
                   + (["--reason", "explicit cancellation"] if command == "close" else []))
    assert exc.value.code == 2
    assert "prior review" in capsys.readouterr().err


def test_no_action_does_not_default_to_live(entry):
    with pytest.raises(SystemExit) as exc:
        entry.main([])
    assert exc.value.code == 2


def test_modes_cannot_silently_replace_old_lab(entry, monkeypatch, tmp_path):
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-fake-v1")
    with pytest.raises(ValueError, match="existing controlled"):
        entry.main(["preflight", "--preparation", str(tmp_path / "NEVER-READ.json")])


def test_close_requires_additional_budget_abandonment_flag(entry, capsys):
    with pytest.raises(SystemExit):
        entry.main(["close", "--live-approved", "--preparation", "NEVER-READ.json",
                    "--reason", "explicit reason"])
    assert "--abandon-remaining-budget" in capsys.readouterr().err


@pytest.mark.parametrize("seconds", ["0", "3601"])
def test_receipt_window_must_be_bounded_before_loading_files(entry, seconds, capsys):
    with pytest.raises(SystemExit):
        entry.main(["serve-receipts", "--live-approved", "--seconds", seconds,
                    "--preparation", "NEVER-READ.json"])
    assert "Receipt window" in capsys.readouterr().err


def test_receipt_cli_routes_only_to_receipt_service(entry, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from msabp_opt.simulation.distributed import control_remote_prepare as preparation
    from msabp_opt.simulation.distributed import control_remote_controller as controller

    monkeypatch.setenv("MSABP_CONTROL_MODE", "")
    root = tmp_path / "job"
    (root / "deployment").mkdir(parents=True)
    for name in ("control.sqlite", "session.dpapi.json", "deployment/maid-runtime.json"):
        (root / name).write_text("test stub", encoding="utf-8")
    frozen = {"source_root": str(entry.ROOT), "output_root": str(root), "job_id": "job"}
    monkeypatch.setattr(preparation, "validate_preparation", lambda *_: frozen)
    monkeypatch.setattr(controller, "open_clients", lambda *_, **__: (
        SimpleNamespace(job={"phase": "RELEASED"}), None, "session", "public", {}))
    for method in ("bind_job", "serve_job", "close_job"):
        monkeypatch.setattr(controller, method, lambda *_, **__: pytest.fail("Not a work/lifecycle entry"))
    calls = []
    monkeypatch.setattr(controller, "serve_receipts", lambda *_, **kw: calls.append(kw["seconds"]))
    assert entry.main(["--shore-root", str(tmp_path), "serve-receipts", "--live-approved",
        "--seconds", "2", "--preparation", str(tmp_path / "mock.json")]) == 0
    assert calls == [2]


def test_bind_enters_serving_in_same_process_instead_of_leaving_lease_unattended(entry, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from msabp_opt.simulation.distributed import control_remote_prepare as preparation
    from msabp_opt.simulation.distributed import control_remote_controller as controller

    # Register the original environment even when absent: main sets the mode,
    # and it must not leak into subsequent legacy transport tests.
    monkeypatch.setenv("MSABP_CONTROL_MODE", "")
    frozen = {"source_root": str(entry.ROOT), "output_root": str(tmp_path / "job"), "job_id": "job"}
    monkeypatch.setattr(preparation, "validate_preparation", lambda *_: frozen)
    gate = SimpleNamespace(job={"phase": "ACTIVE"})
    monkeypatch.setattr(controller, "open_clients", lambda *_, **__: (gate, None, "session", "public", {}))
    calls = []
    monkeypatch.setattr(controller, "bind_job", lambda *_: calls.append("bind"))
    monkeypatch.setattr(controller, "serve_job", lambda *_, **__: calls.append("serve"))
    assert entry.main(["--shore-root", str(tmp_path), "bind", "--live-approved",
                       "--preparation", str(tmp_path / "mock.json")]) == 0
    assert calls == ["bind", "serve"]
