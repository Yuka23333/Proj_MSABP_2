from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.simulation import probe_cst_system_open as probe


class FakeProject:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEnvironment:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False
        self.project = FakeProject()
        self.opened_path: Path | None = None

    def open_project(self, value: str) -> FakeProject:
        self.opened_path = Path(value)
        if self.fail:
            raise RuntimeError("synthetic open failure")
        return self.project

    def close(self) -> None:
        self.closed = True


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_opens_only_an_isolated_copy_and_closes_everything(tmp_path, monkeypatch):
    source = tmp_path / "source.cst"
    source.write_bytes(b"standalone-cst")
    workspace = tmp_path / "evidence"
    environment = FakeEnvironment()
    monkeypatch.setenv("MSABP_CONTROL_MODE", "shore-remote-synthetic-v1")

    report = probe.run_probe(
        source,
        workspace,
        expected_sha256=digest(source),
        environment_factory=lambda: environment,
        module_origin="synthetic/cst/interface.py",
        identity_fn=lambda: "nt authority\\system",
        session_id_fn=lambda: 0,
        process_snapshot_fn=lambda: [],
    )

    assert report["success"] is True
    assert report["identity"] == "nt authority\\system"
    assert report["session_id"] == 0
    assert report["control_mode_was_present"] is True
    assert "MSABP_CONTROL_MODE" not in probe.os.environ
    assert environment.project.closed and environment.closed
    assert environment.opened_path == workspace / "model" / source.name
    assert environment.opened_path.read_bytes() == source.read_bytes()
    assert report["source_sha256_before"] == report["source_sha256_after"]


def test_probe_records_open_failure_and_still_closes_environment(tmp_path):
    source = tmp_path / "source.cst"
    source.write_bytes(b"standalone-cst")
    environment = FakeEnvironment(fail=True)

    report = probe.run_probe(
        source,
        tmp_path / "evidence",
        expected_sha256=None,
        environment_factory=lambda: environment,
        module_origin=None,
        identity_fn=lambda: "system",
        session_id_fn=lambda: 0,
        process_snapshot_fn=lambda: [],
    )

    assert report["success"] is False
    assert report["opened"] is False
    assert report["error_type"] == "RuntimeError"
    assert environment.closed is True


def test_probe_refuses_stale_workspace_and_wrong_source_hash(tmp_path):
    source = tmp_path / "source.cst"
    source.write_bytes(b"standalone-cst")
    stale = tmp_path / "stale"
    stale.mkdir()

    with pytest.raises(FileExistsError):
        probe.run_probe(
            source, stale, expected_sha256=None, environment_factory=FakeEnvironment,
            module_origin=None,
        )
    with pytest.raises(ValueError, match="hash"):
        probe.run_probe(
            source, tmp_path / "new", expected_sha256="0" * 64,
            environment_factory=FakeEnvironment, module_origin=None,
        )
