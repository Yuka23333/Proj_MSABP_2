"""Opt-in supervisor preparation/Princess bridge, no real authority or process."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by name
from __future__ import annotations

import copy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_shore_remote_prepare import (
    prepare_args, remote_worker,  # noqa: F401
    test_prepared_public_inputs_bind_to_strict_remote_runtime_without_network as check_binding,
)
from msabp_opt.simulation.distributed.control_release import (
    RELEASE_SOURCE_FILES, build_release_manifest, public_profile,
)
from msabp_opt.simulation.distributed.control_remote_prepare import prepare_synthetic_job, validate_preparation
from msabp_opt.simulation.distributed.control_supervisor_controller import publish_original_wake


@pytest.fixture
def supervisor_args(prepare_args):
    source = prepare_args["source_root"]
    for name in RELEASE_SOURCE_FILES:
        path = source / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Fixed offline supervisor fixture\n", encoding="utf-8")
    manifest = build_release_manifest(source, prepare_args["shore_source_root"])
    return {**prepare_args, "release_profile": public_profile(manifest, prepare_args["bootstrap_sha256"])}


@pytest.mark.parametrize("points", [1, 4])
def test_profile_frozen_into_job_and_runtime_paths(supervisor_args, points):
    args = {**supervisor_args, "points": points}
    prep = prepare_synthetic_job(**args)
    profile = args["release_profile"]
    manifest = json.loads(Path(prep["paths"]["job_manifest"]).read_bytes())
    assert manifest["supervisor_release_id"] == profile["release_id"]
    assert profile["release_id"] in prep["remote_paths"]["source_root"]
    assert prep["remote_paths"]["python_executable"] == (
        r"C:\ProgramData\ShorePrincess\runtime-msabp\venv\Scripts\python.exe")
    assert set(prep["source_hashes"]) == set(RELEASE_SOURCE_FILES)
    assert prep == validate_preparation(prep["preparation_path"])
    assert (Path(prep["output_root"]) / "job-http-nonces.sqlite").is_file()


def test_profile_runtime_constructed_before_any_wake_signing(supervisor_args, remote_worker):
    check_binding(supervisor_args, remote_worker)


def test_deleted_nonce_ledger_not_reinitialized(supervisor_args):
    prep = prepare_synthetic_job(**supervisor_args)
    root = Path(prep["output_root"])
    before = (root / "preparation.json").read_bytes()
    (root / "job-http-nonces.sqlite").unlink()
    with pytest.raises(ValueError, match="nonce ledger is missing"):
        validate_preparation(root)
    assert not (root / "job-http-nonces.sqlite").exists()
    assert (root / "preparation.json").read_bytes() == before


def test_changed_supervisor_code_rejected(supervisor_args):
    prep = prepare_synthetic_job(**supervisor_args)
    (Path(prep["source_root"]) / RELEASE_SOURCE_FILES[-1]).write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile/source/bootstrap changed"):
        validate_preparation(prep["preparation_path"])


def test_mismatched_profile_rejected_before_output(supervisor_args):
    args = copy.deepcopy(supervisor_args)
    args["release_profile"]["bootstrap_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Approved supervisor profile"):
        prepare_synthetic_job(**args)
    assert not args["output_root"].exists()


@pytest.mark.parametrize("phase", ["DRAINING", "PAUSED_BOUND", "RELEASED"])
def test_no_new_wake_during_drain_or_release(phase):
    queue = Mock()
    publish_original_wake(SimpleNamespace(job={"phase": phase}), queue)
    queue._assignment.assert_not_called()


def test_missing_f_failure_propagates_no_session_fallback():
    queue = Mock()
    queue._assignment.side_effect = PermissionError("F absent")
    with pytest.raises(PermissionError):
        publish_original_wake(SimpleNamespace(job={"phase": "ACTIVE"}), queue)
    queue._assignment.assert_called_once_with()


def test_legacy_preparation_has_no_job_download_service():
    from msabp_opt.simulation.distributed.control_supervisor_controller import make_job_service
    assert make_job_service({}, None, None, None) is None


@pytest.mark.parametrize("points", [1, 4])
def test_supervisor_cli_preparation_pins_dedicated_runtime_without_live_access(
        supervisor_args, monkeypatch, tmp_path, capsys, points):
    # prepare_args already forbids network and process creation. Exercise the
    # same explicit supervisor mode used by the new CLI, using only fixtures.
    source = Path(__file__).resolve().parents[1] / "scripts/simulation/shore_synthetic_princess.py"
    spec = spec_from_file_location("supervisor_preparation_cli", source)
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(entry, "ROOT", supervisor_args["source_root"])
    monkeypatch.setenv("MSABP_CONTROL_MODE", "")
    public = tmp_path / "approved-public-profile.json"
    public.write_text(json.dumps(supervisor_args["release_profile"]), encoding="utf-8")
    common = ["--shore-root", str(supervisor_args["shore_source_root"])]
    assert entry.main([*common, "prepare",
        "--output", str(supervisor_args["output_root"]),
        "--job-id", supervisor_args["job_id"], "--run-id", supervisor_args["run_id"],
        "--points", str(points), "--release-profile", str(public),
        "--bootstrap", str(supervisor_args["bootstrap_path"]),
        "--bootstrap-sha256", supervisor_args["bootstrap_sha256"]], supervisor=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["offline"] is True and report["network"] is report["keys_read"] is False
    prep = validate_preparation(supervisor_args["output_root"])
    assert prep["remote_paths"]["python_executable"] == (
        r"C:\ProgramData\ShorePrincess\runtime-msabp\venv\Scripts\python.exe")
    assert entry.main([*common, "preflight", "--preparation", prep["preparation_path"]],
                      supervisor=True) == 0
    # Changing entrypoint is not a legacy runtime fallback.
    with pytest.raises(ValueError, match="matching legacy or supervisor"):
        entry.main([*common, "preflight", "--preparation", prep["preparation_path"]])
