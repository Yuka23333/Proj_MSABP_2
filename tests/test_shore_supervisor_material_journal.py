"""No live host: conservative file/ledger restart boundary in isolated temp roots."""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from test_shore_remote_worker import remote_worker  # noqa: F401
from test_shore_job_material import materials  # noqa: F401
from msabp_opt.simulation.distributed.control_supervisor import (
    MaintenanceRequired, MaterialJournal, initialize_state,
)


@pytest.fixture
def journal(materials, tmp_path):  # noqa: F811
    config = tmp_path / "config"
    config.mkdir()
    root = tmp_path / "supervisor"
    initialize_state(root, config, materials.profile)
    def reopen():
        return MaterialJournal(root, profile=materials.profile, bootstrap_data=materials.bootstrap_data)
    return root, config, reopen


@pytest.mark.parametrize("points", [1, 4])
def test_download_verify_commit_is_repeatable_and_not_a_launch(journal, materials, points):  # noqa: F811
    root, config, reopen = journal
    case = materials.make(points)
    job = case.binding["job_id"]
    j = reopen()
    assert j.status()["discovery_enabled"] is False
    j.download(job, case.raw, expected_binding=case.binding)
    assert j.status()["jobs"][0]["stage"] == "DOWNLOADED"
    with pytest.raises(MaintenanceRequired, match="verified"):
        j.commit(job)
    j = reopen()
    j.verify(job)
    j.download(job, case.raw, expected_binding=case.binding)
    target = j.commit(job)
    assert target == config / job
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert len(before) == 4
    assert reopen().commit(job) == target
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}
    assert j.status()["jobs"][0]["stage"] == "COMMITTED"
    assert case.runtime["api_token"] not in json.dumps(j.status())


def test_two_instances_compete_without_replacing_lock_or_state(journal):
    root, _, reopen = journal
    before = (root / "instance.lock").stat().st_ino
    with reopen().exclusive():
        with pytest.raises(MaintenanceRequired, match="owns"):
            with reopen().exclusive():
                pytest.fail("Second instance entered")
    assert (root / "instance.lock").stat().st_ino == before
    with reopen().exclusive():
        pass


@pytest.mark.parametrize("file", ["installed.json", "supervisor.sqlite", "instance.lock"])
def test_missing_original_installation_is_not_recreated(journal, file):
    root, _, reopen = journal
    (root / file).unlink()
    with pytest.raises((MaintenanceRequired, FileNotFoundError)):
        reopen()
    assert not (root / file).exists()


def test_reinitialization_refused_and_existing_job_always_blocks_next(journal, materials):  # noqa: F811
    root, config, reopen = journal
    with pytest.raises(FileExistsError):
        initialize_state(root, config, materials.profile)
    one, four = materials.make(1), materials.make(4)
    j = reopen()
    j.download(one.binding["job_id"], one.raw, expected_binding=one.binding)
    with pytest.raises(MaintenanceRequired, match="reconciled"):
        j.download(four.binding["job_id"], four.raw, expected_binding=four.binding)


def test_download_replay_must_be_byte_identical(journal, materials):  # noqa: F811
    case = materials.make()
    j = journal[2]()
    j.download(case.binding["job_id"], case.raw, expected_binding=case.binding)
    with pytest.raises(MaintenanceRequired):
        j.download(case.binding["job_id"], b" " + case.raw, expected_binding=case.binding)


def test_after_directory_rename_crash_reconciles_without_rewriting(journal, materials, monkeypatch):  # noqa: F811
    _, config, reopen = journal
    case = materials.make()
    job = case.binding["job_id"]
    j = reopen()
    j.download(job, case.raw, expected_binding=case.binding)
    j.verify(job)
    rename = os.rename
    def crash(a, b):
        rename(a, b)
        raise OSError("power loss after namespace commit")
    monkeypatch.setattr(os, "rename", crash)
    with pytest.raises(OSError):
        j.commit(job)
    assert j.status()["jobs"][0]["stage"] == "VERIFIED"
    before = {p.name: p.stat().st_mtime_ns for p in (config / job).iterdir()}
    target = reopen().commit(job)
    assert before == {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
    assert j.status()["jobs"][0]["stage"] == "COMMITTED"


@pytest.mark.parametrize("kind", ["missing", "tampered", "unknown", "directory"])
def test_committed_config_never_repaired_silently(journal, materials, kind):  # noqa: F811
    case = materials.make()
    j = journal[2]()
    job = case.binding["job_id"]
    j.download(job, case.raw, expected_binding=case.binding)
    j.verify(job)
    target = j.commit(job)
    if kind == "missing":
        (target / "samples.csv").unlink()
    elif kind == "tampered":
        (target / "samples.csv").write_bytes(b"tampered")
    elif kind == "unknown":
        (target / "extra.py").write_bytes(b"pass")
    else:
        (target / "samples.csv").unlink()
        (target / "samples.csv").mkdir()
    with pytest.raises(MaintenanceRequired):
        journal[2]().commit(job)


def test_orphan_config_is_not_adopted(journal, materials):  # noqa: F811
    _, config, reopen = journal
    case = materials.make()
    (config / case.binding["job_id"]).mkdir()
    with pytest.raises(MaintenanceRequired, match="Unjournaled"):
        reopen().download(case.binding["job_id"], case.raw, expected_binding=case.binding)
    assert reopen().status()["jobs"] == []


def test_verified_record_missing_is_maintenance(journal, materials):  # noqa: F811
    root, _, reopen = journal
    case = materials.make()
    job = case.binding["job_id"]
    j = reopen()
    j.download(job, case.raw, expected_binding=case.binding)
    j.verify(job)
    with sqlite3.connect(root / "supervisor.sqlite") as db:
        db.execute("UPDATE materials SET verified=NULL")
    with pytest.raises(MaintenanceRequired, match="disagree"):
        reopen().commit(job)


def test_journal_primary_key_cannot_relocate_material(journal, materials):  # noqa: F811
    root, _, reopen = journal
    case = materials.make()
    j = reopen()
    j.download(case.binding["job_id"], case.raw, expected_binding=case.binding)
    j.verify(case.binding["job_id"])
    with sqlite3.connect(root / "supervisor.sqlite") as db:
        db.execute("UPDATE materials SET job_id='another-job'")
    with pytest.raises(MaintenanceRequired, match="job ID"):
        reopen().commit("another-job")


def test_interrupted_partial_staging_file_remains_blocked(journal, materials):  # noqa: F811
    root, config, reopen = journal
    case = materials.make()
    job = case.binding["job_id"]
    j = reopen()
    j.download(job, case.raw, expected_binding=case.binding)
    j.verify(job)
    stage = root / "staging" / job
    stage.mkdir(parents=True)
    (stage / "samples.csv").write_bytes(b"sample_id,val")
    with pytest.raises(MaintenanceRequired, match="differ"):
        reopen().commit(job)
    assert not (config / job).exists()
    assert (stage / "samples.csv").read_bytes() == b"sample_id,val"


@pytest.mark.parametrize("damage", [None, "missing-db", "corrupt-db", "missing-config", "changed-config"])
def test_original_readonly_never_locks_writes_or_repairs(journal, materials, monkeypatch, damage):  # noqa: F811
    from test_shore_supervisor_maintenance import require_readonly_connections, tree_state
    root, _, reopen = journal
    case = materials.make()
    job = case.binding["job_id"]
    j = reopen()
    j.download(job, case.raw, expected_binding=case.binding)
    j.verify(job)
    target = j.commit(job)
    if damage == "missing-db":
        (root / "supervisor.sqlite").unlink()
    elif damage == "corrupt-db":
        (root / "supervisor.sqlite").write_bytes(b"damaged original")
    elif damage == "missing-config":
        (target / "samples.csv").unlink()
    elif damage == "changed-config":
        (target / "samples.csv").write_bytes(b"changed")
    before = tree_state(root.parent)
    opened = require_readonly_connections(monkeypatch)
    monkeypatch.setattr(MaterialJournal, "exclusive", lambda *_a: pytest.fail("Readonly handoff cannot lock"))
    if damage is None:
        assert j.original_readonly(job).files == j._verified(job)[0].files
        assert opened
    else:
        with pytest.raises((MaintenanceRequired, OSError)):
            j.original_readonly(job)
    assert tree_state(root.parent) == before
