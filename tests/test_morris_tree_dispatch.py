import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.optimization import prepare_morris_tree as sampler
from scripts.simulation import run_morris_tree as launcher
from test_morris_tree_sampling import config, fake_geometry


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', fake_geometry)
    folder = tmp_path / 'campaign'
    plan = sampler.create_plan(config(tmp_path), folder)
    sampler.append_batch(folder, plan, [], [], 1)
    return folder


def test_dispatch_resume_identical_and_tamper_rejected(campaign):
    plan, dispatch, folder = launcher.prepare_dispatch(campaign, 'd001')
    before = (folder / 'sample.csv').read_bytes()
    assert launcher.prepare_dispatch(campaign, 'd001')[1] == dispatch
    assert (folder / 'sample.csv').read_bytes() == before
    command = launcher.princess_command(campaign, plan, dispatch, folder)
    assert command[command.index('--project')+1] == str(campaign / 'template.cst')
    assert command[command.index('--max-attempts')+1] == '3'
    assert '--device-project-relative-path' not in command
    assert len(dispatch['sample_ids']) == 36
    changed = dict(dispatch, run_id='some-other-run')
    (folder / 'dispatch.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Frozen dispatch'):
        launcher.prepare_dispatch(campaign, 'd001')


def test_retry_keeps_original_point_ids_and_omits_completed(campaign, monkeypatch):
    _, first, path = launcher.prepare_dispatch(campaign, 'd001')
    old = (path / 'dispatch.json').read_bytes()
    done = first['sample_ids'][:10]
    monkeypatch.setattr(launcher, 'collect_results', lambda *a: ({s: {} for s in done}, []))
    _, second, _ = launcher.prepare_dispatch(campaign, 'd002')
    assert second['sample_ids'] == first['sample_ids'][10:]
    assert second['run_id'] != first['run_id']
    assert (path / 'dispatch.json').read_bytes() == old


def test_other_nonterminal_dispatch_blocks_new_and_old_resume(campaign, monkeypatch):
    launcher.prepare_dispatch(campaign, 'd001')
    monkeypatch.setattr(launcher, 'active_tasks', lambda *a: 1)
    with pytest.raises(RuntimeError, match='running/pending'):
        launcher.prepare_dispatch(campaign, 'd002')


def test_read_only_state_handles_pending_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    path = tmp_path / 'simulations/runs/test/princess.sqlite3'
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE tasks(run_id TEXT,status TEXT); CREATE TABLE run_control(run_id TEXT,stop_requested INT);")
        db.execute("INSERT INTO tasks VALUES ('test','pending')")
        db.execute("INSERT INTO run_control VALUES ('test',0)")
    before = path.read_bytes()
    assert launcher.active_tasks('test') == 1
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        db.execute('UPDATE run_control SET stop_requested=1')
    assert launcher.active_tasks('test') == 0


def test_prepare_only_never_probes_or_launches(campaign, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Must not connect or solve')
    monkeypatch.setattr(launcher, 'probe_device', forbidden)
    monkeypatch.setattr(launcher, 'run_and_tee', forbidden)
    assert launcher.main(['--campaign', str(campaign), '--prepare-only']) == 0


def test_failed_preparation_does_not_block_later_batch(campaign):
    orphan = campaign / 'batches/batch_0002'
    orphan.mkdir()
    (orphan / 'candidates.jsonl').write_text('retained failed audit')
    plan, points, batches = sampler.common.load_campaign(campaign)
    batch = sampler.append_batch(campaign, plan, points, batches, 1)
    assert batch['batch_id'] == 3
    assert batch['points'][0]['trajectory_id'] == 2
    assert (orphan / 'candidates.jsonl').read_text() == 'retained failed audit'


def test_remote_probe_supplies_maids_conda_environment(monkeypatch):
    device = SimpleNamespace(id='remote', is_remote=True,
                             python_path=r'C:\User Space\miniforge3\envs\maid\python.exe',
                             repo_root=r'D:\Academic\Proj_MSABP_2')
    expected = {'scripts/geometry/manufacturing.py': 'a'*64}
    def remote(d, script, **kwargs):
        assert d is device
        assert "$env:CONDA_PREFIX='C:\\User Space\\miniforge3\\envs\\maid'" in script
        assert r'maid\Library\bin;' in script and r'maid\Scripts;' in script
        assert '+$env:PATH' in script and ' -B -u -c ' in script
        assert "Set-Location -LiteralPath 'D:\\Academic\\Proj_MSABP_2'" in script
        assert 'exit $LASTEXITCODE' in script
        assert kwargs['timeout'] == 60
        return SimpleNamespace(stdout=json.dumps(expected))
    monkeypatch.setattr(launcher, 'run_remote_powershell', remote)
    before = dict(os.environ)
    assert launcher.probe_device(device, expected) == expected
    assert dict(os.environ) == before


def test_local_probe_supplies_child_only_environment(monkeypatch):
    device = SimpleNamespace(id='local', is_remote=False,
                             python_path=str(Path('conda') / 'envs/maid/python.exe'), repo_root=str(Path.cwd()))
    expected = {'example.py': 'b'*64}
    def execute(command, **kwargs):
        prefix = Path(device.python_path).parent
        assert command[:4] == [device.python_path, '-B', '-u', '-c']
        assert kwargs['env']['CONDA_PREFIX'] == str(prefix)
        assert kwargs['env']['PATH'].startswith(str(prefix) + os.pathsep)
        assert str(prefix / 'Library/bin') in kwargs['env']['PATH']
        assert kwargs['cwd'] == device.repo_root
        return SimpleNamespace(stdout=json.dumps(expected))
    monkeypatch.setattr(launcher.subprocess, 'run', execute)
    before = dict(os.environ)
    assert launcher.probe_device(device, expected) == expected
    assert dict(os.environ) == before


def test_preflight_only_never_launches_or_prompts(campaign, monkeypatch):
    calls = []
    def probe(device, expected):
        calls.append(device.id)
        return expected
    def forbidden(*a, **kw):
        raise AssertionError('Preflight cannot launch or prompt for CST')
    monkeypatch.setattr(launcher, 'probe_device', probe)
    monkeypatch.setattr(launcher, 'run_and_tee', forbidden)
    monkeypatch.setattr('builtins.input', forbidden)
    assert launcher.main(['--campaign', str(campaign), '--preflight-only']) == 0
    assert calls == ['local', 'coconutg2', 'convallariag5']


def test_source_gate_still_rejects_different_code(monkeypatch):
    device = SimpleNamespace(id='remote', is_remote=True, python_path=r'C:\env\python.exe', repo_root=r'D:\repo')
    monkeypatch.setattr(launcher, 'run_remote_powershell',
                        lambda *a, **kw: SimpleNamespace(stdout=json.dumps({'a.py': 'wrong'})))
    with pytest.raises(RuntimeError, match='source mismatch'):
        launcher.probe_device(device, {'a.py': 'right'})
