from copy import deepcopy
import csv
import json

import numpy as np
import pytest

from scripts.optimization import prepare_morris_tree as sampler


def config(tmp_path):
    cfg = sampler.common.read_json(sampler.CONFIG)
    template = tmp_path / 'fixture.cst'
    template.write_bytes(b'not a real CST; test never opens it')
    cfg['project'] = str(template)
    cfg['initial_trajectories'] = 1
    cfg['candidate_pool_factor'] = 1
    cfg['max_candidates_per_batch'] = 6
    return cfg


def fake_geometry(request, quantum=.01):
    normalized = sampler.builder.normalize_request(request)
    # Deterministic fixture identity; implementation is exercised separately.
    return {'request': normalized, 'substrate_bounds': [-33.5, 0, 33.5, 40.6],
            'manufactured_copper_sha256': sampler.common.digest(normalized),
            'reflector': {'exterior': [[0, 0], [1, 0], [1, 1]], 'holes': []},
            'quantum_mm': quantum, 'manufacturing': {'version': 'fixture'}}


def test_default_35d_topology_and_ranges():
    v, request = sampler.variable_spec(sampler.common.read_json(sampler.CONFIG))
    assert len(v) == 35
    assert sum(x['kind'] == 'abs' for x in v) == 7
    assert list(request['tree']) == ['U1', 'U1/L1', 'U1/R1', 'D1', 'D1/L1', 'D1/R1']
    assert all(x['lower'] == 0 and x['upper'] == 1 for x in v if x['kind'] == 'k')


@pytest.mark.parametrize('levels', [4, 6, 8])
def test_oat_grid_and_signed_steps(levels):
    x = sampler.trajectory(35, levels, np.random.default_rng(4))
    dx = np.diff(x, axis=0)
    assert x.shape == (36, 35)
    assert np.all((x >= 0) & (x <= 1))
    assert np.allclose(x * (levels-1), np.round(x * (levels-1)))
    assert np.all(np.count_nonzero(dx, axis=1) == 1)
    assert np.all(np.count_nonzero(dx, axis=0) == 1)
    assert np.allclose(abs(dx[dx != 0]), levels / (2 * (levels-1)))
    assert np.array_equal(x, sampler.trajectory(35, levels, np.random.default_rng(4)))


def test_actual_geometry_full_track():
    cfg = sampler.common.read_json(sampler.CONFIG)
    v, baseline = sampler.variable_spec(cfg)
    for x in sampler.trajectory(35, 4, np.random.default_rng(1)):
        raw = sampler.request_from_unit({'variables': v, 'baseline_request': baseline}, x)
        g = sampler.builder.prepare_geometry(raw)
        assert g['manufacturing']['version'] == 'tree-manufacturing-v2'


def test_append_is_incremental_and_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', fake_geometry)
    folder = tmp_path / 'campaign'
    plan = sampler.create_plan(config(tmp_path), folder)
    first = sampler.append_batch(folder, plan, [], [], 1)
    old_bytes = (folder / 'batches/batch_0001/batch.json').read_bytes()
    plan_bytes = (folder / 'plan.json').read_bytes()
    plan, points, batches = sampler.common.load_campaign(folder, check_sources=True)
    second = sampler.append_batch(folder, plan, points, batches, 1)
    assert (folder / 'batches/batch_0001/batch.json').read_bytes() == old_bytes
    assert (folder / 'plan.json').read_bytes() == plan_bytes
    assert first['points'][0]['trajectory_id'] == 1
    assert second['points'][0]['trajectory_id'] == 2
    assert len(sampler.common.load_campaign(folder)[1]) == 72
    (folder / 'template.cst').write_bytes(b'changed')
    with pytest.raises(ValueError, match='template changed'):
        sampler.common.load_campaign(folder)


def test_reject_whole_track_not_replace_one_point(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', fake_geometry)
    folder = tmp_path / 'campaign'
    plan = sampler.create_plan(config(tmp_path), folder)
    calls = 0
    def sometimes_bad(request, quantum=.01):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sampler.builder.model.manufacturing.ManufacturingError('test rejection', {})
        return fake_geometry(request, quantum)
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', sometimes_bad)
    batch = sampler.append_batch(folder, plan, [], [], 1)
    assert batch['candidate_count'] == 2 and calls == 72
    rows = [json.loads(line) for line in (folder / 'batches/batch_0001/candidates.jsonl').read_text().splitlines()]
    assert not rows[0]['accepted'] and len(rows[0]['labels']) == 36
    assert np.allclose([p['unit'] for p in batch['points']], rows[1]['unit'])


def test_exact_model_reuse_keeps_all_trajectory_positions(tmp_path, monkeypatch):
    def collapsed(request, quantum=.01):
        g = fake_geometry(request, quantum)
        g['manufactured_copper_sha256'] = 'identical copper'
        return g
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', collapsed)
    folder = tmp_path / 'campaign'
    plan = sampler.create_plan(config(tmp_path), folder)
    batch = sampler.append_batch(folder, plan, [], [], 1)
    assert len(batch['points']) == 36
    assert len({p['simulation_sample_id'] for p in batch['points']}) == 1
    with (folder / 'batches/batch_0001/sample.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 1
    g = collapsed(plan['baseline_request'])
    other = deepcopy(g)
    other['substrate_bounds'][2] += 1
    assert sampler.simulation_key(g) != sampler.simulation_key(other)


def test_programming_fault_is_not_manufacturing_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', fake_geometry)
    folder = tmp_path / 'campaign'
    plan = sampler.create_plan(config(tmp_path), folder)
    def broken(*args, **kwargs):
        raise ValueError('unexpected bug')
    monkeypatch.setattr(sampler.builder, 'prepare_geometry', broken)
    with pytest.raises(ValueError, match='unexpected bug'):
        sampler.append_batch(folder, plan, [], [], 1)
    assert not (folder / 'batches/batch_0001/batch.json').exists()


def test_35_to_29_is_topology_only():
    cfg = sampler.common.read_json(sampler.CONFIG)
    cfg['topology'] = cfg['topology'][:-2]
    assert len(sampler.variable_spec(cfg)[0]) == 29
