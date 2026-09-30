import json

import numpy as np
import pytest

from scripts.optimization import prepare_krvea_tree as prep


class Space:
    names = ('x', 'y')

    def normalize(self, raw):
        return np.asarray(raw)

    denormalize = normalize

    def request_from_raw(self, raw):
        return {'x': float(raw[0]), 'y': float(raw[1])}

    def raw_from_request(self, request):
        return np.array([request['x'], request['y']])


def fake_geometry(request, quantum):
    return {'request': request, 'substrate_bounds': [0, 0, 1, 1],
            'manufactured_copper_sha256': prep.common.digest(request),
            'reflector': {'same': True}, 'quantum_mm': quantum}


def test_design_disjoint_reproducible_and_contains_reference(monkeypatch):
    monkeypatch.setattr(prep.builder, 'prepare_geometry', fake_geometry)
    frozen = {'config': {'coordinate_quantum_mm': .01}, 'baseline_request': {'x': .5, 'y': .5}}
    design = {'seed': 23, 'training_count': 8, 'holdout_count': 4,
              'include_reference': True, 'max_candidates': 100}
    rows, audit = prep.generate_design(Space(), frozen, [], design)
    assert (rows, audit) == prep.generate_design(Space(), frozen, [], design)
    assert len(rows) == len({r['simulation_key'] for r in rows}) == 12
    assert [r['split'] for r in rows].count('train') == 8
    assert [r['split'] for r in rows].count('holdout') == 4
    assert rows[0]['request'] == {'x': .5, 'y': .5}
    excluded = [{'simulation_key': rows[0]['simulation_key']}]
    next_rows, next_audit = prep.generate_design(Space(), frozen, excluded, design)
    assert len(next_rows) == 12
    assert rows[0]['simulation_key'] not in {r['simulation_key'] for r in next_rows}
    assert next_audit[0]['status'] == 'duplicate_geometry'


def test_unexpected_geometry_errors_are_not_silently_rejected(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('unexpected algorithm bug')
    monkeypatch.setattr(prep.builder, 'prepare_geometry', broken)
    with pytest.raises(ValueError, match='unexpected algorithm bug'):
        prep.generate_design(Space(), {'config': {'coordinate_quantum_mm': .01},
                                      'baseline_request': {'x': .5, 'y': .5}}, [],
                             {'seed': 1, 'training_count': 1, 'holdout_count': 1,
                              'include_reference': True, 'max_candidates': 10})


def test_worklist_only_tree_mode_and_never_overwrites(tmp_path):
    path = tmp_path / 'work.csv'
    prep.write_worklist(path, [{'sample_id': 'a', 'request': {'tree': {}}}])
    import csv
    with path.open(newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['simulation_mode'] == 'antenna_tree'
    assert json.loads(row['tree_json']) == {'tree': {}}
    with pytest.raises(FileExistsError):
        prep.write_worklist(path, [])


def test_dispatch_defaults_to_print_only(monkeypatch, tmp_path):
    config = {'any': 'value'}
    monkeypatch.setattr(prep, 'load_config', lambda _: config)
    monkeypatch.setattr(prep, 'load_prepared', lambda _: (tmp_path, {}))
    monkeypatch.setattr(prep, 'princess_command', lambda *a: ['python', 'princess.py'])
    def forbidden(*a, **k):
        pytest.fail('print-only mode must not probe or launch')
    monkeypatch.setattr(prep, 'preflight_devices', forbidden)
    monkeypatch.setattr(prep.subprocess, 'call', forbidden)
    assert prep.main(['--dispatch', 'train']) == 0
