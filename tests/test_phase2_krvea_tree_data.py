from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from msabp_opt.optimization import phase2_krvea_tree_data as data
from scripts.optimization import prepare_morris_tree as prepare


def make_plan():
    config = data.common.read_json(data.ROOT / 'configs/optimization/morris_tree_35d.json')
    variables, baseline = prepare.variable_spec(config)
    return {
        'schema_version': 1, 'campaign_id': 'test-tree-seed', 'config': config,
        'variables': variables, 'baseline_request': baseline, 'dimension': len(variables),
        'num_levels': 4, 'delta': 2 / 3,
        'metrics': {'band_ghz': [3.1, 4.8], 'reference_area_mm2': 2720.2},
        'source_sha256': data.common.source_fingerprints(),
        'auxiliary_source_sha256': data.common.auxiliary_fingerprints(),
    }


@pytest.fixture
def space():
    return data.TreeInputSpace.from_plan(make_plan())


def test_35d_roundtrip_topology_and_vectorization(space):
    assert len(space.names) == 35
    assert sum(v['kind'] == 'abs' for v in space.variables) == 7
    unit = np.random.default_rng(9).uniform(size=(7, 35))
    raw = space.denormalize(unit)
    np.testing.assert_allclose(space.normalize(raw), unit, atol=1e-14)
    for row in raw:
        request = space.request_from_raw(row)
        np.testing.assert_array_equal(space.raw_from_request(request), row)
        assert request['build_options'] == space.baseline_request['build_options']
        assert request['tree'].keys() == space.baseline_request['tree'].keys()
    assert space.exact_normalized_area(space.normalize(space.nominal)) == pytest.approx(1)
    assert np.shape(space.exact_normalized_area(unit)) == (7,)
    assert not space.lower.flags.writeable


@pytest.mark.parametrize('change,message', [
    (lambda plan: plan['variables'][1].update(name='SLOT_MAIN_LENGTH'), 'Duplicate or inconsistent'),
    (lambda plan: plan['variables'][0].update(lower=0), 'physical domain'),
    (lambda plan: plan['variables'][7].update(upper=1.1), 'physical domain'),
    (lambda plan: plan['variables'][0].update(upper=float('nan')), 'Invalid bounds'),
    (lambda plan: plan['variables'].pop(), 'cover every'),
    (lambda plan: plan['config']['topology'].pop(), 'topology'),
    (lambda plan: plan['config']['build_options'].update(tip_clearance=2), 'build_options'),
])
def test_invalid_plan_rejected(change, message):
    plan = make_plan()
    change(plan)
    with pytest.raises(ValueError, match=message):
        data.TreeInputSpace.from_plan(plan)


def test_request_range_and_frozen_settings_rejected(space):
    with pytest.raises(ValueError, match='bounds'):
        space.denormalize(np.full(35, 1.01))
    with pytest.raises(ValueError, match='finite'):
        space.denormalize(np.full(35, np.nan))
    with pytest.raises(ValueError, match='dimension'):
        space.normalize(np.zeros(11))
    with pytest.raises(ValueError, match='One raw vector'):
        space.request_from_raw(np.array([space.nominal]))
    request = deepcopy(space.baseline_request)
    request['build_options']['tip_clearance'] = 2
    with pytest.raises(ValueError, match='build_options'):
        space.raw_from_request(request)
    request = deepcopy(space.baseline_request)
    del request['tree']['U1/L1']
    with pytest.raises(ValueError, match='topology'):
        space.raw_from_request(request)
    request = deepcopy(space.baseline_request)
    del request['params']['SLOT_MAIN_HEIGHT']
    with pytest.raises(ValueError, match='complete normalized'):
        space.raw_from_request(request)


def test_area_matches_full_tree_geometry_for_continuous_absolute_values(space):
    rng = np.random.default_rng(17)
    for _ in range(12):
        unit = space.normalize(space.nominal)
        unit[:7] = rng.uniform(size=7)
        request = space.request_from_raw(space.denormalize(unit))
        geometry = data.builder.prepare_geometry(request, quantum=space.coordinate_quantum_mm)
        x0, y0, x1, y1 = geometry['substrate_bounds']
        assert space.exact_normalized_area(unit) == pytest.approx(
            (x1 - x0) * (y1 - y0) / space.reference_area_mm2, rel=1e-13)
    contract = space.exact_area_contract()
    assert contract['type'] == 'msabp_tree_quantized_substrate_area_v1'
    assert contract['coordinate_quantum_mm'] == .01
    assert contract['parameter_names'] == list(space.names)


def make_campaign(folder):
    folder.mkdir()
    plan = make_plan()
    space = data.TreeInputSpace.from_plan(plan)
    raw = [space.nominal, space.nominal.copy(), space.nominal.copy()]
    raw[1][0] = space.lower[0]
    # Synthetic alias: dormant request K differs, but frozen geometry remains
    # the same. No extra independent observation may enter GP training.
    raw[2][-1] = .1
    points, geometry, values = [], [], {}
    for index in range(3):
        request = space.request_from_raw(raw[index])
        unit = space.normalize(raw[index])
        width, height = space.substrate_dimensions(unit)
        report = {'request': request, 'manufactured_copper_sha256': str(index % 2) * 64,
                  'substrate_bounds': [-float(width) / 2, 0, float(width) / 2, float(height)],
                  'reflector': {'exterior': [], 'holes': []}, 'quantum_mm': .01}
        identity = {key: report[key] for key in (
            'manufactured_copper_sha256', 'substrate_bounds', 'reflector', 'quantum_mm')}
        point = {'sample_id': f's{index}', 'simulation_sample_id': f's{index}' if index < 2 else 's0',
                 'trajectory_id': 1, 'step_index': index, 'unit': unit.tolist(), 'request': request,
                 'simulation_key': data.common.digest(identity),
                 'manufactured_copper_sha256': report['manufactured_copper_sha256'],
                 'substrate_area_mm2': float(width * height)}
        points.append(point)
        geometry.append(report)
    (folder / 'template.cst').write_bytes(b'test template only')
    plan['template_sha256'] = data.common.file_hash(folder / 'template.cst')
    plan['plan_sha256'] = data.common.digest(plan)
    data.common.write_json(folder / 'plan.json', plan)
    batch_folder = folder / 'batches' / 'batch_0001'
    batch_folder.mkdir(parents=True)
    for name in ('sample.csv', 'candidates.jsonl'):
        (batch_folder / name).write_text('test fixture\n', encoding='utf-8')
    batch = {'plan_sha256': plan['plan_sha256'], 'points': points,
             'csv_sha256': data.common.file_hash(batch_folder / 'sample.csv'),
             'candidates_sha256': data.common.file_hash(batch_folder / 'candidates.jsonl')}
    batch['batch_sha256'] = data.common.digest(batch)
    data.common.write_json(batch_folder / 'batch.json', batch)
    for point, geom in zip(points[:2], geometry[:2], strict=True):
        destination = folder / 'results' / ('case_' + point['sample_id'])
        destination.mkdir(parents=True)
        records = {}
        for key, name, content in (
            ('s11', 'S11.csv', '2 -10\n6 -10\n'),
            ('rad_eff', 'Rad_Eff.csv', '2 -1\n6 -1\n'),
            ('tot_eff', 'Tot_Eff.csv', '2 -2\n6 -2\n'),
            ('farfield_source', 'source.ffs', 'Synthetic FFS hash fixture; not parsed.\n'),
            ('geometry_tree', 'geometry_tree.json', json.dumps(geom)),
        ):
            path = destination / name
            path.write_text(content, encoding='utf-8')
            records[key] = {'path': name, 'sha256': data.common.file_hash(path),
                            'size_bytes': path.stat().st_size}
        manifest = {
            'schema_version': 1, 'status': 'completed', 'dry_run': False,
            'case_id': point['sample_id'], 'simulation_mode': 'antenna_tree',
            'parameters': point['request'], 'source_sha256': plan['source_sha256'],
            'tree_request_sha256': hashlib.sha256(json.dumps(point['request'], sort_keys=True).encode()).hexdigest(),
            'manufactured_copper_sha256': point['manufactured_copper_sha256'],
            'geometry_sha256': records['geometry_tree']['sha256'], 'artifacts': records,
        }
        data.common.write_json(destination / 'manifest.json', manifest)
    values, issues = data.morris.collect_results(plan, points, [folder / 'results'])
    assert not issues
    for value in values.values():
        manifest = data.common.read_json(Path(value['result_manifests'][0]))
        value.update(roi_theta_radiation_gain_linear=.1, roi_theta_radiation_gain_dbi=-10,
                     negative_roi_theta_radiation_gain_dbi=10,
                     ffs_sha256=manifest['artifacts']['farfield_source']['sha256'])
    report = {'status': 'complete', 'result_issues': [], 'campaign_id': plan['campaign_id'],
              'plan_sha256': plan['plan_sha256'], 'template_sha256': plan['template_sha256'],
              'metric_contract': deepcopy(data.ROI_CONTRACT), 'objective_names_minimize': list(data.REPORT_OBJECTIVES),
              'metric_source_sha256': {name: data.common.file_hash(data.ROOT / name) for name in data.METRIC_SOURCE_PATHS},
              'n_positions': 3, 'n_unique_ffs': 2, 'valid_point_data': values}
    report_path = folder / 'analysis_roi' / 'analysis.json'
    report_path.parent.mkdir()
    data.common.write_json(report_path, report)
    return report_path


def test_seed_reuses_physical_geometry_once_preserves_aliases_and_audits(tmp_path):
    folder = tmp_path / 'seed'
    make_campaign(folder)
    dataset = data.load_morris_seed(folder)
    assert dataset.x_raw.shape == (2, 35)
    assert dataset.objectives.shape == (2, 3)
    assert dataset.aliases == {'s0': 's0', 's1': 's1', 's2': 's0'}
    assert dataset.records[0]['aliases'] == ['s0', 's2']
    np.testing.assert_allclose(dataset.objectives[:, 0], 10 ** -.5)
    np.testing.assert_allclose(dataset.objectives[:, 1], 10)
    np.testing.assert_allclose(dataset.objectives[:, 2], dataset.exact_area(dataset.x_unit))
    assert dataset.provenance['artifact_audit'] == 'verified'


@pytest.mark.parametrize('change,message', [
    (lambda r: r.update(template_sha256='0' * 64), 'template_sha256'),
    (lambda r: r['metric_contract'].update(component='E_phi'), 'metric contract'),
    (lambda r: r.update(n_unique_ffs=3), 'independent-solve count'),
    (lambda r: r['valid_point_data'].pop('s2'), 'every planned'),
    (lambda r: r['valid_point_data']['s2'].update(worst_s11_linear=.9), 'Alias observation'),
    (lambda r: r['valid_point_data']['s0'].update(negative_roi_theta_radiation_gain_dbi=-10), 'sign values'),
    (lambda r: r.update(metric_source_sha256={}), 'source provenance'),
])
def test_seed_rejects_stale_or_inconsistent_reports(tmp_path, change, message):
    folder = tmp_path / 'seed'
    path = make_campaign(folder)
    report = data.common.read_json(path)
    change(report)
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        data.load_morris_seed(folder, audit_artifacts=False)


def test_seed_detects_corrupted_ffs_and_changed_geometry_source(tmp_path):
    folder = tmp_path / 'seed'
    make_campaign(folder)
    (folder / 'results' / 'case_s0' / 'source.ffs').write_text('corrupt', encoding='utf-8')
    with pytest.raises(ValueError, match='Artifact size mismatch'):
        data.load_morris_seed(folder)
    path = folder / 'results' / 'case_s1' / 'manifest.json'
    manifest = data.common.read_json(path)
    manifest['source_sha256']['model'] = '0' * 64
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='result audit failed'):
        data.load_morris_seed(folder)


def test_import_never_loads_cst_or_gp_libraries():
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join((str(data.ROOT), str(data.ROOT / 'src')))
    code = ('import sys; import msabp_opt.optimization.phase2_krvea_tree_data; '
            "assert not any(name.split('.')[0] in {'cst','torch','botorch','gpytorch'} for name in sys.modules)")
    completed = subprocess.run([sys.executable, '-c', code], env=environment, cwd=data.ROOT,
                               text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
