import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scripts.simulation import run_propagation_tree_7 as launch
from msabp_opt.simulation.distributed import case_runner
from msabp_opt.simulation.distributed import propagation_case_runner_tree as runner


def sample_row():
    body = {'exterior': [[-2, 0], [2, 0], [2, 4], [-2, 4]],
            'holes': [[[-.5, 1], [.5, 1], [.5, 2], [-.5, 2]]]}
    poly = Polygon(body['exterior'], body['holes'])
    payload = {'geometry': {'copper': [body], 'substrate_bounds': [-3, 0, 3, 5],
                            'reflector': {'exterior': [[-3, 0], [3, 0], [3, 5], [-3, 5]], 'holes': []}},
               'request': {}, 'source_case_id': 'test_source',
               'manufactured_copper_sha256': hashlib.sha256(unary_union([poly]).simplify(0).normalize().wkb).hexdigest()}
    return {'sample_id': 'test', 'simulation_mode': 'propagation_s21',
            'geometry_engine': 'saved_tree_pair_v1', 'tree_geometry_json': json.dumps(payload),
            'tree_geometry_sha256': runner.digest(payload), 'template_cst_sha256': 'a'*64}


def test_saved_geometry_dispatch_dry_run(tmp_path):
    result = case_runner.run_csv_row(sample_row(), project_path=tmp_path/'absent.cst',
                                     output_root=tmp_path/'out', dry_run=True)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest['geometry_engine'] == 'saved_tree_pair_v1'
    assert manifest['status'] == 'dry_run'
    assert result.s21_path is None


def test_saved_geometry_hash_must_match():
    row = sample_row()
    row['tree_geometry_sha256'] = 'bad'
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.validate_payload(row)


def test_pair_mirror_holes_and_history_scope():
    payload = runner.validate_payload(sample_row())
    original = Polygon(payload['geometry']['copper'][0]['exterior'], payload['geometry']['copper'][0]['holes'])
    mirrored = runner.second_shape(original)
    assert mirrored.area == original.area
    assert len(mirrored.interiors) == 1
    assert mirrored.bounds == (-2, 296, 2, 300)
    steps, expected = runner.history_steps(payload['geometry'])
    text = '\n'.join(code for _, code in steps)
    assert len(expected) == 6
    assert text.count('Solid.Subtract') == 2
    for forbidden in ('Connector', 'Port.', 'Boundary', 'Muscle', 'Kevin', 'Material.New'):
        assert forbidden not in text
    assert all(name.startswith(('a1_', 'a2_')) for name, *_ in expected)


def test_refuse_foreign_history_after_owned_suffix(tmp_path):
    model = SimpleNamespace(_GetHistory=lambda: {'list': [
        {'name': runner.PREFIX+'old', 'contents': '', 'error': False},
        {'name': 'manual change', 'contents': '', 'error': False}]})
    with pytest.raises(RuntimeError, match='Foreign/error'):
        runner.build_pair(SimpleNamespace(model3d=model), {}, tmp_path, 15)


def test_inplace_deployment_never_copies_model(tmp_path):
    runtime = object.__new__(launch.InPlacePrincessRuntime)
    runtime.preparation = SimpleNamespace(paths=SimpleNamespace(run_id='test-run', worklist_csv=tmp_path/'samples.csv'))
    runtime.device_project_relative_path = 'simulations/models/msa-bp-propagation.cst'
    device = SimpleNamespace(repo_root='D:\\Academic\\Proj_MSABP_2', id='local')
    pushes = []
    staged = []
    runtime._stage_runtime = lambda d, paths: staged.append(paths) or tmp_path/'runtime.json'
    runtime._push_file = lambda d, src, dst, **kw: pushes.append((src, dst))
    runtime._copy_device_file = lambda *a, **k: pytest.fail('No model copy allowed')
    paths, _ = runtime.deploy_device(device, force_project=True, launch_generation='retry-1')
    assert paths.project_path == 'D:\\Academic\\Proj_MSABP_2\\simulations\\models\\msa-bp-propagation.cst'
    assert staged[0].project_path == paths.project_path
    assert len(pushes) == 2
    assert all(not str(dst).endswith('.cst') for _, dst in pushes)


def test_current_config_selects_all_three_and_inplace():
    config = launch.common.read_json(launch.CONFIG)
    assert config['devices'] == ['local', 'coconutg2', 'convallariag5']
    assert config['candidate_ranks'] == [1, 2, 3, 8, 9, 11, 28]
    assert config['project_mode'] == 'in_place'
