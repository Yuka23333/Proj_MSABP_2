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


def test_phantom_exception_is_exact_and_material_checked_before_legacy_guard():
    code = runner.infrastructure_check_vba()
    phantom_branch, remainder = code.split('        Else\n', 1)
    assert 'If n = "component1:solid1" Then' in phantom_branch
    assert 'If Solid.GetMaterialNameForShape(n) <> "Kevin" Then Err.Raise' in phantom_branch
    assert phantom_branch.index('Unexpected phantom material') < phantom_branch.index('hasKevin = True')
    assert 'Unowned old antenna geometry' not in phantom_branch
    assert 'hasKevin = True' not in remainder
    # Residual legacy antennas (including the second component) stay blocked;
    # use string lengths rather than brittle hand-counted prefix lengths.
    for prefix in ('component1', 'msabp_tree:', runner.OWNER + ':'):
        assert f'Left(n, Len("{prefix}")) = "{prefix}"' in remainder
    assert 'If Not hasKevin Then Err.Raise' in remainder
    for forbidden in ('Solid.Delete', 'Solid.ChangeMaterial', 'Component.Delete', '.Create'):
        assert forbidden not in code


def test_build_pair_uses_phantom_aware_query_before_adding_geometry(tmp_path, monkeypatch):
    baseline = [{'name': 'phantom and connectors', 'contents': 'baseline', 'error': False}]
    history = list(baseline)
    events = []
    def add(name, code, **kwargs):
        events.append('add')
        history.append({'name': name, 'contents': code, 'error': False})
    def query(code, **kwargs):
        events.append('query')
        if len(events) == 1:
            assert code == runner.infrastructure_check_vba()
    model = SimpleNamespace(_GetHistory=lambda: {'list': list(history)}, add_to_history=add)
    project = SimpleNamespace(model3d=model, schematic=SimpleNamespace(execute_vba_code=query))
    monkeypatch.setattr(runner, 'history_steps', lambda geometry: ([('test', 'test geometry')], []))
    report = runner.build_pair(project, {'geometry': {}}, tmp_path, 15)
    assert events == ['query', 'add', 'query']
    assert history[0] == baseline[0]
    assert report['phantom_material'] == 'Kevin'


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
    assert config['export_e_fields'] is False
    assert config['run_id'].endswith('-003')


@pytest.mark.parametrize('fields_enabled', [False, True])
def test_tree_runner_optional_fields(tmp_path, monkeypatch, fields_enabled):
    row = sample_row()
    row['export_e_fields'] = str(fields_enabled)
    events = []
    items = [r'Ports\port1', r'Ports\port2']
    if fields_enabled:
        items += [rf'Field Monitors\e-field (f={f})' for f in (3.1, 4, 4.8)]
    model = SimpleNamespace(
        get_tree_items=lambda **kw: events.append('inspect') or items,
        get_active_solver_name=lambda **kw: 'HF Time Domain',
        is_solver_running=lambda **kw: False,
        run_solver=lambda **kw: events.append('solve'))
    project = SimpleNamespace(model3d=model)
    monkeypatch.setattr(runner.legacy_runner.cst_run_and_export_s11, 'clear_results_on_project',
                        lambda *a, **kw: events.append('clear'))
    monkeypatch.setattr(runner, 'build_pair', lambda *a: events.append('build') or {})
    def export(_path, folder, **kwargs):
        assert kwargs['export_e_fields'] == fields_enabled
        events.append('export')
        folder.mkdir(parents=True)
        (folder/'S21_complex.csv').write_text('test S21')
        return SimpleNamespace(e_field_monitor_count=3 if fields_enabled else 0)
    monkeypatch.setattr(runner.legacy_runner.export_propagation_results, 'export_propagation_results', export)
    result = case_runner.run_csv_row(row, project_path=tmp_path/'model.cst',
                                    output_root=tmp_path/'out', local_artifact_root=tmp_path/'local', project=project)
    assert events == ['inspect', 'clear', 'build', 'inspect', 'solve', 'export']
    manifest = json.loads(result.manifest_path.read_text())
    assert set(manifest['artifacts']) == {'s21'}
    assert manifest['export_e_fields'] == fields_enabled
    assert manifest['local_only']['retained_on_maid'] == fields_enabled
    assert (result.local_e_field_directory is not None) == fields_enabled


def test_s21_only_still_checks_ports_solver_and_busy_state():
    items = [r'Ports\port1', r'Ports\port2']
    model = SimpleNamespace(get_tree_items=lambda **kw: items,
                            get_active_solver_name=lambda **kw: 'HF Time Domain',
                            is_solver_running=lambda **kw: False)
    project = SimpleNamespace(model3d=model)
    inspect = runner.legacy_runner.inspect_propagation_infrastructure
    with pytest.raises(RuntimeError, match='E-field monitors'):
        inspect(project, 60)
    assert inspect(project, 60, require_e_fields=False)['e_field_monitors'] == ()
    items.pop()
    with pytest.raises(RuntimeError, match='missing manually configured ports'):
        inspect(project, 60, require_e_fields=False)
    items.append(r'Ports\port2')
    model.get_active_solver_name = lambda **kw: 'HF Frequency Domain'
    with pytest.raises(RuntimeError, match='unexpected active CST solver'):
        inspect(project, 60, require_e_fields=False)
    model.get_active_solver_name = lambda **kw: 'HF Time Domain'
    model.is_solver_running = lambda **kw: True
    with pytest.raises(RuntimeError, match='already running'):
        inspect(project, 60, require_e_fields=False)


def test_export_fields_config_requires_boolean():
    with pytest.raises(ValueError, match='JSON boolean'):
        launch.prepare({'export_e_fields': 'false'})


def test_preflight_command_fits_windows_limit(monkeypatch):
    import base64
    config = launch.common.read_json(launch.CONFIG)
    bundle = {'files': [], 'directories': []}
    manifest = {'config': config, 'bundle': bundle,
                'sources': {p: 'a'*64 for p in launch.SOURCE_FILES}}
    device = SimpleNamespace(repo_root='D:\\Academic\\Proj_MSABP_2', id='coconutg2',
                             launch_mode=launch.LaunchMode.BELL)
    def remote(_device, script, **kwargs):
        assert len(base64.b64encode(script.encode('utf-16-le'))) < 7500
        return SimpleNamespace(stdout=json.dumps(bundle))
    monkeypatch.setattr(launch, 'run_remote_powershell', remote)
    launch.preflight([device], manifest)


def test_inplace_ignores_results_but_not_model_inputs():
    before = {'files': [{'path': 'project.cst', 'sha256': 'a'},
                        {'path': 'project/Model/design.mod', 'sha256': 'b'},
                        {'path': 'project/Result/Storage.sdb', 'sha256': 'c'}],
              'directories': ['project', 'project/Model', 'project/Result']}
    after = json.loads(json.dumps(before))
    after['files'][-1]['sha256'] = 'result_changed'
    assert launch.comparable_bundle(before, in_place=True) == launch.comparable_bundle(after, in_place=True)
    assert launch.comparable_bundle(before, in_place=False) != launch.comparable_bundle(after, in_place=False)
    after['files'][1]['sha256'] = 'input_changed'
    assert launch.comparable_bundle(before, in_place=True) != launch.comparable_bundle(after, in_place=True)
