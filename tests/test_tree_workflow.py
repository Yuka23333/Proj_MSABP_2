import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from scripts.automation import cst_build_msabp_geometry_tree as builder  # noqa: E402
from scripts.simulation import 基础仿真文件验证_tree as validation  # noqa: E402
from msabp_opt.simulation.distributed import case_runner  # noqa: E402
from msabp_opt.simulation.distributed.princess import PrincessCoordinator  # noqa: E402


@pytest.mark.parametrize('name,count,holes', [('case1',2,0),('case2',1,1),('case3',3,0)])
def test_all_verified_topologies(name, count, holes):
    prepared = builder.prepare_geometry(validation.case_request(name))
    assert len(prepared['copper']) == count
    assert prepared['copper_holes'] == holes
    assert prepared['substrate_bounds'] == pytest.approx([-33.5,0,33.5,40.6])


def test_tree_request_preserves_zero_width_and_reorders_parents():
    raw = validation.case_request('case2')
    raw['tree'] = dict(reversed(list(raw['tree'].items())))
    request = builder.normalize_request(raw)
    assert list(request['tree']).index('U1') < list(request['tree']).index('U1/L1')
    assert request['tree']['U1']['k'][1] == 0
    assert builder.prepare_geometry(request)['copper_holes'] == 1


@pytest.mark.parametrize('bad', [None, {'invalid': 1}, {'tree': {'U1': {'parent':'missing','side':'U','k':[.5]*3}}},
                               {'tree': {'U1': {'parent':'SLOT','side':'U','k':[.5,float('nan'),.5]}}}])
def test_bad_requests(bad):
    with pytest.raises(ValueError):
        builder.prepare_geometry(bad)


class FakeModel:
    def __init__(self, entries):
        self.entries = deepcopy(entries)
        self.resizes = []
        self.rebuilds = 0
        self.commands = []
    def _GetHistory(self):
        return {'list': deepcopy(self.entries)}
    def _ResizeHistory(self, size, timeout=None):
        self.resizes.append(size)
        self.entries = self.entries[:size]
    def full_history_rebuild(self, timeout=None):
        self.rebuilds += 1
        return True
    def add_to_history(self, name, code, timeout=None):
        self.commands.append((name, code))
        self.entries.append({'name':name, 'contents':code, 'error':False})


def entry(name, contents=''):
    return {'name':name, 'contents':contents, 'error':False}


def test_reset_only_owned_suffix(tmp_path):
    baseline = [entry('connector'),entry('material')]
    model = FakeModel(baseline+[entry(builder.HISTORY_PREFIX+'a'),entry(builder.HISTORY_PREFIX+'b')])
    project = SimpleNamespace(model3d=model)
    assert builder.reset_tree_history(project, tmp_path/'history.json') == baseline
    assert model.resizes == [2] and model.rebuilds == 1
    assert (tmp_path/'history.json').is_file()
    model.entries += [entry(builder.HISTORY_PREFIX+'a'),entry('manual change')]
    with pytest.raises(RuntimeError, match='Manual'):
        builder.reset_tree_history(project, tmp_path/'second.json')
    assert model.resizes == [2]


def test_builder_uses_history_hole_subtraction_without_material_redefinition(tmp_path):
    model = FakeModel([entry('material', '.Name "Rogers AD 350A (lossy)"')])
    checks = []
    project = SimpleNamespace(model3d=model, schematic=SimpleNamespace(
        execute_vba_code=lambda code, timeout=None: checks.append(code)))
    report = builder.build_on_project(project, builder.prepare_geometry(validation.case_request('case2')), tmp_path)
    code = '\n'.join(v for _, v in model.commands)
    assert report['copper_components'] == 1 and report['copper_holes'] == 1
    assert 'Solid.Subtract' in code and 'hole_001' in code
    assert 'Material.Delete' not in code and 'With Material' not in code
    assert not any('Solver.Start' in v or 'Save' == v for _, v in model.commands)
    assert all(name.startswith(builder.HISTORY_PREFIX) for name, _ in model.commands)
    assert any('Volume mismatch' in c for c in checks)


def test_maid_tree_dry_run_dispatch(tmp_path):
    request = validation.case_request('case3')
    result = case_runner.run_csv_row({'sample_id':'reference','simulation_mode':'antenna_tree',
                                     'tree_json':json.dumps(request)},
                                    project_path=tmp_path/'not_opened.cst', output_root=tmp_path/'out',dry_run=True)
    manifest = json.loads(result.manifest_path.read_text())
    assert result.simulation_mode == 'antenna_tree'
    assert manifest['geometry']['copper_components'] == 3
    assert manifest['parameters'] == request
    assert manifest['artifacts'] == {} and manifest['dry_run']
    PrincessCoordinator._verify_manifest_artifacts(result.case_directory, manifest)


def test_plan_preparation_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    template = tmp_path/'template.cst'
    template.write_bytes(b'fixture')
    args = SimpleNamespace(run_id='tree-test',band=(3.1,4.8),tolerance_db=.1, request=None,
                           case='case3', project=template, devices_config=validation.comparison.DEVICE_CONFIG)
    plan, folder = validation.prepare(args)
    assert validation.prepare(args) == (plan,folder)
    args.case = 'case2'
    with pytest.raises(ValueError,match='Plan changed'):
        validation.prepare(args)


def test_mock_solve_exports_and_princess_accepts_tree_artifacts(tmp_path, monkeypatch):
    from msabp_opt.simulation.distributed import case_runner_tree as runner
    project_path = tmp_path/'test.cst'
    project_path.write_bytes(b'fixture')
    project = object()
    monkeypatch.setattr(runner.exports, 'clear_results_on_project', lambda *a, **k: None)
    monkeypatch.setattr(runner.exports, 'inspect_recorded_simulation_setup', lambda *a, **k: None)
    monkeypatch.setattr(runner.builder, 'build_on_project', lambda *a, **k: {'copper_components': 3})
    def solve(p, output, **kwargs):
        assert p is project
        kwargs['stage_callback']('solving')
        kwargs['stage_callback']('exporting_1d_results')
        for filename in ('S11.csv', 'Rad_Eff.csv', 'Tot_Eff.csv'):
            (output.parent/filename).write_text('3 -10\n4 -20\n5 -10\n')
        ffs = case_runner.project_farfield_source_path(project_path)
        ffs.parent.mkdir(parents=True)
        ffs.write_text('new FFS')
    monkeypatch.setattr(runner.exports, 'solve_and_export_s11_on_project', solve)
    result = case_runner.run_csv_row({'sample_id':'reference','simulation_mode':'antenna_tree',
                                     'tree_json':json.dumps(validation.case_request('case3'))},
                                    project_path=project_path, output_root=tmp_path/'out', project=project)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest['status'] == 'completed'
    assert 'solving' in manifest['stage_seconds']
    assert manifest['stage_seconds']['solving'] >= 0
    assert set(manifest['artifacts']) == {'s11','rad_eff','tot_eff','farfield_source','geometry_tree'}
    PrincessCoordinator._verify_manifest_artifacts(result.case_directory, manifest)
