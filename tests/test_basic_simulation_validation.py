import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/simulation/基础仿真文件验证.py"
spec = importlib.util.spec_from_file_location("basic_validation", SCRIPT)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_compare_identical_different_grids():
    a = (np.array([3., 4., 5.]), np.array([-10., -20., -30.]))
    b = (np.array([3., 3.5, 4., 4.5, 5.]), np.array([-10., -15., -20., -25., -30.]))
    result = validation.compare_pair(a, b, (3.1, 4.8))
    assert result['max_abs_delta_db'] == pytest.approx(0)
    assert result['rms_delta_db'] == pytest.approx(0)


def test_compare_offset_and_coverage():
    a = (np.array([3., 5.]), np.array([-10., -20.]))
    b = (a[0], a[1] + .2)
    result = validation.compare_pair(a, b, (3.1, 4.8))
    assert result['max_abs_delta_db'] == pytest.approx(.2)
    assert result['rms_delta_db'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='cover'):
        validation.compare_pair(a, b, (2., 4.8))


def test_curve_rejects_nonfinite_and_duplicates(tmp_path):
    path = tmp_path / 'curve.csv'
    for content in ('3 nan\n4 1\n', '3 1\n3 2\n'):
        path.write_text(content)
        with pytest.raises(ValueError):
            validation.read_curve(path)
    path.write_text('Frequency / GHz  S11 / dB\n3 -10\n4 -20\n')
    x, y = validation.read_curve(path)
    assert list(x) == [3, 4] and list(y) == [-10, -20]


def test_pinned_commands_share_one_input(tmp_path):
    plan = {'run_id': 'test', 'devices_config': 'devices.json'}
    commands = [validation.princess_command(plan, d, tmp_path) for d in validation.DEVICE_IDS]
    for device, command in zip(validation.DEVICE_IDS, commands):
        assert command.count('--device') == 1
        assert command[command.index('--device') + 1] == device
        assert command[command.index('--csv') + 1] == str(tmp_path / 'sample.csv')
        assert command[command.index('--project') + 1] == str(tmp_path / 'template.cst')
    assert len({c[c.index('--run-id') + 1] for c in commands}) == 3


def test_manifest_only_never_passes(tmp_path):
    plan = {'parameters': {}, 'band_ghz': [3.1, 4.8], 'tolerance_db': .1}
    assert validation.compare(plan, tmp_path) == 2
    import json
    report = json.loads((tmp_path / 'comparison.json').read_text())
    assert report['passed'] is False and report['complete'] is False
    assert len(report['errors']) == 3


def test_prepare_freezes_inputs_and_rejects_changed_template(tmp_path, monkeypatch):
    from argparse import Namespace
    registry = validation.DEVICE_CONFIG
    monkeypatch.setattr(validation, 'ROOT', tmp_path)
    template = tmp_path / 'source.cst'
    template.write_bytes(b'fixture only, no CST')
    args = Namespace(run_id='test-validation', tolerance_db=.1, band=(3.1, 4.8),
                     parameters=None, devices_config=registry, project=template)
    plan, folder = validation.prepare(args)
    assert (folder / 'template.cst').read_bytes() == template.read_bytes()
    assert validation.prepare(args) == (plan, folder)
    template.write_bytes(b'changed')
    with pytest.raises(ValueError, match='new --run-id'):
        validation.prepare(args)


def test_three_valid_results_and_hash_tampering(tmp_path):
    import json
    params = {'example': .5}
    for device in validation.DEVICE_IDS:
        case = tmp_path / 'results' / device / 'case_reference'
        case.mkdir(parents=True)
        artifacts = {}
        for key, filename in validation.CURVES.items():
            path = case / filename
            path.write_text('3 -10\n4 -20\n5 -10\n')
            artifacts[key] = {'path': filename, 'sha256': validation.sha256(path)}
        (case / 'manifest.json').write_text(json.dumps({
            'status': 'completed', 'dry_run': False, 'parameters': params, 'artifacts': artifacts,
        }))
    plan = {'parameters': params, 'band_ghz': [3.1, 4.8], 'tolerance_db': .1}
    assert validation.compare(plan, tmp_path) == 0
    report = json.loads((tmp_path / 'comparison.json').read_text())
    assert len(report['comparisons']) == 9 and report['passed']
    (tmp_path / 'results/local/case_reference/S11.csv').write_text('3 -99\n4 -99\n5 -99\n')
    assert validation.compare(plan, tmp_path) == 2
