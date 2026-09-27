import csv
import json
from types import SimpleNamespace

import pytest

from scripts.simulation import sanity_check_fd_tree as fd


def test_solver_check_is_opt_in(monkeypatch):
    exports = fd.runner.exports
    setup = SimpleNamespace(solver_name=fd.FD_SOLVER, solver_running=False,
                            ports=(exports.RECORDED_PORT_TREE_ITEM,),
                            farfield_monitors=exports.RECORDED_FARFIELD_MONITOR_TREE_ITEMS)
    monkeypatch.setattr(exports, 'inspect_project', lambda *a: setup)
    with pytest.raises(RuntimeError, match='expected=HF Time Domain'):
        exports.inspect_recorded_simulation_setup(object())
    assert exports.inspect_recorded_simulation_setup(object(), expected_solver_name=fd.FD_SOLVER) is setup
    setup.ports = ()
    with pytest.raises(RuntimeError, match='port is missing'):
        exports.inspect_recorded_simulation_setup(object(), expected_solver_name=fd.FD_SOLVER)


def test_prepare_reuses_samples_and_freezes_template(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, 'ROOT', tmp_path)
    source = tmp_path / 'source'
    source.mkdir()
    request = fd.timing.validation.case_request('case3')
    with (source / 'sample.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'simulation_mode', 'tree_json'])
        writer.writeheader()
        writer.writerow({'sample_id': 'sample_001', 'simulation_mode': 'antenna_tree',
                         'tree_json': json.dumps(request)})
    project = source / 'template.cst'
    project.write_bytes(b'FD template')
    args = SimpleNamespace(run_id='fd-test', source_run=source, project=project, count=1)
    plan, folder, rows = fd.prepare(args)
    assert plan['samples'][0]['parameters'] == request
    assert (folder / 'template.cst').read_bytes() == b'FD template'
    assert fd.prepare(args) == (plan, folder, rows)
    project.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Plan changed'):
        fd.prepare(args)


def test_fd_batch_stops_on_first_error(tmp_path, monkeypatch):
    (tmp_path / 'template.cst').write_bytes(b'FD')
    closed = []
    project = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(fd.runner.exports, 'open_cst_project', lambda p: project)
    monkeypatch.setattr(fd.runner.exports, 'inspect_recorded_simulation_setup',
                        lambda *a, **k: SimpleNamespace(solver_running=False))
    calls = []

    def fail(row, **kwargs):
        calls.append(kwargs['expected_solver_name'])
        kwargs['stage_callback']('solving')
        raise RuntimeError('test failure')

    monkeypatch.setattr(fd.runner, 'run_csv_row', fail)
    rows = [{'sample_id': 'sample_001'}, {'sample_id': 'sample_002'}]
    assert fd.run({}, tmp_path, rows) == 2
    assert calls == [fd.FD_SOLVER] and closed == [True]
    failure = json.loads((tmp_path / 'failure.json').read_text())
    assert failure['case_id'] == 'sample_001' and failure['stage'] == 'solving'
    with pytest.raises(FileExistsError):
        fd.run({}, tmp_path, rows)
