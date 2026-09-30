from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.optimization import run_phase2_krvea_tree as run
from scripts.optimization import prepare_morris_tree as morris


@pytest.fixture
def bootstrap():
    config = run.common.read_json(run.ROOT / 'configs/optimization/morris_tree_35d.json')
    variables, baseline = morris.variable_spec(config)
    frozen = {'config': config, 'variables': variables, 'baseline_request': baseline,
              'dimension': len(variables), 'plan_sha256': 'a' * 64,
              'template_sha256': 'b' * 64, 'source_sha256': {'model': 'c' * 64},
              'metrics': {'band_ghz': [3.1, 4.8], 'reference_area_mm2': 2720.2}}
    space = run.data.TreeInputSpace.from_plan(frozen)
    plan = {'morris_plan': frozen, 'plan_sha256': 'd' * 64, 'campaign_id': 'tree-test',
            'config': {'proposal': {'q': 4, 'seed': 42, 'inner_evaluations': 100,
                                    'compute_device': 'cpu', 'gp_training_steps': 2}}}
    x = np.random.default_rng(42).random((6, 35))
    y = np.column_stack((np.linspace(.2, .8, 6), np.arange(6.), space.exact_normalized_area(x)))
    seed = {'x_unit': x[:2].tolist(), 'objectives': y[:2].tolist(), 'records': []}
    records = [{'sample_id': str(i), 'unit': x[i].tolist(), 'objectives': y[i].tolist(),
                'split': 'train' if i < 4 else 'holdout'} for i in range(2, 6)]
    snapshot = {'records': records, 'counts': {'train': {'planned': 2, 'completed': 2},
                                            'holdout': {'planned': 2, 'completed': 2}}, 'issues': []}
    return space, plan, seed, snapshot


def test_validation_contains_no_holdout_labels_or_optimization_rows(bootstrap):
    space, plan, seed, snapshot = bootstrap
    snapshot['records'].append({'sample_id': 'future', 'unit': [.9] * 35,
                                'objectives': [.1, -5, 1], 'split': 'optimization'})
    payload, holdout = run.validation_inputs(space, plan, seed, snapshot)
    assert len(payload['training']['x_unit']) == 4
    assert payload['training']['y_full_minimize'] == seed['objectives'] + [r['objectives'] for r in snapshot['records'][:2]]
    assert payload['validation'] == {'x_unit': [r['unit'] for r in holdout]}
    assert payload['compute']['dtype'] == 'float64'
    assert payload['tree_provenance']['dimension'] == 35


def test_incomplete_data_never_builds_request(bootstrap, monkeypatch):
    space, plan, seed, snapshot = bootstrap
    snapshot['counts']['holdout']['completed'] = 1
    monkeypatch.setattr(run, 'make_request', lambda *a, **k: pytest.fail('fit request must not be created'))
    with pytest.raises(RuntimeError, match='holdout incomplete'):
        run.validation_inputs(space, plan, seed, snapshot)


def test_validation_prepare_does_not_call_remote(bootstrap, tmp_path, monkeypatch):
    space, plan, seed, snapshot = bootstrap
    monkeypatch.setattr(run.relay, 'relay_remote_validation', lambda **k: pytest.fail('no remote fit'))
    run.validate(tmp_path, plan, space, seed, snapshot, execute=False)
    run.validate(tmp_path, plan, space, seed, snapshot, execute=False)  # JSON tuple/list round trip
    assert (tmp_path / 'validation/request.json').is_file()
    assert not (tmp_path / 'validation/report.json').exists()


def test_validation_metrics_and_exact_area():
    y = np.array([[.2, 3, .9], [.5, 1, 1], [.8, 2, 1.1]])
    mean = y.copy()
    mean[:, 0] += .05
    prediction = {'predicted_mean_minimize': mean, 'predicted_std': [[.1, .1, 0]] * 3}
    score = run.score_validation(y, prediction)
    assert score[run.data.OBJECTIVE_NAMES[0]]['rmse'] == pytest.approx(.05)
    assert score[run.data.OBJECTIVE_NAMES[0]]['coverage_95'] == 1
    assert score[run.data.OBJECTIVE_NAMES[2]]['exact_area_check']
    assert score[run.data.OBJECTIVE_NAMES[2]]['zero_uncertainty_check']


def test_approval_needs_explicit_budget_and_report(bootstrap, tmp_path):
    space, plan, seed, snapshot = bootstrap
    with pytest.raises(RuntimeError, match='validation_approved'):
        run.require_approval({}, tmp_path, plan, space, seed, snapshot)
    for budget in (None, 0, -1, True, 4.5):
        with pytest.raises(ValueError, match='optimization_budget'):
            run.require_approval({'validation_approved': True, 'optimization_budget': budget},
                                 tmp_path, plan, space, seed, snapshot)
    with pytest.raises(FileNotFoundError):
        run.require_approval({'validation_approved': True, 'optimization_budget': 8},
                             tmp_path, plan, space, seed, snapshot)


def test_validation_report_tampering_blocks_assimilation(bootstrap, tmp_path, monkeypatch):
    space, plan, seed, snapshot = bootstrap
    _, heldout = run.validation_inputs(space, plan, seed, snapshot)
    prediction = {'predicted_mean_minimize': [r['objectives'] for r in heldout],
                  'predicted_std': [[.1, .1, 0]] * len(heldout),
                  'diagnostics': {'gp_training_observations': 4, 'validation_observations': 2,
                                  'validation_labels_used_in_fit': False}}
    monkeypatch.setattr(run, 'remote_arguments', lambda *a: {})
    def remote(**_):
        request_path = tmp_path / 'validation/request.json'
        run.freeze_json(tmp_path / 'validation/response.json', {
            'schema_version': run.relay.RESPONSE_SCHEMA_VERSION, 'task': 'validate', 'status': 'completed',
            'request_sha256': run.common.file_hash(request_path), 'validation_result': prediction})
        return prediction
    monkeypatch.setattr(run.relay, 'relay_remote_validation', remote)
    run.validate(tmp_path, plan, space, seed, snapshot, execute=True)
    config = {'validation_approved': True, 'optimization_budget': 8}
    assert run.require_approval(config, tmp_path, plan, space, seed, snapshot) == 8
    altered = deepcopy(snapshot)
    altered['records'][-1]['objectives'][1] += 1
    with pytest.raises(ValueError, match='does not match'):
        run.require_approval(config, tmp_path, plan, space, seed, altered)


def test_geometry_duplicate_guard_is_physical_not_just_unit_x(monkeypatch):
    def geometry(space, raw, sid, split, quantum):
        return {'sample_id': sid, 'simulation_key': 'same-key'}
    monkeypatch.setattr(run.prep, 'geometry_record', geometry)
    space = SimpleNamespace(denormalize=lambda x: x)
    plan = {'morris_plan': {'config': {'coordinate_quantum_mm': .01}}}
    accepted, audit = run.accept_candidates(space, plan, [[.1], [.8]], [], 1)
    assert len(accepted) == 1 and audit[-1]['status'] == 'duplicate_geometry'
    accepted, _ = run.accept_candidates(space, plan, [[.1]], ['same-key'], 1)
    assert accepted == []


def test_frozen_file_no_overwrite(tmp_path):
    path = tmp_path / 'artifact.json'
    run.freeze_json(path, {'a': 1})
    run.freeze_json(path, {'a': 1})
    with pytest.raises(ValueError, match='differs'):
        run.freeze_json(path, {'a': 2})


def test_permission_error_not_misreported_as_empty(tmp_path, monkeypatch):
    def walk(path, onerror):
        onerror(PermissionError('denied'))
        return iter(())
    monkeypatch.setattr(run.os, 'walk', walk)
    with pytest.raises(PermissionError):
        run.result_manifests(tmp_path)


def test_solver_and_baseline_history_are_not_interchangeable(tmp_path):
    baseline = {'expected_solver_name': 'HF Time Domain', 'baseline_history_sha256': 'a' * 64}
    path = run.freeze_json(tmp_path/'valid.json', baseline)
    run.require_matching_baseline([path], baseline)
    for field, value in [('expected_solver_name', 'HF Frequency Domain'), ('baseline_history_sha256', 'b' * 64)]:
        other = run.freeze_json(tmp_path/f'{field}.json', {**baseline, field: value})
        with pytest.raises(ValueError, match='baseline differs'):
            run.require_matching_baseline([other], baseline)


def test_existing_batch_resumes_without_fit_and_no_fit_at_budget(bootstrap, tmp_path, monkeypatch):
    space, plan, seed, snapshot = bootstrap
    monkeypatch.setattr(run, 'require_approval', lambda *a: 1)
    batch = {'points': [{'sample_id': 'unfinished'}], 'run_id': 'test-batch'}
    monkeypatch.setattr(run, 'batch_plans', lambda *a: [(tmp_path, batch)])
    monkeypatch.setattr(run, 'make_request', lambda *a, **k: pytest.fail('no fit'))
    assert run.propose(tmp_path, plan, {}, space, seed, snapshot, execute=True) == (tmp_path, batch)
    snapshot['records'].append({'sample_id': 'unfinished'})
    assert run.propose(tmp_path, plan, {}, space, seed, snapshot, execute=True) is None


def test_underfilled_proposal_does_not_consume_budget(bootstrap, tmp_path, monkeypatch):
    space, plan, seed, snapshot = bootstrap
    monkeypatch.setattr(run, 'require_approval', lambda *a: 4)
    monkeypatch.setattr(run, 'batch_plans', lambda *a: [])
    monkeypatch.setattr(run, 'remote_arguments', lambda *a: {})
    monkeypatch.setattr(run, 'accept_candidates', lambda *a: ([], [{'status': 'manufacturing_rejected'}]))
    monkeypatch.setattr(run.relay, 'relay_remote_proposal', lambda **k: SimpleNamespace(unit_values=[[.4]*35]*4))
    for record in snapshot['records']:
        record['simulation_key'] = record['sample_id']
    with pytest.raises(RuntimeError, match='no batch/budget changed'):
        run.propose(tmp_path, plan, {}, space, seed, snapshot, execute=True)
    assert not (tmp_path / 'batches').exists()
    assert (tmp_path / 'proposals/batch_0001/geometry_audit.json').is_file()


def test_proposal_continues_reference_vector_state(bootstrap, tmp_path, monkeypatch):
    space, plan, seed, snapshot = bootstrap
    monkeypatch.setattr(run, 'require_approval', lambda *a: 8)
    monkeypatch.setattr(run, 'batch_plans', lambda *a: [(tmp_path, {
        'points': [], 'empty_reference_count': 7})])
    run.propose(tmp_path, plan, {}, space, seed, snapshot, execute=False)
    payload = run.common.read_json(tmp_path / 'proposals/batch_0002/request.json')
    assert payload['previous_empty_reference_count'] == 7


def test_partial_batch_staging_does_not_block_resume(bootstrap, tmp_path, monkeypatch):
    _, plan, _, _ = bootstrap
    request = run.freeze_json(tmp_path / 'request.json', {'request': True})
    response = run.freeze_json(tmp_path / 'response.json', {'response': True})
    points = [{'sample_id': 'case1', 'request': {}}]
    with monkeypatch.context() as patch:
        patch.setattr(run.prep, 'write_worklist', lambda *a: (_ for _ in ()).throw(OSError('interrupted')))
        with pytest.raises(OSError, match='interrupted'):
            run.commit_batch(tmp_path, plan, points, 1, request, response, 3)
    assert run.batch_plans(tmp_path, plan) == []
    directory, batch = run.commit_batch(tmp_path, plan, points, 1, request, response, 3)
    assert run.batch_plans(tmp_path, plan) == [(directory, batch)]
    assert batch['empty_reference_count'] == 3
