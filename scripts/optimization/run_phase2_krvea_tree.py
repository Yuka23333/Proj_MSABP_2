"""Staged arbitrary-tree K-RVEA: collect, blind validation, then approved batches.

F5/default only reports status. --execute is required for GPU work or CST.
Initial geometry preparation/dispatch lives in prepare_krvea_tree.py. Old 11-D
entry points, geometry, solver settings, and the historical RoI are unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from scripts.optimization import prepare_krvea_tree as prep  # noqa: E402
from scripts.optimization import morris_tree_common as common  # noqa: E402
from scripts.postprocessing import analyze_morris_tree as base  # noqa: E402
from scripts.postprocessing.analyze_morris_tree_roi import roi_job  # noqa: E402
from scripts.simulation.run_morris_tree import active_tasks, campaign_lock  # noqa: E402
from msabp_opt.optimization import krvea, phase2_krvea_tree_data as data  # noqa: E402
from msabp_opt.optimization import phase2_krvea_tree_relay as relay  # noqa: E402


def freeze_json(path, value):
    """Retry-safe immutable artifact: never replace a different previous result."""
    path = Path(path)
    if path.exists():
        if common.digest(common.read_json(path)) != common.digest(value):
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        common.write_json(temporary, value)
        temporary.rename(path)
    return path


def result_manifests(directory):
    """Permission errors must not masquerade as zero completed observations."""
    def fail(error):
        raise error
    found = []
    if directory.exists():
        for parent, _, names in os.walk(directory, onerror=fail):
            if 'manifest.json' in names:
                found.append(Path(parent) / 'manifest.json')
    return sorted(found)


def batch_plans(folder, plan):
    batches = []
    for directory in sorted((folder / 'batches').glob('batch_*')):
        batch = common.read_json(directory / 'batch.json')
        if batch['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('Optimization batch belongs to another plan')
        if common.digest({k: v for k, v in batch.items() if k != 'batch_sha256'}) != batch['batch_sha256']:
            raise ValueError('Frozen optimization batch changed')
        if common.file_hash(directory / 'sample.csv') != batch['csv_sha256']:
            raise ValueError('Frozen optimization CSV changed')
        batches.append((directory, batch))
    return batches


def require_matching_baseline(manifest_paths, baseline):
    for path in manifest_paths:
        manifest = common.read_json(path)
        for key in ('baseline_history_sha256', 'expected_solver_name'):
            if manifest.get(key) != baseline[key]:
                raise ValueError(f'Simulation baseline differs from Morris ({key}): {path}')


def collect(folder, plan, *, workers=4):
    """Import the sealed Morris seed and independently audit all new solves."""
    space = data.TreeInputSpace.from_plan(plan['morris_plan'])
    seed = common.read_json(folder / 'morris_seed.json')
    # Recheck the metric implementation/report without rehashing 316 old FFS on
    # each batch. Raw artifacts were audited at bootstrap; the seed is hash-sealed.
    report_seed = data.load_morris_seed(ROOT / plan['config']['morris_campaign'], audit_artifacts=False)
    if (seed['records'] != report_seed.records or seed['aliases'] != report_seed.aliases
            or not np.array_equal(seed['x_unit'], report_seed.x_unit)
            or not np.array_equal(seed['objectives'], report_seed.objectives)):
        raise ValueError('Historical seed/report changed since bootstrap')
    points = common.read_json(folder / 'points.json')
    for _, batch in batch_plans(folder, plan):
        points.extend(batch['points'])
    keys = [r['simulation_key'] for r in seed['records']] + [p['simulation_key'] for p in points]
    if len(keys) != len(set(keys)):
        raise ValueError('Physical geometry overlap across seed/train/holdout/optimization')
    paths = result_manifests(folder / 'results')
    values, issues = base.collect_results(plan['morris_plan'], points, paths) if paths else ({}, [])
    baseline = common.read_json(folder/'baseline_provenance.json')
    for value in values.values():
        require_matching_baseline(value['result_manifests'], baseline)
    jobs = [(sid, value['result_manifests'], str(folder / 'analysis' / 'roi_cache'))
            for sid, value in sorted(values.items())]
    if workers == 1:
        roi = dict(map(roi_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            roi = dict(pool.map(roi_job, jobs)) if jobs else {}
    records = []
    for point in points:
        sid = point['sample_id']
        if sid not in values:
            continue
        observed = {**values[sid], **roi[sid]}
        objective = [observed[name] for name in data.REPORT_OBJECTIVES]
        if not np.isfinite(objective).all() or not 0 <= objective[0] <= 1:
            raise ValueError(f'Nonphysical objectives, not an optimization penalty: {sid}')
        if not np.isclose(space.exact_normalized_area(point['unit']), objective[2], rtol=1e-12):
            raise ValueError(f'Exact area and simulated geometry disagree: {sid}')
        records.append({**point, 'objectives': objective,
                        'result_manifests': observed['result_manifests'],
                        'ffs_sha256': observed['ffs_sha256']})
    complete = {r['sample_id'] for r in records}
    counts = {split: {'planned': sum(p['split'] == split for p in points),
                      'completed': sum(r['split'] == split for r in records)}
              for split in ('train', 'holdout', 'optimization')}
    snapshot = {'plan_sha256': plan['plan_sha256'], 'seed_count': len(seed['records']),
                'counts': counts, 'records': records,
                'missing': [p['sample_id'] for p in points if p['sample_id'] not in complete],
                'issues': issues, 'metric_contract': data.ROI_CONTRACT}
    snapshot['snapshot_sha256'] = common.digest(snapshot)
    return space, seed, snapshot


def arrays(seed, records):
    x = np.asarray([*seed['x_unit'], *[r['unit'] for r in records]], dtype=np.float64)
    y = np.asarray([*seed['objectives'], *[r['objectives'] for r in records]], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y[:, 0] > 1) or np.any(y[:, 0] < 0):
        raise ValueError('Nonfinite/out-of-domain training data')
    return x, y


def require_bootstrap_complete(snapshot):
    for split in ('train', 'holdout'):
        count = snapshot['counts'][split]
        if count['completed'] != count['planned']:
            raise RuntimeError(f'{split} incomplete: {count}; no fit/optimization launched')
    # Failed attempts are retained in the audit, but a subsequent valid result
    # may resolve that sample. Ambiguous valid results are excluded by collector.
    if any(i['sample_id'] is None for i in snapshot['issues']):
        raise RuntimeError('Unassigned result audit issues require review')


def make_request(space, plan, x, y, *, iteration=0, remaining=4, validation_x=None,
                 previous_empty_reference_count=None):
    p = plan['config']['proposal']
    kwargs = dict(plan=plan['morris_plan'],
                  config=krvea.KRVEAConfig(n_variables=len(space.names), n_objectives=3,
                                          q=p['q'], seed=p['seed'] + iteration,
                                          inner_evaluations=p['inner_evaluations']),
                  iteration=iteration, remaining_expensive_budget=remaining,
                  previous_empty_reference_count=previous_empty_reference_count,
                  compute_device=p['compute_device'],
                  surrogate_settings=relay.SurrogateFitSettings(
                      gp_training_steps=p['gp_training_steps'],
                      uncertainty_calibration_source='tree_initial_uncalibrated'))
    args = (x, y[:, :2], y, np.zeros(len(x), dtype=bool), space)
    if validation_x is not None:
        return relay.build_validation_request_payload(*args, validation_x_unit=validation_x, **kwargs)
    return relay.build_request_payload(*args, **kwargs)


def remote_arguments(folder, plan, request, response, iteration):
    p = plan['config']['proposal']
    registry = prep.load_device_registry(ROOT / plan['config']['simulation']['devices_config'])
    return dict(device=registry.get_device(p['device_id']),
                remote=relay.RemoteProposalConfig(device_id=p['device_id'], python_path=p['python_path'],
                                                 compute_device=p['compute_device']),
                plan_id=plan['campaign_id'], batch_index=iteration,
                local_request_path=request, local_response_path=response)


def score_validation(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    mean = np.asarray(prediction['predicted_mean_minimize'], dtype=float)
    std = np.asarray(prediction['predicted_std'], dtype=float)
    if truth.shape != mean.shape or mean.shape != std.shape or truth.shape[1] != 3:
        raise ValueError('Validation arrays must have equal (N,3) shapes')
    if not (np.isfinite(truth).all() and np.isfinite(mean).all() and np.isfinite(std).all()) or np.any(std < 0):
        raise ValueError('Invalid validation values')
    result = {}
    for j, name in enumerate(data.OBJECTIVE_NAMES):
        error = mean[:, j] - truth[:, j]
        rmse = float(np.sqrt(np.mean(error ** 2)))
        spread = float(np.std(truth[:, j]))
        varying = np.ptp(truth[:, j]) > 0 and np.ptp(mean[:, j]) > 0
        row = {'rmse': rmse, 'mae': float(np.mean(np.abs(error))),
               'rmse_over_holdout_std': rmse / spread if spread > 0 else None,
               'spearman_rho': float(spearmanr(truth[:, j], mean[:, j]).statistic) if varying else None,
               'bias': float(error.mean())}
        if j < 2:
            row.update(coverage_95=float(np.mean(np.abs(error) <= 1.96 * std[:, j])),
                       mean_predicted_std=float(std[:, j].mean()))
        else:
            row.update(exact_area_check=bool(np.allclose(error, 0, atol=1e-12, rtol=0)),
                       zero_uncertainty_check=bool(np.all(std[:, j] == 0)))
        result[name] = row
    return result


def validation_inputs(space, plan, seed, snapshot):
    require_bootstrap_complete(snapshot)
    training = [r for r in snapshot['records'] if r['split'] == 'train']
    holdout = [r for r in snapshot['records'] if r['split'] == 'holdout']
    x, y = arrays(seed, training)
    payload = make_request(space, plan, x, y, validation_x=np.asarray([r['unit'] for r in holdout]))
    return payload, holdout


def validate(folder, plan, space, seed, snapshot, *, execute):
    payload, holdout = validation_inputs(space, plan, seed, snapshot)
    destination = folder / 'validation'
    request = freeze_json(destination / 'request.json', payload)
    response = destination / 'response.json'
    if not execute:
        print(f'Validation request ready (X only for holdout, no fit): {request}')
        return
    prediction = relay.relay_remote_validation(**remote_arguments(folder, plan, request, response, 0))
    truth = [r['objectives'] for r in holdout]
    report = {'plan_sha256': plan['plan_sha256'], 'request_sha256': common.file_hash(request),
              'response_sha256': common.file_hash(response),
              'holdout_sha256': common.digest(holdout),
              'training_count': len(payload['training']['x_unit']), 'holdout_count': len(holdout),
              'holdout_ids': [r['sample_id'] for r in holdout],
              'used_for_training': False, 'automatic_approval': False,
              'scores': score_validation(truth, prediction)}
    freeze_json(destination / 'report.json', report)
    print(json.dumps(report['scores'], indent=2))
    print('Review validation/report.json before setting validation_approved=true and a new-point budget.')


def require_approval(config, folder, plan, space, seed, snapshot):
    if config.get('validation_approved') is not True:
        raise RuntimeError('Review held-out predictions first; validation_approved is not true')
    budget = config.get('optimization_budget')
    if type(budget) is not int or budget < 1:
        raise ValueError('Set an explicit positive optimization_budget (new points only)')
    payload, holdout = validation_inputs(space, plan, seed, snapshot)
    destination = folder / 'validation'
    report = common.read_json(destination / 'report.json')
    request = destination / 'request.json'
    response = destination / 'response.json'
    if (common.digest(common.read_json(request)) != common.digest(payload) or report['plan_sha256'] != plan['plan_sha256']
            or common.file_hash(request) != report['request_sha256']
            or common.file_hash(response) != report['response_sha256']
            or common.digest(holdout) != report['holdout_sha256']):
        raise ValueError('Validation does not match this plan/data/worker; review again')
    relay.validation_from_response(common.read_json(response), expected_request_sha256=report['request_sha256'], request=payload)
    return budget


def accept_candidates(space, plan, proposed, seen_keys, batch_index):
    """Only physical unique candidates become expensive jobs; never fake labels."""
    accepted, audit = [], []
    seen = set(seen_keys)
    for i, unit in enumerate(proposed):
        try:
            point = prep.geometry_record(space, space.denormalize(unit),
                                         f'tree_bo_{batch_index:04d}_{i:02d}', 'optimization',
                                         plan['morris_plan']['config']['coordinate_quantum_mm'])
        except prep.builder.model.manufacturing.ManufacturingError as exc:
            audit.append({'index': i, 'status': 'manufacturing_rejected', 'report': exc.report})
            continue
        if point['simulation_key'] in seen:
            audit.append({'index': i, 'status': 'duplicate_geometry'})
            continue
        seen.add(point['simulation_key'])
        accepted.append(point)
        audit.append({'index': i, 'status': 'accepted', 'sample_id': point['sample_id']})
    return accepted, audit


def optimization_command(folder, plan, directory, batch):
    command = prep.princess_command(folder, plan, 'train')
    for flag, value in (('--csv', directory / 'sample.csv'), ('--run-id', batch['run_id']),
                        ('--results-root', folder / 'results' / directory.name)):
        command[command.index(flag) + 1] = str(value)
    return command


def commit_batch(folder, plan, points, iteration, request, response, empty_reference_count):
    """Publish only a complete immutable batch; orphan staging dirs are harmless."""
    batches = folder / 'batches'
    batches.mkdir(exist_ok=True)
    directory = batches / f'batch_{iteration:04d}'
    if directory.exists():
        raise FileExistsError(f'Batch already exists: {directory}')
    staging = Path(tempfile.mkdtemp(prefix='.staging_', dir=batches))
    prep.write_worklist(staging / 'sample.csv', points)
    batch = {'plan_sha256': plan['plan_sha256'], 'points': points,
             'run_id': f'{plan["campaign_id"]}-b{iteration:04d}',
             'empty_reference_count': int(empty_reference_count),
             'request_sha256': common.file_hash(request), 'response_sha256': common.file_hash(response),
             'csv_sha256': common.file_hash(staging / 'sample.csv')}
    batch['batch_sha256'] = common.digest(batch)
    common.write_json(staging / 'batch.json', batch)
    staging.rename(directory)
    return directory, batch


def propose(folder, plan, config, space, seed, snapshot, *, execute):
    budget = require_approval(config, folder, plan, space, seed, snapshot)
    batches = batch_plans(folder, plan)
    used = sum(len(b['points']) for _, b in batches)
    if used > budget:
        raise ValueError('Registered optimization points exceed budget; start a new plan')
    completed = {r['sample_id'] for r in snapshot['records']}
    for directory, batch in batches:
        if any(p['sample_id'] not in completed for p in batch['points']):
            print(f'Resume existing batch; no new proposal: {batch["run_id"]}')
            return directory, batch
    if used == budget:
        print(f'Optimization complete: {used}/{budget} registered points, all audited.')
        return None
    # Assimilation happens only after a saved independent validation + approval.
    x, y = arrays(seed, snapshot['records'])
    iteration = len(batches) + 1
    q = min(plan['config']['proposal']['q'], budget - used)
    previous_empty = batches[-1][1]['empty_reference_count'] if batches else None
    payload = make_request(space, plan, x, y, iteration=iteration, remaining=budget-used,
                           previous_empty_reference_count=previous_empty)
    request_folder = folder / 'proposals' / f'batch_{iteration:04d}'
    request = freeze_json(request_folder / 'request.json', payload)
    response = request_folder / 'response.json'
    if not execute:
        print(f'Proposal request ready; no GP fit/CST: {request}')
        return None
    proposed = relay.relay_remote_proposal(
        **remote_arguments(folder, plan, request, response, iteration),
        expected_q=q, expected_dimension=len(space.names), observed_x_unit=x, input_space=space)
    seen = [r['simulation_key'] for r in seed['records'] + snapshot['records']]
    points, audit = accept_candidates(space, plan, proposed.unit_values, seen, iteration)
    freeze_json(request_folder / 'geometry_audit.json', audit)
    if len(points) != q:
        raise RuntimeError(f'Only {len(points)}/{q} proposals are unique manufacturable geometries; '
                           'no batch/budget changed. Review geometry_audit.json before continuing.')
    return commit_batch(folder, plan, points, iteration, request, response,
                        proposed.diagnostics['empty_reference_count'])


def ensure_idle(plan, folder, *, except_run=None):
    run_ids = [f'{plan["campaign_id"]}-{s}' for s in ('train', 'holdout')]
    run_ids += [b['run_id'] for _, b in batch_plans(folder, plan)]
    for run_id in run_ids:
        if run_id != except_run and active_tasks(run_id):
            raise RuntimeError(f'Other dispatch still has running/pending tasks: {run_id}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=prep.CONFIG)
    parser.add_argument('--action', choices=('status', 'collect', 'validate', 'propose', 'run'), default='status')
    parser.add_argument('--execute', action='store_true', help='Permit remote GPU work (and CST only for run)')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.execute and args.action in ('status', 'collect')):
        parser.error('Invalid workers or --execute on a read/collect-only action')
    config = prep.load_config(args.config)
    folder, plan = prep.load_prepared(config)
    if args.execute and not args.yes and input('Type RUN to allow remote work: ').strip() != 'RUN':
        return 0
    with campaign_lock(folder):
        while True:
            space, seed, snapshot = collect(folder, plan, workers=args.workers)
            print(f'Morris seed={snapshot["seed_count"]}; {json.dumps(snapshot["counts"])}', flush=True)
            if args.action == 'status':
                return 0
            freeze_json(folder/'analysis'/f'snapshot-{snapshot["snapshot_sha256"]}.json', snapshot)
            if args.action == 'collect':
                return 0
            if args.action == 'validate':
                ensure_idle(plan, folder)
                validate(folder, plan, space, seed, snapshot, execute=args.execute)
                return 0
            prepared = propose(folder, plan, config, space, seed, snapshot, execute=args.execute)
            if prepared is None:
                return 0
            directory, batch = prepared
            command = optimization_command(folder, plan, directory, batch)
            print(subprocess.list2cmdline(command), flush=True)
            if args.action != 'run' or not args.execute:
                return 0
            ensure_idle(plan, folder, except_run=batch['run_id'])
            prep.preflight_devices(plan)
            code = subprocess.call(command, cwd=ROOT)
            if code:
                return code
            # An all-terminal Princess can include failed points. Keep them
            # unresolved; stop instead of generating false EM penalties/loops.
            _, _, updated = collect(folder, plan, workers=args.workers)
            completed = {r['sample_id'] for r in updated['records']}
            if any(p['sample_id'] not in completed for p in batch['points']):
                raise RuntimeError('Batch has unresolved results. Inspect/retry explicitly; no new fit started.')


if __name__ == '__main__':
    raise SystemExit(main())
