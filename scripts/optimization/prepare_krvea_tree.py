"""Prepare/dispatch the separate 35D tree bootstrap; default is OFFLINE.

128 supplemental training geometries (including the reference) and 64 held-out
geometries augment Morris, without redefining its frozen campaign or objectives.
Rejection/deduplication means retained LHS candidates are not a strict LHS.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from scripts.optimization import morris_tree_common as common  # noqa: E402
from scripts.optimization.prepare_morris_tree import simulation_key  # noqa: E402
from scripts.automation import cst_build_msabp_geometry_tree as builder  # noqa: E402
from msabp_opt.optimization import phase2_krvea_tree_data as data  # noqa: E402
from msabp_opt.simulation.distributed.config import load_device_registry  # noqa: E402
from msabp_opt.simulation.distributed.runtime import validate_run_id  # noqa: E402
from scripts.simulation.run_morris_tree import active_tasks, campaign_lock, probe_device  # noqa: E402

CONFIG = ROOT / 'configs/optimization/phase2_krvea_tree_35d.json'


def load_config(path=CONFIG):
    config = common.read_json(path)
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported tree bootstrap configuration')
    validate_run_id(config['campaign_id'])
    design = config['initial_design']
    for key in ('training_count', 'holdout_count', 'max_candidates'):
        if type(design[key]) is not int or design[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(design['seed']) is not int or design['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(design['include_reference']) is not bool:
        raise ValueError('include_reference must be boolean')
    registry = load_device_registry(ROOT / config['simulation']['devices_config'])
    for name in config['simulation']['devices']:
        registry.get_device(name)
    if not config['simulation']['devices']:
        raise ValueError('No simulation devices selected')
    return config


def geometry_record(space, raw, sample_id, split, quantum):
    request = space.request_from_raw(raw)
    geometry = builder.prepare_geometry(request, quantum=quantum)
    x0, y0, x1, y1 = geometry['substrate_bounds']
    return {'sample_id': sample_id, 'simulation_sample_id': sample_id, 'split': split,
            'request': geometry['request'], 'unit': space.normalize(raw).tolist(),
            'raw': np.asarray(raw).tolist(), 'simulation_key': simulation_key(geometry),
            'manufactured_copper_sha256': geometry['manufactured_copper_sha256'],
            'substrate_area_mm2': (x1-x0)*(y1-y0)}


def generate_design(space, frozen_plan, seed_records, design):
    seen = {r['simulation_key'] for r in seed_records}
    accepted, audit = [], []
    quantum = frozen_plan['config']['coordinate_quantum_mm']
    seed_sequences = np.random.SeedSequence(design['seed']).spawn(2)
    for split, count_key, sequence in zip(('train', 'holdout'), ('training_count', 'holdout_count'), seed_sequences):
        count = design[count_key]
        sampler = qmc.LatinHypercube(d=len(space.names), seed=np.random.default_rng(sequence))
        candidates = []
        if split == 'train' and design['include_reference']:
            candidates.append(('reference', space.raw_from_request(frozen_plan['baseline_request'])))
        found = 0
        while found < count:
            if not candidates:
                candidates = [('lhs', space.denormalize(u)) for u in sampler.random(count-found)]
            for origin, raw in candidates:
                if len(audit) >= design['max_candidates']:
                    raise RuntimeError('Geometry candidate limit reached; no finished bootstrap was written')
                sample_id = f'tree_{split}_{found:04d}'
                event = {'split': split, 'candidate': len(audit), 'origin': origin,
                         'unit': space.normalize(raw).tolist()}
                try:
                    point = geometry_record(space, raw, sample_id, split, quantum)
                except builder.model.manufacturing.ManufacturingError as exc:
                    audit.append({**event, 'status': 'manufacturing_rejected', 'report': exc.report})
                    if origin == 'reference':
                        raise ValueError('Reference violates frozen manufacturing contract') from exc
                    continue
                # Unexpected geometry errors stop preparation, never become training penalties.
                if point['simulation_key'] in seen:
                    audit.append({**event, 'status': 'duplicate_geometry', 'simulation_key': point['simulation_key']})
                    continue
                seen.add(point['simulation_key'])
                accepted.append(point)
                audit.append({**event, 'status': 'accepted', 'sample_id': sample_id})
                found += 1
                if found % 32 == 0 or found == count:
                    print(f'{split}: {found}/{count} new unique geometries', flush=True)
            candidates = []
    return accepted, audit


def write_worklist(path, points):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'simulation_mode', 'tree_json'])
        writer.writeheader()
        writer.writerows({'sample_id': p['sample_id'], 'simulation_mode': 'antenna_tree',
                          'tree_json': json.dumps(p['request'], separators=(',', ':'))} for p in points)


def load_prepared(config, *, check_provenance=True):
    folder = (ROOT / config['output_directory']).resolve()
    plan = common.read_json(folder / 'plan.json')
    if common.digest({k:v for k,v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('Bootstrap plan was modified')
    # Budget and approval are later, explicit decisions, not sampling semantics.
    fixed = {k:v for k,v in config.items() if k not in ('optimization_budget', 'validation_approved')}
    if plan['config'] != fixed:
        raise ValueError('Bootstrap configuration differs; use a new campaign')
    for name, digest in plan['files_sha256'].items():
        if common.file_hash(folder / name) != digest:
            raise ValueError(f'Frozen bootstrap file changed: {name}')
    if common.source_fingerprints() != plan['morris_plan']['source_sha256'] or common.auxiliary_fingerprints() != plan['morris_plan']['auxiliary_source_sha256']:
        raise ValueError('Geometry/simulation sources changed since Morris')
    if check_provenance:
        seal = common.read_json(folder / 'seed_provenance.json')
        if (common.digest({k:v for k,v in seal.items() if k != 'seal_sha256'}) != seal['seal_sha256']
                or seal['plan_sha256'] != plan['plan_sha256']
                or seal['seed_sha256'] != plan['files_sha256']['morris_seed.json']
                or seal['provenance']['artifact_audit'] != 'verified'
                or seal['provenance']['metric_contract'] != data.ROI_CONTRACT):
            raise ValueError('Morris seed provenance seal differs from this bootstrap')
        if common.file_hash(seal['provenance']['roi_report']) != seal['provenance']['roi_report_sha256']:
            raise ValueError('Frozen Morris RoI report changed; re-audit in a new campaign')
        if seal['metric_source_sha256'] != {p:common.file_hash(ROOT/p) for p in data.METRIC_SOURCE_PATHS}:
            raise ValueError('Frozen metric implementation changed; do not mix old/new objectives')
        baseline = common.read_json(folder/'baseline_provenance.json')
        if (common.digest({k:v for k,v in baseline.items() if k != 'seal_sha256'}) != baseline['seal_sha256']
                or baseline['plan_sha256'] != plan['plan_sha256']
                or baseline['seed_sha256'] != plan['files_sha256']['morris_seed.json']):
            raise ValueError('Frozen simulation baseline provenance changed')
    return folder, plan


def seal_seed(folder, plan, seed):
    """Attach a once-written audit receipt, also upgrading an unsolved bootstrap."""
    cached = common.read_json(folder/'morris_seed.json')
    if (cached['records'] != seed.records or cached['aliases'] != seed.aliases
            or not np.array_equal(cached['x_raw'], seed.x_raw)
            or not np.array_equal(cached['x_unit'], seed.x_unit)
            or not np.array_equal(cached['objectives'], seed.objectives)
            or seed.provenance['artifact_audit'] != 'verified'):
        raise ValueError('A live audited seed matching the sealed bootstrap is required')
    seal = {'plan_sha256': plan['plan_sha256'], 'seed_sha256': plan['files_sha256']['morris_seed.json'],
            'provenance': seed.provenance,
            'metric_source_sha256': {p:common.file_hash(ROOT/p) for p in data.METRIC_SOURCE_PATHS}}
    seal['seal_sha256'] = common.digest(seal)
    path = folder/'seed_provenance.json'
    if path.exists():
        if common.read_json(path) != seal:
            raise ValueError('Existing seed provenance differs; no overwrite allowed')
    else:
        common.write_json(path, seal)
    contracts = set()
    for record in seed.records:
        for manifest_path in record['result_manifests']:
            manifest = common.read_json(manifest_path)
            history = manifest.get('baseline_history_sha256', '')
            solver = manifest.get('expected_solver_name', '')
            if len(history) != 64 or any(c not in '0123456789abcdef' for c in history) or not solver:
                raise ValueError('Missing historical simulation baseline/solver provenance')
            contracts.add((history, solver))
    if len(contracts) != 1:
        raise ValueError('Morris seed has multiple simulation baselines; explicit review required')
    history, solver = next(iter(contracts))
    baseline = {'plan_sha256': plan['plan_sha256'], 'seed_sha256': plan['files_sha256']['morris_seed.json'],
                'baseline_history_sha256': history, 'expected_solver_name': solver,
                'template_sha256': plan['morris_plan']['template_sha256']}
    baseline['seal_sha256'] = common.digest(baseline)
    path = folder/'baseline_provenance.json'
    if path.exists():
        if common.read_json(path) != baseline:
            raise ValueError('Existing simulation baseline differs; no overwrite allowed')
    else:
        common.write_json(path, baseline)


def prepare(config):
    folder = (ROOT / config['output_directory']).resolve()
    if (folder / 'plan.json').exists():
        return load_prepared(config)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Incomplete/nonempty bootstrap directory; inspect before retry')
    campaign = (ROOT / config['morris_campaign']).resolve()
    frozen, _, _ = common.load_campaign(campaign, check_sources=True)
    seed = data.load_morris_seed(campaign)
    print(f'Preparing from {len(seed.x_unit)} independent Morris geometries, {len(seed.input_space.names)}D', flush=True)
    points, audit = generate_design(seed.input_space, frozen, seed.records, config['initial_design'])
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(campaign/'template.cst', folder/'template.cst')
    write_worklist(folder/'train.csv', [p for p in points if p['split']=='train'])
    write_worklist(folder/'holdout.csv', [p for p in points if p['split']=='holdout'])
    common.write_json(folder/'points.json', points)
    common.write_json(folder/'candidate_audit.json', audit)
    common.write_json(folder/'morris_seed.json', {
        'x_raw': seed.x_raw.tolist(), 'x_unit': seed.x_unit.tolist(), 'objectives': seed.objectives.tolist(),
        'records': seed.records, 'aliases': seed.aliases,
        'source_campaign': str(campaign), 'plan_sha256': frozen['plan_sha256'],
    })
    names = ('template.cst', 'train.csv', 'holdout.csv', 'points.json', 'candidate_audit.json', 'morris_seed.json')
    plan = {'schema_version': 1, 'campaign_id': config['campaign_id'],
            'config': {k:v for k,v in config.items() if k not in ('optimization_budget', 'validation_approved')},
            'morris_plan': frozen, 'files_sha256': {n:common.file_hash(folder/n) for n in names},
            'objective_names': list(data.OBJECTIVE_NAMES),
            'split_contract': f'{len(seed.records)} canonical seed +{config["initial_design"]["training_count"]} supplement train; '
                              f'{config["initial_design"]["holdout_count"]} holdout excluded until explicit review',
            'sampling': 'independent seeded LHS candidates per split + reference, geometry-filtered/deduplicated; retained set is not strict LHS'}
    plan['plan_sha256'] = common.digest(plan)
    common.write_json(folder/'plan.json', plan)
    seal_seed(folder, plan, seed)
    return folder, plan


def princess_command(folder, plan, split):
    sim = plan['config']['simulation']
    return [sim['python_path'], str(ROOT/'scripts/simulation/princess.py'), 'start',
            '--csv', str(folder/f'{split}.csv'), '--run-id', f'{plan["campaign_id"]}-{split}',
            '--project', str(folder/'template.cst'), '--results-root', str(folder/'results'/split),
            '--devices-config', str(ROOT/sim['devices_config']), '--coordinate-quantum-mm',
            str(plan['morris_plan']['config']['coordinate_quantum_mm']), '--max-attempts', str(sim['max_attempts']),
            *[item for device in sim['devices'] for item in ('--device', device)]]


def preflight_devices(plan):
    frozen = plan['morris_plan']
    expected = {common.SOURCE_FILES[k]: v for k,v in frozen['source_sha256'].items()}
    expected.update({k:v for k,v in frozen['auxiliary_source_sha256'].items() if not k.startswith('scripts/optimization/')})
    registry = load_device_registry(ROOT/plan['config']['simulation']['devices_config'])
    for device in plan['config']['simulation']['devices']:
        probe_device(registry.get_device(device), expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--dispatch', choices=('train', 'holdout', 'all'))
    parser.add_argument('--execute', action='store_true', help='Actually launch Princess; otherwise only print its command')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--audit-seed', action='store_true', help='Verify historical raw artifacts and attach an immutable provenance receipt; no CST')
    args = parser.parse_args(argv)
    if args.execute and not args.dispatch:
        parser.error('--execute requires --dispatch')
    config = load_config(args.config)
    if args.audit_seed:
        if args.execute or args.dispatch:
            parser.error('--audit-seed is offline and cannot dispatch')
        folder, plan = load_prepared(config, check_provenance=False)
        seal_seed(folder, plan, data.load_morris_seed(ROOT/config['morris_campaign']))
        print('Seed artifacts/metrics verified and provenance sealed; no simulation started.')
        return 0
    if args.dispatch:
        folder, plan = load_prepared(config)
        splits = ('train', 'holdout') if args.dispatch == 'all' else (args.dispatch,)
        commands = [princess_command(folder, plan, split) for split in splits]
        for command in commands:
            print(subprocess.list2cmdline(command))
        if args.execute:
            if not args.yes and input('Type RUN to launch CST: ').strip() != 'RUN':
                return 0
            with campaign_lock(folder):
                preflight_devices(plan)
                for split, command in zip(splits, commands):
                    others = [f'{plan["campaign_id"]}-{s}' for s in ('train', 'holdout') if s != split]
                    others += [common.read_json(p)['run_id'] for p in (folder/'batches').glob('batch_*/batch.json')]
                    for run_id in others:
                        if active_tasks(run_id):
                            raise RuntimeError(f'Another dispatch has pending/running tasks: {run_id}')
                    code = subprocess.call(command, cwd=ROOT)
                    if code:
                        return code
                return 0
        return 0
    folder, plan = prepare(config)
    print(f'OFFLINE preparation complete: {folder}; no CST or GP fit started.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
