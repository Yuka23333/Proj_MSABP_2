"""F5 prepares the 35D/16-trajectory campaign OFFLINE; never starts CST.

Random Morris OAT candidates with whole-trajectory manufacturing screening,
then greedy maximin selection from the surviving pool (not a global optimum).
An even p-level grid uses Delta=p/[2(p-1)]. No surrogate or penalty values.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.automation import cst_build_msabp_geometry_tree as builder  # noqa: E402
from scripts.optimization import morris_tree_common as common  # noqa: E402

CONFIG = ROOT / 'configs/optimization/morris_tree_35d.json'


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def bounds(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)
            or not value[0] < value[1]):
        raise ValueError(f'Invalid bounds: {name}')
    return list(value)


def variable_spec(config):
    absolute = bounds(config['absolute_factors'], 'absolute_factors')
    corners = bounds(config['corner_k_bounds'], 'corner_k_bounds')
    branch = bounds(config['branch_k_bounds'], 'branch_k_bounds')
    tree = {}
    for node_id in config['topology']:
        if node_id in tree:
            raise ValueError('Duplicate branch')
        parent, _, leaf = node_id.rpartition('/')
        tree[node_id] = {'parent': parent or 'SLOT', 'side': leaf[0], 'k': [.5, .5, .5]}
    if not tree:
        raise ValueError('Empty topology')
    baseline = builder.normalize_request({'tree': tree, 'build_options': config['build_options']})
    if baseline['build_options']['manufacturing_mode'] != 'repair':
        raise ValueError('This campaign requires the agreed manufacturing repair policy')
    variables = []
    for name, (kind, default) in builder.model.PARAM_SPECS.items():
        lo, hi = [default * x for x in absolute] if kind == 'abs' else corners
        variables.append({'name': name, 'kind': kind, 'lower': lo, 'upper': hi,
                          'parameter': name})
    for node in baseline['tree']:
        for index in range(3):
            variables.append({'name': f'{node}.K{index+1}', 'kind': 'k',
                              'lower': branch[0], 'upper': branch[1],
                              'node': node, 'k_index': index})
    overrides = config.get('range_overrides', {})
    if set(overrides) - {v['name'] for v in variables}:
        raise ValueError('Unknown range override')
    for v in variables:
        if v['name'] in overrides:
            v['lower'], v['upper'] = bounds(overrides[v['name']], v['name'])
        lo, hi = v['lower'], v['upper']
        if (v['kind'] == 'k' and not 0 <= lo < hi <= 1) or (v['kind'] == 'abs' and lo <= 0):
            raise ValueError(f'Bounds outside physical domain: {v["name"]}')
    return variables, baseline


def trajectory(d, levels, rng):
    """A complete random Morris trajectory; integer grid avoids accumulated drift."""
    if levels < 4 or levels % 2:
        raise ValueError('num_levels must be even and >=4')
    base = rng.integers(0, levels // 2, size=d)
    signs = rng.choice([-1, 1], size=d)
    current = base + (signs < 0) * (levels // 2)
    rows = [current.copy()]
    for index in rng.permutation(d):
        current[index] += signs[index] * (levels // 2)
        rows.append(current.copy())
    return np.array(rows, dtype=float) / (levels - 1)


def request_from_unit(plan, unit):
    request = deepcopy(plan['baseline_request'])
    for spec, x in zip(plan['variables'], unit, strict=True):
        value = float(spec['lower'] + x * (spec['upper'] - spec['lower']))
        if 'parameter' in spec:
            request['params'][spec['parameter']] = value
        else:
            request['tree'][spec['node']]['k'][spec['k_index']] = value
    return request


def simulation_key(geometry):
    # All other physical quantities/materials/ports/boundaries belong to the
    # frozen template + frozen builder. Copper alone is not a sufficient key.
    return common.digest({k: geometry[k] for k in (
        'manufactured_copper_sha256', 'substrate_bounds', 'reflector', 'quantum_mm')})


def select_tracks(tracks, count, old_points):
    """Greedy spread; distance is mean Euclidean distance over all point pairs."""
    arrays = [np.array([p['unit'] for p in track]) for track in tracks]
    old = {}
    for point in old_points:
        old.setdefault(point['trajectory_id'], []).append(point['unit'])
    def distance(a, b):
        return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).mean())
    matrix = np.array([[distance(a, b) for b in arrays] for a in arrays])
    scores = (np.array([min(distance(a, np.array(b)) for b in old.values()) for a in arrays])
              if old else matrix.mean(axis=1))
    remaining, chosen = list(range(len(tracks))), []
    for _ in range(count):
        pick = max(remaining, key=lambda i: (scores[i], -i))
        remaining.remove(pick)
        chosen.append(pick)
        scores = np.minimum(scores, matrix[:, pick]) if old or len(chosen) > 1 else matrix[:, pick]
    return sorted(chosen)


def create_plan(config, folder):
    variables, baseline = variable_spec(config)
    for name in ('initial_trajectories', 'num_levels', 'max_candidates_per_batch', 'max_attempts', 'candidate_pool_factor'):
        positive_int(config[name], name)
    trajectory(len(variables), config['num_levels'], np.random.default_rng(0))
    if not isinstance(config['seed'], int) or isinstance(config['seed'], bool) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,49}', config['campaign_id']):
        raise ValueError('Unsafe campaign_id')
    if not config['devices'] or not all(isinstance(x, str) and x for x in config['devices']):
        raise ValueError('Explicit devices required')
    band = bounds(config['band_ghz'], 'band_ghz')
    if not 2 <= band[0] < band[1] <= 6:
        raise ValueError('Metric band must be inside current 2-6 GHz template contract')
    quantum = config['coordinate_quantum_mm']
    g = builder.prepare_geometry(baseline, quantum=quantum)
    b = g['substrate_bounds']
    project = (ROOT / config['project']).resolve()
    # Read the whole source before reserving the campaign path.
    template = project.read_bytes()
    plan = {'schema_version': 1, 'campaign_id': config['campaign_id'],
            'config': config, 'variables': variables, 'baseline_request': baseline,
            'dimension': len(variables), 'num_levels': config['num_levels'],
            'delta': config['num_levels'] / (2 * (config['num_levels'] - 1)),
            'metrics': {'band_ghz': band, 'reference_area_mm2': (b[2]-b[0])*(b[3]-b[1])},
            'source_sha256': common.source_fingerprints(),
            'auxiliary_source_sha256': common.auxiliary_fingerprints(),
            'template_sha256': common.file_hash(project), 'template_source': str(project),
            'sampling': 'random Morris candidates; complete manufacturing-feasible trajectories; greedy maximin selection',
            'failure_policy': 'CST failures remain unresolved; never penalize or replace a trajectory'}
    plan['plan_sha256'] = common.digest(plan)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'template.cst').write_bytes(template)
    common.write_json(folder / 'plan.json', plan)
    return plan


def append_batch(folder, plan, old_points, batches, count):
    positive_int(count, 'trajectories')
    # Interrupted/offline-rejected preparations keep their audit directories;
    # reserve a fresh number rather than overwriting or blocking all extensions.
    numbers = [int(p.name[6:]) for p in (folder / 'batches').glob('batch_*')
               if p.is_dir() and p.name[6:].isdigit()]
    batch_id = max(numbers, default=0) + 1
    destination = folder / 'batches' / f'batch_{batch_id:04d}'
    destination.mkdir(parents=True, exist_ok=False)
    accepted, candidate_ids, attempts = [], [], 0
    pool_size = count * plan['config']['candidate_pool_factor']
    offset = len({p['trajectory_id'] for p in old_points})
    known = {p['simulation_key']: p['simulation_sample_id'] for p in old_points}
    candidates_path = destination / 'candidates.jsonl'
    with candidates_path.open('x', encoding='utf-8') as audit:
        for candidate in range(1, plan['config']['max_candidates_per_batch'] + 1):
            attempts = candidate
            rng = np.random.default_rng(np.random.SeedSequence([plan['config']['seed'], batch_id, candidate]))
            units = trajectory(plan['dimension'], plan['num_levels'], rng)
            evaluated, labels = [], []
            for step, unit in enumerate(units):
                request = request_from_unit(plan, unit)
                try:
                    geometry = builder.prepare_geometry(request, quantum=plan['config']['coordinate_quantum_mm'])
                except builder.model.manufacturing.ManufacturingError as exc:
                    labels.append({'valid': False, 'reason': str(exc), 'report': exc.report})
                    evaluated.append(None)
                    continue
                # Unexpected ValueError/GEOS/programming faults are not silently
                # relabelled as manufacturing rejection; preparation stops.
                b = geometry['substrate_bounds']
                evaluated.append({'step_index': step, 'unit': unit.tolist(), 'request': geometry['request'],
                                  'manufactured_copper_sha256': geometry['manufactured_copper_sha256'],
                                  'simulation_key': simulation_key(geometry),
                                  'substrate_area_mm2': (b[2]-b[0])*(b[3]-b[1])})
                labels.append({'valid': True, 'manufacturing': geometry['manufacturing']})
            good = all(row is not None for row in evaluated)
            audit.write(json.dumps({'candidate': candidate, 'unit': units.tolist(),
                                    'accepted': good, 'labels': labels}, allow_nan=False) + '\n')
            audit.flush()
            if good:
                accepted.append(evaluated)
                candidate_ids.append(candidate)
            print(f'[Morris] batch={batch_id} candidates={candidate} valid_pool={len(accepted)}/{pool_size}', flush=True)
            if len(accepted) == pool_size:
                break
    if len(accepted) < pool_size:
        common.write_json(destination / 'preparation_failure.json', {
            'requested': count, 'accepted': len(accepted), 'candidates': attempts,
            'reason': 'Candidate limit reached; review ranges. No partial batch can be dispatched.'})
        raise RuntimeError('Not enough complete manufacturable trajectories; inspect candidate audit')
    chosen = select_tracks(accepted, count, old_points)
    points = []
    for i, index in enumerate(chosen, start=offset+1):
        for p in accepted[index]:
            p.update(trajectory_id=i, sample_id=f'morris_t{i:06d}_p{p["step_index"]:02d}')
            p['simulation_sample_id'] = known.setdefault(p['simulation_key'], p['sample_id'])
            points.append(p)
    csv_path = destination / 'sample.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'simulation_mode', 'tree_json'])
        writer.writeheader()
        writer.writerows({'sample_id': p['sample_id'], 'simulation_mode': 'antenna_tree',
                          'tree_json': json.dumps(p['request'], separators=(',', ':'))}
                         for p in points if p['simulation_sample_id'] == p['sample_id'])
    batch = {'schema_version': 1, 'batch_id': batch_id, 'plan_sha256': plan['plan_sha256'],
             'trajectory_count': count, 'candidate_count': attempts, 'valid_pool_size': len(accepted),
             'selected_candidates': [candidate_ids[i] for i in chosen], 'points': points,
             'csv_sha256': common.file_hash(csv_path),
             'candidates_sha256': common.file_hash(candidates_path)}
    batch['batch_sha256'] = common.digest(batch)
    common.write_json(destination / 'batch.json', batch)
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--append-trajectories', type=int)
    args = parser.parse_args(argv)
    config = common.read_json(args.config)
    folder = args.campaign or ROOT / 'simulations/runs' / config['campaign_id']
    if (folder / 'plan.json').exists():
        plan, points, batches = common.load_campaign(folder, check_sources=True)
        if config != plan['config']:
            raise ValueError('Config changed; new campaign required')
        if args.append_trajectories is None and batches:
            print(f'Already prepared: {len(points)} trajectory positions; {folder}')
            return 0
    else:
        if args.append_trajectories is not None:
            raise ValueError('Prepare initial campaign before appending')
        plan = create_plan(config, folder)
        points, batches = [], []
    count = args.append_trajectories if args.append_trajectories is not None else config['initial_trajectories']
    batch = append_batch(folder, plan, points, batches, count)
    canonical = sum(p['simulation_sample_id'] == p['sample_id'] for p in batch['points'])
    print(f'Prepared {count} trajectories, {len(batch["points"])} positions, {canonical} new unique solves: {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
