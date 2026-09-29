"""Frozen Morris campaign contracts. No CST connection or solver imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = {
    'builder': 'scripts/automation/cst_build_msabp_geometry_tree.py',
    'model': 'scripts/geometry/shapely_antenna_tree_model.py',
    'manufacturing': 'scripts/geometry/manufacturing.py',
    'runner': 'src/msabp_opt/simulation/distributed/case_runner_tree.py',
}
AUXILIARY_FILES = (
    'scripts/optimization/prepare_morris_tree.py',
    'scripts/optimization/morris_tree_common.py',
    'scripts/automation/cst_build_msabp_geometry.py',
    'scripts/automation/cst_generate_polygen.py',
    'scripts/automation/cst_run_and_export_s11.py',
    'src/msabp_opt/simulation/distributed/case_runner.py',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    """Exclusive creation: existing experiment artifacts are never overwritten."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_fingerprints():
    return {k: source_hash(ROOT / p) for k, p in SOURCE_FILES.items()}


def auxiliary_fingerprints():
    return {p: source_hash(ROOT / p) for p in AUXILIARY_FILES}


def load_campaign(folder, *, check_sources=False):
    folder = Path(folder)
    plan = read_json(folder / 'plan.json')
    stored = plan['plan_sha256']
    if digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) != stored:
        raise ValueError('Frozen plan changed; start a new campaign')
    if file_hash(folder / 'template.cst') != plan['template_sha256']:
        raise ValueError('Frozen CST template changed')
    if check_sources and (source_fingerprints() != plan['source_sha256'] or
                          auxiliary_fingerprints() != plan['auxiliary_source_sha256']):
        raise ValueError('Geometry/simulation source changed since preparation; use a new campaign')
    points, batches = [], []
    for path in sorted((folder / 'batches').glob('batch_*/batch.json')):
        batch = read_json(path)
        if batch['plan_sha256'] != stored or digest({k: v for k, v in batch.items()
                                                   if k != 'batch_sha256'}) != batch['batch_sha256']:
            raise ValueError(f'Frozen batch changed: {path}')
        if file_hash(path.parent / 'sample.csv') != batch['csv_sha256']:
            raise ValueError(f'Frozen batch CSV changed: {path}')
        if file_hash(path.parent / 'candidates.jsonl') != batch['candidates_sha256']:
            raise ValueError(f'Frozen candidate audit changed: {path}')
        points.extend(batch['points'])
        batches.append(batch)
    ids = [p['sample_id'] for p in points]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample identifiers')
    return plan, points, batches
