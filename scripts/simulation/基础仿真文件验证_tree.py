"""F5: same arbitrary-tree sample on local/G5/G2, using the opt-in tree Maid route."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.simulation import 基础仿真文件验证 as comparison  # noqa: E402
from scripts.automation import cst_build_msabp_geometry_tree as builder  # noqa: E402

RUN_ID = 'base-validation-tree-001'
CASE = 'case3'
REQUEST_JSON = None  # {params, tree, build_options}; None uses CASE below.


def case_request(name):
    tree = builder.model.default_tree()
    if name != 'default':
        tree['U1']['k'] = [.15, 0 if name == 'case2' else .3, .8]
        builder.model.add_branch(tree, 'U1', 'L', (.7, .3, 1))
        if name == 'case3':
            builder.model.add_branch(tree, 'U1', 'L', (.3, .2, 1))
    return builder.normalize_request({'tree': tree, 'build_options': {'snap_fraction': .1, 'tip_clearance': 1.0}})


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        raise ValueError('Unsafe run ID')
    import math
    if not all(math.isfinite(v) for v in [*args.band, args.tolerance_db]) or not args.band[0] < args.band[1] or args.tolerance_db < 0:
        raise ValueError('Invalid band/tolerance')
    request = (json.loads(args.request.read_text(encoding='utf-8-sig')) if args.request else case_request(args.case))
    prepared = builder.prepare_geometry(request)
    request = prepared['request']
    registry = comparison.load_device_registry(args.devices_config)
    for device in comparison.DEVICE_IDS:
        registry.get_device(device)
    plan = {'schema_version': 1, 'run_id': args.run_id, 'geometry_route': 'antenna_tree',
            'devices': list(comparison.DEVICE_IDS), 'parameters': request,
            'devices_config': str(args.devices_config.resolve()),
            'devices_config_sha256': comparison.sha256(args.devices_config),
            'template_source': str(args.project.resolve()), 'template_sha256': comparison.sha256(args.project),
            'band_ghz': list(args.band), 'tolerance_db': args.tolerance_db}
    folder = ROOT/'simulations/runs'/args.run_id
    plan_path = folder/'validation_plan.json'
    row = {'sample_id': 'reference', 'simulation_mode': 'antenna_tree',
           'tree_json': json.dumps(request, separators=(',', ':'))}
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan changed; use a new --run-id')
        if comparison.sha256(folder/'template.cst') != plan['template_sha256']:
            raise ValueError('Frozen template changed')
        with (folder/'sample.csv').open(encoding='utf-8', newline='') as stream:
            if list(csv.DictReader(stream)) != [row]:
                raise ValueError('Frozen sample changed')
    else:
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        (folder/'template.cst').write_bytes(args.project.read_bytes())
        with (folder/'sample.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (folder/'geometry_tree.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan, folder


def compare(plan, folder):
    status = comparison.compare(plan, folder)
    report_path = folder/'comparison.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    fingerprints = {}
    for device in comparison.DEVICE_IDS:
        try:
            case = folder/'results'/device/'case_reference'
            manifest = json.loads((case/'manifest.json').read_text(encoding='utf-8'))
            if manifest['simulation_mode'] != 'antenna_tree':
                raise ValueError('Wrong geometry route')
            if comparison.sha256(case/'geometry_tree.json') != manifest['geometry_sha256']:
                raise ValueError('Geometry audit hash mismatch')
            fingerprints[device] = {k: manifest[k] for k in (
                'geometry_sha256', 'tree_request_sha256', 'source_sha256', 'baseline_history_sha256')}
            expected = hashlib.sha256(json.dumps(plan['parameters'], sort_keys=True).encode()).hexdigest()
            if fingerprints[device]['tree_request_sha256'] != expected:
                raise ValueError('Request fingerprint mismatch')
        except (OSError, ValueError, KeyError) as exc:
            report['errors'][device+'/tree'] = str(exc)
    if len(fingerprints) == 3 and any(v != next(iter(fingerprints.values())) for v in fingerprints.values()):
        report['errors']['tree_provenance'] = 'Three machines used different geometry or source code'
    report['tree_fingerprints'] = fingerprints
    report['passed'] = bool(report['passed'] and not report['errors'])
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Tree validation:', 'PASS' if report['passed'] else 'FAIL', report['errors'])
    return status if report['passed'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--case', choices=('default', 'case1', 'case2', 'case3'), default=CASE)
    parser.add_argument('--request', type=Path, default=REQUEST_JSON)
    parser.add_argument('--project', type=Path, default=comparison.PROJECT)
    parser.add_argument('--devices-config', type=Path, default=comparison.DEVICE_CONFIG)
    parser.add_argument('--band', nargs=2, type=float, default=comparison.BAND_GHZ)
    parser.add_argument('--tolerance-db', type=float, default=comparison.TOLERANCE_DB)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)
    plan, folder = prepare(args)
    print('Tree plan:', folder)
    if args.prepare_only:
        return 0
    if not args.compare_only:
        print('Three real CST solves using antenna_tree; close other local CST tasks first.')
        print('Pull the new dispatcher/builder/runner on BOTH Maids before starting.')
        if not args.yes and input('Type RUN to start: ').strip() != 'RUN':
            return 0
        for device in comparison.DEVICE_IDS:
            code = comparison.run_and_tee(comparison.princess_command(plan, device, folder), folder/f'{device}.log')
            if code:
                print(device, 'Princess exit:', code)
    return compare(plan, folder)


if __name__ == '__main__':
    raise SystemExit(main())
