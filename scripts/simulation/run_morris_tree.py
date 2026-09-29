"""F5 dispatches frozen Morris points through Princess, after explicit RUN.

Exhausted CST failures remain unresolved. Reuse a dispatch ID to resume; use a
new dispatch ID to retry unresolved points without touching previous run state.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from scripts.optimization import morris_tree_common as common  # noqa: E402
from scripts.postprocessing.analyze_morris_tree import collect_results  # noqa: E402
from scripts.simulation.run_remote_real_smoke import run_and_tee  # noqa: E402
from msabp_opt.simulation.distributed.config import load_device_registry  # noqa: E402
from msabp_opt.simulation.distributed.transport import run_remote_powershell, _ps_literal  # noqa: E402

CAMPAIGN = ROOT / 'simulations/runs/morris-tree-35d-016-001'
DISPATCH_ID = 'd002'  # Keep for resume; use a new ID for exhausted-point retries.


def active_tasks(run_id):
    path = ROOT / 'simulations/runs' / run_id / 'princess.sqlite3'
    if not path.exists():
        return 0
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        running = db.execute("SELECT count(*) FROM tasks WHERE run_id=? AND status='running'",
                             (run_id,)).fetchone()[0]
        stopped = db.execute('SELECT stop_requested FROM run_control WHERE run_id=?', (run_id,)).fetchone()
        pending = db.execute("SELECT count(*) FROM tasks WHERE run_id=? AND status='pending'",
                             (run_id,)).fetchone()[0]
        return running + (0 if stopped and stopped[0] else pending)


def require_other_dispatches_idle(folder, dispatch_id):
    for path in (folder / 'dispatches').glob('*/dispatch.json'):
        if path.parent.name != dispatch_id:
            old = common.read_json(path)
            if active_tasks(old['run_id']):
                raise RuntimeError('Previous dispatch has running/pending tasks; resume or explicitly stop it first')


@contextmanager
def campaign_lock(folder):
    """OS-released advisory lock serializes this wrapper, including same-ID resume."""
    with (folder / '.dispatch.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another Morris launcher is using this campaign') from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare_dispatch(folder, dispatch_id, devices=None):
    folder = Path(folder).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,19}', dispatch_id):
        raise ValueError('Unsafe dispatch ID')
    plan, points, _ = common.load_campaign(folder, check_sources=True)
    if not points:
        raise ValueError('No complete prepared trajectories')
    device_ids = devices or plan['config']['devices']
    registry_path = (ROOT / plan['config']['devices_config']).resolve()
    registry = load_device_registry(registry_path)
    for device_id in device_ids:
        registry.get_device(device_id)
    destination = folder / 'dispatches' / dispatch_id
    existing = destination / 'dispatch.json'
    require_other_dispatches_idle(folder, dispatch_id)
    if existing.exists():
        dispatch = common.read_json(existing)
        if (common.digest({k: v for k, v in dispatch.items() if k != 'dispatch_sha256'}) != dispatch['dispatch_sha256'] or
                dispatch['run_id'] != f'{plan["campaign_id"]}-{dispatch_id}' or
                dispatch['dispatch_id'] != dispatch_id or dispatch['devices_config'] != str(registry_path) or
                dispatch['plan_sha256'] != plan['plan_sha256'] or dispatch['devices'] != device_ids or
                dispatch['devices_config_sha256'] != common.file_hash(registry_path) or
                dispatch['csv_sha256'] != common.file_hash(destination / 'sample.csv')):
            raise ValueError('Frozen dispatch/config changed; do not overwrite it')
        by_id = {p['sample_id']: p for p in points}
        expected = [{'sample_id': sid, 'simulation_mode': 'antenna_tree',
                     'tree_json': json.dumps(by_id[sid]['request'], separators=(',', ':'))}
                    for sid in dispatch['sample_ids']]
        with (destination / 'sample.csv').open(encoding='utf-8', newline='') as stream:
            if list(csv.DictReader(stream)) != expected:
                raise ValueError('Dispatch rows differ from frozen trajectories')
        return plan, dispatch, destination
    values, issues = collect_results(plan, points, [folder / 'results'])
    pending = [p for p in points if p['sample_id'] == p['simulation_sample_id'] and p['sample_id'] not in values]
    if not pending:
        raise ValueError('All planned points already have verified results; nothing to dispatch')
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / 'sample.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'simulation_mode', 'tree_json'])
        writer.writeheader()
        writer.writerows({'sample_id': p['sample_id'], 'simulation_mode': 'antenna_tree',
                          'tree_json': json.dumps(p['request'], separators=(',', ':'))} for p in pending)
    dispatch = {'schema_version': 1, 'plan_sha256': plan['plan_sha256'],
                'run_id': f'{plan["campaign_id"]}-{dispatch_id}',
                'dispatch_id': dispatch_id, 'devices': device_ids,
                'devices_config': str(registry_path), 'devices_config_sha256': common.file_hash(registry_path),
                'csv_sha256': common.file_hash(csv_path), 'sample_ids': [p['sample_id'] for p in pending],
                'result_issues_at_creation': issues,
                'failure_policy': 'Keep missing; retry these same points with a new dispatch ID, never replace trajectory'}
    dispatch['dispatch_sha256'] = common.digest(dispatch)
    common.write_json(existing, dispatch)
    return plan, dispatch, destination


def probe_device(device, expected):
    """Read-only source gate using the same conda DLL search path as Maid Bell.

    An absolute python.exe path alone does not activate conda on Windows. Only
    this probe process receives the environment; no user/system setting changes.
    """
    code = ('import hashlib,json,pathlib,sys; '
            'sys.dont_write_bytecode=True; '
            f'root=pathlib.Path({device.repo_root!r}); '
            'sys.path[:0]=[str(root),str(root/"src")]; '
            'from msabp_opt.simulation.distributed import case_runner_tree as r; '
            f'paths={list(expected)!r}; '
            'print(json.dumps({p:hashlib.sha256((root/p).read_text(encoding="utf-8").encode()).hexdigest() for p in paths}))')
    if device.is_remote:
        prefix = PureWindowsPath(device.python_path).parent
        path_prefix = ';'.join(str(p) for p in (prefix, prefix / 'Library/bin', prefix / 'Scripts')) + ';'
        encoded = base64.b64encode(code.encode()).decode()
        bootstrap = f"import base64;exec(base64.b64decode('{encoded}'))"
        script = (
            "$ErrorActionPreference='Stop'\n$ProgressPreference='SilentlyContinue'\n"
            f"$env:CONDA_PREFIX={_ps_literal(str(prefix))}\n"
            f"$env:PATH={_ps_literal(path_prefix)}+$env:PATH\n"
            f"Set-Location -LiteralPath {_ps_literal(device.repo_root)}\n"
            f"& {_ps_literal(device.python_path)} -B -u -c {_ps_literal(bootstrap)}\nexit $LASTEXITCODE"
        )
        output = run_remote_powershell(device, script, action='Morris source preflight', timeout=60).stdout
    else:
        prefix = Path(device.python_path).parent
        environment = os.environ.copy()
        environment['CONDA_PREFIX'] = str(prefix)
        environment['PATH'] = os.pathsep.join((str(prefix), str(prefix / 'Library/bin'),
                                              str(prefix / 'Scripts'), environment.get('PATH', '')))
        output = subprocess.run([device.python_path, '-B', '-u', '-c', code],
                                cwd=device.repo_root, env=environment, check=True,
                                capture_output=True, text=True, timeout=60).stdout
    actual = json.loads(output.strip().splitlines()[-1])
    if actual != expected:
        different = [k for k in expected if actual.get(k) != expected[k]]
        raise RuntimeError(f'{device.id}: source mismatch, synchronize before CST: {different}')
    return actual


def princess_command(folder, plan, dispatch, destination):
    command = [sys.executable, str(ROOT / 'scripts/simulation/princess.py'), 'start',
               '--csv', str(destination / 'sample.csv'), '--run-id', dispatch['run_id'],
               '--project', str(folder / 'template.cst'),
               '--results-root', str(folder / 'results' / dispatch['dispatch_id']),
               '--devices-config', dispatch['devices_config'],
               '--coordinate-quantum-mm', str(plan['config']['coordinate_quantum_mm']),
               '--max-attempts', str(plan['config']['max_attempts'])]
    for device_id in dispatch['devices']:
        command += ['--device', device_id]
    return command


def run(args):
    folder = args.campaign.resolve()
    plan, dispatch, destination = prepare_dispatch(folder, args.dispatch_id, args.device)
    command = princess_command(folder, plan, dispatch, destination)
    print(subprocess.list2cmdline(command))
    print(f'{len(dispatch["sample_ids"])} unique points; failures remain unresolved, not penalized.')
    if args.prepare_only:
        return 0
    if not args.preflight_only and not args.yes and input('Type RUN to start real CST solves: ').strip() != 'RUN':
        return 0
    registry = load_device_registry(dispatch['devices_config'])
    expected = {common.SOURCE_FILES[k]: v for k, v in plan['source_sha256'].items()}
    expected.update(plan['auxiliary_source_sha256'])
    # Sampling modules need not be deployed on Maids; physical implementation must.
    expected = {k: v for k, v in expected.items() if not k.startswith('scripts/optimization/')}
    proof = {device: probe_device(registry.get_device(device), expected) for device in dispatch['devices']}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    common.write_json(destination / f'preflight_{stamp}.json', proof)
    if args.preflight_only:
        print('Morris source preflight PASS: ' + ', '.join(dispatch['devices']) + '; no CST solve started.')
        return 0
    require_other_dispatches_idle(folder, args.dispatch_id)
    code = run_and_tee(command, destination / 'princess.log')
    if code:
        print('Campaign incomplete. Inspect infrastructure/CST errors; retry original points via a new dispatch ID.')
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--dispatch-id', default=DISPATCH_ID)
    parser.add_argument('--device', action='append')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true', help='check device imports/source hashes, without starting CST')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)
    with campaign_lock(args.campaign.resolve()):
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
