"""F5: isolated local FD sanity check using the exact frozen 008 samples.

No Princess dispatch, no solver switching, no edits to the source CST file.
The first failure stops the batch and is written to failure.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.simulation import 本地仿真计时_tree as timing  # noqa: E402
from msabp_opt.simulation.distributed import case_runner_tree as runner  # noqa: E402

# ==================== F5 RUN: edit these settings ====================
RUN_ID = 'local-sanity-fd-tree-002'
SOURCE_RUN = ROOT / 'simulations/runs/local-timing-tree-8-008'
PROJECT = SOURCE_RUN / 'template.cst'  # frozen FD template, not live master
COUNT = 8  # first N original samples; no resampling
FD_SOLVER = 'HF Frequency Domain'


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        raise ValueError('Unsafe run ID')
    with (args.source_run / 'sample.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not 1 <= args.count <= len(rows):
        raise ValueError('count must be between 1 and the source sample count')
    rows = rows[:args.count]
    samples = []
    for row in rows:
        if row['simulation_mode'] != 'antenna_tree':
            raise ValueError('Expected antenna_tree samples')
        request = runner.builder.normalize_request(json.loads(row['tree_json']))
        runner.builder.prepare_geometry(request)
        samples.append({'sample_id': row['sample_id'], 'parameters': request})
    if [s['sample_id'] for s in samples] != [f'sample_{i:03d}' for i in range(1, args.count+1)]:
        raise ValueError('Expected consecutive sample_001 ... IDs')
    sha = timing.validation.comparison.sha256
    plan = {'run_id': args.run_id, 'count': len(rows), 'samples': samples,
            'expected_solver_name': FD_SOLVER, 'source_run': str(args.source_run.resolve()),
            'source_csv_sha256': sha(args.source_run / 'sample.csv'),
            'template_source': str(args.project.resolve()), 'template_sha256': sha(args.project)}
    folder = ROOT / 'simulations/runs' / args.run_id
    if folder.exists():
        if json.loads((folder / 'timing_plan.json').read_text()) != plan:
            raise ValueError('Plan changed; use a new --run-id')
        if sha(folder / 'template.cst') != plan['template_sha256']:
            raise ValueError('Frozen template changed')
    else:
        folder.mkdir(parents=True)
        shutil.copy2(args.project, folder / 'template.cst')
        with (folder / 'sample.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (folder / 'timing_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan, folder, rows


def run(plan, folder, rows):
    work = folder / 'work'
    if work.exists():
        raise FileExistsError('This run was already started; use a new --run-id (no automatic retry)')
    work.mkdir()
    project_path = work / 'msa-bp.cst'
    shutil.copy2(folder / 'template.cst', project_path)
    project = None
    case_id = None
    stage = 'opening_project'

    def notify(value):
        nonlocal stage
        stage = value
        message = f'[FD] {case_id or "preflight"}: {value}'
        print(message, flush=True)
        with (folder / 'fd.log').open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')

    try:
        notify(stage)
        project = runner.exports.open_cst_project(str(project_path))
        notify('checking_fd_template')
        setup = runner.exports.inspect_recorded_simulation_setup(project, expected_solver_name=FD_SOLVER)
        if setup.solver_running:
            raise RuntimeError('Solver already running')
        for row in rows:
            case_id = row['sample_id']
            runner.run_csv_row(row, project_path=project_path, project=project,
                              output_root=folder / 'results/local',
                              expected_solver_name=FD_SOLVER, stage_callback=notify)
    except Exception as exc:
        failure = {'case_id': case_id, 'stage': stage, 'error': str(exc),
                   'traceback': traceback.format_exc()}
        (folder / 'failure.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        print(f'[FD] stopped: {exc}', flush=True)
        return 2
    finally:
        if project is not None:
            project.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--count', type=int, default=COUNT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)
    plan, folder, rows = prepare(args)
    print('FD sanity check:', folder)
    if args.prepare_only:
        return 0
    if args.summarize_only:
        return timing.summarize(plan, folder)
    print(f'{len(rows)} local FD solves. Close other local CST tasks first. FFS export must be enabled in the template.')
    if not args.yes and input('Type RUN to start: ').strip() != 'RUN':
        return 0
    code = run(plan, folder, rows)
    summary_code = timing.summarize(plan, folder)
    return code or summary_code


if __name__ == '__main__':
    raise SystemExit(main())
