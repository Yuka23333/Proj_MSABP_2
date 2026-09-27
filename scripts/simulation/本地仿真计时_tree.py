"""F5: eight lightly perturbed local tree solves; retain exports and timing."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import random
from copy import deepcopy
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.simulation import 基础仿真文件验证_tree as validation  # noqa: E402

RUN_ID = 'local-timing-tree-8-011'
REPEATS = 8
CASE = 'case3'
REQUEST_JSON = None
RANDOM_SEED = 20260926
ABSOLUTE_RELATIVE_DELTA = .02
K_DELTA = .05


def draw_samples(raw, count, seed):
    baseline = validation.builder.normalize_request(raw)
    rng = random.Random(seed)
    samples, signatures = [], set()
    for _ in range(count*128):
        request = deepcopy(baseline)
        for name, value in request['params'].items():
            if validation.builder.model.PARAM_SPECS[name][0] == 'k':
                request['params'][name] = rng.uniform(max(0,value-K_DELTA), min(1,value+K_DELTA))
            else:
                request['params'][name] = value*rng.uniform(1-ABSOLUTE_RELATIVE_DELTA,1+ABSOLUTE_RELATIVE_DELTA)
        for node in request['tree'].values():
            node['k'] = [rng.uniform(max(0,v-K_DELTA), min(1,v+K_DELTA)) for v in node['k']]
        try:
            geometry = validation.builder.prepare_geometry(request)
        except ValueError:
            continue
        signature = geometry['manufactured_copper_sha256']
        if signature in signatures:
            continue
        signatures.add(signature)
        samples.append({'sample_id':f'sample_{len(samples)+1:03d}', 'parameters':geometry['request']})
        if len(samples) == count:
            return baseline, samples
    raise ValueError('Could not find enough distinct valid perturbed geometries')


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        raise ValueError('Unsafe run ID')
    if args.count < 1:
        raise ValueError('count must be positive')
    local = validation.comparison.load_device_registry(args.devices_config).get_device('local')
    if local.launch_mode.value != 'local':
        raise ValueError('local device must use local launch mode')
    raw = json.loads(args.request.read_text(encoding='utf-8-sig')) if args.request else validation.case_request(args.case)
    baseline, samples = draw_samples(raw, args.count, args.seed)
    plan = {'run_id': args.run_id, 'count': args.count, 'device': 'local',
            'baseline': baseline, 'samples': samples, 'seed':args.seed,
            'absolute_relative_delta':ABSOLUTE_RELATIVE_DELTA, 'k_delta':K_DELTA,
            'devices_config': str(args.devices_config.resolve()),
            'devices_config_sha256': validation.comparison.sha256(args.devices_config),
            'template_source': str(args.project.resolve()),
            'template_sha256': validation.comparison.sha256(args.project)}
    folder = ROOT/'simulations/runs'/args.run_id
    rows = [{'sample_id': s['sample_id'], 'simulation_mode': 'antenna_tree',
             'tree_json': json.dumps(s['parameters'], separators=(',', ':'))} for s in samples]
    if (folder/'timing_plan.json').exists():
        if json.loads((folder/'timing_plan.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('Changed plan; use a new --run-id')
        if validation.comparison.sha256(folder/'template.cst') != plan['template_sha256']:
            raise ValueError('Frozen template changed')
        with (folder/'sample.csv').open(encoding='utf-8', newline='') as stream:
            if list(csv.DictReader(stream)) != rows:
                raise ValueError('Frozen sample CSV changed')
    else:
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        (folder/'template.cst').write_bytes(args.project.read_bytes())
        with (folder/'sample.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (folder/'timing_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan, folder


def summarize(plan, folder):
    rows, errors, curves = [], {}, []
    for i in range(1, plan['count']+1):
        sample = plan['samples'][i-1]
        case_id = sample['sample_id']
        case = folder/'results/local'/f'case_{case_id}'
        try:
            manifest = json.loads((case/'manifest.json').read_text(encoding='utf-8'))
            if (manifest.get('status') != 'completed' or manifest.get('dry_run') is not False
                    or manifest.get('simulation_mode') != 'antenna_tree'
                    or manifest.get('case_id') != case_id or manifest.get('parameters') != sample['parameters']):
                raise ValueError('Incomplete or mismatched real simulation')
            for key, filename in [('s11','S11.csv'),('farfield_source','Farfield Source [1].ffs')]:
                path = case/filename
                record = manifest.get('artifacts', {}).get(key, {})
                if not path.stat().st_size or record.get('path') != filename or record.get('sha256') != validation.comparison.sha256(path):
                    raise ValueError(f'Missing/invalid artifact: {filename}')
            solver = float(manifest['stage_seconds']['solving'])
            total = float(manifest['elapsed_seconds'])
            if not all(math.isfinite(v) and v > 0 for v in (solver, total)) or solver > total:
                raise ValueError('Invalid timing')
            curve = validation.comparison.read_curve(case/'S11.csv')
            rows.append({'repeat': i, 'solver_seconds': solver, 'case_total_seconds': total,
                         's11': str(case/'S11.csv'), 'ffs': str(case/'Farfield Source [1].ffs')})
            curves.append((case_id, curve))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[case_id] = str(exc)
    def stats(values):
        return {'n':len(values), 'mean_seconds':statistics.mean(values),
                'median_seconds':statistics.median(values),
                'std_seconds':statistics.stdev(values) if len(values)>1 else 0.0,
                'min_seconds':min(values), 'max_seconds':max(values)} if values else None
    report = {'planned':plan['count'], 'completed':len(rows), 'complete':len(rows)==plan['count'],
              'errors': errors, 'solver':stats([r['solver_seconds'] for r in rows]),
              'case_total':stats([r['case_total_seconds'] for r in rows]),
              'solver_excluding_first':stats([r['solver_seconds'] for r in rows if r['repeat'] != 1]),
              'note':'Different perturbed geometries, not a repeatability test. Mean includes run 1. Partial statistics use completed valid cases only. '
                     'Case total includes geometry/result cleanup, build, solve, export; excludes Maid launch and upload.'}
    (folder/'timing_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (folder/'timings.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['repeat','solver_seconds','case_total_seconds','s11','ffs'])
        writer.writeheader()
        writer.writerows(rows)
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
        for case_id, (x,y) in curves:
            ax.plot(x,y,label=case_id,alpha=.75)
        ax.set(xlabel='Frequency (GHz)',ylabel='S11 (dB)',title='Local benchmark: lightly perturbed tree samples')
        ax.grid(alpha=.3)
        ax.legend(ncol=2)
        fig.savefig(folder/'S11_samples.png',dpi=160)
        plt.close(fig)
    print(json.dumps(report,indent=2))
    return 0 if report['complete'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default=RUN_ID)
    parser.add_argument('--count',type=int,default=REPEATS)
    parser.add_argument('--seed',type=int,default=RANDOM_SEED)
    parser.add_argument('--case',choices=('default','case1','case2','case3'),default=CASE)
    parser.add_argument('--request',type=Path,default=REQUEST_JSON)
    parser.add_argument('--project',type=Path,default=validation.comparison.PROJECT)
    parser.add_argument('--devices-config',type=Path,default=validation.comparison.DEVICE_CONFIG)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--yes',action='store_true')
    args = parser.parse_args(argv)
    plan,folder = prepare(args)
    print('Local benchmark:',folder)
    if args.prepare_only:
        return 0
    if not args.summarize_only:
        print(f"{plan['count']} real CST solves on local ONLY. Close other local CST/Princess tasks first.")
        if not args.yes and input('Type RUN to start: ').strip() != 'RUN':
            return 0
        command = validation.comparison.princess_command(plan,'local',folder)
        code = validation.comparison.run_and_tee(command,folder/'princess.log')
        if code:
            print('Princess exit:',code)
    return summarize(plan,folder)


if __name__ == '__main__':
    raise SystemExit(main())
