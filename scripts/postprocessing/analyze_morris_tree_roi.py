"""Apply the historical frozen RoI to completed tree Morris FFS, without CST.

Separate outputs preserve the original four-metric report. The three objectives
are worst linear |S11|, negative RoI radiation gain dBi, and normalized area.
Linear RoI gain is retained as a scale-sensitivity diagnostic, not a fourth goal.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from scripts.postprocessing import analyze_morris_tree as base  # noqa: E402
from msabp_opt.optimization import phase2_krvea_data as legacy  # noqa: E402

OBJECTIVES = ('worst_s11_linear', legacy.ROI_GAIN_LOSS_DBI_COLUMN, 'normalized_substrate_area')
METRICS = (*OBJECTIVES, legacy.ROI_GAIN_LINEAR_COLUMN)


def roi_job(task):
    sample_id, manifest_paths, cache_directory = task
    fingerprints = set()
    selected = None
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        manifest = base._json(path)
        record = manifest['artifacts']['farfield_source']
        ffs = base._artifact(path.parent, record, 'farfield_source')
        fingerprints.add(record['sha256'])
        selected = ffs, manifest
    if len(fingerprints) != 1 or selected is None:
        raise ValueError(f'{sample_id}: ambiguous/missing FFS across accepted attempts')
    ffs, manifest = selected
    linear, dbi, hit = legacy.roi_radiation_gain_scalar(
        ffs, manifest=manifest, cache_directory=cache_directory,
    )
    return sample_id, {
        legacy.ROI_GAIN_LINEAR_COLUMN: linear, legacy.ROI_GAIN_DBI_COLUMN: dbi,
        legacy.ROI_GAIN_LOSS_DBI_COLUMN: -dbi, 'ffs_sha256': next(iter(fingerprints)),
        'ffs_path': str(ffs), 'cache_hit': hit,
    }


def sensitivity(plan, points, values, *, resamples=2000, seed=20260929):
    """All-or-nothing complete trajectories; same estimator as original report."""
    if resamples < 2:
        raise ValueError('At least two bootstrap resamples required')
    groups = defaultdict(list)
    for point in points:
        groups[point['trajectory_id']].append(point)
    matrices, effects, coverage = [], [], defaultdict(int)
    for trajectory_id, track in groups.items():
        track = sorted(track, key=lambda p: p['step_index'])
        steps = base.validate_trajectory(track, len(plan['variables']), plan['delta'], plan['num_levels'])
        outputs = np.array([[values[p['sample_id']][m] for m in METRICS] for p in track])
        if not np.isfinite(outputs).all():
            raise ValueError('Nonfinite output; no penalty or partial trajectory allowed')
        matrix = np.empty((len(plan['variables']), len(METRICS)))
        for before, after, step, delta_y in zip(track, track[1:], steps, np.diff(outputs, axis=0)):
            dimension = np.flatnonzero(np.abs(step) > 1e-12).item()
            matrix[dimension] = delta_y / step[dimension]
            if before['simulation_key'] != after['simulation_key']:
                coverage[plan['variables'][dimension]['name']] += 1
        matrices.append(matrix)
        for dimension, variable in enumerate(plan['variables']):
            effects.append({'trajectory_id': trajectory_id, 'variable': variable['name'],
                            **dict(zip(METRICS, matrix[dimension].tolist(), strict=True))})
    data = np.asarray(matrices)
    if len(data) < 2:
        raise ValueError('Need at least two complete trajectories for this report')
    rng = np.random.default_rng(seed)
    samples = {'mu': [], 'mu_star': [], 'sigma': []}
    for _ in range(resamples):
        draw = data[rng.integers(0, len(data), size=len(data))]
        samples['mu'].append(draw.mean(axis=0))
        samples['mu_star'].append(np.abs(draw).mean(axis=0))
        samples['sigma'].append(draw.std(axis=0, ddof=1))
    estimates = {'mu': data.mean(axis=0), 'mu_star': np.abs(data).mean(axis=0),
                 'sigma': data.std(axis=0, ddof=1)}
    intervals = {k: np.percentile(v, [2.5, 97.5], axis=0) for k, v in samples.items()}
    indices = []
    for j, metric in enumerate(METRICS):
        order = sorted(range(len(plan['variables'])), key=lambda i: (-estimates['mu_star'][i, j], i))
        for rank, i in enumerate(order, 1):
            name = plan['variables'][i]['name']
            row = {'metric': metric, 'variable': name, 'rank': rank, 'n_effective': len(data),
                   'geometry_changes': coverage[name]}
            for stat, estimate in estimates.items():
                row[stat] = float(estimate[i, j])
                row[stat + '_ci95_low'] = float(intervals[stat][0, i, j])
                row[stat + '_ci95_high'] = float(intervals[stat][1, i, j])
            indices.append(row)
    return indices, effects


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_report(output, indices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles = ['Worst linear |S11|', 'Negative frozen-RoI radiation gain (dBi)',
              'Normalized substrate area', 'Linear RoI gain (scale diagnostic only)']
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), layout='constrained')
    for ax, metric, title in zip(axes.flat, METRICS, titles):
        rows = [r for r in indices if r['metric'] == metric][:12 if 'area' not in metric else 7]
        means = np.array([r['mu_star'] for r in rows])
        errors = np.array([[r['mu_star']-r['mu_star_ci95_low'] for r in rows],
                           [r['mu_star_ci95_high']-r['mu_star'] for r in rows]])
        ax.barh(range(len(rows)), means, xerr=errors, capsize=3, color='#49799c')
        ax.set_yticks(range(len(rows)), [r['variable'] for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc='left')
        ax.set_xlabel('Morris mu*; 95% trajectory-bootstrap interval')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Frozen historical RoI | 3.6 GHz | theta 55-85 deg | phi 40-140 deg\n'
                 'E-theta radiation gain: linear spatial average, then dBi; no refitting', fontsize=14)
    fig.savefig(output / 'sensitivity_roi.png', dpi=160)
    plt.close(fig)


def run(args):
    campaign = args.campaign.resolve()
    output = campaign / 'analysis_roi'
    output.mkdir(exist_ok=True)
    plan, points = base.load_campaign(campaign)
    values, issues = base.collect_results(plan, points, [campaign / 'results'])
    if len(values) != len(points) or issues:
        raise ValueError(f'Base result audit incomplete: {len(values)}/{len(points)}; {len(issues)} issues')
    jobs = [(sid, v['result_manifests'], str(output / 'roi_cache'))
            for sid, v in values.items() if not v['reused_simulation']]
    roi_values = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(roi_job, job): job[0] for job in jobs}
        for future in as_completed(futures):
            sid, value = future.result()
            roi_values[sid] = value
            if len(roi_values) % 20 == 0 or len(roi_values) == len(jobs):
                print(f'RoI: {len(roi_values)}/{len(jobs)} verified FFS', flush=True)
    for value in values.values():
        value.update(roi_values[value['simulation_sample_id']])
    indices, effects = sensitivity(plan, points, values)
    metric_contract = base._json(ROOT / 'configs/optimization/phase2_krvea_roi_radiation_gain_64.json')['phase2_metric']
    assert metric_contract['frequency_ghz'] == legacy.ROI_FREQUENCY_GHZ
    assert metric_contract['theta_bounds_deg'] == list(legacy.ROI_THETA_BOUNDS_DEG)
    assert metric_contract['phi_center_deg'] == legacy.ROI_PHI_CENTER_DEG
    assert metric_contract['phi_half_width_deg'] == legacy.ROI_PHI_HALF_WIDTH_DEG
    provenance = {p: base.sha256(ROOT / p) for p in (
        'src/msabp_opt/optimization/phase2_krvea_data.py', 'scripts/postprocessing/cap_gain.py',
        'scripts/postprocessing/prepare_link_ffs_tensor.py', 'scripts/postprocessing/search_link_spherical_roi.py',
        'scripts/postprocessing/analyze_morris_tree.py', 'scripts/postprocessing/analyze_morris_tree_roi.py')}
    report = {'status': 'complete', 'campaign_id': plan['campaign_id'], 'plan_sha256': plan['plan_sha256'],
              'template_sha256': plan['template_sha256'], 'metric_contract': metric_contract,
              'objective_names_minimize': list(OBJECTIVES), 'diagnostic_only': legacy.ROI_GAIN_LINEAR_COLUMN,
              'metric_source_sha256': provenance, 'n_unique_ffs': len(roi_values), 'n_positions': len(points),
              'n_trajectories': len({p['trajectory_id'] for p in points}), 'bootstrap_resamples': 2000,
              'bootstrap_seed': 20260929, 'indices': indices, 'elementary_effects': effects,
              'valid_point_data': values, 'result_issues': issues,
              'interpretation': 'Conditional on frozen sampled trajectories; no new propagation validation or ROI refit.'}
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    write_csv(output / 'indices.csv', indices)
    write_csv(output / 'elementary_effects.csv', effects)
    write_csv(output / 'samples.csv', [{'sample_id': p['sample_id'],
        'simulation_sample_id': values[p['sample_id']]['simulation_sample_id'],
        **{m: values[p['sample_id']][m] for m in (*METRICS, legacy.ROI_GAIN_DBI_COLUMN)}} for p in points])
    plot_report(output, indices)
    print(f'Complete: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'simulations/runs/morris-tree-35d-016-001')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    run(args)
