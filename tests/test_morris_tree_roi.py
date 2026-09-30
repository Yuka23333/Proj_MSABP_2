from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.postprocessing import analyze_morris_tree as base
from scripts.postprocessing import analyze_morris_tree_roi as roi
from test_morris_tree_analysis import plan, points, linear_values


def inputs():
    rows = points()
    values = linear_values(rows)
    for row in rows:
        row['simulation_key'] = row['sample_id']
        x, y = row['unit']
        dbi = 4 + 2*x - 3*y
        values[row['sample_id']].update({roi.legacy.ROI_GAIN_LOSS_DBI_COLUMN: -dbi,
                                        roi.legacy.ROI_GAIN_LINEAR_COLUMN: 10**(dbi/10)})
    return rows, values


def test_matches_original_estimator_and_negative_dbi_sign():
    rows, values = inputs()
    indices, effects = roi.sensitivity(plan(), rows, values, resamples=50)
    original = base.analyze(plan(), rows, values, bootstrap_resamples=50)
    mapped = {(r['metric'], r['variable']): r for r in indices}
    for expected in original['indices']:
        if expected['metric'] not in roi.OBJECTIVES:
            continue
        actual = mapped[expected['metric'], expected['variable']]
        for key in ('mu', 'mu_star', 'sigma', 'mu_star_ci95_low', 'mu_star_ci95_high'):
            assert actual[key] == pytest.approx(expected[key])
    assert mapped[roi.legacy.ROI_GAIN_LOSS_DBI_COLUMN, 'x']['mu'] == pytest.approx(-2)
    assert mapped[roi.legacy.ROI_GAIN_LOSS_DBI_COLUMN, 'y']['mu'] == pytest.approx(3)
    assert len(effects) == 4
    assert all(row['geometry_changes'] == 2 for row in indices)


def test_missing_and_nonfinite_values_are_not_penalized():
    rows, values = inputs()
    bad = deepcopy(values)
    bad.pop(rows[0]['sample_id'])
    with pytest.raises(KeyError):
        roi.sensitivity(plan(), rows, bad, resamples=10)
    values[rows[0]['sample_id']][roi.legacy.ROI_GAIN_LINEAR_COLUMN] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        roi.sensitivity(plan(), rows, values, resamples=10)


def test_ffs_hash_checked_before_metric_or_cache(tmp_path, monkeypatch):
    ffs = tmp_path / 'field.ffs'
    ffs.write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'artifacts': {'farfield_source': {
        'path': ffs.name, 'sha256': 'a'*64, 'size_bytes': 7}}}))
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid FFS must not reach metric/cache')
    monkeypatch.setattr(roi.legacy, 'roi_radiation_gain_scalar', forbidden)
    with pytest.raises(ValueError, match='SHA-256'):
        roi.roi_job(('test', [str(manifest)], str(tmp_path/'cache')))
