"""Audited arbitrary-tree inputs and Morris seeds for three-objective K-RVEA.

No CST connection, GP fit, or solver is performed here. Historical Morris
positions retain their alias map, but only one row per simulated geometry is
offered as an independent observation. Exact area uses the *tree* substrate,
including the builder's coordinate quantization, not the legacy 11-D formula.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.automation import cst_build_msabp_geometry_tree as builder
from scripts.optimization import morris_tree_common as common
from scripts.postprocessing import analyze_morris_tree as morris


ROOT = Path(__file__).resolve().parents[3]
OBJECTIVE_NAMES = (
    'worst_s11_linear_amplitude',
    'negative_roi_theta_radiation_gain_dbi',
    'normalized_substrate_area',
)
REPORT_OBJECTIVES = ('worst_s11_linear', *OBJECTIVE_NAMES[1:])
ROI_CONTRACT = {
    'name': 'frozen_roi_theta_radiation_gain_v1', 'frequency_ghz': 3.6,
    'theta_bounds_deg': [55.0, 85.0], 'phi_center_deg': 90.0,
    'phi_half_width_deg': 50.0, 'component': 'E_theta',
    'gain_type': 'radiation_gain', 'spatial_average': 'linear_power_solid_angle_weighted',
}
METRIC_SOURCE_PATHS = (
    'src/msabp_opt/optimization/phase2_krvea_data.py',
    'scripts/postprocessing/cap_gain.py',
    'scripts/postprocessing/prepare_link_ffs_tensor.py',
    'scripts/postprocessing/search_link_spherical_roi.py',
    'scripts/postprocessing/analyze_morris_tree.py',
    'scripts/postprocessing/analyze_morris_tree_roi.py',
)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name} must be a positive finite number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return value


@dataclass(frozen=True)
class TreeInputSpace:
    """Frozen parameter order, topology, manufacturing rules and physical range."""

    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    variables: tuple[dict[str, Any], ...]
    baseline_request: dict[str, Any]
    coordinate_quantum_mm: float
    reference_area_mm2: float

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any]) -> 'TreeInputSpace':
        baseline = deepcopy(plan['baseline_request'])
        if builder.normalize_request(baseline) != baseline:
            raise ValueError('Plan needs a complete normalized baseline request')
        quantum = _positive(plan['config']['coordinate_quantum_mm'], 'coordinate_quantum_mm')
        options = baseline['build_options']
        builder.model.manufacturing.validate_settings(
            options['min_feature_mm'], options['feature_snap_mm'], quantum,
            options['manufacturing_mode'],
        )
        if options['manufacturing_mode'] != 'repair':
            raise ValueError('Tree optimization requires the frozen manufacturing repair policy')
        if plan['config'].get('build_options') != options:
            raise ValueError('Plan build_options differ from baseline')
        if set(plan['config'].get('topology', [])) != set(baseline['tree']):
            raise ValueError('Plan topology differs from baseline')
        variables = tuple(deepcopy(plan['variables']))
        expected = {('parameter', name) for name in baseline['params']}
        expected.update(('node', node, index) for node in baseline['tree'] for index in range(3))
        seen, names, bounds = set(), [], []
        for spec in variables:
            name, kind = spec['name'], spec['kind']
            if 'parameter' in spec:
                parameter = spec['parameter']
                if parameter not in builder.model.PARAM_SPECS or kind != builder.model.PARAM_SPECS[parameter][0]:
                    raise ValueError(f'Unknown parameter or mismatched kind: {name}')
                identity, expected_name = ('parameter', parameter), parameter
                if 'node' in spec or 'k_index' in spec:
                    raise ValueError(f'Ambiguous parameter mapping: {name}')
            else:
                node, index = spec['node'], spec['k_index']
                if node not in baseline['tree'] or type(index) is not int or index not in (0, 1, 2) or kind != 'k':
                    raise ValueError(f'Invalid branch mapping: {name}')
                identity, expected_name = ('node', node, index), f'{node}.K{index + 1}'
            if name != expected_name or name in names or identity in seen:
                raise ValueError(f'Duplicate or inconsistent variable mapping: {name}')
            lo, hi = spec['lower'], spec['upper']
            if (isinstance(lo, bool) or isinstance(hi, bool) or
                    not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or
                    not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi):
                raise ValueError(f'Invalid bounds: {name}')
            if (kind == 'k' and not 0 <= lo < hi <= 1) or (kind == 'abs' and lo <= 0):
                raise ValueError(f'Bounds outside physical domain: {name}')
            seen.add(identity)
            names.append(name)
            bounds.append((lo, hi))
        if seen != expected or len(names) != plan['dimension']:
            raise ValueError('Variables must cover every parameter and branch K exactly once')
        lower, upper = np.asarray(bounds, dtype=np.float64).T
        lower.setflags(write=False)
        upper.setflags(write=False)
        result = cls(tuple(names), lower, upper, variables, baseline, quantum,
                     _positive(plan['metrics']['reference_area_mm2'], 'reference_area_mm2'))
        result.raw_from_request(baseline)
        return result

    @property
    def nominal(self) -> np.ndarray:
        return self.raw_from_request(self.baseline_request)

    def _array(self, values: Any, *, unit: bool) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != len(self.names) or not np.isfinite(array).all():
            raise ValueError('Input needs finite values and a final axis equal to the input dimension')
        lo, hi = (0.0, 1.0) if unit else (self.lower, self.upper)
        tolerance = 1e-12 * np.maximum(1, np.abs(hi))
        if np.any(array < lo - tolerance) or np.any(array > hi + tolerance):
            raise ValueError('Input outside frozen bounds')
        return np.clip(array, lo, hi)

    def normalize(self, raw: Any) -> np.ndarray:
        return (self._array(raw, unit=False) - self.lower) / (self.upper - self.lower)

    def denormalize(self, unit: Any) -> np.ndarray:
        return self.lower + self._array(unit, unit=True) * (self.upper - self.lower)

    def request_from_raw(self, raw: Any) -> dict[str, Any]:
        values = self._array(raw, unit=False)
        if values.ndim != 1:
            raise ValueError('One raw vector is required to construct a tree request')
        request = deepcopy(self.baseline_request)
        for spec, value in zip(self.variables, values, strict=True):
            if 'parameter' in spec:
                request['params'][spec['parameter']] = float(value)
            else:
                request['tree'][spec['node']]['k'][spec['k_index']] = float(value)
        return request

    def raw_from_request(self, request: Mapping[str, Any]) -> np.ndarray:
        normalized = builder.normalize_request(dict(request))
        if normalized != request:
            raise ValueError('A complete normalized tree request is required')
        if normalized['build_options'] != self.baseline_request['build_options']:
            raise ValueError('Request changes frozen build_options')
        tree, baseline = normalized['tree'], self.baseline_request['tree']
        if set(tree) != set(baseline) or any(
                (tree[node]['parent'], tree[node]['side']) !=
                (baseline[node]['parent'], baseline[node]['side']) for node in tree):
            raise ValueError('Request changes frozen topology')
        raw = [normalized['params'][v['parameter']] if 'parameter' in v else
               tree[v['node']]['k'][v['k_index']] for v in self.variables]
        return self._array(raw, unit=False)

    def substrate_dimensions(self, unit: Any) -> tuple[np.ndarray, np.ndarray]:
        """CST substrate width/height in mm, after the 0.01 mm coordinate grid."""
        raw = self.denormalize(unit)
        p = {name: raw[..., i] for i, name in enumerate(self.names)}
        half_width = np.maximum(p['SLOT_MAIN_LENGTH'] / 2 +
                                (builder.model.FIXED_OFFSET + p['PATCH_BRICK_1_SIDE_MARGIN']),
                                builder.model.PATCH_BRICK_2_WIDTH / 2)
        top = p['SLOT_MAIN_HEIGHT'] / 2 + (
            (builder.model.FIXED_OFFSET + p['PATCH_BRICK_1_TOP_MARGIN']) +
            p['PATCH_BRICK_2_HEIGHT_MARGIN'])
        bottom = -p['SLOT_MAIN_HEIGHT'] / 2 - (
            (builder.model.FIXED_OFFSET + p['PATCH_BRICK_3_BOTTOM_MARGIN']) +
            p['PATCH_BRICK_4_MARGIN'] + builder.model.PATCH_BRICK_4_FIXED)
        quantum = self.coordinate_quantum_mm
        return (2 * np.rint(half_width / quantum) * quantum,
                np.rint((top - bottom) / quantum) * quantum)

    def exact_normalized_area(self, unit: Any) -> np.ndarray | float:
        """Deterministic third objective; input is normalized, never raw mm."""
        width, height = self.substrate_dimensions(unit)
        value = width * height / self.reference_area_mm2
        return float(value) if np.ndim(value) == 0 else value

    def exact_area_contract(self) -> dict[str, Any]:
        """Self-contained constants for geometry-free remote GP proposal workers."""
        return {
            'type': 'msabp_tree_quantized_substrate_area_v1',
            'parameter_names': list(self.names), 'lower': self.lower.tolist(),
            'upper': self.upper.tolist(), 'coordinate_quantum_mm': self.coordinate_quantum_mm,
            'reference_area_mm2': self.reference_area_mm2,
            'fixed_offset_mm': float(builder.model.FIXED_OFFSET),
            'brick2_width_mm': float(builder.model.PATCH_BRICK_2_WIDTH),
            'brick4_fixed_mm': float(builder.model.PATCH_BRICK_4_FIXED),
        }


@dataclass(frozen=True)
class Dataset:
    input_space: TreeInputSpace
    x_raw: np.ndarray
    x_unit: np.ndarray
    objectives: np.ndarray
    records: list[dict[str, Any]]
    aliases: dict[str, str]
    provenance: dict[str, Any]
    objective_names: tuple[str, ...] = OBJECTIVE_NAMES

    @property
    def exact_area(self):
        return self.input_space.exact_normalized_area


def _validate_report(report: Mapping[str, Any], plan: Mapping[str, Any], points: list) -> None:
    if report.get('status') != 'complete' or report.get('result_issues'):
        raise ValueError('A complete RoI report without unresolved result issues is required')
    for key in ('campaign_id', 'plan_sha256', 'template_sha256'):
        if report.get(key) != plan[key]:
            raise ValueError(f'RoI report differs from frozen campaign: {key}')
    if report.get('metric_contract') != ROI_CONTRACT:
        raise ValueError('RoI report metric contract differs from the historical frozen RoI')
    if report.get('objective_names_minimize') != list(REPORT_OBJECTIVES):
        raise ValueError('Unexpected report objective ordering or sign convention')
    if report.get('n_positions') != len(points):
        raise ValueError('RoI report has a different number of Morris positions')
    sources = report.get('metric_source_sha256', {})
    if set(sources) != set(METRIC_SOURCE_PATHS):
        raise ValueError('RoI metric source provenance is incomplete')
    for name, expected in sources.items():
        if common.file_hash(ROOT / name) != expected:
            raise ValueError(f'RoI metric implementation changed; re-audit the report: {name}')


def load_morris_seed(
    campaign: Path, roi_report: Path | None = None, *, audit_artifacts: bool = True,
) -> Dataset:
    """Read a completed frozen campaign, retaining independent physical solves.

    Default audits the existing report against all curve/geometry/FFS bytes and
    manifest provenance; it does not reparse FFS or recompute the RoI. A caller
    may explicitly skip the artifact pass for an already-audited local snapshot;
    this is recorded in ``provenance``. Historical geometry source hashes are
    checked in manifests, not against today's potentially upgraded generator.
    """
    campaign = Path(campaign).resolve()
    plan, points = morris.load_campaign(campaign)
    space = TreeInputSpace.from_plan(plan)
    report_path = Path(roi_report) if roi_report is not None else campaign / 'analysis_roi' / 'analysis.json'
    report = morris._json(report_path)
    _validate_report(report, plan, points)
    values = report['valid_point_data']
    index = {p['sample_id']: p for p in points}
    if set(values) != set(index):
        raise ValueError('RoI report must contain every planned Morris position exactly once')
    aliases, canonical_by_key = {}, {}
    for point in points:
        sid, canonical = point['sample_id'], point['simulation_sample_id']
        if canonical not in index or index[canonical]['simulation_sample_id'] != canonical:
            raise ValueError('Simulation alias must refer directly to a planned canonical point')
        original = index[canonical]
        for field in ('simulation_key', 'manufactured_copper_sha256', 'substrate_area_mm2'):
            if point[field] != original[field]:
                raise ValueError(f'Alias has different physical geometry: {sid}')
        key = point['simulation_key']
        if canonical_by_key.setdefault(key, canonical) != canonical:
            raise ValueError('The same geometry has multiple canonical observations')
        aliases[sid] = canonical
        raw = space.raw_from_request(point['request'])
        if not np.allclose(space.normalize(raw), point['unit'], rtol=0, atol=1e-12):
            raise ValueError(f'Frozen input/request mismatch: {sid}')
        datum = values[sid]
        if datum['simulation_sample_id'] != canonical or datum['reused_simulation'] != (sid != canonical):
            raise ValueError(f'Report alias mapping differs from campaign: {sid}')
        objective = np.asarray([datum[name] for name in REPORT_OBJECTIVES], dtype=float)
        if not np.isfinite(objective).all() or objective[0] < 0:
            raise ValueError(f'Invalid objective value: {sid}')
        gain = _positive(datum['roi_theta_radiation_gain_linear'], 'linear RoI gain')
        dbi = float(datum['roi_theta_radiation_gain_dbi'])
        if not math.isclose(dbi, 10 * math.log10(gain), abs_tol=1e-10) or not math.isclose(-dbi, objective[1], abs_tol=1e-10):
            raise ValueError(f'RoI linear/dBi/sign values disagree: {sid}')
        area = point['substrate_area_mm2'] / space.reference_area_mm2
        if not math.isclose(area, objective[2], rel_tol=1e-12) or not math.isclose(
                space.exact_normalized_area(point['unit']), area, rel_tol=1e-12):
            raise ValueError(f'Exact tree substrate area disagrees: {sid}')
        for field in (*REPORT_OBJECTIVES, 'ffs_sha256'):
            if datum[field] != values[canonical][field]:
                raise ValueError(f'Alias observation differs from canonical observation: {sid}')
    if report.get('n_unique_ffs') != len(canonical_by_key):
        raise ValueError('Report independent-solve count differs from canonical geometry count')

    live_values = None
    if audit_artifacts:
        live_values, issues = morris.collect_results(plan, points, [campaign / 'results'])
        if issues or set(live_values) != set(index):
            raise ValueError(f'Morris result audit failed: {len(issues)} issues, {len(live_values)}/{len(index)} valid positions')

    records, raw_rows, unit_rows, objectives = [], [], [], []
    for point in points:
        sid = point['sample_id']
        if aliases[sid] != sid:
            continue
        datum = values[sid]
        manifests = list(datum['result_manifests'])
        if live_values is not None:
            live = live_values[sid]
            for metric in morris.METRICS:
                if not math.isclose(live[metric], datum[metric], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f'Report metric differs from live results: {sid}/{metric}')
            manifests = list(live['result_manifests'])
            for filename in manifests:
                path = Path(filename)
                manifest = morris._json(path)
                artifact = manifest['artifacts']['farfield_source']
                morris._artifact(path.parent, artifact, 'farfield_source')
                if artifact['sha256'] != datum['ffs_sha256']:
                    raise ValueError(f'Report FFS hash differs from live result: {sid}')
        raw = space.raw_from_request(point['request'])
        objective = [float(datum[name]) for name in REPORT_OBJECTIVES]
        raw_rows.append(raw)
        unit_rows.append(space.normalize(raw))
        objectives.append(objective)
        records.append({
            'sample_id': sid, 'request': deepcopy(point['request']),
            'simulation_key': point['simulation_key'],
            'manufactured_copper_sha256': point['manufactured_copper_sha256'],
            'objectives': objective, 'result_manifests': manifests,
            'ffs_sha256': datum['ffs_sha256'],
            'aliases': [alias for alias, canonical in aliases.items() if canonical == sid],
        })
    return Dataset(space, np.asarray(raw_rows, dtype=np.float64),
                   np.asarray(unit_rows, dtype=np.float64), np.asarray(objectives, dtype=np.float64),
                   records, aliases, {
                       'campaign': str(campaign), 'plan_sha256': plan['plan_sha256'],
                       'template_sha256': plan['template_sha256'],
                       'roi_report': str(report_path.resolve()),
                       'roi_report_sha256': common.file_hash(report_path),
                       'metric_contract': deepcopy(ROI_CONTRACT),
                       'artifact_audit': 'verified' if audit_artifacts else 'explicitly_skipped',
                       'source_sha256': deepcopy(plan['source_sha256']),
                       'n_positions': len(points), 'n_independent_observations': len(records),
                   })
