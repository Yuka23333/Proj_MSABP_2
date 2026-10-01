"""Saved manufactured tree copper -> a history-owned pair -> existing S21 exporter.

Only MSABP_TREE_PROP:: history suffix entries may be removed. Ports, connectors,
phantom, materials, symmetry and solver setup belong to the provisioned template.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from shapely.affinity import scale, translate
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from . import case_runner as common
from . import propagation_case_runner as legacy_runner
from scripts.automation import cst_build_msabp_geometry_tree as tree
from scripts.automation.cst_generate_polygen import _vba_main_body, build_brick_vba, build_extrude_curve_vba

PREFIX = 'MSABP_TREE_PROP::'
OWNER = 'msabp_tree_pair'
CURVES = 'msabp_tree_pair_curves'
OFFSET_MM = 300.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_payload(row):
    payload = json.loads(row['tree_geometry_json'])
    if digest(payload) != row['tree_geometry_sha256']:
        raise ValueError('Saved tree payload hash mismatch')
    geometry = payload['geometry']
    bounds = geometry['substrate_bounds']
    board = box(*bounds)
    copper = [Polygon(c['exterior'], c['holes']) for c in geometry['copper']]
    reflector = Polygon(geometry['reflector']['exterior'], geometry['reflector']['holes'])
    if board.area <= 0 or not copper or any(not p.is_valid or p.is_empty for p in [*copper, reflector]):
        raise ValueError('Invalid saved polygons')
    if any(p.difference(board).area > 1e-8 for p in [*copper, reflector]):
        raise ValueError('Saved conductor extends outside the substrate')
    actual = hashlib.sha256(unary_union(copper).simplify(0).normalize().wkb).hexdigest()
    if actual != payload['manufactured_copper_sha256']:
        raise ValueError('Saved copper does not match the original simulation')
    return payload


def second_shape(polygon):
    return translate(scale(polygon, xfact=1, yfact=-1, origin=(0, 0)), yoff=OFFSET_MM)


def history_steps(geometry):
    """Generate both antennas directly; mirror is y -> 300-y, with x unchanged."""
    material = tree.legacy.DEFAULT_COPPER_MATERIAL_NAME
    thickness = tree.legacy.DEFAULT_COPPER_THICKNESS_MM
    board_z = tree.legacy.DEFAULT_SUBSTRATE_THICKNESS_MM
    steps = [('component', f'Component.New "{OWNER}"')]
    expected = []
    for side in (1, 2):
        transform = (lambda p: p) if side == 1 else second_shape
        board = transform(box(*geometry['substrate_bounds']))
        x0, y0, x1, y1 = board.bounds
        name = f'a{side}_substrate'
        steps.append((name, _vba_main_body(build_brick_vba(name, OWNER,
            tree.legacy.DEFAULT_SUBSTRATE_MATERIAL_NAME, (x0, x1), (y0, y1), (board_z, 0)))))
        expected.append((name, board.area * abs(board_z), tree.legacy.DEFAULT_SUBSTRATE_MATERIAL_NAME,
                         (x0, x1, y0, y1, board_z, 0)))
        bodies = [(f'copper_{i:03d}', body, thickness, 0.0) for i, body in enumerate(geometry['copper'])]
        bodies.append(('reflector', geometry['reflector'], -thickness, board_z))
        for suffix, body, height, zoffset in bodies:
            polygon = orient(transform(Polygon(body['exterior'], body['holes'])), sign=1)
            name = f'a{side}_{suffix}'
            for index, ring in enumerate((polygon.exterior, *polygon.interiors)):
                tool = name if index == 0 else f'{name}_hole_{index}'
                curve = f'{name}_ring_{index}'
                code = tree.polygon_vba(list(orient(Polygon(ring), sign=1).exterior.coords), curve)
                steps.append((curve, _vba_main_body(code).replace('msabp_tree_curves', CURVES)))
                steps.append((tool, _vba_main_body(build_extrude_curve_vba(tool, OWNER, material, height, CURVES, curve))))
                if index:
                    steps.append((f'cut_{tool}', f'Solid.Subtract "{OWNER}:{name}", "{OWNER}:{tool}"'))
            if zoffset:
                steps.append((f'move_{name}', f'''With Transform
.Reset
.Name "{OWNER}:{name}"
.Vector "0", "0", "{zoffset}"
.UsePickedPoints "False"
.InvertPickedPoints "False"
.MultipleObjects "False"
.GroupObjects "False"
.Repetitions "1"
.Transform "Shape", "Translate"
End With'''))
            x0, y0, x1, y1 = polygon.bounds
            expected.append((name, polygon.area * abs(height), material,
                             (x0, x1, y0, y1, min(0, height)+zoffset, max(0, height)+zoffset)))
    return steps, expected


def build_pair(project, payload, directory, timeout):
    model = project.model3d
    history = model._GetHistory()['list']
    first = next((i for i, e in enumerate(history) if e['name'].startswith(PREFIX)), len(history))
    if any(not e['name'].startswith(PREFIX) for e in history[first:]) or any(e['error'] for e in history[:first]):
        raise RuntimeError('Foreign/error history prevents safe pair reset')
    if first < len(history):
        (directory / 'history_before_reset.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        model._ResizeHistory(first, timeout=timeout)
        model.full_history_rebuild(timeout=timeout)
    baseline = model._GetHistory()['list']
    identity = lambda entries: [(e['name'], e['contents']) for e in entries]
    if identity(baseline) != identity(history[:first]) or any(e['error'] for e in baseline):
        raise RuntimeError('Template history changed during reset')
    # Query only: no infrastructure mutation via execute_vba_code.
    project.schematic.execute_vba_code(f'''Sub Main()
Dim i As Long, n As String, hasKevin As Boolean
hasKevin = False
For i = 0 To Solid.GetNumberOfShapes()
    n = Solid.GetNameOfShapeFromIndex(i)
    If n <> "" Then
        If Left(n, {len(OWNER)+1}) = "{OWNER}:" Or Left(n, 10) = "component1" Or Left(n, 11) = "msabp_tree:" Then Err.Raise vbObjectError + 2101, , "Unowned old antenna geometry: " & n
        If Solid.GetMaterialNameForShape(n) = "Kevin" Then hasKevin = True
    End If
Next i
If Not hasKevin Then Err.Raise vbObjectError + 2102, , "Template has no Kevin phantom solid"
End Sub''', timeout=timeout)
    steps, expected = history_steps(payload['geometry'])
    for name, code in steps:
        model.add_to_history(PREFIX + name, code, timeout=timeout)
    checks = ['Sub Main()', 'Dim x0 As Double, x1 As Double, y0 As Double, y1 As Double, z0 As Double, z1 As Double']
    for name, volume, material, bounds in expected:
        ref = f'{OWNER}:{name}'
        checks += [f'If Not Solid.DoesExist("{ref}") Then Err.Raise vbObjectError + 2110, , "Missing {ref}"',
                   f'If Abs(Solid.GetVolume("{ref}")-({volume:.15g})) > {max(1e-6,volume*1e-6):.15g} Then Err.Raise vbObjectError + 2111, , "Volume mismatch"',
                   f'If Solid.GetMaterialNameForShape("{ref}") <> "{material}" Then Err.Raise vbObjectError + 2112, , "Material mismatch"',
                   f'If Not Solid.GetLooseBoundingBoxOfShape("{ref}",x0,x1,y0,y1,z0,z1) Then Err.Raise vbObjectError + 2113, , "Missing bounds"']
        for variable, value in zip(('x0','x1','y0','y1','z0','z1'), bounds):
            checks.append(f'If Abs({variable}-({value:.15g})) > 0.000001 Then Err.Raise vbObjectError + 2114, , "Placement mismatch: {ref}"')
    checks.append('End Sub')
    project.schematic.execute_vba_code('\n'.join(checks), timeout=timeout)
    final = model._GetHistory()['list']
    if identity(final[:first]) != identity(baseline) or any(e['error'] for e in final):
        raise RuntimeError('Baseline changed or pair history has errors')
    return {'baseline_history': baseline, 'pair_solids': len(expected), 'offset_y_mm': OFFSET_MM,
            'phantom_material': 'Kevin', 'symmetry': 'inherited unchanged from template'}


def run_csv_row(row, *, project_path, output_root, local_artifact_root=None, project=None,
                case_id=None, id_width=4, coordinate_quantum_mm=.01, command_timeout=60,
                overwrite=False, save_project_after_case=False, dry_run=False, stage_callback=None):
    started = time.perf_counter()
    sid = common._case_id_from_row(row, case_id)
    folder = Path(output_root) / common._case_directory_name(sid, id_width)
    stage = 'precheck'
    opened = False
    def notify(name):
        nonlocal stage
        stage = name
        common._notify(stage_callback, name)
    try:
        export_e_fields = common._parse_csv_bool(row.get('export_e_fields', True), 'export_e_fields')
        payload = validate_payload(row)
        if folder.exists() and any(folder.iterdir()) and not overwrite:
            raise FileExistsError(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = Path(project_path)
        artifacts, report, infrastructure = {}, {}, {}
        local_case = None
        (folder/'geometry_tree.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        if not dry_run:
            if local_artifact_root is None:
                raise ValueError('Maid-local export directory is required')
            local_case = Path(local_artifact_root) / common._case_directory_name(sid, id_width)
            if project is None:
                notify('opening_project')
                project = legacy_runner.cst_run_and_export_s11.open_cst_project(str(path))
                opened = True
            notify('checking_propagation_infrastructure')
            infrastructure = legacy_runner.inspect_propagation_infrastructure(
                project, command_timeout, require_e_fields=export_e_fields)
            notify('clearing_results')
            legacy_runner.cst_run_and_export_s11.clear_results_on_project(project, timeout=command_timeout)
            notify('building_tree_pair')
            report = build_pair(project, payload, folder, command_timeout)
            if legacy_runner.inspect_propagation_infrastructure(
                    project, command_timeout, require_e_fields=export_e_fields) != infrastructure:
                raise RuntimeError('Ports, monitors or solver changed')
            notify('solving')
            project.model3d.run_solver(timeout=None)
            notify('exporting_s21_and_retaining_fields' if export_e_fields else 'exporting_s21')
            exported = legacy_runner.export_propagation_results.export_propagation_results(
                path, local_case, excitation_port=1, overwrite=True,
                timeout=float(command_timeout or 60), project=project, export_e_fields=export_e_fields)
            shutil.copy2(local_case/'S21_complex.csv', folder/'S21.csv')
            artifacts['s21'] = common._artifact_record(folder/'S21.csv', folder)
            report['e_field_monitor_count'] = exported.e_field_monitor_count
            if save_project_after_case:
                legacy_runner.cst_run_and_export_s11.execute_save_project(project, timeout=command_timeout)
        manifest = {'schema_version': 1, 'simulation_mode': 'propagation_s21', 'geometry_engine': 'saved_tree_pair_v1',
                    'case_id': sid, 'status': 'dry_run' if dry_run else 'completed', 'dry_run': dry_run,
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter()-started,
                    'parameters': payload['request'], 'source_case_id': payload['source_case_id'],
                    'source_manufactured_copper_sha256': payload['manufactured_copper_sha256'],
                    'template_cst_sha256': row['template_cst_sha256'], 'geometry': report,
                    'infrastructure': infrastructure, 'artifacts': artifacts,
                    'export_e_fields': export_e_fields,
                    'local_only': {'retained_on_maid': export_e_fields and not dry_run,
                                   'transferred_to_princess': False,
                                   'directory': str(local_case/'e_field_native') if export_e_fields and not dry_run else None},
                    'tree_geometry_sha256': row['tree_geometry_sha256']}
        common._write_manifest(folder/'manifest.json', manifest)
        notify('completed')
        return common.CaseRunResult(case_id=sid, case_directory=folder, manifest_path=folder/'manifest.json',
            s11_path=None, farfield_source_path=None, dry_run=dry_run, elapsed_seconds=manifest['elapsed_seconds'],
            s21_path=None if dry_run else folder/'S21.csv', simulation_mode='propagation_s21',
            local_e_field_directory=None if dry_run or not export_e_fields else local_case/'e_field_native')
    except Exception as exc:
        raise common.CaseRunError(sid, stage, str(exc)) from exc
    finally:
        if opened:
            project.close()
