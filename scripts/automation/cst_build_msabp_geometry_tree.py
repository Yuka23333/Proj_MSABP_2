"""Arbitrary-tree CST builder: resolved copper bodies/holes and bounded history.

No CST import at module load. Materials, ports and monitors belong to the template.
Only our own contiguous history suffix may be removed. No fallback solid deletion.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from scripts.geometry import shapely_antenna_tree_model as model
from scripts.automation import cst_build_msabp_geometry as legacy
from scripts.automation.cst_generate_polygen import (
    _vba_main_body, build_brick_vba, build_extrude_curve_vba,
)

HISTORY_PREFIX = "MSABP_TREE::"
OWNER = "msabp_tree"


def polygon_vba(points, name):
    """Keep boolean intersections precise, independently of legacy formatting."""
    points = list(points)
    if points[0] != points[-1]:
        points.append(points[0])
    lines = ['Sub Main()', 'With Polygon', '.Reset', f'.Name "{name}"', '.Curve "msabp_tree_curves"']
    for i, (x, y) in enumerate(points):
        lines.append(f'.{"Point" if i == 0 else "LineTo"} "{x:.15g}", "{y:.15g}"')
    return '\n'.join([*lines, '.Create', 'End With', 'End Sub'])


def normalize_request(raw):
    if not isinstance(raw, dict) or set(raw) - {"params", "tree", "build_options"}:
        raise ValueError("Tree request must contain only params, tree, build_options")
    params = model.default_params()
    override = raw.get("params", {})
    if not isinstance(override, dict) or set(override) - set(params):
        raise ValueError("Unknown tree geometry parameter")
    params.update(override)
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid parameter: {name}")
        if model.PARAM_SPECS[name][0] == 'k' and not 0 <= value <= 1:
            raise ValueError(f"Ratio out of range: {name}")
        if model.PARAM_SPECS[name][0] == 'abs' and value < 0:
            raise ValueError(f"Negative dimension: {name}")
    if params['SLOT_MAIN_LENGTH'] <= 0 or params['SLOT_MAIN_HEIGHT'] <= 0:
        raise ValueError("Main slot dimensions must be positive")
    source = raw.get("tree", model.default_tree())
    if not isinstance(source, dict):
        raise ValueError("tree must be an ordered object")
    tree = {}
    pending = dict(source)
    while pending:
        progress = False
        for node_id, node in list(pending.items()):
            if not isinstance(node, dict) or set(node) != {'parent', 'side', 'k'}:
                raise ValueError(f"Invalid node: {node_id}")
            parent, side, k = node['parent'], node['side'], node['k']
            if parent != model.ROOT and parent not in tree:
                continue
            prefix = '' if parent == model.ROOT else parent + '/'
            if side not in model.child_sides(tree, parent) or not re.fullmatch(re.escape(prefix + side) + r'[1-9]\d*', node_id):
                raise ValueError(f"Invalid branch path/direction: {node_id}")
            if not isinstance(k, (list, tuple)) or len(k) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in k
            ):
                raise ValueError(f"Invalid K values: {node_id}")
            tree[node_id] = {'parent': parent, 'side': side, 'k': list(k)}
            del pending[node_id]
            progress = True
        if not progress:
            raise ValueError("Missing parent or cyclic branch tree")
    options = {'snap_fraction': model.BRANCH_K2_SNAP_FRACTION, 'tip_clearance': model.BRANCH_TIP_CLEARANCE}
    supplied = raw.get('build_options', {})
    if not isinstance(supplied, dict) or set(supplied) - set(options):
        raise ValueError("Unknown build option")
    options.update(supplied)
    model.relax_upper_k(0, options['snap_fraction'])
    model._check_tip_clearance(options['tip_clearance'])
    return {'params': params, 'tree': tree, 'build_options': options}


def polygons(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == 'Polygon':
        return [shape]
    if shape.geom_type != 'MultiPolygon':
        raise ValueError(f"Expected polygonal geometry, got {shape.geom_type}")
    return list(shape.geoms)


def prepare_geometry(raw, quantum=.01):
    if not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("Quantum must be positive")
    request = normalize_request(raw)
    shapes = model.build(request['params'], request['tree'], **request['build_options'])
    offset = -shapes['Substrate_Full'].bounds[1]

    def quantize(shape):
        def ring(r):
            return [(round(x/quantum)*quantum, round((y+offset)/quantum)*quantum) for x, y in r.coords]
        result = []
        for p in polygons(shape):
            q = Polygon(ring(p.exterior), [ring(r) for r in p.interiors])
            if not q.is_valid or q.area <= 0:
                raise ValueError("Quantization collapsed or invalidated a source polygon")
            result.append(q)
        return unary_union(result)

    sources = {n: quantize(shapes[n]) for n in ('Patch', 'Slot', 'CPW_Feed_Pin', 'Substrate_Full')}
    copper = sources['Patch'].difference(sources['Slot']).union(sources['CPW_Feed_Pin'])
    if not copper.is_valid or copper.is_empty:
        raise ValueError("Invalid final copper")
    bodies = sorted(polygons(copper), key=lambda p: (-p.intersection(sources['CPW_Feed_Pin']).area, *p.bounds))
    x0, y0, x1, y1 = sources['Substrate_Full'].bounds
    half = (x1-x0)/2 - legacy.DEFAULT_REFLECTOR_CONNECTOR_BOARD_THICKNESS_MM + legacy.DEFAULT_REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_MM
    depth = legacy.DEFAULT_REFLECTOR_CUTOUT_DEPTH_MM
    if not 0 < half <= (x1-x0)/2 or not 0 < depth < y1-y0:
        raise ValueError("Reflector clearance does not fit substrate")
    reflector = box(x0, y0, x1, y1).difference(box(-half, y0, half, y0+depth))
    def curves(p):
        p = orient(p, sign=1)
        return {'exterior': list(p.exterior.coords)[:-1], 'holes': [list(r.coords)[:-1] for r in p.interiors]}
    return {'request': request, 'quantum_mm': quantum, 'y_offset_mm': offset,
            'substrate_bounds': [x0, y0, x1, y1], 'copper': [curves(p) for p in bodies],
            'reflector': curves(reflector), 'copper_area_mm2': copper.area,
            'copper_holes': sum(len(p.interiors) for p in bodies),
            'source_curves': {n: [curves(p) for p in polygons(g)] for n, g in sources.items()}}


def reset_tree_history(project, backup_path, timeout=15):
    history = project.model3d._GetHistory()
    entries = history['list']
    first = next((i for i, e in enumerate(entries) if e['name'].startswith(HISTORY_PREFIX)), len(entries))
    if any(not e['name'].startswith(HISTORY_PREFIX) for e in entries[first:]):
        raise RuntimeError("Manual/foreign steps follow tree history; refusing to truncate")
    if any(e['error'] for e in entries[:first]):
        raise RuntimeError("Template history has errors")
    if first < len(entries):
        Path(backup_path).write_text(json.dumps(history, indent=2), encoding='utf-8')
        project.model3d._ResizeHistory(first, timeout=timeout)
        project.model3d.full_history_rebuild(timeout=timeout)
    baseline = project.model3d._GetHistory()['list']
    if len(baseline) != first or any(e['error'] for e in baseline):
        raise RuntimeError("Template rebuild failed")
    if [(e['name'], e['contents']) for e in baseline] != [(e['name'], e['contents']) for e in entries[:first]]:
        raise RuntimeError("Baseline history changed")
    return baseline


def build_on_project(project, prepared, audit_directory, timeout=15):
    audit = Path(audit_directory)
    audit.mkdir(parents=True, exist_ok=True)
    baseline = reset_tree_history(project, audit/'history_before_reset.json', timeout)
    # Refuse stale geometry rather than deleting unknown/manual objects.
    project.schematic.execute_vba_code('''Sub Main()
Dim i As Long, shapeName As String
For i = 0 To Solid.GetNumberOfShapes()
shapeName = Solid.GetNameOfShapeFromIndex(i)
If shapeName <> "" And Left(shapeName, 10) <> "Connector:" Then Err.Raise vbObjectError + 2000, , "Unexpected baseline solid: " & shapeName
Next i
If Curve.StartCurveNameIteration("all") <> 0 Then Err.Raise vbObjectError + 2001, , "Unexpected baseline curves"
End Sub''', timeout=timeout)
    material_history = '\n'.join(e['contents'] for e in baseline)
    if f'.Name "{legacy.DEFAULT_SUBSTRATE_MATERIAL_NAME}"' not in material_history:
        raise RuntimeError("Template must contain the manual substrate material definition in its history")
    (audit/'geometry_tree.json').write_text(json.dumps(prepared, indent=2), encoding='utf-8')

    def execute(label, code):
        print('[Tree build]', label, flush=True)
        project.model3d.add_to_history(HISTORY_PREFIX + label, _vba_main_body(code), timeout=timeout)

    execute('component', f'Component.New "{OWNER}"')
    x0, y0, x1, y1 = prepared['substrate_bounds']
    thickness = legacy.DEFAULT_COPPER_THICKNESS_MM
    board_z = legacy.DEFAULT_SUBSTRATE_THICKNESS_MM
    substrate_name = 'msabp_substrate_solid'
    execute('substrate', build_brick_vba(substrate_name, OWNER, legacy.DEFAULT_SUBSTRATE_MATERIAL_NAME,
                                       (x0, x1), (y0, y1), (board_z, 0)))
    expected = [(substrate_name, (x1-x0)*(y1-y0)*abs(board_z), legacy.DEFAULT_SUBSTRATE_MATERIAL_NAME, board_z, 0)]

    def extrude_body(body, name, height):
        for i, ring in enumerate([body['exterior'], *body['holes']]):
            tool = name if i == 0 else name + f'_hole_{i:03d}'
            curve = name + f'_ring_{i:03d}'
            pts = list(orient(Polygon(ring), sign=1).exterior.coords)
            execute('curve '+curve, polygon_vba(pts, curve))
            execute('extrude '+tool, build_extrude_curve_vba(tool, OWNER, legacy.DEFAULT_COPPER_MATERIAL_NAME,
                                                          height, 'msabp_tree_curves', curve))
            if i:
                execute('hole '+tool, f'Solid.Subtract "{OWNER}:{name}", "{OWNER}:{tool}"')
        return Polygon(body['exterior'], body['holes']).area*abs(height)

    for i, body in enumerate(prepared['copper']):
        name = 'msabp_patch_solid' if i == 0 else f'msabp_island_{i:03d}'
        volume = extrude_body(body, name, thickness)
        expected.append((name, volume, legacy.DEFAULT_COPPER_MATERIAL_NAME, 0, thickness))
    name = 'msabp_reflector_solid'
    volume = extrude_body(prepared['reflector'], name, -thickness)
    execute('move reflector', f'''With Transform
.Reset
.Name "{OWNER}:{name}"
.Vector "0", "0", "{board_z}"
.UsePickedPoints "False"
.InvertPickedPoints "False"
.MultipleObjects "False"
.GroupObjects "False"
.Repetitions "1"
.Transform "Shape", "Translate"
End With''')
    expected.append((name, volume, legacy.DEFAULT_COPPER_MATERIAL_NAME, board_z-thickness, board_z))
    checks = ['Sub Main()', 'Dim x0 As Double, x1 As Double, y0 As Double, y1 As Double, z0 As Double, z1 As Double']
    checks.append(f'If Solid.GetNumberOfShapes() <> {len(expected)+3} Then Err.Raise vbObjectError + 2010, , "Unexpected solid count"')
    for name, volume, material, zmin, zmax in expected:
        ref = OWNER+':'+name
        checks += [f'If Not Solid.DoesExist("{ref}") Then Err.Raise vbObjectError + 2011, , "Missing {name}"',
                   f'If Abs(Solid.GetVolume("{ref}") - {volume:.15g}) > {max(1e-6,volume*1e-6):.15g} Then Err.Raise vbObjectError + 2012, , "Volume mismatch: {name}"',
                   f'If Solid.GetMaterialNameForShape("{ref}") <> "{material}" Then Err.Raise vbObjectError + 2013, , "Material mismatch"',
                   f'If Not Solid.GetLooseBoundingBoxOfShape("{ref}", x0,x1,y0,y1,z0,z1) Then Err.Raise vbObjectError + 2014, , "Missing bounds"',
                   f'If Abs(z0-({zmin:.15g})) > 0.000001 Or Abs(z1-({zmax:.15g})) > 0.000001 Then Err.Raise vbObjectError + 2015, , "Z placement mismatch"']
    checks.append('End Sub')
    project.schematic.execute_vba_code('\n'.join(checks), timeout=timeout)
    final = project.model3d._GetHistory()['list']
    if any(e['error'] for e in final) or [(e['name'], e['contents']) for e in final[:len(baseline)]] != [(e['name'], e['contents']) for e in baseline]:
        raise RuntimeError("History error or baseline changed during tree build")
    return {'copper_components': len(prepared['copper']), 'copper_holes': prepared['copper_holes'],
            'copper_area_mm2': prepared['copper_area_mm2'], 'baseline_history_count': len(baseline),
            'history_count': len(final), 'substrate_material': legacy.DEFAULT_SUBSTRATE_MATERIAL_NAME,
            'baseline_history': baseline}
