"""Manufacturing projection for the tree model's planar, orthogonal features.

Widths are measured on the FINAL copper and its complement, not branch boxes
or polygon areas. Exact slab cross-sections find axis-aligned thin features;
Facing-boundary tests distinguish true throats from harmless short steps. Local
axis-aligned joints may be widened; unsafe oblique features remain rejected.
This is a conservative DRC for this model, not a general PCB fabrication engine.
"""
from __future__ import annotations

import math

from shapely.affinity import scale, translate
from shapely.geometry import LineString, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, transform, unary_union
from shapely import set_precision

VERSION = 'tree-manufacturing-v2'
MIN_FEATURE_MM = 0.1
SNAP_THRESHOLD_MM = 0.05
QUANTUM_MM = 0.01
TOL = 1e-8


class ManufacturingError(ValueError):
    def __init__(self, message, report):
        super().__init__('Manufacturing constraint: ' + message)
        self.report = report


def polygons(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == 'Polygon':
        return [shape]
    if shape.geom_type == 'MultiPolygon':
        return list(shape.geoms)
    raise ValueError('Expected Polygon or MultiPolygon')


def rings(shape):
    return [ring for p in polygons(shape) for ring in (p.exterior, *p.interiors)]


def quantize(shape, quantum, y_origin=0.0):
    """Pointwise quantization: never let set_precision silently remove a bridge."""
    def snap(x, y, z=None):
        return round(x / quantum) * quantum, round((y-y_origin) / quantum) * quantum + y_origin
    return transform(snap, shape)


def _lines(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == 'LineString':
        return [shape]
    if hasattr(shape, 'geoms'):
        return [line for g in shape.geoms for line in _lines(g)]
    return []


def thin_features(shape, minimum):
    """Local axial widths, not short polygon edges or minimum_clearance.

Return merged rectangles covering thin strips; midpoint probes are exact for
rectilinear boundaries. If a thin interval has sloped ends, refuse its repair.
"""
    if shape.is_empty:
        return []
    coords = [xy for ring in rings(shape) for xy in ring.coords]
    bounds = shape.bounds
    found = {}
    for axis in (0, 1):
        along = 1-axis
        cuts = sorted(set(round(xy[along], 10) for xy in coords))
        for low, high in zip(cuts, cuts[1:]):
            if high-low < TOL:
                continue
            middle = (low+high)/2
            ends = [(bounds[axis]-1, middle), (bounds[axis+2]+1, middle)]
            line = LineString(ends if axis == 0 else [(y,x) for x,y in ends])
            for segment in _lines(shape.intersection(line)):
                a, b = segment.bounds[axis], segment.bounds[axis+2]
                if not TOL < b-a < minimum-TOL:
                    continue
                rect = box(a, low, b, high) if axis == 0 else box(low, a, high, b)
                # A sloped thin interval is not a constant-width rectangle.
                rectangular = rect.difference(shape).area <= TOL
                key = (axis, round(a,10), round(b,10), rectangular)
                found.setdefault(key, []).append((low,high))
    features = []
    for (axis,a,b,rectangular), spans in sorted(found.items()):
        merged = []
        for lo,hi in sorted(spans):
            if merged and abs(merged[-1][1]-lo) <= TOL:
                merged[-1][1] = hi
            else:
                merged.append([lo,hi])
        for lo,hi in merged:
            bounds = [a,lo,b,hi] if axis == 0 else [lo,a,hi,b]
            features.append({'axis':'x' if axis == 0 else 'y', 'width_mm':b-a,
                             'bounds':bounds, 'rectangular':rectangular})
    return features


def _proximity_issues(copper, minimum):
    """Find opposing material faces, including joints between scan slabs.

    With CCW shells / CW holes, the left normal always points INTO copper.
    Both faces must point into the same copper chord (or away from a void
    chord). Euclidean distance between arbitrary corner vertices is not enough:
    that used to misclassify tiny steps and redundant collinear vertices.
    Simplification is only for detection, never a geometry modification.
    """
    segments = []
    for polygon in polygons(copper.simplify(0)):
        polygon = orient(polygon,sign=1)
        for ring in (polygon.exterior,*polygon.interiors):
            coords = list(ring.coords)
            for a,b in zip(coords,coords[1:]):
                edge = LineString([a,b])
                if edge.length > TOL:
                    segments.append((edge, (-(b[1]-a[1])/edge.length, (b[0]-a[0])/edge.length)))
    issues, seen = [], set()
    boundary_halo = copper.boundary.buffer(TOL)
    for i,(a,normal_a) in enumerate(segments):
        for b,normal_b in segments[i+1:]:
            distance = a.distance(b)
            if not TOL < distance < minimum-TOL:
                continue
            p,q = nearest_points(a,b)
            chord = LineString([p,q])
            if chord.difference(boundary_halo).length <= TOL:
                continue
            vx,vy = q.x-p.x,q.y-p.y
            facing_a = normal_a[0]*vx+normal_a[1]*vy
            facing_b = -normal_b[0]*vx-normal_b[1]*vy
            phase = None
            if facing_a > TOL and facing_b > TOL and chord.difference(copper).length <= TOL:
                phase = 'copper'
            elif facing_a < -TOL and facing_b < -TOL and chord.intersection(copper).length <= TOL:
                phase = 'gap'
            if phase is None:
                continue
            points = sorted([list(p.coords[0]),list(q.coords[0])])
            key = (phase,tuple(tuple(round(v,10) for v in point) for point in points))
            if key not in seen:
                seen.add(key)
                issues.append({'phase':phase, 'distance_mm':distance, 'points':points})
    return issues


def _joint_cap(issue, minimum, quantum):
    """A bounded rectangular collar overlaps both parents of an axial throat.

    Unlike widening just the strip, overlap also fixes its endpoint connection.
    Return BOTH mirrors; never repair one half of the antenna independently.
    """
    p,q = issue['points']
    bounds = [math.floor((min(p[j],q[j])-minimum/2)/quantum+TOL)*quantum for j in (0,1)]
    bounds += [math.ceil((max(p[j],q[j])+minimum/2)/quantum-TOL)*quantum for j in (0,1)]
    cap = box(*bounds)
    return cap.union(scale(cap,xfact=-1,origin=(0,0)))


def _signature(shape):
    return shape.normalize().wkb_hex


def _snapshot(shape):
    return {'area_mm2':shape.area, 'components':len(polygons(shape)),
            'holes':sum(len(p.interiors) for p in polygons(shape))}


def validate_settings(minimum,threshold,quantum,mode):
    if mode not in ('repair', 'reject', 'off'):
        raise ValueError('manufacturing_mode must be repair, reject, or off')
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v > 0
               for v in (minimum,threshold,quantum)) or threshold >= minimum:
        raise ValueError('Require 0 < threshold < minimum and positive quantum')
    if quantum > minimum/2 + TOL:
        raise ValueError('Coordinate quantum too coarse for manufacturing features')


def repair_copper(copper, substrate, protected, *, minimum=MIN_FEATURE_MM,
                  threshold=SNAP_THRESHOLD_MM, quantum=QUANTUM_MM,
                  mode='repair', max_passes=12, max_joint_repairs=8):
    """Project tiny metal/void widths to 0 or >= minimum, then validate.

    Operates on full mirrored geometry. Feed/pad mask must remain unchanged.
    Input and output coordinates are in the model frame; y quantization is
    anchored to the substrate bottom, identically to the CST exporter.
    """
    validate_settings(minimum,threshold,quantum,mode)
    if not isinstance(max_passes,int) or max_passes < 1:
        raise ValueError('max_passes must be positive')
    if not isinstance(max_joint_repairs,int) or isinstance(max_joint_repairs,bool) or max_joint_repairs < 0:
        raise ValueError('max_joint_repairs must be a nonnegative integer')
    report = {'version':VERSION, 'mode':mode, 'minimum_mm':minimum, 'threshold_mm':threshold,
              'quantum_mm':quantum, 'status':'pending', 'actions':[], 'before':_snapshot(copper),
              'max_joint_repairs':max_joint_repairs, 'joint_repairs':0}
    def fail(message, **details):
        report.update(status='rejected', reason=message, **details)
        raise ManufacturingError(message, report)
    if copper.is_empty or not copper.is_valid or not substrate.buffer(TOL).covers(copper):
        fail('invalid or out-of-substrate input copper')
    if mode == 'off':
        report.update(status='disabled', after=report['before'])
        return copper, report
    if copper.symmetric_difference(scale(copper,xfact=-1,origin=(0,0))).area > TOL:
        fail('input is not left/right symmetric')

    origin = substrate.bounds[1]
    # Work at y=0 board bottom to make outward grid rounding reproducible.
    raw = set_precision(translate(copper,yoff=-origin),1e-9)
    board = translate(substrate,yoff=-origin)
    mask = translate(protected,yoff=-origin)
    frame = box(*board.bounds).buffer(2*minimum,join_style=2)
    current = raw
    seen = set()
    coordinates_quantized = False
    raw_grid,mask_grid,board_grid = (quantize(g,quantum) for g in (raw,mask,board))
    def touches_protected(candidate):
        reference,region = (raw_grid,mask_grid) if coordinates_quantized else (raw,mask)
        return candidate.symmetric_difference(reference).intersection(region).area > TOL
    for pass_number in range(max_passes):
        signature = _signature(current)
        if signature in seen:
            fail('repair cycle; no safe projection found')
        seen.add(signature)
        void = frame.difference(current)
        features = [(phase,f) for phase,g in [('copper',current),('gap',void)]
                    for f in thin_features(g,minimum)]
        if features and mode == 'reject':
            fail('sub-minimum feature', violations=[{'phase':p,**f} for p,f in features])
        if features:
            # One phase at a time, full mirrored batch. Recheck the other phase
            # afterwards, since widening copper can create a new narrow gap.
            phase = features[0][0]
            selected = [f for p,f in features if p == phase]
            adds, removes = [], []
            material = current if phase == 'copper' else void
            # A tiny rectangular island/hole must be resized in BOTH axes in
            # one operation; two independent strip expansions create a cross.
            for component in polygons(material):
                bounds = list(component.bounds)
                widths = [bounds[2]-bounds[0], bounds[3]-bounds[1]]
                if min(widths) >= minimum-TOL or abs(component.area-widths[0]*widths[1]) > TOL:
                    continue
                selected = [f for f in selected if box(*f['bounds']).difference(component).area > TOL]
                if min(widths) < threshold-TOL:
                    removes.append(component)
                    action = 'remove'
                else:
                    for axis,width in enumerate(widths):
                        if width < minimum-TOL:
                            mid = (bounds[axis]+bounds[axis+2])/2
                            bounds[axis] = math.floor((mid-minimum/2)/quantum+TOL)*quantum
                            bounds[axis+2] = math.ceil((mid+minimum/2)/quantum-TOL)*quantum
                    adds.append(box(*bounds))
                    action = 'widen'
                report['actions'].append({'pass':pass_number, 'phase':phase, 'action':action,
                                          'kind':'rectangular_component', 'bounds':list(component.bounds),
                                          'width_mm':min(widths)})
            for f in selected:
                if not f['rectangular']:
                    fail('oblique narrow feature requires explicit redesign', feature=f)
                rect = box(*f['bounds'])
                if f['width_mm'] < threshold-TOL:
                    removes.append(rect)
                    action = 'remove'
                else:
                    axis = 0 if f['axis'] == 'x' else 1
                    bounds = list(f['bounds'])
                    mid = (bounds[axis]+bounds[axis+2])/2
                    bounds[axis] = math.floor((mid-minimum/2)/quantum+TOL)*quantum
                    bounds[axis+2] = math.ceil((mid+minimum/2)/quantum-TOL)*quantum
                    adds.append(box(*bounds))
                    action = 'widen'
                report['actions'].append({'pass':pass_number, 'phase':phase,
                                          'action':action, **f})
            material = material.difference(unary_union(removes)).union(unary_union(adds))
            candidate = material if phase == 'copper' else frame.difference(material)
            candidate = candidate.intersection(board_grid if coordinates_quantized else board)
            if candidate.is_empty or not candidate.is_valid:
                fail('repair collapsed or invalidated copper')
            # A feature touching immutable feed geometry is rejected, not patched
            # back afterwards (which could silently restore an illegal sliver).
            if touches_protected(candidate):
                fail('repair would modify protected feed or SMA pads')
            current = set_precision(candidate,1e-9)
            continue

        snapped = quantize(current,quantum)
        if snapped.is_empty or not snapped.is_valid:
            fail('coordinate quantization collapsed or invalidated copper')
        # Quantization is deliberately after repair; validate both phases again.
        if _signature(snapped) != signature and (
                thin_features(snapped,minimum) or thin_features(frame.difference(snapped),minimum)):
            current = snapped
            coordinates_quantized = True
            continue
        current = snapped
        coordinates_quantized = True
        if current.symmetric_difference(scale(current,xfact=-1,origin=(0,0))).area > TOL:
            fail('repair lost mirror symmetry')
        if not quantize(board,quantum).buffer(TOL).covers(current):
            fail('quantization moved copper outside the substrate')
        if touches_protected(current):
            fail('quantization changed protected feed beyond normal coordinate rounding')
        proximity = _proximity_issues(current,minimum)
        if proximity:
            if mode == 'reject':
                fail('remaining narrow oblique/corner proximity', violations=proximity)
            issue = proximity[0]
            p,q = issue['points']
            if abs(p[0]-q[0]) > TOL and abs(p[1]-q[1]) > TOL:
                fail('remaining narrow oblique/corner proximity; unsafe diagonal repair', violations=proximity)
            if issue['distance_mm'] < threshold-TOL:
                fail('remaining narrow oblique/corner proximity; joint removal needs explicit redesign', violations=proximity)
            if report['joint_repairs'] >= max_joint_repairs:
                fail('joint repair limit exceeded', violations=proximity)
            cap = _joint_cap(issue,minimum,quantum)
            candidate = current.union(cap) if issue['phase'] == 'copper' else current.difference(cap)
            candidate = candidate.intersection(board_grid)
            if candidate.is_empty or not candidate.is_valid:
                fail('joint repair collapsed or invalidated copper', feature=issue)
            if touches_protected(candidate):
                fail('repair would modify protected feed or SMA pads', feature=issue)
            if candidate.symmetric_difference(current).area <= TOL:
                fail('joint repair made no progress', feature=issue)
            report['actions'].append({'pass':pass_number, 'kind':'joint_cap', 'action':'widen',
                                      **issue, 'cap_bounds':[list(g.bounds) for g in polygons(cap)],
                                      'added_area_mm2':candidate.difference(current).area,
                                      'removed_area_mm2':current.difference(candidate).area})
            report['joint_repairs'] += 1
            current = set_precision(candidate,1e-9)
            # Do NOT return immediately: recheck copper AND void, all proximities,
            # quantization, protected region and topology on the next pass.
            continue
        components = polygons(current)
        if any(a.intersects(b) for i,a in enumerate(components) for b in components[i+1:]):
            fail('zero-width point contact between copper components')
        # Ensure a formerly attached feed component did not become an isolated
        # feed-only fragment. The immutable mask includes the feed/patch junction.
        result = translate(current,yoff=origin)
        report.update(status='passed', after=_snapshot(result),
                      added_area_mm2=result.difference(copper).area,
                      removed_area_mm2=copper.difference(result).area,
                      coordinate_frame='bounds are mm relative to substrate bottom')
        return result,report
    fail('repair pass limit exceeded')
