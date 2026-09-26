"""Extract every connected copper region after the quantized slot subtraction.

The feed-connected region is first; the other regions are sorted geometrically
for stable CST names. 'Detached' means disconnected from the feed in the planar
geometry, not necessarily electrically floating: some pieces may contact SMA
ground pads. All exterior and hole curves are retained, with CCW tool winding.
"""

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def final_conductor(vertices, source_holes=None):
    source_holes = source_holes or {}
    patch, slot, guide = (
        Polygon(vertices[k], source_holes.get(k, ()))
        for k in ("Patch", "Slot", "CPW_Feed_Pin")
    )
    if not all(p.is_valid and not p.is_empty for p in (patch, slot, guide)):
        raise ValueError(
            "Cannot extract conductor components from invalid source curves"
        )
    conductor = patch.difference(slot).union(guide)
    if not isinstance(conductor, (Polygon, MultiPolygon)) or not conductor.is_valid:
        raise ValueError("Slot subtraction did not produce valid polygonal copper")
    return conductor


def _ring_points(ring):
    return [list(p) for p in orient(Polygon(ring), sign=1.0).exterior.coords[:-1]]


def extract_conductor_components(vertices, source_holes=None):
    """Return all components, including a single component if no island exists."""
    conductor = final_conductor(vertices, source_holes)
    parts = (
        list(conductor.geoms) if isinstance(conductor, MultiPolygon) else [conductor]
    )
    guide = Polygon(vertices["CPW_Feed_Pin"])
    parts.sort(key=lambda p: (-p.intersection(guide).area, *p.bounds, -p.area))
    return [
        {
            "role": "feed_connected" if p.intersection(guide).area > 0 else "detached",
            "exterior": _ring_points(p.exterior),
            "holes": [_ring_points(ring) for ring in p.interiors],
            "area_mm2": float(p.area),
        }
        for p in parts
    ]


def validate_component_curves(vertices, components, source_holes=None):
    """Reject stale/incomplete/overlapping component curves in an extended export."""
    polygons = [Polygon(c.exterior, c.holes) for c in components]
    if not polygons or any(not p.is_valid or p.is_empty for p in polygons):
        raise ValueError(
            "Conductor component curves must define valid, nonempty polygons"
        )
    reference = final_conductor(vertices, source_holes)
    combined = unary_union(polygons)
    tolerance = max(1e-9, reference.area * 1e-10)
    if sum(p.area for p in polygons) - combined.area > tolerance:
        raise ValueError("Conductor component curves overlap")
    if reference.symmetric_difference(combined).area > tolerance:
        raise ValueError(
            "Conductor component curves do not match Patch - Slot + CPW_Feed_Pin"
        )
    if not Polygon(vertices["CPW_Feed_Pin"]).within(polygons[0]):
        raise ValueError("First conductor component must contain the feed pin")
    return polygons
