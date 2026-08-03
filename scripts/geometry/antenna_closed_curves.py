"""Extract the three closed antenna curves as explicit ordered point lists.

The returned curves are, in order:

1. Patch exterior after the symmetric outer slots;
2. symmetric inner/CPW slot assembly;
3. symmetric CPW guide.

Each point list repeats its first point at the end, so every curve is
explicitly closed and can be passed directly to a CAD polygon interface.
"""

from __future__ import annotations

from pprint import pprint

from shapely.geometry import Polygon

try:
    from . import antenna_outline
except ImportError:
    import antenna_outline


Point2D = tuple[float, float]
CURVE_NAMES = (
    "patch",
    "symmetric_slot",
    "symmetric_guide",
)


def polygon_exterior_to_closed_points(polygon: Polygon) -> list[Point2D]:
    """Convert one hole-free Polygon exterior into an explicit closed list."""

    if polygon.is_empty:
        raise ValueError("cannot extract points from an empty Polygon")
    if not polygon.is_valid:
        raise ValueError("cannot extract points from an invalid Polygon")
    if len(polygon.interiors) != 0:
        raise ValueError(
            "expected one exterior curve without interior rings, "
            f"got {len(polygon.interiors)} interior ring(s)"
        )

    points = [
        (
            0.0 if float(x) == 0.0 else float(x),
            0.0 if float(y) == 0.0 else float(y),
        )
        for x, y in polygon.exterior.coords
    ]
    if len(points) < 4:
        raise ValueError("a closed polygon curve requires at least four points")
    if points[0] != points[-1]:
        raise ValueError("Shapely exterior coordinate sequence is not closed")
    return points


def generate_closed_curve_point_lists(
    parameters: antenna_outline.AntennaOutlineParameters | None = None,
) -> list[list[Point2D]]:
    """Build the antenna and return its three closed curves as point lists."""

    curves = antenna_outline.generate_complete_antenna_point_lists(parameters)
    if len(curves) != len(CURVE_NAMES):
        raise ValueError(f"expected three curves, got {len(curves)}")
    return curves


def generate_named_closed_curve_points(
    parameters: antenna_outline.AntennaOutlineParameters | None = None,
) -> dict[str, list[Point2D]]:
    """Return the same curves keyed by stable descriptive names."""

    curves = generate_closed_curve_point_lists(parameters)
    return dict(zip(CURVE_NAMES, curves, strict=True))


def main() -> int:
    named_curves = generate_named_closed_curve_points()
    for name, points in named_curves.items():
        print(f"{name}: {len(points)} points (closed={points[0] == points[-1]})")
        pprint(points, sort_dicts=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
