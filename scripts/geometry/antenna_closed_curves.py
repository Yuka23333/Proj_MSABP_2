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


def polygon_exterior_to_closed_points(
    polygon: Polygon,
    *,
    curve_name: str = "antenna polygon",
    quantum_mm: float = antenna_outline.COORDINATE_QUANTUM_MM,
) -> list[Point2D]:
    """Delegate CST-ready quantization and curve validation to the geometry API."""

    return antenna_outline.polygon_exterior_to_closed_points(
        polygon,
        curve_name=curve_name,
        quantum_mm=quantum_mm,
    )


def generate_closed_curve_point_lists(
    parameters: antenna_outline.AntennaOutlineParameters | None = None,
    *,
    quantum_mm: float = antenna_outline.COORDINATE_QUANTUM_MM,
) -> list[list[Point2D]]:
    """Build the antenna and return three quantized, validated closed curves."""

    curves = antenna_outline.generate_complete_antenna_point_lists(
        parameters,
        quantum_mm=quantum_mm,
    )
    if len(curves) != len(CURVE_NAMES):
        raise ValueError(f"expected three curves, got {len(curves)}")
    return curves


def generate_named_closed_curve_points(
    parameters: antenna_outline.AntennaOutlineParameters | None = None,
    *,
    quantum_mm: float = antenna_outline.COORDINATE_QUANTUM_MM,
) -> dict[str, list[Point2D]]:
    """Return the same curves keyed by stable descriptive names."""

    curves = generate_closed_curve_point_lists(parameters, quantum_mm=quantum_mm)
    return dict(zip(CURVE_NAMES, curves, strict=True))


def main() -> int:
    named_curves = generate_named_closed_curve_points()
    for name, points in named_curves.items():
        print(f"{name}: {len(points)} points (closed={points[0] == points[-1]})")
        pprint(points, sort_dicts=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
