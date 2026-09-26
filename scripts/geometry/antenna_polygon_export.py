"""Read the polygon JSON exported by ``shapely_rectangle_test.py``.

The interchange format is::

    {
      "meta": {
        "quantize_step": 0.01,
        "global_min_y_before_shift": -21.0,
        "self_intersection_check": {...}
      },
      "vertices": {
        "Slot": [[x0, y0], [x1, y1], ...],
        "Patch": [[x0, y0], [x1, y1], ...],
        "CPW_Feed_Pin": [[x0, y0], [x1, y1], ...]
      }
    }

Coordinates are millimetres.  Every vertex array is ordered counterclockwise
by the producer and intentionally omits the duplicate closing point.  The CST
Polygon writer closes the last vertex back to the first.  This reader checks
only the transport schema and numeric types; it deliberately does not repeat
the producer's geometric validity, intersection, winding, or quantization
checks. An optional ``conductor_components`` array stores the final copper as
``{"exterior": [[x,y], ...], "holes": [[[x,y], ...], ...]}`` records, feed-connected
first. Optional ``source_holes`` maps source names to interior-ring arrays. Those
exports must include component records, so an internal island is not lost when
the three original source exteriors alone cannot describe the full geometry.
The CST builder validates the extended records against the source Boolean result.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_PATH = (
    REPOSITORY_ROOT / "results" / "processed" / "antenna_polygon_vertices.json"
)
PATCH_KEY = "Patch"
SLOT_KEY = "Slot"
FEED_PIN_KEY = "CPW_Feed_Pin"
REQUIRED_VERTEX_KEYS = (PATCH_KEY, SLOT_KEY, FEED_PIN_KEY)

Point2D = tuple[float, float]


@dataclass(frozen=True)
class ConductorComponentCurves:
    """One final copper region; first record is the feed-connected main body."""

    exterior: tuple[Point2D, ...]
    holes: tuple[tuple[Point2D, ...], ...] = ()


@dataclass(frozen=True)
class AntennaPolygonExport:
    """Three CST source curves plus the coordinate metadata from one export."""

    source_path: Path
    quantize_step_mm: float
    vertices: Mapping[str, tuple[Point2D, ...]]
    conductor_components: tuple[ConductorComponentCurves, ...] = ()
    source_holes: Mapping[str, tuple[tuple[Point2D, ...], ...]] = field(default_factory=dict)

    def points(self, name: str) -> list[Point2D]:
        """Return a mutable copy while preserving the JSON point order exactly."""

        return list(self.vertices[name])

    @property
    def substrate_bounds_mm(self) -> tuple[float, float, float, float]:
        """Use the exported Patch's global extent as the substrate rectangle."""

        patch = self.vertices[PATCH_KEY]
        x_values = tuple(point[0] for point in patch)
        y_values = tuple(point[1] for point in patch)
        return min(x_values), min(y_values), max(x_values), max(y_values)

    @property
    def substrate_size_mm(self) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = self.substrate_bounds_mm
        return max_x - min_x, max_y - min_y

    def substrate_rectangle_points(self) -> list[Point2D]:
        """Return a closed CCW rectangle spanning the exported Patch bounds."""

        min_x, min_y, max_x, max_y = self.substrate_bounds_mm
        return [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _points(value: Any, label: str) -> tuple[Point2D, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a JSON array of [x, y] points")
    result: list[Point2D] = []
    for index, raw_point in enumerate(value):
        if (
            not isinstance(raw_point, Sequence)
            or isinstance(raw_point, (str, bytes))
            or len(raw_point) != 2
        ):
            raise ValueError(f"{label}[{index}] must contain exactly [x, y]")
        x_value, y_value = raw_point
        if isinstance(x_value, bool) or isinstance(y_value, bool):
            raise ValueError(f"{label}[{index}] coordinates must be numbers")
        try:
            point = float(x_value), float(y_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] coordinates must be numbers") from exc
        if not all(math.isfinite(coordinate) for coordinate in point):
            raise ValueError(f"{label}[{index}] coordinates must be finite")
        result.append(point)
    if len(result) < 3:
        raise ValueError(f"{label} must contain at least three vertices")
    return tuple(result)


def polygon_export_from_payload(payload, source_path="<in-memory>") -> AntennaPolygonExport:
    """Decode the old three-curve format or its optional component-curve extension."""
    source = Path(source_path)
    payload = _mapping(payload, "polygon export")
    meta = _mapping(payload.get("meta"), "polygon export.meta")
    quantize_step = meta.get("quantize_step")
    if isinstance(quantize_step, bool):
        raise ValueError("polygon export.meta.quantize_step must be a number")
    try:
        quantize_step_mm = float(quantize_step)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "polygon export.meta.quantize_step must be a number"
        ) from exc
    if not math.isfinite(quantize_step_mm) or quantize_step_mm <= 0.0:
        raise ValueError("polygon export.meta.quantize_step must be positive")

    raw_vertices = _mapping(payload.get("vertices"), "polygon export.vertices")
    missing = set(REQUIRED_VERTEX_KEYS) - set(raw_vertices)
    if missing:
        raise ValueError(f"polygon export is missing curves: {sorted(missing)}")
    vertices = {
        name: _points(raw_vertices[name], f"polygon export.vertices.{name}")
        for name in REQUIRED_VERTEX_KEYS
    }
    raw_components = payload.get("conductor_components", [])
    if not isinstance(raw_components, list):
        raise ValueError("conductor_components must be an array")
    components = []
    for index, raw in enumerate(raw_components):
        raw = _mapping(raw, f"conductor_components[{index}]")
        holes = raw.get("holes", [])
        if not isinstance(holes, list):
            raise ValueError("conductor component holes must be an array")
        components.append(ConductorComponentCurves(
            _points(raw.get("exterior"), f"conductor_components[{index}].exterior"),
            tuple(_points(ring, f"conductor_components[{index}].holes[{j}]")
                  for j, ring in enumerate(holes)),
        ))
    raw_holes = _mapping(payload.get("source_holes", {}), "source_holes")
    source_holes = {}
    for name, rings in raw_holes.items():
        if name not in REQUIRED_VERTEX_KEYS or not isinstance(rings, list):
            raise ValueError("source_holes must map source names to arrays of hole curves")
        source_holes[name] = tuple(_points(ring, f"source_holes.{name}") for ring in rings)
    if any(source_holes.values()) and not components:
        raise ValueError("Exports with source holes require conductor_components")
    return AntennaPolygonExport(
        source_path=source,
        quantize_step_mm=quantize_step_mm,
        vertices=vertices,
        conductor_components=tuple(components),
        source_holes=source_holes,
    )


def load_antenna_polygon_export(
    path: str | Path = DEFAULT_EXPORT_PATH,
) -> AntennaPolygonExport:
    """Load curve data, checking schema/numeric types without geometric checks."""
    source = Path(path).expanduser().resolve()
    return polygon_export_from_payload(json.loads(source.read_text(encoding="utf-8")), source)
