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
checks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
class AntennaPolygonExport:
    """Three CST source curves plus the coordinate metadata from one export."""

    source_path: Path
    quantize_step_mm: float
    vertices: Mapping[str, tuple[Point2D, ...]]

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


def load_antenna_polygon_export(
    path: str | Path = DEFAULT_EXPORT_PATH,
) -> AntennaPolygonExport:
    """Load one exported JSON without performing geometry validation."""

    source = Path(path).expanduser().resolve()
    payload = _mapping(
        json.loads(source.read_text(encoding="utf-8")),
        "polygon export",
    )
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
    return AntennaPolygonExport(
        source_path=source,
        quantize_step_mm=quantize_step_mm,
        vertices=vertices,
    )
