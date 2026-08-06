"""Build, validate, export, and preview the completed planar MSA-BP antenna.

Imports return the three explicit closed polygon point lists without plotting.
Running from a terminal prints only those lists.  An IDE F5/debug run displays
the complete colour-layered geometry preview.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, fields, replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from pprint import pprint
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import scale
from shapely.geometry import LineString, LinearRing, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity


# =============================================================================
# 带 FIXED 的坐标：不进入采样器
# =============================================================================
CPW_GUIDE_P1_X_FIXED_MM = 0.0
CPW_GUIDE_P1_Y_FIXED_MM = 0.0
CPW_GUIDE_P2_X_FIXED_MM = 0.5
CPW_GUIDE_P2_Y_FIXED_MM = 0.0
CPW_GUIDE_P3_Y_FIXED_MM = 0.2
CPW_GUIDE_P7_X_FIXED_MM = 0.0
CPW_SLOT_P0_X_FIXED_MM = 0.0
CPW_SLOT_P0_Y_FIXED_MM = 0.0
CPW_SLOT_P1_Y_FIXED_MM = 0.0
CPW_SLOT_P5_X_FIXED_MM = 0.0
CPW_MATCHING_STUB1_CAP_X_FIXED_MM = 3.5
CPW_MATCHING_STUB1_LOWER_Y_FIXED_MM = 5.5
CPW_MATCHING_STUB1_UPPER_Y_FIXED_MM = 6.5
CPW_MATCHING_STUB2_CAP_X_FIXED_MM = 3.0
CPW_MATCHING_STUB2_LOWER_Y_FIXED_MM = 8.0
CPW_MATCHING_STUB2_UPPER_Y_FIXED_MM = 8.9
REFLECTOR_CONNECTOR_BOARD_THICKNESS_FIXED_MM = 9.52
REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_FIXED_MM = 2.0
REFLECTOR_CUTOUT_DEPTH_FIXED_MM = 0.5
SMA_SOLDER_KEEPOUT_RIGHT_X_FIXED_MM = 4.76
SMA_SOLDER_KEEPOUT_UPPER_Y_FIXED_MM = 4.5
LOWER_OUTER_SLOT_CPW_CLEARANCE_ABS_X_FIXED_MM = 10.0
DOWNWARD_INNER_BRANCH_CPW_CLEARANCE_FIXED_MM = 5.0
INNER_SLOT_ORDER1_LENGTH_RATIO_MIN = 0.2
INNER_SLOT_ORDER1_LENGTH_RATIO_MAX = 0.9
OUTER_SLOT_ORDER1_HEIGHT_RATIO_MIN = 0.1
OUTER_SLOT_ORDER1_HEIGHT_RATIO_MAX = 0.4
OUTER_SLOT_CENTERLINE_PROTECTION_B_FIXED_MM = 10.0
UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MIN_FRACTION = 0.6
UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MAX_FRACTION = 1.0

COORDINATE_QUANTUM_MM = 0.01
GEOMETRY_TOLERANCE_MM2 = 1e-12
GEOMETRY_DISTANCE_TOLERANCE_MM = 1e-9
CPW_FEEDING_INTERFERENCE_ERROR = "这个参数会干扰CPW馈电"

Point2D = tuple[float, float]


@dataclass(frozen=True)
class AntennaOutlineParameters:
    """Independent numerical design parameters exposed to the sampler.

    Formula-only quantities such as centres, buffers, depths, and linked
    coordinates are properties below.  The arithmetic ``2.0`` used for
    half-widths and midpoints is deliberately not a sampled parameter.
    """

    # 常量组 1：外轮廓矩形
    rectangle_length_mm: float = 67.0
    rectangle_width_mm: float = 39.8

    # 常量组 2：外开槽-上部-1阶
    upper_outer_slot_order1_width_mm: float = 10.0
    upper_outer_slot_order1_height_ratio: float = (39.8 - 25.7) / 39.8

    # 常量组 3：外开槽-上部-2阶
    upper_outer_slot_order2_lower_y_ratio: float = (
        27.2 / 39.8 - UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MIN_FRACTION
    ) / (
        UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MAX_FRACTION
        - UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MIN_FRACTION
    )
    upper_outer_slot_order2_length_ratio: float = 12.0 / (67.0 / 2.0 - 10.0 - 10.0)

    # 常量组 4：外开槽-下部-1阶
    lower_outer_slot_order1_opposite_corner_x_mm: float = 22.2
    lower_outer_slot_order1_height_ratio: float = 12.0 / 39.8

    # 常量组 5：外开槽-下部-2阶
    lower_outer_slot_order2_branch1_length_ratio: float = 10.65 / (
        67.0 / 2.0 - (67.0 / 2.0 - 22.2) - 10.0
    )
    lower_outer_slot_order2_branch1_width_ratio: float = 9.0 / 12.0
    lower_outer_slot_order2_branch1_offset_ratio: float = 0.0
    lower_outer_slot_order2_branch2_length_ratio: float = 1.0
    lower_outer_slot_order2_branch2_width_ratio: float = 4.0 / 12.0
    lower_outer_slot_order2_branch2_offset_ratio: float = 0.0

    # 常量组 6：外开槽-Y轴镜像
    outer_slot_symmetry_axis_x_mm: float = 0.0

    # 常量组 7：内开槽-1阶
    inner_slot_order1_left_x_mm: float = 0.0
    inner_slot_order1_length_ratio: float = 26.5 / (67.0 / 2.0)
    inner_slot_order1_lower_y_mm: float = 20.1
    inner_slot_order1_upper_y_mm: float = 22.1

    # 常量组 8：内开槽-2阶
    inner_slot_order2_cap_left_x_mm: float = 5.5
    inner_slot_order2_cap_right_x_mm: float = 17.5
    inner_slot_order2_cap_y_mm: float = 23.9

    # 常量组 9：内开槽-2阶预留枝条
    inner_slot_order2_reserved_up_enabled: bool = False
    inner_slot_order2_reserved_up_anchor_t: float = 22.0 / 26.5
    inner_slot_order2_reserved_up_length_mm: float = 0.0
    inner_slot_order2_reserved_up_width_mm: float = 0.0
    inner_slot_order2_reserved_down1_enabled: bool = False
    inner_slot_order2_reserved_down1_anchor_t: float = 5.0 / 26.5
    inner_slot_order2_reserved_down1_length_mm: float = 0.0
    inner_slot_order2_reserved_down1_width_mm: float = 0.0
    inner_slot_order2_reserved_down2_enabled: bool = False
    inner_slot_order2_reserved_down2_anchor_t: float = 20.0 / 26.5
    inner_slot_order2_reserved_down2_length_mm: float = 0.0
    inner_slot_order2_reserved_down2_width_mm: float = 0.0

    # 常量组 10：CPW guide 可调参数
    cpw_guide_p1_x_mm: float = CPW_GUIDE_P1_X_FIXED_MM
    cpw_guide_p1_y_mm: float = CPW_GUIDE_P1_Y_FIXED_MM
    cpw_guide_p2_x_mm: float = CPW_GUIDE_P2_X_FIXED_MM
    cpw_guide_p2_y_mm: float = CPW_GUIDE_P2_Y_FIXED_MM
    cpw_guide_p3_y_mm: float = CPW_GUIDE_P3_Y_FIXED_MM
    cpw_guide_p7_x_mm: float = CPW_GUIDE_P7_X_FIXED_MM
    cpw_guide_p3_p4_x_mm: float = 1.375
    cpw_guide_p4_y_mm: float = 11.0
    cpw_guide_p5_p6_x_mm: float = 0.5

    # 常量组 11：CPW slot 可调参数
    cpw_slot_p0_x_mm: float = CPW_SLOT_P0_X_FIXED_MM
    cpw_slot_p0_y_mm: float = CPW_SLOT_P0_Y_FIXED_MM
    cpw_slot_p1_y_mm: float = CPW_SLOT_P1_Y_FIXED_MM
    cpw_slot_p5_x_mm: float = CPW_SLOT_P5_X_FIXED_MM
    cpw_slot_p1_p2_x_mm: float = 2.4
    cpw_slot_p2_y_mm: float = 11.0
    cpw_slot_p3_p4_x_mm: float = 1.7

    # 常量组 12：CPW matching stubs（默认固定）
    cpw_matching_stub1_cap_x_mm: float = CPW_MATCHING_STUB1_CAP_X_FIXED_MM
    cpw_matching_stub1_lower_y_mm: float = CPW_MATCHING_STUB1_LOWER_Y_FIXED_MM
    cpw_matching_stub1_upper_y_mm: float = CPW_MATCHING_STUB1_UPPER_Y_FIXED_MM
    cpw_matching_stub2_cap_x_mm: float = CPW_MATCHING_STUB2_CAP_X_FIXED_MM
    cpw_matching_stub2_lower_y_mm: float = CPW_MATCHING_STUB2_LOWER_Y_FIXED_MM
    cpw_matching_stub2_upper_y_mm: float = CPW_MATCHING_STUB2_UPPER_Y_FIXED_MM

    # 常量组 13：反射板连接器避让（默认固定）
    reflector_connector_board_thickness_mm: float = (
        REFLECTOR_CONNECTOR_BOARD_THICKNESS_FIXED_MM
    )
    reflector_cutout_width_adjustment_mm: float = (
        REFLECTOR_CUTOUT_WIDTH_ADJUSTMENT_FIXED_MM
    )
    reflector_cutout_depth_mm: float = REFLECTOR_CUTOUT_DEPTH_FIXED_MM

    @property
    def upper_outer_slot_order1_depth_mm(self) -> float:
        return self.rectangle_width_mm * self.upper_outer_slot_order1_height_ratio

    @property
    def upper_outer_slot_order1_bottom_y_mm(self) -> float:
        return self.rectangle_width_mm - self.upper_outer_slot_order1_depth_mm

    @property
    def upper_outer_slot_order2_buffer_mm(self) -> float:
        return (self.rectangle_width_mm - self.upper_outer_slot_order2_lower_y_mm) / 2.0

    @property
    def upper_outer_slot_order2_lower_y_mm(self) -> float:
        lower_fraction = UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MIN_FRACTION
        fraction_span = (
            UPPER_OUTER_SLOT_ORDER2_LOWER_Y_MAX_FRACTION - lower_fraction
        )
        return self.rectangle_width_mm * (
            lower_fraction + fraction_span * self.upper_outer_slot_order2_lower_y_ratio
        )

    @property
    def upper_outer_slot_order2_center_y_mm(self) -> float:
        return self.rectangle_width_mm - self.upper_outer_slot_order2_buffer_mm

    @property
    def upper_outer_slot_order2_max_length_mm(self) -> float:
        return (
            self.rectangle_length_mm / 2.0
            - self.upper_outer_slot_order1_width_mm
            - OUTER_SLOT_CENTERLINE_PROTECTION_B_FIXED_MM
        )

    @property
    def upper_outer_slot_order2_line_length_mm(self) -> float:
        return (
            self.upper_outer_slot_order2_length_ratio
            * self.upper_outer_slot_order2_max_length_mm
        )

    @property
    def lower_outer_slot_order1_width_mm(self) -> float:
        return (
            self.rectangle_length_mm / 2.0
            - self.lower_outer_slot_order1_opposite_corner_x_mm
        )

    @property
    def lower_outer_slot_order1_height_mm(self) -> float:
        return self.rectangle_width_mm * self.lower_outer_slot_order1_height_ratio

    @property
    def lower_outer_slot_order1_opposite_corner_y_mm(self) -> float:
        return self.lower_outer_slot_order1_height_mm

    @property
    def lower_outer_slot_order1_buffer_mm(self) -> float:
        return self.lower_outer_slot_order1_width_mm / 2.0

    @property
    def lower_outer_slot_order1_center_x_mm(self) -> float:
        return self.rectangle_length_mm / 2.0 - self.lower_outer_slot_order1_buffer_mm

    @property
    def lower_outer_slot_order2_branch1_center_y_mm(self) -> float:
        available_offset_mm = (
            self.lower_outer_slot_order1_height_mm
            - self.lower_outer_slot_order2_branch1_width_mm
        )
        return (
            self.lower_outer_slot_order1_height_mm / 2.0
            + self.lower_outer_slot_order2_branch1_offset_ratio
            * available_offset_mm
        )

    @property
    def lower_outer_slot_order2_branch1_width_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch1_width_ratio
            * self.lower_outer_slot_order1_height_mm
        )

    @property
    def lower_outer_slot_order2_branch1_lower_y_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch1_center_y_mm
            - self.lower_outer_slot_order2_branch1_width_mm / 2.0
        )

    @property
    def lower_outer_slot_order2_branch1_upper_y_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch1_center_y_mm
            + self.lower_outer_slot_order2_branch1_width_mm / 2.0
        )

    @property
    def lower_outer_slot_order2_branch1_buffer_mm(self) -> float:
        return self.lower_outer_slot_order2_branch1_width_mm / 2.0

    @property
    def lower_outer_slot_order2_max_length_mm(self) -> float:
        return (
            self.rectangle_length_mm / 2.0
            - self.lower_outer_slot_order1_width_mm
            - OUTER_SLOT_CENTERLINE_PROTECTION_B_FIXED_MM
        )

    @property
    def lower_outer_slot_order2_branch1_line_length_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch1_length_ratio
            * self.lower_outer_slot_order2_max_length_mm
        )

    @property
    def lower_outer_slot_order2_branch1_inner_x_mm(self) -> float:
        return (
            self.lower_outer_slot_order1_center_x_mm
            - self.lower_outer_slot_order2_branch1_line_length_mm
        )

    @property
    def lower_outer_slot_order2_branch2_center_y_mm(self) -> float:
        available_offset_mm = (
            self.lower_outer_slot_order1_height_mm
            - self.lower_outer_slot_order2_branch2_width_mm
        )
        return (
            self.lower_outer_slot_order1_height_mm / 2.0
            + self.lower_outer_slot_order2_branch2_offset_ratio
            * available_offset_mm
        )

    @property
    def lower_outer_slot_order2_branch2_width_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch2_width_ratio
            * self.lower_outer_slot_order1_height_mm
        )

    @property
    def lower_outer_slot_order2_branch2_lower_y_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch2_center_y_mm
            - self.lower_outer_slot_order2_branch2_width_mm / 2.0
        )

    @property
    def lower_outer_slot_order2_branch2_upper_y_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch2_center_y_mm
            + self.lower_outer_slot_order2_branch2_width_mm / 2.0
        )

    @property
    def lower_outer_slot_order2_branch2_buffer_mm(self) -> float:
        return self.lower_outer_slot_order2_branch2_width_mm / 2.0

    @property
    def lower_outer_slot_order2_branch2_line_length_mm(self) -> float:
        return (
            self.lower_outer_slot_order2_branch2_length_ratio
            * self.lower_outer_slot_order2_max_length_mm
        )

    @property
    def lower_outer_slot_order2_branch2_inner_x_mm(self) -> float:
        return (
            self.lower_outer_slot_order1_center_x_mm
            - self.lower_outer_slot_order2_branch2_line_length_mm
        )

    @property
    def inner_slot_order1_center_y_mm(self) -> float:
        return (
            self.inner_slot_order1_lower_y_mm + self.inner_slot_order1_upper_y_mm
        ) / 2.0

    @property
    def inner_slot_order1_line_length_mm(self) -> float:
        return self.inner_slot_order1_right_x_mm - self.inner_slot_order1_left_x_mm

    @property
    def inner_slot_order1_right_x_mm(self) -> float:
        available_positive_half_span_mm = (
            self.rectangle_length_mm / 2.0 - self.inner_slot_order1_left_x_mm
        )
        return (
            self.inner_slot_order1_left_x_mm
            + self.inner_slot_order1_length_ratio * available_positive_half_span_mm
        )

    @property
    def inner_slot_order1_buffer_mm(self) -> float:
        return (
            self.inner_slot_order1_upper_y_mm - self.inner_slot_order1_lower_y_mm
        ) / 2.0

    @property
    def inner_slot_order2_center_x_mm(self) -> float:
        return (
            self.inner_slot_order2_cap_left_x_mm + self.inner_slot_order2_cap_right_x_mm
        ) / 2.0

    @property
    def inner_slot_order2_start_y_mm(self) -> float:
        return self.inner_slot_order1_center_y_mm

    @property
    def inner_slot_order2_line_length_mm(self) -> float:
        return self.inner_slot_order2_cap_y_mm - self.inner_slot_order2_start_y_mm

    @property
    def inner_slot_order2_buffer_mm(self) -> float:
        return (
            self.inner_slot_order2_cap_right_x_mm - self.inner_slot_order2_cap_left_x_mm
        ) / 2.0

    @property
    def inner_slot_order2_reserved_anchor_y_mm(self) -> float:
        return self.inner_slot_order1_center_y_mm

    def inner_slot_order1_x_at(self, anchor_t: float) -> float:
        """Return a coordinate attached to the order-1 centreline by ratio."""

        return self.inner_slot_order1_left_x_mm + anchor_t * (
            self.inner_slot_order1_right_x_mm - self.inner_slot_order1_left_x_mm
        )

    @property
    def inner_slot_order2_reserved_up_anchor_x_mm(self) -> float:
        return self.inner_slot_order1_x_at(self.inner_slot_order2_reserved_up_anchor_t)

    @property
    def inner_slot_order2_reserved_down1_anchor_x_mm(self) -> float:
        return self.inner_slot_order1_x_at(
            self.inner_slot_order2_reserved_down1_anchor_t
        )

    @property
    def inner_slot_order2_reserved_down2_anchor_x_mm(self) -> float:
        return self.inner_slot_order1_x_at(
            self.inner_slot_order2_reserved_down2_anchor_t
        )

    @property
    def cpw_guide_y2_linked_mm(self) -> float:
        return self.inner_slot_order1_upper_y_mm

    @property
    def cpw_slot_y1_linked_mm(self) -> float:
        return self.inner_slot_order1_lower_y_mm


DEFAULT_ANTENNA_PARAMETERS = AntennaOutlineParameters()
BRANCH_FIELDS: dict[str, Mapping[str, object]] = {
    "reserved_up_1": {
        "enabled": "inner_slot_order2_reserved_up_enabled",
        "parameters": (
            "inner_slot_order2_reserved_up_anchor_t",
            "inner_slot_order2_reserved_up_length_mm",
            "inner_slot_order2_reserved_up_width_mm",
        ),
    },
    "reserved_down_1": {
        "enabled": "inner_slot_order2_reserved_down1_enabled",
        "parameters": (
            "inner_slot_order2_reserved_down1_anchor_t",
            "inner_slot_order2_reserved_down1_length_mm",
            "inner_slot_order2_reserved_down1_width_mm",
        ),
    },
    "reserved_down_2": {
        "enabled": "inner_slot_order2_reserved_down2_enabled",
        "parameters": (
            "inner_slot_order2_reserved_down2_anchor_t",
            "inner_slot_order2_reserved_down2_length_mm",
            "inner_slot_order2_reserved_down2_width_mm",
        ),
    },
}
PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "outline": (
        "rectangle_length_mm",
        "rectangle_width_mm",
    ),
    "outer.upper.order1": (
        "upper_outer_slot_order1_width_mm",
        "upper_outer_slot_order1_height_ratio",
    ),
    "outer.upper.order2": (
        "upper_outer_slot_order2_lower_y_ratio",
        "upper_outer_slot_order2_length_ratio",
    ),
    "outer.lower.order1": (
        "lower_outer_slot_order1_opposite_corner_x_mm",
        "lower_outer_slot_order1_height_ratio",
    ),
    "outer.lower.order2.branch1": (
        "lower_outer_slot_order2_branch1_length_ratio",
        "lower_outer_slot_order2_branch1_width_ratio",
        "lower_outer_slot_order2_branch1_offset_ratio",
    ),
    "outer.lower.order2.branch2": (
        "lower_outer_slot_order2_branch2_length_ratio",
        "lower_outer_slot_order2_branch2_width_ratio",
        "lower_outer_slot_order2_branch2_offset_ratio",
    ),
    "outer.symmetry": ("outer_slot_symmetry_axis_x_mm",),
    "inner.order1": (
        "inner_slot_order1_left_x_mm",
        "inner_slot_order1_length_ratio",
        "inner_slot_order1_lower_y_mm",
        "inner_slot_order1_upper_y_mm",
    ),
    "inner.order2.primary": (
        "inner_slot_order2_cap_left_x_mm",
        "inner_slot_order2_cap_right_x_mm",
        "inner_slot_order2_cap_y_mm",
    ),
    "inner.order2.reserved.up.control": (
        "inner_slot_order2_reserved_up_enabled",
    ),
    "inner.order2.reserved.up.position": (
        "inner_slot_order2_reserved_up_anchor_t",
    ),
    "inner.order2.reserved.up.size": (
        "inner_slot_order2_reserved_up_length_mm",
        "inner_slot_order2_reserved_up_width_mm",
    ),
    "inner.order2.reserved.down1.control": (
        "inner_slot_order2_reserved_down1_enabled",
    ),
    "inner.order2.reserved.down1.position": (
        "inner_slot_order2_reserved_down1_anchor_t",
    ),
    "inner.order2.reserved.down1.size": (
        "inner_slot_order2_reserved_down1_length_mm",
        "inner_slot_order2_reserved_down1_width_mm",
    ),
    "inner.order2.reserved.down2.control": (
        "inner_slot_order2_reserved_down2_enabled",
    ),
    "inner.order2.reserved.down2.position": (
        "inner_slot_order2_reserved_down2_anchor_t",
    ),
    "inner.order2.reserved.down2.size": (
        "inner_slot_order2_reserved_down2_length_mm",
        "inner_slot_order2_reserved_down2_width_mm",
    ),
    "cpw.guide.fixed": (
        "cpw_guide_p1_x_mm",
        "cpw_guide_p1_y_mm",
        "cpw_guide_p2_x_mm",
        "cpw_guide_p2_y_mm",
        "cpw_guide_p3_y_mm",
        "cpw_guide_p7_x_mm",
    ),
    "cpw.guide.design": (
        "cpw_guide_p3_p4_x_mm",
        "cpw_guide_p4_y_mm",
        "cpw_guide_p5_p6_x_mm",
    ),
    "cpw.slot.fixed": (
        "cpw_slot_p0_x_mm",
        "cpw_slot_p0_y_mm",
        "cpw_slot_p1_y_mm",
        "cpw_slot_p5_x_mm",
    ),
    "cpw.slot.design": (
        "cpw_slot_p1_p2_x_mm",
        "cpw_slot_p2_y_mm",
        "cpw_slot_p3_p4_x_mm",
    ),
    "cpw.matching_stub1.fixed": (
        "cpw_matching_stub1_cap_x_mm",
        "cpw_matching_stub1_lower_y_mm",
        "cpw_matching_stub1_upper_y_mm",
    ),
    "cpw.matching_stub2.fixed": (
        "cpw_matching_stub2_cap_x_mm",
        "cpw_matching_stub2_lower_y_mm",
        "cpw_matching_stub2_upper_y_mm",
    ),
    "reflector.clearance.fixed": (
        "reflector_connector_board_thickness_mm",
        "reflector_cutout_width_adjustment_mm",
        "reflector_cutout_depth_mm",
    ),
}
STRUCTURAL_PARAMETER_NAMES = {
    str(details["enabled"]) for details in BRANCH_FIELDS.values()
}
FIXED_BY_DEFAULT_PARAMETER_NAMES = {
    "outer_slot_symmetry_axis_x_mm",
    "inner_slot_order1_left_x_mm",
    *PARAMETER_GROUPS["cpw.guide.fixed"],
    *PARAMETER_GROUPS["cpw.slot.fixed"],
    *PARAMETER_GROUPS["cpw.matching_stub1.fixed"],
    *PARAMETER_GROUPS["cpw.matching_stub2.fixed"],
    *PARAMETER_GROUPS["reflector.clearance.fixed"],
}
ACTIVE_BRANCH_BY_PARAMETER = {
    str(parameter_name): branch_name
    for branch_name, details in BRANCH_FIELDS.items()
    for parameter_name in details["parameters"]  # type: ignore[union-attr]
}
INDEPENDENT_PARAMETER_NAMES = tuple(
    item.name for item in fields(AntennaOutlineParameters)
)
SAMPLABLE_PARAMETER_NAMES = tuple(
    name
    for name in INDEPENDENT_PARAMETER_NAMES
    if name not in STRUCTURAL_PARAMETER_NAMES
    and name not in FIXED_BY_DEFAULT_PARAMETER_NAMES
)
DEFAULT_EXPLORER_LOG_PATH = (
    Path(__file__).resolve().parents[2] / "logs" / "antenna_outline_explorer.log"
)


@dataclass(frozen=True)
class ExplorerSliderSpec:
    """Translate one GUI slider position into an antenna parameter value."""

    parameter_name: str
    mode: str
    minimum: float
    maximum: float
    resolution: float
    reference_value: float

    def parameter_value(self, slider_value: float) -> float:
        if self.mode == "ratio":
            return float(slider_value)
        return self.reference_value * float(slider_value) / 100.0

    def slider_value(self, parameter_value: float) -> float:
        if self.mode == "ratio":
            value = float(parameter_value)
        else:
            value = 100.0 * float(parameter_value) / self.reference_value
        return min(self.maximum, max(self.minimum, value))


def explorer_slider_spec(
    parameter_name: str,
    reference_parameters: AntennaOutlineParameters | None = None,
) -> ExplorerSliderSpec:
    """Return the direct-ratio or 0--200 percent slider definition."""

    reference_parameters = _resolve_parameters(reference_parameters)
    if parameter_name not in SAMPLABLE_PARAMETER_NAMES:
        raise ValueError(f"parameter is not explorer-adjustable: {parameter_name}")
    reference_value = float(getattr(reference_parameters, parameter_name))
    if parameter_name.endswith(("_anchor_t", "_ratio")):
        if parameter_name == "inner_slot_order1_length_ratio":
            minimum = INNER_SLOT_ORDER1_LENGTH_RATIO_MIN
            maximum = INNER_SLOT_ORDER1_LENGTH_RATIO_MAX
        elif parameter_name in {
            "upper_outer_slot_order1_height_ratio",
            "lower_outer_slot_order1_height_ratio",
        }:
            minimum = OUTER_SLOT_ORDER1_HEIGHT_RATIO_MIN
            maximum = OUTER_SLOT_ORDER1_HEIGHT_RATIO_MAX
        elif parameter_name.endswith("_offset_ratio"):
            minimum, maximum = -1.0, 1.0
        else:
            minimum, maximum = 0.0, 1.0
        return ExplorerSliderSpec(
            parameter_name=parameter_name,
            mode="ratio",
            minimum=minimum,
            maximum=maximum,
            resolution=0.01,
            reference_value=reference_value,
        )
    if reference_value <= 0.0:
        raise ValueError(
            f"percent slider requires a positive reference value: {parameter_name}"
        )
    return ExplorerSliderSpec(
        parameter_name=parameter_name,
        mode="percent",
        minimum=0.0,
        maximum=200.0,
        resolution=1.0,
        reference_value=reference_value,
    )


def explorer_parameter_groups(
    reference_parameters: AntennaOutlineParameters | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return active, non-FIXED numerical groups available in the F5 explorer."""

    reference_parameters = _resolve_parameters(reference_parameters)
    result: dict[str, tuple[str, ...]] = {}
    for group_name, parameter_names in PARAMETER_GROUPS.items():
        available: list[str] = []
        for parameter_name in parameter_names:
            if parameter_name not in SAMPLABLE_PARAMETER_NAMES:
                continue
            branch_name = ACTIVE_BRANCH_BY_PARAMETER.get(parameter_name)
            if branch_name is not None:
                enabled_name = str(BRANCH_FIELDS[branch_name]["enabled"])
                if not bool(getattr(reference_parameters, enabled_name)):
                    continue
            try:
                explorer_slider_spec(parameter_name, reference_parameters)
            except ValueError:
                continue
            available.append(parameter_name)
        if available:
            result[group_name] = tuple(available)
    return result


@dataclass(frozen=True)
class GeometryReport:
    """Validated geometric properties of the rectangle."""

    point_count: int
    bounds: tuple[float, float, float, float]
    area_mm2: float
    perimeter_mm: float
    is_valid: bool
    is_simple: bool
    is_counterclockwise: bool


@dataclass(frozen=True)
class UpperOuterSlotReport:
    """Validated properties of the upper first-order outer slot."""

    centerline: tuple[Point2D, Point2D]
    slot_bounds: tuple[float, float, float, float]
    slot_area_mm2: float
    outline_area_mm2: float
    outline_is_valid: bool
    outline_is_simple: bool
    outline_is_counterclockwise: bool


@dataclass(frozen=True)
class UpperOuterSlotOrder2Report:
    """Validated properties after adding the upper second-order outer slot."""

    centerline: tuple[Point2D, Point2D]
    slot_bounds: tuple[float, float, float, float]
    slot_area_mm2: float
    combined_slot_area_mm2: float
    outline_area_mm2: float
    outline_is_valid: bool
    outline_is_simple: bool
    outline_is_counterclockwise: bool


@dataclass(frozen=True)
class LowerOuterSlotOrder1Report:
    """Validated properties after adding the lower first-order outer slot."""

    centerline: tuple[Point2D, Point2D]
    slot_bounds: tuple[float, float, float, float]
    slot_area_mm2: float
    combined_slot_area_mm2: float
    outline_area_mm2: float
    outline_is_valid: bool
    outline_is_simple: bool
    outline_is_counterclockwise: bool


@dataclass(frozen=True)
class LowerOuterSlotOrder2Report:
    """Validated properties after adding both lower second-order branches."""

    centerlines: tuple[
        tuple[Point2D, Point2D],
        tuple[Point2D, Point2D],
    ]
    slot_bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    slot_areas_mm2: tuple[float, float]
    lower_combined_slot_area_mm2: float
    combined_slot_area_mm2: float
    outline_area_mm2: float
    outline_is_valid: bool
    outline_is_simple: bool
    outline_is_counterclockwise: bool


@dataclass(frozen=True)
class SymmetricOuterSlotsReport:
    """Validated properties after mirroring all outer slots about the y-axis."""

    symmetry_axis_x_mm: float
    right_slot_bounds: tuple[float, float, float, float]
    left_slot_bounds: tuple[float, float, float, float]
    right_slot_area_mm2: float
    left_slot_area_mm2: float
    combined_slot_area_mm2: float
    outline_area_mm2: float
    combined_slot_is_symmetric: bool
    outline_is_symmetric: bool
    outline_is_valid: bool
    outline_is_simple: bool
    outline_is_counterclockwise: bool


@dataclass(frozen=True)
class InnerSlotOrder1Report:
    """Validated properties of the unmirrored, non-subtracted inner slot."""

    centerline: tuple[Point2D, Point2D]
    slot_bounds: tuple[float, float, float, float]
    slot_area_mm2: float
    patch_area_before_mm2: float
    patch_area_after_mm2: float
    patch_is_unchanged: bool
    slot_was_subtracted: bool
    slot_is_y_axis_symmetric: bool


@dataclass(frozen=True)
class InnerSlotBranchReservation:
    """Inactive L-system branch parameters reserved for later refinement."""

    name: str
    parent_name: str
    anchor_t: float
    anchor: Point2D
    growth_direction: Point2D
    geometry_type: str
    enabled: bool
    length_mm: float
    width_mm: float

    @property
    def is_active(self) -> bool:
        """Return the explicit geometry activation state."""

        return self.enabled


@dataclass(frozen=True)
class InnerSlotOrder2Report:
    """Validated properties of the Y+ second-order inner-slot branch."""

    growth_direction: Point2D
    centerline: tuple[Point2D, Point2D]
    slot_bounds: tuple[float, float, float, float]
    slot_area_mm2: float
    overlap_with_order1_mm2: float
    combined_inner_slot_area_mm2: float
    patch_area_before_mm2: float
    patch_area_after_mm2: float
    patch_is_unchanged: bool
    slot_was_subtracted: bool
    combined_slot_is_y_axis_symmetric: bool
    reserved_branches: tuple[
        InnerSlotBranchReservation,
        InnerSlotBranchReservation,
        InnerSlotBranchReservation,
    ]
    reserved_active_count: int


@dataclass(frozen=True)
class CpwGuideReport:
    """Validated properties of the unmirrored, non-subtracted CPW guide."""

    parameters: AntennaOutlineParameters
    anchor_points: tuple[
        Point2D,
        Point2D,
        Point2D,
        Point2D,
        Point2D,
        Point2D,
        Point2D,
    ]
    y1_mm: float
    y2_mm: float
    guide_bounds: tuple[float, float, float, float]
    guide_area_mm2: float
    overlap_with_inner_slot_mm2: float
    combined_step3_area_mm2: float
    patch_area_before_mm2: float
    patch_area_after_mm2: float
    patch_is_unchanged: bool
    guide_was_subtracted: bool
    guide_is_y_axis_symmetric: bool
    guide_is_valid: bool
    guide_is_counterclockwise: bool


@dataclass(frozen=True)
class CpwSlotAssemblyReport:
    """Validated CPW slot, matching stubs, inner branches, and mirrored guide."""

    parameters: AntennaOutlineParameters
    anchor_points: tuple[
        Point2D,
        Point2D,
        Point2D,
        Point2D,
        Point2D,
        Point2D,
    ]
    p3_y_mm: float
    y1_mm: float
    slot_bounds: tuple[float, float, float, float]
    slot_area_mm2: float
    stub_centerlines: tuple[
        tuple[Point2D, Point2D],
        tuple[Point2D, Point2D],
    ]
    stub_bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    stub_areas_mm2: tuple[float, float]
    slot_with_stubs_area_mm2: float
    inner_main_branch_count: int
    inner_order2_branch_count: int
    inner_order2_active_count: int
    right_combined_slot_area_mm2: float
    symmetric_slot_area_mm2: float
    right_guide_area_mm2: float
    symmetric_guide_area_mm2: float
    patch_area_before_mm2: float
    patch_area_after_mm2: float
    patch_is_unchanged: bool
    slot_was_subtracted: bool
    slot_assembly_is_y_axis_symmetric: bool
    guide_is_y_axis_symmetric: bool


def _require_finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _resolve_parameters(
    parameters: AntennaOutlineParameters | None,
) -> AntennaOutlineParameters:
    return DEFAULT_ANTENNA_PARAMETERS if parameters is None else parameters


def _reject_lower_outer_slot_cpw_interference(slot: Polygon) -> None:
    """Keep every right-side lower outer slot outside the central CPW corridor."""

    minimum_x_mm = float(slot.bounds[0])
    if minimum_x_mm < LOWER_OUTER_SLOT_CPW_CLEARANCE_ABS_X_FIXED_MM:
        raise ValueError(CPW_FEEDING_INTERFERENCE_ERROR)


def generate_rectangle(
    parameters: AntennaOutlineParameters | None = None,
) -> list[Point2D]:
    """Return five ordered 2D points describing the closed rectangle.

    The lower-left corner is ``(-rectangle_length_mm / 2, 0)``. Point order
    is counterclockwise, the bottom edge lies on the x-axis, and the shape is
    symmetric about the y-axis.
    """

    parameters = _resolve_parameters(parameters)
    length_mm = _require_finite_positive(
        parameters.rectangle_length_mm, "rectangle_length_mm"
    )
    width_mm = _require_finite_positive(
        parameters.rectangle_width_mm, "rectangle_width_mm"
    )
    half_length_mm = length_mm / 2.0

    return [
        (-half_length_mm, 0.0),
        (half_length_mm, 0.0),
        (half_length_mm, width_mm),
        (-half_length_mm, width_mm),
        (-half_length_mm, 0.0),
    ]


def validate_rectangle(
    points: Sequence[Point2D],
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[Polygon, GeometryReport]:
    """Validate closure, dimensions, winding, and Shapely geometry."""

    parameters = _resolve_parameters(parameters)
    length_mm = _require_finite_positive(
        parameters.rectangle_length_mm, "rectangle_length_mm"
    )
    width_mm = _require_finite_positive(
        parameters.rectangle_width_mm, "rectangle_width_mm"
    )
    half_length_mm = length_mm / 2.0

    coordinates = np.asarray(points, dtype=float)
    if coordinates.shape != (5, 2):
        raise ValueError(
            f"rectangle must contain five 2D points, got shape {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("rectangle contains a non-finite coordinate")
    if not np.array_equal(coordinates[0], coordinates[-1]):
        raise ValueError("rectangle point sequence is not closed")

    polygon = Polygon(coordinates)
    expected_bounds = (-half_length_mm, 0.0, half_length_mm, width_mm)
    expected_area = length_mm * width_mm
    expected_perimeter = 2.0 * (length_mm + width_mm)

    checks = {
        "Shapely polygon is valid": polygon.is_valid,
        "exterior ring is simple": polygon.exterior.is_simple,
        "exterior ring is counterclockwise": polygon.exterior.is_ccw,
        "bounds match requested dimensions": np.allclose(
            polygon.bounds, expected_bounds, rtol=0.0, atol=1e-12
        ),
        "area matches width * height": math.isclose(
            polygon.area, expected_area, rel_tol=0.0, abs_tol=1e-9
        ),
        "perimeter matches rectangle perimeter": math.isclose(
            polygon.length, expected_perimeter, rel_tol=0.0, abs_tol=1e-9
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("rectangle validation failed: " + "; ".join(failures))

    report = GeometryReport(
        point_count=len(points),
        bounds=tuple(float(value) for value in polygon.bounds),
        area_mm2=float(polygon.area),
        perimeter_mm=float(polygon.length),
        is_valid=bool(polygon.is_valid),
        is_simple=bool(polygon.exterior.is_simple),
        is_counterclockwise=bool(polygon.exterior.is_ccw),
    )
    return polygon, report


def generate_upper_outer_slot_centerline(
    parameters: AntennaOutlineParameters | None = None,
) -> LineString:
    """Create the vertical construction line for step 2.1.

    The line is positioned at ``L/2 - slot_width/2`` on the positive-x side.
    It runs from the rectangle's top edge down by the slot depth.
    """

    parameters = _resolve_parameters(parameters)
    length_mm = _require_finite_positive(
        parameters.rectangle_length_mm, "rectangle_length_mm"
    )
    rectangle_width_mm = _require_finite_positive(
        parameters.rectangle_width_mm, "rectangle_width_mm"
    )
    slot_width_mm = _require_finite_positive(
        parameters.upper_outer_slot_order1_width_mm,
        "upper_outer_slot_order1_width_mm",
    )
    slot_depth_mm = _require_finite_positive(
        parameters.upper_outer_slot_order1_depth_mm,
        "upper_outer_slot_order1_depth_mm",
    )
    if slot_width_mm > length_mm or slot_depth_mm > rectangle_width_mm:
        raise ValueError("upper outer slot does not fit inside the outer rectangle")

    center_x_mm = length_mm / 2.0 - slot_width_mm / 2.0
    y_top_mm = rectangle_width_mm
    y_bottom_mm = y_top_mm - slot_depth_mm
    return LineString(
        [
            (center_x_mm, y_top_mm),
            (center_x_mm, y_bottom_mm),
        ]
    )


def expand_upper_outer_slot(
    centerline: LineString,
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Expand the step-2.1 centreline by half the slot width on both sides."""

    parameters = _resolve_parameters(parameters)
    slot_width_mm = _require_finite_positive(
        parameters.upper_outer_slot_order1_width_mm,
        "upper_outer_slot_order1_width_mm",
    )
    slot = centerline.buffer(slot_width_mm / 2.0, cap_style="flat")
    if not isinstance(slot, Polygon):
        raise TypeError(f"expected a Polygon slot, got {slot.geom_type}")
    return orient(slot, sign=1.0)


def build_step_2_1_outline(
    rectangle: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[LineString, Polygon, Polygon, UpperOuterSlotReport]:
    """Subtract the expanded upper outer slot from the rectangle."""

    parameters = _resolve_parameters(parameters)
    centerline = generate_upper_outer_slot_centerline(parameters)
    slot = expand_upper_outer_slot(centerline, parameters)
    if not rectangle.covers(slot):
        raise ValueError("expanded upper outer slot lies outside the rectangle")

    outline = rectangle.difference(slot)
    if not isinstance(outline, Polygon):
        raise TypeError(
            "step 2.1 must produce one Polygon, "
            f"but Shapely returned {outline.geom_type}"
        )
    outline = orient(outline, sign=1.0)

    length_mm = parameters.rectangle_length_mm
    rectangle_width_mm = parameters.rectangle_width_mm
    slot_width_mm = parameters.upper_outer_slot_order1_width_mm
    slot_depth_mm = parameters.upper_outer_slot_order1_depth_mm
    expected_center_x_mm = length_mm / 2.0 - slot_width_mm / 2.0
    expected_y_bottom_mm = rectangle_width_mm - slot_depth_mm
    expected_slot_bounds = (
        length_mm / 2.0 - slot_width_mm,
        expected_y_bottom_mm,
        length_mm / 2.0,
        rectangle_width_mm,
    )
    expected_slot_area_mm2 = slot_width_mm * slot_depth_mm
    expected_outline_area_mm2 = rectangle.area - expected_slot_area_mm2
    centerline_coordinates = tuple((float(x), float(y)) for x, y in centerline.coords)

    checks = {
        "centerline x-position": all(
            math.isclose(x, expected_center_x_mm, rel_tol=0.0, abs_tol=1e-12)
            for x, _ in centerline_coordinates
        ),
        "centerline endpoints": np.allclose(
            centerline_coordinates,
            (
                (expected_center_x_mm, rectangle_width_mm),
                (expected_center_x_mm, expected_y_bottom_mm),
            ),
            rtol=0.0,
            atol=1e-12,
        ),
        "expanded slot bounds": np.allclose(
            slot.bounds, expected_slot_bounds, rtol=0.0, atol=1e-12
        ),
        "expanded slot area": math.isclose(
            slot.area, expected_slot_area_mm2, rel_tol=0.0, abs_tol=1e-9
        ),
        "outline area after subtraction": math.isclose(
            outline.area, expected_outline_area_mm2, rel_tol=0.0, abs_tol=1e-9
        ),
        "outline is valid": outline.is_valid,
        "outline exterior is simple": outline.exterior.is_simple,
        "outline exterior is counterclockwise": outline.exterior.is_ccw,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 2.1 validation failed: " + "; ".join(failures))

    report = UpperOuterSlotReport(
        centerline=(
            centerline_coordinates[0],
            centerline_coordinates[1],
        ),
        slot_bounds=tuple(float(value) for value in slot.bounds),
        slot_area_mm2=float(slot.area),
        outline_area_mm2=float(outline.area),
        outline_is_valid=bool(outline.is_valid),
        outline_is_simple=bool(outline.exterior.is_simple),
        outline_is_counterclockwise=bool(outline.exterior.is_ccw),
    )
    return centerline, slot, outline, report


def generate_upper_outer_slot_order2_centerline(
    parameters: AntennaOutlineParameters | None = None,
) -> LineString:
    """Create the horizontal second-order line from order 1 toward the y-axis."""

    parameters = _resolve_parameters(parameters)
    length_mm = _require_finite_positive(
        parameters.rectangle_length_mm, "rectangle_length_mm"
    )
    order1_width_mm = _require_finite_positive(
        parameters.upper_outer_slot_order1_width_mm,
        "upper_outer_slot_order1_width_mm",
    )
    center_y_mm = _require_finite_positive(
        parameters.upper_outer_slot_order2_center_y_mm,
        "upper_outer_slot_order2_center_y_mm",
    )
    line_length_mm = _require_finite_positive(
        parameters.upper_outer_slot_order2_line_length_mm,
        "upper_outer_slot_order2_line_length_mm",
    )

    x_start_mm = length_mm / 2.0 - order1_width_mm / 2.0
    x_stop_mm = x_start_mm - line_length_mm
    if x_stop_mm < 0.0:
        raise ValueError("upper order-2 line crosses beyond the y-axis")

    return LineString(
        [
            (x_start_mm, center_y_mm),
            (x_stop_mm, center_y_mm),
        ]
    )


def expand_upper_outer_slot_order2(
    centerline: LineString,
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Buffer the second-order line above and below using flat end caps."""

    parameters = _resolve_parameters(parameters)
    buffer_mm = _require_finite_positive(
        parameters.upper_outer_slot_order2_buffer_mm,
        "upper_outer_slot_order2_buffer_mm",
    )
    slot = centerline.buffer(buffer_mm, cap_style="flat")
    if not isinstance(slot, Polygon):
        raise TypeError(f"expected a Polygon slot, got {slot.geom_type}")
    return orient(slot, sign=1.0)


def build_step_2_2_outline(
    rectangle: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    LineString,
    Polygon,
    LineString,
    Polygon,
    Polygon,
    Polygon,
    UpperOuterSlotOrder2Report,
]:
    """Add order 2, merge both upper slots, and subtract them from the rectangle."""

    parameters = _resolve_parameters(parameters)
    order1_centerline, order1_slot, _, _ = build_step_2_1_outline(rectangle, parameters)
    order2_centerline = generate_upper_outer_slot_order2_centerline(parameters)
    order2_slot = expand_upper_outer_slot_order2(order2_centerline, parameters)
    if not rectangle.covers(order2_slot):
        raise ValueError("expanded upper order-2 slot lies outside the rectangle")

    combined_slot = order1_slot.union(order2_slot)
    if not isinstance(combined_slot, Polygon):
        raise TypeError(
            "combined upper slot must be one Polygon, "
            f"but Shapely returned {combined_slot.geom_type}"
        )
    combined_slot = orient(combined_slot, sign=1.0)

    outline = rectangle.difference(combined_slot)
    if not isinstance(outline, Polygon):
        raise TypeError(
            "step 2.2 must produce one Polygon, "
            f"but Shapely returned {outline.geom_type}"
        )
    outline = orient(outline, sign=1.0)

    order1_center_x_mm = (
        parameters.rectangle_length_mm / 2.0
        - parameters.upper_outer_slot_order1_width_mm / 2.0
    )
    center_y_mm = parameters.upper_outer_slot_order2_center_y_mm
    line_length_mm = parameters.upper_outer_slot_order2_line_length_mm
    buffer_mm = parameters.upper_outer_slot_order2_buffer_mm
    expected_order2_centerline = (
        (order1_center_x_mm, center_y_mm),
        (order1_center_x_mm - line_length_mm, center_y_mm),
    )
    expected_order2_bounds = (
        order1_center_x_mm - line_length_mm,
        center_y_mm - buffer_mm,
        order1_center_x_mm,
        center_y_mm + buffer_mm,
    )
    expected_order2_area_mm2 = line_length_mm * (2.0 * buffer_mm)
    overlap_area_mm2 = order1_slot.intersection(order2_slot).area
    expected_combined_area_mm2 = (
        order1_slot.area + expected_order2_area_mm2 - overlap_area_mm2
    )
    expected_outline_area_mm2 = rectangle.area - expected_combined_area_mm2
    order2_centerline_coordinates = tuple(
        (float(x), float(y)) for x, y in order2_centerline.coords
    )

    checks = {
        "order-2 centerline endpoints": np.allclose(
            order2_centerline_coordinates,
            expected_order2_centerline,
            rtol=0.0,
            atol=1e-12,
        ),
        "order-2 slot bounds": np.allclose(
            order2_slot.bounds, expected_order2_bounds, rtol=0.0, atol=1e-12
        ),
        "order-2 slot area": math.isclose(
            order2_slot.area,
            expected_order2_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "combined slot area": math.isclose(
            combined_slot.area,
            expected_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline area after both subtractions": math.isclose(
            outline.area,
            expected_outline_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline is valid": outline.is_valid,
        "outline exterior is simple": outline.exterior.is_simple,
        "outline exterior is counterclockwise": outline.exterior.is_ccw,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 2.2 validation failed: " + "; ".join(failures))

    report = UpperOuterSlotOrder2Report(
        centerline=(
            order2_centerline_coordinates[0],
            order2_centerline_coordinates[1],
        ),
        slot_bounds=tuple(float(value) for value in order2_slot.bounds),
        slot_area_mm2=float(order2_slot.area),
        combined_slot_area_mm2=float(combined_slot.area),
        outline_area_mm2=float(outline.area),
        outline_is_valid=bool(outline.is_valid),
        outline_is_simple=bool(outline.exterior.is_simple),
        outline_is_counterclockwise=bool(outline.exterior.is_ccw),
    )
    return (
        order1_centerline,
        order1_slot,
        order2_centerline,
        order2_slot,
        combined_slot,
        outline,
        report,
    )


def generate_lower_outer_slot_order1_centerline(
    parameters: AntennaOutlineParameters | None = None,
) -> LineString:
    """Create the vertical centreline for the step-2.3 lower outer slot."""

    parameters = _resolve_parameters(parameters)
    length_mm = _require_finite_positive(
        parameters.rectangle_length_mm, "rectangle_length_mm"
    )
    rectangle_width_mm = _require_finite_positive(
        parameters.rectangle_width_mm, "rectangle_width_mm"
    )
    opposite_x_mm = _require_finite_positive(
        parameters.lower_outer_slot_order1_opposite_corner_x_mm,
        "lower_outer_slot_order1_opposite_corner_x_mm",
    )
    height_mm = _require_finite_positive(
        parameters.lower_outer_slot_order1_height_mm,
        "lower_outer_slot_order1_height_mm",
    )
    center_x_mm = _require_finite_positive(
        parameters.lower_outer_slot_order1_center_x_mm,
        "lower_outer_slot_order1_center_x_mm",
    )
    right_x_mm = length_mm / 2.0
    if opposite_x_mm >= right_x_mm or height_mm > rectangle_width_mm:
        raise ValueError("lower outer slot does not fit inside the outer rectangle")

    return LineString(
        [
            (center_x_mm, 0.0),
            (center_x_mm, height_mm),
        ]
    )


def expand_lower_outer_slot_order1(
    centerline: LineString,
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Buffer the lower vertical line horizontally using flat end caps."""

    parameters = _resolve_parameters(parameters)
    buffer_mm = _require_finite_positive(
        parameters.lower_outer_slot_order1_buffer_mm,
        "lower_outer_slot_order1_buffer_mm",
    )
    slot = centerline.buffer(buffer_mm, cap_style="flat")
    if not isinstance(slot, Polygon):
        raise TypeError(f"expected a Polygon slot, got {slot.geom_type}")
    return orient(slot, sign=1.0)


def build_step_2_3_outline(
    rectangle: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    LineString,
    Polygon,
    LineString,
    Polygon,
    LineString,
    Polygon,
    Polygon | MultiPolygon,
    Polygon,
    LowerOuterSlotOrder1Report,
]:
    """Add the lower first-order slot to the completed step-2.2 outline."""

    parameters = _resolve_parameters(parameters)
    (
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        upper_combined_slot,
        _,
        _,
    ) = build_step_2_2_outline(rectangle, parameters)
    lower_centerline = generate_lower_outer_slot_order1_centerline(parameters)
    lower_slot = expand_lower_outer_slot_order1(lower_centerline, parameters)
    _reject_lower_outer_slot_cpw_interference(lower_slot)
    if not rectangle.covers(lower_slot):
        raise ValueError("expanded lower order-1 slot lies outside the rectangle")

    combined_slot = upper_combined_slot.union(lower_slot)
    outline = rectangle.difference(combined_slot)
    if not isinstance(outline, Polygon):
        raise TypeError(
            "step 2.3 must produce one Polygon, "
            f"but Shapely returned {outline.geom_type}"
        )
    outline = orient(outline, sign=1.0)

    right_x_mm = parameters.rectangle_length_mm / 2.0
    opposite_x_mm = parameters.lower_outer_slot_order1_opposite_corner_x_mm
    height_mm = parameters.lower_outer_slot_order1_height_mm
    slot_width_mm = right_x_mm - opposite_x_mm
    center_x_mm = right_x_mm - slot_width_mm / 2.0
    expected_centerline = (
        (center_x_mm, 0.0),
        (center_x_mm, height_mm),
    )
    expected_bounds = (opposite_x_mm, 0.0, right_x_mm, height_mm)
    expected_slot_area_mm2 = slot_width_mm * height_mm
    expected_combined_area_mm2 = upper_combined_slot.area + expected_slot_area_mm2
    expected_outline_area_mm2 = rectangle.area - expected_combined_area_mm2
    centerline_coordinates = tuple(
        (float(x), float(y)) for x, y in lower_centerline.coords
    )

    checks = {
        "derived lower slot width": math.isclose(
            parameters.lower_outer_slot_order1_width_mm,
            slot_width_mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "derived lower slot buffer": math.isclose(
            parameters.lower_outer_slot_order1_buffer_mm,
            slot_width_mm / 2.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "lower centerline endpoints": np.allclose(
            centerline_coordinates,
            expected_centerline,
            rtol=0.0,
            atol=1e-12,
        ),
        "lower slot bounds": np.allclose(
            lower_slot.bounds, expected_bounds, rtol=0.0, atol=1e-12
        ),
        "lower slot area": math.isclose(
            lower_slot.area,
            expected_slot_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "lower and upper slots do not overlap": math.isclose(
            upper_combined_slot.intersection(lower_slot).area,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "combined slot area": math.isclose(
            combined_slot.area,
            expected_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline area after all subtractions": math.isclose(
            outline.area,
            expected_outline_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline plus removed slots restores rectangle": rectangle.equals(
            outline.union(combined_slot)
        ),
        "outline is valid": outline.is_valid,
        "outline exterior is simple": outline.exterior.is_simple,
        "outline exterior is counterclockwise": outline.exterior.is_ccw,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 2.3 validation failed: " + "; ".join(failures))

    report = LowerOuterSlotOrder1Report(
        centerline=(centerline_coordinates[0], centerline_coordinates[1]),
        slot_bounds=tuple(float(value) for value in lower_slot.bounds),
        slot_area_mm2=float(lower_slot.area),
        combined_slot_area_mm2=float(combined_slot.area),
        outline_area_mm2=float(outline.area),
        outline_is_valid=bool(outline.is_valid),
        outline_is_simple=bool(outline.exterior.is_simple),
        outline_is_counterclockwise=bool(outline.exterior.is_ccw),
    )
    return (
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_centerline,
        lower_slot,
        combined_slot,
        outline,
        report,
    )


def generate_lower_outer_slot_order2_centerlines(
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[LineString, LineString]:
    """Create both horizontal centrelines for the step-2.4 lower outer slot."""

    parameters = _resolve_parameters(parameters)
    start_x_mm = _require_finite_positive(
        parameters.lower_outer_slot_order1_center_x_mm,
        "lower_outer_slot_order1_center_x_mm",
    )
    rectangle_width_mm = _require_finite_positive(
        parameters.rectangle_width_mm, "rectangle_width_mm"
    )

    branch_specs = (
        (
            "branch 1",
            parameters.lower_outer_slot_order2_branch1_inner_x_mm,
            parameters.lower_outer_slot_order2_branch1_lower_y_mm,
            parameters.lower_outer_slot_order2_branch1_upper_y_mm,
            parameters.lower_outer_slot_order2_branch1_center_y_mm,
        ),
        (
            "branch 2",
            parameters.lower_outer_slot_order2_branch2_inner_x_mm,
            parameters.lower_outer_slot_order2_branch2_lower_y_mm,
            parameters.lower_outer_slot_order2_branch2_upper_y_mm,
            parameters.lower_outer_slot_order2_branch2_center_y_mm,
        ),
    )
    centerlines: list[LineString] = []
    for name, inner_x_mm, lower_y_mm, upper_y_mm, center_y_mm in branch_specs:
        inner_x_mm = _require_finite_positive(inner_x_mm, f"{name} inner x")
        lower_y_mm = _require_finite_positive(lower_y_mm, f"{name} lower y")
        upper_y_mm = _require_finite_positive(upper_y_mm, f"{name} upper y")
        center_y_mm = _require_finite_positive(center_y_mm, f"{name} center y")
        if inner_x_mm >= start_x_mm:
            raise ValueError(f"lower order-2 {name} must extend toward the y-axis")
        if not lower_y_mm < upper_y_mm <= rectangle_width_mm:
            raise ValueError(f"lower order-2 {name} has invalid vertical bounds")

        centerlines.append(
            LineString(
                [
                    (start_x_mm, center_y_mm),
                    (inner_x_mm, center_y_mm),
                ]
            )
        )

    return centerlines[0], centerlines[1]


def expand_lower_outer_slot_order2(
    centerlines: Sequence[LineString],
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[Polygon, Polygon]:
    """Buffer both horizontal lower-order-2 lines with flat end caps."""

    parameters = _resolve_parameters(parameters)
    if len(centerlines) != 2:
        raise ValueError(
            f"lower order-2 requires exactly two centrelines, got {len(centerlines)}"
        )
    buffer_distances_mm = (
        _require_finite_positive(
            parameters.lower_outer_slot_order2_branch1_buffer_mm,
            "lower_outer_slot_order2_branch1_buffer_mm",
        ),
        _require_finite_positive(
            parameters.lower_outer_slot_order2_branch2_buffer_mm,
            "lower_outer_slot_order2_branch2_buffer_mm",
        ),
    )
    slots: list[Polygon] = []
    for centerline, buffer_mm in zip(centerlines, buffer_distances_mm, strict=True):
        slot = centerline.buffer(buffer_mm, cap_style="flat")
        if not isinstance(slot, Polygon):
            raise TypeError(f"expected a Polygon slot, got {slot.geom_type}")
        slots.append(orient(slot, sign=1.0))

    return slots[0], slots[1]


def build_step_2_4_outline(
    rectangle: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    LineString,
    Polygon,
    LineString,
    Polygon,
    LineString,
    Polygon,
    tuple[LineString, LineString],
    tuple[Polygon, Polygon],
    Polygon | MultiPolygon,
    Polygon,
    LowerOuterSlotOrder2Report,
]:
    """Add both lower second-order branches to the completed step-2.3 outline."""

    parameters = _resolve_parameters(parameters)
    (
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        step_2_3_combined_slot,
        _,
        _,
    ) = build_step_2_3_outline(rectangle, parameters)
    lower_order2_centerlines = generate_lower_outer_slot_order2_centerlines(parameters)
    lower_order2_slots = expand_lower_outer_slot_order2(
        lower_order2_centerlines, parameters
    )
    for index, slot in enumerate(lower_order2_slots, start=1):
        _reject_lower_outer_slot_cpw_interference(slot)
        if not rectangle.covers(slot):
            raise ValueError(
                f"expanded lower order-2 branch {index} lies outside the rectangle"
            )

    lower_combined_slot = lower_order1_slot.union(lower_order2_slots[0]).union(
        lower_order2_slots[1]
    )
    if not isinstance(lower_combined_slot, Polygon):
        raise TypeError(
            "combined lower slot must be one Polygon, "
            f"but Shapely returned {lower_combined_slot.geom_type}"
        )
    lower_combined_slot = orient(lower_combined_slot, sign=1.0)

    combined_slot = step_2_3_combined_slot.union(lower_order2_slots[0]).union(
        lower_order2_slots[1]
    )
    outline = rectangle.difference(combined_slot)
    if not isinstance(outline, Polygon):
        raise TypeError(
            "step 2.4 must produce one Polygon, "
            f"but Shapely returned {outline.geom_type}"
        )
    outline = orient(outline, sign=1.0)

    start_x_mm = parameters.lower_outer_slot_order1_center_x_mm
    branch1_center_y_mm = parameters.lower_outer_slot_order2_branch1_center_y_mm
    branch2_center_y_mm = parameters.lower_outer_slot_order2_branch2_center_y_mm
    branch1_inner_x_mm = parameters.lower_outer_slot_order2_branch1_inner_x_mm
    branch2_inner_x_mm = parameters.lower_outer_slot_order2_branch2_inner_x_mm
    branch1_buffer_mm = parameters.lower_outer_slot_order2_branch1_buffer_mm
    branch2_buffer_mm = parameters.lower_outer_slot_order2_branch2_buffer_mm
    branch1_height_mm = 2.0 * branch1_buffer_mm
    branch2_height_mm = 2.0 * branch2_buffer_mm
    expected_centerlines = (
        (
            (start_x_mm, branch1_center_y_mm),
            (branch1_inner_x_mm, branch1_center_y_mm),
        ),
        (
            (start_x_mm, branch2_center_y_mm),
            (branch2_inner_x_mm, branch2_center_y_mm),
        ),
    )
    expected_bounds = (
        (
            branch1_inner_x_mm,
            branch1_center_y_mm - branch1_buffer_mm,
            start_x_mm,
            branch1_center_y_mm + branch1_buffer_mm,
        ),
        (
            branch2_inner_x_mm,
            branch2_center_y_mm - branch2_buffer_mm,
            start_x_mm,
            branch2_center_y_mm + branch2_buffer_mm,
        ),
    )
    expected_slot_areas_mm2 = (
        parameters.lower_outer_slot_order2_branch1_line_length_mm * branch1_height_mm,
        parameters.lower_outer_slot_order2_branch2_line_length_mm * branch2_height_mm,
    )
    branch1_new_width_mm = (
        parameters.lower_outer_slot_order1_opposite_corner_x_mm - branch1_inner_x_mm
    )
    branch2_new_width_mm = branch1_inner_x_mm - branch2_inner_x_mm
    expected_lower_combined_area_mm2 = (
        lower_order1_slot.area
        + branch1_new_width_mm * branch1_height_mm
        + branch2_new_width_mm * branch2_height_mm
    )
    expected_combined_area_mm2 = (
        step_2_3_combined_slot.area
        + branch1_new_width_mm * branch1_height_mm
        + branch2_new_width_mm * branch2_height_mm
    )
    expected_outline_area_mm2 = rectangle.area - expected_combined_area_mm2
    centerline_coordinates = tuple(
        tuple((float(x), float(y)) for x, y in centerline.coords)
        for centerline in lower_order2_centerlines
    )

    checks = {
        "both lower order-2 lines start on the order-1 centreline": all(
            math.isclose(coordinates[0][0], start_x_mm, rel_tol=0.0, abs_tol=1e-12)
            for coordinates in centerline_coordinates
        ),
        "lower order-2 centerline endpoints": all(
            np.allclose(actual, expected, rtol=0.0, atol=1e-12)
            for actual, expected in zip(
                centerline_coordinates, expected_centerlines, strict=True
            )
        ),
        "lower order-2 slot bounds": all(
            np.allclose(slot.bounds, bounds, rtol=0.0, atol=1e-12)
            for slot, bounds in zip(lower_order2_slots, expected_bounds, strict=True)
        ),
        "lower order-2 slot areas": all(
            math.isclose(slot.area, area, rel_tol=0.0, abs_tol=1e-9)
            for slot, area in zip(
                lower_order2_slots, expected_slot_areas_mm2, strict=True
            )
        ),
        "second branch is vertically nested inside first branch": (
            parameters.lower_outer_slot_order2_branch2_lower_y_mm
            >= parameters.lower_outer_slot_order2_branch1_lower_y_mm
            and parameters.lower_outer_slot_order2_branch2_upper_y_mm
            <= parameters.lower_outer_slot_order2_branch1_upper_y_mm
        ),
        "combined lower slot area": math.isclose(
            lower_combined_slot.area,
            expected_lower_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "combined slot area": math.isclose(
            combined_slot.area,
            expected_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline area after all subtractions": math.isclose(
            outline.area,
            expected_outline_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline plus removed slots restores rectangle": rectangle.equals(
            outline.union(combined_slot)
        ),
        "outline is valid": outline.is_valid,
        "outline exterior is simple": outline.exterior.is_simple,
        "outline exterior is counterclockwise": outline.exterior.is_ccw,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 2.4 validation failed: " + "; ".join(failures))

    report = LowerOuterSlotOrder2Report(
        centerlines=(
            (centerline_coordinates[0][0], centerline_coordinates[0][1]),
            (centerline_coordinates[1][0], centerline_coordinates[1][1]),
        ),
        slot_bounds=(
            tuple(float(value) for value in lower_order2_slots[0].bounds),
            tuple(float(value) for value in lower_order2_slots[1].bounds),
        ),
        slot_areas_mm2=(
            float(lower_order2_slots[0].area),
            float(lower_order2_slots[1].area),
        ),
        lower_combined_slot_area_mm2=float(lower_combined_slot.area),
        combined_slot_area_mm2=float(combined_slot.area),
        outline_area_mm2=float(outline.area),
        outline_is_valid=bool(outline.is_valid),
        outline_is_simple=bool(outline.exterior.is_simple),
        outline_is_counterclockwise=bool(outline.exterior.is_ccw),
    )
    return (
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        lower_order2_centerlines,
        lower_order2_slots,
        combined_slot,
        outline,
        report,
    )


def mirror_about_y_axis(
    geometry: BaseGeometry,
    parameters: AntennaOutlineParameters | None = None,
) -> BaseGeometry:
    """Return a reflection of a 2D Shapely geometry across the y-axis."""

    parameters = _resolve_parameters(parameters)
    symmetry_axis_x_mm = float(parameters.outer_slot_symmetry_axis_x_mm)
    if not math.isfinite(symmetry_axis_x_mm):
        raise ValueError("outer_slot_symmetry_axis_x_mm must be finite")
    if geometry.is_empty:
        raise ValueError("cannot mirror an empty geometry")

    return scale(
        geometry,
        xfact=-1.0,
        yfact=1.0,
        origin=(symmetry_axis_x_mm, 0.0),
    )


def build_step_2_5_outline(
    rectangle: Polygon,
    right_combined_slot: Polygon | MultiPolygon | None = None,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    Polygon | MultiPolygon,
    Polygon | MultiPolygon,
    Polygon | MultiPolygon,
    Polygon,
    SymmetricOuterSlotsReport,
]:
    """Mirror every right-side outer slot and subtract the symmetric pair."""

    parameters = _resolve_parameters(parameters)
    if right_combined_slot is None:
        right_combined_slot = build_step_2_4_outline(rectangle, parameters)[8]
    if not rectangle.covers(right_combined_slot):
        raise ValueError("right-side outer slots lie outside the rectangle")

    symmetry_axis_x_mm = float(parameters.outer_slot_symmetry_axis_x_mm)
    if right_combined_slot.bounds[0] < symmetry_axis_x_mm:
        raise ValueError("right-side outer slots cross the y-axis")

    left_combined_slot = mirror_about_y_axis(right_combined_slot, parameters)
    if not isinstance(left_combined_slot, (Polygon, MultiPolygon)):
        raise TypeError(
            "mirrored outer slot must be Polygon or MultiPolygon, "
            f"but Shapely returned {left_combined_slot.geom_type}"
        )
    if not rectangle.covers(left_combined_slot):
        raise ValueError("mirrored left-side outer slots lie outside the rectangle")

    symmetric_combined_slot = right_combined_slot.union(left_combined_slot)
    if not isinstance(symmetric_combined_slot, (Polygon, MultiPolygon)):
        raise TypeError(
            "symmetric outer slots must be Polygon or MultiPolygon, "
            f"but Shapely returned {symmetric_combined_slot.geom_type}"
        )

    outline = rectangle.difference(symmetric_combined_slot)
    if not isinstance(outline, Polygon):
        raise TypeError(
            "step 2.5 must produce one Polygon, "
            f"but Shapely returned {outline.geom_type}"
        )
    outline = orient(outline, sign=1.0)

    right_min_x, right_min_y, right_max_x, right_max_y = right_combined_slot.bounds
    expected_left_bounds = (
        2.0 * symmetry_axis_x_mm - right_max_x,
        right_min_y,
        2.0 * symmetry_axis_x_mm - right_min_x,
        right_max_y,
    )
    expected_combined_area_mm2 = 2.0 * right_combined_slot.area
    expected_outline_area_mm2 = rectangle.area - expected_combined_area_mm2
    mirrored_symmetric_slot = mirror_about_y_axis(symmetric_combined_slot, parameters)
    mirrored_outline = mirror_about_y_axis(outline, parameters)
    combined_slot_is_symmetric = symmetric_combined_slot.equals(mirrored_symmetric_slot)
    outline_is_symmetric = outline.equals(mirrored_outline)

    checks = {
        "mirrored slot bounds": np.allclose(
            left_combined_slot.bounds,
            expected_left_bounds,
            rtol=0.0,
            atol=1e-12,
        ),
        "mirrored slot area": math.isclose(
            left_combined_slot.area,
            right_combined_slot.area,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "right and left slots do not overlap": math.isclose(
            right_combined_slot.intersection(left_combined_slot).area,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "combined symmetric slot area": math.isclose(
            symmetric_combined_slot.area,
            expected_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "outline area after mirrored subtraction": math.isclose(
            outline.area,
            expected_outline_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "combined slot is y-axis symmetric": combined_slot_is_symmetric,
        "outline is y-axis symmetric": outline_is_symmetric,
        "outline plus removed slots restores rectangle": rectangle.equals(
            outline.union(symmetric_combined_slot)
        ),
        "outline is valid": outline.is_valid,
        "outline exterior is simple": outline.exterior.is_simple,
        "outline exterior is counterclockwise": outline.exterior.is_ccw,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 2.5 validation failed: " + "; ".join(failures))

    report = SymmetricOuterSlotsReport(
        symmetry_axis_x_mm=symmetry_axis_x_mm,
        right_slot_bounds=tuple(float(value) for value in right_combined_slot.bounds),
        left_slot_bounds=tuple(float(value) for value in left_combined_slot.bounds),
        right_slot_area_mm2=float(right_combined_slot.area),
        left_slot_area_mm2=float(left_combined_slot.area),
        combined_slot_area_mm2=float(symmetric_combined_slot.area),
        outline_area_mm2=float(outline.area),
        combined_slot_is_symmetric=bool(combined_slot_is_symmetric),
        outline_is_symmetric=bool(outline_is_symmetric),
        outline_is_valid=bool(outline.is_valid),
        outline_is_simple=bool(outline.exterior.is_simple),
        outline_is_counterclockwise=bool(outline.exterior.is_ccw),
    )
    return (
        right_combined_slot,
        left_combined_slot,
        symmetric_combined_slot,
        outline,
        report,
    )


def generate_inner_slot_order1_centerline(
    parameters: AntennaOutlineParameters | None = None,
) -> LineString:
    """Create the horizontal L-system centreline for inner-slot step 3.1."""

    parameters = _resolve_parameters(parameters)
    left_x_mm = float(parameters.inner_slot_order1_left_x_mm)
    right_x_mm = float(parameters.inner_slot_order1_right_x_mm)
    lower_y_mm = float(parameters.inner_slot_order1_lower_y_mm)
    upper_y_mm = float(parameters.inner_slot_order1_upper_y_mm)
    center_y_mm = float(parameters.inner_slot_order1_center_y_mm)
    values = (left_x_mm, right_x_mm, lower_y_mm, upper_y_mm, center_y_mm)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inner order-1 slot constants must be finite")
    if right_x_mm <= left_x_mm:
        raise ValueError("inner order-1 slot must have positive horizontal length")
    if upper_y_mm <= lower_y_mm:
        raise ValueError("inner order-1 slot must have positive vertical width")

    return LineString(
        [
            (left_x_mm, center_y_mm),
            (right_x_mm, center_y_mm),
        ]
    )


def expand_inner_slot_order1(
    centerline: LineString,
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Buffer the inner order-1 centreline without modifying the Patch."""

    parameters = _resolve_parameters(parameters)
    buffer_mm = _require_finite_positive(
        parameters.inner_slot_order1_buffer_mm,
        "inner_slot_order1_buffer_mm",
    )
    slot = centerline.buffer(buffer_mm, cap_style="flat")
    if not isinstance(slot, Polygon):
        raise TypeError(f"expected a Polygon slot, got {slot.geom_type}")
    return orient(slot, sign=1.0)


def build_step_3_1_geometry(
    patch: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[LineString, Polygon, Polygon, InnerSlotOrder1Report]:
    """Build the first inner slot as separate geometry; do not subtract it."""

    parameters = _resolve_parameters(parameters)
    centerline = generate_inner_slot_order1_centerline(parameters)
    slot = expand_inner_slot_order1(centerline, parameters)
    if not patch.covers(slot):
        raise ValueError("inner order-1 slot does not lie completely inside the Patch")

    result_patch = patch
    centerline_coordinates = tuple((float(x), float(y)) for x, y in centerline.coords)
    expected_centerline = (
        (
            parameters.inner_slot_order1_left_x_mm,
            parameters.inner_slot_order1_center_y_mm,
        ),
        (
            parameters.inner_slot_order1_right_x_mm,
            parameters.inner_slot_order1_center_y_mm,
        ),
    )
    expected_bounds = (
        parameters.inner_slot_order1_left_x_mm,
        parameters.inner_slot_order1_lower_y_mm,
        parameters.inner_slot_order1_right_x_mm,
        parameters.inner_slot_order1_upper_y_mm,
    )
    expected_slot_area_mm2 = (
        parameters.inner_slot_order1_line_length_mm
        * 2.0
        * parameters.inner_slot_order1_buffer_mm
    )
    mirrored_slot = mirror_about_y_axis(slot, parameters)
    slot_is_y_axis_symmetric = slot.equals(mirrored_slot)
    patch_is_unchanged = result_patch.equals_exact(patch, tolerance=0.0)

    checks = {
        "inner order-1 centerline endpoints": np.allclose(
            centerline_coordinates,
            expected_centerline,
            rtol=0.0,
            atol=1e-12,
        ),
        "inner order-1 slot bounds": np.allclose(
            slot.bounds, expected_bounds, rtol=0.0, atol=1e-12
        ),
        "inner order-1 slot area": math.isclose(
            slot.area,
            expected_slot_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "inner order-1 slot is valid": slot.is_valid,
        "inner order-1 slot is inside Patch": slot.within(patch),
        "inner order-1 slot is not y-axis mirrored": not slot_is_y_axis_symmetric,
        "Patch object is returned without subtraction": result_patch is patch,
        "Patch geometry is unchanged": patch_is_unchanged,
        "Patch area is unchanged": math.isclose(
            result_patch.area,
            patch.area,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 3.1 validation failed: " + "; ".join(failures))

    report = InnerSlotOrder1Report(
        centerline=(centerline_coordinates[0], centerline_coordinates[1]),
        slot_bounds=tuple(float(value) for value in slot.bounds),
        slot_area_mm2=float(slot.area),
        patch_area_before_mm2=float(patch.area),
        patch_area_after_mm2=float(result_patch.area),
        patch_is_unchanged=bool(patch_is_unchanged),
        slot_was_subtracted=False,
        slot_is_y_axis_symmetric=bool(slot_is_y_axis_symmetric),
    )
    return centerline, slot, result_patch, report


def generate_inner_slot_order2_centerline(
    parameters: AntennaOutlineParameters | None = None,
) -> LineString:
    """Create the single inner-slot order-2 branch growing along Y+."""

    parameters = _resolve_parameters(parameters)
    center_x_mm = float(parameters.inner_slot_order2_center_x_mm)
    start_y_mm = float(parameters.inner_slot_order2_start_y_mm)
    end_y_mm = float(parameters.inner_slot_order2_cap_y_mm)
    values = (center_x_mm, start_y_mm, end_y_mm)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inner order-2 slot constants must be finite")
    if end_y_mm <= start_y_mm:
        raise ValueError("inner order-2 branch must grow along Y+")

    return LineString(
        [
            (center_x_mm, start_y_mm),
            (center_x_mm, end_y_mm),
        ]
    )


def expand_inner_slot_order2(
    centerline: LineString,
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Buffer the Y+ order-2 branch equally in the X direction."""

    parameters = _resolve_parameters(parameters)
    buffer_mm = _require_finite_positive(
        parameters.inner_slot_order2_buffer_mm,
        "inner_slot_order2_buffer_mm",
    )
    slot = centerline.buffer(buffer_mm, cap_style="flat")
    if not isinstance(slot, Polygon):
        raise TypeError(f"expected a Polygon slot, got {slot.geom_type}")
    return orient(slot, sign=1.0)


def generate_inner_slot_order2_reservations(
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    InnerSlotBranchReservation,
    InnerSlotBranchReservation,
    InnerSlotBranchReservation,
]:
    """Return three zero-size branch specifications without creating geometry."""

    parameters = _resolve_parameters(parameters)
    anchor_y_mm = float(parameters.inner_slot_order2_reserved_anchor_y_mm)
    reservations = (
        InnerSlotBranchReservation(
            name="reserved_up_1",
            parent_name="inner_slot_order1",
            anchor_t=float(parameters.inner_slot_order2_reserved_up_anchor_t),
            anchor=(
                float(parameters.inner_slot_order2_reserved_up_anchor_x_mm),
                anchor_y_mm,
            ),
            growth_direction=(0.0, 1.0),
            geometry_type="slot",
            enabled=bool(parameters.inner_slot_order2_reserved_up_enabled),
            length_mm=float(parameters.inner_slot_order2_reserved_up_length_mm),
            width_mm=float(parameters.inner_slot_order2_reserved_up_width_mm),
        ),
        InnerSlotBranchReservation(
            name="reserved_down_1",
            parent_name="inner_slot_order1",
            anchor_t=float(parameters.inner_slot_order2_reserved_down1_anchor_t),
            anchor=(
                float(parameters.inner_slot_order2_reserved_down1_anchor_x_mm),
                anchor_y_mm,
            ),
            growth_direction=(0.0, -1.0),
            geometry_type="slot",
            enabled=bool(parameters.inner_slot_order2_reserved_down1_enabled),
            length_mm=float(parameters.inner_slot_order2_reserved_down1_length_mm),
            width_mm=float(parameters.inner_slot_order2_reserved_down1_width_mm),
        ),
        InnerSlotBranchReservation(
            name="reserved_down_2",
            parent_name="inner_slot_order1",
            anchor_t=float(parameters.inner_slot_order2_reserved_down2_anchor_t),
            anchor=(
                float(parameters.inner_slot_order2_reserved_down2_anchor_x_mm),
                anchor_y_mm,
            ),
            growth_direction=(0.0, -1.0),
            geometry_type="slot",
            enabled=bool(parameters.inner_slot_order2_reserved_down2_enabled),
            length_mm=float(parameters.inner_slot_order2_reserved_down2_length_mm),
            width_mm=float(parameters.inner_slot_order2_reserved_down2_width_mm),
        ),
    )

    names = [reservation.name for reservation in reservations]
    if len(set(names)) != len(names):
        raise ValueError("reserved inner-slot branch names must be unique")
    for reservation in reservations:
        values = (
            reservation.anchor_t,
            *reservation.anchor,
            *reservation.growth_direction,
            reservation.length_mm,
            reservation.width_mm,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{reservation.name} contains a non-finite parameter")
        if reservation.length_mm < 0.0 or reservation.width_mm < 0.0:
            raise ValueError(f"{reservation.name} dimensions cannot be negative")
        if not 0.0 <= reservation.anchor_t <= 1.0:
            raise ValueError(f"{reservation.name} anchor_t must be in [0, 1]")
        if reservation.parent_name != "inner_slot_order1":
            raise ValueError(f"{reservation.name} has an unsupported parent")
        if reservation.geometry_type != "slot":
            raise ValueError(f"{reservation.name} has an unsupported geometry type")
        if reservation.enabled and (
            reservation.length_mm <= 0.0 or reservation.width_mm <= 0.0
        ):
            raise ValueError(
                f"{reservation.name} requires positive length and width when enabled"
            )
        if not (
            parameters.inner_slot_order1_left_x_mm
            <= reservation.anchor[0]
            <= parameters.inner_slot_order1_right_x_mm
        ):
            raise ValueError(f"{reservation.name} anchor lies outside order 1")
        if not math.isclose(
            reservation.anchor[1],
            parameters.inner_slot_order1_center_y_mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{reservation.name} anchor is not on order-1 centreline")
        if reservation.growth_direction not in ((0.0, 1.0), (0.0, -1.0)):
            raise ValueError(f"{reservation.name} must grow along Y+ or Y-")

    return reservations


def build_active_reserved_inner_slot_branches(
    reservations: Sequence[InnerSlotBranchReservation],
) -> tuple[tuple[LineString, ...], tuple[Polygon, ...]]:
    """Build every active reserved branch as a flat-capped buffered line."""

    centerlines: list[LineString] = []
    polygons: list[Polygon] = []
    for reservation in reservations:
        if not reservation.is_active:
            continue
        start_x, start_y = reservation.anchor
        direction_x, direction_y = reservation.growth_direction
        centerline = LineString(
            [
                reservation.anchor,
                (
                    start_x + direction_x * reservation.length_mm,
                    start_y + direction_y * reservation.length_mm,
                ),
            ]
        )
        branch = centerline.buffer(reservation.width_mm / 2.0, cap_style="flat")
        if not isinstance(branch, Polygon):
            raise TypeError(
                f"active reserved branch {reservation.name} is not a Polygon"
            )
        centerlines.append(centerline)
        polygons.append(orient(branch, sign=1.0))
    return tuple(centerlines), tuple(polygons)


def build_sma_solder_keepout() -> Polygon:
    """Return the fixed SMA solder keepout including its Y-axis mirror."""

    return box(
        -SMA_SOLDER_KEEPOUT_RIGHT_X_FIXED_MM,
        0.0,
        SMA_SOLDER_KEEPOUT_RIGHT_X_FIXED_MM,
        SMA_SOLDER_KEEPOUT_UPPER_Y_FIXED_MM,
    )


def build_step_3_2_geometry(
    patch: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    LineString,
    Polygon,
    LineString,
    Polygon,
    Polygon,
    Polygon,
    InnerSlotOrder2Report,
]:
    """Add one Y+ order-2 branch to the separate inner-slot geometry."""

    parameters = _resolve_parameters(parameters)
    (
        order1_centerline,
        order1_slot,
        _,
        _,
    ) = build_step_3_1_geometry(patch, parameters)
    order2_centerline = generate_inner_slot_order2_centerline(parameters)
    order2_slot = expand_inner_slot_order2(order2_centerline, parameters)
    if not patch.covers(order2_slot):
        raise ValueError("inner order-2 slot does not lie completely inside the Patch")

    base_combined_inner_slot = order1_slot.union(order2_slot)
    if not isinstance(base_combined_inner_slot, Polygon):
        raise TypeError(
            "combined inner slot must be one Polygon, "
            f"but Shapely returned {base_combined_inner_slot.geom_type}"
        )
    base_combined_inner_slot = orient(base_combined_inner_slot, sign=1.0)
    result_patch = patch
    reserved_branches = generate_inner_slot_order2_reservations(parameters)
    _, active_reserved_polygons = build_active_reserved_inner_slot_branches(
        reserved_branches
    )
    reserved_active_count = sum(
        reservation.is_active for reservation in reserved_branches
    )
    substrate = Polygon(generate_rectangle(parameters))
    sma_solder_keepout = build_sma_solder_keepout()
    combined_inner_slot: BaseGeometry = base_combined_inner_slot
    for reservation, branch in zip(
        (item for item in reserved_branches if item.is_active),
        active_reserved_polygons,
        strict=True,
    ):
        if not substrate.covers(branch):
            raise ValueError(
                f"active reserved branch {reservation.name} leaves the substrate"
            )
        if (
            reservation.geometry_type == "slot"
            and branch.intersection(sma_solder_keepout).area
            > GEOMETRY_TOLERANCE_MM2
        ):
            raise ValueError(
                f"active slot branch {reservation.name} covers the SMA solder keepout"
            )
        combined_inner_slot = combined_inner_slot.union(branch)
    if not isinstance(combined_inner_slot, Polygon):
        raise TypeError(
            "active reserved branches must remain connected to the inner slot, "
            f"but Shapely returned {combined_inner_slot.geom_type}"
        )
    combined_inner_slot = orient(combined_inner_slot, sign=1.0)

    centerline_coordinates = tuple(
        (float(x), float(y)) for x, y in order2_centerline.coords
    )
    expected_centerline = (
        (
            parameters.inner_slot_order2_center_x_mm,
            parameters.inner_slot_order2_start_y_mm,
        ),
        (
            parameters.inner_slot_order2_center_x_mm,
            parameters.inner_slot_order2_cap_y_mm,
        ),
    )
    expected_bounds = (
        parameters.inner_slot_order2_cap_left_x_mm,
        parameters.inner_slot_order2_start_y_mm,
        parameters.inner_slot_order2_cap_right_x_mm,
        parameters.inner_slot_order2_cap_y_mm,
    )
    expected_slot_area_mm2 = (
        parameters.inner_slot_order2_line_length_mm
        * 2.0
        * parameters.inner_slot_order2_buffer_mm
    )
    overlap_height_mm = max(
        0.0,
        min(
            parameters.inner_slot_order2_cap_y_mm,
            parameters.inner_slot_order1_upper_y_mm,
        )
        - max(
            parameters.inner_slot_order2_start_y_mm,
            parameters.inner_slot_order1_lower_y_mm,
        ),
    )
    expected_overlap_area_mm2 = (
        2.0 * parameters.inner_slot_order2_buffer_mm * overlap_height_mm
    )
    expected_base_combined_area_mm2 = (
        order1_slot.area + expected_slot_area_mm2 - expected_overlap_area_mm2
    )
    combined_slot_is_y_axis_symmetric = combined_inner_slot.equals(
        mirror_about_y_axis(combined_inner_slot, parameters)
    )
    patch_is_unchanged = result_patch.equals_exact(patch, tolerance=0.0)

    checks = {
        "inner order-2 growth direction is Y+": (
            math.isclose(
                centerline_coordinates[0][0],
                centerline_coordinates[1][0],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and centerline_coordinates[1][1] > centerline_coordinates[0][1]
        ),
        "inner order-2 centerline endpoints": np.allclose(
            centerline_coordinates,
            expected_centerline,
            rtol=0.0,
            atol=1e-12,
        ),
        "inner order-2 slot bounds": np.allclose(
            order2_slot.bounds, expected_bounds, rtol=0.0, atol=1e-12
        ),
        "inner order-2 slot area": math.isclose(
            order2_slot.area,
            expected_slot_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "order-1 and order-2 overlap area": math.isclose(
            order1_slot.intersection(order2_slot).area,
            expected_overlap_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "base combined inner-slot area": math.isclose(
            base_combined_inner_slot.area,
            expected_base_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "combined inner slot is valid": combined_inner_slot.is_valid,
        "combined inner slot is inside substrate": substrate.covers(
            combined_inner_slot
        ),
        "combined inner slot intersects Patch": combined_inner_slot.intersects(patch),
        "combined inner slot is not y-axis mirrored": (
            not combined_slot_is_y_axis_symmetric
        ),
        "three inner order-2 branches are reserved": len(reserved_branches) == 3,
        "reserved active count matches generated polygons": (
            reserved_active_count == len(active_reserved_polygons)
        ),
        "reserved branch directions are one Y+ and two Y-": (
            sum(
                reservation.growth_direction == (0.0, 1.0)
                for reservation in reserved_branches
            )
            == 1
            and sum(
                reservation.growth_direction == (0.0, -1.0)
                for reservation in reserved_branches
            )
            == 2
        ),
        "enabled reserved branches have positive dimensions": all(
            not reservation.is_active
            or (reservation.length_mm > 0.0 and reservation.width_mm > 0.0)
            for reservation in reserved_branches
        ),
        "Patch object is returned without subtraction": result_patch is patch,
        "Patch geometry is unchanged": patch_is_unchanged,
        "Patch area is unchanged": math.isclose(
            result_patch.area,
            patch.area,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 3.2 validation failed: " + "; ".join(failures))

    report = InnerSlotOrder2Report(
        growth_direction=(0.0, 1.0),
        centerline=(centerline_coordinates[0], centerline_coordinates[1]),
        slot_bounds=tuple(float(value) for value in order2_slot.bounds),
        slot_area_mm2=float(order2_slot.area),
        overlap_with_order1_mm2=float(order1_slot.intersection(order2_slot).area),
        combined_inner_slot_area_mm2=float(combined_inner_slot.area),
        patch_area_before_mm2=float(patch.area),
        patch_area_after_mm2=float(result_patch.area),
        patch_is_unchanged=bool(patch_is_unchanged),
        slot_was_subtracted=False,
        combined_slot_is_y_axis_symmetric=bool(combined_slot_is_y_axis_symmetric),
        reserved_branches=reserved_branches,
        reserved_active_count=int(reserved_active_count),
    )
    return (
        order1_centerline,
        order1_slot,
        order2_centerline,
        order2_slot,
        combined_inner_slot,
        result_patch,
        report,
    )


def generate_cpw_guide_points(
    parameters: AntennaOutlineParameters | None = None,
) -> list[Point2D]:
    """Return closed P1-P7 CPW-guide coordinates for the given parameters."""

    parameters = _resolve_parameters(parameters)
    p3_p4_x_mm = float(parameters.cpw_guide_p3_p4_x_mm)
    p4_y_mm = float(parameters.cpw_guide_p4_y_mm)
    p5_p6_x_mm = float(parameters.cpw_guide_p5_p6_x_mm)
    p1_x_mm = float(parameters.cpw_guide_p1_x_mm)
    p1_y_mm = float(parameters.cpw_guide_p1_y_mm)
    p2_x_mm = float(parameters.cpw_guide_p2_x_mm)
    p2_y_mm = float(parameters.cpw_guide_p2_y_mm)
    p3_y_mm = float(parameters.cpw_guide_p3_y_mm)
    p7_x_mm = float(parameters.cpw_guide_p7_x_mm)
    values = (
        p1_x_mm,
        p1_y_mm,
        p2_x_mm,
        p2_y_mm,
        p3_y_mm,
        p7_x_mm,
        p3_p4_x_mm,
        p4_y_mm,
        p5_p6_x_mm,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("CPW guide parameters must be finite")
    if p3_p4_x_mm <= max(p2_x_mm, p5_p6_x_mm):
        raise ValueError("CPW P3/P4 x must exceed both P2 x and P5/P6 x")
    if p5_p6_x_mm <= p1_x_mm:
        raise ValueError("CPW P5/P6 x must lie to the right of P1")
    if p4_y_mm <= p3_y_mm:
        raise ValueError("CPW P4 y must lie above P3 y")

    y1_mm = p4_y_mm + (p3_p4_x_mm - p5_p6_x_mm)
    y2_mm = float(parameters.cpw_guide_y2_linked_mm)
    if not p4_y_mm < y1_mm < y2_mm:
        raise ValueError("CPW guide requires P4_y < y1 < linked order-1 top y2")

    return [
        (p1_x_mm, p1_y_mm),
        (p2_x_mm, p2_y_mm),
        (p3_p4_x_mm, p3_y_mm),
        (p3_p4_x_mm, p4_y_mm),
        (p5_p6_x_mm, y1_mm),
        (p5_p6_x_mm, y2_mm),
        (p7_x_mm, y2_mm),
        (p1_x_mm, p1_y_mm),
    ]


def validate_cpw_guide(
    points: Sequence[Point2D],
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Validate closure, linked anchors, winding, and the CPW-guide Polygon."""

    parameters = _resolve_parameters(parameters)
    coordinates = np.asarray(points, dtype=float)
    if coordinates.shape != (8, 2):
        raise ValueError(
            f"CPW guide requires seven anchors plus closure, got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("CPW guide contains a non-finite coordinate")
    if not np.array_equal(coordinates[0], coordinates[-1]):
        raise ValueError("CPW-guide point sequence is not closed")

    expected_coordinates = np.asarray(
        generate_cpw_guide_points(parameters), dtype=float
    )
    polygon = Polygon(coordinates)
    checks = {
        "CPW coordinates match fixed and linked definitions": np.allclose(
            coordinates, expected_coordinates, rtol=0.0, atol=1e-12
        ),
        "CPW Polygon is valid": polygon.is_valid,
        "CPW exterior is simple": polygon.exterior.is_simple,
        "CPW exterior is counterclockwise": polygon.exterior.is_ccw,
        "CPW Polygon has positive area": polygon.area > 0.0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("CPW-guide validation failed: " + "; ".join(failures))
    return orient(polygon, sign=1.0)


def build_step_3_3_geometry(
    patch: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    LineString,
    Polygon,
    LineString,
    Polygon,
    Polygon,
    Polygon,
    Polygon,
    Polygon,
    CpwGuideReport,
]:
    """Add the parameterized CPW guide without mirroring or altering the Patch."""

    parameters = _resolve_parameters(parameters)
    (
        order1_centerline,
        order1_slot,
        order2_centerline,
        order2_slot,
        combined_inner_slot,
        _,
        _,
    ) = build_step_3_2_geometry(patch, parameters)
    cpw_guide_points = generate_cpw_guide_points(parameters)
    cpw_guide = validate_cpw_guide(cpw_guide_points, parameters)
    if not patch.covers(cpw_guide):
        raise ValueError("CPW guide does not lie completely inside the Patch")

    combined_step3_geometry = combined_inner_slot.union(cpw_guide)
    if not isinstance(combined_step3_geometry, Polygon):
        raise TypeError(
            "combined step-3 guide and slot geometry must be one Polygon, "
            f"but Shapely returned {combined_step3_geometry.geom_type}"
        )
    combined_step3_geometry = orient(combined_step3_geometry, sign=1.0)
    result_patch = patch

    p3_p4_x_mm = float(parameters.cpw_guide_p3_p4_x_mm)
    p4_y_mm = float(parameters.cpw_guide_p4_y_mm)
    p5_p6_x_mm = float(parameters.cpw_guide_p5_p6_x_mm)
    y1_mm = p4_y_mm + (p3_p4_x_mm - p5_p6_x_mm)
    y2_mm = float(parameters.inner_slot_order1_upper_y_mm)
    anchor_points = tuple((float(x), float(y)) for x, y in cpw_guide_points[:-1])
    expected_anchor_points = (
        (parameters.cpw_guide_p1_x_mm, parameters.cpw_guide_p1_y_mm),
        (parameters.cpw_guide_p2_x_mm, parameters.cpw_guide_p2_y_mm),
        (p3_p4_x_mm, parameters.cpw_guide_p3_y_mm),
        (p3_p4_x_mm, p4_y_mm),
        (p5_p6_x_mm, y1_mm),
        (p5_p6_x_mm, y2_mm),
        (parameters.cpw_guide_p7_x_mm, y2_mm),
    )
    overlap_height_mm = max(
        0.0,
        min(y2_mm, parameters.inner_slot_order1_upper_y_mm)
        - max(y1_mm, parameters.inner_slot_order1_lower_y_mm),
    )
    overlap_width_mm = max(
        0.0,
        min(p5_p6_x_mm, parameters.inner_slot_order1_right_x_mm)
        - max(
            parameters.cpw_guide_p7_x_mm,
            parameters.inner_slot_order1_left_x_mm,
        ),
    )
    expected_overlap_area_mm2 = overlap_width_mm * overlap_height_mm
    expected_combined_area_mm2 = (
        combined_inner_slot.area + cpw_guide.area - expected_overlap_area_mm2
    )
    guide_is_y_axis_symmetric = cpw_guide.equals(
        mirror_about_y_axis(cpw_guide, parameters)
    )
    patch_is_unchanged = result_patch.equals_exact(patch, tolerance=0.0)

    checks = {
        "CPW P1 and P2 match parameter definitions": (
            anchor_points[0]
            == (parameters.cpw_guide_p1_x_mm, parameters.cpw_guide_p1_y_mm)
            and anchor_points[1]
            == (parameters.cpw_guide_p2_x_mm, parameters.cpw_guide_p2_y_mm)
        ),
        "CPW P3/P4 x values are linked": math.isclose(
            anchor_points[2][0],
            anchor_points[3][0],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "CPW P5/P6 x values are linked": math.isclose(
            anchor_points[4][0],
            anchor_points[5][0],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "CPW y1 follows linked parameter formula": math.isclose(
            anchor_points[4][1],
            p4_y_mm + (p3_p4_x_mm - p5_p6_x_mm),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "CPW-guide y2 follows inner-slot order-1 top": (
            math.isclose(
                anchor_points[5][1],
                parameters.inner_slot_order1_upper_y_mm,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                anchor_points[6][1],
                parameters.inner_slot_order1_upper_y_mm,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "CPW anchors match P1-P7 definitions": np.allclose(
            anchor_points, expected_anchor_points, rtol=0.0, atol=1e-12
        ),
        "CPW guide is inside Patch": cpw_guide.within(patch),
        "CPW guide and inner-slot overlap area": math.isclose(
            cpw_guide.intersection(combined_inner_slot).area,
            expected_overlap_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "combined step-3 geometry area": math.isclose(
            combined_step3_geometry.area,
            expected_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "CPW guide is not y-axis mirrored": not guide_is_y_axis_symmetric,
        "Patch object is returned without subtraction": result_patch is patch,
        "Patch geometry is unchanged": patch_is_unchanged,
        "Patch area is unchanged": math.isclose(
            result_patch.area,
            patch.area,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 3.3 validation failed: " + "; ".join(failures))

    report = CpwGuideReport(
        parameters=parameters,
        anchor_points=(
            anchor_points[0],
            anchor_points[1],
            anchor_points[2],
            anchor_points[3],
            anchor_points[4],
            anchor_points[5],
            anchor_points[6],
        ),
        y1_mm=float(y1_mm),
        y2_mm=float(y2_mm),
        guide_bounds=tuple(float(value) for value in cpw_guide.bounds),
        guide_area_mm2=float(cpw_guide.area),
        overlap_with_inner_slot_mm2=float(
            cpw_guide.intersection(combined_inner_slot).area
        ),
        combined_step3_area_mm2=float(combined_step3_geometry.area),
        patch_area_before_mm2=float(patch.area),
        patch_area_after_mm2=float(result_patch.area),
        patch_is_unchanged=bool(patch_is_unchanged),
        guide_was_subtracted=False,
        guide_is_y_axis_symmetric=bool(guide_is_y_axis_symmetric),
        guide_is_valid=bool(cpw_guide.is_valid),
        guide_is_counterclockwise=bool(cpw_guide.exterior.is_ccw),
    )
    return (
        order1_centerline,
        order1_slot,
        order2_centerline,
        order2_slot,
        combined_inner_slot,
        cpw_guide,
        combined_step3_geometry,
        result_patch,
        report,
    )


def generate_cpw_slot_points(
    parameters: AntennaOutlineParameters | None = None,
) -> list[Point2D]:
    """Return closed P0-P5 coordinates for one side of the CPW slot."""

    parameters = _resolve_parameters(parameters)
    outer_x_mm = float(parameters.cpw_slot_p1_p2_x_mm)
    p2_y_mm = float(parameters.cpw_slot_p2_y_mm)
    inner_x_mm = float(parameters.cpw_slot_p3_p4_x_mm)
    p0_x_mm = float(parameters.cpw_slot_p0_x_mm)
    p0_y_mm = float(parameters.cpw_slot_p0_y_mm)
    p1_y_mm = float(parameters.cpw_slot_p1_y_mm)
    p5_x_mm = float(parameters.cpw_slot_p5_x_mm)
    values = (
        p0_x_mm,
        p0_y_mm,
        p1_y_mm,
        p5_x_mm,
        outer_x_mm,
        p2_y_mm,
        inner_x_mm,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("CPW-slot parameters must be finite")
    if not outer_x_mm > inner_x_mm > max(p0_x_mm, p5_x_mm):
        raise ValueError("CPW slot requires outer x > inner x > P0/P5 x")
    if p2_y_mm <= p1_y_mm:
        raise ValueError("CPW-slot P2 y must lie above P1 y")

    p3_y_mm = p2_y_mm + (outer_x_mm - inner_x_mm)
    y1_mm = float(parameters.cpw_slot_y1_linked_mm)
    if not p2_y_mm < p3_y_mm < y1_mm:
        raise ValueError("CPW slot requires P2_y < P3_y < linked y1")

    return [
        (p0_x_mm, p0_y_mm),
        (outer_x_mm, p1_y_mm),
        (outer_x_mm, p2_y_mm),
        (inner_x_mm, p3_y_mm),
        (inner_x_mm, y1_mm),
        (p5_x_mm, y1_mm),
        (p0_x_mm, p0_y_mm),
    ]


def validate_cpw_slot(
    points: Sequence[Point2D],
    parameters: AntennaOutlineParameters | None = None,
) -> Polygon:
    """Validate the linked P0-P5 coordinates and resulting CPW-slot Polygon."""

    parameters = _resolve_parameters(parameters)
    coordinates = np.asarray(points, dtype=float)
    if coordinates.shape != (7, 2):
        raise ValueError(
            f"CPW slot requires six anchors plus closure, got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("CPW slot contains a non-finite coordinate")
    if not np.array_equal(coordinates[0], coordinates[-1]):
        raise ValueError("CPW-slot point sequence is not closed")

    expected_coordinates = np.asarray(generate_cpw_slot_points(parameters), dtype=float)
    polygon = Polygon(coordinates)
    checks = {
        "CPW-slot anchors match fixed and linked definitions": np.allclose(
            coordinates, expected_coordinates, rtol=0.0, atol=1e-12
        ),
        "CPW-slot Polygon is valid": polygon.is_valid,
        "CPW-slot exterior is simple": polygon.exterior.is_simple,
        "CPW-slot exterior is counterclockwise": polygon.exterior.is_ccw,
        "CPW-slot Polygon has positive area": polygon.area > 0.0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("CPW-slot validation failed: " + "; ".join(failures))
    return orient(polygon, sign=1.0)


def generate_cpw_matching_stub_centerlines(
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[LineString, LineString]:
    """Create both fixed matching stubs from the adjustable CPW-slot outer edge."""

    parameters = _resolve_parameters(parameters)
    start_x_mm = float(parameters.cpw_slot_p1_p2_x_mm)
    stub_specs = (
        (
            parameters.cpw_matching_stub1_cap_x_mm,
            parameters.cpw_matching_stub1_lower_y_mm,
            parameters.cpw_matching_stub1_upper_y_mm,
        ),
        (
            parameters.cpw_matching_stub2_cap_x_mm,
            parameters.cpw_matching_stub2_lower_y_mm,
            parameters.cpw_matching_stub2_upper_y_mm,
        ),
    )
    centerlines: list[LineString] = []
    for index, (cap_x_mm, lower_y_mm, upper_y_mm) in enumerate(stub_specs, start=1):
        if cap_x_mm <= start_x_mm:
            raise ValueError(f"matching stub {index} requires cap x > CPW-slot outer x")
        if upper_y_mm <= lower_y_mm:
            raise ValueError(f"matching stub {index} has invalid vertical bounds")
        center_y_mm = (lower_y_mm + upper_y_mm) / 2.0
        centerlines.append(
            LineString(
                [
                    (start_x_mm, center_y_mm),
                    (cap_x_mm, center_y_mm),
                ]
            )
        )

    return centerlines[0], centerlines[1]


def expand_cpw_matching_stubs(
    centerlines: Sequence[LineString],
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[Polygon, Polygon]:
    """Buffer both matching-stub branches vertically using flat end caps."""

    parameters = _resolve_parameters(parameters)
    if len(centerlines) != 2:
        raise ValueError(f"two matching stubs are required, got {len(centerlines)}")
    buffer_distances_mm = (
        (
            parameters.cpw_matching_stub1_upper_y_mm
            - parameters.cpw_matching_stub1_lower_y_mm
        )
        / 2.0,
        (
            parameters.cpw_matching_stub2_upper_y_mm
            - parameters.cpw_matching_stub2_lower_y_mm
        )
        / 2.0,
    )
    stubs: list[Polygon] = []
    for centerline, buffer_mm in zip(centerlines, buffer_distances_mm, strict=True):
        stub = centerline.buffer(buffer_mm, cap_style="flat")
        if not isinstance(stub, Polygon):
            raise TypeError(f"expected a Polygon stub, got {stub.geom_type}")
        stubs.append(orient(stub, sign=1.0))
    return stubs[0], stubs[1]


def build_step_3_4_geometry(
    patch: Polygon,
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[
    LineString,
    Polygon,
    LineString,
    Polygon,
    Polygon,
    tuple[LineString, LineString],
    tuple[Polygon, Polygon],
    Polygon,
    Polygon,
    Polygon,
    Polygon,
    Polygon,
    Polygon,
    CpwSlotAssemblyReport,
]:
    """Unite CPW slot, stubs, and inner branches; mirror slots and guide."""

    parameters = _resolve_parameters(parameters)
    (
        order1_centerline,
        order1_slot,
        order2_centerline,
        order2_slot,
        combined_inner_slot,
        cpw_guide,
        _,
        _,
        _,
    ) = build_step_3_3_geometry(patch, parameters)

    cpw_slot_points = generate_cpw_slot_points(parameters)
    cpw_slot = validate_cpw_slot(cpw_slot_points, parameters)
    stub_centerlines = generate_cpw_matching_stub_centerlines(parameters)
    stub_slots = expand_cpw_matching_stubs(stub_centerlines, parameters)

    slot_with_stubs = cpw_slot.union(stub_slots[0]).union(stub_slots[1])
    if not isinstance(slot_with_stubs, Polygon):
        raise TypeError(
            "CPW slot and matching stubs must unite into one Polygon, "
            f"but Shapely returned {slot_with_stubs.geom_type}"
        )
    slot_with_stubs = orient(slot_with_stubs, sign=1.0)

    reservations = generate_inner_slot_order2_reservations(parameters)
    _, active_reserved_polygons = build_active_reserved_inner_slot_branches(
        reservations
    )
    for reservation, branch in zip(
        (item for item in reservations if item.is_active),
        active_reserved_polygons,
        strict=True,
    ):
        if (
            reservation.growth_direction == (0.0, -1.0)
            and branch.distance(slot_with_stubs) + GEOMETRY_DISTANCE_TOLERANCE_MM
            < DOWNWARD_INNER_BRANCH_CPW_CLEARANCE_FIXED_MM
        ):
            raise ValueError(CPW_FEEDING_INTERFERENCE_ERROR)

    right_combined_slot = combined_inner_slot.union(slot_with_stubs)
    if not isinstance(right_combined_slot, Polygon):
        raise TypeError(
            "right-side CPW and inner slots must unite into one Polygon, "
            f"but Shapely returned {right_combined_slot.geom_type}"
        )
    right_combined_slot = orient(right_combined_slot, sign=1.0)
    substrate = Polygon(generate_rectangle(parameters))
    if not substrate.covers(right_combined_slot):
        raise ValueError("right-side combined slot geometry leaves the substrate")
    if not right_combined_slot.intersects(patch):
        raise ValueError("right-side combined slot geometry does not intersect the Patch")

    left_combined_slot = mirror_about_y_axis(right_combined_slot, parameters)
    symmetric_slot_geometry = right_combined_slot.union(left_combined_slot)
    if not isinstance(symmetric_slot_geometry, Polygon):
        raise TypeError(
            "mirrored CPW and inner slots must unite into one Polygon, "
            f"but Shapely returned {symmetric_slot_geometry.geom_type}"
        )
    symmetric_slot_geometry = orient(symmetric_slot_geometry, sign=1.0)

    mirrored_guide = mirror_about_y_axis(cpw_guide, parameters)
    symmetric_guide = cpw_guide.union(mirrored_guide)
    if not isinstance(symmetric_guide, Polygon):
        raise TypeError(
            "mirrored CPW guide must unite into one Polygon, "
            f"but Shapely returned {symmetric_guide.geom_type}"
        )
    symmetric_guide = orient(symmetric_guide, sign=1.0)
    result_patch = patch

    outer_x_mm = float(parameters.cpw_slot_p1_p2_x_mm)
    p2_y_mm = float(parameters.cpw_slot_p2_y_mm)
    inner_x_mm = float(parameters.cpw_slot_p3_p4_x_mm)
    p3_y_mm = p2_y_mm + (outer_x_mm - inner_x_mm)
    y1_mm = float(parameters.inner_slot_order1_lower_y_mm)
    anchor_points = tuple((float(x), float(y)) for x, y in cpw_slot_points[:-1])
    expected_anchor_points = (
        (parameters.cpw_slot_p0_x_mm, parameters.cpw_slot_p0_y_mm),
        (outer_x_mm, parameters.cpw_slot_p1_y_mm),
        (outer_x_mm, p2_y_mm),
        (inner_x_mm, p3_y_mm),
        (inner_x_mm, y1_mm),
        (parameters.cpw_slot_p5_x_mm, y1_mm),
    )
    expected_slot_area_mm2 = Polygon(expected_anchor_points).area
    expected_stub_bounds = (
        (
            outer_x_mm,
            parameters.cpw_matching_stub1_lower_y_mm,
            parameters.cpw_matching_stub1_cap_x_mm,
            parameters.cpw_matching_stub1_upper_y_mm,
        ),
        (
            outer_x_mm,
            parameters.cpw_matching_stub2_lower_y_mm,
            parameters.cpw_matching_stub2_cap_x_mm,
            parameters.cpw_matching_stub2_upper_y_mm,
        ),
    )
    expected_stub_areas_mm2 = (
        (parameters.cpw_matching_stub1_cap_x_mm - outer_x_mm)
        * (
            parameters.cpw_matching_stub1_upper_y_mm
            - parameters.cpw_matching_stub1_lower_y_mm
        ),
        (parameters.cpw_matching_stub2_cap_x_mm - outer_x_mm)
        * (
            parameters.cpw_matching_stub2_upper_y_mm
            - parameters.cpw_matching_stub2_lower_y_mm
        ),
    )
    expected_slot_with_stubs_area_mm2 = expected_slot_area_mm2 + sum(
        expected_stub_areas_mm2
    )
    expected_right_combined_area_mm2 = (
        expected_slot_with_stubs_area_mm2 + combined_inner_slot.area
    )
    expected_symmetric_slot_area_mm2 = 2.0 * expected_right_combined_area_mm2
    expected_symmetric_guide_area_mm2 = 2.0 * cpw_guide.area
    active_reserved_count = sum(item.is_active for item in reservations)
    slot_assembly_is_y_axis_symmetric = symmetric_slot_geometry.equals(
        mirror_about_y_axis(symmetric_slot_geometry, parameters)
    )
    guide_is_y_axis_symmetric = symmetric_guide.equals(
        mirror_about_y_axis(symmetric_guide, parameters)
    )
    patch_is_unchanged = result_patch.equals_exact(patch, tolerance=0.0)
    stub_centerline_coordinates = tuple(
        tuple((float(x), float(y)) for x, y in line.coords) for line in stub_centerlines
    )

    checks = {
        "CPW-slot P0-P5 anchors": np.allclose(
            anchor_points, expected_anchor_points, rtol=0.0, atol=1e-12
        ),
        "CPW-slot area": math.isclose(
            cpw_slot.area,
            expected_slot_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "matching-stub bounds": all(
            np.allclose(stub.bounds, bounds, rtol=0.0, atol=1e-12)
            for stub, bounds in zip(stub_slots, expected_stub_bounds, strict=True)
        ),
        "matching-stub areas": all(
            math.isclose(stub.area, area, rel_tol=0.0, abs_tol=1e-9)
            for stub, area in zip(stub_slots, expected_stub_areas_mm2, strict=True)
        ),
        "CPW slot and matching-stub union area": math.isclose(
            slot_with_stubs.area,
            expected_slot_with_stubs_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "one inner main branch and four order-2 specifications": (
            1 == 1 and 1 + len(reservations) == 4
        ),
        "inner order-2 active count is valid": (
            1 <= 1 + active_reserved_count <= 1 + len(reservations)
        ),
        "right combined slot area": math.isclose(
            right_combined_slot.area,
            expected_right_combined_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "symmetric slot area": math.isclose(
            symmetric_slot_geometry.area,
            expected_symmetric_slot_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "symmetric guide area": math.isclose(
            symmetric_guide.area,
            expected_symmetric_guide_area_mm2,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "slot assembly is y-axis symmetric": slot_assembly_is_y_axis_symmetric,
        "guide is y-axis symmetric": guide_is_y_axis_symmetric,
        "Patch object is returned without subtraction": result_patch is patch,
        "Patch geometry is unchanged": patch_is_unchanged,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("step 3.4 validation failed: " + "; ".join(failures))

    report = CpwSlotAssemblyReport(
        parameters=parameters,
        anchor_points=(
            anchor_points[0],
            anchor_points[1],
            anchor_points[2],
            anchor_points[3],
            anchor_points[4],
            anchor_points[5],
        ),
        p3_y_mm=float(p3_y_mm),
        y1_mm=float(y1_mm),
        slot_bounds=tuple(float(value) for value in cpw_slot.bounds),
        slot_area_mm2=float(cpw_slot.area),
        stub_centerlines=(
            (
                stub_centerline_coordinates[0][0],
                stub_centerline_coordinates[0][1],
            ),
            (
                stub_centerline_coordinates[1][0],
                stub_centerline_coordinates[1][1],
            ),
        ),
        stub_bounds=(
            tuple(float(value) for value in stub_slots[0].bounds),
            tuple(float(value) for value in stub_slots[1].bounds),
        ),
        stub_areas_mm2=(float(stub_slots[0].area), float(stub_slots[1].area)),
        slot_with_stubs_area_mm2=float(slot_with_stubs.area),
        inner_main_branch_count=1,
        inner_order2_branch_count=1 + len(reservations),
        inner_order2_active_count=1 + active_reserved_count,
        right_combined_slot_area_mm2=float(right_combined_slot.area),
        symmetric_slot_area_mm2=float(symmetric_slot_geometry.area),
        right_guide_area_mm2=float(cpw_guide.area),
        symmetric_guide_area_mm2=float(symmetric_guide.area),
        patch_area_before_mm2=float(patch.area),
        patch_area_after_mm2=float(result_patch.area),
        patch_is_unchanged=bool(patch_is_unchanged),
        slot_was_subtracted=False,
        slot_assembly_is_y_axis_symmetric=bool(slot_assembly_is_y_axis_symmetric),
        guide_is_y_axis_symmetric=bool(guide_is_y_axis_symmetric),
    )
    return (
        order1_centerline,
        order1_slot,
        order2_centerline,
        order2_slot,
        cpw_slot,
        stub_centerlines,
        stub_slots,
        slot_with_stubs,
        right_combined_slot,
        symmetric_slot_geometry,
        cpw_guide,
        symmetric_guide,
        result_patch,
        report,
    )


def build_antenna_closed_polygons(
    parameters: AntennaOutlineParameters | None = None,
) -> tuple[Polygon, Polygon, Polygon]:
    """Return the final Patch, symmetric slot, and symmetric guide polygons."""

    parameters = _resolve_parameters(parameters)
    rectangle_points = generate_rectangle(parameters)
    rectangle, _ = validate_rectangle(rectangle_points, parameters)
    right_outer_slots = build_step_2_4_outline(rectangle, parameters)[8]
    patch = build_step_2_5_outline(
        rectangle,
        right_outer_slots,
        parameters,
    )[3]
    step_3_4 = build_step_3_4_geometry(patch, parameters)
    symmetric_slot = step_3_4[9]
    symmetric_guide = step_3_4[11]
    result_patch = step_3_4[12]
    if result_patch is not patch:
        raise ValueError("step 3.4 unexpectedly replaced the Patch object")
    return result_patch, symmetric_slot, symmetric_guide


def quantize_coordinate_mm(
    value: float,
    quantum_mm: float = COORDINATE_QUANTUM_MM,
) -> float:
    """Quantize one finite coordinate to the nearest grid using half-up ties."""

    try:
        decimal_value = Decimal(str(value))
        decimal_quantum = Decimal(str(quantum_mm))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("coordinate and quantum must be finite numbers") from exc
    if not decimal_value.is_finite():
        raise ValueError("coordinate must be finite")
    if not decimal_quantum.is_finite() or decimal_quantum <= 0:
        raise ValueError("coordinate quantum must be finite and positive")
    try:
        grid_index = (decimal_value / decimal_quantum).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    except InvalidOperation as exc:
        raise ValueError("coordinate cannot be represented on the requested grid") from exc
    quantized = float(grid_index * decimal_quantum)
    return 0.0 if quantized == 0.0 else quantized


def _closed_curve_signed_area(points: Sequence[Point2D]) -> float:
    return 0.5 * sum(
        x_current * y_next - x_next * y_current
        for (x_current, y_current), (x_next, y_next) in zip(
            points[:-1],
            points[1:],
            strict=True,
        )
    )


def quantize_and_validate_closed_polygon_points(
    points: Sequence[Point2D],
    *,
    curve_name: str,
    quantum_mm: float = COORDINATE_QUANTUM_MM,
) -> list[Point2D]:
    """Return one CST-ready curve after quantization and self-cross checks."""

    raw_points = list(points)
    if len(raw_points) < 4:
        raise ValueError(f"{curve_name} requires at least three vertices plus closure")
    if raw_points[0] != raw_points[-1]:
        raise ValueError(f"{curve_name} is not explicitly closed before quantization")

    quantized: list[Point2D] = []
    for index, point in enumerate(raw_points):
        if len(point) != 2:
            raise ValueError(f"{curve_name} point {index} is not two-dimensional")
        quantized.append(
            (
                quantize_coordinate_mm(point[0], quantum_mm),
                quantize_coordinate_mm(point[1], quantum_mm),
            )
        )
    quantized[-1] = quantized[0]

    for index, (current, following) in enumerate(
        zip(quantized[:-1], quantized[1:], strict=True)
    ):
        if current == following:
            raise ValueError(
                f"{curve_name} edge {index} collapsed after {quantum_mm:g} mm "
                "coordinate quantization"
            )
    if len(set(quantized[:-1])) < 3:
        raise ValueError(
            f"{curve_name} has fewer than three distinct quantized vertices"
        )

    ring = LinearRing(quantized)
    if not ring.is_simple:
        raise ValueError(
            f"{curve_name} self-intersects or self-touches after "
            f"{quantum_mm:g} mm coordinate quantization"
        )
    polygon = Polygon(quantized)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError(
            f"{curve_name} is invalid after coordinate quantization: "
            f"{explain_validity(polygon)}"
        )
    if len(polygon.interiors) != 0:
        raise ValueError(f"{curve_name} unexpectedly contains an interior ring")
    if polygon.area <= GEOMETRY_TOLERANCE_MM2:
        raise ValueError(f"{curve_name} has zero area after coordinate quantization")
    if _closed_curve_signed_area(quantized) <= 0.0:
        raise ValueError(
            f"{curve_name} must retain counterclockwise point order; "
            "the CST extrusion thickness controls its Z direction"
        )
    return quantized


def polygon_exterior_to_closed_points(
    polygon: Polygon,
    *,
    curve_name: str = "antenna polygon",
    quantum_mm: float = COORDINATE_QUANTUM_MM,
) -> list[Point2D]:
    """Convert one hole-free Polygon to a quantized CST-ready closed curve."""

    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("cannot extract points from an empty or invalid Polygon")
    if len(polygon.interiors) != 0:
        raise ValueError("antenna export polygons must not contain interior rings")
    raw_points = [
        (
            0.0 if float(x) == 0.0 else float(x),
            0.0 if float(y) == 0.0 else float(y),
        )
        for x, y in polygon.exterior.coords
    ]
    return quantize_and_validate_closed_polygon_points(
        raw_points,
        curve_name=curve_name,
        quantum_mm=quantum_mm,
    )


def generate_complete_antenna_point_lists(
    parameters: AntennaOutlineParameters | None = None,
    *,
    quantum_mm: float = COORDINATE_QUANTUM_MM,
) -> list[list[Point2D]]:
    """Return the three quantized, self-intersection-free CST point lists."""

    return [
        polygon_exterior_to_closed_points(
            polygon,
            curve_name=curve_name,
            quantum_mm=quantum_mm,
        )
        for curve_name, polygon in zip(
            ("patch", "symmetric slot", "symmetric guide"),
            build_antenna_closed_polygons(parameters),
            strict=True,
        )
    ]


def generate_reflector_outline_points(
    parameters: AntennaOutlineParameters | None = None,
) -> list[Point2D]:
    """Return the reflector exterior with its fixed bottom connector clearance."""

    parameters = _resolve_parameters(parameters)
    half_length = parameters.rectangle_length_mm / 2.0
    cutout_half_width = (
        half_length
        - parameters.reflector_connector_board_thickness_mm
        + parameters.reflector_cutout_width_adjustment_mm
    )
    depth = parameters.reflector_cutout_depth_mm
    return [
        (-half_length, 0.0),
        (-cutout_half_width, 0.0),
        (-cutout_half_width, depth),
        (cutout_half_width, depth),
        (cutout_half_width, 0.0),
        (half_length, 0.0),
        (half_length, parameters.rectangle_width_mm),
        (-half_length, parameters.rectangle_width_mm),
        (-half_length, 0.0),
    ]


def plot_rectangle(
    points: Sequence[Point2D],
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the closed rectangle using an equal-aspect millimetre coordinate system."""

    coordinates = np.asarray(points, dtype=float)
    figure, axes = plt.subplots(figsize=(10.0, 6.5))
    axes.fill(
        coordinates[:, 0],
        coordinates[:, 1],
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=2.2,
        alpha=0.75,
    )
    axes.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.2,
    )
    axes.scatter(
        coordinates[:-1, 0],
        coordinates[:-1, 1],
        color="#dc2626",
        s=28,
        zorder=3,
    )

    axes.axhline(0.0, color="#111827", linewidth=1.0)
    axes.axvline(0.0, color="#6b7280", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_title(
        "Antenna outline step 1: "
        f"{coordinates[1, 0] - coordinates[0, 0]:g} x "
        f"{coordinates[2, 1] - coordinates[1, 1]:g} mm rectangle"
    )
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    width = coordinates[:, 0].max() - coordinates[:, 0].min()
    height = coordinates[:, 1].max() - coordinates[:, 1].min()
    margin = max(width, height) * 0.06
    axes.set_xlim(coordinates[:, 0].min() - margin, coordinates[:, 0].max() + margin)
    axes.set_ylim(-margin, coordinates[:, 1].max() + margin)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_2_3(
    rectangle: Polygon,
    upper_order1_centerline: LineString,
    upper_order1_slot: Polygon,
    upper_order2_centerline: LineString,
    upper_order2_slot: Polygon,
    lower_order1_centerline: LineString,
    lower_order1_slot: Polygon,
    outline: Polygon,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot all three outer-slot stages and the resulting step-2.3 outline."""

    rectangle_coordinates = np.asarray(rectangle.exterior.coords, dtype=float)
    upper_order1_line_coordinates = np.asarray(
        upper_order1_centerline.coords, dtype=float
    )
    upper_order1_slot_coordinates = np.asarray(
        upper_order1_slot.exterior.coords, dtype=float
    )
    upper_order2_line_coordinates = np.asarray(
        upper_order2_centerline.coords, dtype=float
    )
    upper_order2_slot_coordinates = np.asarray(
        upper_order2_slot.exterior.coords, dtype=float
    )
    lower_order1_line_coordinates = np.asarray(
        lower_order1_centerline.coords, dtype=float
    )
    lower_order1_slot_coordinates = np.asarray(
        lower_order1_slot.exterior.coords, dtype=float
    )
    outline_coordinates = np.asarray(outline.exterior.coords, dtype=float)

    figure, axes = plt.subplots(figsize=(10.0, 6.5))
    axes.fill(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=2.4,
        alpha=0.78,
        label="step 2.3 outline",
    )
    axes.plot(
        rectangle_coordinates[:, 0],
        rectangle_coordinates[:, 1],
        color="#6b7280",
        linestyle="--",
        linewidth=1.2,
        label="original rectangle",
    )
    axes.fill(
        upper_order1_slot_coordinates[:, 0],
        upper_order1_slot_coordinates[:, 1],
        facecolor="#fecaca",
        edgecolor="#dc2626",
        linestyle=":",
        linewidth=1.5,
        alpha=0.58,
        label="upper order-1 slot",
    )
    axes.fill(
        upper_order2_slot_coordinates[:, 0],
        upper_order2_slot_coordinates[:, 1],
        facecolor="#fed7aa",
        edgecolor="#ea580c",
        linestyle=":",
        linewidth=1.7,
        alpha=0.62,
        label="upper order-2 slot",
    )
    axes.fill(
        lower_order1_slot_coordinates[:, 0],
        lower_order1_slot_coordinates[:, 1],
        facecolor="#bbf7d0",
        edgecolor="#16a34a",
        linestyle=":",
        linewidth=1.7,
        alpha=0.66,
        label="lower order-1 slot",
    )
    axes.plot(
        upper_order1_line_coordinates[:, 0],
        upper_order1_line_coordinates[:, 1],
        color="#991b1b",
        linestyle="-.",
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        label="upper order-1 line",
    )
    axes.plot(
        upper_order2_line_coordinates[:, 0],
        upper_order2_line_coordinates[:, 1],
        color="#9a3412",
        linestyle="-.",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label="upper order-2 line",
    )
    axes.plot(
        lower_order1_line_coordinates[:, 0],
        lower_order1_line_coordinates[:, 1],
        color="#166534",
        linestyle="-.",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label="lower order-1 line",
    )
    axes.plot(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.4,
    )

    axes.axhline(0.0, color="#111827", linewidth=1.0)
    axes.axvline(0.0, color="#6b7280", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    lower_order1_min_x, lower_order1_min_y, lower_order1_max_x, lower_order1_max_y = (
        lower_order1_slot.bounds
    )
    axes.set_title(
        "Antenna outline step 2.3: lower first-order outer slot "
        f"({lower_order1_max_x - lower_order1_min_x:g} x "
        f"{lower_order1_max_y - lower_order1_min_y:g} mm)"
    )
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes.legend(loc="center left")

    min_x, min_y, max_x, max_y = rectangle.bounds
    margin = max(max_x - min_x, max_y - min_y) * 0.06
    axes.set_xlim(min_x - margin, max_x + margin)
    axes.set_ylim(min_y - margin, max_y + margin)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_2_4(
    rectangle: Polygon,
    upper_order1_centerline: LineString,
    upper_order1_slot: Polygon,
    upper_order2_centerline: LineString,
    upper_order2_slot: Polygon,
    lower_order1_centerline: LineString,
    lower_order1_slot: Polygon,
    lower_order2_centerlines: Sequence[LineString],
    lower_order2_slots: Sequence[Polygon],
    outline: Polygon,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot both lower second-order branches and the resulting step-2.4 outline."""

    if len(lower_order2_centerlines) != 2 or len(lower_order2_slots) != 2:
        raise ValueError(
            "step 2.4 plotting requires exactly two lower order-2 branches"
        )

    figure, axes = plot_step_2_3(
        rectangle,
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        outline,
        show=False,
    )
    for artist in axes.get_children():
        if artist.get_label() == "step 2.3 outline":
            artist.set_label("step 2.4 outline")

    branch_colours = (
        ("#bae6fd", "#0284c7", "#075985"),
        ("#ddd6fe", "#7c3aed", "#5b21b6"),
    )
    for index, (centerline, slot, colours) in enumerate(
        zip(
            lower_order2_centerlines,
            lower_order2_slots,
            branch_colours,
            strict=True,
        ),
        start=1,
    ):
        slot_coordinates = np.asarray(slot.exterior.coords, dtype=float)
        line_coordinates = np.asarray(centerline.coords, dtype=float)
        face_colour, edge_colour, line_colour = colours
        axes.fill(
            slot_coordinates[:, 0],
            slot_coordinates[:, 1],
            facecolor=face_colour,
            edgecolor=edge_colour,
            linestyle=":",
            linewidth=1.7,
            alpha=0.68,
            label=f"lower order-2 branch {index} slot",
            zorder=2,
        )
        axes.plot(
            line_coordinates[:, 0],
            line_coordinates[:, 1],
            color=line_colour,
            linestyle="-.",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=f"lower order-2 branch {index} line",
            zorder=5,
        )

    outline_coordinates = np.asarray(outline.exterior.coords, dtype=float)
    axes.plot(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.4,
        zorder=4,
    )
    axes.set_title(
        "Antenna outline step 2.4: two lower second-order outer-slot branches "
        f"(inner x={lower_order2_slots[0].bounds[0]:g}, "
        f"{lower_order2_slots[1].bounds[0]:g} mm)"
    )
    axes.legend(loc="center left")
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_2_5(
    rectangle: Polygon,
    upper_order1_centerline: LineString,
    upper_order1_slot: Polygon,
    upper_order2_centerline: LineString,
    upper_order2_slot: Polygon,
    lower_order1_centerline: LineString,
    lower_order1_slot: Polygon,
    lower_order2_centerlines: Sequence[LineString],
    lower_order2_slots: Sequence[Polygon],
    outline: Polygon,
    *,
    parameters: AntennaOutlineParameters | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the right-side construction and its reflection across the y-axis."""

    parameters = _resolve_parameters(parameters)
    figure, axes = plot_step_2_4(
        rectangle,
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        lower_order2_centerlines,
        lower_order2_slots,
        outline,
        show=False,
    )
    for artist in axes.get_children():
        if artist.get_label() == "step 2.4 outline":
            artist.set_label("step 2.5 symmetric outline")

    construction_pairs = (
        (
            upper_order1_centerline,
            upper_order1_slot,
            ("#fecaca", "#dc2626", "#991b1b"),
        ),
        (
            upper_order2_centerline,
            upper_order2_slot,
            ("#fed7aa", "#ea580c", "#9a3412"),
        ),
        (
            lower_order1_centerline,
            lower_order1_slot,
            ("#bbf7d0", "#16a34a", "#166534"),
        ),
        (
            lower_order2_centerlines[0],
            lower_order2_slots[0],
            ("#bae6fd", "#0284c7", "#075985"),
        ),
        (
            lower_order2_centerlines[1],
            lower_order2_slots[1],
            ("#ddd6fe", "#7c3aed", "#5b21b6"),
        ),
    )
    for centerline, slot, colours in construction_pairs:
        mirrored_centerline = mirror_about_y_axis(centerline, parameters)
        mirrored_slot = mirror_about_y_axis(slot, parameters)
        if not isinstance(mirrored_centerline, LineString):
            raise TypeError(
                f"expected mirrored LineString, got {mirrored_centerline.geom_type}"
            )
        if not isinstance(mirrored_slot, Polygon):
            raise TypeError(f"expected mirrored Polygon, got {mirrored_slot.geom_type}")

        line_coordinates = np.asarray(mirrored_centerline.coords, dtype=float)
        slot_coordinates = np.asarray(mirrored_slot.exterior.coords, dtype=float)
        face_colour, edge_colour, line_colour = colours
        axes.fill(
            slot_coordinates[:, 0],
            slot_coordinates[:, 1],
            facecolor=face_colour,
            edgecolor=edge_colour,
            linestyle=":",
            linewidth=1.7,
            alpha=0.62,
            label="_nolegend_",
            zorder=2,
        )
        axes.plot(
            line_coordinates[:, 0],
            line_coordinates[:, 1],
            color=line_colour,
            linestyle="-.",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label="_nolegend_",
            zorder=5,
        )

    outline_coordinates = np.asarray(outline.exterior.coords, dtype=float)
    axes.plot(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.4,
        zorder=4,
    )
    axes.axvline(
        parameters.outer_slot_symmetry_axis_x_mm,
        color="#374151",
        linewidth=1.2,
        linestyle="--",
        label="y-axis symmetry line",
    )
    axes.set_title("Antenna outline step 2.5: all outer slots mirrored about y-axis")
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_3_1(
    rectangle: Polygon,
    upper_order1_centerline: LineString,
    upper_order1_slot: Polygon,
    upper_order2_centerline: LineString,
    upper_order2_slot: Polygon,
    lower_order1_centerline: LineString,
    lower_order1_slot: Polygon,
    lower_order2_centerlines: Sequence[LineString],
    lower_order2_slots: Sequence[Polygon],
    patch: Polygon,
    inner_order1_centerline: LineString,
    inner_order1_slot: Polygon,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the unmirrored inner slot over the unchanged symmetric Patch."""

    figure, axes = plot_step_2_5(
        rectangle,
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        lower_order2_centerlines,
        lower_order2_slots,
        patch,
        show=False,
    )
    for artist in axes.get_children():
        if artist.get_label() == "step 2.5 symmetric outline":
            artist.set_label("step 3.1 Patch (unchanged)")

    slot_coordinates = np.asarray(inner_order1_slot.exterior.coords, dtype=float)
    line_coordinates = np.asarray(inner_order1_centerline.coords, dtype=float)
    axes.fill(
        slot_coordinates[:, 0],
        slot_coordinates[:, 1],
        facecolor="#fef08a",
        edgecolor="#ca8a04",
        linestyle=":",
        linewidth=1.9,
        alpha=0.70,
        hatch="//",
        label="inner order-1 geometry (not subtracted)",
        zorder=3,
    )
    axes.plot(
        line_coordinates[:, 0],
        line_coordinates[:, 1],
        color="#854d0e",
        linestyle="-.",
        linewidth=2.2,
        marker="o",
        markersize=4.8,
        label="inner order-1 L-system line",
        zorder=6,
    )
    axes.set_title(
        "Antenna geometry step 3.1: unmirrored inner order-1 slot (Patch unchanged)"
    )
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_3_2(
    rectangle: Polygon,
    upper_order1_centerline: LineString,
    upper_order1_slot: Polygon,
    upper_order2_centerline: LineString,
    upper_order2_slot: Polygon,
    lower_order1_centerline: LineString,
    lower_order1_slot: Polygon,
    lower_order2_centerlines: Sequence[LineString],
    lower_order2_slots: Sequence[Polygon],
    patch: Polygon,
    inner_order1_centerline: LineString,
    inner_order1_slot: Polygon,
    inner_order2_centerline: LineString,
    inner_order2_slot: Polygon,
    reserved_branches: Sequence[InnerSlotBranchReservation],
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the single Y+ inner order-2 branch over the unchanged Patch."""

    figure, axes = plot_step_3_1(
        rectangle,
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        lower_order2_centerlines,
        lower_order2_slots,
        patch,
        inner_order1_centerline,
        inner_order1_slot,
        show=False,
    )
    for artist in axes.get_children():
        if artist.get_label() == "step 3.1 Patch (unchanged)":
            artist.set_label("step 3.2 Patch (unchanged)")

    slot_coordinates = np.asarray(inner_order2_slot.exterior.coords, dtype=float)
    line_coordinates = np.asarray(inner_order2_centerline.coords, dtype=float)
    axes.fill(
        slot_coordinates[:, 0],
        slot_coordinates[:, 1],
        facecolor="#fbcfe8",
        edgecolor="#db2777",
        linestyle=":",
        linewidth=1.9,
        alpha=0.70,
        hatch="\\\\",
        label="inner order-2 geometry (not subtracted)",
        zorder=4,
    )
    axes.plot(
        line_coordinates[:, 0],
        line_coordinates[:, 1],
        color="#9d174d",
        linestyle="-.",
        linewidth=2.2,
        marker="o",
        markersize=4.8,
        label="inner order-2 L-system line (Y+)",
        zorder=7,
    )
    axes.annotate(
        "",
        xy=tuple(line_coordinates[-1]),
        xytext=tuple(line_coordinates[0]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#9d174d",
            "linewidth": 2.0,
        },
        zorder=8,
    )
    if len(reserved_branches) != 3:
        raise ValueError("step 3.2 plotting expects three reserved branches")
    for index, reservation in enumerate(reserved_branches):
        marker = "^" if reservation.growth_direction == (0.0, 1.0) else "v"
        axes.scatter(
            [reservation.anchor[0]],
            [reservation.anchor[1]],
            marker=marker,
            s=85,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.6,
            label=(
                "reserved zero-size order-2 branches" if index == 0 else "_nolegend_"
            ),
            zorder=9,
        )
    axes.set_title(
        "Antenna geometry step 3.2: one active + three zero-size reserved "
        "inner order-2 branches"
    )
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_3_3(
    rectangle: Polygon,
    upper_order1_centerline: LineString,
    upper_order1_slot: Polygon,
    upper_order2_centerline: LineString,
    upper_order2_slot: Polygon,
    lower_order1_centerline: LineString,
    lower_order1_slot: Polygon,
    lower_order2_centerlines: Sequence[LineString],
    lower_order2_slots: Sequence[Polygon],
    patch: Polygon,
    inner_order1_centerline: LineString,
    inner_order1_slot: Polygon,
    inner_order2_centerline: LineString,
    inner_order2_slot: Polygon,
    reserved_branches: Sequence[InnerSlotBranchReservation],
    cpw_guide: Polygon,
    cpw_guide_report: CpwGuideReport,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the parameterized CPW guide over the unchanged step-3 Patch."""

    figure, axes = plot_step_3_2(
        rectangle,
        upper_order1_centerline,
        upper_order1_slot,
        upper_order2_centerline,
        upper_order2_slot,
        lower_order1_centerline,
        lower_order1_slot,
        lower_order2_centerlines,
        lower_order2_slots,
        patch,
        inner_order1_centerline,
        inner_order1_slot,
        inner_order2_centerline,
        inner_order2_slot,
        reserved_branches,
        show=False,
    )
    for artist in axes.get_children():
        if artist.get_label() == "step 3.2 Patch (unchanged)":
            artist.set_label("step 3.3 Patch (unchanged)")

    cpw_guide_coordinates = np.asarray(cpw_guide.exterior.coords, dtype=float)
    anchor_coordinates = np.asarray(cpw_guide_report.anchor_points, dtype=float)
    axes.fill(
        cpw_guide_coordinates[:, 0],
        cpw_guide_coordinates[:, 1],
        facecolor="#99f6e4",
        edgecolor="#0f766e",
        linestyle=":",
        linewidth=2.0,
        alpha=0.72,
        hatch="xx",
        label="parameterized CPW guide",
        zorder=4,
    )
    axes.plot(
        cpw_guide_coordinates[:, 0],
        cpw_guide_coordinates[:, 1],
        color="#0f766e",
        linewidth=2.0,
        zorder=6,
    )
    axes.scatter(
        anchor_coordinates[:, 0],
        anchor_coordinates[:, 1],
        color="#134e4a",
        edgecolors="white",
        linewidths=0.7,
        s=36,
        label="CPW anchors P1-P7",
        zorder=8,
    )
    label_offsets = (
        (-18, -14),
        (4, -14),
        (6, 6),
        (6, 6),
        (-18, 6),
        (-18, 6),
        (6, 6),
    )
    for index, ((x, y), offset) in enumerate(
        zip(anchor_coordinates, label_offsets, strict=True), start=1
    ):
        axes.annotate(
            f"P{index}",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            color="#134e4a",
            fontsize=8,
            fontweight="bold",
            zorder=9,
        )

    parameters = cpw_guide_report.parameters
    axes.set_title(
        "Antenna step 3.3: parameterized CPW guide\n"
        f"P3/P4 x={parameters.cpw_guide_p3_p4_x_mm:g}, "
        f"P5/P6 x={parameters.cpw_guide_p5_p6_x_mm:g}, "
        f"P4 y={parameters.cpw_guide_p4_y_mm:g} mm"
    )
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_3_4(
    rectangle: Polygon,
    patch: Polygon,
    symmetric_slot_geometry: Polygon,
    symmetric_guide: Polygon,
    cpw_slot: Polygon,
    stub_centerlines: Sequence[LineString],
    report: CpwSlotAssemblyReport,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the symmetric CPW-slot assembly and symmetric guide separately."""

    if len(stub_centerlines) != 2:
        raise ValueError("step 3.4 plotting requires two matching stubs")

    rectangle_coordinates = np.asarray(rectangle.exterior.coords, dtype=float)
    patch_coordinates = np.asarray(patch.exterior.coords, dtype=float)
    slot_coordinates = np.asarray(symmetric_slot_geometry.exterior.coords, dtype=float)
    guide_coordinates = np.asarray(symmetric_guide.exterior.coords, dtype=float)
    cpw_slot_coordinates = np.asarray(cpw_slot.exterior.coords, dtype=float)
    anchor_coordinates = np.asarray(report.anchor_points, dtype=float)

    figure, axes = plt.subplots(figsize=(11.5, 7.0))
    axes.fill(
        patch_coordinates[:, 0],
        patch_coordinates[:, 1],
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=2.4,
        alpha=0.76,
        label="step 3.4 Patch (unchanged)",
        zorder=1,
    )
    axes.plot(
        rectangle_coordinates[:, 0],
        rectangle_coordinates[:, 1],
        color="#6b7280",
        linestyle="--",
        linewidth=1.2,
        label="original rectangle",
        zorder=2,
    )
    axes.fill(
        slot_coordinates[:, 0],
        slot_coordinates[:, 1],
        facecolor="#fda4af",
        edgecolor="#be123c",
        linewidth=1.8,
        linestyle=":",
        alpha=0.58,
        hatch="//",
        label="symmetric CPW slot + stubs + inner branches",
        zorder=3,
    )
    axes.fill(
        guide_coordinates[:, 0],
        guide_coordinates[:, 1],
        facecolor="#5eead4",
        edgecolor="#0f766e",
        linewidth=2.0,
        alpha=0.80,
        hatch="xx",
        label="symmetric CPW guide",
        zorder=4,
    )
    axes.plot(
        cpw_slot_coordinates[:, 0],
        cpw_slot_coordinates[:, 1],
        color="#9f1239",
        linewidth=1.6,
        label="right CPW-slot P0-P5 boundary",
        zorder=6,
    )
    axes.scatter(
        anchor_coordinates[:, 0],
        anchor_coordinates[:, 1],
        color="#881337",
        edgecolors="white",
        linewidths=0.7,
        s=34,
        label="right CPW-slot anchors P0-P5",
        zorder=8,
    )
    label_offsets = (
        (-18, -14),
        (4, -14),
        (5, 5),
        (5, 5),
        (5, 5),
        (-20, 5),
    )
    for index, ((x, y), offset) in enumerate(
        zip(anchor_coordinates, label_offsets, strict=True)
    ):
        axes.annotate(
            f"P{index}",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            color="#881337",
            fontsize=8,
            fontweight="bold",
            zorder=9,
        )

    for index, centerline in enumerate(stub_centerlines):
        right_coordinates = np.asarray(centerline.coords, dtype=float)
        mirrored_centerline = mirror_about_y_axis(centerline, report.parameters)
        if not isinstance(mirrored_centerline, LineString):
            raise TypeError(
                f"expected mirrored LineString, got {mirrored_centerline.geom_type}"
            )
        left_coordinates = np.asarray(mirrored_centerline.coords, dtype=float)
        axes.plot(
            right_coordinates[:, 0],
            right_coordinates[:, 1],
            color="#7f1d1d",
            linestyle="-.",
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            label="matching-stub centrelines" if index == 0 else "_nolegend_",
            zorder=7,
        )
        axes.plot(
            left_coordinates[:, 0],
            left_coordinates[:, 1],
            color="#7f1d1d",
            linestyle="-.",
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            label="_nolegend_",
            zorder=7,
        )

    axes.plot(
        patch_coordinates[:, 0],
        patch_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.4,
        zorder=5,
    )
    axes.axhline(0.0, color="#111827", linewidth=1.0)
    axes.axvline(
        0.0,
        color="#374151",
        linewidth=1.2,
        linestyle="--",
        label="y-axis symmetry line",
    )
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_title(
        "Antenna step 3.4: symmetric CPW slot assembly and CPW guide\n"
        f"slot assembly={report.symmetric_slot_area_mm2:g} mm², "
        f"guide={report.symmetric_guide_area_mm2:g} mm²"
    )
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    min_x, min_y, max_x, max_y = rectangle.bounds
    margin = max(max_x - min_x, max_y - min_y) * 0.06
    axes.set_xlim(min_x - margin, max_x + margin)
    axes.set_ylim(min_y - margin, max_y + margin)
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_2_2(
    rectangle: Polygon,
    order1_centerline: LineString,
    order1_slot: Polygon,
    order2_centerline: LineString,
    order2_slot: Polygon,
    outline: Polygon,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot both upper outer-slot orders and the resulting antenna outline."""

    rectangle_coordinates = np.asarray(rectangle.exterior.coords, dtype=float)
    order1_line_coordinates = np.asarray(order1_centerline.coords, dtype=float)
    order1_slot_coordinates = np.asarray(order1_slot.exterior.coords, dtype=float)
    order2_line_coordinates = np.asarray(order2_centerline.coords, dtype=float)
    order2_slot_coordinates = np.asarray(order2_slot.exterior.coords, dtype=float)
    outline_coordinates = np.asarray(outline.exterior.coords, dtype=float)

    figure, axes = plt.subplots(figsize=(10.0, 6.5))
    axes.fill(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=2.4,
        alpha=0.78,
        label="step 2.2 outline",
    )
    axes.plot(
        rectangle_coordinates[:, 0],
        rectangle_coordinates[:, 1],
        color="#6b7280",
        linestyle="--",
        linewidth=1.2,
        label="original rectangle",
    )
    axes.fill(
        order1_slot_coordinates[:, 0],
        order1_slot_coordinates[:, 1],
        facecolor="#fecaca",
        edgecolor="#dc2626",
        linestyle=":",
        linewidth=1.5,
        alpha=0.58,
        label="order-1 slot",
    )
    axes.fill(
        order2_slot_coordinates[:, 0],
        order2_slot_coordinates[:, 1],
        facecolor="#fed7aa",
        edgecolor="#ea580c",
        linestyle=":",
        linewidth=1.7,
        alpha=0.62,
        label="order-2 slot",
    )
    axes.plot(
        order1_line_coordinates[:, 0],
        order1_line_coordinates[:, 1],
        color="#991b1b",
        linestyle="-.",
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        label="order-1 line",
    )
    axes.plot(
        order2_line_coordinates[:, 0],
        order2_line_coordinates[:, 1],
        color="#9a3412",
        linestyle="-.",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label="order-2 line",
    )
    axes.plot(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.4,
    )

    axes.axhline(0.0, color="#111827", linewidth=1.0)
    axes.axvline(0.0, color="#6b7280", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_title(
        "Antenna outline step 2.2: upper second-order outer slot "
        f"(line {order2_centerline.length:g} mm, "
        f"buffer +/-"
        f"{(order2_slot.bounds[3] - order2_slot.bounds[1]) / 2.0:g} mm)"
    )
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes.legend(loc="lower left")

    min_x, min_y, max_x, max_y = rectangle.bounds
    margin = max(max_x - min_x, max_y - min_y) * 0.06
    axes.set_xlim(min_x - margin, max_x + margin)
    axes.set_ylim(min_y - margin, max_y + margin)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_step_2_1(
    rectangle: Polygon,
    centerline: LineString,
    slot: Polygon,
    outline: Polygon,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the original rectangle, construction line, expanded slot, and result."""

    rectangle_coordinates = np.asarray(rectangle.exterior.coords, dtype=float)
    centerline_coordinates = np.asarray(centerline.coords, dtype=float)
    slot_coordinates = np.asarray(slot.exterior.coords, dtype=float)
    outline_coordinates = np.asarray(outline.exterior.coords, dtype=float)

    figure, axes = plt.subplots(figsize=(10.0, 6.5))
    axes.fill(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=2.4,
        alpha=0.78,
        label="step 2.1 outline",
    )
    axes.plot(
        rectangle_coordinates[:, 0],
        rectangle_coordinates[:, 1],
        color="#6b7280",
        linestyle="--",
        linewidth=1.2,
        label="original rectangle",
    )
    axes.fill(
        slot_coordinates[:, 0],
        slot_coordinates[:, 1],
        facecolor="#fecaca",
        edgecolor="#dc2626",
        linestyle=":",
        linewidth=1.7,
        alpha=0.65,
        label="expanded slot",
    )
    axes.plot(
        centerline_coordinates[:, 0],
        centerline_coordinates[:, 1],
        color="#991b1b",
        linestyle="-.",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label="construction line",
    )
    axes.plot(
        outline_coordinates[:, 0],
        outline_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.4,
    )

    axes.axhline(0.0, color="#111827", linewidth=1.0)
    axes.axvline(0.0, color="#6b7280", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_title(
        "Antenna outline step 2.1: upper first-order outer slot "
        f"({slot.bounds[2] - slot.bounds[0]:g} x "
        f"{slot.bounds[3] - slot.bounds[1]:g} mm)"
    )
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axes.legend(loc="lower left")

    min_x, min_y, max_x, max_y = rectangle.bounds
    margin = max(max_x - min_x, max_y - min_y) * 0.06
    axes.set_xlim(min_x - margin, max_x + margin)
    axes.set_ylim(min_y - margin, max_y + margin)

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_complete_antenna(
    parameters: AntennaOutlineParameters | None = None,
    *,
    save_path: Path | None = None,
    show: bool = True,
    figure: plt.Figure | None = None,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot every active construction family in the completed planar antenna."""

    parameters = _resolve_parameters(parameters)
    rectangle_points = generate_rectangle(parameters)
    rectangle, _ = validate_rectangle(rectangle_points, parameters)
    (
        _upper_order1_centerline,
        upper_order1_slot,
        _upper_order2_centerline,
        upper_order2_slot,
        _lower_order1_centerline,
        lower_order1_slot,
        _lower_order2_centerlines,
        lower_order2_slots,
        right_outer_slots,
        _right_outline,
        _lower_order2_report,
    ) = build_step_2_4_outline(rectangle, parameters)
    (
        _right_outer_slots,
        _left_outer_slots,
        _symmetric_outer_slots,
        patch,
        _symmetry_report,
    ) = build_step_2_5_outline(rectangle, right_outer_slots, parameters)
    (
        _inner_order1_centerline,
        inner_order1_slot,
        _inner_order2_centerline,
        inner_order2_slot,
        cpw_slot,
        _stub_centerlines,
        stub_slots,
        _slot_with_stubs,
        _right_combined_slot,
        symmetric_slot,
        _cpw_guide,
        symmetric_guide,
        result_patch,
        report,
    ) = build_step_3_4_geometry(patch, parameters)
    if result_patch is not patch:
        raise ValueError("complete plotting unexpectedly replaced the Patch")

    def mirrored_pair(polygon: Polygon) -> tuple[Polygon, Polygon]:
        mirrored = mirror_about_y_axis(polygon, parameters)
        if not isinstance(mirrored, Polygon):
            raise TypeError(f"expected mirrored Polygon, got {mirrored.geom_type}")
        return polygon, mirrored

    def fill_group(
        polygons: Sequence[Polygon],
        *,
        facecolor: str,
        edgecolor: str,
        label: str,
        zorder: float,
        hatch: str | None = None,
    ) -> None:
        for index, polygon in enumerate(polygons):
            coordinates = np.asarray(polygon.exterior.coords, dtype=float)
            axes.fill(
                coordinates[:, 0],
                coordinates[:, 1],
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.25,
                alpha=0.82,
                hatch=hatch,
                label=label if index == 0 else "_nolegend_",
                zorder=zorder,
            )

    if (figure is None) != (axes is None):
        raise ValueError("figure and axes must be supplied together")
    if figure is None or axes is None:
        figure, axes = plt.subplots(figsize=(12.5, 8.2))
    rectangle_coordinates = np.asarray(rectangle.exterior.coords, dtype=float)
    reflector_coordinates = np.asarray(
        generate_reflector_outline_points(parameters),
        dtype=float,
    )
    patch_coordinates = np.asarray(patch.exterior.coords, dtype=float)
    axes.fill(
        patch_coordinates[:, 0],
        patch_coordinates[:, 1],
        facecolor="#dbeafe",
        edgecolor="#1d4ed8",
        linewidth=2.1,
        alpha=0.86,
        label="Patch after symmetric outer slots",
        zorder=1,
    )
    axes.plot(
        rectangle_coordinates[:, 0],
        rectangle_coordinates[:, 1],
        color="#64748b",
        linestyle="--",
        linewidth=1.3,
        label="substrate outline",
        zorder=2,
    )
    axes.plot(
        reflector_coordinates[:, 0],
        reflector_coordinates[:, 1],
        color="#334155",
        linestyle="-.",
        linewidth=1.5,
        label="reflector outline with connector clearance",
        zorder=2.5,
    )

    fill_group(
        mirrored_pair(upper_order1_slot),
        facecolor="#ffedd5",
        edgecolor="#c2410c",
        label="upper outer slot, order 1",
        zorder=3,
    )
    fill_group(
        mirrored_pair(upper_order2_slot),
        facecolor="#fdba74",
        edgecolor="#9a3412",
        label="upper outer slot, order 2",
        zorder=4,
    )
    fill_group(
        mirrored_pair(lower_order1_slot),
        facecolor="#fee2e2",
        edgecolor="#dc2626",
        label="lower outer slot, order 1",
        zorder=3,
    )
    lower_order2_symmetric = tuple(
        polygon
        for right_polygon in lower_order2_slots
        for polygon in mirrored_pair(right_polygon)
    )
    fill_group(
        lower_order2_symmetric,
        facecolor="#fca5a5",
        edgecolor="#991b1b",
        label="lower outer slot, order 2",
        zorder=4,
    )
    fill_group(
        mirrored_pair(inner_order1_slot),
        facecolor="#f3e8ff",
        edgecolor="#9333ea",
        label="inner slot, order 1",
        zorder=5,
    )
    fill_group(
        mirrored_pair(inner_order2_slot),
        facecolor="#d8b4fe",
        edgecolor="#6b21a8",
        label="inner slot, order 2",
        zorder=6,
    )
    cpw_slot_and_stubs = (
        *mirrored_pair(cpw_slot),
        *(
            polygon
            for right_stub in stub_slots
            for polygon in mirrored_pair(right_stub)
        ),
    )
    fill_group(
        cpw_slot_and_stubs,
        facecolor="#f9a8d4",
        edgecolor="#be185d",
        label="CPW slot and matching stubs",
        zorder=7,
        hatch="//",
    )
    fill_group(
        (symmetric_guide,),
        facecolor="#99f6e4",
        edgecolor="#0f766e",
        label="symmetric CPW guide",
        zorder=8,
        hatch="xx",
    )

    reservations = generate_inner_slot_order2_reservations(parameters)
    reserved_anchors = [
        (x, reservation.anchor[1])
        for reservation in reservations
        for x in (reservation.anchor[0], -reservation.anchor[0])
    ]
    axes.scatter(
        [point[0] for point in reserved_anchors],
        [point[1] for point in reserved_anchors],
        marker="x",
        s=30,
        linewidths=1.4,
        color="#475569",
        label="reserved zero-size inner branches",
        zorder=10,
    )

    symmetric_slot_coordinates = np.asarray(symmetric_slot.exterior.coords, dtype=float)
    guide_coordinates = np.asarray(symmetric_guide.exterior.coords, dtype=float)
    axes.plot(
        symmetric_slot_coordinates[:, 0],
        symmetric_slot_coordinates[:, 1],
        color="#9d174d",
        linewidth=1.5,
        zorder=9,
    )
    axes.plot(
        guide_coordinates[:, 0],
        guide_coordinates[:, 1],
        color="#0f766e",
        linewidth=1.7,
        zorder=9,
    )
    axes.plot(
        patch_coordinates[:, 0],
        patch_coordinates[:, 1],
        color="#1d4ed8",
        linewidth=2.1,
        zorder=9,
    )
    axes.axhline(0.0, color="#0f172a", linewidth=0.9, zorder=0)
    axes.axvline(
        0.0,
        color="#64748b",
        linestyle=":",
        linewidth=1.0,
        zorder=0,
    )
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    axes.set_title(
        "Complete parameterized MSA-BP planar geometry\n"
        f"Patch={patch.area:g} mm², slot={report.symmetric_slot_area_mm2:g} mm², "
        f"guide={report.symmetric_guide_area_mm2:g} mm²"
    )
    axes.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)

    min_x, min_y, max_x, max_y = rectangle.bounds
    margin = max(max_x - min_x, max_y - min_y) * 0.055
    axes.set_xlim(min_x - margin, max_x + margin)
    axes.set_ylim(min_y - margin, max_y + margin)
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=8.5,
        frameon=True,
    )
    figure.tight_layout(rect=(0.015, 0.19, 0.985, 0.98))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def validate_explorer_parameters(parameters: AntennaOutlineParameters) -> None:
    """Run the quantized curve checks used before accepting a GUI adjustment."""

    generate_complete_antenna_point_lists(parameters)
    quantize_and_validate_closed_polygon_points(
        generate_rectangle(parameters),
        curve_name="substrate outline",
    )
    quantize_and_validate_closed_polygon_points(
        generate_reflector_outline_points(parameters),
        curve_name="reflector outline",
    )


def _build_explorer_logger(log_path: Path) -> logging.Logger:
    resolved_path = log_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(
        f"msabp.antenna_outline.explorer.{abs(hash(resolved_path))}"
    )
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(resolved_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


def launch_parameter_explorer(
    parameters: AntennaOutlineParameters | None = None,
    *,
    log_path: Path = DEFAULT_EXPLORER_LOG_PATH,
) -> AntennaOutlineParameters:
    """Launch the IDE-only group/parameter slider explorer."""

    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    reference_parameters = _resolve_parameters(parameters)
    parameter_groups = explorer_parameter_groups(reference_parameters)
    if not parameter_groups:
        raise RuntimeError("antenna explorer has no active adjustable parameters")
    logger = _build_explorer_logger(Path(log_path))

    root = tk.Tk()
    root.title("MSA-BP antenna parameter explorer")
    root.geometry("1420x940")
    root.minsize(1080, 720)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    controls = ttk.Frame(root, padding=(10, 8, 10, 6))
    controls.grid(row=0, column=0, sticky="ew")
    controls.columnconfigure(1, weight=1)
    controls.columnconfigure(3, weight=2)
    plot_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
    plot_frame.grid(row=1, column=0, sticky="nsew")

    first_group = next(iter(parameter_groups))
    group_variable = tk.StringVar(value=first_group)
    parameter_variable = tk.StringVar(value=parameter_groups[first_group][0])
    value_variable = tk.StringVar()
    status_variable = tk.StringVar(value="Ready")

    ttk.Label(controls, text="Group").grid(row=0, column=0, sticky="w", padx=(0, 5))
    group_selector = ttk.Combobox(
        controls,
        textvariable=group_variable,
        values=tuple(parameter_groups),
        state="readonly",
        width=34,
    )
    group_selector.grid(row=0, column=1, sticky="ew", padx=(0, 12))
    ttk.Label(controls, text="Variable").grid(
        row=0,
        column=2,
        sticky="w",
        padx=(0, 5),
    )
    parameter_selector = ttk.Combobox(
        controls,
        textvariable=parameter_variable,
        values=parameter_groups[first_group],
        state="readonly",
        width=54,
    )
    parameter_selector.grid(row=0, column=3, sticky="ew")

    value_label = ttk.Label(controls, textvariable=value_variable)
    value_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
    slider = tk.Scale(
        controls,
        orient=tk.HORIZONTAL,
        showvalue=False,
        highlightthickness=0,
        length=900,
    )
    slider.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(1, 0))
    status_label = ttk.Label(controls, textvariable=status_variable)
    status_label.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
    ttk.Label(
        controls,
        text=f"Invalid attempts: {Path(log_path).expanduser().resolve()}",
    ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(2, 0))

    current_parameters = reference_parameters
    current_canvas: FigureCanvasTkAgg | None = None
    current_figure: Figure | None = None
    pending_callback: str | None = None
    suppress_slider_callback = False
    last_logged_error: tuple[str, str, float, str] | None = None

    def slider_spec() -> ExplorerSliderSpec:
        return explorer_slider_spec(parameter_variable.get(), reference_parameters)

    def update_value_label(raw_slider_value: float) -> float:
        spec = slider_spec()
        parameter_value = spec.parameter_value(raw_slider_value)
        if spec.mode == "ratio":
            value_variable.set(
                f"{spec.parameter_name}: t={parameter_value:.2f}  "
                "(direct relative position, range 0..1)"
            )
        else:
            value_variable.set(
                f"{spec.parameter_name}: {raw_slider_value:.0f}% × "
                f"{spec.reference_value:g} mm = {parameter_value:g} mm"
            )
        return parameter_value

    def build_candidate_figure(
        candidate: AntennaOutlineParameters,
    ) -> Figure:
        validate_explorer_parameters(candidate)
        figure = Figure(figsize=(12.5, 8.2), dpi=100)
        axes = figure.add_subplot(111)
        plot_complete_antenna(
            candidate,
            show=False,
            figure=figure,
            axes=axes,
        )
        return figure

    def install_figure(figure: Figure) -> None:
        nonlocal current_canvas, current_figure
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        if current_canvas is not None:
            current_canvas.get_tk_widget().destroy()
        if current_figure is not None:
            current_figure.clear()
        widget.pack(fill=tk.BOTH, expand=True)
        current_canvas = canvas
        current_figure = figure

    def apply_slider_value(
        group_name: str,
        parameter_name: str,
        raw_slider_value: float,
    ) -> None:
        nonlocal current_parameters, last_logged_error, pending_callback
        pending_callback = None
        if (
            group_name != group_variable.get()
            or parameter_name != parameter_variable.get()
        ):
            return
        spec = explorer_slider_spec(parameter_name, reference_parameters)
        attempted_value = spec.parameter_value(raw_slider_value)
        candidate = replace(current_parameters, **{parameter_name: attempted_value})
        try:
            figure = build_candidate_figure(candidate)
            install_figure(figure)
        except Exception as exc:
            message = (
                "这组变量画不出来 | "
                f"group={group_name} | variable={parameter_name} | "
                f"slider={raw_slider_value:g} | attempted={attempted_value:g} | "
                f"error={type(exc).__name__}: {exc}"
            )
            error_key = (group_name, parameter_name, attempted_value, str(exc))
            if error_key != last_logged_error:
                logger.error(message)
                print(f"[antenna explorer] {message}", file=sys.stderr)
                last_logged_error = error_key
            status_variable.set(message)
            status_label.configure(foreground="#b91c1c")
            return
        current_parameters = candidate
        last_logged_error = None
        status_variable.set(
            f"Accepted | group={group_name} | variable={parameter_name} | "
            f"value={attempted_value:g}"
        )
        status_label.configure(foreground="#166534")

    def on_slider_change(raw_value: str) -> None:
        nonlocal pending_callback
        if suppress_slider_callback:
            return
        raw_slider_value = float(raw_value)
        update_value_label(raw_slider_value)
        if pending_callback is not None:
            root.after_cancel(pending_callback)
        pending_callback = root.after(
            140,
            apply_slider_value,
            group_variable.get(),
            parameter_variable.get(),
            raw_slider_value,
        )

    slider.configure(command=on_slider_change)

    def refresh_slider() -> None:
        nonlocal suppress_slider_callback, pending_callback
        if pending_callback is not None:
            root.after_cancel(pending_callback)
            pending_callback = None
        spec = slider_spec()
        current_value = float(getattr(current_parameters, spec.parameter_name))
        suppress_slider_callback = True
        try:
            slider.configure(
                from_=spec.minimum,
                to=spec.maximum,
                resolution=spec.resolution,
            )
            position = spec.slider_value(current_value)
            slider.set(position)
            update_value_label(position)
        finally:
            suppress_slider_callback = False

    def on_group_selected(_event: object | None = None) -> None:
        parameter_names = parameter_groups[group_variable.get()]
        parameter_selector.configure(values=parameter_names)
        parameter_variable.set(parameter_names[0])
        refresh_slider()

    def on_parameter_selected(_event: object | None = None) -> None:
        refresh_slider()

    group_selector.bind("<<ComboboxSelected>>", on_group_selected)
    parameter_selector.bind("<<ComboboxSelected>>", on_parameter_selected)

    try:
        install_figure(build_candidate_figure(current_parameters))
    except Exception as exc:
        logger.exception("initial antenna explorer geometry cannot be drawn")
        root.destroy()
        raise RuntimeError("initial antenna explorer geometry cannot be drawn") from exc
    refresh_slider()

    def close_explorer() -> None:
        nonlocal pending_callback
        if pending_callback is not None:
            root.after_cancel(pending_callback)
            pending_callback = None
        if current_figure is not None:
            current_figure.clear()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_explorer)
    root.mainloop()
    return current_parameters


def _running_from_ide_f5() -> bool:
    """Detect common IDE run/debug sessions without treating a terminal as F5."""

    if sys.gettrace() is not None:
        return True
    return any(
        os.environ.get(name)
        for name in ("PYCHARM_HOSTED", "SPYDER_KERNEL_ID", "IDLESTARTUP")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return the three completed antenna polygon point lists."
    )
    parser.add_argument(
        "--cpw-guide-p3-p4-x",
        type=float,
        default=DEFAULT_ANTENNA_PARAMETERS.cpw_guide_p3_p4_x_mm,
        help="Linked adjustable x coordinate for CPW-guide P3 and P4.",
    )
    parser.add_argument(
        "--cpw-guide-p4-y",
        type=float,
        default=DEFAULT_ANTENNA_PARAMETERS.cpw_guide_p4_y_mm,
        help="Adjustable y coordinate for CPW-guide P4.",
    )
    parser.add_argument(
        "--cpw-guide-p5-p6-x",
        type=float,
        default=DEFAULT_ANTENNA_PARAMETERS.cpw_guide_p5_p6_x_mm,
        help="Linked adjustable x coordinate for CPW-guide P5 and P6.",
    )
    parser.add_argument(
        "--cpw-slot-p1-p2-x",
        type=float,
        default=DEFAULT_ANTENNA_PARAMETERS.cpw_slot_p1_p2_x_mm,
        help="Linked adjustable x coordinate for CPW-slot P1 and P2.",
    )
    parser.add_argument(
        "--cpw-slot-p2-y",
        type=float,
        default=DEFAULT_ANTENNA_PARAMETERS.cpw_slot_p2_y_mm,
        help="Adjustable y coordinate for CPW-slot P2.",
    )
    parser.add_argument(
        "--cpw-slot-p3-p4-x",
        type=float,
        default=DEFAULT_ANTENNA_PARAMETERS.cpw_slot_p3_p4_x_mm,
        help="Linked adjustable x coordinate for CPW-slot P3 and P4.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    show_ide_plot: bool = False,
) -> list[list[Point2D]]:
    """Return point lists, optionally displaying the full F5-only IDE preview."""

    args = parse_args(argv)
    parameters = replace(
        DEFAULT_ANTENNA_PARAMETERS,
        cpw_guide_p3_p4_x_mm=args.cpw_guide_p3_p4_x,
        cpw_guide_p4_y_mm=args.cpw_guide_p4_y,
        cpw_guide_p5_p6_x_mm=args.cpw_guide_p5_p6_x,
        cpw_slot_p1_p2_x_mm=args.cpw_slot_p1_p2_x,
        cpw_slot_p2_y_mm=args.cpw_slot_p2_y,
        cpw_slot_p3_p4_x_mm=args.cpw_slot_p3_p4_x,
    )
    if show_ide_plot:
        parameters = launch_parameter_explorer(parameters)
    return generate_complete_antenna_point_lists(parameters)


if __name__ == "__main__":
    ide_f5 = _running_from_ide_f5()
    completed_points = main(show_ide_plot=ide_f5)
    if not ide_f5:
        pprint(completed_points, sort_dicts=False, width=120)
