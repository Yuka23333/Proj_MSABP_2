import argparse
import json
from pathlib import Path

from cst_generate_polygen import (
    DEFAULT_PROJECT_PATH,
    create_brick,
    create_extruded_polygon,
    execute_save_project,
    open_cst_project,
)


CURVES_PATH = Path(__file__).with_name("fractal_antenna_curves.json")
DEFAULT_COMPONENT_NAME = "component1"
DEFAULT_MATERIAL_NAME = "Copper (annealed)"
DEFAULT_THICKNESS = 0.035
SUBSTRATE_CURVE_NAME = "outer_rectangle"
SUBSTRATE_SOLID_NAME = "substrate_solid"
SUBSTRATE_MATERIAL_NAME = "FR-4 (lossy)"
SUBSTRATE_THICKNESS = -1.0
REFLECTOR_SOLID_NAME = "reflector_ground_solid"
REFLECTOR_MATERIAL_NAME = "Copper (annealed)"
REFLECTOR_CUTOUT_SOLID_NAME = "reflector_ground_cutout_solid"
REFLECTOR_CUTOUT_X_RANGE = (-8.0, 8.0)
REFLECTOR_CUTOUT_HEIGHT = 0.5


def substrate_extrusion_thickness(value):
    """将正的物理厚度或旧式负厚度统一为沿 -z 的 CST 拉伸厚度。"""
    physical_thickness = abs(float(value))
    if physical_thickness <= 0:
        raise ValueError("基板厚度必须大于 0。")
    return -physical_thickness


def _load_curve_data(curves_path):
    with curves_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("format") != "fractal_antenna_curves_v1":
        raise ValueError(f"不支持的曲线文件格式: {data.get('format')!r}")

    curves = data.get("curves")
    if not isinstance(curves, list) or not curves:
        raise ValueError("曲线文件中没有可导入的 curves 列表。")

    return data


def _safe_cst_name(name):
    safe_chars = []
    for char in name:
        safe_chars.append(char if char.isalnum() or char == "_" else "_")
    return "".join(safe_chars).strip("_")


def _path_to_points(path):
    if len(path) < 4:
        raise ValueError(f"闭合多边形路径至少需要 4 个点，当前只有 {len(path)} 个。")

    return [(float(point[0]), float(point[1])) for point in path]


def polygon_signed_area(points):
    """返回多边形有向面积：正值为逆时针，负值为顺时针。"""
    return 0.5 * sum(
        x_current * y_next - x_next * y_current
        for (x_current, y_current), (x_next, y_next) in zip(points, points[1:] + points[:1])
    )


def ensure_counterclockwise(points):
    """保持闭合路径起点不变，并在需要时将点序反转为逆时针。"""
    normalized_points = list(points)
    signed_area = polygon_signed_area(normalized_points)
    if abs(signed_area) <= 1e-12:
        raise ValueError("多边形有向面积为 0，无法确定点序方向。")
    if signed_area < 0:
        if normalized_points[0] == normalized_points[-1]:
            normalized_points[1:-1] = reversed(normalized_points[1:-1])
        else:
            normalized_points.reverse()
    return normalized_points


def _find_substrate_bottom_y(curve_data):
    for curve in curve_data["curves"]:
        if curve.get("name") != SUBSTRATE_CURVE_NAME:
            continue

        y_values = [
            float(point[1])
            for path in curve.get("paths", [])
            for point in path
        ]
        if not y_values:
            raise ValueError(f"{SUBSTRATE_CURVE_NAME} 没有可用的 y 坐标。")
        return min(y_values)

    raise ValueError(f"曲线文件中找不到 substrate 曲线: {SUBSTRATE_CURVE_NAME}")


def _offset_points(points, y_offset):
    return [(x_value, y_value + y_offset) for x_value, y_value in points]


def _points_bounds(points):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def iter_curve_paths(curve_data):
    substrate_bottom_y = _find_substrate_bottom_y(curve_data)
    y_offset = -substrate_bottom_y

    for curve in curve_data["curves"]:
        curve_name = _safe_cst_name(curve["name"])
        paths = curve.get("paths", [])
        if not paths:
            raise ValueError(f"{curve_name} 没有 paths。")

        for path_index, path in enumerate(paths, start=1):
            object_base_name = curve_name if len(paths) == 1 else f"{curve_name}_{path_index}"
            points = ensure_counterclockwise(_path_to_points(path))
            yield {
                "label": curve.get("label", curve_name),
                "object_base_name": object_base_name,
                "points": _offset_points(points, y_offset),
                "y_offset": y_offset,
            }


def _find_curve_path(curve_paths, object_base_name):
    for curve_path in curve_paths:
        if curve_path["object_base_name"] == object_base_name:
            return curve_path
    raise ValueError(f"找不到曲线路径: {object_base_name}")


def reflector_z_range(substrate_thickness=SUBSTRATE_THICKNESS, reflector_thickness=DEFAULT_THICKNESS):
    z_max = -abs(substrate_thickness)
    z_min = z_max - abs(reflector_thickness)
    return z_min, z_max


def reflector_cutout_ranges(
    substrate_points,
    substrate_thickness=SUBSTRATE_THICKNESS,
    reflector_thickness=DEFAULT_THICKNESS,
    x_range=REFLECTOR_CUTOUT_X_RANGE,
    height=REFLECTOR_CUTOUT_HEIGHT,
):
    if height <= 0:
        raise ValueError("反射底板切口的 y 向高度必须大于 0。")

    _, y_min, _, _ = _points_bounds(substrate_points)
    return (
        tuple(x_range),
        (y_min, y_min + height),
        reflector_z_range(substrate_thickness, reflector_thickness),
    )


def create_reflector_ground(
    project,
    substrate_points,
    component_name=DEFAULT_COMPONENT_NAME,
    material_name=REFLECTOR_MATERIAL_NAME,
    substrate_thickness=SUBSTRATE_THICKNESS,
    reflector_thickness=DEFAULT_THICKNESS,
    solid_name=REFLECTOR_SOLID_NAME,
    cutout_solid_name=REFLECTOR_CUTOUT_SOLID_NAME,
    cutout_x_range=REFLECTOR_CUTOUT_X_RANGE,
    cutout_height=REFLECTOR_CUTOUT_HEIGHT,
    save_project=False,
    timeout=None,
):
    x_min, y_min, x_max, y_max = _points_bounds(substrate_points)
    cutout_x_range, cutout_y_range, z_range = reflector_cutout_ranges(
        substrate_points,
        substrate_thickness=substrate_thickness,
        reflector_thickness=reflector_thickness,
        x_range=cutout_x_range,
        height=cutout_height,
    )
    create_brick(
        project=project,
        solid_name=solid_name,
        component_name=component_name,
        material_name=material_name,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        z_range=z_range,
        save_project=False,
        timeout=timeout,
    )
    create_brick(
        project=project,
        solid_name=cutout_solid_name,
        component_name=component_name,
        material_name=material_name,
        x_range=cutout_x_range,
        y_range=cutout_y_range,
        z_range=z_range,
        save_project=save_project,
        timeout=timeout,
    )


def import_fractal_antenna_curves(
    curves_path=CURVES_PATH,
    project_path=DEFAULT_PROJECT_PATH,
    component_name=DEFAULT_COMPONENT_NAME,
    material_name=DEFAULT_MATERIAL_NAME,
    thickness=DEFAULT_THICKNESS,
    substrate_material_name=SUBSTRATE_MATERIAL_NAME,
    substrate_thickness=None,
    reflector_material_name=REFLECTOR_MATERIAL_NAME,
    reflector_thickness=DEFAULT_THICKNESS,
    save_project=True,
    dry_run=False,
):
    curve_data = _load_curve_data(Path(curves_path))
    if substrate_thickness is None:
        substrate_thickness = curve_data.get(
            "substrate_thickness_mm",
            SUBSTRATE_THICKNESS,
        )
    substrate_thickness = substrate_extrusion_thickness(substrate_thickness)
    curve_paths = list(iter_curve_paths(curve_data))
    substrate_curve_path = _find_curve_path(curve_paths, SUBSTRATE_CURVE_NAME)

    y_offset = curve_paths[0]["y_offset"] if curve_paths else 0.0
    print(f"准备导入 {len(curve_paths)} 条 CST 多边形路径。")
    print(f"已将 substrate 底边平移到 y=0，所有点 y 偏置: {y_offset:g}")
    if dry_run:
        for curve_path in curve_paths:
            object_base_name = curve_path["object_base_name"]
            print(
                f"[dry-run] {object_base_name}: {len(curve_path['points'])} points, "
                f"winding=CCW, thickness={thickness:g}"
            )
        print(
            f"[dry-run] {SUBSTRATE_SOLID_NAME}: {len(substrate_curve_path['points'])} points, "
            f"material={substrate_material_name}, thickness={substrate_thickness:g}"
        )
        z_min, z_max = reflector_z_range(substrate_thickness, reflector_thickness)
        print(
            f"[dry-run] {REFLECTOR_SOLID_NAME}: material={reflector_material_name}, "
            f"z=({z_min:g}, {z_max:g})"
        )
        cutout_x_range, cutout_y_range, cutout_z_range = reflector_cutout_ranges(
            substrate_curve_path["points"],
            substrate_thickness=substrate_thickness,
            reflector_thickness=reflector_thickness,
        )
        print(
            f"[dry-run] {REFLECTOR_CUTOUT_SOLID_NAME}: material={reflector_material_name}, "
            f"x=({cutout_x_range[0]:g}, {cutout_x_range[1]:g}), "
            f"y=({cutout_y_range[0]:g}, {cutout_y_range[1]:g}), "
            f"z=({cutout_z_range[0]:g}, {cutout_z_range[1]:g})"
        )
        return

    project = open_cst_project(project_path)
    create_extruded_polygon(
        project=project,
        points=substrate_curve_path["points"],
        polygon_name="substrate_polygon",
        curve_name="substrate_curve",
        solid_name=SUBSTRATE_SOLID_NAME,
        component_name=component_name,
        material_name=substrate_material_name,
        thickness=substrate_thickness,
        save_project=False,
    )
    create_reflector_ground(
        project=project,
        substrate_points=substrate_curve_path["points"],
        component_name=component_name,
        material_name=reflector_material_name,
        substrate_thickness=substrate_thickness,
        reflector_thickness=reflector_thickness,
        save_project=False,
    )

    for curve_path in curve_paths:
        object_base_name = curve_path["object_base_name"]
        create_extruded_polygon(
            project=project,
            points=curve_path["points"],
            polygon_name=f"{object_base_name}_polygon",
            curve_name=f"{object_base_name}_curve",
            solid_name=f"{object_base_name}_solid",
            component_name=component_name,
            material_name=material_name,
            thickness=thickness,
            save_project=False,
        )

    if save_project:
        execute_save_project(project)
        print("CST 工程已保存。")


def parse_args():
    parser = argparse.ArgumentParser(description="将分形天线 JSON 顶点导入 CST 并拉伸成 1oz 铜厚度。")
    parser.add_argument("--curves", default=str(CURVES_PATH), help="由 Fractal_Antenna.py 导出的曲线 JSON。")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH, help="目标 CST 工程路径。")
    parser.add_argument("--component", default=DEFAULT_COMPONENT_NAME, help="CST component 名称。")
    parser.add_argument("--material", default=DEFAULT_MATERIAL_NAME, help="CST 材料名称。")
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS, help="拉伸厚度，默认 0.035 mm。")
    parser.add_argument("--substrate-material", default=SUBSTRATE_MATERIAL_NAME, help="基板材料名称。")
    parser.add_argument(
        "--substrate-thickness",
        type=float,
        default=None,
        help="覆盖 JSON 中的基板物理厚度；正负输入都会沿 -z 拉伸。",
    )
    parser.add_argument("--reflector-material", default=REFLECTOR_MATERIAL_NAME, help="反射底板材料名称。")
    parser.add_argument("--reflector-thickness", type=float, default=DEFAULT_THICKNESS, help="反射底板厚度，默认等于铜厚 0.035 mm。")
    parser.add_argument("--no-save", action="store_true", help="导入后不自动保存 CST 工程。")
    parser.add_argument("--dry-run", action="store_true", help="只验证并打印将要导入的路径，不打开 CST。")
    return parser.parse_args()


def main():
    args = parse_args()
    import_fractal_antenna_curves(
        curves_path=args.curves,
        project_path=args.project,
        component_name=args.component,
        material_name=args.material,
        thickness=args.thickness,
        substrate_material_name=args.substrate_material,
        substrate_thickness=args.substrate_thickness,
        reflector_material_name=args.reflector_material,
        reflector_thickness=args.reflector_thickness,
        save_project=not args.no_save,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
