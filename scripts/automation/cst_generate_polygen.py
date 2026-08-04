import os


DEFAULT_PROJECT_PATH = r"D:\Academic\Proj_CFSA\cst_test_proj\test.cst"

DEFAULT_POLYGON_POINTS = [
    (1.0, 1.0),
    (2.0, 1.0),
    (2.0, 2.0),
    (1.0, 2.0),
]


def _format_cst_number(value):
    return f"{value:g}"


def _validate_polygon_points(points):
    if len(points) < 3:
        raise ValueError("CST Polygon 至少需要 3 个点。")

    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"第 {index} 个点不是二维坐标: {point!r}")


def _closed_points(points):
    _validate_polygon_points(points)

    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    return closed


def build_polygon_vba(points, polygon_name, curve_name):
    closed_points = _closed_points(points)
    first_x, first_y = closed_points[0]
    line_commands = [
        f'        .Point "{_format_cst_number(first_x)}", "{_format_cst_number(first_y)}"'
    ]

    for x_value, y_value in closed_points[1:]:
        line_commands.append(
            f'        .LineTo "{_format_cst_number(x_value)}", "{_format_cst_number(y_value)}"'
        )

    point_vba = "\n".join(line_commands)
    return f"""
Sub Main()
    With Polygon
        .Reset
        .Name "{polygon_name}"
        .Curve "{curve_name}"
{point_vba}
        .Create
    End With
End Sub
"""


def build_extrude_curve_vba(
    solid_name,
    component_name,
    material_name,
    thickness,
    curve_name,
    polygon_name,
):
    # .Thickness 遵循右手定则：在标准 XoY 平面中，逆时针点序会向上拉伸，
    # 顺时针点序会向下拉伸。这里不自动修改点序，避免改变建模方向。
    return f"""
Sub Main()
    With ExtrudeCurve
        .Reset
        .Name "{solid_name}"
        .Component "{component_name}"
        .Material "{material_name}"
        .Thickness "{_format_cst_number(thickness)}"
        .Twistangle "0.0"
        .Taperangle "0.0"
        .Curve "{curve_name}:{polygon_name}"
        .Create
    End With
End Sub
"""


def build_save_vba():
    return """
Sub Main()
    Save
End Sub
"""


def _ordered_range(first_value, second_value):
    return min(first_value, second_value), max(first_value, second_value)


def build_brick_vba(
    solid_name,
    component_name,
    material_name,
    x_range,
    y_range,
    z_range,
):
    x_min, x_max = _ordered_range(*x_range)
    y_min, y_max = _ordered_range(*y_range)
    z_min, z_max = _ordered_range(*z_range)
    return f"""
Sub Main()
    With Brick
        .Reset
        .Name "{solid_name}"
        .Component "{component_name}"
        .Material "{material_name}"
        .Xrange "{_format_cst_number(x_min)}", "{_format_cst_number(x_max)}"
        .Yrange "{_format_cst_number(y_min)}", "{_format_cst_number(y_max)}"
        .Zrange "{_format_cst_number(z_min)}", "{_format_cst_number(z_max)}"
        .Create
    End With
End Sub
"""


def open_cst_project(project_path=DEFAULT_PROJECT_PATH):
    import cst.interface

    target_path = os.path.normcase(os.path.abspath(os.fspath(project_path)))
    for process_id in cst.interface.running_design_environments():
        try:
            environment = cst.interface.DesignEnvironment.connect(process_id)
        except RuntimeError:
            continue
        for open_path in environment.list_open_projects():
            if os.path.normcase(os.path.abspath(open_path)) == target_path:
                return environment.get_open_project(open_path)

    environment = cst.interface.DesignEnvironment.connect_to_any_or_new()
    return environment.open_project(target_path)


def _vba_main_body(vba_code):
    """Return a VBA snippet suitable for Model3D.add_to_history."""

    lines = vba_code.strip().splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().lower() == "sub main()"
        and lines[-1].strip().lower() == "end sub"
    ):
        return "\n".join(lines[1:-1])
    return vba_code


def execute_project_vba(project, label, vba_code, timeout=None):
    """Execute 3D VBA through the interface exposed by the connected project."""

    schematic = project.schematic
    if schematic is not None:
        schematic.execute_vba_code(vba_code, timeout=timeout)
        return

    model3d = project.model3d
    if model3d is not None:
        model3d.add_to_history(
            label,
            _vba_main_body(vba_code),
            timeout=timeout,
        )
        return

    raise RuntimeError("connected CST project exposes neither Model3D nor Schematic")


def execute_save_project(project, timeout=None):
    try:
        project.save()
    except RuntimeError:
        # Project.save() can fail for a project attached through an existing
        # DesignEnvironment even though the project itself is writable.  CST's
        # native VBA Save command is the proven fallback used by the migrated
        # automation scripts.
        execute_project_vba(
            project,
            "save project (VBA fallback)",
            "Sub Main()\nSave\nEnd Sub",
            timeout=timeout,
        )


def create_extruded_polygon(
    project,
    points,
    polygon_name="polygon1",
    curve_name="curve1",
    solid_name="solid2",
    component_name="component1",
    material_name="Vacuum",
    thickness=2.0,
    save_project=True,
    timeout=None,
):
    polygon_vba = build_polygon_vba(points, polygon_name, curve_name)
    extrude_vba = build_extrude_curve_vba(
        solid_name=solid_name,
        component_name=component_name,
        material_name=material_name,
        thickness=thickness,
        curve_name=curve_name,
        polygon_name=polygon_name,
    )

    print("正在向 CST 发送 VBA 指令...")
    execute_project_vba(project, f"create curve {curve_name}", polygon_vba, timeout)
    execute_project_vba(project, f"extrude solid {solid_name}", extrude_vba, timeout)
    if save_project:
        execute_save_project(project, timeout=timeout)
    print(f"多边形拉伸体 {solid_name} 建立完成。")


def create_brick(
    project,
    solid_name="brick1",
    component_name="component1",
    material_name="PEC",
    x_range=(0.0, 1.0),
    y_range=(0.0, 1.0),
    z_range=(0.0, 1.0),
    save_project=True,
    timeout=None,
):
    brick_vba = build_brick_vba(
        solid_name=solid_name,
        component_name=component_name,
        material_name=material_name,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    )

    print("正在向 CST 发送 Brick VBA 指令...")
    execute_project_vba(project, f"create brick {solid_name}", brick_vba, timeout)
    if save_project:
        execute_save_project(project, timeout=timeout)
    print(f"长方体 {solid_name} 建立完成。")


def create_polygonal_prism(
    points=None,
    project_path=DEFAULT_PROJECT_PATH,
    polygon_name="polygon1",
    curve_name="curve1",
    solid_name="solid2",
    component_name="component1",
    material_name="Vacuum",
    thickness=2.0,
    save_project=True,
):
    if points is None:
        points = DEFAULT_POLYGON_POINTS

    project = open_cst_project(project_path)
    create_extruded_polygon(
        project=project,
        points=points,
        polygon_name=polygon_name,
        curve_name=curve_name,
        solid_name=solid_name,
        component_name=component_name,
        material_name=material_name,
        thickness=thickness,
        save_project=save_project,
    )


if __name__ == "__main__":
    create_polygonal_prism()
