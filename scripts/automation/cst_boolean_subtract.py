import argparse

from cst_generate_polygen import DEFAULT_PROJECT_PATH, execute_save_project, open_cst_project


DEFAULT_COMPONENT_NAME = "component1"
DEFAULT_SUBTRACTIONS = [
    ("reflector_ground_solid", "reflector_ground_cutout_solid"),
    ("outer_rectangle_solid", "outer_third_order_minkowski_frame_solid"),
    ("outer_rectangle_solid", "feed_rectangle_solid"),
    ("outer_second_order_minkowski_solid", "inner_second_order_minkowski_solid"),
]


def _solid_ref(component_name, solid_name):
    return f"{component_name}:{solid_name}"


def build_subtract_vba(target_solid, tool_solid, component_name=DEFAULT_COMPONENT_NAME):
    target_ref = _solid_ref(component_name, target_solid)
    tool_ref = _solid_ref(component_name, tool_solid)
    return f"""
Sub Main()
    ' performs the boolean operation {target_solid} = {target_solid} - {tool_solid}
    Solid.Subtract "{target_ref}", "{tool_ref}"
End Sub
"""


def execute_subtractions(
    project,
    subtractions=DEFAULT_SUBTRACTIONS,
    component_name=DEFAULT_COMPONENT_NAME,
    save_project=True,
    timeout=None,
):
    for target_solid, tool_solid in subtractions:
        print(f"正在执行布尔差集: {target_solid} = {target_solid} - {tool_solid}")
        project.schematic.execute_vba_code(
            build_subtract_vba(target_solid, tool_solid, component_name=component_name),
            timeout=timeout,
        )

    if save_project:
        execute_save_project(project, timeout=timeout)
        print("CST 工程已保存。")


def run_boolean_subtractions(
    project_path=DEFAULT_PROJECT_PATH,
    component_name=DEFAULT_COMPONENT_NAME,
    subtractions=DEFAULT_SUBTRACTIONS,
    save_project=True,
    dry_run=False,
):
    if dry_run:
        for target_solid, tool_solid in subtractions:
            print(build_subtract_vba(target_solid, tool_solid, component_name=component_name).strip())
        return

    project = open_cst_project(project_path)
    execute_subtractions(
        project=project,
        subtractions=subtractions,
        component_name=component_name,
        save_project=save_project,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="对已导入 CST 的分形天线实体执行布尔差集。")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH, help="目标 CST 工程路径。")
    parser.add_argument("--component", default=DEFAULT_COMPONENT_NAME, help="CST component 名称。")
    parser.add_argument("--no-save", action="store_true", help="执行布尔操作后不自动保存 CST 工程。")
    parser.add_argument("--dry-run", action="store_true", help="只打印 VBA，不打开 CST。")
    return parser.parse_args()


def main():
    args = parse_args()
    run_boolean_subtractions(
        project_path=args.project,
        component_name=args.component,
        save_project=not args.no_save,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
