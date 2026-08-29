import argparse

from cst_generate_polygen import DEFAULT_PROJECT_PATH, execute_save_project, open_cst_project


DEFAULT_COMPONENT_NAME = "component1"

IMPORTED_SOLIDS = [
    "substrate_solid",
    "reflector_ground_solid",
    "reflector_ground_cutout_solid",
    "outer_rectangle_solid",
    "outer_third_order_minkowski_frame_solid",
    "feed_rectangle_solid",
    "feed_pin_solid",
    "outer_second_order_minkowski_solid",
    "inner_second_order_minkowski_solid",
]

BOOLEAN_RESULT_SOLIDS = [
    "substrate_solid",
    "reflector_ground_solid",
    "outer_rectangle_solid",
    "feed_pin_solid",
    "outer_second_order_minkowski_solid",
]

DELETE_PRESETS = {
    "imported": IMPORTED_SOLIDS,
    "boolean": BOOLEAN_RESULT_SOLIDS,
}


def _solid_ref(component_name, solid_name):
    return f"{component_name}:{solid_name}"


def build_delete_vba(solid_name, component_name=DEFAULT_COMPONENT_NAME):
    return f"""
Sub Main()
    On Error Resume Next
    Solid.Delete "{_solid_ref(component_name, solid_name)}"
    On Error GoTo 0
End Sub
"""


def execute_deletes(
    project,
    solid_names,
    component_name=DEFAULT_COMPONENT_NAME,
    save_project=True,
):
    for solid_name in solid_names:
        print(f"正在删除实体: {component_name}:{solid_name}")
        project.schematic.execute_vba_code(build_delete_vba(solid_name, component_name=component_name))

    if save_project:
        execute_save_project(project)
        print("CST 工程已保存。")


def run_delete_solids(
    solid_names,
    project_path=DEFAULT_PROJECT_PATH,
    component_name=DEFAULT_COMPONENT_NAME,
    save_project=True,
    dry_run=False,
):
    if dry_run:
        for solid_name in solid_names:
            print(build_delete_vba(solid_name, component_name=component_name).strip())
        return

    project = open_cst_project(project_path)
    execute_deletes(
        project=project,
        solid_names=solid_names,
        component_name=component_name,
        save_project=save_project,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="删除 CST 中用于调试的分形天线实体。")
    parser.add_argument(
        "--preset",
        choices=sorted(DELETE_PRESETS),
        default="imported",
        help="imported 删除原始导入实体；boolean 删除布尔后剩余实体。",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH, help="目标 CST 工程路径。")
    parser.add_argument("--component", default=DEFAULT_COMPONENT_NAME, help="CST component 名称。")
    parser.add_argument("--solid", action="append", help="额外指定要删除的 solid 名称，可重复传入。")
    parser.add_argument("--no-save", action="store_true", help="删除后不自动保存 CST 工程。")
    parser.add_argument("--dry-run", action="store_true", help="只打印 VBA，不打开 CST。")
    return parser.parse_args()


def main():
    args = parse_args()
    solid_names = list(DELETE_PRESETS[args.preset])
    if args.solid:
        solid_names.extend(args.solid)

    run_delete_solids(
        solid_names=solid_names,
        project_path=args.project,
        component_name=args.component,
        save_project=not args.no_save,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
