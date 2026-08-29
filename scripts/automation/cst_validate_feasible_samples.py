import argparse
import csv
import json
import traceback
from pathlib import Path

import Fractal_Antenna as antenna
from cfsa_geometry.model import curve_data_from_geometry
from cfsa_geometry.parameters import (
    PARAMETER_PATHS,
    config_from_parameter_values,
    fixed_parameter_values,
)
from cst_boolean_subtract import DEFAULT_SUBTRACTIONS, execute_subtractions
from cst_delete_fractal_solids import BOOLEAN_RESULT_SOLIDS, IMPORTED_SOLIDS
from cst_generate_polygen import DEFAULT_PROJECT_PATH, create_extruded_polygon, execute_save_project, open_cst_project
from cst_import_fractal_antenna import (
    DEFAULT_COMPONENT_NAME,
    DEFAULT_MATERIAL_NAME,
    DEFAULT_THICKNESS,
    REFLECTOR_MATERIAL_NAME,
    SUBSTRATE_CURVE_NAME,
    SUBSTRATE_MATERIAL_NAME,
    SUBSTRATE_SOLID_NAME,
    SUBSTRATE_THICKNESS,
    create_reflector_ground,
    iter_curve_paths,
    substrate_extrusion_thickness,
)


DEFAULT_SAMPLES_CSV = Path(__file__).with_name("scan_latin_1024.csv")
DEFAULT_FAILURES_CSV = Path(__file__).with_name("cst_feasible_sample_failures.csv")
DEFAULT_RESULTS_CSV = Path(__file__).with_name("cst_feasible_sample_results.csv")
DEFAULT_TEMP_CURVES_DIR = Path(__file__).with_name("_cst_feasible_sample_curves")
DEFAULT_LABEL_COLUMN = "geometry_feasible"

DELETE_SOLIDS = list(dict.fromkeys([*BOOLEAN_RESULT_SOLIDS, *IMPORTED_SOLIDS]))
FIXED_PARAMETER_VALUES = fixed_parameter_values(antenna.DEFAULT_CONFIG)
LEGACY_PARAMETER_DEFAULTS = {
    "substrate_thickness": antenna.DEFAULT_SUBSTRATE_THICKNESS_MM,
}


def _parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", ""}:
        return False
    try:
        return float(normalized) != 0.0
    except ValueError as exc:
        raise ValueError(f"无法识别可行性标签: {value!r}") from exc


def _curve_data_from_config(config):
    return curve_data_from_geometry(antenna.build_antenna_geometry(config))


def _find_curve_path(curve_paths, object_base_name):
    for curve_path in curve_paths:
        if curve_path["object_base_name"] == object_base_name:
            return curve_path
    raise ValueError(f"找不到曲线路径: {object_base_name}")


def _sample_values_from_row(row):
    for path, expected in FIXED_PARAMETER_VALUES.items():
        if path in row and abs(float(row[path]) - expected) > 1e-12:
            raise ValueError(
                f"参数 {path} 已固定为 {expected:g}，当前 CSV 值为 {float(row[path]):g}。"
            )
    missing_columns = [
        path
        for path in PARAMETER_PATHS
        if path not in row and path not in LEGACY_PARAMETER_DEFAULTS
    ]
    if missing_columns:
        preview = ", ".join(missing_columns[:5])
        suffix = "..." if len(missing_columns) > 5 else ""
        raise ValueError(f"样本 CSV 缺少参数列: {preview}{suffix}")
    return [
        float(row[path]) if path in row else float(LEGACY_PARAMETER_DEFAULTS[path])
        for path in PARAMETER_PATHS
    ]


def _config_from_sample(values):
    return config_from_parameter_values(antenna.DEFAULT_CONFIG, values)


def iter_feasible_samples(samples_csv, label_column=DEFAULT_LABEL_COLUMN):
    with Path(samples_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("样本 CSV 没有表头。")
        if label_column not in reader.fieldnames:
            raise ValueError(f"样本 CSV 缺少可行性标签列: {label_column}")

        feasible_index = 0
        for csv_row_number, row in enumerate(reader, start=1):
            if not _parse_bool(row[label_column]):
                continue

            feasible_index += 1
            yield {
                "csv_row_number": csv_row_number,
                "sample_index": csv_row_number - 1,
                "feasible_index": feasible_index,
                "values": _sample_values_from_row(row),
            }


def _write_curve_data(curve_data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(curve_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_delete_vba(solid_names, component_name=DEFAULT_COMPONENT_NAME):
    lines = ["Sub Main", "On Error Resume Next"]
    for solid_name in solid_names:
        lines.append(f'Solid.Delete "{component_name}:{solid_name}"')
    lines.extend(["On Error GoTo 0", "End Sub"])
    return "\n".join(lines)


def delete_existing_fractal_solids(project, component_name=DEFAULT_COMPONENT_NAME, timeout=None):
    project.schematic.execute_vba_code(
        _build_delete_vba(DELETE_SOLIDS, component_name=component_name),
        timeout=timeout,
    )


def import_curve_data_to_cst(
    project,
    curve_data,
    component_name=DEFAULT_COMPONENT_NAME,
    material_name=DEFAULT_MATERIAL_NAME,
    thickness=DEFAULT_THICKNESS,
    substrate_material_name=SUBSTRATE_MATERIAL_NAME,
    substrate_thickness=SUBSTRATE_THICKNESS,
    reflector_material_name=REFLECTOR_MATERIAL_NAME,
    reflector_thickness=DEFAULT_THICKNESS,
    timeout=None,
):
    substrate_thickness = substrate_extrusion_thickness(substrate_thickness)
    curve_paths = list(iter_curve_paths(curve_data))
    substrate_curve_path = _find_curve_path(curve_paths, SUBSTRATE_CURVE_NAME)

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
        timeout=timeout,
    )
    create_reflector_ground(
        project=project,
        substrate_points=substrate_curve_path["points"],
        component_name=component_name,
        material_name=reflector_material_name,
        substrate_thickness=substrate_thickness,
        reflector_thickness=reflector_thickness,
        save_project=False,
        timeout=timeout,
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
            timeout=timeout,
        )


def validate_one_sample(
    project,
    sample,
    component_name=DEFAULT_COMPONENT_NAME,
    save_each=False,
    write_curve_json=False,
    temp_curves_dir=DEFAULT_TEMP_CURVES_DIR,
):
    delete_existing_fractal_solids(project, component_name=component_name)

    config = _config_from_sample(sample["values"])
    curve_data = _curve_data_from_config(config)

    if write_curve_json:
        curve_path = Path(temp_curves_dir) / f"sample_{sample['sample_index']:04d}.json"
        _write_curve_data(curve_data, curve_path)

    import_curve_data_to_cst(
        project=project,
        curve_data=curve_data,
        component_name=component_name,
        substrate_thickness=config.substrate_thickness,
    )
    execute_subtractions(
        project=project,
        subtractions=DEFAULT_SUBTRACTIONS,
        component_name=component_name,
        save_project=False,
    )

    if save_each:
        execute_save_project(project)


def _empty_failure_record(sample, stage, exc):
    return {
        "feasible_index": sample["feasible_index"],
        "csv_row_number": sample["csv_row_number"],
        "sample_index": sample["sample_index"],
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _result_record(sample, status, stage="", error_message=""):
    return {
        "feasible_index": sample["feasible_index"],
        "csv_row_number": sample["csv_row_number"],
        "sample_index": sample["sample_index"],
        "status": status,
        "stage": stage,
        "error_message": error_message,
    }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(
    samples_csv=DEFAULT_SAMPLES_CSV,
    project_path=DEFAULT_PROJECT_PATH,
    component_name=DEFAULT_COMPONENT_NAME,
    label_column=DEFAULT_LABEL_COLUMN,
    failures_csv=DEFAULT_FAILURES_CSV,
    results_csv=DEFAULT_RESULTS_CSV,
    start_feasible_index=1,
    limit=None,
    dry_run=False,
    save_each=False,
    save_last=False,
    write_curve_json=False,
    temp_curves_dir=DEFAULT_TEMP_CURVES_DIR,
):
    feasible_samples = [
        sample
        for sample in iter_feasible_samples(samples_csv, label_column=label_column)
        if sample["feasible_index"] >= start_feasible_index
    ]
    if limit is not None:
        feasible_samples = feasible_samples[:limit]

    print(f"可行样本数: {len(feasible_samples)}")
    print(f"样本 CSV: {samples_csv}")
    print(f"CST 工程: {project_path}")

    failures = []
    results = []

    if dry_run:
        for sample in feasible_samples:
            try:
                config = _config_from_sample(sample["values"])
                curve_data = _curve_data_from_config(config)
                if write_curve_json:
                    curve_path = Path(temp_curves_dir) / f"sample_{sample['sample_index']:04d}.json"
                    _write_curve_data(curve_data, curve_path)
                results.append(_result_record(sample, "dry_run_ok"))
            except Exception as exc:
                failures.append(_empty_failure_record(sample, "build_geometry", exc))
                results.append(_result_record(sample, "failed", "build_geometry", str(exc)))
        project = None
    else:
        project = open_cst_project(project_path)
        for sample in feasible_samples:
            print(
                f"[{sample['feasible_index']}/{feasible_samples[-1]['feasible_index']}] "
                f"CSV row={sample['csv_row_number']}, sample_index={sample['sample_index']}"
            )
            try:
                validate_one_sample(
                    project=project,
                    sample=sample,
                    component_name=component_name,
                    save_each=save_each,
                    write_curve_json=write_curve_json,
                    temp_curves_dir=temp_curves_dir,
                )
                results.append(_result_record(sample, "ok"))
            except Exception as exc:
                failures.append(_empty_failure_record(sample, "cst_pipeline", exc))
                results.append(_result_record(sample, "failed", "cst_pipeline", str(exc)))
                print(f"  CST 报错: {type(exc).__name__}: {exc}")

    if project is not None and save_last:
        execute_save_project(project)

    failure_fields = [
        "feasible_index",
        "csv_row_number",
        "sample_index",
        "stage",
        "error_type",
        "error_message",
        "traceback",
    ]
    result_fields = [
        "feasible_index",
        "csv_row_number",
        "sample_index",
        "status",
        "stage",
        "error_message",
    ]
    write_csv(failures_csv, failures, failure_fields)
    write_csv(results_csv, results, result_fields)

    print(f"完成样本数: {len(results)}")
    print(f"CST 报错数: {len(failures)}")
    print(f"失败编号: {failures_csv}")
    print(f"完整结果: {results_csv}")
    return failures, results


def parse_args():
    parser = argparse.ArgumentParser(description="将几何可行样本逐个导入 CST，记录 CST 失败样本编号。")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES_CSV), help="扫描 CSV，默认 scan_latin_1024.csv。")
    parser.add_argument("--project", default=DEFAULT_PROJECT_PATH, help="目标 CST 工程路径。")
    parser.add_argument("--component", default=DEFAULT_COMPONENT_NAME, help="CST component 名称。")
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN, help="可行性标签列名。")
    parser.add_argument("--failures", default=str(DEFAULT_FAILURES_CSV), help="CST 失败编号输出 CSV。")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_CSV), help="全部样本结果输出 CSV。")
    parser.add_argument("--start-feasible-index", type=int, default=1, help="从第几个可行样本开始跑，1-based。")
    parser.add_argument("--limit", type=int, help="最多跑多少个可行样本，用于小批量调试。")
    parser.add_argument("--dry-run", action="store_true", help="只构造几何，不连接 CST。")
    parser.add_argument("--save-each", action="store_true", help="每个样本完成后保存 CST 工程。")
    parser.add_argument("--save-last", action="store_true", help="全部样本结束后保存一次 CST 工程。")
    parser.add_argument("--write-curve-json", action="store_true", help="为每个样本额外保存一份曲线 JSON。")
    parser.add_argument("--temp-curves-dir", default=str(DEFAULT_TEMP_CURVES_DIR), help="样本曲线 JSON 输出目录。")
    return parser.parse_args()


def main():
    args = parse_args()
    run_validation(
        samples_csv=args.samples,
        project_path=args.project,
        component_name=args.component,
        label_column=args.label_column,
        failures_csv=args.failures,
        results_csv=args.results,
        start_feasible_index=args.start_feasible_index,
        limit=args.limit,
        dry_run=args.dry_run,
        save_each=args.save_each,
        save_last=args.save_last,
        write_curve_json=args.write_curve_json,
        temp_curves_dir=args.temp_curves_dir,
    )


if __name__ == "__main__":
    main()
