"""F5: three pinned Princess runs of one fixed-model sample, then compare curves.

No solver starts without RUN (or --yes). This uses the existing 23-variable
Maid workflow, NOT the arbitrary-tree experimental CST builder.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.geometry.shapely_antenna_model import DEFAULT_PARAMETERS  # noqa: E402
from scripts.simulation.run_remote_real_smoke import run_and_tee  # noqa: E402
from scripts.automation.antenna_sampler import parameters_from_csv_row  # noqa: E402
from msabp_opt.simulation.distributed.config import load_device_registry  # noqa: E402

# F5 settings: keep RUN_ID for resuming; change it for a new experiment.
RUN_ID = "base-validation-001"
DEVICE_IDS = ("local", "convallariag5", "coconutg2")
PARAMETERS_JSON = None  # Optional JSON object of overrides for the fixed model.
PROJECT = ROOT / "simulations/models/msa-bp.cst"
DEVICE_CONFIG = ROOT / "configs/simulation/princess_devices.json"
BAND_GHZ = (3.1, 4.8)
TOLERANCE_DB = 0.1  # Engineering comparison threshold, not a statistical test.
CURVES = {"s11": "S11.csv", "rad_eff": "Rad_Eff.csv", "tot_eff": "Tot_Eff.csv"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_curve(path):
    """Read existing CST ASCII exports; reject nonfinite/ambiguous samples."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = re.split(r"[\s,;]+", line.strip())
        if len(fields) < 2:
            continue
        try:
            row = [float(fields[0]), float(fields[1])]
        except ValueError:
            continue
        if not np.isfinite(row).all():
            raise ValueError(f"Nonfinite curve sample: {path}")
        rows.append(row)
    data = np.asarray(sorted(rows), dtype=float)
    if len(rows) < 2 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"Missing/duplicate frequency samples: {path}")
    return data[:, 0], data[:, 1]


def compare_pair(first, second, band):
    lo, hi = band
    if not np.isfinite(band).all() or not lo < hi:
        raise ValueError("Invalid comparison band")
    for x, _ in (first, second):
        if x[0] > lo + 1e-9 or x[-1] < hi - 1e-9:
            raise ValueError("Curve does not cover the full requested band; no extrapolation allowed")
    grid = np.unique(np.concatenate(([lo, hi], *(x[(x > lo) & (x < hi)] for x, _ in (first, second)))))
    a, b = (np.interp(grid, x, y) for x, y in (first, second))
    delta = b - a
    index = int(np.argmax(np.abs(delta)))
    return {"max_abs_delta_db": float(abs(delta[index])),
            "rms_delta_db": float(np.sqrt(np.trapezoid(delta**2, grid) / (hi-lo))),
            "worst_frequency_ghz": float(grid[index]), "comparison_points": len(grid)}


def princess_command(plan, device, folder):
    return [sys.executable, str(ROOT / "scripts/simulation/princess.py"), "start",
            "--csv", str(folder / "sample.csv"), "--run-id", f"{plan['run_id']}-{device}",
            "--device", device, "--devices-config", plan["devices_config"],
            "--project", str(folder / "template.cst"),
            "--results-root", str(folder / "results" / device),
            "--max-attempts", "1"]


def prepare(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        raise ValueError("Unsafe run ID")
    if not np.isfinite(args.tolerance_db) or args.tolerance_db < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if not np.isfinite(args.band).all() or not args.band[0] < args.band[1]:
        raise ValueError("Invalid comparison band")
    parameters = asdict(DEFAULT_PARAMETERS)
    if args.parameters:
        overrides = json.loads(args.parameters.read_text(encoding="utf-8-sig"))
        if not isinstance(overrides, dict) or set(overrides) - set(parameters):
            raise ValueError("Parameters must be a JSON object containing only fixed-model parameter names")
        parameters.update(overrides)
    parameters = asdict(parameters_from_csv_row(parameters))
    registry = load_device_registry(args.devices_config)
    for device in DEVICE_IDS:
        registry.get_device(device)
    plan = {"schema_version": 1, "run_id": args.run_id,
            "geometry_route": "fixed_model_23_parameters_not_arbitrary_tree",
            "devices": list(DEVICE_IDS), "parameters": parameters,
            "devices_config": str(args.devices_config.resolve()),
            "devices_config_sha256": sha256(args.devices_config),
            "template_source": str(args.project.resolve()), "template_sha256": sha256(args.project),
            "band_ghz": list(args.band), "tolerance_db": args.tolerance_db,
            "curve_units": "Existing CST workflow exports S11 and efficiencies in dB"}
    folder = ROOT / "simulations/runs" / args.run_id
    plan_path = folder / "validation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing plan differs; use a new --run-id")
        if sha256(folder / "template.cst") != plan["template_sha256"]:
            raise ValueError("Frozen template has changed")
    else:
        if folder.exists():
            raise FileExistsError(f"Refusing to reuse non-plan directory: {folder}")
        folder.mkdir(parents=True)
        (folder / "template.cst").write_bytes(args.project.read_bytes())
        with (folder / "sample.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", *parameters])
            writer.writeheader()
            writer.writerow({"sample_id": "reference", **parameters})
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with (folder / "sample.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]['sample_id'] != 'reference' or asdict(parameters_from_csv_row(rows[0])) != parameters:
        raise ValueError("Frozen sample has changed")
    return plan, folder


def load_result(folder, device, parameters):
    case = folder / "results" / device / "case_reference"
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("dry_run") is not False:
        raise ValueError(f"{device}: no completed real solve")
    if manifest.get("parameters") != parameters:
        raise ValueError(f"{device}: parameters differ from the frozen input")
    curves = {}
    for key, filename in CURVES.items():
        path = case / filename
        artifact = manifest.get("artifacts", {}).get(key, {})
        if artifact.get("path") != filename or artifact.get("sha256") != sha256(path):
            raise ValueError(f"{device}: missing/mismatched artifact {filename}")
        curves[key] = read_curve(path)
    return curves


def compare(plan, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, errors = {}, {}
    for device in DEVICE_IDS:
        try:
            results[device] = load_result(folder, device, plan["parameters"])
        except (OSError, ValueError, KeyError) as exc:
            errors[device] = str(exc)
    report = {"complete": len(results) == 3, "errors": errors, "comparisons": [],
              "band_ghz": plan["band_ghz"], "tolerance_db": plan["tolerance_db"]}
    for a, b in combinations(results, 2):
        for key in CURVES:
            try:
                row = compare_pair(results[a][key], results[b][key], plan["band_ghz"])
                report["comparisons"].append({"a": a, "b": b, "metric": key, **row,
                                              "within_tolerance": row["max_abs_delta_db"] <= plan["tolerance_db"]})
            except ValueError as exc:
                errors[f"{a}/{b}/{key}"] = str(exc)
    report["passed"] = report["complete"] and not errors and all(r["within_tolerance"] for r in report["comparisons"])
    (folder / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["comparisons"]:
        with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["comparisons"][0]))
            writer.writeheader()
            writer.writerows(report["comparisons"])
    if results:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
        for ax, key in zip(axes, CURVES):
            for device, curves in results.items():
                ax.plot(*curves[key], label=device)
            ax.set(title=key, xlabel="Frequency (GHz)", ylabel="dB", xlim=plan["band_ghz"])
            ax.grid(alpha=.3)
            ax.legend()
        fig.savefig(folder / "comparison.png", dpi=160)
        plt.close(fig)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--parameters", type=Path, default=PARAMETERS_JSON)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--devices-config", type=Path, default=DEVICE_CONFIG)
    parser.add_argument("--band", type=float, nargs=2, default=BAND_GHZ)
    parser.add_argument("--tolerance-db", type=float, default=TOLERANCE_DB)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    plan, folder = prepare(args)
    print(f"Plan: {folder}\nGeometry: {plan['geometry_route']}")
    if args.prepare_only:
        return 0
    if not args.compare_only:
        print("Three real CST solves, sequential/pinned: local -> convallariag5 -> coconutg2.")
        print("Close unrelated local CST tasks first. Ensure remote code has been pulled.")
        if not args.yes and input("Type RUN to start: ").strip() != "RUN":
            return 0
        for device in DEVICE_IDS:
            command = princess_command(plan, device, folder)
            code = run_and_tee(command, folder / f"{device}.log")
            if code:
                print(f"{device}: Princess exited {code}; retaining logs and trying the next device")
    return compare(plan, folder)


if __name__ == "__main__":
    raise SystemExit(main())
