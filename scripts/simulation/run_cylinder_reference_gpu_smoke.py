"""Run one default-reference propagation solve on coconutg2 with GPU telemetry.

--prepare-only creates and validates the one-row CSV, without SSH or solving.
--snapshot performs one read-only GPU/process/status query.
--run starts Princess and records an initial, every-180-second, and final sample.
"""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation import princess
from scripts.geometry.shapely_antenna_model import DEFAULT_PARAMETERS
from scripts.automation.cst_build_msabp_geometry import build_sampled_polygon_specs

DEFAULT_CONFIG = ROOT / "configs/simulation/cylinder_reference_gpu_smoke.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare(config: dict) -> Path:
    row = {
        "sample_id": "current_default_reference",
        "simulation_mode": "propagation_s21",
        "doe_source": "current_script_default",
        "geometry_valid": "True",
        "geometry_error": "",
        "final_conductor_components": "1",
        **asdict(DEFAULT_PARAMETERS),
    }
    build_sampled_polygon_specs(DEFAULT_PARAMETERS, coordinate_quantum_mm=0.01)
    path = ROOT / config["worklist_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        if len(existing) != 1 or existing[0] != {k: str(v) for k, v in row.items()}:
            raise ValueError("Existing reference CSV differs; use a new run plan instead of overwriting it")
    else:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    return path


def snapshot(config: dict, label: str) -> dict:
    powershell = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$gpu = @(& nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>&1 | ForEach-Object { "$_" })
$gpuExit = $LASTEXITCODE
$apps = @(& nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>&1 | ForEach-Object { "$_" })
$appExit = $LASTEXITCODE
$solver = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match 'Solver|CSTDesignEnvironment|CSTStudio' } | Select-Object Id,ProcessName,CPU,WorkingSet64,PrivateMemorySize64)
$os = Get-CimInstance Win32_OperatingSystem
[pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');gpu_csv=$gpu;gpu_exit=$gpuExit;compute_apps_csv=$apps;compute_apps_exit=$appExit;solver_processes=$solver;free_ram_kib=$os.FreePhysicalMemory;total_ram_kib=$os.TotalVisibleMemorySize} | ConvertTo-Json -Depth 5 -Compress
'''
    encoded = base64.b64encode(powershell.encode("utf-16-le")).decode("ascii")
    record = {"utc": utc_now(), "label": label, "device": "coconutg2"}
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", config["ssh_target"],
             "powershell -NoProfile -EncodedCommand " + encoded],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=50,
        )
        record["ssh_exit"] = result.returncode
        if result.returncode == 0:
            record["remote"] = json.loads(result.stdout.lstrip("\ufeff").strip())
        else:
            record["error"] = result.stderr or result.stdout
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc)
    try:
        record["princess"] = princess.load_status(config["run_id"])
    except (FileNotFoundError, RuntimeError) as exc:
        record["princess_status_unavailable"] = str(exc)
    return record


def command(config: dict, worklist: Path) -> list[str]:
    return [
        sys.executable, "-u", str(ROOT / "scripts/simulation/princess.py"), "start",
        "--csv", str(worklist), "--run-id", config["run_id"],
        "--devices-config", str(ROOT / config["devices_config"]),
        "--device", "coconutg2",
        "--device-project-relative-path", config["device_project_relative_path"],
        "--max-attempts", "1", "--max-consecutive-errors", "1",
        "--max-recovery-launch-attempts", "1",
        "--startup-timeout-seconds", "120", "--command-timeout-seconds", "60",
        "--save-project-after-case",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--snapshot", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config["device_ids"] != ["coconutg2"] or config["ssh_target"] != "telecom@coconutg2":
        raise ValueError("Multilayer phantom campaign is restricted to coconutg2")
    if args.snapshot:
        print(json.dumps(snapshot(config, "manual"), ensure_ascii=False, indent=2))
        return 0
    worklist = prepare(config)
    argv = command(config, worklist)
    if args.prepare_only:
        print(f"Prepared 1 default-reference case: {worklist}")
        print(subprocess.list2cmdline(argv))
        return 0

    log_root = ROOT / "logs" / config["run_id"]
    log_root.mkdir(parents=True, exist_ok=True)
    lock = log_root / "monitor.lock"
    lock_stream = lock.open("x")
    history: list[dict] = []
    process = None
    try:
        (log_root / "plan.json").write_text(json.dumps({**config, "csv_sha256": hashlib.sha256(worklist.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
        with (log_root / "gpu_telemetry.jsonl").open("a", encoding="utf-8") as telemetry, (log_root / "princess.log").open("a", encoding="utf-8") as output:
            def record(label: str) -> None:
                sample = snapshot(config, label)
                history.append(sample)
                telemetry.write(json.dumps(sample, ensure_ascii=False) + "\n")
                telemetry.flush()
                print(f"[GPU monitor] {sample['utc']} {label}: " + str(sample.get("remote", {}).get("gpu_csv", sample.get("error", "no GPU data"))), flush=True)

            record("before_start")
            process = subprocess.Popen(argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            print(f"Princess PID={process.pid}; run={config['run_id']}; interval={config['gpu_poll_seconds']}s", flush=True)
            next_sample = time.monotonic() + float(config["gpu_poll_seconds"])
            while process.poll() is None:
                if time.monotonic() >= next_sample:
                    record("periodic")
                    next_sample += float(config["gpu_poll_seconds"])
                time.sleep(1)
            record("after_exit")
        peaks = []
        for sample in history:
            for line in sample.get("remote", {}).get("gpu_csv", []):
                try:
                    fields = next(csv.reader([line], skipinitialspace=True))
                    peaks.append(float(fields[3]))
                except (ValueError, IndexError):
                    continue
        summary = {"run_id": config["run_id"], "princess_exit": process.returncode,
                   "samples": len(history), "observed_peak_memory_mib": max(peaks, default=None),
                   "note": "180-second snapshots cannot exclude a brief VRAM peak or prove OOM; inspect solver logs.",
                   "last_status": history[-1].get("princess"), "completed_at_utc": utc_now()}
        (log_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return process.returncode
    finally:
        lock_stream.close()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
