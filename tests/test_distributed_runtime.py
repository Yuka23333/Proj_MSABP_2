from __future__ import annotations

import csv
import json
import sys
import threading
import time
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402
from msabp_opt.simulation.distributed.config import (  # noqa: E402
    DeviceConfig,
    DeviceRegistry,
    LaunchMode,
)
from msabp_opt.simulation.distributed.http_api import (  # noqa: E402
    PrincessClient,
    PrincessHttpServer,
)
from msabp_opt.simulation.distributed.maid import (  # noqa: E402
    Maid,
    MaidRuntimeConfig,
)
from msabp_opt.simulation.distributed.princess import (  # noqa: E402
    PrincessCoordinator,
)
from msabp_opt.simulation.distributed.runtime import (  # noqa: E402
    PrincessRunPaths,
    PrincessRuntime,
    PrincessRuntimeError,
    device_run_paths,
    prepare_run,
    prepare_worklist,
    select_devices,
)
from msabp_opt.simulation.distributed.state import PrincessState  # noqa: E402
from msabp_opt.simulation.distributed.transport import (  # noqa: E402
    DoctorReport,
    LaunchReceipt,
    ModuleCheck,
)


def _write_small_csv(path: Path) -> Path:
    path.write_text(
        "sample_id,value,geometry_valid,geometry_error\n"
        "0,1.0,True,\n"
        "1,2.0,False,self intersection\n"
        "2,3.0,yes,\n",
        encoding="utf-8",
    )
    return path


def _registry(device: DeviceConfig, port: int = 8765) -> DeviceRegistry:
    return DeviceRegistry(
        bind_host="127.0.0.1",
        advertise_url=f"http://127.0.0.1:{port}",
        port=port,
        devices=(device,),
    )


def _device(
    *,
    enabled: bool = False,
    launch_mode: LaunchMode = LaunchMode.SSH_PROCESS,
) -> DeviceConfig:
    return DeviceConfig(
        id="maid-a",
        enabled=enabled,
        launch_mode=launch_mode,
        ssh_target="telecom@maid-a",
        repo_root=r"D:\Academic\Proj_MSABP_2",
        python_path=r"C:\Users\telecom\miniforge3\envs\maid\python.exe",
        runtime_config_path=(
            r"D:\Academic\Proj_MSABP_2\simulations\runs"
            r"\active_maid_runtime.json"
        ),
        bell_host="maid-a" if launch_mode is LaunchMode.BELL else None,
    )


def test_prepare_worklist_excludes_invalid_geometry_before_maid(tmp_path: Path) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    paths = PrincessRunPaths.for_run(
        "run-1",
        repository_root=tmp_path,
    )

    prepared = prepare_worklist(source, paths)
    repeated = prepare_worklist(source, paths)

    assert prepared == repeated
    assert prepared.source.row_count == 3
    assert prepared.worklist.row_count == 2
    assert [item.case_id for item in prepared.worklist.rows] == ["0", "2"]
    assert len(prepared.excluded_rows) == 1
    assert prepared.excluded_rows[0].case_id == "1"
    assert prepared.excluded_rows[0].reason == "self intersection"
    assert [row["sample_id"] for row in csv.DictReader(
        paths.worklist_csv.open(encoding="utf-8")
    )] == ["0", "2"]

    source.write_text("sample_id,value\nchanged,4\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        prepare_worklist(source, paths)


def test_explicit_device_selection_opts_in_disabled_device() -> None:
    device = _device(enabled=False)
    registry = _registry(device)

    selected = select_devices(registry, ["maid-a"])

    assert len(selected) == 1 and selected[0].enabled
    assert not registry.devices[0].enabled
    with pytest.raises(PrincessRuntimeError, match="no Maid devices"):
        select_devices(registry)


def test_custom_results_root_is_frozen_for_resume(tmp_path: Path) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    device = _device(enabled=True)
    registry = _registry(device)
    custom_results = tmp_path / "results" / "raw" / "shared-bo-plan"

    preparation = prepare_run(
        source_csv=source,
        run_id="bo-batch-0000",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
        results_root=custom_results,
    )
    payload = json.loads(preparation.paths.runtime_json.read_text(encoding="utf-8"))

    assert payload["results_root"] == str(custom_results.resolve())
    repeated = prepare_run(
        source_csv=source,
        run_id="bo-batch-0000",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
        results_root=custom_results,
    )
    assert not repeated.created
    with pytest.raises(PrincessRuntimeError, match="results_root"):
        prepare_run(
            source_csv=source,
            run_id="bo-batch-0000",
            registry=registry,
            devices=(device,),
            repository_root=tmp_path,
            results_root=tmp_path / "different-results",
        )


def test_deployment_commits_runtime_last_and_refreshes_project_on_resume(
    tmp_path: Path,
) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device)
    preparation = prepare_run(
        source_csv=source,
        run_id="run-1",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    pushes: list[tuple[Path, str]] = []

    def fake_push(_device, source_path, destination_path, **_kwargs):
        pushes.append((Path(source_path), str(destination_path)))

    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )

    def fake_doctor(_device):
        return DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11",
            modules=checks,
        )

    def fake_launch(_device):
        return LaunchReceipt(
            device_id="maid-a",
            launch_mode=LaunchMode.SSH_PROCESS,
            pid=123,
            command=("ssh",),
        )

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        push_file=fake_push,
        doctor=fake_doctor,
        launcher=fake_launch,
    )
    runtime._state = PrincessState(preparation.paths.database)
    try:
        deployment = runtime.deploy_and_launch(device)
        assert deployment.launch.pid == 123
        assert [item[0].name for item in pushes] == [
            "worklist.csv",
            "template.cst",
            "maid_runtime.json",
        ]
        assert pushes[-1][1].endswith(
            r"simulations\runs\run-1\workers\maid-a\maid_runtime.json"
        )

        runtime.state.register_worker("run-1", "maid-a")
        pushes.clear()
        paths, _staged = runtime.deploy_device(
            device,
            launch_generation="resume-test",
            force_project=True,
        )
        assert [item[0].name for item in pushes] == [
            "worklist.csv",
            "template.cst",
            "maid_runtime.json",
        ]
        assert r"launches\resume-test\model\msa-bp.cst" in paths.project_path
    finally:
        runtime.close()


def test_bell_deployment_passes_ephemeral_run_token_to_launcher(
    tmp_path: Path,
) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True, launch_mode=LaunchMode.BELL)
    registry = _registry(device)
    preparation = prepare_run(
        source_csv=source,
        run_id="bell-run",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )
    launch_calls: list[tuple[DeviceConfig, dict[str, str]]] = []

    def fake_launch(
        launch_device: DeviceConfig,
        **kwargs: str,
    ) -> LaunchReceipt:
        launch_calls.append((launch_device, dict(kwargs)))
        return LaunchReceipt(
            device_id=launch_device.id,
            launch_mode=LaunchMode.BELL,
            pid=4321,
            command=("maid-bell",),
            stdout_path=r"D:\Repo\logs\maid.stdout.log",
            stderr_path=r"D:\Repo\logs\maid.stderr.log",
        )

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        push_file=lambda *_args, **_kwargs: None,
        doctor=lambda _device: DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11",
            modules=checks,
            bell_reachable=True,
        ),
        launcher=fake_launch,
    )
    runtime._state = PrincessState(preparation.paths.database)
    try:
        deployment = runtime.deploy_and_launch(device)
        assert deployment.launch.pid == 4321
        assert len(launch_calls) == 1
        launch_device, kwargs = launch_calls[0]
        assert launch_device.runtime_config_path is not None
        assert kwargs == {"bell_token": preparation.api_token}
    finally:
        runtime.close()


def test_launch_generation_isolates_project_output_and_runtime() -> None:
    paths = device_run_paths(
        _device(enabled=True),
        "run-1",
        launch_generation="resume-1",
    )

    assert paths.project_path.endswith(
        r"workers\maid-a\launches\resume-1\model\msa-bp.cst"
    )
    assert paths.output_root.endswith(r"workers\maid-a\launches\resume-1\output")
    assert paths.runtime_config_path.endswith(
        r"workers\maid-a\launches\resume-1\maid_runtime.json"
    )


def test_worker_pid_confirmation_reads_decoded_state_metadata(tmp_path: Path) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    device = _device(enabled=True)
    registry = _registry(device)
    preparation = prepare_run(
        source_csv=source,
        run_id="pid-check",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=tmp_path / "unused.cst",
    )
    runtime._state = PrincessState(preparation.paths.database)
    try:
        cutoff = time.time()
        runtime.state.register_worker(
            "pid-check",
            "maid-a",
            metadata={"pid": 321},
            now=cutoff + 1.0,
        )
        assert runtime._worker_refreshed_after(
            "maid-a",
            cutoff,
            expected_pid=321,
        )
        assert not runtime._worker_refreshed_after(
            "maid-a",
            cutoff,
            expected_pid=999,
        )
    finally:
        runtime.close()


def test_resume_refresh_skips_duplicate_maid_launch(tmp_path: Path) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    device = _device(enabled=True)
    registry = _registry(device, port=0)
    prepare_run(
        source_csv=source,
        run_id="resume",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    preparation = prepare_run(
        source_csv=source,
        run_id="resume",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )

    def forbidden_launch(_device):
        raise AssertionError("a reconnected Maid must not be launched twice")

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=tmp_path / "unused.cst",
        launcher=forbidden_launch,
    )
    try:
        runtime.start_server()
        runtime.state.register_worker(
            "resume",
            "maid-a",
            metadata={"pid": 123},
            now=float(runtime._server_started_at) + 1.0,
        )
        assert runtime.start_workers(resume_grace_seconds=0.0) == ()
    finally:
        runtime.close()


def test_all_unconfirmed_launches_fail_fast_after_hello_deadline(
    tmp_path: Path,
) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device, port=0)
    preparation = prepare_run(
        source_csv=source,
        run_id="no-hello",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )

    def fake_doctor(_device):
        return DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11.13",
            modules=checks,
        )

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        startup_timeout_seconds=0.01,
        max_recovery_launch_attempts=1,
        push_file=lambda *_args, **_kwargs: None,
        doctor=fake_doctor,
        launcher=lambda _device: LaunchReceipt(
            device_id="maid-a",
            launch_mode=LaunchMode.SSH_PROCESS,
            pid=123,
            command=("ssh",),
        ),
    )
    try:
        runtime.start_server()
        with pytest.raises(PrincessRuntimeError, match="no Maid registered"):
            runtime.start_workers()
    finally:
        runtime.close()


def test_start_workers_requires_pid_confirmed_hello(tmp_path: Path) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device, port=0)
    preparation = prepare_run(
        source_csv=source,
        run_id="confirmed",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )
    holder: dict[str, PrincessRuntime] = {}

    def fake_doctor(launch_device):
        assert launch_device.resolved_runtime_config_path.endswith(
            r"workers\maid-a\maid_runtime.json"
        )
        return DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=False,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11.13",
            modules=checks,
        )

    def fake_launch(_device):
        holder["runtime"].state.register_worker(
            "confirmed",
            "maid-a",
            metadata={"pid": 123},
        )
        return LaunchReceipt(
            device_id="maid-a",
            launch_mode=LaunchMode.SSH_PROCESS,
            pid=123,
            command=("ssh",),
        )

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        startup_timeout_seconds=0.1,
        push_file=lambda *_args, **_kwargs: None,
        doctor=fake_doctor,
        launcher=fake_launch,
    )
    holder["runtime"] = runtime
    try:
        runtime.start_server()
        deployments = runtime.start_workers()
        assert len(deployments) == 1
        assert deployments[0].launch.pid == 123
    finally:
        runtime.close()


def test_launcher_receipt_error_is_overridden_by_same_round_hello(
    tmp_path: Path,
) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device, port=0)
    preparation = prepare_run(
        source_csv=source,
        run_id="inconclusive-launch",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )
    holder: dict[str, PrincessRuntime] = {}

    def fake_doctor(_device):
        return DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11.13",
            modules=checks,
        )

    def inconclusive_launch(_device):
        holder["runtime"].state.register_worker(
            "inconclusive-launch",
            "maid-a",
            metadata={"pid": 987},
            now=time.time() + 1.0,
        )
        raise TimeoutError("SSH receipt timed out after remote Start-Process")

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        startup_timeout_seconds=0.01,
        push_file=lambda *_args, **_kwargs: None,
        doctor=fake_doctor,
        launcher=inconclusive_launch,
    )
    holder["runtime"] = runtime
    try:
        runtime.start_server()
        assert runtime.start_workers() == ()
        assert runtime.state.get_worker(
            "inconclusive-launch",
            "maid-a",
        )["status"] == "idle"
        assert "maid-a" not in runtime._launch_failures
        assert "maid-a" not in runtime._disabled_for_run
    finally:
        runtime.close()


def test_deployment_rejects_non_cst_2025_python_version(tmp_path: Path) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device)
    preparation = prepare_run(
        source_csv=source,
        run_id="wrong-python",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )
    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        push_file=lambda *_args, **_kwargs: None,
        doctor=lambda _device: DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.12.4",
            modules=checks,
        ),
        launcher=lambda _device: pytest.fail("launcher must not run"),
    )
    runtime._state = PrincessState(preparation.paths.database)
    try:
        with pytest.raises(PrincessRuntimeError, match="requires 3.11"):
            runtime.deploy_and_launch(device)
    finally:
        runtime.close()


def test_offline_watchdog_relaunches_with_a_fresh_project_generation(
    tmp_path: Path,
) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device)
    preparation = prepare_run(
        source_csv=source,
        run_id="offline",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    pushes: list[str] = []
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )
    holder: dict[str, PrincessRuntime] = {}

    def fake_launch(_device):
        holder["runtime"].state.register_worker(
            "offline",
            "maid-a",
            metadata={"pid": 444},
        )
        return LaunchReceipt(
            device_id="maid-a",
            launch_mode=LaunchMode.SSH_PROCESS,
            pid=444,
            command=("ssh",),
        )

    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        push_file=lambda _d, _s, destination, **_kwargs: pushes.append(
            str(destination)
        ),
        doctor=lambda _device: DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11.13",
            modules=checks,
        ),
        launcher=fake_launch,
    )
    holder["runtime"] = runtime
    runtime._state = PrincessState(preparation.paths.database)
    try:
        runtime.state.register_worker("offline", "maid-a")
        claim = runtime.state.claim_next("offline", "maid-a")
        assert claim is not None
        runtime.state.release_task(
            "offline",
            "maid-a",
            claim.attempt_id,
            claim.lease_token,
            reason="synthetic disconnect",
            worker_status="offline",
        )
        runtime._recovery_due["maid-a"] = 0.0
        now = time.time()
        runtime._schedule_recoveries(
            now,
            runtime.state.progress("offline").as_dict(),
        )
        assert any(
            "\\launches\\recovery-" in path and path.endswith("msa-bp.cst")
            for path in pushes
        )
        runtime._refresh_pending_launches(now + 1.0)
        assert "maid-a" not in runtime._launch_pending
        assert runtime.state.get_worker("offline", "maid-a")["status"] == "idle"
    finally:
        runtime.close()


def test_exhausted_recovery_budget_stops_instead_of_monitoring_forever(
    tmp_path: Path,
) -> None:
    source = _write_small_csv(tmp_path / "samples.csv")
    project = tmp_path / "template.cst"
    project.write_bytes(b"standalone-cst")
    device = _device(enabled=True)
    registry = _registry(device)
    preparation = prepare_run(
        source_csv=source,
        run_id="exhausted",
        registry=registry,
        devices=(device,),
        repository_root=tmp_path,
    )
    checks = tuple(
        ModuleCheck(name=name, available=True)
        for name in ("cst.interface", "shapely", "scipy", "pandas", "numpy")
    )
    runtime = PrincessRuntime(
        preparation=preparation,
        registry=registry,
        devices=(device,),
        project_template=project,
        max_recovery_launch_attempts=1,
        push_file=lambda *_args, **_kwargs: None,
        doctor=lambda _device: DoctorReport(
            device_id="maid-a",
            python_path=device.python_path,
            python_exists=True,
            repo_exists=True,
            project_exists=True,
            runtime_config_exists=True,
            maid_entrypoint_exists=True,
            scheduled_task_exists=None,
            python_version="3.11.13",
            modules=checks,
        ),
        launcher=lambda _device: (_ for _ in ()).throw(OSError("host offline")),
    )
    runtime._state = PrincessState(preparation.paths.database)
    try:
        runtime.state.register_worker("exhausted", "maid-a", now=1.0)
        runtime._recovery_due["maid-a"] = 0.0
        with pytest.raises(PrincessRuntimeError, match="cannot progress"):
            runtime.monitor(interval_seconds=0.001, terminal_grace_seconds=0.0)
        assert runtime._disabled_for_run == {"maid-a"}
    finally:
        runtime.close()


def _write_full_default_csv(path: Path, row_count: int = 2) -> Path:
    parameter_names = tuple(antenna_sampler.PARAMETER_REGISTRY)
    fieldnames = ("sample_id", *parameter_names, "geometry_valid", "geometry_error")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(row_count):
            row = {
                name: getattr(antenna_sampler.antenna_outline.DEFAULT_ANTENNA_PARAMETERS, name)
                for name in parameter_names
            }
            row.update(
                {
                    "sample_id": str(index),
                    "geometry_valid": True,
                    "geometry_error": "",
                }
            )
            writer.writerow(row)
    return path


def test_real_http_fake_cst_end_to_end_completes_dry_run(tmp_path: Path) -> None:
    csv_path = _write_full_default_csv(tmp_path / "samples.csv")
    paths = PrincessRunPaths.for_run("e2e", repository_root=tmp_path)
    worklist = prepare_worklist(csv_path, paths)
    token = "e2e-secret-token-with-at-least-32-characters"

    with PrincessState(paths.database) as state:
        state.initialize_run("e2e", worklist.worklist)
        coordinator = PrincessCoordinator(
            state,
            run_id="e2e",
            csv_sha256=worklist.worklist.sha256,
            results_dir=paths.results_root,
            incoming_dir=paths.incoming_root,
        )
        server = PrincessHttpServer(
            ("127.0.0.1", 0),
            token=token,
            upload_dir=paths.upload_root,
            message_handler=coordinator.handle_message,
            artifact_handler=coordinator.handle_artifact,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = int(server.server_address[1])
            config = MaidRuntimeConfig.from_mapping(
                {
                    "schema_version": 1,
                    "run_id": "e2e",
                    "worker_id": "maid-local-test",
                    "princess_url": f"http://127.0.0.1:{port}",
                    "api_token": token,
                    "csv_path": str(worklist.worklist.path),
                    "csv_sha256": worklist.worklist.sha256,
                    "project_path": str(tmp_path / "unused.cst"),
                    "output_root": str(tmp_path / "maid-output"),
                    "dry_run": True,
                    "heartbeat_seconds": 60.0,
                    "poll_seconds": 0.001,
                }
            )
            worker = Maid(
                config,
                client=PrincessClient(f"http://127.0.0.1:{port}", token),
            )
            assert worker.run() == 0
            assert state.progress("e2e").as_dict() == {
                "total": 2,
                "pending": 0,
                "running": 0,
                "completed": 2,
                "failed": 0,
                "finished": 2,
                "is_terminal": True,
            }
            assert (paths.results_root / "case_0000" / "manifest.json").is_file()
            assert (paths.results_root / "case_0001" / "manifest.json").is_file()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5.0)
