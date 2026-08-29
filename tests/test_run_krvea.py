from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from msabp_opt.optimization import krvea_data, qlogehvi
from scripts.automation import antenna_sampler
from scripts.optimization import run_krvea


def _config(tmp_path: Path, *, q: int = 4, budget: int = 128) -> run_krvea.CampaignConfig:
    source = tmp_path / "history"
    source.mkdir(exist_ok=True)
    return run_krvea.CampaignConfig(
        plan_id="synthetic-krvea-test",
        source_directories=(source,),
        output_directory=tmp_path / "target",
        total_budget=budget,
        sampling_config=run_krvea.SAMPLING_CONFIG,
        device_config=run_krvea.DEVICE_CONFIG,
        project_template=run_krvea.PROJECT_TEMPLATE,
        proposal=run_krvea.krvea.KRVEAConfig(
            n_variables=11,
            n_objectives=4,
            q=q,
            inner_evaluations=120,
            seed=17,
        ),
    )


def _dataset(n: int = 8) -> krvea_data.Dataset:
    space = krvea_data.authoritative_input_space()
    unit = np.linspace(0.05, 0.95, n)[:, None] * np.ones((n, len(space.names)))
    raw = space.denormalize(unit)
    objectives = np.column_stack(
        (
            np.linspace(0.4, 0.7, n),
            np.linspace(0.1, 0.4, n),
            np.asarray([space.exact_normalized_area(row) for row in raw]),
            np.linspace(-2.0, 3.0, n),
        )
    )
    metadata = pd.DataFrame({"has_completed_result": [True] * n})
    return krvea_data.Dataset(raw, unit, objectives, metadata, space)


def _nominal_full() -> dict[str, float]:
    space = krvea_data.authoritative_input_space()
    return run_krvea.full_parameter_mapping(space.values(space.nominal))


def _candidate_frame(case_ids: list[str]) -> pd.DataFrame:
    parameters = _nominal_full()
    return pd.DataFrame(
        [
            {
                "sample_id": case_id,
                **parameters,
                "geometry_valid": True,
                "geometry_error": "",
            }
            for case_id in case_ids
        ]
    )


def _synthetic_history_observations(n: int = 512) -> pd.DataFrame:
    space = krvea_data.authoritative_input_space()
    unit = np.zeros((n, len(space.names)), dtype=np.float64)
    unit[:, 0] = (np.arange(n, dtype=np.float64) + 0.5) / n
    unit[:, 1:] = 0.5
    raw = space.denormalize(unit)
    records = []
    for index, row in enumerate(raw):
        active = space.values(row)
        width, height, area = krvea_data.substrate_dimensions(active)
        records.append(
            {
                "source": "synthetic",
                "source_root": "synthetic",
                "case_id": f"history_{index:04d}",
                "case_directory": f"synthetic/case_{index:04d}",
                "status": "completed",
                "is_penalty": False,
                **active,
                "substrate_width_mm": width,
                "substrate_height_mm": height,
                krvea_data.AREA_COLUMN: area,
                krvea_data.NORMALIZED_AREA_COLUMN: area / 2720.2,
                krvea_data.WORST_S11_COLUMN: 0.5,
                krvea_data.MEAN_TOT_EFF_COLUMN: 0.7,
                krvea_data.TOT_EFF_LOSS_COLUMN: 0.3,
                krvea_data.CAP_GAIN_LINEAR_COLUMN: 1.0,
                krvea_data.CAP_GAIN_DBI_COLUMN: 0.0,
                "cap_cache_hit": False,
                "tot_eff_samples_kept": 2,
                "tot_eff_samples_removed_above_one": 0,
            }
        )
    return pd.DataFrame.from_records(records)


def test_tracked_sampling_config_exactly_matches_authoritative_doe_file() -> None:
    tracked = json.loads(run_krvea.SAMPLING_CONFIG.read_text(encoding="utf-8"))
    local_authoritative = (
        run_krvea.REPOSITORY_ROOT
        / "results"
        / "processed"
        / "doe-11var-branch-up-lhs-512.sampling.json"
    )
    if local_authoritative.is_file():
        authoritative = json.loads(local_authoritative.read_text(encoding="utf-8"))
        assert tracked == authoritative
    plan = antenna_sampler.resolve_sampling_plan(tracked, n_samples=1)
    sampled = [item for item in plan.resolved_parameters if item.effective_sample]
    names = tuple(item.spec.name for item in sampled)
    assert names == krvea_data.ACTIVE_PARAMETER_NAMES
    assert [item.nominal for item in sampled] == pytest.approx(
        krvea_data.ACTIVE_PARAMETER_NOMINAL
    )
    assert [item.lower for item in sampled] == pytest.approx(
        krvea_data.ACTIVE_PARAMETER_LOWER
    )
    assert [item.upper for item in sampled] == pytest.approx(
        krvea_data.ACTIVE_PARAMETER_UPPER
    )


def test_full_parameter_mapping_merges_11_active_and_12_fixed_in_registry_order() -> None:
    space = krvea_data.authoritative_input_space()
    full = run_krvea.full_parameter_mapping(space.values(space.nominal))

    assert tuple(full) == tuple(antenna_sampler.PARAMETER_REGISTRY)
    assert len(full) == 23
    assert all(full[name] == value for name, value in krvea_data.FIXED_PARAMETER_VALUES.items())


def test_history_observations_are_cached_and_must_have_512_distinct_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_directory.mkdir()
    expected = _synthetic_history_observations()
    calls = 0

    def fake_collect(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return expected.copy()

    monkeypatch.setattr(krvea_data, "collect_observations", fake_collect)
    first, first_dataset, _ = run_krvea.load_or_build_history_cache(config)
    second, second_dataset, _ = run_krvea.load_or_build_history_cache(config)

    assert calls == 1
    assert len(first) == len(second) == 512
    assert len(first_dataset.x_unit) == len(second_dataset.x_unit) == 512


def test_history_cache_rejects_any_count_other_than_512(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_directory.mkdir()
    monkeypatch.setattr(
        krvea_data,
        "collect_observations",
        lambda *_args, **_kwargs: _synthetic_history_observations(511),
    )

    with pytest.raises(RuntimeError, match="exactly 512"):
        run_krvea.load_or_build_history_cache(config)


def test_plan_freezes_budget_sources_area_and_cap_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset = _dataset()
    snapshot = {"manifest_count": 512, "manifest_metadata_sha256": "a" * 64}
    plan = run_krvea.load_or_create_plan(config, dataset.input_space, snapshot)

    assert plan["campaign"] == "512 historical + 128 new expensive evaluations"
    assert plan["total_budget"] == 128
    assert plan["q"] == 4
    assert plan["historical_training_count"] == 512
    assert plan["output_is_automatic_training_source"] is True
    assert plan["objectives"][2]["nominal_area_reference_mm2"] == 2720.2
    assert plan["objectives"][2]["penalty"] == 2.0
    assert plan["objectives"][3]["optimization_scalar"] == "dBi"
    assert "linear_power" in plan["objectives"][3]["averaging"]

    changed = run_krvea.replace(config, total_budget=132)
    with pytest.raises(RuntimeError, match="differs"):
        run_krvea.load_or_create_plan(changed, dataset.input_space, snapshot)


def test_create_batch_persists_active_state_before_penalty_and_writes_registry_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.output_directory.mkdir()
    state = run_krvea._default_state(config)
    run_krvea.save_state(config, state)
    dataset = _dataset()
    unit = np.asarray(
        [
            [0.15] * 11,
            [0.35] * 11,
            [0.55] * 11,
            [0.75] * 11,
        ]
    )
    proposal = SimpleNamespace(
        unit_values=unit,
        raw_values=dataset.input_space.denormalize(unit),
        predicted_mean=np.zeros((4, 4), dtype=np.float64),
        predicted_std=np.zeros((4, 4), dtype=np.float64),
        diagnostics={"empty_reference_count": 37, "mode": "exploitation"},
    )
    monkeypatch.setattr(run_krvea, "_request_remote_proposal", lambda *_args, **_kwargs: proposal)
    calls = 0

    def fake_preflight(parameters, **_kwargs):
        nonlocal calls
        calls += 1
        width, height, area = krvea_data.substrate_dimensions(parameters)
        return (
            calls != 1,
            "synthetic invalid" if calls == 1 else "",
            {
                "substrate_width_mm": width,
                "substrate_height_mm": height,
                krvea_data.AREA_COLUMN: area,
            },
        )

    persisted_before_penalty = False

    def fake_penalty(config_arg, **_kwargs):
        nonlocal persisted_before_penalty
        saved = json.loads(
            (config_arg.output_directory / run_krvea.STATE_FILENAME).read_text(
                encoding="utf-8-sig"
            )
        )
        persisted_before_penalty = saved["active_batch"] is not None
        return config_arg.output_directory / "unused"

    monkeypatch.setattr(run_krvea, "preflight_full_parameters", fake_preflight)
    monkeypatch.setattr(run_krvea, "write_penalty_case", fake_penalty)

    batch = run_krvea.create_batch(config, state, dataset, q=4)

    assert persisted_before_penalty
    assert batch["case_ids"] == [f"krvea_{index:04d}" for index in range(4)]
    assert batch["invalid_preflight_case_ids"] == ["krvea_0000"]
    assert state["previous_empty_reference_count"] == 0
    assert state["active_batch"]["proposed_empty_reference_count"] == 37
    candidates = pd.read_csv(batch["candidate_csv"], encoding="utf-8-sig")
    worklist = pd.read_csv(batch["worklist_csv"], encoding="utf-8-sig")
    assert list(candidates.columns[:24]) == [
        "sample_id",
        *antenna_sampler.PARAMETER_REGISTRY,
    ]
    assert len(worklist) == 3


def test_penalty_manifest_makes_all_four_objectives_poor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_directory.mkdir()
    parameters = _nominal_full()
    destination = run_krvea.write_penalty_case(
        config,
        case_id="krvea_0000",
        parameters=parameters,
        failure_stage="geometry_preflight",
        failure_message="synthetic",
    )
    payload = json.loads(
        (destination / qlogehvi.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    objective = payload["optimization_objectives"]

    assert objective[krvea_data.WORST_S11_COLUMN] == 1.0
    assert objective[krvea_data.TOT_EFF_LOSS_COLUMN] == 1.0
    assert objective[krvea_data.NORMALIZED_AREA_COLUMN] == 2.0
    assert objective[krvea_data.CAP_GAIN_DBI_COLUMN] == 10.0
    assert objective[krvea_data.AREA_COLUMN] == pytest.approx(2720.2)
    parsed = krvea_data.parse_manifest(
        destination / qlogehvi.MANIFEST_FILENAME,
        source_root=config.output_directory,
    )
    assert parsed[krvea_data.NORMALIZED_AREA_COLUMN] == 2.0


def test_empty_reference_count_is_promoted_only_after_terminal_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, q=2)
    config.output_directory.mkdir()
    case_ids = ["krvea_0000", "krvea_0001"]
    candidates = tmp_path / "candidates.csv"
    _candidate_frame(case_ids).to_csv(candidates, index=False)
    batch = {
        "batch_index": 0,
        "run_id": "synthetic-krvea-test-batch-0000",
        "status": "proposed",
        "case_ids": case_ids,
        "invalid_preflight_case_ids": case_ids,
        "candidate_csv": str(candidates),
        "worklist_csv": str(tmp_path / "worklist.csv"),
        "proposal_diagnostics": {"empty_reference_count": 41},
        "proposed_empty_reference_count": 41,
    }
    state = run_krvea._default_state(config)
    state["previous_empty_reference_count"] = 29
    state["active_batch"] = batch
    run_krvea.save_state(config, state)
    assert state["previous_empty_reference_count"] == 29
    monkeypatch.setattr(run_krvea, "_task_records", lambda *_args, **_kwargs: {})

    assert run_krvea.finalize_active_batch(config, state, princess_exit_code=None)
    assert state["previous_empty_reference_count"] == 41
    assert state["active_batch"] is None
    assert len(state["completed_batches"]) == 1


def test_completed_case_objective_failure_gets_four_objective_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, q=1)
    config.output_directory.mkdir()
    case_id = "krvea_0000"
    candidates = tmp_path / "candidates.csv"
    _candidate_frame([case_id]).to_csv(candidates, index=False)
    case_directory = config.output_directory / f"case_{case_id}"
    case_directory.mkdir()
    (case_directory / qlogehvi.MANIFEST_FILENAME).write_text(
        json.dumps({"case_id": case_id, "status": "completed", "parameters": _nominal_full()}),
        encoding="utf-8",
    )
    batch = {
        "batch_index": 0,
        "run_id": "synthetic-krvea-test-batch-0000",
        "status": "proposed",
        "case_ids": [case_id],
        "invalid_preflight_case_ids": [],
        "candidate_csv": str(candidates),
        "worklist_csv": str(tmp_path / "worklist.csv"),
        "proposal_diagnostics": {"empty_reference_count": 33},
        "proposed_empty_reference_count": 33,
    }
    state = run_krvea._default_state(config)
    state["active_batch"] = batch
    monkeypatch.setattr(
        run_krvea,
        "_task_records",
        lambda *_args, **_kwargs: {case_id: {"status": "completed"}},
    )
    monkeypatch.setattr(
        krvea_data,
        "parse_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad FFS")),
    )

    assert run_krvea.finalize_active_batch(config, state, princess_exit_code=0)
    sidecar = json.loads(
        (case_directory / qlogehvi.OPTIMIZATION_PENALTY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    objective = sidecar["optimization_objectives"]
    assert objective[krvea_data.WORST_S11_COLUMN] == 1.0
    assert objective[krvea_data.TOT_EFF_LOSS_COLUMN] == 1.0
    assert objective[krvea_data.NORMALIZED_AREA_COLUMN] == 2.0
    assert objective[krvea_data.CAP_GAIN_DBI_COLUMN] == 10.0


def test_completed_task_without_manifest_is_terminally_penalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, q=1)
    config.output_directory.mkdir()
    case_id = "krvea_0000"
    candidates = tmp_path / "candidates.csv"
    _candidate_frame([case_id]).to_csv(candidates, index=False)
    state = run_krvea._default_state(config)
    state["active_batch"] = {
        "batch_index": 0,
        "run_id": "synthetic-krvea-test-batch-0000",
        "status": "proposed",
        "case_ids": [case_id],
        "invalid_preflight_case_ids": [],
        "candidate_csv": str(candidates),
        "worklist_csv": str(tmp_path / "worklist.csv"),
        "proposal_diagnostics": {"empty_reference_count": 34},
        "proposed_empty_reference_count": 34,
    }
    monkeypatch.setattr(
        run_krvea,
        "_task_records",
        lambda *_args, **_kwargs: {case_id: {"status": "completed"}},
    )

    assert run_krvea.finalize_active_batch(config, state, princess_exit_code=0)
    manifest = (
        config.output_directory
        / f"case_{case_id}"
        / qlogehvi.MANIFEST_FILENAME
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "penalized"
    assert payload["failure"]["stage"] == "objective_extraction"
    assert state["completed_batches"][-1]["outcomes"][case_id] == (
        "penalized_missing_manifest"
    )


def test_princess_command_uses_exactly_the_two_remote_maids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = run_krvea.build_princess_command(
        config,
        {
            "worklist_csv": tmp_path / "worklist.csv",
            "run_id": "synthetic-krvea-test-batch-0000",
        },
    )
    selected = [command[index + 1] for index, token in enumerate(command) if token == "--device"]

    assert selected == ["convallariag5", "coconutg2"]
    assert "local" not in selected


def test_main_requires_explicit_run_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fake_campaign(*_args, **_kwargs):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(run_krvea, "run_campaign", fake_campaign)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    result = run_krvea.main(["--output", str(tmp_path / "unused")])

    assert result == 1
    assert not called
