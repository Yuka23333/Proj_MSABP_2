from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msabp_opt.optimization import krvea_data


def _parameters() -> dict[str, float]:
    space = krvea_data.authoritative_input_space()
    return {
        **dict(zip(space.names, space.nominal.tolist())),
        **krvea_data.FIXED_PARAMETER_VALUES,
    }


def _write_curve(path: Path, rows: list[tuple[float, float]]) -> None:
    path.write_text(
        "Frequency / GHz\tValue\n"
        + "\n".join(f"{frequency} {value}" for frequency, value in rows),
        encoding="utf-8",
    )


def _write_completed_case(root: Path, case_id: str) -> Path:
    case = root / f"case_{case_id}"
    case.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "case_id": case_id,
        "status": "completed",
        "parameters": _parameters(),
        "artifacts": {
            "s11": {"path": "S11.csv"},
            "tot_eff": {"path": "Tot_Eff.csv"},
            "farfield_source": {"path": krvea_data.FARFIELD_FILENAME},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_curve(case / "S11.csv", [(3.1, -20.0), (4.0, -6.020599913), (4.8, -10.0)])
    # The middle sample is >1 after dB conversion and must be removed.
    _write_curve(case / "Tot_Eff.csv", [(3.1, -3.010299957), (4.0, 3.010299957), (4.8, -3.010299957)])
    (case / krvea_data.FARFIELD_FILENAME).write_bytes(b"synthetic ffs")
    return case


def _write_penalty_case(root: Path, case_id: str) -> Path:
    case = root / f"case_{case_id}"
    case.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "status": "penalized",
        "parameters": _parameters(),
        "optimization_objectives": {
            krvea_data.WORST_S11_COLUMN: 1.0,
            krvea_data.MEAN_TOT_EFF_COLUMN: 0.0,
        },
        "artifacts": {},
    }
    (case / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return case


def test_authoritative_11d_order_bounds_and_reference_area() -> None:
    space = krvea_data.authoritative_input_space()

    assert space.names == krvea_data.ACTIVE_PARAMETER_NAMES
    assert space.lower.tolist() == pytest.approx(
        [47.7, 5.4, 2.34, 13.5, 0.46818181818181814, 0.7,
         0.2617647058823529, 0.0, 0.05, 0.05, 0.05]
    )
    assert space.upper.tolist() == pytest.approx(
        [58.3, 6.6, 2.86, 16.5, 0.7681818181818181, 1.0,
         0.5617647058823529, 0.3, 1.0, 1.0, 1.0]
    )
    assert krvea_data.reference_substrate_area_mm2() == pytest.approx(2720.2)
    assert space.exact_normalized_area(space.nominal) == pytest.approx(1.0)


def test_cap_scalar_is_cached_and_keeps_linear_and_dbi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffs = tmp_path / krvea_data.FARFIELD_FILENAME
    ffs.write_bytes(b"stable synthetic identity")
    calls = 0

    def fake_uncached(*args: object, **kwargs: object) -> float:
        nonlocal calls
        calls += 1
        return 2.0

    monkeypatch.setattr(krvea_data, "_uncached_cap_gain_linear", fake_uncached)
    first = krvea_data.cap_gain_scalar(ffs, cache_directory=tmp_path / "cache")
    second = krvea_data.cap_gain_scalar(ffs, cache_directory=tmp_path / "cache")

    assert first == pytest.approx((2.0, 3.01029995664, False))
    assert second == pytest.approx((2.0, 3.01029995664, True))
    assert calls == 1


def test_collect_skips_incomplete_and_dataset_prefers_completed_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    _write_penalty_case(source_a, "penalty")
    _write_completed_case(source_b, "complete")
    incomplete = source_b / "case_incomplete"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "incomplete",
                "status": "completed",
                "parameters": _parameters(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        krvea_data,
        "cap_gain_scalar",
        lambda *args, **kwargs: (0.5, -3.01029995664, False),
    )
    skipped: list[str] = []

    observations = krvea_data.collect_observations(
        [source_a, source_b],
        skipped_incomplete=skipped,
    )
    dataset = krvea_data.build_dataset(observations)

    assert len(observations) == 2
    assert len(skipped) == 1
    assert "missing objective artifact" in skipped[0]
    assert dataset.x_raw.shape == (1, 11)
    assert dataset.x_unit.shape == (1, 11)
    assert dataset.objective_names == (
        krvea_data.WORST_S11_COLUMN,
        krvea_data.TOT_EFF_LOSS_COLUMN,
        krvea_data.NORMALIZED_AREA_COLUMN,
        krvea_data.CAP_GAIN_DBI_COLUMN,
    )
    assert dataset.objectives[0] == pytest.approx(
        [0.5, 0.5, 1.0, -3.01029995664], rel=1e-8
    )
    assert dataset.metadata.loc[0, "replicate_count"] == 1
    assert dataset.metadata.loc[0, "discarded_penalty_count"] == 1
    assert bool(dataset.metadata.loc[0, "has_completed_result"])


def test_manifest_from_old_variable_space_is_rejected(tmp_path: Path) -> None:
    case = _write_penalty_case(tmp_path, "wrong-fixed")
    payload = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    payload["parameters"]["SLOT_MAIN_HEIGHT"] = 2.5
    (case / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not in the current fixed 11-D design space"):
        krvea_data.collect_observations([tmp_path])


def test_manifest_missing_fixed_parameter_is_rejected(tmp_path: Path) -> None:
    case = _write_penalty_case(tmp_path, "missing-fixed")
    payload = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    del payload["parameters"]["SLOT_MAIN_HEIGHT"]
    (case / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fixed parameters"):
        krvea_data.collect_observations([tmp_path])


def test_cap_frequency_average_is_arithmetic_mean_in_linear_power(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.postprocessing import cap_gain as cap_module

    monkeypatch.setattr(cap_module, "parse_ffs", lambda path: {"synthetic": True})
    monkeypatch.setattr(
        cap_module,
        "cap_gain",
        lambda ffs, caps: pd.DataFrame(
            {
                "freq_ghz": [3.1, 4.0, 4.8],
                "theta_max_deg": [15, 15, 15],
                "G_realized_dBi": [0.0, 10.0, 0.0],
            }
        ),
    )
    linear = krvea_data._uncached_cap_gain_linear(
        tmp_path / "synthetic.ffs",
        band_ghz=(3.1, 4.8),
        theta_max_deg=15,
    )
    assert linear == pytest.approx(4.0)


def test_penalty_is_bad_in_all_four_objectives_but_keeps_physical_area(
    tmp_path: Path,
) -> None:
    case = _write_penalty_case(tmp_path, "penalty")

    observations = krvea_data.collect_observations([case])
    dataset = krvea_data.build_dataset(observations)

    assert observations.loc[0, krvea_data.AREA_COLUMN] == pytest.approx(2720.2)
    assert dataset.objectives[0] == pytest.approx(
        [1.0, 1.0, krvea_data.PENALTY_NORMALIZED_AREA, 10.0]
    )
