from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from msabp_opt.optimization import krvea
from msabp_opt.optimization import phase2_krvea_relay as legacy
from msabp_opt.optimization import phase2_krvea_tree_relay as relay
from msabp_opt.simulation.distributed.config import DeviceConfig, LaunchMode


def _space_and_plan():
    names = (*relay.ABSOLUTE_PARAMETERS, *(f"corner_{i}" for i in range(10)),
             *(f"{node}.K{k}" for node in ("U1", "U1/L1", "U1/R1", "D1", "D1/L1", "D1/R1")
               for k in (1, 2, 3)))
    baseline = np.array([53, 2, 6, 2.6, 15, 2, 4], dtype=float)
    lower = np.r_[baseline * .9, np.zeros(28)]
    upper = np.r_[baseline * 1.1, np.ones(28)]
    space = relay.WireInputSpace(names, lower, upper)
    plan = {
        "variables": [{"name": name, "lower": lo, "upper": hi}
                      for name, lo, hi in zip(names, lower, upper)],
        "config": {"coordinate_quantum_mm": .01},
        "metrics": {"reference_area_mm2": 2720.2},
        "plan_sha256": "a" * 64, "template_sha256": "b" * 64,
        "source_sha256": {"model": "c" * 64},
    }
    return space, plan


def _request():
    space, plan = _space_and_plan()
    x = np.array([[0.] * 35, [.5] * 35, [1.] * 35])
    area = relay.exact_normalized_area(x, space, relay.exact_area_contract(space, plan=plan))
    full = np.column_stack(([.8, .5, .7], [2., 3., 1.], area))
    return relay.build_request_payload(
        x, full[:, :2], full, np.zeros(3, dtype=bool), space, plan=plan,
        config=krvea.KRVEAConfig(n_variables=35, n_objectives=3, q=2,
                                 inner_evaluations=0, population_size=12, seed=4),
        iteration=0, remaining_expensive_budget=2, compute_device="cpu",
    )


def _fake_fit(monkeypatch):
    calls = []

    def fit(x, y, **kwargs):
        calls.append((x.copy(), y.copy(), kwargs))

        def predict(values):
            return krvea.SurrogatePrediction(
                mean=np.zeros((len(values), 2)), std=np.ones((len(values), 2)) * .2,
            )
        return predict, {"fake": True, "dtype": "float64"}

    monkeypatch.setattr(relay, "_fit_surrogate_predictor", fit)
    return calls


def test_every_absolute_variable_changes_exact_area():
    space, plan = _space_and_plan()
    rows = np.full((8, 35), .5)
    for index in range(7):
        rows[index + 1, index] = .8
    values = relay.exact_normalized_area(rows, space, relay.exact_area_contract(space, plan=plan))[:, 0]
    assert values[0] == pytest.approx(1)
    assert np.all(values[1:] > values[0])
    assert values.dtype == np.float64


def test_exact_area_matches_actual_tree_substrate_and_cst_rounding():
    from scripts.geometry import shapely_antenna_tree_model as model

    space, plan = _space_and_plan()
    contract = relay.exact_area_contract(space, plan=plan)
    rng = np.random.default_rng(329)
    rows = rng.uniform(0, 1, (20, 35))
    # Half-grid cases are sensitive to the builder's arithmetic grouping.
    ties = np.full((7, 35), .5)
    for index in range(7):
        raw = space.denormalize(ties[index])
        raw[index] += .01 if index in (0, 2) else .005
        ties[index] = (raw - space.lower) / (space.upper - space.lower)
    rows = np.vstack((rows, ties))
    result = relay.exact_normalized_area(rows, space, contract)[:, 0]
    # Reuse real defaults for the other geometry dimensions; this comparison
    # deliberately exercises the actual geometry builder, not our formula.
    for row, expected in zip(rows, result):
        params = model.default_params()
        raw = space.denormalize(row[None, :])[0]
        params.update({name: raw[i] for i, name in enumerate(relay.ABSOLUTE_PARAMETERS)})
        shapes = model._build_raw(params, {})
        x0, y0, x1, y1 = shapes["Substrate_Full"].bounds
        width = round(x1 / .01) * .01 - round(x0 / .01) * .01
        height = round((y1 - y0) / .01) * .01
        assert expected == pytest.approx(width * height / 2720.2, rel=1e-14)


def test_exact_area_clips_only_tolerated_unit_boundary_roundoff():
    space, plan = _space_and_plan()
    contract = relay.exact_area_contract(space, plan=plan)
    rows = np.array([[-1e-13] * 35, [1. + 1e-13] * 35])
    np.testing.assert_array_equal(
        relay.exact_normalized_area(rows, space, contract),
        relay.exact_normalized_area(np.clip(rows, 0., 1.), space, contract),
    )


def test_tree_contract_rejects_reordered_or_stale_bounds():
    space, plan = _space_and_plan()
    plan["variables"][1]["upper"] += .1
    with pytest.raises(ValueError, match="bounds differ"):
        relay.exact_area_contract(space, plan=plan)


def test_payload_has_distinct_schema_all_fingerprints_and_float64():
    request = _request()
    assert request["schema_version"] != legacy.REQUEST_SCHEMA_VERSION
    assert request["compute"] == {"device": "cpu", "dtype": "float64"}
    assert request["objective_contract"]["exact_objective"]["coordinate_quantum_mm"] == .01
    assert len(request["input_space"]["parameter_names"]) == 35
    assert set(request["implementation"]) == {
        "krvea_source_sha256", "krvea_relay_source_sha256", "tree_relay_source_sha256",
        "tree_worker_source_sha256",
    }
    assert all(len(value) == 64 for value in request["implementation"].values())
    with pytest.raises(ValueError, match="schema"):
        legacy.run_request_payload(request)


@pytest.mark.parametrize("field", ["area", "schema", "source", "dtype"])
def test_invalid_requests_fail_before_fit(monkeypatch, field):
    request = _request()
    monkeypatch.setattr(relay, "_fit_surrogate_predictor", lambda *_a, **_k: pytest.fail("fit called"))
    if field == "area":
        request["training"]["y_full_minimize"][0][2] += .1
    elif field == "schema":
        request["schema_version"] = legacy.REQUEST_SCHEMA_VERSION
    elif field == "source":
        request["implementation"]["tree_worker_source_sha256"] = "0" * 64
    else:
        request["compute"]["dtype"] = "float32"
    with pytest.raises((ValueError, RuntimeError)):
        relay.run_request_payload(request)


def test_proposal_engine_reused_without_mutating_legacy(monkeypatch):
    calls = _fake_fit(monkeypatch)
    before = (legacy.REQUEST_SCHEMA_VERSION, legacy.RESPONSE_SCHEMA_VERSION,
              legacy.exact_normalized_area, legacy._fit_surrogate_predictor)
    request = _request()
    result = relay.run_request_payload(request)
    assert len(calls) == 1
    assert result.unit_values.shape == (2, 35)
    assert np.all(result.predicted_std[:, 2] == 0)
    expected = relay.exact_normalized_area(
        result.unit_values, relay.input_space_from_payload(request["input_space"]),
        request["objective_contract"]["exact_objective"],
    )
    np.testing.assert_allclose(result.predicted_mean[:, 2], expected[:, 0])
    assert before == (legacy.REQUEST_SCHEMA_VERSION, legacy.RESPONSE_SCHEMA_VERSION,
                      legacy.exact_normalized_area, legacy._fit_surrogate_predictor)


def test_response_area_and_zero_uncertainty_are_verified(monkeypatch):
    _fake_fit(monkeypatch)
    request = _request()
    result = relay.run_request_payload(request)
    payload = relay.response_payload("f" * 64, result)
    kwargs = dict(expected_request_sha256="f" * 64, expected_q=2, expected_dimension=35,
                  input_space=request["input_space"], exact_contract=request["objective_contract"]["exact_objective"])
    relay.result_from_response(payload, **kwargs)
    stale = copy.deepcopy(payload)
    stale["result"]["predicted_mean_minimize"][0][2] += .1
    with pytest.raises(ValueError, match="exact tree geometry"):
        relay.result_from_response(stale, **kwargs)
    payload["result"]["predicted_std"][0][2] = .01
    with pytest.raises(ValueError, match="uncertainty must be zero"):
        relay.result_from_response(payload, **kwargs)


def _validation_request():
    request = _request()
    request["task"] = "validate"
    request["validation"] = {"x_unit": [[.2] * 35, [.7] * 35]}
    return request


def test_validation_never_uses_labels_or_runs_evolution(monkeypatch, tmp_path):
    calls = _fake_fit(monkeypatch)
    request = _validation_request()
    path = relay.write_request(tmp_path / "request.json", request)
    response = tmp_path / "response.json"
    monkeypatch.setattr(krvea.KRVEA, "propose", lambda *_a, **_k: pytest.fail("evolution called"))
    assert not relay.execute_request_file(path, response)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], request["training"]["x_unit"])
    result = relay.validation_from_response(
        json.loads(response.read_text()), expected_request_sha256=relay.sha256_file(path), request=request,
    )
    assert np.asarray(result["predicted_std"]).shape == (2, 3)
    assert np.all(np.asarray(result["predicted_std"])[:, 2] == 0)
    assert result["diagnostics"]["validation_labels_used_in_fit"] is False
    assert relay.execute_request_file(path, response)
    assert len(calls) == 1


def test_validation_guard_rejects_labels_or_reused_training_design():
    request = _validation_request()
    request["validation"]["y_full"] = [[.3, 1, 1], [.4, 1, 1]]
    with pytest.raises(ValueError, match="never labels"):
        relay.validate_request_payload(request)
    request = _validation_request()
    request["validation"]["x_unit"][0] = request["training"]["x_unit"][0]
    with pytest.raises(ValueError, match="overlaps training"):
        relay.validate_request_payload(request)


def test_cached_old_response_is_not_reused(monkeypatch, tmp_path):
    _fake_fit(monkeypatch)
    request = _request()
    path = relay.write_request(tmp_path / "request.json", request)
    response = tmp_path / "response.json"
    result = relay.run_request_payload(request)
    payload = legacy.response_payload(relay.sha256_file(path), result)
    response.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        relay.execute_request_file(path, response)


def test_remote_uses_new_worker_and_bocuda_dll_paths(monkeypatch, tmp_path):
    request = _validation_request()
    path = relay.write_request(tmp_path / "request.json", request)
    device = DeviceConfig(id="coconutg2", enabled=True, launch_mode=LaunchMode.SSH_PROCESS,
                          repo_root="D:/Academic/Proj_MSABP_2",
                          python_path="C:/Users/telecom/miniforge3/envs/maid/python.exe",
                          ssh_target="telecom@coconutg2")
    calls = []
    monkeypatch.setattr(relay, "push_file_atomic", lambda *a, **k: None)
    monkeypatch.setattr(relay, "pull_file_atomic", lambda *a, **k: None)
    monkeypatch.setattr(relay, "run_remote_powershell", lambda *a, **k: (
        calls.append((a, k)) or SimpleNamespace(stdout="")))
    relay._transfer_remote_request(
        device=device, remote=relay.RemoteProposalConfig(), plan_id="tree-001", batch_index=1,
        local_request_path=path, local_response_path=tmp_path / "response.json", expected_task="validate",
    )
    script = calls[0][0][1]
    assert "phase2_krvea_tree_gpu_worker.py" in script
    assert "phase2_krvea_gpu_worker.py" not in script
    assert "CONDA_PREFIX" in script and "Library\\bin" in script
    assert "bocuda" in script and "krvea_tree_gpu" in script


@pytest.mark.parametrize("field,value", [
    ("iteration", -1), ("iteration", True),
    ("remaining_expensive_budget", -1), ("remaining_expensive_budget", 1.5),
    ("previous_empty_reference_count", -1), ("previous_empty_reference_count", 37),
])
def test_invalid_continuation_state_rejected_before_fit(monkeypatch, field, value):
    request = _request()
    request[field] = value
    monkeypatch.setattr(relay, "_fit_surrogate_predictor", lambda *_a, **_k: pytest.fail("fit called"))
    with pytest.raises(ValueError):
        relay.run_request_payload(request)


@pytest.mark.parametrize("field", ["missing", "dimension", "plan", "source", "device", "s11"])
def test_invalid_provenance_or_targets_rejected_without_fit(field):
    request = _request()
    if field == "missing":
        del request["tree_provenance"]
    elif field == "dimension":
        request["tree_provenance"]["dimension"] = 11
    elif field == "plan":
        request["tree_provenance"]["plan_sha256"] = ""
    elif field == "source":
        request["tree_provenance"]["source_sha256"] = {}
    elif field == "device":
        request["compute"]["device"] = " "
    else:
        request["training"]["y_full_minimize"][0][0] = 1.1
        request["training"]["y_expensive_minimize"][0][0] = 1.1
    with pytest.raises(ValueError):
        relay.validate_request_payload(request)


def test_validation_rejects_duplicate_heldout_inputs():
    request = _validation_request()
    request["validation"]["x_unit"][1] = request["validation"]["x_unit"][0]
    with pytest.raises(ValueError, match="duplicate held-out"):
        relay.validate_request_payload(request)


def test_gp_training_requires_distinct_successful_inputs():
    request = _request()
    training = request["training"]
    training["x_unit"][1] = training["x_unit"][0]
    training["y_full_minimize"][1] = training["y_full_minimize"][0]
    training["y_expensive_minimize"][1] = training["y_expensive_minimize"][0]
    training["is_penalty"][2] = True
    with pytest.raises(ValueError, match="distinct successful"):
        relay.validate_request_payload(request)


def test_dynamic_dimension_is_not_hard_coded_to_35(monkeypatch):
    _fake_fit(monkeypatch)
    space, plan = _space_and_plan()
    extra = ("U1/L1/U1.K1", "U1/L1/U1.K2", "U1/L1/U1.K3")
    space = relay.WireInputSpace((*space.names, *extra), np.r_[space.lower, [0., 0., 0.]],
                                np.r_[space.upper, [1., 1., 1.]])
    plan["variables"].extend({"name": name, "lower": 0., "upper": 1.} for name in extra)
    x = np.array([[0.] * 38, [.5] * 38, [1.] * 38])
    area = relay.exact_normalized_area(x, space, relay.exact_area_contract(space, plan=plan))
    full = np.column_stack(([.8, .5, .7], [2., 3., 1.], area))
    request = relay.build_request_payload(
        x, full[:, :2], full, np.zeros(3, dtype=bool), space, plan=plan,
        config=krvea.KRVEAConfig(n_variables=38, n_objectives=3, q=2,
                                 inner_evaluations=0, population_size=12, seed=4),
        iteration=0, remaining_expensive_budget=2, compute_device="cpu",
    )
    result = relay.run_request_payload(request)
    assert result.unit_values.shape == (2, 38)
    assert request["tree_provenance"]["dimension"] == 38


@pytest.mark.parametrize("field,value", [
    ("validation_labels_used_in_fit", True),
    ("gp_training_observations", 5), ("validation_observations", 3),
])
def test_validation_response_rejects_inconsistent_diagnostics(monkeypatch, field, value):
    _fake_fit(monkeypatch)
    request = _validation_request()
    prediction = relay.fit_predict_physical(request, request["validation"]["x_unit"])
    prediction["diagnostics"][field] = value
    response = {"schema_version": relay.RESPONSE_SCHEMA_VERSION, "status": "completed",
                "task": "validate", "request_sha256": "f" * 64, "validation_result": prediction}
    with pytest.raises(ValueError, match="diagnostics disagree"):
        relay.validation_from_response(response, expected_request_sha256="f" * 64, request=request)


def test_proposal_response_rejects_unbounded_s11(monkeypatch):
    _fake_fit(monkeypatch)
    request = _request()
    response = relay.response_payload("f" * 64, relay.run_request_payload(request))
    response["result"]["predicted_mean_minimize"][0][0] = 1.1
    with pytest.raises(ValueError, match="S11 predictions"):
        relay.result_from_response(response, expected_request_sha256="f" * 64,
                                   expected_q=2, expected_dimension=35)


def test_zero_budget_response_round_trips_and_reuses_cache(monkeypatch, tmp_path):
    calls = _fake_fit(monkeypatch)
    request = _request()
    request["remaining_expensive_budget"] = 0
    path = relay.write_request(tmp_path / "request.json", request)
    response = tmp_path / "response.json"
    assert not relay.execute_request_file(path, response)
    assert relay.execute_request_file(path, response)
    result = relay.result_from_response(
        json.loads(response.read_text()), expected_request_sha256=relay.sha256_file(path),
        expected_q=0, expected_dimension=35, input_space=request["input_space"],
        exact_contract=request["objective_contract"]["exact_objective"],
    )
    assert result.unit_values.shape == (0, 35)
    assert result.predicted_mean.shape == (0, 3)
    assert len(calls) == 1


def test_worker_validate_only_does_not_execute_request(monkeypatch, tmp_path, capsys):
    from scripts.optimization import phase2_krvea_tree_gpu_worker as worker

    request = _validation_request()
    path = relay.write_request(tmp_path / "request.json", request)
    response = tmp_path / "response.json"
    monkeypatch.setattr(relay, "execute_request_file", lambda *_a: pytest.fail("execution called"))
    assert worker.main(["--request", str(path), "--response", str(response), "--validate-only"]) == 0
    assert "no fit performed" in capsys.readouterr().out
    assert not response.exists()
