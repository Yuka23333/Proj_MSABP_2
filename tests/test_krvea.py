from __future__ import annotations

import numpy as np
import pytest

from msabp_opt.optimization.krvea import (
    KRVEA,
    KRVEAConfig,
    SurrogatePrediction,
    associate_by_apd,
    das_dennis_reference_directions,
    normalize_objectives,
    sbx_polynomial_offspring,
    standardize_uncertainty,
)


class SmoothPredictor:
    def __call__(self, unit_x: np.ndarray) -> SurrogatePrediction:
        first = np.sum((unit_x - 0.2) ** 2, axis=1)
        second = np.sum((unit_x - 0.8) ** 2, axis=1)
        third = np.mean(unit_x, axis=1)
        mean = np.column_stack((first, second, third))
        std = 0.02 + 0.08 * np.column_stack(
            (unit_x[:, 0], 1.0 - unit_x[:, 1], unit_x[:, 2])
        )
        return SurrogatePrediction(mean=mean, std=std)


def exact_area(unit_x: np.ndarray) -> np.ndarray:
    return (0.5 + unit_x[:, :1]) * (0.75 + unit_x[:, 1:2])


def _archive(seed: int = 19) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.default_rng(seed).random((36, 5))
    prediction = SmoothPredictor()(x)
    y = np.zeros((len(x), 4))
    y[:, (0, 1, 3)] = prediction.mean
    y[:, 2:3] = exact_area(x)
    return x, y


def _optimizer(seed: int = 7, inner_evaluations: int = 160) -> KRVEA:
    return KRVEA(
        KRVEAConfig(
            n_variables=5,
            n_objectives=4,
            reference_partitions=3,
            population_size=28,
            q=4,
            inner_evaluations=inner_evaluations,
            seed=seed,
        ),
        SmoothPredictor(),
        expensive_objective_indices=(0, 1, 3),
        exact_objective=exact_area,
        exact_objective_indices=(2,),
    )


def test_four_objective_seven_partition_reference_set_has_120_directions() -> None:
    directions = das_dennis_reference_directions(4, 7)

    assert directions.shape == (120, 4)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all(directions >= 0.0)


def test_sbx_and_polynomial_mutation_are_bounded_and_deterministic() -> None:
    parents = np.vstack((np.zeros((2, 6)), np.ones((2, 6)), np.full((2, 6), 0.5)))

    first = sbx_polynomial_offspring(
        parents, 101, np.random.default_rng(123), mutation_probability=0.8
    )
    second = sbx_polynomial_offspring(
        parents, 101, np.random.default_rng(123), mutation_probability=0.8
    )

    assert first.shape == (101, 6)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_objective_normalization_and_apd_handle_constant_and_ideal_rows() -> None:
    objectives = np.asarray(
        [
            [1.0, 4.0, 2.0, 9.0],
            [1.0, 4.0, 2.0, 9.0],
            [2.0, 4.0, 3.0, 10.0],
        ]
    )
    directions = das_dennis_reference_directions(4, 2)

    normalized, minimum, scale = normalize_objectives(objectives)
    association, apd, apd_minimum, apd_scale, gamma = associate_by_apd(
        objectives, directions, penalty=4.0
    )

    assert np.isfinite(normalized).all()
    assert scale[1] == 1.0
    assert np.array_equal(minimum, apd_minimum)
    assert np.array_equal(scale, apd_scale)
    assert np.isfinite(apd).all()
    assert np.isfinite(gamma).all()
    assert association[0] == 0
    assert apd[0] == 0.0


def test_uncertainty_is_scaled_to_comparable_objective_units() -> None:
    std = np.asarray([[0.1, 100.0, 2.0, 0.5]])
    scale = np.asarray([1.0, 1000.0, 20.0, 5.0])

    assert np.allclose(standardize_uncertainty(std, scale), 0.1)


def test_exact_objective_is_inserted_with_zero_uncertainty() -> None:
    optimizer = _optimizer(inner_evaluations=0)
    points = np.random.default_rng(3).random((8, 5))

    prediction = optimizer._predict(points)

    assert np.allclose(prediction.mean[:, 2:3], exact_area(points))
    assert np.all(prediction.std[:, 2] == 0.0)
    assert np.all(prediction.std[:, (0, 1, 3)] > 0.0)


def test_proposal_is_deterministic_unique_and_respects_q() -> None:
    archive_x, archive_y = _archive()

    first = _optimizer().propose(
        archive_x,
        archive_y,
        remaining_expensive_budget=128,
        previous_empty_reference_count=0,
    )
    second = _optimizer().propose(
        archive_x,
        archive_y,
        remaining_expensive_budget=128,
        previous_empty_reference_count=0,
    )

    assert first.unit_x.shape == (4, 5)
    assert np.array_equal(first.unit_x, second.unit_x)
    assert len(np.unique(np.round(first.unit_x, 12), axis=0)) == 4
    assert np.all((first.unit_x >= 0.0) & (first.unit_x <= 1.0))
    nearest_archive = np.min(
        np.max(np.abs(first.unit_x[:, None, :] - archive_x[None, :, :]), axis=2),
        axis=1,
    )
    assert np.all(nearest_archive > 1e-10)
    assert first.diagnostics.inner_evaluations_used == 160
    assert first.diagnostics.proposed_count == 4
    assert first.diagnostics.mode == "exploration"


def test_remaining_expensive_budget_is_a_strict_batch_cap() -> None:
    archive_x, archive_y = _archive()
    optimizer = _optimizer()

    limited = optimizer.propose(
        archive_x, archive_y, remaining_expensive_budget=2
    )
    exhausted = optimizer.propose(
        archive_x, archive_y, remaining_expensive_budget=0
    )

    assert limited.unit_x.shape == (2, 5)
    assert limited.diagnostics.proposed_count == 2
    assert exhausted.unit_x.shape == (0, 5)
    assert exhausted.predicted_mean.shape == (0, 4)


def test_degenerate_archive_uses_deterministic_unique_fallback() -> None:
    class PessimisticPredictor:
        def __call__(self, unit_x: np.ndarray) -> SurrogatePrediction:
            return SurrogatePrediction(
                mean=np.full((len(unit_x), 3), 100.0),
                std=np.full((len(unit_x), 3), 0.1),
            )

    rng = np.random.default_rng(91)
    archive_x = rng.random((20, 5))
    archive_y = np.zeros((20, 4), dtype=np.float64)
    optimizer = KRVEA(
        KRVEAConfig(
            n_variables=5,
            n_objectives=4,
            reference_partitions=2,
            population_size=12,
            q=4,
            inner_evaluations=0,
            seed=11,
        ),
        PessimisticPredictor(),
        expensive_objective_indices=(0, 1, 3),
        exact_objective=exact_area,
        exact_objective_indices=(2,),
    )

    first = optimizer.propose(archive_x, archive_y, remaining_expensive_budget=4)
    second = optimizer.propose(archive_x, archive_y, remaining_expensive_budget=4)

    assert first.unit_x.shape == (4, 5)
    assert np.array_equal(first.unit_x, second.unit_x)
    assert first.diagnostics.random_fallback_count == 4
    assert np.all(
        np.min(
            np.max(np.abs(first.unit_x[:, None] - archive_x[None, :]), axis=2),
            axis=1,
        )
        > 1e-10
    )


def test_invalid_surrogate_shapes_and_nonfinite_values_fail_fast() -> None:
    class BadPredictor:
        def __call__(self, unit_x: np.ndarray) -> SurrogatePrediction:
            return SurrogatePrediction(
                mean=np.zeros((len(unit_x), 2)),
                std=np.zeros((len(unit_x), 2)),
            )

    optimizer = KRVEA(
        KRVEAConfig(3, 4, inner_evaluations=1),
        BadPredictor(),
        expensive_objective_indices=(0, 1, 3),
        exact_objective=lambda x: np.sum(x, axis=1, keepdims=True),
        exact_objective_indices=(2,),
    )
    archive_x = np.full((2, 3), 0.5)
    archive_y = np.ones((2, 4))

    with pytest.raises(ValueError, match="surrogate prediction shape"):
        optimizer.propose(archive_x, archive_y, remaining_expensive_budget=1)

    with pytest.raises(ValueError, match="finite"):
        normalize_objectives([[0.0, np.nan], [1.0, 2.0]])
