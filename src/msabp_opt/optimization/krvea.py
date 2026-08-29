"""Deterministic, clean-room building blocks for K-RVEA campaigns.

The implementation follows the published *behaviour* of K-RVEA: an RVEA
search is performed on surrogate predictions and model management selects a
small batch for expensive evaluation.  It is intentionally independent of the
authors' MATLAB source and only depends on NumPy.

All decision vectors use the unit cube and all objective values use the
minimisation convention.  Callers are responsible for changing the sign of
maximisation objectives before constructing an :class:`KRVEA` instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SurrogatePrediction:
    """Mean and standard deviation for the expensive objectives."""

    mean: FloatArray
    std: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.ndim != 2 or std.shape != mean.shape:
            raise ValueError("surrogate mean and std must be equal-size 2-D arrays")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("surrogate predictions must be finite")
        if np.any(std < 0.0):
            raise ValueError("surrogate standard deviations must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)


class SurrogatePredictor(Protocol):
    """Callback that predicts only the expensive objective columns."""

    def __call__(self, unit_x: FloatArray) -> SurrogatePrediction:
        ...


ExactObjective = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class KRVEAConfig:
    """Configuration of one surrogate-evolution and proposal step."""

    n_variables: int
    n_objectives: int
    reference_partitions: int = 7
    q: int = 4
    inner_evaluations: int = 10_000
    population_size: int | None = None
    seed: int = 0
    crossover_probability: float = 0.8
    crossover_eta: float = 20.0
    mutation_probability: float | None = None
    mutation_eta: float = 20.0
    apd_alpha: float = 1.0
    empty_growth_fraction: float = 0.05
    uniqueness_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.n_variables <= 0:
            raise ValueError("n_variables must be positive")
        if self.n_objectives < 2:
            raise ValueError("n_objectives must be at least two")
        if self.reference_partitions <= 0:
            raise ValueError("reference_partitions must be positive")
        if self.q <= 0:
            raise ValueError("q must be positive")
        if self.inner_evaluations < 0:
            raise ValueError("inner_evaluations must be non-negative")
        if self.population_size is not None and self.population_size <= 0:
            raise ValueError("population_size must be positive when supplied")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must lie in [0, 1]")
        if self.crossover_eta <= 0.0 or self.mutation_eta <= 0.0:
            raise ValueError("variation distribution indices must be positive")
        mutation_probability = self.mutation_probability
        if mutation_probability is None:
            mutation_probability = 1.0 / self.n_variables
            object.__setattr__(self, "mutation_probability", mutation_probability)
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1]")
        if self.apd_alpha <= 0.0:
            raise ValueError("apd_alpha must be positive")
        if self.empty_growth_fraction < 0.0:
            raise ValueError("empty_growth_fraction must be non-negative")
        if self.uniqueness_tolerance < 0.0:
            raise ValueError("uniqueness_tolerance must be non-negative")


@dataclass(frozen=True)
class SelectionDiagnostics:
    """Numerical summary of one APD environmental selection."""

    active_reference_count: int
    empty_reference_count: int
    objective_min: tuple[float, ...]
    objective_scale: tuple[float, ...]
    selected_apd: tuple[float, ...]


@dataclass(frozen=True)
class ProposalDiagnostics:
    """Model-management details useful for persistence and debugging."""

    seed: int
    inner_budget: int
    inner_evaluations_used: int
    generations: int
    reference_direction_count: int
    population_size: int
    active_reference_count: int
    empty_reference_count: int
    previous_empty_reference_count: int
    empty_reference_growth: int
    mode: str
    requested_q: int
    expensive_budget_remaining: int
    proposed_count: int
    random_fallback_count: int = 0
    selected_reference_indices: tuple[int, ...] = field(default_factory=tuple)
    selected_apd: tuple[float, ...] = field(default_factory=tuple)
    selected_mean_std: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProposalBatch:
    """Unique unit-cube candidates selected for expensive evaluation."""

    unit_x: FloatArray
    predicted_mean: FloatArray
    predicted_std: FloatArray
    diagnostics: ProposalDiagnostics


def das_dennis_reference_directions(
    n_objectives: int,
    partitions: int,
) -> FloatArray:
    """Return L2-normalised Das-Dennis simplex directions.

    The number of rows is ``comb(n_objectives + partitions - 1,
    n_objectives - 1)``.  For four objectives and seven partitions this is
    120, matching the reference set used by the original K-RVEA experiments.
    """

    if n_objectives < 2:
        raise ValueError("n_objectives must be at least two")
    if partitions <= 0:
        raise ValueError("partitions must be positive")

    compositions: list[list[int]] = []

    def visit(prefix: list[int], remaining: int, dimensions_left: int) -> None:
        if dimensions_left == 1:
            compositions.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, dimensions_left - 1)

    visit([], partitions, n_objectives)
    directions = np.asarray(compositions, dtype=np.float64) / float(partitions)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms


def normalize_objectives(values: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Min-max normalise objective columns with safe constant-column handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("objective values must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError("objective values must be finite")
    minimum = np.min(array, axis=0)
    span = np.max(array, axis=0) - minimum
    safe_span = np.where(span > np.finfo(np.float64).eps, span, 1.0)
    return (array - minimum) / safe_span, minimum, safe_span


def standardize_uncertainty(std: ArrayLike, objective_scale: ArrayLike) -> FloatArray:
    """Express posterior standard deviations in normalized objective units."""

    uncertainty = np.asarray(std, dtype=np.float64)
    scale = np.asarray(objective_scale, dtype=np.float64)
    if uncertainty.ndim != 2:
        raise ValueError("surrogate standard deviations must be a 2-D array")
    if scale.shape != (uncertainty.shape[1],):
        raise ValueError("objective scale does not match uncertainty columns")
    if not np.isfinite(uncertainty).all() or np.any(uncertainty < 0.0):
        raise ValueError("surrogate standard deviations must be finite and non-negative")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("objective scale must be finite and positive")
    return uncertainty / scale


def _reference_angles(reference_directions: FloatArray) -> FloatArray:
    cosine = np.clip(
        reference_directions @ reference_directions.T,
        -1.0,
        1.0,
    )
    np.fill_diagonal(cosine, -np.inf)
    nearest_cosine = np.max(cosine, axis=1)
    return np.maximum(
        np.arccos(np.clip(nearest_cosine, -1.0, 1.0)),
        np.finfo(np.float64).eps,
    )


def associate_by_apd(
    objective_values: ArrayLike,
    reference_directions: ArrayLike,
    *,
    penalty: float,
) -> tuple[NDArray[np.int64], FloatArray, FloatArray, FloatArray, FloatArray]:
    """Associate points to reference directions and calculate robust APD values."""

    if penalty < 0.0 or not math.isfinite(penalty):
        raise ValueError("APD penalty must be finite and non-negative")
    values, minimum, scale = normalize_objectives(objective_values)
    directions = np.asarray(reference_directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != values.shape[1]:
        raise ValueError("reference directions do not match objective count")
    if not np.isfinite(directions).all():
        raise ValueError("reference directions must be finite")
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(direction_norms <= np.finfo(np.float64).eps):
        raise ValueError("reference directions must have non-zero norm")
    directions = directions / direction_norms

    radial_distance = np.linalg.norm(values, axis=1)
    safe_distance = np.where(radial_distance > np.finfo(np.float64).eps,
                             radial_distance, 1.0)
    unit_values = values / safe_distance[:, None]
    cosine = np.clip(unit_values @ directions.T, -1.0, 1.0)
    zero_rows = radial_distance <= np.finfo(np.float64).eps
    if np.any(zero_rows):
        # The ideal point has no angular direction.  Assigning it to the first
        # direction is deterministic and its zero radial distance makes APD 0.
        cosine[zero_rows] = -1.0
        cosine[zero_rows, 0] = 1.0
    association = np.argmax(cosine, axis=1).astype(np.int64, copy=False)
    angle = np.arccos(np.clip(cosine[np.arange(len(values)), association], -1.0, 1.0))
    gamma = _reference_angles(directions)
    apd = radial_distance * (
        1.0 + penalty * angle / gamma[association]
    )
    return association, apd, minimum, scale, gamma


def environmental_selection(
    unit_x: ArrayLike,
    objective_values: ArrayLike,
    reference_directions: ArrayLike,
    *,
    population_size: int,
    penalty: float,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64], SelectionDiagnostics]:
    """Select a population using one APD winner per active direction first."""

    x = np.asarray(unit_x, dtype=np.float64)
    objectives = np.asarray(objective_values, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(objectives):
        raise ValueError("decision and objective populations must have equal rows")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    association, apd, minimum, scale, _ = associate_by_apd(
        objectives,
        reference_directions,
        penalty=penalty,
    )
    direction_count = len(np.asarray(reference_directions))
    selected: list[int] = []
    for direction_index in range(direction_count):
        members = np.flatnonzero(association == direction_index)
        if members.size:
            local = members[np.argmin(apd[members])]
            selected.append(int(local))

    # Sparse surrogate populations may not activate every reference vector.
    # Fill remaining slots by APD without discarding the diversity-first set.
    target = min(population_size, len(x))
    if len(selected) < target:
        selected_set = set(selected)
        for index in np.argsort(apd, kind="stable"):
            integer_index = int(index)
            if integer_index not in selected_set:
                selected.append(integer_index)
                selected_set.add(integer_index)
                if len(selected) == target:
                    break
    elif len(selected) > target:
        selected = sorted(selected, key=lambda index: (apd[index], index))[:target]

    indices = np.asarray(selected, dtype=np.int64)
    active_count = int(np.unique(association).size)
    diagnostics = SelectionDiagnostics(
        active_reference_count=active_count,
        empty_reference_count=direction_count - active_count,
        objective_min=tuple(float(value) for value in minimum),
        objective_scale=tuple(float(value) for value in scale),
        selected_apd=tuple(float(value) for value in apd[indices]),
    )
    return x[indices], objectives[indices], indices, diagnostics


def sbx_polynomial_offspring(
    parents: ArrayLike,
    n_offspring: int,
    rng: np.random.Generator,
    *,
    crossover_probability: float = 1.0,
    crossover_eta: float = 30.0,
    mutation_probability: float | None = None,
    mutation_eta: float = 20.0,
) -> FloatArray:
    """Generate bounded real-valued offspring with SBX and polynomial mutation."""

    population = np.asarray(parents, dtype=np.float64)
    if population.ndim != 2 or population.shape[0] == 0:
        raise ValueError("parents must be a non-empty 2-D array")
    if not np.isfinite(population).all() or np.any((population < 0.0) | (population > 1.0)):
        raise ValueError("parents must be finite and lie in the unit cube")
    if n_offspring < 0:
        raise ValueError("n_offspring must be non-negative")
    if n_offspring == 0:
        return np.empty((0, population.shape[1]), dtype=np.float64)
    if not 0.0 <= crossover_probability <= 1.0:
        raise ValueError("crossover_probability must lie in [0, 1]")
    if crossover_eta <= 0.0 or mutation_eta <= 0.0:
        raise ValueError("distribution indices must be positive")
    dimensions = population.shape[1]
    mutation_probability = (
        1.0 / dimensions if mutation_probability is None else mutation_probability
    )
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("mutation_probability must lie in [0, 1]")

    children: list[FloatArray] = []
    epsilon = np.finfo(np.float64).eps
    while len(children) < n_offspring:
        parent_indices = rng.integers(0, len(population), size=2)
        first = population[parent_indices[0]].copy()
        second = population[parent_indices[1]].copy()
        child_a, child_b = first.copy(), second.copy()
        if rng.random() <= crossover_probability:
            for column in range(dimensions):
                if rng.random() > 0.5 or abs(first[column] - second[column]) <= epsilon:
                    continue
                low, high = sorted((first[column], second[column]))
                random_value = rng.random()
                beta = 1.0 + 2.0 * low / (high - low)
                alpha = 2.0 - beta ** -(crossover_eta + 1.0)
                if random_value <= 1.0 / alpha:
                    beta_q = (random_value * alpha) ** (1.0 / (crossover_eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - random_value * alpha)) ** (
                        1.0 / (crossover_eta + 1.0)
                    )
                value_a = 0.5 * ((low + high) - beta_q * (high - low))
                beta = 1.0 + 2.0 * (1.0 - high) / (high - low)
                alpha = 2.0 - beta ** -(crossover_eta + 1.0)
                if random_value <= 1.0 / alpha:
                    beta_q = (random_value * alpha) ** (1.0 / (crossover_eta + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - random_value * alpha)) ** (
                        1.0 / (crossover_eta + 1.0)
                    )
                value_b = 0.5 * ((low + high) + beta_q * (high - low))
                if rng.random() <= 0.5:
                    value_a, value_b = value_b, value_a
                child_a[column] = np.clip(value_a, 0.0, 1.0)
                child_b[column] = np.clip(value_b, 0.0, 1.0)

        for child in (child_a, child_b):
            mutation_mask = rng.random(dimensions) <= mutation_probability
            for column in np.flatnonzero(mutation_mask):
                value = child[column]
                random_value = rng.random()
                mutation_power = 1.0 / (mutation_eta + 1.0)
                if random_value <= 0.5:
                    delta = value
                    term = 2.0 * random_value + (1.0 - 2.0 * random_value) * (
                        1.0 - delta
                    ) ** (mutation_eta + 1.0)
                    delta_q = term**mutation_power - 1.0
                else:
                    delta = 1.0 - value
                    term = 2.0 * (1.0 - random_value) + 2.0 * (
                        random_value - 0.5
                    ) * (1.0 - delta) ** (mutation_eta + 1.0)
                    delta_q = 1.0 - term**mutation_power
                child[column] = np.clip(value + delta_q, 0.0, 1.0)
            children.append(child)
            if len(children) == n_offspring:
                break
    return np.asarray(children, dtype=np.float64)


def _as_prediction(value: object) -> SurrogatePrediction:
    if isinstance(value, SurrogatePrediction):
        return value
    if isinstance(value, Sequence) and len(value) == 2:
        return SurrogatePrediction(value[0], value[1])  # type: ignore[arg-type]
    raise TypeError("predictor must return SurrogatePrediction or (mean, std)")


def _is_duplicate(candidate: FloatArray, existing: FloatArray, tolerance: float) -> bool:
    if existing.size == 0:
        return False
    return bool(np.any(np.max(np.abs(existing - candidate), axis=1) <= tolerance))


def _partition_directions(
    directions: FloatArray,
    cluster_count: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Small deterministic-with-seed spherical k-means implementation."""

    if cluster_count <= 0 or cluster_count > len(directions):
        raise ValueError("invalid direction cluster count")
    first = int(rng.integers(0, len(directions)))
    centres = [directions[first]]
    while len(centres) < cluster_count:
        similarities = directions @ np.asarray(centres).T
        distance = 1.0 - np.max(similarities, axis=1)
        centres.append(directions[int(np.argmax(distance))])
    centre_array = np.asarray(centres, dtype=np.float64)

    labels = np.zeros(len(directions), dtype=np.int64)
    for _ in range(30):
        new_labels = np.argmax(directions @ centre_array.T, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for cluster in range(cluster_count):
            members = directions[labels == cluster]
            if len(members) == 0:
                # Farthest point from all other centres repairs an empty cluster.
                similarities = directions @ centre_array.T
                centre_array[cluster] = directions[int(np.argmin(np.max(similarities, axis=1)))]
                continue
            centre = np.mean(members, axis=0)
            norm = np.linalg.norm(centre)
            centre_array[cluster] = centre / max(norm, np.finfo(np.float64).eps)
    return labels


class KRVEA:
    """Surrogate RVEA evolution plus K-RVEA-style model management."""

    def __init__(
        self,
        config: KRVEAConfig,
        predictor: SurrogatePredictor,
        *,
        expensive_objective_indices: Sequence[int] | None = None,
        exact_objective: ExactObjective | None = None,
        exact_objective_indices: Sequence[int] = (),
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.reference_directions = das_dennis_reference_directions(
            config.n_objectives,
            config.reference_partitions,
        )
        expensive = (
            tuple(range(config.n_objectives))
            if expensive_objective_indices is None
            else tuple(int(index) for index in expensive_objective_indices)
        )
        exact = tuple(int(index) for index in exact_objective_indices)
        if len(set(expensive)) != len(expensive) or len(set(exact)) != len(exact):
            raise ValueError("objective index groups must not contain duplicates")
        if set(expensive) & set(exact):
            raise ValueError("expensive and exact objective indices must be disjoint")
        if set(expensive) | set(exact) != set(range(config.n_objectives)):
            raise ValueError("objective index groups must cover every objective exactly once")
        if exact and exact_objective is None:
            raise ValueError("exact objective indices require an exact_objective callback")
        if not exact and exact_objective is not None:
            raise ValueError("exact_objective callback requires exact objective indices")
        self.expensive_objective_indices = expensive
        self.exact_objective_indices = exact
        self.exact_objective = exact_objective

    @property
    def population_size(self) -> int:
        return self.config.population_size or len(self.reference_directions)

    def _predict(self, unit_x: FloatArray) -> SurrogatePrediction:
        x = np.asarray(unit_x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.config.n_variables:
            raise ValueError("candidate array has the wrong shape")
        if not np.isfinite(x).all() or np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("candidate values must be finite and lie in the unit cube")
        expensive = _as_prediction(self.predictor(x))
        expected = (len(x), len(self.expensive_objective_indices))
        if expensive.mean.shape != expected:
            raise ValueError(
                f"surrogate prediction shape {expensive.mean.shape} does not match {expected}"
            )
        mean = np.zeros((len(x), self.config.n_objectives), dtype=np.float64)
        std = np.zeros_like(mean)
        mean[:, self.expensive_objective_indices] = expensive.mean
        std[:, self.expensive_objective_indices] = expensive.std
        if self.exact_objective_indices:
            assert self.exact_objective is not None
            exact = np.asarray(self.exact_objective(x), dtype=np.float64)
            exact_expected = (len(x), len(self.exact_objective_indices))
            if exact.shape != exact_expected or not np.isfinite(exact).all():
                raise ValueError(
                    f"exact objective shape must be {exact_expected} and finite"
                )
            mean[:, self.exact_objective_indices] = exact
        return SurrogatePrediction(mean=mean, std=std)

    def propose(
        self,
        archive_x: ArrayLike,
        archive_y: ArrayLike,
        *,
        remaining_expensive_budget: int,
        previous_empty_reference_count: int | None = None,
    ) -> ProposalBatch:
        """Return at most ``min(q, remaining_expensive_budget)`` unique candidates."""

        config = self.config
        x_archive = np.asarray(archive_x, dtype=np.float64)
        y_archive = np.asarray(archive_y, dtype=np.float64)
        if x_archive.ndim != 2 or x_archive.shape[1] != config.n_variables:
            raise ValueError("archive_x has the wrong shape")
        if y_archive.shape != (len(x_archive), config.n_objectives):
            raise ValueError("archive_y has the wrong shape")
        if len(x_archive) == 0:
            raise ValueError("archive must contain at least one evaluated point")
        if not np.isfinite(x_archive).all() or np.any((x_archive < 0.0) | (x_archive > 1.0)):
            raise ValueError("archive_x must be finite and lie in the unit cube")
        if not np.isfinite(y_archive).all():
            raise ValueError("archive_y must be finite")
        if remaining_expensive_budget < 0:
            raise ValueError("remaining_expensive_budget must be non-negative")
        if previous_empty_reference_count is not None and not (
            0 <= previous_empty_reference_count <= len(self.reference_directions)
        ):
            raise ValueError("previous empty-reference count is outside the valid range")

        # Stable archive deduplication preserves the first real observation.
        _, first_indices = np.unique(
            np.round(x_archive, decimals=14), axis=0, return_index=True
        )
        first_indices.sort()
        x_archive = x_archive[first_indices]
        y_archive = y_archive[first_indices]
        rng = np.random.default_rng(config.seed)
        population_target = self.population_size

        if len(x_archive) >= population_target:
            population_x, population_y, _, _ = environmental_selection(
                x_archive,
                y_archive,
                self.reference_directions,
                population_size=population_target,
                penalty=0.0,
            )
        else:
            needed = population_target - len(x_archive)
            random_x = rng.random((needed, config.n_variables))
            random_prediction = self._predict(random_x)
            population_x = np.vstack((x_archive, random_x))
            population_y = np.vstack((y_archive, random_prediction.mean))

        inner_used = 0
        generations = 0
        while inner_used < config.inner_evaluations:
            offspring_count = min(
                population_target,
                config.inner_evaluations - inner_used,
            )
            offspring_x = sbx_polynomial_offspring(
                population_x,
                offspring_count,
                rng,
                crossover_probability=config.crossover_probability,
                crossover_eta=config.crossover_eta,
                mutation_probability=config.mutation_probability,
                mutation_eta=config.mutation_eta,
            )
            offspring_prediction = self._predict(offspring_x)
            inner_used += offspring_count
            generations += 1
            progress = inner_used / max(config.inner_evaluations, 1)
            penalty = config.n_objectives * progress**config.apd_alpha
            combined_x = np.vstack((population_x, offspring_x))
            combined_y = np.vstack((population_y, offspring_prediction.mean))
            population_x, population_y, _, _ = environmental_selection(
                combined_x,
                combined_y,
                self.reference_directions,
                population_size=population_target,
                penalty=penalty,
            )

        prediction = self._predict(population_x)
        association, apd, _, objective_scale, _ = associate_by_apd(
            prediction.mean,
            self.reference_directions,
            penalty=float(config.n_objectives),
        )
        scaled_std = standardize_uncertainty(prediction.std, objective_scale)
        active_indices = np.unique(association)
        empty_count = len(self.reference_directions) - len(active_indices)
        previous_empty = (
            empty_count
            if previous_empty_reference_count is None
            else previous_empty_reference_count
        )
        empty_growth = empty_count - previous_empty
        explore = empty_growth >= config.empty_growth_fraction * len(
            self.reference_directions
        ) and empty_growth > 0
        mode = "exploration" if explore else "exploitation"

        budgeted_q = min(config.q, remaining_expensive_budget)
        selected: list[int] = []
        selected_directions: list[int] = []
        if budgeted_q > 0 and len(active_indices) > 0:
            cluster_count = min(budgeted_q, len(active_indices))
            active_directions = self.reference_directions[active_indices]
            labels = _partition_directions(active_directions, cluster_count, rng)
            uncertainty = np.mean(
                scaled_std[:, self.expensive_objective_indices], axis=1
            )
            for cluster in range(cluster_count):
                directions_in_cluster = active_indices[labels == cluster]
                members = np.flatnonzero(np.isin(association, directions_in_cluster))
                if explore:
                    ranked = sorted(
                        members.tolist(),
                        key=lambda index: (-uncertainty[index], apd[index], index),
                    )
                else:
                    ranked = sorted(
                        members.tolist(),
                        key=lambda index: (apd[index], -uncertainty[index], index),
                    )
                for index in ranked:
                    already_selected = (
                        population_x[np.asarray(selected, dtype=np.int64)]
                        if selected
                        else np.empty((0, config.n_variables), dtype=np.float64)
                    )
                    if _is_duplicate(
                        population_x[index], x_archive, config.uniqueness_tolerance
                    ) or _is_duplicate(
                        population_x[index], already_selected, config.uniqueness_tolerance
                    ):
                        continue
                    selected.append(index)
                    selected_directions.append(int(association[index]))
                    break

            # A cluster can contain only archived designs.  Fill from all
            # remaining model candidates while preserving the active mode.
            if len(selected) < budgeted_q:
                uncertainty = np.mean(
                    scaled_std[:, self.expensive_objective_indices], axis=1
                )
                if explore:
                    fallback = sorted(
                        range(len(population_x)),
                        key=lambda index: (-uncertainty[index], apd[index], index),
                    )
                else:
                    fallback = sorted(
                        range(len(population_x)),
                        key=lambda index: (apd[index], -uncertainty[index], index),
                    )
                for index in fallback:
                    already_selected = (
                        population_x[np.asarray(selected, dtype=np.int64)]
                        if selected
                        else np.empty((0, config.n_variables), dtype=np.float64)
                    )
                    if index in selected or _is_duplicate(
                        population_x[index], x_archive, config.uniqueness_tolerance
                    ) or _is_duplicate(
                        population_x[index], already_selected, config.uniqueness_tolerance
                    ):
                        continue
                    selected.append(index)
                    selected_directions.append(int(association[index]))
                    if len(selected) == budgeted_q:
                        break

        selected_array = np.asarray(selected, dtype=np.int64)
        proposal_x = population_x[selected_array]
        proposal_mean = prediction.mean[selected_array]
        proposal_std = prediction.std[selected_array]
        selected_apd_values = [float(apd[index]) for index in selected_array]
        mean_std_values = (
            np.mean(
                scaled_std[selected_array][:, self.expensive_objective_indices],
                axis=1,
            ).tolist()
            if len(selected_array)
            else []
        )

        # A pessimistic surrogate can leave the complete environmental
        # population equal to archived designs.  K-RVEA still owes the outer
        # loop a full expensive batch, so use a deterministic model-ranked
        # unit-cube pool only for the missing slots.  This is a fail-safe, not
        # part of the normal evolutionary path, and is explicitly diagnosed.
        random_fallback_count = 0
        if len(proposal_x) < budgeted_q:
            pool_size = max(1024, 64 * (budgeted_q - len(proposal_x)))
            fallback_x = rng.random((pool_size, config.n_variables))
            fallback_prediction = self._predict(fallback_x)
            (
                fallback_association,
                fallback_apd,
                _,
                fallback_scale,
                _,
            ) = associate_by_apd(
                fallback_prediction.mean,
                self.reference_directions,
                penalty=float(config.n_objectives),
            )
            fallback_scaled_std = standardize_uncertainty(
                fallback_prediction.std,
                fallback_scale,
            )
            fallback_uncertainty = np.mean(
                fallback_scaled_std[:, self.expensive_objective_indices],
                axis=1,
            )
            if explore:
                fallback_order = sorted(
                    range(pool_size),
                    key=lambda index: (
                        -fallback_uncertainty[index],
                        fallback_apd[index],
                        index,
                    ),
                )
            else:
                fallback_order = sorted(
                    range(pool_size),
                    key=lambda index: (
                        fallback_apd[index],
                        -fallback_uncertainty[index],
                        index,
                    ),
                )
            for index in fallback_order:
                if _is_duplicate(
                    fallback_x[index], x_archive, config.uniqueness_tolerance
                ) or _is_duplicate(
                    fallback_x[index], proposal_x, config.uniqueness_tolerance
                ):
                    continue
                proposal_x = np.vstack((proposal_x, fallback_x[index]))
                proposal_mean = np.vstack(
                    (proposal_mean, fallback_prediction.mean[index])
                )
                proposal_std = np.vstack(
                    (proposal_std, fallback_prediction.std[index])
                )
                selected_directions.append(int(fallback_association[index]))
                selected_apd_values.append(float(fallback_apd[index]))
                mean_std_values.append(float(fallback_uncertainty[index]))
                random_fallback_count += 1
                if len(proposal_x) == budgeted_q:
                    break
            if len(proposal_x) != budgeted_q:
                raise RuntimeError(
                    "K-RVEA could not construct a complete unique proposal batch"
                )
        diagnostics = ProposalDiagnostics(
            seed=config.seed,
            inner_budget=config.inner_evaluations,
            inner_evaluations_used=inner_used,
            generations=generations,
            reference_direction_count=len(self.reference_directions),
            population_size=population_target,
            active_reference_count=len(active_indices),
            empty_reference_count=empty_count,
            previous_empty_reference_count=previous_empty,
            empty_reference_growth=empty_growth,
            mode=mode,
            requested_q=config.q,
            expensive_budget_remaining=remaining_expensive_budget,
            proposed_count=len(proposal_x),
            random_fallback_count=random_fallback_count,
            selected_reference_indices=tuple(selected_directions),
            selected_apd=tuple(selected_apd_values),
            selected_mean_std=tuple(float(value) for value in mean_std_values),
        )
        return ProposalBatch(
            unit_x=proposal_x,
            predicted_mean=proposal_mean,
            predicted_std=proposal_std,
            diagnostics=diagnostics,
        )


__all__ = [
    "KRVEA",
    "KRVEAConfig",
    "ProposalBatch",
    "ProposalDiagnostics",
    "SelectionDiagnostics",
    "SurrogatePrediction",
    "SurrogatePredictor",
    "associate_by_apd",
    "das_dennis_reference_directions",
    "environmental_selection",
    "normalize_objectives",
    "sbx_polynomial_offspring",
    "standardize_uncertainty",
]
