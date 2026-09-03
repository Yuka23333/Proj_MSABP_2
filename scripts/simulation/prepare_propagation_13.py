"""Build the 13-case propagation worklist from the reviewed Pareto shortlist.

The campaign contains the 12 pure-geometry k-medoids plus candidate rank #1.
Candidate #35 is intentionally not added because the user judged it visually
indistinguishable from the already retained #34.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.automation import antenna_sampler  # noqa: E402


SHORTLIST_CSV = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "pareto_candidates_return_loss_gt_7db.csv"
)
KMEDOIDS_CSV = (
    REPOSITORY_ROOT
    / "results"
    / "processed"
    / "pareto_candidates_selected_k12.csv"
)
OUTPUT_CSV = (
    REPOSITORY_ROOT / "data" / "samples" / "propagation_selected_13.csv"
)
KMEDOID_RANKS = (4, 8, 10, 18, 19, 20, 23, 30, 31, 33, 34, 36)
EXTRA_RANKS = (1,)
EXPECTED_RANKS = tuple(sorted(KMEDOID_RANKS + EXTRA_RANKS))
SIMULATION_MODE = "propagation_s21"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _ranked(rows: list[dict[str, str]], path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        rank = int(row["candidate_rank"])
        if rank in result:
            raise ValueError(f"duplicate candidate_rank={rank} in {path}")
        result[rank] = row
    return result


def build_rows() -> list[dict[str, str]]:
    medoids = _ranked(_read_rows(KMEDOIDS_CSV), KMEDOIDS_CSV)
    if tuple(sorted(medoids)) != KMEDOID_RANKS:
        raise ValueError(
            f"k-medoids ranks changed: expected={KMEDOID_RANKS}, "
            f"actual={tuple(sorted(medoids))}"
        )
    shortlist = _ranked(_read_rows(SHORTLIST_CSV), SHORTLIST_CSV)
    source_rows = dict(medoids)
    for rank in EXTRA_RANKS:
        source_rows[rank] = shortlist[rank]

    output: list[dict[str, str]] = []
    for sequence, rank in enumerate(EXPECTED_RANKS, start=1):
        source = source_rows[rank]
        row = {
            "sample_id": f"prop_{sequence:02d}_rank_{rank:02d}",
            "simulation_mode": SIMULATION_MODE,
            "candidate_rank": str(rank),
            "source": source["source"],
            "source_case_id": source["case_id"],
            "geometry_valid": "True",
            "geometry_error": "",
            "final_conductor_components": "1",
        }
        for name in antenna_sampler.PARAMETER_REGISTRY:
            value = float(source[name])
            if not math.isfinite(value):
                raise ValueError(f"rank {rank} parameter {name} is not finite")
            row[name] = format(value, ".17g")
        antenna_sampler.parameters_from_csv_row(row)
        output.append(row)
    if len(output) != 13:
        raise AssertionError(f"expected 13 propagation cases, got {len(output)}")
    return output


def write_worklist(
    destination: str | Path = OUTPUT_CSV,
) -> Path:
    rows = build_rows()
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "simulation_mode",
        "candidate_rank",
        "source",
        "source_case_id",
        "geometry_valid",
        "geometry_error",
        "final_conductor_components",
        *antenna_sampler.PARAMETER_REGISTRY,
    ]
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)
    return destination


def main() -> int:
    destination = write_worklist()
    print(f"Propagation worklist: {destination}")
    print(f"Cases: {len(build_rows())}; ranks: {EXPECTED_RANKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
