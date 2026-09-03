from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
for root in (REPOSITORY_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.simulation import prepare_propagation_debug_on_host as host  # noqa: E402


def test_default_host_local_debug_target_and_candidate() -> None:
    row = host.load_candidate_row(host.WORKLIST_PATH, host.CANDIDATE_RANK)

    assert host.CANDIDATE_RANK == 1
    assert row["sample_id"] == "prop_01_rank_01"
    assert host.PROJECT_PATH.name == "msa-bp-propagation.cst"
    assert "manual-propagation-export-debug-rank01-002" in str(host.PROJECT_PATH)


def test_host_local_script_never_calls_solver_or_save() -> None:
    source = Path(host.__file__).read_text(encoding="utf-8")

    assert ".run_solver(" not in source
    assert "execute_save_project" not in source
    assert "READY FOR MANUAL SOLVE" in source
    assert "hold_session_open()" in source
