"""Thin command-line entrypoint for one device-local CST Maid."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.maid import (  # noqa: E402
    Maid,
    MaidRuntimeConfig,
    doctor,
)


@contextmanager
def _device_process_lock(worker_id: str) -> Iterator[Path]:
    """Hold one stable per-device lock for the lifetime of a Maid process."""

    lock_path = REPOSITORY_ROOT / "simulations" / "runs" / f"maid.{worker_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle: BinaryIO = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"another Maid process already owns device {worker_id!r}"
                ) from exc
            locked = True
        yield lock_path
    finally:
        if locked:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local Maid process. Princess normally invokes this script "
            "through a detached SSH wake-up command."
        )
    )
    parser.add_argument(
        "--runtime-config",
        required=True,
        type=Path,
        help="Princess-distributed Maid runtime JSON",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="validate the local runtime and dependencies without opening CST",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = MaidRuntimeConfig.load(args.runtime_config)
        report = doctor(config)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.doctor:
            return 0
        with _device_process_lock(config.worker_id):
            return Maid(config).run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Maid fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
