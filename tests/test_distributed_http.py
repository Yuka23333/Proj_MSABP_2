from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msabp_opt.simulation.distributed.http_api import (  # noqa: E402
    ApiError,
    PrincessClient,
    PrincessHttpServer,
)
from msabp_opt.simulation.distributed.protocol import make_message  # noqa: E402


@pytest.fixture
def api_server(tmp_path: Path):
    messages: list[dict] = []
    artifacts: list[tuple[str, str, str, bytes, str, int]] = []

    def on_message(message: dict) -> dict:
        messages.append(message)
        return {"reply": message["type"]}

    def on_artifact(
        attempt_id: str,
        worker_id: str,
        lease_token: str,
        temp_path: Path,
        sha256: str,
        size: int,
    ) -> dict:
        data = temp_path.read_bytes()
        artifacts.append((attempt_id, worker_id, lease_token, data, sha256, size))
        temp_path.unlink()
        return {"sha256": sha256, "size": size}

    server = PrincessHttpServer(
        ("127.0.0.1", 0),
        token="test-token",
        upload_dir=tmp_path / "uploads",
        message_handler=on_message,
        artifact_handler=on_artifact,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, messages, artifacts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_message_and_artifact_round_trip(api_server, tmp_path: Path) -> None:
    server, messages, artifacts = api_server
    client = PrincessClient(
        f"http://127.0.0.1:{server.server_port}",
        "test-token",
    )
    assert client.health() == {"status": "ok"}

    message = make_message("hello", run_id="run-1", worker_id="maid-a")
    assert client.send(message) == {"reply": "hello"}
    assert messages == [message]

    archive = tmp_path / "case.zip"
    archive.write_bytes(b"zip-payload")
    response = client.upload_artifact(
        "attempt-1",
        "maid-a",
        "lease-1",
        archive,
    )
    assert response == {
        "sha256": hashlib.sha256(b"zip-payload").hexdigest(),
        "size": len(b"zip-payload"),
    }
    assert artifacts[0][:4] == (
        "attempt-1",
        "maid-a",
        "lease-1",
        b"zip-payload",
    )


def test_wrong_token_is_rejected(api_server) -> None:
    server, _, _ = api_server
    client = PrincessClient(
        f"http://127.0.0.1:{server.server_port}",
        "wrong-token",
    )
    message = make_message("hello", run_id="run-1", worker_id="maid-a")
    with pytest.raises(ApiError, match="invalid bearer token"):
        client.send(message)
