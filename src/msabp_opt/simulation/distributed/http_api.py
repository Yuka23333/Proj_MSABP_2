"""Dependency-free HTTP transport for Princess and Maid.

Remote Maids make outbound requests to Princess after a device-local Maid Bell
wakes them.  Each Maid opens its local headless CST process; no interactive
desktop is required.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .protocol import ProtocolError, validate_message


MESSAGE_PATH = "/api/v1/message"
ARTIFACT_PATH_PREFIX = "/api/v1/artifact/"
HEALTH_PATH = "/health"
DEFAULT_JSON_LIMIT_BYTES = 1024 * 1024
DEFAULT_ARTIFACT_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ApiError(RuntimeError):
    """Raised when a Princess API request fails."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


def _validated_identifier(value: Any, label: str) -> str:
    resolved = str(value).strip()
    if not _SAFE_IDENTIFIER.fullmatch(resolved) or resolved in {".", ".."}:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid {label}")
    return resolved


MessageHandler = Callable[[dict[str, Any]], Mapping[str, Any]]
ArtifactHandler = Callable[[str, str, str, Path, str, int], Mapping[str, Any]]


class PrincessHttpServer(ThreadingHTTPServer):
    """Threaded HTTP server with injected scheduler callbacks."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        token: str,
        upload_dir: str | Path,
        message_handler: MessageHandler,
        artifact_handler: ArtifactHandler,
        json_limit_bytes: int = DEFAULT_JSON_LIMIT_BYTES,
        artifact_limit_bytes: int = DEFAULT_ARTIFACT_LIMIT_BYTES,
    ) -> None:
        if not token:
            raise ValueError("Princess API token must not be empty")
        if json_limit_bytes <= 0 or artifact_limit_bytes <= 0:
            raise ValueError("request size limits must be positive")
        self.api_token = token
        self.upload_dir = Path(upload_dir).resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.message_handler = message_handler
        self.artifact_handler = artifact_handler
        self.json_limit_bytes = int(json_limit_bytes)
        self.artifact_limit_bytes = int(artifact_limit_bytes)
        super().__init__(server_address, PrincessRequestHandler)


class PrincessRequestHandler(BaseHTTPRequestHandler):
    """Serve authenticated messages and streamed attempt archives."""

    server: PrincessHttpServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != HEALTH_PATH:
            self._send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        self._send_json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            self._require_auth()
            path = urlparse(self.path).path
            if path == MESSAGE_PATH:
                self._handle_message()
                return
            if path.startswith(ARTIFACT_PATH_PREFIX):
                self._handle_artifact(path)
                return
            raise ApiError(HTTPStatus.NOT_FOUND, "unknown endpoint")
        except ApiError as exc:
            if exc.status == HTTPStatus.UNAUTHORIZED:
                self._discard_small_request_body()
            self._send_error(exc.status, str(exc))
        except (ProtocolError, ValueError, json.JSONDecodeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - last-resort server boundary
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{type(exc).__name__}: {exc}",
            )

    def log_message(self, format: str, *args: object) -> None:
        # Princess owns structured logging; BaseHTTPRequestHandler's stderr
        # timestamps are noisy when several Maids poll concurrently.
        return

    def _require_auth(self) -> None:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.api_token}"
        if not hmac.compare_digest(supplied, expected):
            raise ApiError(HTTPStatus.UNAUTHORIZED, "invalid bearer token")

    def _discard_small_request_body(self) -> None:
        """Avoid a Windows TCP reset hiding a small authenticated error body."""

        raw = self.headers.get("Content-Length")
        try:
            length = int(raw) if raw is not None else 0
        except ValueError:
            return
        if 0 < length <= self.server.json_limit_bytes:
            self.rfile.read(length)

    def _content_length(self, *, maximum: int) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(raw)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "negative Content-Length")
        if length > maximum:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request is too large")
        return length

    def _handle_message(self) -> None:
        length = self._content_length(maximum=self.server.json_limit_bytes)
        body = self.rfile.read(length)
        if len(body) != length:
            raise ApiError(HTTPStatus.BAD_REQUEST, "incomplete JSON request body")
        try:
            decoded = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "request is not UTF-8") from exc
        message = validate_message(decoded)
        response = dict(self.server.message_handler(message))
        self._send_json(HTTPStatus.OK, response)

    def _handle_artifact(self, path: str) -> None:
        attempt_id = _validated_identifier(
            unquote(path.removeprefix(ARTIFACT_PATH_PREFIX)),
            "attempt id",
        )
        lease_token = self.headers.get("X-MSABP-Lease-Token", "").strip()
        if not lease_token:
            raise ApiError(HTTPStatus.BAD_REQUEST, "missing lease token")
        worker_id = _validated_identifier(
            self.headers.get("X-MSABP-Worker-ID", ""),
            "worker id",
        )
        length = self._content_length(maximum=self.server.artifact_limit_bytes)

        descriptor, temp_name = tempfile.mkstemp(
            prefix=f"{attempt_id}.",
            suffix=".part",
            dir=self.server.upload_dir,
        )
        digest = hashlib.sha256()
        remaining = length
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ApiError(
                            HTTPStatus.BAD_REQUEST,
                            "incomplete artifact request body",
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            temp_path = Path(temp_name)
            response = dict(
                self.server.artifact_handler(
                    attempt_id,
                    worker_id,
                    lease_token,
                    temp_path,
                    digest.hexdigest(),
                    length,
                )
            )
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise
        self._send_json(HTTPStatus.OK, response)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message, "status": int(status)})

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(
            dict(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class PrincessClient:
    """Small outbound-only client used by Maid."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0) -> None:
        if not token:
            raise ValueError("Princess API token must not be empty")
        if timeout <= 0:
            raise ValueError("HTTP timeout must be positive")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Princess URL must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.timeout = float(timeout)

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", HEALTH_PATH, authenticated=False)

    def send(self, message: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_message(message)
        body = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return self._json_request("POST", MESSAGE_PATH, body=body)

    def upload_artifact(
        self,
        attempt_id: str,
        worker_id: str,
        lease_token: str,
        archive_path: str | Path,
    ) -> dict[str, Any]:
        attempt_id = _validated_identifier(attempt_id, "attempt id")
        worker_id = _validated_identifier(worker_id, "worker id")
        if not isinstance(lease_token, str) or not lease_token.strip():
            raise ValueError("lease token must be a non-empty string")
        if "\r" in lease_token or "\n" in lease_token:
            raise ValueError("lease token contains a forbidden line break")
        archive = Path(archive_path)
        if not archive.is_file():
            raise FileNotFoundError(f"artifact archive does not exist: {archive}")
        parsed = urlparse(self.base_url)
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            parsed.hostname,
            parsed.port,
            timeout=self.timeout,
        )
        base_path = parsed.path.rstrip("/")
        request_path = (
            f"{base_path}{ARTIFACT_PATH_PREFIX}{quote(attempt_id, safe='')}"
        )
        size = archive.stat().st_size
        try:
            connection.putrequest("POST", request_path)
            connection.putheader("Authorization", f"Bearer {self.token}")
            connection.putheader("Content-Type", "application/zip")
            connection.putheader("Content-Length", str(size))
            connection.putheader("X-MSABP-Worker-ID", worker_id)
            connection.putheader("X-MSABP-Lease-Token", lease_token)
            connection.endheaders()
            with archive.open("rb") as handle:
                while chunk := handle.read(COPY_CHUNK_BYTES):
                    connection.send(chunk)
            response = connection.getresponse()
            payload = response.read()
            status = response.status
        finally:
            connection.close()
        return self._decode_response(status, payload)

    def _json_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            method=method,
            headers=headers,
        )
        return self._open_json(request)

    def _open_json(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                message = str(json.loads(detail).get("error", detail))
            except json.JSONDecodeError:
                message = detail
            raise ApiError(exc.code, message) from exc
        except URLError as exc:
            raise ConnectionError(f"Princess is unreachable: {exc.reason}") from exc
        return self._decode_response(HTTPStatus.OK, payload)

    @staticmethod
    def _decode_response(status: int, payload: bytes) -> dict[str, Any]:
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, "Princess returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ApiError(HTTPStatus.BAD_GATEWAY, "Princess response is not an object")
        if not 200 <= int(status) < 300:
            raise ApiError(int(status), str(decoded.get("error", f"HTTP {status}")))
        return decoded
