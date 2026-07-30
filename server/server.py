#!/usr/bin/env python3
"""Small authenticated LAN receiver for Plant Capture Android videos.

Uses only the Python standard library. Videos are written atomically and a
JSON sidecar is created with server-computed size and SHA-256.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
CHUNK_SIZE = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def best_lan_ip() -> str:
    """Best-effort LAN address without sending application data."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


def sanitize_filename(raw: str) -> str:
    name = Path(unquote(raw)).name
    name = SAFE_NAME.sub("_", name).strip("._")
    if not name:
        name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    return name[:180]


class PlantCaptureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        output_dir: Path,
        token: str,
        max_bytes: int,
    ) -> None:
        super().__init__(address, handler)
        self.output_dir = output_dir
        self.token = token
        self.max_bytes = max_bytes


class Handler(BaseHTTPRequestHandler):
    server: PlantCaptureServer
    protocol_version = "HTTP/1.1"
    server_version = "PlantCaptureReceiver/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[{utc_now()}] {self.client_address[0]} {fmt % args}\n")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Do not keep a rejected upload connection alive: if the request body
        # has not been consumed, BaseHTTPRequestHandler would otherwise parse
        # the first MP4 bytes ("ftyp...") as another HTTP request.
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        supplied = auth[7:] if auth.startswith("Bearer ") else self.headers.get("X-Upload-Token", "")
        return bool(supplied) and hashlib.sha256(supplied.encode()).digest() == hashlib.sha256(
            self.server.token.encode()
        ).digest()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "plant-capture-receiver",
                    "time_utc": utc_now(),
                    "max_upload_bytes": self.server.max_bytes,
                },
            )
            return
        if path == "/ready":
            if not self.authorized():
                self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
                return
            self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "plant-capture-receiver",
                    "ready_for_upload": True,
                    "max_upload_bytes": self.server.max_bytes,
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        prefix = "/upload/"
        if not path.startswith(prefix):
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
            return

        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return

        content_type = self.headers.get("Content-Type", "")
        if content_type not in {"video/mp4", "application/octet-stream"}:
            self.send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"ok": False, "error": "content_type_must_be_video_mp4"},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_json(HTTPStatus.LENGTH_REQUIRED, {"ok": False, "error": "content_length_required"})
            return

        if length <= 0:
            self.send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "empty_upload"})
            return
        if length > self.server.max_bytes:
            self.send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"ok": False, "error": "upload_too_large", "limit": self.server.max_bytes},
            )
            return

        filename = sanitize_filename(path[len(prefix) :])
        destination = self.server.output_dir / filename
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            destination = destination.with_name(f"{stem}_{datetime.now().strftime('%H%M%S_%f')}{suffix}")

        self.server.output_dir.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path_raw = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=self.server.output_dir
        )
        temp_path = Path(temp_path_raw)
        sha256 = hashlib.sha256()
        remaining = length
        written = 0

        self.connection.settimeout(60)
        try:
            with os.fdopen(temp_fd, "wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        raise ConnectionError(f"connection ended with {remaining} bytes remaining")
                    output.write(chunk)
                    sha256.update(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, destination)
        except Exception as exc:  # network and filesystem errors need cleanup
            try:
                temp_path.unlink(missing_ok=True)
            finally:
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "upload_failed", "detail": str(exc)},
                )
            return

        metadata: dict[str, Any] = {}
        encoded_metadata = self.headers.get("X-Capture-Metadata-B64")
        if encoded_metadata:
            try:
                decoded = base64.b64decode(encoded_metadata, validate=True).decode("utf-8")
                candidate = json.loads(decoded)
                if isinstance(candidate, dict):
                    metadata = candidate
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                metadata = {"metadata_decode_error": True}

        sidecar = {
            "received_at_utc": utc_now(),
            "remote_ip": self.client_address[0],
            "filename": destination.name,
            "bytes": written,
            "sha256": sha256.hexdigest(),
            "content_type": content_type,
            "capture": metadata,
        }
        sidecar_path = destination.with_suffix(destination.suffix + ".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

        self.send_json(
            HTTPStatus.CREATED,
            {
                "ok": True,
                "filename": destination.name,
                "bytes": written,
                "sha256": sha256.hexdigest(),
                "metadata_file": sidecar_path.name,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive Android plant-capture videos over a LAN")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=Path.home() / "PlantCaptures")
    parser.add_argument(
        "--token",
        default=os.environ.get("PLANT_CAPTURE_TOKEN", "change-this-token"),
        help="Shared upload token; also read from PLANT_CAPTURE_TOKEN",
    )
    parser.add_argument("--max-gib", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.token) < 10:
        raise SystemExit("Refusing a token shorter than 10 characters")
    max_bytes = int(args.max_gib * 1024**3)
    output = args.output.expanduser().resolve()
    server = PlantCaptureServer((args.host, args.port), Handler, output, args.token, max_bytes)
    lan_ip = best_lan_ip()
    print(f"Saving captures to: {output}")
    print(f"Phone server URL:   http://{lan_ip}:{args.port}")
    print("Use the same token in the Android app.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
