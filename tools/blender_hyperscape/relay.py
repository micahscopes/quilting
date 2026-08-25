"""Dependency-free background client for the optional local peer relay.

This module never imports ``bpy``. Network I/O stays on one daemon thread;
Blender adapters drain validated deliveries from the main-thread timer.
Delivery cursors are restart/gap evidence only and never become authored
projection revisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from . import protocol
except ImportError:  # Pure-Python tests import this directory directly.
    import protocol  # type: ignore


DEFAULT_RELAY_URL = "http://127.0.0.1:42117"
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_POLL_LIMIT = 256
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class RelayTransportError(RuntimeError):
    """The local relay or one of its delivery batches is invalid."""


@dataclass(frozen=True)
class RelayDelivery:
    cursor: int
    frame: Mapping[str, Any]


@dataclass(frozen=True)
class RelayStatus:
    state: str = "stopped"
    generation: str | None = None
    cursor: int = 0
    sent_frames: int = 0
    received_frames: int = 0
    gaps: int = 0
    restarts: int = 0
    last_error: str | None = None
    last_activity_monotonic_seconds: float | None = None


RequestJson = Callable[[str, str, str | None], Mapping[str, Any]]


class LocalRelayTransport:
    """Authenticated polling transport with bounded cross-thread queues."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        poll_limit: int = DEFAULT_POLL_LIMIT,
        outbound_capacity: int = 256,
        inbound_capacity: int = 4096,
        request_json: RequestJson | None = None,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._token = _validate_token(token)
        if (
            not isinstance(poll_interval_seconds, (int, float))
            or isinstance(poll_interval_seconds, bool)
            or not 0.005 <= float(poll_interval_seconds) <= 10.0
        ):
            raise RelayTransportError("poll interval must be in [0.005, 10] seconds")
        if (
            isinstance(poll_limit, bool)
            or not isinstance(poll_limit, int)
            or not 1 <= poll_limit <= 1024
        ):
            raise RelayTransportError("poll limit must be in [1, 1024]")
        if (
            isinstance(outbound_capacity, bool)
            or not isinstance(outbound_capacity, int)
            or outbound_capacity <= 0
            or isinstance(inbound_capacity, bool)
            or not isinstance(inbound_capacity, int)
            or inbound_capacity <= 0
        ):
            raise RelayTransportError("relay queue capacities must be positive integers")
        self._poll_interval = float(poll_interval_seconds)
        self._poll_limit = poll_limit
        self._outbound: Queue[str] = Queue(maxsize=outbound_capacity)
        # Owned exclusively by the worker. A failed POST remains here so a
        # later queued authored command can never overtake it during retry.
        self._pending_outbound: str | None = None
        self._inbound: Queue[RelayDelivery] = Queue(maxsize=inbound_capacity)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = RelayStatus()
        self._status_lock = threading.Lock()
        self._degraded = False
        self._request_json_impl = request_json or self._http_json

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_status(state="connecting", last_error=None)
        self._thread = threading.Thread(
            target=self._run,
            name="HyperscapeLocalRelay",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        if thread is not None and thread.is_alive():
            self._set_status(
                state="stopping",
                last_error="relay worker did not stop before the timeout",
            )
            return
        self._thread = None
        self._set_status(state="stopped")

    def send(self, frame: Mapping[str, Any]) -> None:
        protocol.validate_local_peer_frame(frame)
        try:
            encoded = json.dumps(
                frame,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            self._outbound.put_nowait(encoded)
        except (Full, TypeError, ValueError) as error:
            raise RelayTransportError("relay outbound queue rejected the frame") from error

    def drain(self, limit: int = 256) -> list[RelayDelivery]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RelayTransportError("drain limit must be a positive integer")
        deliveries: list[RelayDelivery] = []
        for _ in range(limit):
            try:
                deliveries.append(self._inbound.get_nowait())
            except Empty:
                break
        return deliveries

    def status(self) -> RelayStatus:
        with self._status_lock:
            return replace(self._status)

    def _run(self) -> None:
        retry_seconds = self._poll_interval
        while not self._stop.is_set():
            try:
                self._flush_outbound()
                has_more = self._poll_once()
                self._set_status(
                    state="degraded" if self._degraded else "connected",
                    last_error=None,
                )
                retry_seconds = self._poll_interval
                if not has_more:
                    self._stop.wait(self._poll_interval)
            except Exception as error:  # Keep one broken request from killing sync.
                self._set_status(state="error", last_error=str(error))
                self._stop.wait(retry_seconds)
                retry_seconds = min(max(retry_seconds * 2.0, 0.1), 2.0)

    def _flush_outbound(self) -> None:
        for _ in range(32):
            if self._pending_outbound is None:
                try:
                    self._pending_outbound = self._outbound.get_nowait()
                except Empty:
                    return
            response = self._request_json_impl(
                "POST",
                "/v1/frame",
                self._pending_outbound,
            )
            _decimal_cursor(response.get("cursor"), "posted cursor")
            _generation(response.get("generation"))
            self._pending_outbound = None
            self._increment_status("sent_frames")
            self._mark_activity()

    def _poll_once(self) -> bool:
        requested_after = self.status().cursor
        batch = self._request_json_impl(
            "GET",
            f"/v1/frames?after={requested_after}&limit={self._poll_limit}",
            None,
        )
        return self._accept_batch(batch, requested_after)

    def _accept_batch(self, batch: Mapping[str, Any], requested_after: int) -> bool:
        if not isinstance(batch, Mapping):
            raise RelayTransportError("relay poll response must be an object")
        generation = _generation(batch.get("generation"))
        if _decimal_cursor(batch.get("requestedAfter"), "requested cursor") != requested_after:
            raise RelayTransportError("relay response acknowledged the wrong requested cursor")
        latest = _decimal_cursor(batch.get("latestCursor"), "latest cursor")
        resume_after = _decimal_cursor(batch.get("resumeAfter"), "resume cursor")
        if resume_after > latest:
            raise RelayTransportError("relay resume cursor exceeds latest")
        oldest_value = batch.get("oldestCursor")
        oldest = (
            None
            if oldest_value is None
            else _decimal_cursor(oldest_value, "oldest cursor")
        )
        if oldest is not None and (oldest == 0 or oldest > latest):
            raise RelayTransportError("relay oldest cursor is outside retained history")
        gap = batch.get("gap")
        has_more = batch.get("hasMore")
        frames = batch.get("frames")
        if (
            not isinstance(gap, bool)
            or not isinstance(has_more, bool)
            or not isinstance(frames, list)
        ):
            raise RelayTransportError("relay response has invalid gap, pagination, or frames")

        deliveries: list[RelayDelivery] = []
        previous = max(requested_after, resume_after) if gap else requested_after
        for item in frames:
            if not isinstance(item, Mapping):
                raise RelayTransportError("relay delivery must be an object")
            cursor = _decimal_cursor(item.get("cursor"), "delivery cursor")
            frame_json = item.get("frameJson")
            if not isinstance(frame_json, str):
                raise RelayTransportError("relay delivery frame must be exact JSON text")
            try:
                frame = json.loads(frame_json)
            except json.JSONDecodeError as error:
                raise RelayTransportError("relay delivery frame is invalid JSON") from error
            if cursor != previous + 1 or cursor > latest:
                raise RelayTransportError(
                    "relay delivery cursors must be contiguous through latest"
                )
            protocol.validate_local_peer_frame(frame)
            deliveries.append(RelayDelivery(cursor=cursor, frame=frame))
            previous = cursor

        if has_more and (not deliveries or deliveries[-1].cursor >= latest):
            raise RelayTransportError("relay pagination cannot make forward progress")
        if not has_more and deliveries and deliveries[-1].cursor != latest:
            raise RelayTransportError("final relay page must end at latest")

        current = self.status()
        if current.generation is not None and generation != current.generation:
            self._degraded = True
            self._set_status(
                generation=generation,
                cursor=0,
                restarts=current.restarts + 1,
                gaps=current.gaps + 1,
            )
            return True
        if current.generation is None:
            self._set_status(generation=generation)

        if len(deliveries) > self._inbound.maxsize - self._inbound.qsize():
            raise RelayTransportError("relay inbound queue is full")
        for delivery in deliveries:
            self._inbound.put_nowait(delivery)

        next_cursor = deliveries[-1].cursor if deliveries else requested_after
        if gap:
            self._degraded = True
            current = self.status()
            self._set_status(gaps=current.gaps + 1)
            if not deliveries:
                next_cursor = resume_after
        self._set_status(cursor=next_cursor)
        if deliveries:
            self._increment_status("received_frames", len(deliveries))
            self._mark_activity()
        return has_more or next_cursor < latest

    def _http_json(self, method: str, path: str, body: str | None) -> Mapping[str, Any]:
        data = body.encode("utf-8") if body is not None else None
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self._base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=2.0) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise RelayTransportError("relay response exceeds the byte limit")
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", "replace")
            raise RelayTransportError(
                f"relay HTTP {error.code}: {detail[:512]}"
            ) from error
        except (OSError, URLError) as error:
            raise RelayTransportError(f"relay request failed: {error}") from error
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RelayTransportError("relay returned invalid JSON") from error
        if not isinstance(value, Mapping):
            raise RelayTransportError("relay response must be a JSON object")
        return value

    def _set_status(self, **changes: Any) -> None:
        with self._status_lock:
            self._status = replace(self._status, **changes)

    def _increment_status(self, field: str, amount: int = 1) -> None:
        with self._status_lock:
            self._status = replace(
                self._status,
                **{field: getattr(self._status, field) + amount},
            )

    def _mark_activity(self) -> None:
        self._set_status(last_activity_monotonic_seconds=time.monotonic())


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise RelayTransportError("relay URL must be a string")
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise RelayTransportError("relay URL has an invalid host or port") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise RelayTransportError(
            "relay URL must be one HTTP(S) origin without credentials or a path"
        )
    return value.rstrip("/")


def _validate_token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or any(
            not (
                character.isascii()
                and (character.isalnum() or character in "-_.~")
            )
            for character in value
        )
    ):
        raise RelayTransportError("relay token must be 1..256 URL-safe ASCII characters")
    return value


def _decimal_cursor(value: Any, context: str) -> int:
    if (
        not isinstance(value, str)
        or not value
        or (len(value) > 1 and value.startswith("0"))
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise RelayTransportError(f"{context} must be canonical decimal text")
    cursor = int(value)
    if not 0 <= cursor <= protocol.MAX_U64:
        raise RelayTransportError(f"{context} exceeds an unsigned 64-bit integer")
    return cursor


def _generation(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or any(
            not (
                character.isascii()
                and (character.isalnum() or character in "-_.")
            )
            for character in value
        )
    ):
        raise RelayTransportError("relay generation is invalid")
    return value
