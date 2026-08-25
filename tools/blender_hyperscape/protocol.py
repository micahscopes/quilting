"""Dependency-free Hyperscape live-edit protocol codec for Blender.

This module does not choose sockets, WebRTC, IPC, or HHHS. It validates the
same v0.1 JSON fixtures as ``hyperscape-protocol`` and provides the two small
pieces every Blender transport needs: sender-local presence ordering/TTL and
bounded echo suppression for locally originated authored messages.
"""

from __future__ import annotations

from collections import deque
import json
import math
from numbers import Real
import uuid
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = {"major": 0, "minor": 1}
MAX_PRESENCE_TTL_MILLIS = 60_000
MAX_U64 = (1 << 64) - 1


class HyperscapeProtocolError(ValueError):
    """A wire value is incompatible with the Rust protocol contract."""


def _uuid(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise HyperscapeProtocolError(f"{context} must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise HyperscapeProtocolError(f"{context} must be a UUID") from error
    if parsed.int == 0:
        raise HyperscapeProtocolError(f"{context} must not be nil")
    return str(parsed)


def _finite_vector(value: Any, size: int, context: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        raise HyperscapeProtocolError(f"{context} must contain {size} numbers")
    return [_finite_number(component, context) for component in value]


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HyperscapeProtocolError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HyperscapeProtocolError(f"{context} must be finite")
    return result


def _protocol_version(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise HyperscapeProtocolError("protocol version must be an object")
    major = value.get("major")
    minor = value.get("minor")
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or {"major": major, "minor": minor} != PROTOCOL_VERSION
    ):
        raise HyperscapeProtocolError("unsupported protocol version")


def _header(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HyperscapeProtocolError("message header must be an object")
    _protocol_version(value.get("version"))
    _uuid(value.get("message_id"), "message ID")
    _uuid(value.get("sender"), "sender ID")
    sequence = value.get("sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 0 <= sequence <= MAX_U64
    ):
        raise HyperscapeProtocolError(
            "message sequence must be an unsigned 64-bit integer"
        )
    return value


def _transform(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise HyperscapeProtocolError("entity transform must be an object")
    _finite_vector(value.get("translation"), 3, "translation")
    rotation = _finite_vector(value.get("rotation_wxyz"), 4, "rotation")
    scale = _finite_vector(value.get("scale"), 3, "scale")
    if sum(component * component for component in rotation) <= 1.0e-24:
        raise HyperscapeProtocolError("rotation must be nonzero")
    if any(component == 0.0 for component in scale):
        raise HyperscapeProtocolError("scale must be nonzero")


def _asset(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise HyperscapeProtocolError("asset must be an object")
    _uuid(value.get("id"), "asset ID")
    uri = value.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        raise HyperscapeProtocolError("asset URI must not be empty")
    media_type = value.get("media_type")
    if media_type is not None and (
        not isinstance(media_type, str) or not media_type.strip()
    ):
        raise HyperscapeProtocolError("asset media type must not be empty")
    digest = value.get("content_digest")
    if digest is not None:
        if (
            not isinstance(digest, list)
            or len(digest) != 32
            or any(
                isinstance(byte, bool)
                or not isinstance(byte, int)
                or not 0 <= byte <= 255
                for byte in digest
            )
        ):
            raise HyperscapeProtocolError("asset digest must contain 32 bytes")


def validate_authored_envelope(envelope: Any) -> None:
    if not isinstance(envelope, Mapping):
        raise HyperscapeProtocolError("authored envelope must be an object")
    _header(envelope.get("header"))
    command = envelope.get("command")
    if not isinstance(command, Mapping):
        raise HyperscapeProtocolError("authored command must be an object")
    kind = command.get("type")
    if kind == "upsert_asset":
        _asset(command.get("asset"))
    elif kind == "set_entity_transform":
        _uuid(command.get("entity"), "entity ID")
        _transform(command.get("transform"))
    elif kind == "remove_entity":
        _uuid(command.get("entity"), "entity ID")
    else:
        raise HyperscapeProtocolError(f"unknown authored command {kind!r}")


def validate_presence_envelope(envelope: Any) -> None:
    if not isinstance(envelope, Mapping):
        raise HyperscapeProtocolError("presence envelope must be an object")
    _header(envelope.get("header"))
    presence = envelope.get("presence")
    if not isinstance(presence, Mapping):
        raise HyperscapeProtocolError("presence must be an object")
    ttl = presence.get("ttl_millis")
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or not 0 < ttl <= MAX_PRESENCE_TTL_MILLIS
    ):
        raise HyperscapeProtocolError("presence TTL is out of range")
    camera = presence.get("camera")
    if camera is not None:
        if not isinstance(camera, Mapping):
            raise HyperscapeProtocolError("presence camera must be an object")
        _finite_vector(camera.get("eye"), 3, "camera eye")
        forward = _finite_vector(camera.get("forward"), 3, "camera forward")
        up = _finite_vector(camera.get("up"), 3, "camera up")
        if (
            _length(forward) <= 1.0e-12
            or _length(up) <= 1.0e-12
            or _cross_length(forward, up) <= 1.0e-12
        ):
            raise HyperscapeProtocolError("presence camera directions must be independent")
    selection = presence.get("selection", [])
    if not isinstance(selection, list):
        raise HyperscapeProtocolError("presence selection must be an array")
    for entity in selection:
        _uuid(entity, "selected entity ID")
    focus = presence.get("focus")
    if focus is not None:
        if not isinstance(focus, Mapping):
            raise HyperscapeProtocolError("presence focus must be an object")
        _finite_vector(focus.get("center"), 3, "focus center")
        radius = _finite_number(focus.get("radius"), "focus radius")
        if radius <= 0.0:
            raise HyperscapeProtocolError("focus radius must be finite and positive")
        if not isinstance(focus.get("inversion_enabled"), bool):
            raise HyperscapeProtocolError("focus inversion flag must be boolean")
    if presence.get("active_cue") is not None:
        _uuid(presence["active_cue"], "active cue ID")
    if presence.get("animation_seconds") is not None:
        animation_seconds = _finite_number(
            presence["animation_seconds"], "animation time"
        )
        if animation_seconds < 0.0:
            raise HyperscapeProtocolError("animation time must be finite and nonnegative")


def validate_local_peer_frame(frame: Any) -> None:
    """Validate the transport wrapper without collapsing its two lanes."""

    if not isinstance(frame, Mapping):
        raise HyperscapeProtocolError("local peer frame must be an object")
    lane = frame.get("lane")
    if lane == "authored":
        validate_authored_envelope(frame.get("envelope"))
    elif lane == "presence":
        validate_presence_envelope(frame.get("envelope"))
    else:
        raise HyperscapeProtocolError(f"unknown local peer lane {lane!r}")


def local_peer_frame(lane: str, envelope: Mapping[str, Any]) -> dict[str, Any]:
    frame = {"lane": lane, "envelope": envelope}
    validate_local_peer_frame(frame)
    return frame


def canonical_json(value: Mapping[str, Any]) -> str:
    """Match serde_json's checked-in pretty fixture representation."""

    return json.dumps(value, indent=2, separators=(",", ": "), allow_nan=False) + "\n"


def set_transform_envelope(
    *,
    message_id: str,
    sender: str,
    sequence: int,
    entity: str,
    translation: Sequence[float],
    rotation_wxyz: Sequence[float],
    scale: Sequence[float],
) -> dict[str, Any]:
    envelope = {
        "header": {
            "version": dict(PROTOCOL_VERSION),
            "message_id": _uuid(message_id, "message ID"),
            "sender": _uuid(sender, "sender ID"),
            "sequence": sequence,
        },
        "command": {
            "type": "set_entity_transform",
            "entity": _uuid(entity, "entity ID"),
            "transform": {
                "translation": _finite_vector(translation, 3, "translation"),
                "rotation_wxyz": _finite_vector(rotation_wxyz, 4, "rotation"),
                "scale": _finite_vector(scale, 3, "scale"),
            },
        },
    }
    validate_authored_envelope(envelope)
    return envelope


class PresenceInbox:
    """Latest sender-local presence with receipt-relative expiration."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[int, float, Mapping[str, Any]]] = {}

    def accept(self, envelope: Mapping[str, Any], received_at_seconds: float) -> bool:
        validate_presence_envelope(envelope)
        received_at_seconds = _finite_number(
            received_at_seconds, "presence receipt time"
        )
        if received_at_seconds < 0.0:
            raise HyperscapeProtocolError(
                "presence receipt time must be finite and nonnegative"
            )
        header = envelope["header"]
        sender = _uuid(header["sender"], "sender ID")
        sequence = int(header["sequence"])
        current = self._records.get(sender)
        if current is not None and current[0] >= sequence:
            return False
        expiry = received_at_seconds + int(envelope["presence"]["ttl_millis"]) / 1000.0
        self._records[sender] = (sequence, expiry, envelope)
        return True

    def live(self, now_seconds: float) -> list[Mapping[str, Any]]:
        now_seconds = _finite_number(now_seconds, "presence time")
        if now_seconds < 0.0:
            raise HyperscapeProtocolError("presence time must be finite and nonnegative")
        self._records = {
            sender: record
            for sender, record in self._records.items()
            if record[1] > now_seconds
        }
        return [self._records[sender][2] for sender in sorted(self._records)]


class AuthoredEchoGuard:
    """Bounded message-ID memory for transport echo suppression."""

    def __init__(self, capacity: int = 1024) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise HyperscapeProtocolError("echo guard capacity must be positive")
        self._capacity = capacity
        self._ordered: deque[str] = deque()
        self._known: set[str] = set()

    def record_local(self, message_id: str) -> None:
        message_id = _uuid(message_id, "message ID")
        if message_id in self._known:
            return
        if len(self._ordered) == self._capacity:
            self._known.remove(self._ordered.popleft())
        self._ordered.append(message_id)
        self._known.add(message_id)

    def consume_echo(self, message_id: str) -> bool:
        message_id = _uuid(message_id, "message ID")
        if message_id not in self._known:
            return False
        self._known.remove(message_id)
        self._ordered.remove(message_id)
        return True


def _length(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _cross_length(left: Sequence[float], right: Sequence[float]) -> float:
    cross = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    return _length(cross)
