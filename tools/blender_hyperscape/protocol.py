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
MAX_AUTHORING_LEASES_PER_PRESENCE = 256
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


def _authoring_lease(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HyperscapeProtocolError(f"{context} must be an object")
    target = value.get("target")
    if not isinstance(target, Mapping):
        raise HyperscapeProtocolError(f"{context} target must be an object")
    return {
        "lease_id": _uuid(value.get("lease_id"), "authoring lease ID"),
        "target": {
            "asset": _uuid(target.get("asset"), "authoring lease asset ID"),
            "entity": _uuid(target.get("entity"), "authoring lease entity ID"),
        },
    }


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
    authoring_leases = presence.get("authoring_leases", [])
    if not isinstance(authoring_leases, list):
        raise HyperscapeProtocolError("presence authoring leases must be an array")
    if len(authoring_leases) > MAX_AUTHORING_LEASES_PER_PRESENCE:
        raise HyperscapeProtocolError("presence has too many authoring lease claims")
    lease_ids: set[str] = set()
    lease_targets: set[tuple[str, str]] = set()
    for index, lease in enumerate(authoring_leases):
        normalized = _authoring_lease(lease, f"authoring lease {index}")
        lease_id = normalized["lease_id"]
        target = normalized["target"]
        target_identity = (
            target["asset"],
            target["entity"],
        )
        if lease_id in lease_ids:
            raise HyperscapeProtocolError("presence repeats an authoring lease ID")
        if target_identity in lease_targets:
            raise HyperscapeProtocolError("presence repeats an authoring lease target")
        lease_ids.add(lease_id)
        lease_targets.add(target_identity)
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


def presence_envelope(
    *,
    message_id: str,
    sender: str,
    sequence: int,
    ttl_millis: int,
    camera: Mapping[str, Any] | None = None,
    selection: Sequence[str] = (),
    authoring_leases: Sequence[Mapping[str, Any]] = (),
    focus: Mapping[str, Any] | None = None,
    active_cue: str | None = None,
    animation_seconds: float | None = None,
) -> dict[str, Any]:
    presence: dict[str, Any] = {
        "ttl_millis": ttl_millis,
        "selection": [_uuid(entity, "selected entity ID") for entity in selection],
    }
    if authoring_leases:
        presence["authoring_leases"] = [
            _authoring_lease(lease, f"authoring lease {index}")
            for index, lease in enumerate(authoring_leases)
        ]
    if camera is not None:
        presence["camera"] = {
            "eye": list(camera.get("eye", ())),
            "forward": list(camera.get("forward", ())),
            "up": list(camera.get("up", ())),
        }
    if focus is not None:
        presence["focus"] = {
            "center": list(focus.get("center", ())),
            "radius": focus.get("radius"),
            "inversion_enabled": focus.get("inversion_enabled"),
        }
    if active_cue is not None:
        presence["active_cue"] = _uuid(active_cue, "active cue ID")
    if animation_seconds is not None:
        presence["animation_seconds"] = animation_seconds
    envelope = {
        "header": {
            "version": dict(PROTOCOL_VERSION),
            "message_id": _uuid(message_id, "message ID"),
            "sender": _uuid(sender, "sender ID"),
            "sequence": sequence,
        },
        "presence": presence,
    }
    validate_presence_envelope(envelope)
    return envelope


def _require_positive_capacity(capacity: Any, context: str) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise HyperscapeProtocolError(f"{context} capacity must be positive")
    return capacity


class _BoundedMessageMemory:
    """Python equivalent of ``hyperscope_app::BoundedMessageMemory``."""

    def __init__(self, capacity: int, context: str) -> None:
        self._capacity = _require_positive_capacity(capacity, context)
        self._ordered: deque[str] = deque()
        self._known: set[str] = set()

    def contains(self, message_id: str) -> bool:
        return _uuid(message_id, "message ID") in self._known

    def insert(self, message_id: str) -> None:
        message_id = _uuid(message_id, "message ID")
        if message_id in self._known:
            return
        if len(self._ordered) == self._capacity:
            self._known.remove(self._ordered.popleft())
        self._ordered.append(message_id)
        self._known.add(message_id)

    def remove(self, message_id: str) -> bool:
        message_id = _uuid(message_id, "message ID")
        if message_id not in self._known:
            return False
        self._known.remove(message_id)
        self._ordered.remove(message_id)
        return True


class _BoundedSenderSequences:
    """Bounded latest-sequence memory with UUID-normalized sender keys."""

    def __init__(self, capacity: int, context: str) -> None:
        self._capacity = _require_positive_capacity(capacity, context)
        self._ordered: deque[str] = deque()
        self._latest: dict[str, int] = {}

    def is_stale(self, sender: str, sequence: int) -> bool:
        sender = _uuid(sender, "sender ID")
        return self._latest.get(sender, -1) >= sequence

    def observe(self, sender: str, sequence: int) -> None:
        sender = _uuid(sender, "sender ID")
        if sender in self._latest:
            self._latest[sender] = max(self._latest[sender], sequence)
            return
        if len(self._ordered) == self._capacity:
            del self._latest[self._ordered.popleft()]
        self._ordered.append(sender)
        self._latest[sender] = sequence


class PresenceInbox:
    """Match Rust's bounded presence duplicate, stale, echo, and TTL policy."""

    APPLIED = "applied"
    IGNORED_DUPLICATE = "ignored_duplicate"
    IGNORED_STALE = "ignored_stale"
    IGNORED_ECHO = "ignored_echo"

    def __init__(self, capacity: int = 4096) -> None:
        self._records: dict[str, tuple[int, float, Mapping[str, Any]]] = {}
        self._seen = _BoundedMessageMemory(capacity, "presence inbox")
        self._local_echoes = _BoundedMessageMemory(capacity, "presence inbox")
        self._sender_sequences = _BoundedSenderSequences(
            capacity, "presence inbox"
        )

    def record_local(self, envelope: Mapping[str, Any]) -> None:
        validate_presence_envelope(envelope)
        header = envelope["header"]
        self._local_echoes.insert(header["message_id"])
        self._sender_sequences.observe(header["sender"], header["sequence"])

    def admit(self, envelope: Mapping[str, Any], received_at_seconds: float) -> str:
        validate_presence_envelope(envelope)
        received_at_seconds = _finite_number(
            received_at_seconds, "presence receipt time"
        )
        if received_at_seconds < 0.0:
            raise HyperscapeProtocolError(
                "presence receipt time must be finite and nonnegative"
            )
        header = envelope["header"]
        message_id = _uuid(header["message_id"], "message ID")
        sender = _uuid(header["sender"], "sender ID")
        sequence = int(header["sequence"])
        if self._local_echoes.remove(message_id):
            self._seen.insert(message_id)
            self._sender_sequences.observe(sender, sequence)
            return self.IGNORED_ECHO
        if self._seen.contains(message_id):
            return self.IGNORED_DUPLICATE
        if self._sender_sequences.is_stale(sender, sequence):
            self._seen.insert(message_id)
            return self.IGNORED_STALE
        expiry = received_at_seconds + int(envelope["presence"]["ttl_millis"]) / 1000.0
        self._records[sender] = (sequence, expiry, envelope)
        self._seen.insert(message_id)
        self._sender_sequences.observe(sender, sequence)
        return self.APPLIED

    def accept(self, envelope: Mapping[str, Any], received_at_seconds: float) -> bool:
        """Compatibility predicate; use :meth:`admit` for exact disposition."""

        return self.admit(envelope, received_at_seconds) == self.APPLIED

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
        self._messages = _BoundedMessageMemory(capacity, "echo guard")

    def record_local(self, message_id: str) -> None:
        self._messages.insert(message_id)

    def consume_echo(self, message_id: str) -> bool:
        return self._messages.remove(message_id)


class AuthoredInbox:
    """Match Rust's bounded authored duplicate, stale, and echo policy."""

    APPLIED = "applied"
    IGNORED_DUPLICATE = "ignored_duplicate"
    IGNORED_STALE = "ignored_stale"
    IGNORED_ECHO = "ignored_echo"

    def __init__(self, capacity: int = 4096) -> None:
        self._seen = _BoundedMessageMemory(capacity, "authored inbox")
        self._sender_sequences = _BoundedSenderSequences(capacity, "authored inbox")
        self._echoes = AuthoredEchoGuard(capacity)

    def record_local(self, envelope: Mapping[str, Any]) -> None:
        validate_authored_envelope(envelope)
        header = envelope["header"]
        self._echoes.record_local(header["message_id"])
        self._sender_sequences.observe(header["sender"], header["sequence"])

    def accept(self, envelope: Mapping[str, Any]) -> str:
        validate_authored_envelope(envelope)
        header = envelope["header"]
        message_id = _uuid(header["message_id"], "message ID")
        sender = _uuid(header["sender"], "sender ID")
        sequence = int(header["sequence"])
        if self._echoes.consume_echo(message_id):
            self._seen.insert(message_id)
            self._sender_sequences.observe(sender, sequence)
            return self.IGNORED_ECHO
        if self._seen.contains(message_id):
            return self.IGNORED_DUPLICATE
        if self._sender_sequences.is_stale(sender, sequence):
            self._seen.insert(message_id)
            return self.IGNORED_STALE
        self._seen.insert(message_id)
        self._sender_sequences.observe(sender, sequence)
        return self.APPLIED


def _length(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _cross_length(left: Sequence[float], right: Sequence[float]) -> float:
    cross = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    return _length(cross)
