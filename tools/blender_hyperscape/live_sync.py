"""Main-thread Blender adapter for the optional delivery-only local relay.

The transport thread never imports or touches Blender data. Validated frames
cross a bounded queue and are admitted here from ``bpy.app.timers``. This is a
direct, arrival-ordered single-writer demonstration path; durable replication,
repair, and multi-writer convergence remain HHHS responsibilities.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Mapping
import uuid

import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty
from mathutils import Matrix, Quaternion, Vector

from . import authoring_leases, presence_overlay, protocol, relay


TIMER_INTERVAL_SECONDS = 0.05
PRESENCE_REFRESH_SECONDS = 0.5
PRESENCE_TTL_MILLIS = 1500
MATRIX_QUANTIZATION_DIGITS = 9
MATRIX_DECOMPOSITION_TOLERANCE = 1.0e-6


class BlenderLiveSyncError(RuntimeError):
    """The Blender-side relay adapter cannot safely continue."""


@dataclass(frozen=True)
class BlenderLiveSyncStatus:
    active: bool
    state: str
    detail: str | None
    peer_id: str | None
    bound_entities: int
    remote_peers: int
    authored_sent: int
    authored_applied: int
    authored_ignored: int
    lease_claims: int
    lease_contentions: int
    authored_blocked: int
    transport: relay.RelayStatus | None


class BlenderLiveSync:
    """Own Blender-side semantic admission; all methods run on its main thread."""

    def __init__(self) -> None:
        self._main_thread_id: int | None = None
        self._scene_pointer: int | None = None
        self._transport: relay.LocalRelayTransport | None = None
        self._active = False
        self._peer_id: str | None = None
        self._next_sequence = 0
        self._authored = protocol.AuthoredInbox()
        self._presence = protocol.PresenceInbox()
        self._leases = authoring_leases.AuthoringLeaseController()
        self._lease_claims: tuple[dict[str, Any], ...] = ()
        self._lease_contentions: dict[
            str, tuple[authoring_leases.LeaseHolder, ...]
        ] = {}
        self._remote_presence: list[Mapping[str, Any]] = []
        self._observed_matrices: dict[str, tuple[float, ...]] = {}
        self._dirty_entities: set[str] = set()
        self._frame_changed = False
        self._last_presence_signature: tuple[Any, ...] | None = None
        self._last_presence_sent_seconds = -math.inf
        self._bound_entities = 0
        self._authored_sent = 0
        self._authored_applied = 0
        self._authored_ignored = 0
        self._authored_blocked = 0
        self._detail: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    def connect(
        self,
        scene: bpy.types.Scene,
        transport: relay.LocalRelayTransport,
        peer_id: str,
    ) -> None:
        self._require_main_thread(establish=True)
        if self._active:
            raise BlenderLiveSyncError("Blender live sync is already connected")
        normalized_peer = _stable_id(peer_id, "Blender peer ID")
        self._scene_pointer = scene.as_pointer()
        self._transport = transport
        self._peer_id = normalized_peer
        # Wall-clock nanoseconds keep one stable peer monotonic across ordinary
        # restarts without writing preferences for every presence update.
        self._next_sequence = min(time.time_ns(), protocol.MAX_U64)
        self._authored = protocol.AuthoredInbox()
        self._presence = protocol.PresenceInbox()
        self._leases.clear()
        self._lease_claims = ()
        self._lease_contentions = {}
        self._remote_presence = []
        self._dirty_entities.clear()
        self._frame_changed = False
        self._last_presence_signature = None
        self._last_presence_sent_seconds = -math.inf
        self._authored_sent = 0
        self._authored_applied = 0
        self._authored_ignored = 0
        self._authored_blocked = 0
        self._detail = None
        self._refresh_observed(scene)
        try:
            transport.start()
        except Exception as error:
            self._transport = None
            self._scene_pointer = None
            self._peer_id = None
            raise BlenderLiveSyncError(
                f"could not start the local relay transport: {error}"
            ) from error
        self._active = True

    def disconnect(self) -> None:
        self._require_main_thread(establish=self._main_thread_id is None)
        transport = self._transport
        self._active = False
        if transport is not None:
            transport.stop()
        self._transport = None
        self._scene_pointer = None
        self._peer_id = None
        self._remote_presence = []
        self._leases.clear()
        self._lease_claims = ()
        self._lease_contentions = {}
        self._dirty_entities.clear()
        self._authored_blocked = 0
        self._detail = None

    def mark_object_updated(self, obj: bpy.types.Object) -> None:
        self._require_main_thread()
        if not self._active:
            return
        entity = _object_entity_id(obj)
        if entity is not None:
            self._dirty_entities.add(entity)

    def note_frame_change(self) -> None:
        if not self._active:
            return
        self._require_main_thread()
        self._frame_changed = True

    def tick(self, scene: bpy.types.Scene, now_seconds: float | None = None) -> None:
        self._require_main_thread()
        if not self._active or self._transport is None or self._peer_id is None:
            return
        if scene.as_pointer() != self._scene_pointer:
            raise BlenderLiveSyncError("live sync received the wrong Blender scene")
        now = time.monotonic() if now_seconds is None else float(now_seconds)
        if not math.isfinite(now) or now < 0.0:
            raise BlenderLiveSyncError("live sync time must be finite and nonnegative")

        self._detail = None
        objects, duplicates, invalid = _entity_objects(scene)
        self._bound_entities = len(objects)
        if duplicates:
            self._detail = (
                "Duplicate stable entity IDs are excluded: "
                + ", ".join(sorted(duplicates))
            )
        elif invalid:
            self._detail = f"{invalid} bound object(s) have invalid stable IDs"

        self._admit_deliveries(objects, now)
        selection = _selected_entities(scene, objects)
        asset_id = self._authoring_asset_id(scene)
        self._lease_claims = self._leases.synchronize(asset_id, selection)
        self._publish_presence(
            scene,
            selection,
            self._lease_claims,
            now,
        )
        self._remote_presence = [
            envelope
            for envelope in self._presence.live(now)
            if _stable_id(envelope["header"]["sender"], "presence sender")
            != self._peer_id
        ]
        remote_holders = (
            authoring_leases.remote_holders(
                self._remote_presence,
                asset_id,
                exclude_peer=self._peer_id,
            )
            if asset_id is not None
            else {}
        )
        local_targets = {
            claim["target"]["entity"] for claim in self._lease_claims
        }
        self._lease_contentions = {
            entity: holders
            for entity, holders in remote_holders.items()
            if entity in local_targets
        }
        if self._frame_changed:
            # Timeline evaluation belongs in ephemeral animation presence, not
            # a flood of durable transform edits.
            self._refresh_observed(scene, objects)
            self._dirty_entities.clear()
            self._authored_blocked = 0
            self._frame_changed = False
        else:
            self._publish_dirty(objects, set(remote_holders))

    def status(self) -> BlenderLiveSyncStatus:
        transport_status = self._transport.status() if self._transport else None
        return BlenderLiveSyncStatus(
            active=self._active,
            state=transport_status.state if transport_status else "disconnected",
            detail=self._detail,
            peer_id=self._peer_id,
            bound_entities=self._bound_entities,
            remote_peers=len(self._remote_presence),
            authored_sent=self._authored_sent,
            authored_applied=self._authored_applied,
            authored_ignored=self._authored_ignored,
            lease_claims=len(self._lease_claims),
            lease_contentions=len(self._lease_contentions),
            authored_blocked=self._authored_blocked,
            transport=transport_status,
        )

    def remote_presence(self) -> tuple[Mapping[str, Any], ...]:
        """Return detached live peer samples for viewport-only presentation.

        The copies cannot mutate the sender-order/TTL inbox and are never
        written into Blender datablocks, preferences, or the ``.blend`` file.
        A future draw handler can consume this view without turning camera or
        selection presence into authored scene state.
        """

        self._require_main_thread()
        return tuple(copy.deepcopy(envelope) for envelope in self._remote_presence)

    def _admit_deliveries(
        self,
        objects: Mapping[str, bpy.types.Object],
        now_seconds: float,
    ) -> None:
        assert self._transport is not None
        for delivery in self._transport.drain():
            frame = delivery.frame
            lane = frame["lane"]
            envelope = frame["envelope"]
            if lane == "presence":
                self._presence.admit(envelope, now_seconds)
                continue
            disposition = self._authored.accept(envelope)
            if disposition != protocol.AuthoredInbox.APPLIED:
                self._authored_ignored += 1
                continue
            command = envelope["command"]
            if command["type"] != "set_entity_transform":
                self._detail = (
                    f"Blender direct sync does not apply {command['type']!r} commands"
                )
                self._authored_ignored += 1
                continue
            entity = _stable_id(command["entity"], "authored entity ID")
            obj = objects.get(entity)
            if obj is None:
                self._detail = f"No unique bound Blender object for entity {entity}"
                self._authored_ignored += 1
                continue
            try:
                _apply_wire_transform(obj, command["transform"])
            except (RuntimeError, TypeError, ValueError) as error:
                self._detail = f"Could not apply transform to {obj.name!r}: {error}"
                self._authored_ignored += 1
                continue
            self._observed_matrices[entity] = _matrix_signature(obj.matrix_world)
            self._dirty_entities.discard(entity)
            self._authored_applied += 1

    def _publish_dirty(
        self,
        objects: Mapping[str, bpy.types.Object],
        blocked_entities: set[str],
    ) -> None:
        assert self._transport is not None
        assert self._peer_id is not None
        pending = sorted(self._dirty_entities)
        self._authored_blocked = len(set(pending) & blocked_entities)
        for entity in pending:
            if entity in blocked_entities:
                continue
            obj = objects.get(entity)
            if obj is None:
                self._dirty_entities.discard(entity)
                continue
            try:
                signature, transform = _wire_transform(obj)
            except BlenderLiveSyncError as error:
                self._detail = str(error)
                self._dirty_entities.discard(entity)
                continue
            if self._observed_matrices.get(entity) == signature:
                self._dirty_entities.discard(entity)
                continue
            envelope = protocol.set_transform_envelope(
                message_id=str(uuid.uuid4()),
                sender=self._peer_id,
                sequence=self._sequence(),
                entity=entity,
                translation=transform["translation"],
                rotation_wxyz=transform["rotation_wxyz"],
                scale=transform["scale"],
            )
            try:
                self._transport.send(protocol.local_peer_frame("authored", envelope))
            except relay.RelayTransportError:
                raise
            # The network worker may enqueue an echo immediately, but semantic
            # admission cannot drain it until this main-thread call completes.
            self._authored.record_local(envelope)
            self._observed_matrices[entity] = signature
            self._dirty_entities.discard(entity)
            self._authored_sent += 1

    def _publish_presence(
        self,
        scene: bpy.types.Scene,
        selection: list[str],
        lease_claims: tuple[dict[str, Any], ...],
        now_seconds: float,
    ) -> None:
        assert self._transport is not None
        assert self._peer_id is not None
        camera = _camera_presence(scene.camera) if scene.camera is not None else None
        fps = scene.render.fps / scene.render.fps_base
        animation_seconds = max(
            0.0,
            (scene.frame_current_final - scene.frame_start) / fps,
        )
        signature = _presence_signature(
            camera,
            selection,
            lease_claims,
            animation_seconds,
        )
        if (
            signature == self._last_presence_signature
            and now_seconds - self._last_presence_sent_seconds
            < PRESENCE_REFRESH_SECONDS
        ):
            return
        envelope = protocol.presence_envelope(
            message_id=str(uuid.uuid4()),
            sender=self._peer_id,
            sequence=self._sequence(),
            ttl_millis=PRESENCE_TTL_MILLIS,
            camera=camera,
            selection=selection,
            authoring_leases=lease_claims,
            animation_seconds=animation_seconds,
        )
        self._transport.send(protocol.local_peer_frame("presence", envelope))
        # As with authored edits, the transport worker cannot be drained on
        # Blender's main thread until this publication has recorded its echo.
        self._presence.record_local(envelope)
        self._last_presence_signature = signature
        self._last_presence_sent_seconds = now_seconds

    def _authoring_asset_id(self, scene: bpy.types.Scene) -> str | None:
        settings = getattr(scene, "hyperscape", None)
        value = getattr(settings, "asset_id", "").strip()
        if not value:
            self._append_detail("Stable asset ID is required for authoring leases")
            return None
        try:
            return authoring_leases.normalize_stable_id(value, "authoring asset ID")
        except authoring_leases.AuthoringLeaseError as error:
            self._append_detail(str(error))
            return None

    def _append_detail(self, detail: str) -> None:
        if self._detail is None:
            self._detail = detail
        elif detail not in self._detail:
            self._detail = f"{self._detail}; {detail}"

    def _refresh_observed(
        self,
        scene: bpy.types.Scene,
        objects: Mapping[str, bpy.types.Object] | None = None,
    ) -> None:
        if objects is None:
            objects, duplicates, invalid = _entity_objects(scene)
            self._bound_entities = len(objects)
            if duplicates or invalid:
                self._detail = "Some bound entities are excluded from live sync"
        self._observed_matrices = {
            entity: _matrix_signature(obj.matrix_world)
            for entity, obj in objects.items()
        }

    def _sequence(self) -> int:
        if self._next_sequence > protocol.MAX_U64:
            raise BlenderLiveSyncError("Blender peer sequence exhausted")
        value = self._next_sequence
        self._next_sequence += 1
        return value

    def _require_main_thread(self, *, establish: bool = False) -> None:
        current = threading.get_ident()
        if establish and self._main_thread_id is None:
            self._main_thread_id = current
        if self._main_thread_id != current:
            raise BlenderLiveSyncError("Blender live sync must run on its main thread")


def _stable_id(value: Any, context: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise BlenderLiveSyncError(f"{context} must be a UUID") from error
    if parsed.int == 0:
        raise BlenderLiveSyncError(f"{context} must not be nil")
    return str(parsed)


def _object_entity_id(obj: bpy.types.Object) -> str | None:
    binding = getattr(obj, "hyperscape", None)
    if binding is None or not binding.enabled or not binding.stable_id.strip():
        return None
    try:
        return _stable_id(binding.stable_id.strip(), "stable entity ID")
    except BlenderLiveSyncError:
        return None


def _entity_objects(
    scene: bpy.types.Scene,
) -> tuple[dict[str, bpy.types.Object], set[str], int]:
    objects: dict[str, bpy.types.Object] = {}
    duplicates: set[str] = set()
    invalid = 0
    for obj in scene.objects:
        binding = getattr(obj, "hyperscape", None)
        if binding is None or not binding.enabled:
            continue
        try:
            entity = _stable_id(binding.stable_id.strip(), "stable entity ID")
        except BlenderLiveSyncError:
            invalid += 1
            continue
        if entity in objects:
            duplicates.add(entity)
            del objects[entity]
        elif entity not in duplicates:
            objects[entity] = obj
    return objects, duplicates, invalid


def _selected_entities(
    scene: bpy.types.Scene,
    objects: Mapping[str, bpy.types.Object],
) -> list[str]:
    view_layer = (
        bpy.context.view_layer
        if bpy.context.scene == scene
        else scene.view_layers[0]
    )
    return sorted(
        entity
        for entity, obj in objects.items()
        if obj.select_get(view_layer=view_layer)
    )


def _matrix_signature(matrix: Matrix) -> tuple[float, ...]:
    return tuple(
        round(float(component), MATRIX_QUANTIZATION_DIGITS)
        for row in matrix
        for component in row
    )


def _wire_transform(obj: bpy.types.Object) -> tuple[tuple[float, ...], dict[str, list[float]]]:
    matrix = obj.matrix_world.copy()
    values = [float(component) for row in matrix for component in row]
    if not all(math.isfinite(value) for value in values):
        raise BlenderLiveSyncError(f"{obj.name!r} has a non-finite world transform")
    translation, rotation, scale = matrix.decompose()
    if any(abs(component) <= 1.0e-12 for component in scale):
        raise BlenderLiveSyncError(f"{obj.name!r} has a zero world scale")
    recomposed = Matrix.LocRotScale(translation, rotation, scale)
    magnitude = max(1.0, *(abs(value) for value in values))
    error = max(
        abs(float(matrix[row][column] - recomposed[row][column]))
        for row in range(4)
        for column in range(4)
    )
    if error > MATRIX_DECOMPOSITION_TOLERANCE * magnitude:
        raise BlenderLiveSyncError(
            f"{obj.name!r} world transform contains shear that protocol TRS cannot encode"
        )
    return _matrix_signature(matrix), {
        "translation": [float(component) for component in translation],
        "rotation_wxyz": [
            float(rotation.w),
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
        ],
        "scale": [float(component) for component in scale],
    }


def _apply_wire_transform(obj: bpy.types.Object, transform: Mapping[str, Any]) -> None:
    translation = Vector(transform["translation"])
    rotation = Quaternion(transform["rotation_wxyz"])
    rotation.normalize()
    scale = Vector(transform["scale"])
    obj.matrix_world = Matrix.LocRotScale(translation, rotation, scale)


def _camera_presence(camera: bpy.types.Object) -> dict[str, list[float]]:
    matrix = camera.matrix_world
    rotation = matrix.to_quaternion()
    forward = rotation @ Vector((0.0, 0.0, -1.0))
    up = rotation @ Vector((0.0, 1.0, 0.0))
    return {
        "eye": [float(component) for component in matrix.translation],
        "forward": [float(component) for component in forward],
        "up": [float(component) for component in up],
    }


def _presence_signature(
    camera: Mapping[str, Any] | None,
    selection: list[str],
    authoring_lease_claims: tuple[dict[str, Any], ...],
    animation_seconds: float,
) -> tuple[Any, ...]:
    camera_values: tuple[float, ...] = ()
    if camera is not None:
        camera_values = tuple(
            round(float(component), MATRIX_QUANTIZATION_DIGITS)
            for field in ("eye", "forward", "up")
            for component in camera[field]
        )
    return (
        camera_values,
        tuple(selection),
        tuple(
            (
                claim["lease_id"],
                claim["target"]["asset"],
                claim["target"]["entity"],
            )
            for claim in authoring_lease_claims
        ),
        round(animation_seconds, MATRIX_QUANTIZATION_DIGITS),
    )


_RUNTIME = BlenderLiveSync()


def runtime_status() -> BlenderLiveSyncStatus:
    return _RUNTIME.status()


def overlay_status() -> presence_overlay.PresenceOverlayStatus:
    return presence_overlay.status()


def _addon_preferences(context) -> Any | None:
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def _ensure_peer_id(preferences) -> str:
    try:
        return _stable_id(preferences.peer_id.strip(), "Blender peer ID")
    except BlenderLiveSyncError:
        preferences.peer_id = str(uuid.uuid4())
        return preferences.peer_id


def connect(scene: bpy.types.Scene, base_url: str, token: str, peer_id: str) -> None:
    transport = relay.LocalRelayTransport(base_url, token)
    _RUNTIME.connect(scene, transport, peer_id)
    presence_overlay.start()
    presence_overlay.update(scene, _RUNTIME.remote_presence())
    if not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=0.0)


def disconnect() -> None:
    if _RUNTIME.active:
        _RUNTIME.disconnect()
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    presence_overlay.stop()


def _timer() -> float | None:
    if not _RUNTIME.active:
        return None
    window_manager = bpy.context.window_manager
    if window_manager is not None and window_manager.is_interface_locked:
        return TIMER_INTERVAL_SECONDS
    scene = next(
        (
            candidate
            for candidate in bpy.data.scenes
            if candidate.as_pointer() == _RUNTIME._scene_pointer
        ),
        None,
    )
    if scene is None:
        disconnect()
        return None
    try:
        _RUNTIME.tick(scene)
    except Exception as error:  # A UI timer must not take Blender down with it.
        _RUNTIME._detail = str(error)
    presence_overlay.update(scene, _RUNTIME.remote_presence())
    _tag_view3d_redraw()
    return TIMER_INTERVAL_SECONDS


def _tag_view3d_redraw() -> None:
    window_manager = bpy.context.window_manager
    if window_manager is None:
        return
    for window in window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


@persistent
def _depsgraph_update(_scene, depsgraph) -> None:
    if (
        not _RUNTIME.active
        or _RUNTIME._main_thread_id != threading.get_ident()
    ):
        return
    for update in depsgraph.updates:
        candidate = update.id
        if isinstance(candidate, bpy.types.Object) and update.is_updated_transform:
            _RUNTIME.mark_object_updated(getattr(candidate, "original", candidate))


@persistent
def _frame_change(_scene, _depsgraph=None) -> None:
    if (
        not _RUNTIME.active
        or _RUNTIME._main_thread_id != threading.get_ident()
    ):
        return
    _RUNTIME.note_frame_change()


@persistent
def _load_pre(_unused) -> None:
    disconnect()


class HYPERSCAPE_OT_live_sync_connect(bpy.types.Operator):
    bl_idname = "hyperscape.live_sync_connect"
    bl_label = "Connect Local Hyperscope Peer"
    bl_description = "Connect to the optional delivery-only local relay"

    relay_url: StringProperty(name="Relay URL")
    token: StringProperty(
        name="Bearer Token",
        subtype="PASSWORD",
        options={"SKIP_SAVE"},
    )

    def invoke(self, context, _event):
        preferences = _addon_preferences(context)
        self.relay_url = (
            preferences.relay_url
            if preferences is not None
            else relay.DEFAULT_RELAY_URL
        )
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "relay_url")
        layout.prop(self, "token")
        layout.label(text="Token stays in this live operator/transport only.")

    def execute(self, context):
        preferences = _addon_preferences(context)
        if preferences is None:
            self.report({"ERROR"}, "Hyperscape add-on preferences are unavailable")
            return {"CANCELLED"}
        peer_id = _ensure_peer_id(preferences)
        token = self.token
        self.token = ""
        if not token:
            self.report({"ERROR"}, "relay bearer token is required")
            return {"CANCELLED"}
        try:
            connect(context.scene, self.relay_url, token, peer_id)
        except (BlenderLiveSyncError, relay.RelayTransportError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        preferences.relay_url = self.relay_url
        return {"FINISHED"}


class HYPERSCAPE_OT_live_sync_disconnect(bpy.types.Operator):
    bl_idname = "hyperscape.live_sync_disconnect"
    bl_label = "Disconnect Local Hyperscope Peer"

    def execute(self, _context):
        disconnect()
        return {"FINISHED"}


CLASSES = (
    HYPERSCAPE_OT_live_sync_connect,
    HYPERSCAPE_OT_live_sync_disconnect,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    if _depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_update)
    if _frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_change)
    if _load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_load_pre)


def unregister() -> None:
    disconnect()
    for handlers, function in (
        (bpy.app.handlers.load_pre, _load_pre),
        (bpy.app.handlers.frame_change_post, _frame_change),
        (bpy.app.handlers.depsgraph_update_post, _depsgraph_update),
    ):
        if function in handlers:
            handlers.remove(function)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
