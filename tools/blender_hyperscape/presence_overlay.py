"""Ephemeral Hyperscope peer visualization for Blender's 3D viewport.

Geometry construction is dependency-free and testable outside Blender. GPU
and ``bpy`` modules are imported only by the lifecycle adapter, so receiving
presence never creates a camera, helper object, material, collection, or other
saved datablock.
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import math
from typing import Any, Mapping, Sequence
import uuid


Point3 = tuple[float, float, float]
Segment3 = tuple[Point3, Point3]
EntityWireframes = Mapping[str, Sequence[Segment3]]
CAMERA_GLYPH_SCALE = 0.5
FOCUS_RING_SEGMENTS = 48


class PresenceOverlayError(ValueError):
    """A presence sample cannot be represented by the viewport overlay."""


@dataclass(frozen=True)
class PeerOverlayBatch:
    peer_id: str
    color: tuple[float, float, float, float]
    positions: tuple[Point3, ...]

    @property
    def segments(self) -> int:
        return len(self.positions) // 2


@dataclass(frozen=True)
class PresenceOverlayStatus:
    active: bool
    draw_handler: bool
    peers: int
    segments: int
    last_error: str | None


_DRAW_HANDLE: Any | None = None
_BATCHES: tuple[PeerOverlayBatch, ...] = ()
_LAST_ERROR: str | None = None
_ACTIVE = False


def _point3(value: Sequence[float], context: str) -> Point3:
    if len(value) != 3:
        raise PresenceOverlayError(f"{context} must contain three components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise PresenceOverlayError(f"{context} must remain finite")
    return result  # type: ignore[return-value]


def _add(left: Point3, right: Point3) -> Point3:
    return tuple(left[axis] + right[axis] for axis in range(3))  # type: ignore[return-value]


def _sub(left: Point3, right: Point3) -> Point3:
    return tuple(left[axis] - right[axis] for axis in range(3))  # type: ignore[return-value]


def _scale(vector: Point3, factor: float) -> Point3:
    return tuple(component * factor for component in vector)  # type: ignore[return-value]


def _dot(left: Point3, right: Point3) -> float:
    return sum(left[axis] * right[axis] for axis in range(3))


def _cross(left: Point3, right: Point3) -> Point3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalized(vector: Point3, context: str) -> Point3:
    length = math.sqrt(_dot(vector, vector))
    if not math.isfinite(length) or length <= 1.0e-12:
        raise PresenceOverlayError(f"{context} must be nonzero")
    return _scale(vector, 1.0 / length)


def _sphere_reflect_point_and_direction(
    point: Point3,
    direction: Point3,
    center: Point3,
    radius: float,
) -> tuple[Point3, Point3]:
    delta = _sub(point, center)
    norm_squared = _dot(delta, delta)
    if norm_squared <= 1.0e-20:
        raise PresenceOverlayError("remote camera is at the inversion pole")
    if not math.isfinite(radius) or radius <= 0.0:
        raise PresenceOverlayError("remote focus radius must be positive")
    radial = _scale(delta, 1.0 / math.sqrt(norm_squared))
    reflected_point = _add(center, _scale(delta, radius * radius / norm_squared))
    # The positive conformal scale cancels when the transported direction is
    # normalized. Retaining only the Householder factor is more stable across
    # very large and very small inversion radii.
    reflected_direction = _sub(direction, _scale(radial, 2.0 * _dot(radial, direction)))
    return reflected_point, reflected_direction


def source_camera_frame(
    camera: Mapping[str, Any],
    focus: Mapping[str, Any] | None,
) -> tuple[Point3, Point3, Point3]:
    """Return ordinary/source-chart eye, forward, and up.

    Hyperscope publishes its rendered camera in the active output chart. The
    current protocol's only non-identity chart is the shared inversion sphere;
    because sphere reflection is involutive, applying its point differential
    once maps the viewport sample back to Blender's ordinary source chart.
    """

    eye = _point3(camera["eye"], "remote camera eye")
    forward = _point3(camera["forward"], "remote camera forward")
    up = _point3(camera["up"], "remote camera up")
    if focus is not None and bool(focus.get("inversion_enabled")):
        center = _point3(focus["center"], "remote focus center")
        radius = float(focus["radius"])
        source_eye, forward = _sphere_reflect_point_and_direction(
            eye, forward, center, radius
        )
        _unused_eye, up = _sphere_reflect_point_and_direction(
            eye, up, center, radius
        )
        eye = source_eye
    forward = _normalized(forward, "remote camera forward")
    right = _normalized(_cross(forward, up), "remote camera basis")
    up = _normalized(_cross(right, forward), "remote camera up")
    return eye, forward, up


def peer_color(peer_id: str) -> tuple[float, float, float, float]:
    parsed = uuid.UUID(peer_id)
    if parsed.int == 0:
        raise PresenceOverlayError("remote peer ID must not be nil")
    hue = ((parsed.int >> 64) % 360) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.58, 1.0)
    return red, green, blue, 0.82


def _camera_segments(
    eye: Point3,
    forward: Point3,
    up: Point3,
    scale: float,
) -> list[Segment3]:
    right = _normalized(_cross(forward, up), "remote camera right")
    center = _add(eye, _scale(forward, scale * 0.72))
    horizontal = _scale(right, scale * 0.30)
    vertical = _scale(up, scale * 0.21)
    corners = [
        _add(
            _add(center, _scale(horizontal, horizontal_sign)),
            _scale(vertical, vertical_sign),
        )
        for horizontal_sign, vertical_sign in (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
    ]
    result = [(eye, _add(eye, _scale(forward, scale)))]
    result.extend((eye, corner) for corner in corners)
    result.extend((corners[index], corners[(index + 1) % 4]) for index in range(4))
    result.append((eye, _add(eye, _scale(up, scale * 0.45))))
    return result


def _focus_segments(center: Point3, radius: float) -> list[Segment3]:
    if not math.isfinite(radius) or radius <= 0.0:
        raise PresenceOverlayError("remote focus radius must be positive")
    result: list[Segment3] = []
    for fixed_axis in range(3):
        varying = [axis for axis in range(3) if axis != fixed_axis]
        points: list[Point3] = []
        for sample in range(FOCUS_RING_SEGMENTS):
            angle = math.tau * sample / FOCUS_RING_SEGMENTS
            point = list(center)
            point[varying[0]] += radius * math.cos(angle)
            point[varying[1]] += radius * math.sin(angle)
            points.append(tuple(point))  # type: ignore[arg-type]
        result.extend(
            (points[index], points[(index + 1) % FOCUS_RING_SEGMENTS])
            for index in range(FOCUS_RING_SEGMENTS)
        )
    return result


def build_overlay_batches(
    envelopes: Sequence[Mapping[str, Any]],
    entity_wireframes: EntityWireframes | None = None,
) -> tuple[PeerOverlayBatch, ...]:
    wireframes = entity_wireframes if entity_wireframes is not None else {}
    batches: list[PeerOverlayBatch] = []
    for envelope in envelopes:
        try:
            peer_id = str(envelope["header"]["sender"])
            presence = envelope["presence"]
            focus = presence.get("focus")
            segments: list[Segment3] = []
            camera = presence.get("camera")
            if camera is not None:
                eye, forward, up = source_camera_frame(camera, focus)
                scale = CAMERA_GLYPH_SCALE
                if focus is not None:
                    scale = max(1.0e-4, float(focus["radius"]) * 0.25)
                segments.extend(_camera_segments(eye, forward, up, scale))
            if focus is not None:
                segments.extend(
                    _focus_segments(
                        _point3(focus["center"], "remote focus center"),
                        float(focus["radius"]),
                    )
                )
            for entity in presence.get("selection", ()):
                segments.extend(wireframes.get(str(entity), ()))
            positions = tuple(point for segment in segments for point in segment)
            if positions:
                batches.append(
                    PeerOverlayBatch(
                        peer_id=peer_id,
                        color=peer_color(peer_id),
                        positions=positions,
                    )
                )
        except (KeyError, TypeError, ValueError):
            # Protocol admission already rejects malformed data. A defensive
            # draw projection still skips one bad peer rather than breaking all
            # viewport rendering if application state is corrupted in-process.
            continue
    return tuple(batches)


def scene_entity_wireframes(scene: Any) -> dict[str, tuple[Segment3, ...]]:
    from mathutils import Vector

    candidates: dict[str, list[Any]] = {}
    for obj in scene.objects:
        binding = getattr(obj, "hyperscape", None)
        if binding is None or not binding.enabled:
            continue
        try:
            entity = str(uuid.UUID(binding.stable_id.strip()))
        except (AttributeError, TypeError, ValueError):
            continue
        if entity == str(uuid.UUID(int=0)):
            continue
        candidates.setdefault(entity, []).append(obj)

    result: dict[str, tuple[Segment3, ...]] = {}
    for entity, objects in candidates.items():
        if len(objects) != 1:
            continue
        obj = objects[0]
        local = [tuple(float(component) for component in corner) for corner in obj.bound_box]
        world = [obj.matrix_world @ Vector(corner) for corner in local]
        segments: list[Segment3] = []
        for first in range(len(local)):
            for second in range(first + 1, len(local)):
                differing = sum(
                    not math.isclose(local[first][axis], local[second][axis], abs_tol=1.0e-9)
                    for axis in range(3)
                )
                if differing == 1:
                    segments.append(
                        (
                            tuple(float(component) for component in world[first]),
                            tuple(float(component) for component in world[second]),
                        )
                    )
        result[entity] = tuple(segments)
    return result


def update(scene: Any, envelopes: Sequence[Mapping[str, Any]]) -> None:
    global _BATCHES, _LAST_ERROR
    try:
        _BATCHES = build_overlay_batches(envelopes, scene_entity_wireframes(scene))
        _LAST_ERROR = None
    except Exception as error:
        _BATCHES = ()
        _LAST_ERROR = str(error)


def snapshot() -> tuple[PeerOverlayBatch, ...]:
    return _BATCHES


def status() -> PresenceOverlayStatus:
    return PresenceOverlayStatus(
        active=_ACTIVE,
        draw_handler=_DRAW_HANDLE is not None,
        peers=len(_BATCHES),
        segments=sum(batch.segments for batch in _BATCHES),
        last_error=_LAST_ERROR,
    )


def _draw() -> None:
    global _LAST_ERROR
    if not _BATCHES:
        return
    gpu_module: Any | None = None
    previous_blend: Any | None = None
    previous_depth_test: Any | None = None
    previous_depth_mask: bool | None = None
    previous_line_width: float | None = None
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader

        gpu_module = gpu
        previous_blend = gpu.state.blend_get()
        previous_depth_test = gpu.state.depth_test_get()
        previous_depth_mask = gpu.state.depth_mask_get()
        previous_line_width = gpu.state.line_width_get()
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        gpu.state.line_width_set(2.0)
        for peer in _BATCHES:
            batch = batch_for_shader(shader, "LINES", {"pos": peer.positions})
            shader.bind()
            shader.uniform_float("color", peer.color)
            batch.draw(shader)
        _LAST_ERROR = None
    except Exception as error:
        _LAST_ERROR = str(error)
    finally:
        try:
            if gpu_module is not None:
                if previous_line_width is not None:
                    gpu_module.state.line_width_set(previous_line_width)
                if previous_depth_mask is not None:
                    gpu_module.state.depth_mask_set(previous_depth_mask)
                if previous_depth_test is not None:
                    gpu_module.state.depth_test_set(previous_depth_test)
                if previous_blend is not None:
                    gpu_module.state.blend_set(previous_blend)
        except Exception:
            pass


def start() -> None:
    global _ACTIVE, _DRAW_HANDLE, _LAST_ERROR
    _ACTIVE = True
    try:
        import bpy

        if bpy.app.background or _DRAW_HANDLE is not None:
            return
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW"
        )
        _LAST_ERROR = None
    except Exception as error:
        _LAST_ERROR = str(error)


def stop() -> None:
    global _ACTIVE, _BATCHES, _DRAW_HANDLE, _LAST_ERROR
    try:
        if _DRAW_HANDLE is not None:
            import bpy

            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
    except Exception as error:
        _LAST_ERROR = str(error)
    else:
        _LAST_ERROR = None
    _DRAW_HANDLE = None
    _BATCHES = ()
    _ACTIVE = False
