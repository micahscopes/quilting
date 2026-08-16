"""Small dependency-free authoring evaluator for Blender previews.

The Rust runtime remains authoritative. This module mirrors generator order and
inverse-word semantics so editor actions can preview and preserve coordinates.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class ConformalPreviewError(ValueError):
    pass


def _finite3(point: Sequence[float]) -> tuple[float, float, float]:
    if len(point) != 3 or not all(math.isfinite(float(value)) for value in point):
        raise ConformalPreviewError("point must contain three finite numbers")
    return float(point[0]), float(point[1]), float(point[2])


def _qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qconj(q: Sequence[float]) -> tuple[float, float, float, float]:
    return q[0], -q[1], -q[2], -q[3]


def _unit_quaternion(q: Sequence[float]) -> tuple[float, float, float, float]:
    if len(q) != 4:
        raise ConformalPreviewError("rotation quaternion must have four components")
    norm = math.sqrt(sum(float(value) ** 2 for value in q))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ConformalPreviewError("rotation quaternion must be finite and nonzero")
    return tuple(float(value) / norm for value in q)  # type: ignore[return-value]


def apply_generator(generator: Mapping[str, Any], point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _finite3(point)
    kind = generator.get("type")
    if kind == "translation":
        tx, ty, tz = _finite3(generator["offset"])
        return x + tx, y + ty, z + tz
    if kind == "uniform_scale":
        factor = float(generator["factor"])
        if not math.isfinite(factor) or factor == 0.0:
            raise ConformalPreviewError("scale must be finite and nonzero")
        return factor * x, factor * y, factor * z
    if kind == "rotation":
        q = _unit_quaternion(generator["quaternion_wxyz"])
        rotated = _qmul(_qmul(q, (0.0, x, y, z)), _qconj(q))
        return rotated[1], rotated[2], rotated[3]
    if kind == "sphere_reflection":
        cx, cy, cz = _finite3(generator["center"])
        radius = float(generator["radius"])
        if not math.isfinite(radius) or radius <= 0.0:
            raise ConformalPreviewError("sphere radius must be positive")
        dx, dy, dz = x - cx, y - cy, z - cz
        norm_sq = dx * dx + dy * dy + dz * dz
        if norm_sq <= 1.0e-20:
            raise ConformalPreviewError("preview point is at the inversion pole")
        scale = radius * radius / norm_sq
        return cx + scale * dx, cy + scale * dy, cz + scale * dz
    raise ConformalPreviewError(f"unknown generator type {kind!r}")


def apply_word(generators: Sequence[Mapping[str, Any]], point: Sequence[float]) -> tuple[float, float, float]:
    result = _finite3(point)
    for generator in generators:
        result = apply_generator(generator, result)
    return result


def inverse_generator(generator: Mapping[str, Any]) -> dict[str, Any]:
    kind = generator.get("type")
    if kind == "translation":
        return {"type": kind, "offset": [-float(value) for value in generator["offset"]]}
    if kind == "uniform_scale":
        factor = float(generator["factor"])
        if factor == 0.0:
            raise ConformalPreviewError("zero scale has no inverse")
        return {"type": kind, "factor": 1.0 / factor}
    if kind == "rotation":
        w, x, y, z = _unit_quaternion(generator["quaternion_wxyz"])
        return {"type": kind, "quaternion_wxyz": [w, -x, -y, -z]}
    if kind == "sphere_reflection":
        return dict(generator)
    raise ConformalPreviewError(f"unknown generator type {kind!r}")


def inverse_word(generators: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [inverse_generator(generator) for generator in reversed(generators)]


def world_word(frames: Sequence[Mapping[str, Any]], frame_index: int) -> list[dict[str, Any]]:
    if frame_index < 0 or frame_index >= len(frames):
        raise ConformalPreviewError(f"invalid frame {frame_index}")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: int | None = frame_index
    while current is not None:
        if current in seen:
            raise ConformalPreviewError("frame parent cycle")
        seen.add(current)
        frame = frames[current]
        result.extend(dict(generator) for generator in frame.get("generators", []))
        parent = frame.get("parent")
        current = None if parent is None else int(parent)
    return result


def convert_point(
    frames: Sequence[Mapping[str, Any]],
    point: Sequence[float],
    source_frame: int,
    target_frame: int,
) -> tuple[float, float, float]:
    ambient = apply_word(world_word(frames, source_frame), point)
    return apply_word(inverse_word(world_word(frames, target_frame)), ambient)


def preserve_world_reparent_word(
    frames: Sequence[Mapping[str, Any]],
    frame_index: int,
    new_parent: int | None,
) -> list[dict[str, Any]]:
    if new_parent == frame_index:
        raise ConformalPreviewError("a frame cannot parent itself")
    ancestor = new_parent
    while ancestor is not None:
        if ancestor == frame_index:
            raise ConformalPreviewError("reparenting would create a cycle")
        parent = frames[ancestor].get("parent")
        ancestor = None if parent is None else int(parent)
    old_world = world_word(frames, frame_index)
    parent_world = [] if new_parent is None else world_word(frames, new_parent)
    return [*old_world, *inverse_word(parent_world)]


def classify_wall_side(wall: Mapping[str, Any], point: Sequence[float], epsilon: float = 1.0e-7) -> int:
    """Return -1, 0, +1 for the wall's canonical signed side."""

    x, y, z = _finite3(point)
    geometry = wall["geometry"]
    if geometry["type"] == "sphere":
        cx, cy, cz = geometry["center"]
        signed = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 - float(geometry["radius"]) ** 2
    else:
        nx, ny, nz = geometry["unit_normal"]
        signed = nx * x + ny * y + nz * z - float(geometry["offset"])
    if abs(signed) <= epsilon:
        return 0
    return -1 if signed < 0.0 else 1
