"""Pure-Python Hyperscape glTF/GLB extras codec.

This module intentionally has no ``bpy`` dependency. Blender operators use it,
and ordinary Python tests exercise the exact JSON/GLB round trip in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
import struct
import uuid
from typing import Any, Iterable, Mapping

VERSION = "0.1"
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK = 0x4E4F534A


class HyperscapeCodecError(ValueError):
    """Malformed interchange data or a container that cannot preserve it."""


@dataclass(frozen=True)
class GltfContainer:
    is_glb: bool
    chunks: tuple[tuple[int, bytes], ...] = ()


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HyperscapeCodecError(f"{context} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise HyperscapeCodecError(f"{context} must be finite")
    return value


def _index(value: Any, upper: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= upper:
        raise HyperscapeCodecError(f"{context} index {value!r} is outside 0..{upper}")
    return value


def _vector(value: Any, size: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise HyperscapeCodecError(f"{context} must contain {size} numbers")
    return [_finite_number(component, context) for component in value]


def validate_payload(
    payload: Mapping[str, Any],
    node_count: int,
    bindings: Iterable[Mapping[str, Any] | None] | None = None,
) -> None:
    """Validate the v0.1 authoring model at the Blender boundary."""

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 0:
        raise HyperscapeCodecError("node count must be a nonnegative integer")
    if not isinstance(payload, Mapping) or payload.get("version") != VERSION:
        raise HyperscapeCodecError(f"payload version must be {VERSION!r}")
    frames = payload.get("frames", [])
    walls = payload.get("walls", [])
    anchors = payload.get("anchors", [])
    paths = payload.get("paths", [])
    constraints = payload.get("constraints", [])
    for name, values in (
        ("frames", frames),
        ("walls", walls),
        ("anchors", anchors),
        ("paths", paths),
        ("constraints", constraints),
    ):
        if not isinstance(values, list):
            raise HyperscapeCodecError(f"{name} must be an array")

    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise HyperscapeCodecError(f"frame {frame_index} must be an object")
        if not isinstance(frame.get("name"), str):
            raise HyperscapeCodecError(f"frame {frame_index} name must be a string")
        parent = frame.get("parent")
        if parent is not None and (
            isinstance(parent, bool) or not isinstance(parent, int) or parent < 0 or parent >= frame_index
        ):
            raise HyperscapeCodecError(f"frame {frame_index} parent must precede its child")
        generators = frame.get("generators", [])
        if not isinstance(generators, list):
            raise HyperscapeCodecError(f"frame {frame_index} generators must be an array")
        for generator_index, generator in enumerate(generators):
            context = f"frame {frame_index} generator {generator_index}"
            if not isinstance(generator, Mapping):
                raise HyperscapeCodecError(f"{context} must be an object")
            kind = generator.get("type")
            if kind == "translation":
                _vector(generator.get("offset"), 3, f"{context} offset")
            elif kind == "rotation":
                quaternion = _vector(generator.get("quaternion_wxyz"), 4, f"{context} quaternion")
                if sum(component * component for component in quaternion) <= 1.0e-24:
                    raise HyperscapeCodecError(f"{context} quaternion must be nonzero")
            elif kind == "uniform_scale":
                if _finite_number(generator.get("factor"), f"{context} factor") == 0.0:
                    raise HyperscapeCodecError(f"{context} factor must be nonzero")
            elif kind == "sphere_reflection":
                _vector(generator.get("center"), 3, f"{context} center")
                if _finite_number(generator.get("radius"), f"{context} radius") <= 0.0:
                    raise HyperscapeCodecError(f"{context} radius must be positive")
            else:
                raise HyperscapeCodecError(f"{context} has unknown type {kind!r}")

    for wall_index, wall in enumerate(walls):
        if not isinstance(wall, Mapping):
            raise HyperscapeCodecError(f"wall {wall_index} must be an object")
        if not isinstance(wall.get("name"), str):
            raise HyperscapeCodecError(f"wall {wall_index} name must be a string")
        _index(wall.get("frame"), len(frames), f"wall {wall_index} frame")
        geometry = wall.get("geometry")
        if not isinstance(geometry, Mapping):
            raise HyperscapeCodecError(f"wall {wall_index} geometry must be an object")
        if geometry.get("type") == "sphere":
            _vector(geometry.get("center"), 3, f"wall {wall_index} center")
            if _finite_number(geometry.get("radius"), f"wall {wall_index} radius") <= 0.0:
                raise HyperscapeCodecError(f"wall {wall_index} radius must be positive")
        elif geometry.get("type") == "plane":
            normal = _vector(geometry.get("unit_normal"), 3, f"wall {wall_index} normal")
            norm = math.sqrt(sum(component * component for component in normal))
            if abs(norm - 1.0) > 1.0e-5:
                raise HyperscapeCodecError(f"wall {wall_index} normal must be unit length")
            _finite_number(geometry.get("offset"), f"wall {wall_index} offset")
        else:
            raise HyperscapeCodecError(f"wall {wall_index} has unknown geometry type")

    for anchor_index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            raise HyperscapeCodecError(f"anchor {anchor_index} must be an object")
        if not isinstance(anchor.get("name"), str):
            raise HyperscapeCodecError(f"anchor {anchor_index} name must be a string")
        _index(anchor.get("frame"), len(frames), f"anchor {anchor_index} frame")
        flipped = anchor.get("flipped_walls", [])
        if not isinstance(flipped, list):
            raise HyperscapeCodecError(f"anchor {anchor_index} flipped walls must be an array")
        if any(isinstance(wall, bool) or not isinstance(wall, int) for wall in flipped):
            raise HyperscapeCodecError(f"anchor {anchor_index} flipped walls must be integer indices")
        if len(set(flipped)) != len(flipped):
            raise HyperscapeCodecError(f"anchor {anchor_index} flipped walls must be unique")
        for wall in flipped:
            _index(wall, len(walls), f"anchor {anchor_index} wall")

    for path_index, path in enumerate(paths):
        if not isinstance(path, Mapping):
            raise HyperscapeCodecError(f"path {path_index} must be an object")
        if not isinstance(path.get("name"), str):
            raise HyperscapeCodecError(f"path {path_index} name must be a string")
        if not isinstance(path.get("looping", False), bool):
            raise HyperscapeCodecError(f"path {path_index} looping must be a boolean")
        _index(path.get("node"), node_count, f"path {path_index} node")
        if "coordinate_frame" in path:
            _index(path["coordinate_frame"], len(frames), f"path {path_index} coordinate frame")
        keyframes = path.get("keyframes")
        if not isinstance(keyframes, list) or not keyframes:
            raise HyperscapeCodecError(f"path {path_index} needs at least one keyframe")
        previous = -math.inf
        for key_index, keyframe in enumerate(keyframes):
            if not isinstance(keyframe, Mapping):
                raise HyperscapeCodecError(f"path {path_index} key {key_index} must be an object")
            time = _finite_number(keyframe.get("time_seconds"), f"path {path_index} key {key_index} time")
            if time < 0.0 or time <= previous:
                raise HyperscapeCodecError(f"path {path_index} keyframe times must strictly increase")
            previous = time
            _vector(keyframe.get("point"), 3, f"path {path_index} key {key_index} point")
        transitions = path.get("transitions", [])
        if not isinstance(transitions, list):
            raise HyperscapeCodecError(f"path {path_index} transitions must be an array")
        previous_transition = -math.inf
        for transition_index, transition in enumerate(transitions):
            if not isinstance(transition, Mapping):
                raise HyperscapeCodecError(
                    f"path {path_index} transition {transition_index} must be an object"
                )
            time = _finite_number(
                transition.get("time_seconds"),
                f"path {path_index} transition {transition_index} time",
            )
            if time < 0.0 or time <= previous_transition or time > previous:
                raise HyperscapeCodecError(
                    f"path {path_index} transition times must be in-range and strictly increase"
                )
            previous_transition = time
            frame = _index(
                transition.get("frame"),
                len(frames),
                f"path {path_index} transition {transition_index} frame",
            )
            if "anchor" in transition:
                anchor = _index(
                    transition["anchor"],
                    len(anchors),
                    f"path {path_index} transition {transition_index} anchor",
                )
                if anchors[anchor]["frame"] != frame:
                    raise HyperscapeCodecError(
                        f"path {path_index} transition {transition_index} anchor frame must match"
                    )

    for constraint_index, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping):
            raise HyperscapeCodecError(f"constraint {constraint_index} must be an object")
        kind = constraint.get("type")
        _index(constraint.get("node"), node_count, f"constraint {constraint_index} node")
        if kind == "track":
            _index(constraint.get("target_node"), node_count, f"constraint {constraint_index} target")
            _vector(constraint.get("local_offset", [0, 0, 0]), 3, f"constraint {constraint_index} offset")
        elif kind == "projection_camera":
            _index(constraint.get("frame"), len(frames), f"constraint {constraint_index} frame")
        else:
            raise HyperscapeCodecError(f"constraint {constraint_index} has unknown type {kind!r}")

    if bindings is not None:
        bindings = list(bindings)
        if len(bindings) != node_count:
            raise HyperscapeCodecError("node binding count must match the glTF node array")
        stable_ids: set[uuid.UUID] = set()
        for node, binding in enumerate(bindings):
            if binding is None:
                continue
            if not isinstance(binding, Mapping):
                raise HyperscapeCodecError(f"node {node} binding must be an object")
            if "stable_id" in binding:
                try:
                    stable_id = uuid.UUID(binding["stable_id"])
                except (AttributeError, TypeError, ValueError) as error:
                    raise HyperscapeCodecError(
                        f"node {node} stable_id must be a UUID"
                    ) from error
                if stable_id.int == 0:
                    raise HyperscapeCodecError(f"node {node} stable_id must not be nil")
                if stable_id in stable_ids:
                    raise HyperscapeCodecError(
                        f"node {node} repeats stable UUID {stable_id}"
                    )
                stable_ids.add(stable_id)
            _index(binding.get("frame"), len(frames), f"node {node} frame")
            if "anchor" in binding:
                anchor = _index(binding["anchor"], len(anchors), f"node {node} anchor")
                if anchors[anchor]["frame"] != binding["frame"]:
                    raise HyperscapeCodecError(f"node {node} anchor frame must match entity frame")
            if "path" in binding:
                _index(binding["path"], len(paths), f"node {node} path")
        for path_index, path in enumerate(paths):
            binding = bindings[path["node"]]
            if binding is None or binding.get("path") != path_index:
                raise HyperscapeCodecError(
                    f"path {path_index} must be referenced by its authored node binding"
                )
            if any(
                node != path["node"]
                and candidate is not None
                and candidate.get("path") == path_index
                for node, candidate in enumerate(bindings)
            ):
                raise HyperscapeCodecError(
                    f"path {path_index} is referenced by more than its authored node"
                )


def decode_gltf(data: bytes) -> tuple[dict[str, Any], GltfContainer]:
    if not data.startswith(GLB_MAGIC):
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HyperscapeCodecError(f"invalid glTF JSON: {error}") from error
        if not isinstance(document, dict):
            raise HyperscapeCodecError("glTF root must be an object")
        return document, GltfContainer(False)

    if len(data) < 20:
        raise HyperscapeCodecError("GLB header is truncated")
    magic, version, declared = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION or declared != len(data):
        raise HyperscapeCodecError("expected a complete GLB version 2 buffer")
    chunks: list[tuple[int, bytes]] = []
    cursor = 12
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise HyperscapeCodecError("GLB chunk header is truncated")
        length, kind = struct.unpack_from("<II", data, cursor)
        cursor += 8
        end = cursor + length
        if end > len(data):
            raise HyperscapeCodecError("GLB chunk is truncated")
        chunks.append((kind, data[cursor:end]))
        cursor = end
    if not chunks or chunks[0][0] != JSON_CHUNK:
        raise HyperscapeCodecError("first GLB chunk must be JSON")
    try:
        document = json.loads(chunks[0][1].rstrip(b" \x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HyperscapeCodecError(f"invalid GLB JSON chunk: {error}") from error
    if not isinstance(document, dict):
        raise HyperscapeCodecError("GLB JSON root must be an object")
    return document, GltfContainer(True, tuple(chunks))


def encode_gltf(document: Mapping[str, Any], container: GltfContainer) -> bytes:
    if not container.is_glb:
        return json.dumps(document, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    chunks = [(JSON_CHUNK, json_bytes), *container.chunks[1:]]
    if any(len(chunk) % 4 for _, chunk in chunks):
        raise HyperscapeCodecError("GLB chunks must have four-byte-aligned lengths")
    total = 12 + sum(8 + len(chunk) for _, chunk in chunks)
    output = bytearray(struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total))
    for kind, chunk in chunks:
        output.extend(struct.pack("<II", len(chunk), kind))
        output.extend(chunk)
    return bytes(output)


def inject_asset(
    data: bytes,
    payload: Mapping[str, Any],
    bindings: Mapping[int, Mapping[str, Any]],
) -> bytes:
    document, container = decode_gltf(data)
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        raise HyperscapeCodecError("glTF root needs a nodes array")
    binding_array: list[Mapping[str, Any] | None] = [None] * len(nodes)
    for node, binding in bindings.items():
        _index(node, len(nodes), "binding node")
        binding_array[node] = binding
    validate_payload(payload, len(nodes), binding_array)

    document = copy.deepcopy(document)
    extras = document.setdefault("extras", {})
    if not isinstance(extras, dict):
        raise HyperscapeCodecError("cannot add Hyperscape data to non-object root extras")
    extras["hyperscape"] = copy.deepcopy(dict(payload))
    for node_index, node in enumerate(document["nodes"]):
        if not isinstance(node, dict):
            raise HyperscapeCodecError(f"node {node_index} must be an object")
        existing_extras = node.get("extras")
        if isinstance(existing_extras, dict):
            existing_extras.pop("hyperscape", None)
    for node_index, binding in bindings.items():
        node = document["nodes"][node_index]
        node_extras = node.setdefault("extras", {})
        if not isinstance(node_extras, dict):
            raise HyperscapeCodecError(f"cannot add Hyperscape data to node {node_index} extras")
        node_extras["hyperscape"] = copy.deepcopy(dict(binding))
    return encode_gltf(document, container)


def extract_asset(data: bytes) -> tuple[dict[str, Any] | None, list[dict[str, Any] | None]]:
    document, _ = decode_gltf(data)
    nodes = document.get("nodes", [])
    if not isinstance(nodes, list):
        raise HyperscapeCodecError("glTF nodes must be an array")
    root_extras = document.get("extras")
    payload = root_extras.get("hyperscape") if isinstance(root_extras, dict) else None
    bindings: list[dict[str, Any] | None] = []
    for node in nodes:
        extras = node.get("extras") if isinstance(node, dict) else None
        binding = extras.get("hyperscape") if isinstance(extras, dict) else None
        bindings.append(copy.deepcopy(binding) if isinstance(binding, dict) else None)
    if payload is None and any(binding is not None for binding in bindings):
        raise HyperscapeCodecError("node bindings require root extras.hyperscape")
    if payload is not None:
        validate_payload(payload, len(nodes), bindings)
    return copy.deepcopy(payload), bindings


def unique_node_indices_by_name(document: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, node in enumerate(document.get("nodes", [])):
        name = node.get("name") if isinstance(node, Mapping) else None
        if not isinstance(name, str):
            continue
        if name in result:
            duplicates.add(name)
        else:
            result[name] = index
    for name in duplicates:
        result.pop(name, None)
    return result
