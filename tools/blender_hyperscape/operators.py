from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Vector

from . import codec, conformal


IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
GUIDE_COLLECTION = "Hyperscape Guides"


def _generator_dict(generator) -> dict[str, Any]:
    if generator.kind == "TRANSLATION":
        return {"type": "translation", "offset": list(generator.offset)}
    if generator.kind == "ROTATION":
        return {"type": "rotation", "quaternion_wxyz": list(generator.quaternion_wxyz)}
    if generator.kind == "UNIFORM_SCALE":
        return {"type": "uniform_scale", "factor": generator.factor}
    return {
        "type": "sphere_reflection",
        "center": list(generator.center),
        "radius": generator.radius,
    }


def _set_generator(generator, authored: dict[str, Any]) -> None:
    kind = authored["type"]
    generator.kind = kind.upper()
    if kind == "translation":
        generator.offset = authored["offset"]
    elif kind == "rotation":
        generator.quaternion_wxyz = authored["quaternion_wxyz"]
    elif kind == "uniform_scale":
        generator.factor = authored["factor"]
    else:
        generator.center = authored["center"]
        generator.radius = authored["radius"]


def _frames_dict(settings) -> list[dict[str, Any]]:
    frames = []
    for frame in settings.frames:
        encoded = {
            "name": frame.name,
            "parent": None if frame.parent < 0 else frame.parent,
            "generators": [_generator_dict(generator) for generator in frame.generators],
        }
        if frame.stable_id.strip():
            encoded["stable_id"] = frame.stable_id.strip()
        frames.append(encoded)
    return frames


def _parse_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        return sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as error:
        raise codec.HyperscapeCodecError("flipped wall indices must be comma-separated integers") from error


def build_asset(
    settings,
    node_indices: dict[str, int],
    node_count: int,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    frames = _frames_dict(settings)
    walls = []
    for wall in settings.walls:
        if wall.geometry == "SPHERE":
            geometry = {"type": "sphere", "center": list(wall.center), "radius": wall.radius}
        else:
            normal = Vector(wall.unit_normal)
            if normal.length <= 1.0e-12:
                raise codec.HyperscapeCodecError(f"plane wall {wall.name!r} has a zero normal")
            normal.normalize()
            geometry = {"type": "plane", "unit_normal": list(normal), "offset": wall.offset}
        walls.append({"name": wall.name, "frame": wall.frame, "geometry": geometry})
    anchors = [
        {"name": anchor.name, "frame": anchor.frame, "flipped_walls": _parse_indices(anchor.flipped_walls)}
        for anchor in settings.anchors
    ]
    paths = []
    for path in settings.paths:
        if path.subject is None or path.subject.name not in node_indices:
            raise codec.HyperscapeCodecError(f"path {path.name!r} needs an exported subject")
        paths.append(
            {
                "name": path.name,
                "node": node_indices[path.subject.name],
                "looping": path.looping,
                "keyframes": [
                    {"time_seconds": key.time_seconds, "point": list(key.point)}
                    for key in path.keyframes
                ],
                "transitions": [
                    {
                        "time_seconds": transition.time_seconds,
                        "frame": transition.frame,
                        **({"anchor": transition.anchor} if transition.anchor >= 0 else {}),
                    }
                    for transition in path.transitions
                ],
            }
        )
        if path.coordinate_frame >= 0:
            paths[-1]["coordinate_frame"] = path.coordinate_frame
    constraints = []
    for constraint in settings.constraints:
        if constraint.kind == "TRACK":
            if constraint.subject is None or constraint.subject.name not in node_indices:
                raise codec.HyperscapeCodecError("track constraint needs an exported subject")
            if constraint.target is None or constraint.target.name not in node_indices:
                raise codec.HyperscapeCodecError("track constraint needs an exported target")
            constraints.append(
                {
                    "type": "track",
                    "node": node_indices[constraint.subject.name],
                    "target_node": node_indices[constraint.target.name],
                    "local_offset": list(constraint.local_offset),
                }
            )
        elif constraint.kind == "PROJECTION_CAMERA":
            if constraint.subject is None or constraint.subject.name not in node_indices:
                raise codec.HyperscapeCodecError(
                    "projection-camera constraint needs an exported camera"
                )
            constraints.append(
                {
                    "type": "projection_camera",
                    "node": node_indices[constraint.subject.name],
                    "frame": constraint.frame,
                }
            )
        else:
            if constraint.target is None or constraint.target.name not in node_indices:
                raise codec.HyperscapeCodecError(
                    "surface pin needs an exported target mesh"
                )
            target_binding = constraint.target.hyperscape
            if not target_binding.enabled or not target_binding.stable_id.strip():
                raise codec.HyperscapeCodecError(
                    "surface-pin target needs a Conformal Binding and stable entity ID"
                )
            constraints.append(
                {
                    "type": "surface_pin",
                    "frame": constraint.frame,
                    "target_entity": target_binding.stable_id.strip(),
                    "face": constraint.face,
                    "barycentric": list(constraint.barycentric),
                    "normal_sign": int(constraint.normal_sign),
                    "heading_radians": constraint.heading_radians,
                    "uniform_scale": constraint.uniform_scale,
                    "orientation": constraint.orientation.lower(),
                    "local_offset": [
                        _generator_dict(generator)
                        for generator in constraint.local_generators
                    ],
                }
            )
    payload = {
        "version": codec.VERSION,
        "frames": frames,
        "walls": walls,
        "anchors": anchors,
        "paths": paths,
        "constraints": constraints,
    }
    bindings: dict[int, dict[str, Any]] = {}
    for obj in bpy.context.scene.objects:
        binding = obj.hyperscape
        if not binding.enabled or obj.name not in node_indices:
            continue
        encoded: dict[str, Any] = {"frame": binding.frame}
        if binding.stable_id.strip():
            encoded["stable_id"] = binding.stable_id.strip()
        if binding.anchor >= 0:
            encoded["anchor"] = binding.anchor
        if binding.path >= 0:
            encoded["path"] = binding.path
        bindings[node_indices[obj.name]] = encoded
    codec.validate_payload(
        payload,
        node_count,
        [bindings.get(node) for node in range(node_count)],
    )
    return payload, bindings


def _clear_collection(collection) -> None:
    while len(collection):
        collection.remove(len(collection) - 1)


def load_asset(settings, payload: dict[str, Any], bindings, node_objects: dict[int, bpy.types.Object]) -> None:
    for collection in (settings.frames, settings.walls, settings.anchors, settings.paths, settings.constraints):
        _clear_collection(collection)
    for authored in payload.get("frames", []):
        frame = settings.frames.add()
        frame.stable_id = authored.get("stable_id", "")
        frame.name = authored["name"]
        frame.parent = -1 if authored.get("parent") is None else authored["parent"]
        for encoded in authored.get("generators", []):
            _set_generator(frame.generators.add(), encoded)
    for authored in payload.get("walls", []):
        wall = settings.walls.add()
        wall.name = authored["name"]
        wall.frame = authored["frame"]
        geometry = authored["geometry"]
        wall.geometry = geometry["type"].upper()
        if geometry["type"] == "sphere":
            wall.center = geometry["center"]
            wall.radius = geometry["radius"]
        else:
            wall.unit_normal = geometry["unit_normal"]
            wall.offset = geometry["offset"]
    for authored in payload.get("anchors", []):
        anchor = settings.anchors.add()
        anchor.name = authored["name"]
        anchor.frame = authored["frame"]
        anchor.flipped_walls = ",".join(str(index) for index in authored.get("flipped_walls", []))
    for authored in payload.get("paths", []):
        path = settings.paths.add()
        path.name = authored["name"]
        path.subject = node_objects.get(authored["node"])
        path.coordinate_frame = authored.get("coordinate_frame", -1)
        path.looping = authored.get("looping", False)
        for encoded in authored["keyframes"]:
            key = path.keyframes.add()
            key.time_seconds = encoded["time_seconds"]
            key.point = encoded["point"]
        for encoded in authored.get("transitions", []):
            transition = path.transitions.add()
            transition.time_seconds = encoded["time_seconds"]
            transition.frame = encoded["frame"]
            transition.anchor = encoded.get("anchor", -1)
    objects_by_stable_id = {
        binding["stable_id"]: node_objects.get(node)
        for node, binding in enumerate(bindings)
        if binding is not None and "stable_id" in binding
    }
    for authored in payload.get("constraints", []):
        constraint = settings.constraints.add()
        constraint.kind = authored["type"].upper()
        if authored["type"] == "track":
            constraint.subject = node_objects.get(authored["node"])
            constraint.target = node_objects.get(authored["target_node"])
            constraint.local_offset = authored.get("local_offset", (0.0, 0.0, 0.0))
        elif authored["type"] == "projection_camera":
            constraint.subject = node_objects.get(authored["node"])
            constraint.frame = authored["frame"]
        else:
            constraint.frame = authored["frame"]
            constraint.target = objects_by_stable_id.get(authored["target_entity"])
            constraint.face = authored["face"]
            barycentric = authored["barycentric"]
            total = sum(barycentric)
            constraint.barycentric = tuple(value / total for value in barycentric)
            constraint.normal_sign = str(authored.get("normal_sign", 1))
            constraint.heading_radians = authored.get("heading_radians", 0.0)
            constraint.uniform_scale = authored.get("uniform_scale", 1.0)
            constraint.orientation = authored.get("orientation", "inherit").upper()
            for encoded in authored.get("local_offset", []):
                _set_generator(constraint.local_generators.add(), encoded)
    for node, binding in enumerate(bindings):
        obj = node_objects.get(node)
        if obj is None or binding is None:
            continue
        obj.hyperscape.enabled = True
        obj.hyperscape.stable_id = binding.get("stable_id", "")
        obj.hyperscape.frame = binding["frame"]
        obj.hyperscape.anchor = binding.get("anchor", -1)
        obj.hyperscape.path = binding.get("path", -1)
        obj.hyperscape.preview_frame = binding["frame"]
        obj.hyperscape.preview_anchor = binding.get("anchor", -1)
    settings.status = f"Loaded Hyperscape {payload['version']}"


def _imported_node_objects(
    document: dict[str, Any],
    imported: list[bpy.types.Object],
) -> dict[int, bpy.types.Object]:
    """Match glTF node names after Blender applies collision suffixes.

    Bound/animated nodes must have unique source names. Blender object names are
    unique too, but importing into a populated file can turn ``Subject`` into
    ``Subject.001``. Matching only newly created objects keeps that case safe.
    """

    unique_names = codec.unique_node_indices_by_name(document)
    result: dict[int, bpy.types.Object] = {}
    for source_name, node_index in unique_names.items():
        matches = [
            obj
            for obj in imported
            if obj.name == source_name
            or (
                obj.name.startswith(source_name + ".")
                and obj.name[len(source_name) + 1 :].isdigit()
            )
        ]
        if len(matches) == 1:
            result[node_index] = matches[0]
    return result


def _required_object_nodes(payload: dict[str, Any], bindings) -> set[int]:
    nodes = {index for index, binding in enumerate(bindings) if binding is not None}
    nodes.update(path["node"] for path in payload.get("paths", []))
    for constraint in payload.get("constraints", []):
        if constraint["type"] in ("track", "projection_camera"):
            nodes.add(constraint["node"])
        if constraint["type"] == "track":
            nodes.add(constraint["target_node"])
        elif constraint["type"] == "surface_pin":
            target = constraint["target_entity"]
            nodes.update(
                node
                for node, binding in enumerate(bindings)
                if binding is not None and binding.get("stable_id") == target
            )
    return nodes


class HYPERSCAPE_OT_collection_add(bpy.types.Operator):
    bl_idname = "hyperscape.collection_add"
    bl_label = "Add Hyperscape Item"
    bl_options = {"REGISTER", "UNDO"}
    collection: EnumProperty(
        items=tuple(
            (name, name.title(), "")
            for name in ("frames", "walls", "anchors", "paths", "constraints")
        )
    )

    def execute(self, context):
        values = getattr(context.scene.hyperscape, self.collection)
        item = values.add()
        if self.collection == "frames":
            item.name = f"frame-{len(values) - 1}"
            item.parent = len(values) - 2
            item.stable_id = str(uuid.uuid4())
        elif self.collection == "walls":
            item.name = f"wall-{len(values) - 1}"
        elif self.collection == "anchors":
            item.name = f"anchor-{len(values) - 1}"
        elif self.collection == "paths":
            item.name = f"path-{len(values) - 1}"
            item.keyframes.add()
        return {"FINISHED"}


class HYPERSCAPE_OT_collection_remove(bpy.types.Operator):
    bl_idname = "hyperscape.collection_remove"
    bl_label = "Remove Hyperscape Item"
    bl_options = {"REGISTER", "UNDO"}
    collection: StringProperty()

    def execute(self, context):
        settings = context.scene.hyperscape
        values = getattr(settings, self.collection)
        active_name = f"active_{self.collection[:-1] if self.collection.endswith('s') else self.collection}"
        index = min(getattr(settings, active_name, 0), max(len(values) - 1, 0))
        if values:
            values.remove(index)
            setattr(settings, active_name, min(index, max(len(values) - 1, 0)))
        return {"FINISHED"}


class HYPERSCAPE_OT_generator_add(bpy.types.Operator):
    bl_idname = "hyperscape.generator_add"
    bl_label = "Add Generator"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.frames:
            self.report({"ERROR"}, "Add a frame first")
            return {"CANCELLED"}
        frame = settings.frames[min(settings.active_frame, len(settings.frames) - 1)]
        frame.generators.add()
        frame.active_generator = len(frame.generators) - 1
        return {"FINISHED"}


class HYPERSCAPE_OT_generator_remove(bpy.types.Operator):
    bl_idname = "hyperscape.generator_remove"
    bl_label = "Remove Generator"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if settings.frames:
            frame = settings.frames[min(settings.active_frame, len(settings.frames) - 1)]
            if frame.generators:
                frame.generators.remove(min(frame.active_generator, len(frame.generators) - 1))
        return {"FINISHED"}


class HYPERSCAPE_OT_surface_pin_generator_add(bpy.types.Operator):
    bl_idname = "hyperscape.surface_pin_generator_add"
    bl_label = "Add Surface-pin Local Generator"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.constraints:
            return {"CANCELLED"}
        constraint = settings.constraints[
            min(settings.active_constraint, len(settings.constraints) - 1)
        ]
        if constraint.kind != "SURFACE_PIN":
            return {"CANCELLED"}
        constraint.local_generators.add()
        constraint.active_local_generator = len(constraint.local_generators) - 1
        return {"FINISHED"}


class HYPERSCAPE_OT_surface_pin_generator_remove(bpy.types.Operator):
    bl_idname = "hyperscape.surface_pin_generator_remove"
    bl_label = "Remove Surface-pin Local Generator"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.constraints:
            return {"CANCELLED"}
        constraint = settings.constraints[
            min(settings.active_constraint, len(settings.constraints) - 1)
        ]
        if constraint.kind == "SURFACE_PIN" and constraint.local_generators:
            index = min(
                constraint.active_local_generator,
                len(constraint.local_generators) - 1,
            )
            constraint.local_generators.remove(index)
            constraint.active_local_generator = min(
                index, max(len(constraint.local_generators) - 1, 0)
            )
        return {"FINISHED"}


class HYPERSCAPE_OT_keyframe_add(bpy.types.Operator):
    bl_idname = "hyperscape.keyframe_add"
    bl_label = "Add Path Control Point"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.paths:
            return {"CANCELLED"}
        path = settings.paths[min(settings.active_path, len(settings.paths) - 1)]
        key = path.keyframes.add()
        if len(path.keyframes) > 1:
            previous = path.keyframes[-2]
            key.time_seconds = previous.time_seconds + 1.0
            key.point = previous.point
        path.active_keyframe = len(path.keyframes) - 1
        return {"FINISHED"}


class HYPERSCAPE_OT_keyframe_remove(bpy.types.Operator):
    bl_idname = "hyperscape.keyframe_remove"
    bl_label = "Remove Path Control Point"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.paths:
            return {"CANCELLED"}
        path = settings.paths[min(settings.active_path, len(settings.paths) - 1)]
        if not path.keyframes:
            return {"CANCELLED"}
        index = min(path.active_keyframe, len(path.keyframes) - 1)
        path.keyframes.remove(index)
        path.active_keyframe = min(index, max(len(path.keyframes) - 1, 0))
        return {"FINISHED"}


class HYPERSCAPE_OT_transition_add(bpy.types.Operator):
    bl_idname = "hyperscape.transition_add"
    bl_label = "Add Frame/Anchor Transition"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.paths:
            return {"CANCELLED"}
        path = settings.paths[min(settings.active_path, len(settings.paths) - 1)]
        transition = path.transitions.add()
        if len(path.transitions) > 1:
            transition.time_seconds = path.transitions[-2].time_seconds + 1.0
        path.active_transition = len(path.transitions) - 1
        return {"FINISHED"}


class HYPERSCAPE_OT_transition_remove(bpy.types.Operator):
    bl_idname = "hyperscape.transition_remove"
    bl_label = "Remove Frame/Anchor Transition"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.paths:
            return {"CANCELLED"}
        path = settings.paths[min(settings.active_path, len(settings.paths) - 1)]
        if not path.transitions:
            return {"CANCELLED"}
        index = min(path.active_transition, len(path.transitions) - 1)
        path.transitions.remove(index)
        path.active_transition = min(index, max(len(path.transitions) - 1, 0))
        return {"FINISHED"}


class HYPERSCAPE_OT_preview(bpy.types.Operator):
    bl_idname = "hyperscape.preview"
    bl_label = "Evaluate Dual Coordinates"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        frames = _frames_dict(settings)
        try:
            for obj in context.scene.objects:
                binding = obj.hyperscape
                if binding.enabled:
                    binding.preview_frame = binding.frame
                    binding.preview_anchor = binding.anchor
            for path in settings.paths:
                if path.subject is None or not path.keyframes:
                    continue
                keys = sorted(path.keyframes, key=lambda key: key.time_seconds)
                time = settings.preview_time
                if path.looping and keys[-1].time_seconds > 0.0:
                    time %= keys[-1].time_seconds
                if time <= keys[0].time_seconds:
                    sampled = Vector(keys[0].point)
                elif time >= keys[-1].time_seconds:
                    sampled = Vector(keys[-1].point)
                else:
                    left, right = next(
                        (left, right)
                        for left, right in zip(keys, keys[1:])
                        if time <= right.time_seconds
                    )
                    span = right.time_seconds - left.time_seconds
                    alpha = (time - left.time_seconds) / span
                    sampled = Vector(left.point).lerp(Vector(right.point), alpha)
                binding = path.subject.hyperscape
                active_frame = binding.frame
                active_anchor = binding.anchor
                for transition in sorted(path.transitions, key=lambda transition: transition.time_seconds):
                    if transition.time_seconds > time:
                        break
                    active_frame = transition.frame
                    active_anchor = transition.anchor
                coordinate_frame = (
                    binding.frame if path.coordinate_frame < 0 else path.coordinate_frame
                )
                path.subject.location = conformal.convert_point(
                    frames, sampled, coordinate_frame, active_frame
                )
                binding.preview_frame = active_frame
                binding.preview_anchor = active_anchor
            for obj in context.scene.objects:
                binding = obj.hyperscape
                if not binding.enabled:
                    continue
                local = tuple(obj.location)
                ambient = conformal.apply_word(
                    conformal.world_word(frames, binding.preview_frame), local
                )
                binding.local_coordinates = local
                binding.ambient_coordinates = ambient
            settings.status = "Preview evaluated in local and ambient coordinates"
        except (conformal.ConformalPreviewError, IndexError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class HYPERSCAPE_OT_reanchor_object(bpy.types.Operator):
    bl_idname = "hyperscape.reanchor_object"
    bl_label = "Re-anchor Object, Preserve Ambient Point"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.object
        settings = context.scene.hyperscape
        if obj is None or not obj.hyperscape.enabled:
            return {"CANCELLED"}
        target = settings.frame_reparent_target
        if target < 0:
            self.report({"ERROR"}, "Choose a conformal frame index")
            return {"CANCELLED"}
        try:
            point = conformal.convert_point(_frames_dict(settings), obj.location, obj.hyperscape.frame, target)
        except (conformal.ConformalPreviewError, IndexError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        obj.location = point
        obj.hyperscape.frame = target
        return {"FINISHED"}


class HYPERSCAPE_OT_reparent_frame(bpy.types.Operator):
    bl_idname = "hyperscape.reparent_frame"
    bl_label = "Reparent Frame, Preserve World Map"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.frames:
            return {"CANCELLED"}
        index = min(settings.active_frame, len(settings.frames) - 1)
        new_parent = None if settings.frame_reparent_target < 0 else settings.frame_reparent_target
        try:
            word = conformal.preserve_world_reparent_word(_frames_dict(settings), index, new_parent)
        except (conformal.ConformalPreviewError, IndexError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        frame = settings.frames[index]
        _clear_collection(frame.generators)
        for encoded in word:
            _set_generator(frame.generators.add(), encoded)
        frame.parent = -1 if new_parent is None else new_parent
        return {"FINISHED"}


def _guide_collection(scene):
    collection = bpy.data.collections.get(GUIDE_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(GUIDE_COLLECTION)
        scene.collection.children.link(collection)
    for obj in list(collection.objects):
        if obj.get("hyperscape_guide"):
            bpy.data.objects.remove(obj, do_unlink=True)
    return collection


def _move_to_collection(obj, collection) -> None:
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    obj["hyperscape_guide"] = True


class HYPERSCAPE_OT_refresh_guides(bpy.types.Operator):
    bl_idname = "hyperscape.refresh_guides"
    bl_label = "Refresh Wall and Path Controls"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        collection = _guide_collection(context.scene)
        for index, wall in enumerate(settings.walls):
            if wall.geometry == "SPHERE":
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=wall.center)
                guide = context.object
                guide.scale = (wall.radius,) * 3
            else:
                normal = Vector(wall.unit_normal)
                if normal.length <= 1.0e-12:
                    continue
                normal.normalize()
                bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=4.0, location=normal * wall.offset)
                guide = context.object
                guide.rotation_mode = "QUATERNION"
                guide.rotation_quaternion = normal.to_track_quat("Z", "Y")
            guide.name = f"HS_Wall_{index}_{wall.name}"
            guide.display_type = "WIRE"
            guide.color = (0.15, 0.45, 1.0, 0.35) if wall.preview_inside else (1.0, 0.25, 0.1, 0.35)
            guide.show_in_front = True
            guide["hyperscape_wall_index"] = index
            _move_to_collection(guide, collection)
        for path_index, path in enumerate(settings.paths):
            if not path.keyframes:
                continue
            curve = bpy.data.curves.new(f"HS_Path_{path_index}", "CURVE")
            curve.dimensions = "3D"
            spline = curve.splines.new("POLY")
            spline.points.add(len(path.keyframes) - 1)
            for point, key in zip(spline.points, path.keyframes):
                point.co = (*key.point, 1.0)
            guide = bpy.data.objects.new(f"HS_Path_{path_index}_{path.name}", curve)
            collection.objects.link(guide)
            guide["hyperscape_guide"] = True
            guide["hyperscape_path_index"] = path_index
            guide.show_in_front = True
            for key_index, key in enumerate(path.keyframes):
                control = bpy.data.objects.new(f"HS_Control_{path_index}_{key_index}", None)
                control.empty_display_type = "SPHERE"
                control.empty_display_size = 0.12
                control.location = key.point
                control["hyperscape_guide"] = True
                control["hyperscape_path_index"] = path_index
                control["hyperscape_key_index"] = key_index
                collection.objects.link(control)
        return {"FINISHED"}


class HYPERSCAPE_OT_sync_guides(bpy.types.Operator):
    bl_idname = "hyperscape.sync_guides"
    bl_label = "Apply Wall and Path Control Transforms"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        collection = bpy.data.collections.get(GUIDE_COLLECTION)
        if collection is None:
            self.report({"ERROR"}, "Create guides first")
            return {"CANCELLED"}

        changed = 0
        try:
            for guide in collection.objects:
                wall_index = guide.get("hyperscape_wall_index")
                if wall_index is not None:
                    wall = settings.walls[int(wall_index)]
                    if wall.geometry == "SPHERE":
                        scales = [abs(float(value)) for value in guide.scale]
                        radius = sum(scales) / 3.0
                        if radius <= 1.0e-6:
                            raise codec.HyperscapeCodecError(
                                f"sphere guide {guide.name!r} has zero scale"
                            )
                        wall.center = guide.location
                        wall.radius = radius
                        guide.scale = (radius, radius, radius)
                    else:
                        normal = guide.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))
                        if normal.length <= 1.0e-12:
                            raise codec.HyperscapeCodecError(
                                f"plane guide {guide.name!r} has no usable normal"
                            )
                        normal.normalize()
                        wall.unit_normal = normal
                        wall.offset = normal.dot(guide.matrix_world.translation)
                    changed += 1

                path_index = guide.get("hyperscape_path_index")
                key_index = guide.get("hyperscape_key_index")
                if path_index is not None and key_index is not None:
                    settings.paths[int(path_index)].keyframes[int(key_index)].point = (
                        guide.matrix_world.translation
                    )
                    changed += 1
        except (IndexError, codec.HyperscapeCodecError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        settings.status = f"Applied {changed} guide transforms"
        return {"FINISHED"}


class HYPERSCAPE_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "hyperscape.export"
    bl_label = "Export Hyperscape glTF/GLB"
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})

    def execute(self, context):
        destination = Path(self.filepath)
        is_glb = destination.suffix.lower() == ".glb"
        try:
            with tempfile.TemporaryDirectory(prefix="hyperscape-blender-") as directory:
                temporary = Path(directory) / destination.name
                result = bpy.ops.export_scene.gltf(
                    filepath=str(temporary),
                    export_format="GLB" if is_glb else "GLTF_SEPARATE",
                    export_extras=True,
                    export_cameras=True,
                )
                if "FINISHED" not in result:
                    raise codec.HyperscapeCodecError("Blender glTF export did not finish")
                raw = temporary.read_bytes()
                document, _ = codec.decode_gltf(raw)
                node_indices = codec.unique_node_indices_by_name(document)
                payload, bindings = build_asset(
                    context.scene.hyperscape,
                    node_indices,
                    len(document.get("nodes", [])),
                )
                encoded = codec.inject_asset(raw, payload, bindings)
                if not is_glb:
                    for sidecar in Path(directory).iterdir():
                        if sidecar == temporary:
                            continue
                        target = destination.parent / sidecar.name
                        if sidecar.is_dir():
                            shutil.copytree(sidecar, target, dirs_exist_ok=True)
                        else:
                            shutil.copy2(sidecar, target)
                destination.write_bytes(encoded)
        except (OSError, codec.HyperscapeCodecError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        context.scene.hyperscape.status = f"Exported {destination.name}"
        return {"FINISHED"}


class HYPERSCAPE_OT_generate_stable_id(bpy.types.Operator):
    bl_idname = "hyperscape.generate_stable_id"
    bl_label = "Generate Stable Entity ID"
    bl_description = "Assign a new durable UUID to the active bound object"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.object
        if obj is None or not obj.hyperscape.enabled:
            self.report({"ERROR"}, "select an object with Conformal Binding enabled")
            return {"CANCELLED"}
        obj.hyperscape.stable_id = str(uuid.uuid4())
        return {"FINISHED"}


class HYPERSCAPE_OT_generate_frame_stable_id(bpy.types.Operator):
    bl_idname = "hyperscape.generate_frame_stable_id"
    bl_label = "Generate Stable Conformal Frame ID"
    bl_description = "Assign a new durable UUID to the active conformal frame"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.hyperscape
        if not settings.frames:
            self.report({"ERROR"}, "add a conformal frame first")
            return {"CANCELLED"}
        frame = settings.frames[min(settings.active_frame, len(settings.frames) - 1)]
        frame.stable_id = str(uuid.uuid4())
        return {"FINISHED"}


class HYPERSCAPE_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "hyperscape.import"
    bl_label = "Import Hyperscape glTF/GLB"
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})

    def execute(self, context):
        source = Path(self.filepath)
        try:
            raw = source.read_bytes()
            document, _ = codec.decode_gltf(raw)
            payload, bindings = codec.extract_asset(raw)
            if payload is None:
                raise codec.HyperscapeCodecError("file has no root extras.hyperscape payload")
            before = {obj.as_pointer() for obj in bpy.data.objects}
            result = bpy.ops.import_scene.gltf(filepath=str(source))
            if "FINISHED" not in result:
                raise codec.HyperscapeCodecError("Blender glTF import did not finish")
            imported = [obj for obj in bpy.data.objects if obj.as_pointer() not in before]
            node_objects = _imported_node_objects(document, imported)
            missing = sorted(_required_object_nodes(payload, bindings) - node_objects.keys())
            if missing:
                raise codec.HyperscapeCodecError(
                    "could not uniquely match imported objects for glTF nodes "
                    + ", ".join(str(node) for node in missing)
                )
            load_asset(context.scene.hyperscape, payload, bindings, node_objects)
        except (OSError, codec.HyperscapeCodecError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


CLASSES = (
    HYPERSCAPE_OT_collection_add,
    HYPERSCAPE_OT_collection_remove,
    HYPERSCAPE_OT_generator_add,
    HYPERSCAPE_OT_generator_remove,
    HYPERSCAPE_OT_surface_pin_generator_add,
    HYPERSCAPE_OT_surface_pin_generator_remove,
    HYPERSCAPE_OT_keyframe_add,
    HYPERSCAPE_OT_keyframe_remove,
    HYPERSCAPE_OT_transition_add,
    HYPERSCAPE_OT_transition_remove,
    HYPERSCAPE_OT_preview,
    HYPERSCAPE_OT_reanchor_object,
    HYPERSCAPE_OT_reparent_frame,
    HYPERSCAPE_OT_refresh_guides,
    HYPERSCAPE_OT_sync_guides,
    HYPERSCAPE_OT_generate_stable_id,
    HYPERSCAPE_OT_generate_frame_stable_id,
    HYPERSCAPE_OT_export,
    HYPERSCAPE_OT_import,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
