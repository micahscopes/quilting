"""Editable production-shaped example scene for the Blender extension."""

from __future__ import annotations

import math

import bpy

from .operators import _clear_collection


DEMO_COLLECTION = "Hyperscape Demo"


def _demo_collection(scene):
    collection = bpy.data.collections.get(DEMO_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(DEMO_COLLECTION)
        scene.collection.children.link(collection)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    return collection


def _move(obj, collection) -> None:
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    obj["hyperscape_demo"] = True


def _material(name: str, color: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _add_icosphere(collection, name, location, radius, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    _move(obj, collection)
    return obj


def _add_cube(collection, name, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material)
    _move(obj, collection)
    return obj


def create_demo_scene(context) -> None:
    scene = context.scene
    settings = scene.hyperscape
    settings.asset_id = "a0000000-0000-4000-8000-000000000002"
    collection = _demo_collection(scene)
    for values in (settings.frames, settings.walls, settings.anchors, settings.paths, settings.constraints):
        _clear_collection(values)

    blue = _material("HS Blue", (0.08, 0.25, 0.8, 1.0))
    orange = _material("HS Orange", (0.95, 0.22, 0.03, 1.0))
    green = _material("HS Green", (0.08, 0.65, 0.22, 1.0))
    gray = _material("HS Ground", (0.12, 0.14, 0.18, 1.0))

    traveler = _add_icosphere(collection, "HS_Traveler", (-3.0, 1.2, 0.5), 0.45, orange)
    nested = _add_cube(collection, "HS_NestedLandmark", (0.2, 1.0, 0.4), (0.35, 0.35, 0.8), green)
    euclidean = _add_cube(
        collection, "HS_EuclideanLandmark", (-5.0, -2.0, 0.7), (0.6, 0.6, 0.7), blue
    )
    ground = _add_cube(collection, "HS_Ground", (0.0, 0.0, -0.25), (8.0, 8.0, 0.2), gray)

    camera_data = bpy.data.cameras.get("HS_ProjectionCamera") or bpy.data.cameras.new(
        "HS_ProjectionCamera"
    )
    camera = bpy.data.objects.new("HS_ProjectionCamera", camera_data)
    camera.location = (0.0, -10.0, 5.0)
    camera.rotation_euler = (math.radians(67.0), 0.0, 0.0)
    collection.objects.link(camera)
    camera["hyperscape_demo"] = True
    scene.camera = camera

    world = settings.frames.add()
    world.name = "world-euclidean"
    world.parent = -1

    room = settings.frames.add()
    room.name = "inversion-room"
    room.parent = 0
    translation = room.generators.add()
    translation.kind = "TRANSLATION"
    translation.offset = (0.0, 0.0, 0.5)
    inversion = room.generators.add()
    inversion.kind = "SPHERE_REFLECTION"
    inversion.center = (0.0, 0.0, 0.0)
    inversion.radius = 4.0

    nested_frame = settings.frames.add()
    nested_frame.name = "rotated-nested-frame"
    nested_frame.parent = 1
    rotation = nested_frame.generators.add()
    rotation.kind = "ROTATION"
    angle = math.radians(35.0) / 2.0
    rotation.quaternion_wxyz = (math.cos(angle), 0.0, 0.0, math.sin(angle))
    scale = nested_frame.generators.add()
    scale.kind = "UNIFORM_SCALE"
    scale.factor = 0.65

    for name, frame, center, radius in (
        ("outer-room", 0, (0.0, 0.0, 0.0), 4.0),
        ("overlapping-room", 0, (2.6, 0.0, 0.0), 3.0),
        ("nested-room", 1, (0.0, 0.0, 0.0), 1.4),
    ):
        wall = settings.walls.add()
        wall.name = name
        wall.frame = frame
        wall.geometry = "SPHERE"
        wall.center = center
        wall.radius = radius

    floor_wall = settings.walls.add()
    floor_wall.name = "floor-halfspace"
    floor_wall.frame = 0
    floor_wall.geometry = "PLANE"
    floor_wall.unit_normal = (0.0, 0.0, 1.0)
    floor_wall.offset = 0.0

    anchor = settings.anchors.add()
    anchor.name = "inside-out-room"
    anchor.frame = 1
    anchor.flipped_walls = "0"
    nested_anchor = settings.anchors.add()
    nested_anchor.name = "reanchored-overlap"
    nested_anchor.frame = 2
    nested_anchor.flipped_walls = "0,1"

    path = settings.paths.add()
    path.name = "euclidean-to-conformal-traverse"
    path.subject = traveler
    path.coordinate_frame = 0
    path.looping = True
    for time, point in (
        (0.0, (-3.0, 1.2, 0.5)),
        (2.0, (-1.2, 1.2, 0.8)),
        (4.0, (1.0, 1.2, 0.5)),
        (6.0, (3.0, 1.2, 0.7)),
        (8.0, (-3.0, 1.2, 0.5)),
    ):
        key = path.keyframes.add()
        key.time_seconds = time
        key.point = point
    for time, frame, anchor_index in ((2.0, 1, 0), (4.0, 2, 1), (6.0, 0, -1)):
        transition = path.transitions.add()
        transition.time_seconds = time
        transition.frame = frame
        transition.anchor = anchor_index

    track = settings.constraints.add()
    track.kind = "TRACK"
    track.subject = camera
    track.target = traveler
    track.local_offset = (0.0, 0.0, 0.4)
    projection = settings.constraints.add()
    projection.kind = "PROJECTION_CAMERA"
    projection.subject = camera
    projection.frame = 0

    traveler.hyperscape.enabled = True
    traveler.hyperscape.stable_id = "f0000000-0000-4000-8000-000000000001"
    traveler.hyperscape.frame = 0
    traveler.hyperscape.anchor = -1
    traveler.hyperscape.path = 0
    nested.hyperscape.enabled = True
    nested.hyperscape.stable_id = "f0000000-0000-4000-8000-000000000002"
    nested.hyperscape.frame = 2
    nested.hyperscape.anchor = 1
    euclidean.hyperscape.enabled = True
    euclidean.hyperscape.stable_id = "f0000000-0000-4000-8000-000000000004"
    euclidean.hyperscape.frame = 0
    ground.hyperscape.enabled = True
    ground.hyperscape.stable_id = "f0000000-0000-4000-8000-000000000005"
    ground.hyperscape.frame = 0
    camera.hyperscape.enabled = True
    camera.hyperscape.stable_id = "f0000000-0000-4000-8000-000000000003"
    camera.hyperscape.frame = 0

    settings.active_frame = 1
    settings.active_wall = 0
    settings.active_anchor = 0
    settings.active_path = 0
    settings.active_constraint = 0
    settings.preview_time = 0.0
    settings.status = "Created editable nested/overlapping conformal scene"

    bpy.ops.hyperscape.refresh_guides()
    bpy.ops.hyperscape.preview()


class HYPERSCAPE_OT_create_demo(bpy.types.Operator):
    bl_idname = "hyperscape.create_demo"
    bl_label = "Create Editable Conformal Demo"
    bl_description = "Create nested and overlapping walls, an animated subject, and a tracking camera"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        create_demo_scene(context)
        return {"FINISHED"}


CLASSES = (HYPERSCAPE_OT_create_demo,)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
