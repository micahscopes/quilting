from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


GENERATOR_TYPES = (
    ("TRANSLATION", "Translation", "Translate in the parent conformal frame"),
    ("ROTATION", "Rotation", "Quaternion rotation in w,x,y,z order"),
    ("UNIFORM_SCALE", "Uniform Scale", "Nonzero signed scale"),
    ("SPHERE_REFLECTION", "Sphere Reflection", "Reflection/inversion in a round sphere"),
)


class HyperscapeGenerator(bpy.types.PropertyGroup):
    kind: EnumProperty(name="Generator", items=GENERATOR_TYPES)
    offset: FloatVectorProperty(name="Offset", size=3, subtype="TRANSLATION")
    quaternion_wxyz: FloatVectorProperty(
        name="Quaternion (wxyz)", size=4, default=(1.0, 0.0, 0.0, 0.0)
    )
    factor: FloatProperty(name="Factor", default=1.0)
    center: FloatVectorProperty(name="Center", size=3, subtype="XYZ")
    radius: FloatProperty(name="Radius", default=1.0, min=1.0e-6)


class HyperscapeFrame(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="frame")
    parent: IntProperty(name="Parent", default=-1, min=-1)
    generators: CollectionProperty(type=HyperscapeGenerator)
    active_generator: IntProperty(default=0, min=0)


class HyperscapeWall(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="wall")
    frame: IntProperty(name="Frame", default=0, min=0)
    geometry: EnumProperty(
        name="Geometry",
        items=(("SPHERE", "Sphere", "Round sphere wall"), ("PLANE", "Plane", "Oriented plane wall")),
    )
    center: FloatVectorProperty(name="Center", size=3, subtype="XYZ")
    radius: FloatProperty(name="Radius", default=1.0, min=1.0e-6)
    unit_normal: FloatVectorProperty(name="Normal", size=3, default=(0.0, 0.0, 1.0), subtype="DIRECTION")
    offset: FloatProperty(name="Offset", default=0.0)
    preview_inside: BoolProperty(name="Preview Complement", default=False)


class HyperscapeAnchor(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="anchor")
    frame: IntProperty(name="Frame", default=0, min=0)
    flipped_walls: StringProperty(
        name="Flipped Walls",
        description="Comma-separated wall indices whose complementary side is selected",
    )


class HyperscapePathKeyframe(bpy.types.PropertyGroup):
    time_seconds: FloatProperty(name="Time", default=0.0, min=0.0, subtype="TIME")
    point: FloatVectorProperty(name="Point", size=3, subtype="XYZ")


class HyperscapePath(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="path")
    subject: PointerProperty(name="Subject", type=bpy.types.Object)
    looping: BoolProperty(name="Loop", default=False)
    keyframes: CollectionProperty(type=HyperscapePathKeyframe)
    active_keyframe: IntProperty(default=0, min=0)


class HyperscapeConstraint(bpy.types.PropertyGroup):
    kind: EnumProperty(
        name="Constraint",
        items=(
            ("TRACK", "Cross-frame Track", "Aim at a target resolved across conformal frames"),
            ("PROJECTION_CAMERA", "Projection Camera", "Use this camera and conformal frame for extraction"),
        ),
    )
    subject: PointerProperty(name="Subject", type=bpy.types.Object)
    target: PointerProperty(name="Target", type=bpy.types.Object)
    local_offset: FloatVectorProperty(name="Target Offset", size=3, subtype="TRANSLATION")
    frame: IntProperty(name="Projection Frame", default=0, min=0)


class HyperscapeObjectBinding(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Conformal Binding", default=False)
    frame: IntProperty(name="Frame", default=0, min=0)
    anchor: IntProperty(name="Anchor", default=-1, min=-1)
    path: IntProperty(name="Path", default=-1, min=-1)
    local_coordinates: FloatVectorProperty(name="Local", size=3, subtype="XYZ")
    ambient_coordinates: FloatVectorProperty(name="Ambient", size=3, subtype="XYZ")


class HyperscapeSceneSettings(bpy.types.PropertyGroup):
    frames: CollectionProperty(type=HyperscapeFrame)
    active_frame: IntProperty(default=0, min=0)
    frame_reparent_target: IntProperty(name="New Parent", default=-1, min=-1)
    walls: CollectionProperty(type=HyperscapeWall)
    active_wall: IntProperty(default=0, min=0)
    anchors: CollectionProperty(type=HyperscapeAnchor)
    active_anchor: IntProperty(default=0, min=0)
    paths: CollectionProperty(type=HyperscapePath)
    active_path: IntProperty(default=0, min=0)
    constraints: CollectionProperty(type=HyperscapeConstraint)
    active_constraint: IntProperty(default=0, min=0)
    preview_time: FloatProperty(name="Preview Time", default=0.0, min=0.0, subtype="TIME")
    status: StringProperty(name="Status")


CLASSES = (
    HyperscapeGenerator,
    HyperscapeFrame,
    HyperscapeWall,
    HyperscapeAnchor,
    HyperscapePathKeyframe,
    HyperscapePath,
    HyperscapeConstraint,
    HyperscapeObjectBinding,
    HyperscapeSceneSettings,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hyperscape = PointerProperty(type=HyperscapeSceneSettings)
    bpy.types.Object.hyperscape = PointerProperty(type=HyperscapeObjectBinding)


def unregister() -> None:
    del bpy.types.Object.hyperscape
    del bpy.types.Scene.hyperscape
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
