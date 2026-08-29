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


class HyperscapeAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    relay_url: StringProperty(
        name="Local Relay URL",
        description="Loopback origin for the optional delivery-only peer relay",
        default="http://127.0.0.1:42117",
    )
    peer_id: StringProperty(
        name="Stable Blender Peer ID",
        description="Installation-local UUID used as the protocol sender identity",
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "relay_url")
        layout.prop(self, "peer_id")
        layout.label(text="Bearer tokens are held only by the live connect operator.")


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
    stable_id: StringProperty(
        name="Stable Frame ID",
        description="Durable UUID used by Blender, glTF, Hyperscape, and authored edits",
    )
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


class HyperscapePathTransition(bpy.types.PropertyGroup):
    time_seconds: FloatProperty(name="Time", default=0.0, min=0.0, subtype="TIME")
    frame: IntProperty(name="Enter Frame", default=0, min=0)
    anchor: IntProperty(name="Select Anchor", default=-1, min=-1)


class HyperscapePath(bpy.types.PropertyGroup):
    name: StringProperty(name="Name", default="path")
    subject: PointerProperty(name="Subject", type=bpy.types.Object)
    coordinate_frame: IntProperty(
        name="Control-Point Frame",
        description="Stable frame in which all control points are authored; -1 uses the subject's initial frame",
        default=-1,
        min=-1,
    )
    looping: BoolProperty(name="Loop", default=False)
    keyframes: CollectionProperty(type=HyperscapePathKeyframe)
    active_keyframe: IntProperty(default=0, min=0)
    transitions: CollectionProperty(type=HyperscapePathTransition)
    active_transition: IntProperty(default=0, min=0)


class HyperscapeConstraint(bpy.types.PropertyGroup):
    kind: EnumProperty(
        name="Constraint",
        items=(
            ("TRACK", "Cross-frame Track", "Aim at a target resolved across conformal frames"),
            ("PROJECTION_CAMERA", "Projection Camera", "Use this camera and conformal frame for extraction"),
            (
                "SURFACE_PIN",
                "Animated Surface Pin",
                "Drive a conformal frame from a stable point on a posed QB surface",
            ),
        ),
    )
    subject: PointerProperty(name="Subject", type=bpy.types.Object)
    target: PointerProperty(name="Target", type=bpy.types.Object)
    local_offset: FloatVectorProperty(name="Target Offset", size=3, subtype="TRANSLATION")
    frame: IntProperty(name="Conformal Frame", default=0, min=0)
    face: IntProperty(
        name="Entity-local Triangle",
        description="Triangle index within the target entity's exported source topology",
        default=0,
        min=0,
    )
    barycentric: FloatVectorProperty(
        name="Barycentric Point",
        description="Stable source-face weights; values are normalized on load",
        size=3,
        default=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        min=0.0,
        max=1.0,
    )
    normal_sign: EnumProperty(
        name="Surface Side",
        items=(
            ("1", "Positive Normal", "Use the authored surface normal"),
            ("-1", "Negative Normal", "Use the opposite surface normal"),
        ),
        default="1",
    )
    heading_radians: FloatProperty(name="Material Heading", default=0.0, subtype="ANGLE")
    uniform_scale: FloatProperty(name="Conformal Scale", default=1.0, min=1.0e-6)
    orientation: EnumProperty(
        name="World Orientation",
        items=(
            ("INHERIT", "Inherit", "Keep parent and local-offset chart parity"),
            ("RIGHT_SIDE_IN", "Right-side-in", "Force orientation-preserving ambient parity"),
            ("INSIDE_OUT", "Inside-out", "Force orientation-reversing ambient parity"),
        ),
        default="INHERIT",
    )
    local_generators: CollectionProperty(type=HyperscapeGenerator)
    active_local_generator: IntProperty(default=0, min=0)


class HyperscapeObjectBinding(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Conformal Binding", default=False)
    stable_id: StringProperty(
        name="Stable Entity ID",
        description="Durable UUID shared by Blender, Hyperscape, presentations, and authored edits",
    )
    frame: IntProperty(name="Frame", default=0, min=0)
    anchor: IntProperty(name="Anchor", default=-1, min=-1)
    path: IntProperty(name="Path", default=-1, min=-1)
    preview_frame: IntProperty(name="Preview Frame", default=0, min=0)
    preview_anchor: IntProperty(name="Preview Anchor", default=-1, min=-1)
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
    HyperscapeAddonPreferences,
    HyperscapeGenerator,
    HyperscapeFrame,
    HyperscapeWall,
    HyperscapeAnchor,
    HyperscapePathKeyframe,
    HyperscapePathTransition,
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
