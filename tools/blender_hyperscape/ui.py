from __future__ import annotations

import bpy


def _collection_header(layout, settings, collection: str) -> None:
    row = layout.row(align=True)
    add = row.operator("hyperscape.collection_add", text="", icon="ADD")
    add.collection = collection
    remove = row.operator("hyperscape.collection_remove", text="", icon="REMOVE")
    remove.collection = collection


class HYPERSCAPE_PT_scene(bpy.types.Panel):
    bl_label = "Hyperscape"
    bl_idname = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Hyperscape"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hyperscape
        row = layout.row(align=True)
        row.operator("hyperscape.import", icon="IMPORT")
        row.operator("hyperscape.export", icon="EXPORT")
        layout.operator("hyperscape.create_demo", icon="SCENE_DATA")
        row = layout.row(align=True)
        row.operator("hyperscape.preview", icon="PLAY")
        row.operator("hyperscape.refresh_guides", icon="GIZMO")
        layout.operator("hyperscape.sync_guides", icon="CHECKMARK")
        layout.prop(settings, "preview_time")
        if settings.status:
            layout.label(text=settings.status, icon="INFO")


class HYPERSCAPE_PT_object(bpy.types.Panel):
    bl_label = "Entity Binding and Dual Coordinates"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj is None:
            layout.label(text="Select an entity")
            return
        binding = obj.hyperscape
        layout.prop(binding, "enabled")
        if not binding.enabled:
            return
        layout.prop(binding, "frame")
        layout.prop(binding, "anchor")
        layout.prop(binding, "path")
        layout.prop(binding, "local_coordinates")
        layout.prop(binding, "ambient_coordinates")
        settings = context.scene.hyperscape
        layout.prop(settings, "frame_reparent_target", text="Re-anchor To")
        layout.operator("hyperscape.reanchor_object", icon="CON_TRACKTO")


class HYPERSCAPE_PT_frames(bpy.types.Panel):
    bl_label = "Conformal Frame Forest"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hyperscape
        layout.template_list("UI_UL_list", "hyperscape_frames", settings, "frames", settings, "active_frame", rows=3)
        _collection_header(layout, settings, "frames")
        if not settings.frames:
            return
        frame = settings.frames[min(settings.active_frame, len(settings.frames) - 1)]
        layout.prop(frame, "name")
        layout.prop(frame, "parent")
        layout.label(text="Generator word (application order)")
        layout.template_list(
            "UI_UL_list", "hyperscape_generators", frame, "generators", frame, "active_generator", rows=3
        )
        row = layout.row(align=True)
        row.operator("hyperscape.generator_add", text="", icon="ADD")
        row.operator("hyperscape.generator_remove", text="", icon="REMOVE")
        if frame.generators:
            generator = frame.generators[min(frame.active_generator, len(frame.generators) - 1)]
            layout.prop(generator, "kind")
            if generator.kind == "TRANSLATION":
                layout.prop(generator, "offset")
            elif generator.kind == "ROTATION":
                layout.prop(generator, "quaternion_wxyz")
            elif generator.kind == "UNIFORM_SCALE":
                layout.prop(generator, "factor")
            else:
                layout.prop(generator, "center")
                layout.prop(generator, "radius")
        layout.prop(settings, "frame_reparent_target")
        layout.operator("hyperscape.reparent_frame", icon="OUTLINER_DATA_EMPTY")


class HYPERSCAPE_PT_walls(bpy.types.Panel):
    bl_label = "Round Walls and Oriented Sides"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hyperscape
        layout.template_list("UI_UL_list", "hyperscape_walls", settings, "walls", settings, "active_wall", rows=3)
        _collection_header(layout, settings, "walls")
        if not settings.walls:
            return
        wall = settings.walls[min(settings.active_wall, len(settings.walls) - 1)]
        layout.prop(wall, "name")
        layout.prop(wall, "frame")
        layout.prop(wall, "geometry")
        if wall.geometry == "SPHERE":
            layout.prop(wall, "center")
            layout.prop(wall, "radius")
        else:
            layout.prop(wall, "unit_normal")
            layout.prop(wall, "offset")
        layout.prop(wall, "preview_inside", text="Select Complementary Side")


class HYPERSCAPE_PT_anchors(bpy.types.Panel):
    bl_label = "Anchors"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hyperscape
        layout.template_list("UI_UL_list", "hyperscape_anchors", settings, "anchors", settings, "active_anchor", rows=2)
        _collection_header(layout, settings, "anchors")
        if settings.anchors:
            anchor = settings.anchors[min(settings.active_anchor, len(settings.anchors) - 1)]
            layout.prop(anchor, "name")
            layout.prop(anchor, "frame")
            layout.prop(anchor, "flipped_walls")


class HYPERSCAPE_PT_paths(bpy.types.Panel):
    bl_label = "Paths and Control Points"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hyperscape
        layout.template_list("UI_UL_list", "hyperscape_paths", settings, "paths", settings, "active_path", rows=2)
        _collection_header(layout, settings, "paths")
        if not settings.paths:
            return
        path = settings.paths[min(settings.active_path, len(settings.paths) - 1)]
        layout.prop(path, "name")
        layout.prop(path, "subject")
        layout.prop(path, "looping")
        layout.template_list(
            "UI_UL_list", "hyperscape_keys", path, "keyframes", path, "active_keyframe", rows=3
        )
        row = layout.row(align=True)
        row.operator("hyperscape.keyframe_add", text="", icon="ADD")
        row.operator("hyperscape.keyframe_remove", text="", icon="REMOVE")
        if path.keyframes:
            key = path.keyframes[min(path.active_keyframe, len(path.keyframes) - 1)]
            layout.prop(key, "time_seconds")
            layout.prop(key, "point")


class HYPERSCAPE_PT_constraints(bpy.types.Panel):
    bl_label = "Cross-frame Constraints"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hyperscape
        layout.template_list(
            "UI_UL_list", "hyperscape_constraints", settings, "constraints", settings, "active_constraint", rows=2
        )
        _collection_header(layout, settings, "constraints")
        if not settings.constraints:
            return
        constraint = settings.constraints[min(settings.active_constraint, len(settings.constraints) - 1)]
        layout.prop(constraint, "kind")
        layout.prop(constraint, "subject")
        if constraint.kind == "TRACK":
            layout.prop(constraint, "target")
            layout.prop(constraint, "local_offset")
        else:
            layout.prop(constraint, "frame")


CLASSES = (
    HYPERSCAPE_PT_scene,
    HYPERSCAPE_PT_object,
    HYPERSCAPE_PT_frames,
    HYPERSCAPE_PT_walls,
    HYPERSCAPE_PT_anchors,
    HYPERSCAPE_PT_paths,
    HYPERSCAPE_PT_constraints,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
