from __future__ import annotations

import bpy

from . import live_sync


def _collection_header(layout, settings, collection: str) -> None:
    row = layout.row(align=True)
    add = row.operator("hyperscape.collection_add", text="", icon="ADD")
    add.collection = collection
    remove = row.operator("hyperscape.collection_remove", text="", icon="REMOVE")
    remove.collection = collection


def _draw_generator(layout, generator) -> None:
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
        row = layout.row(align=True)
        row.prop(settings, "asset_id")
        row.operator("hyperscape.generate_asset_stable_id", text="", icon="FILE_REFRESH")
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
        row = layout.row(align=True)
        row.prop(binding, "stable_id")
        row.operator("hyperscape.generate_stable_id", text="", icon="FILE_REFRESH")
        layout.prop(binding, "frame")
        layout.prop(binding, "anchor")
        layout.prop(binding, "path")
        row = layout.row(align=True)
        row.prop(binding, "preview_frame")
        row.prop(binding, "preview_anchor")
        layout.prop(binding, "local_coordinates")
        layout.prop(binding, "ambient_coordinates")
        settings = context.scene.hyperscape
        layout.prop(settings, "frame_reparent_target", text="Re-anchor To")
        layout.operator("hyperscape.reanchor_object", icon="CON_TRACKTO")


class HYPERSCAPE_PT_live_sync(bpy.types.Panel):
    bl_label = "Local Blender ↔ Hyperscope"
    bl_parent_id = "HYPERSCAPE_PT_scene"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, _context):
        layout = self.layout
        status = live_sync.runtime_status()
        if status.active:
            layout.operator("hyperscape.live_sync_disconnect", icon="UNLINKED")
        else:
            layout.operator("hyperscape.live_sync_connect", icon="LINKED")
        layout.label(text=f"State: {status.state}")
        if status.peer_id:
            layout.label(text=f"Peer: {status.peer_id[:8]}…")
        layout.label(
            text=(
                f"{status.bound_entities} entities · "
                f"{status.remote_peers} remote peers"
            )
        )
        layout.label(
            text=(
                f"Authored: {status.authored_sent} sent · "
                f"{status.authored_applied} applied · "
                f"{status.authored_ignored} ignored"
            )
        )
        layout.label(
            text=(
                f"Leases: {status.lease_claims} claimed · "
                f"{status.lease_contentions} contended · "
                f"{status.authored_blocked} edits held"
            )
        )
        if status.transport is not None:
            layout.label(
                text=(
                    f"Delivery gaps/restarts: {status.transport.gaps}/"
                    f"{status.transport.restarts}"
                )
            )
        overlay = live_sync.overlay_status()
        layout.label(
            text=(
                f"Viewport overlay: {overlay.peers} peers · "
                f"{overlay.segments} segments"
            ),
            icon="HIDE_OFF" if overlay.active else "HIDE_ON",
        )
        if overlay.last_error:
            layout.label(text=overlay.last_error, icon="ERROR")
        if status.detail:
            layout.label(text=status.detail, icon="ERROR")
        layout.label(text="Direct single-writer demo; HHHS owns durable repair.")


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
        row = layout.row(align=True)
        row.prop(frame, "stable_id")
        row.operator("hyperscape.generate_frame_stable_id", text="", icon="FILE_REFRESH")
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
            _draw_generator(layout, generator)
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
        layout.prop(path, "coordinate_frame")
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
        layout.label(text="Preserve-ambient frame/anchor transitions")
        layout.template_list(
            "UI_UL_list",
            "hyperscape_transitions",
            path,
            "transitions",
            path,
            "active_transition",
            rows=3,
        )
        row = layout.row(align=True)
        row.operator("hyperscape.transition_add", text="", icon="ADD")
        row.operator("hyperscape.transition_remove", text="", icon="REMOVE")
        if path.transitions:
            transition = path.transitions[min(path.active_transition, len(path.transitions) - 1)]
            layout.prop(transition, "time_seconds")
            layout.prop(transition, "frame")
            layout.prop(transition, "anchor")


class HYPERSCAPE_PT_constraints(bpy.types.Panel):
    bl_label = "Constraint Graph"
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
        if constraint.kind == "TRACK":
            layout.prop(constraint, "subject")
            layout.prop(constraint, "target")
            layout.prop(constraint, "local_offset")
        elif constraint.kind == "PROJECTION_CAMERA":
            layout.prop(constraint, "subject")
            layout.prop(constraint, "frame")
        else:
            layout.prop(constraint, "frame", text="Pinned Frame")
            layout.prop(constraint, "target", text="Surface Entity")
            layout.prop(constraint, "face")
            layout.prop(constraint, "barycentric")
            layout.prop(constraint, "normal_sign")
            layout.prop(constraint, "heading_radians")
            layout.prop(constraint, "uniform_scale")
            layout.prop(constraint, "orientation")
            layout.label(text="Pinned frame parent must equal target entity frame")
            layout.label(text="Local conformal offset (application order)")
            layout.template_list(
                "UI_UL_list",
                "hyperscape_surface_pin_generators",
                constraint,
                "local_generators",
                constraint,
                "active_local_generator",
                rows=3,
            )
            row = layout.row(align=True)
            row.operator("hyperscape.surface_pin_generator_add", text="", icon="ADD")
            row.operator("hyperscape.surface_pin_generator_remove", text="", icon="REMOVE")
            if constraint.local_generators:
                generator = constraint.local_generators[
                    min(
                        constraint.active_local_generator,
                        len(constraint.local_generators) - 1,
                    )
                ]
                _draw_generator(layout, generator)


CLASSES = (
    HYPERSCAPE_PT_scene,
    HYPERSCAPE_PT_object,
    HYPERSCAPE_PT_live_sync,
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
