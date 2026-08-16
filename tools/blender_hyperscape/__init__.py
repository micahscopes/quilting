"""Hyperscape conformal-scene authoring for Blender 4.2+."""

from __future__ import annotations

import bpy

from . import demo, operators, properties, ui


bl_info = {
    "name": "Hyperscape Authoring",
    "author": "Hyperspectives contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar > Hyperscape",
    "description": "Author conformal frames, walls, anchors, paths, and constraints",
    "category": "Import-Export",
}


def _import_menu(self, _context) -> None:
    self.layout.operator(operators.HYPERSCAPE_OT_import.bl_idname, text="Hyperscape glTF/GLB (.gltf/.glb)")


def _export_menu(self, _context) -> None:
    self.layout.operator(operators.HYPERSCAPE_OT_export.bl_idname, text="Hyperscape glTF/GLB (.gltf/.glb)")


def register() -> None:
    properties.register()
    operators.register()
    demo.register()
    ui.register()
    bpy.types.TOPBAR_MT_file_import.append(_import_menu)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)


def unregister() -> None:
    bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
    bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
    ui.unregister()
    demo.unregister()
    operators.unregister()
    properties.unregister()
