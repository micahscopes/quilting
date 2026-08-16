"""Headless Blender integration check.

Run with the command documented in the extension README. This script loads the
source extension directly, so installation is not required in CI.
"""

from __future__ import annotations

from pathlib import Path
import sys

import bpy


ADDON_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADDON_PARENT))

import blender_hyperscape  # noqa: E402
from blender_hyperscape import codec  # noqa: E402


def output_path() -> Path:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise SystemExit("expected one output .glb path after --")
    destination = Path(arguments[0]).resolve()
    if destination.suffix.lower() != ".glb":
        raise SystemExit("round-trip destination must end in .glb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


destination = output_path()
blender_hyperscape.register()
bpy.ops.hyperscape.create_demo()

source_blend = destination.with_suffix(".blend")
bpy.ops.wm.save_as_mainfile(filepath=str(source_blend))
result = bpy.ops.hyperscape.export(filepath=str(destination))
assert "FINISHED" in result, result

payload, bindings = codec.extract_asset(destination.read_bytes())
assert payload is not None
assert len(payload["frames"]) == 3
assert len(payload["walls"]) == 4
assert len(payload["anchors"]) == 2
assert len(payload["paths"]) == 1
assert len(payload["constraints"]) == 2
assert sum(binding is not None for binding in bindings) >= 3

blender_hyperscape.unregister()
bpy.ops.wm.read_factory_settings(use_empty=True)
blender_hyperscape.register()
result = getattr(bpy.ops.hyperscape, "import")(filepath=str(destination))
assert "FINISHED" in result, result
settings = bpy.context.scene.hyperscape
assert len(settings.frames) == 3
assert len(settings.walls) == 4
assert len(settings.anchors) == 2
assert len(settings.paths) == 1
assert len(settings.constraints) == 2
assert settings.paths[0].subject is not None
assert settings.constraints[0].target is not None

bpy.ops.wm.save_as_mainfile(filepath=str(destination.with_name(destination.stem + "-roundtrip.blend")))
print(f"Hyperscape Blender round trip passed: {destination}")
