"""Regenerate the checked Hyperscape Blender demo from one authored scene.

Run through Blender so the .blend source and both ordinary glTF containers
receive the same conformal payload and stable entity identities.
"""

from __future__ import annotations

from pathlib import Path
import sys

import bpy


ADDON_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ADDON_PARENT))

import blender_hyperscape  # noqa: E402
from blender_hyperscape import codec  # noqa: E402


def output_path() -> Path:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise SystemExit("expected one output .glb path after --")
    destination = Path(arguments[0]).resolve()
    if destination.suffix.lower() != ".glb":
        raise SystemExit("demo destination must end in .glb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


destination = output_path()
blender_hyperscape.register()
# Factory startup includes an unrelated Cube and Camera. The interactive demo
# operator is intentionally additive, but the checked release fixture must not
# silently inherit those unbound objects.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.hyperscape.create_demo()
bpy.ops.wm.save_as_mainfile(filepath=str(destination.with_suffix(".blend")))

for path in (destination, destination.with_suffix(".gltf")):
    result = bpy.ops.hyperscape.export(filepath=str(path))
    assert "FINISHED" in result, result

payload, bindings = codec.extract_asset(destination.read_bytes())
assert payload is not None
stable_ids = {
    binding["stable_id"] for binding in bindings if binding and "stable_id" in binding
}
assert len(stable_ids) == 5, stable_ids
print(f"Exported stable Hyperscape demo: {destination}")
