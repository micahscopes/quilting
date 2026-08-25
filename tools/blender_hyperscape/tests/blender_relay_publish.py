"""Publish one real Blender transform through the delivery-only relay."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import bpy


ADDON_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADDON_PARENT))

import blender_hyperscape  # noqa: E402
from blender_hyperscape import live_sync, relay  # noqa: E402


relay_url = os.environ["HYPERSCAPE_RELAY_URL"]
relay_token = os.environ["HYPERSCAPE_RELAY_TOKEN"]
entity_id = os.environ["HYPERSCAPE_ENTITY_ID"]
peer_id = os.environ["HYPERSCAPE_PEER_ID"]

blender_hyperscape.register()
runtime = live_sync.BlenderLiveSync()
try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.name = "Relay Cube"
    obj.hyperscape.enabled = True
    obj.hyperscape.stable_id = entity_id

    transport = relay.LocalRelayTransport(
        relay_url,
        relay_token,
        poll_interval_seconds=0.01,
    )
    runtime.connect(bpy.context.scene, transport, peer_id)
    obj.location = (3.0, 4.0, 5.0)
    bpy.context.view_layer.update()
    runtime.mark_object_updated(obj)
    runtime.tick(bpy.context.scene, 10.0)

    deadline = time.monotonic() + 5.0
    while transport.status().sent_frames < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    status = runtime.status()
    assert status.authored_sent == 1
    assert transport.status().sent_frames >= 2
    print(json.dumps({
        "marker": "Hyperscape Blender relay publish passed",
        "authoredSent": status.authored_sent,
        "transportSent": transport.status().sent_frames,
        "entity": entity_id,
    }))
finally:
    if runtime.active:
        runtime.disconnect()
    blender_hyperscape.unregister()
