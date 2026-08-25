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
from blender_hyperscape import live_sync, presence_overlay, relay  # noqa: E402


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

    received_deadline = time.monotonic() + 8.0
    received_started = time.monotonic()
    while runtime.status().remote_peers < 1 and time.monotonic() < received_deadline:
        runtime.tick(
            bpy.context.scene,
            10.0 + (time.monotonic() - received_started),
        )
        time.sleep(0.01)
    remote = runtime.remote_presence()
    assert len(remote) == 1
    assert remote[0]["presence"]["camera"] == {
        "eye": [8.0, 9.0, 10.0],
        "forward": [0.0, 0.0, -1.0],
        "up": [0.0, 1.0, 0.0],
    }
    assert remote[0]["presence"]["selection"] == [entity_id]
    assert remote[0]["presence"]["focus"] == {
        "center": [1.0, 2.0, 3.0],
        "radius": 4.0,
        "inversion_enabled": True,
    }
    assert remote[0]["presence"]["animation_seconds"] == 2.5
    object_count = len(bpy.data.objects)
    presence_overlay.update(bpy.context.scene, remote)
    overlay = presence_overlay.status()
    assert overlay.peers == 1
    assert overlay.segments >= 166
    assert len(bpy.data.objects) == object_count
    print(json.dumps({
        "marker": "Hyperscape Blender relay round trip passed",
        "authoredSent": status.authored_sent,
        "transportSent": transport.status().sent_frames,
        "remotePeers": len(remote),
        "overlaySegments": overlay.segments,
        "entity": entity_id,
    }))
finally:
    if runtime.active:
        runtime.disconnect()
    presence_overlay.stop()
    blender_hyperscape.unregister()
