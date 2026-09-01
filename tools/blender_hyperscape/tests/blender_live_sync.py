"""Headless Blender check for main-thread local live-sync semantics."""

from __future__ import annotations

from pathlib import Path
import sys
import threading

import bpy
from mathutils import Matrix


ADDON_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADDON_PARENT))

import blender_hyperscape  # noqa: E402
from blender_hyperscape import live_sync, presence_overlay, protocol, relay  # noqa: E402


class FakeTransport:
    def __init__(self) -> None:
        self.started = False
        self.sent: list[dict] = []
        self.incoming: list[relay.RelayDelivery] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def send(self, frame) -> None:
        protocol.validate_local_peer_frame(frame)
        self.sent.append(frame)

    def drain(self, limit: int = 256) -> list[relay.RelayDelivery]:
        deliveries = self.incoming[:limit]
        del self.incoming[:limit]
        return deliveries

    def status(self) -> relay.RelayStatus:
        return relay.RelayStatus(state="connected" if self.started else "stopped")


def authored_frames(transport: FakeTransport) -> list[dict]:
    return [frame for frame in transport.sent if frame["lane"] == "authored"]


def presence_frames(transport: FakeTransport) -> list[dict]:
    return [frame for frame in transport.sent if frame["lane"] == "presence"]


blender_hyperscape.register()
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
obj.name = "Live Cube"
obj.hyperscape.enabled = True
obj.hyperscape.stable_id = "10000000-0000-4000-8000-000000000001"
bpy.context.scene.hyperscape.asset_id = "10000000-0000-4000-8000-000000000002"

transport = FakeTransport()
runtime = live_sync.BlenderLiveSync()
runtime.connect(
    bpy.context.scene,
    transport,
    "20000000-0000-4000-8000-000000000001",
)
assert transport.started

thread_errors: list[Exception] = []


def cross_thread_blender_access() -> None:
    try:
        runtime.mark_object_updated(obj)
    except Exception as error:
        thread_errors.append(error)


worker = threading.Thread(target=cross_thread_blender_access)
worker.start()
worker.join()
assert len(thread_errors) == 1
assert "main thread" in str(thread_errors[0])

obj.location = (1.0, 2.0, 3.0)
bpy.context.view_layer.update()
runtime.mark_object_updated(obj)
runtime.tick(bpy.context.scene, 10.0)
local = authored_frames(transport)[-1]
local_presence = presence_frames(transport)[-1]
assert [frame["lane"] for frame in transport.sent[:2]] == ["presence", "authored"]
assert local["envelope"]["command"]["transform"]["translation"] == [1.0, 2.0, 3.0]
assert local_presence["envelope"]["presence"]["authoring_leases"][0]["target"] == {
    "asset": bpy.context.scene.hyperscape.asset_id,
    "entity": obj.hyperscape.stable_id,
}

# A relay echo must not overwrite a newer local edit that has not yet been sent.
obj.location = (2.0, 2.0, 3.0)
bpy.context.view_layer.update()
transport.incoming.append(relay.RelayDelivery(cursor=1, frame=local))
transport.incoming.append(relay.RelayDelivery(cursor=2, frame=local_presence))
runtime.tick(bpy.context.scene, 10.1)
assert tuple(round(value, 6) for value in obj.matrix_world.translation) == (2.0, 2.0, 3.0)
assert runtime._presence.live(10.1) == []

remote = protocol.set_transform_envelope(
    message_id="30000000-0000-4000-8000-000000000001",
    sender="30000000-0000-4000-8000-000000000002",
    sequence=1,
    entity=obj.hyperscape.stable_id,
    translation=[4.0, 5.0, 6.0],
    rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
    scale=[1.0, 1.0, 1.0],
)
transport.incoming.append(
    relay.RelayDelivery(
        cursor=3,
        frame=protocol.local_peer_frame("authored", remote),
    )
)
before_remote_send_count = len(authored_frames(transport))
runtime.tick(bpy.context.scene, 10.2)
assert tuple(round(value, 6) for value in obj.matrix_world.translation) == (4.0, 5.0, 6.0)
runtime.mark_object_updated(obj)
runtime.tick(bpy.context.scene, 10.3)
assert len(authored_frames(transport)) == before_remote_send_count
assert runtime.status().authored_applied == 1
assert runtime.status().authored_ignored == 1

# Remote presence remains ephemeral and never mutates the Blender object.
remote_presence = protocol.presence_envelope(
    message_id="30000000-0000-4000-8000-000000000003",
    sender="30000000-0000-4000-8000-000000000002",
    sequence=2,
    ttl_millis=1500,
    selection=[obj.hyperscape.stable_id],
    authoring_leases=[
        {
            "lease_id": "30000000-0000-4000-8000-000000000004",
            "target": {
                "asset": bpy.context.scene.hyperscape.asset_id,
                "entity": obj.hyperscape.stable_id,
            },
        }
    ],
)
transport.incoming.append(
    relay.RelayDelivery(
        cursor=4,
        frame=protocol.local_peer_frame("presence", remote_presence),
    )
)
runtime.tick(bpy.context.scene, 10.4)
assert runtime.status().remote_peers == 1
remote_samples = runtime.remote_presence()
assert remote_samples[0]["presence"]["selection"] == [obj.hyperscape.stable_id]
remote_samples[0]["presence"]["selection"].clear()
assert runtime.remote_presence()[0]["presence"]["selection"] == [
    obj.hyperscape.stable_id
]
datablocks_before_overlay = (
    len(bpy.data.objects),
    len(bpy.data.cameras),
    len(bpy.data.materials),
    len(bpy.data.collections),
)
presence_overlay.update(bpy.context.scene, runtime.remote_presence())
overlay = presence_overlay.status()
assert overlay.peers == 1
assert overlay.segments >= 12
assert datablocks_before_overlay == (
    len(bpy.data.objects),
    len(bpy.data.cameras),
    len(bpy.data.materials),
    len(bpy.data.collections),
)
assert tuple(round(value, 6) for value in obj.matrix_world.translation) == (4.0, 5.0, 6.0)
obj.location = (8.0, 5.0, 6.0)
bpy.context.view_layer.update()
runtime.mark_object_updated(obj)
before_contended = len(authored_frames(transport))
runtime.tick(bpy.context.scene, 10.5)
assert len(authored_frames(transport)) == before_contended
assert runtime.status().lease_claims == 1
assert runtime.status().lease_contentions == 1
assert runtime.status().authored_blocked == 1
runtime.tick(bpy.context.scene, 12.0)
assert runtime.status().remote_peers == 0
assert len(authored_frames(transport)) == before_contended + 1
assert runtime.status().lease_contentions == 0
assert runtime.status().authored_blocked == 0
presence_overlay.update(bpy.context.scene, runtime.remote_presence())
assert presence_overlay.status().peers == 0

# Deselection publishes a complete presence snapshot with the claim omitted.
obj.select_set(False)
runtime.tick(bpy.context.scene, 12.1)
assert runtime.status().lease_claims == 0
assert "authoring_leases" not in presence_frames(transport)[-1]["envelope"]["presence"]

# Timeline evaluation updates the observed pose but never authors each frame.
obj.location = (7.0, 8.0, 9.0)
bpy.context.view_layer.update()
runtime.mark_object_updated(obj)
runtime.note_frame_change()
before_timeline = len(authored_frames(transport))
runtime.tick(bpy.context.scene, 12.1)
assert len(authored_frames(transport)) == before_timeline

# World shear cannot be represented by the protocol's explicit TRS payload.
class ShearedObject:
    name = "Sheared"
    matrix_world = Matrix.Identity(4)


ShearedObject.matrix_world[0][1] = 0.25
try:
    live_sync._wire_transform(ShearedObject())
except live_sync.BlenderLiveSyncError as error:
    assert "shear" in str(error)
else:
    raise AssertionError("a sheared matrix was encoded as protocol TRS")

runtime.disconnect()
presence_overlay.stop()
assert not transport.started
blender_hyperscape.unregister()
print("Hyperscape Blender live sync passed")
