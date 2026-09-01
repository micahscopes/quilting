# Hyperscape Authoring for Blender

This Blender 4.2+ extension authors the conformal metadata used by Hyperscape
without replacing Blender's ordinary scene graph or glTF exporter. Meshes,
cameras, materials, and normal node transforms remain standard glTF. The
extension adds the versioned `extras.hyperscape` payload described by
[`docs/conformal-scene-model.md`](../../docs/conformal-scene-model.md).

The **Hyperscape** tab in the 3D View sidebar provides:

- a single-parent conformal frame forest and ordered generator words;
- sphere and plane walls with complementary-side previews;
- anchor bitsets, UUID-backed entity bindings, and local/ambient coordinate inspection;
- piecewise-linear animation paths with transformable control guides;
- timed frame/anchor transitions sampled from one stable control-point chart;
- preserve-world frame reparenting and object re-anchoring;
- cross-frame tracking and projection-camera constraints; and
- `.gltf`/`.glb` import/export that preserves ordinary glTF fallback content.

The dependency-free `protocol.py` module also validates the same checked-in
v0.1 authored/presence JSON fixtures as Rust. It supplies sender-local presence
ordering with receipt-relative TTL and bounded duplicate, stale, and echo
suppression for both lanes. The optional local relay remains a delivery-only
adapter; HHHS can still wrap only authored envelopes without receiving
viewport presence.

For local bridge development, start the disabled-by-default relay and copy its
printed bearer token into the browser and Blender adapters:

```sh
cargo run -p hyperscope-web --features local-peer-relay \
  --bin hyperscope-local-peer-relay
```

It binds to `127.0.0.1:42117`, accepts only configured browser origins, retains
a bounded in-memory suffix, and reports restart/history gaps rather than
claiming persistence or repair. Merely running it changes no Blender or browser
behavior: in the **Local Blender ↔ Hyperscope** panel, choose **Connect**, enter
the printed token, and explicitly opt in. The token lives only in that operator
and the active transport; it is not stored in the `.blend` or add-on
preferences.

The network worker only fills bounded queues. A Blender application timer
validates and applies absolute ordinary-world TRS transforms on Blender's main
thread. Bound objects are matched exclusively by their stable entity UUID;
duplicates, invalid IDs, zero scale, and world shear are excluded instead of
being approximated. Camera, selection, and animation time use the ephemeral
presence lane. Timeline evaluation refreshes presence but is not converted
into a stream of authored transforms.

Selected bound objects also publish asset-scoped advisory authoring leases.
Lease IDs remain stable while selected, refresh with presence, and are omitted
on deselection; TTL expiry is the disconnect/crash fallback. If a live remote
peer claims the same entity, Blender retains the local dirty edit and reports
the contention instead of publishing it. Incoming authored edits are still
admitted: leases coordinate editors but are neither capabilities nor ACLs.

Remote browser presence is projected into a transient 3D View overlay: a
camera glyph, focus/inversion sphere, and wire bounds around uniquely bound
selected entities. Hyperscope publishes its rendered camera in the output
chart, so an active spherical inversion is reflected back into Blender's
ordinary source chart before drawing its eye and tangent frame. The overlay is
rebuilt from TTL-filtered samples, creates no Blender datablocks, and is never
saved in the `.blend`.

## Conformal geometry preview boundary

Blender should remain an authoring peer rather than a forked Hyperscope
renderer. A conformal preview therefore has two deliberately different quality
levels:

1. A generated Geometry Nodes modifier may provide an editable, non-destructive
   authoring preview. It subdivides the ordinary input mesh to an explicit
   fixed **Preview Quality**, evaluates the authored generator word pointwise,
   and leaves the source mesh untouched. Preview topology and quality are local
   editor settings; they are not exported as authored scene state or admitted
   to HHHS. This mode is useful for composition, animation blocking, and
   generator gizmos, but it does not claim screen-space, crack-free QB LOD.
2. Hyperscope remains the exact preview and presentation renderer. The existing
   live peer already shares stable object identity, camera, selection, focus,
   inversion, and authored transforms. A later Blender `POST_VIEW` adapter may
   extend the current presence overlay with an atlas-backed wire/debug surface,
   but PBR, rational QB evaluation, pole handling, shared-edge reconciliation,
   and adaptive LOD remain renderer responsibilities.

The old [Hyperblender fork](https://github.com/micahscopes/hyperblender) is
useful design archaeology, not an implementation dependency. Its durable idea
is controlling a conformal modifier with ordinary Blender objects. Its sphere
reflection path repeatedly mutated a CPU BMesh by subdividing edges near the
inversion center and then transformed every resulting vertex. That refinement
was view-independent and did not provide the current atlas's screen-space,
permutation, seam, or rational-patch guarantees, so it should not be ported.

WebGPU does not change this division. It has compute, vertex, and fragment
shader stages rather than geometry or tessellation stages. The production
backend should retain the immutable tessellation atlas and use compute for
same-pose LOD reconciliation, visible-instance compaction, and indirect draw
arguments. Regenerating tessellated vertex/index buffers every frame would add
allocation and synchronization that the resident atlas was designed to avoid.

This bridge is intentionally a direct, arrival-ordered single-writer demo. Its
cursor detects delivery gaps and process restarts but is not a scene revision.
It has no durable storage, repair, capability delegation, or multi-writer
convergence; those remain the future HHHS-backed session boundary.

## Install or build

Install the directory as an extension from Blender's Preferences, or build a
zip from the repository root with Blender available:

```sh
blender --command extension build --source-dir tools/blender_hyperscape
```

Validate the manifest and package with:

```sh
blender --command extension validate tools/blender_hyperscape-0.1.0.zip
```

The extension requests file access for the selected glTF/GLB import/export
path and network access for an explicitly configured local Hyperscope relay.
Live sync remains disabled until **Connect** is chosen, accepts a loopback URL
by default, and does not retain the bearer token in the `.blend` or add-on
preferences.

## Authoring workflow

1. Enable the extension and open **3D View > Sidebar > Hyperscape**.
2. Choose **Create Editable Conformal Demo** for a complete starting scene, or
   add frames and walls manually.
3. Use **Refresh Wall and Path Controls** to create wire spheres, planes, and
   point controls. Transform those objects with Blender's normal gizmos and
   choose **Apply Wall and Path Control Transforms** to write the changes back.
4. Choose **Evaluate Dual Coordinates** to sample paths at Preview Time and
   inspect each bound object's local and ambient coordinates.
5. Export through the Hyperscape panel or File > Export. The result remains an
   ordinary glTF/GLB for unaware viewers and gains conformal metadata for
   Hyperscape/Hyperscope.

Bound objects may carry a **Stable Entity ID**. Generate it once in the object
panel and retain it across exports; import restores the same UUID. This is the
durable identity used by Blender/Hyperscape edit sync and presentation
selection, while glTF node indices remain container-local handles.

The scene may likewise carry a **Stable Asset ID**. Generate it once in the
Hyperscape scene panel and retain it across exports. Collaborative entity
addresses are the pair `(asset ID, entity ID)` so composed assets cannot alias
one another; legacy files without an asset ID remain loadable but cannot make
scoped lease claims.

Generator lists are displayed in application order. Sphere reflection at its
center is a pole and preview evaluation reports it instead of fabricating a
finite position. Frame reparent and object re-anchor actions preserve the
represented ambient point/map. A path's Control-Point Frame remains fixed as
timed transitions select other active frames and anchors, preventing jumps at
Euclidean → conformal → re-anchored → Euclidean boundaries.

## Automated checks

The exact pure-Python codec and conformal evaluator can be tested without
Blender:

```sh
python -m unittest discover -s tools/blender_hyperscape/tests -v
```

The relay's authenticated HTTP surface has its own reproducible smoke:

```sh
node scripts/smoke-local-peer-relay.mjs
```

With Blender installed, the end-to-end carrier smoke starts an isolated relay
and Blender profile, publishes a real object edit with a nanosecond-scale
sequence, admits the exact JSON through generated Rust/WASM, and checks the
resolved packed-scene matrix:

```sh
node scripts/smoke-blender-browser-relay.mjs
```

When Blender is installed, the headless integration script creates the demo,
exports it, imports it into a fresh file, and checks the authored collections:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender_hyperscape/tests/blender_roundtrip.py -- \
  /tmp/hyperscape-roundtrip.glb
```

The live-sync integration check proves local edit publication, authored and
presence echo suppression, remote transform application, advisory lease
refresh/contention/release gating, ephemeral presence expiry and overlay
cleanup, timeline isolation, explicit shear rejection, and the absence of
overlay-created datablocks against a fake transport:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender_hyperscape/tests/blender_live_sync.py
```

Regenerate the checked demo's `.blend`, `.gltf`, `.bin`, and `.glb` from the
same authored scene with:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender_hyperscape/export_demo.py -- \
  examples/hyperscape-blender-demo.glb
```
