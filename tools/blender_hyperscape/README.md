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
ordering with receipt-relative TTL and bounded authored-message echo
suppression, but deliberately supplies no transport semantics. The optional
local relay is a delivery-only adapter; HHHS can still wrap only authored
envelopes without receiving viewport presence.

For local bridge development, start the disabled-by-default relay and copy its
printed bearer token into the browser and Blender adapters:

```sh
cargo run -p hyperscope-web --features local-peer-relay \
  --bin hyperscope-local-peer-relay
```

It binds to `127.0.0.1:42117`, accepts only configured browser origins, retains
a bounded in-memory suffix, and reports restart/history gaps rather than
claiming persistence or repair. The Blender polling/main-thread adapter is a
separate checkpoint; merely running the relay does not change the add-on or
Hyperscope behavior.

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

The extension requests file access only because its import/export operators
read and write the selected glTF/GLB file.

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

When Blender is installed, the headless integration script creates the demo,
exports it, imports it into a fresh file, and checks the authored collections:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender_hyperscape/tests/blender_roundtrip.py -- \
  /tmp/hyperscape-roundtrip.glb
```

Regenerate the checked demo's `.blend`, `.gltf`, `.bin`, and `.glb` from the
same authored scene with:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python tools/blender_hyperscape/export_demo.py -- \
  examples/hyperscape-blender-demo.glb
```
