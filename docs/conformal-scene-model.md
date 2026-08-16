# Hyperscape conformal scene model

Status: implementation contract for the first vertical slice.  Version `0.1`
uses glTF `extras`; a registered vendor extension may replace the container
without changing the data model.

The runtime ECS is the `hyperscape` crate. It pins Bevy `0.18.1` with default
features disabled and uses only `bevy_app`, `bevy_ecs`, and `bevy_time`.
Hyperscope/Quilting remains the renderer; a conformal frame is deliberately not
represented as Bevy's affine transform component.

Hyperscape advances `Time<Virtual>` explicitly. It does not derive authored
time from Bevy's real-time clock: native tests, browser frames, replay, and
offline tools all feed a duration before an ECS update. This also prevents a
headless or `wasm32-unknown-unknown` runtime from depending on a native
`std::time::Instant` implementation.

## Four structures, four jobs

Hyperscape does not overload one parent pointer with four different meanings.

### Entity scene graph

The ordinary glTF node hierarchy owns meshes, cameras, skins, TRS transforms,
and ordinary animation.  It remains valid glTF and remains useful in clients
that know nothing about Hyperscape.

### Conformal frame forest

A conformal frame is a coordinate chart with at most one parent.  Its
`local_to_parent` value is an ordered word of conformal generators.  Following
the unique parent path produces a map from local coordinates to the ambient
Euclidean chart.

The first version is a forest, not a general DAG.  Multiple paths between two
frames can compose to different Möbius maps.  Such a graph needs an explicit
flatness, preferred-path, or holonomy policy; accepting it accidentally would
make re-anchoring path-dependent.

### Round-wall and chamber layer

A wall is an unoriented sphere or plane expressed in one conformal frame.  The
strict negative and positive loci of its defining function are complementary
open sides.  For a sphere, negative is the bounded interior in that Euclidean
chart and positive is the exterior containing chart infinity.

Contact, crossing, tangency, and the unsigned separated relation belong to the
wall skeleton.  An anchor stores sparse side flips separately.  With positive
radii, signed inversive distance distinguishes external separation from
nesting; its absolute value preserves only their union.

### Constraint graph

Path following, look-at, target tracking, projection alignment, and other
control relationships are directed constraints between entities and frames.
They do not create entity or frame parents.  A deterministic schedule evaluates
animation, frame worlds, constraint targets, anchors/chambers, and render
extraction in that order.

## Transform conventions

- A generator array is in application order: element zero acts first.
- A frame chain maps local coordinates to parent coordinates.
- A world chain is the child chain followed by each ancestor chain.
- A point from frame `A` expressed in frame `B` follows `A_to_world`, then
  `world_to_B`.
- An ordinary glTF node transform acts before its referenced conformal frame.
  Mesh local → glTF node world-within-frame → conformal frame world.
- Runtime quaternions are `(w, x, y, z)`; glTF rotations are converted from
  `(x, y, z, w)` at the loader boundary.

`ConformalTransformChain` retains generator words for authoring, inversion,
and animation.  `Mobius {a,b,c,d}` is the collapsed rendering representation.

## Preserve-world operations

If a frame has old world map `W` and receives new parent world map `P`, its new
local map is

```text
P inverse, after W
```

or, in point order, `W` followed by `P⁻¹`.  Thus
`P ∘ new_local = W`: in point order the new local map acts first, followed by
`P`.  Descendants need no edits because their paths still pass
through the same frame-world map.

Re-anchoring a point from chart `A` to chart `B` similarly preserves the
ambient point by applying `A_to_world` followed by `world_to_B`.  Side flips
and chamber coordinates are updated as a separate semantic step.

## Hyperscope extraction

For each visible entity and view:

1. Evaluate the entity's ordinary glTF transform and animation.
2. Resolve the unique entity-frame and view-frame world chains.
3. Build the relative conformal chain from entity frame to view frame.
4. Collapse it to quaternion coefficients `a,b,c,d`.
5. Send those coefficients, orientation parity, and the ordinary view/projection
   matrices through the existing Hyperscope uniform and adaptive-tessellation
   path.

Different entities may therefore use different relative Möbius transforms in
the same view.  Render extraction, rather than the renderer, owns that choice.

The browser bridge (`mr_loadHyperscape` / `mr_tickHyperscape`) retains both the
ordinary glTF subject-node and projection-camera-node identity of every
extracted packet. The mesh loader carries the subject node per triangle, batch
construction includes it in the batch key, and every render, pick, and
highlight draw selects that subject/view packet's Möbius coefficients and
explicit orientation parity. The lowest projection-camera node is selected
deterministically by default; `mr_setHyperscapeCameraNode` switches views.

The diagnostic snapshot exposes the complete `packets` array as well as the
legacy first-packet fields. For Hyperscape assets, mesh control points remain
in authored coordinates: the packet's ordinary affine model and inverse-
transpose normal matrix act first, followed by the subject/view Möbius map.
Path and cross-frame tracking systems can update the affine translation each
tick. Legacy glTF assets keep the historical baked-and-normalized path.

Adaptive LOD uses the same subject state. The worker replays its GPU
classifier for each extracted `[node, affine model, Möbius map]` record, then
copies only that node's face records into the final coherent classification;
unbound legacy faces use the baseline state. This is conservative and exact
for the current disjoint per-node mesh topology, though the cost scales with
the number of distinct visible subject states and remains a measurement and
visibility-culling target.

### Chamber-aware invalidation and diagnostics

`ChamberSignature` is still computed geometrically for every participating
entity against every wall. That full classification is authoritative. A
`ChamberAggregateState` then maintains counts keyed by the complete oriented
signature and changes only the old and new count entries when an entity crosses
a wall, changes anchor, appears, or disappears. Its measurements distinguish
the two costs:

- `classifications_last_tick` is the full point/wall classification count;
- `aggregate_updates_last_tick` is the sparse count-table mutation count;
- `changed_entities` and `changed_walls` identify the invalidated membership;
- `contact_frontier` expands changed walls by one edge in the wall-contact
  graph; and
- `epoch` advances only when a chamber membership changes.

The browser compares each extracted subject's affine/Möbius state and watches
the chamber epoch. Either change schedules an LOD refresh; identical states are
counted and skipped, and a pending refresh survives the 250 ms throttle. The
worker applies every completed coherent classification and coalesces changes
that arrived in flight into one follow-up, so continuous animation cannot
starve the renderer of LOD updates. The prototype's visibility records compare
subject and camera chamber signatures
and report separating walls plus their contact frontier. They always emit
`can_cull = false`: chamber separation is a scheduling and prioritization hint,
not a proof of geometric occlusion. Incidence Möbius inversion can transport
coarse aggregate payloads, but it does not replace depth, projected bounds, or
an occlusion query.

The open Conformal Scene Inspector in `hyperscope.html` reports frame parents
and parity, local versus ambient coordinates, anchors and flipped walls,
contacts, chamber counts and sparse invalidations, visibility/LOD hints,
bounded change-only transform histories, and the Möbius denominator norm at
each affine model origin. The origin value is a pole warning only; the GPU's
per-sample classifier remains authoritative for a whole mesh.

The checked-in Blender demo is also the browser smoke fixture. At path sample
times `2.25`, `4.25`, and `6.25`, node 2 is in conformal frames `1`, `2`, and
`0`, with flipped-wall sets `[0]`, `[0,1]`, and `[]`. Its local coordinates
change charts while its ambient coordinates follow the same continuous path.
At `4.25`, the world-frame projection camera targets the traveler's ambient
point plus the authored `[0,0,0.4]` offset, exercising cross-frame tracking
rather than merely displaying the exported metadata.

The selected authored projection camera also exposes its ordinary eye and
cross-frame tracking target to the browser. The view matrix and LOD projection
therefore follow the tracked entity without conflating that aim constraint
with camera translation.

## glTF interchange v0.1

The Khronos glTF extension registry reserves `KHR` for Khronos extensions and
`EXT` for multi-vendor extensions.  Hyperscape therefore does not publish an
unregistered `EXT_*` name.  During incubation, application data lives in the
root and node `extras` objects, which glTF explicitly provides for
application-specific data.

Root shape:

```json
{
  "extras": {
    "hyperscape": {
      "version": "0.1",
      "frames": [
        {
          "name": "reflection-room",
          "parent": null,
          "generators": [
            { "type": "translation", "offset": [0, 0, 2] },
            {
              "type": "sphere_reflection",
              "center": [0, 0, 0],
              "radius": 3
            }
          ]
        }
      ],
      "walls": [
        {
          "name": "room-wall",
          "frame": 0,
          "geometry": {
            "type": "sphere",
            "center": [0, 0, 0],
            "radius": 3
          }
        }
      ],
      "anchors": [],
      "paths": [],
      "constraints": []
    }
  }
}
```

An entity node refers to a conformal frame without changing its ordinary glTF
parenting:

```json
{
  "name": "TrackedHorse",
  "mesh": 0,
  "extras": {
    "hyperscape": { "frame": 0 }
  }
}
```

A path may keep all control points in one stable chart while changing the
entity's active frame and anchor at discrete times:

```json
{
  "name": "enter-reanchor-exit",
  "node": 0,
  "coordinate_frame": 0,
  "keyframes": [
    { "time_seconds": 0, "point": [-4, 1, 0] },
    { "time_seconds": 8, "point": [4, 1, 0] }
  ],
  "transitions": [
    { "time_seconds": 2, "frame": 1, "anchor": 0 },
    { "time_seconds": 4, "frame": 2, "anchor": 1 },
    { "time_seconds": 6, "frame": 0 }
  ]
}
```

The runtime samples the path in `coordinate_frame` and converts that point to
the active frame selected by the latest transition. Applying the active frame
map therefore recovers the same ambient point on both sides of a transition:
entry, re-anchoring, and exit do not jump. An omitted transition `anchor`
selects the canonical, unflipped sides in that frame. Transition times are
strictly increasing and lie within the path interval. This is a deterministic
timeline, not a second parent edge in the conformal frame forest.

Unknown clients ignore `extras` and render the ordinary node/mesh fallback.
Once the format has independent implementations and a reserved prefix, the
same payload can move under a vendor extension object and be listed in
`extensionsUsed`.  It is only placed in `extensionsRequired` when omission
prevents a meaningful fallback rendering.

The machine-readable schema is
[`schema/hyperscape-0.1.schema.json`](schema/hyperscape-0.1.schema.json), and
[`../examples/hyperscape-track.gltf`](../examples/hyperscape-track.gltf) is a
minimal editable interchange fixture. The checked-in
[`hyperscape-blender-demo.blend`](../examples/hyperscape-blender-demo.blend)
is the editable Blender source for the nested/overlapping full-flow scene;
[`hyperscape-blender-demo.glb`](../examples/hyperscape-blender-demo.glb) and
its separate `.gltf`/`.bin` form are real exports from that file and are loaded
by the Rust integration tests. `quilting-gltf` validates root and node
references and can inject the payload into either JSON glTF or GLB while
preserving ordinary fallback nodes, unrelated object-valued extras, and binary
chunks.

## Vertical-slice event order

1. Load ordinary glTF and Hyperscape extras.
2. Spawn entity nodes, conformal frames, walls, anchors, paths, and constraints.
3. Select the current path frame/anchor state, then sample control points in
   their stable coordinate frame and convert them to that state.
4. Evaluate frame-world chains and reject cycles or invalid generators.
5. Solve tracking/projection constraints in their declared target frames.
6. Apply requested preserve-world structural reparent operations.
7. Update side bits, chamber membership, and sparse payload aggregates.
8. Extract relative Möbius transforms and conservative visibility/LoD hints.
9. Render with Hyperscope and expose transform/contact/pole diagnostics.

The demonstrator must exercise Euclidean travel, conformal entry, cross-frame
tracking, continuous re-anchoring, conformal exit, and a Blender → GLB →
Hyperscape → Hyperscope round trip.

## Deferred semantics

- Multiple inconsistent frame paths and crossing monodromy.
- General sphere-arrangement realizability.
- Reconstructing conformal structure from contact/mereology alone.
- A complete classification of quaternionic fractional-linear matrices.

These are explicit extensions to this contract, not hidden assumptions in the
first implementation.
