# Hyperscape conformal scene model

Status: implementation contract for the first vertical slice.  Version `0.1`
uses glTF `extras`; a registered vendor extension may replace the container
without changing the data model.

The runtime ECS is the `hyperscape` crate. It pins Bevy `0.18.1` with default
features disabled and uses only `bevy_app`, `bevy_ecs`, and `bevy_time`.
Hyperscope/Quilting remains the renderer; a conformal frame is deliberately not
represented as Bevy's affine transform component.

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

Unknown clients ignore `extras` and render the ordinary node/mesh fallback.
Once the format has independent implementations and a reserved prefix, the
same payload can move under a vendor extension object and be listed in
`extensionsUsed`.  It is only placed in `extensionsRequired` when omission
prevents a meaningful fallback rendering.

## Vertical-slice event order

1. Load ordinary glTF and Hyperscape extras.
2. Spawn entity nodes, conformal frames, walls, anchors, paths, and constraints.
3. Advance ordinary and conformal animation.
4. Evaluate frame-world chains and reject cycles or invalid generators.
5. Solve tracking/projection constraints in their declared target frames.
6. Apply requested preserve-world reparent/re-anchor operations.
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
