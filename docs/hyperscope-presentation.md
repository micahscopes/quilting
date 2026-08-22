# Hyperscope presentation data

`hyperscape` owns presentation sequencing and view transitions; a browser is
an adapter for asset I/O, renderer commands, text, and input. The checked-in
[`hacker-night.presentation.json`](../examples/hacker-night.presentation.json)
is the first two-asset fixture, and
[`hyperscope-presentation-0.1.schema.json`](schema/hyperscope-presentation-0.1.schema.json)
describes its portable JSON shape.

Every presentation, asset, scene, layer, view, and cue has one globally unique,
non-nil UUID. Asset layers remain distinct instances: the runtime never merges
GLBs merely to compose a scene. Layer transforms are ordinary TRS values with
quaternions written `(w, x, y, z)`; they are not conformal frame edges.

A view is complete semantic state:

- camera eye, quaternion orientation, control distance, optional finite target,
  and perspective lens;
- the shared source-space focus/inversion sphere and focal field;
- visibility and opacity overrides addressed by stable layer UUID.

A cue selects a scene and view, carries optional display text, animation
directives, educational overlays, and one duration/easing pair. Activating it
returns a resolved `PresentationSnapshot`. The snapshot's `required_assets` is
the desired load set: it includes every `preload` asset plus assets used by the
active scene. Its layer list is already resolved against the active view.

Camera and focus transitions use Hyperscape's virtual clock. If a cue changes
an active inversion sphere, its authored camera endpoint is first expressed in
the current output chart and transported with the sphere. Thus the endpoint is
still the authored view when the transition finishes; the browser must not
run an independent interpolation.

A finite semantic target can be valid in its authored destination while lying
at infinity in an intermediate chart. The runtime crosses that chart using the
equivalent sight tangent and restores the finite target after both camera and
focus transitions settle. A target that is a pole in its own authored view is
still rejected during document validation.

JSON Schema checks structure. Rust validation additionally checks global UUID
uniqueness, references, finite and nondegenerate transforms, lens/focus ranges,
cue-local animation layers, and reflection-pole failures. A view targeting the
center of an enabled inversion sphere is invalid because that point maps to
infinity; omit `semantic_target` to preserve a free line-of-sight tangent.

## Browser adapter

Open `hyperscope.html?presentation=1&glb=horse.glb` to run the checked-in
fixture. Presentation sequencing, cue validation, camera/focus transitions,
and resolved layer state remain Rust-authoritative. The browser adapter fetches
assets and translates the resolved snapshot into renderer commands.

The WebGL backend keeps each asset and layer semantically distinct while
packing their face records into shared immutable GPU buffers. Stable node
ranges retain per-layer visibility and affine transforms, materials keep
asset-local indices through an explicit base offset, picking retains one
scene-wide face ID, and the animation worker continues to update only the
animated primary asset. Occasional full primary LOD snapshots are converted to
sparse primary-range updates so static layers never lose resident topology.
This is ordinary backend packing, not a merge of the source GLBs or their
presentation identities.

Authored nodes marked `extras.hyperscape_guide: true` are hidden in presentation
composition. They remain in the source Blender/GLB asset for diagnostics, but
large wall spheres and path controls therefore do not occlude the staged scene.
The browser exposes asset fetch, resident ranges, hidden-guide counts, packed
face count, active layers, pending capabilities, and failures at
`globalThis.__hyperscopePresentation`.

The Tuesday adapter deliberately supports one animated primary asset plus
static, untextured secondary assets. Animated or textured secondary assets fail
preflight instead of silently rendering incorrectly. Fractional per-layer
opacity is reported as pending; zero opacity and ordinary visibility are fully
applied. A secondary asset's authored Hyperscape ECS graph is not yet ticked as
a second runtime—the geometry is resident and transformed by presentation
state, while multi-runtime ECS synchronization remains follow-up work.
