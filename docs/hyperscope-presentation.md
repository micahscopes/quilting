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

The current fixture and Rust state machine are an interchange/runtime
foundation. Browser multi-asset residency and draw submission remain a
separate adapter milestone and should consume snapshots without changing this
ownership model.
