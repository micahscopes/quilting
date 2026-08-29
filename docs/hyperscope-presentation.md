# Hyperscope presentation data

`hyperscape` owns presentation sequencing and view transitions; a browser is
an adapter for asset I/O, renderer commands, text, and input. The checked-in
[`hacker-night.presentation.json`](../crates/hyperscape/fixtures/hacker-night.presentation.json)
is the checked five-asset fixture, and
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

After `trunk serve`, open `/?presentation=1&glb=horse.glb` to run the checked-in
fixture. For the exact offline build, rehearsal, and recovery procedure, see
the [hacker-night runbook](hacker-night-runbook.md). Presentation sequencing,
cue validation, camera/focus transitions,
and resolved layer state remain Rust-authoritative. The browser adapter fetches
assets and translates the resolved snapshot into renderer commands.

Presentation orchestration has an explicit rollback boundary.
`presentimpl=rust` is the default: it loads the manifest, dispatches cue
actions, and advances transition time only through the application reducer.
The AppStore read model supplies the asset catalog needed by the browser I/O
adapter, so no second presentation controller or semantic JSON parse is
retained. `presentimpl=shadow` compares that complete cue and navigation result
with the browser-orchestrated standalone Rust controller, while
`presentimpl=js` is the explicit serialized rollback.

Every activated cue is serialized as its stable UUID in the `cue` URL
parameter. Copying or reloading the URL therefore re-enters that exact cue
through Rust's validated `jump_to_cue` path; an absent cue starts at cue one,
while a malformed or unknown cue fails visibly instead of silently presenting
the wrong slide.

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
The checked Blender fixture also assigns five deterministic stable entity IDs,
including four pickable mesh nodes. The IDs persist through `.blend`, `.gltf`,
and `.glb` regeneration and form the authored side of the presentation
selection join; source node indices remain container-local handles.
The browser exposes asset fetch, resident ranges, hidden-guide counts, packed
face count, active layers, pending capabilities, and failures at
`globalThis.__hyperscopePresentation`.

Ordinary packed-node matrix extraction has a separate rollback gate:
`sceneimpl=js` retains the incumbent browser composition, `shadow` compares
Rust's backend-neutral extraction without applying it, and `rust` applies the
Rust result to both presentation rendering and LOD. The browser binding now
contains only stable layer/asset identity and renderer-local node/source
metadata; it cannot supply layer TRS, visibility, or opacity. AppStore samples
the active cue, authored Blender projection, and application revision under one
lock, validates every active layer binding exactly once, then returns sorted
matrices plus effective visibility/opacity. A durable entity transform remains
an absolute source-asset world TRS with the presentation layer outermost. The
diagnostics expose cue/revision fences, matrix and opacity comparisons,
authored overrides, unmatched entities, fallbacks, and bounded mismatches at
`globalThis.__hyperscopeSceneExtraction`.

The checked-in story follows the renderer's actual data flow rather than
embedding the historical demo renderer. Four opening cues introduce projected
4D cell complexes before the animated horse: 4-simplex patch boundaries,
tesseract atlas wire topology, the same tesseract under screen-space LOD, and
analytic normals on a projected 16-cell. Three horse cues then show animated
QB patches, shared-edge LOD, and conformal stretch. An eighth cue returns to
PBR and composes the Blender-authored scene. Each surface visualization is
resolved in Rust to Quilting's backend-neutral `RenderStyle` and carried in the
active application snapshot; ambiguous combinations are rejected. The browser
only translates the incumbent `matcap_wire` control spelling to `both` and
reports the separately orthogonal, not-yet-rendered `control_net` capability.
The same snapshot carries Rust-validated density, attenuation enablement, and
pixel-floor values. The browser applies those values exactly; malformed bridge
data fails visibly instead of being silently defaulted or clamped a second
time. Outside cue application, the same complete render policy can be observed
through `renderstateimpl=shadow` or explicitly consumed with
`renderstateimpl=rust`; ordinary links retain JavaScript control authority
until live parity evidence justifies changing that default.
The canonical Rust route registry separately validates the browser spellings
and ranges for those render controls. In particular, URL `mode=both` maps to
the backend-neutral `matcap_wire` style, density and atlas values must be
integers in their supported ranges, and the pixel floor may remain fractional.
On Rust-authoritative startup the route result contains this complete typed
render policy, including canonical defaults. JavaScript only converts the
backend-neutral combined-style name to its legacy control spelling; it does
not default, round, or clamp an admitted value.

The polytope fixtures are deterministic static GLBs generated from
`quilting_core::polytope4`. Their closed 3-cell shells are separated before 4D
projection so the resulting triangle surfaces remain honest manifold inputs to
the ordinary Quilting pipeline. The tesseract comparison deliberately keeps
one camera: its wire cue disables attenuation at a fixed density of 12, then
its LOD cue enables the ordinary path with an 8-pixel subdivision floor. The
horse LOD cue uses the 16-pixel runtime default for a more legible multi-level
split. No presentation-only tessellator or 4D renderer is involved.

`control_net` remains reserved in the interchange format but is not claimed by
the browser adapter yet. A faithful version needs explicit source/control
geometry in the main renderer rather than the historical demo's second bespoke
draw path. Unsupported overlay requests are visible in
`__hyperscopePresentation.unsupportedOverlays`.

The eight checked-in cues have a deterministic Rust replay oracle and a Node
adapter smoke. An isolated staged-release Chrome/WebGL2 rehearsal on 2026-08-27
advanced through every cue and ended with all five assets ready, 4,432 packed
and LOD-resident faces, 12 topology domains, one GPU scene-classification pass,
no application or extraction mismatch, and no console warning or error. A
canonical reload restored the final cue directly. The complete artifact and
startup evidence is recorded in the
[staged-release benchmark](benchmarks/2026-08-27-hacker-night-release.md).

The Tuesday adapter deliberately supports one animated primary asset plus
static, untextured secondary assets. Animated or textured secondary assets fail
preflight instead of silently rendering incorrectly. Fractional per-layer
opacity is reported as pending; zero opacity and ordinary visibility are fully
applied. A secondary asset's authored Hyperscape ECS graph is not yet ticked as
a second runtime—the geometry is resident and transformed by presentation
state, while multi-runtime ECS synchronization remains follow-up work.
