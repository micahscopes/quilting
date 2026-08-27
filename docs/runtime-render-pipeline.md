# Runtime render pipeline and backend boundary

This is the current execution contract for Hyperscope. It distinguishes work
that happens every browser frame from asynchronous topology work and from
load-time preprocessing. `SPEC.md` describes the mathematical model; this file
is the operational map to use when profiling or adding a WebGPU backend.

## Steady-state browser frame

The request-animation-frame callback performs only the following work for an
ordinary glTF scene:

1. Apply a pending canvas resize, if any.
2. Advance the monotonic semantic animation clock. At most one worker pose
   request is in flight and only the newest pending request is retained. Every
   request carries its issued sample time, revision, and continuity epoch;
   pause, scrub, clip switch, model reset, and clip wrap start a new epoch.
   Old-epoch responses are discarded before WASM, and WASM atomically rejects
   stale/invalid stamps before either its CPU pose or retained GPU resources
   change. Source vertices are not sent back per frame.
3. Sample retained mouse/SpaceMouse state once and update the camera and shared
   focus/inversion sphere. The browser currently adapts device axes directly;
   the accepted ownership target is deterministic `hyperscape::FocusNavigation`
   plus a Rust camera rig. An inversion-sphere edit transports the camera eye
   exactly with the conformal point map. An unanchored camera transports its
   sight tangent and frame with the local differential; an explicit semantic
   target may use exact two-point transport. FOV and lens parameters stay
   unchanged. Camera matrices reuse preallocated typed arrays.
4. Schedule adaptive LOD. The scheduler permits one worker job plus one
   coalesced follow-up, so animation or continuous input cannot create an
   unbounded queue.
5. Enter `mr_render` with the view matrices and camera position.

An authored Hyperscape scene additionally ticks its ECS and extracts the
active camera/subject conformal packets. Scene-inspector diagnostics are
presentation data, not render input, and should remain throttled rather than
being serialized at display refresh rate.

`mr_render` retains semantic batch membership and draw commands across frames.
For each resident batch it runs one transform-feedback preparation dispatch:

- fetch immutable source-face positions, rational QB weights, UVs, normals,
  skinning indices, and morph data from renderer-owned textures;
- evaluate the current pose and ordinary node transform;
- compute a conservative current-pose frustum result; and
- write the canonical 52-float prepared-patch record into a persistent buffer.

All later PBR, matcap, wire, normal, stretch, and pick draws consume that
prepared record. An invisible prepared patch returns an out-of-clip vertex
from the main vertex shader. This avoids fragment work and rasterization for
off-camera patches without a CPU visibility readback, although WebGL still
invokes the vertices belonging to their resident atlas topology.

Selection focus adds no CPU geometry pass. The vertex shader carries the posed
ordinary-space surface point before Möbius transformation, and PBR maps its
radius around one frame-global focus sphere to the exact compactified
coordinate `u = 2/pi atan(distance/radius)`. The origin, sphere, and infinity
are 0, 1/2, and 1; inversion complements the coordinate. This dense B-channel
field drives a spheroidal depth-of-field shell and bypasses JFA seed propagation,
then reuses the retained variable-blur passes. Sphere animation uploads only a
few scalar uniforms.

The CPU still issues one preparation dispatch and one or more draw calls per
resident `(material, node, canonical LOD, parity)` bucket. S3 permutation is a
per-instance value: the six permutations do not own six meshes or six draw
buckets. Odd and even permutations remain separate only because raster winding
is fixed per WebGL draw.

## Asynchronous LOD path

LOD selection runs on a dedicated worker with its own WebGL2 context:

1. The worker evaluates the requested animation pose on the CPU and uploads
   only joint matrices and morph weights. The normalized pose is retained by
   exact animation time, so the adjacent visible-pose request and LOD dispatch
   share one joint/morph evaluation and one normalization pass.
2. GPU pass 1 classifies one source face per output pixel. It computes posed
   geometry, conservative image extent, conformal interior demand, pole safety,
   and screen capacity.
3. GPU pass 2 reconciles shared edges, canonicalizes the LOD triple, and emits
   one packed classification per face.
4. Pass 2 losslessly packs its three four-bit exponents, three-bit
   permutation, one-bit visibility, and eight-bit atlas index into one `u32`
   per face; parity is derived from the permutation. The output is copied into
   staging, fenced, and polled without blocking the worker. Only after the
   fence signals is this four-byte topology word read back and validated.
   Renderer-context authority admits its typed topology and visibility fields
   directly into retained Rust state. Worker rollback and exact shadow parity
   alone expand it to the historical six-field CPU record. Staging buffers,
   packed readback vectors, and rollback-only decode vectors live in separate
   retained pools; the steady-state authority path does not create, resize, or
   write the mesh-sized float decode.
   Renderer-context authority compares the packed words with its retained
   snapshot before semantic admission. A model/scope shape change is a full
   publication; otherwise only changed packed records enter reconciliation.
   An exact no-op skips source admission entirely, while an enabled adaptive
   screen frontier still refreshes from the current view.
5. The WASM worker expands and compares that snapshot with its retained
   previous result before creating any JavaScript typed array. The first
   coherent result after a model, animation, remesh, or compute-resource
   boundary transfers all faces; later results cross into JavaScript and
   transfer only changed `(face_index, six-float classification)` records. A classification with no
   changed records does not enter main-renderer batch bookkeeping.
6. The main-thread WASM layer retains the last valid topology for invisible
   faces, re-reconciles shared edges (including exact duplicate glTF seam
   vertices), grades each triangle to a 2:1 maximum edge ratio, and compares
   memberships with retained buckets. Reconciliation starts from the changed
   face frontier and propagates through half-edge twins instead of rescanning
   every mesh edge.
7. Only changed buckets repack and upload their eight-float topology records;
   unchanged GPU buffers and VAOs survive untouched.

The unavoidable current synchronization boundary is step 4: WebGL2 cannot
generate indirect draws or compact instance lists for later draws. The
worker-side GPU readback is 4 bytes per source face per completed
classification, not a readback of animated vertices or tessellated geometry.
After the initial full snapshot, the worker-to-main transfer is 28 bytes per
changed face (a four-byte face ID plus the six-float record). Current-pose
visibility does not cross that boundary at all.

An off-camera face intentionally retains its last crack-free topology. Making
it coarser is safe only when a conservative conformal bound proves that the
entire patch, including pole-driven ballooning between its corners, remains
outside a guarded frustum. Visibility of the three source vertices alone is
not such a proof.

## Visibility and submission contract

Visibility is a current-pose fact, while resident topology is state carried
between asynchronous classifications. Backends must keep those concepts
separate:

| Contract | WebGL2 implementation | Future WebGPU implementation |
|---|---|---|
| Pose source | Retained joint/morph textures in each context | Shared logical pose buffers, uploaded once per device |
| Patch preparation | Transform feedback writes the 52-float prepared record | Compute writes the same logical record to storage |
| Conservative visibility | Prepared flag makes later vertices degenerate | Visibility bit participates in prefix-sum/atomic compaction |
| Resident topology | CPU retains the last valid crack-free LOD and sparse bucket membership | Storage buffers retain per-face edge LOD and reconcile it in compute |
| Submission | Instanced draw per non-empty material/node/LOD/parity bucket | Compacted visible instances plus indirect arguments per `RenderBatchKey` |
| Selection focus | Source-space sphere in the PBR UBO and MRT B-channel mask | Same logical view packet and WGSL field classification |

The backend-neutral `RenderCommand` stream names patch preparation and
current-view visibility resolution separately. A backend is free to memoize
unchanged preparation. WebGL2 resolves visibility into a retained scalar flag
that makes rejected patches degenerate, whereas WebGPU may resolve the same
command by compacting visible instances and emitting indirect draw counts.
The command contract deliberately says nothing about transform feedback,
storage buffers, or GPU handles.

`RenderStyle` also resolves to one canonical ordered `RenderDrawPassPlan`
slice. Each pass names its geometry and semantic batch selection (`all`, PBR
opaque, or PBR non-opaque). `RenderFrame` extraction and the WebGL2 dispatcher
iterate this same plan; shader selection, material binding, and API resources
remain backend-local. A future WebGPU dispatcher can therefore consume the
same ordering and batch selection while lowering submission to compacted
indirect draws.

The invalidation predicate is shared as well: preparation changes with source
pose, resident topology, or the entity's ordinary affine model; visibility
changes with preparation, the view-projection matrix, or the resident batch
revision. A conformal uniform change does not rebuild the prepared 52-float
record, but it still participates in the later patch evaluation and the batch
revision that invalidates visibility.

The opt-in render shadow compares more than aggregate workload. Expected
commands and actual backend submissions stream the exact pass, batch index,
geometry, index count, and instance count through the same deterministic
rolling fingerprint. This catches reordered or misclassified draws without a
per-frame trace allocation or GPU readback. Comparison remains in Rust; the
bounded diagnostic snapshot exposes the 64-bit value as hexadecimal text so
the browser never rounds it through JavaScript's number representation.

A negative visibility result must mean “the complete posed rational patch is
outside the current guarded frustum.” Invalid bounds, a denominator region
touching zero, or a source bound containing the Möbius pole all survive. The
WebGL vertex rejection therefore saves raster and fragment work but is not
true draw compaction: atlas vertices are still invoked. Demoting asynchronous
off-camera faces to LOD 1 would reduce that work but can expose a coarse flash
when a newer camera/pose makes the patch visible before the worker catches up.
The correct WebGPU solution is same-pose compaction, not weakening this
residency rule. A future predictive WebGL policy would need a proven swept
frustum/pose bound and must retain the hidden last-valid topology.

For a genuine rational patch `q = N D^-1`, visibility centers its analytic
ball at the barycentric-center quotient `c = (ΣN_i)(ΣD_i)^-1` and bounds the
residual with `max|N_i-cD_i| / min|D|`. This is translation-local while
remaining conservative over the complete barycentric domain. The later radial
`POSITION_CLAMP` can move a far point outside that local ball, so the renderer
uses the recentered ball only when it is wholly inside the origin-centered
clamp ball; otherwise it fails closed to the clamp ball instead of taking an
invalid minimum between differently centered spheres.

## Work outside the frame loop

- glTF parsing, image decode, static face textures, adjacency, and skinning
  data are produced once per model load.
- The canonical tessellation atlas is restricted to triples reachable under
  the active validated 2:1 or 4:1 grading policy,
  generated hierarchically in one worker transaction, packed once, cached in
  IndexedDB with a versioned key, and uploaded once. The 2:1 policy has three
  irreducible topology families and 4:1 has six; only those roots are
  independently blue-noise sampled. All higher power-of-two descendants use
  exact midpoint subdivision.
  Serialization and cache persistence run after the first model/render startup
  boundary rather than delaying it.
- HDR prefilter and irradiance products are cached in IndexedDB and uploaded
  once per selected environment.
- Batch ownership changes only after a completed LOD classification, material
  or node reassignment, atlas replacement, or conformal packet change.
- Stretch-range sampling, picking, remeshing, and fuzzy-vision postprocessing
  are opt-in paths rather than baseline per-frame CPU work.
- Drawable dimensions are renderer-owned resize state. Optional framebuffer
  paths consume that state rather than synchronously querying the WebGL
  viewport during rendering.

## Backend-neutral contract

The backend-independent pieces already live below the WebGL renderer:

- `quilting_core::batch::{FaceLodClassification, ResidentLod,
  RenderBatchKey, RenderBatchMember}` and its atomic backend-publication
  admission boundary;
- `quilting_core::instance_layout`, including the eight-float topology record
  and 52-float prepared-patch record;
- `quilting_core::render_pipeline`, whose immutable shader-module, bind-group,
  vertex-layout, render-pipeline, and compute-pipeline descriptors define the
  initial WebGL2/WebGPU common subset needed for backend memoization;
- canonical atlas keys and S3 permutation semantics; and
- WGSL surface, preparation, and material shader logic.

The pipeline descriptors are values, not GPU objects. They retain exact shader
source through `Arc<str>`, cache a non-authoritative source fingerprint so
hash-map lookup does not rescan the WGSL, canonicalize definitions and
binding/layout order, and reject duplicate identities. Compiler identity
includes the composable naga-oil module catalog as well as compiler versions;
diagnostic labels do not affect equality. The few floating-point pipeline
constants use a finite wrapper that normalizes negative zero and rejects
NaN/infinity. A descriptor therefore works as a functional rendering value and
a complete key for the supported subset; it never contains a frame uniform,
camera value, resource handle, command encoder, or mutable backend cursor.
Binding arrays, sparse color targets, stage override constants, and
backend-specific primitive extensions remain explicit follow-up work rather
than being silently omitted from a claimed complete WebGPU key.

`quilting_renderer::memo::DeviceMemo` is the effect boundary. It maps a pure
descriptor to a concrete backend resource, inserts only after construction has
fully succeeded, and scopes every entry to an explicit device/context epoch.
Changing epoch returns the old resources for destruction by the backend that
created them. This cache is derived runtime state: FRP may publish a changed
render plan or descriptor, but HHHS must never replicate GPU handles, cache
contents, compilation status, or per-frame bindings.

The binding schema will be normalized before WebGPU execution:

| Group | Stable responsibility |
|---|---|
| 0 | frame and pose: view, joints, skinning, morph, source-face data |
| 1 | entity/batch: Möbius packet, Euclidean model/normal, QB enablement |
| 2 | material/style uniforms and material textures |
| 3 | pass resources: scene color, transmission, blur/focus, highlight |

The current WGSL happens to reuse group 0/binding 1 for the vertex joint block
and a fragment material block. WebGL tolerates this because stage-specific GLSL
block names are bound after linking; a WebGPU pipeline layout cannot. The
descriptor/cache migration must therefore land with reflected `(group,
binding, stage, kind)` metadata and deterministic WebGL lowering, replacing
generated-name substring heuristics before WebGPU is treated as equivalent.

## Atlas topology and grading policy

Crack-free shared-edge equality and within-face grading are deliberately
separate contracts. Shared edges must use identical boundary subdivision.
The default 2:1 maximum within one source face is a conservative quality and
residency policy: it avoids extreme fans, but it also promotes low edges and
propagates a graded LOD halo through neighboring faces. The measured 4:1
experiment is available through the validated `lodratio=4` route and the
reload-to-apply control. It increases the number and size of cached atlas
entries while potentially reducing the much larger per-scene resident draw
workload. Cache identity includes both ratio and maximum exponent, and the
renderer rejects unsupported or post-residency policy changes.

`TessellationAtlas::build_for_keys` makes the topology choice explicit over
the same reachable key set:

- `Direct` independently runs deterministic blue-noise sampling and constrained
  Delaunay triangulation for every key.
- `Hierarchical` samples only irreducible seed families, then derives all
  power-of-two descendants through exact four-way midpoint subdivision.

Both policies retain exact requested boundary counts. The reproducible probe
is:

```sh
cargo run -p quilting-core --release --example bench_runtime_atlas -- \
  --min-exp 6 --max-exp 6 --ratios 2,4 \
  --modes hierarchical,direct --rounds 3
```

It reports reachable keys, independently sampled topology families, aggregate
atlas vertices/triangles/bytes, one `2 / 64 / 64` resident example, boundary
mismatches, a single-peak 24×24-grid promotion halo, minimum and
first-percentile normalized triangle quality, and median construction time.
The probe and live runtime admit only the measured 2:1 and 4:1 policies. The
default remains hierarchical 2:1; 4:1 is an explicit rollback-safe experiment
until representative animated-scene and browser startup/cache measurements
justify a default cutover.
The latest checked results are archived in the
[hacker-night benchmark](benchmarks/2026-08-22-hacker-night-baseline.md).

`MeshDraw`, VAOs, transform-feedback objects, GL textures, UBOs, framebuffers,
and `glow::Context` are explicitly WebGL backend state. They must not migrate
into a shared scene or command model.

## WebGPU migration sequence

1. **Completed shadow slice:** port the two handwritten GLSL LOD passes to
   WGSL, validate them with the pinned Naga compiler, freeze their host
   payloads and packed output against CPU conformance oracles, and execute both
   passes through a retained backend-local `wgpu` device on a native adapter.
2. Implement patch preparation as a compute pass writing the same logical
   52-float record. Keep the WebGL transform-feedback implementation as the
   compatibility backend.
3. Move edge reconciliation, configurable grading, resident-topology
   selection, and visible-instance compaction into storage buffers.
4. Emit indirect draw arguments per `RenderBatchKey`. This removes the
   steady-state topology readback and avoids vertex invocation for invisible
   instances, while retaining the atlas rather than drawing one universal
   maximum-resolution mesh.
5. Keep CPU readback optional and diagnostic-only. Counters can be copied from
   a small, delayed telemetry buffer without gating rendering.
6. Move PBR material binding and postprocess resources behind backend-specific
   implementations after the geometry path is equivalent.

The CPU oracle for steps 3–4 is now frozen in
`quilting_core::render::VisibilityCompactionPlan`. It consumes one binary
current-pose visibility entry per flattened canonical batch member, removes
disabled instances and suppressed retained roots, preserves visible adaptive
replacements, and emits stable survivor source IDs plus one range per
`RenderBatchKey`. `IndexedIndirectArguments` is the exact 20-byte, five-word
indexed-indirect ABI. Its `first_instance` addresses the compacted survivor-ID
stream; `first_index` and `base_vertex` remain zero while each batch binds its
own canonical atlas entry and may later be patched by a packed-atlas backend.

Core tests freeze stable order, zero-instance buckets, retained-root/overlay
replacement, malformed visibility rejection, stale scene revisions, batch
shape validation, and ABI size/alignment. The classifier now has a WebGPU
executor, but compaction does not: no storage-buffer compaction or indirect
submission pipeline consumes this oracle yet, and WebGL2 continues to use
current-pose degenerate-vertex rejection.

### Frozen WebGPU LOD classifier boundary

The classifier source now exists without changing runtime authority:

- `compute/lod_types.wgsl` defines the storage/uniform ABI and the exact
  packed-word helper;
- `compute/lod_pass1.wgsl` runs one face per invocation in 64-thread groups,
  including morph/skin evaluation, dense authored-subject selection,
  conservative conformal image culling, intrinsic density demand, pixel-floor
  capacity, and adaptive priority; and
- `compute/lod_pass2.wgsl` reconciles only visible neighbors, canonicalizes the
  S3 edge order, looks up the resident atlas, and emits the existing four-byte
  classification word.

The host contract is fixed as little-endian word arrays rather than relying on
Rust padding. Face, position, morph, adjacency, and pass-one records are 16
bytes; skinning records are 32 bytes; authored subject rows are 160 bytes; the
dispatch uniform is 272 bytes; and final classifications remain four bytes.
Scene-node IDs are compacted into deterministic dense subject rows, so sparse
authored IDs do not inflate GPU buffers. Pass one binds one uniform and exactly
eight storage buffers, staying within WebGPU's minimum per-stage storage-buffer
limit. Pass two binds one uniform and four storage buffers.

Naga validates both compute entries and the exact record layouts. Renderer
tests freeze every host offset, all six S3 permutations, invisible standby
records, visible-neighbor-only seam promotion, atlas lookup, and the final
packed word. The Naga WGSL writer renames the flattened entries with a trailing
underscore; shader tests freeze those device-visible names so a compiler update
cannot silently invalidate pipeline creation.

`quilting-webgpu` is the first executing device shadow. It is deliberately
outside the WebGL2 authority path and owns only backend resources: a device and
queue supplied by the caller, two retained compute pipelines, immutable model
and atlas buffers, retained dynamic pose/subject buffers, and diagnostic
staging. Uploading a model uses the existing word packers. A dispatch writes
only the current uniform, dense subject rows, referenced joint-matrix prefix,
and morph weights; it then encodes both passes and one diagnostic copy into a
single command buffer. Extra unused glTF skin joints do not inflate retained
storage or invalidate the full pose.

The crate pins `wgpu` 29.0.1. That is the newest release line tested to compile
against this workspace's exact wasm-bindgen 0.2.108 / js-sys 0.3.85 browser
ABI. `wgpu` 30's browser backend uses newer typed JavaScript bindings, so a 30
upgrade belongs to a coordinated binding-stack migration and browser
regression gate rather than this isolated classifier cut.

The narrow device gate now has three proven parts:

- native Radeon 780M / RADV / Vulkan executed both passes and returned the
  exact packed word produced by the CPU pass-two oracle; and
- `cargo check -p quilting-webgpu --target wasm32-unknown-unknown` passes on
  the repository's pinned browser ABI; and
- Chromium 150 requested a real `BrowserWebGpu` adapter and executed the same
  reusable minimum matrix, returning one exact full-pipeline word and ten exact
  coherence words with no console warnings or errors.

Ordinary native tests report an explicit skip when no adapter is visible.
Setting `QUILTING_REQUIRE_WEBGPU=1` makes that condition a failure for a
hardware conformance lane. This is real native and browser device evidence for
the shared minimum matrix, but not yet broad cross-backend numeric parity. The
shared matrix covers a complete two-pass path; all six S3 permutations;
visible-only neighbor promotion; invisible standby records; and atlas/priority
packing. The larger native-only pass-one matrix additionally covers unused
skin joints; current-pose culling; dense authored-subject selection; morph and
joint animation moving a face across the frustum; and maximum-LOD saturation
when a sphere-reflection pole lies inside a triangle. Pass two is bit-exact with
its CPU oracle. Those pass-one cases currently freeze semantic invariants;
finite non-interior poles, pole grazing, multi-face composed scenes,
maximum-atlas boundaries, the expanded Chrome matrix, and exact WebGL2
comparison remain promotion gates.

The WebGL2 GLSL programs and runtime authority remain untouched. Readback in
`quilting-webgpu` is intentionally full and diagnostic; an authoritative
backend must consume packed classifications on-device and copy only bounded,
delayed telemetry to the CPU.

Only after that gate should same-device resident topology, visible-instance
compaction, and indirect arguments be wired together. That later cut is what
removes WebGL2's four-byte-per-face readback and rejected atlas vertex
invocations; merely compiling WGSL does neither.

The old JavaScript renderer's useful idea was memoizing tessellation topology
and prepared meshes. The retained atlas, versioned browser cache, and stable
batch buffers already implement that principle more completely. Per-patch regl
commands should not return, but independently generated blue-noise/Delaunay
topology remains a deterministic offline-quality or background-cache candidate
and is now measured against the hierarchical runtime over identical key sets.

Meshoptimizer opportunities and the boundary between ordinary triangle
clustering and conformal QB fitting are tracked in
[`meshoptimizer-roadmap.md`](meshoptimizer-roadmap.md).

Selection, inversion, device mappings, Rust ownership, Blender live sync, and
game-facing interaction migration are tracked in
[`focus-navigation-roadmap.md`](focus-navigation-roadmap.md).
