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
indexed-indirect ABI. Portable records keep `first_instance` zero because a
nonzero indirect value requires WebGPU's optional `indirect-first-instance`
feature. The corresponding range record retains `compacted_first_instance`;
the vertex stage combines that prefix with its local instance index. The
WebGPU residency therefore includes a device-aligned, static batch-index
uniform table. Each CPU-issued batch draw selects one table record with a
dynamic offset, while the vertex shader reads the GPU-written range and
survivor-ID buffers. This preserves stable source-instance identity without a
topology readback or the optional first-instance feature.
`first_index` and `base_vertex` remain zero while each batch binds its own
canonical atlas entry and may later be patched by a packed-atlas backend.

Core tests freeze stable order, zero-instance buckets, retained-root/overlay
replacement, malformed visibility rejection, stale scene revisions, batch
shape validation, and ABI size/alignment. The shadow WebGPU executor now runs
parallel count, deterministic batch scan, and stable chunked scatter passes;
native Vulkan and browser WebGPU compare survivor, range, and indirect words
exactly with this oracle. The buffers remain device-resident and expose an
application-owned encoder seam; WebGL2 continues to use current-pose
degenerate-vertex rejection until backend integration reaches parity.

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
queue supplied by the caller, five retained compute pipelines, immutable model,
atlas, batch, and eligibility buffers, retained dynamic pose/subject buffers,
and diagnostic staging. Uploading a model or scene uses the existing word
packers. A classifier dispatch writes only the current uniform, dense subject
rows, referenced joint-matrix prefix, and morph weights. Visibility compaction
keeps source flags, survivor IDs, batch ranges, and indirect arguments on the
device; its application-owned encoder seam can sit between a GPU visibility
producer and render passes without a map/copy boundary. Extra unused glTF skin
joints do not inflate retained storage or invalidate the full pose.

The crate pins `wgpu` 29.0.1. That is the newest release line tested to compile
against this workspace's exact wasm-bindgen 0.2.108 / js-sys 0.3.85 browser
ABI. `wgpu` 30's browser backend uses newer typed JavaScript bindings, so a 30
upgrade belongs to a coordinated binding-stack migration and browser
regression gate rather than this isolated classifier cut.

The narrow device gate now has three proven parts:

- native Radeon 780M / RADV / Vulkan executed both classifier passes plus
  count/scan/scatter compaction, matched the Rust oracles exactly, and consumed
  the resulting ranges and survivor IDs through indexed-indirect draws; and
- `cargo check -p quilting-webgpu --target wasm32-unknown-unknown` passes on
  the repository's pinned browser ABI; and
- Chromium 150 requested a real `BrowserWebGpu` adapter and executed the same
  reusable matrix, returning one exact full-pipeline word, ten coherence words,
  89 compacted survivor words, three five-word ranges, three five-word indirect
  records, and three real indirect draws with no console warnings or errors.

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
maximum-atlas boundaries, and exact WebGL2 image/workload comparison remain
promotion gates. The compaction fixture crosses three 64-lane chunks, preserves
stable source order, covers a disabled bucket and retained-root replacement,
and keeps indirect `first_instance` zero for baseline WebGPU portability.

The WebGL2 GLSL programs and runtime authority remain untouched. WebGPU LOD
classification now has separate input-write and command-encoding entry points
that return a borrow-safe device output with a monotonic encoding epoch. The
resident model no longer allocates a staging buffer. Full classifier readback
is an explicit diagnostic wrapper that allocates temporary conformance staging;
the no-readback path performs no copy, map, or await. Compaction diagnostics use
the same temporary-staging policy. Live residency exposes classification,
source visibility, compacted range/survivor, aligned batch-index, and indirect
buffers for an ordered device-only path. An authoritative backend should copy
only bounded, delayed telemetry to the CPU.

Resident crack-free topology now has the same device-only boundary. A seed
pass decanonicalizes the classifier word into face-local edge exponents, ten
ping-pong Jacobi passes restore shared-edge equality plus the selected 2:1 or
4:1 within-face grading, and a final pass re-canonicalizes the result and
selects its atlas entry. Ten passes are a bound, not a convergence guess: the
input exponent lattice is `[0, 9]`, and propagation across another face loses
at least one exponent through grading. The independent queue-based Rust oracle
matches `quilting-core`'s retained fixed-point semantics, including invisible
faces, and native/browser conformance covers a ten-face maximum-range chain for
both policies. Visibility and adaptive priority remain unchanged in the final
packed word. No resident pass uses a counter readback or CPU publication.

Patch preparation now writes the exact 208-byte prepared record on-device, and
the first production WGSL graphics pipeline pulls those records through the
same pure rational-QB/Möbius evaluator used by WebGL2. A validated
`RenderSceneSnapshot`/`RenderFrame` executor coalesces the canonical all-batch
prepare and visibility phases, selects a device-resident conformal frame per
batch, derives winding from semantic orientation plus S3 parity, and issues
portable compacted indexed-indirect normals draws. Native Vulkan and Chromium
both execute a two-batch retained-root/adaptive-overlay frame; the browser gate
reports two shared-frame draws and a nonempty ten-pixel footprint with no
console warning or error. Each draw borrows an element-offset slice of the
existing packed global barycentric/index atlas, keeping indirect `first_index`
zero without allocating one WebGPU buffer per canonical patch. The retained
WebGPU atlas owner now accepts the live seven-word patch metadata and shared
barycentric/triangle/line arrays directly, validates canonical keys, ranges,
finite vertices, and global indices once, and resolves an extracted scene to
borrowed batch views without copying geometry.

The application now has rollback-safe WebGPU residency behind the canonical
`gfx` route. `gfx=webgl2` is the inert default. `gfx=webgpu-shadow` requests a
headless browser device after the incumbent renderer initializes, mirrors each
atomic packed-atlas replacement, and uploads the same prepared composed LOD
model without constructing a redundant WebGL classifier when the worker
remains authoritative. The backend now replaces preparation/compaction/binding
resources as one atomic `PatchRenderScene`, resolves packed-atlas slices during
encoding without a per-frame draw-view allocation, and executes supported live
normals frames into a depth-backed offscreen target. `gfx=webgpu` currently
performs that same residency and shadow execution but
reports `effective=webgl2`, `state=presentation-fallback`, and a concrete reason
until presentation parity exists. A failed device, atlas, or model shadow is a
diagnostic warning and never retires working WebGL resources.

The first live Chrome residency check retained 22 canonical atlas entries,
25,551 barycentric vertices, and the horse's 984-face prepared model through a
`BrowserWebGpu` adapter with one initialization, atlas upload, and model upload.
The canvas continued through WebGL2 with no console errors. The default route
remained disabled and allocated no WebGPU residency; the explicit presentation
request declared its fallback.

The first live normals gate extracted five real horse batches and 984 source
instances at a 496×770 viewport, then submitted five compacted indirect draws
per changed frame with no validation warning or frame failure. Exact retained
input comparison prevents an idle requestAnimationFrame loop from burning a
second renderer: one static run observed 1,604 frames, submitted one, and
skipped 1,603 unchanged inputs. Animated pose changes correctly submit new
frames. The same-frame pose bridge preserves the incumbent renderer's active
morph-weight prefix and zero-fills inactive resident targets through retained
scratch storage, preventing stale GPU suffix weights without a warm-path
allocation. Animated execution also exposes the next bottleneck honestly:
worker-driven animated LOD changes publish a new batch topology almost every
accepted epoch. The retained WebGPU aggregate now updates topology, subject,
batch, and eligibility buffers in place whenever patch, subject, and batch
cardinalities remain compatible; only a shape change constructs replacement
buffers and bind groups. One animated horse run observed 1,161 coherent scene publications:
1,159 retained updates and two model-boundary rebuilds, with zero frame
failures. Visibility is no longer flattened and widened on the CPU after each
batch reorder. The shadow uploads one compact face-indexed bitset only when it
changes, and a 64-lane GPU pass expands those bits through current patch
topology immediately before compaction. For the 984-face horse that input is
31 words, or 124 bytes. A 1,907-frame animated run submitted 1,906 frames,
updated retained topology 1,733 times, and uploaded visibility twice for 248
bytes total with zero failures; the earlier 1,220-frame flattened run uploaded
4,801,920 bytes. The shadow still repacks topology on the CPU from worker
readback, which remains instrumentation rather than the intended authority.

The retained scene now also binds the final packed resident-LOD buffer directly
to that same flattened visibility destination. A 64-lane adapter projects the
packed face visibility bit through current root/adaptive topology, after which
the existing count/scan/scatter passes consume it without a copy, map, or CPU
publication. Every uploaded model receives a monotonic device-local identity;
the encoder rejects a resident result from another model even when both models
have the same face count, and separately checks the exact face domain. Native
conformance classifies a two-face scene with one authored subject translated
offscreen and proves both `[1, 0]` source order and `[0, 1]` reordered topology.
The shared native/browser matrix runs real visible and offscreen
classifier/reconciliation chains and projects both over two adaptive leaves;
Chromium reports `resident_visibility_words=4` with no console warning or
error. The current live WebGL2-authoritative shadow deliberately keeps the
compact CPU bitset adapter until its classifier and draw construction are moved
onto this same device; the direct resident path is ready for that cut and does
not introduce a diagnostic readback into production.

The next device stage now derives a deterministic retained-root geometry plan
from the packed resident words themselves. Its bounded bucket domain is sorted
atlas index × S3 parity (at most `255 × 2`), with one packed source-face
eligibility field for roots replaced by the sparse adaptive overlay. A
64-face-chunk histogram, independent per-bucket chunk prefix, one bounded
global bucket scan, and source-order scatter produce compacted face IDs,
five-word ranges, and portable indexed-indirect records. Local histogram
atomics affect counts only; the scatter rank is defined by source order, so the
GPU output is byte-exact with the independent Rust oracle across chunk
boundaries. Indirect records retain the packed global atlas's real
`first_index`, removing the need to bind a different index-buffer slice for
each root bucket.

This stage intentionally covers retained roots, not adaptive dyadic leaves.
Leaves of one authored face can have distinct edge LOD and permutation, so
pretending to recover them from one source-face resident word would discard
the conformal screen partition. The existing sparse overlay remains a separate
correctness layer until partition and leaf-LOD extraction move onto the GPU.
The two retained chunk tables cost
`2 × ceil(source_faces / 64) × (2 × atlas_entries) × 4` bytes and are rejected
up front if they exceed the selected device's storage-binding limit. Native
Radeon/Vulkan and Chromium `BrowserWebGpu` both match 323 exact output words
across a 137-face, three-atlas, all-S3 fixture with two successive suppression
fields; the browser also runs a real classifier → reconciliation → root-bucket
chain and reports no console warning or error.

The optimized monolithic WASM artifact now measures 7,669,789 bytes with this
backend enabled versus 7,459,748 bytes with the feature disabled: a
210,041-byte (2.82%) release delta. Both artifacts were rebuilt from the same
tree with `wasm-pack --release` and `wasm-opt`; debug artifacts are not
representative, so size gates must continue to compare optimized outputs.

The next cut consumes the root plan in actual graphics execution: prepare one
root record per compacted face, extend the bucket key with material/render
domains where required, and compose the sparse adaptive overlay without
rebuilding the retained baseline from worker readback. After that, attach the
live shared frame executor to a browser surface and add WebGL2 image/workload
parity behind the existing explicit backend switch. PBR, wire, LOD color,
stretch, picking,
material/texture binding, and postprocess commands still reject explicitly
rather than silently lowering to normals. That cut is what finally removes the
live classifier readback, CPU topology repacking, and rejected atlas vertex
invocations from an authoritative runtime.

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
