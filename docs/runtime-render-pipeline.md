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

`RenderFrame::execution` is the single admission seam. A frame built directly
validates the supplied scene and regenerates the expected command sequence as
the compatibility oracle. A frame built by `RenderFrame::from_command_plan`
carries the immutable `RenderCommandPlan` that created it, shares that plan's
exact `Arc<[RenderCommand]>`, and automatically takes the allocation-free path.
That path checks scene-allocation identity, revision, command key, and command
allocation identity without rescanning the scene or allocating an expected
vector. The plan shares one `Arc`-backed `ValidatedRenderScene` epoch and
rebuilds only when the scene, render style, or command-presence key changes.
Camera matrices, pose samples, focus parameters, and uniform-only matcap changes
can therefore remain high-rate data. Both paths reject a stale/distinct scene
or a noncanonical command sequence before device work, then resolve every batch
command to the immutable batch metadata and exact index count it addresses.

The main renderer now creates one `ValidatedRenderScene` allocation per
structural extraction epoch. The WebGL command plan, optional parity observer,
and WebGPU `PatchRenderScene` retain clones of that exact `Arc`-backed scene;
WebGPU no longer rewrites a backend-local revision, clones member tables, or
revalidates a second snapshot. `renderSceneExtractions` and
`renderSceneExtractionFailures` make that low-rate boundary measurable.
The one-shot backend image-evidence request first synchronizes pending retained
batches and preflights PBR support against this same current scene allocation;
it no longer constructs a request-local snapshot that could disagree with the
epoch published on the following frame.
Prepared-patch, focus-PBR, resident-root, and adaptive-overlay entry points all
call the ordinary `RenderFrame::execution`
seam before device work, so a plan-built frame automatically uses retained
admission while a directly built frame remains the rollback and parity oracle.
Atlas admission, ordered draw traversal, highlight admission, and workload
accounting use that admitted execution. Root/adaptive composition additionally
computes its logical submission evidence before queue writes or submission;
invalid scene/frame pairs therefore cannot partially touch device state before
failing. Extending the command enum fails closed until a backend handles the
new command. With `rendershadow=1`,
non-PBR WebGL2 styles preflight every retained batch against that resolved scene
and execute its draw commands directly; a mismatch falls back before the first
diagnostic draw.

The main renderer retains exactly one active `RenderCommandPlan` beside its
shared validated scene. It rebuilds that plan only when the scene allocation,
render style, or command-presence key changes; camera matrices, animation pose,
focus-field values, and uniform-only options construct frames with
`RenderFrame::from_command_plan`. WebGL parity and WebGPU consume this same plan
allocation and the main renderer constructs one high-rate `RenderFrame` for the
current view. The parity observer and WebGPU consume that exact frame. WebGPU
validates it against the scene allocation retained by `PatchRenderScene`; it no
longer owns a parallel plan cache, rebuild counter, or frame constructor.
Scene residency compares both the structural revision and exact shared `Arc`
identity, so dropping and re-extracting a main-renderer cache cannot leave an
equal-numbered but distinct WebGPU epoch silently resident.
`renderCommandPlanBuilds` exposes the sole low-rate cache boundary in
main-renderer diagnostics. The route/shadow smoke oracle rejects any return of
`RenderFrame::build`, `RenderCommandPlan::build`, or
`RenderFrame::from_command_plan` to the ordinary browser WebGPU path.
The main renderer creates this shared scene/plan/frame chain only when render
parity is enabled or WebGPU will actually attempt the current diagnostic style
(plus explicit one-shot PBR evidence). Merely having a ready headless adapter
therefore adds no extraction, plan, or frame work to ordinary incumbent PBR.
`RenderPoseIdentity` now combines the structural asset revision with the exact
animation continuity epoch and local pose revision; camera motion no longer
pretends to be a pose change. Device LOD classification and rendering consume
that same identity and the same renderer-owned pose payload. Classification
therefore reuses pose buffers on camera-only LOD updates, and rendering reuses
the pose that classification just published instead of uploading it twice.
The three-state `PoseUploadPolicy` distinguishes a full dynamic publication,
initialization of newly retained preparation uniforms, and complete reuse; the
resident root and optional adaptive overlay advance as one family. The backend
no longer retains comparison copies of the joint and morph vectors.
`classifierPoseUploads/Reuses`, `fallbackPoseUploads/Initializations/Reuses`,
and `residentPoseUploads/Initializations/Reuses` expose the bounded paths. A
model replacement still forces a full upload, while a scene/preparation change
forces only its local uniform initialization when the device pose is current.
Retained fallback, resident-root, and adaptive-overlay frame state is split
losslessly into one 176-byte `PatchRenderGlobal` row and 80-byte
`PatchRenderDomain` rows. Each table compares exact packed words before queue
publication. An animation-only frame therefore reuses both halves; camera,
focus, and selection motion uploads 176 bytes per retained family instead of
resending 256 bytes per batch/domain; local Möbius or material changes upload
only the compact domain table. A packing failure invalidates only the affected
witness. Fallback packing writes directly into retained staging storage
instead of allocating an intermediate frame vector. The same pair drives
rendering, visibility, focus-field output, and both picking paths.
`frameTableUploads`, `frameTableReuses`, and `frameTableUploadBytes` report the
device-lifetime traffic, including work preceding a surface skip.
`resolvedExecutionFrames`, `resolvedExecutionFallbacks`, and
`lastExecutionError` make that gate observable through the existing render
shadow diagnostics. Shadow scene validation and WebGL PBR lowering now happen
once per structural plan rebuild, not once per frame; `renderCommandPlanBuilds`
records that low-rate work. The default WebGL2 path still consumes the shared
pass plan without retaining or validating this shadow scene, so the measured
cutover adds no default CPU scan. PBR shadow frames reuse the same exact
residency preflight and canonical opaque/transparent order plus
transmission-pyramid and focus-postprocess boundaries. Browser framebuffer,
material, and texture effects remain incumbent, but their control flow follows
that retained plan; a validation failure retains the complete legacy selection
path for the frame.

`RenderStyle` also resolves to one canonical ordered `RenderDrawPassPlan`
slice. Each pass names its geometry and semantic batch selection (`all`, PBR
opaque, or PBR non-opaque). `RenderFrame` construction and the incumbent WebGL2
dispatcher iterate this same plan; shader selection, material binding, and API
resources remain backend-local. The WebGPU dispatcher consumes the resulting
resolved command order while lowering submission to compacted indirect draws.

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

The retained WebGPU focus composer is the first concrete lowering of that
contract. Its seven `RenderPipelineDescriptor` values authoritatively select
the shared bind-group layout, shader entry points, primitive/multisample state,
and color targets; the same ordered vector is the device memo key. WebGPU
formats outside the currently named portable `TextureFormat` subset preserve
the older uncached construction path instead of being folded into an
incomplete key. Resident-root rendering now uses the same rule for its sixteen
style/winding variants, including its shared root layout, PBR atlas/environment
groups, focus MRT, vertex layout, and highlight depth-write policy.
Prepared/adaptive patch rendering now follows that rule as well. Its requested
style family describes the shared prepared-record layout, ordinary PBR texture
table and environment groups, vertex pull layout, triangle winding variants,
single reused wire variant, optional focus MRT, and highlight depth-write
policy. The exact ordered family is retained per device; a hit returns cloned
WebGPU handles without revisiting WGSL flattening or pipeline construction.
Formats outside the portable subset continue through the explicit uncached
builder rather than receiving a lossy key.

All three functional WebGPU families now cross one mechanical effect boundary:
`pipeline_lowering::render_pipeline`. It validates contiguous vertex slots,
materializes borrowed vertex attributes only for the duration of pipeline
creation, and lowers primitive, depth/stencil, multisample, blend, write-mask,
attachment, and shader-entry state from the descriptor. Focus, prepared patch,
and resident-root builders validate their semantic family order, then use that
same operation. Their uncached nonportable-format fallbacks remain explicit
rollback paths and are not presented as descriptor-backed.

Prepared/adaptive rendering also has one semantic pass plan above that effect
boundary. Style support, ordinary versus focus PBR entry points, triangle or
line geometry, highlight depth policy, winding multiplicity, and retained
pipeline kind are selected once in `prepared_patch_pipeline`. Both the pure
descriptor vector and runtime family builder iterate that same plan, so adding
a style cannot silently change the memo key without changing the pipelines it
constructs.

Resident-root rendering applies the same rule to its eight named families.
PBR, focus PBR, matcap, normals, LOD, stretch, wire, and highlight now share
one ordered plan containing shader entry, layout class, geometry, focus MRT,
and depth-write policy. The sixteen winding descriptors and the sixteen
retained WebGPU handles are built by iterating that plan; named fields are only
assigned after the complete family succeeds.

Focus postprocessing completes the pattern with one seven-pass retained-family
plan. Select-weight, JFA init/step, firmness, Kawase, intermediate directional
blur, and final directional blur each name their label, fragment entry, and
attachment class once. Pure descriptors and runtime handles iterate the same
plan, while the separate per-frame `FocusPipelineKind` continues to describe
scheduled executions rather than retained GPU identity.

`quilting_renderer::memo::DeviceMemo` is the effect boundary. It maps a pure
descriptor to a concrete backend resource, inserts only after construction has
fully succeeded, and scopes every entry to an explicit device/context epoch.
Changing epoch returns the old resources for destruction by the backend that
created them. This cache is derived runtime state: FRP may publish a changed
render plan or descriptor, but HHHS must never replicate GPU handles, cache
contents, compilation status, or per-frame bindings.

The primary graphics WGSL now uses one collision-free binding namespace:

| Group | Stable responsibility |
|---|---|
| 0 | pose and source data: joints, skinning, morph, source-face data |
| 1 | entity/batch draw packet: view, Möbius packet, Euclidean model/normal, QB enablement |
| 2 | material/style uniforms and material textures |
| 3 | pass resources: scene color, transmission, blur/focus, highlight |

The current group-1 `Uniforms` block still combines view fields with per-batch
entity state because that is the incumbent WebGL payload; its actual update
frequency is per draw. Splitting immutable frame fields from entity state is a
later payload migration and no longer requires another namespace change.

WebGL program identity retains the original WGSL source name, `(group,
binding, stage)`, complete portable resource policy, emitted GLSL name, and
assigned UBO point or texture unit. Uniform blocks record their exact minimum
size; textures record sample kind, dimension, and multisampling; samplers record
filtering policy. That provenance is checked against Naga's reachable-resource
reflection for the selected composed entry point. The catalog does not claim
inactive `face_data`, suppression-mask, sheen LUT, or blurred-scene bindings
merely because another entry or an unreachable declaration shares the source
module.

Every primary program now has zero cross-stage slot conflicts.
`WebGlBindingPlan::portable_layout` converts the exact reachable interface into
the shared `PipelineLayoutDescriptor`, inserting explicit empty positional
groups only when a program omits an intermediate responsibility. A deliberate
legacy collision fails before conversion. WebGL continues to lower the same
plan to its established UBO points and texture units; the superseded heuristic
name-probing binder has been deleted. This proves that the primary shader
interface is representable by either API. It does not yet claim resource-model
or image equivalence with the live resident-root WebGPU pipelines, whose
storage/atlas layouts remain separately validated until shared extraction is
promoted.

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
portable compacted indexed-indirect diagnostic draws. Native Vulkan and
Chromium both execute a two-batch retained-root/adaptive-overlay frame; the
browser gate reports two shared-frame draws and a nonempty ten-pixel footprint with no
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
diagnostic frames into a depth-backed offscreen target. `gfx=webgpu` instead
claims a separately supplied presentation canvas on a surface-compatible
adapter before that canvas has any context. It specializes the same retained
pipelines to the surface format and presents the same extracted live frame.
The original WebGL2 canvas remains a transparent input/picking and rollback
layer; it becomes visible for unsupported modes or any presentation failure.
No code attempts to repurpose a claimed canvas context. A failed device, atlas,
model, or surface never retires working WebGL resources.

The first live Chrome residency check retained 22 canonical atlas entries,
25,551 barycentric vertices, and the horse's 984-face prepared model through a
`BrowserWebGpu` adapter with one initialization, atlas upload, and model upload.
The default route remains disabled and allocates no WebGPU residency. Shadow
mode keeps WebGL2 visible; direct mode can now make supported WebGPU styles
visibly authoritative after their first successful presentation.

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
error. The browser now sends the same validated full-scene projection, density
floor, subject table, and exact retained pose to a WebGPU classifier request. A
successful epoch stays device-resident through crack-free reconciliation,
visibility expansion, stable compaction, and indirect drawing; ordinary frames
perform neither a visibility upload nor a readback. A partial animated-primary
request explicitly retires that epoch and exposes the compact CPU-bitset
adapter again, so it cannot combine a stale full-scene classification with a
new pose. For normals or LOD-color scenes consisting entirely of unsuppressed
source roots, the live browser backend now continues from that epoch through
device root topology emission, QB preparation, atlas/parity bucketing, and indirect
submission. It does not prepare or compact the CPU-authored batch topology for
that frame. The root scene is published atomically beside the established
retained scene and has separate readiness, upload, frame, fallback, and error
diagnostics. Its exact face-to-domain and domain-state key is independent of
CPU-authored LOD buckets, so ordinary LOD regrouping reuses the same root GPU
buffers; model/atlas replacement or a real authored-domain change invalidates
them. Root pipeline/allocation failure is optional and falls back without
disabling the established WebGPU modes. Adaptive leaves, suppressed roots,
partial animated-primary classification, and wire, stretch, or
material/composite styles deliberately retain the established path; WebGL2 and
the worker remain rollback authority throughout this bounded promotion.

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
Radeon/Vulkan and Chromium `BrowserWebGpu` both match 243 exact output words
across a 137-face, three-atlas, all-S3 fixture with normal, orientation-reversing,
and disabled draw domains plus two successive suppression fields. Disabled
domains fail closed before scatter, while orientation reversal is folded into
the effective S3 parity. The browser also runs a real classifier →
reconciliation → root-bucket chain and reports no console warning or error.

Direct root preparation now consumes that same resident word without a CPU
topology projection. A vertex-clear and face-atomic-maximum pair reconstructs
the continuous compact-vertex density field used by LOD coloring, including
the standby topology of currently invisible faces. A third pass restores the
face-local edge order from the S3 permutation and emits the existing 48-byte
`PatchTopologyRecord` in source-face order. The ordinary animated rational-QB
preparation pass binds that output directly and writes prepared records by
source face. Consequently the compacted face IDs and five-word bucket ranges
are already layout-compatible with the render vertex puller; no second scatter
or widened instance stream is required.

The independent Rust oracle covers shared vertices, every packed topology
field, exact affine-subject rows, incomplete/conflicting extraction, and
source-model validation. Native Radeon/Vulkan and Chromium `BrowserWebGpu`
match 240 topology words and 1,040 prepared-record words across both 2:1 and
4:1 resident closure, with no console warning or error. Diagnostic staging is
allocated only by the conformance wrappers; production encoding remains one
same-device command chain.

The optimized monolithic WASM artifact now measures 7,511,347 bytes with this
backend enabled versus 7,295,009 bytes with the feature disabled: a
216,338-byte (2.97%) release delta. Both artifacts were rebuilt from the same
tree with `wasm-pack --release` and `wasm-opt`; debug artifacts are not
representative, so size gates must continue to compare optimized outputs.

The shared render contract now extracts one immutable draw-domain row per
source root from only the `(material, render node)` pairs actually present in
the scene. Equal pairs must agree on enabled state, PBR class, transform, and
orientation; every source face must resolve exactly once, including a root
temporarily suppressed by the adaptive overlay. Sparse or very large material
IDs therefore do not inflate the domain table. WebGPU retains one four-word
face selector and one 16-byte record per observed domain; native and browser
conformance match the exact 18-word two-domain fixture, including a
million-scale material ID and orientation-reversing state.

An independent state-sorted oracle can still group eligible roots by
`(domain row, atlas, parity, source face)` and emits only occupied triangle and
line ranges. It is a correctness reference for backends that require
state-sorted material draws, not a mandate to build a GPU radix sorter. Baseline
browser WebGPU has no variable-count multi-draw-indirect facility: wgpu can
only emulate a CPU-known fixed count there. The scalable WebGPU route is
therefore domain indirection in the vertex/material stages while retaining the
bounded atlas/parity schedule. Entity orientation folds into effective parity;
only genuinely distinct pass classes such as opaque/non-opaque may multiply
that bounded schedule. This stays independent of material count and adds no
frame readback.

The retained-root path now binds those source-indexed prepared records,
compacted face IDs, bucket ranges, domain tables, and global-atlas indirect
arguments into actual graphics execution. One shared vertex module evaluates
both ordinary prepared-patch draws and device-resident roots, preventing the
QB/conformal vertex contract from drifting between paths. A source root pulls
its compact domain row in the vertex stage; disabled domains become degenerate,
and the domain selects its own conformal/entity frame without a material ×
atlas cross-product. The global barycentric vertex and triangle-index buffers
are bound once. A fixed, browser-portable atlas × effective-parity loop then
issues indexed-indirect draws using each atlas record's real `first_index`.

The direct-render conformance is deliberately end-to-end rather than a staged
fixture: classifier → 2:1 resident reconciliation → root topology → animated
QB preparation → domain-aware histogram/scan/scatter → two fixed indirect
draws are encoded in one command buffer with no intermediate map or copy.
The sparse adaptive layer now projects only replacement batches from the same
fully validated scene. It owns leaf topology, deduplicated affine subjects,
prepared output, current visibility, compacted IDs, and indirect arguments;
the immutable 208-byte source-face table is reference-counted with the root
preparation scene instead of copied. Root suppression is a separate explicit
publication after the candidate overlay is completely allocated, so a failed
candidate cannot punch holes in the currently drawable baseline. Its packed
mask is retained and checked at encode time rather than rewritten every frame.

Composition uses the same classifier epoch and global atlas. Roots clear the
shared color/depth target, then adaptive batches load both attachments and
issue one indirect draw per sparse state bucket. The composed conformance keeps
one resident root, suppresses a second, and replaces it with one dyadic child
inside a single command encoder. Radeon/Vulkan passes the hardware gate;
Chromium `BrowserWebGpu` renders an exact 144-pixel normals footprint and
reports `resident_root_indirect_draws=2`, `adaptive_overlay_patches=1`, and
`adaptive_overlay_indirect_draws=1`, including an orientation-reversing domain,
a million-scale material ID, and no console warning or error.

The backend now owns a real presentation-surface lifecycle as a separate Rust
resource. Browser creation requests an adapter compatible with a previously
unclaimed canvas, chooses the adapter-preferred surface format and FIFO when
available, specializes the graphics pipelines to it, configures a matching
depth attachment, suspends cleanly at zero size, recreates attachments on
resize, retries one outdated acquisition, skips
timeout/occlusion, and reports surface loss for canvas-level recreation. Frame
ownership is a closure over the same `PatchRenderTarget` accepted by offscreen
execution, so promotion does not introduce a second draw API. The browser gate
uses the same device for the complete resident/adaptive matrix and a 256×256
canvas presentation; Chromium selects its preferred `Rgba8Unorm`, reports one
presented frame and one configuration, and emits no warning, error, or
assertion. That frame is not a clear-color probe: the exact two-domain retained
root plus sparse dyadic replacement fixture is classified, reconciled,
prepared, bucketed, and drawn through the shared executor directly into the
surface. Its root/overlay encoding must equal the separately raster-verified
offscreen encoding before the browser gate can pass.
The old nonzero-word footprint check is now replaced by normalized image
evidence over the actual padded texture copy. Chromium reports 144 alpha-covered
pixels and canonical RGBA8 hash `3fe933548a480845`; native Radeon/Vulkan passes
the same coverage/signature construction without assuming byte origin or
channel order.

Hyperscope now has an opt-in presentation cut at `gfx=webgpu`. Browser layout
supplies an unclaimed `cv-webgpu`; Rust owns its adapter/device, surface format,
depth attachment, resize, acquisition, encoding, submission, and presentation.
The existing `cv` remains the event target and hidden WebGL2 rollback layer
while matcap, wire, matcap-plus-wire, normals, LOD color, or conformal stretch
is supported. The adapter reveals it synchronously for every other mode, and
reveals WebGPU only after a successful fresh presentation in the requested
style. The residency
diagnostics carry that presented style, preventing an old normals surface from
being exposed during a mode switch. WebGPU now submits first: a successful or
safely retained surface frame elides
the entire hidden WebGL2 frame—including begin, patch preparation, camera
visibility, and patch-color draw—while unavailable residency or any frame
failure falls through to WebGL2 in the same render call. The incumbent dirty
stamps remain pending, so a later fallback or mode switch refreshes before it
draws. Picking was already independent: `mr_pick` prepares and classifies the
retained WebGL2 buffers against its exact pick camera on demand. This
dual-canvas interval is deliberate input/picking migration debt, but ordinary
WebGPU diagnostic presentation no longer pays for two rendering pipelines.
Runtime diagnostics expose `webglPatchFrames`,
`webgpuPresentationPatchFrames`, and `webgpuRetainedPresentationFrames` so the
ownership cut is measurable rather than inferred from canvas opacity.

The live shadow still draws the same extracted normals frame into its retained
offscreen target. An explicit one-shot diagnostic rerenders the incumbent
WebGL2 frame into a single-sample RGBA8/depth target, tags both backends with the
same source render call, and rejects stale frames or viewport mismatches before
readback. This keeps all readback and duplicate diagnostic work outside the
ordinary frame path.

The first live Chromium comparison covered a 496×826 horse frame. Both paths
submitted exactly 5 draws, 984 instances, and 1,160 triangles with draw-sequence
hash `9227089913e7d75c`; WebGPU physically issued 5 indirect draws over the same
984 source instances. After origin/row normalization, 170 of 409,696 pixels
differed (414 ppm), with a normalized mean absolute error of 3 millionths. Only
2 silhouette pixels disagreed in coverage (4 ppm); the remaining differences
were predominantly one-channel rounding at shared covered pixels. The
diagnostic background preserves the incumbent RGB clear but uses transparent
alpha so coverage is measured rather than inferred from color. Chromium logged
no warning, error, assertion, or failed request. That evidence admitted the
explicit normals cutover. A fresh 496×770 direct run visibly rendered the horse,
reported `effective=webgpu`, 5 physical indirect draws over 984 instances, and
no surface loss or frame failure. Switching to PBR exposed WebGL2 with
`state=unsupported-mode`; switching back advanced the surface presentation
counter from 4 to 5 before WebGPU became visible again. No warning, error,
assertion, or failed request occurred. The same retained scene now selects
matcap, wire, normals, LOD-color, and stretch pipelines that share one shader
module and one binding-layout identity. Wire consumes the packed line atlas and
a line-list pipeline. Visibility is compacted once into one survivor/range
stream plus separate triangle and line indirect tables, so matcap-plus-wire
performs both ordered passes without a scene rebuild or a second
preparation/classification pass. Matcap profile selection is a typed
backend-neutral `RenderFrameOptions` value packed into the existing frame table,
not a second browser uniform path. Native Radeon/Vulkan renders nonempty,
pairwise-distinct images for matcap, LOD, stretch, wire, and matcap-plus-wire
without rebuilding scene resources and validates both indirect geometry
tables. Their live Chrome gate remains pending. Picking and unsupported
PBR/postprocess commands still reject explicitly rather than silently lowering
to another style. Those cuts are what remove the live classifier readback, CPU
topology repacking, and rejected atlas vertex invocations from an authoritative
runtime.

Authored PBR state now crosses the backend boundary as validated
`quilting_core::material::PbrMaterial` records carried by
`RenderSceneSnapshot`. Texture references are typed optional indices rather
than negative sentinels, glTF's infinite attenuation distance remains an
explicit `None`, and both backends use the same material-index fallback rule.
WebGPU retains a stable 160-byte material table and renders the opaque/masked
subset with shared Rust BRDF code, alpha masking, unlit and double-sided
semantics, selection tint, tone mapping, and fade. The expanded record preserves
normal scale, occlusion strength, base/normal UV transforms, transmission and
volume factors, and a compact authored-texture mask. Decoded glTF images remain
in a sparse index-preserving device table with linear and sRGB views. Baseline
per-material bind groups resolve base color, metallic/roughness, normal,
emissive, occlusion, and transmission channels without sampled-texture arrays;
white, black, and flat-normal 1x1 resources give every unavailable slot a
channel-correct fallback. Browser `ImageBitmap` handles copy directly into that
table before closure, without canvas readback or a texture-sized WASM copy.

Image-based lighting now has the same rollback-safe boundary. A validated
backend-neutral asset describes one complete power-of-two RGBA32F prefiltered
cube mip chain plus one irradiance cube. WebGPU validates both payloads and the
device limit before allocation, converts one face at a time into filterable
RGBA16F storage, and publishes one group-two binding only after both cubes are
complete. Missing IBL binds a valid black placeholder while selecting the
analytical fallback. Resident PBR samples diffuse irradiance and the
roughness-selected specular mip through the shared BRDF. The existing browser
environment upload mirrors the same borrowed WASM slices after WebGL succeeds;
it does not cross JavaScript again or read a canvas. Environment replacement
and later scene creation both preserve the last coherent binding epoch.

The required Radeon/Vulkan gate now validates sparse binding residency and
renders both a real base-color texture and independently replaced IBL through
the shared PBR draw; both images are nonempty and differ from their respective
fallback oracles. This is deliberately not a live-browser cut yet:
transmission/blend sequencing, focus postprocessing, and browser image-parity
acceptance remain required, so Hyperscope still keeps WebGL2 visible for PBR.
An explicit one-shot basic-PBR evidence request now admits only the opaque,
non-sheen, non-focus subset on a headless backend with resident IBL, captures
the incumbent framebuffer after its real material pass, and stages the matching
WebGPU frame. Ordinary PBR frames still incur no WebGPU shadow work. Texture,
environment, and scene candidates fail before publication if their device
resources cannot be made coherent with the retained epoch.

Cross-backend image evidence now has one backend-neutral diagnostic contract
in `quilting-core`. It validates exact dimensions and row strides, normalizes
WebGL bottom-left versus staging top-left origin, RGBA versus BGRA channel
order, and padded GPU-copy rows, then produces a canonical RGBA8 hash, coverage
and channel moments. Pairwise comparison reports exact mismatched/coverage
rates, normalized mean error, per-channel maxima, and at most eight pixel
examples against an explicit tolerance. The contract is intentionally absent
from the warm frame path. The live one-shot diagnostic now feeds the same
source frame to both backends. A broader route/asset matrix and the remaining
render modes are still required before WebGPU can become the default.

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
