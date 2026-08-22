# Runtime render pipeline and backend boundary

This is the current execution contract for Hyperscope. It distinguishes work
that happens every browser frame from asynchronous topology work and from
load-time preprocessing. `SPEC.md` describes the mathematical model; this file
is the operational map to use when profiling or adding a WebGPU backend.

## Steady-state browser frame

The request-animation-frame callback performs only the following work for an
ordinary glTF scene:

1. Apply a pending canvas resize, if any.
2. Advance animation time. At most one worker pose request is in flight; its
   returned joint matrices and morph weights are uploaded to retained GPU
   resources. Source vertices are not sent back per frame.
3. Sample retained mouse/SpaceMouse state once and update the camera and shared
   focus/inversion sphere. The browser currently adapts device axes directly;
   the accepted ownership target is deterministic `hyperscape::FocusNavigation`
   plus a Rust camera rig. An inversion-sphere edit transports the camera eye
   exactly with the conformal point map and its frame/scale with the local
   differential. Camera matrices reuse preallocated typed arrays.
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
   six floats per face.
4. The output is copied into staging, fenced, and polled without blocking the
   worker. Only after the fence signals is the compact topology payload read
   back into the worker. Staging buffers and mesh-sized CPU readback vectors
   are retained and reused across jobs; the steady-state path does not create
   or resize either resource.
5. The WASM worker compares that snapshot with its retained previous result
   before creating any JavaScript typed array. The first coherent result after
   a model, animation, remesh, or compute-resource boundary transfers all
   faces; later results cross into JavaScript and transfer only changed
   `(face_index, six-float classification)` records.
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
worker-side GPU readback is 24 bytes per source face per completed
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

## Work outside the frame loop

- glTF parsing, image decode, static face textures, adjacency, and skinning
  data are produced once per model load.
- The canonical tessellation atlas is restricted to 2:1-reachable triples,
  packed once, cached in IndexedDB with a versioned key, and uploaded once.
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

- `quilting_core::batch::{ResidentLod, RenderBatchKey, RenderBatchMember}`;
- `quilting_core::instance_layout`, including the eight-float topology record
  and 52-float prepared-patch record;
- canonical atlas keys and S3 permutation semantics; and
- WGSL surface, preparation, and material shader logic.

`MeshDraw`, VAOs, transform-feedback objects, GL textures, UBOs, framebuffers,
and `glow::Context` are explicitly WebGL backend state. They must not migrate
into a shared scene or command model.

## WebGPU migration sequence

1. Port the two handwritten GLSL LOD passes to WGSL and test their payloads
   against the CPU conformance suite.
2. Implement patch preparation as a compute pass writing the same logical
   52-float record. Keep the WebGL transform-feedback implementation as the
   compatibility backend.
3. Move edge reconciliation, 2:1 grading, resident-topology selection, and
   visible-instance compaction into storage buffers.
4. Emit indirect draw arguments per `RenderBatchKey`. This removes the
   steady-state topology readback and avoids vertex invocation for invisible
   instances, while retaining the atlas rather than drawing one universal
   maximum-resolution mesh.
5. Keep CPU readback optional and diagnostic-only. Counters can be copied from
   a small, delayed telemetry buffer without gating rendering.
6. Move PBR material binding and postprocess resources behind backend-specific
   implementations after the geometry path is equivalent.

The old JavaScript renderer's useful idea was memoizing tessellation topology
and prepared meshes. The retained atlas, versioned browser cache, and stable
batch buffers already implement that principle more completely. Its random
Poisson/Delaunay generation and per-patch regl commands are not appropriate to
reintroduce into the deterministic adaptive runtime.

Meshoptimizer opportunities and the boundary between ordinary triangle
clustering and conformal QB fitting are tracked in
[`meshoptimizer-roadmap.md`](meshoptimizer-roadmap.md).

Selection, inversion, device mappings, Rust ownership, Blender live sync, and
game-facing interaction migration are tracked in
[`focus-navigation-roadmap.md`](focus-navigation-roadmap.md).
