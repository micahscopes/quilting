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
3. Sample retained mouse/SpaceMouse state once and update the camera. Camera
   matrices reuse preallocated typed arrays.
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

The CPU still issues one preparation dispatch and one or more draw calls per
resident `(material, node, canonical LOD, parity)` bucket. S3 permutation is a
per-instance value: the six permutations do not own six meshes or six draw
buckets. Odd and even permutations remain separate only because raster winding
is fixed per WebGL draw.

## Asynchronous LOD path

LOD selection runs on a dedicated worker with its own WebGL2 context:

1. The worker evaluates the requested animation pose on the CPU and uploads
   only joint matrices and morph weights.
2. GPU pass 1 classifies one source face per output pixel. It computes posed
   geometry, conservative image extent, conformal interior demand, pole safety,
   and screen capacity.
3. GPU pass 2 reconciles shared edges, canonicalizes the LOD triple, and emits
   six floats per face.
4. The output is copied into staging, fenced, and polled without blocking the
   worker. Only after the fence signals is the compact topology payload read
   back and transferred to the main thread.
5. The main-thread WASM layer retains the last valid topology for invisible
   faces, re-reconciles shared edges (including exact duplicate glTF seam
   vertices), grades each triangle to a 2:1 maximum edge ratio, and compares
   memberships with retained buckets.
6. Only changed buckets repack and upload their eight-float topology records;
   unchanged GPU buffers and VAOs survive untouched.

The unavoidable current synchronization boundary is step 4: WebGL2 cannot
generate indirect draws or compact instance lists for later draws. The
readback is 24 bytes per source face per completed classification, not a
readback of animated vertices or tessellated geometry. Current-pose visibility
does not cross that boundary at all.

An off-camera face intentionally retains its last crack-free topology. Making
it coarser is safe only when a conservative conformal bound proves that the
entire patch, including pole-driven ballooning between its corners, remains
outside a guarded frustum. Visibility of the three source vertices alone is
not such a proof.

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
