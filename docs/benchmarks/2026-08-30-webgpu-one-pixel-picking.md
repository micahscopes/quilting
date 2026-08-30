# WebGPU one-pixel prepared-patch picking

Date: 2026-08-30

## Outcome

The staged WebGPU backend now has retained prepared-patch query passes for both
ordinary frames and resident-root frames with sparse adaptive replacements.
They reuse the frame tables, posed prepared records, current compacted
visibility or resident bucket stream, packed barycentric atlas, and
indexed-indirect arguments that produced the completed frame. Picking does not
rerun LoD, visibility, preparation, or compaction; publish a second scene; or
recover identity from a consolidated draw key. The shader returns the semantic
node retained in each prepared patch, the source face, source-face
barycentrics after permutation and dyadic-leaf restriction, the evaluated
source-chart surface point, and camera distance in the displayed chart.

Resident queries reproduce visible composition order into the same one-pixel
target: current resident roots clear it, then current sparse adaptive draws
load its color and depth. Suppressed roots therefore cannot win over their
replacements, and unsuppressed roots remain queryable without expanding the
ordinary CPU-authored scene.

A full-viewport pixel is remapped in clip space onto a one-pixel render target.
The retained target contains:

- one 1x1 `Rgba32Uint` attachment for face, semantic node, and two exact f32
  barycentric bit patterns;
- one 1x1 `Rgba32Float` attachment for source position and displayed distance;
  and
- one private 1x1 `Depth24Plus` attachment.

The explicit readback buffer is 512 bytes: two 16-byte texels occupy separate
256-byte WebGPU-aligned rows. No target or readback exists at viewport scale,
and ordinary frames do not copy or map pick data. A query still executes the
visible prepared-patch triangle draws (resident buckets plus an optional sparse
overlay for resident frames), so this is a latency and memory-transfer
improvement over a full ID framebuffer, not vertex-work elimination.

`PatchPickRequest` carries the current `InteractionTargetTable` residency
epoch. `StagedPatchPickReadback` preserves that epoch across asynchronous
mapping. The WASM evidence adapter runs the incumbent WebGL2 surface pick once,
returns that result synchronously, and stages a retained WebGPU query against
the same renderer call. Its asynchronous readback is diagnostic only. Rust
rejects a changed retained-frame revision, source render call, or echoed target
epoch before comparing the samples. The browser additionally discards a report
if the current interaction-target epoch changed while the readback was in
flight.

The backend-neutral comparison packet distinguishes coverage and node/face
identity mismatches from numeric drift. When both paths hit, it records maximum
source barycentric error, source-position error, and displayed-distance error;
a miss cannot masquerade as zero numeric error.

The pick shader module participates in the existing device-local functional
shader memo. The pipeline and 1x1 target are retained by the WASM backend; one
512-byte staging buffer is currently allocated per explicit query. The WASM
adapter admits at most one staged or reading comparison at a time, so repeated
clicks cannot create an unbounded readback queue.

## Browser route and rollback

`pickimpl=js` is the default and exact rollback. It does not initialize WebGPU
solely for picking and follows the incumbent `mr_pickSurface` path.

`pickimpl=shadow` initializes the offscreen WebGPU backend even when
`gfx=webgl2`, but the visible canvas, selected surface, highlight, and semantic
interaction remain driven by the synchronous WebGL2 result. `pickimpl=rust` is
currently treated as the same non-authoritative measurement lane; spelling the
future mode in a URL cannot bypass the promotion gates.

Bounded browser telemetry is available at
`globalThis.__hyperscopeBackendPickDiagnostics`. It retains scalar counters,
maxima, the last report, and the last error rather than accumulating samples.
It records requests, stage rejections, readbacks, stale target epochs,
coverage/identity mismatches, numeric maxima, and stage/readback/total latency
maxima. Warnings are emitted only on power-of-two mismatch/error counts.

## Verification

- all 28 `quilting-shaders` library tests pass, including Naga validation,
  entry-point/resource reflection, standalone WGSL emission, and reparse of the
  ordinary and resident-root pick shaders;
- all 26 `quilting-webgpu` library tests pass, including no-hit, exact packet,
  malformed packet, epoch, bounds, and source-surface decode cases;
- the native WebGPU conformance matrix queries a pixel first proven covered by
  the ordinary prepared-patch render, then separately queries pixels proven to
  come from a suppressed root's dyadic replacement and an unsuppressed
  resident root. Each asserts semantic node, source face, epoch, barycentric
  normalization, and positive distance;
- this shell compiled and started that native test, but its hardware section
  explicitly skipped because no native Vulkan/GL adapter was available;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` and tests passes; and
- no Trunk server, `wasm-pack`, `wasm-opt`, or browser process was launched.

After browser routing was added, the exact WASM check passed again in 8.09
seconds. All 22 `hyperscope-app` settings tests passed, including the opt-in
route/default/rollback contract, and the inline browser module passed a
JavaScript syntax check. The full source smoke reaches the expected stale
generated-package boundary because `pkg/quilting_wasm.js` predates the new
`pickimpl` registry entry. Regenerating that package was deliberately deferred
rather than invoking the user-disabled Trunk/`wasm-pack`/`wasm-opt` path.

All Cargo gates used one low-priority job. The unsupported native
`quilting-wasm` test configuration still reaches pre-existing wasm-only CSR and
WebGL symbols on a host target; it is not evidence against the passing wasm32
adapter build.

## Remaining boundary

This is not live picking authority. The browser does not call the comparison
path by default, and WebGL2 remains the incumbent picker in every route mode.
The opt-in adapter selects the exact last completed ordinary, resident, or
resident-focus binding epoch and refuses overlapping evidence. Only
representative browser correctness and latency evidence should promote it over
WebGL2.

Promotion requires a non-optimized development package and a live browser
matrix covering ordinary, resident-root, adaptive-overlay, focus, animated,
conformally transformed, miss, device-loss, and target-replacement cases. It
must show zero coverage and identity mismatches, establish scene-relative
numeric and latency budgets, and preserve click precision and epoch rejection.
Further measured work is to encode a requested pick in the same frame
submission, replace per-query staging allocation with a bounded in-flight
ring, and throttle future hover queries without weakening exact clicks.
