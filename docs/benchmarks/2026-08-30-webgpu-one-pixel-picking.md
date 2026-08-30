# WebGPU one-pixel prepared-patch picking

Date: 2026-08-30

## Outcome

The staged WebGPU backend now has a retained prepared-patch query pass. It
reuses the ordinary frame table, posed prepared records, compacted visibility
stream, packed barycentric atlas, and indexed-indirect arguments. Picking does
not publish a second scene or recover identity from a consolidated draw key.
The shader returns the semantic node retained in each prepared patch, the
source face, source-face barycentrics after permutation and dyadic-leaf
restriction, the evaluated source-chart surface point, and camera distance in
the displayed chart.

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
visible prepared-patch triangle draws, so this is a latency and memory-transfer
improvement over a full ID framebuffer, not vertex-work elimination.

`PatchPickRequest` carries the current `InteractionTargetTable` residency
epoch. `StagedPatchPickReadback` preserves that epoch across asynchronous
mapping. The browser adapter exposes the opt-in
`mr_queryWebGpuPatch(x, y, targetEpoch)` function, but does not mutate hover or
selection. Its packet maps directly to
`HyperscopeAppShadow.setPackedInteractionHover`; that existing Rust method
rejects stale epochs and joins the transient packed node to current stable
asset/entity identity and source bounds before semantic dispatch.

The pick shader module participates in the existing device-local functional
shader memo. The pipeline and 1x1 target are retained by the WASM backend; one
512-byte staging buffer is currently allocated per explicit query.

## Verification

- all 27 `quilting-shaders` library tests pass, including Naga validation,
  entry-point/resource reflection, standalone WGSL emission, and reparse of the
  pick shader;
- all 26 `quilting-webgpu` library tests pass, including no-hit, exact packet,
  malformed packet, epoch, bounds, and source-surface decode cases;
- the existing native WebGPU conformance matrix now queries a pixel first
  proven covered by the ordinary prepared-patch render and asserts semantic
  node, source face, epoch, barycentric normalization, and positive distance;
- this shell compiled and started that native test, but its hardware section
  explicitly skipped because no native Vulkan/GL adapter was available;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` and tests passes; and
- no Trunk server, `wasm-pack`, `wasm-opt`, or browser process was launched.

All Cargo gates used one low-priority job. The unsupported native
`quilting-wasm` test configuration still reaches pre-existing wasm-only CSR and
WebGL symbols on a host target; it is not evidence against the passing wasm32
adapter build.

## Remaining boundary

This is not live picking authority. The browser does not call the new query by
default, and WebGL2 remains the incumbent picker. The adapter deliberately
rejects a last frame rendered through resident roots: that path combines a
source-face-indexed root domain with a sparse adaptive overlay, while the first
query pass consumes the coherent ordinary prepared-patch scene. Reading the
latter after a resident-root frame would be stale and is therefore forbidden.

The next renderer slice is one shared 1x1 depth/packet target populated in
visible order by both the resident-root triangle buckets and the adaptive
overlay triangle draws. After root/overlay native and live browser parity, the
opt-in packet can enter the existing Rust packed-target join under a shadow
gate. Only representative correctness and latency evidence should promote it
over WebGL2.

Further measured follow-ups are to encode a requested pick in the same frame
submission, replace per-query staging allocation with a bounded in-flight
ring, and throttle hover queries without weakening click precision or epoch
rejection.
