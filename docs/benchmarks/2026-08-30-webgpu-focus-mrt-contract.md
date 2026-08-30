# WebGPU focus MRT contract — 2026-08-30

## Outcome

The WebGPU patch-frame record is now a 256-byte Rust-owned device record. Its
new final `vec4` carries the resolved source-space focus sphere; the existing
selection flags carry the focus-field enable bit without adding another
binding. CPU packing rejects non-finite spheres and non-positive radii.

The shared prepared-patch WGSL module now exposes an additive
`render_patch_pbr_focus` entry point with two outputs:

1. the existing textured PBR scene color; and
2. `(Möbius stretch, logarithmic view depth, normalized S3 geodesic radius,
   1)` in an RGBA16F-compatible payload.

The formulas match the incumbent WebGL2 PBR MRT. The ordinary one-target PBR
entry point is unchanged, and WebGPU focus capability remains disabled until
the dedicated pipeline renders into the retained target and composition has
image evidence.

## Verification

- Naga validates the ordinary and focus PBR entry points and the exact
  256-byte `PatchRenderFrame` layout.
- The resident-root shader also validates against the enlarged shared frame.
- Rust packing tests cover focus enablement and all four sphere components.
- `quilting-shaders` (25 tests) and `quilting-webgpu` (9 tests): passed.
- Strict crate-local Clippy for both changed crates: passed.
- `quilting-wasm` with `leptos-ui,webgpu-backend` on
  `wasm32-unknown-unknown`: passed.
- No browser tab or user-run server was touched.

## Next measured cut

Create the dedicated two-attachment PBR pipeline, encode one complete
offscreen focus frame into the retained target, and compare its output with
the WebGL2 reference before admitting the feature.
