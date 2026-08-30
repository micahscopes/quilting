# Retained WebGPU focus encoder — 2026-08-30

## Outcome

`quilting-webgpu` now lowers the backend-neutral Rust focus schedule into an
actual WebGPU command encoder. A `FocusPostprocessTarget` retains the scene and
raw-field MRT attachments, half-resolution JFA ping/pong textures,
full-resolution mask and blur textures, one aligned uniform table, and every
legal source-pair bind group. Resizing is explicit; ordinary frames do not
allocate textures or bind groups.

Each focus frame performs one bounded queue upload for immutable, dynamically
offset pass records and then records the exact Rust-owned sequence of render
passes. There is no GPU readback in the production encoder. Returned
`FocusPostprocessEncoding` counts provide CPU-side evidence without changing
the command stream.

WebGPU focus capability remains disabled. The target deliberately exposes the
two PBR MRT views, but PBR has not yet been connected to them and no composed
image has been compared with WebGL2.

## Verification

- Pure lowering tests cover spheroidal JFA bypass, conformal JFA parity, legal
  retained source bindings, final-output routing, and aligned uniform offsets.
- `cargo test -p quilting-webgpu focus_postprocess --lib`: 2 passed.
- `cargo test -p quilting-webgpu --test native_lod --no-run`: passed.
- Strict crate-local Clippy with only the established `chunks_exact` allowance:
  passed.
- `cargo check --target wasm32-unknown-unknown -p quilting-wasm --features
  leptos-ui,webgpu-backend --tests`: passed.
- No browser tab or user-run server was touched.

The direct package-only WASM check remains inapplicable without the optional
`browser-surface` dependency: existing `pbr_resources` browser APIs name
`web_sys::ImageBitmap`. The repository's real consumer gate is the
`quilting-wasm` `leptos-ui,webgpu-backend` build and is run separately.

## Next measured cut

Extend the shared patch frame with the resolved focus sphere, add an MRT PBR
entry point, and render an offscreen focus frame into the retained scene/raw
attachments before composition. Capability admission still waits for image
evidence and the resident-root/presentation paths.
