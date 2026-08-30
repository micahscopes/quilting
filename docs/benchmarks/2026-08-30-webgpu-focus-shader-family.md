# WebGPU focus shader family — 2026-08-30

## Outcome

`quilting-shaders` now owns one validated WGSL module for the WebGPU focus
pipeline. Its entry points implement raw MRT weight selection, JFA seed
initialization, nearest-seed propagation, focus firmness, Kawase mask
smoothing, and directional Gaussian composition. The formulas mirror the
incumbent WebGL2 implementation, including the direct spheroidal field,
rational circle-of-confusion response, 48-pixel shader tap cap, and final-only
sharp/blur crossfade.

`quilting-webgpu::FocusPostprocessPipelines` compiles those entry points on the
real wgpu device behind one compatible bind-group layout. Float16
intermediates and separate RGBA8/output-format blur pipelines make the next
resource-residency cut explicit without putting wgpu handles in the shared
Rust render contract.

This does **not** admit focus-enabled WebGPU frames yet. Texture residency,
PBR MRT output, schedule lowering, and image comparison remain required, so
the current capability predicate continues to fall back to WebGL2.

## Verification

- Naga validates all seven entry points and the exact 64-byte uniform layout.
- The pinned Naga WGSL writer preserves every public entry-point name.
- `cargo test -p quilting-shaders compile_focus_postprocess_shader_family --lib` passed.
- `cargo test -p quilting-webgpu --test native_lod --no-run` passed.
- `cargo test -p quilting-webgpu --test native_lod`: 2 passed. This host had no
  suitable graphics adapter, so the established optional-adapter branch
  skipped device creation; browser/device evidence is still required.
- No browser tab or user-run server was touched.

## Next measured cut

Allocate retained full-resolution and half-resolution focus textures, upload
one aligned immutable uniform record per scheduled pass, and encode an
offscreen focus frame without readback. Diagnostic readback remains a separate
explicit evidence operation.
