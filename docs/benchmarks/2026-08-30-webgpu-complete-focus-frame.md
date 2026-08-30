# Complete offscreen WebGPU focus frame — 2026-08-30

## Outcome

`quilting-webgpu` can now encode and submit a complete focus-enabled PBR frame
without leaving the device:

1. prepare QB patches;
2. expand or otherwise produce visibility;
3. compact survivors and indirect arguments;
4. draw opaque textured PBR into retained RGBA8 scene color plus RGBA16F raw
   focus field;
5. run the shared Rust-scheduled selection, JFA, firmness, Kawase, and
   directional Gaussian passes; and
6. write the composed RGBA8 result to a retained offscreen target.

`FocusPbrPatchRenderPipeline` is intentionally distinct from the ordinary
single-attachment pipeline, preventing callers from pairing the MRT shader
with the wrong render pass. `FocusPatchFrameEncoding` reports scene draw and
postprocess pass counts without reading survivor data or pixels back.

The static presentation capability predicate is still unchanged. This path is
available for controlled offscreen evidence, but focus-enabled live frames
continue to fall back to WebGL2 until image comparison and the resident-root
route are complete.

## Verification

- The native conformance fixture now constructs the retained focus resources
  and, when an adapter is available, submits the complete PBR MRT/composition
  frame under a WebGPU validation error scope and checks nonempty output.
- On this host the established optional-adapter branch skipped device
  execution; both native tests passed and the full test binary compiled.
- `quilting-webgpu` library tests (9): passed.
- Strict crate-local Clippy with only the established `chunks_exact`
  allowance: passed.
- `quilting-wasm` with `leptos-ui,webgpu-backend` on
  `wasm32-unknown-unknown`: passed.
- No browser tab or user-run server was touched.

## Next measured cut

Add deterministic offscreen image evidence for raw-field channels and final
composition, then compare the same fixture against the WebGL2 oracle. In
parallel, extend the resident-root path so focus does not force a return to
CPU-authored per-LOD batches.
