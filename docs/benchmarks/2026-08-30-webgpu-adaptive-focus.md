# Sparse adaptive WebGPU focus composition — 2026-08-30

## Outcome

The focus MRT now includes sparse adaptive dyadic replacement patches as well
as device-resident source roots. Rust encodes the sequence atomically:

1. clear and draw unsuppressed resident roots into scene color and raw field;
2. load both attachments and draw the adaptive replacements;
3. preserve the shared depth attachment across the root/overlay boundary; and
4. run focus composition once over the complete image.

`AdaptiveRenderPipelines` keeps ordinary diagnostic families distinct from
the dedicated focus PBR pipeline. Focus and raw-field attachments must be
paired, unsupported passes and source-face highlights fail explicitly, and
the suppression mask must match the retained overlay epoch before encoding.

The retained adaptive conformance fixture now builds exact root and overlay
PBR resources and, on a real adapter, validates one complete focus submission,
draw/pass counts, raw-field coverage, and final-image coverage. No topology,
visibility, or per-LoD batches cross back to the CPU.

## Verification

- Native WebGPU conformance fixture: compiled with the focus extension.
- `quilting-webgpu` library (10) and native (2) tests: passed; this host used
  the established no-adapter skip.
- Strict crate-local Clippy: passed.
- `quilting-wasm` with `leptos-ui,webgpu-backend` on
  `wasm32-unknown-unknown`: passed.
- No browser tab or user-run server was touched.

## Next measured cut

Compose the retained offscreen result into a WebGPU presentation surface, then
run the frozen WebGL2/WebGPU browser image comparison before changing the live
capability predicate.
