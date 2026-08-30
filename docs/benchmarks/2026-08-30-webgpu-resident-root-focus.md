# Device-resident root focus composition — 2026-08-30

## Outcome

Focus-enabled WebGPU PBR no longer requires the CPU-authored per-LoD batch
renderer. `ResidentRootRenderPipeline` now retains a dedicated two-attachment
PBR variant over the source-face-indexed root path. Packed resident LoD,
visibility rejection, atlas/parity bucketing, indirect draws, raw focus MRT,
and focus composition remain ordered in one command encoder.

The ordinary root API still uses its one-attachment pipelines. The focus API
requires a matching postprocess packet, rejects the still-unimplemented
source-face highlight composite, and returns the same logical scene evidence
plus focus-pass counts. Static live-presentation capability remains unchanged.

The native conformance fixture now constructs exact PBR/IBL root bindings and,
when an adapter exists, validates a complete resident-root focus submission,
nonempty raw-field coverage, and the expected draw/pass counts.

## Verification

- Naga resident-root focus entry point and shared device layouts: passed.
- Native WebGPU conformance test binary: compiled.
- `quilting-shaders` (25), `quilting-webgpu` (10), and native WebGPU (2)
  tests: passed; this host used the established no-adapter skip.
- Strict crate-local Clippy: passed.
- `quilting-wasm` with `leptos-ui,webgpu-backend` on
  `wasm32-unknown-unknown`: passed.
- No browser tab or user-run server was touched.

## Next measured cut

Integrate the sparse adaptive overlay with the same MRT when a scene contains
dyadic leaves, then compose into a presentation surface through an offscreen
scene target. Browser image parity remains the capability gate.
