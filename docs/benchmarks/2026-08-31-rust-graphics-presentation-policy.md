# Rust graphics-presentation policy cut

The browser previously interpreted renderer residency independently in three
places: canvas presentation, WebGPU LOD recovery, and device-LOD authority.
That made fallback behavior a browser-owned semantic decision and allowed the
three predicates to drift.

`hyperscope-app::resolve_graphics_presentation` now consumes backend-neutral
residency facts and returns one coherent decision:

- presentation phase and requested-style capability;
- whether the current admitted frame may present through WebGPU;
- whether a complete device LOD epoch may prewarm behind WebGL2;
- whether device LOD may replace the incumbent.

The browser still owns DOM class toggles and platform error text. The route
control `gfxpresentimpl=js|shadow|rust` retains explicit rollback; `rust` is the
default after live shadow parity. The browser oracle remains instrumented and
bounded rather than deleted.

Focus post-processing is an explicit capability input. The current WebGPU
backend composes it only with PBR, so diagnostic styles now report
`unsupported-mode` immediately instead of waiting forever for a frame the
renderer correctly refuses. A supported PBR/Matcap transition without focus is
staged against the retained frame so style-independent device LOD authority is
not retired between the control event and the next render frame.

## Evidence

- `cargo test -p hyperscope-app`: 114 passed.
- `cargo check -p quilting-wasm --tests --target wasm32-unknown-unknown
  --features leptos-ui,webgpu-backend`: passed.
- Rust/browser live policy comparison: 418 decisions, zero mismatches, zero
  adapter errors.
- Live focus composition audit:
  - focused PBR presented through WebGPU;
  - focused Matcap fell back as `unsupported-mode`;
  - Matcap without focus presented through WebGPU;
  - restoring focused PBR recovered WebGPU with no frame failures.
- Live unsupported-debug recovery audit returned from `fz-weight` to PBR and
  recovered current-frame/device-LOD authority.
