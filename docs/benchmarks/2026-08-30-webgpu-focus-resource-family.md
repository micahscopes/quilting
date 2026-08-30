# Retained WebGPU focus resource family

Date: 2026-08-30

## Outcome

`quilting-webgpu` now owns the complete retained resource family for its focus
render graph:

- the resident-root pipeline for fixed source patches;
- the adaptive-overlay PBR MRT pipeline;
- the JFA, firmness, Kawase, and directional-blur composer pipelines; and
- the viewport-sized intermediate target.

The WASM adapter retains one `FocusPbrRenderResources` value instead of
assembling those backend objects itself in separate headless and presentation
startup paths. It selects only the final output format and asks the WebGPU
crate to ensure the current viewport target.

## Lifecycle invariant

Pipeline-family construction is all-or-nothing. Viewport replacement allocates
and validates a candidate target before publishing it; a failure leaves the
last coherent target resident. Repeating the same extent is idempotent and
does not count as a target rebuild. This keeps resource lifetime inside the
backend while frame/focus semantics remain in the shared Rust render contract.

The native conformance path reuses this same family across direct adaptive
patch rendering, resident-root rendering, raw focus-field readback, and final
composition. It therefore exercises the lifecycle abstraction rather than a
test-only collection of independently created pipelines.

## Verification

- `cargo check -p quilting-webgpu --tests` passed;
- all three native WebGPU tests passed, including target create/reuse checks;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` passed; and
- no browser was launched, reloaded, or controlled. Interactive focus image
  parity remains required before WebGPU presentation promotion.
