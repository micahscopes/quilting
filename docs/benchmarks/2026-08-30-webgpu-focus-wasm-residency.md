# WebGPU focus residency in the live WASM backend

Date: 2026-08-30

## Outcome

The browser WebGPU adapter can now retain every resource needed to execute the
shared Rust `RenderFrame` focus packet without CPU image staging:

- a fixed `Rgba8Unorm` + `Rgba16Float` focus MRT pipeline for resident roots;
- a matching focus-only adaptive overlay pipeline and scene-epoch bindings;
- an output-format-specific focus composer for either the parity target or the
  selected presentation surface; and
- a lazily allocated focus target that is rebuilt only when the viewport size
  changes.

The ordinary resident-root pipeline and the focus bundle are separate
publication domains. A focus pipeline, binding, overlay, or target failure
increments explicit focus diagnostics and leaves ordinary WebGPU residency and
the incumbent WebGL2 renderer intact.

## Coherence boundary

Focus root bindings and sparse adaptive overlays are built from the same model,
scene revision, texture table, environment map, preparation scene, geometry
bucket scene, and root-suppression epoch. Scene, texture, and environment
replacement publish a complete focus candidate or clear focus readiness; stale
focus bindings are never paired with a newly published asset epoch.

The live submission path recognizes focus-aware PBR frames and routes them to
`render_offscreen_focus_resident_adaptive` or
`present_focus_resident_adaptive`. Both paths keep root preparation, LOD
classification, sparse replacement rendering, JFA/Kawase/Gaussian scheduling,
and final composition on the device. The one-shot offscreen evidence target is
still the only permitted readback boundary.

## Capability gate

This cut deliberately does **not** open the browser evidence or presentation
gate. The backend now has truthful diagnostics (`focusPipelineReady`,
`focusSceneReady`, `focusTargetReady`, counters, and the last focus error), but
WebGL2 remains authoritative until a focus-enabled WebGL2/WebGPU image oracle
passes. This prevents resource availability from being mistaken for semantic
parity.

## Verification

- `cargo test -p quilting-shaders --lib`: 25 passed.
- `cargo test -p quilting-webgpu --lib`: 10 passed.
- `cargo test -p quilting-webgpu --tests --no-fail-fast`: 10 unit and 3 native
  conformance tests passed (the hardware test uses its established no-adapter
  skip when necessary).
- strict `quilting-webgpu` library Clippy passed.
- exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` passed.

Whole-`quilting-wasm` strict Clippy still reports pre-existing lint debt in
unrelated modules. This cut introduces no `webgpu_backend.rs` Clippy findings.
