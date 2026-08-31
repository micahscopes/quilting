# Rust animation pose frame clock — 2026-08-31

## Boundary

The renderer needs two distinct times for an animated pose: wrapped authored
clip time and monotonic real time used to measure pose-derived motion. Rust
already owned frame advancement, clip wrapping, pose revisions, continuity
epochs, coalescing, and stale completion. The browser nevertheless incremented
a second real-time clock and passed it back into Rust for every pose request.

`AppState` now advances that pose sample clock atomically with `AppEvent::Frame`
while transport is playing. Clip speed deliberately does not scale it. The
Rust-authority WASM port accepts only renderer clip time and samples the
committed frame clock itself; the explicit browser-clock port remains intact
for `js|shadow` comparison and rollback.

## Evidence

- Fourteen focused native animation tests pass, covering cadence partitioning,
  reverse/high-speed clips, paused transport, clip jobs, pose coalescing, and
  presentation residency.
- `scripts/smoke-animation-pose-frame-clock.mjs` freezes the Rust-authority
  bridge, inline-module syntax, the allocation-free read, and the retained
  shadow seam without requiring generated bindings.
- The WebGL2-only `quilting-wasm --features leptos-ui` wasm32 adapter check
  passes. Trunk, wasm-pack, generated bindings, and wasm-opt were not run.
