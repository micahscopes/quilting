# Rust FRP interaction status

Date: 2026-09-01

## Ownership change

Stable selection status is now a read-only `hyperscope-web` projection over
the AppStore's throttled navigation frame. The view derives selected
asset/entity identity, source bound radius, selected pivot, and the exact
hovered source face only when that hover belongs to the selected identity.
It owns no renderer-local packed node, platform input, or mutable state.

The Leptos CSR island subscribes to `AppStore::navigation_signal`; it never
participates in frame integration and emits no actions. In
`selectionimpl=rust`, the browser mounts this island and stops synthesizing the
stable "selected"/"not selected" label from `selectedObject`. The existing
browser element remains as a separate transient-notice lane for walk/anchor
progress and platform errors. If the generated WASM binding is stale or mount
fails, that element retains the complete incumbent stable-label behavior.

This split is intentional:

- Rust/FRP owns durable-for-the-session selection meaning and geometry.
- JavaScript reports short-lived platform adaptation facts.
- The renderer still consumes its immediate frame lane and never waits for a
  Leptos publication.

## Evidence

- Two native `hyperscope-web` tests cover detached and exact-surface selected
  projections.
- The exact `quilting-wasm` WASM feature set compiles with the new CSR island.
- `node scripts/smoke-rust-interaction-status.mjs` checks the projection,
  signal subscription, WASM mount, browser fallback split, accessibility
  attributes, and inline-module syntax.
- No server, browser, GPU adapter, or WebGPU device was started.
