# Atomic Rust startup route

Date: 2026-09-01

`HyperscopeRoute::startup_settings` now resolves every required startup domain
as one `RouteStartupSettings` transaction. Optional selection and animation
clock intent live inside that packet. A diagnostic anywhere in the route
suppresses the complete packet, so an adapter cannot install a Rust camera
while silently recovering renderer, asset, or presentation defaults in
JavaScript.

The WASM route bridge retains focused projections for parity diagnostics, but
exports `startupSettings` as the sole startup-admission fact. The browser keeps
one `initRustStartupSettings` value, passes it to `applyParams`, and reads later
presentation, render, Patch Lab, primary-asset, and animation initialization
from that same value. Generic query decoding remains the explicit JavaScript
rollback and recorded Rust-failure path.

Verification:

- `cargo test -p hyperscope-app`: 144 passed.
- `cargo clippy -p hyperscope-app --lib --no-deps -- -D warnings`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- Development WASM built with `HYPERSCOPE_WASM_OPT=0`.
- Typed renderer-resource, primary-asset, presentation, and Patch Lab route
  smokes passed against the generated WASM.
- The broad historical route smoke passed after its stale synchronous-pick
  source assertion was updated to require the current awaited pick result,
  retained activation request, and exact surface payload path.

No browser, renderer, GPU context, server, or user-owned process was started.
