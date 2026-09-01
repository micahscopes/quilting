# Rust-typed renderer resource route

Date: 2026-09-01

Environment and matcap resource names now leave Rust route admission as one
`RouteRendererAssetSettings` packet. The browser no longer rediscovers either
name from its generic query object after successful Rust admission.

This does not move browser resources into application state. Fetching,
IndexedDB caching, cancellation, HDR decoding, procedural-environment
generation, matcap lookup, and GPU upload remain platform/renderer adapter
work. Rust owns only validation, canonical defaults, and the atomic startup
projection. The explicit JavaScript route fallback retains its old query path.

Verification:

- `cargo test -p hyperscope-app --lib`: 143 passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- `cargo clippy -p hyperscope-app --lib --no-deps -- -D warnings`: passed.
- `node scripts/smoke-rust-renderer-asset-route.mjs`: passed against the
  generated development WASM artifact.

No browser, renderer, GPU context, server, or user-owned process was started.

