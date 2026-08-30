# Rust focus-postprocess policy boundary — 2026-08-30

## Outcome

Focus-aware image composition now has one backend-neutral, reducer-validated
Rust policy. The value is admitted by the Rust route registry, recorded by app
replay 0.24, published through the render FRP projection, serialized by the
WASM route/app adapters, and consumed by the incumbent browser renderer
adapter.

Hyperscape remains authoritative for the geometric spheroidal focus field.
`FocusPostprocessSettings` owns only renderer policy: mode, blur extent and
strength, retained non-spheroidal focus coordinates, normalization, and the
Gaussian/Kawase pass configuration. In spheroidal mode, renderer extraction
must resolve the effective coordinate, aperture, and enable state from the
committed Hyperscape navigation snapshot.

The defaults preserve the established browser behavior: conformal-stretch
mode, disabled, 11-pixel radius, 3.0 strength, one Gaussian pass, three Kawase
passes, and 1.5 Kawase offset.

## Compatibility and rollback

- App replay is versioned from 0.23 to 0.24. Missing policy fields deserialize
  to the exact old default; a non-default policy is rejected under old replay
  schemas instead of being reinterpreted.
- `renderstateimpl=js|shadow|rust` remains available.
- The HTML controls and WebGL2 postprocess remain intact. Rust authority writes
  the same signal packet through a thin platform adapter.
- No browser tab was opened, reloaded, or otherwise disturbed during this
  change.

## Verification

- `cargo test -p hyperscope-app --all-features`: 114 passed.
- `cargo test -p hyperscope-web --all-features`: 38 library and 3 binary tests
  passed.
- `cargo check --target wasm32-unknown-unknown -p quilting-wasm --features leptos-ui,webgpu-backend --tests`: passed.
- Strict no-dependency Clippy passed for the Hyperscope app, Hyperscope web
  library, and wasm32 Quilting WASM library.
- The inline Hyperscope ES module passed `node --input-type=module --check`.

## Next measured cut

The Rust Leptos render island can now expose this policy without inventing new
state. WebGPU focus work should consume a resolved render packet derived from
this policy plus the Hyperscape frame snapshot, then add composition evidence
before it is allowed to replace WebGL2 for focus-enabled frames.
