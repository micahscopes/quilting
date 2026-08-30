# Rust animation-clip request boundary — 2026-08-30

## Outcome

Animation clip requests now have one typed application receipt.
`AppStore::request_animation_clip` commits `SelectClip` through the reducer,
then exposes the Rust-allocated sequence and commit, requested index, exact
typed selection job, exact cancellations, and whether the resulting
active-or-pending intent matches the request.

The job identity keeps typed `RequestId` and `AssetId` values until the final
WASM/CSR projection. `hyperscope-web` delegates to this application port
instead of filtering `AppCommit.effects`, and the ordinary browser adapter
consumes the typed WASM receipt instead of rediscovering reducer semantics in
JavaScript.

## Boundary and rollback

The browser still executes the physical worker switch and returns the exact
job identity through the existing success/failure completion methods. Rust
continues to own catalog validation, active/pending intent, cancellation,
job allocation, and stale completion rejection.

`animclipimpl=js|shadow|rust` is unchanged and `js` remains the default. The
generic `dispatchAnimationClip` WASM method remains available for rollback and
generated-WASM evidence; Hyperscope now uses `requestAnimationClip`.

## Verification

- The core installed-scene job test passes with typed selection, duplicate
  no-op, cancellation back to the active clip, failed completion, stale
  completion, successful completion, and exact clip-time sampling.
- The `hyperscope-web` clip-control test passes while delegating to the same
  typed application receipt.
- `node scripts/smoke-animation-clip-boundary.mjs`: passed. The zero-build
  oracle rejects generic effect filtering in both `hyperscope-web` and the
  ordinary browser request adapter, while preserving the legacy WASM seam.
- The exact `quilting-wasm` wasm32 check with `leptos-ui,webgpu-backend` and
  tests passed at one idle-priority job. Strict no-dependency Clippy passed for
  `hyperscope-app --all-features` and the `hyperscope-web` animation-control
  feature. No bindings, Trunk server, or optimizer were invoked.
