# WebGPU pick authority cutover

Date: 2026-09-01

## Outcome

`pickimpl=rust` now routes pointer and view-center queries through the retained
WebGPU frame instead of executing the WebGL picker and treating WebGPU as
diagnostic evidence. The default `pickimpl=js` rollback and the
`pickimpl=shadow` comparison lane remain available.

The cutover is deliberately asynchronous and has three observable outcomes:

- an accepted WebGPU result is joined in Rust from the renderer-local packed
  node to stable scene identity and a current animated face/barycentric surface
  address;
- a superseded request, replaced interaction-target epoch, or intervening
  manual camera input is ignored without changing selection or anchoring; and
- staging, device/readback, payload-validation, or browser identity-join
  failure falls back to the incumbent WebGL2 surface picker.

The browser no longer invokes the WebGL picker on the successful Rust path.
The WebGPU capture is accepted only when its retained logical renderer-call ID
and viewport equal the current renderer state. Rust owns monotonic request
ordering and the interaction-target epoch fence; JavaScript retains only the
platform-specific manual-input revision fence and the explicit rollback.

## Static and CPU evidence

- 15 `hyperscape::interaction` tests pass, including newest-request,
  target-epoch, error, and resolve-before-accept behavior.
- `quilting-wasm` checks for `wasm32-unknown-unknown` with
  `leptos-ui,webgpu-backend`.
- 32 `hyperscope-app` route/settings tests pass.
- `node scripts/smoke-backend-pick-shadow.mjs` checks both the retained shadow
  lane and the authority lane, their Rust/WASM/browser ownership boundaries,
  the forbidden retired shuttles, and inline-module syntax.
- `git diff --check` passes.

Strict workspace Clippy still reaches pre-existing lint debt in dependency
crates before the adapter crate; no unrelated lint cleanup is included here.

## Deferred live evidence

No browser, server, native GPU test, adapter probe, or WebGPU device was
started for this cutover. Another project is intentionally running heavy
WebGPU tests and may kill shared browser contexts. Live validation is therefore
an explicit follow-up after that contention clears:

1. retained ordinary, resident-root, and sparse-overlay picks;
2. animated surface-address stability;
3. overlapping pointer requests and manual-camera supersession;
4. no-hit behavior without a hidden WebGL pick;
5. forced device loss followed by the documented WebGL2 rollback; and
6. latency/image parity evidence before considering a default-route change.
