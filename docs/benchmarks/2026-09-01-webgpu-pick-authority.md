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

## Semantic activation handoff

An accepted query requested by `selectionimpl=rust` may publish one bounded,
single-use activation token. The token retains the already validated
`InteractionHit` inside Rust; JavaScript receives only its decimal request ID.
New queries and interaction-target replacements invalidate it, a wrong request
cannot consume the current token, and a stale target epoch consumes and rejects
it.

After the adapter verifies that its displayed packed-node identity agrees with
the Rust join, it consumes the token as one
`InteractionAction::ActivatePrimary(hit)`. Hyperscape applies the corresponding
selection/focus navigation at one application-frame boundary. This replaces
the former browser reconstruction of three separate hover, press, and release
inputs and avoids exposing a half-applied interaction state. Platform-stale or
adapter-rejected results explicitly discard the token. If activation or view
projection fails, the existing direct Rust focus-anchor path remains the
measured rollback.

Deselecting now uses the symmetric Rust transaction. `ClearPointer` resets
hover and pressed state while `DetachFocus` removes selection ownership; the
AppStore stages both against a cloned state and integrates them at one current
virtual-frame boundary. Observers therefore cannot see detached focus with an
old activatable hover. The live sphere, focus enablement, inversion, and chart
are preserved. The former `detachFocus` plus zero-tick adapter remains available
when running against stale generated bindings.

## Static and CPU evidence

- 20 `hyperscape::interaction` tests pass, including newest-request,
  target-epoch, resolve-before-accept, bounded token, single-use, wrong-request,
  stale-token, atomic activation, and pointer-clear behavior.
- The `hyperscope-app` exact-pick test proves one semantic input produces one
  interaction activation and one routed navigation anchor after one frame.
- The AppStore selection-clear test proves pointer state and focus ownership
  clear coherently while sphere geometry, focus enablement, inversion, and
  reflection remain unchanged.
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
