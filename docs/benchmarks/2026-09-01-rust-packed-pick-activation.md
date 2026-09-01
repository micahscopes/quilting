# Rust packed-pick activation

Date: 2026-09-01

## Outcome

`selectionimpl=rust` now enters the Hyperscape interaction reducer for ordinary
WebGL/JavaScript picks as well as token-checked WebGPU picks. This closes the
previous default-route gap where `pickimpl=js` resolved a face but the browser
then bypassed interaction semantics and called Rust focus anchoring directly.

The browser adapter supplies only platform/renderer observations:

- the current Rust-owned interaction-target epoch;
- a transient packed node;
- source-chart pivot and displayed-chart distance; and
- an optional exact source face and barycentric coordinates.

`HyperscopeAppShadow::activatePackedInteraction` validates that packet, joins
the packed node through the current `InteractionTargetTable`, and passes the
resolved stable `InteractionHit` into the same verified
`ActivatePrimary(hit)` helper used by accepted WebGPU tokens. One zero-time
application-frame integration produces the selected focus/navigation snapshot.
The adapter verifies stable identity and source focus geometry before applying
that snapshot to the browser view and resident renderer.

The older three-event hover/press/release path remains under
`selectionimpl=shadow` as comparison evidence. Missing stale bindings,
target-epoch rejection, invalid geometry, or adapter projection failure use the
existing direct Rust focus-anchor rollback; direct anchoring is no longer the
ordinary Rust-selection path.

## Evidence

- The shared native Hyperscape/AppStore tests cover target-table identity join,
  exact-surface retention, atomic primary activation, and routed focus
  selection.
- `quilting-wasm` compiles for `wasm32-unknown-unknown` with
  `leptos-ui,webgpu-backend`.
- `node scripts/smoke-backend-pick-shadow.mjs` checks the packed authority
  boundary, shared activation helper, default Rust-selection route, retained
  shadow oracle, rollback, and inline-module syntax.
- No server, browser, native adapter, or WebGPU device was started.
