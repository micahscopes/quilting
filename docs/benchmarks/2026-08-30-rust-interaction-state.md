# Rust interaction state — 2026-08-30

## Outcome

Hyperscape now has a backend-neutral interaction reducer above renderer-specific
picking. Adapters provide a validated `InteractionHit`; the reducer owns
timestamped hover, primary press, release, and cancel semantics.

Each hit carries one asset-scoped stable entity identity, its source-space
focus bound and pivot, displayed-chart camera distance, and optional
face/barycentric detail. Barycentrics are normalized at construction and cannot
silently name a different object. Interaction reach is the larger of a positive
minimum and a configurable multiple of the shared focus-sphere radius.

A same-entity press/release produces an `InteractionActivation`, which enters
the existing `NavigationAction::AnchorFocus` queue in the same ordered ECS
system set. `InteractionState` retains hover and active hits only. Its snapshot
derives selected identity from `FocusNavigation::anchor`, so this work does not
create a competing selection owner or a second focus sphere.

## Determinism and failure policy

- Actions carry monotonically allocated sequences and nondecreasing virtual
  timestamps.
- Integration is invariant to render-frame partitioning.
- Releasing over a different entity cancels activation.
- Invalid or unreachable hits do not partially replace the previous hover.
- Activation and the existing navigation reducer execute in one chained
  `HyperscapeSet::Interaction` schedule before frame mutation and extraction.
- Current-frame activations are effects, cleared on the next reducer update;
  they are not durable authored or HHHS state.

## Evidence

`nice -n 19 ionice -c 3 env CARGO_BUILD_JOBS=1 cargo test -p hyperscape --lib`
passed all 140 tests. The six interaction tests cover barycentric validation,
focus-relative reach, ECS activation routing without duplicate selection,
cross-entity cancellation, invalid-hit atomicity, and cadence invariance.

`hyperscope-app` now retains the controller beside navigation and integrates it
first on each virtual frame. Its 84-test native gate includes a three-action
hover/press/release case proving that activation becomes the existing
`AnchorFocus` action on the same application frame. The opt-in replay feature
also type-checks without claiming a new serialized pointer-action schema.

`HyperscopeAppShadow` exposes validated hover, clear, press, release, cancel,
and snapshot methods. The route source smoke fences their delegation through
`SemanticAction::Interact`; the exact `leptos-ui,webgpu-backend` WASM feature
set passes `cargo check --target wasm32-unknown-unknown`.

The command used one low-priority Cargo job and did not invoke Trunk,
`wasm-pack`, or `wasm-opt`.

## Remaining boundary

This is semantic-core evidence, not live browser authority. WebGL2/WebGPU ray
or shape queries still need thin adapters that resolve packed renderer hits to
the exact stable identity and enqueue these actions. Visualization such as
hover/selection tint remains presentation policy. A `js|shadow|rust` browser
adoption gate and representative live parity evidence should precede any
incumbent removal.
