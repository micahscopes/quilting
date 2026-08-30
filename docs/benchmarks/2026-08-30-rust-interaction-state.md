# Rust interaction state — 2026-08-30

## Outcome

Hyperscape now has a backend-neutral interaction reducer above renderer-specific
picking. Adapters provide a validated `InteractionHit`; the reducer owns
timestamped hover, primary press, release, and cancel semantics.

Each hit carries one asset-scoped stable entity identity, its source-space
focus bound and pivot, displayed-chart camera distance, and optional
face/barycentric detail. Barycentrics are normalized at construction and cannot
silently name a different object. Interaction reach is the larger of a positive
minimum and a configurable multiple of the shared focus-sphere radius for the
explicit `SetProximityHover` action. `SetHover` represents an exact ray/query
result and validates geometry without applying that proximity cap.

A same-entity press/release produces an `InteractionActivation`, which enters
the existing `NavigationAction::AnchorFocus` queue in the same ordered ECS
system set. `InteractionState` retains hover and active hits only. Its snapshot
derives selected identity from `FocusNavigation::anchor`, so this work does not
create a competing selection owner or a second focus sphere.

Renderer residency now enters through `InteractionTargetTable`. Each transient
packed node is registered with its source bound and optional asset-scoped
identity in one validated replacement. A WebGL2 or WebGPU query then supplies
only the packed node, source pivot, displayed distance, and optional
face/barycentrics. Rust performs the join before constructing
`InteractionHit`; an unknown or deliberately unmapped node cannot masquerade as
semantic identity.

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
passes all 142 tests. The eight interaction tests cover barycentric validation,
focus-relative reach, ECS activation routing without duplicate selection,
cross-entity cancellation, invalid-hit atomicity, cadence invariance,
asset-scoped packed-node resolution, and atomic rejection of duplicate,
unknown, unmapped, or non-finite target samples.

`hyperscope-app` now retains the controller beside navigation and integrates it
first on each virtual frame. Replay schema 0.25 records the complete semantic
interaction vocabulary and its read model: asset/entity identity, source bound
and pivot, displayed distance, optional face/barycentrics, hover, active press,
derived selection, sequence fence, and diagnostics. Schema 0.24 remains
readable but rejects interaction events rather than silently reinterpreting
them.

The replay fixture proves that hover/press/release selects through the existing
navigation authority on the same application frame, retains exact surface
identity, round-trips through JSON, produces a stable golden fingerprint, and
is invariant to frame partitioning. A nil asset identity is rejected without a
partial selection. The full replay-enabled native gate passes all 117 library
tests.

`HyperscopeAppShadow` exposes validated hover, clear, press, release, cancel,
and snapshot methods. The route source smoke fences their delegation through
`SemanticAction::Interact`; the exact `leptos-ui,webgpu-backend` WASM feature
set passes `cargo check --target wasm32-unknown-unknown`.

The facade also owns the backend-local target table outside `AppState` and
outside JavaScript semantic state. Source-bound refresh replaces it atomically;
the shadow pick path submits `setPackedInteractionHover` when available. The
old identity-explicit method remains as the stale-artifact/rollback path.

The first page adapter is intentionally shadow-only. The retained WebGL2 pick
returns its exact face/barycentric coordinate plus on-demand animated QB points
in the source and displayed conformal charts. Under `selectionimpl=shadow`,
mouse and view-center picks enter hover/press/release, compare the selected and
hovered stable identity, and publish
`globalThis.__hyperscopeInteractionDiagnostics`. Missing generated methods,
invalid packets, or comparison failures retain the incumbent direct
`AnchorFocus` observer. The default `selectionimpl=rust` path is unchanged.

The command used one low-priority Cargo job and did not invoke Trunk,
`wasm-pack`, or `wasm-opt`.

## Remaining boundary

This is semantic-core evidence, not live browser authority. WebGL2/WebGPU ray
or shape queries beyond this first WebGL2 click path still need thin adapters;
they now share the packed-target resolver rather than owning separate identity
joins. Visualization such as hover/selection tint remains presentation policy.
Representative live shadow parity and a deliberate Rust-default promotion
should precede any incumbent removal.
