# Selected-camera reframe authority — 2026-08-27

## Contract

`NavigationAction::ReframeSelection` owns the complete semantic operation:

- retain the selected `(asset, entity)` and its source-chart bound/pivot;
- project the pivot and bound radius into the active conformal chart;
- fit the sphere in the narrower viewport axis using the current vertical FOV;
- enable an explicit semantic target without moving the initial visible pose;
- interpolate target with smootherstep and positive distance logarithmically;
- reject invalid framing or a reflection-pole projection without partial camera
  mutation.

The camera framing margin is 15%. This is independent from the focus sphere's
10% selection margin.

## Deterministic gates

- 123 `hyperscape` native tests passed, including cadence independence,
  output-chart conformal scale, target-orbit midpoint, invalid framing, missing
  selection, and reflection-pole rollback.
- 68 `hyperscope-app --all-features` tests passed with replay schema 0.15 and
  the updated navigation golden fingerprint.
- The release `wasm-pack` artifact passed
  `scripts/smoke-hyperscope-app-shadow.mjs`. Both generated WASM facades matched
  at queue, midpoint, endpoint, invalid-framing, and pole-rejection boundaries.
- Route, presentation, and surface-walk adapter smokes remained green.

The parity work exposed an incumbent defect: browser framing called
`framedSphereDistance` without the live lens, silently assuming 60 degrees.
The browser rollback path now passes `cameraFovRadians()` explicitly, matching
the camera projection and Rust model.

## Live Chrome evidence

Testing used separate paused-horse tabs on the user-run `localhost:8888`
Trunk server. The existing chess tab was not selected or modified.

| Route | Result |
| --- | --- |
| `navimpl=shadow&selectionimpl=rust` | 1 dispatch accepted; 40 per-frame comparisons; 0 mismatches; maximum error `9.094947017729282e-12`; 0 fallback writes |
| `navimpl=rust&selectionimpl=rust` | 1 dispatch accepted; 39 Rust camera writes; 0 fallback writes; final fit-distance error `0`; no console warnings/errors |

The live camera used a 75-degree vertical FOV. Its independently recomputed
final distance was `2.0305559623876825`, exactly matching the Rust snapshot.

## Rollback and remaining adapter work

`navimpl=js` retains browser authority; `navimpl=shadow` retains browser writes
and records Rust drift; `navimpl=rust` consumes the AppStore camera snapshot.
The browser still acquires the selection gesture, applies renderer/DOM
projections, and owns URL serialization. Animated source-bound production and
the remaining camera-transition initiators are separate migration slices.
