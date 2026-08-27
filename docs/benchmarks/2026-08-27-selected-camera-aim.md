# Selected-camera aim authority gate — 2026-08-27

## Scope

Moving the finite camera control target to the selected object's output-chart
pivot while preserving orientation, lens, and control distance. The migration
keeps `navimpl=js|shadow|rust`: JavaScript is the incumbent, shadow compares
the Rust application transition per frame, and Rust consumes committed camera
snapshots.

## Deterministic gates

- `cargo test -p hyperscape -p hyperscope-app --all-features`: 124 Hyperscape
  and 69 application/replay tests passed.
- Replay schema `hyperscope-app-replay/0.16` rejects the new action in 0.15
  without reinterpreting the old trace.
- The generated release WASM smoke covers the target-orbit midpoint, endpoint,
  cadence parity between both WASM facades, and atomic reflection-pole
  rejection.
- Hacker-night, navigation, and orchestration replay fingerprints all passed.

## Live Chrome evidence

Chrome DevTools MCP observed an isolated paused horse scene on the user-run
release server.

- Shadow: 1 dispatch, 43 transition-frame comparisons, 0 mismatches, 0
  fallbacks, maximum absolute error `9.094947017729282e-12`.
- After the camera-arbitration repair, the nontrivial select/shift-pan/Object
  sequence was repeated in shadow: the selected focus survived, the pointer
  packet matched, and the resulting aim comparison had 0 mismatches with
  `2.220446049250313e-16` maximum error.
- Rust: the object was selected, then the real pointer shift-pan path moved the
  camera off its pivot before Object mode was entered. The selected focus
  survived the pan. Rust completed the return in 45 authority writes with 0
  fallbacks and no navigation diagnostics. Orientation and a control distance
  of 3 were preserved; the independently measured final eye-to-target distance
  was `3.0000000000000013`.

## Adapter defect found by the gate

Pointer `mousedown` previously resynchronized the complete navigation state,
discarding an already mapped selected-focus anchor. Idle SpaceMouse sampling
also marked the Rust camera stale every frame, forcing that resynchronization.
The adapter now retains synchronized Rust camera authority across idle HID
frames and only performs the full boundary synchronization when browser state
was genuinely invalidated. Actual browser-side camera mutations retain their
explicit invalidation sites.

The original chess tab was not selected or modified during this gate.
