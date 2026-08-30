# Rust animation timeline — 2026-08-30

## Decision

`animclockimpl=rust` now mounts the animation timeline as a Leptos CSR island
over a dedicated `AppAnimationSnapshot`. The snapshot contains only revision,
clock, and the allocation-free active-clip frame record. Renderer sampling
continues through the immediate `AppFrameSnapshot`; the control view cannot
delay animation or pose evaluation.

The timeline dispatches an authored clip-time seek directly through
`AppStore`. Rust validates that playback is paused, an active clip exists, and
the sample lies in its authored range before converting it to the clock's
relative time. JavaScript receives only committed sample time, sequence, and
revision for renderer adaptation. The HTML control remains the `js|shadow`
and mount-failure rollback.

## Cadence and traffic

The existing 50-ms UI cadence calls one zero-argument
`flushAnimationReadModel()` boundary. The compact snapshot and FRP notification
remain inside Rust/WASM and Leptos updates the DOM; no camera, asset,
diagnostic, scene, or full-summary object crosses into JavaScript. A paused
seek crosses three scalar callback values only when the user edits the
timeline.

The reducer test proves that a frame advances immediate clock state while the
animation signal, summary, and navigation projections remain unchanged; the
dedicated flush updates only animation. View tests prove authored range
projection, paused seeking, and atomic rejection of playing, out-of-range, and
non-finite edits.

## Gates

- all 82 `hyperscope-app` tests;
- all 41 `hyperscope-web` library/binary tests;
- strict native and wasm32 `hyperscope-web` Clippy;
- exact `quilting-wasm` wasm32 test compilation with
  `leptos-ui,webgpu-backend`;
- presentation source oracle and inline module syntax check.

Live Chrome interaction remains a separate promotion gate because the active
browser tab is reserved by the user.
