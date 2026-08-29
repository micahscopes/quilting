# Animation-clock cutover model gates — 2026-08-29

## Purpose

The live horse gate left four risks before `animclockimpl=rust` could become
the default: reverse playback, presentation cue switching, background cadence,
and a representative long clip. This checkpoint closes the first three at the
Rust/generated-WASM contract while deliberately making no live-browser claim.

## Exact generated-WASM evidence

`scripts/smoke-hyperscope-app-shadow.mjs` now proves:

- a clock at `0.1` seconds with speed `-1` advances by `0.25` seconds to the
  unwrapped value `-0.15` and maps into clip `[0, 2)` as `1.85`;
- one coarse three-second frame produces the same clock as the partition
  `0.25 + 0.5 + 0.75 + 1.5`, including signed speed;
- presentation start atomically installs `{ playing: true, time: 0.375,
  speed: -0.5 }`, and advance atomically replaces it with
  `{ playing: false, time: 1.25, speed: 0.25 }`;
- direct clock and seek inputs use store-allocated sequences, while a rejected
  non-finite clock changes neither application state nor the next sequence.

The complete generated application smoke, eight-cue presentation smoke,
85-control route smoke, WASM32 build check, and all 83 `hyperscope-app` tests
passed with these assertions.

## What remains unproven

Cadence partition invariance proves the application model and generated-WASM
adapter, not browser lifecycle behavior. Before changing the default, capture
live Chrome evidence for a genuinely background-throttled tab and a long clip,
including wrapped time, evaluated pose, continuity epoch, LOD pose stamp,
fallback count, and console errors. Keep `animclockimpl=js|shadow|rust` until
that evidence is clean.
