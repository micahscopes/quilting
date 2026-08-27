# Browser animation-clock authority gate — 2026-08-27

## Boundary

`animclockimpl=js|shadow|rust` controls only the high-rate primary animation
clock. The existing renderer still owns pose evaluation and GPU submission.
Ordinary model and clip installation seed Rust once from the incumbent clip;
thereafter frame events advance the clocks independently. Rust maps its
unwrapped scene time into the renderer-local clip range before comparison or
authority application.

The default remains `js`. Shadow mode cannot change playback time, speed, or
pose. Rust mode consumes the fixed three-`f64` WASM packet and retains the
browser clock as an explicit fallback.

## Live Chrome evidence

On the ordinary animated horse with clip 0:

- Shadow mode completed 416 frame/seek comparisons with zero mismatches,
  fallbacks, or errors. Maximum wrapped-time/speed error was
  `9.447997939560082e-14`.
- The shadow gate paused playback and scrubbed to clip time `0.75`; Rust
  returned `[playing=0, time=0.75, speed=1]` exactly.
- Rust mode completed 432 authoritative sample writes with zero fallback,
  application mismatch, frame error, renderer error, or clock error.
- The Rust gate paused playback and scrubbed to clip time `0.375`; the browser
  slider and Rust sample both reported `0.375` exactly.
- The canonical URL retained the explicit non-default implementation and
  play/pause state. Chrome reported no warnings or errors.

The 80-control Rust route gate, 73 all-feature application tests, generated
WASM application smoke, and eight-cue presentation smoke passed. The temporary
horse tab was closed without selecting or modifying the user's chess tab.

## Remaining cutover work

This gate proves forward playback, multiple loop crossings, pause, and seek for
one ordinary clip. Reverse speed, presentation cue switching under an active
clock, background-throttled cadence, and a representative long clip remain
required before changing the default from `js`.
