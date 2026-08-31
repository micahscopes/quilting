# Rust animation-clock authority cutover — 2026-08-31

## Outcome

`animclockimpl=rust` is the canonical route default. Explicit
`animclockimpl=js` and `animclockimpl=shadow` routes remain available as the
rollback and parity lanes. This changes only primary animation-clock authority;
JavaScript remains the platform adapter for animation evaluation, renderer
publication, and browser lifecycle events.

Rust owns playing state, signed speed, unwrapped scene time, clip-relative
wrapping, semantic sequence allocation, and the compact FRP timeline model.
One committed clock sample drives the UI, renderer pose scheduler, canonical
URL, and animated-LOD revision.

## Defect found by the gate

The shadow pose-scheduler oracle treated two absent follow-up requests as
unequal because its comparator required both values to be truthy. Identical
`{ disposition: immediate, next: null }` completions therefore produced false
mismatches. Null now compares equal only to null; source-level regression
coverage also checks unequal null/stamp pairs, equal stamps, and changed
revisions. No authoritative runtime behavior changed for this defect.

## Live evidence

`scripts/audit-rust-animation-clock.mjs` loads the repository's 75.8-second
`ant.glb` clip in an isolated Chrome target and exercises the full pipeline
under reverse playback. The explicit Rust lane and shadow lane both passed:

- reverse crossing through unwrapped time zero, with the renderer and LOD pose
  wrapping to the same value above 75 seconds;
- a real 3.2-second hidden-tab interval with zero animation frames;
- foreground resumption consuming one clamped 250-ms frame rather than the
  background wall-clock gap;
- pause, exact seek to 37.9 seconds, and reverse resume;
- zero clock, pose-scheduler, or animated-LOD mismatches and zero runtime
  errors.

The shadow lane recorded 17 clock comparisons and 43 pose-scheduler comparisons
with zero error. The explicit JavaScript rollback repeated the rendered
reverse-wrap, hidden-tab, pause, seek, and resume outcomes while every Rust
clock dispatch/comparison/authority counter remained zero.

`scripts/audit-rust-animation-presentation.mjs` then exercised the real hacker
night presentation. It advanced across four paused polytope cues into the first
playing horse cue, observed coherent Rust clock, renderer pose, and LOD
advancement, then reversed into a paused cue and proved time remained stable.
All five preflight assets remained resident; all 4,432 packed faces had LOD
residency; clock, pose, clip, application, LOD, and console error counts were
zero.

The final release gate reruns the clock audit without an implementation query
parameter, proving the canonical default selects Rust while keeping the URL
compact. Native application/web tests, strict no-dependency Clippy, WASM
compilation, route/source smokes, and diff checks accompany that live gate.
