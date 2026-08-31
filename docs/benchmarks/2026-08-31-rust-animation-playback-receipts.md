# Rust animation playback receipts — 2026-08-31

## Outcome

Direct browser playback actions and the Leptos playback control now adapt the
complete committed clock tuple already returned by Rust: playing, unwrapped
time, speed, sequence, and revision. The browser updates its diagnostic cache
from that receipt instead of serializing the complete application after every
play, pause, or toggle.

The Leptos control samples the compact `AppAnimationSnapshot` once after its
semantic action and forwards only the clock tuple required by the renderer
adapter. Timeline and per-frame animation sampling remain on their existing
specialized paths.

## Verification

The presentation oracle requires the compact callback fields and rejects full
application snapshots from both ordinary playback functions and the Leptos
commit callback. The complete presentation smoke, presentation boundary smoke,
and inline JavaScript syntax pass. The focused native Hyperscope Web playback
test passes with the complete clock receipt; wasm32 CSR remains pending.
