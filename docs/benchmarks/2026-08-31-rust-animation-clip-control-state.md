# Rust animation clip-control state — 2026-08-31

## Outcome

The Leptos installed-animation selector now forwards the active and pending
clip indices contained in its typed Rust request receipt. The browser builds
the minimal parity packet needed by the renderer adapter and no longer
serializes the complete application after a selector change.

Clip identity, job allocation, cancellation, stale-result rejection, and the
installed catalog remain AppStore-owned. JavaScript receives only two indices
and the exact platform job receipt required to switch renderer resources.

## Verification

The animation-clip boundary smoke requires both indices in the CSR callback,
checks their browser projection, and rejects complete application snapshots
from the callback. The complete presentation oracle and inline JavaScript
syntax also pass. The native animation-control crate compiles with its focused
playback test; the wasm32-only CSR callback remains pending.
