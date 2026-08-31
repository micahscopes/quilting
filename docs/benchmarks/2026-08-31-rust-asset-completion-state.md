# Rust asset-completion state — 2026-08-31

## Outcome

The typed asset-completion receipt now carries the compact read model for the
completed asset alongside any primary-scene install job. Rust samples that
asset after admitting or rejecting the completion, so stale completions expose
the current authoritative state rather than reconstructing it in JavaScript.

The browser consumes this projection for ready-state, byte-length, and
attribution checks. It updates its diagnostic/fallback credit cache one asset at
a time and no longer crosses WASM to serialize the complete application after
successful or failed acquisition.

The full application snapshot and typed receipt share one `ShadowAsset`
conversion, removing another parallel serialization path.

## Verification

The application test now checks the exact ready asset returned with a decoded
primary load. The asset-completion source smoke requires the AppStore, WASM,
and browser fields, and rejects full application snapshots from both ordinary
success and failure adapters. JavaScript syntax and neighboring typed boundary
smokes pass. Native and wasm32 reruns remain deferred while an unrelated
release build occupies the machine.
