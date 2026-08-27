# Selected-identity route gate — 2026-08-27

## Invariant

An object selection route names one stable asset-scoped entity identity. It
never serializes the transient face, packed-node, draw-batch, or pick-buffer
index used by the current renderer. `selasset` and `selentity` are therefore
an atomic pair: a partial or malformed pair is diagnostic, and an unresolved
or ambiguous pair cannot guess a selection.

The browser may retain a valid request across asynchronous model loading, but
it resolves the pair only after packed node identities and source bounds exist.
Clearing selection removes both keys while preserving the independent focus
sphere and camera target policy.

## Evidence

- `hyperscope-app` has 79 canonical control specifications. `selasset` and
  `selentity` are validated optional UUIDs in stable adjacent order, and a
  route containing only one receives an `invalid_value` diagnostic.
- 71 all-feature application tests passed, including the atomic-pair and
  canonical-order cases.
- The generated-WASM route smoke passed with all 79 Rust/browser defaults and
  serialization positions covered.
- Live Chrome selected the ordinary paused horse and wrote asset
  `f0000000-0000-4000-8000-000000000001` with entity
  `eeeeeeee-0000-4000-8000-000000000001` to the canonical URL.
- Reloading that exact URL restored node 0 after one `primary-model` resolution
  attempt. The application selected-focus identity and source bound matched;
  route fallback, route mismatch, selection mismatch, and renderer error counts
  remained zero.
- Pressing Escape cleared both URL keys atomically. The application selection
  became null, the focus sphere remained unanchored, application mismatches
  remained empty, and Chrome reported no warnings or errors.

The temporary acceptance tab was closed without selecting or modifying the
user's chess tab.
