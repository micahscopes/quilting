# Rust asset-completion boundary — 2026-08-31

## Outcome

The typed asset handshake now continues through decode completion.
`AppStore::complete_asset_load` admits the exact completion, rejects stale work
as before, validates the reducer's effect contract, and exposes an optional
typed primary-install job. Secondary, failed, and stale completions carry no
install authorization.

The ordinary WASM/browser path uses `finishAssetLoaded`,
`finishAssetLoadedWithMetadata`, and `finishAssetFailed`. The platform host's
`beginInstall` accepts the typed job identity directly; it no longer parses or
filters a generic completion commit.

## Platform boundary

Rust owns request ordering, fetch choice, cancellation, completion admission,
stale-result rejection, and whether primary installation may begin. The
browser owns `AbortController`, bytes, decoding, backend upload, and the
serialized resource fence. It must return the same request/asset identity when
installation succeeds or fails.

The old `completeAssetLoaded`, `completeAssetLoadedWithMetadata`, and
`completeAssetFailed` methods remain as rollback/generated-WASM seams. The
`assetimpl` switch and existing default are unchanged.

## Lightweight verification

- The ten platform-host tests pass, including stale queued decode/install work
  and both request-stage cancellation kinds.
- `node scripts/smoke-asset-completion-boundary.mjs` requires the typed
  AppStore/WASM/browser path, rejects all generic effect interpretation in the
  asset platform host, and parses both JavaScript modules.

Focused native/wasm32 and live-browser gates remain deferred until system
headroom permits. This cut invokes neither Trunk nor Binaryen.
