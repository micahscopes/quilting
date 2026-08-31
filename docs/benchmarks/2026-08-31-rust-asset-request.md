# Rust asset-request boundary — 2026-08-31

## Outcome

Ordinary asset acquisition now begins with one typed Rust receipt.
`AppStore::request_asset_load` allocates the local semantic sequence, commits
the request, validates the reducer's effect contract, and returns the exact
fetch plus asset-load and primary-install cancellations.

The WASM adapter exposes that as `requestAssetLoad`. JavaScript no longer
allocates the application sequence or filters a generic `AppCommit.effects`
array to discover cancellations. `BrowserAssetEffectHost` receives typed jobs
and limits itself to platform responsibilities: abort controllers, acquisition
fences, and serialized decode/install work.

The default Rust lane also allocates process-local request IDs and memoized
URI-to-asset IDs inside `AppState`, transactionally with the request commit.
Presentation/authored asset IDs remain explicit inputs. JavaScript retains its
former UUID allocation only behind the `assetimpl=js|shadow` rollback lanes;
the Rust lane reads both identities back from the typed fetch job.

When a startup network candidate succeeds, the browser still reports the
source/provenance fact, but exact manifest URI-to-asset resolution now runs in
`AppStore`. Missing URIs remain session-local, while duplicate exact URIs are
rejected rather than inheriting JavaScript's first-match ordering. The
standalone presentation rollback lane retains the prior helper.

## Rollback and next boundary

The explicitly sequenced `requestAsset` and `requestPrimaryAsset` methods
remain available for generated-WASM evidence, replay, and rollback. The
`assetimpl=js|shadow|rust` switch is unchanged; the existing `rust` default is
preserved.

Decoded primary-asset completion still exposes its install authorization
through the generic completion commit. Moving that second half to a typed
completion receipt is the next adjacent cut; this commit deliberately does not
claim the entire loading pipeline has crossed the typed boundary.

## Lightweight verification

- `node --test tests/asset_effect_host.test.mjs` covers superseded loads,
  superseded installs, shadow behavior, stale queued work, and JS rollback.
- `node scripts/smoke-asset-request-boundary.mjs` requires the typed
  AppStore/WASM/browser path, rejects generic request-effect filtering, and
  parses both browser modules without compiling WASM.

The four focused session-identity native tests and the exact-presentation-URI
test passed, as did the incremental `wasm32-unknown-unknown` `quilting-wasm`
check with `leptos-ui`. The targeted source oracles passed and parsed the
browser module. Live-browser validation still requires a later binding
refresh; no Trunk, binding-generation, or Binaryen step was part of this
source cut.
