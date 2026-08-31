# Rust primary-scene install completion — 2026-08-31

## Outcome

Primary-scene installation now completes through a typed application port.
`AppStore::complete_primary_scene_install` returns the committed transition and
the exact animation-clip cancellation identities caused by replacing the
resident scene. The port rejects any unexpected renderer job category rather
than allowing a platform adapter to silently ignore it.

The WASM boundary exposes `finishPrimarySceneInstalled` and
`finishPrimarySceneInstallFailed`. Ordinary browser loading consumes their
`clipCancellations` receipt; the older `completePrimaryScene*` methods remain
only as rollback and generated-binding compatibility seams.

## Stale renderer-work fence

When Rust cancels a clip job owned by the former resident scene, the browser
advances both the animation-pose and presentation-animation generations. Any
already-awaited worker response therefore fails its existing generation check
before it can mutate the replacement scene. This makes scene replacement—not
incidental loader ordering—the semantic cancellation fence.

## Lightweight verification

`node scripts/smoke-primary-scene-install-boundary.mjs` requires the typed
AppStore/WASM/browser path, validates the generation fence, rejects ordinary
use of the generic completion methods, and parses the inline browser module.
The Rust unit test constructs an in-flight clip switch, replaces the primary
scene, and requires the typed receipt to expose that exact obsolete job.
Compiled and live gates remain deferred while unrelated builds occupy the
machine; this cut starts no Trunk, Cargo, or Binaryen process.
