# Rust presentation-composition plan — 2026-08-31

## Boundary

Presentation composition has two layers. Stable asset identity and ordering are
application semantics; decoded buffers, packed node/face offsets, materials,
textures, worker residency, and GPU handles are renderer adaptation. The live
browser previously chose the primary asset by basename and treated every other
manifest asset as secondary.

`PresentationCompositionPlan` now resolves the primary in Rust, preferring an
already validated presentation-animation residency binding, then exact stable
asset identity, then one unambiguous URI-leaf match to the installed scene.
Ambiguous and missing identities are typed failures. Secondary assets retain
manifest order. The standalone browser resolver remains only for `js|shadow`
rollback.

The manifest identity adapter is also gated by presentation enablement. The
Rust implementation is the bootstrap default even for an ordinary standalone
`?glb=...` route, but that does not imply a presentation has been loaded. A
standalone asset therefore returns no durable presentation identity instead of
calling the application resolver and failing startup with `NoPresentation`.

## Evidence

Two focused native fixtures pass and cover uninstalled state, session-ID URI
resolution, exact binding precedence, secondary ordering, and ambiguous
basenames. The source oracle freezes the typed application/WASM boundary,
Rust-authority adapter, rollback resolver, and inline-module syntax. The
neighboring presentation-dispatch oracle and checked-in generated-WASM
rollback suite also pass. The original WebGL2-only wasm32 adapter check passed:

```text
cargo check --target wasm32-unknown-unknown -p quilting-wasm --features leptos-ui
Finished `dev` profile ... in 11.49s
```

A live Chrome peer regression subsequently loaded the ordinary horse route
without `presentation=1`: 984 faces reached device LOD and one WebGPU normals
frame, with no warning, error, or fallback. This specifically covers the
bootstrap combination that exposed the missing enablement guard. The documented
Trunk development profile rebuilt generated bindings through
`wasm-pack --dev --no-opt`; `wasm-opt` was not run.
