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

## Evidence

Two focused native fixtures pass and cover uninstalled state, session-ID URI
resolution, exact binding precedence, secondary ordering, and ambiguous
basenames. The source oracle freezes the typed application/WASM boundary,
Rust-authority adapter, rollback resolver, and inline-module syntax. The
neighboring presentation-dispatch oracle and checked-in generated-WASM
rollback suite also pass. The WebGL2-only wasm32 adapter check passes:

```text
cargo check --target wasm32-unknown-unknown -p quilting-wasm --features leptos-ui
Finished `dev` profile ... in 11.49s
```

Trunk, wasm-pack, binding generation, and wasm-opt were not run.
