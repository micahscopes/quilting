# Rust-authoritative Patch Lab route cutover

Date: 2026-09-01

## Outcome

Patch Lab deep links now cross the Rust route boundary as one typed
`PatchLabSessionIntent`. The browser no longer reconstructs the Rust-authority
startup session independently from `lab*` strings.

Rust owns:

- active shape and LOD-field decoding;
- periodic phase conversion to integer microradians;
- resident-atlas exponent limits;
- min/max reconciliation;
- the non-triangle/manual-edge fallback;
- the complete startup packet exposed by `canonicalizeHyperscopeRoute`.

The browser remains a platform adapter: it projects the admitted packet into
controls and performs URL percent encoding plus `history.replaceState`. Live
Patch Lab edits already return through `AppStore` before URL synchronization,
so the copied route reflects the committed read model rather than an in-flight
worker result.

## Evidence

- `cargo test -p hyperscope-app`: 141 passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- `cargo clippy -p hyperscope-app --lib --no-deps -- -D warnings`: passed.
- development WASM package built with `HYPERSCOPE_WASM_PROFILE=dev` and
  `HYPERSCOPE_WASM_OPT=0`; no Binaryen optimization was run.
- `node scripts/smoke-rust-patch-lab-route.mjs`: passed against the rebuilt
  WASM bridge and browser source.
- `node scripts/smoke-patch-lab-job-boundary.mjs`: passed.
- `node scripts/smoke-patch-lab-effect-projection.mjs`: passed.
- the extracted `hyperscope.html` module parses under `node --check`.
- `git diff --check`: passed.

## Existing gate drift kept out of scope

The broad `smoke-hyperscope-route-shadow.mjs` reaches and passes the new typed
Patch Lab assertions, then fails on its pre-existing exact source-string for
surface picking (`const pickedSurface = pickSurfaceAtCanvasPixel(x, y);`). The
current browser implementation has already moved that interaction boundary.
The assertion was not weakened as part of this cut.

Current-toolchain workspace-wide rustfmt and strict Clippy also report many
unrelated historical style/lint deltas. No mass formatting or opportunistic
lint rewrite was performed.
