# App-owned Patch Lab wire projection

Date: 2026-09-01

Patch Lab startup and live state now share one output projection in
`hyperscope-app/wire`. `PatchLabControlsWire` supplies the exact shape, field,
edge requests, exponent band, phase, bend, grid, and animation values used by
both `RouteStartupWireSettings` and `PatchLabReadModelWire`. The live projection
also owns decimal-string job identities, installed geometry facts, reconciled
LOD summaries, histograms, and failures.

Input remains intentionally separate. `quilting-wasm` still validates browser
numbers, accepts the historical `edges` alias at the platform boundary, and
rejects a rendered-triangle count that is not an exact nonnegative JavaScript
safe integer before constructing the semantic Rust summary. This checkpoint
does not weaken that admission seam.

The WASM facade decreased from 5,635 to 5,492 lines and the route module from
512 to 480 lines. The application wire module grew from 295 to 497 lines,
including a new exact oracle. Total lines rose slightly because the oracle
constructs a complete installed/pending read model and verifies shared controls,
string encoding of maximum `u64` job IDs, geometry identity, rendered triangle
accounting, and histogram shape. The architectural gain is removing 175 lines
of duplicate output policy from two adapters.

Verification:

- `cargo test -p hyperscope-app --features replay`: 185 passed, including the
  shared Patch Lab read-model/control oracle.
- `cargo clippy -p hyperscope-app --lib --features wire --no-deps -- -D warnings`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- Development WASM built with `HYPERSCOPE_WASM_OPT=0`.
- Generated-WASM app-shadow and route smokes plus focused Patch Lab and render
  settings smokes passed.
- All 40 CPU-only, source, and generated-WASM smoke scripts passed. The three
  process-owning server/build-lifecycle smokes remained intentionally excluded.

No browser, renderer, GPU context, server, or user-owned process was started.
