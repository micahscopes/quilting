# App-owned commit and effect wire ABI

Date: 2026-09-01

`hyperscope-app/wire` now owns the exact serialized form of `AppCommit`, every
`AppEffect` variant, and the typed Patch Lab effect list. The default native
application still enables no serde dependency. `quilting-wasm` remains
responsible for converting the resulting serializable value into `JsValue`,
but no longer interprets commit dispositions, authored proposal roles, asset
jobs, animation jobs, or Patch Lab jobs.

This moved a central exhaustive policy match out of the platform facade. The
facade decreased from 5,865 to 5,635 lines; the new 295-line application module
contains the retained schema and its oracle. As with the route wire migration,
the objective is single ownership and exhaustiveness rather than minimizing the
schema's necessary representation.

The exact oracle exercises `u64::MAX` revision/job values and proves they remain
decimal strings across the JavaScript boundary. It also fixes the nested tagged
shape of an `evaluate_lod` effect, including manual edge requests, atlas level,
and grading ratio. The generated-WASM app smoke covers asset replacement,
authored projection/peer ingress, presentation, navigation, and animation; the
focused Patch Lab and render-settings smokes cover typed job draining and
embedded commits.

Verification:

- `cargo test -p hyperscope-app --features replay`: 184 passed.
- `cargo clippy -p hyperscope-app --lib --features wire --no-deps -- -D warnings`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- Development WASM built with `HYPERSCOPE_WASM_OPT=0`.
- Generated-WASM app-shadow, Patch Lab job, render-settings, and route smokes
  passed.
- All 40 CPU-only, source, and generated-WASM smoke scripts passed. The three
  process-owning server/build-lifecycle smokes remained intentionally excluded.

No browser, renderer, GPU context, server, or user-owned process was started.
