# App-owned asset job and read-model wire ABI

Date: 2026-09-01

`hyperscope-app/wire` now owns the serialized asset-acquisition request,
completion, fetch job, cancellation identity, and asset-status read model.
`quilting-wasm` converts those values to `JsValue`; it no longer restates their
request/asset IDs, descriptor fields, status variants, or application commit.

The boundary remains deliberately effectful in the browser. JavaScript still
fetches or reads dropped bytes, cancels obsolete work, decodes glTF, installs
renderer resources, and returns typed completions. Rust owns which job is
current, replacement ordering, status publication, and the exact packet the
adapter executes.

The asset wire oracle proves maximum-width semantic sequences remain decimal
strings, fetch/cancellation scopes retain both stable IDs, and ready asset
metadata preserves modest author/license attribution. Content digests remain
part of fetch identity; the compact asset-status projection preserves the
established public ABI.

The WASM facade decreased from 5,429 to 5,290 lines. The application wire
module grew from 627 to 835 lines, including the exact request/read-model
oracle. The total increase records executable evidence while removing 139
lines of platform-owned application protocol.

Verification:

- `cargo test -p hyperscope-app --features replay`: 187 passed, including the
  scoped asset-job and credit-metadata oracle.
- `cargo clippy -p hyperscope-app --lib --features wire --no-deps -- -D warnings`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- Development WASM built with `HYPERSCOPE_WASM_OPT=0`.
- Generated-WASM app-shadow and route smokes plus focused render-settings and
  Patch Lab smokes passed.
- Focused asset-request and asset-completion source gates passed against the
  app-owned output and thin WASM adapter.
- All 40 CPU-only, source, and generated-WASM smoke scripts passed. The three
  process-owning server/build-lifecycle smokes remained intentionally excluded.

No browser, renderer, GPU context, server, or user-owned process was started.
