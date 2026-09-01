# App-owned route wire ABI

Date: 2026-09-01

The exact browser-facing route schema now belongs to `hyperscope-app`, behind
the optional `wire` feature. The default native application remains free of
serde. The existing `replay` feature enables `wire`, allowing JSON-shape tests
to exercise the same schema that the WASM bridge publishes.

`quilting-wasm::route_shadow` is now a 26-line platform adapter. It decodes the
JavaScript pair array, invokes `HyperscopeRoute`, and serializes the
application-owned result. The previous 427-line bridge owned roughly 400 lines
of duplicate DTO declarations and conversions. The authoritative wire module
is 512 lines because the schema and its exact shape oracle still have real
surface area; this checkpoint relocates that policy to the correct layer
rather than claiming to eliminate it.

The wire oracle proves that every focused projection in the diagnostic result
is byte-for-byte equal to its counterpart inside `startupSettings`. A route
diagnostic yields a null startup packet while preserving the diagnostic and
focused projections for rollback analysis. The browser continues to treat only
the startup packet as admission authority.

Verification:

- `cargo check -p hyperscope-app`: passed with no wire dependencies enabled.
- `cargo check -p hyperscope-app --features wire`: passed.
- `cargo check -p hyperscope-app --features replay`: passed.
- `cargo test -p hyperscope-app --features replay`: 183 passed, including the
  exact route-wire shape oracle.
- `cargo clippy -p hyperscope-app --lib --features wire --no-deps -- -D warnings`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- Development WASM built with `HYPERSCOPE_WASM_OPT=0`.
- All 40 CPU-only, source, and generated-WASM smoke scripts passed against the
  rebuilt artifact. The three process-owning server/build-lifecycle smokes
  remained intentionally excluded.

No browser, renderer, GPU context, server, or user-owned process was started.
