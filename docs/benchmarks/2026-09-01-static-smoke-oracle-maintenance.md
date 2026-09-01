# Static smoke-oracle maintenance

Date: 2026-09-01

The complete CPU-only/source/WASM smoke lane is green: 40 of the repository's
43 `scripts/smoke-*.mjs` checks passed in one ordered sweep. The three excluded
checks own external processes or a build lifecycle (`smoke-blender-browser-relay`,
`smoke-local-peer-relay`, and `smoke-hyperscope-build-policy`) and were not
started during this maintenance pass.

Three stale exact-source assertions were repaired without reducing semantic
coverage:

- pointer selection now requires the awaited backend-pick packet, exact surface
  payload, and retained activation request;
- shared WebGPU extraction now requires the centralized live/evidence style
  policy, including live-presentation demand;
- surface-walk reflection transport now requires the Rust-projection bypass,
  incumbent transport fallback, atomic control rollback, and renderer commit
  order.

The generated development WASM used by executable smokes was built with
`HYPERSCOPE_WASM_OPT=0`. No browser, GPU context, server, Blender process, or
user-owned process was started.
