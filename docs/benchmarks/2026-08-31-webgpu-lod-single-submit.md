# WebGPU single-submit resident LOD graph

## Question

Why does one browser LOD request submit classification and resident seam/atlas
reconciliation as two separate GPU jobs when no CPU result exists between them?

## Finding

The split was an API convenience, not a dependency boundary. Classifier passes
one and two write a device-local packed table. Resident reconciliation reads
that table, performs the bounded shared-edge closure, and writes the final
resident table. WebGPU command-pass order already provides the required
visibility; no staging copy, map, readback, or CPU decision intervenes.

`classify_and_reconcile_on_device` now publishes/reuses input state, encodes
both classifier passes and resident reconciliation into one command encoder,
and submits it once. The browser uses this complete graph. The lower-level
encode helpers and former separately submitted helpers remain available for
diagnostics, conformance fixtures, and rollback.

## Bounded result

| One browser LOD request | Before | After |
| --- | ---: | ---: |
| Command encoders | 2 | 1 |
| Queue submissions | 2 | 1 |
| CPU-visible copies/maps | 0 | 0 |
| Classifier/reconcile compute dispatches | unchanged | unchanged |
| Packed LOD semantics | unchanged | unchanged |

This removes one encoder allocation and one queue scheduling boundary per LOD
request. It does not claim fewer shader dispatches; those bounded passes are
the actual classification and crack-free reconciliation work.

## Lifetime correction

The reconciled handle semantically borrows the uploaded model, not the local
classification wrapper. Its Rust lifetime now states that relationship
directly, allowing a complete graph helper to return resident output without
extending the lifetime of a temporary token.

## Verification

```sh
node scripts/smoke-webgpu-lod-single-submit.mjs
cargo check -p quilting-webgpu --tests
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
```

These are source/compiler gates. No linked WebGPU test binary, Trunk server,
`wasm-pack`, binding generation, or `wasm-opt` is involved.
