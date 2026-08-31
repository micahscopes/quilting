# WebGPU split frame state

## Question

Can camera/focus updates stop replicating identical view state through every
retained batch and resident draw domain without changing rendered semantics?

## Finding

Yes. The old 256-byte `PatchRenderFrame` mixed two independent lifetimes:

- 176 bytes of frame-global view, focus, selection, and display state; and
- 80 bytes of domain-local Möbius transform and material identity.

Every batch/domain received its own copy of both halves. Exact word
memoization removed animation-only writes, but camera motion still uploaded
`256N` bytes for `N` domains.

## Contract

The device contract is now losslessly decomposed into:

- `PatchRenderGlobal`: 44 words / 176 bytes, one row per retained family; and
- `PatchRenderDomain`: 20 words / 80 bytes, one row per batch or draw domain.

Naga layout tests assert both exact spans. The one-domain footprint remains
176 + 80 = 256 bytes, so the split introduces no padding tax. Rendering,
prepared visibility, resident-root visibility, focus-field output, ordinary
picking, and resident-root picking all bind the same pair.

Each half retains its own exact-word memo. A camera/focus/selection update
publishes only the global row; a conformal/material-domain update publishes
only the local table. Failed local packing invalidates only that table, while
the already-valid global row remains reusable. Queue order plus the absence of
a submitted draw on error preserves coherent presentation.

Adaptive overlay rows now carry their resolved material slot. The previous
monolithic helper silently wrote slot zero for every adaptive batch, which
could shade nonzero-material leaves with the wrong material in WebGPU.

## Bounded traffic result

Let `N` be the number of retained batches/domains.

| Steady-state case | Monolithic row | Split state |
| --- | ---: | ---: |
| First frame | `256N` | `176 + 80N` |
| Animation/LOD only | `0` after memo | `0` |
| Camera/focus/selection only | `256N` | `176` |
| One or more local-domain changes | `256N` | `80N` |
| Both global and local change | `256N` | `176 + 80N` |

For one domain the first/both-change cases are equal. Every additional domain
saves 176 bytes per global update, and camera motion becomes constant-size.
Resident roots and their matching sparse overlay subsequently share that
single global row within one retained aggregate; see
`2026-08-31-webgpu-aggregate-global-frame.md`.

## Verification

Zero-build source oracles:

```sh
node scripts/smoke-webgpu-frame-table-memo.mjs
node scripts/smoke-webgpu-frame-state-split.mjs
```

Bounded compiler and shader gates use one low-priority Cargo job:

```sh
cargo check -p quilting-webgpu --lib
cargo check -p quilting-webgpu --tests
cargo test -p quilting-shaders device_shader --lib
cargo test -p quilting-shaders \
  flattened_visibility_compaction_wgsl_is_standalone_and_reparseable --lib
```

No Trunk server, `wasm-pack`, binding generation, or `wasm-opt` is involved.
