# WebGPU retained LOD classifier state

## Question

Why does every device LOD classification upload every composed-scene subject
transform when camera motion changes only the view/projection metrics?

## Finding

The classifier mixed two independently changing payloads:

- a fixed 272-byte `LodDispatchUniforms` block containing baseline transform,
  camera projection, density policy, viewport, and immutable counts; and
- one 160-byte `LodSubjectState` row per dense authored subject.

`write_lod_classification_state` previously allocated a new subject vector,
packed every row, and unconditionally issued both queue writes on every
classification. This occurred even when the pose identity was already resident
and the subject transforms were bit-for-bit unchanged.

Each uploaded model now owns an exact retained witness for both halves plus a
reusable subject scratch vector. Packing reuses the allocation, compares the
complete words, and publishes the uniform and subject table independently. A
first all-zero payload still uploads because an uninitialized GPU allocation is
not evidence of zero contents.

## Bounded traffic result

Let `S` be the number of dense subject rows. Dynamic joint matrices and morph
weights are excluded because their independent pose-identity policy already
controls those uploads.

| Classifier request | Before | After |
| --- | ---: | ---: |
| First request | `272 + 160S` | `272 + 160S` |
| Exact repeated state | `272 + 160S` | `0` |
| Camera/density/viewport only | `272 + 160S` | `272` |
| Subject transforms only | `272 + 160S` | `160S` |
| Both halves change | `272 + 160S` | `272 + 160S` |

Device-lifetime `lodStateUploads`, `lodStateReuses`, and
`lodStateUploadBytes` counters expose actual publications in browser
diagnostics. Every classifier request records two outcomes, one per half.

## CPU result and remaining work

The 160-byte-row output allocation is now retained and reused. Exact comparison
still visits every subject, which is required before skipping a full-table
upload, and the shared packer currently reconstructs the node-to-dense-row map
with a `BTreeMap` on every request. Retaining that immutable lookup in the
uploaded model is the next bounded CPU reduction.

Classification and resident LOD reconciliation subsequently moved into one
ordered command encoder and submission without readback; see
`2026-08-31-webgpu-lod-single-submit.md`.

## Verification

Zero-build source oracle:

```sh
node scripts/smoke-webgpu-lod-state-memo.mjs
```

Compiler-only gates use one low-priority Cargo job:

```sh
cargo check -p quilting-renderer --tests
cargo check -p quilting-webgpu --tests
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
```

No linked WebGPU test binary, Trunk server, `wasm-pack`, binding generation, or
`wasm-opt` is involved.
