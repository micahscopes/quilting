# WebGPU retained LOD subject layout

## Question

After retaining the packed subject rows, why does every LOD request still
reconstruct the authored-node to dense-row map from all source-face ownership?

## Finding

The mapping is immutable for an uploaded model. `WgslLodSubjectLayout` now
captures it once using the same first-face-occurrence rule that packs immutable
face records. `WgslLodModelWords` carries that layout into
`LodClassifierModel`, and every dynamic subject publication packs against the
retained instance.

The compatibility `pack_wgsl_lod_subject_words` and `_into` functions remain.
They construct a temporary layout for callers that do not retain a model. The
WebGPU production path uses `pack_wgsl_lod_subject_words_with_layout` and never
enters that compatibility branch.

## Bounded CPU result

Let `F` be source faces and `S` dense authored subjects.

| Work per classifier request | Before | After |
| --- | ---: | ---: |
| Scan face-node ownership | `F` | `0` |
| Allocate/build ordered node map | once per request | once per model upload |
| Pack/compare subject rows | `S` | `S` |
| Allocate packed row vector | `0` after prior memo | `0` |

Exact row packing and comparison remains intentionally linear in `S`: it is the
evidence that permits a full subject-table queue upload to be skipped. A future
dirty-subject protocol could make sparse changes cheaper, but would need stable
subject revisions or changed-row evidence from Hyperscape rather than inferred
mutable state.

## Verification

```sh
node scripts/smoke-webgpu-lod-subject-layout.mjs
cargo check -p quilting-renderer --tests
cargo check -p quilting-webgpu --tests
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
```

These gates do not launch a linked WebGPU test binary, Trunk, `wasm-pack`, or
`wasm-opt`.
