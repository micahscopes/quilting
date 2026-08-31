# WebGPU retained frame-table memo

## Question

Does an animation-only frame resend camera and conformal frame records even
when their exact packed values did not change?

## Finding

Yes. Every encoded fallback frame uploaded one 256-byte record per retained
batch. The resident path uploaded one record per root draw domain plus another
table for an adaptive overlay. The fallback path also allocated and collected
a temporary `Vec<PatchRenderFrame>` before copying those records into an
already-retained staging vector.

Pose, LOD, and frame-table state have different lifetimes. Animation can
advance pose and LOD while the view, focus field, selection, material domain,
and conformal entity transforms represented by a frame table remain exact.

## Contract

Each retained global/domain table now owns its last packed words and an
explicit publication witness. Packing compares every complete record with the
resident staging row:

- the first valid table always uploads, including a hypothetically all-zero
  record;
- an exact match records a reuse and issues no `Queue::write_buffer` call;
- any changed row republishes the complete table atomically; and
- a packing failure invalidates the witness, so partially changed staging
  words can never suppress the next valid publication.

Fallback batches pack directly into retained staging storage. No temporary
frame-record vector remains in the production encoder.

## Bounded traffic result

Let `B`, `D`, and `O` be fallback batches, resident root domains, and adaptive
overlay batches. The later split-frame contract stores a 176-byte global row
per family and an 80-byte local row per batch/domain.

| Steady-state case | Before | After |
| --- | ---: | ---: |
| Animation-only fallback frame | `256B` bytes | `0` bytes |
| Animation-only resident roots | `256D` bytes | `0` bytes |
| Animation-only roots + overlay | `256(D+O)` bytes | `0` bytes |
| Camera/focus change, per family | `256N` bytes | `176` bytes |
| Local conformal/material change | `256N` bytes | `80N` bytes |

The first three rows are the exact-word memo result. The final two include the
subsequent lossless split of frame-global camera/focus words from batch-local
Möbius/material words; see `2026-08-31-webgpu-split-frame-state.md`.

Device-lifetime diagnostics expose `frameTableUploads`, `frameTableReuses`, and
`frameTableUploadBytes`, including writes made before a presentation skip.

## Verification

The zero-build source oracle is:

```sh
node scripts/smoke-webgpu-frame-table-memo.mjs
```

Compiler checks use one low-priority Cargo job. No Trunk server, `wasm-pack`,
binding generation, or `wasm-opt` is part of this gate.

```sh
cargo check -p quilting-webgpu --lib
cargo check -p quilting-webgpu --tests
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
```
