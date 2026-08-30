# Textured resident WebGPU PBR — 2026-08-30

## Decision

The adaptive WebGPU layer already preserves one indirect draw boundary per
extracted material batch. It now binds that batch's resolved texture group
instead of slot zero and admits PBR only when every referenced image plus the
environment epoch is resident.

The direct source-root layer intentionally retains coarser atlas/permutation
parity buckets. It can therefore render textured PBR exactly only when every
enabled root domain resolves to the same material slot. Factor-only scenes
continue to share the semantic placeholder group. Textured multi-material root
scenes remain on the complete material-batched path until material identity is
part of a scalable device compaction key; the renderer does not repeat all
vertices per material or substitute missing images.

## Evidence

The native conformance scene now uses a resident blue base-color texture in
material slot one for both its source roots and adaptive replacement leaf. It
asserts that the old factor-only predicate rejects the scene, the broader
exact-PBR predicate accepts it, the resolved root slot is one, both layers
execute in one coherent submission, and the readback's blue-channel energy is
more than twice its red-channel energy. The latter would fail against the old
hard-coded material-zero placeholder.

The strict `quilting-webgpu` Clippy gate and the required Radeon 780M RADV
Vulkan `native_lod` test pass. Browser promotion remains opt-in and still uses
the shared capability predicates; live Chrome parity is a separate gate.

## Portable multi-material direction

`wgpu 29` marks both dynamically indexed texture binding arrays and
GPU-counted multi-draw as native-only. Its browser implementation expands a
fixed multi-draw count into individual `drawIndexedIndirect` calls. A radix/run
compactor can therefore optimize native execution, but cannot by itself remove
the browser's material draw boundary without a readback or Cartesian command
expansion.

The shared path now builds a deterministic paged `texture_2d_array` atlas at
asset upload time. Each stable texture slot retains its exact rectangle,
dimensions, layer, and wrap modes. Native byte uploads and browser
`ImageBitmap` uploads publish both the incumbent individual texture and the
portable atlas; allocation-preserving updates modify both. Radeon conformance
reads packed rectangles back and proves byte equality before and after an
in-place update, including sparse texture-table slots. Manual wrap-aware
filtering and material-indexed shader bindings are the next cutover gate.
