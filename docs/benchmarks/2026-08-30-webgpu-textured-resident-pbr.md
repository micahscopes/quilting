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
