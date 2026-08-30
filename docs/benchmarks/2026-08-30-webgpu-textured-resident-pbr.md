# Textured resident WebGPU PBR — 2026-08-30

## Decision

The adaptive WebGPU layer already preserves one indirect draw boundary per
extracted material batch. It now binds that batch's resolved texture group
instead of slot zero and admits PBR only when every referenced image plus the
environment epoch is resident.

The direct source-root layer intentionally retains coarser atlas/permutation
parity buckets. It now resolves each root domain's material through one
baseline-WebGPU paged `texture_2d_array` atlas plus device-resident texture
descriptors and material-to-texture records. Arbitrary fully resident opaque
material identities can therefore share a GPU-driven atlas/parity bucket
without repeating vertices, splitting draws on the CPU, requiring native-only
binding arrays, or substituting missing images.

## Evidence

The native conformance scene now contains distinct untextured and blue-textured
resident root domains under one device-generated geometry plan. Its adaptive
replacement uses the untextured material, leaving the textured source root to
prove the portable resident shader path. The test asserts exact per-material
residency, executes both layers in one coherent submission, and requires the
readback's blue-channel energy to exceed its red-channel energy by more than
two-to-one. The latter would fail against the old material-zero placeholder.

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

The shared path builds a deterministic paged `texture_2d_array` atlas at asset
upload time. Each stable texture slot retains its exact rectangle, dimensions,
layer, and wrap modes. Native byte uploads and browser `ImageBitmap` uploads
publish both the incumbent individual texture and the portable atlas;
allocation-preserving updates modify both. The resident shader performs exact
clamp, mirrored-repeat, or repeat addressing plus manual bilinear filtering via
`textureLoad`, then indexes the stable material records directly from the root
draw domain. Radeon conformance proves packed byte equality before and after an
in-place update, including sparse texture-table slots, and proves the
multi-material raster result. Exact WASM compilation and live Chrome promotion
remain separate gates.
