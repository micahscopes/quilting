# Functional WebGPU render-shader memo

Date: 2026-08-30

## Outcome

WebGL2 and WebGPU now share Quilting's device-epoch memo primitive. Immutable
`ShaderModuleDescriptor` values remain the pure cache keys; concrete API
handles remain backend-owned effects. `quilting-renderer::memo` is retained as
a compatibility re-export, while the implementation and its lifecycle tests
live in `quilting-core::render_memo`.

The WebGPU prepared-patch, resident-root, and focus-composer modules are keyed
by:

- their composable WGSL entry source;
- the exact compiler and imported-module catalog revision;
- the representative entry point and target; and
- any future shader definitions carried by the descriptor.

Flattening through `quilting-shaders` and `wgpu::ShaderModule` creation now
happen only on a cache miss. Pipeline families may still differ by color
format, depth state, winding, or entry point, but they reuse the underlying
validated shader module when the functional source descriptor is equal.

## Measured lifecycle

Creating the complete focus resource family produces exactly three retained
render modules: prepared patches, resident roots, and focus composition.
Creating a subsequent ordinary PBR patch pipeline leaves the miss and resident
counts at three and records a cache hit. These counters, including failed
creation and invalidation counts, are exposed in the WASM WebGPU diagnostics.

The cache is scoped to the immutable `LodClassifierDevice` generation. Failed
lowerings are never inserted; WebGL context replacement retains the existing
explicit epoch-invalidation behavior.

## Verification

- all 3 backend-neutral memo lifecycle tests passed in `quilting-core`;
- all 75 `quilting-renderer` library tests passed through the compatibility
  path;
- all 10 `quilting-webgpu` library tests passed;
- all 3 native WebGPU conformance tests passed, including exact memo counters;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` passed; and
- no browser was launched, reloaded, or controlled.
