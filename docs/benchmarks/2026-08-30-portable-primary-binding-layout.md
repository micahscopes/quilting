# Portable primary graphics binding layout

Date: 2026-08-30

## Outcome

The primary WebGL WGSL interface no longer relies on stage-specific reuse of
the same `(group, binding)` coordinate. Its stable namespace is:

| Group | Reachable primary resources |
|---|---|
| 0 | joint/morph pose textures and source-face data |
| 1 | the incumbent per-draw view/entity uniform packet |
| 2 | style/PBR uniforms, material textures, and environment textures |
| 3 | scene-color and transmission pass inputs |

The PBR entry's exact reachable layout contains 3 group-0 entries, 1 group-1
entry, 15 group-2 entries, and 4 group-3 entries. The declared but unreachable
sheen LUT and blurred-scene pairs remain absent from program identity.

Each retained WebGL binding now includes the complete policy needed by a
WebGPU layout: uniform minimum size, texture sample kind and view dimension, or
sampler filtering kind. `WebGlBindingPlan::portable_layout` emits Quilting's
backend-neutral `PipelineLayoutDescriptor`; missing positional groups are
explicit and a deliberate cross-stage collision fails closed.

WebGL's concrete binding points and texture units did not change. The obsolete
fallback that guessed bindings from Naga-generated names was removed; all live
program creation already uses exact reflected plans through the device memo.

## Verification

- all 26 `quilting-shaders` tests pass, including standalone GLSL emission for
  the primary render, patch-preparation, and visibility entries;
- all 81 `quilting-renderer` tests pass, including exact source reflection,
  emitted GLSL coverage, transform-feedback binding provenance, complete
  portable resource kinds, contiguous group construction, and intentional
  collision rejection;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` and tests passes;
- no Trunk server, browser, release build, or WASM optimizer was launched.

## Remaining boundary

The live resident-root WebGPU renderer uses a different storage-oriented
resource model even though it consumes the same functional descriptor types.
Image/workload parity and shared extraction remain the promotion gates before
the incumbent WebGL resource payload can be retired.
