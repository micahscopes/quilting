# Functional WebGPU focus pipelines

## Outcome

The seven-pass focus composer no longer has a WebGPU-only pipeline identity.
`focus_postprocess_pipeline_descriptors` returns an ordered family of
backend-neutral `RenderPipelineDescriptor` values containing the exact:

- WGSL source, compiler catalog, stages, and entry points;
- bind-group bindings, visibility, kinds, dynamic-offset policy, and sizes;
- primitive and multisample state; and
- intermediate and final color targets.

The WebGPU device memo uses that family as its key. On a miss, the shared
lowerer consumes the same descriptors to create the bind-group layout and
pipeline fixed state. A hit clones retained `wgpu` handles and does not revisit
shader lowering or pipeline creation. Frame focus parameters, textures,
targets, command encoders, and application state are not cache keys.

The portable texture-format vocabulary now includes the ordinary WebGPU
surface formats BGRA8, sRGB BGRA8, and RGB10A2. If a future adapter supplies a
format outside the portable subset, the existing uncached construction path
still runs; it is not assigned a misleading functional key.

## Lifecycle evidence

- the pure descriptor test verifies all seven passes, the shared four-binding
  layout, intermediate formats, final surface format, and duplicate blur entry
  point with distinct attachment state;
- a semantic-plan test fixes the seven retained handle kinds and attachment
  classes; descriptor generation and named runtime-field assignment both
  iterate that plan instead of maintaining a second index table;
- the native conformance path expects one family miss, then one hit, one
  resident family, and no shader-memo traffic on the hit;
- WASM diagnostics expose focus-pipeline hit, miss, failed-creation,
  invalidation, and resident-entry counters;
- the memo remains owned by one `LodClassifierDevice`, so device teardown is
  also cache teardown.

This cut does not promote WebGPU focus presentation. The frozen user-triggered
WebGL2/WebGPU image oracle remains the visual parity gate.
