# WebGPU resident-root pipeline memo

## Problem

WebGPU startup could request the same resident-root pipeline family once for
ordinary presentation and again for the focus render graph. Shader modules
were already memoized, but every request still recreated the bind-group
layouts, pipeline layouts, and sixteen style/winding render pipelines.

## Boundary

`LodClassifierDevice` now owns one device-scoped `DeviceMemo` keyed by the
complete ordered family of sixteen backend-neutral `RenderPipelineDescriptor`
values. Together they name:

- shader source/catalog, stages, and entry points;
- root, portable-material-atlas, and environment binding schemas;
- source-root vertex layout;
- topology, winding, culling, depth/write, blend, and multisample state; and
- ordinary color plus focus-MRT attachment formats.

On a miss, the shared functional lowerer consumes those same values to create
the WebGPU layouts and fixed state. A cache hit clones reference-counted
`wgpu` handles; it does not lower WGSL or create layouts/pipelines again. A
failed family construction is not published. Formats outside the named
portable subset retain the prior uncached construction path rather than being
assigned an incomplete key.

The cache dies with its `LodClassifierDevice`, which makes device loss and
shutdown the explicit resource-lifecycle boundary. There is no process-global
registry and no frame-varying state in the key.

## Evidence

- `cargo check -p quilting-webgpu --tests`
- the pure family test covers every pass/winding pair, shared versus PBR
  layouts, vertex layout, focus MRT, wire topology, highlight depth policy,
  sample count, format-sensitive identity, and invalid sample rejection;
- the native conformance path asserts one miss followed by one hit for the
  fixed offscreen resident-root family, and asserts that the hit does not
  revisit shader lowering;
- on this run the native test binary compiled, but its GPU body skipped because
  the host exposed no suitable adapter;
- the exact Hyperscope WASM feature set is type-checked separately without
  invoking Trunk, `wasm-pack`, or `wasm-opt`.

The WASM diagnostic packet exposes hit, miss, failed-creation, invalidation,
and resident-entry counters so a user-run browser can verify whether its
presentation and focus formats actually share the family.

## Remaining work

Focus postprocessing now uses the same functional lowering. Prepared/adaptive
patch pipelines remain migration work and should move only with equivalent
immutable keys and lifecycle tests. This cut does not claim WebGL2/WebGPU image
parity.
