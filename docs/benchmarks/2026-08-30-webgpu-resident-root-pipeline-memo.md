# WebGPU resident-root pipeline memo

## Problem

WebGPU startup could request the same resident-root pipeline family once for
ordinary presentation and again for the focus render graph. Shader modules
were already memoized, but every request still recreated the bind-group
layouts, pipeline layouts, and fourteen style/winding render pipelines.

## Boundary

`LodClassifierDevice` now owns one device-scoped `DeviceMemo` keyed by the
complete backend target state that selects this family:

- color attachment format;
- optional depth attachment format;
- multisample count.

Shader source, compiler-catalog revision, target, and entry-point identity
remain in the separate backend-neutral `ShaderModuleDescriptor` key. A cache
hit clones reference-counted `wgpu` handles; it does not lower WGSL or create
layouts/pipelines again. A failed family construction is not published.

The cache dies with its `LodClassifierDevice`, which makes device loss and
shutdown the explicit resource-lifecycle boundary. There is no process-global
registry and no frame-varying state in the key.

## Evidence

- `cargo check -p quilting-webgpu --tests`
- `cargo test -p quilting-webgpu --lib`: 10 passed
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

Prepared/adaptive patch and focus-postprocess pipeline families are not yet in
this memo. They should move only with equivalent immutable keys and lifecycle
tests; this cut does not claim WebGL2/WebGPU image parity.
