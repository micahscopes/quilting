# WASM backend feature boundary — 2026-08-31

## Finding

The wasm32 Hyperscope adapter failed to compile without `webgpu-backend`
because environment-map upload unconditionally resolved the feature-gated
WebGPU module. ImageBitmap mirroring already had the intended conditional
pattern; the environment mirror had simply missed the same guard.

## Correction

The environment mirror call is now compiled only with `webgpu-backend`.
WebGL2 upload remains unconditional and retains incumbent renderer state if a
feature-enabled WebGPU mirror later rejects the maps.

## Evidence

`scripts/smoke-wasm-feature-boundaries.mjs` freezes the module and mirror-call
guards. The broader wasm32 adapter check discovered this issue and will be
rerun without WebGPU after system load returns below the build threshold; the
feature-enabled check remains a separate gate.
