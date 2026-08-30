#!/usr/bin/env bash
set -euo pipefail

# Trunk runs pre-build hooks for both `serve` and release builds.  Keep the
# interactive development loop optimized enough for the renderer, but skip
# Binaryen's expensive whole-module wasm-opt pass unless the caller explicitly
# requested a release build.
wasm_pack_mode=(--release)
if [[ "${TRUNK_PROFILE:-debug}" != "release" ]]; then
    wasm_pack_mode+=(--no-opt)
    export CARGO_BUILD_JOBS="${HYPERSCOPE_BUILD_JOBS:-2}"
fi

exec wasm-pack build \
    --target web \
    --out-dir ../../pkg \
    "${wasm_pack_mode[@]}" \
    crates/quilting-wasm \
    --features leptos-ui,webgpu-backend
