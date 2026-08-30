#!/usr/bin/env bash
set -euo pipefail

# Trunk runs pre-build hooks for both `serve` and release builds. Keep Rust's
# release optimization for renderer performance, but make Binaryen's expensive
# whole-module wasm-opt pass an explicit artifact-production choice. In
# particular, `trunk serve --release` must remain a bounded interactive build.
wasm_pack_mode=(--release)
if [[ "${HYPERSCOPE_WASM_OPT:-0}" != "1" ]]; then
    wasm_pack_mode+=(--no-opt)
fi
export CARGO_BUILD_JOBS="${HYPERSCOPE_BUILD_JOBS:-${CARGO_BUILD_JOBS:-2}}"

exec wasm-pack build \
    --target web \
    --out-dir ../../pkg \
    "${wasm_pack_mode[@]}" \
    crates/quilting-wasm \
    --features leptos-ui,webgpu-backend
