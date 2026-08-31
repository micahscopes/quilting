#!/usr/bin/env bash
set -euo pipefail

# Trunk runs pre-build hooks for both `serve` and release builds. Keep Rust's
# release optimization by default for renderer performance, but offer an
# explicit development-profile lane for binding/UI iteration. Workspace-level
# numeric-crate overrides still apply. Binaryen's expensive whole-module
# wasm-opt pass remains a separate artifact-production choice.
case "${HYPERSCOPE_WASM_PROFILE:-release}" in
    release)
        wasm_pack_mode=(--release)
        ;;
    dev)
        wasm_pack_mode=(--dev)
        ;;
    profiling)
        wasm_pack_mode=(--profiling)
        ;;
    *)
        echo "error: HYPERSCOPE_WASM_PROFILE must be release, dev, or profiling" >&2
        exit 2
        ;;
esac
if [[ "${HYPERSCOPE_WASM_OPT:-0}" == "1" && "${HYPERSCOPE_ARTIFACT_BUILD:-0}" != "1" ]]; then
    echo "error: wasm-opt requires HYPERSCOPE_ARTIFACT_BUILD=1; ordinary Trunk builds keep Binaryen disabled" >&2
    exit 2
fi
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
