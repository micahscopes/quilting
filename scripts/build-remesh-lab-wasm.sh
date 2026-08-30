#!/usr/bin/env bash
set -euo pipefail

# Patch Lab is an interactive target. Keep its Rust code optimized while
# preventing either Cargo fan-out or Binaryen from monopolizing the machine.
export CARGO_BUILD_JOBS="${HYPERSCOPE_BUILD_JOBS:-${CARGO_BUILD_JOBS:-2}}"

exec wasm-pack build \
    --release \
    --no-opt \
    --target web \
    --out-dir ../../pkg \
    crates/quilting-wasm
