# Rust/Wasm atlas measurement reference

This exports the existing Rust `tri_patch` and constrained CDT solely for an
honest performance baseline. It is **not** an implementation behind the Fe demo.
No workers, shaders, geometry copies to the host, or entropy imports are used.
The explicit seed drives the sampler; an attempted entropy request fails.

Build here:

```sh
CARGO_BUILD_JOBS=2 cargo build --release --target wasm32-unknown-unknown
```

Then from `../quilting-fe-fixtures`, run:

```sh
QUILTING_RUST_ATLAS_WASM=../rust-atlas-baseline/target/wasm32-unknown-unknown/release/rust_atlas_baseline.wasm \
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2 cargo test --release \
  --features 'fe-oracle quilting-export' cpu_atlas_triangle_rust_fe_baseline \
  -- --ignored --nocapture
```

The harness compares all 165 canonical triangle keys, seed42, three rounds,
using one Wasmtime engine configuration with fuel instrumentation disabled.
It reports native Rust, Rust/Wasm and Fe/Wasm separately. Sampling-only and
combined sampling/CDT are separate executions; do not call their timing
difference a direct isolated phase measurement. Output counts are reported;
floating-point/platform differences can change the seeded point stream.
