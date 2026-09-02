# Classic Quilting M1 pure-math oracle evidence

Date: 2026-09-01

This is the first bounded M1 slice from
`classic-quilting-demo-ingots-spec.md`. It lands only the dependency-ordered
pure Fe foundation:

1. `quilting_domain`
2. `quilting_quaternion`
3. `quilting_qb`

The QB ingot implements the actual three-control quaternionic-Bezier quotient,
not the separately named optional six-control polynomial triangle. Fixed raster
topology and browser integration remain outside this slice.

## Source pins and isolation

- Fe source: `2cda142ed0ca93492d14fc51d7e68369ed322781`
- read-only Quilting oracle: `4eeb68fb16b634dac732d2697d352fbf403d1377`
- frozen M0 fixture/ABI parent: MB2 commit `64ba274`

The only Fe change after the initially inspected `25590d5b0` pin was an actor
typed-readback test change; no compiler implementation changed. The Quilting
math sources used here (`patch.rs`, `quaternion.rs`, `permutation.rs`, and
`triangle.rs`) have no diff from the M0 `4e009345` pin. The Quilting worktree
was clean when the Rust oracle dependency was captured and built at `4eeb68f`.
Concurrent work later advanced that worktree; this MB2 phase performed no
writes there.

Fe compiler-library and Wasmtime dependencies are behind the standalone
artifact tool's non-default `fe-oracle` feature. Default ABI validation stays
small and independent.

## Covered contracts

`quilting_domain` fixes the equilateral Cartesian/barycentric conversion,
positive-zero boundary admission, `EdgeId::{BC, CA, AB}`, containment,
edge-parameter convention, and the small vector operations needed by patch
differentials.

`quilting_quaternion` fixes the `(w,x,y,z)` scalar layout and Hamilton product.
Its bounded inverse returns a zero quaternion plus `valid = false` whenever the
squared norm is below the caller's threshold.

`quilting_qb` evaluates

```text
X(a,b,c) = (sum_i bary_i * (pure(P_i) * W_i))
           * inverse(sum_i bary_i * W_i)
```

for exactly three controls. It also provides analytic quotient-rule tangents,
a normalized cross-product normal, the current Quilting S3 barycentric order,
and odd-permutation normal parity. A denominator pole returns finite zero
position/tangents and `conditioned = false`.

## Executed gates

All commands ran from `/laboratory/fe-stuff/mb2`.

```text
/laboratory/fe-stuff/fe-worktrees/mb2/target/debug/fe fmt --check ingots/quilting_domain --color never
/laboratory/fe-stuff/fe-worktrees/mb2/target/debug/fe fmt --check ingots/quilting_quaternion --color never
/laboratory/fe-stuff/fe-worktrees/mb2/target/debug/fe fmt --check ingots/quilting_qb --color never
/laboratory/fe-stuff/fe-worktrees/mb2/target/debug/fe fmt --check ingots/classic_quilting_oracle --color never
/laboratory/fe-stuff/fe-worktrees/mb2/target/debug/fe check ingots/classic_quilting_oracle --color never
cargo fmt --manifest-path tools/quilting-fe-fixtures/Cargo.toml -- --check
CARGO_TARGET_DIR=/laboratory/fe-stuff/fe-worktrees/mb2/target cargo test --locked --manifest-path tools/quilting-fe-fixtures/Cargo.toml
CARGO_TARGET_DIR=/laboratory/fe-stuff/fe-worktrees/mb2/target cargo test --locked --manifest-path tools/quilting-fe-fixtures/Cargo.toml --features quilting-export
CARGO_TARGET_DIR=/laboratory/fe-stuff/fe-worktrees/mb2/target cargo test --locked --manifest-path tools/quilting-fe-fixtures/Cargo.toml --features fe-oracle
CARGO_TARGET_DIR=/laboratory/fe-stuff/fe-worktrees/mb2/target cargo clippy --locked --manifest-path tools/quilting-fe-fixtures/Cargo.toml --all-targets --all-features --no-deps -- -D warnings
```

The default ABI suite passes 6 tests, the M0 exporter suite passes 10, and the
combined Fe oracle suite passes 13. The oracle suite compiles the Fe dependency
chain to self-contained Wasm, validates the module, rejects unexpected imports,
and executes it in Wasmtime.

Numerical comparisons cover every barycentric vertex in the committed
`direct-seed42-matrix.cqa`, all three frozen edge-parameter conventions,
positive-zero admission and invalid normalization, independent f32 Hamilton
products and inverses, a barycentric grid over the Rust
`QBTriPatch::eval_differential` oracle, an identity-weight flat patch, a
denominator pole, and all six S3 permutations. Position tolerance is `2e-6`;
tangent and normal tolerance is `4e-6` after the Rust f64 oracle is deliberately
narrowed to the Fe f32 contract.

## Remaining M1 work

The next bounded slice starts at specification step 8:

1. generate the smallest frozen fixture as an expanded, non-indexed
   `TriangleList<N>`;
2. render single-patch position and normal colors through current authored
   raster, then add blue-noise and Delaunay teaching views;
3. compare the same vectors through generated WGSL (Wasm and Rust are covered
   here);
4. add wire, material, and normal display modes; and
5. run the authored-raster Naga, Wasmtime, and wgpu/lavapipe gates.

If current authored raster cannot express that fixed topology without a
compiler hack or an unreflected resource, M1 stops at that gate as required by
the specification.
