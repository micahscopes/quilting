# Quilting in Fe

This subtree is a self-contained Fe workspace for the small, pure parts of
Quilting that benefit from typed functional shader authoring. Rust Quilting
remains authoritative for topology, atlas generation, scene state, resource
lifetime, and backend integration.

The workspace is deliberately layered:

```text
external Fe core
  ├── ingots/foundation/quilting_domain ───────────┐
  │                    │                           │
  │                    ▼                           │
  │       ingots/geometry/quilting_hyperpatch      │
  │                    │                           │
  └── ingots/algebra/quilting_quaternion ──────────┤
                                                   ▼
                                    ingots/geometry/quilting_qb
                                                   │
                                        ┌──────────┴──────────┐
                                        ▼                     ▼
                        ingots/validation/             ingots/demos/
                        classic_quilting_oracle        classic_quilting_fixed_raster
                                                              ▲
                                                       external Fe std
```

- `foundation` owns reference-domain values and invariants.
- `algebra` owns reusable algebra with explicit conditioning behavior.
- `geometry` owns surface semantics built from those smaller authorities.
- `validation` exposes deterministic Wasm entry points for independent Rust
  comparisons; it is not application code.
- `demos` contains bounded visual programs. A demo may depend inward on the
  other layers, but no library ingot may depend on a demo or oracle.
- `tools/classic-quilting-artifact-abi` is a separate Rust crate. It freezes
  binary fixtures and compares Rust, Fe/Wasm, and generated WGSL without
  making Rust tooling part of the Fe dependency graph.

All internal ingot edges use Fe workspace dependencies (`dependency = true`).
Only the workspace root knows where the pinned Fe `core` and `std` live. This
keeps member manifests portable and makes the compiler boundary visible.

## Toolchain

The exact compiler revision and isolation procedure are in
[`toolchain/README.md`](toolchain/README.md). `../.toolchains/fe` is ignored
and must be an isolated checkout; the active Fe development worktree is never
a dependency. Keeping it outside this subtree also means workspace-wide Fe
formatting never traverses compiler sources.

After preparing it, run from this directory:

```sh
../.toolchains/fe/target/release/fe fmt --check ingots --color never
../.toolchains/fe/target/release/fe check . --profile release --color never
CARGO_TARGET_DIR=../.toolchains/fe/target cargo test --release --locked --manifest-path tools/classic-quilting-artifact-abi/Cargo.toml
CARGO_TARGET_DIR=../.toolchains/fe/target cargo test --release --locked --manifest-path tools/classic-quilting-artifact-abi/Cargo.toml --features quilting-export
CARGO_TARGET_DIR=../.toolchains/fe/target cargo test --release --locked --manifest-path tools/classic-quilting-artifact-abi/Cargo.toml --features fe-oracle
```

Debug-mode checks are useful during editing, but they are not accepted as
performance or release evidence. Release evidence uses both the optimized Fe
compiler binary and the Fe `release` project profile; generated Rust, Wasm,
and GPU-oracle hosts are also compiled with Cargo's `--release` profile. The
Fe/Wasm oracle itself requests Fe's highest current backend level, `O2`; the
authored raster runtime package uses its optimized compiler path.

The strict GPU comparison is documented in
[`docs/classic-quilting-M1-fixed-raster.md`](docs/classic-quilting-M1-fixed-raster.md).

## CGA and Clifford-Bézier direction

The primary next surface lane is `quilting_clifford_bezier`, supported by a
clean `quilting_cga` algebra ingot. CGA will give Quilting a precise language
for points, planes, spheres, incidence, inversions, and spherical control
mechanisms; the Clifford-Bézier ingot will use compile-time specialization to
turn its fixed construction into bounded straight-line shader arithmetic.
Neither begins as a copy of an experimental gallery. The existing QB path is
the compact production baseline and shared oracle, not the ceiling of this Fe
workspace. The type and backend boundaries are recorded in
[`docs/hyperpatch-architecture.md`](docs/hyperpatch-architecture.md). See also
[`docs/quilting-cga-direction.md`](docs/quilting-cga-direction.md).

## Evidence ladder

- M0 freezes the atlas artifact ABI and deterministic seed-42 fixtures.
- M1 proves domain, quaternion, and QB math in Fe against Rust, then proves a
  fixed authored raster through Rust, Fe/Wasm, and WGSL.
- C1 is the next principal slice: freeze the Krasauskas–Zubė Clifford
  construction, prove its quaternion-subalgebra restriction, and render one
  genuinely Clifford patch through the same cross-backend oracle.
- M2 adds the blue-noise and Delaunay teaching views around those proven patch
  evaluators.
- Later milestones add seams, LOD/permutations, controls, workers, and browser
  integration only after the preceding oracle remains green.

The detailed build order lives in
[`docs/classic-quilting-demo-ingots-spec.md`](docs/classic-quilting-demo-ingots-spec.md).
