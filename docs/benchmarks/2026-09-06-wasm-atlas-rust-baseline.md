# Unrestricted triangle atlas: first Rust / Fe baseline

This is an algorithm/runtime baseline, not a matched-quality language comparison.
It establishes that the current Fe/Wasm implementation is substantially slower
than the existing native Rust implementation. Browser-worker measurements remain
outstanding. Quad generation has no matched Rust baseline here.

## Work and measurement

- Every canonical triangle key at exponents 0–8: 165 keys, no ratio restriction.
- Seed 42, 30 proposals per active point in both implementations.
- Rust `tri_patch` (not the jittered-grid shortcut), followed by constrained CDT.
- Fe O2 Wasm, Wasmtime default engine, fuel instrumentation OFF.
- Native Rust release; serial execution in the same test process, three rounds.
- Warm initialization excluded; no compilation, audits, packing, messages, upload
  or retained atlas in timed regions. Each tile is discarded after inspection.
- Fe stage 0 measures sampling; stage 1 separately repeats sampling then CDT.
  Their difference is an estimate, NOT an isolated triangulation measurement.
- Raw log: `/laboratory/quilting/scratch/rust-fe-atlas-baseline-20260906.log`.

| Round | Rust sampling | Rust CDT | Rust combined | Fe sampling | Fe combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.8745 s | 0.3291 s | 3.2036 s | 4.7583 s | 11.1673 s |
| 2 | 2.8886 s | 0.3268 s | 3.2155 s | 4.7719 s | 11.2169 s |
| 3 | 2.8840 s | 0.3182 s | 3.2022 s | 4.7153 s | 11.0042 s |

Median combined time: Rust 3.204 s, Fe/Wasm 11.167 s (3.49× slower).
Rust outputs 527,523 points / 1,026,611 triangles; Fe outputs 464,872 points /
901,309 triangles, identically across all three rounds. Fe therefore produces
about 12.2% fewer triangles while taking longer. Neither output count nor valid
CDT establishes equal blue-noise quality or coverage of the density field.

The Fe stage difference is 6.29–6.45 s; Rust CDT alone is 0.318–0.329 s.
That points toward Fe triangulation and its scratch/setup costs as a priority
for direct phase profiling, not a proven compiler cause. Sampling also needs
attention, particularly dense uniform tiles.

First-round examples:

| Exponent key | Rust combined | Fe combined | Rust / Fe triangles |
| --- | ---: | ---: | ---: |
| [0,0,8] | 11.49 ms | 34.43 ms | 3,388 / 2,970 |
| [8,8,8] | 356.67 ms | 1,467.76 ms | 106,600 / 93,962 |

Fe linear-memory high-water was 16,318,464 bytes for these unaudited triangle
stages; this is not total process RSS or retained packed atlas memory. It is not
comparable to the earlier 37.2 MB audited triangle-plus-quad corpus.

## Limitations and next measurements

Rust uses f64 sampling and the `cdt` crate. Fe uses fixed-point sampling, exact
integer predicates, a different proposal stream, a different spatial index,
and a whole-domain exploration pass. Equal keys/seeds do not mean equal points.
The Rust baseline has not passed the Fe sampler's exact exclusion/coverage
audits; do not use its speed as justification to weaken those guarantees.

Next: isolate scratch initialization, point location, insertion, legalization
and predicate work; compare CDT on identical supplied points; compare sampling
quality and accepted-neighbor work; then run both compiled to Wasm in the same
browser, including persistent workers and packed-result transfer.

Reproduce from `fe/tools/quilting-fe-fixtures`:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2 cargo test --release \
  --features 'fe-oracle quilting-export' cpu_atlas_triangle_rust_fe_baseline \
  -- --ignored --nocapture
```

The initial run used geometry at `a32cfef`, with pending packing exports added
to the same validation module. Packing is not called by this benchmark.

After moving the benchmark into its own module, all three rounds passed again:
Rust combined 3.314 / 3.300 / 3.267 s; Fe combined 11.399 / 11.435 / 11.187 s.
Point/triangle counts were unchanged. The second raw log is
`/laboratory/quilting/scratch/rust-fe-atlas-baseline-relocated-20260906.log`.
