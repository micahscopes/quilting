# Classic Quilting M0 fixture/export evidence

Status: direct seed-42 fixture matrix and typed adversarial rejection frozen,
2026-09-01.

## Authority and command

The exporter uses `quilting-core` read-only through a path dependency. It does
not modify or copy the sampler, constrained-Delaunay implementation, atlas, or
Hyperscope application code. The pinned source was Quilting commit
`4e0093457136eacba07dc1ed93f498f99dcada32`; its relevant topology code was
unchanged from the source reviewed for the handoff.

The exact generator command was:

```sh
cargo run --locked --release \
  --manifest-path tools/quilting-fe-fixtures/Cargo.toml \
  --features quilting-export \
  --bin export-classic-quilting-fixtures -- \
  --output fixtures/classic-quilting/v1 \
  --quilting-commit 4e0093457136eacba07dc1ed93f498f99dcada32
```

The exporter calls direct `TessellationAtlas::build_for_keys` with
`PatchConfig { seed: 42, k_candidates: 30 }`, extracts public patch ranges in
canonical key order, snaps/renormalizes barycentrics under the ABI rule, makes
triangle winding counter-clockwise, and cyclically rotates each triangle so
its smallest index is first. It retains Quilting sampler/point order and CDT
triangle order; it does not lexicographically resort topology.

## Frozen artifacts

The machine-readable authority is
`fixtures/classic-quilting/v1/manifest.json`. Payload hashes cover the exact
canonical patch/vertex/triangle bytes after the 128-byte ABI header.

| Key/artifact | Vertices | Triangles | Payload SHA-256 |
| --- | ---: | ---: | --- |
| `[1,1,1]` | 3 | 1 | `9d273b78db9143626019207fbb7822135ad0d3cdd777ecbe2b7a2d01430d1500` |
| `[1,1,2]` | 4 | 2 | `3f36c564af55d421fddc04702b2afa09740cb9cc41535fa0a0f12359d66f3be4` |
| `[1,2,2]` | 5 | 3 | `5ec399100a8ecca495409dad8606f4dc59c6e9117325858fb3e114f5dfddc897` |
| `[2,4,8]` | 21 | 26 | `9afe43705553ab9b08e83f420d7f784a04a6f126d66838d20893a8831ea5d139` |
| combined matrix | 33 | 32 | `fb61279d43d700079866d818795a8206452e4bd1b1cd82b31246aa7324c1cb7c` |

The manifest also freezes every boundary parameter as an exact `f32` bit
pattern. The checked ABI decoder independently asserts the corresponding
`resolution + 1` counts, parameter grids, positive-zero snapping, packed
ranges, finite barycentrics, indices, and winding.

## Determinism and rejection

Pool widths 1, 2, and 4 independently build round-robin key shards. Results
merge only after sorting by canonical key. All three widths produced the exact
same 1,296-byte combined artifact, not merely the same counts or topology.
Tests also regenerate and byte-compare every committed individual fixture.

`near-degenerate-rejection.json` freezes an AB boundary pair separated by
only a few binary64 ULPs. The checked browser-facing facade rejects it before
the incumbent panicking CDT surface with the stable typed code
`near_duplicate_points` and point identities 3 and 4. The facade also catches
unexpected core panics and maps them to typed errors; a prior artifact can
therefore remain resident.

## Reference timing

One warm release generator run on the reference development machine measured
the complete four-key build, canonical merge, ABI validation, packing, and
cross-width byte comparison as:

| Pool width | Elapsed |
| ---: | ---: |
| 1 | 497 us |
| 2 | 294 us |
| 4 | 335 us |

These are local calibration samples, not browser or product claims. The tiny
matrix is too small to justify a default pool width from these numbers.

## Gate result and next slice

M0's direct fixture gate is green: exact boundary invariants, deterministic
bytes, checked malformed-input rejection, source/license metadata, and the
exact generator command are present. The next bounded phase is M1 pure Fe math:
`quilting_domain`, `quilting_quaternion`, and the actual three-control
`quilting_qb`, with independent Rust-oracle vectors before any raster work.
