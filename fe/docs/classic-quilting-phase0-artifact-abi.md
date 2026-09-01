# Classic Quilting phase 0: patch decision and artifact ABI

Status: Q0 closed and the test-first portion of M0 implemented, 2026-09-01.

## Q0 decision

The required patch model is the current three-control quaternionic-Bézier
triangle. In code and UI, `QB` means quaternionic-Bézier. A conventional
six-control polynomial quadratic triangle is optional, must be named
`quadratic_tri`, and must have its own controls, restriction/seam contract,
fixtures, and GPU record. It is not part of the first implementation ladder.

## Schema v1

The independent checked codec lives in
`tools/classic-quilting-artifact-abi`. Run it with:

```sh
cargo test --manifest-path tools/classic-quilting-artifact-abi/Cargo.toml
```

All integers are unsigned little-endian fixed-width values. All offsets are
absolute byte offsets from the beginning of the artifact. Records are aligned
to 16 bytes. `usize`, native struct layout, maps, and arrival order never enter
the bytes.

### Header: 128 bytes

| Offset | Size | Field | v1 rule |
| ---: | ---: | --- | --- |
| 0 | 8 | magic | ASCII `CQATLAS` followed by NUL |
| 8 | 4 | schema version | `1` |
| 12 | 4 | algorithm version | positive; generator-owned |
| 16 | 4 | endianness marker | `0x01020304` |
| 20 | 4 | header bytes | `128` |
| 24 | 8 | master seed | deterministic request seed |
| 32 | 4 | key/patch count | number of 32-byte patch records |
| 36 | 4 | vertex count | number of 16-byte vertex records |
| 40 | 4 | triangle count | number of 16-byte triangle records |
| 44 | 4 | reserved | zero |
| 48 | 8 | patch-table offset | `128` |
| 56 | 8 | vertex offset | immediately after patch table |
| 64 | 8 | triangle offset | immediately after vertices |
| 72 | 8 | payload bytes | exact bytes after the header |
| 80 | 32 | payload hash | SHA-256 of bytes `[128..end)` |
| 112 | 16 | reserved | zero |

Canonical v1 encoding has no gaps, trailing bytes, or alternative offsets.
The hash is an identity and corruption check, not an authentication boundary.

### Patch record: 32 bytes

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 12 | canonical key `a, b, c` as three `u32` values |
| 12 | 4 | first global vertex |
| 16 | 4 | vertex count |
| 20 | 4 | first global triangle |
| 24 | 4 | triangle count |
| 28 | 4 | reserved zero |

Each resolution is a positive power of two, each key is internally sorted,
and patch records are strictly lexicographically sorted by key. Vertex and
triangle ranges are packed in that order with no overlap or gaps.

### Vertex record: 16 bytes

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 12 | barycentric `a, b, c` as three IEEE-754 `f32` bit patterns |
| 12 | 4 | source class: interior `0`, edge `1`, corner `2` |

Components must be finite, nonnegative, at most one, and sum to one within
`2e-6`. Negative zero is rejected. The source class must agree with the exact
number of zero components. Boundary values must already be snapped: edge `e`
has exactly `resolution[e] + 1` vertices whose opposite barycentric component
is exact positive zero, and its ordered parameters must bit-match the `f32`
values `i / resolution[e]`.

### Triangle record: 16 bytes

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 12 | three global `u32` vertex indices |
| 12 | 4 | reserved zero |

Every index must lie in its patch's vertex range. Indices are distinct,
triangles are counter-clockwise in the reference triangle, and each index
triple is cyclically rotated so its smallest index is first. Schema v1 does not
sort triangles lexicographically, preserving the generator's documented
locality/sampler order.

## What is intentionally not claimed yet

This phase freezes and independently validates the transport contract. It does
not claim Quilting fixture generation, CDT panic removal, pool-size
determinism, or the four requested artifact hashes. Those require the current
Rust Quilting source tree and are the next M0 slice. The generator must use
this codec (or byte-for-byte equivalent code), emit keys `[1,1,1]`, `[1,1,2]`,
`[1,2,2]`, and `[2,4,8]` at seed 42, add the near-degenerate typed rejection,
and record its exact command and independent validation summary.

That follow-on fixture/export slice is recorded in
`classic-quilting-M0-fixtures.md`.
