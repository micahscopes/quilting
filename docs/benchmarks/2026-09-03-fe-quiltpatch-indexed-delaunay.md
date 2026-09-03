# Fe QuiltPatch indexed Delaunay restoration

Date: 2026-09-03

Commit under test: `67b0116` (`quilting-fe`)

Environment:

- Chromium 149.0.0.0 on Linux
- WebGPU adapter: AMD RDNA 3, non-fallback
- Portable device limits: 256 compute invocations per workgroup and eight
  storage buffers per shader stage
- Release Fe CLI served by `fe web dev` on `127.0.0.1:8768`
- Default bilinear Clifford-Bezier controls and pullback-blue-noise mode

## Question

The original restoration provider repeatedly searched every pair of resident
faces for the lexicographically first illegal unlocked edge. Does maintaining
one reverse-directed twin lane per resident half-edge preserve that decision
while removing the repeated all-face-pairs search?

## Browser measurements

All GPU measurements await `device.queue.onSubmittedWorkDone()`. Times are
single-run wall-clock observations, retained as directional evidence rather
than a statistically stable benchmark suite.

| Measurement | all-face-pairs provider | indexed provider |
| --- | ---: | ---: |
| Complete warm pullback frame | 3543.8 ms | 234.5 ms |
| CPU encode and submit portion | 7.0 ms | 6.3 ms |
| GPU completion portion | 3536.8 ms | 228.2 ms |
| Chart 0 restoration | 1721.8 ms | 198.4 ms |
| Chart 1 restoration | 51.1 ms | 14.6 ms |

The indexed provider is approximately 15.1 times faster for the complete warm
frame in this observation. An independently encoded 64-round sampling cycle
took 6.5 ms in one command buffer, confirming that topology restoration—not
candidate generation—was the dominant measured cost.

## Convergence receipt

A DevTools-only buffer-usage probe added `COPY_SRC` while constructing resident
storage buffers. It did not change application source or selection semantics.
The probe copied the two eight-word receipts after GPU completion:

```text
chart 0: [1, 64, 24, 1, 102, 1, 200, 0]
chart 1: [1, 26, 24, 1, 26, 1, 48, 0]
```

The fields are sampling-valid, resident-points, boundary-points,
construction-valid, resident-triangles, restoration-valid, flips, and the
reserved final word. Both charts converged with zero remaining violations and
zero invariant failures.

Resident buffer SHA-256 receipts:

| Buffer | Bytes | SHA-256 |
| --- | ---: | --- |
| `points_0` | 768 | `c4a4686adbde14a19c817bec707439f5eb2b74aa94997c2424aae40692f358ad` |
| `triangles_0` | 1856 | `76ee7da40367b5874cee998f561cd90e2f9b9dad5e95dd479209a2e7a30ad859` |
| `twins_0` | 1392 | `b9d2cca1f2096a302cfad250a687038816800106b241de7e9a83be4bb3734b0d` |
| `receipt_0` | 32 | `f5a20c203fc00903c77c89e5af999916bd9d77f4d086a1f28664dad7c6f4144f` |
| `points_1` | 768 | `75904ae06d9786db30a56bc2f20fd881568eea26d448a8b1783d88c701898a90` |
| `triangles_1` | 1856 | `d632ed56f8b984d159a678f67c42c859f7b5ef8e36f9bf94dd69d7f9b9ca0efe` |
| `twins_1` | 1392 | `d663cf9647bbfc9d3282e598f92a1eb7d02c60331beeb00192c71141387b95f9` |
| `receipt_1` | 32 | `a7414aebdbbef5ea390336ad8822218c65890e7500bf3343e5f5e1ffdb794491` |

## Build observation

The cold release web build completed in 395,617 ms with 14 passes, 35,014
Wasm bytes, 668,928 compiler-reported WGSL bytes, and 887,370 emitted bytes.
The page manifest reports 668,128 bytes across 14 shader artifacts. The prior
manifest reported 685,802 bytes, so the indexed implementation did not create
the feared shader-size regression.

## Remaining gates

- Compare indexed and scalar restoration byte-for-byte over a deterministic
  corpus, not only by shared decision rule and successful convergence.
- Replace serial first-edge selection with a deterministic face-disjoint
  parallel proposal schedule while retaining this indexed implementation as
  the intermediate oracle.
- Measure a distribution rather than a single observation after the browser
  benchmark harness can retain timestamp-query and receipt evidence.
- Canonicalize equivalent resource-binding identifiers during shader emission;
  the two chart restoration shaders remain structurally duplicated.
