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

## Interactive follow-up

Commit `6374c7c` adds a resident illegal-edge proposal cache and separates the
pullback raster effect from topology settling. A control drag now deforms the
last valid parameter-space topology immediately; sampling and Delaunay repair
run once the gesture ends.

One scripted `p10.x = 0.05` browser transition measured 5.2 ms while dragging
and 122.7 ms when released. The drag preserved both valid topology receipts.
The settled state retained 64 and 26 points rather than collapsing to the 24
locked boundary points, which had been the earlier behavior whenever a free
edit introduced a grade-three Clifford residual.

An isolated warm pass observation for that edited state was:

| Pass or phase | Time |
| --- | ---: |
| Candidate pullback samples | 2.9 ms |
| Sampling initialization | 2.5 ms |
| 64 proposal/retire/advance cycles | 7.8 ms |
| Point compaction | 2.4 ms |
| Resident point samples | 2.4 ms |
| Initial constrained topology | 2.4 ms |
| Chart 0 Delaunay restoration | 171.4 ms |
| Chart 1 Delaunay restoration | 29.0 ms |

These remain single-run wall-clock observations under concurrent GPU use. The
dominant remaining runtime cost is the sequential, deterministic flip chain,
not candidate sampling, command encoding, or transfer. Submitting the 64
sampling cycles separately measured 6.0 ms versus 3.4 ms in one command
buffer, too small a difference to explain the complete frame.

The same gesture also exposed a distinct correctness failure: chart 0 admitted
66 points into a 64-point resident buffer. Compaction correctly reported
overflow, but presentation then suppressed the whole chart. The resident
capacity is now 128 points, with triangle capacity derived from the 24-vertex
boundary ring by Euler's disk relation. The formerly failing state now reports
`[1, 66, 24, 1, 106, 1, 213, 0]`. A wider control-displacement sweep admitted
58–71 points and kept both charts valid with zero invariant failures. Ordinary
loops still use the actual resident counts; the additional capacity is storage
headroom, not mandatory work.

The default state retained the exact validated triangle prefixes above after
the capacity increase. Its 1,856-byte prefixes hash to the same
`76ee7da40367b5874cee998f561cd90e2f9b9dad5e95dd479209a2e7a30ad859`
and `d632ed56f8b984d159a678f67c42c859f7b5ef8e36f9bf94dd69d7f9b9ca0efe`
values.

The final fresh release-server build reported 14 passes, 35,256 Wasm bytes,
660,039 WGSL bytes, and 880,304 emitted bytes in 348,792 ms. The manifest
reports 659,239 bytes across 14 shaders. The compiler process reached roughly
11 GB resident memory during repeated builds, and restarting `fe web dev`
missed the populated render cache. Persistent cross-process cache reuse and
incremental pass-level lowering are therefore separate, measured authoring
toolchain needs.

## Packed proposals and local twin repair

Commit `9670166` packs the resident illegal-edge membership cache from one
`u32` per half-edge to one bit per half-edge. Each chart's proposal resource
shrinks from 2,760 bytes to 88 bytes. The scalar restoration invocation owns
the cache, so read/modify/write is deterministic and requires no atomics.
Default triangle-prefix hashes remained byte-identical. Packing alone reduced
one chart-1 observation from 13.9 ms to 4.9 ms, but chart 0 remained 163.2 ms;
proposal storage and empty-slot scanning were not the dominant chart-0 cost.

The next measured checkpoint retains the six twin slots touching a face pair
before an edge flip. A flip preserves its four-edge outer boundary, so every
valid post-flip twin must be either one of the six rewritten slots or one of
their six former twins. Reconnecting against that bounded neighborhood
replaces the previous all-half-edge search after every flip.

| Warm phase after local repair | Time |
| --- | ---: |
| Candidate pullback samples | 13.3 ms |
| Sampling initialization | 2.5 ms |
| 64 proposal/retire/advance cycles | 3.1 ms |
| Point compaction | 2.4 ms |
| Resident point samples | 2.4 ms |
| Initial constrained topology | 2.3 ms |
| Chart 0 Delaunay restoration | 32.7 ms |
| Chart 1 Delaunay restoration | 4.9 ms |

The exact 16-backing-pixel `p10` gesture measured 33.4 ms while held and 90.7
ms on release, versus 219.4 ms on release before local repair. The resulting
receipts were `[1, 66, 24, 1, 106, 1, 213, 0]` and
`[1, 26, 24, 1, 26, 1, 47, 0]`. Their used triangle byte ranges retained the
pre-change SHA-256 values
`f300f7f86d22fb316503cf64d4d2388cb089e070ea6e6b01738d8858cfdc28f6`
and `3b3515a94a7665c15220f5b74e6f926956542b80a19f5756944280dc271ab034`.

Seven larger `p10` deformations admitted 58–71 points on chart 0 and converged
in 63.2–110.6 ms for the complete frame. Both charts remained valid with zero
invariant failures, and chart-0 flip counts exactly matched the prior indexed
oracle: 221, 192, 237, 171, 240, 174, and 228.

The fresh release-server build took 333,464 ms and emitted 14 passes, 35,256
Wasm bytes, 680,501 compiler-reported WGSL bytes, and 900,764 total bytes. A
watch-loop defect was observed separately: after a dependency diagnostic,
`fe web dev` served the last-good artifact but did not consume the corrected
dependency event, requiring a server restart. This is a tooling/cache issue,
not part of the runtime timing above.
