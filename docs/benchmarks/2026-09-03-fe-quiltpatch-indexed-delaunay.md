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

The remaining quadratic initialization was then replaced by a sparse
vertex-incidence index: one outgoing-edge head per resident point and one next
link per resident half-edge. Reverse-edge matching now visits only the target
vertex's outgoing list, while retaining the original provider's exact match
count and reciprocal-twin validation. Each chart adds a 3,272-byte typed scratch
resource; together with points, samples, triangles, twins, proposals, and the
receipt, each restoration pass uses seven storage bindings under WebGPU's
portable limit of eight.

| Warm phase with sparse incidence | Time |
| --- | ---: |
| Chart 0 Delaunay restoration | 20.9 ms |
| Chart 1 Delaunay restoration | 4.7 ms |
| Exact 16-pixel gesture release | 52.0 ms |

The default and moved triangle hashes remained byte-identical to both prior
indexed checkpoints. The same seven-state deformation sweep completed in
29.1–39.5 ms per full frame, retained the exact prior flip counts, and reported
zero invariant failures on both charts. The first pullback-mode activation
after a page reload measured 1,463.3 ms; this cold activation cost remains to
be separated into browser pipeline compilation and first-dispatch work.

The sparse-incidence release build took 363,154 ms and emitted 14 passes,
35,258 Wasm bytes, 687,407 compiler-reported WGSL bytes, and 909,256 total
bytes. A discarded open-address-index prototype also exposed a Sonatina SPIR-V
structurizer regression for a valid outcome-controlled probe loop; the
incidence formulation is both simpler and better matched to mesh topology, so
production no longer depends on that rejected shape.

## Prepared pipelines and singular-simplex admission

Quilting commits `1bfa30c` and `3a4268e`, using the Fe pass-preparation
candidate at `3b368f0e9`, move the dormant pullback subgraph's physical pipeline
creation to a visible idle opportunity. Fe-authored pass activation remains
the only execution decision. On the exact release artifact, live entry took
163.2 ms; after a 2.5-second idle window every one of the twelve pullback
compute/raster passes was resident. Switching from the analytic atlas to the
pullback sampler then incurred no first-use pipeline error or driver-scale
pause.

A 20-step geometric-handle drag produced 22 frames at an 18.8 ms mean interval
(44.8 ms maximum, including the first selected frame). Browser-side command
encoding was 0.3--0.4 ms for held-drag frames. The release frame, which alone
reactivates sampling and Delaunay settlement, encoded in 2.4 ms in this warm
observation.

The same browser sequence reproduced the former black-frame failure by moving
one normalized Clifford weight to exactly zero. Its projective denominator is
then undefined at the corresponding corner. Previously only that vertex used
the off-canvas sentinel, so its two finite neighbors formed a viewport-sized
spurious triangle. The shared Fe geometry predicate now admits or rejects all
three vertices as one coherent simplex in every sampler. At weights
`[0, 4/3, 4/3, 4/3]`, the singular incident triangles form an honest hole;
all handles remain visible and draggable, and Chromium reports no WebGPU or
runtime error.

The final bundle contains 14 passes, 35,352 Wasm bytes, and 703,756 manifest-
reported WGSL bytes. A fresh second `fe web dev` process reused the persistent
render bundle in 352 ms, establishing that the earlier cross-process cache miss
is no longer present for this exact source/dependency closure. Incremental
lowering after a source edit remains unresolved: the two measured fresh
lowerings took 313.3 and 339.8 seconds.

## Atlas-filled topology and compact projective sampling

Commits `2d55aa7` and `6455863` make the settled pullback topology the macro
carrier for canonical atlas patches. Two typed indirect draw commands remain
GPU-resident: the vertex count comes from the selected equal-edge atlas patch,
and the instance count comes from each chart's topology receipt. There is no
CPU readback or JavaScript draw scheduler in this path.

The first deformation-aware sampler measured area and exclusion in the
Euclidean affine quotient. An adversarial normalized weight vector
`[0.003549, 0.107878, 0.038364, 3.850209]` exposed the resulting coordinate
singularity: pullback coefficients reached approximately `8.1e5`, proposals
collapsed into a thin band near the affine pole, and endpoint-averaged
exclusion rejected most of the band.

The replacement samples the normalized homogeneous Clifford lift
`[F(s,t) : W(s,t)]` in the round metric on real projective space. The lift
remains finite when the affine denominator vanishes, while direct projective
chord distance supplies Poisson exclusion. The exclusion radius is derived
from the selected atlas edge density and the half-turn diameter of real
projective space; it is no longer expressed in scene world units.

Two exact browser states were retained as adversarial fixtures. The first uses
a `333:1` corner-weight ratio; the second combines a roughly `1085:1` ratio
with displaced, twisting controls. A DevTools-only `COPY_SRC` probe read the
resident results without changing application source. Both charts in both
states reported valid sampling, construction, and restoration. Every used
triangle had legal indices, positive winding, nonzero fixed-point area, and
the signed and absolute area sums both equaled the Q14 reference area
`268435456` exactly.

| Stress state | Chart receipts | Complete GPU frame |
| --- | --- | ---: |
| `333:1` weights | `[1,43,24,1,60,1,78,0]`, `[1,24,24,1,22,1,51,0]` | 31.1 ms |
| displaced `1085:1` controls/weights | `[1,40,24,1,54,1,76,0]`, `[1,24,24,1,22,1,57,0]` | 24.6 ms |

The corresponding draw commands were `[1224,60,0,0]` and `[1224,22,0,0]`
for the first state, then `[1224,54,0,0]` and `[1224,22,0,0]` for the second.
Solid and wire captures showed coherent surfaces without the earlier oversized
affine-sampler wedges. Small omitted simplices remain where the visible affine
quotient itself is undefined; coherent triangle admission prevents those
points from becoming viewport-sized sentinel triangles.

Multiplying all four weights in the first state by `7.25` preserved both
receipts, both draw commands, and both triangle prefixes exactly. One locked
boundary point moved by four Q14 units because a floating-point inverse-arc
tie took the neighboring bisection branch; topology and presentation were
unchanged. The denotation is mathematically scale invariant, while exact
fixed-point tie canonicalization remains a possible determinism refinement.

That release development build emitted 18 passes, 35,362 Wasm bytes, 776,403
compiler-reported WGSL bytes, and 1,039,139 total bytes. Lowering took 426,576
ms and the complete build took 450,327 ms. Incremental lowering latency and
peak compiler memory remain material authoring-toolchain problems.

## Terminating projective topology restoration

A real pointer gesture exposed a flaw hidden by the two settled-state receipts
above. With the displaced `1085:1` state, moving `p11` to
`[-0.860737, -1.189465, 1.702456]` made chart 0 report
`[1,39,24,1,52,0,256,0]`: construction and every topology invariant remained
valid, but the varying pullback-metric edge rule oscillated until it exhausted
the 256-flip budget. The presentation incorrectly treated optimization
nonconvergence as structural invalidity and suppressed the complete chart.

The replacement separates those meanings. Receipt word 5 reports restoration
convergence; receipt word 7 reports invariant failure. Presentation requires
valid construction and zero invariant failures, so a safe topology cannot
become a black frame merely because an optional optimizer reaches its budget.
The edge policy itself is now deformation-aware through a terminating global
objective: a strict flip must replace the current diagonal with a shorter
chord between normalized projective representatives. Every such flip reduces
the sum of resident interior-edge chord lengths. A fixed tolerance and the
canonical atlas diagonal order settle floating ties without reverse flips.

The exact formerly failing state now reports
`[1,39,24,1,52,1,84,0]` and `[1,24,24,1,22,1,52,0]`, with indirect draws
`[1224,52,0,0]` and `[1224,22,0,0]`. A larger three-step pointer drag reports
`[1,38,24,1,50,1,74,0]` and `[1,24,24,1,22,1,53,0]`; both draws remain live.
For the formerly failing state, every used triangle has legal distinct indices
and positive winding. Each chart's signed and absolute Q14 double-area sums
are both exactly `268435456`, proving one-cover topology without overlaps or
holes in the parameter chart. Chromium reports no WebGPU or runtime error, and
the previously black user tab renders the preserved state after reload.

Candidate and resident projective samples now store only the normalized
eight-coordinate representative plus one conditioning bit. The obsolete UV
parameter and three pullback-tensor lanes were removed. The resulting release
development build still has 18 passes and 35,362 Wasm bytes, but falls to
747,779 compiler-reported WGSL bytes and 1,010,517 emitted bytes. Lowering took
396,308 ms; the complete build took 424,472 ms.

## Ranked boundary quantization stress gate

A deterministic `0x6d2b79f5` sweep exercised 64 independently twisted control
cages and normalized positive weight vectors ranging through approximately
`17.8 million:1`. Under the first terminating-restoration artifact, 48 cases
failed during constrained construction. Every failed chart contained duplicate
locked boundary points; no construction failed without a duplicate. One
reduced `562:1` case mapped consecutive boundary ranks to
`[16382,2,0]`, `[16382,2,0]`, then `[16378,6,0]`, `[16378,6,0]`.

The projective arc inverse itself is monotone, but independent Q14 rounding
could collapse adjacent quantiles. The fixed mapping reserves one integer lane
per requested rank while allowing the measured inverse to occupy every
remaining lane:

`q(rank) = rank + floor(q_raw * (16384 - resolution) / 16384)`.

Endpoints remain exact. For monotone `q_raw`, consecutive ranks are strictly
ordered even when several raw quantiles round to the same coordinate; the
maximum displacement at resolution eight is eight Q14 lanes.

Replaying the identical sweep produced zero duplicate boundary points, zero
construction failures, zero restoration failures, and zero dead indirect
draws. All 128 charts converged; the largest restoration used 117 of 256
permitted flips. Complete GPU frames averaged 4.205 ms and reached 7.9 ms at
maximum. A second readback audited all 4,156 resident triangles: every index
was legal and distinct, every winding was positive, and every chart's signed
and absolute Q14 double-area sums were exactly `268435456`. The smallest
nonzero double area was 2.

The final release development build emitted 18 passes, 35,362 Wasm bytes,
748,310 compiler-reported WGSL bytes, and 1,011,048 total bytes. Lowering took
422,875 ms and the complete rebuild took 427,463 ms.
