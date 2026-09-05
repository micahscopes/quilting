# Device-generated atlas pipeline

The delivery requirement is a fast-starting Fe/WebGPU pipeline for adaptive
blue-noise sampling, Delaunay triangulation, atlas residency, and rendering in
both triangular and square reference domains through edge LoD 8. The existing
24,871,552-byte Rust-generated triangle artifact is a legacy oracle, not an
acceptable download or fallback hidden behind the generated mode.

## One implementation, explicit policy differences

Sampling and topology live in `quilting_atlas`; GPU placement lives here.
Triangle keys use S3; square keys use D4, preserving edge adjacency. Packed
coordinates, boundary identity, and the incircle metric remain typed policies.
Square coordinates must never masquerade as triangular barycentrics.

The work sequence is:

1. Enumerate canonical keys and derive bounded job capacities in Fe CTFE.
2. Materialize deterministic candidates and exact locked boundary samples.
3. Select samples using indexed, parallel conflict resolution; compact on GPU.
4. Insert points with cached local face relocation and conflict-safe claims.
5. Repair unconstrained edges in parallel until the Delaunay receipt succeeds.
6. Publish point/index ranges and readiness for the exact key and generation.
7. Draw the resident ranges, applying the inverse requested symmetry.

Geometry stays in GPU storage throughout. Diagnostic readbacks are test tools,
not pipeline dependencies. First-frame readiness and complete-atlas readiness
are different measurements; generating needed keys first does not waive any
canonical combinations or silently clamp their LoDs.

## Current state and remaining work

- Uniform-density sample queries visit at most 50 candidate slots. Mixed-density
  jobs still need a multilevel radius-aware index; a global coarse radius can
  make their current windows large.
- Exact boundary exclusion needs at most six point tests for triangles and
  eight for squares. Corners are checked separately from constant-density open
  edges. An unchanged exhaustive oracle checks the optimization.
- Both domains now share immutable-generation parallel insertion. A pending
  point follows at most six child faces. This relies on split-only insertion:
  edge flips and vertex motion must not occur during that phase.
- Point arbitration and per-face planning still scan global point lists.
  Replace these with real conflict-safe claims, not racy ordinary stores.
- Stable compaction and scan placement still need work-efficient parallel
  realizations. Preserve deterministic order and checked capacity overflow.
- Indexed Delaunay restoration remains serial. Parallel repair needs explicit
  conflict selection, safe adjacency updates, and a convergence certificate.
- Full-key batching, bounded scratch reuse, atlas publication, and replacement
  of the triangle artifact consumers are not implemented.

## Compiler prerequisite

At Fe `8b7ff12ce`, `GpuIntrinsic` exposes only storage load/store. The
`WebGpuAtomics` implementations in `std::webgpu` are declaration-level panic
bodies, not shader atomic lowering. Efficient claims need a real typed atomic
storage capability through Fe, Sonatina, and Naga. Preserve that distinction
when interpreting apparently complete API declarations.

Do not translate CUDA's permissive last-writer idioms to non-atomic WGSL
stores. Required semantics include atomic initialization/access, deterministic
minimum claims or explicitly nondeterministic exchange, stage-separated
publication, resource identity across helper calls, and portable bindings.
Generic compiler work belongs on mb2, not in handwritten demo JavaScript.

The intended atomic slice extends the existing resource family rather than
adding a second buffer broker: an atomic-u32 element and access witness, typed
atomic load/store/min/add operations, effectful Sonatina object operations,
and Naga atomic storage lowering. Ordinary load/store must not accidentally
grant non-atomic access to that resource. Keep compare/exchange semantics
explicit if added; WebGPU's weak CAS must not pretend to be a strong CAS.
Regression gates must include competing claims, exact resource identity through
helpers, rejection on plain or read-only storage, and no lost operations under
optimization. Do not publish declaration-only panic bodies as implemented APIs.

Sonatina `98696420` now implements atomic-u32 add and unsigned minimum through
effectful object instructions and Naga, including helper calls. The 120 shader
backend regressions, verifier, and unused-result optimizer gates pass. The
actual emitted 420-byte shader passed ten Chromium contention cases up to
16,320 invocations, including counter wraparound. The reproducible diagnostic
and saved receipts are in `fe/web/quad-atlas/atomic-claims.*`.
This is a compiler prerequisite, **not** an atlas performance measurement.
Fe resource-family integration is committed as `4c1dafe2f` on mb2: all 43
release actor/WebGPU regressions pass, including typed atomic add/min through
helpers and rejection of mixed ordinary/atomic access. Atomic load/store and
the claim-based atlas passes remain pending; neither the older core atomic
provider stubs nor the full atlas pipeline is thereby declared complete.

### Claim-based insertion phase contract

The replacement for the two global arbitration/planning scans is deliberately
separate from changing the selected triangulation. Keep the current priority:
the smallest eligible pending point ID wins every face it requests.

1. After location, initialize one atomic claim per current face to `u32::MAX`.
2. Each eligible point performs at most two atomic minimum claims: one for a
   face-interior or boundary-edge insertion, two for an interior-edge insertion.
3. In a later dispatch, a point wins only if **all** its requested faces name
   that point. Never publish half of an interior-edge split.
4. Each face reads its claimant directly. A losing claimant means the face
   remains unchanged; a winning claimant must pass the existing exact split
   and ownership checks before its expansion is accepted.
5. Scan expansions, rebuild the immutable next generation, retire winners,
   and advance. Existing location/invariant failures still prevent publication.

This matches the current preceding-conflict rule, not a maximal independent
set: a point that loses one face can conservatively reserve another for this
round. Progress still follows because the smallest eligible point wins every
face it names. Keep separate dispatches between initialization, claims,
winner publication, and face planning. Relaxed atomics alone are not a
cross-workgroup publication barrier for the other arrays.

Claim/planning work becomes O(points + faces), excluding the separate scan and
rebuild. Atomic contention and required round count still need measurements;
this work bound is not a throughput guarantee. Atomic read/initialization must
be real supported operations, not ordinary racing stores or hidden host work.

`face-claims.verify.test.mjs` exhaustively compares the ownership rule against
the preceding-conflict oracle: 14,641 four-point proposal sets, each in all 24
arrival orders (351,384 schedules). It checks winners, face ownership, progress,
and rejection of half-won interior edges. This finite-model test is not evidence
that the atlas currently runs these claim passes on the GPU.

## Algorithm references and adaptation boundaries

[gCDT, section 5](https://min-tang.github.io/gCDT/files/gcdt.pdf) separates
split-only insertion from Delaunay repair and uses insertion history for local
relocation. That motivates the bounded child-face search here. Its CUDA
selection and wider-integer machinery are not drop-in WebGPU implementations;
retain our exact Q14 predicates and explicit race-free synchronization.

[gDel2D](https://www.comp.nus.edu.sg/~tants/gdel3d.html) provides another GPU
insertion/flipping reference. [PixelPie](https://research.nvidia.com/publication/2013-07_pixelpie-maximal-poisson-disk-sampling-rasterization)
demonstrates highly parallel variable-density sampling, but rasterized coverage
is not automatically equivalent to our exact exclusion and locked boundaries.
Published throughput is feasibility evidence, not a benchmark of this pipeline.

## Acceptance

Record downloaded bytes, shader bytes and compilation time, first usable frame,
sampling/insertion/repair time, complete-atlas time, peak scratch/resident memory,
and warm reuse separately. Exercise cold resources, seed/key changes, device
recovery, maximal mixed-density keys, and both domains through LoD 8.

For every published tile verify boundary identity/order, unique points, positive
orientation, area, Euler relation, edge incidence, no crossings, local Delaunay
legality, and capacity/convergence receipts. Failed or incomplete tiles must not
be published as ready. Small-job success is not the full startup acceptance gate.
