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
