# Quilting Fe: controllable patches and high-quality compositional atlases

Proposed consolidated goal, September 7, 2026. This expands the delivery
programme; it does not declare existing work complete or authorize abandoning
its unfinished controls, patch viewers, tests, or gallery.

## Goal to pursue

Deliver a beautiful, technically honest suite of standalone and gallery-hosted
Fe examples that connects controllable geometric patches, continuously varying
edge sampling, reusable primary tessellations, and carefully measured recursive
blue-noise composition. Build concise reusable Fe machinery rather than isolated
demonstrations. Complete the current delivery checklist first, then carry the
Astra experimental path through implemented comparisons and explicit decisions.

Authoritative supporting checklists and evidence:

- [Current delivery checklist](quilting-fe-delivery-goal.md): every unfinished
  requirement remains in scope, including controllable triangular AND quad
  patches, arc widgets, responsiveness, source browsing and performance gates.
- [Composition experiment](composite-atlas-next-experiment.md).
- [Direction review](/laboratory/quilting/scratch/composition-direction-review-response-20260907.md).
- [Astra recursive-atlas report](/laboratory/quilting/scratch/recursive-atlas-astra-report-20260907.md):
  implementation contracts, algorithms, references, fixtures, costs and proposed
  experimental gates. Recommendations are hypotheses until tested here.

## Order and working discipline

1. Finish the current composition interaction and shared sampling maps.
2. Apply that machinery to controllable triangular and quadrangular patches.
3. Complete/refine the shared gallery, standalone and source-inspection delivery.
4. Establish small, reliable quality/reference measurements.
5. Implement constrained Gaussian primaries, composition comparisons and staged
   seam repair; evaluate their quality/cost crossover.
6. Implement boundary-preserving recursive and progressive comparisons using
   the surviving methods. Run a bounded, independent Spectre exploration.
7. Integrate the useful results into the original examples and publish the
   evidence, limitations and decisions. Do not end with disconnected labs.

Measurement helpers may be built when needed by an earlier stage. Compiler
defects may be repaired when they block it. Neither exception permits another
full-GPU-atlas optimization campaign to displace working interactions.

Use Fe types/traits, derived bindings, FCO/CTFE, sparse GA and typed actors/effects
where they remove real duplication or express useful contracts. Keep browser
realization at the shared typed WebIDL boundary. No demo-specific JS algorithm
shims, external generators masquerading as Fe, or required baked full-atlas
downloads. Rust/CUDA/Python references are research oracles, not the delivered
sampler. Prefer Wasm workers for the current generation implementation; add GPU
or native realizations only for a measured benefit without rewriting semantics.

Commit verified slices frequently. Preserve unrelated changes. Land focused
compiler fixes and regressions on shared mb2, not a parallel durable fork. Use
targeted riff-cat and IR evidence for compiler blockers; record missing diagnostic
affordances. Maintain truthful done/doing/todo and exact served-build identity.

## A. Finish continuously graded composition

- Both two-triangle diagonals and the movable-center four-triangle fan remain
  available. Prioritize the fan for independently adjustable spoke densities.
- Four outer LoDs cover 0–8 without hidden ratio promotion. Automatic interior
  choice starts enabled and preserves separately authored diagonal/spoke values.
  Show resolved segment counts, unmet demand and caps.
- Distinguish edge geometry, number of samples, and distribution of samples.
  Density adjustment redistributes samples continuously ALONG a fixed edge;
  it does not implicitly move the fan center or bend the edge.
- Give every undirected edge one canonical identity, resolution, orientation
  rule and sampling map. Reverse canonical sample inputs before evaluation;
  adjacent uses must not independently fit their maps.
- Derive count and spacing from the same declared density/length measure.
  Integrate it, invert the cumulative measure, and extend boundary maps into
  triangle interiors using shared machinery. Reuse `inverse_length`, sampled
  CDFs, interval maps and `TriangleEdgeMaps`. Distinguish a continuous target
  density from its numerical approximation; measure approximation error and
  refine/interpolate it when visible or relevant to quality.
- Include edge skews, concentration, focal position and twist with neutral/reset
  states and a compensated/uncompensated comparison. Account for the final map
  when claiming surface/screen-space compensation; label reference-domain
  policies honestly. Measure interior distortion, density residual and folds.
- Keep per-child canonical/permutation witnesses and compact draw ranges,
  correct winding and shared final boundary evaluation. Persistent prioritized
  mixed-primary generation must not restart on edits or wait for the full atlas.
- Publish geometry, spacing and diagnostics as one compatible visible snapshot.
  Keep the previous one while pending. Use acknowledged efficient uploads with
  recovery, not repeated full-prefix transfers or mutable newer maps paired
  with older draw state. Measure staging work as well as GPU uploads.

Acceptance: exhaustive planning and reversal checks, nonidentity warped seam
tests, validated CPU/Wasm and GPU map agreement, browser interaction/pathology
fixtures, and measured first-visible/edit/frame/memory costs. A source check or
one attractive screenshot is insufficient.

## B. Deliver controllable patches and the existing example suite

Complete all patch/UI/gallery requirements in the existing delivery checklist.
In particular:

- Carry existing oriented Bézier arc endpoint/middle-point widgets into BOTH
  triangular and quad Patch views. Reuse `arc_curve_explorer` interaction and
  `quilting_patch::arc_authoring`; extract shared widgets rather than substituting
  scalar weight sliders. Higher-degree authoring remains an explicit follow-on,
  not a prerequisite for using the existing widgets.
- Verify actual shape/weight semantics and supported compatibility constraints.
  Resolve supported constraints by default with labeled exploration otherwise.
  Separate reparameterization from shape changes and positional agreement from
  tangent continuity. Ground exact-child claims in representation/degree tests.
- Apply the shared sampling maps to actual curved boundaries and surfaces.
  Exercise unequal weights, twists, poles and near-degenerate states safely.
- Finish stable camera/picking/dragging, AA and subtle optional wires, resets,
  validated URL/history state, and responsive collapsible scrollable controls.
- Refresh `tessellation_warp`, `two_triangle_uniformization` and
  `paired_triangle_fans`. Make each independently usable and gallery-accessible.
- Use Fe-authored shared mb2 gallery/source inspection: navigable Fe, highlighted
  WGSL and appropriately readable Wasm, accurate artifact sizes and inactive
  example lifecycle. Avoid marketing clutter and unexplained jargon.

## C. Establish trustworthy measurement and reference fixtures

- Keep separate receipts for boundary conformity, mesh coverage/legality, point
  statistics, density/metric approximation and progressive point identity.
- Add unique-point extraction, pair-distance/exclusion checks, spectrum and
  directional/repetition diagnostics, quantization comparison and gradient
  reference tests. Use adequate numerical precision; do not measure tiny
  spectral differences with unvalidated rendering approximations.
- Declare domains, metrics, density normalization, point counts, fixed points,
  seeds, iteration budgets and stopping conditions. Compare matched workloads
  and ensembles, not unrelated screenshots or one favorable seed.
- Use the existing local Rust Gaussian implementation centrally: capture its
  useful behaviors as named variants, compare small kernels with independent
  math and original author code, and port the selected semantics into Fe.
  Verify differences rather than dismissing the existing implementation or
  silently equating it with another paper's algorithm.
- Record cold generation, optimization, triangulation, assembly/repair,
  serialization, transfer, GPU upload, frame cost, memory and cache behavior
  separately. Attribute build time separately from application startup.

## D. High-quality small primary tessellations

Implement Astra E1 as a focused primary laboratory:

- Compare current sampling, reproduced local GBN behavior and a clearly defined
  constrained Gaussian objective. Start with a small uniform/mixed triangle
  family and a few variants, not the full combinatorial product.
- Fix canonical boundary samples while optimizing interior points in the
  correct metric. Validate finite-domain attraction/integration and gradients,
  boundary admission, adaptive density and optional hard exclusion separately.
- Compare synchronous and serial update rules as distinct algorithms. Bound
  work and expose convergence/stop reasons; no assumed optimizer convergence.
- Quantize into the actual representation, audit collisions and geometry, and
  reuse the existing triangulator. Measure whether gains survive constraints,
  quantization and subsequent composition.

Advance a primary method for its measured benefit, not its name. Preserve a
useful comparator and a reproducible negative result if a constrained variant
does not improve the intended workloads.

## E. Composition, seam quality and cost

Implement Astra E2/E3 through increasing-cost comparison modes:

1. Repeat primaries; vary independent boundary-compatible primaries; then test
   jointly designed corner/edge neighborhoods where measurements justify them.
2. Canonically weld duplicate boundary uses and establish adjacency.
3. Repair connectivity without moving points, in the final declared metric.
4. Resolve cross-seam point conflicts/refill locally as a separate operation.
5. Optimize point neighborhoods around seams AND their incident corner stars
   against a frozen halo; test small global low-frequency corrections separately.

Known seams can avoid cut-finding, not all statistical repair. Do not claim edge
flips improved an unchanged point-set spectrum. Warps may make previously legal
interior edges suspect. Fixed straight seam points may remain statistically
visible even after connectivity repair. Report conflicting constraints honestly.

Start with one worker owning a small repair region. Parallelize only with clear
point ownership, read halos, immutable epochs and barriers. Measure actual active
fractions and the crossover with direct regeneration; Gaussian neighborhoods
can occupy most of a small primary. Prefer a larger primary or whole-patch solve
when narrow-seam repair loses its advantage.

Keep quality/cost thresholds declared before evaluating ensembles. Astra's
suggested tolerances are starting proposals, not automatic product promises.

## F. Boundary-preserving recursion and refinement

- First test legal uniform similarity recursion; then mixed-density requests
  with protected outer contours, interior refinement or explicit transition
  constructions. An LoD0 boundary cannot silently acquire a midpoint.
- Child intervals reference the master's exact interval and map; no independent
  refitting, hidden promotions or hanging-node cracks. Distinguish local LoD,
  absolute boundary count and hierarchy depth in data and UI.
- Preserve hierarchical coordinates/IDs or explicitly audit wider/quantized
  coordinates and predicate bounds. Do not silently reuse Q14 arithmetic outside
  its proven envelope.
- Distinguish reusable discrete snapshots from nested point refinement. Claim
  progressive inclusion only with parent-to-child identity/position witnesses
  and per-level quality checks. Compare frozen-parent insertion with joint
  optimization; a crossfade is not a nested point process.
- Keep continuous density warping, count changes and recursive subdivision
  distinct but composable. Measure artifacts across their transitions.
- Use typed request/cache keys covering sampler, metric, boundaries, neighborhood
  state, variant and algorithm version. Generate lazily; do not eagerly explode
  LoD × colors × corners × shapes × variants.

Deliver a recursive-density explorer showing hierarchy, protected intervals,
retained/new samples, quality and cost, using the same shared controls/rendering.

## G. Bounded Spectre branch and integration

Build a small independent hierarchy/layout and point-rule comparison from
verified constructions. Test against simpler layouts with finite-window and
directional statistics. Aperiodicity alone is not blue noise. Mesh integration
requires a demonstrated benefit; a measured rejection is an acceptable result
of this bounded branch, not a reason to invent further complexity.

Bring surviving primary/composition/repair methods back into the existing
composition and Patch examples. Provide a coherent comparison index linking
the primary, seam, recursive-density and Spectre explorers, source and evidence.

## Completion and stopping rules

The goal is complete only when:

- All current delivery requirements have actual working-browser and test
  evidence, including controllable tri/quad patches and the gallery/source UX.
- The staged primary, composition, seam and recursion comparisons exist and
  have reproducible quality/performance results and explicit method decisions.
- The bounded Spectre comparison has an evidenced adopt/defer/reject outcome.
- Useful results are integrated; unsuccessful experiments have preserved
  fixtures and clear explanations rather than being mislabeled successes.
- Tests cover planning/permutations, warped boundaries, winding/coverage,
  density error, point quality, refinement identity, stale work and recovery.
- Release `fe web dev` and Chrome MCP verify the real served artifacts. First
  usable views meet the existing under-one-minute gate; resident edits target
  subsecond latency. Disclose and address misses rather than redefining metrics.
- Resources are disposed correctly, inactive examples are quiet, source/binary
  sizes are truthful, commits are consolidated and status is evidence-backed.

Failure of an experimental hypothesis is a valid scientific outcome only after
its bounded implementation/comparison has actually been carried out. It does
not waive unfinished product delivery. Do not claim Gaussian quality, arbitrary
mixed-LoD recursion, global Delaunay legality or negligible repair cost without
their separate evidence. Broader remeshing, Blender sync and full direct GPU
quad-atlas optimization remain outside this goal.
