# Efficient, interactive patch tessellation experiments and delivery

Execution goal, September 8, 2026. This replaces the active goal's work order,
not the unfinished product requirements. Read this document before resuming.

## Outcome

Deliver reliable, inexpensive tessellation of the expressive triangular and
quadrangular patches users actually author. Build a coherent suite of usable
standalone and gallery examples, not disconnected experiments. Demonstrate
concise, legible, generic, idiomatic Fe throughout: useful types, traits,
effects, actors, first-class obligations and CTFE derivation where appropriate.
The means and the end product both matter.

Investigate a small surface-aware coarse mesh filled by reusable atlas primaries
alongside cheaper fixed arrangements. Do not assume dynamic topology, an affine
correction, recursive Wang tiles, or any other candidate is the winner.

Planning evidence:
- /laboratory/quilting/scratch/patch-experiment-planning-request-20260908.md
- /laboratory/quilting/scratch/patch-experiment-planning-response-20260908.md
- Claude session 249ebcbf-f5b3-43fc-b151-fa1275820c31, especially its late
  corrections about symmetric-only measurements, excluded corners, and shear.

Preserved delivery scope: docs/composition-and-recursive-atlas-goal.md and
docs/quilting-fe-delivery-goal.md. Their unfinished requirements remain in scope.
Recommendations and proposed tolerances are hypotheses until tested. Current
source, commits, running processes and served artifacts outrank old summaries.

## 1. Establish trustworthy fixtures and one measurement path

- Reconcile Claude's active changes without overwriting them. Fix the current
  varied-triangle fixture: p2 is inactive; perturb an active corner and prove
  the geometry actually changed. Record witnesses for every supposed variation.
- Capture actual user failures with controls, camera, seed, resolved counts,
  method and build identity. Include asymmetric corners, unequal weights,
  boundary/corner-near samples, unequal LoDs, displaced centers, and invalid or
  near-pole states. Preserve the distinct interpretation of free-weight patches.
- Reuse production Fe evaluation, placement, packed primaries and permutations.
  Batch exports and compile-once oracle runs should avoid repeated heavyweight
  compiles and scalar cross-runtime calls. Rust/f64 may independently verify
  results; it must not become the production geometry implementation.
- Measure actual finite 3D triangle shape and its worst/low-tail distribution,
  area allocation, differential anisotropy, outer fidelity, shared IDs and
  reversals, interior-to-edge continuity, folds, intersections, coverage and
  sampled geometric approximation error. State numerical limits and skipped
  samples explicitly. Boundary agreement and summed area alone are insufficient.
- Separate surface-space from screen-space targets; use the actual camera and
  transform path for projected measurements. Separate area, anisotropy and
  approximation error. Name trace- versus determinant-normalized metric tests.
- Establish numerical noise and declare gates before comparison. The review's
  suggested percentage improvements are provisional, not product promises.

## 2. Deliver the first useful interactive comparison immediately

Use current two-diagonal/four-fan quad arrangements and the actual one-primary
triangle baseline. Compare placement enabled/disabled with identical boundary
contracts and matched final triangle budgets or explicit quality-cost curves.
Initial target budgets are roughly 256 and 1,024 triangles where admissible;
disclose quantization and use larger budgets only for surviving methods.

One shared viewer provides handles, camera, same-state A/B, coarse overlay,
actual atlas wires, common-scale shape/error heatmaps, worst-cell highlighting,
counts, validity and capture/replay. Preserve a usable baseline even if every
new hypothesis fails. Do not wait for all research to finish before delivery.

## 3. Execute bounded toe tests, then interactive experiments

Each toe test has a precise hypothesis, smallest input set, finite work budget,
falsifying result, acceptance gate and next implementation slice. Failed ideas
receive a reproducible result and stop; they do not generate an endless pivot.

### Eight-triangle square

Test the midpoint diamond plus square midlines: four corner triangles and four
center triangles. First test literal geometric midpoints with uniform boundary
laws. With nonuniform laws, require each new outer vertex to be an existing
master sample. Separately label a half-index sample-anchored variant; it need
not be a geometric midpoint. Outer LoD0 cannot secretly gain a midpoint.

Compare real atlas meshes at matched budgets on asymmetric and difficult quads.
Only after a useful numerical result add the recipe to the shared viewer.
Generalize the current ten-edge-slot/four-draw assumptions narrowly into bounded
coarse plans, preserving legacy adapters and permutation/winding checks.

### Cheap metric-guided existing topology

Score both diagonals and a small bounded fan-center set (initially 3x3), then
assemble actual fine meshes for the best few candidates. Include metric-based
edge counts on frozen topology as an ablation, so a count-allocation win is not
misreported as a topology win. Ship the smaller policy if it suffices.

### Bounded adaptive coarse mesh

Where fixed candidates still fail, test modest coarse budgets up to sixteen
faces. Use limited interior relocation, flips and insertion, with validity and
actual fine-mesh checks. Reuse existing pullback metrics and predicates, without
claiming global metric-Delaunay optimality from a local criterion.

Warm-start stable plans, coalesce edits, bound proposals, use measured hysteresis,
and report unmet demand. Compare against the BEST fixed candidate. Expose frozen
versus automatic topology, step/pause, caps and changed-edge overlays only after
the toe test demonstrates value. No general-purpose remesher as a prerequisite.

### Optional constant-metric/affine probe

Verify normalized metric matrices across genuinely varied patches and held-out
corner-near samples. Constant stretch ratio alone proves neither one matrix nor
optimality. A nonidentity affine map cannot fix all three triangle corners.
Chart changes must demonstrate that they actually change the physical mesh.

Use a successful fit as a cheap metric model, or test an explicitly nonlinear,
boundary-fixed correction with strict fold/coverage gates. This probe is not a
prerequisite and must not displace the useful viewer or adaptive comparisons.

## 4. Shared contracts and cost discipline

- One canonical edge owner, count, sampling map and revision. Directed uses
  reverse integer indices before shared evaluation. Outer subedges reference
  master intervals and IDs, not independently fitted CDFs. No hidden promotion,
  hanging nodes, unsupported arbitrary split fractions or erased failures.
- Separate geometry, sample count, sample distribution and coarse connectivity.
  Publish compatible geometry/maps/draw ranges/diagnostics atomically; retain the
  last valid snapshot while new atlas work is pending.
- Reuse the prioritized Wasm-worker atlas pool. Do not regenerate full atlases
  on edits. Report retained and replaced interior identities honestly.
- Count surface/metric evaluations, table rebuilds, proposals, changed keys,
  bytes and memory. Measure cold generation, planning, spacing, assembly,
  transfer/upload and frame cost separately from compiler time. No invented
  timing claims. Expensive diagnostics run on edits/requests, not every frame.
- Keep camera-only invalidation separate from world-space geometry work;
  projected LoD work may legitimately depend on the camera. Do not assume a
  moved control affects only a small part of an analytical patch.
- No shims: no demo-specific JS geometry, external source generators or required
  baked full-atlas downloads. Shared typed WebIDL realization is the boundary.
  Repair genuine compiler limitations on shared mb2 with focused regressions.

## 5. Apply the evidence to automatic patch sampling

Integrate validated curved-edge spacing and surface/projected LoD choices into
the controllable tri/quad viewer. First test interior LoD/count allocation with
focus fixed; then test focal position and concentration fitting if residual
quality warrants it. Honor authored concentration without oscillating coupled
controls. Keep twist optional and measured, not a presumed remedy.

Use analytic curve formulas only within proven model laws. For projected or
higher-degree curves use bounded measured/analytic alternatives with explicit
admission and error estimates. Poles, clipping and silhouette degeneracy require
honest failure/cap behavior, not enormous hidden allocations.

## 6. Preserve and finish the broader programme

Finish controllable arc-based tri/quad authoring, reliable lifecycle, responsive
controls, URL/state replay, accurate artifact reporting, navigable Fe source and
highlighted WGSL/readable Wasm in standalone AND gallery forms. Refresh existing
examples with shared capabilities instead of accumulating isolated replacements.

Then carry through the earlier bounded research/delivery stages: measured small
Gaussian primaries (local Rust as reference; production Fe), composition and seam
repair comparisons, boundary-preserving recursive/progressive experiments, and
an independent bounded Spectre comparison. Integrate useful results; preserve
honest reject/defer evidence for unsuccessful experimental hypotheses. These
stages do not block the initial interactive comparisons and are not erased by
the immediate work order.

## Completion evidence and working practice

Commit verified focused slices, push shared compiler fixes promptly, and preserve
other agents' dirty work. Use release fe web dev and Chrome MCP, exact artifact
identity, realistic pathological fixtures and independent numerical checks.
Keep user updates concise and distinguish proposed, implemented, served, tested
and measured. No theorem-shaped claims from one attractive screenshot or one
symmetric configuration. Avoid repetitive heavy builds; reuse evidence runners.

Completion requires the shared interactive suite and preserved delivery scope,
toe-test results with explicit decisions, integration of survivors, correctness
and quality/cost evidence, source browsing, clean lifecycle and consolidated
commits. An honest rejected experiment is complete as an experiment; it does not
waive unfinished product requirements. Do not end by shrinking the goal to the
method easiest to make pass.
