# Quilting Fe: compositional tessellations and controllable patches

Established September 7, 2026, from the user's renewed direction and the
xhigh composition review. This is the implementation checklist for the active
goal, not a claim that the items below have shipped.

## Outcome and order

Deliver a beautiful, understandable suite of standalone and gallery-hosted Fe
examples in which users can explore tessellation composition, independently
control edge detail and sampling, and directly shape both triangular and
quadrangular patches. The examples must demonstrate useful Fe abstractions,
not merely attractive output backed by ad hoc code.

Order matters:

1. Finish the missing interaction in the current composition explorer.
2. Reuse the same machinery to finish controllable tri and quad patches.
3. Refresh existing examples and integrate the gallery/source experience.
4. Validate, document and consolidate the result; keep deeper experiments
   clearly separate from delivery requirements.

Do not substitute another attractive baseline for either of the first two
milestones. Do not return to full direct quad-atlas/GPU optimization as a
prerequisite. A composition of triangles covering a quad is a legitimate
method, but must not be mislabeled as a directly generated quad atlas.

## Starting truth

### Current checkpoint: shared boundaries, interior controls and visible diagnostics

- Shared directed sampling maps now have an executed Fe/Wasm-versus-WebGPU
  regression: 66,820 queries, 267,280 scalar comparisons, maximum error
  5.96e-8. Reversed GPU canonical parameters and positions were bit-identical.
- Interior concentration, twist and normalized focus controls preserve boundary
  positions. Neutral controls match the bypass; strong warps can still fold the
  straight rendered mesh. These controls are not an automatic uniformizer.
- Native actor-owned readouts show resolved diagonal/spoke counts, unmet demand,
  capped edges and pending tiles. Fe computes these with the visible snapshot;
  the host only presents them and rejects input edits. The manual requests
  remain separate and survive automatic/manual switching.
- Chrome65 verified both automatic and manual diagonal counts, the uniform-LoD8
  cap (256 segments versus approximately362 required), and an unequal-density
  fan. No browser errors; the outer document has no extra vertical scroll.
- Shared mb2 `a7c0804ef` supplies the reusable Fe readout declaration and native
  host realization, with compiler ownership gates and70 runtime tests. Release
  composition Wasm policy/retention tests pass in24.52s, including cap checks.
- Independent edge skews now redistribute cumulative density along each of the
  ten canonical boundaries without moving endpoints or changing segment counts.
  Release Wasm checks and200,460 actual GPU boundary queries passed; reversed
  GPU positions remained bit-identical. All native inputs were exercised in
  Chrome65. Details and limitations: `docs/composition-edge-skew.md`.
- Still pending: measured interior distortion and edit/
  upload costs, acknowledged suffix uploads, avoiding redundant staging, and
  the controllable patch/gallery milestones. The current fan visibly contains
  abrupt density transitions and skinny triangles; boundary agreement alone
  does not establish good interior distribution.

Evidence: the readout follow-up and preceding executed checks in
`docs/composite-spacing-browser-receipt-20260907.md`. The following checkpoints
are historical; their then-pending items are superseded where stated above.

### Earlier checkpoint: rendered shared spacing

- The composition viewer now publishes ten canonical 17-sample density tables
  with its compatible visible scene and applies them to shared boundaries and
  triangle interiors. Compensation varies sample positions along fixed edges;
  it does not move the authored fan center. The four-triangle fan is the default.
- Release Wasm tests passed all6,561 outer combinations and retained-state
  checks for both diagonal and fan recipes (9.57s test runtime). These test the
  represented CDF, not exact continuous integration or arbitrary fold freedom.
- Chrome65 rendered uniform4/4/4/4 and unequal1/6/6/1 fans. The compensation
  toggle visibly changes the unequal case. Each toggle state had zero red fold
  pixels and zero exposed-background pixels in a1,036,324-pixel interior crop.
  This is a representative screen-space check, not exhaustive mesh acceptance.
- A diagnostic GPU copy of all170 spacing samples matched an independent
  midpoint-integral calculation to max absolute error4.7362e-6; all ten tables
  were strictly increasing. Diagnostic JavaScript is not application machinery.
- Compiler fixes are pushed in Sonatina9ff01864 and shared mb2 94d3fe9f7.
  The real build has3 passes,196,297 WGSL bytes and358,989 parent Wasm bytes.
  Site compilation took57.9s; the new dev server reused its cache in196ms.
  Generation, edit and frame latency still need separate measurements.
- Standalone startup now uses the existing declarative `fe-surface state="live"`
  API. Shared mb2 fix49c4abaae prevents initial custom-element attribute
  reactions from booting competing Wasm/task instances. All68 runtime tests
  pass. A fresh page load entered live mode with4,296 vertices at452ms, without
  manual activation or console errors (warm browser shader cache, new workers).
  The automatically rendered square also passed the interior pixel census.
  Screenshot capture stalled; actual canvas pixels were inspected through
  Chrome MCP instead. General gallery/inactive lifecycle acceptance remains.
- Still pending: exact GPU warped-seam
  fixtures, distortion/error diagnostics, acknowledged suffix uploads, richer
  density controls and the remaining patch/gallery delivery requirements.

Evidence: `docs/composite-spacing-browser-receipt-20260907.md` and the logs it
links. The following earlier checkpoint records the pre-spacing baseline.

### Earlier checkpoint: shared requests and priority

- Implemented `SquareEdgeLods` and per-child canonical/permutation resolution;
  tested all6,561 outer requests across both diagonals and the interior fan,
  with distinct interior values and explicit invalid-input rejection. This
  verifies request planning, not rendered seam positions or spacing quality.
- Implemented promotion of unclaimed worker jobs plus semantic-key lookup;
  running leases and retained tiles do not move. The release Wasm pool test
  passed, including promotion with live work, idempotent requests, retained
  completion and existing cancellation/recovery ownership checks.
- Evidence: `/laboratory/quilting/scratch/composite-requests-test-20260907.log`
  (two Fe tests passed) and `composite-priority-regression-20260907.log`
  (release Wasm pool regression passed, 14.49s test runtime, not an atlas benchmark).
- Mixed-edge source now includes per-child resident/draw descriptors, all165
  prioritized triangle primaries, four outer controls and automatic/manual
  interior requests. The candidate is served by release `fe web dev` on38331.
  Chrome66 shows the mixed fan with outer2/4/7/7 and a complete165-tile cache;
  no console errors. This is partial visual acceptance, not the full interaction
  and pathology gate. Uneven interior transitions remain visible and unresolved.
- Release Wasm policy/retention acceptance passed (8.57s test runtime): all9
  uniform-density levels, exact outer integrals, preserved manual settings and
  keeping the previous scene while a changed request lacks a tile. Evidence:
  `/laboratory/quilting/scratch/composite-model-regression-20260907.log`.
- The candidate exposed flattened-state resource positioning in the compiler.
  Fix8b9a43050 is committed and pushed on shared mb2, with a focused regression
  and15 existing resource tests passing. Release CLI rebuilt successfully;
  the candidate compiled in65.8s (development cache hit218ms), with3 passes /
  30,193 WGSL bytes. These are build metrics, not generation/frame benchmarks.
- The automatic-demand path now builds reusable17-sample cumulative density
  tables; its total is the same16-midpoint estimate previously used for LoD.
  `InverseCumulativeMap` adapts these tables to the existing interval-map API.
  Release Wasm checks passed all6,561 outer combinations with257 fractions
  each: strict monotonicity and represented-CDF residual below1e-6, plus an
  unequal-density case requiring midpoint skew. This is not an analytic
  integration-error bound and does not yet establish rendered compensation.
  Evidence: `scratch/composite-cdf-regression-20260907.log` in quilting
  (9.89s test runtime). Next: retained table publication and shared boundary /
  interior application in the viewer.

- Independent edge selection now reaches the served viewer; spacing
  compensation and acknowledged suffix uploads remain unfinished.
- Shared directed-boundary recipes and tests exist; those tests do not yet
  establish equality of final warped seam positions.
- Fe/Wasm generation, typed workers, retained tile output, canonical triangle
  keys and permutation witnesses exist. Their use in the new interaction must
  still be verified end to end.
- Existing interval/CDF maps, interior warps, patch/arc examples and derived
  parameter bindings are reusable foundations, not reasons to recreate them.
- The review was source-based. Neither HTTP 200 nor a successful build is
  browser acceptance.

## 1. Complete compositional tessellation interaction

- [ ] Confirm the exact served build and preserve reproducible current views;
  use the user's Chrome MCP and release `fe web dev` without disturbing other
  tabs or replacing a working page with an unverified build.
- [ ] Support both two-triangle diagonals and a four-triangle movable-center
  fan, with a visible decomposition and the actual rendered triangulation.
- [ ] Expose four independent outer-edge LoDs 0–8: 1–256 segments, not a fixed
  ladder of selected combinations or a hidden ratio restriction.
- [ ] Provide an initially enabled **automatic interior LoDs** checkbox. Store
  authored manual diagonal/spoke values separately; toggling automatic mode
  must not destroy them. Display resolved counts and caps.
- [ ] Give each undirected boundary one identity, resolution and parameter rule;
  every child references that record with explicit direction.
- [ ] Give each child its own canonical primary key, permutation, retained tile
  and compact draw range. Map ring edges AB/BC/CA to atlas lanes c/a/b explicitly.
  Preserve winding under odd permutations.
- [ ] Keep a persistent primary cache. Prioritize requested missing keys and
  deduplicate work. An all165-triangle staging path is acceptable if the visible
  request is prioritized and measurements meet the interaction budget. Never
  wait for all1,200 direct triangle/quad entries before displaying a result.
- [ ] Respect the queue's current reverse issue order; add a regression for
  visible-request priority. Avoid pool recreation and 64MiB allocation per edit.
- [ ] Avoid repeatedly transferring all prior geometry on every completion.
  Establish an acknowledged publication/update policy with correct recovery.
- [ ] Publish a compatible composition atomically: no transient old/new seam
  mixtures, mismatched counts/indices, or spurious high/low-LoD flashes.
- [ ] Keep the last compatible rendering visible while requested tiles load,
  with an honest pending indicator rather than silent fallback to wrong LoDs.

## 2. Shared density, edge spacing and warp machinery

- [ ] Implement and test a named initial automatic policy based on interpolated
  directional segment densities and integrated edge demand. Use segment counts
  `2^LoD`, not exponents as linear densities. Account for diagonal/spoke lengths
  and center movement rather than using a blanket maximum.
- [ ] Test outer integrals, the uniform square's diagonal/spoke demand, dyadic
  rounding and the LoD8 cap. Show unmet demand rather than claiming a capped
  result achieved the requested spacing.
- [ ] Derive a single monotone inverse-cumulative spacing map per shared edge.
  Reuse `CumulativeLengthSamples`, `inverse_length`, `IntervalSamplingMap` and
  `TriangleEdgeMaps`; separate numerical approximation from exact boundary
  identity and analytic special cases.
- [ ] Add independent edge skews, movable center/focus, concentration and twist.
  Offer useful neutral/reset states and compare compensated/uncompensated views.
- [ ] Apply shared boundary maps through the canonical boundary bypass before
  the common parent map. Adjacent children must evaluate identical boundary
  positions, not just agree at endpoints.
- [ ] Include the final map's deformation in the chosen measure when claiming
  compensation for it. Label reference-domain density separately from surface
  or screen-space density. Define the coordinate convention explicitly.
- [ ] Preserve shared boundary conditions while extending maps into interiors.
  Measure spacing/area residuals and anisotropy; do not promise that boundary
  uniformization automatically solves interior twisting.
- [ ] Display signed triangle orientation/fold and coverage failures. Distinguish
  a valid continuous warp from its possibly invalid coarse straight mesh.
- [ ] Keep geometry/topology/placement/metric policy/backend realization separate
  using small useful Fe types and traits, not a new general meshing framework.

## 3. Return to controllable triangular and quadrangular patches

- [ ] Inspect current tri/quad patch, curve and weight controls, their tests and
  cited construction papers. Recover the actual mathematical family and control
  meaning before changing weights; do not equate scalar rescaling with arbitrary
  curvature control.
- [ ] Finish usable triangular and quadrangular Patch views, independently
  accessible, with orbit/pan/zoom, visible/selectable/dragable controls, sensible
  picking/depth order, efficient AA, subtle optional wires and reset presets.
- [ ] Make endpoints and edge arc controls intuitive. Reuse the oriented circular
  curve explorer and midpoint/arc construction where it fits the actual patch
  family. Expose direction, magnitude and relevant vector components rather
  than only accidental scalar reparameterization controls.
- [ ] Specifically carry the existing Bézier arc widgets into the patch viewer:
  reuse `arc_curve_explorer`'s endpoint/middle-point picking, camera-plane drag,
  planar/spatial editing, curve rendering, depth/blending and horizon handling,
  together with `quilting_patch::arc_authoring`. Extract shared widgets where
  useful; do not replace this progress with generic weight sliders. The current
  source has three-point arc authoring, not finished higher-degree controls.
- [ ] Keep higher-degree authoring as an explicit follow-on: define the meaning
  of additional controls and supported curve/patch families, then extend the
  widget abstraction with appropriate continuity/degree tests. It must not block
  applying the existing working arc controls to tri and quad patches now.
- [ ] Resolve the supported family's weight/control constraints automatically
  by default, with a clearly labeled exploratory unconstrained mode. Explain
  when a constraint resolution changes shape versus merely its parameterization;
  do not silently imply shape preservation.
- [ ] Use shared controllable edge objects across adjacent patches. Check
  positional agreement, edge orientation and tangent behavior. Claim tangent
  continuity only under the appropriate construction/compatibility conditions;
  numerical evidence is not a proof of the whole analytic surface.
- [ ] Drive patch tessellation with the shared mixed-edge composition machinery,
  including the automatic/manual interior choices, surface-aware edge lengths,
  inverse spacing maps, warp controls and cap diagnostics.
- [ ] Preserve exact parent evaluation when merely subdividing its parameter
  domain. If representing children as independent polynomial/rational patches,
  verify degree, representation closure and control-net conversion; do not infer
  exact decomposition from a seam that happens to look smooth.
- [ ] Compare two-triangle quad decomposition and the movable-center fan on
  the same authored patch. Keep the virtual hierarchy reusable for later nested
  compositions, without forcing recursive refinement into the first delivery.
- [ ] Reproduce unequal-weight, strongly curved/twisted, near-degenerate and
  projection-pole cases. Handle unsupported/singular cases explicitly and
  safely; do not hide missing regions, unstable handles or invalid meshes.
- [ ] Keep interfaces generic over supported Patch models/metrics, with sparse
  typed GA and CTFE specialization where justified. Do not label QCGA/PGA or
  higher-degree support complete without an implemented, checked construction.
- [ ] Save meaningful presets and shareable state including model, control
  points/weights, camera, sampling mode and LoDs; validate decoded values and
  browser back/forward behavior through the existing Fe web facilities.

## 4. Refresh the suite and make it a strong Fe example

- [ ] Refresh `tessellation_warp` with runtime mixed primary access and honest
  composed-quad modes instead of the old uniform-only/LoD3 preview limitations.
- [ ] Refresh `two_triangle_uniformization` with the shared controls and boundary
  descriptors while retaining its curved-patch comparison.
- [ ] Refresh `paired_triangle_fans` with independent outer requests and shared
  seam/metric machinery. Retain useful old controls as explicit comparisons.
- [ ] Use common ingots for reusable geometry, color, drawing, camera/control,
  parameter binding and diagnostics. Prefer existing abstractions over copying.
- [ ] Use derived parameter bindings, typed actor/effect composition and CTFE
  plans where they genuinely remove boilerplate. Keep computation in Fe;
  browser glue should be the shared typed WebIDL/host boundary, not demo logic.
- [ ] Provide a Fe-authored gallery/index using shared mb2 gallery and source
  inspector capabilities; do not create another handwritten gallery runtime.
- [ ] Keep each example standalone and responsive, with a collapsible scrollable
  control panel, usable dropdowns/checkboxes, legible labels, restrained text,
  attractive color and no unintended outer-page scroll or clipped controls.
- [ ] Include navigable Fe sources plus accurately labeled, syntax-highlighted
  generated WGSL and appropriate readable Wasm text/disassembly. Report actual
  artifact sizes and distinguish per-module from total sizes.
- [ ] Explain what each view demonstrates, which controls matter, and its known
  limits without burying the visual in jargon or marketing copy.
- [ ] Check gallery lifecycle: inactive examples must not all generate atlases,
  compete for the GPU or leak resources. Reuse shared-device/lifecycle support.

## 5. Evidence, performance and completion gates

- [ ] Exhaust 729 ordered triangle keys and 6,561 outer quad requests through
  pure planning/lane/permutation checks; use representative bounded GPU cases
  rather than exhaustively rendering huge scenes without need.
- [ ] Test all dyadic seam levels 0–8 with reversal and nonidentity warps,
  including final positions and winding. Test dynamic edits for stale results.
- [ ] Browser-test uniform extremes, alternating/maximally unequal edge LoDs,
  moved centers, warped boundaries, concentration/twist extremes and saved patch
  pathologies. Capture reproducible states and focused screenshots/measurements.
- [ ] Check signed orientation, boundary coverage and necessary crossing tests;
  evaluate cross-seam sample exclusion separately from simple crack freedom.
- [ ] Measure release cold first-visible/requested geometry, warm starts,
  resident edit latency, missing-tile edit latency, generation/assembly/upload
  time, transferred/retained bytes, frame cost, shader sizes and memory growth.
- [ ] First usable tessellations must appear in under one minute, with requested
  low-cost views appearing promptly; target subsecond resident edits and normal
  interactive rendering. Record exact hardware/workload and disclose misses.
  Do not claim the former subsecond full-GPU-atlas aspiration is achieved.
- [ ] Test repeated edits, reload, resize, resource/device loss and disposal;
  preserve input/control state where supported and surface genuine failures.
- [ ] Use targeted compiler profiling/riff-cat when an observed compiler defect
  blocks delivery. Keep matched source/options and causal before/after numbers;
  avoid reopening broad shader optimization without evidence of a blocker.
- [ ] Land principled compiler fixes with focused regressions promptly on shared
  `/laboratory/fe-stuff/fe-worktrees/mb2`, preserving unrelated work. No casual
  durable forks or private toolchain changes masquerading as integrated fixes.
- [ ] Commit verified project slices as work proceeds. Preserve unrelated dirty
  files; do not stash/reset or claim all work is committed without checking.
- [ ] Keep a done/doing/todo checklist with commit/test/browser evidence and
  known limits. Seek another bounded review at an architectural or correctness
  milestone if warranted, not as ceremonial approval.

## Definition of done and deliberate deferrals

Done means the requested compositional controls actually work in the browser,
both controllable patch shapes use the shared pipeline, the selected existing
demos are refreshed, gallery/standalone/source experiences are usable, and the
correctness/performance/commit gates above have evidence. A missing UI, mock
geometry, passing source check alone, or one attractive view is not completion.

Defer full direct quad-atlas GPU optimization, recursive/Wang tile research,
arbitrary-metric/high-degree generalization beyond verified families, mesh
repatching research, Blender/Hyperscape integration and a general adaptive
meshing framework. Preserve useful results and extension points, but do not
let these become prerequisites or displace finishing the controllable patches.

Review evidence: project scratch `composition-direction-review-20260907.md`
and `composition-direction-review-response-20260907.md`. Detailed composition
policy and prior Fable dispositions: `composite-atlas-next-experiment.md`.
