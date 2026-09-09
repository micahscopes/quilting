# Cheap layout ranking: hypothesis and gate

This is a bounded experiment, not an automatic viewer policy.

## Question

Can local surface stretch cheaply shortlist a useful existing coarse layout,
without first generating every layout's fine mesh?

The candidates are the two square diagonals and nine four-fans with centers at
the Cartesian product of `{0.25, 0.5, 0.75}`. Outer counts, uniform UV boundary
maps, seed and the authored surface are held fixed. No warp is enabled.

Each coarse triangle is measured at its centroid and three corner-near sites
with barycentric weights `(0.96,0.02,0.02)` and permutations. Central differences
estimate surface tangents with step 1/1024. Those tangents act on the coarse
triangle's two edge vectors; the shared, scale-normalized triangle measure
returns local shape. The score is the minimum over these sites, with mean
reported separately. No metric normalization or matrix-constancy claim is made.

Cost is exactly 32 surface evaluations for a diagonal candidate, 64 for a fan.
Repeating at half the difference step tests numerical sensitivity; it is not a
proof of derivative accuracy. These sites stay inside the square for the
specified candidates and step bounds.

## Falsification and acceptance

The hypothesis fails as a ready-to-ship selector if its preferred candidate
worsens actual minimum fine-triangle shape against a competing candidate at
the same triangle count. Report count differences rather than comparing fans
and diagonals as equal-cost configurations. Near ties are not robust decisions.

For this first probe, generate every candidate's real atlas mesh for three
previously measured surfaces at outer level 3. This provides an exhaustive
small candidate oracle; a production selector would only assemble shortlisted
candidates. It uses the same Fe surface evaluator, packed mesh decoding and
finite-triangle statistics as the earlier experiment.

A positive result would still need held-out fixtures, unequal edge counts,
geometric approximation and low-tail checks, a frozen-topology count-allocation
ablation, and stable selection under edits before viewer integration. A negative
result should guide the bounded adaptive coarse-mesh experiment, not cause an
unbounded search for a lucky score.

Limitations: coarse shape is not the shape of the unequal-edge atlas primary;
the probe does not model its internal pattern, finite curvature error or
intersections. The authoring fixtures are admitted circular-edge quads, not
evidence for arbitrary free-weight surfaces or screen-space performance.

Runner: `bounded_quad_candidate_ranking_probe` in the release Fe/Wasm harness.
Raw evidence: `/laboratory/quilting/scratch/quad-candidate-ranking-20260908.log`.

## Result: useful diagonal signal, insufficient fan ranking

The release runner passed measurement-integrity checks for 33 candidates
(three surfaces times eleven layouts), including the actual fine meshes.
The maximum change in minimum predicted score on halving the difference step
was 0.00006497. This suggests that the observed large ranking errors are not
explained by that particular step-size change; it does not rule out all
precision issues.

For the strongly bent asymmetric fixture (bulge 2.4, p2 displacement -0.5/0.7),
the score selects diagonal 02: actual minimum shape 0.245792 versus 0.104948
for diagonal 13, both 272 triangles. For the other asymmetric fixture (1.2,
0.7/-0.5), it selects diagonal 13: actual 0.275248 versus 0.259354. The symmetric
diagonals are effectively tied and must not trigger an automatic change on
their tiny score difference.

Fan rankings fail the equal-count gate. Among 376-triangle fans on the strongly
bent fixture, the score prefers candidate 9 (center 0.5/0.75), actual minimum
0.126217. Candidate 5 (center 0.25/0.5) instead reaches 0.165910. On the other
asymmetric fixture it prefers the central fan (actual 0.189498), while candidate
5 reaches 0.203162 at the same count. The coarse differential score does not
capture the fine atlas's worst elements sufficiently.

Two score calls per candidate took 0.018–0.054 ms in this release Wasmtime run,
including host-call overhead; this is neither browser-worker nor GPU timing.
The complete test took 99.35 seconds including compilation/setup and the 33
fine-mesh audits. Candidate topology also changes reference interior counts:
fans produce 376 or 384 triangles, not a universal equal budget.

Decision: do not ship a one-score fan selector. Retain the diagonal signal as a
candidate for held-out testing; use fine-mesh validation of a shortlist before
adopting any more general selector. Next compare frozen-topology metric count
allocation and extend the shared quality audit with approximation/low-tail
evidence, before claiming that adaptive coarse topology is required.
