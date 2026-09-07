# Measured composition quality, not just shared seams

The current interior extension is **not established as a uniformizer**. It
preserves shared boundary placement, but worsens the two measured unequal-density
cases in the requested reference metric. Strong interior controls can also flip
straight triangles. These findings remain unfinished work, not an acceptance pass.

## One placement implementation

`composite_atlas::placement::place` is now shared by GPU rendering and the Fe/Wasm
audit. GPU storage views and owned sampled tables implement the same interval-map
interface. Atlas decoding and screen projection stay outside placement. There is
no Rust/JavaScript replacement for the sampling or placement algorithm.

`quilting_patch::triangle_quality::measure` reports signed double area and mean
ratio `2 sqrt(3) |double area| / sum(edge length squared)`. Shape1 is equilateral;
zero is degenerate. Orientation is separate: a flipped triangle can score well.
Tests check reversal, scale invariance, equilateral and degenerate cases.

`DirectionalDensity` is shared by edge integration and the audit. It interpolates
horizontal density between bottom/top, vertical density between left/right.
Metric shape scales triangle edge vectors by these densities at the centroid.
Triangle mass integrates their product with three edge-midpoint quadrature:
exact for this quadratic field over straight triangles, up to f32 rounding.
Mass CV is standard deviation / mean across triangles; smaller is more even.
Neither metric is an exact curved-surface or screen-space quality measure.

## Results

Real seed42 packed primaries, exact current permutations, actual finite triangles.
Each row compares compensation off→on with unchanged topology and segment counts.

| Case | Triangles | Flips off→on | Mean reference-metric shape off→on | Mass CV off→on |
| --- | ---: | ---: | ---: | ---: |
| Uniform LoD4, centered fan | 1,432 | 0→0 | .7813→.7813 | .2919→.2919 |
| Outer1/6/6/1, centered fan | 8,132 | 0→0 | .6228→.5484 | 1.0365→1.2769 |
| Uniform4, spoke0 skew4 | 1,432 | 0→0 | .7813→.6648 | .2919→.5463 |
| Uniform4, concentration2/twist.4/focus2.5:1:1 | 1,432 | 12→12 | .6067→.6067 | .6613→.6613 |
| Outer1/6/6/1, center(.17,.83) | 9,120 | 0→0 | .5613→.5402 | 1.7121→2.1338 |
| Uniform4, diagonal02 | 1,212 | 0→0 | .7894→.7894 | .4989→.4989 |

Important qualifications:

- The reference metric does not incorporate the additional authored skew/focus/
  twist as a new desired density. Those controls intentionally alter distribution;
  their reference scores are diagnostic, not proof that the controls are wrong.
- The unequal-density cases have neutral optional controls. Their deterioration
  is evidence against the current interior-extension quality claim.
- The folded case's signed area is approximately1, while absolute area is
  1.004874. Signed coverage alone would miss this failure. It also contains36
  triangles with Euclidean shape below.1. Summed area is not an overlap certificate.
- These are six cases, not an exhaustive proof or a performance benchmark.
  The audit regenerates primaries and uses fresh Wasm instances per case.

Both release Wasm tests passed in32.93s, including the prior density, direction,
cap, retention and completion checks. A passing audit test means the observations
are finite/coherent and geometry is accounted for; it deliberately does not assert
that every experimental warp is fold-free or that compensation improves quality.
Evidence: `/laboratory/quilting/scratch/composition-quality-quadrature-tests-20260907.log`.

## Browser check and next decision

Chrome65 ran the refactored renderer (`fe-render-e1fb919e30c176fa.json`). The strong
control case above visibly produced red folds:396 red pixels in the1018×1018
interior crop. This agrees qualitatively with the audit's12 flipped triangles,
not a GPU assertion of that exact count. Neutral controls were restored afterward.
Compiled size:595,120 parent Wasm bytes,143,353 WGSL bytes /3passes; placement
consolidation reduced the WGSL from145,268 bytes without a new rendering stage.

Next: compare boundary-preserving interior extensions against this same corpus
and metric, without changing the primary geometry or quietly weakening boundary
requirements. Keep optional artistic warps distinct from automatic density fitting.
Reject claims of improvement based solely on zero red pixels or signed area1.
Carry the shared placement/measurement boundary into the controllable patch viewer;
do not postpone that entire deliverable for another unbounded atlas optimization.

Dev-server observation: rebuilding replaces content-addressed assets while an
older page may still need them on freeze/live, producing a stale WGSL404. Waiting
for the build and reloading resolved this instance. Retaining assets for active
page generations is a shared dev-host issue, not a geometry fallback to add here.
