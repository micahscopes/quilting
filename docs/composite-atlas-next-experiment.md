# Compositional tessellations: active direction

The user explicitly pivoted to **multiple methods of composition from primary
tessellations** on September 7. The unfinished direct worker-atlas viewer and
matched benchmarks are preserved, not prerequisites for this work. GPU atlas
generation stays parked. Do not resume direct-atlas development merely because
the automatic goal still contains its older priority order.

Aim for visible usable tessellations in under one minute of opening an explorer;
measure this, including runtime primary generation. Never gate the first view on
generating all 1,200 direct atlas entries. The direct generator remains a
reference method, not the mandatory production construction.

## Comparison family

Start with two diagonal splits and a movable-center four-triangle fan sharing
the same primary-tessellation access, oriented boundary rules, parent warp,
renderer, and diagnostics. Show their decomposition next to the actual triangles.
Do not claim that one fan settles the design space.

Then add recursive primary subdivision/tiling and Wang-style compatible tiles
as distinct experiments. Compare generation cost, unique primary requests,
assembly cost, triangle quality, shared boundaries, coverage/folds, storage, and
discrete LoD transitions. Each method should have a focused standalone view,
with a common comparison index rather than one overloaded mode dropdown.

The first affine-composition control is explicitly a distortion control. It
does not claim surface-uniform sampling, global blue noise, or global Delaunay.
Keep that distinction visible while adding shape-aware and warped variants.

## Four-child shape-aware comparison

Implementation checkpoint: `quilting_atlas::composition` defines both diagonal
recipes and the interior fan, with canonical endpoint identities and directed
boundary uses. Its Fe test checks outer-edge coverage, unique reversed seam
incidence, and matching sample indices for every dyadic boundary level 0–8.
This is a topological contract only: placement, primary-atlas lane adaptation,
runtime generation, warping, and the rendered comparison are not implemented by
that module. It supplies no guarantee of interior triangle quality.

Construct a square from four triangular children meeting at its center. Generate
primary children using their actual placed shape's metric for both sampling and
incircle predicates. Reuse exact internal boundary chains with explicit reversal,
then evaluate one continuous parent-domain warp across all children.

Compare direct quad, correctly shaped four-child composition, and affine-warped
equilateral children as a distortion control. Include mixed densities `[0,0,0,8]`,
`[0,8,0,8]`, uniform extremes and an asymmetric key. Check valid shape symmetries
rather than assuming every triangular child's keys admit all six permutations.

Measure generation, assembly/materialization, copies/upload, draw instances,
triangle count, boundary equality, orientation/coverage, cross-child exclusion,
and angle/distortion distributions. Match external boundary requests and disclose
the different interior density fields. Fewer expensive primary generation jobs
is a hypothesis for savings, not a benchmark result.

Exact shared boundary positions establish stitching, not global blue noise or
Delaunay legality after arbitrary warping. Spatial continuity across children
does not make discrete LoD swaps temporally continuous. Sampled diagnostics are
not certified bounds over an entire analytic patch.

## Recursive progressive alternative

The user specifically identified [Kopf et al., Recursive Wang Tiles for
Real-Time Blue Noise](https://johanneskopf.de/publications/blue_noise/). This is
the literature direction behind Fable's proposal D, not its four-child proposal
B. The paper's progressive recursive point sets support adaptive density and
reuse; constrained triangulations, exact atlas boundary ladders and cross-level
mesh stitching are additional work. Explicit point output still costs work
proportional to output size. Do not repeat the review's unqualified perimeter-only
runtime claim.

The user's follow-up extends this beyond point-set reuse: investigate tiles
carrying compatible interpolation and warping. First show two adjacent tiles
sharing one edge parameterization (with explicit orientation reversal), with
independent interior focal/concentration/twist controls. Check the entire shared
boundary rule, not just its endpoints. Then test whether recursive refinement
can inherit those rules without cracks or lost spacing control. Show actual
triangles, spacing distortion and fold/overlap diagnostics. This is a proposed
extension, not a guarantee supplied by the Wang-tile paper: exact edge matching
does not establish interior injectivity, blue noise or Delaunay after warping.

## Review provenance

Read-only Fable review and lead corrections are retained in project scratch:
`fable-composite-atlas-response-20260906.md`,
`fable-composite-atlas-rigour-response-20260906.md`, and
`fable-composite-atlas-disposition-20260906.md`. The disposition rejects several
overstated mathematical guarantees in the raw reports; read it alongside them.
