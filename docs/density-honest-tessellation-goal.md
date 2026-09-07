# Density-honest patch tessellation

Goal set September 7, 2026, after boundary spacing became measured and the
interior density field was established in closed form. This narrows and
sequences the interior half of the composition programme. It does not replace
[the delivery checklist](quilting-fe-delivery-goal.md) or
[the composition and recursive atlas goal](composition-and-recursive-atlas-goal.md),
whose unfinished requirements all remain in scope.

## Goal

Make the tessellation of a curved patch honest about the surface rather than
about its chart: uniform surface triangle area under one declared target, no
density discontinuity at any internal seam, and every claim carried by a
measurement of the surface rather than of the reference square. Do it by
reusing the precomputed atlas, with maps published once per geometry edit, and
without assuming a Clifford signature.

## What is established

Boundary spacing is measured, not reference-domain. Outer sides carry an exact
nested arc-length table; interior spokes and diagonals carry an integrated one.
Worst adjacent-chord deviation at 64 segments fell from 0.197562 to 0.000726 on
spokes and 0.276497 to 0.001783 on diagonals. Sliding the authored midpoint
along its own arc, a pure reparameterization, moves boundary samples by
0.000000466 rather than 0.558928188.

Surface area density is closed form. For the triangle it is a constant over the
denominator norm squared, verified against central differences to 0.000019; the
quad adds a total-degree-two correction. The norm is a Gram contraction, and
the Gram comes from the algebra's own norm by polarization, so no signature is
assumed. Depth selection needs no logarithm: one subdivision level is exactly
one factor of two in the norm, and whole regions are certified through the
Bernstein hull, which fails closed and refines.

Three things are measured and settled. Recursion has 0.81 levels to act on at
the default authoring and 2.9 to 6.3 at working curvatures, so it is a no-op on
mild patches and useful on the ones being authored. Finer fan counts do not
help, flat from 2 to 16 children, because fan children all meet at the centre
and each spans the full radial range of a field whose level sets are conics.
Inverted surface chords need both extreme curvature and coarse resolution,
reaching 16.7 percent only at level 2 on an extreme triangle and vanishing by
level 4, so the visible interior defect is not chord inversion.

## The defect this goal addresses

Interior points are placed by composing three one-dimensional boundary laws.
The closed-form density is consulted nowhere in placement, the audited density
is an admitted reference-domain heuristic, and the fold witness counts signed
area in the reference square. So after the boundary became surface-measured the
interior still interpolates the rim. Two consequences follow, and the evidence
above says they are the ones that matter:

- Interior triangle area varies with the interior-to-rim ratio of the field,
  which no boundary-interpolated map can equalize, since one-dimensional
  density inversion is exact while two-dimensional area equalization is not an
  inversion.
- The slice composition applies edges in a fixed order, so the two sides of an
  internal seam receive different interior densities while sharing bit-identical
  boundary points. That is a density crease, not a geometric crack.

Twist is not a separate problem. For a conformal map the rotation field's
gradient equals the log-density gradient, so resolving density resolves twist
without an explicit frame.

## Order

1. **Publish a two-dimensional interior map driven by the density.** A function
   of the assembled domain, not of the child, which removes the seam crease by
   construction. Smallest form on existing machinery: three-direction slice
   composition whose one-dimensional law along each direction is the inverse
   cumulative of the square root of the density, published like the arc tables
   and blended by the existing locality weight. Boundary and spoke points do not
   move. Compare against a Dacorogna-Moser map from a small Poisson solve before
   accepting the slice form as final.
2. **Add surface-space witnesses.** Chord normal against the analytic normal,
   minimum certified norm per triangle as the pole witness, and triangle mass
   against the closed-form density reported as coefficient of variation and
   max-over-min. Replace the reference-domain heuristic in the quality audit.
3. **Carry the boundary flag from assembly into the warp** instead of
   re-deriving barycentrics and testing exact zeros, which drifts at about
   3e-8 for arbitrary parameters.
4. **Drive levels from one declared target** rather than four manual sliders,
   using the certified per-region norm interval.
5. **Then choose between recursion and ranking**, on evidence. The depth field
   already exists. Recursion's ceiling is a factor of two in area, because the
   fractional part of depth spreads evenly once the span passes two levels;
   closing that needs ranked tiles, whose parent inclusion is what breaks when
   better primaries are substituted.
6. **Refresh the other demos** onto the current atlas and maps, the level-of-
   detail atlas landscape first.

## Constraints

Every undirected edge keeps one canonical identity, resolution, orientation rule
and sampling map, and both adjacent uses produce bit-identical points. Maps are
published once per geometry edit and shared by CPU audit and GPU, never solved
per configuration or per frame. Point relaxation, if any, belongs in the atlas
build in the reference frame, once. Reuse the 165 canonical keys; do not
generate a tessellation per patch configuration.

No construction may rest on Euclidean-only classification. The patches are
Dupin cyclides under the Euclidean instance and something else under another,
so sphere and cyclide facts are not available as design assumptions. Where a
metric is required for a claim to mean anything, say so at the claim.

Publication revisions must change when the authored weights change, not only
when reference levels or skews do.

## Non-goals

Screen-space or pixel uniformity, which is not conformal and re-couples every
field to the camera. Twist in any automatic solve. Fitted skew or concentration
parameters as a correction mechanism; both are one-parameter Möbius families
whose derivative is a constant over a perfect square, so they can match a
curved segment's endpoints and never its interior. Curvature-driven refinement,
which answers a sag question rather than an area question.

## Completion

- Interior triangle mass, measured on the surface, has a coefficient of
  variation and a max-over-min ratio reported per published snapshot, and both
  improve against the current pipeline on the same authoring.
- No density discontinuity is measurable across any internal seam.
- Seam points remain bit-identical from both sides, verified as now.
- Every audit that claims a surface property calls the surface evaluator.
- The browser shows it on the real served build, not only in tests.
- Where an experiment fails, its fixture and negative result are preserved
  rather than deleted, as with fan count above.
