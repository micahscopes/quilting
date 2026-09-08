# Even tessellation on curved patches

Goal set September 7, 2026. Scope is the interior of a composed patch. It does
not replace [the delivery checklist](quilting-fe-delivery-goal.md) or
[the composition and recursive atlas goal](composition-and-recursive-atlas-goal.md).

## What you will be able to do

Open the composed patch demo, drag an arc handle as far as you like, and the
triangles stay about the same size on the surface. No slider touching. Today
that only holds along the four outer sides; the interior still bunches and
thins, worst near the pole, and the fix is to go hunting in a panel of
forty-four controls for the ones that happen to compensate.

Specifically, after this:

- Every slider at neutral gives the best tessellation, not a starting point to
  correct from.
- No visible density step where two children meet at a spoke or diagonal.
- Visibly less twisting at neutral, without a twist control being involved.
- One target size control instead of four level sliders, with levels derived.
- The panel reports how even the surface actually is, so you can see it work.

## What ships

- An interior placement table in the published spacing buffer, built once per
  geometry edit from the patch's own density, read identically by the Wasm
  audit and the WebGPU draw. Same shape as the edge tables that shipped today.
- `placement.fe` reads that table instead of composing three edge laws inward.
- Surface-space audits: triangle mass against the closed-form density, chord
  normal against the analytic normal, and a per-triangle pole-proximity bound.
  These replace the reference-domain heuristic that currently scores quality.
- A resolved-size readout beside the existing segment counts.
- The level-of-detail atlas landscape demo running on the current atlas and maps.

Nothing regenerates per patch configuration. Connectivity does not change,
index buffers are reused, and no triangulator runs at edit or draw time.

## Acceptance

Measured on the surface, not in the reference square, on the same authoring
before and after:

| Check | Now | Target |
|---|---|---|
| Interior triangle area, max over min | see below | four, the dyadic floor; two needs sub-dyadic control |
| Density step across an internal seam | visible | not measurable |
| Outer-side chord deviation | 0.000019 | unchanged |
| Interior segment chord deviation | 0.00178 | unchanged |
| Seam points identical from both sides | exact | exact |

Plus: the browser shows it on the real served build, and every audit that
claims a surface property calls the surface evaluator.

## Measured so far

Surface triangle area over the whole domain, comparing points left where the
reference grid puts them against points placed by the density law. The
reference column is a floor rather than the shipped path, which carries
measured edge laws inward and so sits between the two columns.

| Patch and curvature | Reference max/min | Density max/min | Reference CV | Density CV |
|---|---|---|---|---|
| triangle, 0.6 | 3.27 | 1.98 | 0.242 | 0.116 |
| triangle, 2.4 | 64.2 | 8.12 | 0.822 | 0.338 |
| triangle, 4.8 | 601 | 27.7 | 1.514 | 0.529 |
| quad, 0.6 | 2.59 | 1.55 | 0.188 | 0.088 |
| quad, 2.4 | 53.3 | 11.0 | 0.959 | 0.449 |
| quad, 4.8 | 553 | 52.0 | 2.018 | 0.744 |

So the separable slice form meets the target at the default authoring and does
not meet it at aggressive curvature, where it still improves the ratio eight to
twenty times.

Adding a third slice direction along the diagonal, which the review originally
prescribed, was measured and is strictly worse at every curvature: on the quad
at midpoint offset 2.4 the area ratio went from 10.96 to 174.78, and at 19.2
from 2307 to 13579. Each pass equalizes along its own direction as if the others
had not acted, so composing more of them interferes rather than refines.

That closes the design question from both sides. Two passes are worth shipping,
more passes are not, and the remaining residual needs a map that is not a
product of one-dimensional corrections at all. The Poisson-solve map is
therefore required rather than optional, and it is the only remaining route.

## Order

1. Interior placement from the density rather than from carried edge laws.
   Closed. The two-pass slice form ships. A third slice direction, damped
   transport, and transport with the boundary re-imposed afterwards were each
   built and measured, and each is worse. Placement alone cannot close the
   residual, for the reason recorded above.
2. Add the three surface-space audits and retire the reference-domain heuristic.
3. Carry the boundary flag from assembly into the warp rather than re-deriving
   barycentrics and testing exact zeros, which drifts near 3e-8.
4. Derive levels from one target size using the certified norm intervals.
5. Refresh the other demos, landscape first.
6. Recursion, measured above as the only construction bounded across curvature,
   and therefore the one to build into the composition rather than an optional
   later branch. Its depth field is built and certified. Its residual is dyadic
   rounding, and closing that needs ranked tiles, which stays research.

## Why placement first, and why it was not enough

Recursion has 0.81 levels of density to act on at the default authoring and 2.9
to 6.3 at working curvatures, so it does nothing on mild patches and cannot do
better than a factor of two in area anywhere, because the fractional part of
depth spreads evenly. Finer fan counts do nothing at all: worst per-child span
is flat from 2 through 16 children, since every fan child meets at the centre
and spans the full radial range. And the visible interior defect is not the
surface folding over: inverted chords need extreme curvature and coarse
resolution together, reaching 16.7 percent only at level 2 and vanishing by
level 4.

What is left is placement. Interior points are positioned by composing three
one-dimensional boundary laws, and the closed-form density is consulted nowhere
in placement, so the interior still interpolates the rim. The seam step follows
from the same code applying those laws in a fixed order, so two children
disagree just inside a seam they agree on exactly.

Twisting is not separate. For an angle-preserving map the local rotation and
the local stretch are locked together, so fixing density fixes twist.

## Constraints

Every undirected edge keeps one canonical identity, resolution, orientation
rule and sampling map, and both adjacent uses produce bit-identical points.
Maps are published once per geometry edit, never solved per configuration or
per frame. Point relaxation, if it ever happens, belongs in the atlas build in
the reference frame, once. Reuse the 165 canonical keys.

No construction may rest on Euclidean-only classification: these patches are
Dupin cyclides under the Euclidean instance and something else under another,
so sphere and cyclide facts are not design assumptions. Where a claim needs a
definite metric to mean anything, say so at the claim.

Publication revisions change when the authored weights change, not only when
levels or skews do.

## Non-goals

Screen-space or pixel uniformity, which is not angle-preserving and re-couples
every field to the camera. Twist in any automatic solve. Fitted skew or
concentration parameters as a correction mechanism, since both are
one-parameter families that can match a curved segment's endpoints and never
its interior. Curvature-driven refinement, which answers a sag question rather
than an area question.

## Already true

Boundary spacing is measured rather than reference-domain, and the density
field is closed form and cheap. Sliding the authored midpoint along its own arc
moves boundary samples 0.000000466 instead of 0.558928188. Interior segment
deviation is 0.00178, down from 0.276. Depth for recursion is already
computable and certified over whole regions. Evidence and method are in
[the density document](patch-area-density.md) and
[the spacing document](dyadic-circular-edge-spacing.md).
