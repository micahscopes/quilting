# Circular-edge spacing without trigonometry

This is a reusable Fe kernel. The composite viewer now uses it for the
four outer sides; see Integration below for what is and is not covered.

## Construction

For the supported circular rational arcs, parameter speed has the form
`K / |w(t)|²`, with positive constant K and an affine denominator w. Its norm
must be positive definite on the relevant denominator plane; this is not an
automatic consequence of using an arbitrary Clifford metric.

For an admitted interval `[a,b]`, its equal-arc-length midpoint is

```
m = a + (b-a) |w(a)| / (|w(a)| + |w(b)|).
```

Reason: the direction halfway between two unit denominator vectors is their
normalized sum. The expression above makes the two endpoint contributions
have equal magnitudes, so it bisects their angle. The derivative of denominator
angle is a constant divided by `|w(t)|²`, the same normalized measure as arc
length. The collinear, same-direction case gives the continuous straight-line
limit. An interval crossing a denominator zero must be rejected.

Repeated midpoint construction gives precisely the fractions used by dyadic
atlas boundaries. Parents are not moved when refining. The implementation
normalizes the two magnitudes before computing their ratio to avoid overflow
of their sum, and rejects collapsed representable intervals and invalid norms.
It additionally asks the model to conservatively admit every final interval.

## Fe boundaries

- `quilting_patch::arc_spacing::CircularArcMeasure` states the required speed
  law and interval-admission contract.
- `dyadic_arc_samples<M,N>` builds a `2^L+1` parameter table in bounded work:
  one magnitude per sample and one regularity check per final interval.
- `IntervalMapSamples` permits CPU-owned arrays and future GPU storage views
  to share direct interpolation without a duplicate implementation.
- `EuclideanArcMeasure` uses the existing FCO-derived sparse Clifford norm and
  conservative component-interval chart bound. It does not reuse the isotropic
  model's norm or assume all projective models are Euclidean.

Selecting represented dyadic fractions returns their table values directly.
Linear interpolation at other fractions is only an approximation of the
continuous inverse map. Intentional edge skewing also intentionally changes
the resulting physical spacing. Neither should be advertised as exact uniform
arc length at every real-valued input.

## Verified evidence

Release Fe/Wasm oracle, September 7, 2026:

- 18 denominator families × 257 samples, magnitude ratios 0.1/1/10 and angles
  0/0.01/0.3/1.5/2.8/3.1 radians. Independent f64 inverse-line geometry checks
  cumulative angle and chord spacing; zero-length reference chords are skipped.
- Worst normalized cumulative-position error: `0.000006315`.
- Worst relative chord error in that stress set: `0.001055299` (about 0.106%).
- 129-sample tables agree bit-for-bit with every other sample of 257-sample
  tables; this is a nesting witness, not a statistical blue-noise claim.
- Zero endpoint norm, an interior pole, and NaN input reject.
- All seven default authored triangle/quad edges were also evaluated with the
  actual sparse-Clifford weights and patch evaluator. Their worst relative
  chord deviation from each edge's mean was `0.000027090` (about 0.0027%).

The complete test passed in 30.53 seconds including Fe compilation and many
repeated constructions to query individual samples. This is not a production
map-generation benchmark. Receipt:
`/laboratory/quilting/scratch/arc-spacing-geometric-wasm-20260907.log`.

## Integration: outer sides, September 7, 2026

The composite viewer publishes one 2^8+1 parameter table per outer side beside
the existing reference cumulatives, behind the `automatic_density` control.
Because the construction is nested, every coarser LoD selects its own samples
out of that single table exactly, so one published size serves LoD0 to LoD8.
A side whose arc the measure refuses, and the flat view, keep the reference map
through a leading sentinel rather than silently falling back to uniform UV.

Publication now also invalidates on the authored denominator weights, not only
on reference LoDs and skews. Without that an arc drag kept spacing built for the
previous surface.

Acceptance beyond chord uniformity: sliding the authored midpoint along its own
arc reparameterizes that edge without moving it, so measured spacing must return
the same points. Over slides of 0.30 to 0.70 on the default quad, whose edge
spans two units, uniform-parameter spacing moved boundary samples by up to
`0.558928188` while measured spacing moved them by `0.000000466`.

## Interior segments: measured integration, September 7, 2026

Diagonals and fan spokes restrict through the bilinear domain, so their
denominator is quadratic rather than affine and this kernel's law does not
apply. Under `automatic_density` they instead receive an integrated cumulative:
sixteen published spans, each accumulated from four substeps of actual surface
chord length through the same composition warp and patch evaluation the
renderer uses. The existing cumulative inverse then places their samples, and
the skew control biases that uniform base rather than correcting it.

Worst relative deviation of adjacent surface chords at 64 segments on the
default curved quad:

| Interior segment | Uniform parameter | Measured length |
|---|---|---|
| spokes 0 and 1 to centre | 0.197562 | 0.013415 |
| spokes 2 and 3 to centre | 0.246315 | 0.013836 |
| diagonals 0-2 and 1-3 | 0.276497 | 0.027125 |

Unlike the outer-side tables this carries a quadrature residual, and the two
diagonals show roughly twice the spoke residual because they are longer and
bend more inside each of the sixteen spans. Raising the published span count is
the direct lever if that becomes visible; the substep count already controls
length accuracy rather than inverse resolution.

## Integration still required

Build each canonical physical boundary map once per compatible geometric
snapshot. All adjacent triangles must share it, reversing the input before
canonical evaluation. The publication revision must change when weights change,
not only when reference-domain LoDs/skews change. Do not pair old tables with
new weights or silently replace an inadmissible arc map with uniform UV spacing.

The triangle's three edges and quad's four outer edges have this linear
denominator structure. Quad diagonals and arbitrary spokes generally do not:
their restriction through a bilinear parameter domain can be higher degree.
They now take the measured integration above rather than an exact restriction.
In a triangular patch every straight reference line does keep the affine
structure, so its interior segments could take the exact law; that case is not
yet split out from the shared interior path.
Interior distortion, finite-triangle coverage near poles, and extreme f32
parameter conditioning remain open requirements, not solved by this kernel.
