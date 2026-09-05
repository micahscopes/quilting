# Clifford-weighted patches and arc authoring

Two different edits must not be confused:

- Multiplying the four fixed paper weights by positive, separable corner
  scalars changes the coordinates on that same surface.
- Changing the directions of the relative weights can change the curves and
  the surface itself. The arc-authoring interface derives these weights from
  geometric handles instead of exposing an unstructured coefficient editor.

## An endpoint–on-arc-point–endpoint construction

`quilting_patch::arc_authoring::arc_through(algebra, start, through, end,
minimum)` returns either valid controls or `ArcConstruction::Degenerate`.
The supplied `through` point is the curve value at parameter `1/2`, not an
off-curve Bézier control point. It need not be the halfway point in arc length.

For points `a`, `f`, and `b`, the relative end weight is
`(b - f)^-1 * (f - a)` and the start weight is the identity. Product order
matters. Evaluation is

```text
C(t) = [a (1-t) + b w t] [(1-t) + w t]^-1
```

`arc_from_tangent` instead derives `w = (b-a)^-1 * tangent`, specifying
the derivative at the start. Positive scaling of that tangent changes
parameter speed on the same arc; changing its direction changes geometry.

The construction and interpolation live in the generic Patch ingot. The
Clifford implementation supplies sparse products specialized through
`ga_expr` FCO derivation:

| Provider | Metric | Meaning |
| --- | --- | --- |
| `euclidean::EuclideanCliffordAlgebra` | Cl(3,0,0) | Ordinary Euclidean circular arcs or lines |
| `IsotropicCliffordAlgebra` | Cl(2,0,1) | Isotropic arcs; not necessarily Euclidean circles |

Both use the same sparse expression descriptions and coefficient storage.
There is no runtime blade algebra, metric switch inside the products, or
handwritten shader implementation.

## Guarantees and limits

- Accepted, defined evaluations interpolate the endpoints and on-arc handle.
- Coincident or metric-null required displacements are rejected. In particular,
  an isotropic purely vertical displacement is null, while a Euclidean one is not.
- The quotient's `defined` flag is a separate check at each evaluation. A
  collinear exterior handle can select a curve that crosses infinity; accepted
  controls do **not** certify a finite affine curve over the whole interval.
- Rendering policy: keep both finite branches, fade their approach to the
  view's horizon, and never connect a tessellation segment across a pole. An
  undefined sample is not grounds for dropping the complete arc. The fade
  belongs to presentation geometry, not the magnitude of an arbitrarily
  scaled homogeneous weight. This policy is not yet an implemented arc widget.
- Model-specific point admission remains separate from division. A small
  denominator or non-vector residual must not silently become a drawable point.
- Four individually valid arc constructions do **not** by themselves prove a
  compatible quad patch. Closure and interior point-space conditions need their
  own construction and evidence before these controls can author a whole quad.
- Exact restriction of an existing bilinear numerator/denominator to two
  triangles already exists in `restrict_bilinear_projective_controls_to_triangle`.
  Its children are degree-two triangular fields; the shared diagonal need not
  be circular. Arc authoring does not replace or approximate that restriction.

## Sources and checks

Krasauskas and Zubė, *Bézier-like Parametrizations of Spheres and Cyclides Using
Geometric Algebra*, p. 3, gives the on-arc-point construction. Their *Rational
Bézier Formulas with Quaternion and Clifford Algebra Weights*, pp. 6–10,
distinguishes reparameterization, tangent authoring, circularity, and whole-patch
compatibility. Copies are in the user's `sync/PDFs/math/QB Surfaces` collection
and `sync/PDFs/math/clifford patches.pdf`.

`clifford_oracle::dense_arc_authoring_interpolates_handles_in_both_metrics`
uses an independent dense eight-blade f64 algebra. The companion
`fe_oracle::authored_arc_weights_wasm_match_dense_algebras_and_geometric_handles`
compares compiled production Fe evaluations, including null inputs, a pole,
and non-finite handles. Neither test claims whole-quad compatibility or a
global sampling guarantee.
