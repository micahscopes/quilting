# Closed-form surface area density

Measured September 7, 2026. The interior density field that a variable-density
tessellation would consume is closed form and cheap in both patch families. It
needs no runtime differentiation and no measurement of an existing mesh.

## Structure

The denominator weights are Bernstein-blended points in the four-dimensional
rotor space, so the squared norm is a contraction of their Gram matrix with the
blend basis. The patch map is an inversion of that blend, and an inversion is
conformal, so its area factor is the reciprocal fourth power of that norm. What
remains is the blend's own Jacobian.

```
triangle   dA = K / |W(lambda)|^4                    K constant
           |W|^2 = lambda^T G lambda                 G = 3x3 Gram, 6 entries

quad       dA = sqrt(Q(u,v)) / |W(u,v)|^4            Q total degree two
           |W|^2 = B(u,v)^T G B(u,v)                 G = 4x4 Gram, 10 entries
```

`Q` is total degree two rather than tensor-product degree (2,2), so it carries
six coefficients and not nine. The nine-node check below is therefore sufficient
but not minimal; the degree ladder in the higher-order section is what pins it.

The triangular blend is affine, so its Jacobian is constant and the reciprocal
fourth power is the entire law. The bilinear blend's Jacobian varies with
position, which is the whole of the quad's departure.

## Verified evidence

Release Fe/Wasm oracle against central differences of the actual evaluator,
step 1/512, on the default authored patches.

| Claim | Samples | Result |
|---|---|---|
| Triangle density is exactly the reciprocal fourth power | 55 | worst relative spread of the ratio `0.000019` |
| Quad departs from that law | 121 | worst relative spread `0.171676` |
| Quad's squared correction is a tensor-product quadratic | 81 | worst relative error vs a nine-node interpolant `0.00011456` |
| Degree-two patch's squared correction is total degree two | 27 per line, 3 lines | see the degree ladder below |

The quad check builds a three-node-per-axis Lagrange interpolant from nine
samples and reproduces every other sample, which is a structural claim about
the correction's degree rather than a fit quality. Residuals at this level are
central-difference truncation, not model error.

## Higher order

The degree-two case uses the exact triangular restriction of the authored quad,
so it is real higher-order geometry rather than a degree-elevated rewrite. A
degree ladder along three interior lines gives the minimal fitting degree:

| Fitted degree | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Worst relative error | 0.1519 | 0.00732 | 0.0000940 | 0.000159 | 0.0000792 |

Degree two fits and degree one does not, and past two the numbers are the
central-difference noise floor rather than improvement. So the squared
correction is total degree two.

That is lower than a naive count predicts. Differentiating a total-degree-`d`
blend gives partials of degree `d-1`, suggesting a cross product of degree
`2(d-1)` and a square of degree `4(d-1)`. The leading terms cancel instead. For
a bilinear blend `A + Bu + Cv + Duv` the partials are `B + Dv` and `C + Du`, and
their cross product loses its top term because `D x D` vanishes, leaving an
affine cross product and a degree-two square.

So `4(d-1)` is a safe upper bound and the realized degree can be lower. Whether
that particular cancellation persists beyond the bilinear case is not
established here.

**The exact boundary arc law does not extend.** It requires an affine rotor
path along the edge, which only the degree-one blend provides. A higher-degree
boundary is not a circle, so the dyadic midpoint construction does not apply
and such edges need the measured integration used for interior segments.

## Depth range

Recursive substitution can only act on whole octaves of this field, so its range
decides whether recursion is worth building. Measured by sweeping the authored
midpoint's out-of-plane offset, which is what "how curved is this patch" means
for the default control net:

| Midpoint offset | Triangle depth span | Quad depth span |
|---|---|---|
| 0.6, the default | 0.81 | 0.65 |
| 1.2 | 1.72 | 1.49 |
| 2.4 | 2.90 | 2.78 |
| 4.8 | 4.50 | 4.47 |
| 9.6 | 6.34 | 6.38 |

At the default authoring the span is under one level, so recursion would be a
no-op there and any visible interior imbalance is sub-dyadic. The span crosses
one level at roughly 0.8 and then grows by about 1.3 to 1.5 levels per doubling.

The fractional part of depth is spread nearly uniformly across quartiles once
the span exceeds about two levels. So recursion alone leaves a residual up to a
factor of two in area, which is its ceiling and the place a ranking or a derived
residual map would have to earn its keep.

## Cost

Per geometry edit, precompute the Gram matrix, six dot products for a triangle
or ten for a quad, plus the quad's nine correction coefficients. Per sample the
triangle costs one quadratic form and a reciprocal square, with no transcendental
at all; the quad adds a second small polynomial and one square root.

This is why the field is affordable to evaluate at every candidate point of a
variable-density construction. Computing it instead by differentiating the
evaluator, or by measuring triangle areas after meshing, is both far more
expensive and, in the second case, circular.

## Limits

These are the surface densities. Projected, pixel-space density is a different
object: perspective projection is not conformal, so it does not inherit the
scalar isotropic structure and it re-couples the field to the camera.

The measurements above are the default authored patches. The structural claims
follow from the blend degree and should hold generally, but unequal authored
weights and near-degenerate poles are not separately swept here.
