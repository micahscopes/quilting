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

quad       dA = sqrt(Q(u,v)) / |W(u,v)|^4            Q tensor-product quadratic
           |W|^2 = B(u,v)^T G B(u,v)                 G = 4x4 Gram, 10 entries
```

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

The quad check builds a three-node-per-axis Lagrange interpolant from nine
samples and reproduces every other sample, which is a structural claim about
the correction's degree rather than a fit quality. Residuals at this level are
central-difference truncation, not model error.

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
