# Conformal mereology experiment

This small Lean project tests the precise relationship between:

- geometric Möbius transformations of a conformal sphere;
- the halfspace pocset cut out by round spheres;
- changing the distinguished point at infinity; and
- Möbius inversion in the incidence algebra of a containment poset.

It intentionally proves only the relationship that is actually justified:

1. a geometric Möbius transformation, being a bijection, preserves region
   inclusion and complementary sides;
2. exact open sides live in the regular-open Boolean algebra, whose complement
   is the interior of ordinary set complement;
3. changing the point called infinity complements exactly the walls separating
   the two points, and transforms a separating wall total by `total - old`;
4. poset Möbius inversion is equivariant under the induced order isomorphism
   because inverse operators remain inverse under conjugation; and
5. pivoting a containment chain changes the zeta kernel and Möbius kernel
   together—in the full reversal case, the new Möbius kernel is the transpose
   of the old one.

The finite payload/chamber layer now additionally proves:

- finite signed payload mass is preserved by bijective transport, including
  compactified spherical inversion;
- regular-open complement accounting is `total - old` when payload positions
  avoid the common wall boundary;
- a three-wall nested arrangement has four chambers, including background;
  the concrete honest re-anchor is `(14,12,8)`, not the old three-label shuffle
  `(7,6,4)`;
- Möbius inversion works on any finite poset without requiring a global bottom;
  this permits a genuine background top; and
- every indexed finite laminar family canonically yields a background-aware
  `ChamberModel (WithTop W) X`: each point is owned by its unique least
  containing region, or by top when it is outside every original region;
- the resulting geometric chambers are pairwise disjoint, cover the ambient
  space, and reconstruct every region as the union of its lower chambers; and
- cumulative geometric-region mass is exactly the zeta transform of direct
  geometric-chamber mass, and the incidence Möbius transform recovers those
  direct masses;
- a semantic `ChamberReassignment` records which indices in two different
  containment orders own the same physical chambers; and
- honest re-anchoring is proved to factor as `Z_new * R * M_old`. These
  transports compose along re-anchor paths by composing `R`, while each
  intermediate zeta/Möbius pair cancels. Full reversal is proved for an
  arbitrary `Fin n` chain, including its background chamber; and
- for every containment poset whose intervals are chains, the incidence
  Möbius kernel is proved to be exactly `I - C`, with `C` the cover kernel.
  Between two matched orders, `mu_new - mu_old = C_old - C_new`, so kernel
  changes are supported only on changed covers.

The spherical-inversion layer is now concrete rather than axiomatic:

- `extendedInversion` swaps an arbitrary affine pole with infinity and is
  proved involutive on `OnePoint P`;
- on a proper Euclidean-type space, that equivalence is proved continuous at
  finite non-pole points, at the pole, and at infinity, yielding
  `extendedInversionHomeomorph`;
- compactified open round sides are represented by open balls, strict
  exteriors containing infinity, and affine half-spaces; every constructor is
  proved open and regular open and can be packaged as a `RegularOpen` element;
- the exact, dimension-independent sphere-power formula is proved for
  mathlib's Euclidean inversion;
- for a sphere not through the pole, the image centre and radius are proved to
  be
  `c + (R² / (dist(a,c)² - r²)) • (a - c)` and
  `R² r / |dist(a,c)² - r²|`;
- if the pole is outside, an open ball maps to an open ball;
- if the pole is inside, an open ball maps to the complement of the image's
  closed ball (the strict exterior); and
- if the boundary passes through the pole, it maps to an affine hyperplane
  and the ball maps to a half-space;
- exact compactified image theorems include the pole and infinity rather than
  stopping on the punctured affine chart; arbitrary-centre inversion therefore
  preserves `IsOpenRoundSide`, both directly and through `pushRegion`;
- translation, nonzero uniform scale, every orthogonal linear equivalence
  (including rotations), and arbitrary-centre inversion are certified
  `RoundSideAutomorphism` generators, and certified words compose;
- two-sphere inversive power is proved to transform by the exact signed factor
  `R⁴ / (δ₁δ₂)`, while absolute signed inversive distance is invariant; and
- the coarse separated/tangent/crossing classification is invariant. Absolute
  separation is proved to split into external disjointness or nesting (for
  positive radii); the absolute invariant preserves their union, while signed
  side/orientation data distinguishes the two branches. For an
  indexed family, inversion acts on the signed inversive Gram matrix by
  diagonal sign conjugation `D * G * D`; a one-wall flip changes only that
  row and column; and
- any finite matrix update supported on one row and column is explicitly the
  sum of two rank-one outer products, hence has rank at most two. The
  single-wall Gram flip is an immediate specialization.

The incidence-algebra layer also states the coordinate-change formula
directly: a containment pivot is `Z_new * M_old`, its reverse is
`Z_old * M_new`, and successive pivots telescope.

It does **not** identify geometric inversion with incidence Möbius inversion.
Instead it treats the geometric operation as changing the containment order
whose incidence inverse is then the relevant Möbius kernel.

Build it with:

```sh
cd formal
nix shell nixpkgs#lean4 -c lake update
nix shell nixpkgs#lean4 -c lake exe cache get
nix shell nixpkgs#lean4 -c lake build
```

The explicit `nix shell` wrapper is needed on NixOS; on a conventional Linux
installation with `elan`, the corresponding bare `lake` commands are enough.
If Nix path differences invalidate upstream cache traces, the two project
modules can be checked directly without rebuilding unrelated mathlib tools:

```sh
nix shell nixpkgs#lean4 -c lake env lean ConformalMereology/SphericalInversion.lean
nix shell nixpkgs#lean4 -c lake env lean ConformalMereology.lean
```
