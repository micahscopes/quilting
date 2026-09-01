# `quilting_cga`: conformal authoring without a runtime algebra tax

Status: API direction; implementation is stop-gated on independent oracles.

Quilting does not need a general-purpose Clifford algebra interpreter in its
render loop. It would benefit from a precise conformal vocabulary at the
authoring and interaction layers, especially for spherical focus/inversion
controls, anchored frames, incidence queries, and graphs of conformal
transformations.

The proposed `quilting_cga` ingot belongs under `ingots/algebra` and depends
only on `quilting_domain` and `quilting_quaternion`. It does not depend on the
experimental `gaplay`, `sparse_clifford`, or `ga_expr` ingots. Its public
surface should expose typed geometric values rather than public blade arrays:

- Euclidean point, normalized plane, oriented sphere, and round;
- explicit finite/ideal/degenerate classification;
- translation, rotation, dilation, reflection, and inversion constructors;
- unambiguously ordered conformal-map composition and typed action on
  geometric values;
- incidence and signed-side predicates with documented conditioning;
- conversion to the compact quaternion/Möbius coefficients consumed by
  Quilting's current hot path; and
- deterministic interpolation policies for UI controls, with singular cases
  represented as values rather than NaNs.

The first compact map should mirror the renderer's quaternionic coefficient
order and right-quotient convention:

```text
MobiusF { a, b, c, d: QuatF }

mobius_identity
mobius_translation
mobius_uniform_scale_admit
mobius_rotation_axis_angle_admit
mobius_inversion
mobius_sphere_reflection_admit
mobius_compose_after(outer, inner)
mobius_apply_point_bounded
mobius_apply_tangent_bounded
```

The API deliberately says `compose_after`, not `compose`. It calls the compact
value a conformal map, not a versor: claiming a general sandwich
representation would be dishonest at this stage. Bounded actions return a
finite payload, status, and denominator norm; pole/ideal outcomes never
masquerade as finite points and NaN is never control flow.

The initial geometry vocabulary is similarly small:

```text
Point3
UnitPlane { unit_normal, offset }
Sphere { center, positive_radius }
RoundGeometry = Sphere | Plane
OrientedRound { geometry, orientation }

admit_plane(normal, offset, min_norm_squared)
admit_sphere(center, radius)
round_signed_value(round, point)
classify_signed(value, epsilon)
```

Sphere sign is `|x - center|² - radius²`; plane sign is
`unit_normal · x - offset`. These definitions remain distinct from raw CGA
dual-sphere incidence, whose conventional scale differs. Static zero-sized
frame markers may wrap these values in Fe, but dynamic frame IDs, scene
ownership, and the conformal forest remain Rust authorities.

The layering should be:

```text
quilting_domain       Euclidean/reference-domain primitives
quilting_quaternion   compact quaternion arithmetic and render ABI
quilting_cga          typed conformal authoring and control semantics
quilting_qb           surface evaluation; consumes only the narrow algebra it needs
```

`quilting_qb` should not be forced through a full multivector representation
per vertex. CGA can instead author or compose a transformation and lower it to
the compact representation already used by the surface evaluator.

Before implementation, freeze cross-language vectors for point/sphere/plane
embedding, incidence, each generator, composition order, round trips to
Möbius coefficients, ideal elements, near-null inputs, and every singular
failure mode. The Fe implementation may use compile-time expression
specialization, but generated Wasm/WGSL must contain straight-line scalar
arithmetic rather than a runtime expression tree or sparse blade loop.

The existing Fe `sparse_clifford`, `ga_expr`, `gaplay`, `cga3d`, and `qcga`
work are research inputs only. Reuse a mechanism only when its denotation and
generated code survive the Quilting-specific oracle; do not inherit gallery
state, presentation logic, or experimental naming.

Implement compact map construction/composition, bounded point/tangent action,
admitted planes/spheres, oriented incidence, and static frame mapping first.
Stop-gate arbitrary versor products, general meet/join, point pairs and
circles, arbitrary round transformation, versor interpolation, and any dynamic
Fe scene graph until their independent oracles and product need are concrete.

## Principal Clifford-Bézier surface lane

The principal next surface ingot, `quilting_clifford_bezier`, tests whether the
richer patch model can be both expressive and operationally small. It remains
separate from `quilting_qb`: QB supplies a compact baseline and restriction
oracle, while the new demo and interaction work leans deliberately into the
Clifford construction. Comparisons must never change the established patch ABI
or silently put dense multivector work into an existing shader.

This is a paired literature comparison, not an invented family resemblance.
Zubė's 2013 *Quaternionic Bézier curves, surfaces and volume* establishes the
quaternionic construction; Krasauskas and Zubė's 2014 *Rational Bézier
Formulas with Quaternion and Clifford Algebra Weights* places quaternion and
Clifford weights in the same research program. The exact equations,
association conventions, and construction-specific domains must still be
checked against the primary texts before becoming an ABI. In particular, the
current QB fixture is triangular and three-control, while cited Clifford
constructions may be bilinear, four-control, and convention-sensitive. Shared
fixtures must make those differences explicit rather than calling one model a
larger version of the other.

The intended experiment follows a functional-graphics discipline:

1. state the patch denotation once as typed, immutable composition;
2. specialize its fixed basis, grade, degree, and control layout at compile
   time;
3. lower the result to bounded straight-line scalar arithmetic;
4. expose independent CPU/Rust, Fe/Wasm, WGSL, and eventually GLSL
   interpretations; and
5. compare both numerical output and operational shape.

Promotion requires more than matching pictures. Gates must cap generated
scalar operation count, live temporaries/register pressure, shader source and
module size, compile/link latency, dispatch dimensions, bytes uploaded per
frame, nonfinite outputs, device errors, and repeated context/device recovery.
The old failure mode—dense multivector evaluation at every sample repeatedly
crashing a WebGL context—is an explicit anti-regression fixture.

The first Clifford gate should restrict the carrier to the quaternion
subalgebra and reproduce every QB fixture. Only after that equality holds may
the suite add one genuinely Clifford-only construction from the primary
literature.

Parallel structure should be derived from the fixed patch expression where
Fe can do so honestly. Patch instances and sample lanes are independent work;
reductions inside a lane must preserve a pinned association policy. No claim
of automatic parallelism is accepted until generated code and GPU traces show
the intended schedule.
