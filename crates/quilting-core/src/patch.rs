use crate::quaternion::{Quat, Mobius};

/// A QB (Quaternionic-Bézier) triangle patch.
///
/// Each vertex has a position (as a pure imaginary quaternion representing
/// a 3D point) and a weight (a general quaternion). Surface evaluation
/// uses the rational quaternion formula:
///
///   X(u,v) = Im( (Σ λᵢ pᵢ wᵢ) / (Σ λᵢ wᵢ) )
///
/// where λᵢ are barycentric coordinates. This produces Möbius-invariant
/// surfaces: applying a Möbius transformation only requires transforming
/// control points and weights.
#[derive(Debug, Clone, Copy)]
pub struct QBTriPatch {
    /// Vertex positions (pure imaginary quaternions = 3D points)
    pub positions: [Quat; 3],
    /// Vertex weights (general quaternions)
    pub weights: [Quat; 3],
}

/// Result of evaluating a surface patch at a parameter point.
#[derive(Debug, Clone, Copy)]
pub struct SurfacePoint {
    pub position: [f64; 3],
    pub normal: [f64; 3],
}

/// Exact first-order parameter frame of a rational QB triangle.
#[derive(Debug, Clone, Copy)]
pub struct PatchDifferential {
    pub position: [f64; 3],
    pub tangent_u: [f64; 3],
    pub tangent_v: [f64; 3],
}

/// One triangle in the barycentric parameter domain of a source QB patch.
///
/// Keeping the source-domain corners beside a restricted patch lets render
/// extraction interpolate UVs, normals, skinning inputs, and stable surface
/// addresses without inventing child mesh identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QBPatchDomain {
    pub corners: [[f64; 3]; 3],
}

impl QBPatchDomain {
    pub const FULL: Self = Self {
        corners: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    };

    /// Map barycentrics in this child triangle back to the source patch.
    pub fn map_barycentric(&self, child: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|coordinate| {
            child[0] * self.corners[0][coordinate]
                + child[1] * self.corners[1][coordinate]
                + child[2] * self.corners[2][coordinate]
        })
    }

    /// Express a domain nested inside this domain in source coordinates.
    pub fn compose(&self, child: Self) -> Self {
        Self {
            corners: child.corners.map(|corner| self.map_barycentric(corner)),
        }
    }

    /// Dyadic 1-to-4 split. Every child retains the parent's orientation and
    /// neighboring children contain numerically identical shared endpoints.
    pub fn quarter(&self) -> [Self; 4] {
        let midpoint = |a: [f64; 3], b: [f64; 3]| {
            std::array::from_fn(|coordinate| 0.5 * (a[coordinate] + b[coordinate]))
        };
        let [p0, p1, p2] = self.corners;
        let p01 = midpoint(p0, p1);
        let p02 = midpoint(p0, p2);
        let p12 = midpoint(p1, p2);
        [
            Self { corners: [p0, p01, p02] },
            Self { corners: [p01, p1, p12] },
            Self { corners: [p02, p12, p2] },
            Self { corners: [p01, p12, p02] },
        ]
    }
}

/// Exact rational restriction paired with its location in the source patch.
#[derive(Debug, Clone, Copy)]
pub struct RestrictedQBTriPatch {
    pub patch: QBTriPatch,
    pub domain: QBPatchDomain,
}

impl RestrictedQBTriPatch {
    /// Split again while retaining coordinates in the original source patch.
    pub fn quarter(&self) -> [Self; 4] {
        QBPatchDomain::FULL.quarter().map(|local_domain| {
            let patch = self.patch.restrict(local_domain).patch;
            let domain = self.domain.compose(local_domain);
            Self { patch, domain }
        })
    }
}

impl QBTriPatch {
    pub fn new(positions: [Quat; 3], weights: [Quat; 3]) -> Self {
        Self { positions, weights }
    }

    /// Create a flat patch (all weights = 1) from 3D vertex positions.
    pub fn flat(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Self {
        Self {
            positions: [
                Quat::from_point(p0[0], p0[1], p0[2]),
                Quat::from_point(p1[0], p1[1], p1[2]),
                Quat::from_point(p2[0], p2[1], p2[2]),
            ],
            weights: [Quat::ONE, Quat::ONE, Quat::ONE],
        }
    }

    /// Evaluate the rational quaternion surface at barycentric (u, v).
    /// The third barycentric coordinate is w = 1 - u - v.
    #[inline]
    pub fn eval(&self, u: f64, v: f64) -> Quat {
        let w = 1.0 - u - v;
        let [p0, p1, p2] = self.positions;
        let [w0, w1, w2] = self.weights;

        let top = w * (p0 * w0) + u * (p1 * w1) + v * (p2 * w2);
        let bottom = w * w0 + u * w1 + v * w2;

        top * bottom.inv()
    }

    /// Evaluate the patch and its exact quotient-rule derivatives.
    ///
    /// The parameter directions match [`Self::eval`]: `u` moves from control
    /// 0 toward control 1 and `v` moves from control 0 toward control 2.
    pub fn eval_differential(&self, u: f64, v: f64) -> PatchDifferential {
        let bary_0 = 1.0 - u - v;
        let [p0, p1, p2] = self.positions;
        let [w0, w1, w2] = self.weights;
        let pw0 = p0 * w0;
        let pw1 = p1 * w1;
        let pw2 = p2 * w2;
        let top = bary_0 * pw0 + u * pw1 + v * pw2;
        let bottom = bary_0 * w0 + u * w1 + v * w2;
        let bottom_inverse = bottom.inv();
        let point = top * bottom_inverse;
        let tangent_u = (pw1 - pw0 - point * (w1 - w0)) * bottom_inverse;
        let tangent_v = (pw2 - pw0 - point * (w2 - w0)) * bottom_inverse;
        PatchDifferential {
            position: point.to_point(),
            tangent_u: tangent_u.to_point(),
            tangent_v: tangent_v.to_point(),
        }
    }

    /// Restrict this degree-one rational patch to a barycentric subtriangle.
    ///
    /// Numerator and denominator controls are restricted homogeneously before
    /// converting back to `(position, weight)`. The result is another exact QB
    /// patch, not a sampled polynomial approximation.
    pub fn restrict(&self, domain: QBPatchDomain) -> RestrictedQBTriPatch {
        let weighted_positions: [Quat; 3] = std::array::from_fn(|index| {
            self.positions[index] * self.weights[index]
        });
        let mut positions = [Quat::ZERO; 3];
        let mut weights = [Quat::ZERO; 3];
        for (corner_index, barycentric) in domain.corners.into_iter().enumerate() {
            let numerator = weighted_positions[0] * barycentric[0]
                + weighted_positions[1] * barycentric[1]
                + weighted_positions[2] * barycentric[2];
            let denominator = self.weights[0] * barycentric[0]
                + self.weights[1] * barycentric[1]
                + self.weights[2] * barycentric[2];
            positions[corner_index] = numerator * denominator.inv();
            weights[corner_index] = denominator;
        }
        RestrictedQBTriPatch {
            patch: Self { positions, weights },
            domain,
        }
    }

    /// Exact dyadic 1-to-4 restriction in the full source domain.
    pub fn quarter(&self) -> [RestrictedQBTriPatch; 4] {
        QBPatchDomain::FULL.quarter().map(|domain| self.restrict(domain))
    }

    /// Evaluate surface point with normal (via finite differences).
    pub fn eval_with_normal(&self, u: f64, v: f64) -> SurfacePoint {
        let eps = 1e-4;
        let p = self.eval(u, v);
        let pu = self.eval(u + eps, v);
        let pv = self.eval(u, v + eps);

        let du = [pu.x - p.x, pu.y - p.y, pu.z - p.z];
        let dv = [pv.x - p.x, pv.y - p.y, pv.z - p.z];

        // Normal = cross(du, dv), normalized
        let nx = du[1] * dv[2] - du[2] * dv[1];
        let ny = du[2] * dv[0] - du[0] * dv[2];
        let nz = du[0] * dv[1] - du[1] * dv[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let normal = if len > 1e-12 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        SurfacePoint {
            position: p.to_point(),
            normal,
        }
    }

    /// Apply a Möbius transformation to this patch.
    /// Transforms positions and weights according to equations (4) and (5).
    pub fn transform(&self, m: &Mobius) -> Self {
        let mut new_positions = [Quat::ZERO; 3];
        let mut new_weights = [Quat::ZERO; 3];
        for i in 0..3 {
            new_positions[i] = m.apply(self.positions[i]);
            new_weights[i] = m.transform_weight(self.positions[i], self.weights[i]);
        }
        Self {
            positions: new_positions,
            weights: new_weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq_3(a: [f64; 3], b: [f64; 3]) -> bool {
        (a[0]-b[0]).abs() < EPS && (a[1]-b[1]).abs() < EPS && (a[2]-b[2]).abs() < EPS
    }

    #[test]
    fn flat_patch_is_planar() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // Evaluate at centroid
        let sp = patch.eval_with_normal(1.0/3.0, 1.0/3.0);
        assert!(approx_eq_3(sp.position, [1.0/3.0, 1.0/3.0, 0.0]));
        // Normal should be (0,0,1) for a flat patch in XY plane
        assert!(approx_eq_3(sp.normal, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn flat_patch_vertices() {
        let p0 = [1.0, 2.0, 3.0];
        let p1 = [4.0, 5.0, 6.0];
        let p2 = [7.0, 8.0, 9.0];
        let patch = QBTriPatch::flat(p0, p1, p2);

        // u=0, v=0 → p0
        assert!(approx_eq_3(patch.eval(0.0, 0.0).to_point(), p0));
        // u=1, v=0 → p1
        assert!(approx_eq_3(patch.eval(1.0, 0.0).to_point(), p1));
        // u=0, v=1 → p2
        assert!(approx_eq_3(patch.eval(0.0, 1.0).to_point(), p2));
    }

    #[test]
    fn mobius_transform_preserves_surface() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let t = Mobius::translation(Quat::from_point(5.0, 0.0, 0.0));
        let transformed = patch.transform(&t);

        let original = patch.eval(0.3, 0.3).to_point();
        let moved = transformed.eval(0.3, 0.3).to_point();

        // Translation by (5,0,0) should shift x by 5
        assert!((moved[0] - original[0] - 5.0).abs() < EPS);
        assert!((moved[1] - original[1]).abs() < EPS);
        assert!((moved[2] - original[2]).abs() < EPS);
    }

    #[test]
    fn nonunit_weights_curve_surface() {
        // A patch with non-unit weights should NOT be flat
        let patch = QBTriPatch::new(
            [
                Quat::from_point(0.0, 0.0, 0.0),
                Quat::from_point(1.0, 0.0, 0.0),
                Quat::from_point(0.0, 1.0, 0.0),
            ],
            [
                Quat::ONE,
                Quat::ONE,
                Quat::new(0.0, 0.0, 0.0, 1.0), // weight = k (pure quaternion)
            ],
        );

        let flat = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // At (0.3, 0.3), the weighted patch should differ from the flat one
        let p_weighted = patch.eval(0.3, 0.3).to_point();
        let p_flat = flat.eval(0.3, 0.3).to_point();

        let diff = (0..3).map(|i| (p_weighted[i] - p_flat[i]).abs()).sum::<f64>();
        assert!(diff > 0.01, "non-unit weights should deform the surface, diff={}", diff);
    }

    /// Interior barycentric samples, avoiding the corners (where the identity
    /// is trivial) and the edges.
    const INTERIOR: [(f64, f64); 6] = [
        (1.0/3.0, 1.0/3.0),
        (0.1, 0.1),
        (0.7, 0.2),
        (0.2, 0.7),
        (0.45, 0.05),
        (0.05, 0.45),
    ];

    /// The theorem the whole system rests on: a Möbius map commutes with QB
    /// evaluation. Transforming the control points and weights and *then*
    /// evaluating gives the same point as evaluating and then transforming.
    ///
    /// It holds exactly, not approximately. Writing `X = top·bot⁻¹`, the
    /// transform rules give `top' = a·top + b·bot` and `bot' = c·top + d·bot`,
    /// because each control point's `(c·pᵢ + d)⁻¹` cancels against the weight
    /// rule `w'ᵢ = (c·pᵢ + d)·wᵢ`. Then
    /// `top'·bot'⁻¹ = (a·top + b·bot)(c·top + d·bot)⁻¹ = (aX + b)(cX + d)⁻¹`.
    ///
    /// This is why the GPU can fold the transform into the rational form and
    /// spend one quaternion inverse per vertex instead of one per control
    /// point, and why dragging a Möbius slider needs no CPU work.
    ///
    /// The interesting case is `c ≠ 0`. With `c = 0` the map is affine and the
    /// identity collapses to linearity of the numerator, which proves much
    /// less — so these cases all use sphere reflections and inversions.
    #[test]
    fn mobius_commutes_with_evaluation_when_c_is_nonzero() {
        // Poles sit at the reflection centres, so keep those off the patch.
        let flat = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let curved = QBTriPatch::new(
            [
                Quat::from_point(0.0, 0.0, 0.0),
                Quat::from_point(1.0, 0.0, 0.0),
                Quat::from_point(0.0, 1.0, 0.0),
            ],
            [
                Quat::new(0.8, 0.1, -0.2, 0.3),
                Quat::new(1.0, 0.0, 0.0, 0.0),
                Quat::new(0.4, -0.5, 0.6, 0.2),
            ],
        );

        let cases: [(&str, Mobius); 3] = [
            (
                "sphere reflection",
                Mobius::sphere_reflection(Quat::from_point(0.4, -0.3, 1.9), 1.7),
            ),
            (
                "sphere inversion (two reflections)",
                Mobius::sphere_inversion(
                    Quat::from_point(0.2, 0.5, -1.4), 0.9,
                    Quat::from_point(-0.6, 0.1, 2.2), 1.3,
                ),
            ),
            (
                "reflection composed with rotation",
                Mobius::sphere_reflection(Quat::from_point(-1.1, 0.7, 1.5), 2.0)
                    .compose(&Mobius::rotation(0.3, 1.0, -0.2, 0.9)),
            ),
        ];

        for (name, m) in cases {
            assert!(!m.is_affine(), "{name}: c must be nonzero for this to test anything");
            for (patch_name, patch) in [("flat", flat), ("curved weights", curved)] {
                let transformed = patch.transform(&m);
                for (u, v) in INTERIOR {
                    let via_patch = transformed.eval(u, v);
                    let via_point = m.apply(patch.eval(u, v));
                    // Scale tolerance with magnitude: sphere reflections push
                    // points a long way, so a fixed absolute epsilon would be
                    // meaninglessly strict out there and vacuous near zero.
                    let scale = via_point.norm().max(1.0);
                    let err = (via_patch - via_point).norm();
                    assert!(
                        err < 1e-9 * scale,
                        "{name} / {patch_name} at ({u}, {v}): \
                         transform-then-eval {via_patch:?} != eval-then-transform \
                         {via_point:?} (err {err:e})"
                    );
                }
            }
        }
    }

    /// Corners are fixed points of the identity above, but they also pin down
    /// the transform rule itself: control point i must land exactly on the
    /// Möbius image of the original control point.
    #[test]
    fn transform_moves_control_points_to_their_images() {
        let m = Mobius::sphere_reflection(Quat::from_point(0.3, 0.2, 2.5), 1.4);
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let transformed = patch.transform(&m);
        for i in 0..3 {
            let expected = m.apply(patch.positions[i]);
            assert!(
                (transformed.positions[i] - expected).norm() < 1e-12,
                "control point {i} misplaced"
            );
        }
    }

    #[test]
    fn analytic_differential_matches_inverted_patch_finite_differences() {
        let patch = QBTriPatch::new(
            [
                Quat::from_point(0.2, -0.1, 1.5),
                Quat::from_point(1.1, 0.2, 1.8),
                Quat::from_point(-0.3, 1.0, 2.1),
            ],
            [
                Quat::new(1.0, 0.1, 0.0, 0.0),
                Quat::new(0.8, 0.0, -0.2, 0.1),
                Quat::new(1.2, -0.1, 0.1, 0.0),
            ],
        )
        .transform(&Mobius::inversion());
        let (u, v) = (0.31, 0.27);
        let differential = patch.eval_differential(u, v);
        let epsilon = 1.0e-6;
        let point = patch.eval(u, v).to_point();
        let finite_u = patch
            .eval(u + epsilon, v)
            .to_point()
            .map(|value| value / epsilon);
        let finite_v = patch
            .eval(u, v + epsilon)
            .to_point()
            .map(|value| value / epsilon);
        let point_rate = point.map(|value| value / epsilon);
        assert!(approx_eq_3(differential.position, point));
        assert!(approx_eq_3(
            differential.tangent_u,
            std::array::from_fn(|axis| finite_u[axis] - point_rate[axis]),
        ));
        assert!(approx_eq_3(
            differential.tangent_v,
            std::array::from_fn(|axis| finite_v[axis] - point_rate[axis]),
        ));
    }

    fn assert_quat_near(actual: Quat, expected: Quat, tolerance: f64) {
        let scale = expected.norm().max(1.0);
        let error = (actual - expected).norm();
        assert!(
            error <= tolerance * scale,
            "actual={actual:?} expected={expected:?} error={error:e}"
        );
    }

    #[test]
    fn rational_restriction_reproduces_parent_parameterization() {
        let patch = QBTriPatch::flat(
            [-0.7, -0.4, -3.0],
            [0.9, -0.2, -3.4],
            [-0.1, 0.8, -2.8],
        )
        .transform(&Mobius::sphere_reflection(
            Quat::from_point(0.2, -0.1, -1.4),
            0.9,
        ));
        let domain = QBPatchDomain {
            corners: [
                [0.65, 0.20, 0.15],
                [0.10, 0.75, 0.15],
                [0.15, 0.10, 0.75],
            ],
        };
        let child = patch.restrict(domain);
        for local in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.3, 0.5],
            [0.72, 0.11, 0.17],
        ] {
            let source = domain.map_barycentric(local);
            let expected = patch.eval(source[1], source[2]);
            let actual = child.patch.eval(local[1], local[2]);
            assert_quat_near(actual, expected, 2.0e-12);
        }
    }

    #[test]
    fn rational_restriction_commutes_with_mobius_transform() {
        let patch = QBTriPatch::flat(
            [-0.8, -0.5, -3.2],
            [0.9, -0.4, -3.0],
            [-0.3, 0.8, -3.4],
        );
        let domain = QBPatchDomain {
            corners: [
                [0.5, 0.3, 0.2],
                [0.1, 0.8, 0.1],
                [0.2, 0.1, 0.7],
            ],
        };
        let transform =
            Mobius::sphere_reflection(Quat::from_point(0.1, -0.1, -2.2), 0.8);
        let restrict_then_transform = patch.restrict(domain).patch.transform(&transform);
        let transform_then_restrict = patch.transform(&transform).restrict(domain).patch;
        for local in [[0.2, 0.3, 0.5], [0.7, 0.1, 0.2], [0.1, 0.8, 0.1]] {
            assert_quat_near(
                restrict_then_transform.eval(local[1], local[2]),
                transform_then_restrict.eval(local[1], local[2]),
                3.0e-12,
            );
        }
    }

    #[test]
    fn recursive_quarters_retain_source_domains_and_shared_edges() {
        let patch = QBTriPatch::new(
            [
                Quat::from_point(-0.5, -0.2, -2.0),
                Quat::from_point(0.8, -0.1, -2.5),
                Quat::from_point(-0.1, 0.9, -2.2),
            ],
            [
                Quat::new(1.0, 0.1, 0.0, 0.0),
                Quat::new(0.9, 0.0, -0.1, 0.1),
                Quat::new(1.1, -0.1, 0.1, 0.0),
            ],
        );
        let parent_domain = QBPatchDomain {
            corners: [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.1, 0.7],
            ],
        };
        let children = patch.restrict(parent_domain).quarter();
        assert_eq!(children[0].domain.corners[1], children[3].domain.corners[0]);
        assert_eq!(children[0].domain.corners[2], children[3].domain.corners[2]);

        for t in [0.0, 0.2, 0.5, 0.9, 1.0] {
            // Child 0 edge p1->p2 is child 3 edge p0->p2.
            assert_quat_near(
                children[0].patch.eval(1.0 - t, t),
                children[3].patch.eval(0.0, t),
                3.0e-12,
            );
        }
        for child in children {
            let local = [0.2, 0.3, 0.5];
            let source = child.domain.map_barycentric(local);
            assert_quat_near(
                child.patch.eval(local[1], local[2]),
                patch.eval(source[1], source[2]),
                3.0e-12,
            );
        }
    }
}
