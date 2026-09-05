//! Independent dense f64 oracle for the sparse Fe Clifford patch kernel.
//!
//! This deliberately does not mirror Fe's sparse carrier or FCO product
//! planner. It evaluates the complete eight-blade algebra and then checks the
//! projected patch laws.

#[derive(Clone, Copy, Debug, Default)]
struct Multivector([f64; 8]);

impl Multivector {
    fn scalar(value: f64) -> Self {
        let mut coefficients = [0.0; 8];
        coefficients[0] = value;
        Self(coefficients)
    }

    fn vector(x: f64, y: f64, z: f64) -> Self {
        let mut coefficients = [0.0; 8];
        coefficients[1] = x;
        coefficients[2] = y;
        coefficients[4] = z;
        Self(coefficients)
    }

    fn even(scalar: f64, e12: f64, e13: f64, e23: f64) -> Self {
        let mut coefficients = [0.0; 8];
        coefficients[0] = scalar;
        coefficients[3] = e12;
        coefficients[5] = e13;
        coefficients[6] = e23;
        Self(coefficients)
    }

    fn scale(self, scale: f64) -> Self {
        Self(self.0.map(|coefficient| coefficient * scale))
    }

    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|index| self.0[index] + other.0[index]))
    }

    fn reverse(self) -> Self {
        Self(std::array::from_fn(|blade| {
            let grade = blade.count_ones();
            let negative = grade * grade.saturating_sub(1) / 2 % 2 == 1;
            if negative {
                -self.0[blade]
            } else {
                self.0[blade]
            }
        }))
    }

    fn product(self, other: Self, metric: [f64; 3]) -> Self {
        let mut result = [0.0; 8];
        for left_blade in 0_usize..8 {
            for right_blade in 0_usize..8 {
                let mut coefficient = self.0[left_blade] * other.0[right_blade];
                if coefficient == 0.0 {
                    continue;
                }

                let shared = left_blade & right_blade;
                for (generator, square) in metric.iter().copied().enumerate() {
                    if shared & (1 << generator) != 0 {
                        coefficient *= square;
                    }
                }

                let swaps = (0..3)
                    .filter(|generator| left_blade & (1 << generator) != 0)
                    .map(|generator| (right_blade & ((1 << generator) - 1)).count_ones())
                    .sum::<u32>();
                if swaps % 2 == 1 {
                    coefficient = -coefficient;
                }
                result[left_blade ^ right_blade] += coefficient;
            }
        }
        Self(result)
    }

    fn inverse_even(self, metric: [f64; 3]) -> Option<Self> {
        let reverse = self.reverse();
        let norm = self.product(reverse, metric);
        if norm.0[1..].iter().any(|value| value.abs() > 1.0e-12) || norm.0[0].abs() < 1.0e-12 {
            return None;
        }
        Some(reverse.scale(norm.0[0].recip()))
    }
}

#[derive(Clone, Copy)]
struct Control {
    point: Multivector,
    weight: Multivector,
}

fn bilinear(values: [Multivector; 4], s: f64, t: f64) -> Multivector {
    values[0]
        .scale((1.0 - s) * (1.0 - t))
        .add(values[1].scale(s * (1.0 - t)))
        .add(values[2].scale((1.0 - s) * t))
        .add(values[3].scale(s * t))
}

#[derive(Clone, Copy)]
struct TriangleDomain {
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
}

#[derive(Clone, Copy)]
struct QuadraticTriangle([Multivector; 6]);

fn midpoint(first: [f64; 2], second: [f64; 2]) -> [f64; 2] {
    [0.5 * (first[0] + second[0]), 0.5 * (first[1] + second[1])]
}

fn quadratic_edge_control(
    first: Multivector,
    middle: Multivector,
    second: Multivector,
) -> Multivector {
    middle.scale(2.0).add(first.scale(-0.5)).add(second.scale(-0.5))
}

fn restrict_bilinear(values: [Multivector; 4], domain: TriangleDomain) -> QuadraticTriangle {
    let at = |point: [f64; 2]| bilinear(values, point[0], point[1]);
    let aa = at(domain.a);
    let bb = at(domain.b);
    let cc = at(domain.c);
    QuadraticTriangle([
        aa,
        quadratic_edge_control(aa, at(midpoint(domain.a, domain.b)), bb),
        bb,
        quadratic_edge_control(aa, at(midpoint(domain.a, domain.c)), cc),
        quadratic_edge_control(bb, at(midpoint(domain.b, domain.c)), cc),
        cc,
    ])
}

fn evaluate_quadratic_triangle(value: QuadraticTriangle, bary: [f64; 3]) -> Multivector {
    let [a, b, c] = bary;
    value.0[0]
        .scale(a * a)
        .add(value.0[1].scale(2.0 * a * b))
        .add(value.0[2].scale(b * b))
        .add(value.0[3].scale(2.0 * a * c))
        .add(value.0[4].scale(2.0 * b * c))
        .add(value.0[5].scale(c * c))
}

#[derive(Clone, Copy)]
struct QuadraticTriangleDifferential {
    value: Multivector,
    tangent_b: Multivector,
    tangent_c: Multivector,
}

fn evaluate_quadratic_triangle_differential(
    value: QuadraticTriangle,
    bary: [f64; 3],
) -> QuadraticTriangleDifferential {
    let [a, b, c] = bary;
    QuadraticTriangleDifferential {
        value: evaluate_quadratic_triangle(value, bary),
        tangent_b: value.0[0]
            .scale(-2.0 * a)
            .add(value.0[1].scale(2.0 * (a - b)))
            .add(value.0[2].scale(2.0 * b))
            .add(value.0[3].scale(-2.0 * c))
            .add(value.0[4].scale(2.0 * c)),
        tangent_c: value.0[0]
            .scale(-2.0 * a)
            .add(value.0[1].scale(-2.0 * b))
            .add(value.0[3].scale(2.0 * (a - c)))
            .add(value.0[4].scale(2.0 * b))
            .add(value.0[5].scale(2.0 * c)),
    }
}

fn triangle_parameter(domain: TriangleDomain, bary: [f64; 3]) -> [f64; 2] {
    [
        domain.a[0] * bary[0] + domain.b[0] * bary[1] + domain.c[0] * bary[2],
        domain.a[1] * bary[0] + domain.b[1] * bary[1] + domain.c[1] * bary[2],
    ]
}

fn triangle_domains(center: [f64; 2]) -> [TriangleDomain; 4] {
    [
        TriangleDomain { a: [0.0, 0.0], b: [1.0, 0.0], c: center },
        TriangleDomain { a: [1.0, 0.0], b: [1.0, 1.0], c: center },
        TriangleDomain { a: [1.0, 1.0], b: [0.0, 1.0], c: center },
        TriangleDomain { a: [0.0, 1.0], b: [0.0, 0.0], c: center },
    ]
}

fn triangle_fields(
    controls: [Control; 4],
    domain: TriangleDomain,
) -> (QuadraticTriangle, QuadraticTriangle) {
    let metric = [1.0, 1.0, 0.0];
    let numerator = controls.map(|control| control.point.product(control.weight, metric));
    let denominator = controls.map(|control| control.weight);
    (restrict_bilinear(numerator, domain), restrict_bilinear(denominator, domain))
}

fn evaluate_triangle(
    controls: [Control; 4],
    domain: TriangleDomain,
    bary: [f64; 3],
) -> Multivector {
    let metric = [1.0, 1.0, 0.0];
    let (numerator, denominator) = triangle_fields(controls, domain);
    let numerator = evaluate_quadratic_triangle(numerator, bary);
    let denominator = evaluate_quadratic_triangle(denominator, bary);
    numerator.product(
        denominator.inverse_even(metric).expect("conditioned triangular denominator"),
        metric,
    )
}

fn evaluate_triangle_differential(
    controls: [Control; 4],
    domain: TriangleDomain,
    bary: [f64; 3],
) -> QuadraticTriangleDifferential {
    let metric = [1.0, 1.0, 0.0];
    let (numerator, denominator) = triangle_fields(controls, domain);
    let numerator = evaluate_quadratic_triangle_differential(numerator, bary);
    let denominator = evaluate_quadratic_triangle_differential(denominator, bary);
    let denominator_inverse = denominator
        .value
        .inverse_even(metric)
        .expect("conditioned triangular denominator");
    let value = numerator.value.product(denominator_inverse, metric);
    let quotient_tangent = |numerator_tangent: Multivector,
                            denominator_tangent: Multivector| {
        numerator_tangent
            .add(value.product(denominator_tangent, metric).scale(-1.0))
            .product(denominator_inverse, metric)
    };
    QuadraticTriangleDifferential {
        value,
        tangent_b: quotient_tangent(numerator.tangent_b, denominator.tangent_b),
        tangent_c: quotient_tangent(numerator.tangent_c, denominator.tangent_c),
    }
}

fn triangle_normal(differential: QuadraticTriangleDifferential) -> Option<[f64; 3]> {
    let b = [
        differential.tangent_b.0[1],
        differential.tangent_b.0[2],
        differential.tangent_b.0[4],
    ];
    let c = [
        differential.tangent_c.0[1],
        differential.tangent_c.0[2],
        differential.tangent_c.0[4],
    ];
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    let length = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    (length > 1.0e-12).then(|| cross.map(|component| component / length))
}

fn assert_multivector_close(actual: Multivector, expected: Multivector, tolerance: f64) {
    for blade in 0..8 {
        assert_close(actual.0[blade], expected.0[blade], tolerance);
    }
}

fn evaluate(controls: [Control; 4], s: f64, t: f64) -> Multivector {
    let metric = [1.0, 1.0, 0.0];
    let numerator = bilinear(
        controls.map(|control| control.point.product(control.weight, metric)),
        s,
        t,
    );
    let denominator = bilinear(controls.map(|control| control.weight), s, t);
    numerator.product(
        denominator
            .inverse_even(metric)
            .expect("conditioned denominator"),
        metric,
    )
}

fn grade_three_pair(left: Control, right: Control) -> f64 {
    let metric = [1.0, 1.0, 0.0];
    left.point
        .product(left.weight, metric)
        .product(right.weight.reverse(), metric)
        .add(
            right
                .point
                .product(right.weight, metric)
                .product(left.weight.reverse(), metric),
        )
        .0[7]
}

fn paper_example(unscaled_fourth_weight: bool) -> [Control; 4] {
    [
        Control {
            point: Multivector::vector(0.0, 2.0, 0.0),
            weight: Multivector::scalar(1.0),
        },
        Control {
            point: Multivector::vector(0.0, 1.0, 0.0),
            weight: Multivector::even(0.0, 1.0, 0.0, -1.0),
        },
        Control {
            point: Multivector::vector(0.0, -2.0, 0.0),
            // The paper's coefficient matrix disambiguates an OCR artifact:
            // this is 2e12 + e23, followed by `w3 = -3(...)`.
            weight: Multivector::even(0.0, 2.0, 0.0, 1.0),
        },
        Control {
            point: Multivector::vector(0.0, -1.0, 0.0),
            weight: if unscaled_fourth_weight {
                Multivector::even(2.0, 0.0, 1.0, 0.0)
            } else {
                Multivector::even(-6.0, 0.0, -3.0, 0.0)
            },
        },
    ]
}

fn scaled_paper_example(scales: [f64; 4]) -> [Control; 4] {
    std::array::from_fn(|index| {
        let control = paper_example(false)[index];
        Control {
            point: control.point,
            weight: control.weight.scale(scales[index]),
        }
    })
}

/// Equal-budget experiment using the exact resident atlas, not a substitute
/// regular grid. Chord error is sampled in the *warped* parameter triangles.
#[test]
fn radial_atlas_concentration_is_a_budget_neutral_tradeoff() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../ingots/demos/classic_quilting_lod/assets/sha256/695152acd242f5b88a6ac4074f6855fea35b12bf16436308ba925b8cc962803e.bin");
    let atlas = crate::decode(&std::fs::read(path).expect("resident atlas artifact")).unwrap();
    let tile = atlas.patches.iter().find(|p| p.key == crate::AtlasKey::new(8,8,8)).unwrap();
    let samples = [[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5],[1.0/3.0;3]];
    let corners = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
    for (name,weights,center) in [
        ("paper-centered",[1.0;4],[0.5,0.5]),
        ("paper-offset",[1.0;4],[0.12,0.76]),
        ("squeezed-centered",[0.02,0.4,0.4,8.0],[0.5,0.5]),
    ] {
        let controls = scaled_paper_example(weights);
        let position = |uv: [f64;2]| {
            let p = evaluate(controls,uv[0],uv[1]);
            [p.0[1],p.0[2],p.0[4]]
        };
        for strength in [0.0625,0.25,1.0,4.0,16.0] {
            let mut worst = 0.0_f64;
            let mut weighted_squared = 0.0_f64;
            let mut covered_area = 0.0_f64;
            let mut triangles = 0;
            for child in 0..4 {
                let vertex_uv = |index: u32| {
                    let [a,b,c] = atlas.vertices[index as usize].barycentric.map(f64::from);
                    let d = a+b+strength*c;
                    std::array::from_fn::<_,2,_>(|axis|
                        (a*corners[child][axis]+b*corners[(child+1)%4][axis]+strength*c*center[axis])/d)
                };
                for triangle in &atlas.triangles[tile.first_triangle as usize..(tile.first_triangle+tile.triangle_count) as usize] {
                    triangles += 1;
                    let uv = triangle.indices.map(vertex_uv);
                    let p = uv.map(position);
                    let area = 0.5*((uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[1][1]-uv[0][1])*(uv[2][0]-uv[0][0]));
                    assert!(area>0.0,"no parameter-space triangle inversion");
                    covered_area += area;
                    for bary in samples {
                        let at = std::array::from_fn(|axis| (0..3).map(|i| bary[i]*uv[i][axis]).sum());
                        let exact = position(at);
                        let flat: [f64;3] = std::array::from_fn(|axis| (0..3).map(|i| bary[i]*p[i][axis]).sum());
                        let error_squared = (0..3).map(|i| (exact[i]-flat[i]).powi(2)).sum::<f64>();
                        worst = worst.max(error_squared.sqrt());
                        weighted_squared += area*error_squared/(samples.len() as f64);
                    }
                }
            }
            assert_eq!(triangles,4*tile.triangle_count);
            assert!((covered_area-1.0).abs()<0.000002);
            assert!(worst.is_finite() && weighted_squared.is_finite());
            eprintln!("{name}: k={strength}, triangles={triangles}, sampled_max_chord={worst:.8}, parameter_area_rms={:.8}",(weighted_squared/covered_area).sqrt());
        }
    }
}

fn denominator_norm_squared(controls: [Control; 4], s: f64, t: f64) -> f64 {
    let metric = [1.0, 1.0, 0.0];
    let denominator = bilinear(controls.map(|control| control.weight), s, t);
    denominator.product(denominator.reverse(), metric).0[0]
}

fn reparameterize(value: f64, first: f64, second: f64) -> f64 {
    second * value / (first * (1.0 - value) + second * value)
}

pub(crate) fn paper_sample(s: f64, t: f64) -> ([f64; 3], f64) {
    let value = evaluate(paper_example(false), s, t);
    ([value.0[1], value.0[2], value.0[4]], value.0[7])
}

pub(crate) fn paper_reconciliation_scale() -> f64 {
    let controls = paper_example(true);
    -grade_three_pair(controls[1], controls[2]) / grade_three_pair(controls[0], controls[3])
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn paper_reconciliation_recovers_negative_three_scale() {
    assert_close(paper_reconciliation_scale(), -3.0, 1.0e-12);
}

#[test]
fn paper_patch_is_vector_valued_and_lies_on_its_quartic() {
    let controls = paper_example(false);
    for s_step in 0..=8 {
        for t_step in 0..=8 {
            let sample = evaluate(controls, f64::from(s_step) / 8.0, f64::from(t_step) / 8.0);
            assert_close(sample.0[7], 0.0, 2.0e-12);
            let [x, y, z] = [sample.0[1], sample.0[2], sample.0[4]];
            let radius_squared = x * x + y * y;
            let implicit =
                radius_squared * radius_squared - 8.0 * x * x - 5.0 * y * y + 12.0 * z * z + 4.0;
            assert_close(implicit, 0.0, 2.0e-9);
        }
    }
}

#[test]
fn rank_one_corner_scaling_only_reparameterizes_the_paper_patch() {
    // A separable corner scale a_ij = lambda_i * mu_j is exactly the
    // rank-one relation a00*a11 = a10*a01. Bilinear interpolation factors
    // into a positive scalar times the original homogeneous patch at the two
    // rationally reparameterized coordinates below.
    let scales = [2.807_198_5, 0.140_730_57, 0.096_135_378, 0.0];
    let scales = [
        scales[0],
        scales[1],
        scales[2],
        scales[1] * scales[2] / scales[0],
    ];
    assert_close(scales[0] * scales[3], scales[1] * scales[2], 1.0e-14);

    let scaled = scaled_paper_example(scales);
    let lambda_0 = 1.0;
    let lambda_1 = scales[1] / scales[0];
    let mu_0 = scales[0];
    let mu_1 = scales[2];
    for s_step in 0..=16 {
        for t_step in 0..=16 {
            let s = f64::from(s_step) / 16.0;
            let t = f64::from(t_step) / 16.0;
            let mapped_s = reparameterize(s, lambda_0, lambda_1);
            let mapped_t = reparameterize(t, mu_0, mu_1);
            let actual = evaluate(scaled, s, t);
            let expected = evaluate(paper_example(false), mapped_s, mapped_t);
            for blade in 0..8 {
                assert_close(actual.0[blade], expected.0[blade], 2.0e-9);
            }
        }
    }
}

#[test]
fn captured_independent_weights_leave_the_vector_valued_locus() {
    let cases: [[f64; 4]; 2] = [
        [2.807_198_5, 0.140_730_57, 0.096_135_378, 0.955_935_66],
        [0.055_783_328, 0.024_856_215, 3.487_446_8, 0.431_913_82],
    ];
    for scales in cases {
        assert!(
            (scales[0] * scales[3] - scales[1] * scales[2]).abs() > 1.0e-3,
            "captured state unexpectedly satisfies the rank-one relation"
        );
        let controls = scaled_paper_example(scales);
        let mut maximum_grade_three = 0.0_f64;
        for s_step in 0..=256 {
            for t_step in 0..=256 {
                let sample = evaluate(
                    controls,
                    f64::from(s_step) / 256.0,
                    f64::from(t_step) / 256.0,
                );
                maximum_grade_three = maximum_grade_three.max(sample.0[7].abs());
            }
        }
        assert!(
            maximum_grade_three > 0.1,
            "captured state should expose a material grade-three residual; got {maximum_grade_three}"
        );
    }
}

#[test]
fn positive_paper_weight_scales_have_no_affine_pole() {
    // For the paper weights, |W|^2 = a^2 + b^2 with
    //   a = w00(1-s)(1-t) - 6*w11*s*t
    //   b = w10*s(1-t) + 2*w01*(1-s)*t.
    // With positive scales, b can vanish on the closed square only at (0,0)
    // and (1,1); a is respectively w00 and -6*w11 there. Thus the norm has no
    // zero even though it may become small enough to magnify an invalid
    // grade-one projection dramatically.
    let cases: [[f64; 4]; 4] = [
        [1.0, 1.0, 1.0, 1.0],
        [2.807_198_5, 0.140_730_57, 0.096_135_378, 0.955_935_66],
        [0.055_783_328, 0.024_856_215, 3.487_446_8, 0.431_913_82],
        [1.9, 1.9, 0.001, 0.001],
    ];
    for scales in cases {
        let controls = scaled_paper_example(scales);
        for s_step in 0..=128 {
            for t_step in 0..=128 {
                let s = f64::from(s_step) / 128.0;
                let t = f64::from(t_step) / 128.0;
                let scalar = scales[0] * (1.0 - s) * (1.0 - t) - 6.0 * scales[3] * s * t;
                let e12 = scales[1] * s * (1.0 - t) + 2.0 * scales[2] * (1.0 - s) * t;
                let expected = scalar * scalar + e12 * e12;
                let actual = denominator_norm_squared(controls, s, t);
                assert_close(actual, expected, 2.0e-12);
                assert!(actual > 0.0, "positive scales reached a denominator pole");
            }
        }
    }
}

#[test]
fn moving_center_four_triangle_fan_is_an_exact_patch_decomposition() {
    let cases = [
        ([1.0, 1.0, 1.0, 1.0], [0.5, 0.5]),
        ([0.02, 8.0, 8.0, 8.0], [0.13, 0.79]),
        ([8.0, 0.02, 8.0, 8.0], [0.86, 0.18]),
    ];
    for (scales, center) in cases {
        let controls = scaled_paper_example(scales);
        for domain in triangle_domains(center) {
            for b_step in 0..=16 {
                for c_step in 0..=(16 - b_step) {
                    let b = f64::from(b_step) / 16.0;
                    let c = f64::from(c_step) / 16.0;
                    let bary = [1.0 - b - c, b, c];
                    let parameter = triangle_parameter(domain, bary);
                    assert_multivector_close(
                        evaluate_triangle(controls, domain, bary),
                        evaluate(controls, parameter[0], parameter[1]),
                        2.0e-9,
                    );
                }
            }
        }
    }
}

#[test]
fn neighboring_fan_patches_share_the_complete_curved_edge() {
    let controls = scaled_paper_example([0.02, 8.0, 8.0, 8.0]);
    let domains = triangle_domains([0.17, 0.83]);
    for first in 0..4 {
        let second = (first + 1) % 4;
        for step in 0..=64 {
            let t = f64::from(step) / 64.0;
            // Each neighboring pair shares its first patch's B--C edge and
            // its second patch's A--C edge, with identical orientation from
            // their common outer corner toward the movable center.
            let left = evaluate_triangle(controls, domains[first], [0.0, 1.0 - t, t]);
            let right = evaluate_triangle(controls, domains[second], [1.0 - t, 0.0, t]);
            assert_multivector_close(left, right, 2.0e-9);
        }
    }
}

#[test]
fn neighboring_fan_patches_share_the_surface_tangent_plane() {
    let controls = paper_example(false);
    let domains = triangle_domains([0.17, 0.83]);
    for first in 0..4 {
        let second = (first + 1) % 4;
        for step in 1..64 {
            let t = f64::from(step) / 64.0;
            let left = evaluate_triangle_differential(
                controls,
                domains[first],
                [0.0, 1.0 - t, t],
            );
            let right = evaluate_triangle_differential(
                controls,
                domains[second],
                [1.0 - t, 0.0, t],
            );
            assert_multivector_close(left.value, right.value, 2.0e-9);
            let left_normal = triangle_normal(left).expect("left child has a tangent plane");
            let right_normal = triangle_normal(right).expect("right child has a tangent plane");
            let agreement = left_normal[0] * right_normal[0]
                + left_normal[1] * right_normal[1]
                + left_normal[2] * right_normal[2];
            assert_close(agreement, 1.0, 2.0e-10);
        }
    }
}

#[test]
fn exact_restriction_reduces_local_display_error_on_regular_twisted_regions() {
    let saddle = [
        Control { point: Multivector::vector(0.0, 0.0, 0.0), weight: Multivector::scalar(1.0) },
        Control { point: Multivector::vector(1.0, 0.0, 0.0), weight: Multivector::scalar(1.0) },
        Control { point: Multivector::vector(0.0, 1.0, 0.0), weight: Multivector::scalar(1.0) },
        Control { point: Multivector::vector(1.0, 1.0, 8.0), weight: Multivector::scalar(1.0) },
    ];
    // Rank-one scaling keeps the squeezed example vector-valued. It isolates
    // difficult parameterization from the separately tested invalid weights.
    let cases = [
        ("saddle", saddle),
        ("paper", paper_example(false)),
        ("squeezed-valid", scaled_paper_example([0.02, 0.4, 0.4, 8.0])),
    ];
    let root_domain = triangle_domains([0.5, 0.5])[0];
    let anchor = triangle_parameter(root_domain, [1.0 / 3.0; 3]);
    for (name, controls) in cases {
        let mut errors = Vec::new();
        let mut normal_spreads = Vec::new();
        for level in 0..=6 {
            let scale = 2.0_f64.powi(-level);
            let shrink = |point: [f64; 2]| {
                [
                    anchor[0] + scale * (point[0] - anchor[0]),
                    anchor[1] + scale * (point[1] - anchor[1]),
                ]
            };
            let domain = TriangleDomain {
                a: shrink(root_domain.a),
                b: shrink(root_domain.b),
                c: shrink(root_domain.c),
            };
            let corners = [
                evaluate_triangle(controls, domain, [1.0, 0.0, 0.0]),
                evaluate_triangle(controls, domain, [0.0, 1.0, 0.0]),
                evaluate_triangle(controls, domain, [0.0, 0.0, 1.0]),
            ];
            let center = evaluate_triangle_differential(controls, domain, [1.0 / 3.0; 3]);
            let normal = triangle_normal(center).expect("regular region at anchor");
            let mut error = 0.0_f64;
            let mut normal_spread = 0.0_f64;
            // Interior checks avoid boundary singularities of the paper's
            // authored patch; this gate concerns regular regions only.
            for b_step in 1..8 {
                for c_step in 1..(8 - b_step) {
                    let b = f64::from(b_step) / 8.0;
                    let c = f64::from(c_step) / 8.0;
                    let bary = [1.0 - b - c, b, c];
                    let actual = evaluate_triangle_differential(controls, domain, bary);
                    assert_close(actual.value.0[7], 0.0, 1.0e-10);
                    let flat = corners[0].scale(bary[0])
                        .add(corners[1].scale(bary[1]))
                        .add(corners[2].scale(bary[2]));
                    let difference = actual.value.add(flat.scale(-1.0));
                    let distance = (difference.0[1].powi(2)
                        + difference.0[2].powi(2) + difference.0[4].powi(2)).sqrt();
                    error = error.max(distance);
                    let actual_normal = triangle_normal(actual).expect("regular interior sample");
                    let cosine: f64 = normal.iter().zip(actual_normal).map(|(a, b)| a * b).sum();
                    normal_spread = normal_spread.max(1.0 - cosine);
                }
            }
            errors.push(error);
            normal_spreads.push(normal_spread);
        }
        eprintln!("{name}: local chord error by restriction depth {errors:?}");
        eprintln!("{name}: local normal spread (1-cos) {normal_spreads:?}");
        assert!(errors[6] < errors[0] / 100.0, "{name}: refinement failed to reduce local error");
        assert!(normal_spreads[6] < normal_spreads[0] / 100.0,
            "{name}: refinement failed to reduce local normal variation");
        if name == "saddle" {
            // For X(u,v)=(u,v,8uv), affine restriction scales the quadratic
            // departure from a chord exactly by scale squared.
            for level in 1..errors.len() {
                assert_close(errors[level] / errors[level - 1], 0.25, 1.0e-8);
            }
        }
    }
}

#[test]
fn even_euclidean_clifford_subalgebra_is_hamilton_quaternions() {
    let metric = [1.0, 1.0, 1.0];
    let left = [0.7, -1.1, 0.4, 2.3];
    let right = [-0.2, 1.7, -0.6, 0.9];

    // Hamilton (w, x, y, z) maps to 1, -e23, -e31, -e12. Since the
    // canonical stored blade is e13=-e31, its coefficient is +y.
    let embed = |q: [f64; 4]| Multivector::even(q[0], -q[3], q[2], -q[1]);
    let product = embed(left).product(embed(right), metric);
    let hamilton = [
        left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3],
        left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0],
    ];
    let expected = embed(hamilton);
    for blade in 0..8 {
        assert_close(product.0[blade], expected.0[blade], 1.0e-12);
    }
}
