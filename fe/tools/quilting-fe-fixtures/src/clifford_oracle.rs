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
