//! Independent dense f64 Cl(4,1) oracle for the sparse Fe CGA map.
//!
//! The production ingot uses a named null basis and FCO-compiled vector dots.
//! This oracle instead expands values into an orthogonal 32-blade algebra with
//! signature (4,1), evaluates the complete sphere sandwich, and only then
//! converts the result back to the null basis.

#[derive(Clone, Copy, Debug, Default)]
struct Multivector([f64; 32]);

impl Multivector {
    fn product(self, other: Self) -> Self {
        let metric = [1.0, 1.0, 1.0, 1.0, -1.0];
        let mut result = [0.0; 32];
        for left_blade in 0_usize..32 {
            for right_blade in 0_usize..32 {
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
                let swaps = (0..5)
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
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReflectionSample {
    pub(crate) position: [f64; 3],
    pub(crate) weight: f64,
    pub(crate) null_residual: f64,
    pub(crate) nonvector_residual: f64,
}

fn null_vector(coefficients: [f64; 5]) -> Multivector {
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let [e1, e2, e3, eo, ei] = coefficients;
    let mut value = [0.0; 32];
    value[1] = e1;
    value[2] = e2;
    value[4] = e3;
    value[8] = (eo - ei) * inverse_sqrt_two;
    value[16] = (eo + ei) * inverse_sqrt_two;
    Multivector(value)
}

fn as_null_vector(value: Multivector) -> [f64; 5] {
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    [
        value.0[1],
        value.0[2],
        value.0[4],
        (value.0[8] + value.0[16]) * inverse_sqrt_two,
        (value.0[16] - value.0[8]) * inverse_sqrt_two,
    ]
}

fn dot(left: [f64; 5], right: [f64; 5]) -> f64 {
    null_vector(left).product(null_vector(right)).0[0]
}

fn embedded_point(point: [f64; 3]) -> [f64; 5] {
    let [x, y, z] = point;
    [x, y, z, 1.0, 0.5 * (x * x + y * y + z * z)]
}

fn dual_sphere(center: [f64; 3], radius: f64) -> [f64; 5] {
    let [x, y, z] = center;
    [
        x,
        y,
        z,
        1.0,
        0.5 * (x * x + y * y + z * z - radius * radius),
    ]
}

pub(crate) fn sphere_incidence(point: [f64; 3], center: [f64; 3], radius: f64) -> f64 {
    dot(embedded_point(point), dual_sphere(center, radius))
}

pub(crate) fn sphere_reflection(
    point: [f64; 3],
    center: [f64; 3],
    radius: f64,
) -> ReflectionSample {
    let point_value = null_vector(embedded_point(point));
    let sphere_value = null_vector(dual_sphere(center, radius));
    let sandwich = sphere_value.product(point_value).product(sphere_value);
    let homogeneous = as_null_vector(sandwich);
    let mut nonvector_residual: f64 = 0.0;
    for (blade, coefficient) in sandwich.0.iter().copied().enumerate() {
        if blade != 1 && blade != 2 && blade != 4 && blade != 8 && blade != 16 {
            nonvector_residual = nonvector_residual.max(coefficient.abs());
        }
    }
    let weight = homogeneous[3];
    let inverse_weight = weight.recip();
    let position = [
        homogeneous[0] * inverse_weight,
        homogeneous[1] * inverse_weight,
        homogeneous[2] * inverse_weight,
    ];
    let spatial_norm = homogeneous[0] * homogeneous[0]
        + homogeneous[1] * homogeneous[1]
        + homogeneous[2] * homogeneous[2];
    let null_residual =
        (spatial_norm - 2.0 * homogeneous[3] * homogeneous[4]) * inverse_weight * inverse_weight;
    ReflectionSample {
        position,
        weight,
        null_residual,
        nonvector_residual,
    }
}

fn closed_reflection(point: [f64; 3], center: [f64; 3], radius: f64) -> [f64; 3] {
    let displacement = sub(point, center);
    let scale = radius * radius / dot3(displacement, displacement);
    add(center, scale3(displacement, scale))
}

pub(crate) fn finite_difference_tangent(
    point: [f64; 3],
    tangent: [f64; 3],
    center: [f64; 3],
    radius: f64,
) -> [f64; 3] {
    let h = 1.0e-5;
    let lower = sphere_reflection(sub(point, scale3(tangent, h)), center, radius).position;
    let upper = sphere_reflection(add(point, scale3(tangent, h)), center, radius).position;
    scale3(sub(upper, lower), 0.5 / h)
}

pub(crate) fn xy_normal(point: [f64; 3], center: [f64; 3], radius: f64) -> [f64; 3] {
    normalize(cross(
        finite_difference_tangent(point, [1.0, 0.0, 0.0], center, radius),
        finite_difference_tangent(point, [0.0, 1.0, 0.0], center, radius),
    ))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|lane| left[lane] + right[lane])
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|lane| left[lane] - right[lane])
}

fn scale3(value: [f64; 3], scale: f64) -> [f64; 3] {
    value.map(|lane| lane * scale)
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(value: [f64; 3]) -> [f64; 3] {
    scale3(value, dot3(value, value).sqrt().recip())
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn dense_sandwich_is_the_classic_sphere_reflection_and_an_involution() {
    let centers = [[0.0, 0.0, 0.0], [0.35, -0.7, 1.1], [-1.3, 0.2, 0.4]];
    let points = [[1.0, 0.5, -0.25], [-0.8, 1.4, 2.0], [3.0, -2.0, 0.7]];
    let radii = [0.5, 1.25, 2.0];
    for center in centers {
        for point in points {
            for radius in radii {
                let sample = sphere_reflection(point, center, radius);
                let expected = closed_reflection(point, center, radius);
                for lane in 0..3 {
                    assert_close(sample.position[lane], expected[lane], 2.0e-12);
                }
                assert_close(sample.null_residual, 0.0, 2.0e-12);
                assert_close(sample.nonvector_residual, 0.0, 2.0e-15);
                let restored = sphere_reflection(sample.position, center, radius).position;
                for lane in 0..3 {
                    assert_close(restored[lane], point[lane], 2.0e-11);
                }
            }
        }
    }
}

#[test]
fn dense_null_basis_has_the_expected_sphere_incidence() {
    let center = [0.4, -0.2, 0.7];
    let radius = 1.75;
    let directions = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
    for direction in directions {
        let point = add(center, scale3(direction, radius));
        assert_close(sphere_incidence(point, center, radius), 0.0, 2.0e-15);
    }
    assert_close(
        dot(dual_sphere(center, radius), dual_sphere(center, radius)),
        radius * radius,
        2.0e-15,
    );
}

#[test]
fn dense_finite_differences_match_the_closed_reflection_jacobian() {
    let point = [0.7, -1.1, 1.6];
    let center = [-0.2, 0.4, 0.1];
    let radius = 1.3;
    let displacement = sub(point, center);
    let inverse_distance_squared = dot3(displacement, displacement).recip();
    let scale = radius * radius * inverse_distance_squared;
    for tangent in [[1.0, 0.0, 0.0], [0.3, -0.8, 0.5], [0.0, 0.0, 1.0]] {
        let expected = scale3(
            sub(
                tangent,
                scale3(
                    displacement,
                    2.0 * dot3(displacement, tangent) * inverse_distance_squared,
                ),
            ),
            scale,
        );
        let finite = finite_difference_tangent(point, tangent, center, radius);
        for lane in 0..3 {
            assert_close(finite[lane], expected[lane], 2.0e-10);
        }
    }
}
