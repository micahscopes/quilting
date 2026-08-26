//! Exact conservative screen-domain classification for rational QB patches.
//!
//! Perspective projection is unbounded at the camera plane, and a Möbius map
//! is unbounded at its pole. A screen-uniform tessellator must not interpret
//! either boundary as a request for infinite LoD. Instead, it first classifies
//! each exact rational child against the shader's fully-faded denominator band
//! and the six homogeneous view-frustum planes.
//!
//! For a transformed QB patch, the Möbius denominator is affine quaternion
//! data `D(b)`. Its squared norm is quadratic over the barycentric simplex.
//! The homogeneous numerator of every projected clip plane is quadratic too:
//! `plane.xyz · imag(N(b) conj(D(b))) + plane.w |D(b)|²`. We find the true
//! extrema of those quadratics on the closed triangle, including edge and
//! interior stationary points, rather than relying on a finite sample stencil.

use crate::patch::QBTriPatch;
use crate::quaternion::Quat;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenDomainPolicy {
    /// Denominator norm² at or below which the vertex shader's smooth fade is
    /// exactly zero. The current shader uses `smoothstep(1e-4, 1e-3, |D|²)`.
    pub fade_zero_norm_sq: f64,
    /// Relative tolerance for a homogeneous clip-plane sign decision.
    pub clip_relative_epsilon: f64,
}

impl Default for ScreenDomainPolicy {
    fn default() -> Self {
        Self {
            fade_zero_norm_sq: 1.0e-4,
            clip_relative_epsilon: 1.0e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenDomainClass {
    /// The complete leaf is in front of every clip plane and outside the fully
    /// faded Möbius-pole band, so its screen metric is finite.
    FiniteVisible,
    /// The complete leaf lies in the shader's exactly-zero fade band.
    FullyFaded,
    /// The complete leaf lies outside at least one homogeneous frustum plane.
    OutsideFrustum,
    /// The leaf crosses the exactly-zero fade boundary and must be restricted
    /// before ordinary screen metrics are meaningful.
    FadeBoundary,
    /// The leaf crosses one or more view-frustum boundaries.
    ClipBoundary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenDomainDiagnostic {
    pub class: ScreenDomainClass,
    pub denominator_norm_sq_range: [f64; 2],
    /// Ranges for left, right, bottom, top, near, and far homogeneous planes.
    pub clip_plane_ranges: [[f64; 2]; 6],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenDomainError {
    InvalidPolicy,
    InvalidProjection,
    NonFinitePatch,
}

impl std::fmt::Display for ScreenDomainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(formatter, "invalid screen-domain policy"),
            Self::InvalidProjection => write!(formatter, "invalid screen-domain projection"),
            Self::NonFinitePatch => write!(formatter, "non-finite QB patch controls"),
        }
    }
}

impl std::error::Error for ScreenDomainError {}

fn segment_origin_minimum(a: Quat, b: Quat) -> (f64, f64) {
    let direction = b - a;
    let parameter = (-a.dot(direction) / direction.norm_sq().max(1.0e-300)).clamp(0.0, 1.0);
    ((a + direction * parameter).norm_sq(), parameter)
}

/// Exact minimum of `|Σ bᵢ weightᵢ|²` over the closed barycentric simplex,
/// together with one barycentric point attaining it.
///
/// For a Möbius-transformed Euclidean triangle this is also the point of
/// greatest conformal dilation. Returning the witness lets screen-space
/// diagnostics probe that interior extremum instead of hoping a fixed sample
/// stencil happens to land near it.
pub fn denominator_norm_sq_minimum(patch: &QBTriPatch) -> (f64, [f64; 3]) {
    let [a, b, c] = patch.weights;
    let ab = b - a;
    let ac = c - a;
    let scale = ab.norm_sq().max(ac.norm_sq()).max(1.0e-300);
    let gram = ab.norm_sq() * ac.norm_sq() - ab.dot(ac).powi(2);
    if gram <= 1.0e-24 * scale * scale {
        let (ab_value, ab_parameter) = segment_origin_minimum(a, b);
        let (ac_value, ac_parameter) = segment_origin_minimum(a, c);
        let (bc_value, bc_parameter) = segment_origin_minimum(b, c);
        return [
            (ab_value, [1.0 - ab_parameter, ab_parameter, 0.0]),
            (ac_value, [1.0 - ac_parameter, 0.0, ac_parameter]),
            (bc_value, [0.0, 1.0 - bc_parameter, bc_parameter]),
        ]
        .into_iter()
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .expect("three simplex edges are always present");
    }

    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a.norm_sq(), [1.0, 0.0, 0.0]);
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b.norm_sq(), [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let parameter = d1 / (d1 - d3);
        return (
            (a + ab * parameter).norm_sq(),
            [1.0 - parameter, parameter, 0.0],
        );
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c.norm_sq(), [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let parameter = d2 / (d2 - d6);
        return (
            (a + ac * parameter).norm_sq(),
            [1.0 - parameter, 0.0, parameter],
        );
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = c - b;
        return (
            (b + bc * parameter).norm_sq(),
            [0.0, 1.0 - parameter, parameter],
        );
    }

    let inverse_sum = 1.0 / (va + vb + vc);
    let barycentric = [va * inverse_sum, vb * inverse_sum, vc * inverse_sum];
    (
        (a + ab * barycentric[1] + ac * barycentric[2]).norm_sq(),
        barycentric,
    )
}

/// Exact range of `|Σ bᵢ weightᵢ|²` over the closed barycentric simplex.
/// The minimum is the squared distance from the origin to the affine
/// quaternion triangle. Convexity places the maximum at a vertex.
pub fn denominator_norm_sq_range(patch: &QBTriPatch) -> [f64; 2] {
    let [a, b, c] = patch.weights;
    let minimum = denominator_norm_sq_minimum(patch).0;
    [minimum, a.norm_sq().max(b.norm_sq()).max(c.norm_sq())]
}

fn quadratic_value(coefficients: [f64; 6], u: f64, v: f64) -> f64 {
    let [constant, linear_u, linear_v, square_u, cross_uv, square_v] = coefficients;
    constant
        + linear_u * u
        + linear_v * v
        + square_u * u * u
        + 2.0 * cross_uv * u * v
        + square_v * v * v
}

fn edge_stationary(start: f64, middle: f64, end: f64) -> Option<f64> {
    let square = 2.0 * (end + start - 2.0 * middle);
    let linear = end - start - square;
    let scale = start.abs().max(middle.abs()).max(end.abs()).max(1.0e-300);
    (square.abs() > 1.0e-14 * scale)
        .then(|| -linear / (2.0 * square))
        .filter(|parameter| *parameter > 0.0 && *parameter < 1.0)
}

/// Exact extrema of a scalar quadratic on the closed unit simplex, recovered
/// from its values at three vertices and three edge midpoints.
fn quadratic_simplex_range(values: [f64; 6]) -> [f64; 2] {
    let [at_0, at_1, at_2, mid_01, mid_02, mid_12] = values;
    let square_u = 2.0 * (at_1 + at_0 - 2.0 * mid_01);
    let square_v = 2.0 * (at_2 + at_0 - 2.0 * mid_02);
    let linear_u = at_1 - at_0 - square_u;
    let linear_v = at_2 - at_0 - square_v;
    let cross_uv =
        2.0 * (mid_12 - at_0 - 0.5 * (linear_u + linear_v) - 0.25 * (square_u + square_v));
    let coefficients = [at_0, linear_u, linear_v, square_u, cross_uv, square_v];
    let mut candidates = vec![at_0, at_1, at_2];
    if let Some(parameter) = edge_stationary(at_0, mid_01, at_1) {
        candidates.push(quadratic_value(coefficients, parameter, 0.0));
    }
    if let Some(parameter) = edge_stationary(at_0, mid_02, at_2) {
        candidates.push(quadratic_value(coefficients, 0.0, parameter));
    }
    if let Some(parameter) = edge_stationary(at_2, mid_12, at_1) {
        candidates.push(quadratic_value(coefficients, parameter, 1.0 - parameter));
    }
    let determinant = square_u * square_v - cross_uv * cross_uv;
    let coefficient_scale = square_u
        .abs()
        .max(square_v.abs())
        .max(cross_uv.abs())
        .max(1.0e-300);
    if determinant.abs() > 1.0e-14 * coefficient_scale * coefficient_scale {
        let u = 0.5 * (cross_uv * linear_v - square_v * linear_u) / determinant;
        let v = 0.5 * (cross_uv * linear_u - square_u * linear_v) / determinant;
        if u > 0.0 && v > 0.0 && u + v < 1.0 {
            candidates.push(quadratic_value(coefficients, u, v));
        }
    }
    candidates
        .into_iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |range, value| {
            [range[0].min(value), range[1].max(value)]
        })
}

fn homogeneous_plane_value(patch: &QBTriPatch, plane: [f64; 4], bary: [f64; 3]) -> f64 {
    let numerator = (patch.positions[0] * patch.weights[0]) * bary[0]
        + (patch.positions[1] * patch.weights[1]) * bary[1]
        + (patch.positions[2] * patch.weights[2]) * bary[2];
    let denominator =
        patch.weights[0] * bary[0] + patch.weights[1] * bary[1] + patch.weights[2] * bary[2];
    let homogeneous_position = numerator * denominator.conj();
    plane[0] * homogeneous_position.x
        + plane[1] * homogeneous_position.y
        + plane[2] * homogeneous_position.z
        + plane[3] * denominator.norm_sq()
}

fn clip_planes(view_projection: &[f64; 16]) -> [[f64; 4]; 6] {
    let row = |index: usize| {
        [
            view_projection[index],
            view_projection[4 + index],
            view_projection[8 + index],
            view_projection[12 + index],
        ]
    };
    let add = |left: [f64; 4], right: [f64; 4], sign: f64| {
        std::array::from_fn(|index| left[index] + sign * right[index])
    };
    let row_0 = row(0);
    let row_1 = row(1);
    let row_2 = row(2);
    let row_3 = row(3);
    [
        add(row_3, row_0, 1.0),
        add(row_3, row_0, -1.0),
        add(row_3, row_1, 1.0),
        add(row_3, row_1, -1.0),
        add(row_3, row_2, 1.0),
        add(row_3, row_2, -1.0),
    ]
}

pub fn diagnose_screen_domain(
    transformed_patch: &QBTriPatch,
    view_projection: &[f64; 16],
    policy: ScreenDomainPolicy,
) -> Result<ScreenDomainDiagnostic, ScreenDomainError> {
    if !policy.fade_zero_norm_sq.is_finite()
        || policy.fade_zero_norm_sq < 0.0
        || !policy.clip_relative_epsilon.is_finite()
        || policy.clip_relative_epsilon < 0.0
    {
        return Err(ScreenDomainError::InvalidPolicy);
    }
    if view_projection.iter().any(|value| !value.is_finite()) {
        return Err(ScreenDomainError::InvalidProjection);
    }
    if transformed_patch
        .positions
        .iter()
        .chain(transformed_patch.weights.iter())
        .flat_map(|quaternion| [quaternion.w, quaternion.x, quaternion.y, quaternion.z])
        .any(|value| !value.is_finite())
    {
        return Err(ScreenDomainError::NonFinitePatch);
    }

    let denominator_norm_sq_range = denominator_norm_sq_range(transformed_patch);
    let sample_barycentrics = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let clip_plane_ranges = clip_planes(view_projection).map(|plane| {
        quadratic_simplex_range(
            sample_barycentrics
                .map(|barycentric| homogeneous_plane_value(transformed_patch, plane, barycentric)),
        )
    });

    let class = if denominator_norm_sq_range[1] <= policy.fade_zero_norm_sq {
        ScreenDomainClass::FullyFaded
    } else if denominator_norm_sq_range[0] < policy.fade_zero_norm_sq {
        ScreenDomainClass::FadeBoundary
    } else if clip_plane_ranges.iter().any(|range| {
        let scale = range[0].abs().max(range[1].abs()).max(1.0e-300);
        range[1] < -policy.clip_relative_epsilon * scale
    }) {
        ScreenDomainClass::OutsideFrustum
    } else if clip_plane_ranges.iter().all(|range| {
        let scale = range[0].abs().max(range[1].abs()).max(1.0e-300);
        range[0] >= -policy.clip_relative_epsilon * scale
    }) {
        ScreenDomainClass::FiniteVisible
    } else {
        ScreenDomainClass::ClipBoundary
    };
    Ok(ScreenDomainDiagnostic {
        class,
        denominator_norm_sq_range,
        clip_plane_ranges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::{Mobius, Quat};

    fn perspective() -> [f64; 16] {
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.000_200_02,
            -1.0,
            0.0,
            0.0,
            -0.020_002_000_2,
            0.0,
        ]
    }

    #[test]
    fn quadratic_range_includes_edge_and_interior_stationary_points() {
        let quadratic = |u: f64, v: f64| 3.0 * (u - 0.2).powi(2) + 2.0 * (v - 0.3).powi(2) - 0.75;
        let values = [
            quadratic(0.0, 0.0),
            quadratic(1.0, 0.0),
            quadratic(0.0, 1.0),
            quadratic(0.5, 0.0),
            quadratic(0.0, 0.5),
            quadratic(0.5, 0.5),
        ];
        let range = quadratic_simplex_range(values);
        assert!((range[0] + 0.75).abs() < 1.0e-12);
        for u_step in 0..=100 {
            for v_step in 0..=100 - u_step {
                let value = quadratic(u_step as f64 / 100.0, v_step as f64 / 100.0);
                assert!(value >= range[0] - 1.0e-12 && value <= range[1] + 1.0e-12);
            }
        }
    }

    #[test]
    fn finite_front_patch_and_offscreen_patch_are_distinguished() {
        let visible = QBTriPatch::flat([-0.2, -0.2, -2.0], [0.2, -0.2, -2.0], [0.0, 0.2, -2.0]);
        let outside = QBTriPatch::flat([4.0, -0.2, -2.0], [5.0, -0.2, -2.0], [4.5, 0.2, -2.0]);
        let policy = ScreenDomainPolicy::default();
        assert_eq!(
            diagnose_screen_domain(&visible, &perspective(), policy)
                .unwrap()
                .class,
            ScreenDomainClass::FiniteVisible,
        );
        assert_eq!(
            diagnose_screen_domain(&outside, &perspective(), policy)
                .unwrap()
                .class,
            ScreenDomainClass::OutsideFrustum,
        );
    }

    #[test]
    fn camera_plane_crossing_is_a_boundary_not_infinite_lod() {
        let crossing = QBTriPatch::flat([-0.2, -0.2, -2.0], [0.2, -0.2, -2.0], [0.0, 0.2, 1.0]);
        assert_eq!(
            diagnose_screen_domain(&crossing, &perspective(), ScreenDomainPolicy::default(),)
                .unwrap()
                .class,
            ScreenDomainClass::ClipBoundary,
        );
    }

    #[test]
    fn exact_fade_range_separates_cutout_and_boundary() {
        let positions = [
            Quat::from_point(-0.2, -0.2, -2.0),
            Quat::from_point(0.2, -0.2, -2.0),
            Quat::from_point(0.0, 0.2, -2.0),
        ];
        let faded = QBTriPatch::new(positions, [Quat::new(0.005, 0.0, 0.0, 0.0); 3]);
        let boundary = QBTriPatch::new(
            positions,
            [
                Quat::new(0.001, 0.0, 0.0, 0.0),
                Quat::new(0.1, 0.0, 0.0, 0.0),
                Quat::new(0.1, 0.0, 0.0, 0.0),
            ],
        );
        let policy = ScreenDomainPolicy::default();
        assert_eq!(
            diagnose_screen_domain(&faded, &perspective(), policy)
                .unwrap()
                .class,
            ScreenDomainClass::FullyFaded,
        );
        assert_eq!(
            diagnose_screen_domain(&boundary, &perspective(), policy)
                .unwrap()
                .class,
            ScreenDomainClass::FadeBoundary,
        );
    }

    #[test]
    fn inverted_patch_plane_ranges_bound_dense_samples() {
        let patch =
            QBTriPatch::flat([-0.8, -0.5, -3.2], [0.9, -0.4, -3.0], [-0.3, 0.8, -3.4]).transform(
                &Mobius::sphere_reflection(Quat::from_point(0.1, -0.1, -2.2), 0.8),
            );
        let diagnostic = diagnose_screen_domain(
            &patch,
            &perspective(),
            ScreenDomainPolicy {
                fade_zero_norm_sq: 0.0,
                ..ScreenDomainPolicy::default()
            },
        )
        .unwrap();
        let planes = clip_planes(&perspective());
        for u_step in 0..=80 {
            for v_step in 0..=80 - u_step {
                let barycentric = [
                    1.0 - (u_step + v_step) as f64 / 80.0,
                    u_step as f64 / 80.0,
                    v_step as f64 / 80.0,
                ];
                for (plane, range) in planes.iter().zip(diagnostic.clip_plane_ranges) {
                    let value = homogeneous_plane_value(&patch, *plane, barycentric);
                    assert!(value >= range[0] - 1.0e-10 && value <= range[1] + 1.0e-10);
                }
            }
        }
    }
}
