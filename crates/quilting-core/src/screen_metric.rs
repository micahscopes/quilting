//! Exact local screen-space metric for rational QB patches.
//!
//! A transformed patch followed by perspective projection is a smooth map
//! from two barycentric parameters to screen pixels away from a Möbius pole
//! and the camera plane. Its Jacobian is available analytically: QB tangents
//! come from the rational quotient rule and perspective contributes the exact
//! derivative of `clip.xy / clip.w`. The pullback metric `G = JᵀJ` is the
//! quantity a screen-uniform tessellator should bound.

use crate::patch::{PatchDifferential, QBTriPatch};

/// Logical triangle edge order used by the tessellation and LOD pipelines.
///
/// `A` is opposite control point 0 (`p1 -> p2`), `B` is opposite control
/// point 1 (`p0 -> p2`), and `C` is opposite control point 2 (`p0 -> p1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchEdge {
    A,
    B,
    C,
}

impl PatchEdge {
    /// Patch coordinates and their derivative for a unit edge parameter.
    pub fn parameter_frame(self, parameter: f64) -> ([f64; 2], [f64; 2]) {
        match self {
            Self::A => ([1.0 - parameter, parameter], [-1.0, 1.0]),
            Self::B => ([0.0, parameter], [0.0, 1.0]),
            Self::C => ([parameter, 0.0], [1.0, 0.0]),
        }
    }
}

/// Error and sampling policy for a projected edge arc-length map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenArcOptions {
    /// Absolute error target for the whole edge, in pixels.
    pub tolerance_px: f64,
    /// Mandatory subdivisions guard against a narrow stretch peak falling
    /// between the first Simpson samples.
    pub min_depth: u8,
    /// Hard recursion bound near a projection singularity.
    pub max_depth: u8,
}

impl Default for ScreenArcOptions {
    fn default() -> Self {
        Self {
            tolerance_px: 0.05,
            min_depth: 2,
            max_depth: 14,
        }
    }
}

/// A monotone knot in a projected edge arc-length map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenArcKnot {
    pub parameter: f64,
    pub cumulative_px: f64,
}

/// Error-controlled numerical integral of the exact local screen metric.
///
/// The local derivative is analytic. Arc length itself generally has no
/// elementary closed form after perspective, so it is integrated adaptively.
/// The knots can be inverted to redistribute a uniform tessellation along the
/// actual projected curve rather than its barycentric parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenEdgeArc {
    pub length_px: f64,
    pub estimated_error_px: f64,
    /// Largest midpoint arc-length error introduced by treating one accepted
    /// knot interval as linear during inversion.
    pub max_mapping_error_px: f64,
    pub evaluations: u32,
    pub max_depth_reached: bool,
    pub knots: Vec<ScreenArcKnot>,
}

impl ScreenEdgeArc {
    /// Convert a normalized screen arc-length fraction back to the patch edge
    /// parameter. Linear interpolation is confined to one accepted adaptive
    /// interval; reducing [`ScreenArcOptions::tolerance_px`] tightens it.
    pub fn parameter_at_fraction(&self, fraction: f64) -> f64 {
        let fraction = fraction.clamp(0.0, 1.0);
        if self.length_px <= 0.0 || self.knots.len() < 2 {
            return fraction;
        }
        let target = fraction * self.length_px;
        let upper = self
            .knots
            .partition_point(|knot| knot.cumulative_px < target)
            .clamp(1, self.knots.len() - 1);
        let lower = upper - 1;
        let a = self.knots[lower];
        let b = self.knots[upper];
        let span = b.cumulative_px - a.cumulative_px;
        if span <= f64::EPSILON {
            return 0.5 * (a.parameter + b.parameter);
        }
        let local = ((target - a.cumulative_px) / span).clamp(0.0, 1.0);
        a.parameter + local * (b.parameter - a.parameter)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScreenArcError {
    InvalidOptions,
    UnprojectableSample { parameter: f64 },
}

impl std::fmt::Display for ScreenArcError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions => write!(formatter, "invalid screen arc integration options"),
            Self::UnprojectableSample { parameter } => write!(
                formatter,
                "patch edge is not projectable at parameter {parameter}"
            ),
        }
    }
}

impl std::error::Error for ScreenArcError {}

/// Exact first-order image of a patch parameter frame in screen pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenMetric {
    /// Projected sample position in viewport pixels, with the origin at the
    /// viewport centre. Translation is irrelevant to the metric.
    pub position_px: [f64; 2],
    /// Screen-pixel derivatives for the patch's `(u, v)` directions.
    pub tangent_u_px: [f64; 2],
    pub tangent_v_px: [f64; 2],
    /// Symmetric pullback metric `[g_uu, g_uv, g_vv] = JᵀJ`.
    pub metric: [f64; 3],
    /// Singular values of `J`, ascending: minimum and maximum local pixel
    /// stretch per unit parameter displacement.
    pub principal_stretch: [f64; 2],
    /// Absolute Jacobian determinant: local pixel area per parameter area.
    pub area_scale: f64,
}

impl ScreenMetric {
    /// Pixel length predicted for a parameter-space displacement.
    pub fn length(&self, delta: [f64; 2]) -> f64 {
        let [g_uu, g_uv, g_vv] = self.metric;
        (g_uu * delta[0] * delta[0] + 2.0 * g_uv * delta[0] * delta[1] + g_vv * delta[1] * delta[1])
            .max(0.0)
            .sqrt()
    }

    /// Local screen anisotropy. Infinite means one parameter direction has
    /// collapsed under projection.
    pub fn anisotropy(&self) -> f64 {
        if self.principal_stretch[0] > 0.0 {
            self.principal_stretch[1] / self.principal_stretch[0]
        } else {
            f64::INFINITY
        }
    }
}

/// Project an exact QB differential through a column-major WebGL
/// view-projection matrix. Returns `None` on/behind the camera plane or for a
/// non-finite frame; callers must subdivide or conservatively bound such a
/// crossing rather than manufacture a finite metric.
pub fn project_patch_differential(
    differential: PatchDifferential,
    view_projection: &[f64; 16],
    viewport: [f64; 2],
) -> Option<ScreenMetric> {
    if viewport[0] <= 0.0
        || viewport[1] <= 0.0
        || viewport.iter().any(|value| !value.is_finite())
        || view_projection.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let multiply = |point: [f64; 3], homogeneous_w: f64| -> [f64; 4] {
        std::array::from_fn(|row| {
            view_projection[row] * point[0]
                + view_projection[4 + row] * point[1]
                + view_projection[8 + row] * point[2]
                + view_projection[12 + row] * homogeneous_w
        })
    };
    let clip = multiply(differential.position, 1.0);
    if clip.iter().any(|value| !value.is_finite()) || clip[3] <= 1.0e-12 {
        return None;
    }

    let derivative = |tangent: [f64; 3]| -> Option<[f64; 2]> {
        let delta = multiply(tangent, 0.0);
        let denominator = clip[3] * clip[3];
        let result = [
            0.5 * viewport[0] * (delta[0] * clip[3] - clip[0] * delta[3]) / denominator,
            0.5 * viewport[1] * (delta[1] * clip[3] - clip[1] * delta[3]) / denominator,
        ];
        result
            .iter()
            .all(|value| value.is_finite())
            .then_some(result)
    };
    let tangent_u_px = derivative(differential.tangent_u)?;
    let tangent_v_px = derivative(differential.tangent_v)?;
    let g_uu = tangent_u_px[0].powi(2) + tangent_u_px[1].powi(2);
    let g_uv = tangent_u_px[0] * tangent_v_px[0] + tangent_u_px[1] * tangent_v_px[1];
    let g_vv = tangent_v_px[0].powi(2) + tangent_v_px[1].powi(2);
    let trace = g_uu + g_vv;
    let discriminant = ((g_uu - g_vv).powi(2) + 4.0 * g_uv.powi(2)).sqrt();
    let eigen_min = (0.5 * (trace - discriminant)).max(0.0);
    let eigen_max = (0.5 * (trace + discriminant)).max(0.0);
    let area_scale = (tangent_u_px[0] * tangent_v_px[1] - tangent_u_px[1] * tangent_v_px[0]).abs();
    let position_px = [
        0.5 * viewport[0] * clip[0] / clip[3],
        0.5 * viewport[1] * clip[1] / clip[3],
    ];
    let result = ScreenMetric {
        position_px,
        tangent_u_px,
        tangent_v_px,
        metric: [g_uu, g_uv, g_vv],
        principal_stretch: [eigen_min.sqrt(), eigen_max.sqrt()],
        area_scale,
    };
    result
        .position_px
        .iter()
        .chain(result.metric.iter())
        .chain(result.principal_stretch.iter())
        .chain(std::iter::once(&result.area_scale))
        .all(|value| value.is_finite())
        .then_some(result)
}

/// Evaluate the exact local screen metric of an already transformed QB patch.
/// Applying a Möbius map through [`QBTriPatch::transform`] before this call
/// keeps the rational surface and its quotient-rule derivatives exact.
pub fn patch_screen_metric(
    transformed_patch: &QBTriPatch,
    u: f64,
    v: f64,
    view_projection: &[f64; 16],
    viewport: [f64; 2],
) -> Option<ScreenMetric> {
    project_patch_differential(
        transformed_patch.eval_differential(u, v),
        view_projection,
        viewport,
    )
}

struct EdgeArcIntegrator<'a> {
    patch: &'a QBTriPatch,
    edge: PatchEdge,
    view_projection: &'a [f64; 16],
    viewport: [f64; 2],
    options: ScreenArcOptions,
    evaluations: u32,
}

impl EdgeArcIntegrator<'_> {
    fn speed(&mut self, parameter: f64) -> Result<f64, ScreenArcError> {
        self.evaluations += 1;
        let ([u, v], derivative) = self.edge.parameter_frame(parameter);
        patch_screen_metric(self.patch, u, v, self.view_projection, self.viewport)
            .map(|metric| metric.length(derivative))
            .filter(|speed| speed.is_finite())
            .ok_or(ScreenArcError::UnprojectableSample { parameter })
    }

    fn integrate_interval(
        &mut self,
        start: f64,
        end: f64,
        speed_start: f64,
        speed_middle: f64,
        speed_end: f64,
        coarse: f64,
        depth: u8,
        tolerance_px: f64,
        leaves: &mut Vec<(f64, f64, f64, f64, bool)>,
    ) -> Result<(), ScreenArcError> {
        let middle = 0.5 * (start + end);
        let left_middle = 0.5 * (start + middle);
        let right_middle = 0.5 * (middle + end);
        let speed_left_middle = self.speed(left_middle)?;
        let speed_right_middle = self.speed(right_middle)?;
        let left = simpson(start, middle, speed_start, speed_left_middle, speed_middle);
        let right = simpson(middle, end, speed_middle, speed_right_middle, speed_end);
        let refined = left + right;
        let error = (refined - coarse).abs() / 15.0;
        let mapping_error = 0.5 * (left - right).abs();
        let at_limit = depth >= self.options.max_depth;
        if (depth >= self.options.min_depth
            && error <= tolerance_px
            && mapping_error <= self.options.tolerance_px)
            || at_limit
        {
            let corrected = (refined + (refined - coarse) / 15.0).max(0.0);
            leaves.push((end, corrected, error, mapping_error, at_limit));
            return Ok(());
        }

        self.integrate_interval(
            start,
            middle,
            speed_start,
            speed_left_middle,
            speed_middle,
            left,
            depth + 1,
            tolerance_px * 0.5,
            leaves,
        )?;
        self.integrate_interval(
            middle,
            end,
            speed_middle,
            speed_right_middle,
            speed_end,
            right,
            depth + 1,
            tolerance_px * 0.5,
            leaves,
        )
    }
}

fn simpson(start: f64, end: f64, speed_start: f64, speed_middle: f64, speed_end: f64) -> f64 {
    (end - start) * (speed_start + 4.0 * speed_middle + speed_end) / 6.0
}

/// Integrate one projected patch edge and retain a monotone map suitable for
/// screen-uniform edge reparameterization.
///
/// Adjacent faces must identify their shared edge in one canonical endpoint
/// order. Reversing the order reverses `parameter`, but preserves both total
/// length and physical subdivision positions.
pub fn patch_screen_edge_arc(
    transformed_patch: &QBTriPatch,
    edge: PatchEdge,
    view_projection: &[f64; 16],
    viewport: [f64; 2],
    options: ScreenArcOptions,
) -> Result<ScreenEdgeArc, ScreenArcError> {
    if !options.tolerance_px.is_finite()
        || options.tolerance_px <= 0.0
        || options.min_depth > options.max_depth
        || options.max_depth > 24
    {
        return Err(ScreenArcError::InvalidOptions);
    }

    let mut integrator = EdgeArcIntegrator {
        patch: transformed_patch,
        edge,
        view_projection,
        viewport,
        options,
        evaluations: 0,
    };
    let speed_start = integrator.speed(0.0)?;
    let speed_middle = integrator.speed(0.5)?;
    let speed_end = integrator.speed(1.0)?;
    let coarse = simpson(0.0, 1.0, speed_start, speed_middle, speed_end);
    let mut leaves = Vec::new();
    integrator.integrate_interval(
        0.0,
        1.0,
        speed_start,
        speed_middle,
        speed_end,
        coarse,
        0,
        options.tolerance_px,
        &mut leaves,
    )?;

    let mut cumulative_px = 0.0;
    let mut estimated_error_px = 0.0;
    let mut max_mapping_error_px: f64 = 0.0;
    let mut max_depth_reached = false;
    let mut knots = Vec::with_capacity(leaves.len() + 1);
    knots.push(ScreenArcKnot {
        parameter: 0.0,
        cumulative_px: 0.0,
    });
    for (parameter, length_px, error_px, mapping_error_px, at_limit) in leaves {
        cumulative_px += length_px;
        estimated_error_px += error_px;
        max_mapping_error_px = max_mapping_error_px.max(mapping_error_px);
        max_depth_reached |= at_limit;
        knots.push(ScreenArcKnot {
            parameter,
            cumulative_px,
        });
    }
    Ok(ScreenEdgeArc {
        length_px: cumulative_px,
        estimated_error_px,
        max_mapping_error_px,
        evaluations: integrator.evaluations,
        max_depth_reached,
        knots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::{Mobius, Quat};

    fn perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> [f64; 16] {
        let scale = 1.0 / (0.5 * fov_y).tan();
        let range = 1.0 / (near - far);
        [
            scale / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            scale,
            0.0,
            0.0,
            0.0,
            0.0,
            (near + far) * range,
            -1.0,
            0.0,
            0.0,
            2.0 * near * far * range,
            0.0,
        ]
    }

    fn project(point: [f64; 3], matrix: &[f64; 16], viewport: [f64; 2]) -> [f64; 2] {
        let clip: [f64; 4] = std::array::from_fn(|row| {
            matrix[row] * point[0]
                + matrix[4 + row] * point[1]
                + matrix[8 + row] * point[2]
                + matrix[12 + row]
        });
        [
            0.5 * viewport[0] * clip[0] / clip[3],
            0.5 * viewport[1] * clip[1] / clip[3],
        ]
    }

    #[test]
    fn orthographic_identity_metric_has_expected_pixel_scales() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let metric = patch_screen_metric(
            &patch,
            0.2,
            0.3,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            [200.0, 100.0],
        )
        .unwrap();
        assert_eq!(metric.tangent_u_px, [100.0, 0.0]);
        assert_eq!(metric.tangent_v_px, [0.0, 50.0]);
        assert_eq!(metric.metric, [10_000.0, 0.0, 2_500.0]);
        assert_eq!(metric.principal_stretch, [50.0, 100.0]);
        assert_eq!(metric.area_scale, 5_000.0);
    }

    #[test]
    fn analytic_inverted_perspective_metric_matches_finite_differences() {
        let source = QBTriPatch::flat([-0.8, -0.5, -3.2], [0.9, -0.4, -3.0], [-0.3, 0.8, -3.4]);
        let transform = Mobius::sphere_reflection(Quat::from_point(0.1, -0.1, -2.2), 0.8);
        let patch = source.transform(&transform);
        let matrix = perspective(75.0_f64.to_radians(), 16.0 / 9.0, 0.01, 100.0);
        let viewport = [1600.0, 900.0];
        let u = 0.27;
        let v = 0.31;
        let metric = patch_screen_metric(&patch, u, v, &matrix, viewport).unwrap();
        let epsilon = 1.0e-6;
        let centre = project(patch.eval(u, v).to_point(), &matrix, viewport);
        let sample_u = project(patch.eval(u + epsilon, v).to_point(), &matrix, viewport);
        let sample_v = project(patch.eval(u, v + epsilon).to_point(), &matrix, viewport);
        let finite_u = [
            (sample_u[0] - centre[0]) / epsilon,
            (sample_u[1] - centre[1]) / epsilon,
        ];
        let finite_v = [
            (sample_v[0] - centre[0]) / epsilon,
            (sample_v[1] - centre[1]) / epsilon,
        ];
        for (analytic, finite) in metric
            .tangent_u_px
            .into_iter()
            .zip(finite_u)
            .chain(metric.tangent_v_px.into_iter().zip(finite_v))
        {
            let relative = (analytic - finite).abs() / analytic.abs().max(1.0);
            assert!(relative < 2.0e-5, "analytic={analytic} finite={finite}");
        }
    }

    #[test]
    fn camera_plane_crossing_has_no_finite_screen_metric() {
        let patch = QBTriPatch::flat([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]);
        let matrix = perspective(75.0_f64.to_radians(), 1.0, 0.01, 100.0);
        assert!(patch_screen_metric(&patch, 0.2, 0.2, &matrix, [100.0; 2]).is_none());
    }

    #[test]
    fn perspective_edge_arc_inverts_pixel_length_instead_of_parameter() {
        let patch = QBTriPatch::flat([-1.0, 0.0, -2.0], [1.0, 0.0, -6.0], [0.0, 1.0, -2.0]);
        let matrix = perspective(90.0_f64.to_radians(), 1.0, 0.01, 100.0);
        let arc = patch_screen_edge_arc(
            &patch,
            PatchEdge::C,
            &matrix,
            [200.0; 2],
            ScreenArcOptions {
                tolerance_px: 1.0e-7,
                min_depth: 3,
                max_depth: 18,
            },
        )
        .unwrap();
        let half_parameter = arc.parameter_at_fraction(0.5);
        assert!((arc.length_px - 66.666_666_666_7).abs() < 1.0e-6);
        assert!((half_parameter - 0.25).abs() < 1.0e-5);
        assert!(arc.estimated_error_px < 1.0e-7);
        assert!(!arc.max_depth_reached);
    }

    #[test]
    fn adjacent_reversed_edges_have_one_physical_arc_map() {
        let a = [-0.8, -0.4, -3.2];
        let b = [0.9, -0.2, -4.1];
        let left = QBTriPatch::flat(a, b, [-0.2, 0.9, -3.5]);
        let right = QBTriPatch::flat(b, a, [0.3, -1.0, -3.7]);
        let transform = Mobius::sphere_reflection(Quat::from_point(0.1, 0.2, -1.2), 0.9);
        let left = left.transform(&transform);
        let right = right.transform(&transform);
        let matrix = perspective(78.0_f64.to_radians(), 16.0 / 9.0, 0.01, 100.0);
        let options = ScreenArcOptions {
            tolerance_px: 1.0e-5,
            min_depth: 4,
            max_depth: 18,
        };
        let left_arc =
            patch_screen_edge_arc(&left, PatchEdge::C, &matrix, [1600.0, 900.0], options).unwrap();
        let right_arc =
            patch_screen_edge_arc(&right, PatchEdge::C, &matrix, [1600.0, 900.0], options).unwrap();
        assert!((left_arc.length_px - right_arc.length_px).abs() < 1.0e-7);
        for fraction in [0.1, 0.25, 0.5, 0.8] {
            let reversed_sum = left_arc.parameter_at_fraction(fraction)
                + right_arc.parameter_at_fraction(1.0 - fraction);
            assert!((reversed_sum - 1.0).abs() < 2.0e-5);
        }
    }

    #[test]
    fn edge_arc_rejects_camera_plane_crossing() {
        let patch = QBTriPatch::flat([0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [0.0, 1.0, -1.0]);
        let matrix = perspective(75.0_f64.to_radians(), 1.0, 0.01, 100.0);
        let error = patch_screen_edge_arc(
            &patch,
            PatchEdge::C,
            &matrix,
            [100.0; 2],
            ScreenArcOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(error, ScreenArcError::UnprojectableSample { .. }));
    }
}
