//! Exact local screen-space metric for rational QB patches.
//!
//! A transformed patch followed by perspective projection is a smooth map
//! from two barycentric parameters to screen pixels away from a Möbius pole
//! and the camera plane. Its Jacobian is available analytically: QB tangents
//! come from the rational quotient rule and perspective contributes the exact
//! derivative of `clip.xy / clip.w`. The pullback metric `G = JᵀJ` is the
//! quantity a screen-uniform tessellator should bound.

use crate::patch::{PatchDifferential, QBTriPatch};

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
        (g_uu * delta[0] * delta[0]
            + 2.0 * g_uv * delta[0] * delta[1]
            + g_vv * delta[1] * delta[1])
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
            0.5 * viewport[0] * (delta[0] * clip[3] - clip[0] * delta[3])
                / denominator,
            0.5 * viewport[1] * (delta[1] * clip[3] - clip[1] * delta[3])
                / denominator,
        ];
        result.iter().all(|value| value.is_finite()).then_some(result)
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
    let area_scale = (tangent_u_px[0] * tangent_v_px[1]
        - tangent_u_px[1] * tangent_v_px[0])
        .abs();
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
        let patch = QBTriPatch::flat(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        let metric = patch_screen_metric(
            &patch,
            0.2,
            0.3,
            &[
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
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
        let source = QBTriPatch::flat(
            [-0.8, -0.5, -3.2],
            [0.9, -0.4, -3.0],
            [-0.3, 0.8, -3.4],
        );
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
        let patch = QBTriPatch::flat(
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        );
        let matrix = perspective(75.0_f64.to_radians(), 1.0, 0.01, 100.0);
        assert!(patch_screen_metric(&patch, 0.2, 0.2, &matrix, [100.0; 2]).is_none());
    }
}
