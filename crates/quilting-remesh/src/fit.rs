//! QB patch fitting: least-squares position fit + Gauss-Newton quaternion weight optimization.
//!
//! RESEARCH BASELINE, NOT ON THE PRODUCT PATH. Nothing outside the test-only
//! [`crate::remesh`] pipeline, the round-trip tests in [`crate::roundtrip`], and
//! `examples/experiments.rs` calls this. Optimizing w₁ and w₂ freely (8 DOF) is
//! under-constrained: the objective has a flat gauge direction and unconstrained
//! steps push the denominator `Σλᵢwᵢ` through zero, so the surface blows up. The
//! shipped fitter, [`crate::c_estimator`], searches a single quaternion instead.
//! Kept because the derivation below is the reference for any future rethink.
//!
//! Weight constraints from the literature (Krasauskas & Zubė):
//! - For the QB parametrization to produce points in R³ (not R⁴), the expression
//!   (Σ λᵢ pᵢ wᵢ) * (Σ λᵢ wᵢ)^-1 must be pure imaginary (Re = 0).
//! - For triangular patches with 3 control points, this is automatically satisfied
//!   when positions are pure imaginary quaternions (3D points) and we extract Im().
//! - The gauge freedom: multiplying all weights by a common quaternion q doesn't
//!   change the surface. We fix w0 = Quat::ONE to remove this redundancy.
//! - Circular arc boundaries arise naturally when adjacent patches share edge weights.

use quilting_core::quaternion::Quat;
use quilting_core::patch::QBTriPatch;
use crate::geometry;
use crate::linalg::solve_gauss;

#[derive(Debug, Clone)]
pub struct FitConfig {
    pub gauss_newton_iterations: usize,
    pub lambda_position: f64,
    pub lambda_normal: f64,
    pub convergence_tol: f64,
    /// If provided, use these as initial control points instead of least-squares fit.
    pub corner_positions: Option<[[f64; 3]; 3]>,
    /// Regularization weight pulling weights toward identity. Default 0.1.
    /// Set to 0.0 to disable regularization (for round-trip recovery tests).
    pub regularization: f64,
    /// If provided, use these as initial weights instead of identity.
    pub initial_weights: Option<[Quat; 3]>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            gauss_newton_iterations: 5,
            lambda_position: 1.0,
            lambda_normal: 0.05,
            convergence_tol: 1e-6,
            corner_positions: None,
            regularization: 0.1,
            initial_weights: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitResult {
    pub patch: QBTriPatch,
    pub rms_position_error: f64,
    pub max_position_error: f64,
    pub rms_normal_error_degrees: f64,
    pub max_normal_error_degrees: f64,
}

/// Fit a QB patch to sample data.
///
/// 1. Linear least-squares to find optimal control point positions (weights = identity).
/// 2. Gauss-Newton to optimize quaternion weights w1 and w2 (w0 pinned to identity).
pub fn fit_qb_patch(
    sample_positions: &[[f64; 3]],
    sample_normals: &[[f64; 3]],
    sample_bary: &[[f64; 3]],
    config: &FitConfig,
) -> FitResult {
    let n = sample_positions.len();
    assert_eq!(n, sample_bary.len());
    assert_eq!(n, sample_normals.len());

    // Phase 1: Control point positions — either from corners or least-squares
    let control_points = match config.corner_positions {
        Some(corners) => corners,
        None => fit_positions(sample_positions, sample_bary),
    };

    // Build initial patch
    let initial_weights = config.initial_weights.unwrap_or([Quat::ONE, Quat::ONE, Quat::ONE]);
    let mut patch = QBTriPatch::new(
        [
            Quat::from_point(control_points[0][0], control_points[0][1], control_points[0][2]),
            Quat::from_point(control_points[1][0], control_points[1][1], control_points[1][2]),
            Quat::from_point(control_points[2][0], control_points[2][1], control_points[2][2]),
        ],
        initial_weights,
    );

    // Phase 2: Gauss-Newton weight optimization
    if config.gauss_newton_iterations > 0 && n >= 4 {
        patch = optimize_weights(patch, sample_positions, sample_normals, sample_bary, config);
    }

    // Compute final error metrics
    let (rms_pos, max_pos, rms_norm, max_norm) =
        compute_errors(&patch, sample_positions, sample_normals, sample_bary);

    FitResult {
        patch,
        rms_position_error: rms_pos,
        max_position_error: max_pos,
        rms_normal_error_degrees: rms_norm,
        max_normal_error_degrees: max_norm,
    }
}

/// Fit control point positions via least-squares.
/// Minimizes Σ ||(1-u-v)*P0 + u*P1 + v*P2 - xi||²
fn fit_positions(samples: &[[f64; 3]], bary: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let n = samples.len();

    // Build 3x3 normal equations: A * [P0, P1, P2]^T = B
    // Basis functions: φ0 = bary[0] = 1-u-v, φ1 = bary[1] = u, φ2 = bary[2] = v
    let mut ata = [[0.0f64; 3]; 3]; // 3x3
    let mut atb = [[0.0f64; 3]; 3]; // 3x3 (one column per spatial dimension)

    for i in 0..n {
        let b = bary[i];
        for r in 0..3 {
            for c in 0..3 {
                ata[r][c] += b[r] * b[c];
            }
            for d in 0..3 {
                atb[r][d] += b[r] * samples[i][d];
            }
        }
    }

    // Solve 3x3 system for each spatial dimension
    let mut result = [[0.0; 3]; 3];
    let inv = invert_3x3(ata);
    for d in 0..3 {
        for r in 0..3 {
            result[r][d] = inv[r][0] * atb[0][d] + inv[r][1] * atb[1][d] + inv[r][2] * atb[2][d];
        }
    }

    result
}

/// Gauss-Newton optimization of quaternion weights.
/// We fix w0 = Quat::ONE and optimize w1, w2 (8 parameters).
fn optimize_weights(
    mut patch: QBTriPatch,
    samples: &[[f64; 3]],
    normals: &[[f64; 3]],
    bary: &[[f64; 3]],
    config: &FitConfig,
) -> QBTriPatch {
    let n = samples.len();
    // Parameters: [w1.w, w1.x, w1.y, w1.z, w2.w, w2.x, w2.y, w2.z]
    let num_params = 8;
    let num_residuals = n * 3; // position residuals only for stability
    // (normal residuals added as separate term)

    let eps = 1e-5; // finite difference step

    for _iter in 0..config.gauss_newton_iterations {
        // Compute residuals
        let residuals = compute_residuals(&patch, samples, bary, config.lambda_position);

        // Compute Jacobian by finite differences
        let mut jacobian = vec![vec![0.0; num_params]; num_residuals];

        for p in 0..num_params {
            let mut patch_plus = patch;
            perturb_weight(&mut patch_plus, p, eps);
            let res_plus = compute_residuals(&patch_plus, samples, bary, config.lambda_position);

            for r in 0..num_residuals {
                jacobian[r][p] = (res_plus[r] - residuals[r]) / eps;
            }
        }

        // Normal equations: (J^T J) * delta = -J^T * r
        let mut jtj = [[0.0f64; 8]; 8];
        let mut jtr = [0.0f64; 8];

        for r in 0..num_residuals {
            for i in 0..num_params {
                jtr[i] -= jacobian[r][i] * residuals[r];
                for j in 0..num_params {
                    jtj[i][j] += jacobian[r][i] * jacobian[r][j];
                }
            }
        }

        // Add normal error (compute Jacobian efficiently — O(params*n), not O(params^2*n))
        if config.lambda_normal > 0.0 {
            let norm_residuals = compute_normal_residuals(&patch, normals, bary, config.lambda_normal);
            let n_norm_res = norm_residuals.len();
            let mut norm_jac = vec![vec![0.0; num_params]; n_norm_res];
            for p in 0..num_params {
                let mut patch_plus = patch;
                perturb_weight(&mut patch_plus, p, eps);
                let nr_plus = compute_normal_residuals(&patch_plus, normals, bary, config.lambda_normal);
                for r in 0..n_norm_res {
                    norm_jac[r][p] = (nr_plus[r] - norm_residuals[r]) / eps;
                }
            }
            for r in 0..n_norm_res {
                for i in 0..num_params {
                    jtr[i] -= norm_jac[r][i] * norm_residuals[r];
                    for j in 0..num_params {
                        jtj[i][j] += norm_jac[r][i] * norm_jac[r][j];
                    }
                }
            }
        }

        // Regularize: penalize weight deviation from identity
        let reg_weight = config.regularization;
        let w1 = patch.weights[1];
        let w2 = patch.weights[2];
        let w_dev = [w1.w - 1.0, w1.x, w1.y, w1.z, w2.w - 1.0, w2.x, w2.y, w2.z];
        for i in 0..num_params {
            jtr[i] -= reg_weight * w_dev[i];
            jtj[i][i] += reg_weight;
        }

        // Levenberg-Marquardt damping for stability
        for i in 0..num_params {
            jtj[i][i] += 1e-6 * (jtj[i][i] + 1e-8);
        }

        // Solve the 8x8 normal equations
        let delta = solve_gauss(jtj, jtr);

        // Check convergence
        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if delta_norm < config.convergence_tol { break; }

        // Line search
        let old_err: f64 = residuals.iter().map(|r| r * r).sum();
        let mut alpha = 1.0;
        for _ in 0..8 {
            let mut trial = patch;
            apply_delta(&mut trial, &delta, alpha);
            let trial_res = compute_residuals(&trial, samples, bary, config.lambda_position);
            let new_err: f64 = trial_res.iter().map(|r| r * r).sum();
            if new_err < old_err {
                patch = trial;
                break;
            }
            alpha *= 0.5;
        }
    }

    patch
}

fn compute_residuals(patch: &QBTriPatch, samples: &[[f64; 3]], bary: &[[f64; 3]], weight: f64) -> Vec<f64> {
    let n = samples.len();
    let mut res = Vec::with_capacity(n * 3);
    let w_sqrt = weight.sqrt();
    for i in 0..n {
        let p = patch.eval(bary[i][1], bary[i][2]); // eval takes (u, v) where u=bary[1], v=bary[2]
        res.push(w_sqrt * (p.x - samples[i][0]));
        res.push(w_sqrt * (p.y - samples[i][1]));
        res.push(w_sqrt * (p.z - samples[i][2]));
    }
    res
}

fn compute_normal_residuals(patch: &QBTriPatch, normals: &[[f64; 3]], bary: &[[f64; 3]], weight: f64) -> Vec<f64> {
    let n = normals.len();
    let mut res = Vec::with_capacity(n * 3);
    let w_sqrt = weight.sqrt();
    for i in 0..n {
        let sp = safe_eval_with_normal(patch, bary[i]);
        res.push(w_sqrt * (sp.normal[0] - normals[i][0]));
        res.push(w_sqrt * (sp.normal[1] - normals[i][1]));
        res.push(w_sqrt * (sp.normal[2] - normals[i][2]));
    }
    res
}

/// Evaluate patch normal using finite differences that stay inside the triangle.
/// bary = [w, u, v] where w = 1 - u - v.
fn safe_eval_with_normal(patch: &QBTriPatch, bary: [f64; 3]) -> quilting_core::patch::SurfacePoint {
    let u = bary[1];
    let v = bary[2];
    let eps = 1e-4;

    // Use one-sided differences if near boundary
    let (u_plus, u_minus) = if u + eps + v <= 1.0 {
        (u + eps, u - eps.min(u))
    } else {
        (u, u - eps.min(u))
    };
    let (v_plus, v_minus) = if v + eps + u <= 1.0 {
        (v + eps, v - eps.min(v))
    } else {
        (v, v - eps.min(v))
    };

    let p = patch.eval(u, v);
    let du_len = u_plus - u_minus;
    let dv_len = v_plus - v_minus;

    if du_len < 1e-8 || dv_len < 1e-8 {
        // Degenerate — fall back to centroid normal
        return patch.eval_with_normal(0.333, 0.333);
    }

    let pu = patch.eval(u_plus, v);
    let pu_neg = patch.eval(u_minus, v);
    let pv = patch.eval(u, v_plus);
    let pv_neg = patch.eval(u, v_minus);

    let du = [(pu.x - pu_neg.x) / du_len, (pu.y - pu_neg.y) / du_len, (pu.z - pu_neg.z) / du_len];
    let dv = [(pv.x - pv_neg.x) / dv_len, (pv.y - pv_neg.y) / dv_len, (pv.z - pv_neg.z) / dv_len];

    let normal = geometry::vec3_normalize(geometry::vec3_cross(du, dv));

    quilting_core::patch::SurfacePoint {
        position: p.to_point(),
        normal,
    }
}

/// Perturb a weight parameter by epsilon.
/// Parameters 0-3 = w1 (w,x,y,z), 4-7 = w2 (w,x,y,z)
fn perturb_weight(patch: &mut QBTriPatch, param: usize, eps: f64) {
    let (weight_idx, component) = (param / 4 + 1, param % 4); // +1 because w0 is fixed
    match component {
        0 => patch.weights[weight_idx].w += eps,
        1 => patch.weights[weight_idx].x += eps,
        2 => patch.weights[weight_idx].y += eps,
        3 => patch.weights[weight_idx].z += eps,
        _ => unreachable!(),
    }
}

fn apply_delta(patch: &mut QBTriPatch, delta: &[f64; 8], alpha: f64) {
    // w1
    patch.weights[1].w += alpha * delta[0];
    patch.weights[1].x += alpha * delta[1];
    patch.weights[1].y += alpha * delta[2];
    patch.weights[1].z += alpha * delta[3];
    // w2
    patch.weights[2].w += alpha * delta[4];
    patch.weights[2].x += alpha * delta[5];
    patch.weights[2].y += alpha * delta[6];
    patch.weights[2].z += alpha * delta[7];
}

pub fn compute_errors(
    patch: &QBTriPatch,
    samples: &[[f64; 3]],
    normals: &[[f64; 3]],
    bary: &[[f64; 3]],
) -> (f64, f64, f64, f64) {
    let n = samples.len();
    let mut sum_pos_sq = 0.0;
    let mut max_pos = 0.0_f64;
    let mut sum_norm_sq = 0.0;
    let mut max_norm = 0.0_f64;

    for i in 0..n {
        // Position from direct eval
        let p = patch.eval(bary[i][1], bary[i][2]);
        let pos_err = geometry::vec3_dist(p.to_point(), samples[i]);
        sum_pos_sq += pos_err * pos_err;
        max_pos = max_pos.max(pos_err);

        // Normal from safe eval (handles boundary points)
        let sp = safe_eval_with_normal(patch, bary[i]);
        let cos_angle = geometry::vec3_dot(sp.normal, normals[i]).clamp(-1.0, 1.0);
        let angle_deg = cos_angle.acos().to_degrees();
        sum_norm_sq += angle_deg * angle_deg;
        max_norm = max_norm.max(angle_deg);
    }

    let rms_pos = (sum_pos_sq / n as f64).sqrt();
    let rms_norm = (sum_norm_sq / n as f64).sqrt();
    (rms_pos, max_pos, rms_norm, max_norm)
}

/// Invert a 3x3 matrix (Cramer's rule).
fn invert_3x3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        // Singular: return identity
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}
