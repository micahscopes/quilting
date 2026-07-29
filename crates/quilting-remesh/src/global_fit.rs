//! Fitting curved QB patches onto a QEM-simplified mesh.
//!
//! [`per_patch_fit`] is the live product path: each simplified face gets its own
//! Möbius `c` parameter from [`crate::c_estimator`], independently of its
//! neighbours. Patches therefore meet at shared corner positions but have no
//! continuity of curvature across edges.
//!
//! [`global_fit`] is the research alternative — it optimizes one quaternion
//! weight per *vertex* so adjacent patches share weights and are C0 by
//! construction. It is reachable only from `examples/experiments.rs`; the
//! coupled Gauss-Newton never beat per-patch fitting for visual output, and the
//! shared-vertex coupling means one bad patch drags its neighbours with it.

use quilting_core::quaternion::Quat;
use quilting_core::patch::QBTriPatch;
use crate::c_estimator::CEstimatorConfig;
use crate::geometry;
use crate::linalg::solve_gauss;

/// Tuning for the blow-up guard.
///
/// A QB patch is a rational map: the surface is `(Σλᵢpᵢwᵢ)(Σλᵢwᵢ)⁻¹`, so
/// wherever the denominator approaches zero inside the parameter triangle the
/// patch runs off toward infinity. Rather than constrain the fit itself we probe
/// the fitted patch and fall back to a flat triangle when it misbehaves. These
/// numbers get retuned regularly, which is why they live here rather than inline.
#[derive(Debug, Clone)]
pub struct BlowUpGuard {
    /// Barycentric grid resolution used to probe the patch. Higher resolution
    /// catches more localized blow-ups, at linear cost.
    pub grid: usize,
    /// Reject if a probe lands further than this multiple of the control
    /// triangle's corner radius away from its centroid.
    pub radius_factor: f64,
    /// Reject if |Σλᵢwᵢ| drops below this — the patch is about to invert.
    pub min_denominator: f64,
}

impl Default for BlowUpGuard {
    fn default() -> Self {
        Self { grid: 8, radius_factor: 1.5, min_denominator: 0.1 }
    }
}

impl BlowUpGuard {
    /// True if `patch` misbehaves anywhere over the control triangle `cp`.
    pub fn rejects(&self, patch: &QBTriPatch, cp: &[[f64; 3]; 3]) -> bool {
        let n = self.grid.max(1);
        let cx = (cp[0][0] + cp[1][0] + cp[2][0]) / 3.0;
        let cy = (cp[0][1] + cp[1][1] + cp[2][1]) / 3.0;
        let cz = (cp[0][2] + cp[1][2] + cp[2][2]) / 3.0;
        let mut max_r = 0.0_f64;
        for c in cp {
            max_r = max_r.max(((c[0] - cx).powi(2) + (c[1] - cy).powi(2) + (c[2] - cz).powi(2)).sqrt());
        }
        let thresh = max_r * self.radius_factor;

        for i in 0..=n {
            for j in 0..=(n - i) {
                let u = i as f64 / n as f64;
                let v = j as f64 / n as f64;
                let w = 1.0 - u - v;

                let denom = w * patch.weights[0] + u * patch.weights[1] + v * patch.weights[2];
                if denom.norm() < self.min_denominator {
                    return true;
                }

                let p = patch.eval(u, v).to_point();
                if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                    return true;
                }
                let d = ((p[0] - cx).powi(2) + (p[1] - cy).powi(2) + (p[2] - cz).powi(2)).sqrt();
                if d > thresh {
                    return true;
                }
            }
        }
        false
    }
}

/// Everything tuned about the curved fitting path, defaulting to the values the
/// live `remesh_simplified_curved` pipeline ships with.
#[derive(Debug, Clone)]
pub struct CurvedFitConfig {
    /// Inner Gauss-Newton solve for the per-face Möbius `c`.
    pub c_estimator: CEstimatorConfig,
    /// Post-fit sanity check; a rejected patch falls back to flat.
    pub guard: BlowUpGuard,
    /// Minimum number of projected samples before a face is worth fitting. Below
    /// this the 4-DOF `c` solve is underdetermined and we keep the flat triangle.
    pub min_samples: usize,
    /// Barycentric slack when deciding whether an original-mesh vertex projects
    /// into a simplified triangle. Slack is needed because QEM collapse moves the
    /// simplified triangle off the original surface, so strict containment would
    /// starve boundary faces of samples.
    pub sample_margin: f64,
}

impl Default for CurvedFitConfig {
    fn default() -> Self {
        Self {
            c_estimator: CEstimatorConfig::default(),
            guard: BlowUpGuard::default(),
            min_samples: 4,
            sample_margin: 0.05,
        }
    }
}

/// Result of global fitting.
pub struct GlobalFitResult {
    /// Fitted QB patches (one per simplified face)
    pub patches: Vec<QBTriPatch>,
    /// Per-vertex weights
    pub vertex_weights: Vec<Quat>,
    /// RMS position error across all samples
    pub rms_error: f64,
    /// Max position error
    pub max_error: f64,
}

/// Fit curved QB patches with shared vertex weights using c-estimator initialization.
/// Runs per-face c-estimation, averages weights at shared vertices for continuity,
/// then optionally refines with global Gauss-Newton (though init-only often works best).
///
/// Research baseline only — the live path is [`per_patch_fit`].
///
/// `simp_pos` / `simp_faces` — the simplified (coarse) mesh
/// `orig_pos` / `orig_normals` — original mesh vertices and normals for fitting
pub fn global_fit(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    orig_pos: &[[f64; 3]],
    orig_normals: &[[f64; 3]],
    max_iterations: usize,
    config: &CurvedFitConfig,
) -> GlobalFitResult {
    let n_verts = simp_pos.len();
    let n_faces = simp_faces.len();

    // Collect samples per face: which original vertices project into each simplified triangle
    let face_samples: Vec<FaceSamples> = simp_faces.iter().map(|face| {
        let tri = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
        collect_face_samples(&tri, orig_pos, orig_normals, config.sample_margin)
    }).collect();

    // Initialize weights from per-face c-estimator, averaged at shared vertices.
    let mut weights = initialize_from_c_estimator(simp_pos, simp_faces, &face_samples, config);

    // Gauge fix: vertex 0 is always Quat::ONE — rescale all others relative to it
    if weights[0].norm() > 1e-10 {
        let w0_inv = weights[0].inv();
        for w in &mut weights {
            *w = *w * w0_inv;
        }
    }
    weights[0] = Quat::ONE;

    // Free parameters: vertices 1..n_verts, 4 params each
    let n_params = (n_verts - 1) * 4;

    if n_params == 0 {
        return build_result(simp_pos, simp_faces, &weights, &face_samples, &config.guard);
    }

    let eps = 1e-6;

    for _iter in 0..max_iterations {
        // Compute residuals for ALL faces
        let residuals = compute_all_residuals(simp_pos, simp_faces, &weights, &face_samples);
        let n_res = residuals.len();

        if n_res == 0 { break; }

        // Compute Jacobian: d(residuals)/d(weight params)
        // Each free vertex v (index 1..n) has 4 params at offset (v-1)*4
        let mut jtj = vec![vec![0.0f64; n_params]; n_params];
        let mut jtr = vec![0.0f64; n_params];

        // For efficiency, compute Jacobian one vertex at a time
        // (perturbing vertex v only affects faces containing v)
        for vi in 1..n_verts {
            let param_offset = (vi - 1) * 4;

            // Find which faces contain this vertex
            let affected_faces: Vec<usize> = (0..n_faces)
                .filter(|&fi| simp_faces[fi].contains(&vi))
                .collect();

            if affected_faces.is_empty() { continue; }

            for dim in 0..4 {
                let pi = param_offset + dim;

                // Perturb weight[vi]
                let mut w_plus = weights.clone();
                match dim {
                    0 => w_plus[vi].w += eps,
                    1 => w_plus[vi].x += eps,
                    2 => w_plus[vi].y += eps,
                    _ => w_plus[vi].z += eps,
                }

                let res_plus = compute_all_residuals(simp_pos, simp_faces, &w_plus, &face_samples);

                // Jacobian column pi
                for r in 0..n_res {
                    let j = (res_plus[r] - residuals[r]) / eps;
                    jtr[pi] -= j * residuals[r];
                    // Only fill the diagonal block for efficiency (approximate)
                    jtj[pi][pi] += j * j;
                }

                // Cross-terms between different dims of same vertex
                for dim2 in 0..dim {
                    let pj = param_offset + dim2;
                    let mut w_plus2 = weights.clone();
                    match dim2 {
                        0 => w_plus2[vi].w += eps,
                        1 => w_plus2[vi].x += eps,
                        2 => w_plus2[vi].y += eps,
                        _ => w_plus2[vi].z += eps,
                    }
                    let res_plus2 = compute_all_residuals(simp_pos, simp_faces, &w_plus2, &face_samples);
                    for r in 0..n_res {
                        let ji = (res_plus[r] - residuals[r]) / eps;
                        let jj = (res_plus2[r] - residuals[r]) / eps;
                        jtj[pi][pj] += ji * jj;
                        jtj[pj][pi] += ji * jj;
                    }
                }
            }
        }

        // Regularization: gently pull weights toward identity
        let reg = 0.001;
        for vi in 1..n_verts {
            let offset = (vi - 1) * 4;
            let w = weights[vi];
            let dev = [w.w - 1.0, w.x, w.y, w.z];
            for d in 0..4 {
                jtr[offset + d] -= reg * dev[d];
                jtj[offset + d][offset + d] += reg;
            }
        }

        // LM damping
        for i in 0..n_params {
            jtj[i][i] += 1e-4 * (jtj[i][i] + 1e-6);
        }

        // Solve the system using block-diagonal approximation
        // (each vertex is independent in the approximate Hessian)
        let mut delta = vec![0.0f64; n_params];
        for vi in 1..n_verts {
            let off = (vi - 1) * 4;
            let mut block = [[0.0f64; 4]; 4];
            let mut rhs = [0.0f64; 4];
            for i in 0..4 {
                for j in 0..4 { block[i][j] = jtj[off + i][off + j]; }
                rhs[i] = jtr[off + i];
            }
            let d = solve_gauss(block, rhs);
            for i in 0..4 { delta[off + i] = d[i]; }
        }

        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if delta_norm < 1e-10 { break; }

        // Apply with line search
        let old_err: f64 = residuals.iter().map(|r| r * r).sum();
        let mut alpha = 1.0;
        for _ in 0..8 {
            let mut trial = weights.clone();
            for vi in 1..n_verts {
                let off = (vi - 1) * 4;
                trial[vi].w += alpha * delta[off];
                trial[vi].x += alpha * delta[off + 1];
                trial[vi].y += alpha * delta[off + 2];
                trial[vi].z += alpha * delta[off + 3];
            }
            let trial_res = compute_all_residuals(simp_pos, simp_faces, &trial, &face_samples);
            let new_err: f64 = trial_res.iter().map(|r| r * r).sum();
            if new_err < old_err {
                weights = trial;
                break;
            }
            alpha *= 0.5;
        }
    }

    build_result(simp_pos, simp_faces, &weights, &face_samples, &config.guard)
}

/// Initialize per-vertex weights using per-face c-estimation, averaged at shared vertices.
fn initialize_from_c_estimator(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    face_samples: &[FaceSamples],
    config: &CurvedFitConfig,
) -> Vec<Quat> {
    let n_verts = simp_pos.len();

    // Strategy: run c-estimator per face, convert to per-vertex weights,
    // average at shared vertices, then clamp for stability.
    let mut c_sum = vec![Quat::ZERO; n_verts];
    let mut c_count = vec![0.0f64; n_verts];

    for (fi, face) in simp_faces.iter().enumerate() {
        let samples = &face_samples[fi];
        if samples.positions.len() < config.min_samples { continue; }

        let cp = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
        let c = crate::c_estimator::estimate_c_parameter(
            &cp, &samples.positions, &samples.bary, &samples.normals, &config.c_estimator,
        );

        // Accumulate c (not weights!) at each vertex for averaging
        for &vi in face {
            c_sum[vi] = c_sum[vi] + c;
            c_count[vi] += 1.0;
        }
    }

    // Average c per vertex, then compute weights w = c_avg * P + 1
    let mut weights = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        if c_count[i] > 0.0 {
            let c_avg = Quat::new(
                c_sum[i].w / c_count[i],
                c_sum[i].x / c_count[i],
                c_sum[i].y / c_count[i],
                c_sum[i].z / c_count[i],
            );
            let pi = Quat::from_point(simp_pos[i][0], simp_pos[i][1], simp_pos[i][2]);
            let wi = c_avg * pi + Quat::ONE;
            weights.push(wi);
        } else {
            weights.push(Quat::ONE);
        }
    }

    weights
}

pub(crate) struct FaceSamples {
    pub(crate) positions: Vec<[f64; 3]>,
    pub(crate) normals: Vec<[f64; 3]>,
    pub(crate) bary: Vec<[f64; 3]>,
}

/// Collect the original-mesh vertices that project into `tri`, together with
/// their barycentric coordinates and normals. `margin` is the barycentric slack
/// allowed outside the triangle (see [`CurvedFitConfig::sample_margin`]).
pub(crate) fn collect_face_samples(
    tri: &[[f64; 3]; 3],
    orig_pos: &[[f64; 3]],
    orig_normals: &[[f64; 3]],
    margin: f64,
) -> FaceSamples {
    let p0 = tri[0];
    let e1 = geometry::vec3_sub(tri[1], p0);
    let e2 = geometry::vec3_sub(tri[2], p0);
    let d11 = geometry::vec3_dot(e1, e1);
    let d12 = geometry::vec3_dot(e1, e2);
    let d22 = geometry::vec3_dot(e2, e2);
    let det = d11 * d22 - d12 * d12;

    let mut samples = FaceSamples {
        positions: Vec::new(), normals: Vec::new(), bary: Vec::new(),
    };

    if det.abs() < 1e-20 { return samples; }

    for (i, p) in orig_pos.iter().enumerate() {
        let d = geometry::vec3_sub(*p, p0);
        let b1 = geometry::vec3_dot(d, e1);
        let b2 = geometry::vec3_dot(d, e2);
        let u = (d22 * b1 - d12 * b2) / det;
        let v = (d11 * b2 - d12 * b1) / det;
        let w = 1.0 - u - v;

        let lo = -margin;
        let hi = 1.0 + margin;
        if u >= lo && v >= lo && w >= lo && u <= hi && v <= hi {
            samples.positions.push(*p);
            samples.normals.push(orig_normals[i]);
            samples.bary.push([w, u, v]);
        }
    }
    samples
}

fn compute_all_residuals(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    weights: &[Quat],
    face_samples: &[FaceSamples],
) -> Vec<f64> {
    let mut residuals = Vec::new();

    for (fi, face) in simp_faces.iter().enumerate() {
        let positions = [
            Quat::from_point(simp_pos[face[0]][0], simp_pos[face[0]][1], simp_pos[face[0]][2]),
            Quat::from_point(simp_pos[face[1]][0], simp_pos[face[1]][1], simp_pos[face[1]][2]),
            Quat::from_point(simp_pos[face[2]][0], simp_pos[face[2]][1], simp_pos[face[2]][2]),
        ];
        let w = [weights[face[0]], weights[face[1]], weights[face[2]]];
        let patch = QBTriPatch::new(positions, w);

        let samples = &face_samples[fi];
        for (k, bary) in samples.bary.iter().enumerate() {
            let eval = patch.eval(bary[1], bary[2]).to_point();
            residuals.push(samples.positions[k][0] - eval[0]);
            residuals.push(samples.positions[k][1] - eval[1]);
            residuals.push(samples.positions[k][2] - eval[2]);
        }
    }
    residuals
}

/// Per-patch independent fitting: each patch gets its own c parameter.
/// No vertex sharing, so no continuity guarantee, but each patch is individually stable.
/// Uses the c-estimator with a blow-up guard.
pub fn per_patch_fit(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    orig_pos: &[[f64; 3]],
    orig_normals: &[[f64; 3]],
    config: &CurvedFitConfig,
) -> GlobalFitResult {
    let face_samples: Vec<FaceSamples> = simp_faces.iter().map(|face| {
        let tri = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
        collect_face_samples(&tri, orig_pos, orig_normals, config.sample_margin)
    }).collect();

    let mut patches = Vec::with_capacity(simp_faces.len());

    for (fi, face) in simp_faces.iter().enumerate() {
        let cp = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
        let samples = &face_samples[fi];

        if samples.positions.len() >= config.min_samples {
            let mut fitted = crate::c_estimator::fit_patch_from_samples(
                cp, &samples.positions, &samples.bary, &samples.normals, &config.c_estimator,
            );

            // Normal consistency: check that the patch curves in the right direction
            // by comparing normal at centroid with average sample normal
            let centroid_sp = fitted.eval_with_normal(1.0/3.0, 1.0/3.0);
            let mut avg_normal = [0.0f64; 3];
            for n in &samples.normals {
                avg_normal[0] += n[0]; avg_normal[1] += n[1]; avg_normal[2] += n[2];
            }
            let nn = samples.normals.len() as f64;
            avg_normal[0] /= nn; avg_normal[1] /= nn; avg_normal[2] /= nn;
            let dot = centroid_sp.normal[0] * avg_normal[0]
                + centroid_sp.normal[1] * avg_normal[1]
                + centroid_sp.normal[2] * avg_normal[2];
            if dot < 0.0 {
                // Normal is flipped — swap two vertices to fix winding
                let mut fixed = fitted;
                fixed.positions.swap(1, 2);
                fixed.weights.swap(1, 2);
                fitted = fixed;
            }

            // Probe the fitted patch; a blow-up falls back to the flat triangle.
            if config.guard.rejects(&fitted, &cp) {
                patches.push(QBTriPatch::flat(cp[0], cp[1], cp[2]));
            } else {
                patches.push(fitted);
            }
        } else {
            patches.push(QBTriPatch::flat(cp[0], cp[1], cp[2]));
        }
    }

    // Compute error
    let mut sum_err = 0.0_f64;
    let mut max_err = 0.0_f64;
    let mut count = 0;
    for (fi, _) in simp_faces.iter().enumerate() {
        let patch = &patches[fi];
        for (k, bary) in face_samples[fi].bary.iter().enumerate() {
            let eval = patch.eval(bary[1], bary[2]).to_point();
            let dx = face_samples[fi].positions[k][0] - eval[0];
            let dy = face_samples[fi].positions[k][1] - eval[1];
            let dz = face_samples[fi].positions[k][2] - eval[2];
            let d = (dx*dx + dy*dy + dz*dz).sqrt();
            sum_err += d * d;
            max_err = max_err.max(d);
            count += 1;
        }
    }

    GlobalFitResult {
        patches,
        vertex_weights: vec![], // not meaningful for per-patch fit
        rms_error: if count > 0 { (sum_err / count as f64).sqrt() } else { 0.0 },
        max_error: max_err,
    }
}

/// Apply the same blow-up guard [`per_patch_fit`] uses to a globally-fitted patch
/// set, resetting offenders to flat.
///
/// Note the wart inherent to the shared-weight formulation: resetting one patch
/// also resets the weights at its corners, which silently changes every other
/// patch touching those vertices.
fn validate_and_fix_patches(
    patches: &mut [QBTriPatch],
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    weights: &mut [Quat],
    guard: &BlowUpGuard,
) {
    for (fi, face) in simp_faces.iter().enumerate() {
        let cp = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
        if guard.rejects(&patches[fi], &cp) {
            patches[fi] = QBTriPatch::flat(cp[0], cp[1], cp[2]);
            for &vi in face {
                weights[vi] = Quat::ONE;
            }
        }
    }
}

fn build_result(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    weights: &[Quat],
    face_samples: &[FaceSamples],
    guard: &BlowUpGuard,
) -> GlobalFitResult {
    let mut vertex_weights = weights.to_vec();
    let mut patches: Vec<QBTriPatch> = simp_faces.iter().map(|face| {
        QBTriPatch::new(
            [
                Quat::from_point(simp_pos[face[0]][0], simp_pos[face[0]][1], simp_pos[face[0]][2]),
                Quat::from_point(simp_pos[face[1]][0], simp_pos[face[1]][1], simp_pos[face[1]][2]),
                Quat::from_point(simp_pos[face[2]][0], simp_pos[face[2]][1], simp_pos[face[2]][2]),
            ],
            [weights[face[0]], weights[face[1]], weights[face[2]]],
        )
    }).collect();

    // Validate: reset blow-up patches to flat
    validate_and_fix_patches(&mut patches, simp_pos, simp_faces, &mut vertex_weights, guard);

    // Compute error stats
    let mut sum_err = 0.0_f64;
    let mut max_err = 0.0_f64;
    let mut count = 0;
    for (fi, _face) in simp_faces.iter().enumerate() {
        let patch = &patches[fi];
        for (k, bary) in face_samples[fi].bary.iter().enumerate() {
            let eval = patch.eval(bary[1], bary[2]).to_point();
            let dx = face_samples[fi].positions[k][0] - eval[0];
            let dy = face_samples[fi].positions[k][1] - eval[1];
            let dz = face_samples[fi].positions[k][2] - eval[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            sum_err += d * d;
            max_err = max_err.max(d);
            count += 1;
        }
    }

    GlobalFitResult {
        patches,
        vertex_weights,
        rms_error: if count > 0 { (sum_err / count as f64).sqrt() } else { 0.0 },
        max_error: max_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roundtrip::spherical_patch_via_inversion;

    fn control_triangle(patch: &QBTriPatch) -> [[f64; 3]; 3] {
        [
            patch.positions[0].to_point(),
            patch.positions[1].to_point(),
            patch.positions[2].to_point(),
        ]
    }

    /// A flat patch is the fallback the guard resets to, so it must never be
    /// rejected — otherwise a rejection would have nowhere safe to land.
    #[test]
    fn test_guard_accepts_flat_patch() {
        let cp = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let patch = QBTriPatch::flat(cp[0], cp[1], cp[2]);
        assert!(!BlowUpGuard::default().rejects(&patch, &cp));
    }

    /// A genuinely curved, well-behaved patch (spherical inversion of a flat
    /// triangle) must survive the guard, or curvature would never reach the
    /// renderer at all.
    #[test]
    fn test_guard_accepts_well_behaved_curved_patch() {
        let (patch, _) = spherical_patch_via_inversion(
            [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0],
        );
        let cp = control_triangle(&patch);
        assert!(!BlowUpGuard::default().rejects(&patch, &cp),
            "an inverted-but-tame patch should pass the guard");
    }

    /// The pathological case the guard exists for: weights whose barycentric
    /// combination `Σλᵢwᵢ` passes through zero, so the rational map has a pole
    /// strictly inside the parameter triangle.
    #[test]
    fn test_guard_rejects_vanishing_denominator() {
        let cp = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let positions = [
            Quat::from_point(cp[0][0], cp[0][1], cp[0][2]),
            Quat::from_point(cp[1][0], cp[1][1], cp[1][2]),
            Quat::from_point(cp[2][0], cp[2][1], cp[2][2]),
        ];
        // λ₀ = λ₁ = ½ gives Σλᵢwᵢ = ½(1) + ½(-1) = 0.
        let pole = QBTriPatch::new(positions, [Quat::ONE, -Quat::ONE, Quat::ONE]);
        assert!(BlowUpGuard::default().rejects(&pole, &cp),
            "a patch with an interior pole must be rejected");
    }

    /// Both thresholds must actually be consulted, not just carried around.
    #[test]
    fn test_guard_thresholds_are_wired_up() {
        let (patch, _) = spherical_patch_via_inversion(
            [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0],
        );
        let cp = control_triangle(&patch);

        let tight_radius = BlowUpGuard { radius_factor: 0.01, ..Default::default() };
        assert!(tight_radius.rejects(&patch, &cp), "radius_factor should be honoured");

        let tight_denom = BlowUpGuard { min_denominator: 1e6, ..Default::default() };
        assert!(tight_denom.rejects(&patch, &cp), "min_denominator should be honoured");
    }

    /// End-to-end on the live path: fitting a simplified sphere should produce
    /// curvature that survives the guard, and every emitted patch should itself
    /// pass the guard (nothing pathological escapes into the render).
    #[test]
    fn test_per_patch_fit_sphere_curves_without_blowing_up() {
        let (orig_pos, orig_faces) = crate::test_shapes::sphere(2);
        let orig_normals = geometry::compute_vertex_normals(&orig_pos, &orig_faces);
        let (simp_pos, simp_faces) = crate::simplify::simplify(&orig_pos, &orig_faces, 20);

        let config = CurvedFitConfig::default();
        let result = per_patch_fit(&simp_pos, &simp_faces, &orig_pos, &orig_normals, &config);

        let curved = result.patches.iter()
            .filter(|p| (p.weights[1] - Quat::ONE).norm() > 1e-6)
            .count();
        assert!(curved > 0,
            "the guard should not flatten every patch on a smooth sphere ({} of {} curved)",
            curved, result.patches.len());

        for (fi, face) in simp_faces.iter().enumerate() {
            let cp = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
            assert!(!config.guard.rejects(&result.patches[fi], &cp),
                "patch {} escaped the guard", fi);
        }
    }
}
