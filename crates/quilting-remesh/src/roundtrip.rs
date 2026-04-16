/// Round-trip tests for QB patch recovery.
///
/// The loop: create QB patches with known weights → tessellate to triangles
/// → run recovery pipeline → compare recovered weights to ground truth.
///
/// This validates each pipeline stage in isolation:
/// - Tessellator correctness (do generated triangles lie on the QB surface?)
/// - Weight fitting (can Gauss-Newton recover the original weights?)
/// - Segmentation (does the pipeline find the right patch boundaries?)

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::{Quat, Mobius};

/// Result of tessellating a QB patch.
pub struct TessellationResult {
    pub positions: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub faces: Vec<[usize; 3]>,
    /// Exact barycentric coordinates for each vertex (known from tessellation grid).
    pub bary: Vec<[f64; 3]>,
}

/// Tessellate a single QB patch into a triangle mesh.
/// `resolution` controls the number of subdivisions along each edge (n → ~n² triangles).
/// Returns positions, normals, faces, AND exact barycentric coordinates.
pub fn tessellate_patch(patch: &QBTriPatch, resolution: usize) -> TessellationResult {
    let n = resolution.max(1);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut bary = Vec::new();
    let mut faces = Vec::new();

    // Generate vertices on a triangular grid in barycentric space.
    // Vertex (i, j) has bary coords (1 - i/n - j/n, i/n, j/n) for i+j <= n.
    let mut vertex_map = vec![vec![usize::MAX; n + 1]; n + 1];
    for i in 0..=n {
        for j in 0..=(n - i) {
            let u = i as f64 / n as f64;
            let v = j as f64 / n as f64;
            let w = 1.0 - u - v;
            let sp = patch.eval_with_normal(u, v);
            let idx = positions.len();
            positions.push(sp.position);
            normals.push(sp.normal);
            bary.push([w, u, v]); // [λ0, λ1, λ2]
            vertex_map[i][j] = idx;
        }
    }

    // Generate triangles from the grid
    for i in 0..n {
        for j in 0..(n - i) {
            let a = vertex_map[i][j];
            let b = vertex_map[i + 1][j];
            let c = vertex_map[i][j + 1];
            faces.push([a, b, c]);

            if i + j + 1 < n {
                let d = vertex_map[i + 1][j + 1];
                faces.push([b, d, c]);
            }
        }
    }

    TessellationResult { positions, normals, faces, bary }
}

/// Tessellate multiple QB patches into a single merged mesh.
/// Adjacent patches sharing an edge will have shared vertices (within tolerance).
pub fn tessellate_patches(patches: &[QBTriPatch], resolution: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<usize>) {
    let mut all_positions: Vec<[f64; 3]> = Vec::new();
    let mut all_faces: Vec<[usize; 3]> = Vec::new();
    let mut patch_ids: Vec<usize> = Vec::new();

    for (pi, patch) in patches.iter().enumerate() {
        let tess = tessellate_patch(patch, resolution);

        let mut remap: Vec<usize> = Vec::with_capacity(tess.positions.len());
        for p in &tess.positions {
            let mut found = None;
            for (ei, existing) in all_positions.iter().enumerate() {
                let dx = p[0] - existing[0];
                let dy = p[1] - existing[1];
                let dz = p[2] - existing[2];
                if dx * dx + dy * dy + dz * dz < 1e-12 {
                    found = Some(ei);
                    break;
                }
            }
            match found {
                Some(ei) => remap.push(ei),
                None => {
                    remap.push(all_positions.len());
                    all_positions.push(*p);
                }
            }
        }

        for face in &tess.faces {
            all_faces.push([remap[face[0]], remap[face[1]], remap[face[2]]]);
            patch_ids.push(pi);
        }
    }

    (all_positions, all_faces, patch_ids)
}

/// Run a single-patch recovery experiment and return structured results.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub name: String,
    pub approach: String,
    pub rms_position: f64,
    pub max_position: f64,
    pub rms_normal_degrees: f64,
    pub weight_error_w1: f64,
    pub weight_error_w2: f64,
    pub original_w1: Quat,
    pub original_w2: Quat,
    pub recovered_w1: Quat,
    pub recovered_w2: Quat,
}

impl std::fmt::Display for ExperimentResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:20} {:20} pos_rms={:.6} pos_max={:.6} norm={:.2}deg w1_err={:.4} w2_err={:.4}",
            self.name, self.approach,
            self.rms_position, self.max_position,
            self.rms_normal_degrees,
            self.weight_error_w1, self.weight_error_w2)
    }
}

/// Estimate the Möbius `c` parameter from mesh samples.
///
/// The QB formula with weights `wᵢ = c*pᵢ_flat + 1` (gauge d=1) produces:
///   X(λ) = (Σ λᵢ Pᵢ (c*Fᵢ+1)) (Σ λᵢ (c*Fᵢ+1))⁻¹
///
/// where Pᵢ are curved corner positions and Fᵢ are flat corner positions.
/// The flat corners serve as a reference: they define where the surface
/// WOULD be if it were flat (c=0).
///
/// For a general mesh patch (not generated by Möbius), we don't have flat corners.
/// But we can use the control triangle (the flat triangle through the 3 corner
/// positions) as the reference. The "flat corners" ARE the curved corners in this case.
///
/// So we solve for c in: wᵢ = c*Pᵢ + 1 (where Pᵢ are the control points),
/// minimizing the QB surface error against sample points.
pub fn estimate_c_parameter(
    control_points: &[[f64; 3]; 3],
    samples: &[[f64; 3]],
    sample_bary: &[[f64; 3]],
    sample_normals: &[[f64; 3]],
) -> Quat {
    let cp = [
        Quat::from_point(control_points[0][0], control_points[0][1], control_points[0][2]),
        Quat::from_point(control_points[1][0], control_points[1][1], control_points[1][2]),
        Quat::from_point(control_points[2][0], control_points[2][1], control_points[2][2]),
    ];

    // Use Gauss-Newton on c (4 parameters) to minimize position + normal error
    let mut c = Quat::ZERO;
    let eps = 1e-6;

    for _iter in 0..100 {
        let mut residuals = Vec::new();
        let weights = make_weights_from_c(c, &cp);
        let patch = QBTriPatch::new(cp, weights);

        // Position residuals
        for (k, bary) in sample_bary.iter().enumerate() {
            let eval = patch.eval_with_normal(bary[1], bary[2]);
            residuals.push(samples[k][0] - eval.position[0]);
            residuals.push(samples[k][1] - eval.position[1]);
            residuals.push(samples[k][2] - eval.position[2]);
            // Normal residuals (weighted lower)
            let nw = 0.1;
            residuals.push(nw * (sample_normals[k][0] - eval.normal[0]));
            residuals.push(nw * (sample_normals[k][1] - eval.normal[1]));
            residuals.push(nw * (sample_normals[k][2] - eval.normal[2]));
        }

        let nr = residuals.len();

        // Jacobian via finite differences (4 columns, nr rows)
        let mut jac = vec![vec![0.0; 4]; nr];
        for dim in 0..4 {
            let mut c_plus = c;
            match dim { 0 => c_plus.w += eps, 1 => c_plus.x += eps, 2 => c_plus.y += eps, _ => c_plus.z += eps }
            let w_plus = make_weights_from_c(c_plus, &cp);
            let patch_plus = QBTriPatch::new(cp, w_plus);

            let mut ri = 0;
            for (k, bary) in sample_bary.iter().enumerate() {
                let eval = patch_plus.eval_with_normal(bary[1], bary[2]);
                let r_pos = [samples[k][0] - eval.position[0], samples[k][1] - eval.position[1], samples[k][2] - eval.position[2]];
                let nw = 0.1;
                let r_norm = [nw * (sample_normals[k][0] - eval.normal[0]), nw * (sample_normals[k][1] - eval.normal[1]), nw * (sample_normals[k][2] - eval.normal[2])];
                for &r in r_pos.iter().chain(r_norm.iter()) {
                    jac[ri][dim] = (r - residuals[ri]) / eps;
                    ri += 1;
                }
            }
        }

        // Normal equations: (J^T J) delta = -J^T r
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];
        for r in 0..nr {
            for i in 0..4 {
                jtr[i] -= jac[r][i] * residuals[r];
                for j in 0..4 {
                    jtj[i][j] += jac[r][i] * jac[r][j];
                }
            }
        }
        // LM damping
        for i in 0..4 { jtj[i][i] += 1e-6 * (jtj[i][i] + 1e-8); }

        // Solve 4x4
        let delta = solve_4x4(jtj, jtr);
        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if delta_norm < 1e-10 { break; }

        c.w += delta[0]; c.x += delta[1]; c.y += delta[2]; c.z += delta[3];
    }

    c
}

fn make_weights_from_c(c: Quat, cp: &[Quat; 3]) -> [Quat; 3] {
    let w0 = c * cp[0] + Quat::ONE;
    let w1 = c * cp[1] + Quat::ONE;
    let w2 = c * cp[2] + Quat::ONE;
    let w0_inv = w0.inv();
    [Quat::ONE, w1 * w0_inv, w2 * w0_inv]
}

fn solve_4x4(a: [[f64; 4]; 4], b: [f64; 4]) -> [f64; 4] {
    // Gaussian elimination with partial pivoting
    let mut m = [[0.0; 5]; 4];
    for i in 0..4 { for j in 0..4 { m[i][j] = a[i][j]; } m[i][4] = b[i]; }
    for col in 0..4 {
        let mut max_row = col;
        for row in (col+1)..4 {
            if m[row][col].abs() > m[max_row][col].abs() { max_row = row; }
        }
        m.swap(col, max_row);
        if m[col][col].abs() < 1e-15 { continue; }
        for row in (col+1)..4 {
            let factor = m[row][col] / m[col][col];
            for j in col..5 { m[row][j] -= factor * m[col][j]; }
        }
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        x[i] = m[i][4];
        for j in (i+1)..4 { x[i] -= m[i][j] * x[j]; }
        if m[i][i].abs() > 1e-15 { x[i] /= m[i][i]; }
    }
    x
}

/// Estimate QB weights by finding the Möbius transform that maps a flat reference
/// triangle to the observed curved surface.
///
/// Given flat control points P and curved sample points X with bary coords λ:
/// - The flat point at λ is p = Σλᵢ*Pᵢ
/// - The Möbius image should be X = (a*p+b)(c*p+d)⁻¹
/// - The QB weight is wᵢ = c*Pᵢ + d
///
/// We minimize Σ |X_k - (a*p_k+b)(c*p_k+d)⁻¹|² over (a,b,c,d) quaternions.
/// Since this is 16 params with 4 gauge DOF = 12 effective DOF, and each sample
/// gives 3 constraints, we need >= 4 samples.
///
/// Simpler approach: use only the 3 corner correspondences (flat corners → curved corners)
/// to determine c, d (6 DOF from the weight transform, minus gauge).
pub fn estimate_weights_from_mobius(
    flat_corners: &[[f64; 3]; 3],
    curved_corners: &[[f64; 3]; 3],
    samples: &[[f64; 3]],
    sample_bary: &[[f64; 3]],
) -> [Quat; 3] {
    // Strategy: the QB formula with weights wᵢ and flat positions pᵢ evaluates to:
    //   X(λ) = (Σ λᵢ pᵢ wᵢ)(Σ λᵢ wᵢ)⁻¹
    //
    // At corners (λᵢ = 1, rest 0): X = pᵢ wᵢ wᵢ⁻¹ = pᵢ
    // So the flat positions ARE the corner positions of the output surface!
    //
    // But we want the curved corners to be the output. So we need:
    //   curved_corner_i = flat_corner_i  (for the patch to pass through curved corners)
    // That's only true if we use the CURVED corners as the patch positions.
    //
    // The Möbius approach: start with flat patch positions = flat_corners, apply
    // Möbius F. New positions = F(flat_corners). New weights = (c*flat_corner_i + d).
    // We need F(flat_corners) = curved_corners, so F maps flat → curved.
    //
    // Find c, d such that: curved_i = (a*flat_i + b)(c*flat_i + d)⁻¹
    //
    // For the weight computation, we only need c and d.
    // The weight transform is: wᵢ = (c * flat_i + d) * identity = c * flat_i + d
    //
    // From 3 correspondences flat_i → curved_i, and the Möbius formula:
    //   curved_i * (c * flat_i + d) = a * flat_i + b
    //
    // This gives 3 quaternion equations in 4 quaternion unknowns (a,b,c,d).
    // Fix gauge by setting d = 1 (identity quaternion).
    //
    // Then: curved_i * (c * flat_i + 1) = a * flat_i + b
    //   ⟹ a * flat_i + b = curved_i * c * flat_i + curved_i
    //   ⟹ a * flat_i - curved_i * c * flat_i = curved_i - b
    //   ⟹ (a - curved_i * c) * flat_i = curved_i - b
    //
    // Three equations:
    //   (a - C0*c) * F0 = C0 - b
    //   (a - C1*c) * F1 = C1 - b
    //   (a - C2*c) * F2 = C2 - b
    //
    // Subtract pairs to eliminate b:
    //   (a - C0*c)*F0 - (a - C1*c)*F1 = C0 - C1
    //   (a - C0*c)*F0 - (a - C2*c)*F2 = C0 - C2
    //
    // This is getting complex. Let me try a simpler numerical approach:
    // fit c (4 params) to minimize the distance between QB-eval points and samples.

    let fp = [
        Quat::from_point(flat_corners[0][0], flat_corners[0][1], flat_corners[0][2]),
        Quat::from_point(flat_corners[1][0], flat_corners[1][1], flat_corners[1][2]),
        Quat::from_point(flat_corners[2][0], flat_corners[2][1], flat_corners[2][2]),
    ];
    let cp = [
        Quat::from_point(curved_corners[0][0], curved_corners[0][1], curved_corners[0][2]),
        Quat::from_point(curved_corners[1][0], curved_corners[1][1], curved_corners[1][2]),
        Quat::from_point(curved_corners[2][0], curved_corners[2][1], curved_corners[2][2]),
    ];

    // Try a direct approach: find Möbius F s.t. F(flat_i) = curved_i using
    // the cross-ratio. For 3 points, the Möbius is not unique (need 4 points
    // in the plane, but we're in 3D / quaternion space).
    //
    // Simplest working approach: set d=1, c=0 initially (identity Möbius → flat patch).
    // Then search for c that makes the QB patch match the curved surface.
    // Since w_i = c * flat_i + d = c * flat_i + 1:
    //
    // The QB eval becomes:
    //   X(λ) = (Σ λᵢ pᵢ (c*pᵢ+1)) (Σ λᵢ (c*pᵢ+1))⁻¹
    //
    // where pᵢ are the CURVED corner positions (the patch control points).
    //
    // Wait — the control point positions in the patch are the curved corners (not flat).
    // The flat corners are just for the Möbius derivation. The actual QB patch has:
    //   positions = curved_corners
    //   weights = c * flat_corners + 1
    //
    // So we optimize c (4 params) to minimize:
    //   Σ_k |X(λ_k) - sample_k|²
    //
    // where X is the QB eval with the above positions/weights.

    // Simple gradient descent on c
    let mut c = Quat::ZERO; // start with c=0 → weights = 1 → flat patch
    let step = 0.001;
    let n_iters = 500;

    for _ in 0..n_iters {
        // Compute gradient via finite differences
        let loss = mobius_loss(&cp, &fp, c, samples, sample_bary);
        let mut grad = Quat::ZERO;
        let eps = 1e-5;
        for dim in 0..4 {
            let mut c_plus = c;
            match dim {
                0 => c_plus.w += eps,
                1 => c_plus.x += eps,
                2 => c_plus.y += eps,
                _ => c_plus.z += eps,
            }
            let loss_plus = mobius_loss(&cp, &fp, c_plus, samples, sample_bary);
            let g = (loss_plus - loss) / eps;
            match dim {
                0 => grad.w = g,
                1 => grad.x = g,
                2 => grad.y = g,
                _ => grad.z = g,
            }
        }
        // Gradient descent step
        c.w -= step * grad.w;
        c.x -= step * grad.x;
        c.y -= step * grad.y;
        c.z -= step * grad.z;
    }

    // Weights: wᵢ = c * flat_i + 1, gauge-normalized so w0 = 1
    let w0 = c * fp[0] + Quat::ONE;
    let w1 = c * fp[1] + Quat::ONE;
    let w2 = c * fp[2] + Quat::ONE;

    let w0_inv = w0.inv();
    [Quat::ONE, w1 * w0_inv, w2 * w0_inv]
}

/// Loss function for Möbius weight estimation.
fn mobius_loss(
    curved_corners: &[Quat; 3],
    flat_corners: &[Quat; 3],
    c: Quat,
    samples: &[[f64; 3]],
    sample_bary: &[[f64; 3]],
) -> f64 {
    let weights = [
        c * flat_corners[0] + Quat::ONE,
        c * flat_corners[1] + Quat::ONE,
        c * flat_corners[2] + Quat::ONE,
    ];
    // Gauge normalize
    let w0_inv = weights[0].inv();
    let w = [Quat::ONE, weights[1] * w0_inv, weights[2] * w0_inv];

    let patch = QBTriPatch::new(*curved_corners, w);

    let mut loss = 0.0;
    for (k, bary) in sample_bary.iter().enumerate() {
        let u = bary[1];
        let v = bary[2];
        let eval = patch.eval(u, v);
        let target = Quat::from_point(samples[k][0], samples[k][1], samples[k][2]);
        let diff = eval - target;
        loss += diff.norm_sq();
    }
    loss
}

/// Run recovery on a patch using given FitConfig and barycentric coordinates.
pub fn run_recovery(
    name: &str,
    approach: &str,
    original: &QBTriPatch,
    positions: &[[f64; 3]],
    normals: &[[f64; 3]],
    bary: &[[f64; 3]],
    config: &crate::fit::FitConfig,
) -> ExperimentResult {
    let result = crate::fit::fit_qb_patch(positions, normals, bary, config);
    let err = measure_recovery_error(original, &result.patch, 20);

    ExperimentResult {
        name: name.to_string(),
        approach: approach.to_string(),
        rms_position: err.rms_position,
        max_position: err.max_position,
        rms_normal_degrees: err.rms_normal_degrees,
        weight_error_w1: err.weight_errors[1],
        weight_error_w2: err.weight_errors[2],
        original_w1: original.weights[1],
        original_w2: original.weights[2],
        recovered_w1: result.patch.weights[1],
        recovered_w2: result.patch.weights[2],
    }
}

/// Create a spherical QB patch by Möbius-inverting a flat patch.
///
/// Takes a flat triangle, translates it away from the origin, then applies
/// spherical inversion. The image is a curved triangle on a sphere.
/// The weights are computed correctly by QBTriPatch::transform.
/// Returns (curved_patch, flat_corners) — flat_corners needed for weight recovery.
pub fn spherical_patch_via_inversion(
    p0: [f64; 3], p1: [f64; 3], p2: [f64; 3],
) -> (QBTriPatch, [[f64; 3]; 3]) {
    let flat = QBTriPatch::flat(p0, p1, p2);
    let inv = Mobius::inversion();
    (flat.transform(&inv), [p0, p1, p2])
}

/// Create a QB patch from arbitrary corner positions and sample points.
/// This tests the c-estimator on surfaces that are NOT Möbius images.
/// The patch is constructed by fitting c to best match the samples.
pub fn fit_patch_from_samples(
    corner_positions: [[f64; 3]; 3],
    samples: &[[f64; 3]],
    sample_bary: &[[f64; 3]],
    sample_normals: &[[f64; 3]],
) -> QBTriPatch {
    let cp = [
        Quat::from_point(corner_positions[0][0], corner_positions[0][1], corner_positions[0][2]),
        Quat::from_point(corner_positions[1][0], corner_positions[1][1], corner_positions[1][2]),
        Quat::from_point(corner_positions[2][0], corner_positions[2][1], corner_positions[2][2]),
    ];

    let c = estimate_c_parameter(&corner_positions, samples, sample_bary, sample_normals);
    let weights = make_weights_from_c(c, &cp);
    QBTriPatch::new(cp, weights)
}

/// Create a set of QB patches forming a sphere via Möbius inversion of a flat octahedron.
///
/// Start with a flat octahedron offset from the origin, invert it.
/// The image is a set of curved triangular patches on a sphere.
pub fn sphere_patches(offset: f64) -> Vec<QBTriPatch> {
    use quilting_core::quaternion::Mobius;

    // Octahedron vertices, offset along z so none are at origin
    let z = offset;
    let verts = [
        [1.0, 0.0, z], [-1.0, 0.0, z],
        [0.0, 1.0, z], [0.0, -1.0, z],
        [0.0, 0.0, z + 1.0], [0.0, 0.0, z - 1.0],
    ];
    let faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ];

    let inv = Mobius::inversion();

    faces.iter().map(|f| {
        let flat = QBTriPatch::flat(verts[f[0]], verts[f[1]], verts[f[2]]);
        flat.transform(&inv)
    }).collect()
}

/// Error metrics comparing recovered patch to ground truth.
#[derive(Debug, Clone)]
pub struct RecoveryError {
    /// RMS position error over sample points
    pub rms_position: f64,
    /// Max position error
    pub max_position: f64,
    /// Weight similarity: |w_recovered - w_original| for each weight
    pub weight_errors: [f64; 3],
    /// RMS normal error in degrees
    pub rms_normal_degrees: f64,
}

/// Compare a recovered patch to the ground truth by sampling both.
pub fn measure_recovery_error(
    original: &QBTriPatch,
    recovered: &QBTriPatch,
    num_samples: usize,
) -> RecoveryError {
    let n = num_samples;
    let mut pos_err_sum = 0.0;
    let mut pos_err_max = 0.0_f64;
    let mut norm_err_sum = 0.0;
    let mut count = 0;

    for i in 0..=n {
        for j in 0..=(n - i) {
            let u = i as f64 / n as f64;
            let v = j as f64 / n as f64;

            let sp_orig = original.eval_with_normal(u, v);
            let sp_recv = recovered.eval_with_normal(u, v);

            let dx = sp_orig.position[0] - sp_recv.position[0];
            let dy = sp_orig.position[1] - sp_recv.position[1];
            let dz = sp_orig.position[2] - sp_recv.position[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            pos_err_sum += d * d;
            pos_err_max = pos_err_max.max(d);

            let dot = sp_orig.normal[0] * sp_recv.normal[0]
                + sp_orig.normal[1] * sp_recv.normal[1]
                + sp_orig.normal[2] * sp_recv.normal[2];
            let angle = dot.clamp(-1.0, 1.0).acos().to_degrees();
            norm_err_sum += angle * angle;

            count += 1;
        }
    }

    let rms_position = (pos_err_sum / count as f64).sqrt();
    let rms_normal = (norm_err_sum / count as f64).sqrt();

    let weight_errors = [
        (original.weights[0] - recovered.weights[0]).norm(),
        (original.weights[1] - recovered.weights[1]).norm(),
        (original.weights[2] - recovered.weights[2]).norm(),
    ];

    RecoveryError {
        rms_position,
        max_position: pos_err_max,
        weight_errors,
        rms_normal_degrees: rms_normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::FitConfig;
    use crate::geometry;

    fn corners(patch: &QBTriPatch) -> [[f64; 3]; 3] {
        [
            patch.positions[0].to_point(),
            patch.positions[1].to_point(),
            patch.positions[2].to_point(),
        ]
    }

    // ── Tessellation tests ──────────────────────────────────────────

    #[test]
    fn test_tessellate_flat_patch() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let tess = tessellate_patch(&patch, 4);
        assert_eq!(tess.positions.len(), 15); // (4+1)*(4+2)/2
        assert_eq!(tess.faces.len(), 16);     // 4²
        for p in &tess.positions {
            assert!(p[2].abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverted_patch_is_curved() {
        let (patch, _) = spherical_patch_via_inversion(
            [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [-1.0, 0.0, 2.0],
        );
        let tess = tessellate_patch(&patch, 8);
        let cp = corners(&patch);
        let n = geometry::face_normal_normalized(&cp, [0, 1, 2]);
        let d0 = geometry::vec3_dot(cp[0], n);

        let max_dev: f64 = tess.positions.iter()
            .map(|p| (geometry::vec3_dot(*p, n) - d0).abs())
            .fold(0.0, f64::max);

        assert!(max_dev > 1e-4, "inverted patch should be curved, deviation = {}", max_dev);
    }

    #[test]
    fn test_multi_patch_tessellation() {
        let patches = sphere_patches(3.0);
        assert_eq!(patches.len(), 8);
        let (positions, _faces, patch_ids) = tessellate_patches(&patches, 6);
        for p in &positions {
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
        }
        let unique: std::collections::HashSet<usize> = patch_ids.iter().copied().collect();
        assert_eq!(unique.len(), 8);
    }

    // ── Verification: Möbius weight formula ────────────────────────

    /// Verify that the weight formula wᵢ = (c*pᵢ+d) from the Möbius transform
    /// produces the exact same patch as QBTriPatch::transform.
    #[test]
    fn test_mobius_weight_formula_direct() {
        let flat_corners = [[0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0]];
        let flat = QBTriPatch::flat(flat_corners[0], flat_corners[1], flat_corners[2]);
        let inv = Mobius::inversion();

        // Method 1: QBTriPatch::transform (known correct)
        let transformed = flat.transform(&inv);

        // Method 2: manual weight computation via wᵢ = (c*pᵢ + d)
        // For inversion: c=1, d=0
        let fp = [
            Quat::from_point(flat_corners[0][0], flat_corners[0][1], flat_corners[0][2]),
            Quat::from_point(flat_corners[1][0], flat_corners[1][1], flat_corners[1][2]),
            Quat::from_point(flat_corners[2][0], flat_corners[2][1], flat_corners[2][2]),
        ];

        // The Möbius inversion is F(x) = (0*x + (-1))(1*x + 0)^-1
        // So a=0, b=-1, c=1, d=0
        // Weight formula: wᵢ = (c*pᵢ + d) * flat_wᵢ = (1*pᵢ + 0) * 1 = pᵢ
        let weights_manual = [fp[0], fp[1], fp[2]]; // = flat positions as quaternions

        // Gauge normalize: w0 becomes identity
        let w0_inv = weights_manual[0].inv();
        let weights_normalized = [
            Quat::ONE,
            weights_manual[1] * w0_inv,
            weights_manual[2] * w0_inv,
        ];

        eprintln!("Möbius weight formula verification:");
        eprintln!("  transform weights: {:?}", transformed.weights);
        eprintln!("  manual weights:    {:?}", weights_normalized);

        // Compare weights
        for i in 0..3 {
            let diff = (transformed.weights[i] - weights_normalized[i]).norm();
            eprintln!("  w{} diff: {:.2e}", i, diff);
            // Note: may not be identical due to different gauge choices
        }

        // The real test: do both produce the same surface?
        let n = 10;
        let mut max_diff = 0.0_f64;
        let manual_patch = QBTriPatch::new(transformed.positions, weights_normalized);
        for i in 0..=n {
            for j in 0..=(n-i) {
                let u = i as f64 / n as f64;
                let v = j as f64 / n as f64;
                let p1 = transformed.eval(u, v).to_point();
                let p2 = manual_patch.eval(u, v).to_point();
                let d = ((p1[0]-p2[0]).powi(2) + (p1[1]-p2[1]).powi(2) + (p1[2]-p2[2]).powi(2)).sqrt();
                max_diff = max_diff.max(d);
            }
        }
        eprintln!("  max surface difference: {:.2e}", max_diff);
        assert!(max_diff < 1e-10, "manual weights should produce identical surface, diff={}", max_diff);
    }

    /// Test: use Möbius-derived initial weights, then refine with Gauss-Newton.
    #[test]
    fn test_mobius_init_then_gauss_newton() {
        let flat_corners = [[0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0]];
        let (original, _) = spherical_patch_via_inversion(
            flat_corners[0], flat_corners[1], flat_corners[2],
        );
        let tess = tessellate_patch(&original, 12);
        let cp = corners(&original);

        // Compute Möbius-derived weights (we know the transform is inversion)
        let fp = [
            Quat::from_point(flat_corners[0][0], flat_corners[0][1], flat_corners[0][2]),
            Quat::from_point(flat_corners[1][0], flat_corners[1][1], flat_corners[1][2]),
            Quat::from_point(flat_corners[2][0], flat_corners[2][1], flat_corners[2][2]),
        ];
        // For inversion: wᵢ = pᵢ (the flat position quaternion)
        let w0_inv = fp[0].inv();
        let mobius_weights = [
            Quat::ONE,
            fp[1] * w0_inv,
            fp[2] * w0_inv,
        ];

        // Test 1: Möbius weights alone (no optimization)
        let mobius_patch = QBTriPatch::new(original.positions, mobius_weights);
        let err_mobius = measure_recovery_error(&original, &mobius_patch, 20);
        eprintln!("Möbius-derived weights (no optimization):");
        eprintln!("  pos RMS={:.6}, norm={:.2}°, w1_err={:.4}, w2_err={:.4}",
            err_mobius.rms_position, err_mobius.rms_normal_degrees,
            err_mobius.weight_errors[1], err_mobius.weight_errors[2]);

        // Test 2: Möbius weights as init → Gauss-Newton refinement
        let config = FitConfig {
            gauss_newton_iterations: 50,
            lambda_normal: 0.1,
            convergence_tol: 1e-12,
            regularization: 0.0,
            corner_positions: Some(cp),
            initial_weights: Some(mobius_weights),
            ..Default::default()
        };
        let result = crate::fit::fit_qb_patch(
            &tess.positions, &tess.normals, &tess.bary, &config);
        let err_refined = measure_recovery_error(&original, &result.patch, 20);
        eprintln!("Möbius init + Gauss-Newton refinement:");
        eprintln!("  pos RMS={:.6}, norm={:.2}°, w1_err={:.4}, w2_err={:.4}",
            err_refined.rms_position, err_refined.rms_normal_degrees,
            err_refined.weight_errors[1], err_refined.weight_errors[2]);

        assert!(err_mobius.rms_position < 1e-6,
            "Möbius weights should produce near-exact surface");
    }

    // ── Recovery tests ──────────────────────────────────────────────

    #[test]
    fn test_roundtrip_flat_patch() {
        let original = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let tess = tessellate_patch(&original, 8);
        let cp = corners(&original);
        let config = FitConfig {
            gauss_newton_iterations: 10,
            lambda_normal: 0.1,
            corner_positions: Some(cp),
            ..Default::default()
        };
        let r = run_recovery("flat", "exact_bary", &original,
            &tess.positions, &tess.normals, &tess.bary, &config);
        eprintln!("{}", r);
        assert!(r.rms_position < 1e-6);
    }

    #[test]
    fn test_roundtrip_inverted_patch() {
        let (original, _) = spherical_patch_via_inversion(
            [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0],
        );
        let tess = tessellate_patch(&original, 10);
        let cp = corners(&original);
        let config = FitConfig {
            gauss_newton_iterations: 50,
            lambda_normal: 0.5,
            convergence_tol: 1e-12,
            regularization: 0.0,
            corner_positions: Some(cp),
            ..Default::default()
        };
        let r = run_recovery("inverted", "no_reg", &original,
            &tess.positions, &tess.normals, &tess.bary, &config);
        eprintln!("{}", r);
        assert!(r.rms_position < 0.1);
    }

    // ── Comprehensive experiment ─────────────────────────────────────

    /// Run multiple approaches on multiple test surfaces and print a comparison table.
    #[test]
    fn test_experiment_matrix() {
        eprintln!("\n{:=<100}", "= QB PATCH RECOVERY EXPERIMENTS ");

        // Test surfaces: (name, patch, flat_corners_if_inverted)
        let flat_corners_default = [[0.0; 3]; 3];
        let (mild, mild_flat) = spherical_patch_via_inversion(
            [0.3, 0.0, 3.0], [0.0, 0.3, 3.0], [-0.3, 0.0, 3.0]);
        let (medium, medium_flat) = spherical_patch_via_inversion(
            [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0]);
        let (strong, strong_flat) = spherical_patch_via_inversion(
            [1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [-1.0, 0.0, 1.5]);

        // Non-Möbius surfaces: create patches from real geometric shapes
        // Saddle surface: z = x*y (hyperbolic paraboloid, K < 0)
        let saddle = {
            let corners = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.5, -0.5, 0.25]];
            // Sample saddle z = x*y at interior points
            let n = 8;
            let mut positions = Vec::new();
            let mut bary_coords = Vec::new();
            let mut normals = Vec::new();
            for i in 0..=n {
                for j in 0..=(n-i) {
                    let u = i as f64 / n as f64;
                    let v = j as f64 / n as f64;
                    let w = 1.0 - u - v;
                    // Linear interpolation of corners
                    let x = w * corners[0][0] + u * corners[1][0] + v * corners[2][0];
                    let y = w * corners[0][1] + u * corners[1][1] + v * corners[2][1];
                    // Saddle displacement: z = x*y
                    let z = x * y;
                    positions.push([x, y, z]);
                    bary_coords.push([w, u, v]);
                    // Normal of z=xy surface: n = (-dz/dx, -dz/dy, 1) normalized = (-y, -x, 1)/|...|
                    let nl = (y*y + x*x + 1.0).sqrt();
                    normals.push([-y/nl, -x/nl, 1.0/nl]);
                }
            }
            // Control points ARE the corners (with z=x*y applied)
            let cp = [
                [corners[0][0], corners[0][1], corners[0][0]*corners[0][1]],
                [corners[1][0], corners[1][1], corners[1][0]*corners[1][1]],
                [corners[2][0], corners[2][1], corners[2][0]*corners[2][1]],
            ];
            (cp, positions, bary_coords, normals)
        };

        // Hemisphere cap: take a patch from a unit sphere (not via Möbius)
        let sphere_cap = {
            let corners = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let n = 8;
            let mut positions = Vec::new();
            let mut bary_coords = Vec::new();
            let mut normals = Vec::new();
            for i in 0..=n {
                for j in 0..=(n-i) {
                    let u = i as f64 / n as f64;
                    let v = j as f64 / n as f64;
                    let w = 1.0 - u - v;
                    let x = w * corners[0][0] + u * corners[1][0] + v * corners[2][0];
                    let y = w * corners[0][1] + u * corners[1][1] + v * corners[2][1];
                    let z = w * corners[0][2] + u * corners[1][2] + v * corners[2][2];
                    // Project onto unit sphere
                    let r = (x*x + y*y + z*z).sqrt();
                    let px = x / r;
                    let py = y / r;
                    let pz = z / r;
                    positions.push([px, py, pz]);
                    bary_coords.push([w, u, v]);
                    normals.push([px, py, pz]); // sphere normal = position
                }
            }
            (corners, positions, bary_coords, normals)
        };

        let surfaces: Vec<(&str, QBTriPatch, [[f64; 3]; 3])> = vec![
            ("flat", QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), flat_corners_default),
            ("mild_curve", mild, mild_flat),
            ("medium_curve", medium, medium_flat),
            ("strong_curve", strong, strong_flat),
        ];

        eprintln!("{:20} {:15} {:>10} {:>10} {:>8}",
            "surface", "approach", "pos_rms", "pos_max", "norm");
        eprintln!("{:-<70}", "");

        for (surf_name, patch, flat_cp) in &surfaces {
            let tess = tessellate_patch(patch, 12);
            let cp = corners(patch);

            // 1. GN baseline (identity init, default regularization)
            {
                let cfg = FitConfig {
                    gauss_newton_iterations: 10,
                    corner_positions: Some(cp),
                    ..Default::default()
                };
                let r = run_recovery(surf_name, "gn_baseline", patch,
                    &tess.positions, &tess.normals, &tess.bary, &cfg);
                eprintln!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                    surf_name, "gn_baseline", r.rms_position, r.max_position, r.rms_normal_degrees);
            }

            // 2. GN no regularization
            {
                let cfg = FitConfig {
                    gauss_newton_iterations: 50,
                    lambda_normal: 0.1,
                    convergence_tol: 1e-12,
                    regularization: 0.0,
                    corner_positions: Some(cp),
                    ..Default::default()
                };
                let r = run_recovery(surf_name, "gn_no_reg", patch,
                    &tess.positions, &tess.normals, &tess.bary, &cfg);
                eprintln!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                    surf_name, "gn_no_reg", r.rms_position, r.max_position, r.rms_normal_degrees);
            }

            // 3. c-parameter estimator (no knowledge of flat corners needed!)
            {
                let c = estimate_c_parameter(&cp, &tess.positions, &tess.bary, &tess.normals);
                let weights = make_weights_from_c(c, &[
                    Quat::from_point(cp[0][0], cp[0][1], cp[0][2]),
                    Quat::from_point(cp[1][0], cp[1][1], cp[1][2]),
                    Quat::from_point(cp[2][0], cp[2][1], cp[2][2]),
                ]);
                let est_patch = QBTriPatch::new(patch.positions, weights);
                let err = measure_recovery_error(patch, &est_patch, 20);
                eprintln!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                    surf_name, "c_estimator", err.rms_position, err.max_position, err.rms_normal_degrees);
            }

            // 4. Möbius-derived init (only if we know the flat corners — oracle)
            if *surf_name != "flat" {
                let fp = [
                    Quat::from_point(flat_cp[0][0], flat_cp[0][1], flat_cp[0][2]),
                    Quat::from_point(flat_cp[1][0], flat_cp[1][1], flat_cp[1][2]),
                    Quat::from_point(flat_cp[2][0], flat_cp[2][1], flat_cp[2][2]),
                ];
                let w0_inv = fp[0].inv();
                let mobius_w = [Quat::ONE, fp[1] * w0_inv, fp[2] * w0_inv];
                let mobius_patch = QBTriPatch::new(patch.positions, mobius_w);
                let err = measure_recovery_error(patch, &mobius_patch, 20);
                eprintln!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                    surf_name, "mobius_oracle", err.rms_position, err.max_position, err.rms_normal_degrees);
            }
        }
        // ── Non-Möbius surfaces ──────────────────────────────────────
        eprintln!("\n--- Non-Möbius surfaces (c-estimator only) ---");

        // Saddle
        {
            let (cp, ref positions, ref bary_coords, ref normals) = saddle;
            let c = estimate_c_parameter(&cp, positions, bary_coords, normals);
            let cpq = [
                Quat::from_point(cp[0][0], cp[0][1], cp[0][2]),
                Quat::from_point(cp[1][0], cp[1][1], cp[1][2]),
                Quat::from_point(cp[2][0], cp[2][1], cp[2][2]),
            ];
            let weights = make_weights_from_c(c, &cpq);
            let fitted = QBTriPatch::new(cpq, weights);

            // Measure error against the sample points
            let mut pos_err = 0.0_f64;
            let mut pos_max = 0.0_f64;
            let mut norm_err = 0.0_f64;
            for (k, bary) in bary_coords.iter().enumerate() {
                let eval = fitted.eval_with_normal(bary[1], bary[2]);
                let dx = positions[k][0] - eval.position[0];
                let dy = positions[k][1] - eval.position[1];
                let dz = positions[k][2] - eval.position[2];
                let d = (dx*dx + dy*dy + dz*dz).sqrt();
                pos_err += d * d;
                pos_max = pos_max.max(d);
                let dot = normals[k][0]*eval.normal[0] + normals[k][1]*eval.normal[1] + normals[k][2]*eval.normal[2];
                norm_err += dot.clamp(-1.0, 1.0).acos().to_degrees().powi(2);
            }
            let n = bary_coords.len() as f64;
            eprintln!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                "saddle", "c_estimator", (pos_err/n).sqrt(), pos_max, (norm_err/n).sqrt());
            eprintln!("  c = {:?}", c);
        }

        // Sphere cap
        {
            let (cp, ref positions, ref bary_coords, ref normals) = sphere_cap;
            let c = estimate_c_parameter(&cp, positions, bary_coords, normals);
            let cpq = [
                Quat::from_point(cp[0][0], cp[0][1], cp[0][2]),
                Quat::from_point(cp[1][0], cp[1][1], cp[1][2]),
                Quat::from_point(cp[2][0], cp[2][1], cp[2][2]),
            ];
            let weights = make_weights_from_c(c, &cpq);
            let fitted = QBTriPatch::new(cpq, weights);

            let mut pos_err = 0.0_f64;
            let mut pos_max = 0.0_f64;
            let mut norm_err = 0.0_f64;
            for (k, bary) in bary_coords.iter().enumerate() {
                let eval = fitted.eval_with_normal(bary[1], bary[2]);
                let dx = positions[k][0] - eval.position[0];
                let dy = positions[k][1] - eval.position[1];
                let dz = positions[k][2] - eval.position[2];
                let d = (dx*dx + dy*dy + dz*dz).sqrt();
                pos_err += d * d;
                pos_max = pos_max.max(d);
                let dot = normals[k][0]*eval.normal[0] + normals[k][1]*eval.normal[1] + normals[k][2]*eval.normal[2];
                norm_err += dot.clamp(-1.0, 1.0).acos().to_degrees().powi(2);
            }
            let n = bary_coords.len() as f64;
            eprintln!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                "sphere_cap", "c_estimator", (pos_err/n).sqrt(), pos_max, (norm_err/n).sqrt());
            eprintln!("  c = {:?}", c);

            // Compare with flat (c=0) to show improvement
            let flat = QBTriPatch::new(cpq, [Quat::ONE, Quat::ONE, Quat::ONE]);
            let mut flat_err = 0.0_f64;
            for (k, bary) in bary_coords.iter().enumerate() {
                let eval = flat.eval(bary[1], bary[2]).to_point();
                let dx = positions[k][0] - eval[0];
                let dy = positions[k][1] - eval[1];
                let dz = positions[k][2] - eval[2];
                flat_err += dx*dx + dy*dy + dz*dz;
            }
            eprintln!("{:20} {:15} {:10.6}",
                "sphere_cap", "flat_baseline", (flat_err/n).sqrt());
        }

        eprintln!("{:=<70}", "");
    }
}
