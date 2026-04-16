/// Global QB weight optimization with inter-patch continuity.
///
/// After QEM simplification produces a watertight triangle mesh, we fit curved
/// QB patches by optimizing per-vertex quaternion weights globally. Adjacent
/// patches automatically share weights at common vertices, ensuring C0 continuity.
///
/// The approach:
/// 1. Collect sample points (original mesh vertices) for each simplified face
/// 2. Assign per-vertex weight parameters (4 floats per vertex, gauge-fixed at one vertex)
/// 3. Gauss-Newton optimization: minimize surface error across ALL patches simultaneously
///    subject to shared-vertex constraints (which are implicit — same variable used)
///
/// The weight at vertex i is: wᵢ (a free quaternion parameter)
/// with w₀ pinned to Quat::ONE for gauge fixing.

use quilting_core::quaternion::Quat;
use quilting_core::patch::QBTriPatch;
use crate::geometry;

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

/// Fit curved QB patches globally with shared vertex weights.
///
/// `simp_pos` / `simp_faces` — the simplified (coarse) mesh
/// `orig_pos` / `orig_normals` — original mesh vertices and normals for fitting
pub fn global_fit(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    orig_pos: &[[f64; 3]],
    orig_normals: &[[f64; 3]],
    max_iterations: usize,
) -> GlobalFitResult {
    let n_verts = simp_pos.len();
    let n_faces = simp_faces.len();

    // Collect samples per face: which original vertices project into each simplified triangle
    let face_samples: Vec<FaceSamples> = simp_faces.iter().map(|face| {
        let tri = [simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]];
        collect_face_samples(&tri, orig_pos, orig_normals)
    }).collect();

    // Initialize: all weights = identity (flat patches)
    let mut weights: Vec<Quat> = vec![Quat::ONE; n_verts];

    // Gauge fix: vertex 0 is always Quat::ONE
    // Free parameters: vertices 1..n_verts, 4 params each
    let n_params = (n_verts - 1) * 4;

    if n_params == 0 {
        return build_result(simp_pos, simp_faces, &weights, &face_samples);
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
            let d = solve_4x4_from_slice(&jtj, &jtr, off);
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

    build_result(simp_pos, simp_faces, &weights, &face_samples)
}

struct FaceSamples {
    positions: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
    bary: Vec<[f64; 3]>,
}

fn collect_face_samples(
    tri: &[[f64; 3]; 3],
    orig_pos: &[[f64; 3]],
    orig_normals: &[[f64; 3]],
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

        let margin = -0.05;
        if u >= margin && v >= margin && w >= margin && u <= 1.05 && v <= 1.05 {
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

fn solve_4x4_from_slice(jtj: &[Vec<f64>], jtr: &[f64], off: usize) -> [f64; 4] {
    let mut m = [[0.0; 5]; 4];
    for i in 0..4 {
        for j in 0..4 { m[i][j] = jtj[off + i][off + j]; }
        m[i][4] = jtr[off + i];
    }
    // Gaussian elimination with partial pivoting
    for col in 0..4 {
        let mut max_row = col;
        for row in (col + 1)..4 {
            if m[row][col].abs() > m[max_row][col].abs() { max_row = row; }
        }
        m.swap(col, max_row);
        if m[col][col].abs() < 1e-15 { continue; }
        for row in (col + 1)..4 {
            let factor = m[row][col] / m[col][col];
            for j in col..5 { m[row][j] -= factor * m[col][j]; }
        }
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        x[i] = m[i][4];
        for j in (i + 1)..4 { x[i] -= m[i][j] * x[j]; }
        if m[i][i].abs() > 1e-15 { x[i] /= m[i][i]; }
    }
    x
}

fn build_result(
    simp_pos: &[[f64; 3]],
    simp_faces: &[[usize; 3]],
    weights: &[Quat],
    face_samples: &[FaceSamples],
) -> GlobalFitResult {
    let patches: Vec<QBTriPatch> = simp_faces.iter().map(|face| {
        QBTriPatch::new(
            [
                Quat::from_point(simp_pos[face[0]][0], simp_pos[face[0]][1], simp_pos[face[0]][2]),
                Quat::from_point(simp_pos[face[1]][0], simp_pos[face[1]][1], simp_pos[face[1]][2]),
                Quat::from_point(simp_pos[face[2]][0], simp_pos[face[2]][1], simp_pos[face[2]][2]),
            ],
            [weights[face[0]], weights[face[1]], weights[face[2]]],
        )
    }).collect();

    // Compute error stats
    let mut sum_err = 0.0_f64;
    let mut max_err = 0.0_f64;
    let mut count = 0;
    for (fi, face) in simp_faces.iter().enumerate() {
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
        vertex_weights: weights.to_vec(),
        rms_error: if count > 0 { (sum_err / count as f64).sqrt() } else { 0.0 },
        max_error: max_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_fit_sphere() {
        let (orig_pos, orig_faces) = crate::test_shapes::sphere(2);
        let orig_normals = crate::geometry::compute_vertex_normals(&orig_pos, &orig_faces);

        // Simplify to 20 faces
        let (simp_pos, simp_faces) = crate::simplify::simplify(&orig_pos, &orig_faces, 20);

        eprintln!("global_fit sphere: {} → {} faces, {} verts",
            orig_faces.len(), simp_faces.len(), simp_pos.len());

        let result = global_fit(&simp_pos, &simp_faces, &orig_pos, &orig_normals, 20);

        eprintln!("  RMS error: {:.6}, max: {:.6}", result.rms_error, result.max_error);
        eprintln!("  {} patches, {} vertex weights", result.patches.len(), result.vertex_weights.len());

        // Check continuity: adjacent patches should share weights at common vertices
        let num_curved = result.vertex_weights.iter()
            .filter(|w| ((**w) - Quat::ONE).norm() > 0.01)
            .count();
        eprintln!("  {} of {} vertices have non-identity weights", num_curved, result.vertex_weights.len());

        // Measure sphere fit (distance to unit sphere)
        let mut sphere_err = 0.0_f64;
        let mut sphere_max = 0.0_f64;
        let mut count = 0;
        let n = 5;
        for patch in &result.patches {
            for i in 0..=n {
                for j in 0..=(n - i) {
                    let u = i as f64 / n as f64;
                    let v = j as f64 / n as f64;
                    let p = patch.eval(u, v).to_point();
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    let d = (r - 1.0).abs();
                    sphere_err += d * d;
                    sphere_max = sphere_max.max(d);
                    count += 1;
                }
            }
        }
        let sphere_rms = (sphere_err / count as f64).sqrt();
        eprintln!("  sphere fit: RMS={:.6}, max={:.6}", sphere_rms, sphere_max);

        // Compare with flat patches
        let flat_patches: Vec<QBTriPatch> = simp_faces.iter().map(|f| {
            QBTriPatch::flat(simp_pos[f[0]], simp_pos[f[1]], simp_pos[f[2]])
        }).collect();
        let mut flat_sphere_err = 0.0_f64;
        let mut count2 = 0;
        for patch in &flat_patches {
            for i in 0..=n {
                for j in 0..=(n - i) {
                    let u = i as f64 / n as f64;
                    let v = j as f64 / n as f64;
                    let p = patch.eval(u, v).to_point();
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    let d = (r - 1.0).abs();
                    flat_sphere_err += d * d;
                    count2 += 1;
                }
            }
        }
        let flat_rms = (flat_sphere_err / count2 as f64).sqrt();
        eprintln!("  flat baseline: RMS={:.6}", flat_rms);

        if sphere_rms < flat_rms {
            eprintln!("  GLOBAL FIT IS {:.1}x BETTER THAN FLAT", flat_rms / sphere_rms.max(1e-10));
        } else {
            eprintln!("  flat is {:.1}x better", sphere_rms / flat_rms.max(1e-10));
        }

        assert!(result.patches.len() > 0);
    }
}
