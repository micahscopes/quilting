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
use quilting_core::quaternion::Quat;

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
pub fn spherical_patch_via_inversion(
    p0: [f64; 3], p1: [f64; 3], p2: [f64; 3],
) -> QBTriPatch {
    use quilting_core::quaternion::Mobius;

    // Start with a flat patch
    let flat = QBTriPatch::flat(p0, p1, p2);

    // Inversion: x ↦ -x⁻¹ maps planes not through origin to spheres
    let inv = Mobius::inversion();
    flat.transform(&inv)
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
        let patch = spherical_patch_via_inversion(
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
        let original = spherical_patch_via_inversion(
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

        // Test surfaces
        let surfaces: Vec<(&str, QBTriPatch)> = vec![
            ("flat", QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])),
            ("mild_curve", spherical_patch_via_inversion(
                [0.3, 0.0, 3.0], [0.0, 0.3, 3.0], [-0.3, 0.0, 3.0])),
            ("medium_curve", spherical_patch_via_inversion(
                [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0])),
            ("strong_curve", spherical_patch_via_inversion(
                [1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [-1.0, 0.0, 1.5])),
        ];

        // Approaches: vary GN iterations, regularization (via lambda_normal), etc.
        let approaches: Vec<(&str, FitConfig)> = vec![
            ("baseline", FitConfig {
                gauss_newton_iterations: 10,
                lambda_normal: 0.05,
                ..Default::default()
            }),
            ("more_iters", FitConfig {
                gauss_newton_iterations: 50,
                lambda_normal: 0.05,
                convergence_tol: 1e-12,
                ..Default::default()
            }),
            ("no_reg", FitConfig {
                gauss_newton_iterations: 50,
                lambda_normal: 0.1,
                convergence_tol: 1e-12,
                regularization: 0.0,
                ..Default::default()
            }),
            ("low_reg", FitConfig {
                gauss_newton_iterations: 50,
                lambda_normal: 0.1,
                convergence_tol: 1e-12,
                regularization: 0.001,
                ..Default::default()
            }),
            ("heavy_normal_no_reg", FitConfig {
                gauss_newton_iterations: 50,
                lambda_normal: 1.0,
                convergence_tol: 1e-12,
                regularization: 0.0,
                ..Default::default()
            }),
        ];

        // Also test with "oracle" initial weights (perturbed ground truth)
        // to verify the optimizer CAN recover if started near the right basin
        eprintln!("\n--- Oracle init (perturbed ground truth) ---");
        for (surf_name, patch) in &surfaces {
            let tess = tessellate_patch(patch, 12);
            let cp = corners(patch);

            // Test exact true weights (zero perturbation) — verify it's a minimum
            let perturbed = patch.weights;

            let cfg = FitConfig {
                gauss_newton_iterations: 50,
                lambda_normal: 0.1,
                convergence_tol: 1e-12,
                regularization: 0.0,
                corner_positions: Some(cp),
                initial_weights: Some(perturbed),
                ..Default::default()
            };

            let r = run_recovery(surf_name, "oracle_init", patch,
                &tess.positions, &tess.normals, &tess.bary, &cfg);

            eprintln!("{:20} {:20} {:10.6} {:10.6} {:7.2}° {:8.4} {:8.4}",
                surf_name, "oracle_init",
                r.rms_position, r.max_position,
                r.rms_normal_degrees,
                r.weight_error_w1, r.weight_error_w2);
        }

        eprintln!("{:20} {:20} {:>10} {:>10} {:>8} {:>8} {:>8}",
            "surface", "approach", "pos_rms", "pos_max", "norm", "w1_err", "w2_err");
        eprintln!("{:-<100}", "");

        for (surf_name, patch) in &surfaces {
            let tess = tessellate_patch(patch, 12);
            let cp = corners(patch);

            for (app_name, config) in &approaches {
                let mut cfg = config.clone();
                cfg.corner_positions = Some(cp);

                let r = run_recovery(surf_name, app_name, patch,
                    &tess.positions, &tess.normals, &tess.bary, &cfg);

                eprintln!("{:20} {:20} {:10.6} {:10.6} {:7.2}° {:8.4} {:8.4}",
                    surf_name, app_name,
                    r.rms_position, r.max_position,
                    r.rms_normal_degrees,
                    r.weight_error_w1, r.weight_error_w2);
            }
        }
        eprintln!("{:=<100}", "");
    }
}
