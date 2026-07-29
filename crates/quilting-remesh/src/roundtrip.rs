//! Tessellation and round-trip ground truth for QB patch recovery.
//!
//! The loop: create QB patches with known weights → tessellate to triangles
//! → run recovery → compare recovered weights and surface to ground truth.
//! The tests here are the reference for the underlying math (Krasauskas–Zubė
//! eq. 5): they pin down that a Möbius image of a flat triangle really is a QB
//! patch whose weights are `wᵢ = c*pᵢ + d`, and that our tessellator agrees.
//!
//! [`tessellate_patch`] is also on the live path — the WASM remesh viewer uses
//! it to turn fitted patches into drawable triangles.
//!
//! The production fitter itself lives in [`crate::c_estimator`]; the
//! Gauss-Newton weight fitter in [`crate::fit`] that these tests exercise is a
//! research baseline that was never good enough to ship.

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
}
