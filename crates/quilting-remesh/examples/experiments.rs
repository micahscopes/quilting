//! Comparison harness for QB patch recovery approaches.
//!
//! These used to be `#[test]` functions with zero assertions — tables that ran on
//! every `cargo test` and that nobody diffed. They are exploratory measurements,
//! not invariants, so they live here instead.
//!
//! Usage:
//!   cargo run --example experiments             # everything
//!   cargo run --example experiments -- recovery # single-patch recovery matrix
//!   cargo run --example experiments -- global   # global vs per-patch fitting

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::Quat;
use quilting_remesh::c_estimator::{estimate_c_parameter, make_weights_from_c, CEstimatorConfig};
use quilting_remesh::fit::FitConfig;
use quilting_remesh::global_fit::{global_fit, CurvedFitConfig};
use quilting_remesh::roundtrip::{
    measure_recovery_error, run_recovery, spherical_patch_via_inversion, tessellate_patch,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    match which {
        "recovery" => recovery_matrix(),
        "global" => global_fit_comparison(),
        "all" => {
            recovery_matrix();
            global_fit_comparison();
        }
        other => {
            eprintln!("unknown experiment `{}`; expected one of: recovery, global, all", other);
            std::process::exit(2);
        }
    }
}

fn corners(patch: &QBTriPatch) -> [[f64; 3]; 3] {
    [
        patch.positions[0].to_point(),
        patch.positions[1].to_point(),
        patch.positions[2].to_point(),
    ]
}

fn as_quats(cp: &[[f64; 3]; 3]) -> [Quat; 3] {
    [
        Quat::from_point(cp[0][0], cp[0][1], cp[0][2]),
        Quat::from_point(cp[1][0], cp[1][1], cp[1][2]),
        Quat::from_point(cp[2][0], cp[2][1], cp[2][2]),
    ]
}

/// Run multiple approaches on multiple test surfaces and print a comparison table.
fn recovery_matrix() {
    println!("\n{:=<100}", "= QB PATCH RECOVERY EXPERIMENTS ");

    let c_cfg = CEstimatorConfig::default();

    // Test surfaces: (name, patch, flat_corners_if_inverted)
    let flat_corners_default = [[0.0; 3]; 3];
    let (mild, mild_flat) = spherical_patch_via_inversion(
        [0.3, 0.0, 3.0], [0.0, 0.3, 3.0], [-0.3, 0.0, 3.0]);
    let (medium, medium_flat) = spherical_patch_via_inversion(
        [0.5, 0.0, 2.0], [0.0, 0.5, 2.0], [-0.5, 0.0, 2.0]);
    let (strong, strong_flat) = spherical_patch_via_inversion(
        [1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [-1.0, 0.0, 1.5]);

    // Non-Möbius surfaces: create patches from real geometric shapes.
    // Saddle surface: z = x*y (hyperbolic paraboloid, K < 0)
    let saddle = {
        let corners = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.5, -0.5, 0.25]];
        let n = 8;
        let mut positions = Vec::new();
        let mut bary_coords = Vec::new();
        let mut normals = Vec::new();
        for i in 0..=n {
            for j in 0..=(n - i) {
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
                let nl = (y * y + x * x + 1.0).sqrt();
                normals.push([-y / nl, -x / nl, 1.0 / nl]);
            }
        }
        // Control points ARE the corners (with z=x*y applied)
        let cp = [
            [corners[0][0], corners[0][1], corners[0][0] * corners[0][1]],
            [corners[1][0], corners[1][1], corners[1][0] * corners[1][1]],
            [corners[2][0], corners[2][1], corners[2][0] * corners[2][1]],
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
            for j in 0..=(n - i) {
                let u = i as f64 / n as f64;
                let v = j as f64 / n as f64;
                let w = 1.0 - u - v;
                let x = w * corners[0][0] + u * corners[1][0] + v * corners[2][0];
                let y = w * corners[0][1] + u * corners[1][1] + v * corners[2][1];
                let z = w * corners[0][2] + u * corners[1][2] + v * corners[2][2];
                // Project onto unit sphere
                let r = (x * x + y * y + z * z).sqrt();
                let (px, py, pz) = (x / r, y / r, z / r);
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

    println!("{:20} {:15} {:>10} {:>10} {:>8}", "surface", "approach", "pos_rms", "pos_max", "norm");
    println!("{:-<70}", "");

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
            println!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
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
            println!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                surf_name, "gn_no_reg", r.rms_position, r.max_position, r.rms_normal_degrees);
        }

        // 3. c-parameter estimator (no knowledge of flat corners needed)
        {
            let c = estimate_c_parameter(&cp, &tess.positions, &tess.bary, &tess.normals, &c_cfg);
            let weights = make_weights_from_c(c, &as_quats(&cp));
            let est_patch = QBTriPatch::new(patch.positions, weights);
            let err = measure_recovery_error(patch, &est_patch, 20);
            println!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                surf_name, "c_estimator", err.rms_position, err.max_position, err.rms_normal_degrees);
        }

        // 4. Möbius-derived init (only if we know the flat corners — oracle)
        if *surf_name != "flat" {
            let fp = as_quats(flat_cp);
            let w0_inv = fp[0].inv();
            let mobius_w = [Quat::ONE, fp[1] * w0_inv, fp[2] * w0_inv];
            let mobius_patch = QBTriPatch::new(patch.positions, mobius_w);
            let err = measure_recovery_error(patch, &mobius_patch, 20);
            println!("{:20} {:15} {:10.6} {:10.6} {:7.2}°",
                surf_name, "mobius_oracle", err.rms_position, err.max_position, err.rms_normal_degrees);
        }
    }

    // ── Non-Möbius surfaces ──────────────────────────────────────────
    println!("\n--- Non-Möbius surfaces (c-estimator only) ---");

    for (label, (cp, positions, bary_coords, normals)) in
        [("saddle", &saddle), ("sphere_cap", &sphere_cap)]
    {
        let c = estimate_c_parameter(cp, positions, bary_coords, normals, &c_cfg);
        let cpq = as_quats(cp);
        let fitted = QBTriPatch::new(cpq, make_weights_from_c(c, &cpq));

        let (rms, max, norm_rms) = sample_error(&fitted, positions, bary_coords, normals);
        println!("{:20} {:15} {:10.6} {:10.6} {:7.2}°", label, "c_estimator", rms, max, norm_rms);
        println!("  c = {:?}", c);

        // Flat (c=0) baseline for comparison
        let flat = QBTriPatch::new(cpq, [Quat::ONE, Quat::ONE, Quat::ONE]);
        let (flat_rms, _, _) = sample_error(&flat, positions, bary_coords, normals);
        println!("{:20} {:15} {:10.6}", label, "flat_baseline", flat_rms);
    }

    // ── Realistic test: actual mesh pipeline ─────────────────────────
    println!("\n--- Realistic: QEM sphere simplification ---");
    for (subdivs, target) in &[(2, 20), (3, 40), (3, 20)] {
        let (positions, faces) = quilting_remesh::test_shapes::sphere(*subdivs);
        let flat_result = quilting_remesh::remesh_simplified(&positions, &faces, *target).unwrap();
        let curved_result =
            quilting_remesh::remesh_simplified_curved(&positions, &faces, *target).unwrap();

        let (flat_rms, flat_max) = measure_sphere_fit(&flat_result.patches);
        let (curved_rms, curved_max) = measure_sphere_fit(&curved_result.patches);

        let label = format!("sph{}→{}", faces.len(), target);
        println!("{:20} {:15} {:10.6} {:10.6}", label, "flat", flat_rms, flat_max);
        println!("{:20} {:15} {:10.6} {:10.6}", label, "curved", curved_rms, curved_max);

        // |w1 - 1| is a rough proxy for how much curvature each patch took on
        let c_norms: Vec<f64> = curved_result.patches.iter()
            .map(|p| (p.weights[1] - Quat::ONE).norm())
            .collect();
        let avg_c = c_norms.iter().sum::<f64>() / c_norms.len() as f64;
        let max_c = c_norms.iter().fold(0.0_f64, |a, &b| a.max(b));
        println!("  avg |w1-1|={:.4}, max={:.4}", avg_c, max_c);

        if curved_rms < flat_rms {
            println!("  curved is {:.1}x better than flat", flat_rms / curved_rms.max(1e-10));
        } else {
            println!("  flat is {:.1}x better", curved_rms / flat_rms.max(1e-10));
        }
    }

    println!("{:=<70}", "");
}

/// Compare flat / c-init / c-init+Gauss-Newton on QEM-simplified spheres.
fn global_fit_comparison() {
    println!("\n{:=<80}", "= GLOBAL FIT EXPERIMENTS ");
    println!("{:20} {:15} {:>10} {:>10} {:>10}",
        "config", "approach", "sph_rms", "sph_max", "samp_rms");
    println!("{:-<80}", "");

    let cfg = CurvedFitConfig::default();

    for (subdivs, target) in &[(2, 20), (3, 20), (3, 40)] {
        let (orig_pos, orig_faces) = quilting_remesh::test_shapes::sphere(*subdivs);
        let orig_normals = quilting_remesh::geometry::compute_vertex_normals(&orig_pos, &orig_faces);
        let (simp_pos, simp_faces) =
            quilting_remesh::simplify::simplify(&orig_pos, &orig_faces, *target);

        let label = format!("sph{}→{}", orig_faces.len(), simp_faces.len());

        // Flat baseline
        let flat_patches: Vec<QBTriPatch> = simp_faces.iter()
            .map(|f| QBTriPatch::flat(simp_pos[f[0]], simp_pos[f[1]], simp_pos[f[2]]))
            .collect();
        let (flat_rms, flat_max) = measure_sphere_fit(&flat_patches);
        println!("{:20} {:15} {:10.6} {:10.6} {:>10}", label, "flat", flat_rms, flat_max, "—");

        // c-init only (0 GN iterations) — tests the initialization quality
        let init_only = global_fit(&simp_pos, &simp_faces, &orig_pos, &orig_normals, 0, &cfg);
        let (init_rms, init_max) = measure_sphere_fit(&init_only.patches);
        println!("{:20} {:15} {:10.6} {:10.6} {:10.6}",
            "", "c_init_only", init_rms, init_max, init_only.rms_error);

        // c-init + 10 GN iterations
        let refined = global_fit(&simp_pos, &simp_faces, &orig_pos, &orig_normals, 10, &cfg);
        let (ref_rms, ref_max) = measure_sphere_fit(&refined.patches);
        println!("{:20} {:15} {:10.6} {:10.6} {:10.6}",
            "", "c_init+gn10", ref_rms, ref_max, refined.rms_error);

        // Per-patch fit, the live product path, for reference
        let per_patch = quilting_remesh::global_fit::per_patch_fit(
            &simp_pos, &simp_faces, &orig_pos, &orig_normals, &cfg);
        let (pp_rms, pp_max) = measure_sphere_fit(&per_patch.patches);
        println!("{:20} {:15} {:10.6} {:10.6} {:10.6}",
            "", "per_patch", pp_rms, pp_max, per_patch.rms_error);

        if ref_rms < flat_rms {
            println!("{:20} → global fit is {:.1}x better than flat", "", flat_rms / ref_rms);
        }
    }
    println!("{:=<80}", "");
}

/// RMS / max deviation of the patch set from the unit sphere.
fn measure_sphere_fit(patches: &[QBTriPatch]) -> (f64, f64) {
    let mut err = 0.0_f64;
    let mut max_d = 0.0_f64;
    let mut count = 0;
    let n = 5;
    for patch in patches {
        for i in 0..=n {
            for j in 0..=(n - i) {
                let u = i as f64 / n as f64;
                let v = j as f64 / n as f64;
                let p = patch.eval(u, v).to_point();
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let d = (r - 1.0).abs();
                err += d * d;
                max_d = max_d.max(d);
                count += 1;
            }
        }
    }
    ((err / count as f64).sqrt(), max_d)
}

/// (position RMS, position max, normal RMS in degrees) of a patch against samples.
fn sample_error(
    patch: &QBTriPatch,
    positions: &[[f64; 3]],
    bary: &[[f64; 3]],
    normals: &[[f64; 3]],
) -> (f64, f64, f64) {
    let mut pos_err = 0.0_f64;
    let mut pos_max = 0.0_f64;
    let mut norm_err = 0.0_f64;
    for (k, b) in bary.iter().enumerate() {
        let eval = patch.eval_with_normal(b[1], b[2]);
        let dx = positions[k][0] - eval.position[0];
        let dy = positions[k][1] - eval.position[1];
        let dz = positions[k][2] - eval.position[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        pos_err += d * d;
        pos_max = pos_max.max(d);
        let dot = normals[k][0] * eval.normal[0]
            + normals[k][1] * eval.normal[1]
            + normals[k][2] * eval.normal[2];
        norm_err += dot.clamp(-1.0, 1.0).acos().to_degrees().powi(2);
    }
    let n = bary.len() as f64;
    ((pos_err / n).sqrt(), pos_max, (norm_err / n).sqrt())
}
