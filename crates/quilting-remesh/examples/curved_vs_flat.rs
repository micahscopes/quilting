//! Curved (linear global fit) vs flat baseline, on inputs of increasing realism.
//!
//! Run: `cargo run --example curved_vs_flat`  (add `--release` to iterate faster;
//! it also runs in debug).
//!
//! For each input we fit curved QB patches with one shared quaternion weight per
//! coarse vertex via [`quilting_remesh::linear_fit::linear_global_fit`], build the
//! flat baseline (same faces, all weights = ONE), and report:
//!
//!   * patch count
//!   * curved / flat RMS + max surface error (patch-at-sample-bary vs target)
//!   * the improvement ratio (flat RMS / curved RMS)
//!   * `max_v |wᵥ − 1|`  — if ≈ 0 the solve did nothing and "curved" == flat
//!   * curvature deviation — max |curved − flat| at patch centroids (proves the
//!     patches are genuinely non-planar)
//!   * the maximum C0 edge gap — sampled from BOTH patches sharing each edge;
//!     ≈ machine-epsilon proves watertightness
//!   * the accepted sparse LSQR relative normal residual
//!   * blow-up count — samples where the rational patch ran off to infinity

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::{Mobius, Quat};
use quilting_remesh::geometry;
use quilting_remesh::linear_fit::{self, LinearFitConfig, Sample};
use quilting_remesh::{simplify, test_shapes};
use std::collections::HashMap;

fn main() {
    let cases = vec![
        sphere_ground_truth_case(),
        sphere_qem_case(),
        cylinder_qem_case(),
        torus_qem_case(),
    ];

    println!();
    println!("Linear global QB weight fit  —  curved vs flat");
    println!("errors are absolute (shapes are ~unit scale); RMS/max are patch-at-sample-bary vs target\n");

    // Error / improvement table.
    println!(
        "{:<26} {:>7} {:>11} {:>11} {:>11} {:>11} {:>8}",
        "case", "patches", "curvedRMS", "curvedMax", "flatRMS", "flatMax", "ratio"
    );
    println!("{}", "-".repeat(26 + 7 + 11 * 4 + 8 + 6));
    for c in &cases {
        let ratio = if c.curved_rms > 0.0 { c.flat_rms / c.curved_rms } else { f64::INFINITY };
        println!(
            "{:<26} {:>7} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>8.1}",
            c.name, c.n_patches, c.curved_rms, c.curved_max, c.flat_rms, c.flat_max, ratio
        );
    }

    // Diagnostics table.
    println!();
    println!(
        "{:<26} {:>10} {:>11} {:>11} {:>11} {:>9} {:>8}",
        "case", "max|w-1|", "curvatureDev", "sample|bot|", "C0 gap", "normalRes", "blowups"
    );
    println!("{}", "-".repeat(26 + 10 + 11 * 3 + 9 + 8 + 6));
    for c in &cases {
        println!(
            "{:<26} {:>10.3} {:>11.3e} {:>11.3e} {:>11.3e} {:>9.2e} {:>8}",
            c.name, c.max_weight_dev, c.curvature_dev, c.min_bot, c.c0_gap, c.relative_normal_residual, c.blowups
        );
    }
    println!();

    // Plain-language verdict lines.
    for c in &cases {
        let curved = if c.max_weight_dev > 1e-6 { "genuinely curved" } else { "COLLAPSED TO FLAT" };
        let watertight = if c.c0_gap < 1e-6 { "watertight" } else { "CRACKED" };
        let better = if c.flat_rms > c.curved_rms * 1.5 {
            format!("{:.1}x better than flat", c.flat_rms / c.curved_rms.max(1e-30))
        } else if c.curved_rms < c.flat_rms {
            "marginally better than flat".to_string()
        } else {
            "NOT better than flat".to_string()
        };
        println!("  {:<26} {} | {} | {}", c.name, curved, watertight, better);
    }
    println!();

    tikhonov_sweep();
}

/// Everything measured for one input.
struct CaseResult {
    name: String,
    n_patches: usize,
    curved_rms: f64,
    curved_max: f64,
    flat_rms: f64,
    flat_max: f64,
    max_weight_dev: f64,
    curvature_dev: f64,
    c0_gap: f64,
    relative_normal_residual: f64,
    blowups: usize,
    min_bot: f64,
}

fn run_case(
    name: &str,
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    samples: &[Sample],
    cfg: &LinearFitConfig,
) -> CaseResult {
    let res = linear_fit::linear_global_fit_full(coarse_pos, coarse_faces, samples, cfg)
        .expect("shared QB fixture fit");
    let flat = flat_patches(coarse_pos, coarse_faces);

    let scale = geometry::bounding_radius(coarse_pos).max(1e-9);
    let (curved_rms, curved_max, blowups) = sample_error(&res.patches, samples, scale);
    let (flat_rms, flat_max, _) = sample_error(&flat, samples, scale);

    CaseResult {
        name: name.to_string(),
        n_patches: coarse_faces.len(),
        curved_rms,
        curved_max,
        flat_rms,
        flat_max,
        max_weight_dev: res.max_weight_dev,
        curvature_dev: curvature_deviation(&res.patches, &flat),
        c0_gap: max_c0_edge_gap(&res.patches, coarse_faces, 16),
        relative_normal_residual: res.relative_normal_residual,
        blowups,
        min_bot: min_denominator(&res.patches, samples),
    }
}

/// Minimum |bot(λ)| = |Σ λᵥ wᵥ| over all samples for the fitted patches. The QB
/// map is `top·bot⁻¹`; as this approaches 0 the patch runs off to infinity, so a
/// small value here is the mechanism behind curved-worse-than-flat.
fn min_denominator(patches: &[QBTriPatch], samples: &[Sample]) -> f64 {
    let mut min = f64::INFINITY;
    for s in samples {
        let p = &patches[s.face_index];
        let bot = s.bary[0] * p.weights[0] + s.bary[1] * p.weights[1] + s.bary[2] * p.weights[2];
        min = min.min(bot.norm());
    }
    min
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// Favorable canary: a sphere built by inverting a flat octahedron is *exactly*
/// a set of QB patches with one globally-consistent shared weight per vertex.
/// The linear solve should recover it essentially to machine precision.
fn sphere_ground_truth_case() -> CaseResult {
    let z = 2.0;
    let verts = [
        [1.0, 0.0, z], [-1.0, 0.0, z],
        [0.0, 1.0, z], [0.0, -1.0, z],
        [0.0, 0.0, z + 1.0], [0.0, 0.0, z - 1.0],
    ];
    let faces_arr: [[usize; 3]; 8] = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ];
    let inv = Mobius::inversion();
    let coarse_pos: Vec<[f64; 3]> = verts
        .iter()
        .map(|v| inv.apply(Quat::from_point(v[0], v[1], v[2])).to_point())
        .collect();
    let coarse_faces: Vec<[usize; 3]> = faces_arr.to_vec();

    let mut samples = Vec::new();
    for (fi, f) in faces_arr.iter().enumerate() {
        let flat = QBTriPatch::flat(verts[f[0]], verts[f[1]], verts[f[2]]);
        let patch = flat.transform(&inv);
        let tess = quilting_remesh::roundtrip::tessellate_patch(&patch, 5);
        for (k, bary) in tess.bary.iter().enumerate() {
            samples.push(Sample { face_index: fi, bary: *bary, target: tess.positions[k] });
        }
    }
    // Tiny Tikhonov to show the near-exactness the canary is capable of.
    let cfg = LinearFitConfig {
        tikhonov: 1e-12,
        ..LinearFitConfig::default()
    };
    run_case("sphere (QB ground truth)", &coarse_pos, &coarse_faces, &samples, &cfg)
}

/// Realistic sphere: dense icosphere, QEM-simplified, samples by projecting the
/// dense vertices onto the coarse faces (noisy correspondence).
fn sphere_qem_inputs() -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<Sample>) {
    let (dense_pos, dense_faces) = test_shapes::sphere(3); // 1280 faces
    let (coarse_pos, coarse_faces) = simplify::simplify(&dense_pos, &dense_faces, 80);
    let samples = linear_fit::collect_samples(&coarse_pos, &coarse_faces, &dense_pos, 0.05);
    (coarse_pos, coarse_faces, samples)
}

fn sphere_qem_case() -> CaseResult {
    let (coarse_pos, coarse_faces, samples) = sphere_qem_inputs();
    run_case("sphere (icosphere→QEM)", &coarse_pos, &coarse_faces, &samples, &LinearFitConfig::default())
}

/// Does more regularization rescue the noisy case? Sweep Tikhonov τ and watch
/// curved RMS relative to the flat baseline. Large τ just pulls the fit back to
/// flat — it never gets meaningfully below the flat error.
fn tikhonov_sweep() {
    let (coarse_pos, coarse_faces, samples) = sphere_qem_inputs();
    let flat = flat_patches(&coarse_pos, &coarse_faces);
    let scale = geometry::bounding_radius(&coarse_pos).max(1e-9);
    let (flat_rms, _, _) = sample_error(&flat, &samples, scale);

    println!("Tikhonov sweep on sphere (icosphere→QEM)   flat RMS = {flat_rms:.3e}");
    println!("{:>12} {:>12} {:>12} {:>12}", "tau", "curvedRMS", "min|bot|", "max|w-1|");
    for &tau in &[1e-12, 1e-8, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0] {
        let cfg = LinearFitConfig {
            tikhonov: tau,
            ..LinearFitConfig::default()
        };
        let res = linear_fit::linear_global_fit_full(&coarse_pos, &coarse_faces, &samples, &cfg)
            .expect("shared QB sweep fit");
        let (rms, _, _) = sample_error(&res.patches, &samples, scale);
        let mb = min_denominator(&res.patches, &samples);
        println!("{:>12.1e} {:>12.3e} {:>12.3e} {:>12.3e}", tau, rms, mb, res.max_weight_dev);
    }
    println!();
}

/// Anisotropic curvature: a closed cylinder, QEM-simplified.
fn cylinder_qem_case() -> CaseResult {
    let (dense_pos, dense_faces) = test_shapes::cylinder(40, 12, 2.0, 1.0);
    let (coarse_pos, coarse_faces) = simplify::simplify(&dense_pos, &dense_faces, 90);
    let samples = linear_fit::collect_samples(&coarse_pos, &coarse_faces, &dense_pos, 0.05);
    run_case("cylinder (→QEM)", &coarse_pos, &coarse_faces, &samples, &LinearFitConfig::default())
}

/// Saddle / doubly-curved: a torus, QEM-simplified.
fn torus_qem_case() -> CaseResult {
    let (dense_pos, dense_faces) = torus(48, 24, 1.0, 0.35);
    let (coarse_pos, coarse_faces) = simplify::simplify(&dense_pos, &dense_faces, 140);
    let samples = linear_fit::collect_samples(&coarse_pos, &coarse_faces, &dense_pos, 0.05);
    run_case("torus (→QEM)", &coarse_pos, &coarse_faces, &samples, &LinearFitConfig::default())
}

// ---------------------------------------------------------------------------
// Shape generators / metrics
// ---------------------------------------------------------------------------

/// Triangle torus in the XZ plane, tube around Y. `major`/`minor` radii.
fn torus(nu: usize, nv: usize, major: f64, minor: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut pos = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        let u = 2.0 * std::f64::consts::PI * (i as f64 / nu as f64);
        for j in 0..nv {
            let v = 2.0 * std::f64::consts::PI * (j as f64 / nv as f64);
            let r = major + minor * v.cos();
            pos.push([r * u.cos(), minor * v.sin(), r * u.sin()]);
        }
    }
    let idx = |i: usize, j: usize| -> usize { (i % nu) * nv + (j % nv) };
    let mut faces = Vec::with_capacity(nu * nv * 2);
    for i in 0..nu {
        for j in 0..nv {
            let a = idx(i, j);
            let b = idx(i + 1, j);
            let c = idx(i + 1, j + 1);
            let d = idx(i, j + 1);
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
    (pos, faces)
}

fn flat_patches(pos: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<QBTriPatch> {
    faces.iter().map(|f| QBTriPatch::flat(pos[f[0]], pos[f[1]], pos[f[2]])).collect()
}

/// RMS + max error of each patch evaluated at its samples' barycentric coords,
/// against the target positions. Also counts blow-ups (non-finite or a point
/// flung more than 10× the model radius from its target — the rational map's
/// denominator went through zero).
fn sample_error(patches: &[QBTriPatch], samples: &[Sample], scale: f64) -> (f64, f64, usize) {
    let mut sum = 0.0;
    let mut max = 0.0f64;
    let mut blowups = 0;
    let mut n = 0;
    for s in samples {
        let p = patches[s.face_index].eval(s.bary[1], s.bary[2]).to_point();
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            blowups += 1;
            continue;
        }
        let d = ((p[0] - s.target[0]).powi(2)
            + (p[1] - s.target[1]).powi(2)
            + (p[2] - s.target[2]).powi(2))
        .sqrt();
        if d > 10.0 * scale {
            blowups += 1;
        }
        sum += d * d;
        max = max.max(d);
        n += 1;
    }
    ((sum / n.max(1) as f64).sqrt(), max, blowups)
}

/// Max over patches of |curved(centroid) − flat(centroid)|. Nonzero proves the
/// patches genuinely bulge off the chord plane.
fn curvature_deviation(curved: &[QBTriPatch], flat: &[QBTriPatch]) -> f64 {
    let mut max = 0.0f64;
    for (cp, fp) in curved.iter().zip(flat) {
        let a = cp.eval(1.0 / 3.0, 1.0 / 3.0).to_point();
        let b = fp.eval(1.0 / 3.0, 1.0 / 3.0).to_point();
        let d =
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        max = max.max(d);
    }
    max
}

/// Sample each shared edge from BOTH adjacent patches and return the max
/// position difference. By construction (shared endpoint weights) ≈ 0.
fn max_c0_edge_gap(patches: &[QBTriPatch], faces: &[[usize; 3]], t_steps: usize) -> f64 {
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for i in 0..3 {
            let a = f[i];
            let b = f[(i + 1) % 3];
            edge_faces.entry((a.min(b), a.max(b))).or_default().push(fi);
        }
    }
    let mut max_gap = 0.0f64;
    for (&(g0, g1), fs) in &edge_faces {
        if fs.len() < 2 {
            continue;
        }
        for step in 0..=t_steps {
            let t = step as f64 / t_steps as f64;
            let p0 = eval_edge(&patches[fs[0]], &faces[fs[0]], g0, g1, t);
            let p1 = eval_edge(&patches[fs[1]], &faces[fs[1]], g0, g1, t);
            let d = ((p0[0] - p1[0]).powi(2) + (p0[1] - p1[1]).powi(2) + (p0[2] - p1[2]).powi(2))
                .sqrt();
            max_gap = max_gap.max(d);
        }
    }
    max_gap
}

fn eval_edge(patch: &QBTriPatch, face: &[usize; 3], g0: usize, g1: usize, t: f64) -> [f64; 3] {
    let l0 = face.iter().position(|&v| v == g0).unwrap();
    let l1 = face.iter().position(|&v| v == g1).unwrap();
    let mut bary = [0.0f64; 3];
    bary[l0] = 1.0 - t;
    bary[l1] = t;
    patch.eval(bary[1], bary[2]).to_point()
}
