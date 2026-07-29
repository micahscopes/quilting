//! Direct linear global fit of shared per-vertex QB weights.
//!
//! The other curved fitters in this crate ([`crate::c_estimator`],
//! [`crate::global_fit`]) treat weight fitting as a nonlinear least-squares
//! problem and lean on Gauss-Newton. This module rests on a sharper
//! observation: the *algebraic* patch residual is **linear** in the weights, so
//! one dense linear solve recovers a globally-consistent, C0-watertight weight
//! per vertex — no iteration, no initialization sensitivity.
//!
//! ## The residual is linear in the weights
//!
//! A QB triangle patch evaluates to `X(λ) = Im[ top(λ) · bot(λ)⁻¹ ]` with
//! `top(λ) = Σ λᵢ pᵢ wᵢ` and `bot(λ) = Σ λᵢ wᵢ`. For a sample point `Xₖ` known
//! to sit at barycentric `λₖ` on the face `(a,b,c)`, define the algebraic
//! residual
//!
//! ```text
//!   rₖ = top(λₖ) − Xₖ · bot(λₖ)
//!      = Σ_{v∈{a,b,c}} λ_{k,v} (pᵥ − Xₖ) wᵥ .
//! ```
//!
//! `rₖ = 0` is *exactly* `X(λₖ) = Xₖ` (multiply on the right by `bot⁻¹`). Because
//! quaternion left-multiplication `q · w` is a linear map on the four real
//! components of `w` — the 4×4 matrix [`left_mul_matrix`] — `rₖ` is linear in the
//! stacked vector of all `wᵥ`. Stacking every sample gives an overdetermined
//! homogeneous system `A w = 0`.
//!
//! ## Killing the trivial and gauge nullspaces
//!
//! `w = 0` solves `A w = 0`, and so does any right-multiplication `wᵥ ↦ wᵥ·s`
//! (both `top` and `bot` pick up the same right factor `s`, which cancels in
//! `top·bot⁻¹`). Both are removed by **pinning** one vertex weight to
//! `Quat::ONE`: its contribution moves to the right-hand side `b`, leaving a
//! `4·(V−1)` unknown system `A_free w_free = b`. A small Tikhonov pull of every
//! free weight toward `ONE` regularizes vertices that are starved of samples and
//! makes the normal matrix `AᵀA` symmetric positive-definite. One dense solve
//! finishes it.
//!
//! Sharing one weight per vertex is what makes the result watertight: two faces
//! meeting on an edge share that edge's two endpoint weights, and a QB edge curve
//! depends only on its two endpoints, so the boundary curves coincide to machine
//! precision (see [`crate::linear_fit`] tests and the `curved_vs_flat` example).

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::Quat;

/// A target sample: a 3D point known to lie at barycentric `bary` on face
/// `face_index` of the coarse mesh.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Index into the coarse `faces` array.
    pub face_index: usize,
    /// Barycentric coordinates `[λ₀, λ₁, λ₂]` matching the face's vertex order.
    pub bary: [f64; 3],
    /// Target 3D position the patch should pass through at `bary`.
    pub target: [f64; 3],
}

/// Tuning for [`linear_global_fit_full`].
#[derive(Debug, Clone, Copy)]
pub struct LinearFitConfig {
    /// Tikhonov weight pulling every free vertex weight toward `Quat::ONE`.
    /// Kept tiny so it only breaks ties / stabilizes under-sampled vertices
    /// rather than biasing the fit toward flat.
    pub tikhonov: f64,
    /// Iterations for the power / inverse-power condition-number estimate.
    /// Zero skips the estimate (reports `condition_number = 0.0`).
    pub cond_iters: usize,
}

impl Default for LinearFitConfig {
    fn default() -> Self {
        Self { tikhonov: 1e-8, cond_iters: 200 }
    }
}

/// Full result of the linear global fit, including diagnostics.
pub struct LinearFitResult {
    /// One fitted QB patch per coarse face.
    pub patches: Vec<QBTriPatch>,
    /// One solved quaternion weight per coarse vertex (vertex 0 is pinned to `ONE`).
    pub vertex_weights: Vec<Quat>,
    /// `max_v |wᵥ − ONE|`. If ≈ 0 the solve collapsed to flat and the "curved"
    /// result is a lie; a healthy curved fit reports a clearly nonzero value.
    pub max_weight_dev: f64,
    /// Estimated condition number of the normal matrix `AᵀA` (λmax / λmin).
    /// `0.0` if the estimate was skipped or failed.
    pub condition_number: f64,
    /// Number of free unknowns (`4·(V−1)`).
    pub dim: usize,
}

/// Convenience wrapper matching the assessment's signature: fit and return just
/// the patches. Uses [`LinearFitConfig::default`].
pub fn linear_global_fit(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    samples: &[Sample],
) -> Vec<QBTriPatch> {
    linear_global_fit_full(coarse_pos, coarse_faces, samples, &LinearFitConfig::default()).patches
}

/// The full fit. See the module docs for the derivation.
pub fn linear_global_fit_full(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    samples: &[Sample],
    config: &LinearFitConfig,
) -> LinearFitResult {
    let v = coarse_pos.len();
    // Vertex 0 is pinned to ONE; the remaining V-1 vertices are free (4 real
    // unknowns each). free_index maps a global vertex to its reduced block.
    let free = |vtx: usize| -> Option<usize> { if vtx == 0 { None } else { Some(vtx - 1) } };
    let dim = 4 * v.saturating_sub(1);

    // Normal equations AᵀA x = Aᵀb, accumulated block-by-block. AtA is dense
    // row-major (dim is small: 4·(V-1), tens–low hundreds for coarse meshes).
    let mut ata = vec![0.0f64; dim * dim];
    let mut atb = vec![0.0f64; dim];

    for s in samples {
        let face = coarse_faces[s.face_index];
        let xk = Quat::from_point(s.target[0], s.target[1], s.target[2]);

        // Per-vertex 4×4 blocks Bᵥ = λ_{k,v} · L(pᵥ − Xₖ).
        let mut blocks: [[[f64; 4]; 4]; 3] = [[[0.0; 4]; 4]; 3];
        for i in 0..3 {
            let vtx = face[i];
            let p = Quat::from_point(coarse_pos[vtx][0], coarse_pos[vtx][1], coarse_pos[vtx][2]);
            let l = left_mul_matrix(p - xk);
            let lam = s.bary[i];
            for r in 0..4 {
                for c in 0..4 {
                    blocks[i][r][c] = lam * l[r][c];
                }
            }
        }

        // Pinned contribution → right-hand side. w₀ = vec(ONE) = [1,0,0,0], so
        // B₀·vec(ONE) is just the first column of B₀. The equation Σ Bᵥ wᵥ = 0
        // becomes Σ_{free} Bᵥ wᵥ = d with d = −(pinned column).
        let mut d = [0.0f64; 4];
        for i in 0..3 {
            if free(face[i]).is_none() {
                for r in 0..4 {
                    d[r] -= blocks[i][r][0];
                }
            }
        }

        // Accumulate AᵀA and Aᵀb over the free vertices of this face.
        for i in 0..3 {
            let Some(ri) = free(face[i]) else { continue };
            let bi = &blocks[i];
            // Aᵀb block: Biᵀ · d
            for a in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += bi[k][a] * d[k];
                }
                atb[4 * ri + a] += acc;
            }
            for j in 0..3 {
                let Some(rj) = free(face[j]) else { continue };
                let bj = &blocks[j];
                // AᵀA block (ri,rj): Biᵀ · Bj
                for a in 0..4 {
                    for b in 0..4 {
                        let mut acc = 0.0;
                        for k in 0..4 {
                            acc += bi[k][a] * bj[k][b];
                        }
                        ata[(4 * ri + a) * dim + (4 * rj + b)] += acc;
                    }
                }
            }
        }
    }

    // Tikhonov: add τ·I to the diagonal, pull toward vec(ONE) = [1,0,0,0].
    let tau = config.tikhonov;
    for r in 0..(dim / 4) {
        for c in 0..4 {
            let idx = 4 * r + c;
            ata[idx * dim + idx] += tau;
        }
        atb[4 * r] += tau; // pull real part toward 1
    }

    // Solve.
    let lu = DenseLu::factor(ata.clone(), dim);
    let x = lu.solve(&atb);

    // Reconstruct per-vertex weights.
    let mut weights = vec![Quat::ONE; v];
    for vtx in 1..v {
        let r = vtx - 1;
        weights[vtx] = Quat::new(x[4 * r], x[4 * r + 1], x[4 * r + 2], x[4 * r + 3]);
    }

    // Diagnostic: how far the weights actually moved from flat.
    let max_weight_dev = weights.iter().map(|w| (*w - Quat::ONE).norm()).fold(0.0, f64::max);

    // Reconstruct patches.
    let patches = coarse_faces
        .iter()
        .map(|face| {
            let positions = [
                Quat::from_point(coarse_pos[face[0]][0], coarse_pos[face[0]][1], coarse_pos[face[0]][2]),
                Quat::from_point(coarse_pos[face[1]][0], coarse_pos[face[1]][1], coarse_pos[face[1]][2]),
                Quat::from_point(coarse_pos[face[2]][0], coarse_pos[face[2]][1], coarse_pos[face[2]][2]),
            ];
            QBTriPatch::new(positions, [weights[face[0]], weights[face[1]], weights[face[2]]])
        })
        .collect();

    let condition_number = if config.cond_iters > 0 && dim > 0 {
        estimate_condition(&ata, &lu, dim, config.cond_iters)
    } else {
        0.0
    };

    LinearFitResult { patches, vertex_weights: weights, max_weight_dev, condition_number, dim }
}

/// The 4×4 real matrix `L(q)` of quaternion **left**-multiplication: for any
/// quaternion `w`, `L(q) · vec(w) = vec(q · w)` where `vec(w) = [w, x, y, z]`.
///
/// This is the linchpin of the whole module — it is what makes the algebraic
/// residual linear in the weights.
pub fn left_mul_matrix(q: Quat) -> [[f64; 4]; 4] {
    // Derived from q·w = (qw+qxi+qyj+qzk)(ww+wxi+wyj+wzk):
    [
        [q.w, -q.x, -q.y, -q.z],
        [q.x,  q.w, -q.z,  q.y],
        [q.y,  q.z,  q.w, -q.x],
        [q.z, -q.y,  q.x,  q.w],
    ]
}

/// Dense LU factorization with partial pivoting, kept around so the same factors
/// serve both the main solve and inverse-power iteration.
struct DenseLu {
    lu: Vec<f64>,
    perm: Vec<usize>,
    n: usize,
    singular: bool,
}

impl DenseLu {
    fn factor(mut a: Vec<f64>, n: usize) -> Self {
        const EPS: f64 = 1e-300;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut singular = false;
        for col in 0..n {
            // partial pivot
            let mut max_row = col;
            let mut max_val = a[col * n + col].abs();
            for row in (col + 1)..n {
                let val = a[row * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < EPS {
                singular = true;
                continue;
            }
            if max_row != col {
                for k in 0..n {
                    a.swap(col * n + k, max_row * n + k);
                }
                perm.swap(col, max_row);
            }
            let pivot = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                a[row * n + col] = factor;
                for k in (col + 1)..n {
                    a[row * n + k] -= factor * a[col * n + k];
                }
            }
        }
        Self { lu: a, perm, n, singular }
    }

    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        // Apply row permutation.
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            y[i] = b[self.perm[i]];
        }
        // Forward substitution (unit lower).
        for i in 0..n {
            let mut sum = y[i];
            for k in 0..i {
                sum -= self.lu[i * n + k] * y[k];
            }
            y[i] = sum;
        }
        // Back substitution (upper).
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= self.lu[i * n + k] * y[k];
            }
            let d = self.lu[i * n + i];
            y[i] = if d.abs() > 1e-300 { sum / d } else { 0.0 };
        }
        y
    }
}

/// Standalone dense solve of `A x = b` (A row-major `n×n`). Public so callers /
/// tests can reuse it independently of the fitter.
pub fn solve_dense(a: Vec<f64>, b: &[f64], n: usize) -> Vec<f64> {
    DenseLu::factor(a, n).solve(b)
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Estimate `cond(AᵀA) = λmax / λmin` for the SPD normal matrix by power
/// iteration (λmax) and inverse-power iteration through the cached LU (λmin).
fn estimate_condition(ata: &[f64], lu: &DenseLu, n: usize, iters: usize) -> f64 {
    if lu.singular {
        return f64::INFINITY;
    }
    // Largest eigenvalue: power iteration.
    let mut x = vec![1.0f64; n];
    let mut lambda_max = 0.0;
    for _ in 0..iters {
        let y = matvec(ata, &x, n);
        let ny = norm(&y);
        if ny < 1e-300 {
            break;
        }
        lambda_max = y.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>()
            / x.iter().map(|a| a * a).sum::<f64>();
        for i in 0..n {
            x[i] = y[i] / ny;
        }
    }
    // Smallest eigenvalue: inverse-power iteration (solve AᵀA y = x).
    let mut z = vec![1.0f64; n];
    let nz = norm(&z);
    for i in 0..n {
        z[i] /= nz;
    }
    let mut lambda_min = 0.0;
    for _ in 0..iters {
        let y = lu.solve(&z);
        let ny = norm(&y);
        if ny < 1e-300 || !ny.is_finite() {
            break;
        }
        // Rayleigh quotient of AᵀA with the new iterate.
        let ay = matvec(ata, &y, n);
        lambda_min = y.iter().zip(&ay).map(|(a, b)| a * b).sum::<f64>()
            / y.iter().map(|a| a * a).sum::<f64>();
        for i in 0..n {
            z[i] = y[i] / ny;
        }
    }
    if lambda_min.abs() < 1e-300 {
        f64::INFINITY
    } else {
        lambda_max / lambda_min
    }
}

/// Reuse the crate's mesh-projection sampler to build [`Sample`]s: project each
/// original vertex onto the coarse faces it falls in and record the barycentric
/// coordinates and the original position as the target. Thin wrapper over
/// [`crate::global_fit`]'s `collect_face_samples` — the same correspondence the
/// production curved fitter uses.
pub fn collect_samples(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    orig_pos: &[[f64; 3]],
    margin: f64,
) -> Vec<Sample> {
    let dummy_normals = vec![[0.0f64; 3]; orig_pos.len()];
    let mut out = Vec::new();
    for (fi, face) in coarse_faces.iter().enumerate() {
        let tri = [coarse_pos[face[0]], coarse_pos[face[1]], coarse_pos[face[2]]];
        let fs = crate::global_fit::collect_face_samples(&tri, orig_pos, &dummy_normals, margin);
        for (k, bary) in fs.bary.iter().enumerate() {
            out.push(Sample {
                face_index: fi,
                bary: *bary,
                target: fs.positions[k],
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::quaternion::Mobius;

    /// The linearity claim the whole module rests on: `L(q)·vec(w) = vec(q·w)`.
    #[test]
    fn left_mul_matrix_matches_quaternion_product() {
        let cases = [
            (Quat::new(0.3, -1.2, 0.7, 2.1), Quat::new(-0.5, 0.9, 1.3, -0.2)),
            (Quat::from_point(1.0, -2.0, 0.5), Quat::ONE),
            (Quat::new(2.0, 0.0, -1.0, 0.4), Quat::new(0.0, 0.0, 0.0, 1.0)),
            (Quat::I, Quat::J),
        ];
        for (q, w) in cases {
            let l = left_mul_matrix(q);
            let wv = [w.w, w.x, w.y, w.z];
            let got = [
                l[0][0] * wv[0] + l[0][1] * wv[1] + l[0][2] * wv[2] + l[0][3] * wv[3],
                l[1][0] * wv[0] + l[1][1] * wv[1] + l[1][2] * wv[2] + l[1][3] * wv[3],
                l[2][0] * wv[0] + l[2][1] * wv[1] + l[2][2] * wv[2] + l[2][3] * wv[3],
                l[3][0] * wv[0] + l[3][1] * wv[1] + l[3][2] * wv[2] + l[3][3] * wv[3],
            ];
            let expect = q * w;
            let expect_v = [expect.w, expect.x, expect.y, expect.z];
            for i in 0..4 {
                assert!(
                    (got[i] - expect_v[i]).abs() < 1e-12,
                    "L(q)·vec(w) mismatch: {:?} vs {:?}",
                    got,
                    expect_v
                );
            }
        }
    }

    #[test]
    fn solve_dense_reproduces_rhs() {
        // 3×3 SPD system.
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let b = [1.0, -2.0, 3.0];
        let x = solve_dense(a.clone(), &b, 3);
        for i in 0..3 {
            let axi: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!((axi - b[i]).abs() < 1e-10, "row {i}: {axi} vs {}", b[i]);
        }
    }

    /// Build coarse mesh + samples from a known curved QB ground truth (a sphere
    /// obtained by inverting a flat octahedron), fit, and check we recover it to
    /// machine precision — this is only possible because a Möbius image of a
    /// plane admits an *exact* globally-consistent shared-weight assignment.
    fn octahedron_sphere() -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<Sample>, Vec<QBTriPatch>) {
        let z = 2.0;
        let verts = [
            [1.0, 0.0, z], [-1.0, 0.0, z],
            [0.0, 1.0, z], [0.0, -1.0, z],
            [0.0, 0.0, z + 1.0], [0.0, 0.0, z - 1.0],
        ];
        let faces_arr = [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ];
        let inv = Mobius::inversion();
        // Coarse mesh = images of the octahedron vertices (shared under one map).
        let coarse_pos: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| inv.apply(Quat::from_point(v[0], v[1], v[2])).to_point())
            .collect();
        let coarse_faces: Vec<[usize; 3]> = faces_arr.iter().map(|f| [f[0], f[1], f[2]]).collect();

        // Ground-truth patches and samples from tessellating them (exact bary).
        let mut truth = Vec::new();
        let mut samples = Vec::new();
        for (fi, f) in faces_arr.iter().enumerate() {
            let flat = QBTriPatch::flat(verts[f[0]], verts[f[1]], verts[f[2]]);
            let patch = flat.transform(&inv);
            truth.push(patch);
            let tess = crate::roundtrip::tessellate_patch(&patch, 4);
            for (k, bary) in tess.bary.iter().enumerate() {
                samples.push(Sample { face_index: fi, bary: *bary, target: tess.positions[k] });
            }
        }
        (coarse_pos, coarse_faces, samples, truth)
    }

    #[test]
    fn sphere_ground_truth_recovered_near_exactly() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let cfg = LinearFitConfig { tikhonov: 1e-12, cond_iters: 0 };
        let res = linear_global_fit_full(&coarse_pos, &coarse_faces, &samples, &cfg);

        // Residual: patch at each sample bary should hit the target.
        let mut max_err = 0.0f64;
        for s in &samples {
            let p = res.patches[s.face_index].eval(s.bary[1], s.bary[2]).to_point();
            let d = ((p[0] - s.target[0]).powi(2)
                + (p[1] - s.target[1]).powi(2)
                + (p[2] - s.target[2]).powi(2))
            .sqrt();
            max_err = max_err.max(d);
        }
        assert!(max_err < 1e-6, "sphere ground truth not recovered: max_err={max_err:e}");
        // And the fit must be genuinely curved, not a collapse to flat.
        assert!(
            res.max_weight_dev > 0.1,
            "weights barely moved ({}), fit collapsed to flat",
            res.max_weight_dev
        );
    }

    #[test]
    fn sphere_curved_beats_flat_by_large_margin() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let res = linear_global_fit_full(
            &coarse_pos,
            &coarse_faces,
            &samples,
            &LinearFitConfig::default(),
        );
        let curved = sample_rms(&res.patches, &coarse_faces, &samples);
        let flat_patches = flat_patches(&coarse_pos, &coarse_faces);
        let flat = sample_rms(&flat_patches, &coarse_faces, &samples);
        assert!(
            flat > curved * 50.0,
            "curved RMS {curved:e} should be ≫ better than flat RMS {flat:e}"
        );
    }

    #[test]
    fn c0_edge_gap_is_machine_epsilon() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let res = linear_global_fit_full(
            &coarse_pos,
            &coarse_faces,
            &samples,
            &LinearFitConfig::default(),
        );
        let gap = max_c0_edge_gap(&res.patches, &coarse_faces, 12);
        assert!(gap < 1e-9, "shared-weight patches should be C0 watertight, gap={gap:e}");
    }

    // --- small test helpers (mirrored, simply, in the example) ---

    fn flat_patches(pos: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<QBTriPatch> {
        faces
            .iter()
            .map(|f| QBTriPatch::flat(pos[f[0]], pos[f[1]], pos[f[2]]))
            .collect()
    }

    fn sample_rms(patches: &[QBTriPatch], _faces: &[[usize; 3]], samples: &[Sample]) -> f64 {
        let mut sum = 0.0;
        for s in samples {
            let p = patches[s.face_index].eval(s.bary[1], s.bary[2]).to_point();
            sum += (p[0] - s.target[0]).powi(2)
                + (p[1] - s.target[1]).powi(2)
                + (p[2] - s.target[2]).powi(2);
        }
        (sum / samples.len().max(1) as f64).sqrt()
    }

    fn max_c0_edge_gap(patches: &[QBTriPatch], faces: &[[usize; 3]], t_steps: usize) -> f64 {
        use std::collections::HashMap;
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
            let f0 = fs[0];
            let f1 = fs[1];
            for step in 0..=t_steps {
                let t = step as f64 / t_steps as f64;
                let p0 = eval_edge(&patches[f0], &faces[f0], g0, g1, t);
                let p1 = eval_edge(&patches[f1], &faces[f1], g0, g1, t);
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
}
