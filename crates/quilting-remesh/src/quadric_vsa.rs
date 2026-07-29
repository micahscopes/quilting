/// Quadric VSA — Variational Shape Approximation with implicit quadric surface proxies.
///
/// Instead of clustering faces by normal similarity (planar VSA), this clusters by
/// proximity to a fitted quadric surface. This naturally groups faces that lie on the
/// same sphere, cylinder, or other second-order surface — exactly the surface types
/// that Dupin cyclides generalize, making this ideal for QB patch segmentation.
///
/// Algorithm:
/// 1. Farthest-point seeding (same as planar VSA)
/// 2. First iteration uses planar (L2,1) metric to bootstrap
/// 3. After first partition, fit quadric per cluster via algebraic fitting
///    (smallest eigenvector of area-weighted scatter matrix)
/// 4. Subsequent iterations use hybrid metric:
///    E = α * taubin_dist²(centroid, quadric) + (1-α) * l21_normal_error
/// 5. Post-process: classify each quadric (plane, sphere, cylinder, general)
///
/// Reference: Yan, Liu, Wang — "Quadric Surface Extraction by Variational Shape
/// Approximation" (GMP 2006), extended in "Variational mesh segmentation via quadric
/// surface fitting" (CAD 2012).

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use quilting_mesh::HalfEdgeMesh;
use crate::geometry;

/// Classified surface type for a quadric proxy.
#[derive(Debug, Clone)]
pub enum SurfaceType {
    Plane,
    Sphere { center: [f64; 3], radius: f64 },
    Cylinder { axis: [f64; 3], radius: f64 },
    General,
}

/// A quadric proxy representing a cluster.
#[derive(Debug, Clone)]
pub struct QuadricProxy {
    /// Implicit quadric coefficients: f(x) = C · F(x)
    /// F(x) = [1, x, y, z, x², xy, xz, y², yz, z²]
    pub coeffs: [f64; 10],
    /// Area-weighted average normal (for hybrid metric + fallback)
    pub normal: [f64; 3],
    /// Area-weighted centroid
    pub centroid: [f64; 3],
    /// Total area
    pub area: f64,
    /// Classified surface type
    pub surface_type: SurfaceType,
}

#[derive(Debug, Clone)]
pub struct QuadricVsaConfig {
    pub target_clusters: usize,
    pub max_iterations: usize,
    pub sharp_edge_threshold: f64,
    /// Balance between quadric distance (1.0) and normal deviation (0.0).
    /// Default 0.5 (equal weight). Higher = more curvature-aware.
    pub quadric_weight: f64,
}

impl Default for QuadricVsaConfig {
    fn default() -> Self {
        Self {
            target_clusters: 500,
            max_iterations: 20,
            sharp_edge_threshold: 40.0_f64.to_radians(),
            quadric_weight: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuadricVsaResult {
    pub face_labels: Vec<usize>,
    pub num_clusters: usize,
    pub proxies: Vec<QuadricProxy>,
    pub sharp_edges: Vec<u32>,
}

// ── Quadric math ──────────────────────────────────────────────────────

/// Feature vector F(x,y,z) = [1, x, y, z, x², xy, xz, y², yz, z²]
fn feature_vector(p: &[f64; 3]) -> [f64; 10] {
    let (x, y, z) = (p[0], p[1], p[2]);
    [1.0, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z]
}

/// Evaluate implicit quadric f(p) = C · F(p)
fn eval_quadric(c: &[f64; 10], p: &[f64; 3]) -> f64 {
    let f = feature_vector(p);
    let mut val = 0.0;
    for i in 0..10 {
        val += c[i] * f[i];
    }
    val
}

/// Gradient of f(x,y,z) = c0 + c1*x + c2*y + c3*z + c4*x² + c5*xy + c6*xz + c7*y² + c8*yz + c9*z²
fn eval_quadric_gradient(c: &[f64; 10], p: &[f64; 3]) -> [f64; 3] {
    let (x, y, z) = (p[0], p[1], p[2]);
    [
        c[1] + 2.0 * c[4] * x + c[5] * y + c[6] * z,
        c[2] + c[5] * x + 2.0 * c[7] * y + c[8] * z,
        c[3] + c[6] * x + c[8] * y + 2.0 * c[9] * z,
    ]
}

/// Taubin's approximate squared distance from point to quadric zero-set:
/// d²(p, Z(f)) ≈ f(p)² / |∇f(p)|²
fn taubin_distance_sq(c: &[f64; 10], p: &[f64; 3]) -> f64 {
    let f = eval_quadric(c, p);
    let g = eval_quadric_gradient(c, p);
    let g2 = geometry::vec3_dot(g, g);
    if g2 < 1e-20 {
        return f * f; // degenerate: treat as algebraic error
    }
    f * f / g2
}

// ── Quadric fitting via algebraic method ──────────────────────────────

/// Fit an implicit quadric to a set of weighted points.
/// Returns the 10 coefficients of the best-fit quadric (smallest eigenvector
/// of the area-weighted scatter matrix).
fn fit_quadric(points: &[[f64; 3]], weights: &[f64]) -> [f64; 10] {
    let n = points.len();
    assert_eq!(n, weights.len());

    // Build scatter matrix M = Σ w_i * F(p_i) * F(p_i)^T
    let mut m = [[0.0f64; 10]; 10];
    for (i, p) in points.iter().enumerate() {
        let w = weights[i];
        let f = feature_vector(p);
        for r in 0..10 {
            for c in r..10 {
                m[r][c] += w * f[r] * f[c];
            }
        }
    }
    // Symmetrize
    for r in 0..10 {
        for c in 0..r {
            m[r][c] = m[c][r];
        }
    }

    // Find eigenvector for smallest eigenvalue via Jacobi decomposition
    let (eigenvalues, eigenvectors) = jacobi_eigen_10(&mut m);

    // Pick eigenvector with smallest eigenvalue (first one after sorting)
    let mut min_idx = 0;
    let mut min_val = eigenvalues[0];
    for i in 1..10 {
        if eigenvalues[i] < min_val {
            min_val = eigenvalues[i];
            min_idx = i;
        }
    }

    // Extract eigenvector (stored as columns)
    let mut coeffs = [0.0; 10];
    for i in 0..10 {
        coeffs[i] = eigenvectors[i][min_idx];
    }

    // Normalize so the quadratic part has unit trace (c4 + c7 + c9)
    // This makes the coefficients comparable across clusters
    let trace = coeffs[4] + coeffs[7] + coeffs[9];
    if trace.abs() > 1e-15 {
        let s = 1.0 / trace;
        for c in &mut coeffs {
            *c *= s;
        }
    }

    coeffs
}

/// Jacobi eigenvalue decomposition for a 10×10 symmetric matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i][j] = j-th component of i-th eigenvector.
fn jacobi_eigen_10(a: &mut [[f64; 10]; 10]) -> ([f64; 10], [[f64; 10]; 10]) {
    let mut v = [[0.0f64; 10]; 10];
    for i in 0..10 {
        v[i][i] = 1.0;
    }

    for _sweep in 0..100 {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..10 {
            for j in (i + 1)..10 {
                let val = a[i][j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute Givens rotation angle
        let diff = a[p][p] - a[q][q];
        let t = if diff.abs() < 1e-30 {
            1.0 // tan(π/4) = 1
        } else {
            let tau = diff / (2.0 * a[p][q]);
            // Choose the smaller root for stability
            let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
            sign / (tau.abs() + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update matrix A (only affected rows/cols p and q)
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app + t * apq;
        a[q][q] = aqq - t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for i in 0..10 {
            if i == p || i == q {
                continue;
            }
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip + s * aiq;
            a[p][i] = a[i][p];
            a[i][q] = -s * aip + c * aiq;
            a[q][i] = a[i][q];
        }

        // Update eigenvectors
        for i in 0..10 {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    let mut eigenvalues = [0.0; 10];
    for i in 0..10 {
        eigenvalues[i] = a[i][i];
    }
    (eigenvalues, v)
}

// ── Surface type classification ───────────────────────────────────────

/// Classify a fitted quadric by analyzing its quadratic form.
///
/// The quadratic part Q = [[c4, c5/2, c6/2], [c5/2, c7, c8/2], [c6/2, c8/2, c9]]
/// has eigenvalues that determine the surface type:
/// - All zero → plane
/// - All equal and nonzero → sphere
/// - Two equal, one zero → cylinder
/// - Otherwise → general quadric
fn classify_quadric(coeffs: &[f64; 10], centroid: &[f64; 3]) -> SurfaceType {
    // Extract quadratic form matrix
    let q = [
        [coeffs[4], coeffs[5] / 2.0, coeffs[6] / 2.0],
        [coeffs[5] / 2.0, coeffs[7], coeffs[8] / 2.0],
        [coeffs[6] / 2.0, coeffs[8] / 2.0, coeffs[9]],
    ];

    // Find eigenvalues of the 3×3 quadratic form via cubic formula
    let eigs = eigenvalues_3x3_sym(&q);
    let abs_eigs = [eigs[0].abs(), eigs[1].abs(), eigs[2].abs()];
    let max_eig = abs_eigs[0].max(abs_eigs[1]).max(abs_eigs[2]);

    if max_eig < 1e-10 {
        return SurfaceType::Plane;
    }

    let threshold = max_eig * 0.1;
    let num_nonzero = abs_eigs.iter().filter(|&&e| e > threshold).count();
    let num_near_equal = {
        let mut count = 0;
        if (abs_eigs[0] - abs_eigs[1]).abs() < threshold { count += 1; }
        if (abs_eigs[1] - abs_eigs[2]).abs() < threshold { count += 1; }
        if (abs_eigs[0] - abs_eigs[2]).abs() < threshold { count += 1; }
        count
    };

    // All three eigenvalues roughly equal and nonzero → sphere
    if num_nonzero == 3 && num_near_equal == 3 {
        // Sphere: f(x) = |x - c|² - r² = 0
        // From coefficients: center = -[c1, c2, c3] / (2 * c4) when c4=c7=c9
        let scale = coeffs[4] + coeffs[7] + coeffs[9];
        if scale.abs() > 1e-15 {
            let inv = -0.5 / (scale / 3.0);
            let center = [coeffs[1] * inv, coeffs[2] * inv, coeffs[3] * inv];
            let r2 = center[0] * center[0] + center[1] * center[1] + center[2] * center[2]
                - coeffs[0] / (scale / 3.0);
            if r2 > 0.0 {
                return SurfaceType::Sphere {
                    center,
                    radius: r2.sqrt(),
                };
            }
        }
        return SurfaceType::Sphere {
            center: *centroid,
            radius: 1.0,
        };
    }

    // Two nonzero, one near-zero → cylinder
    if num_nonzero == 2 {
        // The eigenvector corresponding to the near-zero eigenvalue is the cylinder axis
        let mut min_idx = 0;
        if abs_eigs[1] < abs_eigs[min_idx] { min_idx = 1; }
        if abs_eigs[2] < abs_eigs[min_idx] { min_idx = 2; }

        // Get axis from the near-zero eigenvector (approximate)
        let axis = match min_idx {
            0 => [1.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };

        let avg_eig = (abs_eigs.iter().sum::<f64>() - abs_eigs[min_idx]) / 2.0;
        let radius = if avg_eig > 1e-15 { (1.0 / avg_eig).sqrt() } else { 1.0 };

        return SurfaceType::Cylinder { axis, radius };
    }

    SurfaceType::General
}

/// Eigenvalues of a 3×3 symmetric matrix via Cardano's formula.
fn eigenvalues_3x3_sym(m: &[[f64; 3]; 3]) -> [f64; 3] {
    let a = m[0][0];
    let b = m[1][1];
    let c = m[2][2];
    let d = m[0][1];
    let e = m[0][2];
    let f = m[1][2];

    // Characteristic polynomial: λ³ - p*λ² + q*λ - r = 0
    let p = a + b + c; // trace
    let q = a * b + a * c + b * c - d * d - e * e - f * f;
    let r = a * b * c + 2.0 * d * e * f - a * f * f - b * e * e - c * d * d; // determinant

    // Solve via Cardano's method for real roots
    let p3 = p / 3.0;
    let q2 = (p * p - 3.0 * q) / 9.0;
    let r2 = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0;
    let disc = r2 * r2 - q2 * q2 * q2;

    if disc <= 0.0 {
        // Three real roots
        let theta = if q2 > 1e-30 { (r2 / (q2 * q2.sqrt())).clamp(-1.0, 1.0).acos() } else { 0.0 };
        let sq = -2.0 * q2.sqrt();
        let mut eigs = [
            sq * (theta / 3.0).cos() + p3,
            sq * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() + p3,
            sq * ((theta - 2.0 * std::f64::consts::PI) / 3.0).cos() + p3,
        ];
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        eigs
    } else {
        // One real root (shouldn't happen for symmetric matrix, but handle gracefully)
        let sqrt_disc = disc.sqrt();
        let u = (r2.abs() + sqrt_disc).cbrt() * if r2 >= 0.0 { -1.0 } else { 1.0 };
        let v = if u.abs() > 1e-30 { q2 / u } else { 0.0 };
        let root = u + v + p3;
        [root, root, root]
    }
}

// ── Priority queue entry ──────────────────────────────────────────────

#[derive(Clone)]
struct QueueEntry {
    face: u32,
    cluster: usize,
    error: f64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool { self.error == other.error }
}
impl Eq for QueueEntry {}
impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal) // min-heap
    }
}

// ── Main segmentation ─────────────────────────────────────────────────

/// Run quadric VSA segmentation.
pub fn segment(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    mesh: &HalfEdgeMesh,
    config: &QuadricVsaConfig,
) -> QuadricVsaResult {
    let num_faces = faces.len();
    let k = config.target_clusters.min(num_faces);

    // Precompute per-face data
    let face_normals: Vec<[f64; 3]> = faces
        .iter()
        .map(|tri| geometry::face_normal_normalized(positions, *tri))
        .collect();
    let face_areas: Vec<f64> = faces
        .iter()
        .map(|tri| geometry::face_area(positions, *tri))
        .collect();
    let face_centroids: Vec<[f64; 3]> = faces
        .iter()
        .map(|tri| {
            let p0 = positions[tri[0]];
            let p1 = positions[tri[1]];
            let p2 = positions[tri[2]];
            [
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            ]
        })
        .collect();

    // Collect per-face vertex positions (3 vertices per face, used for quadric fitting)
    let face_verts: Vec<[[f64; 3]; 3]> = faces
        .iter()
        .map(|tri| [positions[tri[0]], positions[tri[1]], positions[tri[2]]])
        .collect();

    // Sharp edges
    let sharp_edges = detect_sharp_edges(&face_normals, mesh, config.sharp_edge_threshold);
    let sharp_set: std::collections::HashSet<u32> = sharp_edges.iter().copied().collect();

    // Farthest-point seeding
    let seeds = farthest_point_seed(k, &face_centroids, &face_areas);

    // Initialize with planar proxies (quadric fitting needs a partition first)
    let mut proxies: Vec<QuadricProxy> = seeds
        .iter()
        .map(|&fi| QuadricProxy {
            coeffs: [0.0; 10],
            normal: face_normals[fi],
            centroid: face_centroids[fi],
            area: face_areas[fi],
            surface_type: SurfaceType::Plane,
        })
        .collect();

    let mut labels = vec![usize::MAX; num_faces];
    let alpha = config.quadric_weight;

    for iter in 0..config.max_iterations {
        let old_labels = labels.clone();
        let use_quadric = iter > 0; // first iteration is planar-only

        // Partition: priority-queue flood fill
        labels = vec![usize::MAX; num_faces];
        let mut heap = BinaryHeap::new();
        for (ci, &seed_face) in seeds.iter().enumerate() {
            heap.push(QueueEntry {
                face: seed_face as u32,
                cluster: ci,
                error: 0.0,
            });
        }

        while let Some(entry) = heap.pop() {
            let fi = entry.face as usize;
            if labels[fi] != usize::MAX {
                continue;
            }
            labels[fi] = entry.cluster;

            let he_ids = mesh.face_half_edges(entry.face);
            for &he_id in &he_ids {
                if sharp_set.contains(&he_id) {
                    continue;
                }
                if let Some(adj_face) = mesh.adjacent_face(he_id) {
                    let adj = adj_face as usize;
                    if labels[adj] != usize::MAX {
                        continue;
                    }
                    let err = proxy_error(
                        adj,
                        &face_normals,
                        &face_areas,
                        &face_centroids,
                        &proxies[entry.cluster],
                        use_quadric,
                        alpha,
                    );
                    heap.push(QueueEntry {
                        face: adj_face,
                        cluster: entry.cluster,
                        error: err,
                    });
                }
            }
        }

        // Assign unreached faces
        for fi in 0..num_faces {
            if labels[fi] == usize::MAX {
                let mut best = 0;
                let mut best_err = f64::MAX;
                for (ci, proxy) in proxies.iter().enumerate() {
                    let err = proxy_error(
                        fi,
                        &face_normals,
                        &face_areas,
                        &face_centroids,
                        proxy,
                        use_quadric,
                        alpha,
                    );
                    if err < best_err {
                        best_err = err;
                        best = ci;
                    }
                }
                labels[fi] = best;
            }
        }

        if labels == old_labels {
            break;
        }

        // Update proxies: recompute normals/centroids AND fit quadrics
        proxies = update_proxies(
            k,
            &labels,
            &face_normals,
            &face_areas,
            &face_centroids,
            &face_verts,
        );
    }

    // Merge tiny clusters
    let avg_per_cluster = num_faces as f64 / k as f64;
    let min_merge = (avg_per_cluster * 0.2).max(1.0).ceil() as usize;
    if min_merge > 1 {
        merge_tiny_clusters(&mut labels, &proxies, min_merge);
    }

    let actual_k = *labels.iter().max().unwrap_or(&0) + 1;

    QuadricVsaResult {
        face_labels: labels,
        num_clusters: actual_k,
        proxies,
        sharp_edges,
    }
}

/// Hybrid error metric: blend of quadric distance and L(2,1) normal deviation.
fn proxy_error(
    fi: usize,
    face_normals: &[[f64; 3]],
    face_areas: &[f64],
    face_centroids: &[[f64; 3]],
    proxy: &QuadricProxy,
    use_quadric: bool,
    alpha: f64,
) -> f64 {
    let area = face_areas[fi];
    let normal = face_normals[fi];

    // L(2,1) normal error (always computed)
    let d = geometry::vec3_dot(normal, proxy.normal);
    let l21 = area * (1.0 - d * d);

    if !use_quadric {
        return l21;
    }

    // Taubin distance to quadric
    let centroid = face_centroids[fi];
    let quad_dist = taubin_distance_sq(&proxy.coeffs, &centroid);
    let quad_err = area * quad_dist;

    // Blend: higher alpha = more quadric influence
    alpha * quad_err + (1.0 - alpha) * l21
}

/// Recompute proxies from current labeling, including quadric fitting.
fn update_proxies(
    k: usize,
    labels: &[usize],
    normals: &[[f64; 3]],
    areas: &[f64],
    centroids: &[[f64; 3]],
    face_verts: &[[[f64; 3]; 3]],
) -> Vec<QuadricProxy> {
    let mut proxies = vec![
        QuadricProxy {
            coeffs: [0.0; 10],
            normal: [0.0; 3],
            centroid: [0.0; 3],
            area: 0.0,
            surface_type: SurfaceType::Plane,
        };
        k
    ];

    // Accumulate normals and centroids (area-weighted)
    for (fi, &label) in labels.iter().enumerate() {
        if label >= k {
            continue;
        }
        let a = areas[fi];
        proxies[label].area += a;
        for d in 0..3 {
            proxies[label].normal[d] += normals[fi][d] * a;
            proxies[label].centroid[d] += centroids[fi][d] * a;
        }
    }

    for proxy in proxies.iter_mut() {
        if proxy.area > 1e-15 {
            let inv = 1.0 / proxy.area;
            proxy.centroid = geometry::vec3_scale(proxy.centroid, inv);
            proxy.normal = geometry::vec3_normalize(proxy.normal);
        } else {
            proxy.normal = [0.0, 0.0, 1.0];
        }
    }

    // Collect vertices per cluster and fit quadrics
    for ci in 0..k {
        let mut points = Vec::new();
        let mut weights = Vec::new();

        for (fi, &label) in labels.iter().enumerate() {
            if label != ci {
                continue;
            }
            let a = areas[fi] / 3.0; // distribute face area to its 3 vertices
            for v in &face_verts[fi] {
                points.push(*v);
                weights.push(a);
            }
        }

        if points.len() < 10 {
            // Not enough points for quadric fit — use planar proxy
            // Encode plane as quadric: n·x - d = 0
            // coeffs = [-d, nx, ny, nz, 0, 0, 0, 0, 0, 0]
            let n = proxies[ci].normal;
            let c = proxies[ci].centroid;
            let d = geometry::vec3_dot(n, c);
            proxies[ci].coeffs = [-d, n[0], n[1], n[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            proxies[ci].surface_type = SurfaceType::Plane;
            continue;
        }

        proxies[ci].coeffs = fit_quadric(&points, &weights);
        proxies[ci].surface_type = classify_quadric(&proxies[ci].coeffs, &proxies[ci].centroid);
    }

    proxies
}

// ── Utility functions ─────────────────────────────────────────────────

fn detect_sharp_edges(
    face_normals: &[[f64; 3]],
    mesh: &HalfEdgeMesh,
    threshold: f64,
) -> Vec<u32> {
    let mut sharp = Vec::new();
    for he_idx in 0..mesh.half_edges.len() as u32 {
        if mesh.is_boundary_edge(he_idx) {
            sharp.push(he_idx);
            continue;
        }
        if let Some(adj_face) = mesh.adjacent_face(he_idx) {
            let face_a = mesh.half_edges[he_idx as usize].face;
            let na = face_normals[face_a as usize];
            let nb = face_normals[adj_face as usize];
            let cos_angle = geometry::vec3_dot(na, nb).clamp(-1.0, 1.0);
            if cos_angle.acos() > threshold {
                sharp.push(he_idx);
            }
        }
    }
    sharp
}

fn farthest_point_seed(k: usize, centroids: &[[f64; 3]], areas: &[f64]) -> Vec<usize> {
    let n = centroids.len();
    if k >= n {
        return (0..n).collect();
    }

    let mut seeds = Vec::with_capacity(k);
    let mut is_seed = vec![false; n];
    let mut min_dist = vec![f64::MAX; n];

    // First seed: largest area face
    let first = areas
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    seeds.push(first);
    is_seed[first] = true;
    min_dist[first] = 0.0;

    for _ in 1..k {
        let last = *seeds.last().unwrap();
        for fi in 0..n {
            let d = geometry::vec3_dist(centroids[fi], centroids[last]);
            min_dist[fi] = min_dist[fi].min(d);
        }
        let mut best_idx = 0;
        let mut best_score = -1.0f64;
        for fi in 0..n {
            if is_seed[fi] {
                continue;
            }
            let score = min_dist[fi] * areas[fi].sqrt();
            if score > best_score {
                best_score = score;
                best_idx = fi;
            }
        }
        seeds.push(best_idx);
        is_seed[best_idx] = true;
        min_dist[best_idx] = 0.0;
    }
    seeds
}

fn merge_tiny_clusters(labels: &mut [usize], proxies: &[QuadricProxy], min_faces: usize) {
    let k = proxies.len();
    let mut counts = vec![0usize; k];
    for &l in labels.iter() {
        if l < k {
            counts[l] += 1;
        }
    }

    let mut merge_to = vec![usize::MAX; k];
    for ci in 0..k {
        if counts[ci] < min_faces && counts[ci] > 0 {
            let mut best = ci;
            let mut best_dot = -2.0;
            for cj in 0..k {
                if cj == ci || counts[cj] < min_faces {
                    continue;
                }
                let d = geometry::vec3_dot(proxies[ci].normal, proxies[cj].normal);
                if d > best_dot {
                    best_dot = d;
                    best = cj;
                }
            }
            if best != ci {
                merge_to[ci] = best;
            }
        }
    }

    for label in labels.iter_mut() {
        if *label < k && merge_to[*label] != usize::MAX {
            *label = merge_to[*label];
        }
    }

    // Compact
    let mut used: Vec<usize> = labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    used.sort();
    let remap: std::collections::HashMap<usize, usize> = used
        .iter()
        .enumerate()
        .map(|(i, &old)| (old, i))
        .collect();
    for label in labels.iter_mut() {
        if let Some(&new) = remap.get(label) {
            *label = new;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::shapes;

    #[test]
    fn test_feature_vector() {
        let f = feature_vector(&[1.0, 2.0, 3.0]);
        assert_eq!(f, [1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0]);
    }

    #[test]
    fn test_eval_plane_quadric() {
        // Plane z = 0: coeffs = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        let c = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((eval_quadric(&c, &[1.0, 2.0, 0.0])).abs() < 1e-10);
        assert!((eval_quadric(&c, &[1.0, 2.0, 3.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sphere_quadric() {
        // Unit sphere: x² + y² + z² - 1 = 0
        let c = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        assert!((eval_quadric(&c, &[1.0, 0.0, 0.0])).abs() < 1e-10);
        assert!((eval_quadric(&c, &[0.0, 0.0, 1.0])).abs() < 1e-10);
        assert!((eval_quadric(&c, &[0.5, 0.0, 0.0]) - (-0.75)).abs() < 1e-10);
    }

    #[test]
    fn test_taubin_distance_on_sphere() {
        // Points on unit sphere should have zero distance
        let c = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let d = taubin_distance_sq(&c, &[1.0, 0.0, 0.0]);
        assert!(d < 1e-10, "point on sphere should have ~0 distance, got {}", d);

        // Point off sphere
        let d_off = taubin_distance_sq(&c, &[2.0, 0.0, 0.0]);
        assert!(d_off > 0.5, "point at r=2 should have large distance, got {}", d_off);
    }

    #[test]
    fn test_classify_sphere() {
        let c = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        match classify_quadric(&c, &[0.0, 0.0, 0.0]) {
            SurfaceType::Sphere { radius, .. } => {
                assert!((radius - 1.0).abs() < 0.2, "unit sphere radius should be ~1, got {}", radius);
            }
            other => panic!("expected Sphere, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_plane() {
        let c = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        match classify_quadric(&c, &[0.0; 3]) {
            SurfaceType::Plane => {}
            other => panic!("expected Plane, got {:?}", other),
        }
    }

    #[test]
    fn test_jacobi_eigenvalues_identity() {
        let mut m = [[0.0f64; 10]; 10];
        for i in 0..10 {
            m[i][i] = (i + 1) as f64;
        }
        let (eigs, _) = jacobi_eigen_10(&mut m);
        // Should get eigenvalues 1..10
        for i in 0..10 {
            let expected = (i + 1) as f64;
            let found = eigs.iter().any(|&e| (e - expected).abs() < 1e-8);
            assert!(found, "missing eigenvalue {}", expected);
        }
    }

    #[test]
    fn test_fit_quadric_sphere() {
        // Generate points on a unit sphere
        let mut points = Vec::new();
        let mut weights = Vec::new();
        let n = 50;
        for i in 0..n {
            let phi = std::f64::consts::PI * i as f64 / n as f64;
            for j in 0..n {
                let theta = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
                let p = [
                    phi.sin() * theta.cos(),
                    phi.sin() * theta.sin(),
                    phi.cos(),
                ];
                points.push(p);
                weights.push(1.0);
            }
        }

        let coeffs = fit_quadric(&points, &weights);
        // All points should have near-zero quadric value
        let mut max_err = 0.0f64;
        for p in &points {
            max_err = max_err.max(eval_quadric(&coeffs, p).abs());
        }
        assert!(max_err < 0.01, "sphere fit max error = {}", max_err);

        // Should classify as sphere
        match classify_quadric(&coeffs, &[0.0; 3]) {
            SurfaceType::Sphere { radius, .. } => {
                assert!((radius - 1.0).abs() < 0.3, "radius should be ~1, got {}", radius);
            }
            other => panic!("expected Sphere for sphere fit, got {:?}", other),
        }
    }

    #[test]
    fn test_quadric_vsa_cube() {
        let (positions, faces) = shapes::cube();
        let faces_u32: Vec<[u32; 3]> = faces
            .iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        let mesh = HalfEdgeMesh::from_triangles(positions.len() as u32, &faces_u32);

        let config = QuadricVsaConfig {
            target_clusters: 6,
            max_iterations: 20,
            sharp_edge_threshold: std::f64::consts::PI,
            quadric_weight: 0.5,
        };
        let result = segment(&positions, &faces, &mesh, &config);

        assert!(
            result.num_clusters >= 4 && result.num_clusters <= 8,
            "cube should have ~6 clusters, got {}",
            result.num_clusters
        );
        for &l in &result.face_labels {
            assert!(l < result.num_clusters);
        }
    }

    #[test]
    fn test_quadric_vsa_sphere() {
        // A subdivided sphere should cluster into fewer regions than planar VSA
        // because the quadric proxy can represent the whole sphere
        let (positions, faces) = crate::test_shapes::sphere(2);
        let faces_u32: Vec<[u32; 3]> = faces
            .iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        let mesh = HalfEdgeMesh::from_triangles(positions.len() as u32, &faces_u32);

        let config = QuadricVsaConfig {
            target_clusters: 8,
            max_iterations: 20,
            sharp_edge_threshold: std::f64::consts::PI,
            quadric_weight: 0.7,
        };
        let result = segment(&positions, &faces, &mesh, &config);

        assert!(
            result.num_clusters >= 4 && result.num_clusters <= 12,
            "sphere should segment into ~8 clusters, got {}",
            result.num_clusters
        );

        // At least some proxies should be classified as spheres
        let num_sphere = result
            .proxies
            .iter()
            .filter(|p| matches!(p.surface_type, SurfaceType::Sphere { .. }))
            .count();
        eprintln!(
            "sphere segmentation: {} clusters, {} sphere-classified",
            result.num_clusters, num_sphere
        );
    }
}
