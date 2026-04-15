/// Discrete curvature estimation for cyclide-aware mesh simplification.
///
/// A Dupin cyclide has constant principal curvatures along each family of
/// curvature lines. The key insight: if a surface region has nearly constant
/// principal curvatures, it can be represented by a single QB patch with
/// appropriate quaternion weights.
///
/// We compute:
/// 1. Per-vertex principal curvatures (κ₁, κ₂) and directions via the
///    discrete shape operator (cotangent-weighted)
/// 2. Per-edge "cyclide compatibility" — how similar the curvatures are
///    across an edge. High compatibility = the edge can be collapsed
///    without losing cyclide structure.

use crate::geometry;

/// Per-vertex curvature data.
#[derive(Clone, Debug)]
pub struct VertexCurvature {
    /// Principal curvatures (κ₁ ≥ κ₂)
    pub k1: f64,
    pub k2: f64,
    /// Principal directions (unit vectors in tangent plane)
    pub dir1: [f64; 3],
    pub dir2: [f64; 3],
    /// Mean curvature H = (κ₁ + κ₂) / 2
    pub mean: f64,
    /// Gaussian curvature K = κ₁ * κ₂
    pub gaussian: f64,
}

/// Compute per-vertex principal curvatures via the discrete shape operator.
///
/// Uses the cotangent-weighted Laplacian: the mean curvature normal is
/// H·n = (1/2A) Σ (cot α_ij + cot β_ij)(p_j - p_i)
/// where the sum is over the 1-ring neighborhood.
pub fn compute_vertex_curvatures(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
) -> Vec<VertexCurvature> {
    let n_verts = positions.len();
    let vertex_normals = geometry::compute_vertex_normals(positions, faces);

    // Compute per-vertex area (1/3 of incident face areas)
    let mut vertex_area = vec![0.0f64; n_verts];
    for face in faces {
        let area = geometry::face_area(positions, *face);
        for &vi in face {
            vertex_area[vi] += area / 3.0;
        }
    }

    // Build adjacency: per-vertex list of (neighbor_vertex, face_idx, local_opposite)
    let mut adjacency: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_verts];
    for (fi, face) in faces.iter().enumerate() {
        for local in 0..3 {
            let vi = face[local];
            let vj = face[(local + 1) % 3];
            adjacency[vi].push((vj, fi, (local + 2) % 3));
        }
    }

    let mut curvatures = Vec::with_capacity(n_verts);

    for vi in 0..n_verts {
        let pi = positions[vi];
        let ni = vertex_normals[vi];
        let area = vertex_area[vi];

        if area < 1e-15 || geometry::vec3_len(ni) < 0.5 {
            curvatures.push(VertexCurvature {
                k1: 0.0, k2: 0.0,
                dir1: [1.0, 0.0, 0.0], dir2: [0.0, 1.0, 0.0],
                mean: 0.0, gaussian: 0.0,
            });
            continue;
        }

        // Compute mean curvature vector via cotangent Laplacian
        let mut laplacian = [0.0f64; 3];
        for &(vj, fi, opp_local) in &adjacency[vi] {
            let cot = geometry::cotangent_at_vertex(positions, faces[fi], opp_local);
            let w = cot.max(0.0); // clamp negative cotangents
            let edge = geometry::vec3_sub(positions[vj], pi);
            laplacian[0] += w * edge[0];
            laplacian[1] += w * edge[1];
            laplacian[2] += w * edge[2];
        }

        // H·n ≈ (1/2A) * laplacian
        let inv_2a = if area > 1e-15 { 1.0 / (2.0 * area) } else { 0.0 };
        let hn = geometry::vec3_scale(laplacian, inv_2a);
        let mean = geometry::vec3_dot(hn, ni); // signed mean curvature

        // Estimate Gaussian curvature via angle defect
        // K = (2π - Σ θ_i) / A
        let mut angle_sum = 0.0;
        for &(_vj, fi, _) in &adjacency[vi] {
            let face = faces[fi];
            let local_vi = face.iter().position(|&v| v == vi).unwrap_or(0);
            let va = positions[face[(local_vi + 1) % 3]];
            let vb = positions[face[(local_vi + 2) % 3]];
            let e1 = geometry::vec3_sub(va, pi);
            let e2 = geometry::vec3_sub(vb, pi);
            let l1 = geometry::vec3_len(e1);
            let l2 = geometry::vec3_len(e2);
            if l1 > 1e-15 && l2 > 1e-15 {
                let cos_a = (geometry::vec3_dot(e1, e2) / (l1 * l2)).clamp(-1.0, 1.0);
                angle_sum += cos_a.acos();
            }
        }
        // Deduplicate: each face contributes the angle once, but adjacency
        // lists edges, not faces. Divide by 2 since each face has 2 edges from vi.
        angle_sum /= 2.0;
        let gaussian = (2.0 * std::f64::consts::PI - angle_sum) / area.max(1e-15);

        // Principal curvatures from H and K:
        // κ₁, κ₂ = H ± sqrt(H² - K)
        let disc = (mean * mean - gaussian).max(0.0).sqrt();
        let k1 = mean + disc;
        let k2 = mean - disc;

        // Principal directions: project edges onto tangent plane, weight by curvature
        // This is approximate — uses the direction of maximum normal curvature
        let mut max_kn = 0.0_f64;
        let mut dir1 = [1.0, 0.0, 0.0];
        for &(vj, _, _) in &adjacency[vi] {
            let edge = geometry::vec3_sub(positions[vj], pi);
            let edge_len = geometry::vec3_len(edge);
            if edge_len < 1e-15 { continue; }
            // Project edge onto tangent plane
            let proj_n = geometry::vec3_dot(edge, ni);
            let tangent = [
                edge[0] - proj_n * ni[0],
                edge[1] - proj_n * ni[1],
                edge[2] - proj_n * ni[2],
            ];
            let t_len = geometry::vec3_len(tangent);
            if t_len < 1e-15 { continue; }
            // Normal curvature in this direction ≈ 2 * (n · (pj - pi)) / |pj - pi|²
            let kn = (2.0 * proj_n / (edge_len * edge_len)).abs();
            if kn > max_kn {
                max_kn = kn;
                dir1 = geometry::vec3_normalize(tangent);
            }
        }
        // dir2 = n × dir1
        let dir2 = geometry::vec3_normalize(geometry::vec3_cross(ni, dir1));

        curvatures.push(VertexCurvature {
            k1, k2, dir1, dir2, mean, gaussian,
        });
    }

    curvatures
}

/// Compute the "cyclide compatibility" score for an edge collapse.
///
/// High score = the two vertices have similar curvature properties,
/// meaning they belong to the same cyclide region and the edge can
/// be collapsed without losing cyclide structure.
///
/// Low score = curvature changes sharply across this edge — it should
/// be preserved as a patch boundary.
///
/// Returns a value in [0, 1] where:
/// - 1.0 = perfectly compatible (same curvature type and magnitude)
/// - 0.0 = completely incompatible (curvature changes sign or type)
pub fn cyclide_compatibility(c0: &VertexCurvature, c1: &VertexCurvature) -> f64 {
    // Compare mean curvatures
    let mean_diff = (c0.mean - c1.mean).abs();
    let mean_avg = (c0.mean.abs() + c1.mean.abs()) * 0.5 + 1e-6;
    let mean_compat = 1.0 / (1.0 + (mean_diff / mean_avg).powi(2));

    // Compare Gaussian curvatures
    let gauss_diff = (c0.gaussian - c1.gaussian).abs();
    let gauss_avg = (c0.gaussian.abs() + c1.gaussian.abs()) * 0.5 + 1e-6;
    let gauss_compat = 1.0 / (1.0 + (gauss_diff / gauss_avg).powi(2));

    // Compare principal curvature magnitudes
    let k1_diff = (c0.k1 - c1.k1).abs();
    let k2_diff = (c0.k2 - c1.k2).abs();
    let k_avg = (c0.k1.abs() + c0.k2.abs() + c1.k1.abs() + c1.k2.abs()) * 0.25 + 1e-6;
    let k_compat = 1.0 / (1.0 + ((k1_diff + k2_diff) / k_avg).powi(2));

    // Combined score — geometric mean gives balanced weighting
    (mean_compat * gauss_compat * k_compat).cbrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_curvature() {
        // Unit sphere should have κ₁ = κ₂ = 1 everywhere
        let (positions, faces) = crate::test_shapes::sphere(2);
        let curvatures = compute_vertex_curvatures(&positions, &faces);

        // Check a few vertices
        let mut mean_h = 0.0;
        for c in &curvatures {
            mean_h += c.mean;
        }
        mean_h /= curvatures.len() as f64;
        // Mean curvature magnitude of unit sphere should be ~1 (sign depends on normal convention)
        assert!((mean_h.abs() - 1.0).abs() < 0.5,
            "mean curvature magnitude of unit sphere should be ~1, got {:.3}", mean_h);
    }

    #[test]
    fn test_flat_curvature() {
        // Flat plane should have κ₁ = κ₂ = 0
        let positions = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [0.5, 0.5, 0.0],
        ];
        let faces = vec![[0,1,4], [1,3,4], [3,2,4], [2,0,4]];
        let curvatures = compute_vertex_curvatures(&positions, &faces);

        // Interior vertex should have near-zero curvature
        let c = &curvatures[4]; // center vertex
        assert!(c.mean.abs() < 0.1, "flat plane mean curvature should be ~0, got {:.3}", c.mean);
    }

    #[test]
    fn test_compatibility_same_curvature() {
        let c = VertexCurvature {
            k1: 1.0, k2: 1.0, dir1: [1.0,0.0,0.0], dir2: [0.0,1.0,0.0],
            mean: 1.0, gaussian: 1.0,
        };
        let compat = cyclide_compatibility(&c, &c);
        assert!((compat - 1.0).abs() < 0.01, "identical curvatures should be fully compatible");
    }

    #[test]
    fn test_compatibility_different_curvature() {
        let c0 = VertexCurvature {
            k1: 1.0, k2: 1.0, dir1: [1.0,0.0,0.0], dir2: [0.0,1.0,0.0],
            mean: 1.0, gaussian: 1.0,
        };
        let c1 = VertexCurvature {
            k1: 0.0, k2: 0.0, dir1: [1.0,0.0,0.0], dir2: [0.0,1.0,0.0],
            mean: 0.0, gaussian: 0.0,
        };
        let compat = cyclide_compatibility(&c0, &c1);
        assert!(compat < 0.5, "sphere vs flat should have low compatibility, got {:.3}", compat);
    }
}
