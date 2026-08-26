//! Recovering a small set of curved QB patches from a dense triangle mesh.
//!
//! Two pipelines live here, and they are not equally load-bearing.
//!
//! **Product path.** [`remesh_simplified`] / [`remesh_simplified_curved`]:
//! QEM edge collapse ([`simplify`], [`curvature`]) produces a coarse watertight
//! mesh, then each simplified triangle is fitted independently with the Möbius
//! c-parameter estimator ([`c_estimator`], driven by [`global_fit::per_patch_fit`])
//! and screened by a blow-up guard. [`quadric_vsa`] and [`roundtrip::tessellate_patch`]
//! are also called directly from the WASM bindings.
//!
//! **Research path.** [`remesh`] — VSA segmentation ([`vsa`]) → cluster extraction
//! ([`cluster`]) → harmonic parameterization ([`parameterize`], [`sparse`]) →
//! Gauss-Newton weight fitting ([`fit`]). It is exercised only by tests, and note
//! that it no longer fits anything: it hand-builds a flat patch through each
//! cluster's three corner vertices. It survives as a segmentation baseline.
//!
//! **QB-aware candidate path.** [`coarse_complex`] validates stable authored
//! identity and oriented topology, turns boundaries/domains/explicit locks into
//! corner-cut simplification charts, and prevents a backend from welding those
//! ownership seams. [`conformal_optimizer`] supplies deterministic clustering
//! and conformal scoring; [`linear_fit`] supplies the sparse global shared-weight
//! fit. Topology reduction and provenance correspondence are being connected
//! between those foundations before this path becomes a product entry point.
//!
//! Fitting curved QB patches to arbitrary meshes is an open problem in this
//! codebase, not a solved one. The candidate path therefore remains explicit
//! and measurable instead of replacing the legacy path prematurely.

pub mod geometry;
pub mod linalg;
pub mod sparse;
pub mod vsa;
pub mod quadric_vsa;
pub mod cluster;
pub mod parameterize;
pub mod fit;
pub mod simplify;
pub mod curvature;
pub mod test_shapes;
pub mod roundtrip;
pub mod c_estimator;
pub mod global_fit;
pub mod linear_fit;
pub mod conformal_optimizer;
pub mod coarse_complex;
#[cfg(feature = "meshopt-prototype")]
pub mod coarse_reduction;

use quilting_core::patch::QBTriPatch;

/// Configuration for the remeshing pipeline.
#[derive(Debug, Clone)]
pub struct RemeshConfig {
    /// Target number of output patches.
    pub target_patches: usize,
    /// Maximum Lloyd iterations for VSA.
    pub vsa_iterations: usize,
    /// Dihedral angle threshold for sharp edge detection (radians).
    pub sharp_edge_angle: f64,
    /// Maximum faces per cluster before adaptive splitting.
    pub max_cluster_size: usize,
}

impl Default for RemeshConfig {
    fn default() -> Self {
        Self {
            target_patches: 500,
            vsa_iterations: 20,
            sharp_edge_angle: 40.0_f64.to_radians(),
            max_cluster_size: 20,
        }
    }
}

/// Statistics from the remeshing pipeline.
#[derive(Debug, Clone)]
pub struct RemeshStats {
    pub original_faces: usize,
    pub num_clusters: usize,
    pub num_patches: usize,
    pub avg_position_error: f64,
    pub max_position_error: f64,
    pub avg_normal_error_degrees: f64,
    pub max_normal_error_degrees: f64,
    pub reduction_ratio: f64,
    pub num_skipped: usize,
    pub num_flipped: usize,
    /// Per-patch normal RMS error in degrees.
    pub per_patch_normal_error: Vec<f64>,
}

/// Result of the remeshing pipeline.
#[derive(Debug)]
pub struct RemeshResult {
    pub patches: Vec<QBTriPatch>,
    pub patch_uvs: Vec<[[f32; 2]; 3]>,
    pub patch_normals: Vec<[[f32; 3]; 3]>,
    /// Cluster ID for each original face (for visualization).
    pub face_cluster_ids: Vec<usize>,
    pub stats: RemeshStats,
}

/// Errors from the remeshing pipeline.
#[derive(Debug)]
pub enum RemeshError {
    TooFewFaces,
    SegmentationFailed(String),
    ParameterizationFailed(String),
    FittingFailed(String),
}

impl std::fmt::Display for RemeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewFaces => write!(f, "mesh has too few faces to remesh"),
            Self::SegmentationFailed(s) => write!(f, "segmentation failed: {}", s),
            Self::ParameterizationFailed(s) => write!(f, "parameterization failed: {}", s),
            Self::FittingFailed(s) => write!(f, "fitting failed: {}", s),
        }
    }
}

impl std::error::Error for RemeshError {}

/// Simplified remeshing via QEM edge collapse.
/// Produces a watertight coarse mesh of flat QB patches.
pub fn remesh_simplified(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    target_patches: usize,
) -> Result<RemeshResult, RemeshError> {
    remesh_simplified_inner(positions, faces, target_patches, None)
}

/// Like [`remesh_simplified`] but fits a curved QB patch inside each simplified
/// triangle with the c-parameter estimator, using the shipped default tuning.
pub fn remesh_simplified_curved(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    target_patches: usize,
) -> Result<RemeshResult, RemeshError> {
    remesh_simplified_curved_with(positions, faces, target_patches, &Default::default())
}

/// [`remesh_simplified_curved`] with explicit fitting tuning — the knobs that
/// used to be inline literals in the fitter (see [`global_fit::CurvedFitConfig`]).
pub fn remesh_simplified_curved_with(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    target_patches: usize,
    config: &global_fit::CurvedFitConfig,
) -> Result<RemeshResult, RemeshError> {
    remesh_simplified_inner(positions, faces, target_patches, Some(config))
}

fn remesh_simplified_inner(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    target_patches: usize,
    fit_curved: Option<&global_fit::CurvedFitConfig>,
) -> Result<RemeshResult, RemeshError> {
    if faces.len() < 4 {
        return Err(RemeshError::TooFewFaces);
    }

    let (simp_pos, simp_faces) = simplify::simplify(positions, faces, target_patches);

    let orig_normals = geometry::compute_vertex_normals(positions, faces);
    let vertex_normals = geometry::compute_vertex_normals(&simp_pos, &simp_faces);
    let mut patches = Vec::with_capacity(simp_faces.len());
    let mut patch_uvs = Vec::with_capacity(simp_faces.len());
    let mut patch_normals = Vec::with_capacity(simp_faces.len());

    if let Some(fit_config) = fit_curved {
        // Per-patch fit: each patch gets its own c parameter with a blow-up guard.
        // per_patch_fit is more stable than global_fit for visual output.
        let fit_result = global_fit::per_patch_fit(
            &simp_pos, &simp_faces, positions, &orig_normals, fit_config,
        );
        for (fi, face) in simp_faces.iter().enumerate() {
            patches.push(fit_result.patches[fi]);
            patch_uvs.push([[0.0f32, 0.0]; 3]);
            patch_normals.push([
                [vertex_normals[face[0]][0] as f32, vertex_normals[face[0]][1] as f32, vertex_normals[face[0]][2] as f32],
                [vertex_normals[face[1]][0] as f32, vertex_normals[face[1]][1] as f32, vertex_normals[face[1]][2] as f32],
                [vertex_normals[face[2]][0] as f32, vertex_normals[face[2]][1] as f32, vertex_normals[face[2]][2] as f32],
            ]);
        }
    } else {
        for face in &simp_faces {
            patches.push(quilting_core::patch::QBTriPatch::flat(
                simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]],
            ));
            patch_uvs.push([[0.0f32, 0.0]; 3]);
            patch_normals.push([
                [vertex_normals[face[0]][0] as f32, vertex_normals[face[0]][1] as f32, vertex_normals[face[0]][2] as f32],
                [vertex_normals[face[1]][0] as f32, vertex_normals[face[1]][1] as f32, vertex_normals[face[1]][2] as f32],
                [vertex_normals[face[2]][0] as f32, vertex_normals[face[2]][1] as f32, vertex_normals[face[2]][2] as f32],
            ]);
        }
    }

    // Error estimate
    let mut max_err = 0.0_f64;
    let mut sum_err = 0.0;
    for p in positions {
        let mut min_dist = f64::MAX;
        for face in &simp_faces {
            let d = point_to_triangle_dist(*p, simp_pos[face[0]], simp_pos[face[1]], simp_pos[face[2]]);
            min_dist = min_dist.min(d);
        }
        sum_err += min_dist;
        max_err = max_err.max(min_dist);
    }
    let avg_err = sum_err / positions.len().max(1) as f64;

    let face_cluster_ids = vec![0; faces.len()];
    let num_patches = patches.len();

    Ok(RemeshResult {
        patches,
        patch_uvs,
        patch_normals,
        face_cluster_ids,
        stats: RemeshStats {
            original_faces: faces.len(),
            num_clusters: simp_faces.len(),
            num_patches,
            avg_position_error: avg_err,
            max_position_error: max_err,
            avg_normal_error_degrees: 0.0,
            max_normal_error_degrees: 0.0,
            reduction_ratio: faces.len() as f64 / num_patches.max(1) as f64,
            num_skipped: 0,
            num_flipped: 0,
            per_patch_normal_error: vec![],
        },
    })
}

/// Distance from point to triangle (approximate — uses projection to plane).
fn point_to_triangle_dist(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = geometry::vec3_sub(b, a);
    let ac = geometry::vec3_sub(c, a);
    let n = geometry::vec3_cross(ab, ac);
    let len = geometry::vec3_len(n);
    if len < 1e-15 { return geometry::vec3_dist(p, a); }
    let d = geometry::vec3_dot(geometry::vec3_sub(p, a), n).abs() / len;
    d
}

/// Run the VSA segmentation pipeline: segment → extract clusters → parameterize
/// → emit one patch per cluster.
///
/// Test-only, and it does NOT fit QB patches despite the machinery it drags in.
/// The fitting step was removed when free-weight Gauss-Newton proved unusable;
/// what remains hand-builds a flat [`fit::FitResult`] through each cluster's
/// three corner vertices (with a winding fix), then reports how far the original
/// surface strays from it. Useful as a segmentation-quality baseline and as the
/// only consumer of [`vsa`], [`cluster`], [`parameterize`] and [`sparse`]. The
/// live entry points are [`remesh_simplified`] and [`remesh_simplified_curved`].
pub fn remesh(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    normals: Option<&[[f64; 3]]>,
    uvs: Option<&[[f64; 2]]>,
    config: &RemeshConfig,
) -> Result<RemeshResult, RemeshError> {
    if faces.len() < 4 {
        return Err(RemeshError::TooFewFaces);
    }

    // Step 1: Compute vertex normals if not provided
    let computed_normals;
    let vertex_normals = match normals {
        Some(n) => n,
        None => {
            computed_normals = geometry::compute_vertex_normals(positions, faces);
            &computed_normals
        }
    };

    // Step 2: Build half-edge mesh
    let num_verts = positions.len() as u32;
    let faces_u32: Vec<[u32; 3]> = faces.iter()
        .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
        .collect();
    let he_mesh = quilting_mesh::HalfEdgeMesh::from_triangles(num_verts, &faces_u32);

    // Step 3: VSA segmentation
    let vsa_config = vsa::VsaConfig {
        target_clusters: config.target_patches,
        max_iterations: config.vsa_iterations,
        sharp_edge_threshold: config.sharp_edge_angle,
    };
    let mut vsa_result = vsa::segment(positions, faces, &he_mesh, &vsa_config);

    // Step 3b: Adaptive splitting — break oversized clusters
    // Scale max size relative to the average cluster size so we don't explode
    // the cluster count for large meshes with small target counts
    let effective_max = if config.max_cluster_size > 0 {
        let avg = faces.len() / config.target_patches.max(1);
        // Cap at max_cluster_size but ensure at least avg size to avoid over-splitting
        config.max_cluster_size.max(avg)
    } else {
        usize::MAX
    };
    if effective_max < usize::MAX {
        split_large_clusters(
            &mut vsa_result,
            positions,
            faces,
            effective_max,
        );
    }

    // Step 4: Extract clusters and fit QB patches
    let clusters = cluster::extract_clusters(
        positions, faces, &he_mesh, &vsa_result,
    );

    let mut patches = Vec::with_capacity(clusters.len());
    let mut patch_uvs = Vec::with_capacity(clusters.len());
    let mut patch_normals_out = Vec::with_capacity(clusters.len());
    let mut total_pos_err = 0.0;
    let mut max_pos_err = 0.0_f64;
    let mut total_norm_err = 0.0;
    let mut max_norm_err = 0.0_f64;
    let mut num_fitted = 0usize;
    let mut num_skipped = 0usize;
    let mut num_flipped = 0usize;
    let mut per_patch_normal_error = Vec::new();

    for cl in &clusters {
        // Parameterize cluster onto reference triangle
        let param = match parameterize::parameterize_cluster(positions, faces, cl, &he_mesh) {
            Ok(p) => p,
            Err(_) => { num_skipped += 1; continue; }
        };

        // Collect sample positions and normals in cluster
        let sample_positions: Vec<[f64; 3]> = cl.vertex_indices.iter()
            .map(|&vi| positions[vi])
            .collect();
        let sample_normals: Vec<[f64; 3]> = cl.vertex_indices.iter()
            .map(|&vi| vertex_normals[vi])
            .collect();

        // Create QB patch directly from corner vertices (no fitting needed).
        // The corners are actual mesh vertices; the QB patch with identity weights
        // is just a flat triangle through these 3 points.
        let mut fit_result = fit::FitResult {
            patch: quilting_core::patch::QBTriPatch::flat(
                positions[cl.corner_vertices[0]],
                positions[cl.corner_vertices[1]],
                positions[cl.corner_vertices[2]],
            ),
            rms_position_error: 0.0,
            max_position_error: 0.0,
            rms_normal_error_degrees: 0.0,
            max_normal_error_degrees: 0.0,
        };

        // Check normal consistency: compare the control triangle's geometric
        // normal against the average of the original mesh vertex normals.
        // Use the control triangle normal (not QB eval) since identity-weight
        // QB patches evaluate as flat triangles anyway.
        let p0 = fit_result.patch.positions[0].to_point();
        let p1 = fit_result.patch.positions[1].to_point();
        let p2 = fit_result.patch.positions[2].to_point();
        let tri_normal = geometry::face_normal_normalized(&[p0, p1, p2], [0, 1, 2]);

        // Area-weighted average of original face normals (more robust than vertex normals)
        let mut cluster_normal = [0.0; 3];
        for &fi in &cl.face_indices {
            let fn_ = geometry::face_normal(positions, faces[fi]);
            cluster_normal = geometry::vec3_add(cluster_normal, fn_);
        }
        cluster_normal = geometry::vec3_normalize(cluster_normal);

        if geometry::vec3_dot(tri_normal, cluster_normal) < 0.0 {
            fit_result.patch.positions.swap(1, 2);
            fit_result.patch.weights.swap(1, 2);
            num_flipped += 1;
        }

        // Recompute errors after potential normal flip
        let (rms_pos, max_pos_local, rms_norm, max_norm_local) =
            fit::compute_errors(&fit_result.patch, &sample_positions, &sample_normals, &param.vertex_bary);

        total_pos_err += rms_pos;
        max_pos_err = max_pos_err.max(max_pos_local);
        total_norm_err += rms_norm;
        max_norm_err = max_norm_err.max(max_norm_local);
        per_patch_normal_error.push(rms_norm);
        num_fitted += 1;

        patches.push(fit_result.patch);

        // Transfer UVs from original mesh corners
        let corner_uvs = if let Some(uv_data) = uvs {
            [
                [uv_data[cl.corner_vertices[0]][0] as f32, uv_data[cl.corner_vertices[0]][1] as f32],
                [uv_data[cl.corner_vertices[1]][0] as f32, uv_data[cl.corner_vertices[1]][1] as f32],
                [uv_data[cl.corner_vertices[2]][0] as f32, uv_data[cl.corner_vertices[2]][1] as f32],
            ]
        } else {
            [[0.0, 0.0]; 3]
        };
        patch_uvs.push(corner_uvs);

        // Corner normals
        let cn = [
            [vertex_normals[cl.corner_vertices[0]][0] as f32,
             vertex_normals[cl.corner_vertices[0]][1] as f32,
             vertex_normals[cl.corner_vertices[0]][2] as f32],
            [vertex_normals[cl.corner_vertices[1]][0] as f32,
             vertex_normals[cl.corner_vertices[1]][1] as f32,
             vertex_normals[cl.corner_vertices[1]][2] as f32],
            [vertex_normals[cl.corner_vertices[2]][0] as f32,
             vertex_normals[cl.corner_vertices[2]][1] as f32,
             vertex_normals[cl.corner_vertices[2]][2] as f32],
        ];
        patch_normals_out.push(cn);
    }

    let n = num_fitted.max(1) as f64;
    let stats = RemeshStats {
        original_faces: faces.len(),
        num_clusters: vsa_result.num_clusters,
        num_patches: patches.len(),
        avg_position_error: total_pos_err / n,
        max_position_error: max_pos_err,
        avg_normal_error_degrees: total_norm_err / n,
        max_normal_error_degrees: max_norm_err,
        reduction_ratio: faces.len() as f64 / patches.len().max(1) as f64,
        num_skipped,
        num_flipped,
        per_patch_normal_error,
    };

    Ok(RemeshResult {
        patches,
        patch_uvs,
        patch_normals: patch_normals_out,
        face_cluster_ids: vsa_result.face_labels,
        stats,
    })
}

/// Split clusters that exceed max_cluster_size by re-running VSA on their faces.
fn split_large_clusters(
    result: &mut vsa::VsaResult,
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    max_size: usize,
) {
    // Count faces per cluster
    let mut cluster_faces: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (fi, &label) in result.face_labels.iter().enumerate() {
        cluster_faces.entry(label).or_default().push(fi);
    }

    let mut next_label = result.num_clusters;
    let mut changed = false;

    for (&_cluster_id, face_list) in &cluster_faces {
        if face_list.len() <= max_size { continue; }

        // This cluster is too big — subdivide it
        let num_sub = (face_list.len() + max_size - 1) / max_size;
        if num_sub < 2 { continue; }

        let face_centroids: Vec<[f64; 3]> = face_list.iter()
            .map(|&fi| {
                let tri = faces[fi];
                let p = [positions[tri[0]], positions[tri[1]], positions[tri[2]]];
                [(p[0][0]+p[1][0]+p[2][0])/3.0,
                 (p[0][1]+p[1][1]+p[2][1])/3.0,
                 (p[0][2]+p[1][2]+p[2][2])/3.0]
            })
            .collect();

        // Simple spatial k-means on centroids
        let sub_labels = spatial_kmeans(&face_centroids, num_sub, 10);

        // Assign new labels
        for (local_i, &fi) in face_list.iter().enumerate() {
            let sub = sub_labels[local_i];
            if sub > 0 {
                result.face_labels[fi] = next_label + sub - 1;
                changed = true;
            }
            // sub == 0 keeps the original label
        }
        next_label += num_sub - 1;
    }

    if changed {
        // Compact labels
        let mut used: Vec<usize> = result.face_labels.iter().copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        used.sort();
        let remap: std::collections::HashMap<usize, usize> = used.iter()
            .enumerate().map(|(i, &old)| (old, i)).collect();
        for label in result.face_labels.iter_mut() {
            if let Some(&new) = remap.get(label) {
                *label = new;
            }
        }
        result.num_clusters = used.len();
    }
}

/// Simple spatial k-means on 3D points.
fn spatial_kmeans(points: &[[f64; 3]], k: usize, max_iter: usize) -> Vec<usize> {
    let n = points.len();
    if k >= n { return (0..n).collect(); }

    // Initialize centers by stride
    let mut centers: Vec<[f64; 3]> = (0..k)
        .map(|i| points[i * n / k])
        .collect();

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f64::MAX;
            for (ci, c) in centers.iter().enumerate() {
                let d = geometry::vec3_dist(*p, *c);
                if d < best_dist {
                    best_dist = d;
                    best = ci;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed { break; }

        // Update centers
        let mut counts = vec![0usize; k];
        let mut sums = vec![[0.0; 3]; k];
        for (i, p) in points.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            sums[c] = geometry::vec3_add(sums[c], *p);
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                centers[ci] = geometry::vec3_scale(sums[ci], 1.0 / counts[ci] as f64);
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remesh_icosahedron() {
        let (positions, faces) = quilting_core::shapes::icosahedron();
        let config = RemeshConfig {
            target_patches: 5,
            vsa_iterations: 10,
            sharp_edge_angle: std::f64::consts::PI, // no sharp edges
            ..Default::default()
        };
        let result = remesh(&positions, &faces, None, None, &config).unwrap();
        assert!(result.patches.len() >= 3, "should produce at least 3 patches");
        assert!(result.stats.num_clusters >= 3);
        assert!(result.stats.avg_position_error.is_finite());
    }

    #[test]
    fn test_remesh_cube() {
        let (positions, faces) = quilting_core::shapes::cube();
        let config = RemeshConfig {
            target_patches: 6,
            vsa_iterations: 10,
            sharp_edge_angle: std::f64::consts::PI,
            ..Default::default()
        };
        let result = remesh(&positions, &faces, None, None, &config).unwrap();
        assert!(result.patches.len() >= 4);
        // Cube has 12 triangles, should reduce
        assert!(result.stats.reduction_ratio > 1.0);
    }

    #[test]
    fn test_remesh_to_render_pipeline() {
        // End-to-end: remesh → compute_instances_from_patches → FaceInstance
        let (positions, faces) = quilting_core::shapes::icosahedron();
        let config = RemeshConfig {
            target_patches: 5,
            ..Default::default()
        };
        let result = remesh(&positions, &faces, None, None, &config).unwrap();

        // Feed into the renderer integration point
        let transform = quilting_core::quaternion::Mobius::identity();
        let instances = quilting_core::evaluate::compute_instances_from_patches(
            &result.patches,
            &result.patch_uvs,
            &result.patch_normals,
            &transform,
            4,
        );

        assert_eq!(instances.len(), result.patches.len());

        // Verify instances have valid data
        for inst in &instances {
            for p in &inst.positions {
                assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
            }
            for w in &inst.weights {
                assert!(w.w.is_finite());
            }
            // Check the instance can be packed to f32
            let packed = inst.to_f32_array();
            assert_eq!(packed.len(), 52);
            for val in &packed {
                assert!(val.is_finite(), "packed value is not finite: {}", val);
            }
        }
    }

    #[test]
    fn test_remesh_with_mobius_transform() {
        // Verify remeshed patches transform correctly under Möbius
        let (positions, faces) = quilting_core::shapes::octahedron();
        let config = RemeshConfig {
            target_patches: 4,
            ..Default::default()
        };
        let result = remesh(&positions, &faces, None, None, &config).unwrap();

        // Apply a translation
        let t = quilting_core::quaternion::Mobius::translation(
            quilting_core::quaternion::Quat::from_point(1.0, 0.0, 0.0),
        );
        let instances = quilting_core::evaluate::compute_instances_from_patches(
            &result.patches,
            &result.patch_uvs,
            &result.patch_normals,
            &t,
            4,
        );

        // All positions should be shifted by (1, 0, 0)
        for (i, inst) in instances.iter().enumerate() {
            let original = result.patches[i].positions[0].to_point();
            let shifted = inst.positions[0].to_point();
            assert!(
                (shifted[0] - original[0] - 1.0).abs() < 1e-6,
                "x should be shifted by 1.0"
            );
        }
    }

    #[test]
    fn test_remesh_simplified_curved_sphere() {
        // Simplify a sphere and fit curved QB patches
        let (positions, faces) = crate::test_shapes::sphere(2);
        let result = remesh_simplified_curved(&positions, &faces, 20).unwrap();

        eprintln!("curved QEM sphere: {} patches from {} faces",
            result.stats.num_patches, result.stats.original_faces);
        eprintln!("  avg pos error: {:.6}, max: {:.6}",
            result.stats.avg_position_error, result.stats.max_position_error);

        // Should have reasonable patches
        assert!(result.patches.len() <= 30);
        assert!(result.patches.len() >= 10);

        // Check that at least some patches have non-identity weights
        let num_curved = result.patches.iter().filter(|p| {
            let w1_diff = (p.weights[1] - quilting_core::quaternion::Quat::ONE).norm();
            w1_diff > 0.01
        }).count();
        eprintln!("  {} of {} patches have non-identity weights (curved)",
            num_curved, result.patches.len());
    }

    #[test]
    fn test_too_few_faces() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let config = RemeshConfig::default();
        let result = remesh(&positions, &faces, None, None, &config);
        assert!(result.is_err());
    }
}
