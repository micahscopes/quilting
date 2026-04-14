/// Variational Shape Approximation — clusters mesh faces by normal similarity.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use quilting_mesh::HalfEdgeMesh;
use crate::geometry;

#[derive(Debug, Clone)]
pub struct VsaConfig {
    pub target_clusters: usize,
    pub max_iterations: usize,
    pub sharp_edge_threshold: f64,
}

impl Default for VsaConfig {
    fn default() -> Self {
        Self {
            target_clusters: 500,
            max_iterations: 20,
            sharp_edge_threshold: 40.0_f64.to_radians(),
        }
    }
}

/// A planar proxy representing a cluster.
#[derive(Debug, Clone)]
pub struct Proxy {
    pub normal: [f64; 3],
    pub centroid: [f64; 3],
    pub area: f64,
}

#[derive(Debug, Clone)]
pub struct VsaResult {
    pub face_labels: Vec<usize>,
    pub num_clusters: usize,
    pub proxies: Vec<Proxy>,
    pub sharp_edges: Vec<u32>,
}

/// Entry for the priority queue during flood fill.
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
        // Min-heap: reverse ordering so smaller error has higher priority
        other.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal)
    }
}

/// Run VSA segmentation on the mesh.
pub fn segment(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    mesh: &HalfEdgeMesh,
    config: &VsaConfig,
) -> VsaResult {
    let num_faces = faces.len();
    let k = config.target_clusters.min(num_faces);

    // Precompute face normals and areas
    let face_normals: Vec<[f64; 3]> = faces.iter()
        .map(|tri| geometry::face_normal_normalized(positions, *tri))
        .collect();
    let face_areas: Vec<f64> = faces.iter()
        .map(|tri| geometry::face_area(positions, *tri))
        .collect();
    let face_centroids: Vec<[f64; 3]> = faces.iter()
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

    // Detect sharp edges (reuse precomputed normals)
    let sharp_edges = detect_sharp_edges_with_normals(&face_normals, mesh, config.sharp_edge_threshold);
    let sharp_set: std::collections::HashSet<u32> = sharp_edges.iter().copied().collect();

    // Farthest-point seeding
    let seeds = farthest_point_seed(k, &face_centroids, &face_areas);
    // eprintln!("VSA: {} faces, {} target clusters, {} seeds: {:?}", num_faces, k, seeds.len(), seeds);

    // Initialize proxies from seeds
    let mut proxies: Vec<Proxy> = seeds.iter().map(|&fi| {
        Proxy {
            normal: face_normals[fi],
            centroid: face_centroids[fi],
            area: face_areas[fi],
        }
    }).collect();

    let mut labels = vec![usize::MAX; num_faces];

    // Lloyd iteration
    for _iter in 0..config.max_iterations {
        let old_labels = labels.clone();

        // Partition: priority-queue flood fill
        labels = vec![usize::MAX; num_faces];
        let mut heap = BinaryHeap::new();

        // Seed the queue with proxy faces
        for (ci, &seed_face) in seeds.iter().enumerate() {
            heap.push(QueueEntry { face: seed_face as u32, cluster: ci, error: 0.0 });
        }

        while let Some(entry) = heap.pop() {
            let fi = entry.face as usize;
            if labels[fi] != usize::MAX { continue; }
            labels[fi] = entry.cluster;

            // Expand to adjacent faces
            let he_ids = mesh.face_half_edges(entry.face);
            for &he_id in &he_ids {
                // Skip sharp edges
                if sharp_set.contains(&he_id) { continue; }
                if let Some(adj_face) = mesh.adjacent_face(he_id) {
                    let adj = adj_face as usize;
                    if labels[adj] != usize::MAX { continue; }

                    let err = l21_error(&face_normals[adj], face_areas[adj], &proxies[entry.cluster]);
                    heap.push(QueueEntry { face: adj_face, cluster: entry.cluster, error: err });
                }
            }
        }

        // Assign any unreached faces to nearest proxy
        for fi in 0..num_faces {
            if labels[fi] == usize::MAX {
                let mut best = 0;
                let mut best_err = f64::MAX;
                for (ci, proxy) in proxies.iter().enumerate() {
                    let err = l21_error(&face_normals[fi], face_areas[fi], proxy);
                    if err < best_err {
                        best_err = err;
                        best = ci;
                    }
                }
                labels[fi] = best;
            }
        }

        // Debug: count labels

        // Check convergence
        if labels == old_labels { break; }

        // Update proxies
        proxies = update_proxies(k, &labels, &face_normals, &face_areas, &face_centroids);
    }

    // Post-process: merge tiny clusters (only if we have enough faces per cluster on average)
    let avg_faces_per_cluster = num_faces as f64 / k as f64;
    let min_merge_size = (avg_faces_per_cluster * 0.2).max(1.0).ceil() as usize;
    if min_merge_size > 1 {
        merge_tiny_clusters(&mut labels, &proxies, &face_normals, &face_areas, min_merge_size);
    }

    // Recount clusters
    let actual_k = *labels.iter().max().unwrap_or(&0) + 1;

    VsaResult {
        face_labels: labels,
        num_clusters: actual_k,
        proxies,
        sharp_edges,
    }
}

/// L(2,1) error metric: area * (1 - dot(face_normal, proxy_normal)^2)
fn l21_error(face_normal: &[f64; 3], face_area: f64, proxy: &Proxy) -> f64 {
    let d = geometry::vec3_dot(*face_normal, proxy.normal);
    face_area * (1.0 - d * d)
}

/// Detect sharp edges using precomputed face normals.
fn detect_sharp_edges_with_normals(
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

/// Farthest-point seeding: pick k well-distributed seed faces.
fn farthest_point_seed(k: usize, centroids: &[[f64; 3]], areas: &[f64]) -> Vec<usize> {
    let n = centroids.len();
    if k >= n { return (0..n).collect(); }

    let mut seeds = Vec::with_capacity(k);
    let mut is_seed = vec![false; n];
    let mut min_dist = vec![f64::MAX; n];

    // First seed: face with largest area
    let first = areas.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    seeds.push(first);
    is_seed[first] = true;
    min_dist[first] = 0.0;

    for _ in 1..k {
        // Update min distances from the last seed added
        let last = seeds[seeds.len() - 1];
        for fi in 0..n {
            let d = geometry::vec3_dist(centroids[fi], centroids[last]);
            min_dist[fi] = min_dist[fi].min(d);
        }

        // Pick farthest non-seed face (weighted by area)
        let mut best_idx = 0;
        let mut best_score = -1.0_f64;
        for fi in 0..n {
            if is_seed[fi] { continue; }
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

/// Recompute proxies from current labeling.
fn update_proxies(
    k: usize,
    labels: &[usize],
    normals: &[[f64; 3]],
    areas: &[f64],
    centroids: &[[f64; 3]],
) -> Vec<Proxy> {
    let mut proxies = vec![Proxy { normal: [0.0; 3], centroid: [0.0; 3], area: 0.0 }; k];

    for (fi, &label) in labels.iter().enumerate() {
        if label >= k { continue; }
        let a = areas[fi];
        proxies[label].area += a;
        for d in 0..3 {
            proxies[label].normal[d] += normals[fi][d] * a;
            proxies[label].centroid[d] += centroids[fi][d] * a;
        }
    }

    for proxy in &mut proxies {
        if proxy.area > 1e-15 {
            let inv_a = 1.0 / proxy.area;
            proxy.centroid = geometry::vec3_scale(proxy.centroid, inv_a);
            proxy.normal = geometry::vec3_normalize(proxy.normal);
        } else {
            proxy.normal = [0.0, 0.0, 1.0];
        }
    }

    proxies
}

/// Merge clusters with fewer than `min_faces` into their most similar neighbor.
fn merge_tiny_clusters(
    labels: &mut [usize],
    proxies: &[Proxy],
    _face_normals: &[[f64; 3]],
    _face_areas: &[f64],
    min_faces: usize,
) {
    let k = proxies.len();
    let mut counts = vec![0usize; k];
    for &l in labels.iter() {
        if l < k { counts[l] += 1; }
    }

    // Build merge map: tiny cluster -> best large neighbor
    let mut merge_to = vec![usize::MAX; k];
    for ci in 0..k {
        if counts[ci] < min_faces && counts[ci] > 0 {
            // Find most similar non-tiny cluster
            let mut best = ci;
            let mut best_dot = -2.0;
            for cj in 0..k {
                if cj == ci || counts[cj] < min_faces { continue; }
                let d = geometry::vec3_dot(proxies[ci].normal, proxies[cj].normal);
                if d > best_dot {
                    best_dot = d;
                    best = cj;
                }
            }
            if best != ci { merge_to[ci] = best; }
        }
    }

    // Apply merges
    for label in labels.iter_mut() {
        if *label < k && merge_to[*label] != usize::MAX {
            *label = merge_to[*label];
        }
    }

    // Compact labels so they're contiguous 0..n
    let mut used: Vec<usize> = labels.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
    used.sort();
    let remap: std::collections::HashMap<usize, usize> = used.iter().enumerate().map(|(i, &old)| (old, i)).collect();
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
    fn test_vsa_cube() {
        let (positions, faces_usize) = shapes::cube();
        let faces: Vec<[usize; 3]> = faces_usize;
        let faces_u32: Vec<[u32; 3]> = faces.iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        let mesh = HalfEdgeMesh::from_triangles(positions.len() as u32, &faces_u32);

        // Cube has 90° dihedral angles, so use a higher threshold to allow
        // clustering across the cube's sharp edges (testing normal-based clustering)
        let config = VsaConfig {
            target_clusters: 6,
            max_iterations: 20,
            sharp_edge_threshold: std::f64::consts::PI, // no sharp edge barriers
        };
        let result = segment(&positions, &faces, &mesh, &config);

        // Cube has 6 flat faces (12 triangles) — should cluster into ~6
        assert!(result.num_clusters >= 4 && result.num_clusters <= 8,
            "cube should have ~6 clusters, got {}", result.num_clusters);

        // Every face should be assigned
        for &l in &result.face_labels {
            assert!(l < result.num_clusters);
        }
    }

    #[test]
    fn test_vsa_cube_with_sharp_edges() {
        let (positions, faces_usize) = shapes::cube();
        let faces: Vec<[usize; 3]> = faces_usize;
        let faces_u32: Vec<[u32; 3]> = faces.iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        let mesh = HalfEdgeMesh::from_triangles(positions.len() as u32, &faces_u32);

        // With 40° threshold, cube edges (90°) are all sharp barriers
        // Each face pair (2 triangles sharing coplanar edge) should form a cluster
        let config = VsaConfig {
            target_clusters: 6,
            max_iterations: 20,
            sharp_edge_threshold: 40.0_f64.to_radians(),
        };
        let result = segment(&positions, &faces, &mesh, &config);

        // Should get 6 clusters (one per cube face, each containing 2 coplanar triangles)
        assert!(result.num_clusters >= 5 && result.num_clusters <= 12,
            "cube with sharp edges should have ~6 clusters, got {}", result.num_clusters);
    }

    #[test]
    fn test_vsa_icosahedron() {
        let (positions, faces) = shapes::icosahedron();
        let faces_u32: Vec<[u32; 3]> = faces.iter()
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        let mesh = HalfEdgeMesh::from_triangles(positions.len() as u32, &faces_u32);

        let config = VsaConfig {
            target_clusters: 5,
            max_iterations: 20,
            sharp_edge_threshold: std::f64::consts::PI, // no sharp edges on icosahedron
        };
        let result = segment(&positions, &faces, &mesh, &config);

        assert!(result.num_clusters >= 3 && result.num_clusters <= 7,
            "icosahedron with 5 targets should get ~5 clusters, got {}", result.num_clusters);
    }
}
