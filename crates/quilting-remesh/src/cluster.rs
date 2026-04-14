/// Cluster extraction: boundary loops, junction vertices, and corner selection.

use quilting_mesh::HalfEdgeMesh;
use crate::geometry;

/// A cluster of mesh faces with boundary and corner information.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: usize,
    pub face_indices: Vec<usize>,
    /// All unique vertex indices in this cluster (global mesh indices).
    pub vertex_indices: Vec<usize>,
    /// Ordered boundary vertex loop (global mesh indices).
    pub boundary_loop: Vec<usize>,
    /// 3 corner vertices picked on the boundary (global mesh indices).
    pub corner_vertices: [usize; 3],
    /// Indices into boundary_loop where corners live.
    pub corner_loop_positions: [usize; 3],
    /// Interior (non-boundary) vertices.
    pub interior_vertices: Vec<usize>,
}

/// Find junction vertices: mesh vertices where 3+ clusters meet.
/// These are the natural corners for watertight patch tiling.
fn find_junction_vertices(
    mesh: &HalfEdgeMesh,
    labels: &[usize],
    num_vertices: usize,
) -> Vec<usize> {
    // For each vertex, collect the set of cluster labels of its incident faces
    let mut vertex_clusters: Vec<std::collections::HashSet<usize>> =
        vec![std::collections::HashSet::new(); num_vertices];

    for fi in 0..labels.len() {
        let verts = mesh.face_vertices(fi as u32);
        for &v in &verts {
            vertex_clusters[v as usize].insert(labels[fi]);
        }
    }

    // Junction = vertex with 3+ different clusters, or 2+ clusters on mesh boundary
    let mut junctions = Vec::new();
    for vi in 0..num_vertices {
        let n_clusters = vertex_clusters[vi].len();
        if n_clusters >= 3 {
            junctions.push(vi);
        } else if n_clusters >= 2 {
            // Check if this is on the mesh boundary
            let outgoing = mesh.vertex_outgoing(vi as u32);
            let on_boundary = outgoing.iter().any(|&he| mesh.is_boundary_edge(he));
            if on_boundary {
                junctions.push(vi);
            }
        }
    }

    junctions
}

/// Extract clusters from VSA result, using junction vertices as shared corners.
pub fn extract_clusters(
    positions: &[[f64; 3]],
    _faces: &[[usize; 3]],
    mesh: &HalfEdgeMesh,
    vsa_result: &crate::vsa::VsaResult,
) -> Vec<Cluster> {
    let labels = &vsa_result.face_labels;
    let num_clusters = vsa_result.num_clusters;
    let num_vertices = positions.len();

    // Find junction vertices (shared between 3+ clusters)
    let junctions = find_junction_vertices(mesh, labels, num_vertices);
    let junction_set: std::collections::HashSet<usize> = junctions.iter().copied().collect();

    // Group faces by cluster
    let mut cluster_faces: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
    for (fi, &label) in labels.iter().enumerate() {
        if label < num_clusters {
            cluster_faces[label].push(fi);
        }
    }

    let mut clusters = Vec::with_capacity(num_clusters);

    for ci in 0..num_clusters {
        if cluster_faces[ci].is_empty() { continue; }

        let face_indices = &cluster_faces[ci];

        // Collect all vertices and identify boundary
        let mut vertex_set = std::collections::HashSet::new();
        let mut boundary_he: Vec<u32> = Vec::new();

        for &fi in face_indices {
            let verts = mesh.face_vertices(fi as u32);
            for &v in &verts {
                vertex_set.insert(v as usize);
            }

            let he_ids = mesh.face_half_edges(fi as u32);
            for &he_id in &he_ids {
                let is_cluster_boundary = if mesh.is_boundary_edge(he_id) {
                    true
                } else if let Some(adj_face) = mesh.adjacent_face(he_id) {
                    labels[adj_face as usize] != ci
                } else {
                    true
                };
                if is_cluster_boundary {
                    boundary_he.push(he_id);
                }
            }
        }

        let vertex_indices: Vec<usize> = vertex_set.iter().copied().collect();

        // Chain boundary half-edges into an ordered loop
        let boundary_loop = match chain_boundary_loop(&boundary_he, mesh) {
            Some(loop_verts) => loop_verts,
            None => {
                let bv: Vec<usize> = boundary_he.iter()
                    .flat_map(|&he| {
                        let (from, to) = mesh.edge_vertices(he);
                        vec![from as usize, to as usize]
                    })
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                if bv.len() < 3 { continue; }
                bv
            }
        };

        if boundary_loop.len() < 3 { continue; }

        // Find junction vertices on this cluster's boundary
        let mut junction_positions: Vec<usize> = Vec::new();
        for (loop_idx, &vi) in boundary_loop.iter().enumerate() {
            if junction_set.contains(&vi) {
                junction_positions.push(loop_idx);
            }
        }

        // Pick 3 corners: prefer junction vertices, fall back to farthest-point
        let corner_loop_positions = if junction_positions.len() >= 3 {
            // Pick 3 well-separated junctions
            pick_best_three_junctions(positions, &boundary_loop, &junction_positions)
        } else if junction_positions.len() == 2 {
            // Have 2 junctions, pick a 3rd via farthest-point from both
            let c0 = junction_positions[0];
            let c1 = junction_positions[1];
            let c2 = find_farthest_from_two(positions, &boundary_loop, c0, c1);
            let mut corners = [c0, c1, c2];
            corners.sort();
            corners
        } else {
            // No junctions — fall back to farthest-point sampling
            farthest_point_corners(positions, &boundary_loop)
        };

        let corner_vertices = [
            boundary_loop[corner_loop_positions[0]],
            boundary_loop[corner_loop_positions[1]],
            boundary_loop[corner_loop_positions[2]],
        ];

        // Interior vertices
        let boundary_set: std::collections::HashSet<usize> = boundary_loop.iter().copied().collect();
        let interior_vertices: Vec<usize> = vertex_indices.iter()
            .copied()
            .filter(|v| !boundary_set.contains(v))
            .collect();

        clusters.push(Cluster {
            id: ci,
            face_indices: face_indices.clone(),
            vertex_indices,
            boundary_loop,
            corner_vertices,
            corner_loop_positions,
            interior_vertices,
        });
    }

    clusters
}

/// Pick 3 well-separated junction vertices from a boundary loop.
fn pick_best_three_junctions(
    positions: &[[f64; 3]],
    boundary: &[usize],
    junction_positions: &[usize],
) -> [usize; 3] {
    let n = junction_positions.len();
    if n < 3 { return [0, 1, 2]; }

    // Try all triples and pick the one with maximum minimum arc distance
    let arc_lengths = compute_arc_lengths(positions, boundary);
    let total_len = total_arc_length_from(positions, boundary, &arc_lengths);

    let mut best = [junction_positions[0], junction_positions[1], junction_positions[2]];
    let mut best_score = 0.0_f64;

    // For small junction counts, try all triples
    if n <= 20 {
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let a = junction_positions[i];
                    let b = junction_positions[j];
                    let c = junction_positions[k];
                    let min_dist = arc_dist(&arc_lengths, total_len, a, b)
                        .min(arc_dist(&arc_lengths, total_len, b, c))
                        .min(arc_dist(&arc_lengths, total_len, a, c));
                    if min_dist > best_score {
                        best_score = min_dist;
                        best = [a, b, c];
                    }
                }
            }
        }
    } else {
        // For many junctions, use farthest-point on the junction subset
        best[0] = junction_positions[0];
        best[1] = junction_positions.iter().copied()
            .max_by(|&a, &b| {
                arc_dist(&arc_lengths, total_len, best[0], a)
                    .partial_cmp(&arc_dist(&arc_lengths, total_len, best[0], b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(junction_positions[1]);
        best[2] = junction_positions.iter().copied()
            .filter(|&p| p != best[0] && p != best[1])
            .max_by(|&a, &b| {
                let da = arc_dist(&arc_lengths, total_len, best[0], a)
                    .min(arc_dist(&arc_lengths, total_len, best[1], a));
                let db = arc_dist(&arc_lengths, total_len, best[0], b)
                    .min(arc_dist(&arc_lengths, total_len, best[1], b));
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(junction_positions[2 % n]);
    }

    best.sort();
    best
}

/// Find a third corner that's farthest from two given corners.
fn find_farthest_from_two(
    positions: &[[f64; 3]],
    boundary: &[usize],
    c0: usize,
    c1: usize,
) -> usize {
    let n = boundary.len();
    let arc_lengths = compute_arc_lengths(positions, boundary);
    let total_len = total_arc_length_from(positions, boundary, &arc_lengths);

    (0..n)
        .filter(|&i| i != c0 && i != c1)
        .max_by(|&a, &b| {
            let da = arc_dist(&arc_lengths, total_len, c0, a)
                .min(arc_dist(&arc_lengths, total_len, c1, a));
            let db = arc_dist(&arc_lengths, total_len, c0, b)
                .min(arc_dist(&arc_lengths, total_len, c1, b));
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(n / 2)
}

fn compute_arc_lengths(positions: &[[f64; 3]], boundary: &[usize]) -> Vec<f64> {
    let n = boundary.len();
    let mut arc_lengths = vec![0.0; n];
    for i in 1..n {
        arc_lengths[i] = arc_lengths[i - 1]
            + geometry::vec3_dist(positions[boundary[i]], positions[boundary[i - 1]]);
    }
    arc_lengths
}

fn total_arc_length_from(positions: &[[f64; 3]], boundary: &[usize], arc_lengths: &[f64]) -> f64 {
    let n = boundary.len();
    if n < 2 { return 0.0; }
    arc_lengths[n - 1] + geometry::vec3_dist(positions[boundary[n - 1]], positions[boundary[0]])
}

fn arc_dist(arc_lengths: &[f64], total_len: f64, a: usize, b: usize) -> f64 {
    let d1 = (arc_lengths[b] - arc_lengths[a]).abs();
    let d2 = total_len - d1;
    d1.min(d2)
}

/// Fallback: Pick 3 well-separated corners on the boundary loop using farthest-point sampling.
fn farthest_point_corners(positions: &[[f64; 3]], boundary: &[usize]) -> [usize; 3] {
    let n = boundary.len();
    if n < 3 { return [0, n.min(1), n.min(2)]; }

    let arc_lengths = compute_arc_lengths(positions, boundary);
    let total_len = total_arc_length_from(positions, boundary, &arc_lengths);

    let c0 = 0;
    let c1 = (0..n).max_by(|&a, &b| {
        arc_dist(&arc_lengths, total_len, c0, a)
            .partial_cmp(&arc_dist(&arc_lengths, total_len, c0, b))
            .unwrap_or(std::cmp::Ordering::Equal)
    }).unwrap_or(n / 3);

    let c2 = (0..n)
        .filter(|&i| i != c0 && i != c1)
        .max_by(|&a, &b| {
            let da = arc_dist(&arc_lengths, total_len, c0, a)
                .min(arc_dist(&arc_lengths, total_len, c1, a));
            let db = arc_dist(&arc_lengths, total_len, c0, b)
                .min(arc_dist(&arc_lengths, total_len, c1, b));
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(n / 2);

    let mut corners = [c0, c1, c2];
    corners.sort();
    corners
}

/// Chain boundary half-edges into an ordered vertex loop.
fn chain_boundary_loop(boundary_he: &[u32], mesh: &HalfEdgeMesh) -> Option<Vec<usize>> {
    if boundary_he.is_empty() { return None; }

    let mut next_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &he_id in boundary_he {
        let (from, to) = mesh.edge_vertices(he_id);
        next_map.insert(from as usize, to as usize);
    }

    if next_map.is_empty() { return None; }

    let (from, _) = mesh.edge_vertices(boundary_he[0]);
    let start = from as usize;
    let mut loop_verts = Vec::new();
    let mut current = start;
    let max_steps = next_map.len() + 1;

    for _ in 0..max_steps {
        loop_verts.push(current);
        match next_map.get(&current) {
            Some(&next) => {
                if next == start { break; }
                current = next;
            }
            None => break,
        }
    }

    if loop_verts.len() >= 3 { Some(loop_verts) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farthest_point_corners_simple() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let boundary = vec![0, 1, 2, 3];
        let corners = farthest_point_corners(&positions, &boundary);
        assert_eq!(corners.len(), 3);
        assert!(corners[0] < corners[1] && corners[1] < corners[2]);
    }
}
