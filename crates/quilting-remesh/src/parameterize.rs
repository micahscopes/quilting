/// Harmonic parameterization: maps a cluster onto the reference triangle.

use crate::cluster::Cluster;
use crate::geometry;
use crate::sparse::{CsrMatrix, solve_cg};
use quilting_mesh::HalfEdgeMesh;

/// Result of parameterizing a cluster.
#[derive(Debug)]
pub struct Parameterization {
    /// Barycentric coordinates for each vertex in the cluster (same order as cluster.vertex_indices).
    pub vertex_bary: Vec<[f64; 3]>,
}

/// Parameterize a cluster onto the reference triangle using harmonic mapping.
///
/// - Boundary vertices are mapped to the triangle boundary proportionally by arc length.
/// - Interior vertices are solved via cotangent Laplacian (conjugate gradient).
pub fn parameterize_cluster(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    cluster: &Cluster,
    _mesh: &HalfEdgeMesh,
) -> Result<Parameterization, String> {
    let n_verts = cluster.vertex_indices.len();
    if n_verts < 3 {
        return Err("cluster has fewer than 3 vertices".into());
    }

    // Build local index map: global vertex ID -> local index
    let mut global_to_local: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (li, &gi) in cluster.vertex_indices.iter().enumerate() {
        global_to_local.insert(gi, li);
    }

    // Classify vertices as boundary or interior
    let boundary_set: std::collections::HashSet<usize> = cluster.boundary_loop.iter().copied().collect();
    let is_boundary: Vec<bool> = cluster.vertex_indices.iter()
        .map(|vi| boundary_set.contains(vi))
        .collect();

    // Compute boundary barycentric coordinates
    let mut bary = vec![[0.0, 0.0, 0.0]; n_verts];
    assign_boundary_bary(positions, cluster, &global_to_local, &mut bary);

    // If no interior vertices, we're done
    let interior_indices: Vec<usize> = (0..n_verts).filter(|&i| !is_boundary[i]).collect();
    if interior_indices.is_empty() {
        return Ok(Parameterization { vertex_bary: bary });
    }

    // Build local interior index map
    let mut local_to_interior: Vec<usize> = vec![usize::MAX; n_verts];
    for (ii, &li) in interior_indices.iter().enumerate() {
        local_to_interior[li] = ii;
    }

    let n_interior = interior_indices.len();

    // Collect cluster-local triangles
    let local_tris: Vec<[usize; 3]> = cluster.face_indices.iter()
        .map(|&fi| {
            let tri = faces[fi];
            [
                *global_to_local.get(&tri[0]).unwrap(),
                *global_to_local.get(&tri[1]).unwrap(),
                *global_to_local.get(&tri[2]).unwrap(),
            ]
        })
        .collect();

    // Build cotangent Laplacian and solve for u and v coordinates
    // For each interior vertex i, the Laplacian equation is:
    //   sum_j w_ij * (u_i - u_j) = 0
    // where w_ij = (cot(alpha_ij) + cot(beta_ij)) / 2

    // Accumulate edge weights
    let mut edge_weights: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
    let local_positions: Vec<[f64; 3]> = cluster.vertex_indices.iter()
        .map(|&gi| positions[gi])
        .collect();

    for tri in &local_tris {
        for local_k in 0..3 {
            let i = tri[(local_k + 1) % 3];
            let j = tri[(local_k + 2) % 3];
            let cot = geometry::cotangent_at_vertex(&local_positions, *tri, local_k);
            // Clamp to prevent negative weights from very obtuse triangles
            let w = cot.max(0.01) * 0.5;
            let key = (i.min(j), i.max(j));
            *edge_weights.entry(key).or_insert(0.0) += w;
        }
    }

    // Build sparse system: A * x_interior = b
    let mut triplets = Vec::new();
    let mut rhs_u = vec![0.0; n_interior];
    let mut rhs_v = vec![0.0; n_interior];

    for (&(vi, vj), &w) in &edge_weights {
        let i_interior = local_to_interior[vi] != usize::MAX;
        let j_interior = local_to_interior[vj] != usize::MAX;

        if i_interior && j_interior {
            let ii = local_to_interior[vi];
            let jj = local_to_interior[vj];
            triplets.push((ii, ii, w));
            triplets.push((jj, jj, w));
            triplets.push((ii, jj, -w));
            triplets.push((jj, ii, -w));
        } else if i_interior && !j_interior {
            // j is boundary: move to RHS
            let ii = local_to_interior[vi];
            triplets.push((ii, ii, w));
            rhs_u[ii] += w * bary[vj][0];
            rhs_v[ii] += w * bary[vj][1];
        } else if !i_interior && j_interior {
            let jj = local_to_interior[vj];
            triplets.push((jj, jj, w));
            rhs_u[jj] += w * bary[vi][0];
            rhs_v[jj] += w * bary[vi][1];
        }
    }

    if triplets.is_empty() {
        // Fallback: project onto proxy plane
        return Ok(Parameterization { vertex_bary: bary });
    }

    let a = CsrMatrix::from_triplets(n_interior, &triplets);

    // Solve for u coordinates
    let mut sol_u = vec![0.0; n_interior];
    solve_cg(&a, &rhs_u, &mut sol_u, 200, 1e-8);

    // Solve for v coordinates
    let mut sol_v = vec![0.0; n_interior];
    solve_cg(&a, &rhs_v, &mut sol_v, 200, 1e-8);

    // Write interior results back
    for (ii, &li) in interior_indices.iter().enumerate() {
        let u = sol_u[ii].clamp(0.0, 1.0);
        let v = sol_v[ii].clamp(0.0, 1.0);
        let w = (1.0 - u - v).max(0.0);
        // Normalize to ensure sum = 1
        let sum = u + v + w;
        if sum > 1e-15 {
            bary[li] = [u / sum, v / sum, w / sum];
        } else {
            bary[li] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        }
    }

    Ok(Parameterization { vertex_bary: bary })
}

/// Assign barycentric coordinates to boundary vertices.
/// The boundary loop is divided into 3 arcs by the 3 corner vertices.
/// Corner 0 -> (1,0,0), Corner 1 -> (0,1,0), Corner 2 -> (0,0,1)
/// Intermediate boundary vertices get linearly interpolated by arc length.
fn assign_boundary_bary(
    positions: &[[f64; 3]],
    cluster: &Cluster,
    global_to_local: &std::collections::HashMap<usize, usize>,
    bary: &mut [[f64; 3]],
) {
    let loop_verts = &cluster.boundary_loop;
    let n = loop_verts.len();
    let corners = cluster.corner_loop_positions;

    // Assign corners
    if let Some(&li) = global_to_local.get(&cluster.corner_vertices[0]) {
        bary[li] = [1.0, 0.0, 0.0];
    }
    if let Some(&li) = global_to_local.get(&cluster.corner_vertices[1]) {
        bary[li] = [0.0, 1.0, 0.0];
    }
    if let Some(&li) = global_to_local.get(&cluster.corner_vertices[2]) {
        bary[li] = [0.0, 0.0, 1.0];
    }

    // Interpolate along each arc
    let arcs = [
        (corners[0], corners[1], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // edge 0->1
        (corners[1], corners[2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), // edge 1->2
        (corners[2], corners[0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), // edge 2->0
    ];

    for &(start_idx, end_idx, bary_start, bary_end) in &arcs {
        // Collect vertices along this arc
        let mut arc_verts = Vec::new();
        let mut idx = start_idx;
        loop {
            arc_verts.push(idx);
            if idx == end_idx { break; }
            idx = (idx + 1) % n;
            if arc_verts.len() > n + 1 { break; } // safety
        }

        if arc_verts.len() <= 2 { continue; } // just the two corners

        // Compute cumulative arc lengths
        let mut arc_lengths = vec![0.0];
        for i in 1..arc_verts.len() {
            let prev_gi = loop_verts[arc_verts[i - 1]];
            let curr_gi = loop_verts[arc_verts[i]];
            arc_lengths.push(arc_lengths[i - 1] + geometry::vec3_dist(positions[prev_gi], positions[curr_gi]));
        }
        let total = *arc_lengths.last().unwrap();
        if total < 1e-15 { continue; }

        // Interpolate interior boundary vertices
        for i in 1..arc_verts.len() - 1 {
            let t = arc_lengths[i] / total;
            let gi = loop_verts[arc_verts[i]];
            if let Some(&li) = global_to_local.get(&gi) {
                bary[li] = [
                    bary_start[0] * (1.0 - t) + bary_end[0] * t,
                    bary_start[1] * (1.0 - t) + bary_end[1] * t,
                    bary_start[2] * (1.0 - t) + bary_end[2] * t,
                ];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_bary_simple() {
        // A simple 4-vertex cluster with 1 interior vertex
        // Triangle: 0-1-2, 0-2-3 (vertex 3 is interior? No, let's make it simpler)
        // Actually, for a meaningful test we need interior vertices.
        // Just verify the function doesn't panic on a simple cluster.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];

        let cluster = Cluster {
            id: 0,
            face_indices: vec![0],
            vertex_indices: vec![0, 1, 2],
            boundary_loop: vec![0, 1, 2],
            corner_vertices: [0, 1, 2],
            corner_loop_positions: [0, 1, 2],
            interior_vertices: vec![],
        };

        let faces = vec![[0, 1, 2]];
        let mesh = quilting_mesh::HalfEdgeMesh::from_triangles(3, &[[0, 1, 2]]);

        let result = parameterize_cluster(&positions, &faces, &cluster, &mesh);
        assert!(result.is_ok());
        let param = result.unwrap();

        // Corner vertices should have exact bary coords
        assert!((param.vertex_bary[0][0] - 1.0).abs() < 1e-8); // vertex 0 -> (1,0,0)
        assert!((param.vertex_bary[1][1] - 1.0).abs() < 1e-8); // vertex 1 -> (0,1,0)
        assert!((param.vertex_bary[2][2] - 1.0).abs() < 1e-8); // vertex 2 -> (0,0,1)
    }
}
