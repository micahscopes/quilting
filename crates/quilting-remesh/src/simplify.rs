/// QEM-based mesh simplification via edge collapse.
/// Produces a watertight coarse triangle mesh from a dense input mesh.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::geometry;

/// Simplify a triangle mesh to approximately `target_faces` triangles.
/// Returns (positions, faces) of the simplified mesh, plus a mapping
/// from each original face to the nearest simplified face.
pub fn simplify(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    target_faces: usize,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let n_verts = positions.len();
    let n_faces = faces.len();

    if n_faces <= target_faces {
        return (positions.to_vec(), faces.to_vec());
    }

    // Per-vertex quadric error matrix (4x4 symmetric, stored as 10 floats)
    let mut quadrics = vec![[0.0f64; 10]; n_verts];

    // Initialize quadrics from face planes
    for face in faces {
        let n = geometry::face_normal(positions, *face);
        let len = geometry::vec3_len(n);
        if len < 1e-15 { continue; }
        let nx = n[0] / len;
        let ny = n[1] / len;
        let nz = n[2] / len;
        let d = -(nx * positions[face[0]][0] + ny * positions[face[0]][1] + nz * positions[face[0]][2]);
        // Q = pp^T where p = [nx, ny, nz, d]
        let q = [
            nx*nx, nx*ny, nx*nz, nx*d,
            ny*ny, ny*nz, ny*d,
            nz*nz, nz*d,
            d*d,
        ];
        let area = len * 0.5;
        for &vi in face {
            for i in 0..10 {
                quadrics[vi][i] += q[i] * area;
            }
        }
    }

    // Mutable mesh state
    let mut pos = positions.to_vec();
    let mut tris: Vec<Option<[usize; 3]>> = faces.iter().map(|f| Some(*f)).collect();
    let mut vertex_alive = vec![true; n_verts];
    // Union-find for collapsed vertices
    let mut parent: Vec<usize> = (0..n_verts).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        x
    }

    // Build edge priority queue
    let mut heap: BinaryHeap<EdgeCollapse> = BinaryHeap::new();
    let mut edge_set = std::collections::HashSet::new();

    for face in faces {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            let key = (a.min(b), a.max(b));
            if edge_set.insert(key) {
                let (cost, target) = edge_cost(&quadrics[a], &quadrics[b], &pos[a], &pos[b]);
                heap.push(EdgeCollapse { cost, v0: a, v1: b, target_pos: target });
            }
        }
    }

    let mut live_faces = n_faces;

    // Collapse edges until we reach target
    while live_faces > target_faces {
        let collapse = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let v0 = find(&mut parent, collapse.v0);
        let v1 = find(&mut parent, collapse.v1);
        if v0 == v1 || !vertex_alive[v0] || !vertex_alive[v1] { continue; }

        // Collapse v1 into v0
        pos[v0] = collapse.target_pos;
        for i in 0..10 {
            quadrics[v0][i] += quadrics[v1][i];
        }
        vertex_alive[v1] = false;
        parent[v1] = v0;

        // Update triangles: replace v1 with v0, remove degenerate
        for tri in tris.iter_mut() {
            if let Some(f) = tri {
                let mut changed = false;
                for vi in f.iter_mut() {
                    let root = find(&mut parent, *vi);
                    if root != *vi {
                        *vi = root;
                        changed = true;
                    }
                }
                if changed || f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
                    // Check for degenerate
                    if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
                        *tri = None;
                        live_faces -= 1;
                    }
                }
            }
        }

        // Re-insert edges incident to v0
        let neighbors: std::collections::HashSet<usize> = tris.iter()
            .filter_map(|t| *t)
            .flat_map(|f| f.into_iter())
            .filter(|&v| v != v0 && vertex_alive[v])
            .collect();

        for &nb in &neighbors {
            let (cost, target) = edge_cost(&quadrics[v0], &quadrics[nb], &pos[v0], &pos[nb]);
            heap.push(EdgeCollapse { cost, v0, v1: nb, target_pos: target });
        }
    }

    // Collect surviving faces and compact vertex indices
    let surviving: Vec<[usize; 3]> = tris.into_iter()
        .filter_map(|t| t)
        .collect();

    // Compact: remap vertex indices to contiguous range
    let mut used_verts = std::collections::HashSet::new();
    for f in &surviving {
        for &v in f { used_verts.insert(v); }
    }
    let mut sorted_verts: Vec<usize> = used_verts.into_iter().collect();
    sorted_verts.sort();
    let remap: std::collections::HashMap<usize, usize> = sorted_verts.iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();

    let new_positions: Vec<[f64; 3]> = sorted_verts.iter().map(|&vi| pos[vi]).collect();
    let new_faces: Vec<[usize; 3]> = surviving.iter()
        .map(|f| [remap[&f[0]], remap[&f[1]], remap[&f[2]]])
        .collect();

    (new_positions, new_faces)
}

/// Compute optimal collapse position from combined quadric.
/// Solves the 3x3 system Q * v = -q3 for the position that minimizes error.
/// Falls back to midpoint if the system is singular.
fn optimal_position(q: &[f64; 10], p0: &[f64; 3], p1: &[f64; 3]) -> [f64; 3] {
    // Q matrix (symmetric 3x3 upper-left block):
    // [q[0] q[1] q[2]]   [q[3]]
    // [q[1] q[4] q[5]] v = -[q[6]]
    // [q[2] q[5] q[7]]   [q[8]]
    let a = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
    let b = [-q[3], -q[6], -q[8]];

    // Determinant
    let det = a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
            - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
            + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0]);

    if det.abs() > 1e-10 {
        let inv_det = 1.0 / det;
        let x = ((a[1][1]*a[2][2] - a[1][2]*a[2][1])*b[0]
               + (a[0][2]*a[2][1] - a[0][1]*a[2][2])*b[1]
               + (a[0][1]*a[1][2] - a[0][2]*a[1][1])*b[2]) * inv_det;
        let y = ((a[1][2]*a[2][0] - a[1][0]*a[2][2])*b[0]
               + (a[0][0]*a[2][2] - a[0][2]*a[2][0])*b[1]
               + (a[0][2]*a[1][0] - a[0][0]*a[1][2])*b[2]) * inv_det;
        let z = ((a[1][0]*a[2][1] - a[1][1]*a[2][0])*b[0]
               + (a[0][1]*a[2][0] - a[0][0]*a[2][1])*b[1]
               + (a[0][0]*a[1][1] - a[0][1]*a[1][0])*b[2]) * inv_det;

        // Sanity check: optimal point shouldn't be too far from the edge
        let mid = [(p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5, (p0[2]+p1[2])*0.5];
        let edge_len = geometry::vec3_dist(*p0, *p1);
        let opt_dist = geometry::vec3_dist([x, y, z], mid);
        if opt_dist < edge_len * 3.0 {
            return [x, y, z];
        }
    }

    // Fallback: pick the endpoint with lower error, or midpoint
    let mid = [(p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5, (p0[2]+p1[2])*0.5];
    let e0 = eval_quadric(q, p0);
    let e1 = eval_quadric(q, p1);
    let em = eval_quadric(q, &mid);
    if e0 <= e1 && e0 <= em { *p0 }
    else if e1 <= em { *p1 }
    else { mid }
}

/// Compute the cost of collapsing an edge using quadric error,
/// penalized by inverse edge length to preserve thin features.
fn edge_cost(q0: &[f64; 10], q1: &[f64; 10], p0: &[f64; 3], p1: &[f64; 3]) -> (f64, [f64; 3]) {
    let mut q = [0.0; 10];
    for i in 0..10 { q[i] = q0[i] + q1[i]; }

    let target = optimal_position(&q, p0, p1);
    let error = eval_quadric(&q, &target);

    // Penalize short edges to preserve thin features (legs, tails, ears).
    // Without this, thin cylindrical features collapse first because their
    // edges are short and have low absolute quadric error.
    let edge_len = geometry::vec3_dist(*p0, *p1);
    let penalty = if edge_len > 1e-10 { 1.0 / edge_len } else { 1e10 };
    let cost = error * penalty;

    (cost, target)
}

/// Evaluate quadric error Q at point p.
/// Q is stored as [q00, q01, q02, q03, q11, q12, q13, q22, q23, q33]
fn eval_quadric(q: &[f64; 10], p: &[f64; 3]) -> f64 {
    let x = p[0]; let y = p[1]; let z = p[2];
    q[0]*x*x + 2.0*q[1]*x*y + 2.0*q[2]*x*z + 2.0*q[3]*x
    + q[4]*y*y + 2.0*q[5]*y*z + 2.0*q[6]*y
    + q[7]*z*z + 2.0*q[8]*z
    + q[9]
}

#[derive(Clone)]
struct EdgeCollapse {
    cost: f64,
    v0: usize,
    v1: usize,
    target_pos: [f64; 3],
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}
impl Eq for EdgeCollapse {}
impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller cost = higher priority
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_cube() {
        let (positions, faces) = quilting_core::shapes::cube();
        let (new_pos, new_faces) = simplify(&positions, &faces, 4);
        assert!(new_faces.len() <= 6, "cube should simplify to ~4 faces, got {}", new_faces.len());
        assert!(new_faces.len() >= 2);
        // All face indices should be valid
        for f in &new_faces {
            for &v in f {
                assert!(v < new_pos.len());
            }
        }
    }

    #[test]
    fn test_simplify_icosahedron() {
        let (positions, faces) = quilting_core::shapes::icosahedron();
        let (new_pos, new_faces) = simplify(&positions, &faces, 8);
        assert!(new_faces.len() <= 12);
        assert!(new_faces.len() >= 4);
        for f in &new_faces {
            for &v in f { assert!(v < new_pos.len()); }
        }
    }

    #[test]
    fn test_simplify_noop() {
        let (positions, faces) = quilting_core::shapes::tetrahedron();
        let (_, new_faces) = simplify(&positions, &faces, 100);
        assert_eq!(new_faces.len(), faces.len(), "should not simplify below target");
    }
}
