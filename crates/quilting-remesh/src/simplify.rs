/// QEM-based mesh simplification via edge collapse.
/// Produces a watertight coarse triangle mesh from a dense input mesh.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::geometry;

/// Simplify a triangle mesh to approximately `target_faces` triangles.
/// Uses cyclide-aware edge costs: edges where both vertices have similar
/// curvature (same cyclide region) are cheaper to collapse.
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

    // Compute curvatures for cyclide-aware cost
    let curvatures = crate::curvature::compute_vertex_curvatures(positions, faces);

    // Per-vertex quadric error matrix (symmetric 4x4, stored as 10 floats)
    let mut quadrics = vec![[0.0f64; 10]; n_verts];
    for face in faces {
        let n = geometry::face_normal(positions, *face);
        let len = geometry::vec3_len(n);
        if len < 1e-15 { continue; }
        let nx = n[0] / len; let ny = n[1] / len; let nz = n[2] / len;
        let d = -(nx * positions[face[0]][0] + ny * positions[face[0]][1] + nz * positions[face[0]][2]);
        let q = [nx*nx, nx*ny, nx*nz, nx*d, ny*ny, ny*nz, ny*d, nz*nz, nz*d, d*d];
        let area = len * 0.5;
        for &vi in face {
            for i in 0..10 { quadrics[vi][i] += q[i] * area; }
        }
    }

    let mut pos = positions.to_vec();
    let mut tris: Vec<Option<[usize; 3]>> = faces.iter().map(|f| Some(*f)).collect();
    let mut vertex_alive = vec![true; n_verts];
    let mut parent: Vec<usize> = (0..n_verts).collect();

    // Per-vertex adjacency: which face indices reference this vertex
    let mut vert_faces: Vec<Vec<usize>> = vec![Vec::new(); n_verts];
    for (fi, face) in faces.iter().enumerate() {
        for &v in face { vert_faces[v].push(fi); }
    }

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    // Build edge priority queue
    let mut heap: BinaryHeap<EdgeCollapse> = BinaryHeap::new();
    let mut edge_set = std::collections::HashSet::new();

    for face in faces {
        for i in 0..3 {
            let a = face[i]; let b = face[(i + 1) % 3];
            let key = (a.min(b), a.max(b));
            if edge_set.insert(key) {
                let (cost, target) = edge_cost(
                    &quadrics[a], &quadrics[b], &pos[a], &pos[b],
                    &curvatures[a], &curvatures[b],
                );
                heap.push(EdgeCollapse { cost, v0: a, v1: b, target_pos: target });
            }
        }
    }

    let mut live_faces = n_faces;

    while live_faces > target_faces {
        let collapse = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let v0 = find(&mut parent, collapse.v0);
        let v1 = find(&mut parent, collapse.v1);
        if v0 == v1 || !vertex_alive[v0] || !vertex_alive[v1] { continue; }

        // Check if collapse would invert any triangle normal
        let collapse_valid = check_collapse_valid(
            v0, v1, &collapse.target_pos, &pos, &tris, &vert_faces,
        );
        if !collapse_valid { continue; }

        // Collapse v1 into v0
        pos[v0] = collapse.target_pos;
        for i in 0..10 { quadrics[v0][i] += quadrics[v1][i]; }
        vertex_alive[v1] = false;
        parent[v1] = v0;

        // Merge v1's face list into v0's
        let v1_faces = std::mem::take(&mut vert_faces[v1]);
        vert_faces[v0].extend_from_slice(&v1_faces);

        // Update only affected triangles (those referencing v0 or v1)
        let affected: Vec<usize> = vert_faces[v0].clone();
        vert_faces[v0].clear();

        for &fi in &affected {
            if let Some(f) = &mut tris[fi] {
                // Update vertex references
                for vi in f.iter_mut() {
                    let root = find(&mut parent, *vi);
                    *vi = root;
                }
                // Remove degenerate triangles
                if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
                    tris[fi] = None;
                    live_faces -= 1;
                } else {
                    // Re-register in adjacency
                    for &v in f.iter() {
                        if !vert_faces[v].contains(&fi) {
                            vert_faces[v].push(fi);
                        }
                    }
                }
            }
        }

        // Re-insert edges incident to v0 with updated costs
        let mut neighbors = std::collections::HashSet::new();
        for &fi in &vert_faces[v0] {
            if let Some(f) = &tris[fi] {
                for &v in f {
                    if v != v0 && vertex_alive[v] { neighbors.insert(v); }
                }
            }
        }

        for &nb in &neighbors {
            let (cost, target) = edge_cost(
                &quadrics[v0], &quadrics[nb], &pos[v0], &pos[nb],
                &curvatures[v0], &curvatures[nb],
            );
            heap.push(EdgeCollapse { cost, v0, v1: nb, target_pos: target });
        }
    }

    // Collect surviving faces and compact
    let surviving: Vec<[usize; 3]> = tris.into_iter().filter_map(|t| t).collect();

    let mut used_verts = std::collections::HashSet::new();
    for f in &surviving { for &v in f { used_verts.insert(v); } }
    let mut sorted_verts: Vec<usize> = used_verts.into_iter().collect();
    sorted_verts.sort();
    let remap: std::collections::HashMap<usize, usize> = sorted_verts.iter()
        .enumerate().map(|(new, &old)| (old, new)).collect();

    let new_positions: Vec<[f64; 3]> = sorted_verts.iter().map(|&vi| pos[vi]).collect();
    let new_faces: Vec<[usize; 3]> = surviving.iter()
        .map(|f| [remap[&f[0]], remap[&f[1]], remap[&f[2]]])
        .collect();

    (new_positions, new_faces)
}

/// Check if collapsing v1 into v0 (placing at target_pos) would invert any triangle.
fn check_collapse_valid(
    v0: usize, v1: usize, target_pos: &[f64; 3],
    pos: &[[f64; 3]], tris: &[Option<[usize; 3]>], vert_faces: &[Vec<usize>],
) -> bool {
    // Check all triangles incident to v0 or v1 that will survive the collapse
    let check_faces = |v: usize| -> bool {
        for &fi in &vert_faces[v] {
            if let Some(f) = &tris[fi] {
                // Skip triangles that will become degenerate (contain both v0 and v1)
                let has_v0 = f.contains(&v0);
                let has_v1 = f.contains(&v1);
                if has_v0 && has_v1 { continue; }

                // Compute normal before collapse
                let p0 = pos[f[0]]; let p1 = pos[f[1]]; let p2 = pos[f[2]];
                let n_before = geometry::vec3_cross(
                    geometry::vec3_sub(p1, p0), geometry::vec3_sub(p2, p0),
                );

                // Compute normal after collapse (replace v0 or v1 with target_pos)
                let mut fp = [p0, p1, p2];
                for i in 0..3 {
                    if f[i] == v0 || f[i] == v1 { fp[i] = *target_pos; }
                }
                let n_after = geometry::vec3_cross(
                    geometry::vec3_sub(fp[1], fp[0]), geometry::vec3_sub(fp[2], fp[0]),
                );

                // If normal flips, reject this collapse
                if geometry::vec3_dot(n_before, n_after) < 0.0 {
                    return false;
                }

                // If triangle becomes degenerate (zero area), reject
                if geometry::vec3_len(n_after) < 1e-15 {
                    return false;
                }
            }
        }
        true
    };

    check_faces(v0) && check_faces(v1)
}

/// Compute optimal collapse position and cost.
/// Cost is scaled by cyclide compatibility: edges within the same cyclide
/// region (similar curvature) are cheaper to collapse. Edges at curvature
/// boundaries are expensive — they should become patch boundaries.
fn edge_cost(
    q0: &[f64; 10], q1: &[f64; 10], p0: &[f64; 3], p1: &[f64; 3],
    c0: &crate::curvature::VertexCurvature, c1: &crate::curvature::VertexCurvature,
) -> (f64, [f64; 3]) {
    let mut q = [0.0; 10];
    for i in 0..10 { q[i] = q0[i] + q1[i]; }

    let target = optimal_position(&q, p0, p1);
    let error = eval_quadric(&q, &target);

    // Penalize short edges to preserve thin features
    let edge_len = geometry::vec3_dist(*p0, *p1);
    let len_penalty = if edge_len > 1e-10 { 1.0 / edge_len } else { 1e10 };

    // Cyclide compatibility: high compatibility = low cost multiplier.
    // Edges at curvature boundaries get 5x higher cost.
    let compat = crate::curvature::cyclide_compatibility(c0, c1);
    let curvature_penalty = 1.0 + 4.0 * (1.0 - compat);

    (error * len_penalty * curvature_penalty, target)
}

fn optimal_position(q: &[f64; 10], p0: &[f64; 3], p1: &[f64; 3]) -> [f64; 3] {
    // Use the endpoint with lower quadric error.
    // This keeps vertices on the original surface instead of drifting inward
    // (midpoint of a chord on a sphere is inside the sphere).
    let e0 = eval_quadric(q, p0);
    let e1 = eval_quadric(q, p1);
    if e0 <= e1 { *p0 } else { *p1 }
}

fn eval_quadric(q: &[f64; 10], p: &[f64; 3]) -> f64 {
    let (x, y, z) = (p[0], p[1], p[2]);
    q[0]*x*x + 2.0*q[1]*x*y + 2.0*q[2]*x*z + 2.0*q[3]*x
    + q[4]*y*y + 2.0*q[5]*y*z + 2.0*q[6]*y
    + q[7]*z*z + 2.0*q[8]*z + q[9]
}

#[derive(Clone)]
struct EdgeCollapse {
    cost: f64,
    v0: usize,
    v1: usize,
    target_pos: [f64; 3],
}

impl PartialEq for EdgeCollapse { fn eq(&self, o: &Self) -> bool { self.cost == o.cost } }
impl Eq for EdgeCollapse {}
impl PartialOrd for EdgeCollapse { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for EdgeCollapse {
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_cube() {
        let (positions, faces) = quilting_core::shapes::cube();
        let (new_pos, new_faces) = simplify(&positions, &faces, 4);
        assert!(new_faces.len() <= 6, "got {}", new_faces.len());
        assert!(new_faces.len() >= 2);
        for f in &new_faces { for &v in f { assert!(v < new_pos.len()); } }
    }

    #[test]
    fn test_simplify_icosahedron() {
        let (positions, faces) = quilting_core::shapes::icosahedron();
        let (new_pos, new_faces) = simplify(&positions, &faces, 8);
        assert!(new_faces.len() <= 12);
        assert!(new_faces.len() >= 4);
        for f in &new_faces { for &v in f { assert!(v < new_pos.len()); } }
    }

    #[test]
    fn test_simplify_noop() {
        let (positions, faces) = quilting_core::shapes::tetrahedron();
        let (_, new_faces) = simplify(&positions, &faces, 100);
        assert_eq!(new_faces.len(), faces.len());
    }

    #[test]
    fn test_simplify_sphere() {
        let (positions, faces) = crate::test_shapes::sphere(2);
        assert_eq!(faces.len(), 320);
        let (new_pos, new_faces) = simplify(&positions, &faces, 20);
        assert!(new_faces.len() <= 30, "got {}", new_faces.len());
        for f in &new_faces { for &v in f { assert!(v < new_pos.len()); } }
    }
}
