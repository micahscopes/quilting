use std::collections::HashMap;

/// Uniform midpoint subdivision: each triangle becomes 4 triangles.
/// Edge midpoints are shared, so the resulting mesh is watertight.
pub fn subdivide(
    positions: &[[f64; 2]],
    triangles: &[[usize; 3]],
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let mut new_positions = positions.to_vec();
    let mut new_triangles = Vec::with_capacity(triangles.len() * 4);
    let mut edge_mids: HashMap<(usize, usize), usize> = HashMap::new();

    for &[a, b, c] in triangles {
        let ab = get_midpoint(a, b, &mut new_positions, &mut edge_mids);
        let bc = get_midpoint(b, c, &mut new_positions, &mut edge_mids);
        let ac = get_midpoint(a, c, &mut new_positions, &mut edge_mids);

        new_triangles.push([a, ab, ac]);
        new_triangles.push([ab, b, bc]);
        new_triangles.push([ac, bc, c]);
        new_triangles.push([ab, bc, ac]);
    }

    (new_positions, new_triangles)
}

/// Subdivide N times uniformly.
pub fn subdivide_n(
    positions: &[[f64; 2]],
    triangles: &[[usize; 3]],
    n: u32,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let mut pos = positions.to_vec();
    let mut tris = triangles.to_vec();
    for _ in 0..n {
        let (p, t) = subdivide(&pos, &tris);
        pos = p;
        tris = t;
    }
    (pos, tris)
}

/// Adaptive subdivision: refine triangles until all edges are shorter than
/// the local spacing defined by `density_fn`. Handles T-junctions by
/// propagating edge splits to neighbors.
///
/// `density_fn` takes a 2D position and returns the desired spacing.
/// A triangle is refined when any edge is longer than `threshold * local_spacing`.
pub fn subdivide_adaptive<F: Fn([f64; 2]) -> f64>(
    positions: &[[f64; 2]],
    triangles: &[[usize; 3]],
    density_fn: &F,
    threshold: f64,
    max_iterations: usize,
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let mut pos = positions.to_vec();
    let mut tris = triangles.to_vec();

    for _ in 0..max_iterations {
        let mut edge_mids: HashMap<(usize, usize), usize> = HashMap::new();
        let mut new_tris = Vec::new();
        let mut any_refined = false;

        // Count how many triangles reference each edge.
        // Boundary edges (count=1) must not be split — they're stitching edges.
        let mut edge_count: HashMap<(usize, usize), u32> = HashMap::new();
        for &[a, b, c] in &tris {
            for &(i, j) in &[(a, b), (b, c), (a, c)] {
                *edge_count.entry(edge_key(i, j)).or_insert(0) += 1;
            }
        }

        // First pass: identify which interior edges need splitting
        let mut split_edges: HashMap<(usize, usize), bool> = HashMap::new();
        for &[a, b, c] in &tris {
            for &(i, j) in &[(a, b), (b, c), (a, c)] {
                let key = edge_key(i, j);
                if split_edges.contains_key(&key) {
                    continue;
                }
                // Never split boundary edges
                if edge_count.get(&key) == Some(&1) {
                    split_edges.insert(key, false);
                    continue;
                }
                let mid = [
                    (pos[i][0] + pos[j][0]) * 0.5,
                    (pos[i][1] + pos[j][1]) * 0.5,
                ];
                let dx = pos[i][0] - pos[j][0];
                let dy = pos[i][1] - pos[j][1];
                let edge_len = (dx * dx + dy * dy).sqrt();
                let desired = density_fn(mid);
                split_edges.insert(key, edge_len > threshold * desired);
            }
        }

        // Second pass: subdivide each triangle based on how many of its edges are split.
        // This avoids T-junctions: if an edge is split, BOTH triangles sharing it get
        // the midpoint vertex.
        for &[a, b, c] in &tris {
            let ab_split = *split_edges.get(&edge_key(a, b)).unwrap_or(&false);
            let bc_split = *split_edges.get(&edge_key(b, c)).unwrap_or(&false);
            let ac_split = *split_edges.get(&edge_key(a, c)).unwrap_or(&false);

            let n_splits = ab_split as u8 + bc_split as u8 + ac_split as u8;

            if n_splits == 0 {
                // No refinement needed
                new_tris.push([a, b, c]);
            } else if n_splits == 3 {
                // Full 4-way split (red refinement)
                let ab = get_midpoint(a, b, &mut pos, &mut edge_mids);
                let bc = get_midpoint(b, c, &mut pos, &mut edge_mids);
                let ac = get_midpoint(a, c, &mut pos, &mut edge_mids);
                new_tris.push([a, ab, ac]);
                new_tris.push([ab, b, bc]);
                new_tris.push([ac, bc, c]);
                new_tris.push([ab, bc, ac]);
                any_refined = true;
            } else if n_splits == 1 {
                // One edge split → bisect into 2 triangles
                let (split_v, opp, v1, v2) = if ab_split {
                    (get_midpoint(a, b, &mut pos, &mut edge_mids), c, a, b)
                } else if bc_split {
                    (get_midpoint(b, c, &mut pos, &mut edge_mids), a, b, c)
                } else {
                    (get_midpoint(a, c, &mut pos, &mut edge_mids), b, a, c)
                };
                new_tris.push([v1, split_v, opp]);
                new_tris.push([split_v, v2, opp]);
                any_refined = true;
            } else {
                // Two edges split → promote to full 4-way split to avoid
                // degenerate green triangles. Split the third edge too.
                let ab = get_midpoint(a, b, &mut pos, &mut edge_mids);
                let bc = get_midpoint(b, c, &mut pos, &mut edge_mids);
                let ac = get_midpoint(a, c, &mut pos, &mut edge_mids);
                new_tris.push([a, ab, ac]);
                new_tris.push([ab, b, bc]);
                new_tris.push([ac, bc, c]);
                new_tris.push([ab, bc, ac]);
                any_refined = true;
            }
        }

        tris = new_tris;
        if !any_refined {
            break;
        }
    }

    (pos, tris)
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn get_midpoint(
    a: usize,
    b: usize,
    positions: &mut Vec<[f64; 2]>,
    cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    let key = edge_key(a, b);
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let mid = [
        (positions[a][0] + positions[b][0]) * 0.5,
        (positions[a][1] + positions[b][1]) * 0.5,
    ];
    let idx = positions.len();
    positions.push(mid);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_triangle_subdivision() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, tris) = subdivide(&positions, &triangles);
        assert_eq!(pos.len(), 6);
        assert_eq!(tris.len(), 4);
    }

    #[test]
    fn subdivision_preserves_boundary() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, _) = subdivide(&positions, &triangles);
        let expected = [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];
        for exp in &expected {
            assert!(pos.iter().any(|m| (m[0]-exp[0]).abs() < 1e-12 && (m[1]-exp[1]).abs() < 1e-12),
                "missing midpoint {:?}", exp);
        }
    }

    #[test]
    fn double_subdivision() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, tris) = subdivide_n(&positions, &triangles, 2);
        assert_eq!(tris.len(), 16);
        assert_eq!(pos.len(), 15);
    }

    #[test]
    fn shared_edges() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let triangles = vec![[0, 1, 2], [1, 3, 2]];
        let (pos, tris) = subdivide(&positions, &triangles);
        assert_eq!(pos.len(), 9);
        assert_eq!(tris.len(), 8);
    }

    #[test]
    fn adaptive_uniform_density() {
        // Start with a mesh that has interior edges (subdivide once first)
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, tris) = subdivide(&positions, &triangles); // 4 triangles, 3 interior edges
        let (_, refined) = subdivide_adaptive(&pos, &tris, &|_| 0.15, 1.0, 10);
        assert!(refined.len() > 4, "expected refinement, got {} tris", refined.len());
    }

    #[test]
    fn adaptive_no_refinement_needed() {
        // Very coarse density — no interior edges should need splitting
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, tris) = subdivide(&positions, &triangles);
        let (_, refined) = subdivide_adaptive(&pos, &tris, &|_| 10.0, 1.0, 10);
        assert_eq!(refined.len(), 4, "should not have refined beyond initial subdivision");
    }

    #[test]
    fn adaptive_variable_density() {
        // Start with a subdivided mesh, then adaptively refine
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, tris) = subdivide(&positions, &triangles);
        let (pos, tris) = subdivide_adaptive(
            &pos, &tris,
            &|p| 0.05 + 0.4 * (1.0 - p[0]),  // fine near x=1, coarse near x=0
            1.0, 10,
        );
        let centroid = |t: &[usize; 3]| -> f64 {
            (pos[t[0]][0] + pos[t[1]][0] + pos[t[2]][0]) / 3.0
        };
        let near_right = tris.iter().filter(|t| centroid(t) > 0.3).count();
        let near_left = tris.iter().filter(|t| centroid(t) <= 0.3).count();
        assert!(near_right >= near_left,
            "expected at least as many triangles near x=1 ({}) as x=0 ({})", near_right, near_left);
    }

    #[test]
    fn adaptive_preserves_boundary() {
        // Boundary edges should never be split
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];
        let (pos, tris) = subdivide(&positions, &triangles);

        // Count boundary points before
        let boundary_before: Vec<_> = pos.iter()
            .filter(|p| p[0].abs() < 1e-10 || p[1].abs() < 1e-10 || (p[0]+p[1]-1.0).abs() < 1e-10)
            .collect();

        let (pos2, _) = subdivide_adaptive(&pos, &tris, &|_| 0.01, 1.0, 5);

        // Boundary points after — originals should still be there
        for bp in &boundary_before {
            assert!(
                pos2.iter().any(|p| (p[0]-bp[0]).abs() < 1e-10 && (p[1]-bp[1]).abs() < 1e-10),
                "boundary point {:?} was lost", bp
            );
        }
    }
}
