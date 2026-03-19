use std::collections::HashMap;

/// Uniform midpoint subdivision: each triangle becomes 4 triangles.
/// Edge midpoints are shared, so the resulting mesh is watertight.
/// O(V + T) — no spatial index, no randomness.
pub fn subdivide(
    positions: &[[f64; 2]],
    triangles: &[[usize; 3]],
) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
    let mut new_positions = positions.to_vec();
    let mut new_triangles = Vec::with_capacity(triangles.len() * 4);
    let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();

    for &[a, b, c] in triangles {
        let ab = midpoint_index(a, b, &mut new_positions, &mut edge_midpoints);
        let bc = midpoint_index(b, c, &mut new_positions, &mut edge_midpoints);
        let ac = midpoint_index(a, c, &mut new_positions, &mut edge_midpoints);

        new_triangles.push([a, ab, ac]);
        new_triangles.push([ab, b, bc]);
        new_triangles.push([ac, bc, c]);
        new_triangles.push([ab, bc, ac]);
    }

    (new_positions, new_triangles)
}

fn midpoint_index(
    a: usize,
    b: usize,
    positions: &mut Vec<[f64; 2]>,
    cache: &mut HashMap<(usize, usize), usize>,
) -> usize {
    let key = if a < b { (a, b) } else { (b, a) };
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

/// Subdivide N times.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_triangle_subdivision() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let (pos, tris) = subdivide(&positions, &triangles);
        // 3 original + 3 midpoints = 6 vertices
        assert_eq!(pos.len(), 6);
        // 1 triangle -> 4 triangles
        assert_eq!(tris.len(), 4);
    }

    #[test]
    fn subdivision_preserves_boundary() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let (pos, _) = subdivide(&positions, &triangles);
        // Check midpoints are correct
        let midpoints: Vec<[f64; 2]> = pos[3..].to_vec();
        let expected = vec![[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];
        for exp in &expected {
            assert!(
                midpoints.iter().any(|m| (m[0] - exp[0]).abs() < 1e-12 && (m[1] - exp[1]).abs() < 1e-12),
                "missing midpoint {:?}", exp
            );
        }
    }

    #[test]
    fn double_subdivision() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let (pos, tris) = subdivide_n(&positions, &triangles, 2);
        // After 2 subdivisions: 1 -> 4 -> 16 triangles
        assert_eq!(tris.len(), 16);
        // Vertices: 3 -> 6 -> 15
        assert_eq!(pos.len(), 15);
    }

    #[test]
    fn shared_edges() {
        // Two triangles sharing an edge
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let triangles = vec![[0, 1, 2], [1, 3, 2]];

        let (pos, tris) = subdivide(&positions, &triangles);
        // 4 original + 5 unique edges = 9 vertices
        assert_eq!(pos.len(), 9);
        // 2 triangles -> 8 triangles
        assert_eq!(tris.len(), 8);
    }
}
