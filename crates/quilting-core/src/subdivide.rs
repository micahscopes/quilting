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
}
