use serde::{Deserialize, Serialize};
use crate::triangle;

/// The 6 permutations of S3 (symmetric group on 3 elements).
///
/// On the equilateral reference triangle, these correspond to geometric
/// symmetries: cyclic perms → 120°/240° rotations, transpositions → reflections.
pub const S3_PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2], // identity
    [0, 2, 1], // swap 1,2 (reflect across altitude from A)
    [1, 0, 2], // swap 0,1 (reflect across altitude from C)
    [1, 2, 0], // cycle right (120° rotation)
    [2, 0, 1], // cycle left (240° rotation)
    [2, 1, 0], // swap 0,2 (reflect across altitude from B)
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalKey {
    pub res: [u32; 3],
    pub perm_index: usize,
}

/// Given edge resolutions (a, b, c), find the canonical (sorted) form
/// and the permutation index that maps canonical back to original.
pub fn canonical_form(res: [u32; 3]) -> CanonicalKey {
    let mut sorted = res;
    sorted.sort();

    for (idx, perm) in S3_PERMUTATIONS.iter().enumerate() {
        let permuted = [sorted[perm[0]], sorted[perm[1]], sorted[perm[2]]];
        if permuted == res {
            return CanonicalKey {
                res: sorted,
                perm_index: idx,
            };
        }
    }

    CanonicalKey {
        res: sorted,
        perm_index: 0,
    }
}

/// Apply a permutation to vertex indices.
pub fn apply_perm(perm_index: usize, verts: [usize; 3]) -> [usize; 3] {
    let perm = S3_PERMUTATIONS[perm_index];
    [verts[perm[0]], verts[perm[1]], verts[perm[2]]]
}

/// Get inverse permutation index.
pub fn inverse_perm(perm_index: usize) -> usize {
    let perm = S3_PERMUTATIONS[perm_index];
    for (idx, candidate) in S3_PERMUTATIONS.iter().enumerate() {
        let composed = [
            candidate[perm[0]],
            candidate[perm[1]],
            candidate[perm[2]],
        ];
        if composed == [0, 1, 2] {
            return idx;
        }
    }
    0
}

/// Returns +1 for even permutations (rotations), -1 for odd (reflections).
pub fn perm_sign(perm_index: usize) -> i32 {
    // Even: identity [0,1,2], cycles [1,2,0] [2,0,1]
    // Odd: transpositions [0,2,1] [1,0,2] [2,1,0]
    let perm = S3_PERMUTATIONS[perm_index];
    let inversions = (0..3).flat_map(|i| (i+1..3).map(move |j| (i, j)))
        .filter(|&(i, j)| perm[i] > perm[j])
        .count();
    if inversions % 2 == 0 { 1 } else { -1 }
}

/// Remap positions under a triangle vertex permutation.
///
/// Converts to barycentric, permutes the bary coords, converts back.
/// On the equilateral triangle this corresponds to actual rotations/reflections.
pub fn remap_position(perm_index: usize, pos: [f64; 2]) -> [f64; 2] {
    let bary = triangle::cartesian_to_bary(pos[0], pos[1]);
    let perm = S3_PERMUTATIONS[perm_index];
    let new_bary = [bary[perm[0]], bary[perm[1]], bary[perm[2]]];
    triangle::bary_to_cartesian(new_bary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::{VERTICES, VERTEX_A, VERTEX_B, VERTEX_C};

    #[test]
    fn s3_group_closure() {
        for &a in &S3_PERMUTATIONS {
            for &b in &S3_PERMUTATIONS {
                let composed = [a[b[0]], a[b[1]], a[b[2]]];
                assert!(
                    S3_PERMUTATIONS.contains(&composed),
                    "composition {:?} o {:?} = {:?} not in S3",
                    a, b, composed
                );
            }
        }
    }

    #[test]
    fn canonical_always_sorted() {
        let cases = [[4, 8, 2], [8, 4, 2], [2, 4, 8], [8, 8, 4], [4, 4, 4]];
        for res in cases {
            let key = canonical_form(res);
            assert!(
                key.res[0] <= key.res[1] && key.res[1] <= key.res[2],
                "canonical {:?} not sorted for input {:?}",
                key.res, res
            );
        }
    }

    #[test]
    fn canonical_roundtrip() {
        let cases = [[4, 8, 2], [8, 4, 16], [2, 2, 8], [4, 4, 4]];
        for res in cases {
            let key = canonical_form(res);
            let perm = S3_PERMUTATIONS[key.perm_index];
            let recovered = [key.res[perm[0]], key.res[perm[1]], key.res[perm[2]]];
            assert_eq!(recovered, res);
        }
    }

    #[test]
    fn inverse_perm_is_inverse() {
        for (idx, _) in S3_PERMUTATIONS.iter().enumerate() {
            let inv = inverse_perm(idx);
            let perm = S3_PERMUTATIONS[idx];
            let inv_perm = S3_PERMUTATIONS[inv];
            let composed = [inv_perm[perm[0]], inv_perm[perm[1]], inv_perm[perm[2]]];
            assert_eq!(composed, [0, 1, 2]);
        }
    }

    #[test]
    fn remap_identity() {
        // A point inside the equilateral triangle
        let pos = [0.1, 0.2];
        let remapped = remap_position(0, pos);
        assert!(
            (remapped[0] - pos[0]).abs() < 1e-12 && (remapped[1] - pos[1]).abs() < 1e-12,
            "identity remap changed position: {:?} -> {:?}",
            pos, remapped
        );
    }

    #[test]
    fn remap_preserves_triangle() {
        let positions = [
            VERTEX_A, VERTEX_B, VERTEX_C,
            [0.0, 0.0],    // centroid
            [0.0, -0.5],   // midpoint of BC
            [0.1, 0.2],
        ];
        for perm_idx in 0..6 {
            for &pos in &positions {
                let [x, y] = remap_position(perm_idx, pos);
                assert!(
                    triangle::contains(x + 1e-10, y + 1e-10)
                        || triangle::contains(x - 1e-10, y - 1e-10)
                        || triangle::contains(x, y),
                    "remap({}, {:?}) = [{}, {}] outside equilateral triangle",
                    perm_idx, pos, x, y
                );
            }
        }
    }

    #[test]
    fn remap_vertices_permuted() {
        for perm_idx in 0..6 {
            let remapped: Vec<[f64; 2]> = VERTICES
                .iter()
                .map(|&v| remap_position(perm_idx, v))
                .collect();
            for r in &remapped {
                let close_to_vertex = VERTICES.iter().any(|v| {
                    (r[0] - v[0]).abs() < 1e-10 && (r[1] - v[1]).abs() < 1e-10
                });
                assert!(
                    close_to_vertex,
                    "remap({}) produced {:?} which isn't a vertex",
                    perm_idx, r
                );
            }
        }
    }

    #[test]
    fn remap_120_rotation_is_cyclic() {
        // Perm [1,2,0] (cycle right) should be a 120° rotation
        let p = [0.1, 0.3];
        let r1 = remap_position(3, p); // [1,2,0]
        let r2 = remap_position(3, r1);
        let r3 = remap_position(3, r2);
        // Three applications should return to original
        assert!(
            (r3[0] - p[0]).abs() < 1e-10 && (r3[1] - p[1]).abs() < 1e-10,
            "3x 120° rotation didn't return to start: {:?} -> {:?}",
            p, r3
        );
    }
}
