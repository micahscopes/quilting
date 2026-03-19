use serde::{Deserialize, Serialize};

/// The 6 permutations of S3 (symmetric group on 3 elements).
pub const S3_PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2], // identity
    [0, 2, 1], // swap 1,2
    [1, 0, 2], // swap 0,1
    [1, 2, 0], // cycle right
    [2, 0, 1], // cycle left
    [2, 1, 0], // swap 0,2
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

    // Fallback (shouldn't happen for valid inputs)
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

/// Remap positions under a triangle vertex permutation.
///
/// The unit triangle vertices are A=(0,0), B=(1,0), C=(0,1).
/// Permuting vertices means remapping the 2D coordinates.
pub fn remap_position(perm_index: usize, pos: [f64; 2]) -> [f64; 2] {
    let [x, y] = pos;
    // Barycentric: u = 1-x-y, v = x, w = y
    let bary = [1.0 - x - y, x, y];
    let inv = inverse_perm(perm_index);
    let perm = S3_PERMUTATIONS[inv];
    let new_bary = [bary[perm[0]], bary[perm[1]], bary[perm[2]]];
    // Back to 2D: x = new_bary[1], y = new_bary[2]
    [new_bary[1], new_bary[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_group_closure() {
        // Composing any two permutations should yield another permutation in the group
        for &a in &S3_PERMUTATIONS {
            for &b in &S3_PERMUTATIONS {
                let composed = [a[b[0]], a[b[1]], a[b[2]]];
                assert!(
                    S3_PERMUTATIONS.contains(&composed),
                    "composition {:?} o {:?} = {:?} not in S3",
                    a,
                    b,
                    composed
                );
            }
        }
    }

    #[test]
    fn canonical_always_sorted() {
        let cases = [
            [4, 8, 2],
            [8, 4, 2],
            [2, 4, 8],
            [8, 8, 4],
            [4, 4, 4],
        ];
        for res in cases {
            let key = canonical_form(res);
            assert!(
                key.res[0] <= key.res[1] && key.res[1] <= key.res[2],
                "canonical {:?} not sorted for input {:?}",
                key.res,
                res
            );
        }
    }

    #[test]
    fn canonical_roundtrip() {
        let cases = [
            [4, 8, 2],
            [8, 4, 16],
            [2, 2, 8],
            [4, 4, 4],
        ];
        for res in cases {
            let key = canonical_form(res);
            let perm = S3_PERMUTATIONS[key.perm_index];
            let recovered = [key.res[perm[0]], key.res[perm[1]], key.res[perm[2]]];
            assert_eq!(
                recovered, res,
                "roundtrip failed for {:?}: canonical={:?}, perm_index={}, recovered={:?}",
                res, key.res, key.perm_index, recovered
            );
        }
    }

    #[test]
    fn inverse_perm_is_inverse() {
        for (idx, _) in S3_PERMUTATIONS.iter().enumerate() {
            let inv = inverse_perm(idx);
            let perm = S3_PERMUTATIONS[idx];
            let inv_perm = S3_PERMUTATIONS[inv];
            let composed = [inv_perm[perm[0]], inv_perm[perm[1]], inv_perm[perm[2]]];
            assert_eq!(
                composed,
                [0, 1, 2],
                "inverse_perm({}) = {} is not the inverse",
                idx,
                inv
            );
        }
    }

    #[test]
    fn remap_identity() {
        let pos = [0.3, 0.2];
        let remapped = remap_position(0, pos); // identity permutation
        assert!(
            (remapped[0] - pos[0]).abs() < 1e-12 && (remapped[1] - pos[1]).abs() < 1e-12,
            "identity remap changed position: {:?} -> {:?}",
            pos,
            remapped
        );
    }

    #[test]
    fn remap_preserves_triangle() {
        // All remapped positions should stay inside the unit triangle
        let positions = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.3],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.25, 0.25],
        ];
        for perm_idx in 0..6 {
            for &pos in &positions {
                let [x, y] = remap_position(perm_idx, pos);
                assert!(
                    x >= -1e-10 && y >= -1e-10 && x + y <= 1.0 + 1e-10,
                    "remap({}, {:?}) = [{}, {}] outside triangle",
                    perm_idx,
                    pos,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn remap_vertices_permuted() {
        // Remapping should send triangle vertices to other triangle vertices
        let vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for perm_idx in 0..6 {
            let remapped: Vec<[f64; 2]> = vertices.iter().map(|&v| remap_position(perm_idx, v)).collect();
            // Each remapped vertex should be close to one of the original vertices
            for r in &remapped {
                let close_to_vertex = vertices.iter().any(|v| {
                    (r[0] - v[0]).abs() < 1e-10 && (r[1] - v[1]).abs() < 1e-10
                });
                assert!(
                    close_to_vertex,
                    "remap({}) produced {:?} which isn't a vertex",
                    perm_idx,
                    r
                );
            }
        }
    }
}
