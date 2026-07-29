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

    /// Signed area of a 2D triangle; positive when the winding is CCW.
    fn signed_area(t: [[f64; 2]; 3]) -> f64 {
        let e1 = [t[1][0] - t[0][0], t[1][1] - t[0][1]];
        let e2 = [t[2][0] - t[0][0], t[2][1] - t[0][1]];
        0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
    }

    /// `perm_sign` has to mean something geometric, not just count inversions:
    /// odd permutations are reflections of the reference triangle, so remapping
    /// its vertices through one reverses the winding order.
    ///
    /// This is what SPEC invariant 4 turns on. The renderer feeds `perm_sign`
    /// to the shader as `perm_parity`; if it disagreed with the actual winding
    /// of the remapped tessellation, every face using an odd permutation would
    /// light with an inside-out normal.
    #[test]
    fn odd_permutations_reverse_winding() {
        let reference = [VERTEX_A, VERTEX_B, VERTEX_C];
        assert!(signed_area(reference) > 0.0, "reference triangle should be CCW");

        for perm_idx in 0..S3_PERMUTATIONS.len() {
            let remapped = [
                remap_position(perm_idx, VERTEX_A),
                remap_position(perm_idx, VERTEX_B),
                remap_position(perm_idx, VERTEX_C),
            ];
            let area = signed_area(remapped);
            // Same shape, so only the orientation can change.
            assert!(
                (area.abs() - signed_area(reference)).abs() < 1e-12,
                "perm {perm_idx} changed triangle area: {area}"
            );
            let winding = if area > 0.0 { 1 } else { -1 };
            assert_eq!(
                winding,
                perm_sign(perm_idx),
                "perm {perm_idx} ({:?}) has sign {} but remaps to winding {winding}",
                S3_PERMUTATIONS[perm_idx],
                perm_sign(perm_idx),
            );
        }
    }

    #[test]
    fn perm_sign_matches_the_group_structure() {
        // Identity and the two 3-cycles are even; the three transpositions odd.
        assert_eq!(perm_sign(0), 1, "identity");
        assert_eq!(perm_sign(1), -1, "swap 1,2");
        assert_eq!(perm_sign(2), -1, "swap 0,1");
        assert_eq!(perm_sign(3), 1, "120 deg rotation");
        assert_eq!(perm_sign(4), 1, "240 deg rotation");
        assert_eq!(perm_sign(5), -1, "swap 0,2");

        // sign is a homomorphism: sign(a∘b) = sign(a)·sign(b).
        for (i, &a) in S3_PERMUTATIONS.iter().enumerate() {
            for (j, &b) in S3_PERMUTATIONS.iter().enumerate() {
                let composed = [a[b[0]], a[b[1]], a[b[2]]];
                let k = S3_PERMUTATIONS.iter().position(|p| *p == composed).unwrap();
                assert_eq!(
                    perm_sign(k),
                    perm_sign(i) * perm_sign(j),
                    "sign not multiplicative for {i} o {j}"
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
