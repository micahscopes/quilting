/// Compute interpolated spacing for a point inside a triangular patch.
///
/// `bary` = [x, y, z] barycentric coordinates.
///
/// `res` = [res_a, res_b, res_c] edge resolutions:
///   - res_a = resolution of edge A (the edge connecting vertices y,z)
///   - res_b = resolution of edge B (the edge connecting vertices x,z)
///   - res_c = resolution of edge C (the edge connecting vertices x,y)
///
/// Each edge's influence is the product of its two endpoint bary coords (y*z
/// for edge A, x*z for edge B, x*y for edge C). This peaks at the edge
/// midpoint and falls to zero at the opposite vertex — true edge-based
/// interpolation rather than vertex-based.
///
/// Resolution is blended via weighted geometric mean (interpolation in log
/// space). Since LODs are powers of 2, this linearly blends the exponent,
/// producing perceptually uniform density transitions.
///
/// Returns the local spacing (inverse of local density).
pub fn tri_edge_weight(bary: [f64; 3], res: [f64; 3]) -> f64 {
    let [x, y, z] = bary;
    // Edge influence = product of the two endpoint bary coords
    let e0 = y * z; // edge A
    let e1 = x * z; // edge B
    let e2 = x * y; // edge C
    let sum = e0 + e1 + e2;
    if sum < 1e-30 {
        // At a vertex, geometric mean of adjacent edges
        let max_res = res[0].max(res[1]).max(res[2]);
        return 1.0 / max_res;
    }
    // Weighted geometric mean: exp(Σ w_i * ln(res_i) / Σ w_i)
    let log_res = (e0 * res[0].ln() + e1 * res[1].ln() + e2 * res[2].ln()) / sum;
    1.0 / log_res.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri_edge_midpoint_dominated_by_edge() {
        // At midpoint of edge A (bary = [0, 0.5, 0.5]):
        // e0 = 0.5*0.5 = 0.25, e1 = 0*0.5 = 0, e2 = 0*0.5 = 0
        // Only edge A contributes → spacing = 1/res_a exactly
        let res = [10.0, 2.0, 2.0];
        let spacing = tri_edge_weight([0.0, 0.5, 0.5], res);
        assert!(
            (spacing - 1.0 / 10.0).abs() < 1e-12,
            "at edge A midpoint, spacing should be exactly 1/res_a, got {}",
            spacing
        );
    }

    #[test]
    fn tri_centroid_equal_contribution() {
        // At centroid with equal resolutions, result should match 1/res
        let res = [8.0, 8.0, 8.0];
        let spacing = tri_edge_weight([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], res);
        assert!((spacing - 1.0 / 8.0).abs() < 1e-12);
    }

    #[test]
    fn tri_symmetry() {
        let res_abc = [4.0, 8.0, 16.0];
        let bary = [0.2, 0.3, 0.5];
        let w1 = tri_edge_weight(bary, res_abc);
        // Swap y,z and swap res_b, res_c
        let w2 = tri_edge_weight([0.2, 0.5, 0.3], [4.0, 16.0, 8.0]);
        assert!(
            (w1 - w2).abs() < 1e-12,
            "symmetry broken: {} vs {}",
            w1,
            w2
        );
    }

    #[test]
    fn tri_opposite_vertex_gets_zero_influence() {
        // Near vertex x=1 (y→0, z→0): all edge products → 0
        // The function should handle this gracefully
        let res = [16.0, 2.0, 2.0];
        let spacing = tri_edge_weight([0.99, 0.005, 0.005], res);
        // Near vertex, all edges have low influence; should not crash
        assert!(spacing > 0.0 && spacing.is_finite());
    }
}
