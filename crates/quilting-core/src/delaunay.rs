use crate::triangle;

pub struct Triangulation {
    pub positions: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
}

/// Constrained Delaunay triangulation within the reference triangle.
///
/// Uses barycentric coordinates to identify boundary points, builds a
/// contour around the reference triangle edges, and constrains the
/// triangulation so no triangle can cross the boundary — eliminating fins.
pub fn triangulate_2d_constrained(
    points: &[[f64; 2]],
    bary: &[[f64; 3]],
) -> Triangulation {
    if points.len() < 3 {
        return Triangulation { positions: points.to_vec(), triangles: vec![] };
    }

    const EDGE_EPS: f64 = 1e-6;

    // Classify each point: corner, edge interior, or interior.
    // bary[0] ≈ 0 → on edge BC (opposite vertex A)
    // bary[1] ≈ 0 → on edge AC (opposite vertex B)
    // bary[2] ≈ 0 → on edge AB (opposite vertex C)
    let mut idx_a = None; // bary ≈ (1,0,0)
    let mut idx_b = None; // bary ≈ (0,1,0)
    let mut idx_c = None; // bary ≈ (0,0,1)

    let mut edge_ab = Vec::new(); // bary[2] ≈ 0
    let mut edge_bc = Vec::new(); // bary[0] ≈ 0
    let mut edge_ca = Vec::new(); // bary[1] ≈ 0

    for (i, b) in bary.iter().enumerate() {
        let on_bc = b[0].abs() < EDGE_EPS;
        let on_ac = b[1].abs() < EDGE_EPS;
        let on_ab = b[2].abs() < EDGE_EPS;

        if on_ab && on_ac {
            idx_a = Some(i);
        } else if on_ab && on_bc {
            idx_b = Some(i);
        } else if on_bc && on_ac {
            idx_c = Some(i);
        } else if on_ab {
            edge_ab.push(i);
        } else if on_bc {
            edge_bc.push(i);
        } else if on_ac {
            edge_ca.push(i);
        }
        // else: interior point, no action needed
    }

    let a = idx_a.expect("missing vertex A on boundary");
    let b = idx_b.expect("missing vertex B on boundary");
    let c = idx_c.expect("missing vertex C on boundary");

    // Sort edge points by parameter along each edge.
    // Edge AB: A(1,0,0) → B(0,1,0), parameter = bary[1]
    edge_ab.sort_by(|&i, &j| bary[i][1].partial_cmp(&bary[j][1]).unwrap());
    // Edge BC: B(0,1,0) → C(0,0,1), parameter = bary[2]
    edge_bc.sort_by(|&i, &j| bary[i][2].partial_cmp(&bary[j][2]).unwrap());
    // Edge CA: C(0,0,1) → A(1,0,0), parameter = bary[0]
    edge_ca.sort_by(|&i, &j| bary[i][0].partial_cmp(&bary[j][0]).unwrap());

    // Build closed contour: A → B → C → A
    let mut contour = Vec::with_capacity(3 + edge_ab.len() + edge_bc.len() + edge_ca.len() + 1);
    contour.push(a);
    contour.extend_from_slice(&edge_ab);
    contour.push(b);
    contour.extend_from_slice(&edge_bc);
    contour.push(c);
    contour.extend_from_slice(&edge_ca);
    contour.push(a); // close

    let pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();

    let result = cdt::triangulate_contours(&pts, &[contour])
        .expect("CDT triangulation failed");

    let triangles: Vec<[usize; 3]> = result.iter()
        .map(|t| [t.0, t.1, t.2])
        .collect();

    Triangulation {
        positions: points.to_vec(),
        triangles,
    }
}

/// Constrained triangulation from cartesian points alone (computes bary internally).
/// Backward-compatible wrapper for callers that don't pass bary.
pub fn triangulate_2d_clipped(points: &[[f64; 2]]) -> Triangulation {
    let bary: Vec<[f64; 3]> = points.iter()
        .map(|&[x, y]| {
            let mut b = triangle::cartesian_to_bary(x, y);
            for c in &mut b {
                if c.abs() < 1e-10 { *c = 0.0; }
            }
            let sum = b[0] + b[1] + b[2];
            if sum > 0.0 { b[0] /= sum; b[1] /= sum; b[2] /= sum; }
            b
        })
        .collect();
    triangulate_2d_constrained(points, &bary)
}

/// Unconstrained Delaunay (for subdivision paths that don't need boundary constraints).
pub fn triangulate_2d(points: &[[f64; 2]]) -> Triangulation {
    let pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
    let result = cdt::triangulate_points(&pts)
        .expect("CDT triangulation failed");
    let triangles: Vec<[usize; 3]> = result.iter()
        .map(|t| [t.0, t.1, t.2])
        .collect();
    Triangulation { positions: points.to_vec(), triangles }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulate_square() {
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri = triangulate_2d(&points);
        assert_eq!(tri.triangles.len(), 2, "expected 2 triangles for a quad");
        for t in &tri.triangles {
            for &idx in t {
                assert!(idx < points.len(), "triangle index {} out of range", idx);
            }
        }
    }

    #[test]
    fn triangulate_triangle() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
        let tri = triangulate_2d(&points);
        assert_eq!(tri.triangles.len(), 1, "expected 1 triangle for 3 points");
        assert_eq!(tri.positions.len(), 3);
    }

    #[test]
    fn constrained_stays_inside() {
        use crate::sampling::{tri_patch, PatchConfig};
        let config = PatchConfig::default();
        let sample = tri_patch([4.0, 4.0, 4.0], &config);
        let tri = triangulate_2d_constrained(&sample.positions, &sample.bary);

        // Every triangle vertex must be inside (or on boundary of) reference triangle
        for t in &tri.triangles {
            for &idx in t {
                let [u, v, w] = triangle::cartesian_to_bary(
                    tri.positions[idx][0],
                    tri.positions[idx][1],
                );
                assert!(
                    u >= -1e-10 && v >= -1e-10 && w >= -1e-10,
                    "vertex {} outside triangle: bary=[{}, {}, {}]",
                    idx, u, v, w
                );
            }
        }

        // Every triangle centroid must be inside
        for t in &tri.triangles {
            let cx = (tri.positions[t[0]][0] + tri.positions[t[1]][0] + tri.positions[t[2]][0]) / 3.0;
            let cy = (tri.positions[t[0]][1] + tri.positions[t[1]][1] + tri.positions[t[2]][1]) / 3.0;
            let [u, v, w] = triangle::cartesian_to_bary(cx, cy);
            assert!(
                u >= -1e-10 && v >= -1e-10 && w >= -1e-10,
                "centroid outside triangle: bary=[{}, {}, {}]",
                u, v, w
            );
        }
    }
}
