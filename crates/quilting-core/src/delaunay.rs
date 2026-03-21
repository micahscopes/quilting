use delaunator::{triangulate, Point};
use crate::triangle;

pub struct Triangulation {
    pub positions: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
}

/// General Delaunay triangulation (no filtering).
pub fn triangulate_2d(points: &[[f64; 2]]) -> Triangulation {
    let del_points: Vec<Point> = points.iter().map(|p| Point { x: p[0], y: p[1] }).collect();
    let result = triangulate(&del_points);
    let triangles: Vec<[usize; 3]> = result.triangles
        .chunks(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    Triangulation { positions: points.to_vec(), triangles }
}

/// Delaunay triangulation with exterior + sliver removal (default threshold 0.01).
pub fn triangulate_2d_clipped(points: &[[f64; 2]]) -> Triangulation {
    triangulate_2d_filtered(points, 0.01)
}

/// Delaunay triangulation with configurable sliver threshold.
/// `sliver_threshold`: compactness ratio below which triangles are removed.
/// 0.0 = keep all interior triangles, 0.01 = default, higher = more aggressive.
pub fn triangulate_2d_filtered(points: &[[f64; 2]], sliver_threshold: f64) -> Triangulation {
    let mut tri = triangulate_2d(points);
    tri.triangles.retain(|t| {
        let p0 = points[t[0]];
        let p1 = points[t[1]];
        let p2 = points[t[2]];

        // Reject triangles with ANY vertex outside the reference triangle
        for p in &[p0, p1, p2] {
            let [u, v, w] = triangle::cartesian_to_bary(p[0], p[1]);
            if u < -0.02 || v < -0.02 || w < -0.02 {
                return false;
            }
        }

        // Reject triangles with centroid outside
        let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
        let cy = (p0[1] + p1[1] + p2[1]) / 3.0;
        let [u, v, w] = triangle::cartesian_to_bary(cx, cy);
        if u < -0.01 || v < -0.01 || w < -0.01 {
            return false;
        }

        if sliver_threshold <= 0.0 {
            return true;
        }

        let e0_sq = (p1[0]-p0[0]).powi(2) + (p1[1]-p0[1]).powi(2);
        let e1_sq = (p2[0]-p1[0]).powi(2) + (p2[1]-p1[1]).powi(2);
        let e2_sq = (p0[0]-p2[0]).powi(2) + (p0[1]-p2[1]).powi(2);

        let area2 = ((p1[0]-p0[0]) * (p2[1]-p0[1]) - (p2[0]-p0[0]) * (p1[1]-p0[1])).abs();
        let perimeter = e0_sq.sqrt() + e1_sq.sqrt() + e2_sq.sqrt();

        if perimeter < 1e-15 { return false; }

        let compactness = area2 / (perimeter * perimeter);
        compactness > sliver_threshold
    });
    tri
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulate_square() {
        let points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri = triangulate_2d(&points);
        // 4 points in general position -> 2 triangles
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
    fn triangulate_many_points() {
        // Grid of points inside unit triangle
        let mut points = Vec::new();
        for i in 0..=5 {
            for j in 0..=(5 - i) {
                let x = i as f64 / 5.0;
                let y = j as f64 / 5.0;
                points.push([x, y]);
            }
        }
        let tri = triangulate_2d(&points);
        assert!(tri.triangles.len() > 0, "should produce triangles");
        // Euler: for n points with h on hull, triangles = 2n - h - 2
        // Just verify all indices are valid
        for t in &tri.triangles {
            for &idx in t {
                assert!(idx < points.len(), "index {} out of range", idx);
            }
        }
    }
}
