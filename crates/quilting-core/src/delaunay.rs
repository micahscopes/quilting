use delaunator::{triangulate, Point};

pub struct Triangulation {
    pub positions: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
}

pub fn triangulate_2d(points: &[[f64; 2]]) -> Triangulation {
    let del_points: Vec<Point> = points.iter().map(|p| Point { x: p[0], y: p[1] }).collect();

    let result = triangulate(&del_points);

    let triangles: Vec<[usize; 3]> = result
        .triangles
        .chunks(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();

    Triangulation {
        positions: points.to_vec(),
        triangles,
    }
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
