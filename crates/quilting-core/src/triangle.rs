/// Reference equilateral triangle geometry.
///
/// Vertices inscribed in the unit circle, centered at the origin:
///   A = (0, 1)          — top
///   B = (-√3/2, -1/2)   — bottom-left
///   C = (√3/2, -1/2)    — bottom-right
///
/// Side length = √3. S3 permutations correspond to geometric symmetries:
/// cyclic permutations → 120°/240° rotations, transpositions → reflections.

pub const SQRT3: f64 = 1.732_050_808_068_887_2;
pub const SQRT3_OVER_2: f64 = 0.866_025_403_784_438_6;
pub const ONE_OVER_SQRT3: f64 = 0.577_350_269_189_625_8;

pub const VERTEX_A: [f64; 2] = [0.0, 1.0];
pub const VERTEX_B: [f64; 2] = [-SQRT3_OVER_2, -0.5];
pub const VERTEX_C: [f64; 2] = [SQRT3_OVER_2, -0.5];

pub const VERTICES: [[f64; 2]; 3] = [VERTEX_A, VERTEX_B, VERTEX_C];

/// Bounding box: x in [-√3/2, √3/2], y in [-0.5, 1.0]
pub const X_MIN: f64 = -SQRT3_OVER_2;
pub const X_MAX: f64 = SQRT3_OVER_2;
pub const Y_MIN: f64 = -0.5;
pub const Y_MAX: f64 = 1.0;
pub const WIDTH: f64 = SQRT3;
pub const HEIGHT: f64 = 1.5;

/// Convert 2D Cartesian to barycentric coordinates for the reference triangle.
#[inline]
pub fn cartesian_to_bary(x: f64, y: f64) -> [f64; 3] {
    let u = (1.0 + 2.0 * y) / 3.0;
    let v = (1.0 - y) / 3.0 - x * ONE_OVER_SQRT3;
    let w = (1.0 - y) / 3.0 + x * ONE_OVER_SQRT3;
    [u, v, w]
}

/// Convert barycentric coordinates to 2D Cartesian for the reference triangle.
#[inline]
pub fn bary_to_cartesian(bary: [f64; 3]) -> [f64; 2] {
    let x = SQRT3_OVER_2 * (bary[2] - bary[1]);
    let y = (3.0 * bary[0] - 1.0) / 2.0;
    [x, y]
}

/// Point-in-triangle test via barycentric coordinates.
#[inline]
pub fn contains(x: f64, y: f64) -> bool {
    let [u, v, w] = cartesian_to_bary(x, y);
    u >= 0.0 && v >= 0.0 && w >= 0.0
}

/// Linear interpolation between two 2D points.
#[inline]
pub fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertices_on_unit_circle() {
        for v in &VERTICES {
            let r = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "vertex {:?} not on unit circle: r={}", v, r);
        }
    }

    #[test]
    fn bary_roundtrip() {
        let test_points = [
            [0.0, 0.0],   // centroid
            VERTEX_A,
            VERTEX_B,
            VERTEX_C,
            [0.0, -0.5],  // midpoint of BC
            [0.2, 0.3],
        ];
        for &p in &test_points {
            let bary = cartesian_to_bary(p[0], p[1]);
            let back = bary_to_cartesian(bary);
            assert!(
                (back[0] - p[0]).abs() < 1e-12 && (back[1] - p[1]).abs() < 1e-12,
                "roundtrip failed for {:?}: bary={:?}, back={:?}", p, bary, back
            );
        }
    }

    #[test]
    fn vertex_bary_coords() {
        // A should be [1,0,0], B=[0,1,0], C=[0,0,1]
        let ba = cartesian_to_bary(VERTEX_A[0], VERTEX_A[1]);
        assert!((ba[0] - 1.0).abs() < 1e-12 && ba[1].abs() < 1e-12 && ba[2].abs() < 1e-12);

        let bb = cartesian_to_bary(VERTEX_B[0], VERTEX_B[1]);
        assert!(bb[0].abs() < 1e-12 && (bb[1] - 1.0).abs() < 1e-12 && bb[2].abs() < 1e-12);

        let bc = cartesian_to_bary(VERTEX_C[0], VERTEX_C[1]);
        assert!(bc[0].abs() < 1e-12 && bc[1].abs() < 1e-12 && (bc[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn centroid_bary() {
        let bary = cartesian_to_bary(0.0, 0.0);
        for &b in &bary {
            assert!((b - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn containment() {
        assert!(contains(0.0, 0.0)); // centroid
        assert!(contains(0.0, 0.9)); // near A
        assert!(!contains(0.0, 1.1)); // above A
        assert!(!contains(1.0, 0.0)); // outside right
        assert!(!contains(-1.0, 0.0)); // outside left
    }
}
