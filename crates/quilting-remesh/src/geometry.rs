/// 3D geometry utilities for mesh processing.

#[inline]
pub fn vec3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn vec3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn vec3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn vec3_len(a: [f64; 3]) -> f64 {
    vec3_dot(a, a).sqrt()
}

#[inline]
pub fn vec3_normalize(a: [f64; 3]) -> [f64; 3] {
    let l = vec3_len(a);
    if l < 1e-15 { return [0.0, 0.0, 0.0]; }
    vec3_scale(a, 1.0 / l)
}

#[inline]
pub fn vec3_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    vec3_len(vec3_sub(a, b))
}

/// Compute the (unnormalized) face normal for a triangle.
pub fn face_normal(positions: &[[f64; 3]], tri: [usize; 3]) -> [f64; 3] {
    let e1 = vec3_sub(positions[tri[1]], positions[tri[0]]);
    let e2 = vec3_sub(positions[tri[2]], positions[tri[0]]);
    vec3_cross(e1, e2)
}

/// Compute face normal (normalized).
pub fn face_normal_normalized(positions: &[[f64; 3]], tri: [usize; 3]) -> [f64; 3] {
    vec3_normalize(face_normal(positions, tri))
}

/// Compute the area of a triangle.
pub fn face_area(positions: &[[f64; 3]], tri: [usize; 3]) -> f64 {
    vec3_len(face_normal(positions, tri)) * 0.5
}

/// Compute dihedral angle between two faces sharing a half-edge.
/// Returns the angle in radians (0 = coplanar, PI = folded flat).
pub fn dihedral_angle(
    positions: &[[f64; 3]],
    face_a_verts: [usize; 3],
    face_b_verts: [usize; 3],
) -> f64 {
    let na = face_normal_normalized(positions, face_a_verts);
    let nb = face_normal_normalized(positions, face_b_verts);
    let cos_angle = vec3_dot(na, nb).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Cotangent of the angle at vertex `opposite_local` in triangle `tri`.
/// Used for the cotangent Laplacian weight.
/// Returns cot(angle) = cos/sin = dot(e1,e2) / |cross(e1,e2)|
pub fn cotangent_at_vertex(positions: &[[f64; 3]], tri: [usize; 3], opposite_local: usize) -> f64 {
    let v = positions[tri[opposite_local]];
    let a = positions[tri[(opposite_local + 1) % 3]];
    let b = positions[tri[(opposite_local + 2) % 3]];
    let e1 = vec3_sub(a, v);
    let e2 = vec3_sub(b, v);
    let cos_val = vec3_dot(e1, e2);
    let sin_val = vec3_len(vec3_cross(e1, e2));
    if sin_val < 1e-15 { return 0.0; }
    cos_val / sin_val
}

/// Compute angle-weighted vertex normals for the entire mesh.
pub fn compute_vertex_normals(positions: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<[f64; 3]> {
    let mut normals = vec![[0.0; 3]; positions.len()];

    for tri in faces {
        let fn_raw = face_normal(positions, *tri);
        // Angle at each vertex for weighting
        for local in 0..3 {
            let v = positions[tri[local]];
            let a = positions[tri[(local + 1) % 3]];
            let b = positions[tri[(local + 2) % 3]];
            let e1 = vec3_sub(a, v);
            let e2 = vec3_sub(b, v);
            let l1 = vec3_len(e1);
            let l2 = vec3_len(e2);
            if l1 < 1e-15 || l2 < 1e-15 { continue; }
            let cos_a = (vec3_dot(e1, e2) / (l1 * l2)).clamp(-1.0, 1.0);
            let angle = cos_a.acos();
            normals[tri[local]] = vec3_add(normals[tri[local]], vec3_scale(fn_raw, angle));
        }
    }

    for n in &mut normals {
        *n = vec3_normalize(*n);
    }
    normals
}

/// Compute the bounding sphere radius of a set of positions.
pub fn bounding_radius(positions: &[[f64; 3]]) -> f64 {
    if positions.is_empty() { return 1.0; }
    // Compute centroid
    let n = positions.len() as f64;
    let mut center = [0.0; 3];
    for p in positions {
        center[0] += p[0];
        center[1] += p[1];
        center[2] += p[2];
    }
    center = vec3_scale(center, 1.0 / n);
    positions.iter().map(|p| vec3_dist(*p, center)).fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_normal() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let n = face_normal_normalized(&positions, [0, 1, 2]);
        assert!((n[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_face_area() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let a = face_area(&positions, [0, 1, 2]);
        assert!((a - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cotangent_right_angle() {
        // Right angle at vertex 0
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let cot = cotangent_at_vertex(&positions, [0, 1, 2], 0);
        assert!(cot.abs() < 1e-10, "cot(90°) should be 0, got {}", cot);
    }

    #[test]
    fn test_cotangent_45_degree() {
        // Isoceles right triangle: 45° at vertices 1 and 2
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let cot = cotangent_at_vertex(&positions, [0, 1, 2], 1);
        assert!((cot - 1.0).abs() < 1e-10, "cot(45°) should be 1, got {}", cot);
    }

    #[test]
    fn test_vertex_normals_flat() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let normals = compute_vertex_normals(&positions, &faces);
        for n in &normals {
            assert!((n[2] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dihedral_coplanar() {
        let positions = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.0],
        ];
        let angle = dihedral_angle(&positions, [0, 1, 2], [1, 0, 3]);
        assert!(angle < 1e-10, "coplanar faces should have angle ~0, got {}", angle);
    }
}
