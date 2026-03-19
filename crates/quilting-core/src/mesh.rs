use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TessellationMesh {
    pub positions: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
    pub normals: Vec<[f64; 3]>,
}

impl TessellationMesh {
    /// Create a flat 2D tessellation mesh (normals all point up in z).
    pub fn from_2d(positions: Vec<[f64; 2]>, triangles: Vec<[usize; 3]>) -> Self {
        let normals = vec![[0.0, 0.0, 1.0]; positions.len()];
        Self {
            positions,
            triangles,
            normals,
        }
    }

    /// Compute angle-weighted vertex normals from 3D positions.
    pub fn compute_normals_3d(
        positions_3d: &[[f64; 3]],
        triangles: &[[usize; 3]],
    ) -> Vec<[f64; 3]> {
        let mut normals = vec![[0.0f64; 3]; positions_3d.len()];

        for tri in triangles {
            let [i, j, k] = *tri;
            let a = positions_3d[i];
            let b = positions_3d[j];
            let c = positions_3d[k];

            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];

            // Face normal (magnitude = 2*area)
            let face_normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];

            let angle_a = angle_between(ab, ac);
            let angle_b = angle_between([-ab[0], -ab[1], -ab[2]], bc);
            let angle_c = angle_between([-ac[0], -ac[1], -ac[2]], [-bc[0], -bc[1], -bc[2]]);

            for d in 0..3 {
                normals[i][d] += angle_a * face_normal[d];
                normals[j][d] += angle_b * face_normal[d];
                normals[k][d] += angle_c * face_normal[d];
            }
        }

        // Normalize
        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }

        normals
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let cos = (dot / (la * lb)).clamp(-1.0, 1.0);
    cos.acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_2d_sets_normals() {
        let mesh = TessellationMesh::from_2d(
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![[0, 1, 2]],
        );
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        for n in &mesh.normals {
            assert_eq!(*n, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn normals_3d_flat_plane() {
        // Flat triangle in XY plane -> normals should be [0,0,1]
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let triangles = [[0, 1, 2]];
        let normals = TessellationMesh::compute_normals_3d(&positions, &triangles);
        for n in &normals {
            assert!((n[2] - 1.0).abs() < 1e-10, "expected z-up normal, got {:?}", n);
        }
    }

    #[test]
    fn normals_3d_unit_length() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let triangles = [[0, 1, 3], [1, 2, 3], [2, 0, 3]];
        let normals = TessellationMesh::compute_normals_3d(&positions, &triangles);
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "normal not unit length: {:?} (len={})",
                n,
                len
            );
        }
    }
}
