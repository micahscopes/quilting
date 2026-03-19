use crate::quaternion::{Quat, Mobius};

/// Per-face instance data for instanced rendering.
/// Each face has 3 transformed positions and 3 transformed weights (quaternions).
/// The vertex shader uses these with shared tessellation geometry to evaluate
/// the QB surface: X = (Σ λᵢ pᵢ wᵢ) / (Σ λᵢ wᵢ)
#[derive(Debug, Clone)]
pub struct FaceInstance {
    /// Transformed vertex positions as quaternions (pure imaginary)
    pub positions: [Quat; 3],
    /// Transformed vertex weights
    pub weights: [Quat; 3],
}

/// Compute per-face instance data for an entire mesh under a Möbius transformation.
///
/// Input: source mesh vertices + faces + transformation
/// Output: one FaceInstance per face, ready to upload as instance attributes
pub fn compute_instances(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
) -> Vec<FaceInstance> {
    // Pre-transform all vertices and their weights
    let transformed: Vec<(Quat, Quat)> = vertices.iter().map(|v| {
        let p = Quat::from_point(v[0], v[1], v[2]);
        let p_prime = transform.apply(p);
        let w_prime = transform.transform_weight(p, Quat::ONE);
        (p_prime, w_prime)
    }).collect();

    faces.iter().map(|face| {
        let (p0, w0) = transformed[face[0]];
        let (p1, w1) = transformed[face[1]];
        let (p2, w2) = transformed[face[2]];
        FaceInstance {
            positions: [p0, p1, p2],
            weights: [w0, w1, w2],
        }
    }).collect()
}

impl FaceInstance {
    /// Pack as 24 f32s: [p0.w,p0.x,p0.y,p0.z, p1..., p2..., w0..., w1..., w2...]
    pub fn to_f32_array(&self) -> [f32; 24] {
        let mut out = [0.0f32; 24];
        for (i, p) in self.positions.iter().enumerate() {
            out[i*4]   = p.w as f32;
            out[i*4+1] = p.x as f32;
            out[i*4+2] = p.y as f32;
            out[i*4+3] = p.z as f32;
        }
        for (i, w) in self.weights.iter().enumerate() {
            out[12+i*4]   = w.w as f32;
            out[12+i*4+1] = w.x as f32;
            out[12+i*4+2] = w.y as f32;
            out[12+i*4+3] = w.z as f32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    #[test]
    fn identity_preserves_positions() {
        let (verts, faces) = shapes::cube();
        let instances = compute_instances(&verts, &faces, &Mobius::identity());

        // Under identity, positions should match original and weights should be 1
        for (face, inst) in faces.iter().zip(instances.iter()) {
            for i in 0..3 {
                let orig = verts[face[i]];
                let got = inst.positions[i].to_point();
                assert!((got[0] - orig[0]).abs() < 1e-10);
                assert!((got[1] - orig[1]).abs() < 1e-10);
                assert!((got[2] - orig[2]).abs() < 1e-10);
            }
            for w in &inst.weights {
                assert!((w.w - 1.0).abs() < 1e-10);
                assert!(w.x.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn translation_shifts() {
        let (verts, faces) = shapes::tetrahedron();
        let t = Mobius::translation(Quat::from_point(10.0, 0.0, 0.0));
        let instances = compute_instances(&verts, &faces, &t);

        for (face, inst) in faces.iter().zip(instances.iter()) {
            for i in 0..3 {
                let orig = verts[face[i]];
                let got = inst.positions[i].to_point();
                assert!((got[0] - orig[0] - 10.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn sphere_reflection_changes_weights() {
        let (verts, faces) = shapes::octahedron();
        let m = Mobius::sphere_reflection(Quat::ZERO, 1.0);
        let instances = compute_instances(&verts, &faces, &m);

        // Sphere reflection should produce non-identity weights
        let has_nontrivial = instances.iter().any(|inst| {
            inst.weights.iter().any(|w| (w.w - 1.0).abs() > 0.01 || w.x.abs() > 0.01)
        });
        assert!(has_nontrivial, "sphere reflection should produce non-unit weights");
    }

    #[test]
    fn f32_packing() {
        let inst = FaceInstance {
            positions: [
                Quat::from_point(1.0, 2.0, 3.0),
                Quat::from_point(4.0, 5.0, 6.0),
                Quat::from_point(7.0, 8.0, 9.0),
            ],
            weights: [Quat::ONE, Quat::ONE, Quat::ONE],
        };
        let packed = inst.to_f32_array();
        assert_eq!(packed.len(), 24);
        // p0 = (w=0, x=1, y=2, z=3)
        assert_eq!(packed[0], 0.0);
        assert_eq!(packed[1], 1.0);
        assert_eq!(packed[2], 2.0);
        assert_eq!(packed[3], 3.0);
        // w0 = (1, 0, 0, 0)
        assert_eq!(packed[12], 1.0);
    }
}
