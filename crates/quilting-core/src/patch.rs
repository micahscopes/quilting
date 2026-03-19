use crate::quaternion::{Quat, Mobius};

/// A QB (Quaternionic-Bézier) triangle patch.
///
/// Each vertex has a position (as a pure imaginary quaternion representing
/// a 3D point) and a weight (a general quaternion). Surface evaluation
/// uses the rational quaternion formula:
///
///   X(u,v) = Im( (Σ λᵢ pᵢ wᵢ) / (Σ λᵢ wᵢ) )
///
/// where λᵢ are barycentric coordinates. This produces Möbius-invariant
/// surfaces: applying a Möbius transformation only requires transforming
/// control points and weights.
#[derive(Debug, Clone, Copy)]
pub struct QBTriPatch {
    /// Vertex positions (pure imaginary quaternions = 3D points)
    pub positions: [Quat; 3],
    /// Vertex weights (general quaternions)
    pub weights: [Quat; 3],
}

/// Result of evaluating a surface patch at a parameter point.
#[derive(Debug, Clone, Copy)]
pub struct SurfacePoint {
    pub position: [f64; 3],
    pub normal: [f64; 3],
}

impl QBTriPatch {
    pub fn new(positions: [Quat; 3], weights: [Quat; 3]) -> Self {
        Self { positions, weights }
    }

    /// Create a flat patch (all weights = 1) from 3D vertex positions.
    pub fn flat(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Self {
        Self {
            positions: [
                Quat::from_point(p0[0], p0[1], p0[2]),
                Quat::from_point(p1[0], p1[1], p1[2]),
                Quat::from_point(p2[0], p2[1], p2[2]),
            ],
            weights: [Quat::ONE, Quat::ONE, Quat::ONE],
        }
    }

    /// Evaluate the rational quaternion surface at barycentric (u, v).
    /// The third barycentric coordinate is w = 1 - u - v.
    #[inline]
    pub fn eval(&self, u: f64, v: f64) -> Quat {
        let w = 1.0 - u - v;
        let [p0, p1, p2] = self.positions;
        let [w0, w1, w2] = self.weights;

        let top = w * (p0 * w0) + u * (p1 * w1) + v * (p2 * w2);
        let bottom = w * w0 + u * w1 + v * w2;

        top * bottom.inv()
    }

    /// Evaluate surface point with normal (via finite differences).
    pub fn eval_with_normal(&self, u: f64, v: f64) -> SurfacePoint {
        let eps = 1e-4;
        let p = self.eval(u, v);
        let pu = self.eval(u + eps, v);
        let pv = self.eval(u, v + eps);

        let du = [pu.x - p.x, pu.y - p.y, pu.z - p.z];
        let dv = [pv.x - p.x, pv.y - p.y, pv.z - p.z];

        // Normal = cross(du, dv), normalized
        let nx = du[1] * dv[2] - du[2] * dv[1];
        let ny = du[2] * dv[0] - du[0] * dv[2];
        let nz = du[0] * dv[1] - du[1] * dv[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let normal = if len > 1e-12 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        SurfacePoint {
            position: p.to_point(),
            normal,
        }
    }

    /// Apply a Möbius transformation to this patch.
    /// Transforms positions and weights according to equations (4) and (5).
    pub fn transform(&self, m: &Mobius) -> Self {
        let mut new_positions = [Quat::ZERO; 3];
        let mut new_weights = [Quat::ZERO; 3];
        for i in 0..3 {
            new_positions[i] = m.apply(self.positions[i]);
            new_weights[i] = m.transform_weight(self.positions[i], self.weights[i]);
        }
        Self {
            positions: new_positions,
            weights: new_weights,
        }
    }
}

/// A QB quad patch (bilinear, 4 control points).
#[derive(Debug, Clone, Copy)]
pub struct QBQuadPatch {
    /// Control positions: [p00, p10, p01, p11]
    pub positions: [Quat; 4],
    /// Control weights
    pub weights: [Quat; 4],
}

impl QBQuadPatch {
    pub fn new(positions: [Quat; 4], weights: [Quat; 4]) -> Self {
        Self { positions, weights }
    }

    /// Evaluate at (s, t) in [0,1]².
    #[inline]
    pub fn eval(&self, s: f64, t: f64) -> Quat {
        let s0 = 1.0 - s;
        let s1 = s;
        let t0 = 1.0 - t;
        let t1 = t;

        let [p00, p10, p01, p11] = self.positions;
        let [w00, w10, w01, w11] = self.weights;

        let top = (s0 * t0) * (p00 * w00)
                + (s1 * t0) * (p10 * w10)
                + (s0 * t1) * (p01 * w01)
                + (s1 * t1) * (p11 * w11);
        let bottom = (s0 * t0) * w00
                   + (s1 * t0) * w10
                   + (s0 * t1) * w01
                   + (s1 * t1) * w11;

        top * bottom.inv()
    }

    pub fn eval_with_normal(&self, s: f64, t: f64) -> SurfacePoint {
        let eps = 1e-4;
        let p = self.eval(s, t);
        let ps = self.eval(s + eps, t);
        let pt = self.eval(s, t + eps);

        let ds = [ps.x - p.x, ps.y - p.y, ps.z - p.z];
        let dt = [pt.x - p.x, pt.y - p.y, pt.z - p.z];

        let nx = ds[1] * dt[2] - ds[2] * dt[1];
        let ny = ds[2] * dt[0] - ds[0] * dt[2];
        let nz = ds[0] * dt[1] - ds[1] * dt[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let normal = if len > 1e-12 {
            [nx / len, ny / len, nz / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        SurfacePoint {
            position: p.to_point(),
            normal,
        }
    }

    pub fn transform(&self, m: &Mobius) -> Self {
        let mut new_positions = [Quat::ZERO; 4];
        let mut new_weights = [Quat::ZERO; 4];
        for i in 0..4 {
            new_positions[i] = m.apply(self.positions[i]);
            new_weights[i] = m.transform_weight(self.positions[i], self.weights[i]);
        }
        Self { positions: new_positions, weights: new_weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq_3(a: [f64; 3], b: [f64; 3]) -> bool {
        (a[0]-b[0]).abs() < EPS && (a[1]-b[1]).abs() < EPS && (a[2]-b[2]).abs() < EPS
    }

    #[test]
    fn flat_patch_is_planar() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // Evaluate at centroid
        let sp = patch.eval_with_normal(1.0/3.0, 1.0/3.0);
        assert!(approx_eq_3(sp.position, [1.0/3.0, 1.0/3.0, 0.0]));
        // Normal should be (0,0,1) for a flat patch in XY plane
        assert!(approx_eq_3(sp.normal, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn flat_patch_vertices() {
        let p0 = [1.0, 2.0, 3.0];
        let p1 = [4.0, 5.0, 6.0];
        let p2 = [7.0, 8.0, 9.0];
        let patch = QBTriPatch::flat(p0, p1, p2);

        // u=0, v=0 → p0
        assert!(approx_eq_3(patch.eval(0.0, 0.0).to_point(), p0));
        // u=1, v=0 → p1
        assert!(approx_eq_3(patch.eval(1.0, 0.0).to_point(), p1));
        // u=0, v=1 → p2
        assert!(approx_eq_3(patch.eval(0.0, 1.0).to_point(), p2));
    }

    #[test]
    fn mobius_transform_preserves_surface() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let t = Mobius::translation(Quat::from_point(5.0, 0.0, 0.0));
        let transformed = patch.transform(&t);

        let original = patch.eval(0.3, 0.3).to_point();
        let moved = transformed.eval(0.3, 0.3).to_point();

        // Translation by (5,0,0) should shift x by 5
        assert!((moved[0] - original[0] - 5.0).abs() < EPS);
        assert!((moved[1] - original[1]).abs() < EPS);
        assert!((moved[2] - original[2]).abs() < EPS);
    }

    #[test]
    fn nonunit_weights_curve_surface() {
        // A patch with non-unit weights should NOT be flat
        let patch = QBTriPatch::new(
            [
                Quat::from_point(0.0, 0.0, 0.0),
                Quat::from_point(1.0, 0.0, 0.0),
                Quat::from_point(0.0, 1.0, 0.0),
            ],
            [
                Quat::ONE,
                Quat::ONE,
                Quat::new(0.0, 0.0, 0.0, 1.0), // weight = k (pure quaternion)
            ],
        );

        let flat = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // At (0.3, 0.3), the weighted patch should differ from the flat one
        let p_weighted = patch.eval(0.3, 0.3).to_point();
        let p_flat = flat.eval(0.3, 0.3).to_point();

        let diff = (0..3).map(|i| (p_weighted[i] - p_flat[i]).abs()).sum::<f64>();
        assert!(diff > 0.01, "non-unit weights should deform the surface, diff={}", diff);
    }
}
