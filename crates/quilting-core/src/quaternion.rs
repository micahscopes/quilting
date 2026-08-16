use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul, Neg, Div};

/// Squared norm below which a quaternion counts as a Möbius pole and `inv`
/// returns a sentinel instead of dividing. Equivalently `|q| < 1e-10`.
///
/// This constant is mirrored by `qinv` in `shaders/math/quaternion.wgsl` and
/// the two must stay equal, even though the CPU runs f64 and the GPU runs f32.
/// The GPU is the binding constraint:
///
/// - f32 `dot(q, q)` for `|q| ~ 1e-10` is `~1e-20`, still a normal float with
///   ample mantissa. Push the cutoff much lower and the squares slide toward
///   f32's subnormal range (min normal `1.2e-38`) and the inverse loses all
///   precision before the guard ever fires.
/// - The sentinel then propagates into positions, cross products and
///   normalizations. `1e10` squares to `1e20`, eighteen orders below f32's
///   `3.4e38` ceiling, so the analytic-normal path stays finite.
///
/// f64 could take a far tighter cutoff (`1e-30` was the historical value), but
/// a CPU that admits poles the GPU rejects is worse than a slightly blunt one:
/// the CPU computes the LODs and smooth normals for exactly the geometry the
/// GPU draws, so disagreeing about where the pole is puts a discontinuity
/// precisely where the artifacts are most visible. Coordinates are model-scale
/// (order 1), so `|c*x + d| < 1e-10` already means the point maps ten billion
/// units away — nothing renderable is lost by treating it as infinity.
pub const SINGULARITY_NORM_SQ: f64 = 1e-20;

/// Scalar part returned by [`Quat::inv`] at a pole. Chosen as
/// `1 / sqrt(SINGULARITY_NORM_SQ)` so `|q⁻¹|` is continuous across the guard.
pub const SINGULARITY_SENTINEL: f64 = 1e10;

/// Squared norm of the Möbius denominator `c·x + d` below which a sampled
/// point counts as pole-adjacent for tessellation purposes and the face LOD
/// saturates to the maximum. Mirrored by the `min_bot_sq` check in
/// `quilting-renderer/shaders/lod_compute.vert.glsl`.
///
/// This is deliberately far above [`SINGULARITY_NORM_SQ`]: the sentinel keeps
/// the *arithmetic* finite, but it cannot keep the *geometry* honest. When the
/// pole lands on a sampled point, rounding cancels the imaginary part of the
/// numerator together with the denominator (the point and the pole share the
/// same float bits), so even the sentinel puts the deformed point at the
/// origin instead of far away. A collapsed point fakes a small deformed
/// median, which would hand the most conformally stretched face the *least*
/// tessellation. `|bot|² < 1e-8` means the point maps ≥ 10⁴ model units out —
/// unconditionally "maximum distortion" — so overriding the median there
/// changes nothing that the median math was getting right.
pub const POLE_PROXIMITY_NORM_SQ: f64 = 1e-8;

/// Squared norm of `c` below which a Möbius transform is treated as affine
/// (no conformal curvature) for CPU geometry preprocessing. The live shader
/// now evaluates the full differential even when `c = 0`, because that branch
/// still contains rotations and signed uniform scales. All non-affine
/// constructors in this crate produce `|c|` of order 1, so the band below
/// 1e-3 only contains transforms that are conformally flat to ~3% per model
/// unit.
pub const AFFINE_C_NORM_SQ: f64 = 1e-3;

/// Quaternion: q = w + xi + yj + zk
///
/// Following the convention in Krasauskas & Zubė where R³ is identified
/// with the imaginary quaternions Im(H), so a 3D point (x,y,z) corresponds
/// to the pure quaternion xi + yj + zk (w=0).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const ZERO: Self = Self { w: 0.0, x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
    pub const I: Self = Self { w: 0.0, x: 1.0, y: 0.0, z: 0.0 };
    pub const J: Self = Self { w: 0.0, x: 0.0, y: 1.0, z: 0.0 };
    pub const K: Self = Self { w: 0.0, x: 0.0, y: 0.0, z: 1.0 };

    #[inline]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Pure imaginary quaternion from a 3D point.
    #[inline]
    pub fn from_point(x: f64, y: f64, z: f64) -> Self {
        Self { w: 0.0, x, y, z }
    }

    /// Extract the 3D point (imaginary part).
    #[inline]
    pub fn to_point(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Scalar (real) part.
    #[inline]
    pub fn re(self) -> f64 { self.w }

    /// Imaginary (vector) part as [x, y, z].
    #[inline]
    pub fn im(self) -> [f64; 3] { [self.x, self.y, self.z] }

    /// Conjugate: q̄ = w - xi - yj - zk
    #[inline]
    pub fn conj(self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    /// Squared norm: |q|² = qq̄ = w² + x² + y² + z²
    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Norm: |q|
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Inverse: q⁻¹ = q̄/|q|²
    #[inline]
    pub fn inv(self) -> Self {
        let n2 = self.norm_sq();
        if n2 < SINGULARITY_NORM_SQ {
            // At a Möbius pole. Return a large but finite value so callers get
            // "very far away" rather than NaN.
            return Self { w: SINGULARITY_SENTINEL, x: 0.0, y: 0.0, z: 0.0 };
        }
        let inv_n2 = 1.0 / n2;
        Self {
            w: self.w * inv_n2,
            x: -self.x * inv_n2,
            y: -self.y * inv_n2,
            z: -self.z * inv_n2,
        }
    }

    /// Normalize to unit quaternion.
    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    /// Dot product (as 4D vectors).
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }
}

// q * q' = (rr' - v·v') + (rv' + r'v + v×v')
impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl Mul<f64> for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self { w: self.w * s, x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Mul<Quat> for f64 {
    type Output = Quat;
    #[inline]
    fn mul(self, q: Quat) -> Quat { q * self }
}

impl Div for Quat {
    type Output = Self;
    /// q / r = q * r⁻¹ (right division)
    #[inline]
    fn div(self, rhs: Self) -> Self { self * rhs.inv() }
}

impl Div<f64> for Quat {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        let inv = 1.0 / s;
        Self { w: self.w * inv, x: self.x * inv, y: self.y * inv, z: self.z * inv }
    }
}

impl Add for Quat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { w: self.w + rhs.w, x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl Sub for Quat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { w: self.w - rhs.w, x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl Neg for Quat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { w: -self.w, x: -self.x, y: -self.y, z: -self.z }
    }
}

/// Möbius transformation in R³: F(x) = (ax + b)(cx + d)⁻¹
///
/// Represented as a 2×2 quaternion matrix `[[a, b], [c, d]]`.
/// Composition corresponds to matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mobius {
    pub a: Quat,
    pub b: Quat,
    pub c: Quat,
    pub d: Quat,
}

impl Mobius {
    pub fn new(a: Quat, b: Quat, c: Quat, d: Quat) -> Self {
        Self { a, b, c, d }
    }

    /// Identity transformation.
    pub fn identity() -> Self {
        Self::new(Quat::ONE, Quat::ZERO, Quat::ZERO, Quat::ONE)
    }

    /// Check if this is an identity or near-identity transform.
    /// Returns true if c ≈ 0 (no conformal curvature — affine transform).
    /// The threshold matches the shader's predicate; see [`AFFINE_C_NORM_SQ`].
    pub fn is_affine(&self) -> bool {
        self.c.norm_sq() < AFFINE_C_NORM_SQ
    }


    /// Translation: x ↦ x + t
    pub fn translation(t: Quat) -> Self {
        Self::new(Quat::ONE, t, Quat::ZERO, Quat::ONE)
    }

    /// Uniform scaling: x ↦ λx
    pub fn scale(s: f64) -> Self {
        Self::new(Quat::new(s, 0.0, 0.0, 0.0), Quat::ZERO, Quat::ZERO, Quat::ONE)
    }

    /// Inversion: x ↦ -x⁻¹ (inversion in the unit sphere)
    pub fn inversion() -> Self {
        Self::new(Quat::ZERO, -Quat::ONE, Quat::ONE, Quat::ZERO)
    }

    /// Apply transformation to a point (pure imaginary quaternion).
    #[inline]
    pub fn apply(&self, x: Quat) -> Quat {
        (self.a * x + self.b) * (self.c * x + self.d).inv()
    }

    /// Local conformal length scale at `x`.
    ///
    /// For `F(x) = (a x + b) (c x + d)⁻¹`, the differential sends a tangent
    /// vector `v` to `(a - F(x)c) v (c x + d)⁻¹`. Quaternion norms are
    /// multiplicative, so every direction has scale
    /// `|a - F(x)c| / |c x + d|`. The value is infinite at an exact pole.
    pub fn conformal_scale_at(&self, x: Quat) -> f64 {
        let denominator = self.c * x + self.d;
        let denominator_norm_sq = denominator.norm_sq();
        if denominator_norm_sq == 0.0 {
            return f64::INFINITY;
        }
        let mapped = (self.a * x + self.b) * denominator.inv();
        let left = self.a - mapped * self.c;
        (left.norm_sq() / denominator_norm_sq).sqrt()
    }

    /// Transform a weight under this Möbius transformation.
    /// w' = (cx + d) * w  (equation 5 from the paper)
    #[inline]
    pub fn transform_weight(&self, x: Quat, w: Quat) -> Quat {
        (self.c * x + self.d) * w
    }

    /// Rotation: x ↦ qxq̄ where q is a unit quaternion.
    /// Axis (ax,ay,az), angle in radians.
    pub fn rotation(ax: f64, ay: f64, az: f64, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let c = half.cos();
        let len = (ax * ax + ay * ay + az * az).sqrt();
        let (nx, ny, nz) = if len > 1e-12 {
            (ax / len, ay / len, az / len)
        } else {
            (0.0, 0.0, 1.0)
        };
        let q = Quat::new(c, s * nx, s * ny, s * nz);
        // F(x) = qxq̄ = (qx)(q̄)⁻¹ · ... but actually for rotations:
        // F(x) = (qx + 0)(0x + q̄⁻¹)⁻¹ ... no.
        // Rotation via Möbius: a=q, b=0, c=0, d=q̄ doesn't work because
        // F(x) = qx(q̄)⁻¹ = qxq (since q̄⁻¹ = q for unit quaternions).
        // That's wrong — we want qxq̄.
        // Actually: F(x) = (ax+b)(cx+d)⁻¹ with a=q, b=0, c=0, d=1
        // gives F(x) = qx which is NOT a rotation (it's a left multiplication).
        //
        // The correct Möbius for rotation qxq̄:
        // This is a similarity, expressible as F(x) = qxq̄ = q·x·q̄.
        // In Möbius form: not directly (ax+b)(cx+d)⁻¹ because that's
        // left-linear, but rotation is a conjugation.
        //
        // However, for PURE IMAGINARY x (3D points), qxq̄ IS a Möbius map.
        // We need: (ax+b)(cx+d)⁻¹ = qxq̄ for all pure imaginary x.
        // Setting c=0, d=q̄: F(x) = (ax+b)·q = axq + bq
        // We need axq = qx, so a = qxq⁻¹·x⁻¹... that depends on x.
        //
        // The trick: for pure imaginary quaternions, qxq̄ = qx·q̄.
        // As a Möbius: a=q, b=0, c=0, d=conj(q)⁻¹ = q (for unit q, q̄⁻¹=q).
        // Then F(x) = qx · q⁻¹ = qx · (q̄)⁻¹... wait.
        // d = conj(q), d⁻¹ = conj(conj(q))/|conj(q)|² = q (unit).
        // F(x) = (qx + 0)(0 + conj(q))⁻¹ = qx · q = qxq. NOT qxq̄.
        //
        // For unit quaternions: q̄ = q⁻¹, so qxq̄ = qxq⁻¹.
        // Möbius: a=q, b=0, c=0, d=q. Then F(x) = qx·q⁻¹ = qxq̄. ✓
        Self::new(q, Quat::ZERO, Quat::ZERO, q)
    }

    /// Sphere reflection: x ↦ c + r²(x-c)/|x-c|²
    ///
    /// For pure imaginary quaternions (3D points), this is:
    ///   F(x) = c - r²·(x-c)⁻¹ = (cx - c² - r²)(x - c)⁻¹
    ///
    /// So a=c, b=-(c²+r²), c_coeff=1, d=-c.
    /// Note: this is an IMPROPER (orientation-reversing) transformation.
    pub fn sphere_reflection(center: Quat, r: f64) -> Self {
        let c_sq = center * center; // c² (for pure imaginary c, this is -|c|²)
        let r_sq = Quat::new(r * r, 0.0, 0.0, 0.0);
        Self::new(
            center,             // a = c
            -(c_sq + r_sq),     // b = -(c² + r²)
            Quat::ONE,          // c_coeff = 1
            -center,            // d = -c
        )
    }

    /// Sphere inversion: compose two sphere reflections for a proper
    /// (orientation-preserving) Möbius transformation.
    pub fn sphere_inversion(c1: Quat, r1: f64, c2: Quat, r2: f64) -> Self {
        let s1 = Self::sphere_reflection(c1, r1);
        let s2 = Self::sphere_reflection(c2, r2);
        s2.compose(&s1)
    }

    /// Compose two Möbius transformations (matrix multiplication).
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: Quat, b: Quat) -> bool {
        (a.w - b.w).abs() < EPS && (a.x - b.x).abs() < EPS
            && (a.y - b.y).abs() < EPS && (a.z - b.z).abs() < EPS
    }

    #[test]
    fn quaternion_product() {
        // ij = k
        assert!(approx_eq(Quat::I * Quat::J, Quat::K));
        // jk = i
        assert!(approx_eq(Quat::J * Quat::K, Quat::I));
        // ki = j
        assert!(approx_eq(Quat::K * Quat::I, Quat::J));
        // i² = -1
        assert!(approx_eq(Quat::I * Quat::I, -Quat::ONE));
    }

    #[test]
    fn inverse() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        let qi = q.inv();
        assert!(approx_eq(q * qi, Quat::ONE));
        assert!(approx_eq(qi * q, Quat::ONE));
    }

    /// The pole guard should hand back a finite continuation rather than a
    /// discontinuity: `|q⁻¹|` at the threshold is `1/sqrt(threshold)`, which is
    /// exactly the sentinel. See `SINGULARITY_NORM_SQ` for why these particular
    /// values, and `qinv` in `shaders/math/quaternion.wgsl` for the GPU copy.
    #[test]
    fn pole_sentinel_is_continuous_with_the_real_inverse() {
        assert!(
            (SINGULARITY_SENTINEL - 1.0 / SINGULARITY_NORM_SQ.sqrt()).abs()
                < 1e-6 * SINGULARITY_SENTINEL,
            "sentinel should be 1/sqrt(threshold)"
        );

        // Just above the threshold: the real inverse, magnitude ~= sentinel.
        let just_above = Quat::from_point(SINGULARITY_NORM_SQ.sqrt() * 1.001, 0.0, 0.0);
        let norm_above = just_above.inv().norm();
        assert!(
            (norm_above / SINGULARITY_SENTINEL - 1.0).abs() < 0.01,
            "inverse magnitude {norm_above:e} should be near the sentinel"
        );

        // Just below: the guard, exactly the sentinel and still finite.
        let just_below = Quat::from_point(SINGULARITY_NORM_SQ.sqrt() * 0.999, 0.0, 0.0);
        assert_eq!(just_below.inv().norm(), SINGULARITY_SENTINEL);
        assert!(Quat::ZERO.inv().norm().is_finite());
    }

    /// The sentinel must survive being squared in f32, since that is what the
    /// shader does when it normalizes an analytic QB normal.
    #[test]
    fn pole_sentinel_stays_in_f32_range_when_squared() {
        let s = SINGULARITY_SENTINEL as f32;
        assert!(s.is_finite() && (s * s).is_finite(), "sentinel overflows f32 when squared");
    }

    #[test]
    fn conjugate_product() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        // |q|² = q * q̄
        let n2 = q.norm_sq();
        let qq = q * q.conj();
        assert!((qq.w - n2).abs() < EPS);
        assert!(qq.x.abs() < EPS && qq.y.abs() < EPS && qq.z.abs() < EPS);
    }

    #[test]
    fn mobius_identity() {
        let m = Mobius::identity();
        let p = Quat::from_point(1.0, 2.0, 3.0);
        assert!(approx_eq(m.apply(p), p));
    }

    #[test]
    fn mobius_translation() {
        let t = Quat::from_point(1.0, 0.0, 0.0);
        let m = Mobius::translation(t);
        let p = Quat::from_point(0.0, 1.0, 0.0);
        let result = m.apply(p);
        assert!(approx_eq(result, Quat::from_point(1.0, 1.0, 0.0)));
    }

    #[test]
    fn conformal_scale_covers_affine_and_inversive_generators() {
        let p = Quat::from_point(2.0, 0.0, 0.0);
        assert!((Mobius::identity().conformal_scale_at(p) - 1.0).abs() < EPS);
        assert!((Mobius::translation(Quat::I).conformal_scale_at(p) - 1.0).abs() < EPS);
        assert!((Mobius::scale(-3.0).conformal_scale_at(p) - 3.0).abs() < EPS);
        assert!((Mobius::rotation(0.0, 0.0, 1.0, 0.7).conformal_scale_at(p) - 1.0).abs() < EPS);
        assert!((Mobius::inversion().conformal_scale_at(p) - 0.25).abs() < EPS);
        assert!(Mobius::inversion().conformal_scale_at(Quat::ZERO).is_infinite());
    }

    #[test]
    fn mobius_inversion() {
        // Inversion of a unit point: inv(p) = -p⁻¹ = -(-p)/|p|² = p/|p|²
        // For p = (1,0,0): inv(p) = -(1,0,0)⁻¹ = -(-(1,0,0)/1) = (1,0,0)
        let m = Mobius::inversion();
        let p = Quat::from_point(1.0, 0.0, 0.0);
        let result = m.apply(p);
        // F(x) = (0*x + (-1))(1*x + 0)⁻¹ = -x⁻¹
        // For pure imag x: x⁻¹ = -x/|x|², so -x⁻¹ = x/|x|²
        let expected = Quat::from_point(1.0, 0.0, 0.0); // |p|²=1
        assert!(approx_eq(result, expected));
    }

    #[test]
    fn mobius_composition() {
        let t = Mobius::translation(Quat::from_point(1.0, 0.0, 0.0));
        let s = Mobius::scale(2.0);
        let ts = t.compose(&s); // first scale, then translate

        let p = Quat::from_point(1.0, 0.0, 0.0);
        let direct = t.apply(s.apply(p));
        let composed = ts.apply(p);
        assert!(approx_eq(direct, composed));
    }

    #[test]
    fn weight_transform_identity() {
        let m = Mobius::identity();
        let p = Quat::from_point(1.0, 2.0, 3.0);
        let w = Quat::ONE;
        // Under identity: w' = (0*p + 1)*w = w
        let w_prime = m.transform_weight(p, w);
        assert!(approx_eq(w_prime, w));
    }
}
