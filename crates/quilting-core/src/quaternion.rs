use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul, Neg, Div};

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
        debug_assert!(n2 > 1e-30, "inverting near-zero quaternion");
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
/// Represented as a 2×2 quaternion matrix [[a,b],[c,d]].
/// Composition corresponds to matrix multiplication.
#[derive(Debug, Clone, Copy)]
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

    /// Transform a weight under this Möbius transformation.
    /// w' = (cx + d) * w  (equation 5 from the paper)
    #[inline]
    pub fn transform_weight(&self, x: Quat, w: Quat) -> Quat {
        (self.c * x + self.d) * w
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
