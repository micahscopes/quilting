/// Conformal Geometric Algebra for 3D Euclidean space (CGA3).
///
/// Basis: e1, e2, e3, e+, e- with signatures (+,+,+,+,-)
/// Null basis: no = (e- - e+)/2 (origin), ni = e- + e+ (infinity)
///
/// A 3D point x ∈ R³ is embedded as: X = x + ½|x|²ni + no
/// Versors (even-grade elements) represent all conformal transformations
/// via the sandwich product: X' = V X V̄
///
/// We store CGA3 multivectors as 32 f64 coefficients (2^5 basis blades).
/// Only a subset of operations are implemented — those needed for
/// constructing and composing conformal transformations.

use crate::quaternion::{Quat, Mobius};

/// Indices for the 32 basis blades of Cl(4,1).
/// Ordered by grade: scalar, vectors, bivectors, trivectors, 4-vectors, pseudoscalar.
///
/// We use a bitmap encoding: blade index = bitmask of basis vectors present.
/// e1=0b00001, e2=0b00010, e3=0b00100, ep=0b01000, em=0b10000
const E1: usize = 0b00001;
const E2: usize = 0b00010;
const E3: usize = 0b00100;
const EP: usize = 0b01000; // e+
const EM: usize = 0b10000; // e-

/// Metric signature: e1²=+1, e2²=+1, e3²=+1, e+²=+1, e-²=-1
fn basis_sq(i: usize) -> f64 {
    if i == EM { -1.0 } else { 1.0 }
}

/// Compute sign of the geometric product of two basis blades.
/// Returns (sign, result_blade).
fn blade_product(a: usize, b: usize) -> (f64, usize) {
    let mut sign = 1.0;
    let mut result = a;

    // Move each bit of b through the bits of result (current a),
    // counting transpositions and contracting matching pairs.
    for bit in 0..5 {
        let mask = 1 << bit;
        if b & mask == 0 { continue; }

        // Count how many bits in result are above this bit (need to swap past them)
        let above = result >> (bit + 1);
        let swaps = above.count_ones();
        if swaps % 2 == 1 { sign = -sign; }

        if result & mask != 0 {
            // Same basis vector — contract: eᵢeᵢ = ±1
            sign *= basis_sq(mask);
            result ^= mask; // remove the bit
        } else {
            result |= mask; // add the bit
        }
    }

    (sign, result)
}

/// A CGA3 multivector with 32 components.
#[derive(Clone, Copy)]
pub struct Cga3 {
    pub data: [f64; 32],
}

impl Cga3 {
    pub const ZERO: Self = Self { data: [0.0; 32] };

    pub fn scalar(s: f64) -> Self {
        let mut mv = Self::ZERO;
        mv.data[0] = s;
        mv
    }

    /// Basis vector e1
    pub fn e1() -> Self { Self::basis(E1) }
    pub fn e2() -> Self { Self::basis(E2) }
    pub fn e3() -> Self { Self::basis(E3) }

    /// Origin: no = (e- - e+) / 2
    pub fn no() -> Self {
        let mut mv = Self::ZERO;
        mv.data[EM] = 0.5;
        mv.data[EP] = -0.5;
        mv
    }

    /// Infinity: ni = e- + e+
    pub fn ni() -> Self {
        let mut mv = Self::ZERO;
        mv.data[EM] = 1.0;
        mv.data[EP] = 1.0;
        mv
    }

    fn basis(idx: usize) -> Self {
        let mut mv = Self::ZERO;
        mv.data[idx] = 1.0;
        mv
    }

    /// Embed a 3D point into CGA3: X = x + ½|x|²ni + no
    pub fn up_point(x: f64, y: f64, z: f64) -> Self {
        let r2 = x * x + y * y + z * z;
        let mut mv = Self::ZERO;
        mv.data[E1] = x;
        mv.data[E2] = y;
        mv.data[E3] = z;
        // ni = e+ + e-, no = (e- - e+)/2
        // X = x + ½r²(e+ + e-) + ½(e- - e+)
        //   = x + (½r² - ½)e+ + (½r² + ½)e-
        mv.data[EP] = 0.5 * r2 - 0.5;
        mv.data[EM] = 0.5 * r2 + 0.5;
        mv
    }

    /// Project a CGA3 conformal point back to R³.
    /// Normalizes so that the coefficient of no is 1.
    pub fn down_point(&self) -> [f64; 3] {
        // Inner product with ni gives -2 * (coefficient of no in the point)
        // The no component: coefficient of e- minus coefficient of e+ (divided by... )
        // For a conformal point X = αx + α½|x|²ni + αno,
        // the e- component = α(½|x|² + ½), the e+ component = α(½|x|² - ½)
        // α = e- - e+ = 1/(e- component - e+ component) ? No...
        // no·X = -α, so α = -(no·X)
        // Actually, for a normalized point with no coefficient = 1:
        // x_i = coefficient of e_i
        let no_coeff = self.data[EM] - self.data[EP]; // coefficient in the no direction
        if no_coeff.abs() < 1e-30 {
            return [0.0, 0.0, 0.0]; // point at infinity
        }
        let inv = 1.0 / no_coeff;
        [self.data[E1] * inv, self.data[E2] * inv, self.data[E3] * inv]
    }

    /// Geometric product: self * other
    pub fn gp(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..32 {
            if self.data[i] == 0.0 { continue; }
            for j in 0..32 {
                if other.data[j] == 0.0 { continue; }
                let (sign, blade) = blade_product(i, j);
                result.data[blade] += sign * self.data[i] * other.data[j];
            }
        }
        result
    }

    /// Reverse: reverses the order of basis vectors in each blade.
    /// For a k-blade, reverse multiplies by (-1)^(k(k-1)/2).
    pub fn reverse(&self) -> Self {
        let mut result = *self;
        for i in 0..32u32 {
            let grade = i.count_ones();
            // (-1)^(k(k-1)/2): negate for grades 2,3,6,7,10,11,...
            let sign = if grade >= 2 && (grade * (grade - 1)) / 2 % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            result.data[i as usize] *= sign;
        }
        result
    }

    /// Grade involution: negates odd-grade components.
    pub fn involute(&self) -> Self {
        let mut result = *self;
        for i in 0..32u32 {
            if i.count_ones() % 2 == 1 {
                result.data[i as usize] = -result.data[i as usize];
            }
        }
        result
    }

    /// Conjugate: reverse composed with grade involution.
    pub fn conjugate(&self) -> Self {
        self.reverse().involute()
    }

    /// Sandwich product: self * x * self.reverse()
    /// Used to apply a versor transformation to a point.
    pub fn sandwich(&self, x: &Self) -> Self {
        self.gp(x).gp(&self.reverse())
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        let mut result = *self;
        for v in &mut result.data { *v *= s; }
        result
    }

    /// Add two multivectors.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..32 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Scalar part.
    pub fn scalar_part(&self) -> f64 { self.data[0] }

    // --- Versor constructors for common conformal transformations ---

    /// Translator: T = 1 - ½t·ni, where t is a translation vector.
    /// Sandwich: TXT̃ translates X by t.
    pub fn translator(tx: f64, ty: f64, tz: f64) -> Self {
        // T = 1 - ½(t₁e₁ + t₂e₂ + t₃e₃)ni
        // ni = e+ + e-
        // t·ni = t₁(e₁e+ + e₁e-) + t₂(e₂e+ + e₂e-) + t₃(e₃e+ + e₃e-)
        let mut mv = Self::scalar(1.0);
        let half_t = [-0.5 * tx, -0.5 * ty, -0.5 * tz];
        let e_bases = [E1, E2, E3];
        for i in 0..3 {
            let (_, blade_p) = blade_product(e_bases[i], EP);
            let (sign_p, _) = blade_product(e_bases[i], EP);
            mv.data[blade_p] += half_t[i] * sign_p;

            let (sign_m, blade_m) = blade_product(e_bases[i], EM);
            mv.data[blade_m] += half_t[i] * sign_m;
        }
        mv
    }

    /// Rotor: R = cos(θ/2) - sin(θ/2)(B), where B is a unit bivector.
    /// For rotation around an axis (ax, ay, az) by angle θ:
    pub fn rotor(ax: f64, ay: f64, az: f64, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let c = half.cos();
        let len = (ax * ax + ay * ay + az * az).sqrt();
        let (nx, ny, nz) = if len > 1e-12 {
            (ax / len, ay / len, az / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Bivector for the rotation plane:
        // axis (nx,ny,nz) → bivector = nx(e2e3) + ny(e3e1) + nz(e1e2)
        let mut mv = Self::scalar(c);
        let (s23, b23) = blade_product(E2, E3);
        let (s31, b31) = blade_product(E3, E1);
        let (s12, b12) = blade_product(E1, E2);
        mv.data[b23] -= s * nx * s23;
        mv.data[b31] -= s * ny * s31;
        mv.data[b12] -= s * nz * s12;
        mv
    }

    /// Dilator: D = cosh(λ/2) + sinh(λ/2)(no∧ni)
    /// Produces uniform scaling by e^λ.
    pub fn dilator(log_scale: f64) -> Self {
        let half = log_scale / 2.0;
        let mut mv = Self::scalar(half.cosh());
        // no∧ni = ½(e- - e+)(e- + e+) = ½(e-² - e+²) = ½(-1-1) ... no wait
        // no = (e- - e+)/2, ni = e+ + e-
        // no∧ni = ½(e- - e+)(e+ + e-) ... need outer product
        // = ½(e-e+ + e-e- - e+e+ - e+e-)
        // = ½(e-e+ + (-1) - (+1) - e+e-)
        // = ½(e-e+ - e+e- - 2) = ½(2·e-e+ - 2) ... no, e-e+ = -e+e-
        // Actually: no∧ni outer product of two vectors = no·ni + no∧ni
        // Let me just compute: no∧ni as the grade-2 part of no*ni
        let no = Self::no();
        let ni = Self::ni();
        let product = no.gp(&ni);
        // Extract bivector (grade 2) part
        for i in 0..32u32 {
            if i.count_ones() == 2 {
                mv.data[i as usize] += half.sinh() * product.data[i as usize];
            }
        }
        mv
    }

    /// Convert a CGA3 versor to a Möbius transformation (2×2 quaternion matrix).
    ///
    /// Given a versor V that acts as X' = VXV̄, extract the equivalent
    /// Möbius transformation F(x) = (ax+b)(cx+d)⁻¹.
    pub fn to_mobius(&self) -> Mobius {
        // Apply the versor to the canonical basis points to extract a,b,c,d.
        // The Möbius transformation maps:
        //   0 → b·d⁻¹
        //   ∞ → a·c⁻¹
        //   eᵢ → (a·eᵢ + b)(c·eᵢ + d)⁻¹
        //
        // Strategy: transform the origin and three unit points,
        // then solve for a,b,c,d.
        //
        // Simpler approach: use the known relationship between CGA versors
        // and 2×2 quaternion matrices. For even versors, the map is:
        //
        // Transform no → extract b,d from the result
        // Transform ni → extract a,c from the result

        let origin = Self::up_point(0.0, 0.0, 0.0);
        let p1 = Self::up_point(1.0, 0.0, 0.0);
        let p2 = Self::up_point(0.0, 1.0, 0.0);
        let p3 = Self::up_point(0.0, 0.0, 1.0);

        let o_t = self.sandwich(&origin).down_point();
        let p1_t = self.sandwich(&p1).down_point();
        let p2_t = self.sandwich(&p2).down_point();
        let p3_t = self.sandwich(&p3).down_point();

        // F(0) = b·d⁻¹ → b = F(0)·d
        // F(eᵢ) = (a·eᵢ + b)(c·eᵢ + d)⁻¹
        // For unit weights, d=1, c=0 gives similarity; for full Möbius we need
        // to solve the system. Use 4 point correspondences.
        //
        // Actually, for most practical cases (rotations, translations, dilations),
        // c=0 and the Möbius reduces to an affine map F(x) = ax + b with d⁻¹ = 1.
        // For inversions, c≠0.
        //
        // General extraction: use the approach from Dorst's GA textbook.
        // For now, use a numerical approach: solve from 4 point correspondences.

        let o = Quat::from_point(o_t[0], o_t[1], o_t[2]);
        let f1 = Quat::from_point(p1_t[0], p1_t[1], p1_t[2]);
        let f2 = Quat::from_point(p2_t[0], p2_t[1], p2_t[2]);
        let f3 = Quat::from_point(p3_t[0], p3_t[1], p3_t[2]);

        // F(x) = (ax+b)(cx+d)⁻¹
        // Assume d=1 (can always normalize). Then:
        // F(0) = b(d)⁻¹ = b → b = o
        // F(eᵢ) = (a·eᵢ + b)(c·eᵢ + 1)⁻¹
        //
        // For conformal (not full Möbius with inversion): c=0
        // Then F(eᵢ) = a·eᵢ + b
        // So a·e1 = f1 - b = f1 - o
        // Similarly for e2, e3.
        //
        // Check if c=0 works (similarity transformation):
        let a_times_e1 = f1 - o;
        let a_times_e2 = f2 - o;
        let a_times_e3 = f3 - o;

        // a is the quaternion such that a·eᵢ gives the right result.
        // a·i = a_times_e1, a·j = a_times_e2, a·k = a_times_e3
        // From quaternion multiplication:
        // a·i: if a = w+xi+yj+zk, then a·i = -x + wi + zj - yk
        // We have 3 equations, 4 unknowns, but the system is overdetermined
        // in practice. Use the e1 and e2 results:
        let ae1 = a_times_e1;
        let ae2 = a_times_e2;
        let _ae3 = a_times_e3;

        // a * i = ae1 → a = ae1 * i⁻¹ = ae1 * (-i) = -ae1 * i
        let a = ae1 * (-Quat::I);

        // Verify with e2: a * j should = ae2
        let check = a * Quat::J;
        let err = (check - ae2).norm();

        if err < 1e-6 {
            // Similarity transformation (c=0)
            Mobius::new(a, o, Quat::ZERO, Quat::ONE)
        } else {
            // Full Möbius — need to solve for c too.
            // Use an additional point (e.g., -e1) for more constraints.
            // For now, fall back to numerical solve with 4 points.
            // This handles inversions and general conformal maps.
            solve_mobius_4pt(
                [Quat::ZERO, Quat::I, Quat::J, Quat::K],
                [o, f1, f2, f3],
            )
        }
    }
}

/// Solve for Möbius coefficients from 4 point correspondences.
/// F(xᵢ) = yᵢ where F(x) = (ax+b)(cx+d)⁻¹
fn solve_mobius_4pt(x: [Quat; 4], y: [Quat; 4]) -> Mobius {
    // F(x₀) = b·d⁻¹ = y₀ (when x₀ = 0)
    // Set d = 1. Then b = y₀.
    // F(xᵢ) = (a·xᵢ + b)(c·xᵢ + 1)⁻¹ = yᵢ
    // → a·xᵢ + b = yᵢ·(c·xᵢ + 1)
    // → a·xᵢ - yᵢ·c·xᵢ = yᵢ - b
    //
    // For x₀=0: b = y₀, confirmed.
    // For i=1,2,3: a·xᵢ - yᵢ·c·xᵢ = yᵢ - y₀
    //
    // This is a system of 3 quaternion equations in a,c (8 real unknowns).
    // With 3×4=12 real equations, it's overdetermined.
    //
    // Simplification: try c=0 first (affine), then if residual is large,
    // solve the full system iteratively.

    let b = y[0];
    let d = Quat::ONE;

    // Try c=0: a·xᵢ = yᵢ - b for i=1,2,3
    // Since x₁=i, a·i = y₁-b → a = (y₁-b)·(-i)
    let a = (y[1] - b) * (-x[1]); // right-multiply by inverse of x[1]

    // Check residual
    let mut max_err: f64 = 0.0;
    for i in 1..4 {
        let predicted = a * x[i] + b;
        max_err = max_err.max((predicted - y[i]).norm());
    }

    if max_err < 1e-6 {
        return Mobius::new(a, b, Quat::ZERO, d);
    }

    // Full Möbius with c≠0:
    // For each i: (a - yᵢ·c)·xᵢ = yᵢ - b
    // With xᵢ being unit quaternions (i, j, k), we can solve:
    //
    // Let u = a, v = c. Then for i=1,2,3:
    //   (u - yᵢ·v)·xᵢ = yᵢ - b
    //   u·xᵢ - yᵢ·v·xᵢ = yᵢ - b
    //
    // This is linear in u, v (8 unknowns, 12 equations).
    // Use least squares via the normal equations, or just use
    // two equations to solve for u and v.

    // From i=1 (x₁=i): u·i - y₁·v·i = y₁ - b
    // From i=2 (x₂=j): u·j - y₂·v·j = y₂ - b
    // Two quat equations → 8 real equations, 8 unknowns. Solvable!

    let r1 = y[1] - b;
    let r2 = y[2] - b;

    // u·i = r1 + y₁·v·i
    // u·j = r2 + y₂·v·j
    // u = (r1 + y₁·v·i)·(-i) = -r1·i + y₁·v·i·(-i) = -r1·i - y₁·v
    // Substitute into second: (-r1·i - y₁·v)·j = r2 + y₂·v·j
    // -r1·i·j - y₁·v·j = r2 + y₂·v·j
    // -r1·k - y₁·v·j - y₂·v·j = r2
    // -(y₁ + y₂)·v·j = r2 + r1·k
    // v = -(y₁ + y₂)⁻¹ · (r2 + r1·k) · j⁻¹
    // v = -(y₁ + y₂)⁻¹ · (r2 + r1·k) · (-j)

    let sum_y = y[1] + y[2];
    let rhs = r2 + r1 * Quat::K;
    let c = -(sum_y.inv() * rhs * (-Quat::J));
    let a = -(r1 * Quat::I) - y[1] * c;

    Mobius::new(a, b, c, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-8;

    fn approx_eq_3(a: [f64; 3], b: [f64; 3]) -> bool {
        (a[0]-b[0]).abs() < EPS && (a[1]-b[1]).abs() < EPS && (a[2]-b[2]).abs() < EPS
    }

    #[test]
    fn up_down_roundtrip() {
        let pts = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.5, 7.0]];
        for p in &pts {
            let up = Cga3::up_point(p[0], p[1], p[2]);
            let down = up.down_point();
            assert!(approx_eq_3(down, *p), "roundtrip failed: {:?} -> {:?}", p, down);
        }
    }

    #[test]
    fn translator_versor() {
        let t = Cga3::translator(3.0, 0.0, 0.0);
        let p = Cga3::up_point(1.0, 2.0, 0.0);
        let result = t.sandwich(&p).down_point();
        assert!(approx_eq_3(result, [4.0, 2.0, 0.0]),
            "translation failed: {:?}", result);
    }

    #[test]
    fn rotor_versor() {
        // 90° rotation around z-axis: (1,0,0) → (0,1,0)
        let r = Cga3::rotor(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let p = Cga3::up_point(1.0, 0.0, 0.0);
        let result = r.sandwich(&p).down_point();
        assert!(approx_eq_3(result, [0.0, 1.0, 0.0]),
            "rotation failed: {:?}", result);
    }

    #[test]
    fn translator_to_mobius() {
        let t = Cga3::translator(5.0, 0.0, 0.0);
        let m = t.to_mobius();
        let p = Quat::from_point(1.0, 2.0, 3.0);
        let result = m.apply(p).to_point();
        assert!(approx_eq_3(result, [6.0, 2.0, 3.0]),
            "Möbius translation: {:?}", result);
    }

    #[test]
    fn rotor_to_mobius() {
        let r = Cga3::rotor(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let m = r.to_mobius();
        let p = Quat::from_point(1.0, 0.0, 0.0);
        let result = m.apply(p).to_point();
        assert!(approx_eq_3(result, [0.0, 1.0, 0.0]),
            "Möbius rotation: {:?}", result);
    }

    #[test]
    fn composition_versor() {
        // Translate then rotate = single versor
        let t = Cga3::translator(1.0, 0.0, 0.0);
        let r = Cga3::rotor(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let composed = r.gp(&t); // rotate ∘ translate

        let p = Cga3::up_point(0.0, 0.0, 0.0);
        let direct = r.sandwich(&t.sandwich(&p)).down_point();
        let via_comp = composed.sandwich(&p).down_point();
        assert!(approx_eq_3(direct, via_comp),
            "composition: direct={:?} composed={:?}", direct, via_comp);
    }
}
