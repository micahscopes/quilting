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

// CGA3 operates independently — no Möbius conversion needed.
// For QB patch evaluation, use Mobius directly from quaternion.rs.

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

    /// Outer product (wedge) of two multivectors — grade-raising part.
    pub fn outer(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..32u32 {
            if self.data[i as usize] == 0.0 { continue; }
            let gi = i.count_ones();
            for j in 0..32u32 {
                if other.data[j as usize] == 0.0 { continue; }
                let gj = j.count_ones();
                let (sign, blade) = blade_product(i as usize, j as usize);
                // Outer product keeps only the grade gi+gj part
                if (blade as u32).count_ones() == gi + gj {
                    result.data[blade] += sign * self.data[i as usize] * other.data[j as usize];
                }
            }
        }
        result
    }

    /// Inner product (left contraction).
    pub fn inner(&self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..32u32 {
            if self.data[i as usize] == 0.0 { continue; }
            let gi = i.count_ones() as i32;
            for j in 0..32u32 {
                if other.data[j as usize] == 0.0 { continue; }
                let gj = j.count_ones() as i32;
                let (sign, blade) = blade_product(i as usize, j as usize);
                // Left contraction keeps grade |gj - gi| when gj >= gi
                if gj >= gi && (blade as u32).count_ones() as i32 == gj - gi {
                    result.data[blade] += sign * self.data[i as usize] * other.data[j as usize];
                }
            }
        }
        result
    }

    /// Extract a specific grade from the multivector.
    pub fn grade(&self, g: u32) -> Self {
        let mut result = Self::ZERO;
        for i in 0..32u32 {
            if i.count_ones() == g {
                result.data[i as usize] = self.data[i as usize];
            }
        }
        result
    }

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

    /// Construct a sphere in CGA3 with center (cx,cy,cz) and radius r.
    ///
    /// In the conformal model, a sphere is a grade-1 vector:
    ///   S = C - ½r²ni
    /// where C = up_point(center) is the conformal embedding of the center.
    /// This is the "direct" representation (the sphere itself, not its dual).
    pub fn sphere(cx: f64, cy: f64, cz: f64, r: f64) -> Self {
        let mut s = Self::up_point(cx, cy, cz);
        // Subtract ½r²ni
        let half_r2 = 0.5 * r * r;
        let ni = Self::ni();
        for i in 0..32 {
            s.data[i] -= half_r2 * ni.data[i];
        }
        s
    }

    /// Construct a plane in CGA3 with normal (nx,ny,nz) and distance d from origin.
    ///
    /// A plane is: π = n + d·ni, where n = nx·e1 + ny·e2 + nz·e3.
    /// (n should be unit length.)
    pub fn plane(nx: f64, ny: f64, nz: f64, d: f64) -> Self {
        let len = (nx*nx + ny*ny + nz*nz).sqrt();
        let (nx, ny, nz) = if len > 1e-12 {
            (nx/len, ny/len, nz/len)
        } else {
            (0.0, 0.0, 1.0)
        };
        let mut mv = Self::ZERO;
        mv.data[E1] = nx;
        mv.data[E2] = ny;
        mv.data[E3] = nz;
        let ni = Self::ni();
        for i in 0..32 {
            mv.data[i] += d * ni.data[i];
        }
        mv
    }

    /// Reflection through a sphere (or plane). The sphere/plane versor S
    /// reflects points via the sandwich product: X' = S X S⁻¹.
    ///
    /// This is a single reflection — an *improper* conformal transformation
    /// (orientation-reversing). Compose two sphere reflections for a proper
    /// (orientation-preserving) Möbius transformation: inversion through
    /// a sphere pair, or equivalently, the composition S₂ S₁.
    ///
    /// For the unit sphere at origin: equivalent to the classical inversion
    /// x ↦ x/|x|².
    pub fn sphere_reflection(cx: f64, cy: f64, cz: f64, r: f64) -> Self {
        // The versor is the sphere itself — applying it as a sandwich
        // reflects through the sphere.
        Self::sphere(cx, cy, cz, r)
    }

    /// Inversion through a sphere pair: compose two sphere reflections
    /// to get a proper (orientation-preserving) Möbius transformation.
    ///
    /// Geometrically: reflects through sphere1, then through sphere2.
    pub fn sphere_inversion(
        c1: [f64; 3], r1: f64,
        c2: [f64; 3], r2: f64,
    ) -> Self {
        let s1 = Self::sphere(c1[0], c1[1], c1[2], r1);
        let s2 = Self::sphere(c2[0], c2[1], c2[2], r2);
        s2.gp(&s1) // apply s1 first, then s2
    }

    /// Transform a 3D point using this versor via CGA sandwich product.
    /// This works for ANY conformal transformation — no Möbius extraction needed.
    pub fn transform_point(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let p = Self::up_point(x, y, z);
        self.sandwich(&p).down_point()
    }
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
    fn transform_point_translate() {
        let t = Cga3::translator(5.0, 0.0, 0.0);
        let result = t.transform_point(1.0, 2.0, 3.0);
        assert!(approx_eq_3(result, [6.0, 2.0, 3.0]),
            "translate: {:?}", result);
    }

    #[test]
    fn transform_point_rotate() {
        let r = Cga3::rotor(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        let result = r.transform_point(1.0, 0.0, 0.0);
        assert!(approx_eq_3(result, [0.0, 1.0, 0.0]),
            "rotate: {:?}", result);
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

    #[test]
    fn unit_sphere_reflection() {
        // Reflection through the unit sphere at origin: x ↦ x/|x|²
        // Point (2,0,0) should map to (0.5, 0, 0)
        let s = Cga3::sphere_reflection(0.0, 0.0, 0.0, 1.0);
        let p = Cga3::up_point(2.0, 0.0, 0.0);
        let result = s.sandwich(&p).down_point();
        assert!(approx_eq_3(result, [0.5, 0.0, 0.0]),
            "unit sphere reflection of (2,0,0): {:?}", result);
    }

    #[test]
    fn sphere_reflection_on_sphere() {
        // A point ON the sphere should map to itself
        let s = Cga3::sphere_reflection(0.0, 0.0, 0.0, 2.0);
        let p = Cga3::up_point(2.0, 0.0, 0.0);
        let result = s.sandwich(&p).down_point();
        assert!(approx_eq_3(result, [2.0, 0.0, 0.0]),
            "point on sphere should be fixed: {:?}", result);
    }

    #[test]
    fn offset_sphere_reflection() {
        // Sphere centered at (1,0,0) with radius 1
        // Point (1,0,0) is the center — reflection maps center to infinity
        // Point (2,0,0) is on the sphere — should map to itself
        let s = Cga3::sphere_reflection(1.0, 0.0, 0.0, 1.0);
        let p = Cga3::up_point(2.0, 0.0, 0.0);
        let result = s.sandwich(&p).down_point();
        assert!(approx_eq_3(result, [2.0, 0.0, 0.0]),
            "point on offset sphere: {:?}", result);
    }

    #[test]
    fn sphere_inversion_proper() {
        // Two concentric sphere reflections = a dilation
        // Unit sphere then sphere of radius 2 at origin:
        // First: x ↦ x/|x|², then: x ↦ 4x/|x|²
        // Composed: x ↦ 4x (scaling by 4)
        let v = Cga3::sphere_inversion([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0], 2.0);
        let p = Cga3::up_point(1.0, 0.0, 0.0);
        let result = v.sandwich(&p).down_point();
        assert!(approx_eq_3(result, [4.0, 0.0, 0.0]),
            "double inversion should scale by r2²/r1²=4: {:?}", result);
    }

    #[test]
    fn sphere_reflection_transform_point() {
        let s = Cga3::sphere_reflection(0.0, 0.0, 0.0, 1.0);
        let result = s.transform_point(2.0, 0.0, 0.0);
        assert!(approx_eq_3(result, [0.5, 0.0, 0.0]),
            "sphere reflection: {:?}", result);
    }

    #[test]
    fn sphere_inversion_transform_point() {
        let v = Cga3::sphere_inversion([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0], 2.0);
        let result = v.transform_point(1.0, 2.0, 3.0);
        assert!(approx_eq_3(result, [4.0, 8.0, 12.0]),
            "sphere inversion (4x scale): {:?}", result);
    }
}
