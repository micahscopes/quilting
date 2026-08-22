//! Conformal-dilation-aware LoD density and culling primitive.
//!
//! The Möbius denominator `bot(q) = c·q + d` is affine in position, so `|bot|²`
//! is a convex quadratic. Therefore the point of maximum conformal dilation over
//! a triangle (its closest approach to the pole `h`, where the local isotropic
//! scale `λ = k/|q−h|²` is largest) is the closed-form closest point of the
//! triangle to `h`. Sampling only the triangle's rim (as the plain LoD pass does)
//! misses this peak whenever `h` projects into the triangle interior — the
//! spike-fan / funnel case. This module computes the exact quantities so both the
//! LoD density and a conformally-aware image bound can key on them.
//!
//! The GLSL twin lives in `crates/quilting-renderer/shaders/lod_compute.vert.glsl`
//! and MUST mirror [`closest_point_triangle`]'s branch order exactly.

use crate::quaternion::{Mobius, Quat, POLE_PROXIMITY_NORM_SQ};

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared distance from point `p` to the segment `[a, b]`.
pub fn seg_dist_sq(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let denom = dot(ab, ab).max(1e-30);
    let t = (dot(sub(p, a), ab) / denom).clamp(0.0, 1.0);
    let q = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    dot(sub(p, q), sub(p, q))
}

/// Closest point on triangle `abc` to `p` (Ericson, *Real-Time Collision
/// Detection* §5.1.5). The GLSL twin must keep this exact branch order so the
/// CPU and GPU LoD passes agree.
pub fn closest_point_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub(c, b);
        return [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
    }
    let denom = 1.0 / (va + vb + vc);
    [
        a[0] + ab[0] * (vb * denom) + ac[0] * (vc * denom),
        a[1] + ab[1] * (vb * denom) + ac[1] * (vc * denom),
        a[2] + ab[2] * (vb * denom) + ac[2] * (vc * denom),
    ]
}

/// Closed-form per-face conformal-dilation quantities for a triangle `(v0,v1,v2)`.
/// Edge indexing matches the LoD passes: edge a = (v1,v2), b = (v0,v2), c = (v0,v1).
#[derive(Clone, Copy, Debug)]
pub struct ConformalPatch {
    /// Exact `min |c·q + d|²` over the *solid* triangle (≤ any boundary sample).
    /// Replaces the 6-sample `min_bot_sq` used by the pole guard.
    pub min_bot_sq: f64,
    /// Peak conformal dilation `λ* = k / d_T²` over the face.
    pub lambda_star: f64,
    /// Rest-space point of maximum dilation (closest point of the face to `h`).
    pub x_star: [f64; 3],
    /// Per-edge closest-approach² to the pole, `[a, b, c]`.
    pub edge_dist_sq: [f64; 3],
    /// Interior gate `g = 1 − d_T²/d_∂² ∈ [0,1]`; `0` unless `h` projects into the
    /// interior (then the rim samples miss the dilation and this rises).
    pub interior_gate: f64,
}

impl ConformalPatch {
    /// Compute for `(v0,v1,v2)`. `None` for (near-)affine transforms (no finite
    /// pole ⇒ no interior dilation to correct for).
    pub fn new(m: &Mobius, v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Option<Self> {
        let h = m.pole()?;
        // Degenerate (collinear / needle) faces have no well-defined closest point
        // — closest_point_triangle would divide by zero. Bail so callers fall back
        // to the rim, rather than propagate NaN through the pole guard and floor.
        let e1 = sub(v1, v0);
        let e2 = sub(v2, v0);
        let nrm = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        if dot(nrm, nrm) < 1e-24 {
            return None;
        }
        let k = m.power();
        let hw2 = h.re() * h.re();
        let him = h.im();
        let c2 = m.c.norm_sq();

        let d_a = hw2 + seg_dist_sq(him, v1, v2);
        let d_b = hw2 + seg_dist_sq(him, v0, v2);
        let d_c = hw2 + seg_dist_sq(him, v0, v1);
        let d_boundary = d_a.min(d_b).min(d_c);

        let xs = closest_point_triangle(him, v0, v1, v2);
        let dv = sub(xs, him);
        let d_t = hw2 + dot(dv, dv);

        Some(Self {
            min_bot_sq: c2 * d_t,
            lambda_star: k / d_t.max(1e-300),
            x_star: xs,
            edge_dist_sq: [d_a, d_b, d_c],
            interior_gate: (1.0 - d_t / d_boundary.max(1e-300)).max(0.0),
        })
    }
}

/// Conservative bound on the Möbius *image* of a triangle, for culling. These
/// maps are conformal and preserve R³, so a sphere enclosing the (rest) triangle
/// maps to a sphere; its image is found by the antipodal construction along the
/// line through the pole.
#[derive(Clone, Copy, Debug)]
pub enum ImageBound {
    /// The pole lies inside (or within 5% of) the enclosing sphere: the image can
    /// reach arbitrarily far (toward ∞), so the face must never be culled.
    NeverCull,
    /// A ball provably containing the entire Möbius image of the triangle.
    Ball { center: [f64; 3], radius: f64 },
}

/// Image bound for triangle `(v0,v1,v2)` under `m`.
pub fn image_ball(m: &Mobius, v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> ImageBound {
    let ctr = [
        (v0[0] + v1[0] + v2[0]) / 3.0,
        (v0[1] + v1[1] + v2[1]) / 3.0,
        (v0[2] + v1[2] + v2[2]) / 3.0,
    ];
    let s = dot(sub(v0, ctr), sub(v0, ctr))
        .max(dot(sub(v1, ctr), sub(v1, ctr)))
        .max(dot(sub(v2, ctr), sub(v2, ctr)))
        .sqrt();

    // Antipode direction: through the pole if there is one, else arbitrary
    // (affine ⇒ similarity ⇒ any diameter maps to a diameter).
    let u = match m.pole() {
        Some(h) => {
            let him = h.im();
            let hw2 = h.re() * h.re();
            let to_ctr = sub(ctr, him);
            let pole_dist_sq = hw2 + dot(to_ctr, to_ctr);
            let sr = 1.05 * s;
            if pole_dist_sq <= sr * sr {
                return ImageBound::NeverCull;
            }
            let inv = 1.0 / dot(to_ctr, to_ctr).sqrt().max(1e-30);
            [to_ctr[0] * inv, to_ctr[1] * inv, to_ctr[2] * inv]
        }
        None => [1.0, 0.0, 0.0],
    };

    let apply = |p: [f64; 3]| m.apply(Quat::from_point(p[0], p[1], p[2])).to_point();
    let fp = apply([ctr[0] + s * u[0], ctr[1] + s * u[1], ctr[2] + s * u[2]]);
    let fm = apply([ctr[0] - s * u[0], ctr[1] - s * u[1], ctr[2] - s * u[2]]);
    let center = [(fp[0] + fm[0]) * 0.5, (fp[1] + fm[1]) * 0.5, (fp[2] + fm[2]) * 0.5];
    let radius = dot(sub(fp, fm), sub(fp, fm)).sqrt() * 0.5;
    ImageBound::Ball { center, radius }
}

/// Return true only when a world-space ball lies wholly outside one clip plane.
///
/// `vp` is a column-major WebGL view-projection matrix. The six homogeneous
/// clip planes are `row3 ± row{0,1,2}`; scaling each plane by the radius of its
/// spatial normal makes the test independent of plane normalization.
pub fn ball_outside_frustum(vp: &[f64; 16], center: [f64; 3], radius: f64) -> bool {
    if radius < 0.0
        || !radius.is_finite()
        || center.iter().any(|component| !component.is_finite())
        || vp.iter().any(|component| !component.is_finite())
    {
        return false;
    }

    let row = |index: usize| [vp[index], vp[4 + index], vp[8 + index], vp[12 + index]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let plane = |axis: [f64; 4], sign: f64| {
        [
            r3[0] + sign * axis[0],
            r3[1] + sign * axis[1],
            r3[2] + sign * axis[2],
            r3[3] + sign * axis[3],
        ]
    };

    [
        plane(r0, 1.0),
        plane(r0, -1.0),
        plane(r1, 1.0),
        plane(r1, -1.0),
        plane(r2, 1.0),
        plane(r2, -1.0),
    ]
    .into_iter()
    .any(|p| {
        let signed = p[0] * center[0] + p[1] * center[1] + p[2] * center[2] + p[3];
        let normal_len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        normal_len > 0.0 && signed < -radius * normal_len
    })
}

#[inline]
fn q_from(p: [f64; 3]) -> Quat {
    Quat::from_point(p[0], p[1], p[2])
}
#[inline]
fn d3(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}
#[inline]
fn s2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}
#[inline]
fn snap_pow2(v: f64) -> f64 {
    2f64.powf(v.max(1.0).log2().round())
}

/// Interior-complete per-edge LoDs for triangle `(v0,v1,v2)` under `m`, projected
/// to screen pixels by `project` (a *deformed* world point → screen px, or `None`
/// if behind the camera), targeting ~`min_px` on screen per subdivision.
///
/// Returns three edge LoDs in canonical order `[a=(v1,v2), b=(v0,v2), c=(v0,v1)]`.
/// On top of the rim screen-arcs (corner→deformed-midpoint→corner), it adds:
///  - a per-edge **peak boost** `max(1, (k/d_e²)·L_e / warc_e)` so uniform-in-t
///    seeds don't overshoot min_px where an edge grazes the pole, and
///  - a gated **interior floor** at the max-dilation point `x*` (all three edges,
///    since the atlas grades interior density as the geometric mean of the edges),
///  - and the exact pole guard `|c|²·d_T² < POLE_PROXIMITY_NORM_SQ ⇒ max_lod`.
///
/// The GLSL twin in `lod_compute.vert.glsl` mirrors this; `min_lod`/`max_lod` are
/// the clamp bounds (`max_lod` = the built atlas cap on the GPU path).
pub fn conformal_edge_lods(
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    m: &Mobius,
    project: impl Fn(Quat) -> Option<[f64; 2]>,
    min_px: f64,
    min_lod: u32,
    max_lod: u32,
) -> [u32; 3] {
    let clampl = |v: f64| (snap_pow2(v) as u32).clamp(min_lod, max_lod);
    let mid = |a: [f64; 3], b: [f64; 3]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];

    let d0 = m.apply(q_from(v0));
    let d1 = m.apply(q_from(v1));
    let d2q = m.apply(q_from(v2));
    let dma = m.apply(q_from(mid(v1, v2)));
    let dmb = m.apply(q_from(mid(v0, v2)));
    let dmc = m.apply(q_from(mid(v0, v1)));

    // rim screen arc: corner → deformed edge-midpoint → corner (Fix-A style)
    let rim = |x: Quat, mm: Quat, y: Quat| -> Option<f64> {
        Some(s2(project(x)?, project(mm)?) + s2(project(mm)?, project(y)?))
    };
    let px_a = rim(d1, dma, d2q);
    let px_b = rim(d0, dmb, d2q);
    let px_c = rim(d0, dmc, d1);

    let patch = match ConformalPatch::new(m, v0, v1, v2) {
        Some(p) => p,
        None => {
            // Affine (no finite pole): rim drive only.
            let f = |px: Option<f64>| px.map(|p| clampl(p / min_px)).unwrap_or(min_lod);
            return [f(px_a), f(px_b), f(px_c)];
        }
    };
    if patch.min_bot_sq < POLE_PROXIMITY_NORM_SQ {
        return [max_lod, max_lod, max_lod];
    }

    let le = [d3(v1, v2), d3(v0, v2), d3(v0, v1)];
    let l_max = le[0].max(le[1]).max(le[2]);

    // Interior floor — robust form, replacing the ill-conditioned gate +
    // finite-difference-of-F. The LoD that makes the tessellated sub-edge at the
    // max-dilation point x* about min_px on screen is  λ* · ρ · L_max, where:
    //   • λ* = patch.lambda_star = k / d_T²  is the CLOSED-FORM peak conformal
    //     scale (no differencing of the near-singular map — this was the source of
    //     the patchy, per-face-inconsistent LoD), and
    //   • ρ is the projection Jacobian at the well-conditioned IMAGE point
    //     y* = F(x*), found by finite differences there (a benign point, not the
    //     pole).
    // Applied to all three edges via `max`: the peak point needs them all, and
    // faces far from the pole have a tiny λ* so it stays inert (no gate needed).
    // If y* falls behind the camera the face wraps past the near plane — leave it
    // to the rim + the cull rather than emit a spurious spike.
    let mut px_int = 0.0;
    let y_star = m.apply(q_from(patch.x_star));
    if let Some(s0) = project(y_star) {
        let yc = y_star.to_point();
        let eps = 1e-3;
        let mut rho = 0.0f64;
        for dir in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5774, 0.5774, 0.5774]] {
            let yp = Quat::from_point(yc[0] + eps * dir[0], yc[1] + eps * dir[1], yc[2] + eps * dir[2]);
            if let Some(sp) = project(yp) {
                rho = rho.max(s2(sp, s0) / eps);
            }
        }
        px_int = patch.lambda_star * rho * l_max;
    }

    let drive = |px: Option<f64>| -> u32 {
        let base = px.unwrap_or(0.0).max(px_int);
        if base <= 0.0 {
            return min_lod;
        }
        clampl(base / min_px)
    };
    [drive(px_a), drive(px_b), drive(px_c)]
}
