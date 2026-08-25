//! Property tests for the conformal-dilation primitive (`conformal_lod`),
//! verifying the closed-form identities the LoD-density and cull design rest on,
//! against the real `Mobius`/`Quat` implementation. No wasm, no GL — native f64.
//!
//! Ported from Fable's verification harness; every identity below was checked to
//! machine precision across single reflections and 2-/3-fold compositions.

use quilting_core::conformal_lod::{
    ball_outside_frustum, closest_point_triangle, image_ball, ConformalPatch, ImageBound,
};
use quilting_core::quaternion::{Mobius, Quat};

/// Deterministic xorshift PRNG so failures reproduce exactly.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        (0..6).map(|_| self.next()).sum::<f64>() - 3.0
    }
    fn vec3(&mut self) -> [f64; 3] {
        [self.gauss() * 2.0, self.gauss() * 2.0, self.gauss() * 2.0]
    }
}

fn pt(p: [f64; 3]) -> Quat {
    Quat::from_point(p[0], p[1], p[2])
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Build a spread of transforms: single reflections + 2-/3-fold compositions.
fn transforms(rng: &mut Rng) -> Vec<Mobius> {
    let refl = |rng: &mut Rng| {
        let c = rng.vec3();
        Mobius::sphere_reflection(pt(c), 0.3 + rng.next() * 2.0)
    };
    let mut ts = Vec::new();
    for _ in 0..6 {
        ts.push(refl(rng));
    }
    for _ in 0..6 {
        let (a, b) = (refl(rng), refl(rng));
        ts.push(a.compose(&b));
    }
    for _ in 0..4 {
        let (a, b, c) = (refl(rng), refl(rng), refl(rng));
        ts.push(a.compose(&b).compose(&c));
    }
    ts
}

#[test]
fn pole_and_power_match_sphere_reflection() {
    // sphere_reflection(C, r): pole = C exactly, power = r² exactly.
    let m = Mobius::sphere_reflection(Quat::from_point(0.5, -0.2, 0.3), 2.0);
    let h = m.pole().expect("reflection has a pole");
    assert!(h.re().abs() < 1e-12, "pole must be a pure R³ point, re={}", h.re());
    assert!((h.im()[0] - 0.5).abs() < 1e-12);
    assert!((h.im()[1] + 0.2).abs() < 1e-12);
    assert!((h.im()[2] - 0.3).abs() < 1e-12);
    assert!((m.power() - 4.0).abs() < 1e-10, "power should be r²=4, got {}", m.power());
}

#[test]
fn pole_is_a_zero_of_bot_and_pure() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for m in transforms(&mut rng) {
        let h = m.pole().expect("constructed transforms all have a pole");
        let bot = m.c * h + m.d; // bot(h) must vanish
        assert!(bot.norm() < 1e-9 * (1.0 + m.d.norm()), "|bot(pole)| = {}", bot.norm());
        assert!(h.re().abs() < 1e-9 * (1.0 + h.norm()), "pole not pure: re={}", h.re());
    }
    // near-affine ⇒ no pole
    let affine = Mobius::new(Quat::I, Quat::ONE, Quat::ZERO, Quat::ONE);
    assert!(affine.pole().is_none());
}

#[test]
fn metric_identity_holds() {
    // |F(x) - F(y)| = k |x - y| / (|x - h| |y - h|)
    let mut rng = Rng(0xDEADBEEF01234567);
    let mut worst = 0.0f64;
    for m in transforms(&mut rng) {
        let h = m.pole().unwrap();
        let k = m.power();
        for _ in 0..50 {
            let (x, y) = (pt(rng.vec3()), pt(rng.vec3()));
            let lhs = (m.apply(x) - m.apply(y)).norm();
            let rhs = k * (x - y).norm() / ((x - h).norm() * (y - h).norm());
            worst = worst.max((lhs - rhs).abs() / (rhs.abs() + 1e-12));
        }
    }
    assert!(worst < 1e-9, "metric identity max rel err = {worst:.3e}");
}

#[test]
fn lambda_star_matches_repo_conformal_scale() {
    // λ*(x) = k/|x-h|² must equal Mobius::conformal_scale_at at the same point.
    let mut rng = Rng(0x1122334455667788);
    let mut worst = 0.0f64;
    for m in transforms(&mut rng) {
        let h = m.pole().unwrap();
        let k = m.power();
        for _ in 0..40 {
            let x = pt(rng.vec3());
            let lam_closed = k / (x - h).norm_sq();
            let lam_repo = m.conformal_scale_at(x);
            worst = worst.max((lam_closed - lam_repo).abs() / lam_repo);
        }
    }
    assert!(worst < 1e-9, "λ vs conformal_scale_at max rel err = {worst:.3e}");
}

#[test]
fn min_bot_sq_is_the_exact_minimum_over_triangle() {
    // Exactness proved two ways: (1) the minimum is ACHIEVED at x_star — a point
    // on the triangle — so |bot(x_star)|² must equal the closed form exactly;
    // (2) it lower-bounds a dense grid, so nothing on the triangle is smaller.
    // Together: closed form = the true minimum.
    let mut rng = Rng(0x0F1E2D3C4B5A6978);
    for m in transforms(&mut rng) {
        for _ in 0..12 {
            let (a, b, c) = (rng.vec3(), rng.vec3(), rng.vec3());
            let patch = ConformalPatch::new(&m, a, b, c).unwrap();
            let bot_at_xstar = (m.c * pt(patch.x_star) + m.d).norm_sq();
            let rel = (bot_at_xstar - patch.min_bot_sq).abs() / (patch.min_bot_sq + 1e-300);
            assert!(rel < 1e-9, "min not achieved at x_star: rel err {rel:.3e}");

            let n = 100;
            let mut grid_min = f64::INFINITY;
            for i in 0..=n {
                for j in 0..=(n - i) {
                    let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
                    let w = 1.0 - u - v;
                    let p = [a[0] * w + b[0] * u + c[0] * v, a[1] * w + b[1] * u + c[1] * v, a[2] * w + b[2] * u + c[2] * v];
                    grid_min = grid_min.min((m.c * pt(p) + m.d).norm_sq());
                }
            }
            assert!(patch.min_bot_sq <= grid_min * (1.0 + 1e-9), "closed {} > grid {}", patch.min_bot_sq, grid_min);
        }
    }
}

#[test]
fn interior_gate_zero_when_pole_projects_outside() {
    // A triangle far from and to one side of the pole: closest approach is on the
    // boundary ⇒ gate 0 (rim sampling already suffices there).
    let m = Mobius::sphere_reflection(Quat::from_point(0.0, 0.0, 0.0), 1.0);
    let patch = ConformalPatch::new(&m, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]).unwrap();
    assert!(patch.interior_gate.abs() < 1e-9, "gate={}", patch.interior_gate);

    // Pole (origin) hovering just above a triangle centered on it in z=0.3:
    // closest point is interior ⇒ gate > 0.
    let tri = ([1.0, 0.0, 0.3], [-0.5, 0.87, 0.3], [-0.5, -0.87, 0.3]);
    let p2 = ConformalPatch::new(&m, tri.0, tri.1, tri.2).unwrap();
    assert!(p2.interior_gate > 0.5, "interior pole gate={}", p2.interior_gate);
    // x_star should be ~directly under the pole (z=0.3, xy≈0)
    assert!(p2.x_star[2] > 0.29 && p2.x_star[0].abs() < 0.2 && p2.x_star[1].abs() < 0.2);
}

#[test]
fn closest_point_triangle_is_optimal() {
    // The returned point must be at least as close as any grid point, up to the
    // grid's own spacing (~diam/n). That plus the algorithm returning genuine
    // triangle points (vertex/edge/interior) makes it the exact projection.
    let mut rng = Rng(0xABCDEF0123456789);
    for _ in 0..500 {
        let (a, b, c, p) = (rng.vec3(), rng.vec3(), rng.vec3(), rng.vec3());
        let cp = closest_point_triangle(p, a, b, c);
        let n = 120usize;
        let mut brute = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=(n - i) {
                let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
                let w = 1.0 - u - v;
                let q = [a[0] * w + b[0] * u + c[0] * v, a[1] * w + b[1] * u + c[1] * v, a[2] * w + b[2] * u + c[2] * v];
                brute = brute.min(len(sub(q, p)));
            }
        }
        let diam = len(sub(a, b)).max(len(sub(b, c))).max(len(sub(a, c)));
        assert!(
            len(sub(cp, p)) <= brute + 2.0 * diam / n as f64,
            "closest point not optimal: {} vs grid {}",
            len(sub(cp, p)),
            brute
        );
    }
}

#[test]
fn image_ball_contains_the_whole_deformed_triangle() {
    // The cull ball must provably contain every image point of the triangle.
    let mut rng = Rng(0x55AA55AA55AA55AA);
    let mut worst_overflow = 0.0f64;
    let mut ball_cases = 0;
    for m in transforms(&mut rng) {
        for _ in 0..20 {
            // triangle placed away from the pole so we exercise the Ball branch
            let base = rng.vec3();
            let (a, b, c) = (
                [base[0] + rng.gauss() * 0.3, base[1] + rng.gauss() * 0.3, base[2] + rng.gauss() * 0.3],
                [base[0] + rng.gauss() * 0.3, base[1] + rng.gauss() * 0.3, base[2] + rng.gauss() * 0.3],
                [base[0] + rng.gauss() * 0.3, base[1] + rng.gauss() * 0.3, base[2] + rng.gauss() * 0.3],
            );
            let ImageBound::Ball { center, radius } = image_ball(&m, a, b, c) else { continue };
            ball_cases += 1;
            let n = 24;
            for i in 0..=n {
                for j in 0..=(n - i) {
                    let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
                    let w = 1.0 - u - v;
                    let p = [a[0] * w + b[0] * u + c[0] * v, a[1] * w + b[1] * u + c[1] * v, a[2] * w + b[2] * u + c[2] * v];
                    let img = m.apply(pt(p)).to_point();
                    let d = len(sub(img, center));
                    worst_overflow = worst_overflow.max((d - radius).max(0.0) / (radius + 1e-12));
                }
            }
        }
    }
    assert!(ball_cases > 50, "too few Ball cases exercised: {ball_cases}");
    // 5% inflation baked into image_ball's radius covers discretization; any
    // point escaping the ball is a real containment failure.
    assert!(worst_overflow < 1e-6, "image escaped cull ball by {worst_overflow:.3e}");
}

#[test]
fn image_ball_never_culls_when_pole_is_inside() {
    // A triangle straddling the pole: image reaches ∞, must be NeverCull.
    let m = Mobius::sphere_reflection(Quat::from_point(0.0, 0.0, 0.0), 1.0);
    let bound = image_ball(&m, [1.0, 0.0, 0.0], [-0.5, 0.87, 0.0], [-0.5, -0.87, 0.0]);
    assert!(matches!(bound, ImageBound::NeverCull), "pole-straddling face must never cull");
}

#[test]
fn ball_frustum_test_is_conservative_at_clip_boundaries() {
    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    assert!(!ball_outside_frustum(&identity, [0.0, 0.0, 0.0], 0.25));
    assert!(!ball_outside_frustum(&identity, [1.4, 0.0, 0.0], 0.5));
    assert!(ball_outside_frustum(&identity, [1.6, 0.0, 0.0], 0.5));
    assert!(ball_outside_frustum(&identity, [0.0, -2.0, 0.0], 0.25));
    assert!(ball_outside_frustum(&identity, [0.0, 0.0, 2.0], 0.25));

    // Column-major translation: x=3 maps to the clip origin, while x=0 is left.
    let translated = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -3.0, 0.0, 0.0, 1.0,
    ];
    assert!(!ball_outside_frustum(&translated, [3.0, 0.0, 0.0], 0.25));
    assert!(ball_outside_frustum(&translated, [0.0, 0.0, 0.0], 0.25));

    // Invalid bounds disable culling rather than risking a false negative.
    assert!(!ball_outside_frustum(&identity, [f64::NAN, 0.0, 0.0], 0.5));
    assert!(!ball_outside_frustum(&identity, [10.0, 0.0, 0.0], f64::INFINITY));
}

// ---- Acceptance oracle: build the real atlas patch, map it through the Möbius
// ---- + projection, and check the max on-screen sub-edge. ----

use quilting_core::atlas::TessellationAtlas;
use quilting_core::conformal_lod::conformal_edge_lods;
use quilting_core::sampling::PatchConfig;
use quilting_core::triangle::cartesian_to_bary;

fn sd(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Camera at origin looking down −z; simple pinhole. `None` if behind.
fn projector(q: Quat) -> Option<[f64; 2]> {
    let p = q.to_point();
    if p[2] >= -0.05 {
        return None;
    }
    let f = 800.0;
    Some([500.0 + f * p[0] / -p[2], 500.0 + f * p[1] / -p[2]])
}

/// Largest on-screen edge of the tessellated patch mapped onto triangle (v0,v1,v2).
fn worst_screen_subedge(
    atlas: &TessellationAtlas,
    res: [u32; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    m: &Mobius,
) -> f64 {
    let mesh = atlas.get_patch(res).expect("atlas has this patch");
    let screens: Vec<Option<[f64; 2]>> = mesh
        .positions
        .iter()
        .map(|&[x, y]| {
            let b = cartesian_to_bary(x, y);
            let rest = [
                b[0] * v0[0] + b[1] * v1[0] + b[2] * v2[0],
                b[0] * v0[1] + b[1] * v1[1] + b[2] * v2[1],
                b[0] * v0[2] + b[1] * v1[2] + b[2] * v2[2],
            ];
            projector(m.apply(pt(rest)))
        })
        .collect();
    let mut worst = 0.0f64;
    for t in &mesh.triangles {
        for (i, j) in [(0usize, 1usize), (1, 2), (0, 2)] {
            if let (Some(a), Some(b)) = (screens[t[i]], screens[t[j]]) {
                worst = worst.max(sd(a, b));
            }
        }
    }
    worst
}

/// The old rim-only drive (no interior floor / peak boost): the pre-fix behavior.
fn rim_only_lods(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], m: &Mobius, min_px: f64, cap: u32) -> [u32; 3] {
    let mid = |a: [f64; 3], b: [f64; 3]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let d0 = m.apply(pt(v0));
    let d1 = m.apply(pt(v1));
    let d2 = m.apply(pt(v2));
    let dma = m.apply(pt(mid(v1, v2)));
    let dmb = m.apply(pt(mid(v0, v2)));
    let dmc = m.apply(pt(mid(v0, v1)));
    let arc = |x: Quat, mm: Quat, y: Quat| -> f64 {
        match (projector(x), projector(mm), projector(y)) {
            (Some(a), Some(b), Some(c)) => sd(a, b) + sd(b, c),
            _ => 0.0,
        }
    };
    let snap = |px: f64| {
        let l = 2f64.powf((px / min_px).max(1.0).log2().round()) as u32;
        l.clamp(2, cap)
    };
    [snap(arc(d1, dma, d2)), snap(arc(d0, dmb, d2)), snap(arc(d0, dmc, d1))]
}

#[test]
fn interior_floor_catches_the_spike_fan() {
    let atlas = TessellationAtlas::build(&[1, 2, 4, 8, 16, 32, 64], &PatchConfig::default());
    let cap = 64u32;
    let min_px = 16.0;

    // Roughly equilateral face in front of the camera at z = -3.
    let s = 1.5;
    let v0 = [0.0, s, -3.0];
    let v1 = [-s * 0.866, -s * 0.5, -3.0];
    let v2 = [s * 0.866, -s * 0.5, -3.0];

    // Sphere reflection whose pole (= center) sits behind the face centroid, so it
    // projects into the interior: rim samples see mild dilation, the barycentric
    // centre a larger one — but mild enough that the fix resolves below the cap.
    let m = Mobius::sphere_reflection(Quat::from_point(0.0, 0.0, -3.7), 0.8);

    let rim = rim_only_lods(v0, v1, v2, &m, min_px, cap);
    let conf = conformal_edge_lods(v0, v1, v2, &m, projector, min_px, 2, cap, 1000.0 * std::f64::consts::SQRT_2);

    // The interior floor must lift the LoD above what the rim alone asked for.
    assert!(
        *conf.iter().max().unwrap() > *rim.iter().max().unwrap(),
        "interior floor didn't boost: conformal {conf:?} vs rim {rim:?}"
    );

    let worst_conf = worst_screen_subedge(&atlas, conf, v0, v1, v2, &m);
    let worst_rim = worst_screen_subedge(&atlas, rim, v0, v1, v2, &m);

    // OLD: the interior sub-triangles overshoot min_px badly — the spike-fan.
    assert!(
        worst_rim > 2.0 * min_px,
        "expected rim-only to under-tessellate; worst sub-edge {worst_rim:.1}px (target {:.0})",
        2.0 * min_px
    );
    // NEW: every sub-edge is within ~2·min_px — allowing the √2 slack that
    // power-of-2 LoD snapping can add (the ideal LoD lands between 2^k and
    // 2^(k+1); rounding to the nearer level leaves a sub-edge up to √2 over the
    // target). Beyond that the fix has genuinely under-tessellated. Skip only if
    // the demand legitimately saturated the atlas cap.
    let capped = conf.iter().any(|&l| l >= cap);
    let tol = 2.0 * min_px * std::f64::consts::SQRT_2;
    assert!(
        capped || worst_conf <= tol,
        "conformal LoD still under-tessellates: worst {worst_conf:.1}px > {tol:.0} (lods {conf:?})"
    );
    // And it must be a dramatic improvement over the rim-only spike-fan.
    assert!(
        worst_conf < worst_rim * 0.5,
        "conformal barely helped: {worst_conf:.0}px vs rim {worst_rim:.0}px"
    );
    println!("spike-fan oracle: rim lods {rim:?} worst {worst_rim:.0}px  →  conformal {conf:?} worst {worst_conf:.0}px");
}

// ---- Regression: LoD must be smooth across smooth geometry near the pole ----
// The peak-boost (∝1/d_e) and the interior-floor finite-diff Jacobian both go
// ill-conditioned near the pole; this catches the "patchy LoD on a smooth
// surface" glitch (the two triangles of one quad are near-identical geometry,
// so their LoD must not differ by more than one pow2 level).

fn grid_projector(q: Quat) -> Option<[f64; 2]> {
    let p = q.to_point();
    if p[2] >= -0.05 { return None; }
    let f = 800.0;
    Some([500.0 + f * p[0] / -p[2], 500.0 + f * p[1] / -p[2]])
}

#[test]
fn lod_is_smooth_across_a_grid_straddling_the_pole() {
    let n = 24usize;
    let step = 0.12;
    let idx = |i: usize, j: usize| j * (n + 1) + i;
    let mut verts = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            verts.push([(i as f64 - n as f64 / 2.0) * step, (j as f64 - n as f64 / 2.0) * step, -3.0]);
        }
    }
    // pole just in front of the grid centre, offset so it projects into the interior
    let m = Mobius::sphere_reflection(Quat::from_point(0.05, 0.0, -2.9), 0.4);
    let min_px = 8.0;
    let lod = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> u32 {
        *conformal_edge_lods(a, b, c, &m, grid_projector, min_px, 2, 256, 1000.0 * std::f64::consts::SQRT_2).iter().max().unwrap()
    };

    let mut worst = 1.0f64;
    let mut worst_at = (0usize, 0usize);
    for j in 0..n {
        for i in 0..n {
            // the two triangles of quad (i,j) share the diagonal — near-identical geometry
            let t0 = lod(verts[idx(i, j)], verts[idx(i + 1, j)], verts[idx(i, j + 1)]);
            let t1 = lod(verts[idx(i + 1, j)], verts[idx(i + 1, j + 1)], verts[idx(i, j + 1)]);
            let r = (t0.max(t1) as f64) / (t0.min(t1) as f64);
            if r > worst {
                worst = r;
                worst_at = (i, j);
            }
        }
    }
    println!("worst adjacent-triangle LoD ratio = {worst} at quad {worst_at:?}");
    assert!(
        worst <= 2.0,
        "patchy LoD on smooth geometry: two triangles of one quad differ by {worst}× at {worst_at:?}"
    );
}

#[test]
fn interior_floor_holds_for_off_center_poles() {
    // Fable's finding 3: the old linear gate undershot as the pole slid off the
    // centroid toward an edge. The robust (ungated, analytic-λ*) floor must keep
    // the worst sub-edge bounded for every pole position, not just the centre.
    let atlas = TessellationAtlas::build(&[1, 2, 4, 8, 16, 32, 64], &PatchConfig::default());
    let cap = 64u32;
    let min_px = 16.0;
    let s = 1.5;
    let v0 = [0.0, s, -3.0];
    let v1 = [-s * 0.866, -s * 0.5, -3.0];
    let v2 = [s * 0.866, -s * 0.5, -3.0];
    let tol = 2.0 * min_px * std::f64::consts::SQRT_2;
    for &(px, py, pz, r) in &[
        (0.0, 0.0, -3.7, 0.8),   // centred
        (0.0, -0.40, -3.7, 0.8), // toward an edge
        (0.0, -0.70, -3.6, 0.6), // near an edge (the case that broke before)
        (0.30, -0.30, -3.6, 0.6),
        (0.0, 0.60, -3.6, 0.6), // toward a vertex
    ] {
        let m = Mobius::sphere_reflection(Quat::from_point(px, py, pz), r);
        let conf = conformal_edge_lods(v0, v1, v2, &m, projector, min_px, 2, cap, 1000.0 * std::f64::consts::SQRT_2);
        let worst = worst_screen_subedge(&atlas, conf, v0, v1, v2, &m);
        let capped = conf.iter().any(|&l| l >= cap);
        assert!(
            capped || worst <= tol,
            "off-center pole ({px},{py},{pz}) r={r}: worst {worst:.0}px > {tol:.0} lods {conf:?}"
        );
    }
}

#[test]
fn degenerate_face_falls_back_without_nan() {
    // Fable's finding 5: collinear/needle faces must not NaN the pole guard/floor.
    let m = Mobius::sphere_reflection(Quat::from_point(0.0, 0.0, -3.1), 0.5);
    let lods = conformal_edge_lods([0.0, 0.0, -3.0], [1.0, 0.0, -3.0], [2.0, 0.0, -3.0], &m, projector, 16.0, 2, 64, 1000.0 * std::f64::consts::SQRT_2);
    for l in lods {
        assert!((2..=64).contains(&l) && l.is_power_of_two(), "degenerate produced garbage LoD: {lods:?}");
    }
}

#[test]
fn exact_pole_is_capped_by_raster_capacity() {
    let v0 = [-1.0, -1.0, -3.0];
    let v1 = [1.0, -1.0, -3.0];
    let v2 = [0.0, 1.0, -3.0];
    let m = Mobius::sphere_reflection(Quat::from_point(0.0, 0.0, -3.0), 1.0);
    let lods = conformal_edge_lods(v0, v1, v2, &m, projector, 64.0, 1, 128, 800.0);
    assert_eq!(lods, [8, 8, 8]);
}
