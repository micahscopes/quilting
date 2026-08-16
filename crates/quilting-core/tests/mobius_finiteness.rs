//! Deterministic repro harness for the "degenerate vertices" bug: spikes,
//! collapses, and NaN normals when a Möbius pole passes near the mesh.
//!
//! The CPU pipeline runs f64 and the GPU runs f32; the bug lives in the gap,
//! so asserting on f64 values proves nothing. This file mirrors the shader
//! math in f32 (see the `shader mirror` section) and checks three things per
//! Möbius transform:
//!   (a) the CPU instance packing stays finite,
//!   (b) CPU LODs stay powers of two in range, and pole-adjacent faces get
//!       MAX_LOD rather than collapsing to a tiny LOD,
//!   (c) the GPU's f32 evaluation stays finite and bounded.
//!
//! Key design point learned the hard way: a *uniform grid* sweep of Möbius
//! parameters finds nothing on a dense mesh — 27 pole positions x 3 radii over
//! both demo meshes gave zero failures — because the pole almost never lands
//! near a vertex by chance. The sweep must be ADVERSARIAL: walk the pole onto
//! actual mesh vertices and edge midpoints at a ladder of offsets straddling
//! the f32 cancellation cliff, and sample patch INTERIORS (barycentric grid),
//! not just corners.
//!
//! Run with:  cargo test -p quilting-core --test mobius_finiteness

use quilting_core::evaluate::{compute_instances_with_uvs, MAX_LOD};
use quilting_core::quaternion::{Mobius, Quat};
use quilting_mesh::HalfEdgeMesh;

// ---------------------------------------------------------------------------
// Shader mirror: f32 copies of shaders/math/quaternion.wgsl and the fused
// Möbius-QB evaluation in shaders/vertex/main.wgsl. Every formula here must
// track the shader line-for-line — when the shader changes, change this too.
// ---------------------------------------------------------------------------
type Q = [f32; 4]; // (w, x, y, z) — matches the WGSL vec4 layout

fn qmul(a: Q, b: Q) -> Q {
    [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
     a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
     a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
     a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]
}
fn qdot(a: Q, b: Q) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3] }
fn qadd(a: Q, b: Q) -> Q { [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]] }
fn qsub(a: Q, b: Q) -> Q { [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]] }
fn qs(a: Q, s: f32) -> Q { [a[0]*s, a[1]*s, a[2]*s, a[3]*s] }
fn qconj(a: Q) -> Q { [a[0], -a[1], -a[2], -a[3]] }
fn f32q(q: Quat) -> Q { [q.w as f32, q.x as f32, q.y as f32, q.z as f32] }
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
const IDENTITY_WEIGHT: Q = [1.0, 0.0, 0.0, 0.0]; // instance_layout constant weights

// Mirror of qinv in shaders/math/quaternion.wgsl (sentinel convention).
fn qinv(q: Q) -> Q {
    let d = qdot(q, q);
    if d < 1e-20 { return [1e10, 0.0, 0.0, 0.0]; }
    [q[0]/d, -q[1]/d, -q[2]/d, -q[3]/d]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Mirror of POSITION_CLAMP in vertex/main.wgsl.
const POSITION_CLAMP: f32 = 1.0e4;

/// Budget for a "healthy" frame. The vertex shader clamps evaluated positions
/// to POSITION_CLAMP (models are normalized to unit half-extent), so anything
/// past it — or non-finite — is a bug, not extreme-but-valid geometry.
const MAX_POSITION_MAG: f32 = POSITION_CLAMP;

#[derive(Debug, Default, Clone, Copy)]
struct PatchReport {
    face: usize,
    bary: [f32; 3],
    /// Smallest |bot|^2 seen anywhere (diagnostic only — it is an input
    /// condition, not an output defect; poles ON the mesh legitimately give 0).
    bot_sq_min: f32,
    position_mag: f32,
    /// dot(n, n) for the analytic normal — this is what overflows first.
    normal_dot: f32,
    fade: f32,
    nonfinite_position: bool,
    nonfinite_normal: bool,
    /// The smooth-normal Möbius differential path (main.wgsl has_smooth branch)
    /// went non-finite.
    nonfinite_smooth: bool,
}

impl PatchReport {
    fn failed(&self) -> bool {
        self.nonfinite_position || self.nonfinite_normal || self.nonfinite_smooth
            || self.position_mag > MAX_POSITION_MAG * 1.001
    }
}

/// Evaluate the fused Möbius-QB surface over a barycentric grid, exactly as
/// `vertex/main.wgsl` does (position, analytic normal, smooth-normal
/// differential, fade), and return the worst sample.
///
/// `pole` enables a prefilter: faces further than a few edge lengths from the
/// pole can only see |bot| bounded away from zero (bot is linear over the
/// patch), so they get a cheap corner-only check while faces near the pole get
/// the full interior grid. `grid` should be >= 8 — the pole usually lands
/// inside a face, and corner-only checks miss it.
fn worst_sample(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    mobius: &Mobius,
    grid: usize,
    pole: [f64; 3],
) -> PatchReport {
    let (a, b, c, d) = (f32q(mobius.a), f32q(mobius.b), f32q(mobius.c), f32q(mobius.d));
    let mut worst = PatchReport { bot_sq_min: f32::INFINITY, ..Default::default() };

    for (fi, t) in faces.iter().enumerate() {
        let pf64: [[f64; 3]; 3] = std::array::from_fn(|k| positions[t[k]]);
        let p: [Q; 3] = std::array::from_fn(|k| {
            [0.0, pf64[k][0] as f32, pf64[k][1] as f32, pf64[k][2] as f32]
        });
        // Prefilter: dense grid only when the pole is within ~3 edge lengths.
        let d3 = |a: &[f64; 3], b: &[f64; 3]| -> f64 {
            ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt()
        };
        let max_edge = d3(&pf64[0], &pf64[1]).max(d3(&pf64[1], &pf64[2])).max(d3(&pf64[0], &pf64[2]));
        let near = (0..3).map(|k| d3(&pf64[k], &pole)).fold(f64::INFINITY, f64::min)
            <= 3.0 * max_edge;
        let g = if near { grid } else { 1 };

        // The live instance buffer supplies identity weights; the conformal
        // weight is derived in-shader from the Möbius uniforms.
        let pw: [Q; 3] = std::array::from_fn(|i| qmul(qadd(qmul(a, p[i]), b), IDENTITY_WEIGHT));
        let bw: [Q; 3] = std::array::from_fn(|i| qmul(qadd(qmul(c, p[i]), d), IDENTITY_WEIGHT));

        // Smooth-normal Möbius differential per corner (main.wgsl is_mobius
        // branch): rn_i = (a - M_i*c) * n_i * inv(bot_i). Any unit normal
        // exercises the arithmetic; use the face normal for all corners.
        let e01 = [p[1][1]-p[0][1], p[1][2]-p[0][2], p[1][3]-p[0][3]];
        let e02 = [p[2][1]-p[0][1], p[2][2]-p[0][2], p[2][3]-p[0][3]];
        let fnrm = cross3(e01, e02);
        let fl = (fnrm[0]*fnrm[0] + fnrm[1]*fnrm[1] + fnrm[2]*fnrm[2]).sqrt();
        let sn = if fl > 1e-12 { [fnrm[0]/fl, fnrm[1]/fl, fnrm[2]/fl] } else { [0.0, 0.0, 1.0] };
        let rn: [[f32; 3]; 3] = std::array::from_fn(|i| {
            let bot_i = qadd(qmul(c, p[i]), d);
            let m_i = qmul(qadd(qmul(a, p[i]), b), qinv(bot_i));
            let a_i = qsub(a, qmul(m_i, c));
            let r = qmul(qmul(a_i, [0.0, sn[0], sn[1], sn[2]]), qinv(bot_i));
            [r[1], r[2], r[3]]
        });

        for i in 0..=g {
            for j in 0..=(g - i) {
                let l = [i as f32 / g as f32, j as f32 / g as f32, 0.0];
                let l = [l[0], l[1], 1.0 - l[0] - l[1]];
                let top = qadd(qadd(qs(pw[0], l[0]), qs(pw[1], l[1])), qs(pw[2], l[2]));
                let bot = qadd(qadd(qs(bw[0], l[0]), qs(bw[1], l[1])), qs(bw[2], l[2]));
                let bot_sq = qdot(bot, bot);
                let bi = qinv(bot);
                let x = qmul(top, bi);
                let mut pos = [x[1], x[2], x[3]];
                // vs_main clamps evaluated positions to POSITION_CLAMP.
                let pr = (pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]).sqrt();
                if pr > POSITION_CLAMP {
                    for v in pos.iter_mut() { *v *= POSITION_CLAMP / pr; }
                }
                let mag = pos[0].abs().max(pos[1].abs()).max(pos[2].abs());

                // Analytic normal via the quotient rule (main.wgsl
                // eval_mobius_qb): right-multiplied by conj(bot), which keeps
                // the direction of the bot⁻¹ form without its 1/|bot|²
                // magnitude.
                let cbq = qconj(bot);
                let xdu = qmul(qsub(qsub(pw[1], pw[0]), qmul(x, qsub(bw[1], bw[0]))), cbq);
                let xdv = qmul(qsub(qsub(pw[2], pw[0]), qmul(x, qsub(bw[2], bw[0]))), cbq);
                let n = cross3([xdu[1], xdu[2], xdu[3]], [xdv[1], xdv[2], xdv[3]]);
                // WGSL length(n) is sqrt(dot(n, n)); dot is the overflow risk.
                let ndot = n[0]*n[0] + n[1]*n[1] + n[2]*n[2];

                // Smooth-normal blend (main.wgsl: max-component prescale, then
                // normalize).
                let ns: [f32; 3] = std::array::from_fn(|k| {
                    l[0]*rn[0][k] + l[1]*rn[1][k] + l[2]*rn[2][k]
                });
                let nmax = ns[0].abs().max(ns[1].abs()).max(ns[2].abs());
                let sdot = if nmax > 1e-20 {
                    let nn = [ns[0]/nmax, ns[1]/nmax, ns[2]/nmax];
                    nn[0]*nn[0] + nn[1]*nn[1] + nn[2]*nn[2]
                } else {
                    0.0 // shader keeps the analytic normal
                };

                let cand = PatchReport {
                    face: fi,
                    bary: l,
                    bot_sq_min: bot_sq.min(worst.bot_sq_min),
                    position_mag: if mag.is_finite() { mag } else { f32::INFINITY },
                    normal_dot: ndot,
                    fade: smoothstep(0.0001, 0.001, bot_sq),
                    nonfinite_position: !mag.is_finite(),
                    nonfinite_normal: !ndot.is_finite(),
                    nonfinite_smooth: !sdot.is_finite(),
                };
                let worse = (cand.failed() && !worst.failed())
                    || (cand.failed() == worst.failed() && cand.position_mag > worst.position_mag);
                if worse {
                    let keep_min = worst.bot_sq_min.min(cand.bot_sq_min);
                    worst = cand;
                    worst.bot_sq_min = keep_min;
                } else {
                    worst.bot_sq_min = worst.bot_sq_min.min(bot_sq);
                }
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// f32 mirror of quilting-renderer/shaders/lod_compute.vert.glsl (GPU LOD pass)
// ---------------------------------------------------------------------------

fn snap_pow2(v: f32) -> f32 { v.max(1.0).log2().round().exp2() }

/// Per-face uniform LOD as the GLSL pass computes it (screen attenuation off).
/// Returns the per-face LOD.
fn glsl_lod_pass(
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    mobius: &Mobius,
    density: f32,
    mesh_radius: f32,
    max_lod: f32,
) -> Vec<f32> {
    let (a, b, c, d) = (f32q(mobius.a), f32q(mobius.b), f32q(mobius.c), f32q(mobius.d));
    let mob = |p: [f32; 3], min_bot_sq: &mut f32| -> [f32; 3] {
        let q = [0.0, p[0], p[1], p[2]];
        let top = qadd(qmul(a, q), b);
        let bot = qadd(qmul(c, q), d);
        *min_bot_sq = min_bot_sq.min(qdot(bot, bot));
        let r = qmul(top, qinv(bot));
        [r[1], r[2], r[3]]
    };
    let d3 = |a: [f32; 3], b: [f32; 3]| -> f32 {
        ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt()
    };
    faces.iter().map(|t| {
        let p: [[f32; 3]; 3] = std::array::from_fn(|k| {
            let v = positions[t[k]];
            [v[0] as f32, v[1] as f32, v[2] as f32]
        });
        let mid = |x: [f32; 3], y: [f32; 3]| -> [f32; 3] {
            [(x[0]+y[0])*0.5, (x[1]+y[1])*0.5, (x[2]+y[2])*0.5]
        };
        let mut mb = f32::INFINITY;
        let dv: [[f32; 3]; 3] = std::array::from_fn(|k| mob(p[k], &mut mb));
        let dm = [
            mob(mid(p[1], p[2]), &mut mb),
            mob(mid(p[0], p[2]), &mut mb),
            mob(mid(p[0], p[1]), &mut mb),
        ];
        let med = d3(dv[0], dm[0]).max(d3(dv[1], dm[1])).max(d3(dv[2], dm[2]));
        let target = mesh_radius / density;
        let lod = snap_pow2(med / target).clamp(2.0, max_lod);
        // Pole proximity saturation (mirrors the min_bot_sq check in the GLSL).
        if mb < 1e-8 { max_lod } else { lod }
    }).collect()
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

fn load_normalized(path: &str) -> Option<(Vec<[f64; 3]>, Vec<[usize; 3]>)> {
    let bytes = std::fs::read(path).ok()?;
    let scene = quilting_gltf::load_gltf(&bytes).ok()?;
    let (mut pos, mut tris) = (Vec::new(), Vec::new());
    for m in &scene.meshes {
        for pr in &m.primitives {
            let off = pos.len();
            pos.extend_from_slice(&pr.positions);
            for f in &pr.triangles { tris.push([f[0]+off, f[1]+off, f[2]+off]); }
        }
    }
    if pos.is_empty() { return None; }
    // Mirrors load_gltf_data's normalization (quilting-wasm): centre the bbox
    // and scale the largest half-extent to 1.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &pos { for i in 0..3 { lo[i] = lo[i].min(p[i]); hi[i] = hi[i].max(p[i]); } }
    let ctr = [(lo[0]+hi[0])*0.5, (lo[1]+hi[1])*0.5, (lo[2]+hi[2])*0.5];
    let ext = ((hi[0]-lo[0]).max(hi[1]-lo[1]).max(hi[2]-lo[2])) * 0.5;
    let s = if ext > 1e-10 { 1.0/ext } else { 1.0 };
    Some((pos.iter().map(|p| [(p[0]-ctr[0])*s, (p[1]-ctr[1])*s, (p[2]-ctr[2])*s]).collect(), tris))
}

/// Adversarial pole placements: sampled vertices and edge midpoints, each at a
/// ladder of offsets straddling the f32 cancellation cliff.
///
/// The ladder matters: 1e-2 is where visible spikes start on a unit model,
/// 1e-5 is where the analytic normal used to overflow f32, and at ~1e-7 the
/// f32 rounding of `c*p + d` cancels to exactly zero (nine orders above the
/// qinv guard). 0.0 is the bit-exact hit.
fn adversarial_poles(pos: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<(Mobius, [f64; 3])> {
    let mut out = Vec::new();
    let vstep = (pos.len() / 8).max(1);
    let fstep = (faces.len() / 8).max(1);
    let offsets = [1e-1f64, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0];
    let radii = [0.5f64, 2.0];
    for v in pos.iter().step_by(vstep) {
        for &e in &offsets { for &r in &radii {
            let p = [v[0]+e, v[1], v[2]];
            out.push((Mobius::sphere_reflection(Quat::from_point(p[0], p[1], p[2]), r), p));
        }}
    }
    for f in faces.iter().step_by(fstep) {
        let (a, b) = (pos[f[0]], pos[f[1]]);
        let m = [(a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5];
        for &e in &offsets {
            let p = [m[0], m[1]+e, m[2]];
            out.push((Mobius::sphere_reflection(Quat::from_point(p[0], p[1], p[2]), 1.0), p));
        }
    }
    out
}

/// Faces the CPU LOD pass must saturate: any of the 6 sampled bot values
/// (3 vertices + 3 edge midpoints, bot is linear so midpoint bots are
/// averages) within the pole-proximity threshold. Mirrors evaluate.rs.
fn pole_saturated_faces(
    pos: &[[f64; 3]],
    faces: &[[usize; 3]],
    m: &Mobius,
) -> Vec<usize> {
    let bot = |p: [f64; 3]| -> Quat { m.c * Quat::from_point(p[0], p[1], p[2]) + m.d };
    faces.iter().enumerate().filter_map(|(fi, f)| {
        let b: [Quat; 3] = std::array::from_fn(|k| bot(pos[f[k]]));
        let min = (0..3).map(|k| {
            b[k].norm_sq().min(((b[k] + b[(k+1)%3]) * 0.5).norm_sq())
        }).fold(f64::INFINITY, f64::min);
        // Strictly inside the threshold so borderline rungs don't flake.
        if min < 0.99e-8 { Some(fi) } else { None }
    }).collect()
}

fn check_model(name: &str, pos: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<String> {
    let f32s: Vec<[u32; 3]> = faces.iter().map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();
    let he = HalfEdgeMesh::from_triangles(pos.len() as u32, &f32s);
    let mut failures = Vec::new();

    for (m, pole) in adversarial_poles(pos, faces) {
        // (a) CPU packing must be finite.
        let inst = compute_instances_with_uvs(pos, faces, &m, None, Some(&he), None, None);
        for (fi, i) in inst.iter().enumerate() {
            let arr = i.to_f32_array();
            if let Some((k, v)) = arr.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                failures.push(format!(
                    "{name}: CPU pack non-finite at face {fi} float {k} = {v} (pole {pole:?})"));
                break;
            }
        }
        // (b) LODs stay powers of two and in range, and faces the pole
        // touches saturate to MAX_LOD — catastrophic cancellation must not
        // fake a small median on the most distorted face.
        for (fi, i) in inst.iter().enumerate() {
            for &l in &i.edge_lods {
                if !l.is_power_of_two() || l < 2 || l > MAX_LOD {
                    failures.push(format!("{name}: bad LOD {l} on face {fi} (pole {pole:?})"));
                }
            }
        }
        for fi in pole_saturated_faces(pos, faces, &m) {
            if inst[fi].edge_lods != [MAX_LOD; 3] {
                failures.push(format!(
                    "{name}: pole-adjacent face {fi} got LODs {:?}, want [{MAX_LOD}; 3] \
                     (pole {pole:?})", inst[fi].edge_lods));
            }
        }
        // (c) The GPU's f32 evaluation must stay finite and bounded.
        let w = worst_sample(pos, faces, &m, 8, pole);
        if w.failed() {
            failures.push(format!(
                "{name}: GPU f32 blow-up at face {} bary {:?}: |bot|^2_min={:.3e} |X|={:.3e} \
                 dot(n,n)={:.3e} fade={:.3} nonfinite pos/normal/smooth={}/{}/{} (pole {pole:?})",
                w.face, w.bary, w.bot_sq_min, w.position_mag, w.normal_dot, w.fade,
                w.nonfinite_position, w.nonfinite_normal, w.nonfinite_smooth));
        }
        if failures.len() > 20 { break; } // don't drown the output
    }
    failures
}

#[test]
fn demo_meshes_stay_finite_under_mobius_sweep() {
    let mut all = Vec::new();
    for path in ["../../horse.glb", "../../ant.glb"] {
        match load_normalized(path) {
            Some((pos, faces)) => {
                let name = path.rsplit('/').next().unwrap();
                all.extend(check_model(name, &pos, &faces));
            }
            // Tracked demo assets; if they're gone the test should say so,
            // not silently pass.
            None => panic!("could not load demo mesh {path}"),
        }
    }
    // Also cover an analytic shape with no dev-dependency at all.
    let (v, f) = quilting_core::shapes::icosahedron();
    all.extend(check_model("icosahedron", &v, &f));

    assert!(all.is_empty(), "Möbius sweep produced degenerate geometry:\n  {}",
        all.join("\n  "));
}

/// Regression guard for the specific overflow chain in `vertex/main.wgsl`:
/// |bot|^2 ~ 1e-10 makes dot(n, n) overflow f32 to +inf, `length()` returns
/// inf, `n / inf` is the zero vector, and `normalize(mv * vec4(0,0,0,0))`
/// yields NaN in `normal_vs`. This is ten orders of magnitude above the qinv
/// guard at 1e-20, so the guard never fires.
#[test]
fn analytic_normal_does_not_overflow_before_the_qinv_guard() {
    let (v, f) = quilting_core::shapes::icosahedron();
    let pole = [v[0][0] + 1e-5, v[0][1], v[0][2]];
    let m = Mobius::sphere_reflection(Quat::from_point(pole[0], pole[1], pole[2]), 1.0);
    let w = worst_sample(&v, &f, &m, 8, pole);
    assert!(w.normal_dot.is_finite(),
        "dot(n,n) overflowed to {} at |bot|^2_min={:.3e} (qinv guard is 1e-20 and never fired)",
        w.normal_dot, w.bot_sq_min);
    assert!(!w.nonfinite_smooth,
        "smooth-normal Möbius differential went non-finite at |bot|^2_min={:.3e}",
        w.bot_sq_min);
}

/// Regression guard for the GPU LOD pass: a pole exactly on a vertex (or
/// within f32 cancellation range of one) must saturate the face LOD, not
/// collapse the deformed median and hand the most distorted face on screen
/// the LEAST tessellation.
#[test]
fn gpu_lod_pass_saturates_at_the_pole() {
    let (v, f) = quilting_core::shapes::icosahedron();
    for offset in [1e-7f64, 0.0] {
        let m = Mobius::sphere_reflection(
            Quat::from_point(v[0][0] + offset, v[0][1], v[0][2]), 1.0);
        let lods = glsl_lod_pass(&v, &f, &m, 20.0, 1.0, 512.0);
        let max = lods.iter().cloned().fold(0.0f32, f32::max);
        assert_eq!(max, 512.0,
            "pole at offset {offset:e} from a vertex: GPU LOD pass gave max LOD {max}, \
             want 512 (per-face LODs {lods:?})");
    }
}

/// Regression guard for the CPU preprocessing boundary. The live shader no
/// longer uses this as a normal-transform shortcut: rotations and signed
/// scales have `c = 0` but still act on directions.
#[test]
fn affine_predicate_uses_the_documented_cpu_boundary() {
    for k in [1e-1f64, 1e-2, 1e-3, 1e-5, 1e-9] {
        let m = Mobius::new(Quat::ONE, Quat::ZERO,
                            Quat::new(k, 0., 0., 0.), Quat::new(k, 0., 0., 0.));
        let cpu = m.is_affine();
        let expected = m.c.norm_sq() < quilting_core::quaternion::AFFINE_C_NORM_SQ;
        assert_eq!(cpu, expected,
            "|c|^2={:.3e}: core says affine={cpu}, documented boundary says {expected}",
            m.c.norm_sq());
    }
}
