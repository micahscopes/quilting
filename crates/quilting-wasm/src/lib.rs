pub mod main_renderer;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use quilting_core::atlas::{TessellationAtlas, BuildMode};
use quilting_renderer::compute::LodCompute;
use quilting_core::evaluate::{compute_instances, compute_instances_no_lod, ScreenInfo};
use quilting_core::permutation::{canonical_form, perm_sign};
use quilting_core::quaternion::{Quat, Mobius};
use quilting_core::sampling::PatchConfig;
use quilting_core::shapes;
use quilting_core::triangle;
use quilting_mesh::HalfEdgeMesh;
use std::cell::RefCell;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    let mut config = wasm_tracing::WasmLayerConfig::new();
    config.set_max_level(tracing::Level::INFO);
    wasm_tracing::set_as_global_default_with_config(config);
}

/// Stored glTF data for animation switching without re-parsing.
struct StoredGltfData {
    animations: Vec<quilting_gltf::animation::Animation>,
    skins: Vec<quilting_gltf::animation::Skin>,
    nodes: Vec<quilting_gltf::scene::Node>,
    combined: quilting_gltf::mesh::Primitive,
    face_material_indices: Vec<Option<usize>>,
    primary_skin_idx: Option<usize>,
    active_animation: usize,
    /// Per-frame evaluator for GPU skinning (replaces prebake).
    evaluator: Option<quilting_gltf::evaluator::AnimationEvaluator>,
    /// Normalization: center and 1/extent for rest-pose bounding box.
    norm_center: [f64; 3],
    norm_scale: f64,
}

thread_local! {
    static ATLAS: RefCell<Option<TessellationAtlas>> = RefCell::new(None);
    static GPU_COMPUTE: RefCell<Option<(glow::Context, LodCompute)>> = RefCell::new(None);
    /// Cached half-edge mesh — built once per shape, reused across frames.
    static MESH_CACHE: RefCell<Option<CachedMesh>> = RefCell::new(None);
    /// Track which (canonical_lod, perm_parity) tessellation keys have been sent to JS.
    /// JS caches the GPU buffers, so we skip re-sending the bary/triangle data.
    static SENT_TESS: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());
}

struct CachedMesh {
    half_edge: HalfEdgeMesh,
    verts: Vec<[f64; 3]>,
    tris: Vec<[usize; 3]>,
}

/// Build the tessellation atlas client-side. Call once at init.
/// max_lod_exp: build LODs from 2^0 to 2^max_lod_exp (e.g., 8 → up to 256)
/// mode: "direct" or "hierarchical"
/// Set the sliver filter threshold. 0.0 = no filtering, 0.01 = default.
/// Must call build_atlas after changing to take effect.
#[wasm_bindgen]
pub fn set_sliver_threshold(threshold: f64) {
    quilting_core::atlas::set_sliver_threshold(threshold);
}

#[wasm_bindgen]
pub fn build_atlas(max_lod_exp: u32, mode: &str) -> f64 {
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let lods: Vec<u32> = (0..=max_lod_exp).map(|n| 1u32 << n).collect();
    let build_mode = match mode {
        "hierarchical" => BuildMode::Hierarchical,
        _ => BuildMode::Direct,
    };

    let start = js_sys::Date::now();
    let atlas = TessellationAtlas::build_with_mode(&lods, &config, build_mode);
    let elapsed = js_sys::Date::now() - start;

    ATLAS.with(|a| *a.borrow_mut() = Some(atlas));
    // New atlas means new tessellation data — clear the sent cache
    SENT_TESS.with(|s| s.borrow_mut().clear());

    elapsed
}

/// Initialize GPU compute context using OffscreenCanvas.
/// Call once per worker — creates a WebGL2 context for transform feedback LOD computation.
#[wasm_bindgen]
pub fn init_gpu_compute(max_faces: u32) -> bool {
    let canvas = web_sys::OffscreenCanvas::new(1, 1);
    let canvas = match canvas {
        Ok(c) => c,
        Err(_) => return false,
    };

    let gl_ctx = canvas.get_context("webgl2");
    let gl_ctx = match gl_ctx {
        Ok(Some(ctx)) => ctx.dyn_into::<web_sys::WebGl2RenderingContext>().ok(),
        _ => None,
    };
    let gl_ctx = match gl_ctx {
        Some(c) => c,
        None => return false,
    };

    let gl = glow::Context::from_webgl2_context(gl_ctx);

    match LodCompute::new(&gl, max_faces as usize) {
        Ok(compute) => {
            GPU_COMPUTE.with(|gc| *gc.borrow_mut() = Some((gl, compute)));
            true
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("GPU compute init failed: {e}").into());
            false
        }
    }
}

/// Run GPU LOD computation via transform feedback.
/// Positions must have been uploaded via upload_positions_to_compute() first.
/// mobius: 16 floats (a,b,c,d quaternions)
/// Returns: flat f32 array, 2 floats per face (atlas_index, perm_index)
#[wasm_bindgen]
pub fn gpu_compute_lods(
    num_faces: u32,
    num_vertices: u32,
    mobius: &[f32],
    density: f32,
    mesh_radius: f32,
) -> Vec<f32> {
    GPU_COMPUTE.with(|gc| {
        let mut gc = gc.borrow_mut();
        let (gl, compute) = match gc.as_mut() {
            Some(pair) => pair,
            None => return vec![],
        };
        let mut mob = [0.0f32; 16];
        for (i, &v) in mobius.iter().take(16).enumerate() {
            mob[i] = v;
        }
        let identity_vp = [0.0f32; 16];
        let max_lod = LOD_MAX.with(|m| *m.borrow());
        let n = compute.compute_lods(gl, num_faces as usize, num_vertices, 0, 0,
            mob, density, mesh_radius, 0.0, max_lod, &identity_vp, 0.0, 0.0);
        compute.read_back(gl, n)
    })
}

thread_local! {
    /// Cached half-edge mesh for edge coherence reconciliation after GPU LOD readback.
    static LOD_HALF_EDGE: RefCell<Option<HalfEdgeMesh>> = RefCell::new(None);
    /// Sorted atlas keys — maps atlas_index back to canonical LOD triples.
    static LOD_ATLAS_KEYS: RefCell<Vec<[u32; 3]>> = RefCell::new(Vec::new());
    /// Mesh bounding sphere radius (from rest-pose positions, for GPU density scaling).
    static LOD_MESH_RADIUS: RefCell<f64> = RefCell::new(1.0);
    /// Maximum LOD the atlas supports — clamp here instead of falling to LOD 2.
    static LOD_MAX: RefCell<f32> = RefCell::new(512.0);
}

/// Upload static animation data to the GPU compute context for per-frame LOD.
/// Call once after loading a model. Uploads:
/// - Rest-pose positions as a texture
/// - Face indices for the LOD compute shader
/// - Skinning texture (joint indices + weights) if skeletal animation
/// - Morph delta texture if morph target animation
/// - Atlas LUT for LOD → atlas index mapping
/// Returns true if upload succeeded (GPU compute context available).
#[wasm_bindgen]
pub fn upload_model_to_compute() -> bool {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            None => return false,
        };

        GPU_COMPUTE.with(|gc| {
            let mut gc = gc.borrow_mut();
            let (gl, compute) = match gc.as_mut() {
                Some(pair) => pair,
                None => return false,
            };

            let combined = &data.combined;
            let nv = combined.positions.len();

            // Upload rest-pose positions (normalized)
            let center = data.norm_center;
            let s = data.norm_scale;
            let mut pos_f32 = Vec::with_capacity(nv * 3);
            for pos in &combined.positions {
                pos_f32.push(((pos[0] - center[0]) * s) as f32);
                pos_f32.push(((pos[1] - center[1]) * s) as f32);
                pos_f32.push(((pos[2] - center[2]) * s) as f32);
            }
            compute.upload_positions_texture(gl, &pos_f32, nv);

            // Upload face indices
            let face_indices_f32: Vec<f32> = combined.triangles.iter()
                .flat_map(|f| [f[0] as f32, f[1] as f32, f[2] as f32])
                .collect();
            compute.upload_face_indices(gl, &face_indices_f32);

            // Upload skinning data — real or identity fallback for static models
            if let (Some(ji), Some(jw)) = (&combined.joint_indices, &combined.joint_weights) {
                compute.upload_skinning_texture(gl, ji, jw);
            } else {
                // Static model: all vertices → joint 0, weight 1.0
                let ji_default: Vec<[u16; 4]> = vec![[0, 0, 0, 0]; nv];
                let jw_default: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; nv];
                compute.upload_skinning_texture(gl, &ji_default, &jw_default);
            }

            // Upload morph deltas — always upload even if empty, to clear stale data
            if !combined.morph_targets.is_empty() {
                let num_targets = combined.morph_targets.len();
                let ns = data.norm_scale as f32;
                let mut deltas = Vec::with_capacity(num_targets * nv * 3);
                for target in &combined.morph_targets {
                    for vi in 0..nv {
                        if vi < target.len() {
                            deltas.push(target[vi][0] as f32 * ns);
                            deltas.push(target[vi][1] as f32 * ns);
                            deltas.push(target[vi][2] as f32 * ns);
                        } else {
                            deltas.extend_from_slice(&[0.0, 0.0, 0.0]);
                        }
                    }
                }
                compute.upload_morph_deltas(gl, &deltas, nv, num_targets);
            } else {
                // No morph targets: upload a 1x1 zero texture to clear stale data
                compute.upload_morph_deltas(gl, &[0.0; 3], 1, 1);
            }

            // Upload atlas LUT and cache keys for readback mapping
            ATLAS.with(|atlas_cell| {
                let atlas = atlas_cell.borrow();
                if let Some(atlas) = atlas.as_ref() {
                    const LUT_SIZE: usize = 1200;
                    let mut lut = vec![255u8; LUT_SIZE];
                    let mut keys: Vec<[u32; 3]> = atlas.patches.keys().copied().collect();
                    keys.sort();
                    for (idx, key) in keys.iter().enumerate() {
                        if idx >= 255 { break; }
                        let ea = (key[0] as f64).log2().round() as usize;
                        let eb = (key[1] as f64).log2().round() as usize;
                        let ec = (key[2] as f64).log2().round() as usize;
                        let lut_key = ea + eb * 10 + ec * 100;
                        if lut_key < LUT_SIZE { lut[lut_key] = idx as u8; }
                    }
                    compute.upload_atlas_lut(gl, &lut);
                    let max_lod = keys.last().map(|k| *k.iter().max().unwrap()).unwrap_or(512) as f32;
                    LOD_MAX.with(|m| *m.borrow_mut() = max_lod);
                    LOD_ATLAS_KEYS.with(|ak| *ak.borrow_mut() = keys);
                }
            });

            // Build half-edge mesh for edge coherence (once)
            let faces_u32: Vec<[u32; 3]> = combined.triangles.iter()
                .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
                .collect();
            LOD_HALF_EDGE.with(|he| {
                *he.borrow_mut() = Some(HalfEdgeMesh::from_triangles(nv as u32, &faces_u32));
            });

            // Compute mesh_radius from normalized positions
            let (mut cx, mut cy, mut cz) = (0.0f64, 0.0, 0.0);
            let n = nv as f64;
            for i in 0..nv {
                cx += pos_f32[i*3] as f64; cy += pos_f32[i*3+1] as f64; cz += pos_f32[i*3+2] as f64;
            }
            cx /= n; cy /= n; cz /= n;
            let mut max_r = 0.0f64;
            for i in 0..nv {
                let dx = pos_f32[i*3] as f64 - cx;
                let dy = pos_f32[i*3+1] as f64 - cy;
                let dz = pos_f32[i*3+2] as f64 - cz;
                max_r = max_r.max((dx*dx + dy*dy + dz*dz).sqrt());
            }
            LOD_MESH_RADIUS.with(|r| *r.borrow_mut() = max_r.max(1e-6));

            true
        })
    })
}

/// Per-frame GPU LOD computation with edge coherence reconciliation.
///
/// 1. Evaluates animation (morph + skeletal) at time t
/// 2. Uploads joint matrices + morph weights to worker GPU
/// 3. Runs transform feedback LOD compute (Möbius + medians + snap)
/// 4. Reads back raw (atlas_index, perm_index) per face
/// 5. Inverts permutation to recover unsorted per-edge LODs
/// 6. Enforces edge coherence via half-edge mesh (max across shared edges)
/// 7. Re-canonicalizes to (canonical_lod, perm_index, parity) per face
///
/// Returns Float32Array with 5 floats per face: [canon_a, canon_b, canon_c, perm_index, parity]
/// Sample Möbius stretch at the midpoint of each face (from instance data).
/// Returns [min_stretch, max_stretch] as sigmoid-mapped values matching the vertex shader.
/// Runs on CPU, async-safe — call from a worker without blocking rendering.
#[wasm_bindgen]
pub fn sample_stretch_range(mobius: &[f32], instances: &[f32], num_faces: u32) -> Vec<f32> {
    if mobius.len() < 16 { return vec![0.5, 0.5]; }
    let c = [mobius[8], mobius[9], mobius[10], mobius[11]];
    let d = [mobius[12], mobius[13], mobius[14], mobius[15]];
    let c_len2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2] + c[3]*c[3];
    if c_len2 < 0.001 { return vec![0.5, 0.5]; } // identity Möbius, no stretch

    let stride = 40usize; // INSTANCE_STRIDE floats per face
    let nf = num_faces as usize;
    let mut min_s = f32::INFINITY;
    let mut max_s = f32::NEG_INFINITY;

    for fi in 0..nf {
        let base = fi * stride;
        if base + 12 > instances.len() { break; }
        // Instance data layout: [p0.x, p0.y, p0.z, w0, p1.x, p1.y, p1.z, w1, p2.x, p2.y, p2.z, w2, ...]
        // Actually: positions are at offsets 0-2 (p0), 4-6 (p1), 8-10 (p2) within the instance
        let p0 = [instances[base+0], instances[base+1], instances[base+2]];
        let p1 = [instances[base+4], instances[base+5], instances[base+6]];
        let p2 = [instances[base+8], instances[base+9], instances[base+10]];
        // Centroid
        let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
        let cy = (p0[1] + p1[1] + p2[1]) / 3.0;
        let cz = (p0[2] + p1[2] + p2[2]) / 3.0;
        // bot = c*p + d (quaternion multiply, p as pure imaginary (0, cx, cy, cz))
        // For pure imaginary p: c*p = (c0*0 - c1*cx - c2*cy - c3*cz, c0*cx + c2*cz - c3*cy, c0*cy + c3*cx - c1*cz, c0*cz + c1*cy - c2*cx)
        let bw = -c[1]*cx - c[2]*cy - c[3]*cz + d[0];
        let bx = c[0]*cx + c[2]*cz - c[3]*cy + d[1];
        let by = c[0]*cy + c[3]*cx - c[1]*cz + d[2];
        let bz = c[0]*cz + c[1]*cy - c[2]*cx + d[3];
        let bot_len2 = bw*bw + bx*bx + by*by + bz*bz;
        let stretch = 1.0 / bot_len2.max(0.001);
        // Sigmoid mapping matching vertex shader
        let log_s = stretch.log2();
        let sig = 1.0 / (1.0 + (-log_s * 0.25_f32).exp());
        min_s = min_s.min(sig);
        max_s = max_s.max(sig);
    }

    if min_s.is_infinite() { return vec![0.5, 0.5]; }
    vec![min_s, max_s]
}

/// Uses mesh_radius computed at upload time (not hardcoded).
#[wasm_bindgen]
pub fn compute_animated_lods(
    t: f64,
    mobius: &[f32],
    density: f32,
    min_px: f32,
    vp_matrix: &[f32],
    vp_width: f32,
    vp_height: f32,
) -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = match gd.try_borrow() {
            Ok(d) => d,
            Err(_) => return JsValue::NULL, // concurrent borrow during model load
        };
        let data = match data.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };

        let num_faces = data.combined.triangles.len();
        let num_vertices = data.combined.positions.len() as u32;
        let mesh_radius = LOD_MESH_RADIUS.with(|r| *r.borrow()) as f32;

        // 1. Evaluate animation pose at time t
        let (joint_matrices, morph_weights, num_joints, num_morph) =
            if let Some(ref eval) = data.evaluator {
                let pose = eval.evaluate(t);

                let c = data.norm_center;
                let s = data.norm_scale;
                let si = if s.abs() > 1e-10 { 1.0 / s } else { 1.0 };
                let sf = s as f32; let sif = si as f32;
                let cx = c[0] as f32; let cy = c[1] as f32; let cz = c[2] as f32;
                let norm: [f32; 16] = [
                    sf, 0.0, 0.0, 0.0, 0.0, sf, 0.0, 0.0,
                    0.0, 0.0, sf, 0.0, -cx*sf, -cy*sf, -cz*sf, 1.0,
                ];
                let unnorm: [f32; 16] = [
                    sif, 0.0, 0.0, 0.0, 0.0, sif, 0.0, 0.0,
                    0.0, 0.0, sif, 0.0, cx, cy, cz, 1.0,
                ];
                fn mat4_mul_f32(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
                    let mut out = [0.0f32; 16];
                    for col in 0..4 {
                        for row in 0..4 {
                            out[col * 4 + row] = a[row] * b[col * 4]
                                + a[4 + row] * b[col * 4 + 1]
                                + a[8 + row] * b[col * 4 + 2]
                                + a[12 + row] * b[col * 4 + 3];
                        }
                    }
                    out
                }

                let mut normalized_mats = Vec::with_capacity(pose.joint_matrices.len());
                let nj = pose.joint_matrices.len() / 16;
                for ji in 0..nj {
                    let m: [f32; 16] = pose.joint_matrices[ji*16..(ji+1)*16].try_into().unwrap();
                    let nmu = mat4_mul_f32(&norm, &mat4_mul_f32(&m, &unnorm));
                    normalized_mats.extend_from_slice(&nmu);
                }

                let nm = pose.morph_weights.len() as u32;
                (normalized_mats, pose.morph_weights, nj as u32, nm)
            } else {
                (vec![], vec![], 0u32, 0u32)
            };

        // 2-3. GPU compute
        let gpu_class = GPU_COMPUTE.with(|gc| {
            let mut gc = gc.borrow_mut();
            let (gl, compute) = match gc.as_mut() {
                Some(pair) => pair,
                None => return vec![],
            };

            if !joint_matrices.is_empty() {
                compute.upload_joint_matrices(gl, &joint_matrices);
            }
            if !morph_weights.is_empty() {
                compute.upload_morph_weights(gl, &morph_weights);
            }

            let mut mob = [0.0f32; 16];
            for (i, &v) in mobius.iter().take(16).enumerate() { mob[i] = v; }
            let mut vp = [0.0f32; 16];
            for (i, &v) in vp_matrix.iter().take(16).enumerate() { vp[i] = v; }

            let max_lod = LOD_MAX.with(|m| *m.borrow());
            let n = compute.compute_lods(
                gl, num_faces, num_vertices,
                num_joints, num_morph,
                mob, density, mesh_radius, min_px, max_lod,
                &vp, vp_width, vp_height,
            );
            compute.read_back(gl, n)
        });

        if gpu_class.is_empty() { return JsValue::NULL; }

        // 4-5. Invert permutation to recover unsorted per-face LODs
        let nf = num_faces;
        let stride = quilting_renderer::compute::FLOATS_PER_FACE_OUTPUT;
        let mut face_lods = vec![[2u32; 3]; nf];

        LOD_ATLAS_KEYS.with(|ak| {
            let keys = ak.borrow();
            for fi in 0..nf {
                let rb = fi * stride;
                if rb + 1 < gpu_class.len() {
                    let atlas_idx = gpu_class[rb] as usize;
                    let perm_idx = gpu_class[rb + 1] as usize;
                    if atlas_idx < keys.len() && perm_idx < 6 {
                        let canonical = keys[atlas_idx];
                        let perm = quilting_core::permutation::S3_PERMUTATIONS[perm_idx];
                        face_lods[fi] = [canonical[perm[0]], canonical[perm[1]], canonical[perm[2]]];
                    }
                }
            }
        });

        // 6. Edge coherence: shared edges must have matching LODs
        // Guard: skip if half-edge mesh doesn't match current model (stale from previous load)
        LOD_HALF_EDGE.with(|he_cell| {
            let he_opt = he_cell.borrow();
            if let Some(he) = he_opt.as_ref() {
                if he.num_faces as usize != nf { return; } // stale mesh
                for fi in 0..nf {
                    let hes = he.face_half_edges(fi as u32);
                    for (hi, &he_id) in hes.iter().enumerate() {
                        let lod_idx = (hi + 2) % 3;
                        if let Some(twin_id) = quilting_mesh::unpack_twin(he.half_edges[he_id as usize].twin) {
                            let adj_fi = he.half_edges[twin_id as usize].face as usize;
                            if adj_fi < nf {
                                let adj_hes = he.face_half_edges(adj_fi as u32);
                                for (adj_hi, &adj_he_id) in adj_hes.iter().enumerate() {
                                    if adj_he_id == twin_id {
                                        let adj_lod_idx = (adj_hi + 2) % 3;
                                        let max_lod = face_lods[fi][lod_idx].max(face_lods[adj_fi][adj_lod_idx]);
                                        face_lods[fi][lod_idx] = max_lod;
                                        face_lods[adj_fi][adj_lod_idx] = max_lod;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        // 7. Re-canonicalize to (canonical_lod, perm_index, parity)
        let mut result = Vec::with_capacity(nf * 5);
        for fi in 0..nf {
            let ck = canonical_form(face_lods[fi]);
            let parity = perm_sign(ck.perm_index);
            result.push(ck.res[0] as f32);
            result.push(ck.res[1] as f32);
            result.push(ck.res[2] as f32);
            result.push(ck.perm_index as f32);
            result.push(parity as f32);
        }

        let arr = js_sys::Float32Array::new_with_length(result.len() as u32);
        arr.copy_from(&result);
        arr.into()
    })
}

/// Set tessellation parameters.
/// density: target mesh_radius/density = triangle size in deformed world units (default 100)
/// screen_atten: whether to attenuate LOD for distant/small screen faces
#[wasm_bindgen]
pub fn set_tess_params(density: f64, screen_atten: bool) {
    quilting_core::evaluate::set_tess_params(density, screen_atten);
}

/// Set minimum pixels per subdivision for screen attenuation.
#[wasm_bindgen]
pub fn set_min_px_per_sub(px: f64) {
    quilting_core::evaluate::set_min_px_per_sub(px);
}

/// Export all atlas patches as pre-computed bary data for GPU upload.
/// Returns a JS array of { key: [a,b,c], perm: n, bary: Float64Array, tris: Uint32Array }.
/// Call once after atlas build — JS creates GPU buffers from this, then
/// the rendering pipeline never needs to send tessellation data again.
#[wasm_bindgen]
pub fn export_all_patches() -> JsValue {
    ATLAS.with(|atlas_cell| {
        let atlas = atlas_cell.borrow();
        let atlas = match atlas.as_ref() {
            Some(a) => a,
            None => return JsValue::NULL,
        };

        let result = js_sys::Array::new();
        for &key in atlas.patches.keys() {
            // get_patch with a sorted key returns canonical (unpermuted) mesh
            let mesh = match atlas.get_patch(key) {
                Some(m) => m,
                None => continue,
            };

            // Convert cartesian to bary once (canonical, perm 0)
            let base_bary: Vec<[f64; 3]> = mesh.positions.iter().map(|p| {
                let mut b = triangle::cartesian_to_bary(p[0], p[1]);
                for c in &mut b { if c.abs() < 1e-10 { *c = 0.0; } }
                let sum = b[0] + b[1] + b[2];
                if sum > 0.0 { b[0] /= sum; b[1] /= sum; b[2] /= sum; }
                b
            }).collect();

            // Emit all 6 permutations
            for perm in 0u32..6 {
                let bary: Vec<f64> = base_bary.iter().map(|b| {
                    match perm {
                        1 => [b[0], b[2], b[1]],
                        2 => [b[1], b[0], b[2]],
                        3 => [b[1], b[2], b[0]],
                        4 => [b[2], b[0], b[1]],
                        5 => [b[2], b[1], b[0]],
                        _ => *b,
                    }
                }).flat_map(|b| [b[0], b[1], b[2]]).collect();

                // Odd permutations (1,2,5) reverse 2D winding of barycentric coords.
                // Swap two indices per triangle to restore consistent CCW screen winding.
                let is_odd_perm = matches!(perm, 1 | 2 | 5);
                let tris: Vec<u32> = mesh.triangles.iter()
                    .flat_map(|t| {
                        if is_odd_perm {
                            [t[0] as u32, t[2] as u32, t[1] as u32]
                        } else {
                            [t[0] as u32, t[1] as u32, t[2] as u32]
                        }
                    }).collect();

                let obj = js_sys::Object::new();
                let s = |k: &str, v: JsValue| { js_sys::Reflect::set(&obj, &k.into(), &v).ok(); };
                s("key", serde_wasm_bindgen::to_value(&key).unwrap());
                s("perm", JsValue::from(perm));
                s("bary", js_sys::Float64Array::from(&bary[..]).into());
                s("tris", js_sys::Uint32Array::from(&tris[..]).into());
                s("n_verts", JsValue::from(base_bary.len() as u32));
                s("n_tris", JsValue::from(mesh.triangles.len() as u32));
                result.push(&obj);
            }
        }
        result.into()
    })
}

/// Build a subset of the atlas (for parallel construction).
#[wasm_bindgen]
pub fn build_atlas_subset(max_lod_exp: u32, mode: &str, worker_index: u32, num_workers: u32) -> f64 {
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let lods: Vec<u32> = (0..=max_lod_exp).map(|n| 1u32 << n).collect();
    let build_mode = match mode {
        "hierarchical" => BuildMode::Hierarchical,
        _ => BuildMode::Direct,
    };
    let start = js_sys::Date::now();
    let atlas = TessellationAtlas::build_subset(&lods, &config, build_mode, worker_index as usize, num_workers as usize);
    let elapsed = js_sys::Date::now() - start;
    ATLAS.with(|a| *a.borrow_mut() = Some(atlas));
    SENT_TESS.with(|s| s.borrow_mut().clear());
    elapsed
}

/// Merge another atlas (from bytes) into the current one.
#[wasm_bindgen]
pub fn merge_atlas_bytes(bytes: &[u8]) -> bool {
    match TessellationAtlas::from_bytes(bytes) {
        Ok(other) => {
            ATLAS.with(|a| {
                if let Some(atlas) = a.borrow_mut().as_mut() {
                    atlas.merge_from(&other);
                }
            });
            SENT_TESS.with(|s| s.borrow_mut().clear());
            true
        }
        Err(_) => false,
    }
}

/// Export the atlas as bytes for sharing with other workers.
#[wasm_bindgen]
pub fn export_atlas_bytes() -> Vec<u8> {
    ATLAS.with(|a| {
        a.borrow().as_ref().map(|atlas| atlas.to_bytes()).unwrap_or_default()
    })
}

/// Import an atlas from bytes (built by another worker).
#[wasm_bindgen]
pub fn import_atlas_bytes(bytes: &[u8]) -> bool {
    match TessellationAtlas::from_bytes(bytes) {
        Ok(atlas) => {
            ATLAS.with(|a| *a.borrow_mut() = Some(atlas));
            SENT_TESS.with(|s| s.borrow_mut().clear());
            true
        }
        Err(_) => false,
    }
}

/// Extend the atlas to include a new LOD level by subdividing from existing parents.
/// Much faster than rebuilding — each new patch is one subdivision step.
#[wasm_bindgen]
pub fn extend_atlas(new_lod: u32) -> f64 {
    let start = js_sys::Date::now();
    ATLAS.with(|atlas_cell| {
        let mut atlas_opt = atlas_cell.borrow_mut();
        if let Some(atlas) = atlas_opt.as_mut() {
            // Collect all LOD levels currently in the atlas + the new one
            let mut levels: Vec<u32> = atlas.lod_levels.clone();
            if !levels.contains(&new_lod) {
                levels.push(new_lod);
                levels.sort();
            }

            // Generate all canonical triples that include new_lod
            let mut new_triples = Vec::new();
            for &a in &levels {
                for &b in &levels {
                    for &c in &levels {
                        if a > b || b > c { continue; }
                        if a != new_lod && b != new_lod && c != new_lod { continue; }
                        let key = [a, b, c];
                        if !atlas.patches.contains_key(&key) {
                            new_triples.push(key);
                        }
                    }
                }
            }

            for key in &new_triples {
                ensure_patch(atlas, *key);
            }

            atlas.lod_levels = levels;
        }
    });
    SENT_TESS.with(|s| s.borrow_mut().clear());
    js_sys::Date::now() - start
}

/// Generate a single tessellation patch via hierarchical subdivision and store
/// it in the atlas. Recursively generates ancestor patches if needed (e.g.,
/// [4,256,512] needs parent [2,128,256] which needs grandparent [1,64,128]).
/// Falls back to direct Poisson generation for base triples that can't be halved.
#[wasm_bindgen]
pub fn generate_and_store_patch(res_a: u32, res_b: u32, res_c: u32) {
    ATLAS.with(|atlas_cell| {
        let mut atlas_opt = atlas_cell.borrow_mut();
        // Initialize empty atlas if none exists
        let atlas = atlas_opt.get_or_insert_with(|| TessellationAtlas {
            positions: Vec::new(),
            triangles: Vec::new(),
            patches: HashMap::default(),
            lod_levels: Vec::new(),
        });
        let mut key = [res_a, res_b, res_c];
        key.sort();
        ensure_patch(atlas, key);
    });
}

fn insert_patch(atlas: &mut TessellationAtlas, key: [u32; 3], positions: Vec<[f64; 2]>, triangles: Vec<[usize; 3]>) {
    let base_vertex = atlas.positions.len();
    let base_triangle = atlas.triangles.len();
    atlas.positions.extend_from_slice(&positions);
    for t in &triangles {
        atlas.triangles.push([t[0] + base_vertex, t[1] + base_vertex, t[2] + base_vertex]);
    }
    atlas.patches.insert(key, quilting_core::atlas::PatchEntry {
        base_vertex,
        vertex_count: positions.len(),
        base_triangle,
        triangle_count: triangles.len(),
    });
}

fn get_patch_local(atlas: &TessellationAtlas, key: &[u32; 3]) -> Option<(Vec<[f64; 2]>, Vec<[usize; 3]>)> {
    let entry = atlas.patches.get(key)?;
    let positions = atlas.positions[entry.base_vertex..entry.base_vertex + entry.vertex_count].to_vec();
    let triangles: Vec<[usize; 3]> = atlas.triangles
        [entry.base_triangle..entry.base_triangle + entry.triangle_count]
        .iter()
        .map(|t| [t[0] - entry.base_vertex, t[1] - entry.base_vertex, t[2] - entry.base_vertex])
        .collect();
    Some((positions, triangles))
}

/// Generate a patch: try hierarchical subdivision, fall back to Poisson.
fn ensure_patch(atlas: &mut TessellationAtlas, key: [u32; 3]) -> bool {
    if atlas.patches.contains_key(&key) { return true; }

    // Try subdivision if all components are even
    if key[0] % 2 == 0 && key[1] % 2 == 0 && key[2] % 2 == 0 && key[0] > 0 {
        let parent = [key[0] / 2, key[1] / 2, key[2] / 2];
        if ensure_patch(atlas, parent) {
            if let Some((pos, tris)) = get_patch_local(atlas, &parent) {
                let (new_pos, new_tris) = quilting_core::subdivide::subdivide(&pos, &tris);
                insert_patch(atlas, key, new_pos, new_tris);
                return true;
            }
        }
    }

    // Poisson fallback
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let res = [key[0] as f64, key[1] as f64, key[2] as f64];
    let sample = quilting_core::sampling::tri_patch(res, &config);
    if sample.positions.len() < 3 { return false; }
    let tri = quilting_core::delaunay::triangulate_2d_constrained(&sample.positions, &sample.bary);
    insert_patch(atlas, key, tri.positions, tri.triangles);
    true
}

/// Get a built-in shape.
#[wasm_bindgen]
pub fn get_shape(name: &str) -> JsValue {
    let (verts, faces) = match name {
        "tetrahedron" => shapes::tetrahedron(),
        "octahedron" => shapes::octahedron(),
        "icosahedron" => shapes::icosahedron(),
        _ => shapes::cube(),
    };
    let positions: Vec<f64> = verts.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
    let indices: Vec<u32> = faces.iter().flat_map(|f| [f[0] as u32, f[1] as u32, f[2] as u32]).collect();
    serde_wasm_bindgen::to_value(&ShapeData {
        positions, faces: indices,
        num_verts: verts.len(), num_faces: faces.len(),
    }).unwrap()
}

#[derive(serde::Serialize)]
struct ShapeData {
    positions: Vec<f64>,
    faces: Vec<u32>,
    num_verts: usize,
    num_faces: usize,
}

/// Compute batched mesh data using the precomputed atlas.
#[wasm_bindgen]
pub fn compute_mesh_batches(
    positions: &[f64],
    faces: &[u32],
    transform_type: &str,
    params: &[f64],
    override_res: u32,
    vp_matrix: &[f64],     // 16 doubles: column-major view-projection matrix
    viewport_width: f64,
    viewport_height: f64,
) -> JsValue {
    let verts: Vec<[f64; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]]).collect();
    let tris: Vec<[usize; 3]> = faces.chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();

    // Build or reuse cached half-edge mesh. Rebuild only when topology changes.
    MESH_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        let needs_rebuild = match cache.as_ref() {
            Some(c) => c.tris.len() != tris.len() || c.verts.len() != verts.len(),
            None => true,
        };
        if needs_rebuild {
            let faces_u32: Vec<[u32; 3]> = tris.iter()
                .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
                .collect();
            *cache = Some(CachedMesh {
                half_edge: HalfEdgeMesh::from_triangles(verts.len() as u32, &faces_u32),
                verts: verts.clone(),
                tris: tris.clone(),
            });
        }
    });

    let transform = match transform_type {
        "sphere_reflection" if params.len() >= 4 && params[3] > 0.001 => {
            Mobius::sphere_reflection(
                Quat::from_point(params[0], params[1], params[2]),
                params[3],
            )
        }
        "rotation" if params.len() >= 4 => {
            Mobius::rotation(params[0], params[1], params[2], params[3])
        }
        "translation" if params.len() >= 3 => {
            Mobius::translation(Quat::from_point(params[0], params[1], params[2]))
        }
        _ => Mobius::identity(),
    };

    let screen = if vp_matrix.len() >= 16 && viewport_width > 0.0 {
        let mut m = [0.0f64; 16];
        m.copy_from_slice(&vp_matrix[..16]);
        Some(ScreenInfo { vp_matrix: m, width: viewport_width, height: viewport_height })
    } else {
        None
    };

    let t0 = js_sys::Date::now();

    let instances_orig = compute_instances_no_lod(&verts, &tris);
    let t1 = js_sys::Date::now();

    let instances_xform = MESH_CACHE.with(|cache_cell| {
        let cache = cache_cell.borrow();
        let mesh_ref = cache.as_ref().map(|c| &c.half_edge);
        compute_instances(&verts, &tris, &transform, screen.as_ref(), mesh_ref)
    });
    let t2 = js_sys::Date::now();

    // Group by (canonical LOD, perm_index)
    let mut groups: FxHashMap<([u32; 3], usize), Vec<usize>> = FxHashMap::default();
    for (fi, inst) in instances_xform.iter().enumerate() {
        let lod = if override_res > 0 {
            [override_res, override_res, override_res]
        } else {
            inst.edge_lods
        };
        let key = canonical_form(lod);
        groups.entry((key.res, key.perm_index)).or_default().push(fi);
    }

    let mut batches = Vec::new();

    // Try atlas lookup, fall back to the mesh stored in atlas
    ATLAS.with(|atlas_cell| {
        let atlas_ref = atlas_cell.borrow();

        for (&(canonical_lod, perm_index), face_indices) in &groups {
            // Find the best available LOD that preserves edge stitching.
            // Safe fallback: uniform patch at min(edge LODs). This ensures
            // shared edges still match — the min-LOD edges are correct and
            // higher-LOD edges just get undersampled.
            let (mesh, used_lod) = {
                let mut found = None;
                if let Some(atlas) = atlas_ref.as_ref() {
                    // Try exact match
                    if let Some(m) = atlas.get_patch(canonical_lod) {
                        found = Some((m, canonical_lod));
                    } else {
                        // Fall back to uniform at min edge LOD, then halve
                        let min_lod = canonical_lod[0]; // sorted, so [0] is min
                        let mut try_res = min_lod;
                        while try_res >= 1 {
                            let uniform = [try_res, try_res, try_res];
                            if let Some(m) = atlas.get_patch(uniform) {
                                found = Some((m, uniform));
                                break;
                            }
                            if try_res <= 1 { break; }
                            try_res /= 2;
                        }
                    }
                }
                found.unwrap_or_else(|| {
                    web_sys::console::warn_1(&format!(
                        "ATLAS FALLBACK: wanted {:?}, no match found — using [1,1,1]",
                        canonical_lod
                    ).into());
                    let config = PatchConfig { k_candidates: 30, seed: 42 };
                    let sample = quilting_core::sampling::tri_patch([1.0, 1.0, 1.0], &config);
                    let tri = quilting_core::delaunay::triangulate_2d_clipped(&sample.positions);
                    (quilting_core::mesh::TessellationMesh::from_2d(tri.positions, tri.triangles), [1, 1, 1])
                })
            };

            let is_fallback = used_lod != canonical_lod;
            if is_fallback {
                web_sys::console::warn_1(&format!(
                    "LOD MISMATCH: wanted {:?}, got {:?} ({} faces)",
                    canonical_lod, used_lod, face_indices.len()
                ).into());
            }
            let parity = perm_sign(perm_index);

            let actual_lod = if override_res > 0 {
                [override_res, override_res, override_res]
            } else {
                instances_xform[face_indices[0]].edge_lods
            };

            // Tess cache key includes perm_index — each permutation gets its own
            // pre-remapped bary buffer for exact edge stitching.
            // Key by used_lod (not canonical_lod) so fallback data and correct
            // data get separate cache entries — no invalidation needed.
            let tess_key = format!("{},{},{}/{}", used_lod[0], used_lod[1], used_lod[2], perm_index);

            let already_sent = SENT_TESS.with(|s| s.borrow().contains(&tess_key));

            let (bary_data, tess_tris, n_verts, n_tris) = if already_sent {
                (vec![], vec![], mesh.positions.len(), mesh.triangles.len())
            } else {
                // Convert to bary first, then permute by swapping components.
                // This is exact (no arithmetic error) unlike the old approach
                // of remapping in 2D cartesian then converting to bary.
                let bary: Vec<f64> = mesh.positions.iter().map(|p| {
                    let mut b = triangle::cartesian_to_bary(p[0], p[1]);
                    // Snap near-zero bary to exact 0.0
                    for c in &mut b { if c.abs() < 1e-10 { *c = 0.0; } }
                    let sum = b[0] + b[1] + b[2];
                    if sum > 0.0 { b[0] /= sum; b[1] /= sum; b[2] /= sum; }
                    // Permute in bary space — just component swap, bit-identical
                    match perm_index {
                        1 => [b[0], b[2], b[1]],
                        2 => [b[1], b[0], b[2]],
                        3 => [b[1], b[2], b[0]],
                        4 => [b[2], b[0], b[1]],
                        5 => [b[2], b[1], b[0]],
                        _ => b,
                    }
                }).flat_map(|b| [b[0], b[1], b[2]]).collect();

                let tris: Vec<u32> = mesh.triangles.iter()
                    .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

                let nv = bary.len() / 3;
                let nt = tris.len() / 3;
                SENT_TESS.with(|s| s.borrow_mut().insert(tess_key));
                (bary, tris, nv, nt)
            };

            let nf = face_indices.len();
            let mut orig_data = vec![0.0f32; nf * 52];
            let mut xform_data = vec![0.0f32; nf * 52];

            for (i, &fi) in face_indices.iter().enumerate() {
                let o = instances_orig[fi].to_f32_array();
                let x = instances_xform[fi].to_f32_array();
                orig_data[i*52..(i+1)*52].copy_from_slice(&o);
                xform_data[i*52..(i+1)*52].copy_from_slice(&x);
            }

            batches.push(BatchData {
                lod: actual_lod,
                wanted_lod: [canonical_lod[0], canonical_lod[1], canonical_lod[2]],
                used_lod: [used_lod[0], used_lod[1], used_lod[2]],
                is_fallback,
                perm_parity: parity,
                perm_index,
                material_index: -1,
                instances_orig: orig_data,
                instances_xform: xform_data,
                tess_bary: bary_data,
                tess_triangles: tess_tris,
                face_indices: face_indices.iter().map(|&i| i as u32).collect(),
                num_faces: face_indices.len(),
                verts_per_face: n_verts,
                tris_per_face: n_tris,
            });
        }
    });

    let t3 = js_sys::Date::now();

    let result = serde_wasm_bindgen::to_value(&MeshBatches {
        batches,
        total_faces: tris.len(),
        num_batches: groups.len(),
        timings: [t1 - t0, t2 - t1, t3 - t2],
    }).unwrap();

    let t4 = js_sys::Date::now();
    // Log timings: [orig_instances_ms, xform_lod_ms, batching_ms, serde_ms]
    web_sys::console::log_1(&format!(
        "compute_mesh_batches: orig={:.1}ms lod={:.1}ms batch={:.1}ms serde={:.1}ms total={:.1}ms",
        t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0
    ).into());

    result
}

#[derive(serde::Serialize)]
struct BatchData {
    lod: [u32; 3],
    wanted_lod: [u32; 3],
    used_lod: [u32; 3],
    is_fallback: bool,
    perm_parity: i32,  // +1 for even permutations, -1 for odd (normal flip)
    perm_index: usize, // S3 permutation index (0-5) for vertex shader bary remapping
    material_index: i32, // material index (-1 = default)
    instances_orig: Vec<f32>,
    instances_xform: Vec<f32>,
    tess_bary: Vec<f64>,
    tess_triangles: Vec<u32>,
    face_indices: Vec<u32>, // original mesh face indices for pick identification
    num_faces: usize,
    verts_per_face: usize,
    tris_per_face: usize,
}

#[derive(serde::Serialize)]
struct MeshBatches {
    batches: Vec<BatchData>,
    total_faces: usize,
    num_batches: usize,
    timings: [f64; 3], // [orig_ms, lod_ms, batch_ms]
}

// --- Shader compilation via quilting-shaders ---

/// Compile the vertex shader for native OpenGL/WebGL rendering
/// (no Y-flip or Z-remap -- suitable for direct use with WebGL2).
#[wasm_bindgen]
pub fn get_vertex_glsl() -> String {
    match quilting_shaders::compile_vertex_glsl_native() {
        Ok(glsl) => glsl,
        Err(e) => format!("// ERROR: {}", e),
    }
}

/// Compile a fragment shader for native OpenGL/WebGL rendering.
/// mode: "matcap", "wire", or "normals"
#[wasm_bindgen]
pub fn get_fragment_glsl(mode: &str) -> String {
    match quilting_shaders::compile_fragment_glsl_native(mode) {
        Ok(glsl) => glsl,
        Err(e) => format!("// ERROR: {}", e),
    }
}

thread_local! {
    /// Per-face material indices for the current loaded model.
    /// Index i -> material index for face i (None = default material).
    static FACE_MATERIALS: RefCell<Vec<Option<usize>>> = RefCell::new(Vec::new());
    /// Stored glTF data for animation switching without re-parsing.
    static GLTF_DATA: RefCell<Option<StoredGltfData>> = RefCell::new(None);
}

/// Set per-face material indices (for KHR_materials_variants switching).
#[wasm_bindgen]
pub fn set_face_materials(materials: &[i32]) {
    FACE_MATERIALS.with(|fm| {
        *fm.borrow_mut() = materials.iter().map(|&m| {
            if m >= 0 { Some(m as usize) } else { None }
        }).collect();
    });
}

/// List available animations for the current glTF model.
/// Returns a JS array of { index, name, duration, t_min, t_max }.
#[wasm_bindgen]
pub fn list_animations() -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };
        let info = quilting_gltf::evaluator::list_animations(&data.animations);
        let arr = js_sys::Array::new();
        for i in &info {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &"index".into(), &JsValue::from_f64(i.index as f64)).unwrap();
            let fallback = format!("Animation {}", i.index);
            let name = i.name.as_deref().unwrap_or(&fallback);
            js_sys::Reflect::set(&obj, &"name".into(), &JsValue::from_str(name)).unwrap();
            js_sys::Reflect::set(&obj, &"duration".into(), &JsValue::from_f64(i.duration)).unwrap();
            js_sys::Reflect::set(&obj, &"t_min".into(), &JsValue::from_f64(i.t_min)).unwrap();
            js_sys::Reflect::set(&obj, &"t_max".into(), &JsValue::from_f64(i.t_max)).unwrap();
            arr.push(&obj);
        }
        arr.into()
    })
}

/// Switch to a different animation by index. Rebakes the HyperMesh.
/// Returns a JS object with { time_min, time_max, num_vertices, num_faces }
/// or null on failure.
#[wasm_bindgen]
pub fn set_active_animation(index: u32) -> JsValue {
    let index = index as usize;

    GLTF_DATA.with(|gd| {
        let mut data = gd.borrow_mut();
        let data = match data.as_mut() {
            Some(d) => d,
            None => return JsValue::NULL,
        };

        if index >= data.animations.len() {
            web_sys::console::warn_1(&format!(
                "set_active_animation: index {} out of range ({})", index, data.animations.len()
            ).into());
            return JsValue::NULL;
        }

        data.active_animation = index;

        // Rebuild evaluator for the new animation
        if let Some(si) = data.primary_skin_idx {
            if si < data.skins.len() {
                let num_morph = data.combined.morph_targets.len();
                data.evaluator = Some(quilting_gltf::evaluator::AnimationEvaluator::new(
                    data.animations[index].clone(),
                    Some(data.skins[si].clone()),
                    data.nodes.clone(),
                    num_morph,
                ));
            }
        }

        // Get animation time range from evaluator
        let anim_info = quilting_gltf::evaluator::list_animations(&data.animations);
        let (time_min, time_max) = anim_info.get(index)
            .map(|a| (a.t_min, a.t_max))
            .unwrap_or((0.0, 0.0));
        let n_verts = data.combined.positions.len();
        let n_faces = data.combined.triangles.len();

        FACE_MATERIALS.with(|fm| *fm.borrow_mut() = data.face_material_indices.clone());
        SENT_TESS.with(|s| s.borrow_mut().clear());

        web_sys::console::log_1(&format!(
            "set_active_animation: switched to animation {} — {} verts, {} faces, time [{:.3}, {:.3}]",
            index, n_verts, n_faces, time_min, time_max
        ).into());

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"time_min".into(), &JsValue::from_f64(time_min)).unwrap();
        js_sys::Reflect::set(&result, &"time_max".into(), &JsValue::from_f64(time_max)).unwrap();
        js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(n_verts as f64)).unwrap();
        js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(n_faces as f64)).unwrap();
        result.into()
    })
}

/// Evaluate skeletal animation at time t and return joint matrices as a flat f32 array.
/// Returns null if no evaluator is available.
/// The matrices are skin matrices (joint_world × inverse_bind), column-major, ready for UBO upload.
#[wasm_bindgen]
pub fn evaluate_animation_frame(t: f64) -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };
        let evaluator = match data.evaluator.as_ref() {
            Some(e) => e,
            None => return JsValue::NULL,
        };
        let pose = evaluator.evaluate(t);

        let result = js_sys::Object::new();

        // Joint matrices (skeletal skinning)
        if !pose.joint_matrices.is_empty() {
            // Sandwich each joint matrix with normalization:
            //   norm_M = norm * M * unnorm
            let c = data.norm_center;
            let s = data.norm_scale;
            let si = if s.abs() > 1e-10 { 1.0 / s } else { 1.0 };
            let sf = s as f32;
            let sif = si as f32;
            let cx = c[0] as f32; let cy = c[1] as f32; let cz = c[2] as f32;
            let norm: [f32; 16] = [
                sf, 0.0, 0.0, 0.0, 0.0, sf, 0.0, 0.0,
                0.0, 0.0, sf, 0.0, -cx*sf, -cy*sf, -cz*sf, 1.0,
            ];
            let unnorm: [f32; 16] = [
                sif, 0.0, 0.0, 0.0, 0.0, sif, 0.0, 0.0,
                0.0, 0.0, sif, 0.0, cx, cy, cz, 1.0,
            ];

            fn mat4_mul_f32(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
                let mut out = [0.0f32; 16];
                for col in 0..4 {
                    for row in 0..4 {
                        out[col * 4 + row] = a[row] * b[col * 4]
                            + a[4 + row] * b[col * 4 + 1]
                            + a[8 + row] * b[col * 4 + 2]
                            + a[12 + row] * b[col * 4 + 3];
                    }
                }
                out
            }

            let num_joints = pose.joint_matrices.len() / 16;
            let mut result_mats = Vec::with_capacity(pose.joint_matrices.len());
            for ji in 0..num_joints {
                let m: [f32; 16] = pose.joint_matrices[ji*16..(ji+1)*16].try_into().unwrap();
                let mu = mat4_mul_f32(&m, &unnorm);
                let nmu = mat4_mul_f32(&norm, &mu);
                result_mats.extend_from_slice(&nmu);
            }
            let arr = js_sys::Float32Array::new_with_length(result_mats.len() as u32);
            arr.copy_from(&result_mats);
            js_sys::Reflect::set(&result, &"matrices".into(), &arr).unwrap();
        }

        // Morph weights (morph target animation)
        if !pose.morph_weights.is_empty() {
            let arr = js_sys::Float32Array::new_with_length(pose.morph_weights.len() as u32);
            arr.copy_from(&pose.morph_weights);
            js_sys::Reflect::set(&result, &"morph_weights".into(), &arr).unwrap();
        }

        result.into()
    })
}

/// Get per-vertex animation data (skinning + morph targets) for the current model.
/// Returns a JS object with:
///   { joint_indices?, joint_weights?, num_vertices, num_joints, num_morph_targets,
///     morph_deltas?, t_min, t_max, duration }
/// Returns null if no animated model is loaded.
#[wasm_bindgen]
pub fn get_skinning_data() -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) if d.evaluator.is_some() => d,
            _ => return JsValue::NULL,
        };

        let nv = data.combined.positions.len();
        let has_skin = data.combined.joint_indices.is_some();
        let num_morph = data.combined.morph_targets.len();

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(nv as f64)).unwrap();
        js_sys::Reflect::set(&result, &"num_morph_targets".into(), &JsValue::from_f64(num_morph as f64)).unwrap();

        // Joint data (optional — only for skeletal skinning)
        if has_skin {
            let ji = data.combined.joint_indices.as_ref().unwrap();
            let jw = data.combined.joint_weights.as_ref().unwrap();
            let mut indices_f32 = Vec::with_capacity(nv * 4);
            for idx in ji {
                indices_f32.extend_from_slice(&[idx[0] as f32, idx[1] as f32, idx[2] as f32, idx[3] as f32]);
            }
            let mut weights_f32 = Vec::with_capacity(nv * 4);
            for w in jw {
                weights_f32.extend_from_slice(&[w[0], w[1], w[2], w[3]]);
            }
            let ji_arr = js_sys::Float32Array::new_with_length(indices_f32.len() as u32);
            ji_arr.copy_from(&indices_f32);
            let jw_arr = js_sys::Float32Array::new_with_length(weights_f32.len() as u32);
            jw_arr.copy_from(&weights_f32);
            js_sys::Reflect::set(&result, &"joint_indices".into(), &ji_arr).unwrap();
            js_sys::Reflect::set(&result, &"joint_weights".into(), &jw_arr).unwrap();
        }

        // Morph target deltas (optional — only for morph target animation)
        // Deltas are scaled by norm_scale to match normalized position space.
        if num_morph > 0 {
            let ns = data.norm_scale as f32;
            let mut deltas = Vec::with_capacity(num_morph * nv * 3);
            for target in &data.combined.morph_targets {
                for vi in 0..nv {
                    if vi < target.len() {
                        deltas.push(target[vi][0] as f32 * ns);
                        deltas.push(target[vi][1] as f32 * ns);
                        deltas.push(target[vi][2] as f32 * ns);
                    } else {
                        deltas.extend_from_slice(&[0.0, 0.0, 0.0]);
                    }
                }
            }
            let delta_arr = js_sys::Float32Array::new_with_length(deltas.len() as u32);
            delta_arr.copy_from(&deltas);
            js_sys::Reflect::set(&result, &"morph_deltas".into(), &delta_arr).unwrap();
        }

        // Evaluator metadata
        if let Some(ref eval) = data.evaluator {
            let (t_min, t_max) = eval.time_range();
            js_sys::Reflect::set(&result, &"num_joints".into(), &JsValue::from_f64(eval.num_joints() as f64)).unwrap();
            js_sys::Reflect::set(&result, &"t_min".into(), &JsValue::from_f64(t_min)).unwrap();
            js_sys::Reflect::set(&result, &"t_max".into(), &JsValue::from_f64(t_max)).unwrap();
            js_sys::Reflect::set(&result, &"duration".into(), &JsValue::from_f64(eval.duration())).unwrap();
        }
        result.into()
    })
}

/// Build rest-pose instance data for GPU skinning (static, uploaded once).
/// LODs are computed from skinned positions at time `lod_time` for accuracy.
/// Returns a Float32Array with 40 floats per face (compact stride).
/// Vertex indices are packed in p0.x/p1.x/p2.x for skinning texture lookup.
/// Returns null if no skinned model is loaded.
#[wasm_bindgen]
pub fn get_rest_pose_instances(lod_time: f64) -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            _ => return JsValue::NULL,
        };

        let combined = &data.combined;
        let nf = combined.triangles.len();
        let has_uvs = combined.uvs.is_some();

        const COMPACT_STRIDE: usize = 40;
        let mut instances = vec![0.0f32; nf * COMPACT_STRIDE];

        // Compute smooth normals from rest pose
        let n_verts = combined.positions.len();
        let mut vn = vec![[0.0f64; 3]; n_verts];
        for face in &combined.triangles {
            let v0 = combined.positions[face[0]];
            let v1 = combined.positions[face[1]];
            let v2 = combined.positions[face[2]];
            let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
            let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
            let fn_ = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
            for &vi in face { vn[vi][0] += fn_[0]; vn[vi][1] += fn_[1]; vn[vi][2] += fn_[2]; }
        }
        // Normalize
        for n in &mut vn {
            let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
            if len > 1e-10 { n[0] /= len; n[1] /= len; n[2] /= len; }
        }
        // Use glTF normals if available
        let normals = combined.normals.as_ref();

        // Normalize positions
        let center = data.norm_center;
        let norm_scale = data.norm_scale;

        let norm_positions: Vec<[f64; 3]> = combined.positions.iter().map(|v| {
            [(v[0]-center[0])*norm_scale, (v[1]-center[1])*norm_scale, (v[2]-center[2])*norm_scale]
        }).collect();

        let tris_usize: Vec<[usize; 3]> = combined.triangles.clone();

        // Use the REAL LOD computation from evaluate.rs — half-edge coherent,
        // Möbius-deformed medians, screen attenuation, proper snap to power of 2.
        // Identity Möbius for initial load; async recompute handles Möbius-dependent LOD.
        let vertex_uvs_f32: Option<Vec<[f32; 2]>> = combined.uvs.as_ref().map(|uvs| {
            uvs.iter().map(|uv| [uv[0] as f32, uv[1] as f32]).collect()
        });
        let vertex_normals_f32: Option<Vec<[f32; 3]>> = normals.map(|ns| {
            ns.iter().map(|n| [n[0] as f32, n[1] as f32, n[2] as f32]).collect()
        }).or_else(|| {
            // Compute smooth normals if glTF doesn't provide them
            Some(vn.iter().map(|n| [n[0] as f32, n[1] as f32, n[2] as f32]).collect())
        });

        let lod_instances = quilting_core::evaluate::compute_instances_with_uvs(
            &norm_positions,
            &tris_usize,
            &Mobius::identity(),
            None, // no screen info for initial load
            None, // will build half-edge mesh internally
            vertex_uvs_f32.as_deref(),
            vertex_normals_f32.as_deref(),
        );

        // Pack into compact 40-float format for GPU animation path
        for (fi, face) in combined.triangles.iter().enumerate() {
            let b = fi * COMPACT_STRIDE;
            let inst = &lod_instances[fi];

            // p0/p1/p2: vertex_index in .x, normalized rest-pose xyz in .yzw
            for (vi, &vert_idx) in face.iter().enumerate() {
                let off = b + vi * 4;
                instances[off]   = vert_idx as f32;
                instances[off+1] = norm_positions[vert_idx][0] as f32;
                instances[off+2] = norm_positions[vert_idx][1] as f32;
                instances[off+3] = norm_positions[vert_idx][2] as f32;
            }
            // Edge LODs from compute_instances (half-edge coherent)
            instances[b + 12] = inst.edge_lods[0] as f32;
            instances[b + 13] = inst.edge_lods[1] as f32;
            instances[b + 14] = inst.edge_lods[2] as f32;
            // Vertex LODs
            instances[b + 16] = inst.vertex_lods[0] as f32;
            instances[b + 17] = inst.vertex_lods[1] as f32;
            instances[b + 18] = inst.vertex_lods[2] as f32;
            // UVs at offset 20
            instances[b+20] = inst.uvs[0][0]; instances[b+21] = inst.uvs[0][1];
            instances[b+22] = inst.uvs[1][0]; instances[b+23] = inst.uvs[1][1];
            instances[b+24] = inst.uvs[2][0]; instances[b+25] = inst.uvs[2][1];
            // Normals at offset 28
            for vi in 0..3 {
                let off = b + 28 + vi * 4;
                instances[off]   = inst.normals[vi][0];
                instances[off+1] = inst.normals[vi][1];
                instances[off+2] = inst.normals[vi][2];
            }
        }

        let arr = js_sys::Float32Array::new_with_length(instances.len() as u32);
        arr.copy_from(&instances);

        // Build per-face LOD classification for batch grouping
        // Each face: [canonical_a, canonical_b, canonical_c, perm_index, parity]
        let mut face_lods = Vec::with_capacity(nf * 5);
        for fi in 0..nf {
            let b = fi * COMPACT_STRIDE;
            let l0 = instances[b + 12] as u32;
            let l1 = instances[b + 13] as u32;
            let l2 = instances[b + 14] as u32;
            let lods = [l0, l1, l2];
            let ck = quilting_core::permutation::canonical_form(lods);
            let canonical = ck.res;
            let perm_idx = ck.perm_index;
            let parity = quilting_core::permutation::perm_sign(perm_idx);
            face_lods.push(canonical[0] as f32);
            face_lods.push(canonical[1] as f32);
            face_lods.push(canonical[2] as f32);
            face_lods.push(perm_idx as f32);
            face_lods.push(parity as f32);
        }
        let lod_arr = js_sys::Float32Array::new_with_length(face_lods.len() as u32);
        lod_arr.copy_from(&face_lods);

        // Compute bounding box for camera framing
        let mut bb_min = [f64::INFINITY; 3];
        let mut bb_max = [f64::NEG_INFINITY; 3];
        for pos in &combined.positions {
            for i in 0..3 {
                bb_min[i] = bb_min[i].min(pos[i]);
                bb_max[i] = bb_max[i].max(pos[i]);
            }
        }
        let extent = (bb_max[0]-bb_min[0]).max(bb_max[1]-bb_min[1]).max(bb_max[2]-bb_min[2]);

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"instances".into(), &arr).unwrap();
        js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(nf as f64)).unwrap();
        js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(n_verts as f64)).unwrap();
        js_sys::Reflect::set(&result, &"stride".into(), &JsValue::from_f64(COMPACT_STRIDE as f64)).unwrap();
        js_sys::Reflect::set(&result, &"extent".into(), &JsValue::from_f64(extent)).unwrap();
        js_sys::Reflect::set(&result, &"face_lods".into(), &lod_arr).unwrap();
        result.into()
    })
}

/// Load a glTF/GLB file from raw bytes, bake animation into a HyperMesh,
/// and store it for slicing.  Accepts both GLB (binary) and glTF with
/// embedded base64 buffers — `gltf::import_slice` handles both.
///
/// Walks ALL scene nodes, applies world transforms, merges all meshes.
/// Returns a JS object with:
///   { time_min, time_max, num_vertices, num_faces, materials, textures,
///     base_color, metallic, roughness }
/// Built with js_sys to avoid serde overhead on large data.
#[wasm_bindgen]
pub fn load_gltf_data(data: &[u8]) -> JsValue {
    let t0 = js_sys::Date::now();

    let scene = match quilting_gltf::load_gltf_raw(data) {
        Ok(s) => s,
        Err(e) => {
            web_sys::console::warn_1(&format!("load_gltf_data: could not load: {e}").into());
            return JsValue::NULL;
        }
    };

    let t_parse = js_sys::Date::now();
    web_sys::console::log_1(&format!(
        "load_gltf_data: parsed in {:.0}ms — {} meshes, {} materials, {} animations, {} skins, {} nodes, {} raw textures",
        t_parse - t0, scene.meshes.len(), scene.materials.len(),
        scene.animations.len(), scene.skins.len(), scene.nodes.len(), scene.raw_textures.len()
    ).into());

    // Determine the active scene
    let active_scene = if let Some(si) = scene.default_scene {
        scene.scenes.get(si)
    } else {
        scene.scenes.first()
    };

    // Collect all (mesh_idx, skin_idx, world_transform) from the scene graph
    let mut mesh_nodes: Vec<MeshNodeRef> = Vec::new();

    if let Some(active) = active_scene {
        let world_transforms = quilting_gltf::scene::compute_world_transforms(&scene.nodes, active);
        for (node_idx, node) in scene.nodes.iter().enumerate() {
            if let Some(mi) = node.mesh {
                mesh_nodes.push(MeshNodeRef {
                    mesh_idx: mi,
                    skin_idx: node.skin,
                    world_transform: world_transforms[node_idx],
                });
            }
        }
    } else {
        // No scene graph — just use all meshes at identity
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        for (mi, _) in scene.meshes.iter().enumerate() {
            mesh_nodes.push(MeshNodeRef {
                mesh_idx: mi,
                skin_idx: None,
                world_transform: identity,
            });
        }
    }

    if mesh_nodes.is_empty() && !scene.meshes.is_empty() {
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        mesh_nodes.push(MeshNodeRef {
            mesh_idx: 0,
            skin_idx: None,
            world_transform: identity,
        });
    }

    if mesh_nodes.is_empty() {
        web_sys::console::error_1(&"load_gltf_data: no meshes found".into());
        return JsValue::NULL;
    }

    web_sys::console::log_1(&format!(
        "load_gltf_data: {} mesh nodes in scene", mesh_nodes.len()
    ).into());

    // Find the primary skinned node for animation baking
    let primary_skinned = mesh_nodes.iter()
        .find(|mn| mn.skin_idx.is_some())
        .or(mesh_nodes.first());

    let _primary_mesh_idx = primary_skinned.map(|mn| mn.mesh_idx).unwrap_or(0);
    let primary_skin_idx = primary_skinned.and_then(|mn| mn.skin_idx);

    // Merge all meshes with world transforms, tracking per-triangle material index.
    // For animation baking we use the primary mesh; for static we merge all.
    let has_animation = !scene.animations.is_empty() && (
        primary_skin_idx.is_some() || scene.animations[0].channels.iter()
            .any(|c| c.property == quilting_gltf::animation::AnimationProperty::MorphTargetWeights)
    );

    // Track per-triangle material indices across the merged mesh
    let mut face_material_indices: Vec<Option<usize>> = Vec::new();

    let combined = if has_animation {
        // For animated meshes, merge ALL mesh nodes sharing the primary skin.
        // Many glTF models split a skinned character into multiple meshes
        // (body, fur, wings, etc.) that all reference the same skeleton.
        let target_skin = primary_skin_idx;
        let skinned_meshes: Vec<usize> = mesh_nodes.iter()
            .filter(|mn| mn.skin_idx == target_skin)
            .map(|mn| mn.mesh_idx)
            .collect();
        web_sys::console::log_1(&format!(
            "load_gltf_data: merging {} skinned meshes for bake", skinned_meshes.len()
        ).into());
        flatten_multi_mesh_for_bake(&scene.meshes, &skinned_meshes, &mut face_material_indices)
    } else {
        // For static models, merge ALL mesh nodes with world transforms
        merge_all_mesh_nodes(&scene, &mesh_nodes, &mut face_material_indices)
    };

    let n_verts = combined.positions.len();
    let n_tris = combined.triangles.len();
    web_sys::console::log_1(&format!(
        "load_gltf_data: merged — {n_verts} vertices, {n_tris} triangles, {} materials",
        scene.materials.len()
    ).into());

    // Get time range from animation if available, otherwise default
    let (time_min, time_max) = if !scene.animations.is_empty() {
        let info = quilting_gltf::evaluator::list_animations(&scene.animations);
        if let Some(a) = info.first() { (a.t_min, a.t_max) } else { (0.0, 0.0) }
    } else {
        (0.0, 0.0)
    };

    // Build materials array using js_sys
    let js_materials = js_sys::Array::new();
    for mat in &scene.materials {
        let obj = js_sys::Object::new();

        let bc = js_sys::Array::new();
        for &c in &mat.base_color_factor {
            bc.push(&JsValue::from_f64(c));
        }
        js_sys::Reflect::set(&obj, &"base_color".into(), &bc).unwrap();
        js_sys::Reflect::set(&obj, &"metallic".into(), &JsValue::from_f64(mat.metallic_factor)).unwrap();
        js_sys::Reflect::set(&obj, &"roughness".into(), &JsValue::from_f64(mat.roughness_factor)).unwrap();

        // Texture index (resolved: texture -> image)
        let tex_idx = mat.base_color_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = tex_idx {
            js_sys::Reflect::set(&obj, &"base_color_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"base_color_texture_index".into(), &JsValue::NULL).unwrap();
        }

        // Metallic-roughness texture
        let mr_idx = mat.metallic_roughness_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = mr_idx {
            js_sys::Reflect::set(&obj, &"metallic_roughness_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"metallic_roughness_texture_index".into(), &JsValue::NULL).unwrap();
        }

        // Normal texture + scale
        let normal_idx = mat.normal_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = normal_idx {
            js_sys::Reflect::set(&obj, &"normal_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"normal_texture_index".into(), &JsValue::NULL).unwrap();
        }
        js_sys::Reflect::set(&obj, &"normal_scale".into(), &JsValue::from_f64(mat.normal_scale)).unwrap();

        // Emissive
        let emissive = js_sys::Array::new();
        for &c in &mat.emissive_factor {
            emissive.push(&JsValue::from_f64(c));
        }
        js_sys::Reflect::set(&obj, &"emissive_factor".into(), &emissive).unwrap();
        let emissive_idx = mat.emissive_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = emissive_idx {
            js_sys::Reflect::set(&obj, &"emissive_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"emissive_texture_index".into(), &JsValue::NULL).unwrap();
        }

        // Occlusion
        let occ_idx = mat.occlusion_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = occ_idx {
            js_sys::Reflect::set(&obj, &"occlusion_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"occlusion_texture_index".into(), &JsValue::NULL).unwrap();
        }
        js_sys::Reflect::set(&obj, &"occlusion_strength".into(), &JsValue::from_f64(mat.occlusion_strength)).unwrap();

        // Alpha
        let alpha_mode_str = match mat.alpha_mode {
            quilting_gltf::material::AlphaMode::Opaque => "OPAQUE",
            quilting_gltf::material::AlphaMode::Mask => "MASK",
            quilting_gltf::material::AlphaMode::Blend => "BLEND",
        };
        js_sys::Reflect::set(&obj, &"alpha_mode".into(), &JsValue::from_str(alpha_mode_str)).unwrap();
        js_sys::Reflect::set(&obj, &"alpha_cutoff".into(), &JsValue::from_f64(mat.alpha_cutoff)).unwrap();
        js_sys::Reflect::set(&obj, &"double_sided".into(), &JsValue::from_bool(mat.double_sided)).unwrap();
        js_sys::Reflect::set(&obj, &"unlit".into(), &JsValue::from_bool(mat.unlit)).unwrap();

        // KHR_materials_sheen
        let sheen = js_sys::Array::new();
        for &c in &mat.sheen_color_factor { sheen.push(&JsValue::from_f64(c)); }
        js_sys::Reflect::set(&obj, &"sheen_color".into(), &sheen).unwrap();
        js_sys::Reflect::set(&obj, &"sheen_roughness".into(), &JsValue::from_f64(mat.sheen_roughness_factor)).unwrap();

        // KHR_materials_specular
        let spec = js_sys::Array::new();
        for &c in &mat.specular_color_factor { spec.push(&JsValue::from_f64(c)); }
        js_sys::Reflect::set(&obj, &"specular_color".into(), &spec).unwrap();

        // KHR_materials_ior / transmission / volume
        js_sys::Reflect::set(&obj, &"ior".into(), &JsValue::from_f64(mat.ior)).unwrap();
        js_sys::Reflect::set(&obj, &"transmission_factor".into(), &JsValue::from_f64(mat.transmission_factor)).unwrap();
        let trans_tex_idx = mat.transmission_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = trans_tex_idx {
            js_sys::Reflect::set(&obj, &"transmission_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"transmission_texture_index".into(), &JsValue::NULL).unwrap();
        }
        js_sys::Reflect::set(&obj, &"thickness_factor".into(), &JsValue::from_f64(mat.thickness_factor)).unwrap();
        let attn = js_sys::Array::new();
        for &c in &mat.attenuation_color { attn.push(&JsValue::from_f64(c)); }
        js_sys::Reflect::set(&obj, &"attenuation_color".into(), &attn).unwrap();
        js_sys::Reflect::set(&obj, &"attenuation_distance".into(), &JsValue::from_f64(mat.attenuation_distance)).unwrap();

        // KHR_texture_transform on normal map
        let nuv_s = js_sys::Array::new();
        nuv_s.push(&JsValue::from_f64(mat.normal_uv_scale[0]));
        nuv_s.push(&JsValue::from_f64(mat.normal_uv_scale[1]));
        js_sys::Reflect::set(&obj, &"normal_uv_scale".into(), &nuv_s).unwrap();
        let nuv_o = js_sys::Array::new();
        nuv_o.push(&JsValue::from_f64(mat.normal_uv_offset[0]));
        nuv_o.push(&JsValue::from_f64(mat.normal_uv_offset[1]));
        js_sys::Reflect::set(&obj, &"normal_uv_offset".into(), &nuv_o).unwrap();
        js_sys::Reflect::set(&obj, &"normal_uv_rotation".into(), &JsValue::from_f64(mat.normal_uv_rotation)).unwrap();

        // KHR_texture_transform on base color texture
        let buv_s = js_sys::Array::new();
        buv_s.push(&JsValue::from_f64(mat.base_uv_scale[0]));
        buv_s.push(&JsValue::from_f64(mat.base_uv_scale[1]));
        js_sys::Reflect::set(&obj, &"base_uv_scale".into(), &buv_s).unwrap();
        let buv_o = js_sys::Array::new();
        buv_o.push(&JsValue::from_f64(mat.base_uv_offset[0]));
        buv_o.push(&JsValue::from_f64(mat.base_uv_offset[1]));
        js_sys::Reflect::set(&obj, &"base_uv_offset".into(), &buv_o).unwrap();
        js_sys::Reflect::set(&obj, &"base_uv_rotation".into(), &JsValue::from_f64(mat.base_uv_rotation)).unwrap();

        js_materials.push(&obj);
    }

    // Build textures array — send raw image blobs for browser-native decoding
    let js_textures = js_sys::Array::new();
    for tex_info in &scene.raw_textures {
        let obj = js_sys::Object::new();
        // Send raw encoded bytes (PNG/JPEG) for browser-native decoding
        let raw = js_sys::Uint8Array::new_with_length(tex_info.blob.data.len() as u32);
        raw.copy_from(&tex_info.blob.data);
        js_sys::Reflect::set(&obj, &"raw_data".into(), &raw).unwrap();
        js_sys::Reflect::set(&obj, &"mime_type".into(), &JsValue::from_str(&tex_info.blob.mime_type)).unwrap();
        // Sampler wrap modes
        js_sys::Reflect::set(&obj, &"wrap_s".into(), &JsValue::from_f64(tex_info.wrap_s as f64)).unwrap();
        js_sys::Reflect::set(&obj, &"wrap_t".into(), &JsValue::from_f64(tex_info.wrap_t as f64)).unwrap();

        js_textures.push(&obj);
    }

    // Build per-face material index array (Int32Array, -1 for no material)
    let js_face_materials = js_sys::Int32Array::new_with_length(face_material_indices.len() as u32);
    let mat_indices: Vec<i32> = face_material_indices.iter()
        .map(|mi| mi.map(|i| i as i32).unwrap_or(-1))
        .collect();
    js_face_materials.copy_from(&mat_indices);

    // Extract first material as default (backward compat).
    // With raw image blobs (browser-native decode), we use base_color_factor directly.
    let (base_color, metallic, roughness) = if !scene.materials.is_empty() {
        let mat = &scene.materials[0];
        let bc = [
            mat.base_color_factor[0] as f32, mat.base_color_factor[1] as f32,
            mat.base_color_factor[2] as f32, mat.base_color_factor[3] as f32,
        ];
        (bc, mat.metallic_factor as f32, mat.roughness_factor as f32)
    } else {
        ([0.9, 0.75, 0.6, 1.0], 0.0, 0.4)
    };

    // Build AnimationEvaluator for GPU animation (skeletal and/or morph targets)
    let evaluator = if !scene.animations.is_empty() {
        let skin = primary_skin_idx.and_then(|si| scene.skins.get(si).cloned());
        let num_morph = combined.morph_targets.len();
        Some(quilting_gltf::evaluator::AnimationEvaluator::new(
            scene.animations[0].clone(),
            skin,
            scene.nodes.clone(),
            num_morph,
        ))
    } else { None };

    // Compute rest-pose bounding box for normalization
    let (norm_center, norm_scale) = {
        let mut bb_min = [f64::INFINITY; 3];
        let mut bb_max = [f64::NEG_INFINITY; 3];
        for pos in &combined.positions {
            for i in 0..3 {
                bb_min[i] = bb_min[i].min(pos[i]);
                bb_max[i] = bb_max[i].max(pos[i]);
            }
        }
        let center = [(bb_min[0]+bb_max[0])*0.5, (bb_min[1]+bb_max[1])*0.5, (bb_min[2]+bb_max[2])*0.5];
        let extent = ((bb_max[0]-bb_min[0]).max(bb_max[1]-bb_min[1]).max(bb_max[2]-bb_min[2])) * 0.5;
        let scale = if extent > 1e-10 { 1.0 / extent } else { 1.0 };
        (center, scale)
    };

    // Store glTF data for animation switching
    GLTF_DATA.with(|gd| {
        *gd.borrow_mut() = Some(StoredGltfData {
            animations: scene.animations.clone(),
            skins: scene.skins.clone(),
            nodes: scene.nodes.clone(),
            combined: combined.clone(),
            face_material_indices: face_material_indices.clone(),
            primary_skin_idx,
            active_animation: 0,
            evaluator,
            norm_center,
            norm_scale,
        });
    });

    FACE_MATERIALS.with(|fm| *fm.borrow_mut() = face_material_indices);
    SENT_TESS.with(|s| s.borrow_mut().clear());

    let t_end = js_sys::Date::now();
    web_sys::console::log_1(&format!(
        "load_gltf_data: total {:.0}ms — {} verts, {} faces, {} materials, {} textures, time [{:.3}, {:.3}]",
        t_end - t0, n_verts, n_tris, scene.materials.len(), scene.raw_textures.len(), time_min, time_max
    ).into());

    // Build result object using js_sys (not serde) for speed
    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"time_min".into(), &JsValue::from_f64(time_min)).unwrap();
    js_sys::Reflect::set(&result, &"time_max".into(), &JsValue::from_f64(time_max)).unwrap();
    js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(n_verts as f64)).unwrap();
    js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(n_tris as f64)).unwrap();
    js_sys::Reflect::set(&result, &"materials".into(), &js_materials).unwrap();
    js_sys::Reflect::set(&result, &"textures".into(), &js_textures).unwrap();
    js_sys::Reflect::set(&result, &"face_material_indices".into(), &js_face_materials).unwrap();

    // Backward-compat scalar fields
    let bc_arr = js_sys::Array::new();
    for &c in &base_color { bc_arr.push(&JsValue::from_f64(c as f64)); }
    js_sys::Reflect::set(&result, &"base_color".into(), &bc_arr).unwrap();
    js_sys::Reflect::set(&result, &"metallic".into(), &JsValue::from_f64(metallic as f64)).unwrap();
    js_sys::Reflect::set(&result, &"roughness".into(), &JsValue::from_f64(roughness as f64)).unwrap();

    // KHR_materials_variants: extract from raw glTF JSON
    // Parse the GLB header to get the JSON chunk and extract variant data.
    let js_variants = js_sys::Array::new();
    let js_variant_mappings = js_sys::Array::new();
    if data.len() > 12 {
        // Try to parse as GLB and extract variant info
        if let Ok(gltf_json) = {
            let mut pos = 12usize; // skip GLB header
            if pos + 8 <= data.len() {
                let chunk_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
                pos += 8; // skip chunk header
                if pos + chunk_len <= data.len() {
                    serde_json::from_slice::<serde_json::Value>(&data[pos..pos+chunk_len])
                } else {
                    Err(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, "bad chunk")))
                }
            } else {
                Err(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, "bad glb")))
            }
        } {
            // Extract variant names
            if let Some(vars) = gltf_json.pointer("/extensions/KHR_materials_variants/variants")
                .and_then(|v| v.as_array())
            {
                for v in vars {
                    let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("unnamed");
                    js_variants.push(&JsValue::from_str(name));
                }
            }
            // Extract per-primitive variant mappings
            if let Some(meshes) = gltf_json.get("meshes").and_then(|v| v.as_array()) {
                for mesh in meshes {
                    if let Some(prims) = mesh.get("primitives").and_then(|v| v.as_array()) {
                        for prim in prims {
                            if let Some(mappings) = prim.pointer("/extensions/KHR_materials_variants/mappings")
                                .and_then(|v| v.as_array())
                            {
                                for m in mappings {
                                    let obj = js_sys::Object::new();
                                    let mat = m.get("material").and_then(|v| v.as_u64()).unwrap_or(0);
                                    js_sys::Reflect::set(&obj, &"material".into(), &JsValue::from_f64(mat as f64)).unwrap();
                                    let vi = js_sys::Array::new();
                                    if let Some(vars) = m.get("variants").and_then(|v| v.as_array()) {
                                        for v in vars { vi.push(&JsValue::from_f64(v.as_u64().unwrap_or(0) as f64)); }
                                    }
                                    js_sys::Reflect::set(&obj, &"variants".into(), &vi).unwrap();
                                    js_variant_mappings.push(&obj);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    js_sys::Reflect::set(&result, &"variant_names".into(), &js_variants).unwrap();
    js_sys::Reflect::set(&result, &"variant_mappings".into(), &js_variant_mappings).unwrap();

    // Animation list for selector UI
    let anim_info = quilting_gltf::evaluator::list_animations(&scene.animations);
    let js_animations = js_sys::Array::new();
    for info in &anim_info {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"index".into(), &JsValue::from_f64(info.index as f64)).unwrap();
        let fallback = format!("Animation {}", info.index);
        let name = info.name.as_deref().unwrap_or(&fallback);
        js_sys::Reflect::set(&obj, &"name".into(), &JsValue::from_str(name)).unwrap();
        js_sys::Reflect::set(&obj, &"duration".into(), &JsValue::from_f64(info.duration)).unwrap();
        js_sys::Reflect::set(&obj, &"t_min".into(), &JsValue::from_f64(info.t_min)).unwrap();
        js_sys::Reflect::set(&obj, &"t_max".into(), &JsValue::from_f64(info.t_max)).unwrap();
        js_animations.push(&obj);
    }
    js_sys::Reflect::set(&result, &"animations".into(), &js_animations).unwrap();

    result.into()
}

/// Flatten all primitives in a mesh into a single Primitive suitable for baking.
/// Merges positions, triangles, joint data with proper index offsets.
/// Tracks per-triangle material indices in the output vec.
/// Merge multiple meshes (by index) into one Primitive for skinned animation baking.
/// All meshes should share the same skin/skeleton.
fn flatten_multi_mesh_for_bake(
    meshes: &[quilting_gltf::mesh::Mesh],
    mesh_indices: &[usize],
    face_materials: &mut Vec<Option<usize>>,
) -> quilting_gltf::mesh::Primitive {
    let mut positions = Vec::new();
    let mut normals_all: Option<Vec<[f64; 3]>> = None;
    let mut uvs_all: Option<Vec<[f64; 2]>> = None;
    let mut triangles = Vec::new();
    let mut joint_indices_all: Option<Vec<[u16; 4]>> = None;
    let mut joint_weights_all: Option<Vec<[f32; 4]>> = None;

    for &mi in mesh_indices {
        if mi >= meshes.len() { continue; }
        let mesh = &meshes[mi];
        for prim in &mesh.primitives {
            let offset = positions.len();
            positions.extend_from_slice(&prim.positions);
            let new_tris: Vec<[usize; 3]> = prim.triangles.iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset])
                .collect();
            for _ in &new_tris {
                face_materials.push(prim.material_index);
            }
            triangles.extend(new_tris);

            if let Some(ref n) = prim.normals {
                normals_all.get_or_insert_with(Vec::new).extend_from_slice(n);
            }
            if let Some(ref uv) = prim.uvs {
                uvs_all.get_or_insert_with(Vec::new).extend_from_slice(uv);
            }
            if let Some(ref ji) = prim.joint_indices {
                joint_indices_all.get_or_insert_with(Vec::new).extend_from_slice(ji);
            }
            if let Some(ref jw) = prim.joint_weights {
                joint_weights_all.get_or_insert_with(Vec::new).extend_from_slice(jw);
            }
        }
    }

    // Use morph targets from the first mesh's first primitive (if any)
    let morph_targets = mesh_indices.first()
        .and_then(|&mi| meshes.get(mi))
        .and_then(|m| m.primitives.first())
        .map(|p| p.morph_targets.clone())
        .unwrap_or_default();

    quilting_gltf::mesh::Primitive {
        positions,
        normals: normals_all,
        uvs: uvs_all,
        triangles,
        material_index: None,
        joint_indices: joint_indices_all,
        joint_weights: joint_weights_all,
        morph_targets,
        tangents: None,
    }
}

/// Merge all mesh nodes in the scene into one combined Primitive,
/// applying world transforms to positions and normals.
fn merge_all_mesh_nodes(
    scene: &quilting_gltf::GltfSceneRaw,
    mesh_nodes: &[MeshNodeRef],
    face_materials: &mut Vec<Option<usize>>,
) -> quilting_gltf::mesh::Primitive {
    let mut positions = Vec::new();
    let mut normals_all: Option<Vec<[f64; 3]>> = None;
    let mut uvs_all: Option<Vec<[f64; 2]>> = None;
    let mut triangles = Vec::new();

    for mn in mesh_nodes {
        let mesh = &scene.meshes[mn.mesh_idx];
        let m = &mn.world_transform;

        // Extract 3x3 normal matrix (inverse transpose of upper-left 3x3)
        // For uniform/no-shear transforms, the upper-left 3x3 works directly
        let nm = [
            m[0], m[1], m[2],
            m[4], m[5], m[6],
            m[8], m[9], m[10],
        ];

        for prim in &mesh.primitives {
            let offset = positions.len();

            // Transform positions by world matrix
            for &pos in &prim.positions {
                let x = m[0]*pos[0] + m[4]*pos[1] + m[8]*pos[2]  + m[12];
                let y = m[1]*pos[0] + m[5]*pos[1] + m[9]*pos[2]  + m[13];
                let z = m[2]*pos[0] + m[6]*pos[1] + m[10]*pos[2] + m[14];
                positions.push([x, y, z]);
            }

            // Transform normals — pad with defaults if primitive lacks them
            {
                let out = normals_all.get_or_insert_with(Vec::new);
                if let Some(ref normals) = prim.normals {
                    for &n in normals {
                        let nx = nm[0]*n[0] + nm[3]*n[1] + nm[6]*n[2];
                        let ny = nm[1]*n[0] + nm[4]*n[1] + nm[7]*n[2];
                        let nz = nm[2]*n[0] + nm[5]*n[1] + nm[8]*n[2];
                        let len = (nx*nx + ny*ny + nz*nz).sqrt();
                        if len > 1e-10 {
                            out.push([nx/len, ny/len, nz/len]);
                        } else {
                            out.push([0.0, 1.0, 0.0]);
                        }
                    }
                } else {
                    // Pad with up vector for vertices without normals
                    out.resize(out.len() + prim.positions.len(), [0.0, 1.0, 0.0]);
                }
            }

            // UVs — pad with zeros if primitive lacks them
            {
                let out = uvs_all.get_or_insert_with(Vec::new);
                if let Some(ref uv) = prim.uvs {
                    out.extend_from_slice(uv);
                } else {
                    out.resize(out.len() + prim.positions.len(), [0.0, 0.0]);
                }
            }

            let new_tris: Vec<[usize; 3]> = prim.triangles.iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset])
                .collect();
            for _ in &new_tris {
                face_materials.push(prim.material_index);
            }
            triangles.extend(new_tris);
        }
    }

    quilting_gltf::mesh::Primitive {
        positions,
        normals: normals_all,
        uvs: uvs_all,
        triangles,
        material_index: None,
        joint_indices: None,
        joint_weights: None,
        morph_targets: vec![],
        tangents: None,
    }
}

/// Helper struct to pass mesh node info to merge_all_mesh_nodes.
struct MeshNodeRef {
    mesh_idx: usize,
    #[allow(dead_code)]
    skin_idx: Option<usize>,
    world_transform: [f64; 16],
}


