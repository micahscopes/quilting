use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use quilting_core::atlas::{TessellationAtlas, BuildMode};
use quilting_renderer::compute::LodCompute;
use quilting_core::evaluate::{compute_instances, compute_instances_no_lod, compute_instances_no_lod_with_uvs, compute_instances_with_uvs, ScreenInfo};
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
}

struct PrebakeInfo {
    num_frames: usize,
    num_vertices: usize,
    num_faces: usize,
    time_min: f64,
    time_max: f64,
    dt: f64,
    mesh_radius: f64,
    /// CPU copy of all prebaked positions: [frame][vertex] = [x,y,z] as f32
    positions: Vec<f32>, // num_frames * num_vertices * 3
}

thread_local! {
    static ATLAS: RefCell<Option<TessellationAtlas>> = RefCell::new(None);
    static GPU_COMPUTE: RefCell<Option<(glow::Context, LodCompute)>> = RefCell::new(None);
    static PREBAKE_INFO: RefCell<Option<PrebakeInfo>> = RefCell::new(None);
    /// Pre-built flat f32 instance buffers from the prebake path.
    static FLAT_INSTANCE_DATA: RefCell<Option<(Vec<f32>, Vec<f32>)>> = RefCell::new(None);
    /// GPU classification: [atlas_index, perm_index] per face (2 floats each).
    static GPU_CLASSIFICATION: RefCell<Option<Vec<f32>>> = RefCell::new(None);
    /// Reusable sorted buffers — grow but never shrink to avoid per-frame allocation.
    static SORTED_BUFS: RefCell<(Vec<f32>, Vec<f32>)> = RefCell::new((Vec::new(), Vec::new()));
    /// Cached half-edge mesh — built once per shape, reused across frames.
    static MESH_CACHE: RefCell<Option<CachedMesh>> = RefCell::new(None);
    /// Sorted atlas patch keys — built once during LUT upload, reused for GPU readback.
    /// Guarantees stable index↔key mapping regardless of HashMap iteration order.
    static ATLAS_KEYS: RefCell<Vec<[u32; 3]>> = RefCell::new(Vec::new());
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

/// Pre-evaluate all animation frames and upload to GPU texture.
/// Returns the number of frames baked, or 0 on failure.
/// After this, slice_and_transform uses GPU texture lookup instead of CPU trajectory eval.
#[wasm_bindgen]
pub fn prebake_animation(num_frames: u32, time_min: f64, time_max: f64) -> u32 {
    let nf = num_frames as usize;
    let dt = if nf > 1 { (time_max - time_min) / (nf - 1) as f64 } else { 0.0 };

    HYPER_MESH.with(|hm| {
        let mesh_opt = hm.borrow();
        let mesh = match mesh_opt.as_ref() {
            Some(m) => m,
            None => return 0,
        };

        let num_verts = mesh.num_vertices as usize;

        // Pre-evaluate all frames
        let mut all_positions = Vec::with_capacity(nf * num_verts * 3);
        for fi in 0..nf {
            let t = time_min + fi as f64 * dt;
            let verts = mesh.positions_at(t);
            for v in &verts {
                all_positions.push(v[0] as f32);
                all_positions.push(v[1] as f32);
                all_positions.push(v[2] as f32);
            }
        }

        // Face indices as floats (for vertex attrib)
        let face_indices_f32: Vec<f32> = mesh.faces.iter()
            .flat_map(|f| [f[0] as f32, f[1] as f32, f[2] as f32])
            .collect();
        let num_faces = mesh.faces.len();

        // Upload to GPU
        GPU_COMPUTE.with(|gc| {
            let mut gc = gc.borrow_mut();
            let (gl, compute) = match gc.as_mut() {
                Some(pair) => pair,
                None => return 0,
            };

            compute.upload_positions_texture(gl, &all_positions, num_verts, nf);
            compute.upload_face_indices(gl, &face_indices_f32);

            // Build and upload atlas LUT: exponent triple → atlas index
            // key = exp_a + exp_b*10 + exp_c*100 where exp = log2(lod)
            // LUT size 1200 handles exponents up to 10 (key up to 1110).
            ATLAS.with(|atlas_cell| {
                let atlas = atlas_cell.borrow();
                if let Some(atlas) = atlas.as_ref() {
                    const LUT_SIZE: usize = 1200;
                    let mut lut = vec![255u8; LUT_SIZE]; // 255 = no entry
                    let mut keys: Vec<[u32; 3]> = atlas.patches.keys().copied().collect();
                    keys.sort();
                    for (idx, key) in keys.iter().enumerate() {
                        if idx >= 255 { break; } // u8 index limit
                        // key is sorted canonical [a,b,c]
                        let ea = (key[0] as f64).log2().round() as usize;
                        let eb = (key[1] as f64).log2().round() as usize;
                        let ec = (key[2] as f64).log2().round() as usize;
                        let lut_key = ea + eb * 10 + ec * 100;
                        if lut_key < LUT_SIZE {
                            lut[lut_key] = idx as u8;
                        }
                    }
                    compute.upload_atlas_lut(gl, &lut);
                    ATLAS_KEYS.with(|ak| *ak.borrow_mut() = keys.clone());
                    web_sys::console::log_1(&format!(
                        "Atlas LUT: {} entries mapped", keys.len()
                    ).into());
                }
            });

            // Compute mesh radius from frame 0
            let mesh_radius = {
                let (mut cx, mut cy, mut cz) = (0.0f64, 0.0, 0.0);
                let n = num_verts as f64;
                for i in 0..num_verts {
                    cx += all_positions[i*3] as f64;
                    cy += all_positions[i*3+1] as f64;
                    cz += all_positions[i*3+2] as f64;
                }
                cx /= n; cy /= n; cz /= n;
                let mut max_r = 0.0f64;
                for i in 0..num_verts {
                    let dx = all_positions[i*3] as f64 - cx;
                    let dy = all_positions[i*3+1] as f64 - cy;
                    let dz = all_positions[i*3+2] as f64 - cz;
                    max_r = max_r.max((dx*dx + dy*dy + dz*dz).sqrt());
                }
                max_r.max(1e-6)
            };

            // Store prebake info
            PREBAKE_INFO.with(|pi| *pi.borrow_mut() = Some(PrebakeInfo {
                num_frames: nf,
                num_vertices: num_verts,
                num_faces,
                time_min, time_max, dt,
                mesh_radius,
                positions: all_positions.clone(),
            }));

            web_sys::console::log_1(&format!(
                "Prebaked {} frames × {} verts × {} faces = {:.1}MB on GPU",
                nf, num_verts, num_faces,
                all_positions.len() as f64 * 4.0 / 1e6
            ).into());

            nf as u32
        })
    })
}

/// Run GPU LOD computation via transform feedback.
/// control_points: flat f32 array, 12 floats per face (3 quaternions wxyz)
/// mobius: 16 floats (a,b,c,d quaternions)
/// Returns: flat f32 array, 2 floats per face (atlas_index, perm_index)
#[wasm_bindgen]
pub fn gpu_compute_lods(
    control_points: &[f32],
    mobius: &[f32],
    density: f32,
    mesh_radius: f32,
) -> Vec<f32> {
    let num_faces = control_points.len() / 12;
    GPU_COMPUTE.with(|gc| {
        let mut gc = gc.borrow_mut();
        let (gl, compute) = match gc.as_mut() {
            Some(pair) => pair,
            None => return vec![],
        };
        compute.upload_control_points(gl, control_points);
        let mut mob = [0.0f32; 16];
        for (i, &v) in mobius.iter().take(16).enumerate() {
            mob[i] = v;
        }
        let identity_vp = [0.0f32; 16];
        let n = compute.compute(gl, num_faces, mob, density, mesh_radius, 0.0, &identity_vp, 0.0, 0.0);
        compute.read_back(gl, n)
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
/// slice_and_transform never needs to send tessellation data again.
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

            let tris: Vec<u32> = mesh.triangles.iter()
                .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

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

            let used_flat = FLAT_INSTANCE_DATA.with(|fid| {
                let fid = fid.borrow();
                if let Some((ref flat_orig, ref flat_xform)) = *fid {
                    for (i, &fi) in face_indices.iter().enumerate() {
                        let src = fi * 52;
                        if src + 52 <= flat_orig.len() {
                            orig_data[i*52..(i+1)*52].copy_from_slice(&flat_orig[src..src+52]);
                            xform_data[i*52..(i+1)*52].copy_from_slice(&flat_xform[src..src+52]);
                        }
                    }
                    true
                } else {
                    false
                }
            });
            if !used_flat {
                for (i, &fi) in face_indices.iter().enumerate() {
                    let o = instances_orig[fi].to_f32_array();
                    let x = instances_xform[fi].to_f32_array();
                    orig_data[i*52..(i+1)*52].copy_from_slice(&o);
                    xform_data[i*52..(i+1)*52].copy_from_slice(&x);
                }
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

// --- Spacetime slicing ---

struct CachedSlice {
    normal: [f64; 4],
    offset: f64,
    verts: Vec<[f64; 3]>,
    tris: Vec<[usize; 3]>,
    weights: Vec<[f64; 4]>,  // per-vertex conformal weights from 4D Möbius
    uvs: Vec<[f32; 2]>,      // per-vertex texture coordinates
    normals: Vec<[f32; 3]>,  // per-vertex smooth normals
    half_edge: HalfEdgeMesh,
    /// Maps sliced face index -> original HyperMesh face index.
    source_face_indices: Vec<usize>,
    /// Cached original (untransformed) instances — recomputed only when slice changes.
    orig_instances: Vec<quilting_core::evaluate::FaceInstance>,
}

thread_local! {
    static HYPER_MESH: RefCell<Option<quilting_spacetime::HyperMesh>> = RefCell::new(None);
    static SLICE_CACHE: RefCell<Option<CachedSlice>> = RefCell::new(None);
    /// Per-face material indices for the current loaded model.
    /// Index i -> material index for HyperMesh face i (None = default material).
    static FACE_MATERIALS: RefCell<Vec<Option<usize>>> = RefCell::new(Vec::new());
}

#[derive(serde::Serialize)]
struct SpacetimeLayerData {
    positions: Vec<f64>,
    faces: Vec<u32>,
    times: Vec<f64>,
}

#[derive(serde::Serialize)]
struct SpacetimeSliceData {
    layers: Vec<SpacetimeLayerData>,
}

#[derive(serde::Serialize)]
struct HyperMeshInfo {
    time_min: f64,
    time_max: f64,
    num_vertices: u32,
    num_faces: usize,
    // PBR material from glTF (if available)
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
}

/// Initialize a hypermesh by name. Stores it in a thread-local.
/// Names: "rotating_cube", "breathing_sphere", "colliding_spheres", "twisting_torus"
#[wasm_bindgen]
pub fn create_hypermesh(name: &str) -> JsValue {
    use quilting_spacetime::synthesize;

    let mut mesh = match name {
        "rotating_cube" => synthesize::rotating_cube(2.0, std::f64::consts::TAU, 128),
        "breathing_sphere" => synthesize::breathing_sphere(2.0, 1.0, 0.3, 6),
        "colliding_spheres" => synthesize::colliding_spheres(2.0, 1.5, 4.0, 4),
        "twisting_torus" => synthesize::twisting_torus(2.0, 2.0, 2.0, 0.5, 32, 24),
        "galloping_horse" => synthesize::galloping_horse(2.0),
        _ => synthesize::rotating_cube(2.0, std::f64::consts::TAU, 128),
    };

    // Capture animation range before padding
    let (time_min, time_max) = mesh.time_range();

    // Toroidal embedding handles periodicity — no loop padding needed
    let info = HyperMeshInfo {
        time_min,
        time_max,
        num_vertices: mesh.num_vertices,
        num_faces: mesh.faces.len(),
        base_color: [0.9, 0.75, 0.6, 1.0], // default warm clay
        metallic: 0.0,
        roughness: 0.4,
    };

    HYPER_MESH.with(|hm| *hm.borrow_mut() = Some(mesh));
    FACE_MATERIALS.with(|fm| *fm.borrow_mut() = Vec::new()); // built-in shapes have no materials
    // Clear sent-tess cache so JS gets fresh tessellation data after shape change
    SENT_TESS.with(|s| s.borrow_mut().clear());

    serde_wasm_bindgen::to_value(&info).unwrap()
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

    let scene = match quilting_gltf::load_gltf(data) {
        Ok(s) => s,
        Err(e) => {
            web_sys::console::warn_1(&format!("load_gltf_data: could not load: {e}").into());
            return JsValue::NULL;
        }
    };

    let t_parse = js_sys::Date::now();
    web_sys::console::log_1(&format!(
        "load_gltf_data: parsed in {:.0}ms — {} meshes, {} materials, {} animations, {} skins, {} nodes, {} images",
        t_parse - t0, scene.meshes.len(), scene.materials.len(),
        scene.animations.len(), scene.skins.len(), scene.nodes.len(), scene.images.len()
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

    let primary_mesh_idx = primary_skinned.map(|mn| mn.mesh_idx).unwrap_or(0);
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
        // For animated meshes, use just the primary mesh (skinning needs single mesh)
        let mesh = &scene.meshes[primary_mesh_idx];
        flatten_primitives_for_bake(mesh, &mut face_material_indices)
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

    let num_samples = 32usize;
    let hyper = if let Some(si) = primary_skin_idx {
        if !scene.animations.is_empty() && !scene.skins.is_empty() && si < scene.skins.len() {
            let skin = &scene.skins[si];
            let anim = &scene.animations[0];
            web_sys::console::log_1(&format!(
                "load_gltf_data: baking skinned animation with {} samples ({} joints)",
                num_samples, skin.joints.len()
            ).into());
            quilting_gltf::bake::bake_skinned_animation(
                &combined, skin, anim, &scene.nodes, num_samples,
            )
        } else {
            build_static_hypermesh(&combined)
        }
    } else if has_animation {
        let has_morph = scene.animations[0].channels.iter()
            .any(|c| c.property == quilting_gltf::animation::AnimationProperty::MorphTargetWeights);
        if has_morph && !combined.morph_targets.is_empty() {
            web_sys::console::log_1(&format!(
                "load_gltf_data: baking morph animation ({} targets, 32 samples)",
                combined.morph_targets.len()
            ).into());
            quilting_gltf::bake::bake_morph_animation(
                &combined,
                &scene.animations[0],
                &combined.morph_targets,
                32,
            )
        } else {
            build_static_hypermesh(&combined)
        }
    } else {
        build_static_hypermesh(&combined)
    };

    let mut hyper = normalize_hypermesh(hyper);

    // Set per-vertex UVs on the HyperMesh if the combined primitive has them
    if let Some(ref uvs) = combined.uvs {
        hyper.vertex_uvs = uvs.iter().map(|uv| [uv[0] as f32, uv[1] as f32]).collect();
    }

    // Set per-vertex normals on the HyperMesh if the combined primitive has them
    if let Some(ref norms) = combined.normals {
        hyper.vertex_normals = norms.iter().map(|n| [n[0] as f32, n[1] as f32, n[2] as f32]).collect();
    }

    let (time_min, time_max) = hyper.time_range();

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

    // Build textures array — send image data + average color for each image
    let js_textures = js_sys::Array::new();
    for img in &scene.images {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"width".into(), &JsValue::from_f64(img.width as f64)).unwrap();
        js_sys::Reflect::set(&obj, &"height".into(), &JsValue::from_f64(img.height as f64)).unwrap();

        // Compute average color (for fallback when not sampling textures)
        let pixel_count = (img.width * img.height) as f64;
        if pixel_count > 0.0 {
            let mut r_sum: f64 = 0.0;
            let mut g_sum: f64 = 0.0;
            let mut b_sum: f64 = 0.0;
            let mut a_sum: f64 = 0.0;
            for chunk in img.pixels.chunks(4) {
                if chunk.len() == 4 {
                    // sRGB to linear for averaging
                    r_sum += (chunk[0] as f64 / 255.0).powf(2.2);
                    g_sum += (chunk[1] as f64 / 255.0).powf(2.2);
                    b_sum += (chunk[2] as f64 / 255.0).powf(2.2);
                    a_sum += chunk[3] as f64 / 255.0;
                }
            }
            let avg_color = js_sys::Array::new();
            // Convert back from linear to sRGB for display
            avg_color.push(&JsValue::from_f64((r_sum / pixel_count).powf(1.0 / 2.2)));
            avg_color.push(&JsValue::from_f64((g_sum / pixel_count).powf(1.0 / 2.2)));
            avg_color.push(&JsValue::from_f64((b_sum / pixel_count).powf(1.0 / 2.2)));
            avg_color.push(&JsValue::from_f64(a_sum / pixel_count));
            js_sys::Reflect::set(&obj, &"avg_color".into(), &avg_color).unwrap();
        }

        // Send pixel data as Uint8Array
        let pixels = js_sys::Uint8Array::new_with_length(img.pixels.len() as u32);
        pixels.copy_from(&img.pixels);
        js_sys::Reflect::set(&obj, &"data".into(), &pixels).unwrap();

        js_textures.push(&obj);
    }

    // Build per-face material index array (Int32Array, -1 for no material)
    let js_face_materials = js_sys::Int32Array::new_with_length(face_material_indices.len() as u32);
    let mat_indices: Vec<i32> = face_material_indices.iter()
        .map(|mi| mi.map(|i| i as i32).unwrap_or(-1))
        .collect();
    js_face_materials.copy_from(&mat_indices);

    // Extract first material as default (backward compat with existing HyperMeshInfo)
    let (base_color, metallic, roughness) = if !scene.materials.is_empty() {
        let mat = &scene.materials[0];
        let mut bc = [
            mat.base_color_factor[0] as f32, mat.base_color_factor[1] as f32,
            mat.base_color_factor[2] as f32, mat.base_color_factor[3] as f32,
        ];
        // If this material has a texture, use the average color instead of the factor
        if let Some(tex_ref) = &mat.base_color_texture {
            if let Some(&img_idx) = scene.texture_to_image.get(tex_ref.index) {
                if let Some(img) = scene.images.get(img_idx) {
                    let pixel_count = (img.width * img.height) as f64;
                    if pixel_count > 0.0 {
                        let mut r: f64 = 0.0;
                        let mut g: f64 = 0.0;
                        let mut b: f64 = 0.0;
                        for chunk in img.pixels.chunks(4) {
                            if chunk.len() == 4 {
                                r += (chunk[0] as f64 / 255.0).powf(2.2);
                                g += (chunk[1] as f64 / 255.0).powf(2.2);
                                b += (chunk[2] as f64 / 255.0).powf(2.2);
                            }
                        }
                        // Multiply texture average with base_color_factor
                        bc[0] = (bc[0] as f64 * (r / pixel_count).powf(1.0 / 2.2)) as f32;
                        bc[1] = (bc[1] as f64 * (g / pixel_count).powf(1.0 / 2.2)) as f32;
                        bc[2] = (bc[2] as f64 * (b / pixel_count).powf(1.0 / 2.2)) as f32;
                    }
                }
            }
        }
        (bc, mat.metallic_factor as f32, mat.roughness_factor as f32)
    } else {
        ([0.9, 0.75, 0.6, 1.0], 0.0, 0.4)
    };

    HYPER_MESH.with(|hm| *hm.borrow_mut() = Some(hyper));
    FACE_MATERIALS.with(|fm| *fm.borrow_mut() = face_material_indices);
    SENT_TESS.with(|s| s.borrow_mut().clear());
    // Clear slice cache — use try_borrow_mut to avoid panic if a previous
    // WASM panic left the RefCell in a borrowed state.
    SLICE_CACHE.with(|c| {
        if let Ok(mut cache) = c.try_borrow_mut() {
            *cache = None;
        }
    });

    let t_end = js_sys::Date::now();
    web_sys::console::log_1(&format!(
        "load_gltf_data: total {:.0}ms — {} verts, {} faces, {} materials, {} textures, time [{:.3}, {:.3}]",
        t_end - t0, n_verts, n_tris, scene.materials.len(), scene.images.len(), time_min, time_max
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

    result.into()
}

/// Flatten all primitives in a mesh into a single Primitive suitable for baking.
/// Merges positions, triangles, joint data with proper index offsets.
/// Tracks per-triangle material indices in the output vec.
fn flatten_primitives_for_bake(
    mesh: &quilting_gltf::mesh::Mesh,
    face_materials: &mut Vec<Option<usize>>,
) -> quilting_gltf::mesh::Primitive {
    let mut positions = Vec::new();
    let mut normals_all: Option<Vec<[f64; 3]>> = None;
    let mut uvs_all: Option<Vec<[f64; 2]>> = None;
    let mut triangles = Vec::new();
    let mut joint_indices_all: Option<Vec<[u16; 4]>> = None;
    let mut joint_weights_all: Option<Vec<[f32; 4]>> = None;

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

    // Use morph targets from the first primitive (if any)
    let morph_targets = if let Some(first) = mesh.primitives.first() {
        first.morph_targets.clone()
    } else {
        vec![]
    };

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
    scene: &quilting_gltf::GltfScene,
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

            // Transform normals
            if let Some(ref normals) = prim.normals {
                let out = normals_all.get_or_insert_with(Vec::new);
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
            }

            // UVs pass through unchanged (not affected by world transform)
            if let Some(ref uv) = prim.uvs {
                uvs_all.get_or_insert_with(Vec::new).extend_from_slice(uv);
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

/// Build a static HyperMesh (no animation) from a primitive — 2 identical keyframes.
fn build_static_hypermesh(prim: &quilting_gltf::mesh::Primitive) -> quilting_spacetime::HyperMesh {
    use quilting_spacetime::trajectory::{HermiteSegment, VertexTrajectory};

    let faces: Vec<[u32; 3]> = prim.triangles.iter()
        .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    let trajectories: Vec<VertexTrajectory> = prim.positions.iter().map(|&pos| {
        VertexTrajectory {
            segments: vec![HermiteSegment {
                t_start: 0.0,
                t_end: 2.0,
                pos_start: pos,
                pos_end: pos,
                vel_start: [0.0, 0.0, 0.0],
                vel_end: [0.0, 0.0, 0.0],
            }],
        }
    }).collect();

    quilting_spacetime::HyperMesh::new(faces, trajectories)
}

/// Normalize a HyperMesh so positions fit roughly in [-1, 1].
/// Computes bounding box across all trajectory keyframes, then scales.
fn normalize_hypermesh(mut mesh: quilting_spacetime::HyperMesh) -> quilting_spacetime::HyperMesh {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];

    for traj in &mesh.trajectories {
        for seg in &traj.segments {
            for i in 0..3 {
                min[i] = min[i].min(seg.pos_start[i]).min(seg.pos_end[i]);
                max[i] = max[i].max(seg.pos_start[i]).max(seg.pos_end[i]);
            }
        }
    }

    let center = [(min[0]+max[0])*0.5, (min[1]+max[1])*0.5, (min[2]+max[2])*0.5];
    let extent = ((max[0]-min[0]).max(max[1]-min[1]).max(max[2]-min[2])) * 0.5;
    if extent < 1e-10 {
        return mesh;
    }
    let scale = 1.0 / extent;

    for traj in &mut mesh.trajectories {
        for seg in &mut traj.segments {
            for i in 0..3 {
                seg.pos_start[i] = (seg.pos_start[i] - center[i]) * scale;
                seg.pos_end[i] = (seg.pos_end[i] - center[i]) * scale;
                seg.vel_start[i] *= scale;
                seg.vel_end[i] *= scale;
            }
        }
    }

    mesh
}

/// Slice the current hypermesh with a hyperplane.
/// normal: [nx, ny, nz, nt] (4 floats)
/// offset: f64
/// Returns { layers: [{ positions: [f64], faces: [u32], times: [f64] }] }
#[wasm_bindgen]
pub fn slice_hypermesh(normal: &[f64], offset: f64) -> JsValue {
    use quilting_spacetime::HyperplaneSlicer;

    let n = if normal.len() >= 4 {
        [normal[0], normal[1], normal[2], normal[3]]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };

    let result = HYPER_MESH.with(|hm| {
        let mesh_opt = hm.borrow();
        let mesh = match mesh_opt.as_ref() {
            Some(m) => m,
            None => return SpacetimeSliceData { layers: vec![] },
        };

        let slicer = HyperplaneSlicer::new(n, offset);
        let slice = slicer.slice_marching(mesh);

        let layers = slice.layers.into_iter().map(|layer| {
            let positions: Vec<f64> = layer.positions.iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            let faces: Vec<u32> = layer.faces.iter()
                .flat_map(|f| [f[0], f[1], f[2]])
                .collect();
            SpacetimeLayerData {
                positions,
                faces,
                times: layer.times,
            }
        }).collect();

        SpacetimeSliceData { layers }
    });

    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Slice the current hypermesh, then apply a Möbius transform to the result.
/// Returns the same BatchData format that index.html's rendering pipeline expects.
#[wasm_bindgen]
pub fn slice_and_transform(
    normal: &[f64],
    offset: f64,
    transform_type: &str,
    toroidal: bool,
    params: &[f64],
    override_res: u32,
    vp_matrix: &[f64],
    viewport_width: f64,
    viewport_height: f64,
) -> JsValue {
    let n = if normal.len() >= 4 {
        [normal[0], normal[1], normal[2], normal[3]]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };

    let empty = || serde_wasm_bindgen::to_value(&MeshBatches {
        batches: vec![], total_faces: 0, num_batches: 0, timings: [0.0, 0.0, 0.0],
    }).unwrap();

    // Performance marks
    let perf_early: web_sys::Performance = js_sys::Reflect::get(
        &js_sys::global(), &"performance".into()
    ).ok().and_then(|p| p.dyn_into().ok()).unwrap();
    perf_early.mark("sat:fn-start").ok();

    let _t_start = js_sys::Date::now();

    let has_prebake = PREBAKE_INFO.with(|pi| pi.borrow().is_some());
    let cache_hit = SLICE_CACHE.with(|c| {
        let cache = c.borrow();
        match cache.as_ref() {
            Some(cs) => {
                // With prebake, only check normal (time comes from GPU texture)
                if has_prebake {
                    cs.normal == n
                } else {
                    cs.normal == n && (cs.offset - offset).abs() < 1e-12
                }
            }
            None => false,
        }
    });

    // With prebake, update the cached offset without rebuilding the slice
    if has_prebake && cache_hit {
        SLICE_CACHE.with(|c| {
            if let Some(cs) = c.borrow_mut().as_mut() {
                cs.offset = offset;
            }
        });
    }

    if !cache_hit {
        // Classic mode fast path: pure time slice (normal = [0,0,0,1]) without
        // tilt or toroidal embedding. Just evaluate each vertex's trajectory at
        // time t — no marching tetrahedra needed. O(V) instead of O(V*segs*F).
        let is_classic = !toroidal
            && n[0].abs() < 1e-10 && n[1].abs() < 1e-10 && n[2].abs() < 1e-10
            && (n[3] - 1.0).abs() < 1e-10;

        if is_classic {
            let built = HYPER_MESH.with(|hm| {
                let mesh_opt = hm.borrow();
                let mesh = match mesh_opt.as_ref() {
                    Some(m) => m,
                    None => return false,
                };
                let verts = mesh.positions_at(offset);
                let tris: Vec<[usize; 3]> = mesh.faces.iter()
                    .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
                    .collect();
                let uvs = mesh.vertex_uvs.clone();
                let normals = mesh.vertex_normals.clone();
                let source_faces: Vec<usize> = (0..tris.len()).collect();

                if verts.is_empty() || tris.is_empty() { return false; }

                let faces_u32: Vec<[u32; 3]> = tris.iter()
                    .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
                    .collect();
                let half_edge = HalfEdgeMesh::from_triangles(verts.len() as u32, &faces_u32);

                SLICE_CACHE.with(|c| *c.borrow_mut() = Some(CachedSlice {
                    normal: n, offset, verts, tris,
                    weights: vec![[0.0; 4]; 0], // no 4D weights in classic mode
                    uvs, normals, half_edge,
                    source_face_indices: source_faces,
                    orig_instances: Vec::new(),
                }));
                true
            });
            if !built { return empty(); }
        } else {
            // Full marching tetrahedra slice for tilted/toroidal hyperplanes
            let slice_result = HYPER_MESH.with(|hm| {
                let mesh_opt = hm.borrow();
                let mesh = match mesh_opt.as_ref() {
                    Some(m) => m,
                    None => return None,
                };
                let mut slicer = quilting_spacetime::HyperplaneSlicer::new(n, offset);
                if toroidal {
                    slicer = slicer.with_toroidal(2.0, mesh.period);
                }
                Some(slicer.slice_marching(mesh))
            });

            let slice = match slice_result {
                Some(s) if !s.layers.is_empty() => s,
                _ => return empty(),
            };

            // Merge ALL layers
            let mut verts: Vec<[f64; 3]> = Vec::new();
            let mut vert_weights: Vec<[f64; 4]> = Vec::new();
            let mut vert_uvs: Vec<[f32; 2]> = Vec::new();
            let mut vert_normals: Vec<[f32; 3]> = Vec::new();
            let mut tris: Vec<[usize; 3]> = Vec::new();
            let mut slice_source_faces: Vec<usize> = Vec::new();
            for layer in slice.layers {
                let base = verts.len();
                verts.extend_from_slice(&layer.positions);
                vert_weights.extend_from_slice(&layer.weights);
                vert_uvs.extend_from_slice(&layer.uvs);
                vert_normals.extend_from_slice(&layer.normals);
                for f in &layer.faces {
                    tris.push([f[0] as usize + base, f[1] as usize + base, f[2] as usize + base]);
                }
                slice_source_faces.extend_from_slice(&layer.source_face_indices);
            }

            if tris.is_empty() || verts.is_empty() {
                SLICE_CACHE.with(|c| *c.borrow_mut() = None);
                return empty();
            }

            let faces_u32: Vec<[u32; 3]> = tris.iter()
                .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
                .collect();
            let half_edge = HalfEdgeMesh::from_triangles(verts.len() as u32, &faces_u32);

            SLICE_CACHE.with(|c| *c.borrow_mut() = Some(CachedSlice {
                normal: n, offset, verts, tris, weights: vert_weights, uvs: vert_uvs,
                normals: vert_normals, half_edge, source_face_indices: slice_source_faces,
                orig_instances: Vec::new(),
            }));
        }
    }

    // Classic 3D Möbius — applied to each frame's spatial positions
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

    // Performance marks for DevTools User Timing
    let perf: web_sys::Performance = js_sys::Reflect::get(
        &js_sys::global(), &"performance".into()
    ).ok().and_then(|p| p.dyn_into().ok()).unwrap();
    let pm = |label: &str| { perf.mark(label).ok(); };

    pm("sat:mobius-start");

    // Use cached slice for Möbius computation
    let gpu_available = GPU_COMPUTE.with(|gc| gc.borrow().is_some());
    // Log once which path we're using
    thread_local! { static LOGGED_PATH: RefCell<bool> = RefCell::new(false); }
    LOGGED_PATH.with(|l| {
        if !*l.borrow() {
            *l.borrow_mut() = true;
            let path = if gpu_available && !transform.is_affine() { "GPU" } else { "CPU" };
            web_sys::console::log_1(&format!("LOD path: {path} (gpu={gpu_available}, affine={})", transform.is_affine()).into());
        }
    });

    // Pack Möbius as 16 floats (shared by both paths)
    let mob_f32: [f32; 16] = [
        transform.a.w as f32, transform.a.x as f32, transform.a.y as f32, transform.a.z as f32,
        transform.b.w as f32, transform.b.x as f32, transform.b.y as f32, transform.b.z as f32,
        transform.c.w as f32, transform.c.x as f32, transform.c.y as f32, transform.c.z as f32,
        transform.d.w as f32, transform.d.x as f32, transform.d.y as f32, transform.d.z as f32,
    ];
    let tess_density = quilting_core::evaluate::get_tess_density();

    // Check if we have prebaked animation data on GPU
    // Prebake info + frame positions for the current time
    let prebake_frame_data = PREBAKE_INFO.with(|pi| {
        let pi = pi.borrow();
        let p = match pi.as_ref() {
            Some(p) => p,
            None => return None,
        };
        let frame = if p.dt > 0.0 {
            (((offset - p.time_min) / p.dt).round() as usize).min(p.num_frames - 1)
        } else { 0 };

        // Extract this frame's vertex positions as f64
        let base = frame * p.num_vertices * 3;
        let verts: Vec<[f64; 3]> = (0..p.num_vertices).map(|i| {
            let off = base + i * 3;
            [p.positions[off] as f64, p.positions[off+1] as f64, p.positions[off+2] as f64]
        }).collect();

        Some((p.num_frames, p.num_vertices, p.num_faces,
              p.time_min, p.dt, p.mesh_radius, frame, verts))
    });

    struct RawBatch {
        actual_lod: [u32; 3],
        canonical_lod: [u32; 3],
        used_lod: [u32; 3],
        parity: i32,
        perm_index: usize,
        material_index: i32,
        face_indices: Vec<u32>,
        num_faces: usize,
        n_verts: usize,
        n_tris: usize,
    }

    let (num_tris, source_faces, raw_batches, all_orig, all_xform) = if let Some((_pb_nf, pb_nv, pb_faces, _pb_tmin, _pb_dt, pb_radius, frame, frame_verts)) = prebake_frame_data {
        // MERGED GPU+BATCH PATH: GPU compute → readback → group → single-pass sorted write.
        // Eliminates intermediate unsorted buffers entirely.

        // 1. FIRE GPU compute (non-blocking — flush starts the work)
        let gpu_n = {
            GPU_COMPUTE.with(|gc| {
                let mut gc = gc.borrow_mut();
                let (gl, compute) = gc.as_mut().unwrap();
                let min_px_val = if quilting_core::evaluate::get_screen_atten() {
                    quilting_core::evaluate::get_min_px_per_sub() as f32
                } else { 0.0 };
                let vp_f32: [f32; 16] = screen.as_ref().map(|s| {
                    let mut m = [0.0f32; 16];
                    for i in 0..16 { m[i] = s.vp_matrix[i] as f32; }
                    m
                }).unwrap_or([0.0; 16]);
                let (vpw, vph) = screen.as_ref().map(|s| (s.width as f32, s.height as f32)).unwrap_or((0.0, 0.0));
                compute.compute_with_texture(
                    gl, pb_faces, frame as u32, pb_nv as u32,
                    mob_f32, tess_density as f32, pb_radius as f32,
                    min_px_val, &vp_f32, vpw, vph,
                )
            })
        };

        SLICE_CACHE.with(|c| {
            let cache = c.borrow();
            let cs = cache.as_ref().unwrap();
            let nf = cs.tris.len();
            let src_faces = cs.source_face_indices.clone();

            // 2. CPU WORK while GPU runs: pre-compute orig smooth normals
            let orig_vnormals: Vec<[f32; 3]> = if !cs.normals.is_empty() {
                // glTF normals — already per-vertex
                cs.normals.clone()
            } else {
                // Compute smooth vertex normals from geometry
                let num_verts = frame_verts.len();
                let mut vn = vec![[0.0f64; 3]; num_verts];
                for face in &cs.tris {
                    let v0 = frame_verts[face[0]]; let v1 = frame_verts[face[1]]; let v2 = frame_verts[face[2]];
                    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
                    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
                    let fn_ = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
                    for &vi in face { vn[vi][0] += fn_[0]; vn[vi][1] += fn_[1]; vn[vi][2] += fn_[2]; }
                }
                vn.iter().map(|n| {
                    let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                    if len > 1e-10 { [(n[0]/len) as f32, (n[1]/len) as f32, (n[2]/len) as f32] }
                    else { [0.0, 0.0, 1.0] }
                }).collect()
            };

            // 3. GPU READBACK — 5 floats per face: [atlas_index, perm_index, nx, ny, nz]
            let gpu_class = if gpu_n > 0 {
                GPU_COMPUTE.with(|gc| {
                    let gc = gc.borrow();
                    let (gl, compute) = gc.as_ref().unwrap();
                    compute.read_back(gl, gpu_n)
                })
            } else { vec![] };

            let stride = quilting_renderer::compute::FLOATS_PER_FACE_OUTPUT;

            // 4. Accumulate deformed smooth normals from GPU face normals
            let xform_vnormals: Vec<[f32; 3]> = if !transform.is_affine() && gpu_class.len() >= nf * stride {
                let num_verts = frame_verts.len();
                let mut vn = vec![[0.0f32; 3]; num_verts];
                for (fi, face) in cs.tris.iter().enumerate() {
                    let rb = fi * stride + 2;
                    let fn_ = [gpu_class[rb], gpu_class[rb+1], gpu_class[rb+2]];
                    for &vi in face { vn[vi][0] += fn_[0]; vn[vi][1] += fn_[1]; vn[vi][2] += fn_[2]; }
                }
                let sign: f32 = -1.0;
                vn.iter().map(|n| {
                    let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                    if len > 1e-10 { [sign*n[0]/len, sign*n[1]/len, sign*n[2]/len] }
                    else { [0.0, 0.0, sign] }
                }).collect()
            } else {
                orig_vnormals.clone()
            };

            // 5. Group faces by (canonical_lod, perm_index, material) from GPU classification
            let face_mat_indices: Vec<i32> = FACE_MATERIALS.with(|fm| {
                let mats = fm.borrow();
                (0..nf).map(|fi| {
                    if fi < src_faces.len() {
                        let orig_fi = src_faces[fi];
                        if orig_fi < mats.len() { mats[orig_fi].map(|m| m as i32).unwrap_or(-1) }
                        else { -1 }
                    } else { -1 }
                }).collect()
            });

            let groups: Vec<(([u32; 3], usize, i32), Vec<usize>)> = ATLAS_KEYS.with(|ak| {
                let keys = ak.borrow();
                let mut map: FxHashMap<([u32; 3], usize, i32), Vec<usize>> = FxHashMap::default();
                for fi in 0..nf {
                    let rb = fi * stride;
                    if rb + 1 < gpu_class.len() {
                        let atlas_idx = gpu_class[rb] as usize;
                        let perm_idx = gpu_class[rb + 1] as usize;
                        let mat_idx = face_mat_indices[fi];
                        if atlas_idx < keys.len() {
                            let canonical_lod = keys[atlas_idx];
                            map.entry((canonical_lod, perm_idx, mat_idx)).or_default().push(fi);
                        }
                    }
                }
                map.into_iter().collect()
            });

            // 6. SINGLE-PASS SORTED WRITE — write flat buffers directly in batch order.
            // Reuse preallocated buffers from previous frame.
            let (mut sorted_orig, mut sorted_xform) = SORTED_BUFS.with(|sb| {
                let mut sb = sb.borrow_mut();
                let needed = nf * 52;
                if sb.0.len() < needed { sb.0.resize(needed, 0.0); }
                if sb.1.len() < needed { sb.1.resize(needed, 0.0); }
                (std::mem::take(&mut sb.0), std::mem::take(&mut sb.1))
            });

            let mut write_pos = 0usize;
            let mut raw_batches: Vec<RawBatch> = Vec::with_capacity(groups.len());
            let has_uvs = !cs.uvs.is_empty();

            for &((canonical_lod, perm_index, mat_idx), ref face_indices) in &groups {
                let batch_offset = write_pos;
                let batch_nf = face_indices.len();
                let parity = perm_sign(perm_index);

                // Look up LODs for this batch from atlas keys
                let lods = ATLAS_KEYS.with(|ak| {
                    let keys = ak.borrow();
                    let rb = face_indices[0] * stride;
                    let atlas_idx = gpu_class[rb] as usize;
                    if atlas_idx < keys.len() { keys[atlas_idx] } else { canonical_lod }
                });

                for &fi in face_indices {
                    let face = &cs.tris[fi];
                    let dst = write_pos * 52;

                    // Positions as quaternions (w=0, x,y,z)
                    for (vi, &vert_idx) in face.iter().enumerate() {
                        let v = frame_verts[vert_idx];
                        let off = dst + vi * 4;
                        sorted_orig[off]   = 0.0;
                        sorted_orig[off+1] = v[0] as f32;
                        sorted_orig[off+2] = v[1] as f32;
                        sorted_orig[off+3] = v[2] as f32;
                    }

                    // Weights = identity (w=1, x=0, y=0, z=0)
                    sorted_orig[dst + 12] = 1.0;
                    sorted_orig[dst + 16] = 1.0;
                    sorted_orig[dst + 20] = 1.0;

                    // Edge LODs + vertex LODs
                    sorted_orig[dst + 24] = lods[0] as f32;
                    sorted_orig[dst + 25] = lods[1] as f32;
                    sorted_orig[dst + 26] = lods[2] as f32;
                    sorted_orig[dst + 28] = lods[0] as f32;
                    sorted_orig[dst + 29] = lods[1] as f32;
                    sorted_orig[dst + 30] = lods[2] as f32;

                    // UVs
                    if has_uvs {
                        let uv0 = cs.uvs[face[0]]; let uv1 = cs.uvs[face[1]]; let uv2 = cs.uvs[face[2]];
                        sorted_orig[dst + 32] = uv0[0]; sorted_orig[dst + 33] = uv0[1];
                        sorted_orig[dst + 34] = uv1[0]; sorted_orig[dst + 35] = uv1[1];
                        sorted_orig[dst + 36] = uv2[0]; sorted_orig[dst + 37] = uv2[1];
                    }

                    // Orig normals
                    for vi in 0..3 {
                        let n = orig_vnormals[face[vi]];
                        let off = dst + 40 + vi * 4;
                        sorted_orig[off] = n[0]; sorted_orig[off+1] = n[1]; sorted_orig[off+2] = n[2];
                    }

                    // xform: copy everything from orig, then overwrite normals
                    sorted_xform[dst..dst+52].copy_from_slice(&sorted_orig[dst..dst+52]);

                    // Xform deformed normals
                    for vi in 0..3 {
                        let n = xform_vnormals[face[vi]];
                        let off = dst + 40 + vi * 4;
                        sorted_xform[off] = n[0]; sorted_xform[off+1] = n[1]; sorted_xform[off+2] = n[2];
                    }

                    write_pos += 1;
                }

                let (n_verts, n_tris) = ATLAS.with(|atlas_cell| {
                    let atlas = atlas_cell.borrow();
                    if let Some(atlas) = atlas.as_ref() {
                        if let Some(entry) = atlas.patches.get(&canonical_lod) {
                            return (entry.vertex_count, entry.triangle_count);
                        }
                    }
                    (0, 0)
                });

                raw_batches.push(RawBatch {
                    actual_lod: [lods[0], lods[1], lods[2]],
                    canonical_lod: [canonical_lod[0], canonical_lod[1], canonical_lod[2]],
                    used_lod: [canonical_lod[0], canonical_lod[1], canonical_lod[2]],
                    parity, perm_index, material_index: mat_idx,
                    face_indices: face_indices.iter().map(|&i| {
                        if i < src_faces.len() { src_faces[i] as u32 } else { i as u32 }
                    }).collect(),
                    num_faces: batch_nf, n_verts, n_tris,
                });
            }

            (nf, src_faces, raw_batches, sorted_orig, sorted_xform)
        })
    } else {
        web_sys::console::warn_1(&"slice_and_transform: no prebake data! Using empty result.".into());
        (0, vec![], vec![], vec![], vec![])
    };
    pm("sat:mobius-end");

    pm("sat:group-start");
    pm("sat:group-end");
    pm("sat:batch-start");
    pm("sat:batch-end");

    pm("sat:serialize-start");

    let num_batches = raw_batches.len();
    let mut batch_meta = vec![0i32; num_batches * 16];
    let mut all_face_indices: Vec<u32> = Vec::with_capacity(num_tris);

    let mut write_pos = 0usize;
    for (bi, b) in raw_batches.iter().enumerate() {
        let batch_offset = write_pos;
        write_pos += b.num_faces;

        let m = bi * 16;
        batch_meta[m]    = b.actual_lod[0] as i32;
        batch_meta[m+1]  = b.actual_lod[1] as i32;
        batch_meta[m+2]  = b.actual_lod[2] as i32;
        batch_meta[m+3]  = b.canonical_lod[0] as i32;
        batch_meta[m+4]  = b.canonical_lod[1] as i32;
        batch_meta[m+5]  = b.canonical_lod[2] as i32;
        batch_meta[m+6]  = b.used_lod[0] as i32;
        batch_meta[m+7]  = b.used_lod[1] as i32;
        batch_meta[m+8]  = b.used_lod[2] as i32;
        batch_meta[m+9]  = b.parity;
        batch_meta[m+10] = b.perm_index as i32;
        batch_meta[m+11] = b.material_index;
        batch_meta[m+12] = b.num_faces as i32;
        batch_meta[m+13] = b.n_verts as i32;
        batch_meta[m+14] = b.n_tris as i32;
        batch_meta[m+15] = batch_offset as i32;

        all_face_indices.extend_from_slice(&b.face_indices);
    }

    // Allocate + copy. new_with_length + copy_from is faster than
    // Float32Array::from() which goes through wasm-bindgen's JS shim.
    // Worker transfers these (zero-copy to main), then they're gone.
    let js_orig = js_sys::Float32Array::new_with_length(all_orig.len() as u32);
    js_orig.copy_from(&all_orig);
    let js_xform = js_sys::Float32Array::new_with_length(all_xform.len() as u32);
    js_xform.copy_from(&all_xform);

    // Recycle buffers into SORTED_BUFS for next frame (avoids re-allocation)
    SORTED_BUFS.with(|sb| {
        let mut sb = sb.borrow_mut();
        sb.0 = all_orig;
        sb.1 = all_xform;
    });
    let js_meta = js_sys::Int32Array::new_with_length(batch_meta.len() as u32);
    js_meta.copy_from(&batch_meta);
    let js_face_idx = js_sys::Uint32Array::new_with_length(all_face_indices.len() as u32);
    js_face_idx.copy_from(&all_face_indices);

    pm("sat:serialize-end");

    perf.measure_with_start_mark_and_end_mark("Slice + Cache", "sat:fn-start", "sat:mobius-start").ok();
    perf.measure_with_start_mark_and_end_mark("Möbius + LOD", "sat:mobius-start", "sat:mobius-end").ok();
    perf.measure_with_start_mark_and_end_mark("Materials + Grouping", "sat:group-start", "sat:group-end").ok();
    perf.measure_with_start_mark_and_end_mark("Batch Assembly", "sat:batch-start", "sat:batch-end").ok();
    perf.measure_with_start_mark_and_end_mark("Serialize to JS", "sat:serialize-start", "sat:serialize-end").ok();
    perf.measure_with_start_mark_and_end_mark("Total (WASM)", "sat:fn-start", "sat:serialize-end").ok();

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"total_faces".into(), &JsValue::from(num_tris as u32)).ok();
    js_sys::Reflect::set(&result, &"num_batches".into(), &JsValue::from(num_batches as u32)).ok();
    js_sys::Reflect::set(&result, &"all_orig".into(), &js_orig).ok();
    js_sys::Reflect::set(&result, &"all_xform".into(), &js_xform).ok();
    js_sys::Reflect::set(&result, &"batch_meta".into(), &js_meta).ok();
    js_sys::Reflect::set(&result, &"face_indices".into(), &js_face_idx).ok();

    result.into()
}
