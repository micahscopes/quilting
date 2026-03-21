use wasm_bindgen::prelude::*;
use quilting_core::atlas::{TessellationAtlas, BuildMode};
use quilting_core::evaluate::{compute_instances, compute_instances_no_lod, ScreenInfo};
use quilting_core::mesh::TessellationMesh;
use quilting_core::permutation::{canonical_form, remap_position, perm_sign};
use quilting_core::quaternion::{Quat, Mobius};
use quilting_core::sampling::PatchConfig;
use quilting_core::shapes;
use quilting_core::triangle;
use quilting_mesh::HalfEdgeMesh;
use std::cell::RefCell;
use std::collections::HashMap;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

thread_local! {
    static ATLAS: RefCell<Option<TessellationAtlas>> = RefCell::new(None);
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

/// Generate a single tessellation patch and store it in the atlas.
/// Used for progressive refinement — generates missing LODs on demand.
#[wasm_bindgen]
pub fn generate_and_store_patch(res_a: u32, res_b: u32, res_c: u32) {
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let res = [res_a as f64, res_b as f64, res_c as f64];
    let sample = quilting_core::sampling::tri_patch(res, &config);
    if sample.positions.len() < 3 { return; }
    let tri = quilting_core::delaunay::triangulate_2d_clipped(&sample.positions);

    ATLAS.with(|atlas_cell| {
        let mut atlas_opt = atlas_cell.borrow_mut();
        if let Some(atlas) = atlas_opt.as_mut() {
            let mut key = [res_a, res_b, res_c];
            key.sort();
            if atlas.patches.contains_key(&key) { return; }

            let base_vertex = atlas.positions.len();
            let base_triangle = atlas.triangles.len();
            atlas.positions.extend_from_slice(&tri.positions);
            for t in &tri.triangles {
                atlas.triangles.push([t[0] + base_vertex, t[1] + base_vertex, t[2] + base_vertex]);
            }
            atlas.patches.insert(key, quilting_core::atlas::PatchEntry {
                base_vertex,
                vertex_count: tri.positions.len(),
                base_triangle,
                triangle_count: tri.triangles.len(),
            });
        }
    });
}

/// Generate a single tessellation patch for a given LOD triple.
/// Returns { bary, triangles, n_verts, n_tris }
#[wasm_bindgen]
pub fn generate_patch(res_a: u32, res_b: u32, res_c: u32) -> JsValue {
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let sample = quilting_core::sampling::tri_patch(
        [res_a as f64, res_b as f64, res_c as f64], &config,
    );
    if sample.positions.len() < 3 {
        return serde_wasm_bindgen::to_value(&PatchData {
            bary: vec![], triangles: vec![], n_verts: 0, n_tris: 0,
        }).unwrap();
    }
    let tri = quilting_core::delaunay::triangulate_2d_clipped(&sample.positions);

    let bary: Vec<f64> = sample.bary.iter().flat_map(|b| [b[0], b[1], b[2]]).collect();
    let triangles: Vec<u32> = tri.triangles.iter()
        .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

    serde_wasm_bindgen::to_value(&PatchData {
        bary,
        triangles,
        n_verts: sample.positions.len(),
        n_tris: tri.triangles.len(),
    }).unwrap()
}

/// Store a patch into the atlas from worker results.
/// bary: flat [u0,v0,w0, u1,v1,w1, ...]
/// triangles: flat [i0,j0,k0, ...]
#[wasm_bindgen]
pub fn store_patch(res_a: u32, res_b: u32, res_c: u32, positions_2d: &[f64], triangles: &[u32]) {
    let pos: Vec<[f64; 2]> = positions_2d.chunks(2).map(|c| [c[0], c[1]]).collect();
    let tris: Vec<[usize; 3]> = triangles.chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();

    ATLAS.with(|atlas_cell| {
        let mut atlas_opt = atlas_cell.borrow_mut();
        if atlas_opt.is_none() {
            *atlas_opt = Some(TessellationAtlas {
                positions: Vec::new(),
                triangles: Vec::new(),
                patches: HashMap::new(),
                lod_levels: Vec::new(),
            });
        }
        let atlas = atlas_opt.as_mut().unwrap();
        let key = {
            let mut k = [res_a, res_b, res_c];
            k.sort();
            k
        };

        let base_vertex = atlas.positions.len();
        let base_triangle = atlas.triangles.len();
        atlas.positions.extend_from_slice(&pos);
        for t in &tris {
            atlas.triangles.push([t[0] + base_vertex, t[1] + base_vertex, t[2] + base_vertex]);
        }
        atlas.patches.insert(key, quilting_core::atlas::PatchEntry {
            base_vertex,
            vertex_count: pos.len(),
            base_triangle,
            triangle_count: tris.len(),
        });
    });
}

#[derive(serde::Serialize)]
struct PatchData {
    bary: Vec<f64>,
    triangles: Vec<u32>,
    n_verts: usize,
    n_tris: usize,
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
    let mut groups: HashMap<([u32; 3], usize), Vec<usize>> = HashMap::new();
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
                    let config = PatchConfig { k_candidates: 30, seed: 42 };
                    let sample = quilting_core::sampling::tri_patch([1.0, 1.0, 1.0], &config);
                    let tri = quilting_core::delaunay::triangulate_2d_clipped(&sample.positions);
                    (quilting_core::mesh::TessellationMesh::from_2d(tri.positions, tri.triangles), [1, 1, 1])
                })
            };

            let is_fallback = used_lod != canonical_lod;
            let parity = perm_sign(perm_index);

            let actual_lod = if override_res > 0 {
                [override_res, override_res, override_res]
            } else {
                instances_xform[face_indices[0]].edge_lods
            };

            // Tess cache key includes perm_index — each permutation gets its own
            // pre-remapped bary buffer for exact edge stitching.
            let tess_key = format!("{},{},{}/{}", canonical_lod[0], canonical_lod[1], canonical_lod[2], perm_index);

            let already_sent = SENT_TESS.with(|s| s.borrow().contains(&tess_key));

            let (bary_data, tess_tris, n_verts, n_tris) = if already_sent {
                (vec![], vec![], mesh.positions.len(), mesh.triangles.len())
            } else {
                // CPU-side permutation remapping ensures shared edge vertices
                // have bit-identical bary coords across adjacent faces.
                let bary: Vec<f64> = mesh.positions.iter().map(|p| {
                    let remapped = if perm_index == 0 { *p } else { remap_position(perm_index, *p) };
                    let mut b = triangle::cartesian_to_bary(remapped[0], remapped[1]);
                    // Snap near-zero bary to exact 0.0 — prevents epsilon * infinity gaps at Möbius poles
                    for c in &mut b { if c.abs() < 1e-10 { *c = 0.0; } }
                    let sum = b[0] + b[1] + b[2];
                    if sum > 0.0 { b[0] /= sum; b[1] /= sum; b[2] /= sum; }
                    b
                }).flat_map(|b| [b[0], b[1], b[2]]).collect();

                let tris: Vec<u32> = mesh.triangles.iter()
                    .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

                let nv = bary.len() / 3;
                let nt = tris.len() / 3;
                SENT_TESS.with(|s| s.borrow_mut().insert(tess_key));
                (bary, tris, nv, nt)
            };

            let orig_data: Vec<f32> = face_indices.iter()
                .flat_map(|&fi| instances_orig[fi].to_f32_array()).collect();
            let xform_data: Vec<f32> = face_indices.iter()
                .flat_map(|&fi| instances_xform[fi].to_f32_array()).collect();

            batches.push(BatchData {
                lod: actual_lod,
                wanted_lod: [canonical_lod[0], canonical_lod[1], canonical_lod[2]],
                used_lod: [used_lod[0], used_lod[1], used_lod[2]],
                is_fallback,
                perm_parity: parity,
                perm_index,
                instances_orig: orig_data,
                instances_xform: xform_data,
                tess_bary: bary_data,
                tess_triangles: tess_tris,
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
    instances_orig: Vec<f32>,
    instances_xform: Vec<f32>,
    tess_bary: Vec<f64>,
    tess_triangles: Vec<u32>,
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

/// Compile the production vertex shader (WGSL -> GLSL ES 300).
/// Returns the GLSL source string for use with WebGL2.
#[wasm_bindgen]
pub fn compile_vertex_shader() -> String {
    match quilting_shaders::compile_vertex_glsl() {
        Ok(glsl) => glsl,
        Err(e) => format!("// ERROR: {}", e),
    }
}

/// Compile a production fragment shader (WGSL -> GLSL ES 300).
/// mode: "matcap", "wire", or "normals"
/// Returns the GLSL source string for use with WebGL2.
#[wasm_bindgen]
pub fn compile_fragment_shader(mode: &str) -> String {
    match quilting_shaders::compile_fragment_glsl(mode) {
        Ok(glsl) => glsl,
        Err(e) => format!("// ERROR: {}", e),
    }
}

// --- Spacetime slicing ---

thread_local! {
    static HYPER_MESH: RefCell<Option<quilting_spacetime::HyperMesh>> = RefCell::new(None);
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

    // Loop trajectories for periodic animations (cube, sphere, torus).
    // Don't loop the horse — it's not a perfect cycle.
    if name != "galloping_horse" {
        for traj in &mut mesh.trajectories {
            traj.loop_pad(1);
        }
    }
    let info = HyperMeshInfo {
        time_min,
        time_max,
        num_vertices: mesh.num_vertices,
        num_faces: mesh.faces.len(),
    };

    HYPER_MESH.with(|hm| *hm.borrow_mut() = Some(mesh));

    serde_wasm_bindgen::to_value(&info).unwrap()
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
