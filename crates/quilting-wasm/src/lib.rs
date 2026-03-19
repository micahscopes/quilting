use wasm_bindgen::prelude::*;
use quilting_core::atlas::{TessellationAtlas, BuildMode};
use quilting_core::evaluate::compute_instances;
use quilting_core::mesh::TessellationMesh;
use quilting_core::permutation::{canonical_form, remap_position};
use quilting_core::quaternion::{Quat, Mobius};
use quilting_core::sampling::PatchConfig;
use quilting_core::shapes;
use quilting_core::triangle;
use std::cell::RefCell;
use std::collections::HashMap;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

thread_local! {
    static ATLAS: RefCell<Option<TessellationAtlas>> = RefCell::new(None);
}

/// Build the tessellation atlas client-side. Call once at init.
/// max_lod_exp: build LODs from 2^0 to 2^max_lod_exp (e.g., 8 → up to 256)
/// mode: "direct" or "hierarchical"
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
) -> JsValue {
    let verts: Vec<[f64; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]]).collect();
    let tris: Vec<[usize; 3]> = faces.chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();

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

    let instances_orig = compute_instances(&verts, &tris, &Mobius::identity());
    let instances_xform = compute_instances(&verts, &tris, &transform);

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

            let (bary_data, tess_tris) = {
                let bary: Vec<f64> = mesh.positions.iter().map(|p| {
                    let remapped = if perm_index == 0 { *p } else { remap_position(perm_index, *p) };
                    triangle::cartesian_to_bary(remapped[0], remapped[1])
                }).flat_map(|b| [b[0], b[1], b[2]]).collect();

                let tris: Vec<u32> = mesh.triangles.iter()
                    .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

                (bary, tris)
            };

            let n_verts = bary_data.len() / 3;
            let n_tris = tess_tris.len() / 3;

            let orig_data: Vec<f32> = face_indices.iter()
                .flat_map(|&fi| instances_orig[fi].to_f32_array()).collect();
            let xform_data: Vec<f32> = face_indices.iter()
                .flat_map(|&fi| instances_xform[fi].to_f32_array()).collect();

            let actual_lod = if override_res > 0 {
                [override_res, override_res, override_res]
            } else {
                instances_xform[face_indices[0]].edge_lods
            };

            batches.push(BatchData {
                lod: actual_lod,
                wanted_lod: [canonical_lod[0], canonical_lod[1], canonical_lod[2]],
                used_lod: [used_lod[0], used_lod[1], used_lod[2]],
                is_fallback,
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

    serde_wasm_bindgen::to_value(&MeshBatches {
        batches,
        total_faces: tris.len(),
        num_batches: groups.len(),
    }).unwrap()
}

#[derive(serde::Serialize)]
struct BatchData {
    lod: [u32; 3],
    wanted_lod: [u32; 3],
    used_lod: [u32; 3],
    is_fallback: bool,
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
}
