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

    let n_patches = atlas.patches.len();
    ATLAS.with(|a| *a.borrow_mut() = Some(atlas));

    elapsed
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
            // Look up from atlas
            let mesh_opt = atlas_ref.as_ref()
                .and_then(|atlas| atlas.get_patch(canonical_lod));

            let (bary_data, tess_tris) = if let Some(mesh) = mesh_opt {
                // Remap positions through the permutation
                let bary: Vec<f64> = mesh.positions.iter().map(|p| {
                    let remapped = if perm_index == 0 { *p } else { remap_position(perm_index, *p) };
                    triangle::cartesian_to_bary(remapped[0], remapped[1])
                }).flat_map(|b| [b[0], b[1], b[2]]).collect();

                let tris: Vec<u32> = mesh.triangles.iter()
                    .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

                (bary, tris)
            } else {
                // Not in atlas — generate on the fly (shouldn't happen if atlas is built)
                let config = PatchConfig { k_candidates: 30, seed: 42 };
                let sample = quilting_core::sampling::tri_patch(
                    [canonical_lod[0] as f64, canonical_lod[1] as f64, canonical_lod[2] as f64],
                    &config,
                );
                let tri_result = quilting_core::delaunay::triangulate_2d(&sample.positions);

                let bary: Vec<f64> = sample.bary.iter().map(|b| {
                    if perm_index == 0 { *b } else {
                        let p = triangle::bary_to_cartesian(*b);
                        let r = remap_position(perm_index, p);
                        triangle::cartesian_to_bary(r[0], r[1])
                    }
                }).flat_map(|b| [b[0], b[1], b[2]]).collect();

                let tris: Vec<u32> = tri_result.triangles.iter()
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
