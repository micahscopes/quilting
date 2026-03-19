use wasm_bindgen::prelude::*;
use quilting_core::delaunay::triangulate_2d;
use quilting_core::evaluate::compute_instances;
use quilting_core::permutation::{canonical_form, remap_position};
use quilting_core::quaternion::{Quat, Mobius};
use quilting_core::sampling::{tri_patch, PatchConfig};
use quilting_core::shapes;
use quilting_core::triangle;
use std::collections::HashMap;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get a built-in shape. Returns flat arrays [positions, faces].
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
        positions,
        faces: indices,
        num_verts: verts.len(),
        num_faces: faces.len(),
    }).unwrap()
}

#[derive(serde::Serialize)]
struct ShapeData {
    positions: Vec<f64>,
    faces: Vec<u32>,
    num_verts: usize,
    num_faces: usize,
}

/// Compute transform instances for a mesh.
/// positions: flat [x0,y0,z0, x1,y1,z1, ...]
/// faces: flat [a0,b0,c0, a1,b1,c1, ...]
/// transform: "identity" | "sphere_reflection"
/// params: [cx, cy, cz, r] for sphere_reflection
///
/// Returns batched data grouped by LOD triple.
#[wasm_bindgen]
pub fn compute_mesh_batches(
    positions: &[f64],
    faces: &[u32],
    transform_type: &str,
    params: &[f64],
    override_res: u32,
) -> JsValue {
    let verts: Vec<[f64; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let tris: Vec<[usize; 3]> = faces.chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    let transform = match transform_type {
        "sphere_reflection" if params.len() >= 4 => {
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

    // Group by (canonical sorted LOD, permutation index) so each batch
    // gets the right tessellation with edges mapped correctly.
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

    let config = PatchConfig { k_candidates: 30, seed: 42 };

    // Cache tessellations by canonical key (shared across permutations)
    let mut tess_cache: HashMap<[u32; 3], (Vec<[f64; 3]>, Vec<[f64; 2]>, Vec<[usize; 3]>)> = HashMap::new();
    let mut batches = Vec::new();

    for (&(canonical_lod, perm_index), face_indices) in &groups {
        // Get or generate the canonical tessellation
        let (bary_data, pos_data, tri_data) = tess_cache
            .entry(canonical_lod)
            .or_insert_with(|| {
                let tess = tri_patch(
                    [canonical_lod[0] as f64, canonical_lod[1] as f64, canonical_lod[2] as f64],
                    &config,
                );
                let tri_result = triangulate_2d(&tess.positions);
                (tess.bary, tri_result.positions, tri_result.triangles)
            });

        // Apply the permutation to remap bary coords so edges match the face's LOD order
        let remapped_bary: Vec<f64> = if perm_index == 0 {
            // Identity — no remapping needed
            bary_data.iter().flat_map(|b| [b[0], b[1], b[2]]).collect()
        } else {
            // Remap each position through the permutation, then recompute bary
            pos_data.iter().map(|p| {
                let remapped = remap_position(perm_index, *p);
                triangle::cartesian_to_bary(remapped[0], remapped[1])
            }).flat_map(|b| [b[0], b[1], b[2]]).collect()
        };

        let tess_tris: Vec<u32> = tri_data.iter()
            .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();

        let orig_data: Vec<f32> = face_indices.iter()
            .flat_map(|&fi| instances_orig[fi].to_f32_array()).collect();
        let xform_data: Vec<f32> = face_indices.iter()
            .flat_map(|&fi| instances_xform[fi].to_f32_array()).collect();

        // The actual LOD for this batch (unsorted, matches the face's edge order)
        let actual_lod = if override_res > 0 {
            [override_res, override_res, override_res]
        } else {
            instances_xform[face_indices[0]].edge_lods
        };

        batches.push(BatchData {
            lod: actual_lod,
            instances_orig: orig_data,
            instances_xform: xform_data,
            tess_bary: remapped_bary,
            tess_triangles: tess_tris,
            num_faces: face_indices.len(),
            verts_per_face: bary_data.len(),
            tris_per_face: tri_data.len(),
        });
    }

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
