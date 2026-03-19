use wasm_bindgen::prelude::*;
use quilting_core::atlas::TessellationAtlas;
use quilting_core::sampling::{tri_patch, PatchConfig};
use quilting_core::delaunay::triangulate_2d;
use quilting_core::mesh::TessellationMesh;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Generate a single tessellated triangle patch.
/// Returns { positions: Float64Array, triangles: Uint32Array, vertex_count, triangle_count }
#[wasm_bindgen]
pub fn generate_patch(res_a: f64, res_b: f64, res_c: f64, seed: u64) -> JsValue {
    let config = PatchConfig {
        k_candidates: 30,
        seed,
    };
    let sample = tri_patch([res_a, res_b, res_c], &config);

    if sample.positions.len() < 3 {
        return serde_wasm_bindgen::to_value(&PatchResult::empty()).unwrap();
    }

    let tri = triangulate_2d(&sample.positions);
    let mesh = TessellationMesh::from_2d(tri.positions, tri.triangles);

    let result = PatchResult::from_mesh(&mesh);
    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Build a full tessellation atlas for the given LOD levels.
/// Returns serialized atlas as Uint8Array.
#[wasm_bindgen]
pub fn build_atlas(lod_levels: &[u32], seed: u64) -> Vec<u8> {
    let config = PatchConfig {
        k_candidates: 30,
        seed,
    };
    let atlas = TessellationAtlas::build(lod_levels, &config);
    atlas.to_bytes()
}

/// Look up a patch from a serialized atlas.
/// Returns { positions, triangles, vertex_count, triangle_count }
#[wasm_bindgen]
pub fn get_patch_from_atlas(atlas_bytes: &[u8], res_a: u32, res_b: u32, res_c: u32) -> JsValue {
    let atlas = match TessellationAtlas::from_bytes(atlas_bytes) {
        Ok(a) => a,
        Err(_) => return serde_wasm_bindgen::to_value(&PatchResult::empty()).unwrap(),
    };

    match atlas.get_patch([res_a, res_b, res_c]) {
        Some(mesh) => {
            let result = PatchResult::from_mesh(&mesh);
            serde_wasm_bindgen::to_value(&result).unwrap()
        }
        None => serde_wasm_bindgen::to_value(&PatchResult::empty()).unwrap(),
    }
}

#[derive(serde::Serialize)]
struct PatchResult {
    positions: Vec<f64>,    // flat [x0, y0, x1, y1, ...]
    triangles: Vec<u32>,    // flat [i0, j0, k0, i1, j1, k1, ...]
    vertex_count: usize,
    triangle_count: usize,
}

impl PatchResult {
    fn empty() -> Self {
        Self {
            positions: vec![],
            triangles: vec![],
            vertex_count: 0,
            triangle_count: 0,
        }
    }

    fn from_mesh(mesh: &TessellationMesh) -> Self {
        let positions: Vec<f64> = mesh.positions.iter()
            .flat_map(|p| [p[0], p[1]])
            .collect();
        let triangles: Vec<u32> = mesh.triangles.iter()
            .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
            .collect();
        Self {
            vertex_count: mesh.positions.len(),
            triangle_count: mesh.triangles.len(),
            positions,
            triangles,
        }
    }
}
