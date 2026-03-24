//! Main-thread rendering module for the production hyperscope.
//!
//! All exports are prefixed with `mr_` in JS to distinguish from
//! worker-side exports in lib.rs.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use tracing::info;

use glow::HasContext;
use quilting_renderer::buffer::{
    MeshBuffers, TessBuffers, PbrParams, EnvironmentMaps,
};
use quilting_renderer::pass::{Camera, RenderBatch, RenderMode};
use quilting_renderer::Renderer;
use quilting_renderer::texture::TextureCache;
use quilting_core::batch;

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
    }
}

struct MainState {
    renderer: Renderer,
    texture_cache: TextureCache,
    env_maps: EnvironmentMaps,
    batches: Vec<GpuBatch>,
    cached_instances: Vec<f32>,
    face_materials: Vec<usize>,
    materials: Vec<PbrParams>,
    num_faces: usize,
    render_mode: RenderMode,
    mobius: [f32; 16],
}

struct GpuBatch {
    mesh: MeshBuffers,
    perm_parity: f32,
    perm_index: i32,
    material_index: usize,
    lod: [u32; 3],
}

thread_local! {
    static STATE: RefCell<Option<MainState>> = RefCell::new(None);
    static TESS_CACHE: RefCell<std::collections::HashMap<String, TessBuffers>> =
        RefCell::new(std::collections::HashMap::new());
}

const IDENTITY_MOBIUS: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
];

#[wasm_bindgen(js_name = "mr_init")]
pub fn mr_init(canvas_id: &str) -> bool {
    let document = web_sys::window().and_then(|w| w.document());
    let document = match document { Some(d) => d, None => return false };

    let canvas = match document.get_element_by_id(canvas_id) {
        Some(c) => match c.dyn_into::<web_sys::HtmlCanvasElement>() {
            Ok(c) => c, Err(_) => return false,
        },
        None => return false,
    };

    let gl_ctx = canvas.get_context_with_context_options(
        "webgl2",
        &js_sys::JSON::parse(r#"{"antialias":true}"#).unwrap(),
    );
    let gl_ctx = match gl_ctx {
        Ok(Some(ctx)) => match ctx.dyn_into::<web_sys::WebGl2RenderingContext>() {
            Ok(c) => c, Err(_) => return false,
        },
        _ => return false,
    };

    let gl = glow::Context::from_webgl2_context(gl_ctx);
    let renderer = match Renderer::new(gl) {
        Ok(r) => r,
        Err(e) => {
            web_sys::console::error_1(&format!("Renderer init: {e}").into());
            return false;
        }
    };
    let texture_cache = match TextureCache::new(renderer.gl()) {
        Ok(tc) => tc,
        Err(e) => {
            web_sys::console::error_1(&format!("TextureCache: {e}").into());
            return false;
        }
    };

    STATE.with(|s| {
        *s.borrow_mut() = Some(MainState {
            renderer, texture_cache,
            env_maps: EnvironmentMaps::default(),
            batches: Vec::new(),
            cached_instances: Vec::new(),
            face_materials: Vec::new(),
            materials: Vec::new(),
            num_faces: 0,
            render_mode: RenderMode::Matcap,
            mobius: IDENTITY_MOBIUS,
        });
    });
    info!("Renderer initialized on canvas '{}'", canvas_id);
    true
}

#[wasm_bindgen(js_name = "mr_resize")]
pub fn mr_resize(width: i32, height: i32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() { st.renderer.resize(width, height); }
    });
}

#[wasm_bindgen(js_name = "mr_setRenderMode")]
pub fn mr_set_render_mode(mode: &str) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.render_mode = match mode {
                "pbr" => RenderMode::Pbr, "matcap" => RenderMode::Matcap,
                "wire" => RenderMode::Wire, "normals" => RenderMode::Normals,
                "both" => RenderMode::Both, "lod" => RenderMode::Lod,
                _ => RenderMode::Pbr,
            };
        }
    });
}

#[wasm_bindgen(js_name = "mr_setMobius")]
pub fn mr_set_mobius(mobius: &[f32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            for (i, &v) in mobius.iter().take(16).enumerate() { st.mobius[i] = v; }
        }
    });
}

#[wasm_bindgen(js_name = "mr_setInstanceData")]
pub fn mr_set_instance_data(instances: &[f32], num_faces: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.cached_instances = instances.to_vec();
            st.num_faces = num_faces as usize;
        }
    });
}

#[wasm_bindgen(js_name = "mr_setFaceMaterials")]
pub fn mr_set_face_materials(materials: &[i32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.face_materials = materials.iter().map(|&m| if m >= 0 { m as usize } else { 0 }).collect();
        }
    });
}

#[wasm_bindgen(js_name = "mr_uploadTessPatch")]
pub fn mr_upload_tess_patch(key: &str, bary: &[f32], tri_idx: &[u32], line_idx: &[u32]) {
    STATE.with(|s| {
        let state = s.borrow();
        let state = match state.as_ref() { Some(s) => s, None => return };
        if let Ok(tess) = TessBuffers::new(state.renderer.gl(), bary, tri_idx, line_idx) {
            TESS_CACHE.with(|tc| { tc.borrow_mut().insert(key.to_string(), tess); });
        }
    });
}

/// Upload skinning texture (joint indices + weights) to main-thread GL.
/// `joint_indices`: flat f32 array [j0,j1,j2,j3] × num_vertices
/// `joint_weights`: flat f32 array [w0,w1,w2,w3] × num_vertices
#[wasm_bindgen(js_name = "mr_uploadSkinningTexture")]
pub fn mr_upload_skinning_texture(joint_indices: &[f32], joint_weights: &[f32], num_vertices: u32) {
    STATE.with(|s| {
        let state = s.borrow();
        let state = match state.as_ref() { Some(s) => s, None => return };
        let gl = state.renderer.gl();
        let nv = num_vertices as usize;
        // Convert flat f32 arrays to the format SkinningTexture expects
        let ji: Vec<[u16; 4]> = (0..nv).map(|i| {
            [joint_indices[i*4] as u16, joint_indices[i*4+1] as u16,
             joint_indices[i*4+2] as u16, joint_indices[i*4+3] as u16]
        }).collect();
        let jw: Vec<[f32; 4]> = (0..nv).map(|i| {
            [joint_weights[i*4], joint_weights[i*4+1], joint_weights[i*4+2], joint_weights[i*4+3]]
        }).collect();
        if let Ok(tex) = quilting_renderer::buffer::SkinningTexture::new(gl, &ji, &jw) {
            // Bind to texture unit 15 (matches prototype)
            tex.bind(gl, 15);
            info!("Skinning texture uploaded: {} vertices", nv);
        }
    });
}

/// Upload morph target delta texture to main-thread GL.
/// `deltas`: flat f32 array [dx,dy,dz] × num_vertices × num_targets
#[wasm_bindgen(js_name = "mr_uploadMorphTexture")]
pub fn mr_upload_morph_texture(deltas: &[f32], num_vertices: u32, num_targets: u32) {
    STATE.with(|s| {
        let state = s.borrow();
        let state = match state.as_ref() { Some(s) => s, None => return };
        let gl = state.renderer.gl();
        let nv = num_vertices as usize;
        let nt = num_targets as usize;
        // Pack into RGBA32F texture: width=num_vertices, height=num_targets
        let mut rgba = vec![0.0f32; nv * nt * 4];
        for t in 0..nt {
            for v in 0..nv {
                let src = (t * nv + v) * 3;
                let dst = (t * nv + v) * 4;
                if src + 2 < deltas.len() {
                    rgba[dst] = deltas[src];
                    rgba[dst+1] = deltas[src+1];
                    rgba[dst+2] = deltas[src+2];
                }
            }
        }
        unsafe {
            let tex = match gl.create_texture() {
                Ok(t) => t,
                Err(_) => return,
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                nv as i32, nt as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            // Bind to texture unit 14 (matches prototype)
            gl.active_texture(glow::TEXTURE0 + 14);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            info!("Morph texture uploaded: {} vertices × {} targets", nv, nt);
        }
    });
}

#[wasm_bindgen(js_name = "mr_buildBatches")]
pub fn mr_build_batches(face_lods: &[f32]) {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return };
        let gl = state.renderer.gl();

        for b in state.batches.drain(..) { b.mesh.destroy(gl); }

        let logical = batch::group_into_batches(
            face_lods, &state.cached_instances, &state.face_materials, state.num_faces,
        );

        let mut built = 0;
        let mut missing = 0;
        TESS_CACHE.with(|tc| {
            let tc = tc.borrow();
            for lb in &logical {
                let key = lb.tess_key.as_string();
                let tess = match tc.get(&key) {
                    Some(t) => t,
                    None => { missing += 1; continue; }
                };
                let mesh = match MeshBuffers::new(gl, tess, &lb.instance_data, lb.face_indices.len() as i32) {
                    Ok(m) => m, Err(_) => continue,
                };
                state.batches.push(GpuBatch {
                    mesh, perm_parity: lb.parity, perm_index: lb.perm_index as i32,
                    material_index: lb.material_index, lod: lb.lod,
                });
                built += 1;
            }
        });
        info!("Built {} GPU batches ({} logical, {} missing tess)", built, logical.len(), missing);
    });
}

#[wasm_bindgen(js_name = "mr_uploadAnimationPose")]
pub fn mr_upload_animation_pose(matrices: &[f32], morph_weights: &[f32], skin_tex_w: i32) {
    STATE.with(|s| {
        if let Some(ref st) = *s.borrow() {
            st.renderer.joint_ubo().upload(st.renderer.gl(), matrices, morph_weights, skin_tex_w);
        }
    });
}

#[wasm_bindgen(js_name = "mr_render")]
pub fn mr_render(mvp: &[f32], mv: &[f32], camera_pos: &[f32]) {
    STATE.with(|s| {
        let state = s.borrow();
        let state = match state.as_ref() { Some(s) => s, None => return };
        if state.batches.is_empty() { return; }

        let camera = Camera {
            mvp: mvp[..16].try_into().unwrap_or([0.0; 16]),
            mv: mv[..16].try_into().unwrap_or([0.0; 16]),
            mobius: state.mobius,
            camera_pos: [
                camera_pos.get(0).copied().unwrap_or(0.0),
                camera_pos.get(1).copied().unwrap_or(0.0),
                camera_pos.get(2).copied().unwrap_or(0.0),
            ],
        };

        let render_batches: Vec<RenderBatch> = state.batches.iter().map(|b| {
            RenderBatch {
                mesh: &b.mesh, perm_parity: b.perm_parity, perm_index: b.perm_index,
                wire_color: lod_color(&b.lod), material_index: b.material_index,
            }
        }).collect();

        let gl = state.renderer.gl();
        state.renderer.begin_frame();
        state.renderer.joint_ubo().bind(gl);
        state.renderer.matcap_ubo().upload(gl, false); // no matcap texture → use heatmap
        state.renderer.matcap_ubo().bind(gl);
        state.renderer.render(state.render_mode, &camera, &render_batches);
        state.renderer.end_frame();
    });
}

fn lod_color(lod: &[u32; 3]) -> [f32; 3] {
    let max_lod = *lod.iter().max().unwrap_or(&1) as f32;
    let t = (max_lod.log2() / 8.0).clamp(0.0, 1.0);
    if t < 0.5 { let s = t * 2.0; [0.0, s, 1.0 - s] }
    else { let s = (t - 0.5) * 2.0; [s, 1.0 - s, 0.0] }
}
