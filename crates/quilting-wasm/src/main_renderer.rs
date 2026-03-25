//! Main-thread rendering module for the production hyperscope.
//!
//! All exports are prefixed with `mr_` in JS to distinguish from
//! worker-side exports in lib.rs.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use tracing::{info, debug};

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
    // Screen-space refraction: framebuffer copy for transmission
    scene_color_fbo: Option<glow::Framebuffer>,
    scene_color_tex: Option<glow::Texture>,
    scene_color_size: (i32, i32),
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
            render_mode: RenderMode::Pbr,
            mobius: IDENTITY_MOBIUS,
            scene_color_fbo: None,
            scene_color_tex: None,
            scene_color_size: (0, 0),
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
/// Set PBR material parameters from JS material objects.
/// Each material is a flat f32 array packed as:
/// [base_r, base_g, base_b, base_a, metallic, roughness, normal_scale,
///  occlusion_strength, alpha_cutoff, alpha_mode, unlit,
///  emissive_r, emissive_g, emissive_b,
///  has_base_color_tex, has_mr_tex, has_normal_tex, has_emissive_tex, has_occlusion_tex,
///  sheen_r, sheen_g, sheen_b, has_sheen, sheen_roughness,
///  specular_r, specular_g, specular_b, has_specular,
///  normal_uv_scale_x, normal_uv_scale_y, normal_uv_offset_x, normal_uv_offset_y, normal_uv_rotation,
///  base_uv_scale_x, base_uv_scale_y, base_uv_rotation]
/// 36 floats per material.
#[wasm_bindgen(js_name = "mr_setMaterials")]
pub fn mr_set_materials(data: &[f32], num_materials: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.materials.clear();
            let stride = 48;
            for i in 0..num_materials as usize {
                let o = i * stride;
                if o + stride > data.len() { break; }
                let d = &data[o..o + stride];
                st.materials.push(PbrParams {
                    base_color: [d[0], d[1], d[2], d[3]],
                    metallic: d[4],
                    roughness: d[5],
                    normal_scale: d[6],
                    occlusion_strength: d[7],
                    alpha_cutoff: d[8],
                    alpha_mode: d[9],
                    unlit: d[10] > 0.5,
                    emissive_factor: [d[11], d[12], d[13]],
                    has_base_color_tex: d[14] > 0.5,
                    has_metallic_roughness_tex: d[15] > 0.5,
                    has_normal_tex: d[16] > 0.5,
                    has_emissive_tex: d[17] > 0.5,
                    has_occlusion_tex: d[18] > 0.5,
                    sheen_color: [d[19], d[20], d[21]],
                    has_sheen: d[22] > 0.5,
                    sheen_roughness: d[23],
                    specular_color: [d[24], d[25], d[26]],
                    has_specular: d[27] > 0.5,
                    normal_uv_scale: [d[28], d[29]],
                    normal_uv_offset: [d[30], d[31]],
                    normal_uv_rotation: d[32],
                    base_uv_scale: [d[33], d[34]],
                    base_uv_rotation: d[35],
                    base_color_tex_idx: d[36] as i32,
                    metallic_roughness_tex_idx: d[37] as i32,
                    normal_tex_idx: d[38] as i32,
                    emissive_tex_idx: d[39] as i32,
                    occlusion_tex_idx: d[40] as i32,
                    ior: d[41],
                    transmission_factor: d[42],
                    thickness_factor: d[43],
                    attenuation_color: [d[44], d[45], d[46]],
                    attenuation_distance: d[47],
                    ..PbrParams::default()
                });
            }
            info!("Set {} PBR materials", st.materials.len());
        }
    });
}

/// Upload glTF images to main-thread GL texture cache.
/// `pixels`: flat RGBA8 data for all images concatenated
/// `widths`/`heights`: per-image dimensions
#[wasm_bindgen(js_name = "mr_uploadImages")]
pub fn mr_upload_images(pixels: &[u8], widths: &[u32], heights: &[u32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            let gl = st.renderer.gl();
            let mut images = Vec::new();
            let mut offset = 0usize;
            for i in 0..widths.len() {
                let w = widths[i];
                let h = heights[i];
                let size = (w * h * 4) as usize;
                if offset + size <= pixels.len() {
                    images.push((w, h, &pixels[offset..offset + size]));
                }
                offset += size;
            }
            st.texture_cache.upload_images(gl, &images);
            info!("Uploaded {} textures to main-thread GL", widths.len());
        }
    });
}

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
            debug!("Skinning texture uploaded: {} vertices", nv);
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
            debug!("Morph texture uploaded: {} vertices × {} targets", nv, nt);
        }
    });
}

/// Upload environment cubemaps for IBL as float data (RGBA16F).
/// `prefiltered`: ALL mip levels concatenated. Each mip = 6 faces × mipSize×mipSize × 4 floats.
///   Mip 0 = full size, mip 1 = size/2, etc. Total mips = floor(log2(pf_size)) + 1.
/// `irradiance`: 6 faces × ir_size×ir_size × 4 floats (single level).
#[wasm_bindgen(js_name = "mr_uploadEnvMaps")]
pub fn mr_upload_env_maps(
    prefiltered: &[f32], pf_size: u32,
    irradiance: &[f32], ir_size: u32,
) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            let gl = st.renderer.gl();
            unsafe {
                if let Some(t) = st.env_maps.prefiltered { gl.delete_texture(t); }
                if let Some(t) = st.env_maps.irradiance { gl.delete_texture(t); }

                let num_mips = (pf_size as f32).log2().floor() as i32 + 1;

                // Prefiltered specular cubemap with per-mip GGX-filtered data
                let pf_tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(pf_tex));
                // Allocate storage for all mip levels
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MAX_LEVEL, num_mips - 1);

                let mut offset = 0usize;
                for mip in 0..num_mips {
                    let mip_size = (pf_size >> mip as u32).max(1);
                    let face_floats = (mip_size * mip_size * 4) as usize;
                    for face in 0..6u32 {
                        let end = offset + face_floats;
                        if end <= prefiltered.len() {
                            let bytes = bytemuck_cast_slice(&prefiltered[offset..end]);
                            gl.tex_image_2d(
                                glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                                mip, glow::RGBA16F as i32,
                                mip_size as i32, mip_size as i32, 0,
                                glow::RGBA, glow::FLOAT,
                                glow::PixelUnpackData::Slice(Some(bytes)),
                            );
                        }
                        offset = end;
                    }
                }
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

                // Irradiance cubemap (single level)
                let ir_tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(ir_tex));
                let ir_face_floats = (ir_size * ir_size * 4) as usize;
                for face in 0..6u32 {
                    let off = face as usize * ir_face_floats;
                    let end = off + ir_face_floats;
                    if end <= irradiance.len() {
                        let bytes = bytemuck_cast_slice(&irradiance[off..end]);
                        gl.tex_image_2d(
                            glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                            0, glow::RGBA16F as i32,
                            ir_size as i32, ir_size as i32, 0,
                            glow::RGBA, glow::FLOAT,
                            glow::PixelUnpackData::Slice(Some(bytes)),
                        );
                    }
                }
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

                st.env_maps.prefiltered = Some(pf_tex);
                st.env_maps.irradiance = Some(ir_tex);
                st.env_maps.mip_count = (num_mips - 1) as f32;

                info!("Uploaded env maps: prefiltered {}x{} ({} mips), irradiance {}x{}",
                    pf_size, pf_size, st.env_maps.mip_count as u32, ir_size, ir_size);
            }
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
        if missing > 0 {
            info!("Built {} GPU batches, {} missing tess patches", built, missing);
        }
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
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return };
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

        // Mode-specific UBO setup
        let has_env = state.env_maps.prefiltered.is_some();
        let env_mips = state.env_maps.mip_count;
        let white = state.texture_cache.placeholder();
        let black = state.texture_cache.placeholder_black();
        let cube = state.texture_cache.placeholder_cube();

        match state.render_mode {
            RenderMode::Pbr => {
                // Env cubemaps: bind once (shared across all batches)
                unsafe {
                    gl.active_texture(glow::TEXTURE0 + 5);
                    gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(
                        state.env_maps.prefiltered.unwrap_or(cube)
                    ));
                    gl.active_texture(glow::TEXTURE0 + 6);
                    gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(
                        state.env_maps.irradiance.unwrap_or(cube)
                    ));
                }

                // PBR: two-pass rendering — opaque first, then transparent
                unsafe { gl.use_program(Some(state.renderer.programs().pbr)); }
                let default_mat = PbrParams::default();

                let get_mat = |batch: &RenderBatch| -> PbrParams {
                    let base = if batch.material_index < state.materials.len() {
                        state.materials[batch.material_index].clone()
                    } else if !state.materials.is_empty() {
                        state.materials[0].clone()
                    } else {
                        default_mat.clone()
                    };
                    let mut m = base;
                    m.has_env_map = has_env;
                    m.env_mip_count = env_mips;
                    m
                };

                // Pass 1: opaque non-transmission
                for batch in &render_batches {
                    let mat = get_mat(batch);
                    if mat.alpha_mode > 1.5 || mat.transmission_factor > 0.0 { continue; }
                    // Get this batch's material
                    let base_mat = if batch.material_index < state.materials.len() {
                        &state.materials[batch.material_index]
                    } else if !state.materials.is_empty() {
                        &state.materials[0]
                    } else {
                        &default_mat
                    };
                    let mut mat = base_mat.clone();
                    mat.has_env_map = has_env;
                    mat.env_mip_count = env_mips;

                    // Upload per-material PBR UBO
                    state.renderer.pbr_ubo().upload(gl, &mat);
                    state.renderer.pbr_ubo().bind(gl);

                    // Bind placeholders first, then override with actual textures
                    let tex_defaults = [white, black, black, black, white];
                    for (unit, &tex) in tex_defaults.iter().enumerate() {
                        unsafe {
                            gl.active_texture(glow::TEXTURE0 + unit as u32);
                            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                        }
                    }
                    let bind_tex = |unit: u32, idx: i32| {
                        if idx >= 0 {
                            let tex = state.texture_cache.get(Some(idx as usize));
                            unsafe {
                                gl.active_texture(glow::TEXTURE0 + unit);
                                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                            }
                        }
                    };
                    bind_tex(0, mat.base_color_tex_idx);
                    bind_tex(1, mat.metallic_roughness_tex_idx);
                    bind_tex(2, mat.normal_tex_idx);
                    bind_tex(3, mat.emissive_tex_idx);
                    bind_tex(4, mat.occlusion_tex_idx);

                    // Upload vertex UBO and draw this batch
                    quilting_renderer::pass::upload_batch_ubo(
                        gl, state.renderer.vtx_ubo(), &camera,
                        batch.perm_parity, batch.perm_index, 1,
                    );
                    unsafe {
                        gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                        gl.draw_elements_instanced(
                            glow::TRIANGLES, batch.mesh.num_tri_indices,
                            glow::UNSIGNED_INT, 0, batch.mesh.num_instances,
                        );
                    }
                }
                // --- Blit opaque framebuffer to scene_color texture for refraction ---
                let has_transmission = render_batches.iter().any(|b| {
                    let m = get_mat(b);
                    m.transmission_factor > 0.0
                });
                if has_transmission {
                    unsafe {
                        // Get canvas size from viewport
                        let mut vp_buf = [0i32; 4];
                        gl.get_parameter_i32_slice(glow::VIEWPORT, &mut vp_buf);
                        let vw = vp_buf[2].max(1);
                        let vh = vp_buf[3].max(1);

                        // Create/resize scene color texture if needed
                        if state.scene_color_size != (vw, vh) {
                            if let Some(fbo) = state.scene_color_fbo { gl.delete_framebuffer(fbo); }
                            if let Some(tex) = state.scene_color_tex { gl.delete_texture(tex); }

                            let tex = gl.create_texture().unwrap();
                            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                            gl.tex_image_2d(
                                glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
                                vw, vh, 0, glow::RGBA, glow::UNSIGNED_BYTE,
                                glow::PixelUnpackData::Slice(None),
                            );
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

                            let fbo = gl.create_framebuffer().unwrap();
                            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                            gl.framebuffer_texture_2d(
                                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                glow::TEXTURE_2D, Some(tex), 0,
                            );
                            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

                            state.scene_color_fbo = Some(fbo);
                            state.scene_color_tex = Some(tex);
                            state.scene_color_size = (vw, vh);
                        }

                        // Blit current framebuffer (opaques) → scene_color_tex
                        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None); // read from default FB
                        let dst_fbo = state.scene_color_fbo.unwrap();
                        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(dst_fbo));
                        gl.blit_framebuffer(
                            0, 0, vw, vh, 0, 0, vw, vh,
                            glow::COLOR_BUFFER_BIT, glow::NEAREST,
                        );
                        // Restore default framebuffer for drawing
                        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
                        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);

                        // Bind scene color and generate mipmaps for roughness blur
                        gl.active_texture(glow::TEXTURE0 + 8);
                        gl.bind_texture(glow::TEXTURE_2D, Some(state.scene_color_tex.unwrap()));
                        gl.generate_mipmap(glow::TEXTURE_2D);
                    }
                }

                // Pass 2: transmission + transparent
                for batch in &render_batches {
                    let mat = get_mat(batch);
                    if mat.alpha_mode < 1.5 && mat.transmission_factor <= 0.0 { continue; }

                    // Opaque-transmission: no blend, write depth (solid glass)
                    // Alpha-blend: blend on, no depth write (transparent overlay)
                    let is_blend = mat.alpha_mode > 1.5;
                    unsafe {
                        if is_blend {
                            gl.enable(glow::BLEND);
                            gl.blend_func_separate(
                                glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA,
                                glow::ONE, glow::ONE_MINUS_SRC_ALPHA,
                            );
                            gl.depth_mask(false);
                        } else {
                            gl.disable(glow::BLEND);
                            gl.depth_mask(true);
                        }
                    }

                    state.renderer.pbr_ubo().upload(gl, &mat);
                    state.renderer.pbr_ubo().bind(gl);

                    let tex_defaults = [white, black, black, black, white];
                    for (unit, &tex) in tex_defaults.iter().enumerate() {
                        unsafe {
                            gl.active_texture(glow::TEXTURE0 + unit as u32);
                            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                        }
                    }
                    let bind_tex = |unit: u32, idx: i32| {
                        if idx >= 0 {
                            let tex = state.texture_cache.get(Some(idx as usize));
                            unsafe {
                                gl.active_texture(glow::TEXTURE0 + unit);
                                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                            }
                        }
                    };
                    bind_tex(0, mat.base_color_tex_idx);
                    bind_tex(1, mat.metallic_roughness_tex_idx);
                    bind_tex(2, mat.normal_tex_idx);
                    bind_tex(3, mat.emissive_tex_idx);
                    bind_tex(4, mat.occlusion_tex_idx);

                    quilting_renderer::pass::upload_batch_ubo(
                        gl, state.renderer.vtx_ubo(), &camera,
                        batch.perm_parity, batch.perm_index, 1,
                    );
                    unsafe {
                        gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                        gl.draw_elements_instanced(
                            glow::TRIANGLES, batch.mesh.num_tri_indices,
                            glow::UNSIGNED_INT, 0, batch.mesh.num_instances,
                        );
                    }
                }
                unsafe {
                    gl.depth_mask(true);
                }

                state.renderer.end_frame();
                return;
            }
            RenderMode::Lod => {
                state.renderer.matcap_ubo().upload(gl, 0.0); // heatmap
                state.renderer.matcap_ubo().bind(gl);
            }
            _ => {
                state.renderer.matcap_ubo().upload(gl, 2.0); // procedural matcap
                state.renderer.matcap_ubo().bind(gl);
            }
        }

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
