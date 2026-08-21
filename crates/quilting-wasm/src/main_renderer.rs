//! Main-thread rendering module for the production hyperscope.
//!
//! All exports are prefixed with `mr_` in JS to distinguish from
//! worker-side exports in lib.rs.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use tracing::{info, debug};
use crate::{perf_mark, perf_measure};

use glow::HasContext;
use quilting_renderer::buffer::{
    MeshBuffers, TessBuffers, PbrParams, EnvironmentMaps, PersistentInstances,
};
use quilting_renderer::pass::{
    affine_normal_matrix, affine_orientation_sign, apply_batch_winding,
    camera_for_batch, Camera, RenderBatch, RenderMode, IDENTITY_MATRIX,
};
use quilting_renderer::Renderer;
use quilting_renderer::texture::TextureCache;
use quilting_core::batch;
use quilting_core::instance_layout;
use hyperscape::interchange::{
    GltfHyperscopePacket, HyperscapeGltfRuntime, RuntimeDiagnosticSnapshot,
};
use hyperscape::{ChamberSide, ContactClassification};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Duration;

/// Floats per material in the array `mr_setMaterials` receives.
const MATERIAL_STRIDE: usize = 50;

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
    persistent_buf: Option<PersistentInstances>,
    face_materials: Vec<usize>,
    /// Stable ordinary glTF node index for each source triangle.
    face_nodes: Vec<usize>,
    materials: Vec<PbrParams>,
    num_faces: usize,
    render_mode: RenderMode,
    mobius: [f32; 16],
    /// Explicit parity from the authoring generator word. Legacy direct matrix
    /// input falls back to the old `c != 0` heuristic.
    mobius_orientation: i8,
    /// Extracted state keyed by `(projection camera node, subject node)`.
    hyperscape_packets: BTreeMap<(usize, usize), EntityConformalState>,
    active_hyperscape_camera: Option<usize>,
    // Screen-space refraction: framebuffer copy for transmission
    /// Mip-chained scene copy for the transmission/refraction blur pyramid.
    /// Distinct from `fuzzy_scene_*`: this one carries a full mip chain with
    /// LINEAR_MIPMAP_LINEAR filtering, which the fuzzy paths must not inherit.
    scene_color_fbo: Option<glow::Framebuffer>,
    scene_color_tex: Option<glow::Texture>,
    scene_color_size: (i32, i32),
    /// Flat (single-level) scratch target for the fuzzy-vision weight paths.
    /// Kept separate from `scene_color_*` because both keyed reallocation on
    /// viewport size alone, so whichever path ran first silently imposed its
    /// texture shape on the other for the rest of the session.
    fuzzy_scene_fbo: Option<glow::Framebuffer>,
    fuzzy_scene_tex: Option<glow::Texture>,
    fuzzy_scene_size: (i32, i32),
    // Gaussian-blurred scene color for rough transmission
    blur_fbo: Option<glow::Framebuffer>,
    blur_tex: Option<glow::Texture>,
    blur_program: Option<glow::Program>,
    blur_vao: Option<glow::VertexArray>,
    // MRT: PBR renders to this FBO with color + weight attachments
    pbr_fbo: Option<glow::Framebuffer>,
    pbr_color_tex: Option<glow::Texture>,
    pbr_depth_rb: Option<glow::Renderbuffer>,
    pbr_fbo_size: (i32, i32),
    // Fuzzy-vision JFA blur pipeline
    fuzzy: Option<fuzzy_vision::JfaPipeline>,
    fuzzy_enabled: bool,
    fuzzy_mode: u32, // 0=radial, 1=conformal
    fuzzy_debug: u32, // 0=off, 1=smoothed weight, 2=jfa, 3=firmness
    fuzzy_weight_fbo: Option<glow::Framebuffer>,
    fuzzy_weight_tex: Option<glow::Texture>,
    fuzzy_weight_size: (i32, i32),
    // Pick buffer for face inspection
    pick_fbo: Option<glow::Framebuffer>,
    pick_tex: Option<glow::Texture>,
    pick_depth: Option<glow::Renderbuffer>,
    pick_size: (i32, i32),
    highlight_face: i32, // -1 = none
    highlight_prog: Option<glow::Program>,
    highlight_vao: Option<glow::VertexArray>,
}

struct GpuBatch {
    mesh: MeshBuffers,
    shared_buf: bool, // true = mesh uses shared instance buffer, don't delete instance_buf
    perm_parity: f32,
    perm_index: i32,
    material_index: usize,
    node_index: usize,
    lod: [u32; 3],
}

#[derive(Clone, Copy)]
struct EntityConformalState {
    mobius: [f32; 16],
    orientation_sign: i8,
    euclidean_model: [f32; 16],
}

/// Create a separable Gaussian blur program (GLSL ES 300, not WGSL).
fn create_blur_program(gl: &glow::Context) -> Result<(glow::Program, glow::VertexArray), String> {
    unsafe {
        let vs_src = "#version 300 es\n\
            out vec2 v_uv;\n\
            void main() {\n\
                float x = float((gl_VertexID & 1) << 2) - 1.0;\n\
                float y = float((gl_VertexID & 2) << 1) - 1.0;\n\
                v_uv = vec2(x, y) * 0.5 + 0.5;\n\
                gl_Position = vec4(x, y, 0.0, 1.0);\n\
            }";
        let fs_src = "#version 300 es\n\
            precision highp float;\n\
            uniform sampler2D u_tex;\n\
            uniform vec2 u_dir;\n\
            in vec2 v_uv;\n\
            out vec4 o_color;\n\
            void main() {\n\
                vec3 c = textureLod(u_tex, v_uv, 0.0).rgb * 0.227027;\n\
                c += textureLod(u_tex, v_uv + u_dir * 1.384615, 0.0).rgb * 0.316216;\n\
                c += textureLod(u_tex, v_uv - u_dir * 1.384615, 0.0).rgb * 0.316216;\n\
                c += textureLod(u_tex, v_uv + u_dir * 3.230769, 0.0).rgb * 0.070270;\n\
                c += textureLod(u_tex, v_uv - u_dir * 3.230769, 0.0).rgb * 0.070270;\n\
                o_color = vec4(c, 1.0);\n\
            }";

        let vs = gl.create_shader(glow::VERTEX_SHADER).map_err(|e| format!("{e}"))?;
        gl.shader_source(vs, vs_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            let log = gl.get_shader_info_log(vs);
            return Err(format!("blur VS: {log}"));
        }
        let fs = gl.create_shader(glow::FRAGMENT_SHADER).map_err(|e| format!("{e}"))?;
        gl.shader_source(fs, fs_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            let log = gl.get_shader_info_log(fs);
            return Err(format!("blur FS: {log}"));
        }
        let prog = gl.create_program().map_err(|e| format!("{e}"))?;
        gl.attach_shader(prog, vs);
        gl.attach_shader(prog, fs);
        gl.link_program(prog);
        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            return Err(format!("blur link: {log}"));
        }
        gl.detach_shader(prog, vs); gl.delete_shader(vs);
        gl.detach_shader(prog, fs); gl.delete_shader(fs);

        // Set texture unit
        gl.use_program(Some(prog));
        if let Some(loc) = gl.get_uniform_location(prog, "u_tex") {
            gl.uniform_1_i32(Some(&loc), 0);
        }
        gl.use_program(None);

        let vao = gl.create_vertex_array().map_err(|e| format!("{e}"))?;

        Ok((prog, vao))
    }
}

thread_local! {
    static STATE: RefCell<Option<MainState>> = RefCell::new(None);
    static TESS_CACHE: RefCell<std::collections::HashMap<String, TessBuffers>> =
        RefCell::new(std::collections::HashMap::new());
    static HYPERSCAPE_RUNTIME: RefCell<Option<HyperscapeGltfRuntime>> = RefCell::new(None);
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

    let fuzzy = fuzzy_vision::JfaPipeline::new(renderer.gl(), fuzzy_vision::JfaConfig::default()).ok();

    STATE.with(|s| {
        *s.borrow_mut() = Some(MainState {
            renderer, texture_cache,
            env_maps: EnvironmentMaps::default(),
            batches: Vec::new(),
            cached_instances: Vec::new(),
            persistent_buf: None,
            face_materials: Vec::new(),
            face_nodes: Vec::new(),
            materials: Vec::new(),
            num_faces: 0,
            render_mode: RenderMode::Pbr,
            mobius: IDENTITY_MOBIUS,
            mobius_orientation: 1,
            hyperscape_packets: BTreeMap::new(),
            active_hyperscape_camera: None,
            scene_color_fbo: None,
            scene_color_tex: None,
            scene_color_size: (0, 0),
            fuzzy_scene_fbo: None,
            fuzzy_scene_tex: None,
            fuzzy_scene_size: (0, 0),
            blur_fbo: None,
            blur_tex: None,
            blur_program: None,
            blur_vao: None,
            pbr_fbo: None,
            pbr_color_tex: None,
            pbr_depth_rb: None,
            pbr_fbo_size: (0, 0),
            fuzzy,
            fuzzy_enabled: false,
            fuzzy_mode: 0,
            fuzzy_debug: 0,
            fuzzy_weight_fbo: None,
            fuzzy_weight_tex: None,
            fuzzy_weight_size: (0, 0),
            pick_fbo: None,
            pick_tex: None,
            pick_depth: None,
            pick_size: (0, 0),
            highlight_face: -1,
            highlight_prog: None,
            highlight_vao: None,
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
                "both" => RenderMode::Both, "lod" => RenderMode::Lod, "stretch" => RenderMode::Stretch,
                _ => RenderMode::Pbr,
            };
        }
    });
}

#[wasm_bindgen(js_name = "mr_setMobius")]
pub fn mr_set_mobius(mobius: &[f32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.hyperscape_packets.clear();
            st.active_hyperscape_camera = None;
            for (i, &v) in mobius.iter().take(16).enumerate() { st.mobius[i] = v; }
            let c_len_sq = st.mobius[8..12].iter().map(|value| value * value).sum::<f32>();
            st.mobius_orientation = if c_len_sq > 0.001 { -1 } else { 1 };
        }
    });
}

#[wasm_bindgen(js_name = "mr_setMobiusWithParity")]
pub fn mr_set_mobius_with_parity(mobius: &[f32], orientation_sign: i32) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.hyperscape_packets.clear();
            renderer.active_hyperscape_camera = None;
            for (index, &value) in mobius.iter().take(16).enumerate() {
                renderer.mobius[index] = value;
            }
            renderer.mobius_orientation = if orientation_sign < 0 { -1 } else { 1 };
        }
    });
}

fn clear_hyperscape_packets() {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.hyperscape_packets.clear();
            renderer.active_hyperscape_camera = None;
        }
    });
}

fn apply_hyperscape_packets(packets: &[GltfHyperscopePacket]) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(renderer) = state.as_mut() else { return };

        renderer.hyperscape_packets.clear();
        for extracted in packets {
            renderer.hyperscape_packets.insert(
                (extracted.camera_node, extracted.subject_node),
                EntityConformalState {
                    mobius: extracted.packet.mobius,
                    orientation_sign: extracted.packet.orientation_sign,
                    euclidean_model: extracted.packet.euclidean_model,
                },
            );
        }

        let retained_camera = renderer.active_hyperscape_camera.filter(|camera| {
            renderer
                .hyperscape_packets
                .keys()
                .any(|(candidate, _)| candidate == camera)
        });
        renderer.active_hyperscape_camera = retained_camera.or_else(|| {
            renderer
                .hyperscape_packets
                .keys()
                .map(|(camera, _)| *camera)
                .min()
        });

        // Preserve legacy global consumers while batches use the complete
        // subject/view map below.
        if let Some(camera) = renderer.active_hyperscape_camera {
            if let Some((_, packet)) = renderer
                .hyperscape_packets
                .range((camera, 0)..=(camera, usize::MAX))
                .next()
            {
                renderer.mobius = packet.mobius;
                renderer.mobius_orientation = packet.orientation_sign;
            }
        }
    });
}

fn conformal_state_for_node(renderer: &MainState, node_index: usize) -> EntityConformalState {
    if let Some(camera) = renderer.active_hyperscape_camera {
        return renderer
            .hyperscape_packets
            .get(&(camera, node_index))
            .copied()
            .unwrap_or(EntityConformalState {
                mobius: IDENTITY_MOBIUS,
                orientation_sign: 1,
                euclidean_model: IDENTITY_MATRIX,
            });
    }
    EntityConformalState {
            mobius: renderer.mobius,
            orientation_sign: renderer.mobius_orientation,
            euclidean_model: IDENTITY_MATRIX,
    }
}

fn camera_for_node(camera: &Camera, renderer: &MainState, node_index: usize) -> Camera {
    let conformal = conformal_state_for_node(renderer, node_index);
    Camera {
        mvp: camera.mvp,
        mv: camera.mv,
        mobius: conformal.mobius,
        camera_pos: camera.camera_pos,
    }
}

#[wasm_bindgen(js_name = "mr_loadHyperscape")]
pub fn mr_load_hyperscape(data: &[u8]) -> bool {
    let (nodes, asset) = match quilting_gltf::load_hyperscape_graph(data) {
        Ok(graph) => graph,
        Err(error) => {
            web_sys::console::warn_1(&format!("Hyperscape graph: {error}").into());
            HYPERSCAPE_RUNTIME.with(|runtime| *runtime.borrow_mut() = None);
            clear_hyperscape_packets();
            return false;
        }
    };
    let Some(asset) = asset else {
        HYPERSCAPE_RUNTIME.with(|runtime| *runtime.borrow_mut() = None);
        clear_hyperscape_packets();
        return false;
    };
    let runtime = match HyperscapeGltfRuntime::new(&nodes, &asset) {
        Ok(runtime) => runtime,
        Err(error) => {
            web_sys::console::warn_1(&format!("Hyperscape runtime: {error}").into());
            HYPERSCAPE_RUNTIME.with(|runtime| *runtime.borrow_mut() = None);
            clear_hyperscape_packets();
            return false;
        }
    };
    apply_hyperscape_packets(&runtime.packets_by_node());
    HYPERSCAPE_RUNTIME.with(|slot| *slot.borrow_mut() = Some(runtime));
    true
}

/// Select which authored projection-camera node supplies subject-relative
/// conformal chains. Returns false when that camera has no extracted packets.
#[wasm_bindgen(js_name = "mr_setHyperscapeCameraNode")]
pub fn mr_set_hyperscape_camera_node(node_index: i32) -> bool {
    if node_index < 0 {
        return false;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(renderer) = state.as_mut() else { return false };
        let node_index = node_index as usize;
        if renderer
            .hyperscape_packets
            .keys()
            .any(|(camera, _)| *camera == node_index)
        {
            renderer.active_hyperscape_camera = Some(node_index);
            true
        } else {
            false
        }
    })
}

#[wasm_bindgen(js_name = "mr_tickHyperscape")]
pub fn mr_tick_hyperscape(delta_seconds: f64) -> JsValue {
    let delta_seconds = if delta_seconds.is_finite() {
        delta_seconds.clamp(0.0, 0.25)
    } else {
        0.0
    };
    let snapshot = HYPERSCAPE_RUNTIME.with(|slot| {
        let mut runtime = slot.borrow_mut();
        let runtime = runtime.as_mut()?;
        runtime.tick(Duration::from_secs_f64(delta_seconds));
        Some((
            runtime.packets_by_node(),
            runtime.diagnostics().to_vec(),
            runtime.diagnostic_snapshot(),
        ))
    });
    let Some((packets, diagnostics, scene_diagnostics)) = snapshot else {
        return JsValue::NULL;
    };
    apply_hyperscape_packets(&packets);
    let active_camera = STATE.with(|state| {
        state
            .borrow()
            .as_ref()
            .and_then(|renderer| renderer.active_hyperscape_camera)
    });

    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"packet_count".into(),
        &JsValue::from_f64(packets.len() as f64),
    ).ok();
    if let Some(camera) = active_camera {
        js_sys::Reflect::set(
            &result,
            &"active_camera_node".into(),
            &JsValue::from_f64(camera as f64),
        ).ok();
    }
    let primary = active_camera
        .and_then(|camera| packets.iter().find(|packet| packet.camera_node == camera))
        .or_else(|| packets.first());
    if let Some(extracted) = primary {
        let packet = &extracted.packet;
        js_sys::Reflect::set(
            &result,
            &"orientation_sign".into(),
            &JsValue::from_f64(packet.orientation_sign as f64),
        ).ok();
        let mobius = js_sys::Float32Array::from(packet.mobius.as_slice());
        let model = js_sys::Float32Array::from(packet.euclidean_model.as_slice());
        let camera_eye = js_sys::Float32Array::from(packet.camera_eye.as_slice());
        js_sys::Reflect::set(&result, &"mobius".into(), &mobius).ok();
        js_sys::Reflect::set(&result, &"euclidean_model".into(), &model).ok();
        js_sys::Reflect::set(&result, &"camera_eye".into(), &camera_eye).ok();
        if let Some(target) = packet.camera_target {
            let target = js_sys::Float32Array::from(target.as_slice());
            js_sys::Reflect::set(&result, &"camera_target".into(), &target).ok();
        }
    }
    let packet_snapshots = js_sys::Array::new();
    for extracted in packets {
        let packet = js_sys::Object::new();
        js_sys::Reflect::set(
            &packet,
            &"subject_node".into(),
            &JsValue::from_f64(extracted.subject_node as f64),
        ).ok();
        js_sys::Reflect::set(
            &packet,
            &"camera_node".into(),
            &JsValue::from_f64(extracted.camera_node as f64),
        ).ok();
        js_sys::Reflect::set(
            &packet,
            &"orientation_sign".into(),
            &JsValue::from_f64(extracted.packet.orientation_sign as f64),
        ).ok();
        js_sys::Reflect::set(
            &packet,
            &"origin_pole_denominator_norm_sq".into(),
            &JsValue::from_f64(extracted.packet.origin_pole_denominator_norm_sq),
        ).ok();
        let mobius = js_sys::Float32Array::from(extracted.packet.mobius.as_slice());
        let model = js_sys::Float32Array::from(extracted.packet.euclidean_model.as_slice());
        let camera_eye = js_sys::Float32Array::from(extracted.packet.camera_eye.as_slice());
        js_sys::Reflect::set(&packet, &"mobius".into(), &mobius).ok();
        js_sys::Reflect::set(&packet, &"euclidean_model".into(), &model).ok();
        js_sys::Reflect::set(&packet, &"camera_eye".into(), &camera_eye).ok();
        if let Some(target) = extracted.packet.camera_target {
            let target = js_sys::Float32Array::from(target.as_slice());
            js_sys::Reflect::set(&packet, &"camera_target".into(), &target).ok();
        }
        packet_snapshots.push(&packet);
    }
    js_sys::Reflect::set(&result, &"packets".into(), &packet_snapshots).ok();
    let messages = js_sys::Array::new();
    for diagnostic in diagnostics {
        messages.push(&JsValue::from_str(&diagnostic));
    }
    js_sys::Reflect::set(&result, &"diagnostics".into(), &messages).ok();
    js_sys::Reflect::set(
        &result,
        &"scene".into(),
        &runtime_diagnostic_snapshot_to_js(&scene_diagnostics),
    ).ok();
    result.into()
}

fn chamber_side_name(side: ChamberSide) -> &'static str {
    match side {
        ChamberSide::Negative => "negative",
        ChamberSide::OnWall => "on_wall",
        ChamberSide::Positive => "positive",
    }
}

fn runtime_diagnostic_snapshot_to_js(snapshot: &RuntimeDiagnosticSnapshot) -> JsValue {
    let frames = snapshot
        .frames
        .iter()
        .map(|frame| {
            serde_json::json!({
                "frame": frame.frame,
                "name": frame.name,
                "parent": frame.parent,
                "generator_count": frame.generator_count,
                "local_orientation_sign": frame.local_orientation_sign,
                "world_orientation_sign": frame.world_orientation_sign,
            })
        })
        .collect::<Vec<_>>();
    let entities = snapshot
        .entities
        .iter()
        .map(|entity| {
            let chamber = entity
                .chamber
                .iter()
                .map(|(wall, side)| {
                    serde_json::json!({ "wall": wall, "side": chamber_side_name(*side) })
                })
                .collect::<Vec<_>>();
            let history = entity
                .history
                .iter()
                .map(|sample| {
                    serde_json::json!({
                        "elapsed_seconds": sample.elapsed_seconds,
                        "frame": sample.frame.0,
                        "local": sample.local,
                        "euclidean": sample.euclidean,
                        "anchor_frame": sample.anchor_frame.map(|frame| frame.0),
                        "flipped_walls": sample.flipped_walls.iter().map(|wall| wall.0).collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "node": entity.node,
                "name": entity.name,
                "frame": entity.frame,
                "local": entity.local,
                "euclidean": entity.euclidean,
                "anchor_frame": entity.anchor_frame,
                "flipped_walls": entity.flipped_walls,
                "chamber": chamber,
                "history": history,
            })
        })
        .collect::<Vec<_>>();
    let contacts = snapshot
        .contacts
        .iter()
        .map(|contact| {
            let classification = match contact.classification {
                ContactClassification::Known(relation) => format!("{relation:?}"),
                ContactClassification::RequiresCommonChart => "requires_common_chart".into(),
            };
            serde_json::json!({
                "first": contact.first.0,
                "second": contact.second.0,
                "classification": classification,
            })
        })
        .collect::<Vec<_>>();
    let chamber_counts = snapshot
        .chamber_aggregates
        .counts
        .iter()
        .map(|(key, count)| {
            let signature = key
                .iter()
                .map(|(wall, side)| {
                    serde_json::json!({ "wall": wall.0, "side": chamber_side_name(*side) })
                })
                .collect::<Vec<_>>();
            serde_json::json!({ "signature": signature, "count": count })
        })
        .collect::<Vec<_>>();
    let visibility_hints = snapshot
        .visibility_hints
        .iter()
        .map(|hint| {
            serde_json::json!({
                "subject_node": hint.subject_node,
                "camera_node": hint.camera_node,
                "comparable_chambers": hint.comparable_chambers,
                "same_chamber": hint.same_chamber,
                "separating_walls": hint.separating_walls,
                "contact_frontier": hint.contact_frontier,
                "can_cull": hint.can_cull,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "elapsed_seconds": snapshot.elapsed_seconds,
        "frames": frames,
        "entities": entities,
        "contacts": contacts,
        "chamber_aggregates": {
            "epoch": snapshot.chamber_aggregates.epoch,
            "counts": chamber_counts,
            "changed_nodes": snapshot.chamber_aggregates.changed_nodes,
            "changed_walls": snapshot.chamber_aggregates.changed_walls,
            "contact_frontier": snapshot.chamber_aggregates.contact_frontier,
            "classifications_last_tick": snapshot.chamber_aggregates.classifications_last_tick,
            "aggregate_updates_last_tick": snapshot.chamber_aggregates.aggregate_updates_last_tick,
        },
        "visibility_hints": visibility_hints,
        "transform_history_epoch": snapshot.transform_history_epoch,
    });
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

#[wasm_bindgen(js_name = "mr_setFuzzy")]
pub fn mr_set_fuzzy(enabled: bool, max_distance: f32, blur_strength: f32, mode: u32, focus: f32, bandwidth: f32, normalize: bool, stretch_min: f32, stretch_max: f32, blur_passes: u32, kawase_passes: u32, kawase_offset: f32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.fuzzy_enabled = enabled;
            st.fuzzy_mode = mode;
            if let Some(ref mut fv) = st.fuzzy {
                let mut cfg = fv.config().clone();
                cfg.max_distance = max_distance;
                cfg.blur_strength = blur_strength;
                cfg.focus = focus;
                cfg.bandwidth = bandwidth;
                cfg.normalize = normalize;
                cfg.cpu_stretch_min = stretch_min;
                cfg.cpu_stretch_max = stretch_max;
                cfg.blur_passes = blur_passes.max(1);
                cfg.kawase_passes = kawase_passes;
                cfg.kawase_offset = kawase_offset;
                cfg.blur_mode = mode;
                fv.set_config(cfg);
            }
        }
    });
}

#[wasm_bindgen(js_name = "mr_setFuzzyDebug")]
pub fn mr_set_fuzzy_debug(debug_stage: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.fuzzy_debug = debug_stage;
        }
    });
}

#[wasm_bindgen(js_name = "mr_highlightFace")]
pub fn mr_highlight_face(face_id: i32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.highlight_face = face_id;
        }
    });
}

/// Pick the face under pixel (x, y). Renders a pick pass and reads back the face ID.
/// Returns face ID (>= 0) or -1 if no face at that pixel.
/// Also logs face info (LOD, edge lengths, instance data) to console.
#[wasm_bindgen(js_name = "mr_pick")]
pub fn mr_pick(mvp: &[f32], mv: &[f32], camera_pos: &[f32], x: i32, y: i32) -> i32 {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return -1 };
        // Lazy-init highlight program
        if state.highlight_prog.is_none() {
            let gl = state.renderer.gl();
            unsafe {
                let vs = gl.create_shader(glow::VERTEX_SHADER).unwrap();
                gl.shader_source(vs, HIGHLIGHT_VS);
                gl.compile_shader(vs);
                let fs = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
                gl.shader_source(fs, HIGHLIGHT_FS);
                gl.compile_shader(fs);
                let prog = gl.create_program().unwrap();
                gl.attach_shader(prog, vs);
                gl.attach_shader(prog, fs);
                gl.link_program(prog);
                gl.delete_shader(vs);
                gl.delete_shader(fs);
                state.highlight_prog = Some(prog);
                state.highlight_vao = Some(gl.create_vertex_array().unwrap());
            }
        }

        let gl = state.renderer.gl();

        unsafe {
            let mut vp = [0i32; 4];
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut vp);
            let vw = vp[2].max(1);
            let vh = vp[3].max(1);

            // Create/resize pick FBO
            if state.pick_fbo.is_none() || state.pick_size != (vw, vh) {
                if let Some(f) = state.pick_fbo { gl.delete_framebuffer(f); }
                if let Some(t) = state.pick_tex { gl.delete_texture(t); }
                if let Some(r) = state.pick_depth { gl.delete_renderbuffer(r); }

                let tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                    glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);

                let depth = gl.create_renderbuffer().unwrap();
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth));
                gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, vw, vh);

                let fbo = gl.create_framebuffer().unwrap();
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D, Some(tex), 0);
                gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT,
                    glow::RENDERBUFFER, Some(depth));

                state.pick_fbo = Some(fbo);
                state.pick_tex = Some(tex);
                state.pick_depth = Some(depth);
                state.pick_size = (vw, vh);
            }

            // Render pick pass
            gl.bind_framebuffer(glow::FRAMEBUFFER, state.pick_fbo);
            gl.viewport(0, 0, vw, vh);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);

            gl.use_program(Some(state.renderer.programs().pick));

            let camera = quilting_renderer::pass::Camera {
                mvp: mvp[..16].try_into().unwrap_or([0.0; 16]),
                mv: mv[..16].try_into().unwrap_or([0.0; 16]),
                mobius: state.mobius,
                camera_pos: [
                    camera_pos.get(0).copied().unwrap_or(0.0),
                    camera_pos.get(1).copied().unwrap_or(0.0),
                    camera_pos.get(2).copied().unwrap_or(0.0),
                ],
            };

            let mut face_offset = 0i32;
            for batch in &state.batches {
                let batch_camera = camera_for_node(&camera, state, batch.node_index);
                let conformal = conformal_state_for_node(state, batch.node_index);
                apply_batch_winding(
                    gl,
                    conformal.orientation_sign
                        * affine_orientation_sign(&conformal.euclidean_model),
                );
                let euclidean_normal = affine_normal_matrix(&conformal.euclidean_model);
                let vtx_ubo = state.renderer.vtx_ubo();
                vtx_ubo.upload(
                    gl, &batch_camera.mvp, &batch_camera.mv,
                    batch.perm_parity, batch.perm_index, 1,
                    &batch_camera.mobius, &batch_camera.camera_pos,
                    &conformal.euclidean_model, &euclidean_normal,
                );
                vtx_ubo.set_face_offset(gl, face_offset);
                vtx_ubo.bind(gl);

                gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                gl.draw_elements_instanced(
                    glow::TRIANGLES, batch.mesh.num_tri_indices,
                    glow::UNSIGNED_INT, 0, batch.mesh.num_instances,
                );
                face_offset += batch.mesh.num_instances;
            }

            // Read pixel (flip Y for framebuffer coords)
            let fy = vh - 1 - y;
            let mut px = [0u8; 4];
            gl.read_pixels(x, fy, 1, 1, glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut px)));

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, vw, vh);

            // Decode 24-bit face ID from RGB
            if px[3] == 0 {
                return -1; // no geometry (alpha=0 from clear)
            }
            let face_id = (px[0] as i32) | ((px[1] as i32) << 8) | ((px[2] as i32) << 16);

            // Log face info
            let base = face_id as usize * instance_layout::STRIDE
                + instance_layout::offset::POSITIONS;
            if base + 12 <= state.cached_instances.len() {
                // Instance layout: [vi, x, y, z] per control point — skip vertex index at offset 0
                let p0 = &state.cached_instances[base+1..base+4];
                let p1 = &state.cached_instances[base+5..base+8];
                let p2 = &state.cached_instances[base+9..base+12];
                info!("Pick face {}: p0=[{:.4},{:.4},{:.4}] p1=[{:.4},{:.4},{:.4}] p2=[{:.4},{:.4},{:.4}]",
                    face_id, p0[0],p0[1],p0[2], p1[0],p1[1],p1[2], p2[0],p2[1],p2[2]);

                // Compute edge lengths and medians
                let d = |a: &[f32], b: &[f32]| ((a[0]-b[0]).powi(2)+(a[1]-b[1]).powi(2)+(a[2]-b[2]).powi(2)).sqrt();
                let mid = |a: &[f32], b: &[f32]| [(a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0];
                let ea = d(p1, p2); let eb = d(p0, p2); let ec = d(p0, p1);
                let ma = d(p0, &mid(p1, p2)); let mb = d(p1, &mid(p0, p2)); let mc = d(p2, &mid(p0, p1));
                info!("  edges: a={:.4} b={:.4} c={:.4}", ea, eb, ec);
                info!("  medians: a={:.4} b={:.4} c={:.4}", ma, mb, mc);
                info!("  bary at click: ({:.2}, {:.2})", px[2] as f32 / 255.0, px[3] as f32 / 255.0);

                // Find batch and report LOD triple
                let mut fo = 0i32;
                for batch in &state.batches {
                    let bs = batch.mesh.num_instances;
                    if face_id >= fo && face_id < fo + bs {
                        info!("  LOD triple: [{}, {}, {}] perm={} parity={:.0}",
                            batch.lod[0], batch.lod[1], batch.lod[2],
                            batch.perm_index, batch.perm_parity);
                        info!("  glTF node: {}", batch.node_index);
                        break;
                    }
                    fo += bs;
                }
            }

            // Set highlight
            state.highlight_face = face_id;
            face_id
        }
    })
}

#[wasm_bindgen(js_name = "mr_setInstanceData")]
pub fn mr_set_instance_data(instances: &[f32], num_faces: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.cached_instances = instances.to_vec();
            st.num_faces = num_faces as usize;
            if st.face_nodes.len() != st.num_faces {
                st.face_nodes = vec![0; st.num_faces];
            }
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

#[wasm_bindgen(js_name = "mr_setFaceNodes")]
pub fn mr_set_face_nodes(nodes: &[i32]) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.face_nodes = nodes
                .iter()
                .map(|&node| if node >= 0 { node as usize } else { 0 })
                .collect();
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

/// Set PBR material parameters from JS material objects.
/// Each material is a flat f32 array of `MATERIAL_STRIDE` floats packed as:
/// [base_r, base_g, base_b, base_a, metallic, roughness, normal_scale,
///  occlusion_strength, alpha_cutoff, alpha_mode, unlit,
///  emissive_r, emissive_g, emissive_b,
///  has_base_color_tex, has_mr_tex, has_normal_tex, has_emissive_tex, has_occlusion_tex,
///  sheen_r, sheen_g, sheen_b, has_sheen, sheen_roughness,
///  specular_r, specular_g, specular_b, has_specular,
///  normal_uv_scale_x, normal_uv_scale_y, normal_uv_offset_x, normal_uv_offset_y, normal_uv_rotation,
///  base_uv_scale_x, base_uv_scale_y, base_uv_rotation,
///  base_color_tex_idx, mr_tex_idx, normal_tex_idx, emissive_tex_idx, occlusion_tex_idx,
///  ior, transmission_factor, thickness_factor,
///  attenuation_r, attenuation_g, attenuation_b, attenuation_distance,
///  transmission_tex_idx, double_sided]
///
/// `hyperscope.html` packs the matching array.
#[wasm_bindgen(js_name = "mr_setMaterials")]
pub fn mr_set_materials(data: &[f32], num_materials: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.materials.clear();
            let stride = MATERIAL_STRIDE;
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
                    transmission_tex_idx: d[48] as i32,
                    double_sided: d[49] > 0.5,
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
/// `wrap_modes`: pairs of [wrap_s, wrap_t] per image (GL enum values)
#[wasm_bindgen(js_name = "mr_uploadImages")]
pub fn mr_upload_images(pixels: &[u8], widths: &[u32], heights: &[u32], wrap_modes: &[u32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            let gl = st.renderer.gl();
            let mut images = Vec::new();
            let mut offset = 0usize;
            for i in 0..widths.len() {
                let w = widths[i];
                let h = heights[i];
                let ws = wrap_modes.get(i * 2).copied().unwrap_or(glow::REPEAT);
                let wt = wrap_modes.get(i * 2 + 1).copied().unwrap_or(glow::REPEAT);
                let size = (w * h * 4) as usize;
                if offset + size <= pixels.len() {
                    images.push((w, h, &pixels[offset..offset + size], ws, wt));
                }
                offset += size;
            }
            st.texture_cache.upload_images(gl, &images);
            info!("Uploaded {} textures to main-thread GL", widths.len());
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

        perf_mark("batch-destroy-start");
        for b in state.batches.drain(..) {
            if b.shared_buf {
                b.mesh.destroy_vaos_only(gl);
            } else {
                b.mesh.destroy(gl);
            }
        }
        perf_mark("batch-destroy-end");
        perf_measure("batch-destroy-old", "batch-destroy-start", "batch-destroy-end");

        // Phase 1: bucket sort to get face groupings (fast O(n), no instance data copy)
        perf_mark("batch-group-start");
        let nf = state.num_faces;
        // Model metadata can arrive before its instance buffer. Validate at
        // the consumption boundary so a previous model's face count cannot
        // destroy a correct node map for the incoming model.
        if state.face_nodes.len() != nf {
            web_sys::console::warn_1(
                &format!(
                    "face-node count {} does not match triangle count {}; unmatched entries use node 0",
                    state.face_nodes.len(), nf
                )
                .into(),
            );
            state.face_nodes.resize(nf, 0);
        }
        let mut buckets: BTreeMap<(usize, usize, usize, usize), Vec<u32>> = BTreeMap::new();
        for fi in 0..nf {
            let lo = fi * batch::FACE_LOD_STRIDE;
            if lo + 5 >= face_lods.len() { break; }
            let atlas_idx = face_lods[lo + 5] as usize;
            let perm = face_lods[lo + 3] as usize;
            let mat = if fi < state.face_materials.len() { state.face_materials[fi] } else { 0 };
            let node = state.face_nodes.get(fi).copied().unwrap_or(0);
            buckets.entry((atlas_idx, perm, mat, node)).or_default().push(fi as u32);
        }
        perf_mark("batch-group-end");
        perf_measure("batch-group", "batch-group-start", "batch-group-end");

        // Phase 2: write each batch's instance data contiguously into the persistent GPU buffer
        // and create lightweight VAOs pointing at the right offset
        perf_mark("batch-upload-start");

        // Ensure persistent buffer is large enough
        let total_bytes = nf * instance_layout::STRIDE * 4;
        match state.persistent_buf {
            Some(ref pb) => {
                if total_bytes > pb.capacity {
                    // Reallocate
                    pb.destroy(gl);
                    state.persistent_buf = None;
                }
            }
            None => {}
        }
        if state.persistent_buf.is_none() {
            let dummy = vec![0.0f32; nf * instance_layout::STRIDE];
            match PersistentInstances::new(gl, &dummy) {
                Ok(pb) => state.persistent_buf = Some(pb),
                Err(e) => { info!("Failed to create persistent buf: {e}"); return; }
            }
        }
        let pb = state.persistent_buf.as_ref().unwrap();

        // Stream each batch's instance data into contiguous regions of the GPU buffer
        let mut gpu_offset: usize = 0; // current write position in floats
        let mut built = 0;
        let mut missing = 0;
        let stride = instance_layout::STRIDE;

        // Pre-allocate a batch-sized CPU staging buffer (reused across batches)
        let max_batch_size = buckets.values().map(Vec::len).max().unwrap_or(0);
        let mut staging = vec![0.0f32; max_batch_size * stride];

        TESS_CACHE.with(|tc| {
            let tc = tc.borrow();
            for (&(_atlas_idx, perm, mat, node_index), faces) in &buckets {
                let fi0 = faces[0] as usize;
                let lo = fi0 * batch::FACE_LOD_STRIDE;
                let ca = face_lods[lo] as u32;
                let cb = face_lods[lo + 1] as u32;
                let cc = face_lods[lo + 2] as u32;
                let parity = face_lods[lo + 4];
                let tess_key = batch::TessKey { lod: [ca, cb, cc], perm_index: perm as u32 };

                let tess = match tc.get(&tess_key.as_string()) {
                    Some(t) => t,
                    None => { missing += 1; continue; }
                };

                // Gather this batch's instance data into staging buffer
                let batch_floats = faces.len() * stride;
                let permutation = quilting_core::permutation::S3_PERMUTATIONS[perm.min(5)];
                for (i, &fi) in faces.iter().enumerate() {
                    let src = fi as usize * stride;
                    let dst = i * stride;
                    staging[dst..dst + stride]
                        .copy_from_slice(&state.cached_instances[src..src + stride]);

                    // The cached instance contains the load-time identity-Mobius
                    // edge LODs. Refresh them from this computation so density
                    // visualization follows camera and conformal LOD changes.
                    let lod_offset = fi as usize * batch::FACE_LOD_STRIDE;
                    let canonical = [
                        face_lods[lod_offset],
                        face_lods[lod_offset + 1],
                        face_lods[lod_offset + 2],
                    ];
                    let instance_lod_offset = dst + instance_layout::offset::EDGE_LODS;
                    staging[instance_lod_offset..instance_lod_offset + 3].copy_from_slice(&[
                        canonical[permutation[0]],
                        canonical[permutation[1]],
                        canonical[permutation[2]],
                    ]);
                }

                // Upload this batch's data to the GPU at the current offset
                let byte_offset = gpu_offset * 4;
                unsafe {
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(pb.buf));
                    gl.buffer_sub_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        byte_offset as i32,
                        bytemuck_cast_slice(&staging[..batch_floats]),
                    );
                }

                // Create VAOs pointing at this offset in the shared buffer
                let mesh = match MeshBuffers::from_shared(
                    gl, tess, &pb.buf, byte_offset as i32, faces.len() as i32,
                ) {
                    Ok(m) => m, Err(_) => continue,
                };

                state.batches.push(GpuBatch {
                    mesh, shared_buf: true, perm_parity: parity,
                    perm_index: perm as i32,
                    material_index: mat, node_index, lod: [ca, cb, cc],
                });
                built += 1;
                gpu_offset += batch_floats;
            }
        });
        perf_mark("batch-upload-end");
        perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");

        let mut mat_counts = BTreeMap::new();
        let mut node_counts = BTreeMap::new();
        for b in &state.batches {
            *mat_counts.entry(b.material_index).or_insert(0usize) += b.mesh.num_instances as usize;
            *node_counts.entry(b.node_index).or_insert(0usize) += b.mesh.num_instances as usize;
        }
        info!(
            "Built {} GPU batches ({} missing), material→faces: {:?}, node→faces: {:?}",
            built, missing, mat_counts, node_counts
        );
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

/// Reset skeletal and morph animation state before accepting a new model.
/// Without this boundary, a static glTF loaded after an animated one is
/// deformed by the previous model's joint UBO and animation textures.
#[wasm_bindgen(js_name = "mr_clearAnimationState")]
pub fn mr_clear_animation_state() {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow().as_ref() {
            renderer
                .renderer
                .joint_ubo()
                .clear(renderer.renderer.gl());
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
            let conformal = conformal_state_for_node(state, b.node_index);
            let euclidean_orientation = affine_orientation_sign(&conformal.euclidean_model);
            RenderBatch {
                mesh: &b.mesh, perm_parity: b.perm_parity, perm_index: b.perm_index,
                wire_color: lod_color(&b.lod), material_index: b.material_index,
                mobius: conformal.mobius,
                orientation_sign: conformal.orientation_sign * euclidean_orientation,
                euclidean_model: conformal.euclidean_model,
                euclidean_normal: affine_normal_matrix(&conformal.euclidean_model),
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

                // Set up MRT FBO for PBR: color (attachment 0) + weight (attachment 1)
                // Only for conformal mode (mode 1) — radial mode blits from default FB.
                if state.fuzzy_enabled && true /* all fuzzy modes use MRT */ {
                    unsafe {
                        let mut vp = [0i32; 4];
                        gl.get_parameter_i32_slice(glow::VIEWPORT, &mut vp);
                        let vw = vp[2].max(1);
                        let vh = vp[3].max(1);

                        if state.pbr_fbo.is_none() || state.pbr_fbo_size != (vw, vh) {
                            // Clean up old
                            if let Some(f) = state.pbr_fbo { gl.delete_framebuffer(f); }
                            if let Some(t) = state.pbr_color_tex { gl.delete_texture(t); }
                            if let Some(r) = state.pbr_depth_rb { gl.delete_renderbuffer(r); }
                            if let Some(t) = state.fuzzy_weight_tex { gl.delete_texture(t); }
                            if let Some(f) = state.fuzzy_weight_fbo { gl.delete_framebuffer(f); }

                            // Color texture (attachment 0)
                            let color_tex = gl.create_texture().unwrap();
                            gl.bind_texture(glow::TEXTURE_2D, Some(color_tex));
                            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

                            // Weight texture (attachment 1)
                            let weight_tex = gl.create_texture().unwrap();
                            gl.bind_texture(glow::TEXTURE_2D, Some(weight_tex));
                            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

                            // Depth renderbuffer
                            let depth_rb = gl.create_renderbuffer().unwrap();
                            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth_rb));
                            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, vw, vh);

                            // FBO with both attachments
                            let fbo = gl.create_framebuffer().unwrap();
                            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                glow::TEXTURE_2D, Some(color_tex), 0);
                            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT1,
                                glow::TEXTURE_2D, Some(weight_tex), 0);
                            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT,
                                glow::RENDERBUFFER, Some(depth_rb));
                            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);

                            state.pbr_fbo = Some(fbo);
                            state.pbr_color_tex = Some(color_tex);
                            state.pbr_depth_rb = Some(depth_rb);
                            state.pbr_fbo_size = (vw, vh);
                            state.fuzzy_weight_tex = Some(weight_tex);
                            state.fuzzy_weight_fbo = None; // not needed separately
                            state.fuzzy_weight_size = (vw, vh);
                        }

                        // Bind MRT FBO and clear attachments separately:
                        // attachment 0 (color) = scene background
                        // attachment 1 (weight) = 0.5 (neutral stretch, won't pollute min/max)
                        gl.bind_framebuffer(glow::FRAMEBUFFER, state.pbr_fbo);
                        // Clear color attachment only
                        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::NONE]);
                        gl.clear_color(0.2, 0.2, 0.3, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                        // Clear weight attachment: R=0.5 (neutral stretch), G=0.5 (neutral depth)
                        // Both channels at 0.5 = "no effect" for JFA — background pixels
                        // won't bias toward blur or sharpness in either mode.
                        gl.draw_buffers(&[glow::NONE, glow::COLOR_ATTACHMENT1]);
                        gl.clear_color(0.5, 0.5, 0.0, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT);
                        // Re-enable both for rendering
                        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);
                    }
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
                    // m.debug_output reserved for future normals debug view
                    m
                };

                // Pass 1: opaque non-transmission
                for batch in &render_batches {
                    let mat = get_mat(batch);
                    if mat.alpha_mode > 1.5 || mat.transmission_factor > 0.0 { continue; }

                    // glTF spec: cull back faces for single-sided materials
                    unsafe {
                        if !mat.double_sided {
                            gl.enable(glow::CULL_FACE);
                            gl.cull_face(glow::BACK);
                        } else {
                            gl.disable(glow::CULL_FACE);
                        }
                    }

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
                    let batch_camera = camera_for_batch(&camera, batch);
                    apply_batch_winding(gl, batch.orientation_sign);
                    quilting_renderer::pass::upload_batch_ubo(
                        gl, state.renderer.vtx_ubo(), &batch_camera,
                        batch.perm_parity, batch.perm_index, 1,
                        &batch.euclidean_model, &batch.euclidean_normal,
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

                        // Create/resize scene color texture with mip chain
                        if state.scene_color_size != (vw, vh) {
                            if let Some(fbo) = state.scene_color_fbo { gl.delete_framebuffer(fbo); }
                            if let Some(tex) = state.scene_color_tex { gl.delete_texture(tex); }

                            let num_mips = ((vw.max(vh) as f32).log2().floor() as i32 + 1).min(8);
                            let tex = gl.create_texture().unwrap();
                            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                            // Allocate all mip levels
                            for mip in 0..num_mips {
                                let mw = (vw >> mip).max(1);
                                let mh = (vh >> mip).max(1);
                                gl.tex_image_2d(
                                    glow::TEXTURE_2D, mip, glow::RGBA8 as i32,
                                    mw, mh, 0, glow::RGBA, glow::UNSIGNED_BYTE,
                                    glow::PixelUnpackData::Slice(None),
                                );
                            }
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, num_mips - 1);

                            // FBO for blitting into mip 0
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

                        // --- Gaussian blur pyramid: blur each mip level from the previous ---
                        if state.blur_program.is_none() {
                            if let Ok((prog, vao)) = create_blur_program(gl) {
                                state.blur_program = Some(prog);
                                state.blur_vao = Some(vao);
                            }
                        }
                        // One temp texture for ping-pong during H/V separable blur
                        if state.blur_tex.is_none() || state.scene_color_size != (vw, vh) {
                            if let Some(f) = state.blur_fbo { gl.delete_framebuffer(f); }
                            if let Some(t) = state.blur_tex { gl.delete_texture(t); }
                            let t = gl.create_texture().unwrap();
                            gl.bind_texture(glow::TEXTURE_2D, Some(t));
                            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                            let f = gl.create_framebuffer().unwrap();
                            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(f));
                            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(t), 0);
                            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                            state.blur_fbo = Some(f);
                            state.blur_tex = Some(t);
                        }

                        if let (Some(prog), Some(vao), Some(blur_fbo), Some(blur_tex), Some(sc_fbo)) =
                            (state.blur_program, state.blur_vao, state.blur_fbo, state.blur_tex, state.scene_color_fbo)
                        {
                            let sc_tex = state.scene_color_tex.unwrap();
                            gl.use_program(Some(prog));
                            gl.bind_vertex_array(Some(vao));
                            gl.disable(glow::DEPTH_TEST);
                            gl.disable(glow::BLEND);
                            let dir_loc = gl.get_uniform_location(prog, "u_dir");

                            // For each mip level 1..N: Gaussian blur from mip (level-1)
                            let num_mips = ((vw.max(vh) as f32).log2().floor() as i32 + 1).min(8);
                            for mip in 1..num_mips {
                                let mw = (vw >> mip).max(1);
                                let mh = (vh >> mip).max(1);
                                let px = 1.0 / mw as f32;
                                let py = 1.0 / mh as f32;

                                // H pass: read mip (level-1) from scene_color → write to blur_tex
                                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(blur_fbo));
                                // Resize blur_tex for this mip size
                                gl.active_texture(glow::TEXTURE0);
                                gl.bind_texture(glow::TEXTURE_2D, Some(blur_tex));
                                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, mw, mh, 0,
                                    glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                    glow::TEXTURE_2D, Some(blur_tex), 0);
                                gl.viewport(0, 0, mw, mh);
                                // Read from scene_color mip (level-1)
                                gl.active_texture(glow::TEXTURE0);
                                gl.bind_texture(glow::TEXTURE_2D, Some(sc_tex));
                                // Force sampling from specific mip level
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, mip - 1);
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, mip - 1);
                                if let Some(ref loc) = dir_loc { gl.uniform_2_f32(Some(loc), px, 0.0); }
                                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                                // Restore mip range
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, 0);
                                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, num_mips - 1);

                                // V pass: read blur_tex → write to scene_color mip (level)
                                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(sc_fbo));
                                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                    glow::TEXTURE_2D, Some(sc_tex), mip);
                                gl.active_texture(glow::TEXTURE0);
                                gl.bind_texture(glow::TEXTURE_2D, Some(blur_tex));
                                if let Some(ref loc) = dir_loc { gl.uniform_2_f32(Some(loc), 0.0, py); }
                                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                            }

                            // Restore FBO to mip 0 for future blits
                            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                glow::TEXTURE_2D, Some(sc_tex), 0);
                            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                            gl.viewport(0, 0, vw, vh);
                            gl.enable(glow::DEPTH_TEST);
                            gl.use_program(Some(state.renderer.programs().pbr));
                        }

                        // Bind scene_color with mip chain to unit 8
                        gl.active_texture(glow::TEXTURE0 + 8);
                        gl.bind_texture(glow::TEXTURE_2D, Some(state.scene_color_tex.unwrap()));
                    }
                }

                // Pass 2: transmission + transparent
                for batch in &render_batches {
                    let mat = get_mat(batch);
                    if mat.alpha_mode < 1.5 && mat.transmission_factor <= 0.0 { continue; }

                    let is_blend = mat.alpha_mode > 1.5;
                    let is_transmission = mat.transmission_factor > 0.0;
                    unsafe {
                        if is_blend {
                            gl.enable(glow::BLEND);
                            gl.blend_func_separate(
                                glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA,
                                glow::ONE, glow::ONE_MINUS_SRC_ALPHA,
                            );
                            gl.depth_mask(false);
                            gl.disable(glow::CULL_FACE);
                        } else {
                            gl.disable(glow::BLEND);
                            gl.depth_mask(true);
                            // Cull back faces for transmission to prevent overdraw ghosting
                            if is_transmission {
                                gl.enable(glow::CULL_FACE);
                                gl.cull_face(glow::BACK);
                            } else {
                                gl.disable(glow::CULL_FACE);
                            }
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
                    bind_tex(10, mat.transmission_tex_idx);

                    let batch_camera = camera_for_batch(&camera, batch);
                    apply_batch_winding(gl, batch.orientation_sign);
                    quilting_renderer::pass::upload_batch_ubo(
                        gl, state.renderer.vtx_ubo(), &batch_camera,
                        batch.perm_parity, batch.perm_index, 1,
                        &batch.euclidean_model, &batch.euclidean_normal,
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
                    gl.disable(glow::CULL_FACE);
                    gl.disable(glow::BLEND);
                }

                // --- Fuzzy-vision post-process ---
                if state.fuzzy_enabled {
                    if let Some(ref mut fv) = state.fuzzy {
                        unsafe {
                            let mut vp = [0i32; 4];
                            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut vp);
                            let vw = vp[2].max(1);
                            let vh = vp[3].max(1);
                            fv.resize(gl, vw, vh);

                            // When MRT is active (conformal mode), PBR rendered to pbr_fbo.
                            // Color is in pbr_color_tex, weight in fuzzy_weight_tex.
                            // When radial mode, PBR rendered to default FB — need to blit.
                            let (scene_tex, weight_tex) = if true /* all fuzzy modes use MRT */ && state.pbr_color_tex.is_some() {
                                // Conformal: MRT gave us raw stretch in fuzzy_weight_tex.
                                // Run focused weight generator to apply Gaussian band selection.
                                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                                gl.draw_buffers(&[glow::BACK]);
                                let stretch_tex = state.fuzzy_weight_tex.unwrap();
                                // Need a separate FBO for the focused weight output
                                if state.fuzzy_scene_tex.is_none() || state.fuzzy_scene_size != (vw, vh) {
                                    if let Some(old) = state.fuzzy_scene_fbo { gl.delete_framebuffer(old); }
                                    if let Some(old) = state.fuzzy_scene_tex { gl.delete_texture(old); }
                                    let tex = gl.create_texture().unwrap();
                                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                    let fbo = gl.create_framebuffer().unwrap();
                                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                                    gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                        glow::TEXTURE_2D, Some(tex), 0);
                                    state.fuzzy_scene_fbo = Some(fbo);
                                    state.fuzzy_scene_tex = Some(tex);
                                    state.fuzzy_scene_size = (vw, vh);
                                }
                                // Generate focused weight from raw stretch
                                fv.generate_conformal_weight(
                                    gl, stretch_tex,
                                    state.fuzzy_scene_fbo.unwrap(), vw, vh,
                                );
                                // fuzzy_scene_tex now has the focused weight; use it for JFA
                                (state.pbr_color_tex.unwrap(), state.fuzzy_scene_tex.unwrap())
                            } else {
                                // Radial: blit scene from default FB to a texture
                                if state.fuzzy_scene_tex.is_none() || state.fuzzy_scene_size != (vw, vh) {
                                    if let Some(old) = state.fuzzy_scene_fbo { gl.delete_framebuffer(old); }
                                    if let Some(old) = state.fuzzy_scene_tex { gl.delete_texture(old); }
                                    let tex = gl.create_texture().unwrap();
                                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                    let fbo = gl.create_framebuffer().unwrap();
                                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                                    gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                        glow::TEXTURE_2D, Some(tex), 0);
                                    state.fuzzy_scene_fbo = Some(fbo);
                                    state.fuzzy_scene_tex = Some(tex);
                                    state.fuzzy_scene_size = (vw, vh);
                                }
                                // Ensure radial weight texture
                                if state.fuzzy_weight_tex.is_none() || state.fuzzy_weight_size != (vw, vh) {
                                    if let Some(old) = state.fuzzy_weight_fbo { gl.delete_framebuffer(old); }
                                    if let Some(old) = state.fuzzy_weight_tex { gl.delete_texture(old); }
                                    let tex = gl.create_texture().unwrap();
                                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                    let fbo = gl.create_framebuffer().unwrap();
                                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                                    gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                                        glow::TEXTURE_2D, Some(tex), 0);
                                    state.fuzzy_weight_fbo = Some(fbo);
                                    state.fuzzy_weight_tex = Some(tex);
                                    state.fuzzy_weight_size = (vw, vh);
                                }
                                let sc_fbo = state.fuzzy_scene_fbo.unwrap();
                                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
                                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(sc_fbo));
                                gl.blit_framebuffer(0, 0, vw, vh, 0, 0, vw, vh,
                                    glow::COLOR_BUFFER_BIT, glow::NEAREST);
                                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                                // Generate radial weight
                                fv.generate_radial_weight(gl, state.fuzzy_weight_fbo.unwrap(), vw, vh);
                                (state.fuzzy_scene_tex.unwrap(), state.fuzzy_weight_tex.unwrap())
                            };

                            // Run JFA blur → default FB
                            if state.fuzzy_debug > 0 {
                                // Debug: run pipeline but override output with intermediate texture
                                fv.run(gl, weight_tex, scene_tex, None);
                                fv.debug_blit(gl, state.fuzzy_debug, None);
                            } else {
                                fv.run(gl, weight_tex, scene_tex, None);
                            }
                            gl.viewport(0, 0, vw, vh);
                            gl.enable(glow::DEPTH_TEST);
                        }
                    }
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

        // Highlight pass: overlay picked QB patch with cyan
        if state.highlight_face >= 0 && state.highlight_prog.is_some() {
            render_highlight(state.renderer.gl(), state, &camera);
        }

        state.renderer.end_frame();
    });
}

const HIGHLIGHT_VS: &str = r#"#version 300 es
out vec2 v_uv;
void main() {
    float x = (gl_VertexID == 1) ? 3.0 : -1.0;
    float y = (gl_VertexID == 2) ? 3.0 : -1.0;
    v_uv = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"#;

const HIGHLIGHT_FS: &str = r#"#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 o_color;
uniform sampler2D u_pick;
uniform vec3 u_target;

void main() {
    vec4 px = texture(u_pick, v_uv);
    if (px.a < 0.5) discard; // background
    if (abs(px.r - u_target.r) > 0.003 ||
        abs(px.g - u_target.g) > 0.003 ||
        abs(px.b - u_target.b) > 0.003) discard;
    o_color = vec4(0.0, 1.0, 1.0, 0.5);
}
"#;


/// Render highlight overlay for the picked face.
/// Requires pick FBO + highlight program to exist (created in mr_pick).
/// Renders pick pass to FBO, then fullscreen overlay where face ID matches.
fn render_highlight(gl: &glow::Context, state: &MainState, camera: &quilting_renderer::pass::Camera) {
    let target_id = state.highlight_face;
    let (highlight_prog, highlight_vao, pick_fbo, pick_tex) = match (
        state.highlight_prog, state.highlight_vao, state.pick_fbo, state.pick_tex,
    ) {
        (Some(p), Some(v), Some(f), Some(t)) => (p, v, f, t),
        _ => return, // not initialized (mr_pick hasn't been called yet)
    };

    unsafe {
        let (vw, vh) = state.pick_size;
        if vw == 0 { return; }

        // Render pick pass to FBO
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(pick_fbo));
        gl.viewport(0, 0, vw, vh);
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        gl.enable(glow::DEPTH_TEST);
        gl.disable(glow::BLEND);
        gl.use_program(Some(state.renderer.programs().pick));

        let mut face_offset = 0i32;
        for batch in &state.batches {
            let batch_camera = camera_for_node(camera, state, batch.node_index);
            let conformal = conformal_state_for_node(state, batch.node_index);
            apply_batch_winding(
                gl,
                conformal.orientation_sign
                    * affine_orientation_sign(&conformal.euclidean_model),
            );
            let euclidean_normal = affine_normal_matrix(&conformal.euclidean_model);
            quilting_renderer::pass::upload_batch_ubo(
                gl, state.renderer.vtx_ubo(), &batch_camera,
                batch.perm_parity, batch.perm_index, 1,
                &conformal.euclidean_model, &euclidean_normal,
            );
            state.renderer.vtx_ubo().set_face_offset(gl, face_offset);
            state.renderer.vtx_ubo().bind(gl);

            gl.bind_vertex_array(Some(batch.mesh.tri_vao));
            gl.draw_elements_instanced(glow::TRIANGLES, batch.mesh.num_tri_indices,
                glow::UNSIGNED_INT, 0, batch.mesh.num_instances);
            face_offset += batch.mesh.num_instances;
        }

        state.renderer.vtx_ubo().set_face_offset(gl, 0);

        // Fullscreen overlay: cyan where pick matches target face
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.viewport(0, 0, vw, vh);
        gl.use_program(Some(highlight_prog));
        gl.disable(glow::DEPTH_TEST);
        gl.enable(glow::BLEND);
        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(pick_tex));
        if let Some(loc) = gl.get_uniform_location(highlight_prog, "u_pick") {
            gl.uniform_1_i32(Some(&loc), 0);
        }
        let tr = (target_id & 255) as f32 / 255.0;
        let tg = ((target_id >> 8) & 255) as f32 / 255.0;
        let tb = ((target_id >> 16) & 255) as f32 / 255.0;
        if let Some(loc) = gl.get_uniform_location(highlight_prog, "u_target") {
            gl.uniform_3_f32(Some(&loc), tr, tg, tb);
        }

        gl.bind_vertex_array(Some(highlight_vao));
        gl.draw_arrays(glow::TRIANGLES, 0, 3);

        gl.disable(glow::BLEND);
        gl.enable(glow::DEPTH_TEST);
    }
}

fn lod_color(lod: &[u32; 3]) -> [f32; 3] {
    let max_lod = *lod.iter().max().unwrap_or(&1) as f32;
    let t = (max_lod.log2() / 8.0).clamp(0.0, 1.0);
    if t < 0.5 { let s = t * 2.0; [0.0, s, 1.0 - s] }
    else { let s = (t - 0.5) * 2.0; [s, 1.0 - s, 0.0] }
}
