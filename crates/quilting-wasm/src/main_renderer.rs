//! Main-thread rendering module for the production hyperscope.
//!
//! All exports are prefixed with `mr_` in JS to distinguish from
//! worker-side exports in lib.rs.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use tracing::{info, debug, warn};
use crate::{perf_mark, perf_measure};
use crate::auxiliary_programs::{
    auxiliary_program_descriptor, AuxiliaryProgram, POST_PROCESS_UNIFORMS_BINDING,
};

use glow::HasContext;
use quilting_renderer::buffer::{
    create_patch_input_vao, EnvironmentMaps, MeshBuffers, MeshDraw, PbrParams,
    PersistentBatchInstances, TessAtlasBuffers, TessBuffers,
};
use quilting_renderer::pass::{
    affine_normal_matrix, affine_orientation_sign, apply_batch_winding,
    camera_for_batch, record_indexed_submission, same_vertex_uniform_state, Camera, RenderBatch,
    RenderMode, IDENTITY_MATRIX,
};
use quilting_renderer::Renderer;
use quilting_renderer::texture::TextureCache;
use quilting_core::batch;
use quilting_core::instance_layout;
use quilting_core::render::{
    FocusFieldPacket, PbrDrawClass, RenderBatchSnapshot, RenderEntityTransform,
    RenderFrameOptions, RenderGeometry, RenderSceneSnapshot, RenderStyle,
    RenderSubmissionStats, RenderView,
};
use quilting_core::source_bounds::source_focus_bounds;
use crate::render_shadow::RenderShadowObserver;
use crate::round_shadow::{browser_now_ms, RoundShadowObserver};
use crate::surface_runtime::{
    validate_pose_stamp, ComposedSurfaceWalkSnapshot, SurfaceRuntime, SurfaceRuntimeSnapshot,
    SurfaceWalkReflectionTransportSnapshot,
};
use crate::surface_walk::{ComposedSurfaceWalkResultJs, SurfaceWalkReflectionTransportResultJs};
use hyperscape::interchange::{
    GltfHyperscopePacket, HyperscapeGltfRuntime, RuntimeDiagnosticSnapshot,
};
use hyperscape::{
    CameraBasis, CameraRig, ChamberSide, ContactClassification, FocusSphere, PerspectiveLens,
    SphereReflectionState, SurfaceWalkControls, SurfaceWalkInput,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

/// Floats per material in the array `mr_setMaterials` receives.
const MATERIAL_STRIDE: usize = 50;

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
    }
}

fn pbr_material_for_index<'a>(
    materials: &'a [PbrParams],
    default_material: &'a PbrParams,
    requested: usize,
) -> (usize, &'a PbrParams) {
    if let Some(material) = materials.get(requested) {
        (requested, material)
    } else if let Some(material) = materials.first() {
        (0, material)
    } else {
        (usize::MAX, default_material)
    }
}

fn bind_pbr_material_state(
    gl: &glow::Context,
    renderer: &Renderer,
    texture_cache: &TextureCache,
    material: &PbrParams,
    texture_defaults: &[glow::Texture; 5],
    has_env_map: bool,
    env_mip_count: f32,
    bind_transmission: bool,
    selected: bool,
    focus_sphere: [f32; 4],
    focus_field_enabled: bool,
) {
    let selection_tint = if selected {
        [0.16, 0.78, 1.0, 0.13]
    } else {
        [0.0; 4]
    };
    renderer.pbr_ubo().upload_with_frame_state(
        gl,
        material,
        has_env_map,
        env_mip_count,
        selection_tint,
        focus_sphere,
        [if focus_field_enabled { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
    );
    renderer.pbr_ubo().bind(gl);

    for (unit, &texture) in texture_defaults.iter().enumerate() {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        }
    }
    let bind_texture = |unit: u32, index: i32| {
        if index >= 0 {
            let texture = texture_cache.get(Some(index as usize));
            unsafe {
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            }
        }
    };
    bind_texture(0, material.base_color_tex_idx);
    bind_texture(1, material.metallic_roughness_tex_idx);
    bind_texture(2, material.normal_tex_idx);
    bind_texture(3, material.emissive_tex_idx);
    bind_texture(4, material.occlusion_tex_idx);
    if bind_transmission {
        bind_texture(10, material.transmission_tex_idx);
    }
}

/// Shared non-program resources for the two no-attribute fullscreen passes.
/// Their programs are non-owning handles retained by the Renderer memo.
struct FullscreenAuxResources {
    vao: glow::VertexArray,
    params_ubo: glow::Buffer,
}

impl FullscreenAuxResources {
    fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let vao = gl.create_vertex_array()
                .map_err(|error| format!("fullscreen auxiliary VAO: {error}"))?;
            let params_ubo = match gl.create_buffer() {
                Ok(buffer) => buffer,
                Err(error) => {
                    gl.delete_vertex_array(vao);
                    return Err(format!("fullscreen auxiliary UBO: {error}"));
                }
            };
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(params_ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, 16, glow::DYNAMIC_DRAW);
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);
            Ok(Self { vao, params_ubo })
        }
    }

    fn upload_and_bind(&self, gl: &glow::Context, value: [f32; 4]) {
        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.params_ubo));
            gl.buffer_sub_data_u8_slice(
                glow::UNIFORM_BUFFER,
                0,
                bytemuck_cast_slice(&value),
            );
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                POST_PROCESS_UNIFORMS_BINDING,
                Some(self.params_ubo),
            );
        }
    }

    fn destroy(self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                POST_PROCESS_UNIFORMS_BINDING,
                None,
            );
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.params_ubo);
        }
    }
}

struct MainState {
    renderer: Renderer,
    /// Authoritative drawable size, updated with the GL viewport by mr_resize.
    /// Optional passes consume this instead of synchronously querying GL state.
    viewport_size: (i32, i32),
    texture_cache: TextureCache,
    env_maps: EnvironmentMaps,
    batches: BTreeMap<batch::RenderBatchKey, GpuBatch>,
    /// Retained WebGL draw views derived from backend-neutral batch keys.
    /// Rebuilt only when membership or per-node conformal state changes.
    render_batches: Vec<RenderBatch>,
    render_commands_dirty: bool,
    render_command_builds: u64,
    render_calls: u64,
    pbr_draw_calls: u64,
    pbr_material_updates: u64,
    pbr_vertex_uniform_updates: u64,
    /// Actual indexed patch work submitted by the most recent scene render.
    /// Picking, highlighting, and fullscreen post-processing are separate
    /// auxiliary passes and are deliberately excluded.
    last_render_submission: RenderSubmissionStats,
    /// Saturating session totals for explicit diagnostic queries.
    render_submission_totals: RenderSubmissionStats,
    /// Opt-in comparison between backend-neutral frame commands and actual
    /// indexed WebGL patch submissions.
    render_shadow: RenderShadowObserver,
    render_shadow_scene_dirty: bool,
    batch_groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    batch_staging: Vec<f32>,
    cached_instances: Vec<f32>,
    /// Last valid topology uploaded for each face. Worker visibility is
    /// asynchronous, so an invisible sentinel must not demote a patch that the
    /// current-pose render GPU may need to resurrect in the same frame.
    resident_face_lods: Vec<Option<batch::ResidentLod>>,
    /// Current C0-continuous visualization LODs, indexed by source face.
    resident_vertex_lods: Vec<[u32; 3]>,
    resident_vertex_lod_scratch: Vec<u32>,
    /// Validated within-face grading policy. Shared-edge equality remains an
    /// independent invariant; this controls only anisotropy/promotion halos.
    lod_grading: batch::FaceLodGrading,
    /// Visibility from the latest worker classification, retained separately
    /// from topology because invisible faces deliberately keep their last LOD.
    classified_face_visibility: Vec<bool>,
    classified_culled_faces: usize,
    /// Opt-in observer for the conservative CPU round hierarchy. It never
    /// changes batch membership or draw calls.
    round_shadow: RoundShadowObserver,
    /// Faces whose latest asynchronous classification changed. This sparse
    /// frontier seeds half-edge reconciliation without rescanning every edge.
    lod_dirty_faces: Vec<usize>,
    lod_balance_scratch: batch::ResidentLodBalanceScratch,
    /// Source-mesh adjacency used to keep retained asynchronous topology
    /// crack-free, including exact duplicate vertices at glTF attribute seams.
    lod_topology: Option<quilting_mesh::HalfEdgeMesh>,
    /// Rust-authoritative stable source address and output-chart walker. The
    /// adjacency layer is built lazily on first attachment.
    surface_runtime: SurfaceRuntime,
    /// Material/node/atlas changes require a rebuild even when face topology
    /// classifications are unchanged.
    batch_layout_dirty: bool,
    batch_update_stats: BatchUpdateStats,
    face_materials: Vec<usize>,
    /// Stable ordinary glTF node index for each source triangle.
    face_nodes: Vec<usize>,
    materials: Vec<PbrParams>,
    num_faces: usize,
    render_mode: RenderMode,
    /// Analytic character-matcap profile selected by the browser shell.
    matcap_style: f32,
    mobius: [f32; 16],
    /// Explicit parity from the authoring generator word. Legacy direct matrix
    /// input falls back to the old `c != 0` heuristic.
    mobius_orientation: i8,
    /// Extracted state keyed by `(projection camera node, subject node)`.
    hyperscape_packets: BTreeMap<(usize, usize), EntityConformalState>,
    active_hyperscape_camera: Option<usize>,
    /// Presentation-layer state is independent of authored Hyperscape frames.
    /// It keeps distinct asset/node identity while the WebGL backend packs all
    /// resident faces into shared buffers.
    presentation_nodes: BTreeMap<usize, PresentationNodeState>,
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
    fullscreen_aux: Option<FullscreenAuxResources>,
    // MRT: PBR renders to this FBO with color + weight attachments
    pbr_fbo: Option<glow::Framebuffer>,
    pbr_color_tex: Option<glow::Texture>,
    pbr_depth_rb: Option<glow::Renderbuffer>,
    pbr_fbo_size: (i32, i32),
    // Fuzzy-vision JFA blur pipeline
    fuzzy: Option<fuzzy_vision::JfaPipeline>,
    fuzzy_enabled: bool,
    fuzzy_mode: u32, // 0=DoF, 1=conformal, 2=hybrid, 3=selection field
    fuzzy_debug: u32, // 0=off, 1=smoothed weight, 2=jfa, 3=firmness
    fuzzy_weight_fbo: Option<glow::Framebuffer>,
    fuzzy_weight_tex: Option<glow::Texture>,
    fuzzy_weight_size: (i32, i32),
    // Pick buffer for face inspection
    pick_fbo: Option<glow::Framebuffer>,
    pick_tex: Option<glow::Texture>,
    pick_bary_tex: Option<glow::Texture>,
    pick_depth: Option<glow::Renderbuffer>,
    pick_size: (i32, i32),
    last_pick_barycentric: Option<[f32; 3]>,
    highlight_face: i32, // -1 = none
    /// Stable glTF node receiving the lightweight per-patch selection tint.
    selected_node: i32, // -1 = none
    /// Persistent posed ordinary-space sphere shared by focus and inversion.
    focus_sphere: [f32; 4],
    focus_field_enabled: bool,
    highlight_prog: Option<glow::Program>,
}

impl MainState {
    /// Retire one complete renderer epoch with the WebGL context that created
    /// its resources. This must run before a replacement state becomes visible.
    fn retire(mut self) {
        let gl = self.renderer.gl();

        unsafe {
            gl.use_program(None);
            gl.bind_vertex_array(None);
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, None);
            gl.bind_buffer(glow::UNIFORM_BUFFER, None);
        }

        // RenderBatch/MeshDraw values are non-owning views into GpuBatch.
        self.render_batches.clear();
        for (_, batch) in std::mem::take(&mut self.batches) {
            batch.destroy(gl);
        }

        // Patch views and batch VAOs reference the immutable atlas buffers.
        // Retire every view before deleting their old-context owner.
        TESS_CACHE.with(|cache| cache.borrow_mut().clear());
        TESS_ATLAS.with(|atlas| {
            if let Some(atlas) = atlas.borrow_mut().take() {
                atlas.destroy(gl);
            }
        });

        // Delete framebuffer objects before their attached textures and
        // renderbuffers so attachment references cannot prolong allocation.
        unsafe {
            for framebuffer in [
                self.scene_color_fbo.take(),
                self.fuzzy_scene_fbo.take(),
                self.blur_fbo.take(),
                self.pbr_fbo.take(),
                self.fuzzy_weight_fbo.take(),
                self.pick_fbo.take(),
            ].into_iter().flatten() {
                gl.delete_framebuffer(framebuffer);
            }
        }

        // The delegated owner follows the same framebuffer-before-attachment
        // order for its internal render graph.
        if let Some(fuzzy) = self.fuzzy.take() {
            fuzzy.destroy(gl);
        }

        // Application-owned fullscreen bindings retire before Renderer drops
        // the memo-owned programs and their shared shader modules.
        if let Some(resources) = self.fullscreen_aux.take() {
            resources.destroy(gl);
        }
        let _ = self.blur_program.take();
        let _ = self.highlight_prog.take();

        unsafe {
            for renderbuffer in [self.pbr_depth_rb.take(), self.pick_depth.take()]
                .into_iter().flatten()
            {
                gl.delete_renderbuffer(renderbuffer);
            }
            for texture in [
                self.scene_color_tex.take(),
                self.fuzzy_scene_tex.take(),
                self.blur_tex.take(),
                self.pbr_color_tex.take(),
                self.fuzzy_weight_tex.take(),
                self.pick_tex.take(),
                self.pick_bary_tex.take(),
                self.env_maps.prefiltered.take(),
                self.env_maps.irradiance.take(),
                self.env_maps.sheen_lut.take(),
            ].into_iter().flatten() {
                gl.delete_texture(texture);
            }
        }

        self.texture_cache.destroy(gl);
        // `self` now drops. Renderer is the only remaining GL owner and its
        // Drop implementation tears down its transform feedback, program memo,
        // UBOs, and animation textures with this same context.
    }
}

fn resolve_auxiliary_program(
    state: &mut MainState,
    kind: AuxiliaryProgram,
) -> Result<glow::Program, String> {
    let descriptor = auxiliary_program_descriptor(kind)
        .map_err(|error| format!("{kind:?} descriptor: {error}"))?;
    let program = state.renderer.resolve_program(descriptor)
        .map_err(|error| format!("{kind:?} program: {error}"))?;
    if state.fullscreen_aux.is_none() {
        state.fullscreen_aux = Some(FullscreenAuxResources::new(state.renderer.gl())?);
    }
    Ok(program)
}

#[derive(Default)]
struct BatchUpdateStats {
    calls: u64,
    unchanged_calls: u64,
    retained_buckets: u64,
    updated_buckets: u64,
    created_buckets: u64,
    reallocated_buckets: u64,
    retired_buckets: u64,
    uploaded_instances: u64,
    last_culled_faces: u64,
    last_lod_corrections: u64,
    last_missing_atlas_entries: u64,
    last_gpu_failures: u64,
}

struct GpuBatch {
    instances: PersistentBatchInstances,
    mesh: MeshBuffers,
    prepare_vao: glow::VertexArray,
    members: Vec<batch::RenderBatchMember>,
    perm_parity: f32,
    material_index: usize,
    node_index: usize,
}

impl GpuBatch {
    fn destroy(self, gl: &glow::Context) {
        unsafe {
            gl.delete_vertex_array(self.prepare_vao);
        }
        // Both render VAOs reference the persistent prepared-instance buffer.
        self.mesh.destroy_vaos_only(gl);
        self.instances.destroy(gl);
    }

    fn can_fit(&self, instance_count: usize) -> bool {
        instance_count <= self.instances.capacity_instances
    }

    fn upload_members(
        &mut self,
        gl: &glow::Context,
        members: &[batch::RenderBatchMember],
        instance_data: &[f32],
    ) -> Result<(), String> {
        self.instances.upload_active(gl, instance_data)?;
        self.mesh.num_instances = members.len() as i32;
        self.members.clear();
        self.members.extend_from_slice(members);
        Ok(())
    }
}

fn fill_batch_instance_data(
    residents: &[Option<batch::ResidentLod>],
    key: batch::RenderBatchKey,
    members: &[batch::RenderBatchMember],
    staging: &mut [f32],
) -> Result<usize, String> {
    let stride = instance_layout::BATCH_TOPOLOGY_STRIDE;
    let required = members.len().checked_mul(stride)
        .ok_or_else(|| "batch instance size overflow".to_string())?;
    if staging.len() < required {
        return Err("batch staging buffer is too small".to_string());
    }

    for (instance_index, member) in members.iter().enumerate() {
        let face = member.face_index as usize;
        let resident = residents.get(face).and_then(|resident| *resident)
            .ok_or_else(|| format!("face {face} has no resident LOD"))?;
        if resident.canonical != key.lod
            || resident.parity_bucket.min(1) as u8 != key.parity_bucket
            || resident.perm_index.min(5) as u8 != member.permutation_index
        {
            return Err(format!("face {face} no longer matches its batch key"));
        }

        let destination = instance_index * stride;
        let edge_lods = resident.edge_lods().map(|lod| lod as f32);
        let vertex_lods = member.vertex_lods.map(|lod| lod as f32);
        staging[destination..destination + stride].copy_from_slice(&[
            edge_lods[0],
            edge_lods[1],
            edge_lods[2],
            member.permutation_index as f32,
            member.face_index as f32,
            vertex_lods[0],
            vertex_lods[1],
            vertex_lods[2],
        ]);
    }
    Ok(required)
}

fn create_gpu_batch(
    gl: &glow::Context,
    tess: &TessBuffers,
    key: batch::RenderBatchKey,
    members: &[batch::RenderBatchMember],
    instance_data: &[f32],
) -> Result<GpuBatch, String> {
    let capacity_instances = members.len().max(1).next_power_of_two();
    let instances = PersistentBatchInstances::with_capacity(gl, instance_data, capacity_instances)?;
    let prepare_vao = match create_patch_input_vao(gl, &instances.topology_buf, 0) {
        Ok(vao) => vao,
        Err(error) => {
            instances.destroy(gl);
            return Err(error);
        }
    };
    let mesh = match MeshBuffers::from_shared(
        gl,
        tess,
        &instances.prepared_buf,
        0,
        members.len() as i32,
    ) {
        Ok(mesh) => mesh,
        Err(error) => {
            unsafe { gl.delete_vertex_array(prepare_vao); }
            instances.destroy(gl);
            return Err(error);
        }
    };
    Ok(GpuBatch {
        instances,
        mesh,
        prepare_vao,
        members: members.to_vec(),
        perm_parity: key.parity(),
        material_index: key.material_index,
        node_index: key.node_index,
    })
}

#[derive(Clone, Copy, PartialEq)]
struct EntityConformalState {
    mobius: [f32; 16],
    orientation_sign: i8,
    euclidean_model: [f32; 16],
}

#[derive(Clone, Copy, PartialEq)]
struct PresentationNodeState {
    euclidean_model: [f32; 16],
    visible: bool,
    opacity: f32,
}

thread_local! {
    static STATE: RefCell<Option<MainState>> = RefCell::new(None);
    static TESS_CACHE: RefCell<std::collections::HashMap<[u32; 3], TessBuffers>> =
        RefCell::new(std::collections::HashMap::new());
    /// Owns the immutable buffers referenced by `TESS_CACHE` patch slices.
    static TESS_ATLAS: RefCell<Option<TessAtlasBuffers>> = RefCell::new(None);
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

    STATE.with(|state| {
        let replacement = MainState {
            viewport_size: (canvas.width() as i32, canvas.height() as i32),
            renderer, texture_cache,
            env_maps: EnvironmentMaps::default(),
            batches: BTreeMap::new(),
            render_batches: Vec::new(),
            render_commands_dirty: true,
            render_command_builds: 0,
            render_calls: 0,
            pbr_draw_calls: 0,
            pbr_material_updates: 0,
            pbr_vertex_uniform_updates: 0,
            last_render_submission: RenderSubmissionStats::default(),
            render_submission_totals: RenderSubmissionStats::default(),
            render_shadow: RenderShadowObserver::default(),
            render_shadow_scene_dirty: true,
            batch_groups: BTreeMap::new(),
            batch_staging: Vec::new(),
            cached_instances: Vec::new(),
            resident_face_lods: Vec::new(),
            resident_vertex_lods: Vec::new(),
            resident_vertex_lod_scratch: Vec::new(),
            lod_grading: batch::FaceLodGrading::default(),
            classified_face_visibility: Vec::new(),
            classified_culled_faces: 0,
            round_shadow: RoundShadowObserver::default(),
            lod_dirty_faces: Vec::new(),
            lod_balance_scratch: batch::ResidentLodBalanceScratch::default(),
            lod_topology: None,
            surface_runtime: SurfaceRuntime::default(),
            batch_layout_dirty: true,
            batch_update_stats: BatchUpdateStats::default(),
            face_materials: Vec::new(),
            face_nodes: Vec::new(),
            materials: Vec::new(),
            num_faces: 0,
            render_mode: RenderMode::Pbr,
            matcap_style: 1.0,
            mobius: IDENTITY_MOBIUS,
            mobius_orientation: 1,
            hyperscape_packets: BTreeMap::new(),
            active_hyperscape_camera: None,
            presentation_nodes: BTreeMap::new(),
            scene_color_fbo: None,
            scene_color_tex: None,
            scene_color_size: (0, 0),
            fuzzy_scene_fbo: None,
            fuzzy_scene_tex: None,
            fuzzy_scene_size: (0, 0),
            blur_fbo: None,
            blur_tex: None,
            blur_program: None,
            fullscreen_aux: None,
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
            pick_bary_tex: None,
            pick_depth: None,
            pick_size: (0, 0),
            last_pick_barycentric: None,
            highlight_face: -1,
            selected_node: -1,
            focus_sphere: [0.5, 0.0, 0.0, 2.0],
            focus_field_enabled: false,
            highlight_prog: None,
        };
        let previous = state.borrow_mut().take();
        if let Some(previous) = previous {
            previous.retire();
        }
        *state.borrow_mut() = Some(replacement);
    });
    info!("Renderer initialized on canvas '{}'", canvas_id);
    true
}

#[wasm_bindgen(js_name = "mr_resize")]
pub fn mr_resize(width: i32, height: i32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.viewport_size = (width.max(1), height.max(1));
            st.renderer.resize(width, height);
        }
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

#[wasm_bindgen(js_name = "mr_setMatcapStyle")]
pub fn mr_set_matcap_style(style: &str) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.matcap_style = match style {
                "aqua" => 0.0,
                "citric-acid" => 1.0,
                "golden-soft" => 2.0,
                "soft-studio" => 3.0,
                _ => 1.0,
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
            st.render_commands_dirty = true;
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
            renderer.render_commands_dirty = true;
        }
    });
}

/// Apply resolved presentation-layer state to stable renderer node IDs.
/// Records are `[node, visible, opacity, column_major_model_matrix x16]`.
/// Opacity is retained for the material adapter; zero opacity already behaves
/// as hidden. The global Möbius map remains shared presentation view state.
#[wasm_bindgen(js_name = "mr_setPresentationNodeStates")]
pub fn mr_set_presentation_node_states(records: &[f32]) -> bool {
    const STRIDE: usize = 19;
    if records.len() % STRIDE != 0 {
        warn!("Presentation node-state payload has an invalid length");
        return false;
    }
    let mut next = BTreeMap::new();
    for record in records.chunks_exact(STRIDE) {
        let node = record[0];
        let opacity = record[2];
        if !node.is_finite()
            || node < 0.0
            || node.fract() != 0.0
            || !opacity.is_finite()
            || !record[3..19].iter().all(|value| value.is_finite())
        {
            warn!("Presentation node-state payload contains invalid values");
            return false;
        }
        let mut euclidean_model = [0.0; 16];
        euclidean_model.copy_from_slice(&record[3..19]);
        next.insert(
            node as usize,
            PresentationNodeState {
                euclidean_model,
                visible: record[1] > 0.5 && opacity > 0.0,
                opacity: opacity.clamp(0.0, 1.0),
            },
        );
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return false;
        };
        if state.presentation_nodes != next {
            state.presentation_nodes = next;
            state.render_commands_dirty = true;
        }
        true
    })
}

fn clear_hyperscape_packets() {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.hyperscape_packets.clear();
            renderer.active_hyperscape_camera = None;
            renderer.render_commands_dirty = true;
        }
    });
}

fn apply_hyperscape_packets(packets: &[GltfHyperscopePacket]) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(renderer) = state.as_mut() else { return };

        let packet_state = |extracted: &GltfHyperscopePacket| EntityConformalState {
            mobius: extracted.packet.mobius,
            orientation_sign: extracted.packet.orientation_sign,
            euclidean_model: extracted.packet.euclidean_model,
        };
        let packets_changed = renderer.hyperscape_packets.len() != packets.len()
            || packets.iter().any(|extracted| {
                renderer
                    .hyperscape_packets
                    .get(&(extracted.camera_node, extracted.subject_node))
                    != Some(&packet_state(extracted))
            });
        if packets_changed {
            renderer.hyperscape_packets.clear();
            for extracted in packets {
                renderer.hyperscape_packets.insert(
                    (extracted.camera_node, extracted.subject_node),
                    packet_state(extracted),
                );
            }
        }

        let previous_camera = renderer.active_hyperscape_camera;
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
        let previous_global = (renderer.mobius, renderer.mobius_orientation);
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
        renderer.render_commands_dirty |= packets_changed
            || renderer.active_hyperscape_camera != previous_camera
            || (renderer.mobius, renderer.mobius_orientation) != previous_global;
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
    let euclidean_model = renderer
        .presentation_nodes
        .get(&node_index)
        .map_or(IDENTITY_MATRIX, |state| state.euclidean_model);
    EntityConformalState {
            mobius: renderer.mobius,
            orientation_sign: renderer.mobius_orientation,
            euclidean_model,
    }
}

/// Derive selection and scene-scale bounds from the same Euclidean model
/// matrices that the renderer, picker, and surface walker consume. Hidden
/// presentation nodes are excluded so guides cannot distort navigation scale.
#[wasm_bindgen(js_name = "mr_sourceFocusBounds")]
pub fn mr_source_focus_bounds() -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return JsValue::NULL;
        };
        let mut models = BTreeMap::new();
        for &node in &state.face_nodes {
            models.entry(node).or_insert_with(|| {
                if state.presentation_nodes.get(&node)
                    .is_some_and(|presentation| !presentation.visible)
                {
                    None
                } else {
                    Some(conformal_state_for_node(state, node).euclidean_model)
                }
            });
        }
        match source_focus_bounds(&state.cached_instances, &state.face_nodes, &models) {
            Ok(snapshot) => serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL),
            Err(error) => {
                warn!("Could not derive renderer source focus bounds: {error}");
                JsValue::NULL
            }
        }
    })
}

/// Synchronize the retained draw-command list with GPU batch ownership and
/// authored conformal state. Camera and animation pose are deliberately absent
/// from these commands, so ordinary frames can reuse them byte-for-byte.
fn sync_render_batches(renderer: &mut MainState) {
    if !renderer.render_commands_dirty {
        return;
    }

    let mut render_batches = std::mem::take(&mut renderer.render_batches);
    render_batches.clear();
    render_batches.reserve(renderer.batches.len());
    for batch in renderer.batches.values() {
        let conformal = conformal_state_for_node(renderer, batch.node_index);
        let euclidean_orientation = affine_orientation_sign(&conformal.euclidean_model);
        let mut mesh = MeshDraw::from(&batch.mesh);
        if renderer
            .presentation_nodes
            .get(&batch.node_index)
            .is_some_and(|state| !state.visible || state.opacity <= 0.0)
        {
            mesh.num_instances = 0;
        }
        render_batches.push(RenderBatch {
            mesh,
            perm_parity: batch.perm_parity,
            material_index: batch.material_index,
            node_index: batch.node_index,
            mobius: conformal.mobius,
            orientation_sign: conformal.orientation_sign * euclidean_orientation,
            euclidean_model: conformal.euclidean_model,
            euclidean_normal: affine_normal_matrix(&conformal.euclidean_model),
        });
    }
    renderer.render_batches = render_batches;
    renderer.render_commands_dirty = false;
    renderer.render_shadow_scene_dirty = true;
    renderer.render_command_builds += 1;
}

fn extract_render_scene(renderer: &MainState) -> Result<RenderSceneSnapshot, String> {
    if renderer.batches.len() != renderer.render_batches.len() {
        return Err(format!(
            "retained GPU batches ({}) do not match render views ({})",
            renderer.batches.len(),
            renderer.render_batches.len(),
        ));
    }
    let default_material = PbrParams::default();
    let mut batches = Vec::with_capacity(renderer.batches.len());
    for ((&key, gpu_batch), render_batch) in
        renderer.batches.iter().zip(&renderer.render_batches)
    {
        if gpu_batch.material_index != render_batch.material_index
            || gpu_batch.node_index != render_batch.node_index
        {
            return Err("retained GPU batch metadata does not match its render view".to_string());
        }
        let triangle_index_count = u32::try_from(render_batch.mesh.num_tri_indices)
            .map_err(|_| "render batch has a negative triangle index count".to_string())?;
        let line_index_count = u32::try_from(render_batch.mesh.num_line_indices)
            .map_err(|_| "render batch has a negative line index count".to_string())?;
        let (_, material) = pbr_material_for_index(
            &renderer.materials,
            &default_material,
            render_batch.material_index,
        );
        let pbr_class = if material.transmission_factor > 0.0 {
            PbrDrawClass::Transmission
        } else if material.alpha_mode > 1.5 {
            PbrDrawClass::Blend
        } else {
            PbrDrawClass::Opaque
        };
        batches.push(RenderBatchSnapshot {
            key,
            members: gpu_batch.members.clone(),
            triangle_index_count,
            line_index_count,
            transform: RenderEntityTransform {
                mobius: render_batch.mobius,
                orientation_sign: render_batch.orientation_sign,
                euclidean_model: render_batch.euclidean_model,
                euclidean_normal: render_batch.euclidean_normal,
            },
            enabled: render_batch.mesh.num_instances > 0,
            pbr_class,
        });
    }
    Ok(RenderSceneSnapshot {
        revision: 0,
        batches,
    })
}

fn refresh_render_shadow_scene(renderer: &mut MainState) {
    if !renderer.render_shadow.is_enabled() || !renderer.render_shadow_scene_dirty {
        return;
    }
    let extracted = extract_render_scene(renderer);
    renderer.render_shadow_scene_dirty = false;
    match extracted {
        Ok(scene) => renderer.render_shadow.replace_scene(scene),
        Err(error) => renderer.render_shadow.record_extraction_error(error),
    }
}

fn render_style(mode: RenderMode) -> RenderStyle {
    match mode {
        RenderMode::Pbr => RenderStyle::Pbr,
        RenderMode::Matcap => RenderStyle::Matcap,
        RenderMode::Wire => RenderStyle::Wire,
        RenderMode::Normals => RenderStyle::Normals,
        RenderMode::Both => RenderStyle::MatcapWire,
        RenderMode::Lod => RenderStyle::Lod,
        RenderMode::Stretch => RenderStyle::Stretch,
    }
}

fn observe_render_submission(
    renderer: &mut MainState,
    camera: &Camera,
    actual: RenderSubmissionStats,
) {
    if !renderer.render_shadow.is_enabled() {
        return;
    }
    let selected_node = usize::try_from(renderer.selected_node).ok();
    let highlight_face = u32::try_from(renderer.highlight_face).ok();
    renderer.render_shadow.observe(
        render_style(renderer.render_mode),
        RenderView {
            viewport: [
                renderer.viewport_size.0.max(0) as u32,
                renderer.viewport_size.1.max(0) as u32,
            ],
            mvp: camera.mvp,
            model_view: camera.mv,
            camera_position: camera.camera_pos,
            selected_node,
            focus: FocusFieldPacket {
                sphere: renderer.focus_sphere,
                enabled: renderer.focus_field_enabled,
            },
        },
        RenderFrameOptions {
            focus_postprocess: renderer.fuzzy_enabled,
            highlight_face,
        },
        actual,
    );
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
            if renderer.active_hyperscape_camera != Some(node_index) {
                renderer.active_hyperscape_camera = Some(node_index);
                renderer.render_commands_dirty = true;
            }
            true
        } else {
            false
        }
    })
}

#[wasm_bindgen(js_name = "mr_tickHyperscape")]
pub fn mr_tick_hyperscape(delta_seconds: f64, include_scene_diagnostics: bool) -> JsValue {
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
            include_scene_diagnostics.then(|| runtime.diagnostic_snapshot()),
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
    if let Some(scene_diagnostics) = scene_diagnostics {
        js_sys::Reflect::set(
            &result,
            &"scene".into(),
            &runtime_diagnostic_snapshot_to_js(&scene_diagnostics),
        ).ok();
    }
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

#[wasm_bindgen(js_name = "mr_setSelectedNode")]
pub fn mr_set_selected_node(node_id: i32) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.selected_node = node_id.max(-1);
        }
    });
}

/// Atomically install a complete application-owned focus/selection packet.
///
/// This is crate-visible so the Rust application adapter can hand state to the
/// renderer without serializing the packet through JavaScript. Returning
/// `false` means either validation failed or no renderer is resident.
pub(crate) fn apply_focus_packet(
    sphere: [f32; 4],
    enabled: bool,
    selected_node: i32,
) -> bool {
    if !sphere.iter().all(|value| value.is_finite())
        || sphere[3] <= 0.0
        || selected_node < -1
    {
        return false;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(renderer) = state.as_mut() else {
            return false;
        };
        renderer.focus_sphere = sphere;
        renderer.focus_field_enabled = enabled;
        renderer.selected_node = selected_node;
        true
    })
}

#[wasm_bindgen(js_name = "mr_setFocusSphere")]
pub fn mr_set_focus_sphere(
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    enabled: bool,
) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            let center = [center_x, center_y, center_z];
            if center.iter().all(|value| value.is_finite()) && radius.is_finite() {
                renderer.focus_sphere = [center_x, center_y, center_z, radius.max(1e-4)];
            }
            renderer.focus_field_enabled = enabled;
        }
    });
}

/// Compact readback of the retained focus/selection packet for migration
/// parity checks. This reads CPU-side renderer state and performs no GPU
/// synchronization.
#[wasm_bindgen(js_name = "mr_debugFocusState")]
pub fn mr_debug_focus_state() -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(renderer) = state.as_ref() else {
            return JsValue::NULL;
        };
        let result = js_sys::Object::new();
        let center = js_sys::Float32Array::from(&renderer.focus_sphere[..3]);
        js_sys::Reflect::set(&result, &"center".into(), &center).ok();
        js_sys::Reflect::set(
            &result,
            &"radius".into(),
            &JsValue::from_f64(f64::from(renderer.focus_sphere[3])),
        )
        .ok();
        js_sys::Reflect::set(
            &result,
            &"enabled".into(),
            &JsValue::from_bool(renderer.focus_field_enabled),
        )
        .ok();
        js_sys::Reflect::set(
            &result,
            &"selectedNode".into(),
            &JsValue::from_f64(f64::from(renderer.selected_node)),
        )
        .ok();
        result.into()
    })
}

/// CPU-only readback of descriptor memo counters. No GL query, fence, or
/// buffer synchronization is performed.
#[wasm_bindgen(js_name = "mr_programCacheDiagnostics")]
pub fn mr_program_cache_diagnostics() -> JsValue {
    fn counters(
        diagnostics: quilting_renderer::memo::DeviceMemoDiagnostics,
    ) -> js_sys::Object {
        let result = js_sys::Object::new();
        for (name, value) in [
            ("hits", diagnostics.hits),
            ("misses", diagnostics.misses),
            ("failedCreations", diagnostics.failed_creations),
            ("invalidations", diagnostics.invalidations),
            ("residentEntries", diagnostics.resident_entries as u64),
        ] {
            js_sys::Reflect::set(
                &result,
                &JsValue::from_str(name),
                &JsValue::from_f64(value as f64),
            )
            .expect("setting a property on a new plain object must succeed");
        }
        result
    }

    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return JsValue::NULL;
        };
        let diagnostics = state.renderer.program_memo_diagnostics();
        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("deviceEpoch"),
            &JsValue::from_f64(diagnostics.device_epoch as f64),
        )
        .expect("setting a property on a new plain object must succeed");
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("shaders"),
            &counters(diagnostics.shaders),
        )
        .expect("setting a property on a new plain object must succeed");
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("programs"),
            &counters(diagnostics.programs),
        )
        .expect("setting a property on a new plain object must succeed");
        result.into()
    })
}

/// Pick the face under pixel (x, y). Renders a pick pass and reads back the face ID.
/// Returns face ID (>= 0) or -1 if no face at that pixel.
/// Also logs face info (LOD, edge lengths, instance data) to console.
#[wasm_bindgen(js_name = "mr_pick")]
pub fn mr_pick(mvp: &[f32], mv: &[f32], camera_pos: &[f32], x: i32, y: i32) -> i32 {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return -1 };
        sync_render_batches(state);
        // Lazy-init highlight program
        if state.highlight_prog.is_none() {
            match resolve_auxiliary_program(state, AuxiliaryProgram::Highlight) {
                Ok(program) => state.highlight_prog = Some(program),
                Err(error) => {
                    warn!("Selection highlight unavailable; picking continues: {error}");
                }
            }
        }

        let gl = state.renderer.gl();

        unsafe {
            let (vw, vh) = state.viewport_size;

            // Create/resize pick FBO
            if state.pick_fbo.is_none() || state.pick_size != (vw, vh) {
                if let Some(f) = state.pick_fbo { gl.delete_framebuffer(f); }
                if let Some(t) = state.pick_tex { gl.delete_texture(t); }
                if let Some(t) = state.pick_bary_tex { gl.delete_texture(t); }
                if let Some(r) = state.pick_depth { gl.delete_renderbuffer(r); }

                let tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                    glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);

                let bary_tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_2D, Some(bary_tex));
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
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT1,
                    glow::TEXTURE_2D, Some(bary_tex), 0);
                gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT,
                    glow::RENDERBUFFER, Some(depth));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);

                state.pick_fbo = Some(fbo);
                state.pick_tex = Some(tex);
                state.pick_bary_tex = Some(bary_tex);
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

            // Picking can run between an LOD membership update and the next
            // animation frame. Expand the compact topology stream here so it
            // never consumes stale prepared records.
            state.renderer.joint_ubo().bind(gl);
            state.renderer.bind_vertex_textures();
            for (batch, render_batch) in state.batches.values().zip(&state.render_batches) {
                state.renderer.prepare_patch_batch(
                    &camera,
                    render_batch,
                    batch.prepare_vao,
                    batch.instances.prepared_buf,
                    0,
                );
            }
            gl.use_program(Some(state.renderer.programs().pick));

            for batch in &state.render_batches {
                let batch_camera = camera_for_batch(&camera, batch);
                apply_batch_winding(
                    gl,
                    batch.orientation_sign,
                    batch.perm_parity,
                );
                let vtx_ubo = state.renderer.vtx_ubo();
                vtx_ubo.upload(
                    gl, &batch_camera.mvp, &batch_camera.mv,
                    1,
                    &batch_camera.mobius, &batch_camera.camera_pos,
                    &batch.euclidean_model, &batch.euclidean_normal,
                );
                vtx_ubo.bind(gl);

                gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                gl.draw_elements_instanced(
                    glow::TRIANGLES, batch.mesh.num_tri_indices,
                    glow::UNSIGNED_INT, batch.mesh.tri_index_offset, batch.mesh.num_instances,
                );
            }

            // Read pixel (flip Y for framebuffer coords)
            let fy = vh - 1 - y;
            let mut px = [0u8; 4];
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            gl.read_pixels(x, fy, 1, 1, glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut px)));
            let mut bary_px = [0u8; 4];
            gl.read_buffer(glow::COLOR_ATTACHMENT1);
            gl.read_pixels(x, fy, 1, 1, glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut bary_px)));
            gl.read_buffer(glow::COLOR_ATTACHMENT0);

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, vw, vh);

            // Decode 24-bit face ID from RGB
            if px[3] == 0 {
                state.last_pick_barycentric = None;
                return -1; // no geometry (alpha=0 from clear)
            }
            let face_id = (px[0] as i32) | ((px[1] as i32) << 8) | ((px[2] as i32) << 16);
            let mut barycentric = [
                bary_px[0] as f32 / 255.0,
                bary_px[1] as f32 / 255.0,
                bary_px[2] as f32 / 255.0,
            ];
            let barycentric_sum = barycentric.iter().sum::<f32>();
            if bary_px[3] == 0 || !barycentric_sum.is_finite() || barycentric_sum <= 1e-6 {
                state.last_pick_barycentric = None;
            } else {
                for coordinate in &mut barycentric {
                    *coordinate /= barycentric_sum;
                }
                state.last_pick_barycentric = Some(barycentric);
            }

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
                if let Some(barycentric) = state.last_pick_barycentric {
                    info!("  bary at click: ({:.3}, {:.3}, {:.3})",
                        barycentric[0], barycentric[1], barycentric[2]);
                }

                let node = state.face_nodes.get(face_id as usize).copied().unwrap_or(0);
                info!("  glTF node: {}", node);
            }

            // Set highlight
            state.highlight_face = face_id;
            face_id
        }
    })
}

/// Pick a stable source-surface address. The face ID and barycentric seed are
/// captured by the same depth-tested pass, so animated or conformally mapped
/// geometry cannot produce a face/coordinate mismatch.
#[wasm_bindgen(js_name = "mr_pickSurface")]
pub fn mr_pick_surface(
    mvp: &[f32],
    mv: &[f32],
    camera_pos: &[f32],
    x: i32,
    y: i32,
) -> JsValue {
    let face = mr_pick(mvp, mv, camera_pos, x, y);
    if face < 0 {
        return JsValue::NULL;
    }
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return JsValue::NULL;
        };
        let Some(barycentric) = state.last_pick_barycentric else {
            return JsValue::NULL;
        };
        let result = js_sys::Object::new();
        let barycentric_array = js_sys::Float32Array::new_with_length(3);
        barycentric_array.copy_from(&barycentric);
        let node = state.face_nodes.get(face as usize).copied().unwrap_or(0);
        js_sys::Reflect::set(&result, &"face".into(), &JsValue::from_f64(face as f64)).ok();
        js_sys::Reflect::set(&result, &"node".into(), &JsValue::from_f64(node as f64)).ok();
        js_sys::Reflect::set(&result, &"barycentric".into(), &barycentric_array).ok();
        result.into()
    })
}

fn surface_snapshot_to_js(snapshot: Result<SurfaceRuntimeSnapshot, String>) -> JsValue {
    match snapshot {
        Ok(snapshot) => serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL),
        Err(error) => {
            warn!("Surface walker: {error}");
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"status".into(), &"error".into()).ok();
            js_sys::Reflect::set(&result, &"error".into(), &error.into()).ok();
            result.into()
        }
    }
}

fn composed_surface_snapshot_to_js(
    snapshot: Result<ComposedSurfaceWalkSnapshot, String>,
) -> ComposedSurfaceWalkResultJs {
    let value = match snapshot {
        Ok(snapshot) => serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL),
        Err(error) => {
            warn!("Composed surface walker: {error}");
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"status".into(), &"error".into()).ok();
            js_sys::Reflect::set(&result, &"error".into(), &error.into()).ok();
            result.into()
        }
    };
    value.unchecked_into()
}

fn exact_vector3(values: &[f64], label: &str) -> Result<[f64; 3], String> {
    values
        .try_into()
        .map_err(|_| format!("{label} must contain exactly three numbers"))
}

fn surface_walk_camera(
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    parameters: &[f64],
) -> Result<CameraRig, String> {
    let [control_distance, vertical_fov_radians, near, far]: [f64; 4] = parameters
        .try_into()
        .map_err(|_| "surface-walk camera parameters must contain exactly four numbers")?;
    CameraRig::new(
        exact_vector3(eye, "surface-walk camera eye")?,
        CameraBasis::from_forward_up(
            exact_vector3(forward, "surface-walk camera forward")?,
            exact_vector3(up, "surface-walk camera up")?,
        )
        .map_err(|error| error.to_string())?,
        control_distance,
        None,
        PerspectiveLens {
            vertical_fov_radians,
            near,
            far,
        },
    )
    .map_err(|error| error.to_string())
}

fn surface_walk_controls(values: &[f64]) -> Result<SurfaceWalkControls, String> {
    let [
        base_radii_per_second,
        base_eye_height,
        speed_octave_steps,
        body_scale_octave_steps,
        eye_height_octave_steps,
        smoothing_seconds,
        tangent_pull_fraction,
        fast_multiplier,
        default_near,
        minimum_near,
        near_eye_fraction,
    ]: [f64; 11] = values
        .try_into()
        .map_err(|_| "surface-walk controls must contain exactly eleven numbers")?;
    Ok(SurfaceWalkControls {
        base_radii_per_second,
        base_eye_height,
        speed_octave_steps,
        body_scale_octave_steps,
        eye_height_octave_steps,
        smoothing_seconds,
        tangent_pull_fraction,
        fast_multiplier,
        default_near,
        minimum_near,
        near_eye_fraction,
    })
}

fn surface_walk_reflection_state(
    enabled: bool,
    center: &[f64],
    radius: f64,
) -> Result<SphereReflectionState, String> {
    let center = exact_vector3(center, "surface-walk reflection center")?;
    if !center.into_iter().all(f64::is_finite) || !radius.is_finite() {
        return Err("surface-walk reflection must be finite".to_string());
    }
    if enabled {
        FocusSphere::new(center, radius)
            .map(SphereReflectionState::Sphere)
            .map_err(|error| error.to_string())
    } else {
        Ok(SphereReflectionState::Identity)
    }
}

fn surface_walk_reflection_transport_to_js(
    snapshot: SurfaceWalkReflectionTransportSnapshot,
) -> Result<SurfaceWalkReflectionTransportResultJs, JsValue> {
    serde_wasm_bindgen::to_value(&snapshot)
        .map(JsCast::unchecked_into)
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Atomically carry topology-side state, the composed contact follower, and
/// previous animated-contact samples into a new spherical-reflection chart.
/// This export is deliberately independent of the render backend.
#[wasm_bindgen(js_name = "mr_transportSurfaceWalkReflection")]
pub fn mr_transport_surface_walk_reflection(
    previous_enabled: bool,
    previous_center: &[f64],
    previous_radius: f64,
    next_enabled: bool,
    next_center: &[f64],
    next_radius: f64,
) -> Result<SurfaceWalkReflectionTransportResultJs, JsValue> {
    let previous =
        surface_walk_reflection_state(previous_enabled, previous_center, previous_radius)
            .map_err(|error| JsValue::from_str(&error))?;
    let next = surface_walk_reflection_state(next_enabled, next_center, next_radius)
        .map_err(|error| JsValue::from_str(&error))?;
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return Ok(JsValue::NULL.unchecked_into());
        };
        let snapshot = state
            .surface_runtime
            .transport_between_reflections(previous, next)
            .map_err(|error| JsValue::from_str(&error))?;
        surface_walk_reflection_transport_to_js(snapshot)
    })
}

/// Attach the Rust walker to a source face selected by `mr_pickSurface`.
#[wasm_bindgen(js_name = "mr_attachSurface")]
pub fn mr_attach_surface(
    face: u32,
    barycentric: &[f32],
    eye_height: f32,
    camera_position: &[f32],
) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let Some(&node) = state.face_nodes.get(face as usize) else {
            return surface_snapshot_to_js(Err("surface face has no node".into()));
        };
        let conformal = conformal_state_for_node(state, node);
        let barycentric = [
            barycentric.first().copied().unwrap_or(0.0) as f64,
            barycentric.get(1).copied().unwrap_or(0.0) as f64,
            barycentric.get(2).copied().unwrap_or(0.0) as f64,
        ];
        let camera_position = [
            camera_position.first().copied().unwrap_or(0.0) as f64,
            camera_position.get(1).copied().unwrap_or(0.0) as f64,
            camera_position.get(2).copied().unwrap_or(0.0) as f64,
        ];
        let orientation_sign = conformal.orientation_sign
            * affine_orientation_sign(&conformal.euclidean_model);
        let MainState {
            surface_runtime,
            cached_instances,
            face_nodes,
            num_faces,
            ..
        } = state;
        surface_snapshot_to_js(surface_runtime.attach(
            cached_instances,
            *num_faces,
            face_nodes,
            face,
            barycentric,
            eye_height as f64,
            camera_position,
            orientation_sign,
            conformal.mobius,
            conformal.euclidean_model,
        ))
    })
}

/// Candidate Rust-authoritative surface-walk attachment. The legacy walker
/// remains available beside this instance only for `js|shadow|rust` migration
/// comparison; both borrow the same immutable geometry and pose data.
#[wasm_bindgen(js_name = "mr_attachSurfaceWalk")]
#[allow(clippy::too_many_arguments)]
pub fn mr_attach_surface_walk(
    face: u32,
    barycentric: &[f64],
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    camera_parameters: &[f64],
    scene_radius: f64,
    controls: &[f64],
    transition_duration_seconds: f64,
) -> ComposedSurfaceWalkResultJs {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL.unchecked_into();
        };
        let Some(&node) = state.face_nodes.get(face as usize) else {
            return composed_surface_snapshot_to_js(Err("surface face has no node".into()));
        };
        let request = (|| {
            let barycentric = exact_vector3(barycentric, "surface barycentric")?;
            let camera = surface_walk_camera(eye, forward, up, camera_parameters)?;
            let controls = surface_walk_controls(controls)?;
            let conformal = conformal_state_for_node(state, node);
            let orientation_sign = conformal.orientation_sign
                * affine_orientation_sign(&conformal.euclidean_model);
            let MainState {
                surface_runtime,
                cached_instances,
                face_nodes,
                num_faces,
                ..
            } = state;
            surface_runtime.attach_composed(
                cached_instances,
                *num_faces,
                face_nodes,
                face,
                barycentric,
                camera,
                scene_radius,
                controls,
                transition_duration_seconds,
                orientation_sign,
                conformal.mobius,
                conformal.euclidean_model,
            )
        })();
        composed_surface_snapshot_to_js(request)
    })
}

/// Advance relative to the current animated surface in the displayed output
/// chart. The runtime adds measured surface/frame velocity before applying the
/// Jacobian pseudoinverse, so zero input remains stuck to moving geometry.
#[wasm_bindgen(js_name = "mr_stepSurface")]
pub fn mr_step_surface(delta_seconds: f64, relative_output_velocity: &[f32]) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let Some(face) = state.surface_runtime.attachment_face() else {
            return surface_snapshot_to_js(Err("surface walker is detached".into()));
        };
        let Some(&node) = state.face_nodes.get(face as usize) else {
            return surface_snapshot_to_js(Err("attached face has no node".into()));
        };
        let conformal = conformal_state_for_node(state, node);
        let orientation_sign = conformal.orientation_sign
            * affine_orientation_sign(&conformal.euclidean_model);
        let velocity = [
            relative_output_velocity.first().copied().unwrap_or(0.0) as f64,
            relative_output_velocity.get(1).copied().unwrap_or(0.0) as f64,
            relative_output_velocity.get(2).copied().unwrap_or(0.0) as f64,
        ];
        let MainState {
            surface_runtime,
            cached_instances,
            face_nodes,
            ..
        } = state;
        surface_snapshot_to_js(surface_runtime.step_relative(
            cached_instances,
            face_nodes,
            delta_seconds,
            velocity,
            orientation_sign,
            conformal.mobius,
            conformal.euclidean_model,
        ))
    })
}

/// Apply one complete semantic walking frame to the composed Rust candidate.
/// The returned packet contains topology, metrics, target contact camera, and
/// the camera to render after the optional anchor glide.
#[wasm_bindgen(js_name = "mr_stepSurfaceWalk")]
#[allow(clippy::too_many_arguments)]
pub fn mr_step_surface_walk(
    delta_seconds: f64,
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    camera_parameters: &[f64],
    scene_radius: f64,
    controls: &[f64],
    forward_axis: f64,
    right_axis: f64,
    fast: bool,
    orient: bool,
    capture_relative_view: bool,
) -> ComposedSurfaceWalkResultJs {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL.unchecked_into();
        };
        let request = (|| {
            let face = state
                .surface_runtime
                .composed_attachment_face()
                .ok_or_else(|| "composed surface walker is detached".to_string())?;
            let node = *state
                .face_nodes
                .get(face as usize)
                .ok_or_else(|| "attached face has no node".to_string())?;
            let camera = surface_walk_camera(eye, forward, up, camera_parameters)?;
            let controls = surface_walk_controls(controls)?;
            let conformal = conformal_state_for_node(state, node);
            let orientation_sign = conformal.orientation_sign
                * affine_orientation_sign(&conformal.euclidean_model);
            let MainState {
                surface_runtime,
                cached_instances,
                face_nodes,
                ..
            } = state;
            surface_runtime.step_composed(
                cached_instances,
                face_nodes,
                delta_seconds,
                camera,
                scene_radius,
                controls,
                SurfaceWalkInput {
                    forward_axis,
                    right_axis,
                    fast,
                },
                orient,
                capture_relative_view,
                orientation_sign,
                conformal.mobius,
                conformal.euclidean_model,
            )
        })();
        composed_surface_snapshot_to_js(request)
    })
}

#[wasm_bindgen(js_name = "mr_detachSurface")]
pub fn mr_detach_surface() -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        serde_wasm_bindgen::to_value(&state.surface_runtime.detach()).unwrap_or(JsValue::NULL)
    })
}

fn build_instance_lod_topology(
    instances: &[f32],
    num_faces: usize,
    face_nodes: &[usize],
) -> Option<quilting_mesh::HalfEdgeMesh> {
    let required = num_faces.checked_mul(instance_layout::STRIDE)?;
    if instances.len() < required || face_nodes.len() != num_faces {
        return None;
    }

    type PositionKey = [u32; 3];
    let mut source_positions = HashMap::<(usize, u32), PositionKey>::new();
    let mut welded_vertices = HashMap::<(usize, PositionKey), u32>::new();
    let mut faces = Vec::<[u32; 3]>::with_capacity(num_faces);
    for face in 0..num_faces {
        let node = face_nodes[face];
        let base = face * instance_layout::STRIDE + instance_layout::offset::POSITIONS;
        let mut compact_face = [0u32; 3];
        for corner in 0..3 {
            let offset = base + corner * 4;
            let source = instances[offset];
            let xyz = [instances[offset + 1], instances[offset + 2], instances[offset + 3]];
            if !source.is_finite()
                || source < 0.0
                || source.fract() != 0.0
                || !xyz.iter().all(|coordinate| coordinate.is_finite())
            {
                return None;
            }
            let source = source as u32;
            let position = xyz.map(|coordinate| {
                if coordinate == 0.0 {
                    0.0_f32.to_bits()
                } else {
                    coordinate.to_bits()
                }
            });
            if let Some(previous) = source_positions.insert((node, source), position) {
                if previous != position {
                    return None;
                }
            }
            let next_vertex = welded_vertices.len() as u32;
            let compact = *welded_vertices
                .entry((node, position))
                .or_insert(next_vertex);
            compact_face[corner] = compact;
        }
        faces.push(compact_face);
    }

    Some(quilting_mesh::HalfEdgeMesh::from_triangles(
        welded_vertices.len() as u32,
        &faces,
    ))
}

#[wasm_bindgen(js_name = "mr_setRoundShadowEnabled")]
pub fn mr_set_round_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let MainState {
            round_shadow,
            cached_instances,
            num_faces,
            ..
        } = state;
        round_shadow.set_enabled(enabled, cached_instances, *num_faces)
    })
}

#[wasm_bindgen(js_name = "mr_setRenderShadowEnabled")]
pub fn mr_set_render_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let changed = state.render_shadow.is_enabled() != enabled;
        state.render_shadow.set_enabled(enabled);
        if enabled && changed {
            state.render_shadow_scene_dirty = true;
            sync_render_batches(state);
            refresh_render_shadow_scene(state);
        }
        state.render_shadow.to_js()
    })
}

#[wasm_bindgen(js_name = "mr_renderShadowDiagnostics")]
pub fn mr_render_shadow_diagnostics() -> JsValue {
    STATE.with(|state| {
        state
            .borrow()
            .as_ref()
            .map_or(JsValue::NULL, |state| state.render_shadow.to_js())
    })
}

#[wasm_bindgen(js_name = "mr_roundShadowDiagnostics")]
pub fn mr_round_shadow_diagnostics() -> JsValue {
    STATE.with(|state| {
        state
            .borrow()
            .as_ref()
            .map_or(JsValue::NULL, |state| state.round_shadow.to_js())
    })
}

/// Compare the conservative CPU hierarchy with the authoritative GPU
/// visibility classification for the same completed LOD request. This is
/// telemetry only: candidate membership never changes renderer state.
#[allow(clippy::too_many_arguments)] // Flat arrays form the browser/WASM ABI.
#[wasm_bindgen(js_name = "mr_compareRoundShadow")]
pub fn mr_compare_round_shadow(
    view_projection: &[f32],
    transform_kind: &str,
    transform_center: &[f32],
    transform_radius: f32,
    animated: bool,
    authored_scene: bool,
    authoritative_full_snapshot: bool,
    pose_time: f64,
    pose_joint_matrices: &[f32],
    pose_morph_weights: &[f32],
) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let MainState {
            round_shadow,
            classified_face_visibility,
            surface_runtime,
            cached_instances,
            num_faces,
            ..
        } = state;
        let pose_reconstruct_started = browser_now_ms();
        let posed_patches = if animated {
            surface_runtime
                .patch_controls_for_pose(
                    cached_instances,
                    *num_faces,
                    pose_joint_matrices,
                    pose_morph_weights,
                )
                .map(Some)
        } else {
            Ok(None)
        };
        let pose_reconstruct_ms = if animated {
            browser_now_ms() - pose_reconstruct_started
        } else {
            0.0
        };
        round_shadow.compare(
            view_projection,
            transform_kind,
            transform_center,
            transform_radius,
            animated,
            authored_scene,
            authoritative_full_snapshot,
            classified_face_visibility,
            pose_time,
            pose_reconstruct_ms,
            posed_patches,
        )
    })
}
#[wasm_bindgen(js_name = "mr_setInstanceData")]
pub fn mr_set_instance_data(instances: &[f32], num_faces: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            let num_faces = num_faces as usize;
            if let Err(error) = st.renderer.upload_face_data_texture(instances, num_faces) {
                warn!("Could not upload immutable source-face data: {error}");
                return;
            }
            st.cached_instances = instances.to_vec();
            st.num_faces = num_faces;
            st.render_shadow.asset_changed();
            st.render_shadow_scene_dirty = true;
            st.presentation_nodes.clear();
            st.surface_runtime.reset_geometry();
            st.batch_groups.clear();
            st.batch_staging.clear();
            st.resident_face_lods = vec![None; st.num_faces];
            st.resident_vertex_lods = vec![[1; 3]; st.num_faces];
            st.resident_vertex_lod_scratch.clear();
            st.classified_face_visibility = vec![false; st.num_faces];
            st.classified_culled_faces = st.num_faces;
            st.lod_dirty_faces.clear();
            st.lod_balance_scratch = batch::ResidentLodBalanceScratch::default();
            if st.face_nodes.len() != st.num_faces {
                st.face_nodes = vec![0; st.num_faces];
            }
            st.lod_topology = build_instance_lod_topology(
                &st.cached_instances,
                st.num_faces,
                &st.face_nodes,
            );
            st.batch_update_stats = BatchUpdateStats::default();
            if let Some(topology) = st.lod_topology.as_ref() {
                info!(
                    "LOD topology: {} faces, {} boundary half-edges after exact seam welding",
                    topology.num_faces,
                    topology.num_boundary_edges(),
                );
            } else {
                warn!("Could not construct resident LOD topology from instance data");
            }
            let MainState {
                round_shadow,
                cached_instances,
                num_faces,
                ..
            } = st;
            round_shadow.rebuild(cached_instances, *num_faces);
            st.batch_layout_dirty = true;
        }
    });
}

/// Read-only topology diagnostics for browser regression checks.
#[wasm_bindgen(js_name = "mr_debugResidentLodEdges")]
pub fn mr_debug_resident_lod_edges() -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else { return JsValue::NULL };
        let Some(topology) = state.lod_topology.as_ref() else { return JsValue::NULL };

        let mut shared_edges = 0u32;
        let mut mismatched_edges = 0u32;
        let mut missing_residents = 0u32;
        let mut roundtrip_failures = 0u32;
        let mut anisotropic_faces = 0u32;
        let mut max_edge_ratio = 1u32;
        let mut same_max_density_jumps = 0u32;
        let mut max_adjacent_triangle_ratio = 1.0f64;
        let mut lod_histogram = BTreeMap::<String, u32>::new();
        let mut examples = Vec::<String>::new();
        let resident_triangle_counts = TESS_CACHE.with(|cache| {
            let cache = cache.borrow();
            state.resident_face_lods.iter().map(|resident| {
                resident.and_then(|resident| cache.get(&resident.canonical))
                    .map(|patch| patch.num_tri_indices as u32 / 3)
            }).collect::<Vec<_>>()
        });
        let resident_triangles: u64 = resident_triangle_counts
            .iter()
            .flatten()
            .map(|&count| count as u64)
            .sum();
        for resident in &state.resident_face_lods {
            match resident {
                Some(resident) => {
                    if batch::ResidentLod::from_edge_lods(resident.edge_lods()) != *resident {
                        roundtrip_failures += 1;
                    }
                    let [minimum, middle, maximum] = resident.canonical;
                    *lod_histogram
                        .entry(format!("{minimum}/{middle}/{maximum}"))
                        .or_default() += 1;
                    let ratio = maximum / minimum.max(1);
                    max_edge_ratio = max_edge_ratio.max(ratio);
                    if ratio > 1 {
                        anisotropic_faces += 1;
                    }
                }
                None => missing_residents += 1,
            }
        }
        for half_edge in 0..topology.half_edges.len() as u32 {
            let Some(twin) = topology.twin(half_edge) else { continue };
            if half_edge > twin {
                continue;
            }
            shared_edges += 1;
            let face = topology.half_edges[half_edge as usize].face as usize;
            let twin_face = topology.half_edges[twin as usize].face as usize;
            let Some(face_lods) = state
                .resident_face_lods
                .get(face)
                .and_then(|resident| resident.map(batch::ResidentLod::edge_lods))
            else {
                continue;
            };
            let Some(twin_lods) = state
                .resident_face_lods
                .get(twin_face)
                .and_then(|resident| resident.map(batch::ResidentLod::edge_lods))
            else {
                continue;
            };
            let edge_index = (half_edge as usize % 3 + 2) % 3;
            let twin_edge_index = (twin as usize % 3 + 2) % 3;
            if face_lods[edge_index] != twin_lods[twin_edge_index] {
                mismatched_edges += 1;
                if examples.len() < 8 {
                    examples.push(format!(
                        "f{face}:e{edge_index}={} != f{twin_face}:e{twin_edge_index}={}",
                        face_lods[edge_index],
                        twin_lods[twin_edge_index],
                    ));
                }
            }
            let face_resident = state.resident_face_lods[face].unwrap();
            let twin_resident = state.resident_face_lods[twin_face].unwrap();
            let face_max = *face_resident.canonical.iter().max().unwrap_or(&1);
            let twin_max = *twin_resident.canonical.iter().max().unwrap_or(&1);
            if face_max == twin_max {
                if let (Some(face_triangles), Some(twin_triangles)) = (
                    resident_triangle_counts[face],
                    resident_triangle_counts[twin_face],
                ) {
                    let smaller = face_triangles.min(twin_triangles).max(1);
                    let larger = face_triangles.max(twin_triangles);
                    let ratio = larger as f64 / smaller as f64;
                    max_adjacent_triangle_ratio = max_adjacent_triangle_ratio.max(ratio);
                    if ratio >= 4.0 {
                        same_max_density_jumps += 1;
                        if examples.len() < 8 {
                            examples.push(format!(
                                "same-max f{face} {:?} ({face_triangles} tris) vs f{twin_face} {:?} ({twin_triangles} tris)",
                                face_resident.canonical,
                                twin_resident.canonical,
                            ));
                        }
                    }
                }
            }
        }

        let result = js_sys::Object::new();
        let set_number = |name: &str, value: u32| {
            js_sys::Reflect::set(&result, &name.into(), &JsValue::from(value)).ok();
        };
        set_number("faces", state.num_faces as u32);
        set_number("sharedEdges", shared_edges);
        set_number("boundaryHalfEdges", topology.num_boundary_edges() as u32);
        set_number("mismatchedEdges", mismatched_edges);
        set_number("missingResidents", missing_residents);
        set_number("roundtripFailures", roundtrip_failures);
        set_number("anisotropicFaces", anisotropic_faces);
        set_number("maxEdgeRatio", max_edge_ratio);
        set_number("gradingRatio", state.lod_grading.ratio());
        set_number("sameMaxDensityJumps", same_max_density_jumps);
        js_sys::Reflect::set(
            &result,
            &"activeBatches".into(),
            &JsValue::from(state.batches.len() as f64),
        ).ok();
        let batch_stats = &state.batch_update_stats;
        for (name, value) in [
            ("batchBuildCalls", batch_stats.calls),
            ("unchangedBatchBuildCalls", batch_stats.unchanged_calls),
            ("retainedBatchBuckets", batch_stats.retained_buckets),
            ("updatedBatchBuckets", batch_stats.updated_buckets),
            ("createdBatchBuckets", batch_stats.created_buckets),
            ("reallocatedBatchBuckets", batch_stats.reallocated_buckets),
            ("retiredBatchBuckets", batch_stats.retired_buckets),
            ("uploadedBatchInstances", batch_stats.uploaded_instances),
            (
                "uploadedBatchBytes",
                batch_stats.uploaded_instances
                    * instance_layout::BATCH_TOPOLOGY_STRIDE_BYTES as u64,
            ),
            (
                "sourceFaceTextureBytes",
                state.num_faces as u64 * instance_layout::STRIDE_BYTES as u64,
            ),
            ("lastCulledFaces", batch_stats.last_culled_faces),
            ("lastLodCorrections", batch_stats.last_lod_corrections),
            (
                "lastMissingAtlasEntries",
                batch_stats.last_missing_atlas_entries,
            ),
            ("lastGpuBatchFailures", batch_stats.last_gpu_failures),
            ("renderCommandBuilds", state.render_command_builds),
            ("renderCalls", state.render_calls),
            ("pbrDrawCalls", state.pbr_draw_calls),
            ("pbrMaterialUpdates", state.pbr_material_updates),
            (
                "pbrVertexUniformUpdates",
                state.pbr_vertex_uniform_updates,
            ),
            (
                "lastPatchDrawCalls",
                state.last_render_submission.draw_calls,
            ),
            (
                "lastZeroInstancePatchDrawCalls",
                state.last_render_submission.zero_instance_draw_calls,
            ),
            (
                "lastInvalidPatchDrawCalls",
                state.last_render_submission.invalid_draw_calls,
            ),
            (
                "lastSubmittedPatchInstances",
                state.last_render_submission.submitted_instances,
            ),
            (
                "lastSubmittedPatchTriangles",
                state.last_render_submission.triangles,
            ),
            (
                "lastSubmittedPatchLines",
                state.last_render_submission.lines,
            ),
            (
                "patchDrawCalls",
                state.render_submission_totals.draw_calls,
            ),
            (
                "zeroInstancePatchDrawCalls",
                state.render_submission_totals.zero_instance_draw_calls,
            ),
            (
                "invalidPatchDrawCalls",
                state.render_submission_totals.invalid_draw_calls,
            ),
            (
                "submittedPatchInstances",
                state.render_submission_totals.submitted_instances,
            ),
            (
                "submittedPatchTriangles",
                state.render_submission_totals.triangles,
            ),
            (
                "submittedPatchLines",
                state.render_submission_totals.lines,
            ),
        ] {
            js_sys::Reflect::set(
                &result,
                &name.into(),
                &JsValue::from(value as f64),
            ).ok();
        }
        js_sys::Reflect::set(
            &result,
            &"maxSameMaxAdjacentTriangleRatio".into(),
            &JsValue::from(max_adjacent_triangle_ratio),
        ).ok();
        js_sys::Reflect::set(
            &result,
            &"residentTriangles".into(),
            &JsValue::from(resident_triangles as f64),
        ).ok();
        js_sys::Reflect::set(
            &result,
            &"lodHistogram".into(),
            &serde_wasm_bindgen::to_value(&lod_histogram.into_iter().collect::<Vec<_>>())
                .unwrap_or(JsValue::NULL),
        ).ok();
        js_sys::Reflect::set(
            &result,
            &"examples".into(),
            &serde_wasm_bindgen::to_value(&examples).unwrap_or(JsValue::NULL),
        ).ok();
        result.into()
    })
}

#[wasm_bindgen(js_name = "mr_setFaceMaterials")]
pub fn mr_set_face_materials(materials: &[i32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.face_materials = materials.iter().map(|&m| if m >= 0 { m as usize } else { 0 }).collect();
            st.batch_layout_dirty = true;
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
            renderer.batch_layout_dirty = true;
        }
    });
}

/// Select the measured within-face LOD grading policy before model residency.
/// A live switch would require recovering the unpromoted classifier snapshot
/// and replacing the atlas atomically, so the browser intentionally exposes
/// this as a reload-to-apply setting.
#[wasm_bindgen(js_name = "mr_setLodGradingRatio")]
pub fn mr_set_lod_grading_ratio(ratio: u32) -> bool {
    let Some(grading) = batch::FaceLodGrading::from_ratio(ratio) else {
        return false;
    };
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else { return false };
        if state.num_faces != 0 && state.lod_grading != grading {
            web_sys::console::warn_1(
                &"LOD grading ratio changes require a reload before model residency".into(),
            );
            return false;
        }
        state.lod_grading = grading;
        true
    })
}

/// Upload one packed canonical atlas. `patches` contains seven u32s per patch:
/// `[lod_a, lod_b, lod_c, tri_start, tri_count, line_start, line_count]`.
#[wasm_bindgen(js_name = "mr_uploadTessAtlas")]
pub fn mr_upload_tess_atlas(
    patches: &[u32],
    bary: &[f32],
    tri_idx: &[u32],
    line_idx: &[u32],
) -> bool {
    if patches.len() % 7 != 0 || bary.len() % 3 != 0 {
        web_sys::console::error_1(&"packed tessellation metadata has an invalid length".into());
        return false;
    }
    let ranges_valid = patches.chunks_exact(7).all(|patch| {
        patch[3].checked_add(patch[4]).is_some_and(|end| end as usize <= tri_idx.len())
            && patch[5].checked_add(patch[6]).is_some_and(|end| end as usize <= line_idx.len())
    });
    if !ranges_valid {
        web_sys::console::error_1(&"packed tessellation metadata has an invalid index range".into());
        return false;
    }

    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return false };
        let atlas = match TessAtlasBuffers::new(state.renderer.gl(), bary, tri_idx, line_idx) {
            Ok(atlas) => atlas,
            Err(error) => {
                web_sys::console::error_1(&format!("tessellation atlas upload: {error}").into());
                return false;
            }
        };

        // Patch VAOs retain element-buffer bindings. Invalidate them before an
        // atlas replacement so no live draw references the retired owner.
        for (_, batch) in std::mem::take(&mut state.batches) {
            batch.destroy(state.renderer.gl());
        }
        state.render_commands_dirty = true;
        state.batch_layout_dirty = true;
        TESS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.clear();
            for patch in patches.chunks_exact(7) {
                cache.insert(
                    [patch[0], patch[1], patch[2]],
                    atlas.patch(patch[3], patch[4], patch[5], patch[6]),
                );
            }
        });
        TESS_ATLAS.with(|owner| {
            if let Some(old) = owner.borrow_mut().replace(atlas) {
                old.destroy(state.renderer.gl());
            }
        });
        true
    })
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
fn decode_pbr_material(d: &[f32]) -> Option<PbrParams> {
    (d.len() >= MATERIAL_STRIDE).then(|| PbrParams {
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
    })
}

#[wasm_bindgen(js_name = "mr_setMaterials")]
pub fn mr_set_materials(data: &[f32], num_materials: u32) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.materials.clear();
            for i in 0..num_materials as usize {
                let offset = i * MATERIAL_STRIDE;
                let Some(record) = data.get(offset..offset + MATERIAL_STRIDE) else {
                    break;
                };
                let Some(material) = decode_pbr_material(record) else {
                    break;
                };
                st.materials.push(material);
            }
            info!("Set {} PBR materials", st.materials.len());
            st.render_shadow_scene_dirty = true;
        }
    });
}

/// Append materials for an additional presentation asset. Texture indices in
/// the records must already be offset into the shared texture cache.
#[wasm_bindgen(js_name = "mr_appendMaterials")]
pub fn mr_append_materials(data: &[f32], num_materials: u32) -> u32 {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return 0;
        };
        let base = state.materials.len() as u32;
        for index in 0..num_materials as usize {
            let offset = index * MATERIAL_STRIDE;
            let Some(record) = data.get(offset..offset + MATERIAL_STRIDE) else {
                break;
            };
            let Some(material) = decode_pbr_material(record) else {
                break;
            };
            state.materials.push(material);
        }
        info!(
            "Appended {} PBR materials at base {}",
            state.materials.len() as u32 - base,
            base,
        );
        state.render_shadow_scene_dirty = true;
        base
    })
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

/// Upload transferable browser-decoded images directly to WebGL. This avoids
/// a 4-byte-per-pixel canvas readback, cross-thread transfer, and WASM staging
/// allocation for large glTF texture sets.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "mr_uploadImageBitmaps")]
pub fn mr_upload_image_bitmaps(
    bitmaps: &js_sys::Array,
    wrap_modes: &[u32],
) -> Result<u32, JsValue> {
    let mut images = Vec::with_capacity(bitmaps.length() as usize);
    for index in 0..bitmaps.length() {
        let bitmap = bitmaps.get(index);
        let wrap_s = wrap_modes
            .get(index as usize * 2)
            .copied()
            .unwrap_or(glow::REPEAT);
        let wrap_t = wrap_modes
            .get(index as usize * 2 + 1)
            .copied()
            .unwrap_or(glow::REPEAT);
        if bitmap.is_null() || bitmap.is_undefined() {
            images.push(None);
        } else {
            images.push(Some((
                bitmap.dyn_into::<web_sys::ImageBitmap>().map_err(|_| {
                    JsValue::from_str(&format!("texture {index} is not an ImageBitmap"))
                })?,
                wrap_s,
                wrap_t,
            )));
        }
    }

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let uploaded = state
            .texture_cache
            .upload_image_bitmaps(state.renderer.gl(), &images);
        info!("Uploaded {uploaded}/{} ImageBitmap textures directly to WebGL", images.len());
        Ok(uploaded)
    })
}

/// Upload skinning texture (joint indices + weights) to main-thread GL.
/// `joint_indices`: flat f32 array [j0,j1,j2,j3] × num_vertices
/// `joint_weights`: flat f32 array [w0,w1,w2,w3] × num_vertices
#[wasm_bindgen(js_name = "mr_uploadSkinningTexture")]
pub fn mr_upload_skinning_texture(joint_indices: &[f32], joint_weights: &[f32], num_vertices: u32) {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return };
        let nv = num_vertices as usize;
        let Some(component_count) = nv.checked_mul(4) else {
            warn!("Skinning texture dimensions overflow: {} vertices", nv);
            return;
        };
        if joint_indices.len() < component_count || joint_weights.len() < component_count {
            warn!(
                "Skinning texture input is truncated: expected {} components, got {} indices and {} weights",
                component_count, joint_indices.len(), joint_weights.len()
            );
            return;
        }
        // Convert flat f32 arrays to the format SkinningTexture expects
        let ji: Vec<[u16; 4]> = joint_indices[..component_count].chunks_exact(4)
            .map(|row| [row[0] as u16, row[1] as u16, row[2] as u16, row[3] as u16])
            .collect();
        let jw: Vec<[f32; 4]> = joint_weights[..component_count].chunks_exact(4)
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect();
        match state.renderer.upload_skinning_texture(&ji, &jw) {
            Ok(()) => {
                state.surface_runtime.set_skinning(&ji, &jw);
                debug!("Skinning texture uploaded: {} vertices", nv);
            }
            Err(error) => warn!("Skinning texture upload failed: {}", error),
        }
    });
}

/// Upload morph target delta texture to main-thread GL.
/// `deltas`: flat f32 array [dx,dy,dz] × num_vertices × num_targets
#[wasm_bindgen(js_name = "mr_uploadMorphTexture")]
pub fn mr_upload_morph_texture(deltas: &[f32], num_vertices: u32, num_targets: u32) {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return };
        let nv = num_vertices as usize;
        let nt = num_targets as usize;
        match state.renderer.upload_morph_texture(deltas, nv, nt) {
            Ok(()) => {
                state.surface_runtime.set_morph_targets(deltas, nv, nt);
                debug!("Morph texture uploaded: {} vertices × {} targets", nv, nt);
            }
            Err(error) => warn!("Morph texture upload failed: {}", error),
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
    update_batches(face_lods, None);
}

/// Apply changed worker classifications without copying or scanning a full
/// six-float record for every source face. The first classification after a
/// model/animation boundary still uses `mr_buildBatches` as a full snapshot.
#[wasm_bindgen(js_name = "mr_updateBatches")]
pub fn mr_update_batches(face_indices: &[u32], face_lods: &[f32]) {
    if face_indices.is_empty() {
        return;
    }
    let Some(required) = face_indices.len().checked_mul(batch::FACE_LOD_STRIDE) else {
        web_sys::console::warn_1(&"Sparse LOD update size overflow".into());
        return;
    };
    if face_lods.len() < required {
        web_sys::console::warn_1(
            &format!(
                "Sparse LOD update has {} floats for {} face records",
                face_lods.len(),
                face_indices.len(),
            )
            .into(),
        );
        return;
    }
    update_batches(face_lods, Some(face_indices));
}

fn update_batches(face_lods: &[f32], face_indices: Option<&[u32]>) {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return };
        let gl = state.renderer.gl();
        state.batch_update_stats.calls += 1;

        // Phase 1: bucket sort to get face groupings (fast O(n), no instance data copy)
        perf_mark("batch-group-start");
        perf_mark("batch-retain-start");
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
        if state.resident_face_lods.len() != nf {
            state.resident_face_lods.resize(nf, None);
        }
        if state.classified_face_visibility.len() != nf {
            state.classified_face_visibility = vec![false; nf];
            state.classified_culled_faces = nf;
        }
        // Use LOD 2 for faces that have never had a visible classification if
        // the atlas provides it. This preserves the historical safe fallback
        // while allowing genuinely visible, sub-pixel faces to select LOD 1.
        let initial_resident_lod = TESS_CACHE.with(|tc| {
            let tc = tc.borrow();
            tc.keys()
                .filter(|lod| lod[0] >= 2)
                .min_by_key(|lod| {
                    (lod[0].saturating_mul(lod[1]).saturating_mul(lod[2]), **lod)
                })
                .or_else(|| tc.keys().min_by_key(|lod| {
                    (lod[0].saturating_mul(lod[1]).saturating_mul(lod[2]), **lod)
                }))
                .copied()
        }).unwrap_or([1; 3]);
        let initial_resident = batch::ResidentLod {
            canonical: initial_resident_lod,
            perm_index: 0,
            parity_bucket: 0,
        };
        let full_snapshot = face_indices.is_none();
        let mut culled = if full_snapshot {
            0
        } else {
            state.classified_culled_faces
        };
        let mut topology_changed = state.batch_layout_dirty;
        state.lod_dirty_faces.clear();
        let record_count = face_indices.map_or(nf, <[u32]>::len);
        for record_index in 0..record_count {
            let fi = face_indices
                .map_or(record_index, |indices| indices[record_index] as usize);
            if fi >= nf {
                continue;
            }
            let cpu_visible = batch::face_is_visible(face_lods, record_index);
            if full_snapshot {
                state.classified_face_visibility[fi] = cpu_visible;
                culled += usize::from(!cpu_visible);
            } else {
                let previously_visible = state.classified_face_visibility[fi];
                if previously_visible != cpu_visible {
                    if cpu_visible {
                        culled = culled.saturating_sub(1);
                    } else {
                        culled = culled.saturating_add(1).min(nf);
                    }
                    state.classified_face_visibility[fi] = cpu_visible;
                }
            }
            let previous = state.resident_face_lods[fi];
            let resident = batch::ResidentLod::from_visible_payload(face_lods, record_index)
                .or(previous)
                .unwrap_or(initial_resident);
            state.resident_face_lods[fi] = Some(resident);
            if previous != Some(resident) {
                topology_changed = true;
                state.lod_dirty_faces.push(fi);
            }
        }
        state.classified_culled_faces = culled;
        perf_mark("batch-retain-end");
        perf_measure("batch-retain", "batch-retain-start", "batch-retain-end");

        perf_mark("batch-balance-start");
        let lod_corrections = if let Some(topology) = state.lod_topology.as_ref() {
            batch::balance_resident_lods_from_faces_with_grading(
                &mut state.resident_face_lods,
                topology,
                &state.lod_dirty_faces,
                &mut state.lod_balance_scratch,
                state.lod_grading,
            )
        } else {
            0
        };
        perf_mark("batch-balance-end");
        perf_measure(
            "batch-balance",
            "batch-balance-start",
            "batch-balance-end",
        );
        topology_changed |= lod_corrections != 0;
        state.batch_update_stats.last_culled_faces = culled as u64;
        state.batch_update_stats.last_lod_corrections = lod_corrections as u64;
        state.batch_update_stats.last_missing_atlas_entries = 0;
        state.batch_update_stats.last_gpu_failures = 0;

        if !topology_changed {
            state.batch_update_stats.unchanged_calls += 1;
            perf_mark("batch-bucket-start");
            perf_mark("batch-bucket-end");
            perf_measure("batch-bucket", "batch-bucket-start", "batch-bucket-end");
            perf_mark("batch-group-end");
            perf_measure("batch-group", "batch-group-start", "batch-group-end");
            return;
        }

        let force_upload = state.batch_layout_dirty;
        perf_mark("batch-bucket-start");
        if let Some(topology) = state.lod_topology.as_ref() {
            batch::rebuild_resident_vertex_lods(
                &state.resident_face_lods,
                topology,
                initial_resident,
                &mut state.resident_vertex_lod_scratch,
                &mut state.resident_vertex_lods,
            );
        } else {
            state.resident_vertex_lods.resize(nf, [1; 3]);
            for (face, vertex_lods) in state.resident_vertex_lods.iter_mut().enumerate() {
                let edges = state.resident_face_lods[face]
                    .unwrap_or(initial_resident)
                    .edge_lods();
                *vertex_lods = [
                    edges[1].max(edges[2]),
                    edges[0].max(edges[2]),
                    edges[0].max(edges[1]),
                ];
            }
        }
        batch::group_resident_faces_into(
            &state.resident_face_lods,
            &state.resident_vertex_lods,
            &state.face_materials,
            &state.face_nodes,
            initial_resident,
            &mut state.batch_groups,
        );
        perf_mark("batch-bucket-end");
        perf_measure("batch-bucket", "batch-bucket-start", "batch-bucket-end");
        perf_mark("batch-group-end");
        perf_measure("batch-group", "batch-group-start", "batch-group-end");

        // Phase 2: retain batches whose key and ordered (face, permutation)
        // membership remain valid. Only changed buckets repack or upload their
        // source streams; capacity growth rebuilds that bucket alone.
        perf_mark("batch-upload-start");
        let mut previous_batches = std::mem::take(&mut state.batches);
        let mut next_batches = BTreeMap::<batch::RenderBatchKey, GpuBatch>::new();
        let mut retained = 0usize;
        let mut updated = 0usize;
        let mut created = 0usize;
        let mut reallocated = 0usize;
        let mut retired = 0usize;
        let mut uploaded_instances = 0usize;
        let mut missing = 0;
        let mut failed = 0;
        let stride = instance_layout::BATCH_TOPOLOGY_STRIDE;

        // One reusable CPU scratch allocation serves only buckets whose
        // membership actually changed.
        let max_batch_size = state.batch_groups.values().map(Vec::len).max().unwrap_or(0);
        state.batch_staging.resize(max_batch_size * stride, 0.0);

        TESS_CACHE.with(|tc| {
            let tc = tc.borrow();
            for (&key, members) in &state.batch_groups {
                let tess = match tc.get(&key.lod) {
                    Some(t) => t,
                    None => { missing += 1; continue; }
                };
                let previous = previous_batches.remove(&key);
                let membership_changed = previous.as_ref()
                    .is_none_or(|batch| batch.members != *members);
                if !force_upload && !membership_changed {
                    retained += 1;
                    next_batches.insert(key, previous.unwrap());
                    continue;
                }

                let batch_floats = match fill_batch_instance_data(
                    &state.resident_face_lods,
                    key,
                    members,
                    &mut state.batch_staging,
                ) {
                    Ok(floats) => floats,
                    Err(error) => {
                        warn!("Failed to pack retained batch {:?}: {error}", key);
                        if let Some(batch) = previous { batch.destroy(gl); retired += 1; }
                        failed += 1;
                        continue;
                    }
                };
                let instance_data = &state.batch_staging[..batch_floats];

                if let Some(mut gpu_batch) = previous {
                    if gpu_batch.can_fit(members.len()) {
                        match gpu_batch.upload_members(gl, members, instance_data) {
                            Ok(()) => {
                                updated += 1;
                                uploaded_instances += members.len();
                                next_batches.insert(key, gpu_batch);
                            }
                            Err(error) => {
                                warn!("Failed to update retained batch {:?}: {error}", key);
                                gpu_batch.destroy(gl);
                                retired += 1;
                                failed += 1;
                            }
                        }
                        continue;
                    }
                    gpu_batch.destroy(gl);
                    retired += 1;
                    reallocated += 1;
                }

                match create_gpu_batch(gl, tess, key, members, instance_data) {
                    Ok(gpu_batch) => {
                        created += 1;
                        uploaded_instances += members.len();
                        next_batches.insert(key, gpu_batch);
                    }
                    Err(error) => {
                        warn!("Failed to create retained batch {:?}: {error}", key);
                        failed += 1;
                    }
                }
            }
        });

        for (_, batch) in previous_batches {
            batch.destroy(gl);
            retired += 1;
        }
        state.batches = next_batches;
        state.render_commands_dirty = true;
        state.batch_update_stats.retained_buckets += retained as u64;
        state.batch_update_stats.updated_buckets += updated as u64;
        state.batch_update_stats.created_buckets += created as u64;
        state.batch_update_stats.reallocated_buckets += reallocated as u64;
        state.batch_update_stats.retired_buckets += retired as u64;
        state.batch_update_stats.uploaded_instances += uploaded_instances as u64;
        state.batch_update_stats.last_missing_atlas_entries = missing as u64;
        state.batch_update_stats.last_gpu_failures = failed as u64;
        perf_mark("batch-upload-end");
        perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");

        debug!(
            "Updated {} GPU batches ({} unchanged, {} uploaded in place, {} created, {} capacity reallocations, {} retired; {} CPU-culled faces, {} LOD corrections, {} atlas entries missing, {} failures)",
            state.batches.len(), retained, updated, created, reallocated, retired,
            culled, lod_corrections, missing, failed,
        );
        state.batch_layout_dirty = missing != 0 || failed != 0;
    });
}

#[wasm_bindgen(js_name = "mr_uploadAnimationPose")]
pub fn mr_upload_animation_pose(
    matrices: &[f32],
    morph_weights: &[f32],
    skin_tex_w: i32,
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
) -> Result<bool, JsValue> {
    validate_pose_stamp(
        clip_time_seconds,
        sample_time_seconds,
        revision,
        continuity_epoch,
    )
    .map_err(|error| JsValue::from_str(&error))?;
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let Some(st) = state.as_mut() else {
            return Ok(false);
        };
        let accepted = st
            .surface_runtime
            .set_timed_pose(
                matrices,
                morph_weights,
                clip_time_seconds,
                sample_time_seconds,
                revision,
                continuity_epoch,
            )
            .map_err(|error| JsValue::from_str(&error))?;
        if !accepted {
            return Ok(false);
        }
        st.renderer
            .joint_ubo()
            .upload(st.renderer.gl(), matrices, morph_weights, skin_tex_w);
        st.render_shadow.pose_changed();
        Ok(true)
    })
}

/// Reset skeletal and morph animation state before accepting a new model.
/// Without this boundary, a static glTF loaded after an animated one is
/// deformed by the previous model's joint UBO and animation textures.
#[wasm_bindgen(js_name = "mr_clearAnimationState")]
pub fn mr_clear_animation_state() {
    STATE.with(|state| {
        if let Some(state) = state.borrow_mut().as_mut() {
            state.renderer.joint_ubo().clear(state.renderer.gl());
            state.renderer.clear_animation_textures();
            state.surface_runtime.clear_animation();
            state.render_shadow.pose_changed();
        }
    });
}

#[wasm_bindgen(js_name = "mr_render")]
pub fn mr_render(mvp: &[f32], mv: &[f32], camera_pos: &[f32]) {
    STATE.with(|s| {
        let mut state = s.borrow_mut();
        let state = match state.as_mut() { Some(s) => s, None => return };
        if state.batches.is_empty() { return; }
        state.render_calls += 1;

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

        sync_render_batches(state);
        refresh_render_shadow_scene(state);
        let has_transmission = if matches!(state.render_mode, RenderMode::Pbr) {
            let default_material = PbrParams::default();
            state.render_batches.iter().any(|batch| {
                let (_, material) = pbr_material_for_index(
                    &state.materials,
                    &default_material,
                    batch.material_index,
                );
                material.transmission_factor > 0.0
            })
        } else {
            false
        };
        if has_transmission && state.blur_program.is_none() {
            match resolve_auxiliary_program(state, AuxiliaryProgram::Blur) {
                Ok(program) => state.blur_program = Some(program),
                Err(error) => {
                    warn!("Transmission blur unavailable; rendering continues: {error}");
                }
            }
        }
        let render_batches = state.render_batches.as_slice();

        let gl = state.renderer.gl();
        state.renderer.begin_frame();
        state.renderer.joint_ubo().bind(gl);
        state.renderer.bind_vertex_textures();

        // Deform and classify each patch once. Every subsequent material,
        // wire, normal, and pick draw consumes these prepared records without
        // CPU visibility readback or repeated position skinning.
        for (gpu_batch, render_batch) in state.batches.values().zip(render_batches) {
            state.renderer.prepare_patch_batch(
                &camera,
                render_batch,
                gpu_batch.prepare_vao,
                gpu_batch.instances.prepared_buf,
                0,
            );
        }

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
                        let (vw, vh) = state.viewport_size;

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
                        // Clear weight attachment: R/G remain neutral for conformal/DoF.
                        // B=1 places the empty scene outside the spherical focus field.
                        gl.draw_buffers(&[glow::NONE, glow::COLOR_ATTACHMENT1]);
                        gl.clear_color(0.5, 0.5, 1.0, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT);
                        // Re-enable both for rendering
                        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);
                    }
                }

                // PBR: two-pass rendering — opaque first, then transparent
                unsafe { gl.use_program(Some(state.renderer.programs().pbr)); }
                let default_mat = PbrParams::default();
                let texture_defaults = [white, black, black, black, white];
                let mut submission_stats = RenderSubmissionStats::default();
                let mut material_updates = 0u64;
                let mut vertex_uniform_updates = 0u64;
                let mut active_material: Option<(usize, bool)> = None;
                let mut active_vertex_state: Option<&RenderBatch> = None;

                // Pass 1: opaque non-transmission
                for batch in render_batches {
                    let (material_slot, mat) = pbr_material_for_index(
                        &state.materials,
                        &default_mat,
                        batch.material_index,
                    );
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

                    let selected = state.selected_node >= 0
                        && batch.node_index == state.selected_node as usize;
                    if active_material != Some((material_slot, selected)) {
                        bind_pbr_material_state(
                            gl,
                            &state.renderer,
                            &state.texture_cache,
                            mat,
                            &texture_defaults,
                            has_env,
                            env_mips,
                            false,
                            selected,
                            state.focus_sphere,
                            state.focus_field_enabled,
                        );
                        active_material = Some((material_slot, selected));
                        material_updates += 1;
                    }

                    apply_batch_winding(gl, batch.orientation_sign, batch.perm_parity);
                    if active_vertex_state
                        .is_none_or(|previous| !same_vertex_uniform_state(previous, batch))
                    {
                        let batch_camera = camera_for_batch(&camera, batch);
                        quilting_renderer::pass::upload_batch_ubo(
                            gl, state.renderer.vtx_ubo(), &batch_camera,
                            1,
                            &batch.euclidean_model, &batch.euclidean_normal,
                        );
                        active_vertex_state = Some(batch);
                        vertex_uniform_updates += 1;
                    }
                    unsafe {
                        gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                        gl.draw_elements_instanced(
                            glow::TRIANGLES, batch.mesh.num_tri_indices,
                            glow::UNSIGNED_INT, batch.mesh.tri_index_offset, batch.mesh.num_instances,
                        );
                    }
                    record_indexed_submission(
                        &mut submission_stats,
                        RenderGeometry::Triangles,
                        batch.mesh.num_tri_indices,
                        batch.mesh.num_instances,
                    );
                }
                // --- Blit opaque framebuffer to scene_color texture for refraction ---
                if has_transmission {
                    unsafe {
                        // Get canvas size from viewport
                        let (vw, vh) = state.viewport_size;

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

                        if let (Some(prog), Some(resources), Some(blur_fbo), Some(blur_tex), Some(sc_fbo)) =
                            (state.blur_program, state.fullscreen_aux.as_ref(), state.blur_fbo, state.blur_tex, state.scene_color_fbo)
                        {
                            let sc_tex = state.scene_color_tex.unwrap();
                            gl.use_program(Some(prog));
                            gl.bind_vertex_array(Some(resources.vao));
                            gl.disable(glow::DEPTH_TEST);
                            gl.disable(glow::BLEND);

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
                                resources.upload_and_bind(gl, [px, 0.0, 0.0, 0.0]);
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
                                resources.upload_and_bind(gl, [0.0, py, 0.0, 0.0]);
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
                active_material = None;
                active_vertex_state = None;
                for batch in render_batches {
                    let (material_slot, mat) = pbr_material_for_index(
                        &state.materials,
                        &default_mat,
                        batch.material_index,
                    );
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

                    let selected = state.selected_node >= 0
                        && batch.node_index == state.selected_node as usize;
                    if active_material != Some((material_slot, selected)) {
                        bind_pbr_material_state(
                            gl,
                            &state.renderer,
                            &state.texture_cache,
                            mat,
                            &texture_defaults,
                            has_env,
                            env_mips,
                            true,
                            selected,
                            state.focus_sphere,
                            state.focus_field_enabled,
                        );
                        active_material = Some((material_slot, selected));
                        material_updates += 1;
                    }

                    apply_batch_winding(gl, batch.orientation_sign, batch.perm_parity);
                    if active_vertex_state
                        .is_none_or(|previous| !same_vertex_uniform_state(previous, batch))
                    {
                        let batch_camera = camera_for_batch(&camera, batch);
                        quilting_renderer::pass::upload_batch_ubo(
                            gl, state.renderer.vtx_ubo(), &batch_camera,
                            1,
                            &batch.euclidean_model, &batch.euclidean_normal,
                        );
                        active_vertex_state = Some(batch);
                        vertex_uniform_updates += 1;
                    }
                    unsafe {
                        gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                        gl.draw_elements_instanced(
                            glow::TRIANGLES, batch.mesh.num_tri_indices,
                            glow::UNSIGNED_INT, batch.mesh.tri_index_offset, batch.mesh.num_instances,
                        );
                    }
                    record_indexed_submission(
                        &mut submission_stats,
                        RenderGeometry::Triangles,
                        batch.mesh.num_tri_indices,
                        batch.mesh.num_instances,
                    );
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
                            let (vw, vh) = state.viewport_size;
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

                state.pbr_draw_calls = state
                    .pbr_draw_calls
                    .saturating_add(submission_stats.draw_calls);
                state.pbr_material_updates += material_updates;
                state.pbr_vertex_uniform_updates += vertex_uniform_updates;
                state.last_render_submission = submission_stats;
                state.render_submission_totals.merge(submission_stats);
                observe_render_submission(state, &camera, submission_stats);
                state.renderer.end_frame();
                return;
            }
            RenderMode::Lod => {
                state
                    .renderer
                    .matcap_ubo()
                    .upload(gl, 0.0, state.matcap_style); // heatmap
                state.renderer.matcap_ubo().bind(gl);
            }
            _ => {
                state.renderer.matcap_ubo().upload(gl, 1.0, state.matcap_style);
                state.renderer.matcap_ubo().bind(gl);
            }
        }

        let submission_stats = state
            .renderer
            .render(state.render_mode, &camera, render_batches);
        observe_render_submission(state, &camera, submission_stats);

        // Highlight pass: overlay picked QB patch with cyan
        if state.highlight_face >= 0 && state.highlight_prog.is_some() {
            render_highlight(state.renderer.gl(), state, &camera);
        }

        state.last_render_submission = submission_stats;
        state.render_submission_totals.merge(submission_stats);
        state.renderer.end_frame();
    });
}

/// Render highlight overlay for the picked face.
/// Requires pick FBO + highlight program to exist (created in mr_pick).
/// Renders pick pass to FBO, then fullscreen overlay where face ID matches.
fn render_highlight(gl: &glow::Context, state: &MainState, camera: &quilting_renderer::pass::Camera) {
    let target_id = state.highlight_face;
    let (highlight_prog, resources, pick_fbo, pick_tex) = match (
        state.highlight_prog, state.fullscreen_aux.as_ref(), state.pick_fbo, state.pick_tex,
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

        for batch in &state.render_batches {
            let batch_camera = camera_for_batch(camera, batch);
            apply_batch_winding(
                gl,
                batch.orientation_sign,
                batch.perm_parity,
            );
            quilting_renderer::pass::upload_batch_ubo(
                gl, state.renderer.vtx_ubo(), &batch_camera,
                1,
                &batch.euclidean_model, &batch.euclidean_normal,
            );
            state.renderer.vtx_ubo().bind(gl);

            gl.bind_vertex_array(Some(batch.mesh.tri_vao));
            gl.draw_elements_instanced(glow::TRIANGLES, batch.mesh.num_tri_indices,
                glow::UNSIGNED_INT, batch.mesh.tri_index_offset, batch.mesh.num_instances);
        }

        // Fullscreen overlay: cyan where pick matches target face
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.viewport(0, 0, vw, vh);
        gl.use_program(Some(highlight_prog));
        gl.disable(glow::DEPTH_TEST);
        gl.enable(glow::BLEND);
        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(pick_tex));
        let tr = (target_id & 255) as f32 / 255.0;
        let tg = ((target_id >> 8) & 255) as f32 / 255.0;
        let tb = ((target_id >> 16) & 255) as f32 / 255.0;
        resources.upload_and_bind(gl, [tr, tg, tb, 0.0]);

        gl.bind_vertex_array(Some(resources.vao));
        gl.draw_arrays(glow::TRIANGLES, 0, 3);

        gl.disable(glow::BLEND);
        gl.enable(glow::DEPTH_TEST);
    }
}
