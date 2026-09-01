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
    create_patch_input_vao, create_patch_visibility_input_vao, EnvironmentMaps, MeshBuffers,
    MeshDraw, PersistentBatchInstances, TessAtlasBuffers, TessBuffers,
};
use quilting_renderer::compute::{
    apply_lod_classification_publication, build_composed_lod_model,
    compare_lod_classifications, diff_packed_lod_classifications,
    exact_f32_slice_fingerprint, prepare_lod_atlas_lookup, prepare_lod_dispatch_state,
    prepare_lod_model, prepared_lod_model_fingerprint,
    unpack_lod_classification_fields, LodAtlasLookup, LodCompute, LodModelResidency,
    StagedLodReadback,
    FLOATS_PER_FACE_OUTPUT, PACKED_LOD_OUTPUT_BYTES_PER_FACE,
};
use quilting_renderer::pass::{
    affine_normal_matrix, affine_orientation_sign, apply_batch_winding,
    camera_for_batch, record_indexed_submission, same_vertex_uniform_state, Camera, RenderBatch,
    webgl_pbr_draws, WebGlPbrCommandPlan, IDENTITY_MATRIX,
};
use quilting_renderer::Renderer;
use quilting_renderer::texture::TextureCache;
use quilting_core::batch;
use quilting_core::instance_layout;
use quilting_core::material::{
    pbr_material_for_index, PbrAlphaMode, PbrMaterial, PbrTextureReferences,
};
use quilting_core::render::{
    patch_preparation_needed, patch_visibility_needed, FocusFieldPacket, FocusPostprocessMode,
    FocusPostprocessPacket, MatcapStyle, PbrDrawClass, RenderBatchSnapshot,
    RenderCommandPlan, RenderEntityTransform, RenderFrame, RenderFrameOptions, RenderPass,
    RenderPoseIdentity, RenderSceneSnapshot, RenderStyle, RenderSubmissionStats, RenderView,
    ValidatedRenderScene, DEFAULT_RENDER_CLEAR_COLOR,
};
#[cfg(feature = "webgpu-backend")]
use quilting_core::render::RenderSubmissionMismatch;
#[cfg(feature = "webgpu-backend")]
use quilting_core::render_evidence::{
    compare_render_images, RenderImageChannelOrder, RenderImageComparison, RenderImageOrigin,
    RenderPickComparison, RenderPickEvidenceReport, RenderPickHit, Rgba8ImageView,
};
use quilting_core::screen_partition::ScreenPartitionPolicy;
use quilting_core::screen_leaf_lod::{
    ScreenMeshNeighborhoodScratch, ScreenMeshTopologyCache,
};
use quilting_core::screen_plan::{
    select_adaptive_screen_faces, AdaptiveScreenFaceCandidate,
    AdaptiveScreenFaceSelectionPolicy, SelectedScreenPatch,
};
use quilting_core::source_bounds::source_focus_bounds;
use crate::adaptive_screen::{
    measure_adaptive_screen_patch, AdaptivePickedConfig, AdaptivePickedRefreshSnapshot,
    AdaptivePickedRuntime, AdaptiveRootShadow, AdaptiveScreenRequest, AdaptiveScreenSelection,
};
use crate::render_shadow::RenderShadowObserver;
use crate::round_shadow::{browser_now_ms, RoundShadowObserver};
use crate::surface_runtime::{
    validate_pose_stamp, ComposedSurfaceWalkSnapshot, SurfaceCameraAnchorSnapshot, SurfaceRuntime,
    SurfaceRuntimeSnapshot, SurfaceWalkReflectionTransportSnapshot,
};
use crate::surface_walk::{ComposedSurfaceWalkResultJs, SurfaceWalkReflectionTransportResultJs};
use crate::timing::{TimingDistribution, TimingDistributionSnapshot};
use hyperscape::interchange::{
    GltfHyperscopePacket, GltfSurfaceFramePinRequest, HyperscapeGltfRuntime,
    RuntimeDiagnosticSnapshot,
};
use hyperscape::{
    CameraBasis, CameraRig, ChamberSide, ContactClassification, FocusSphere, PerspectiveLens,
    SphereReflectionState, SurfaceSample, SurfaceWalkControls, SurfaceWalkInput,
};
use hyperscape_protocol::{ConformalFrameId, WireConformalGenerator};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

/// Floats per material in the array `mr_setMaterials` receives.
const MATERIAL_STRIDE: usize = 50;
const RUNTIME_TIMING_WINDOW_CAPACITY: usize = 2_048;

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
    }
}

fn pbr_draw_class(material: &PbrMaterial) -> PbrDrawClass {
    if material.transmission_factor > 0.0 {
        PbrDrawClass::Transmission
    } else if material.alpha_mode == PbrAlphaMode::Blend {
        PbrDrawClass::Blend
    } else {
        PbrDrawClass::Opaque
    }
}

fn bind_pbr_material_state(
    gl: &glow::Context,
    renderer: &Renderer,
    texture_cache: &TextureCache,
    material: &PbrMaterial,
    texture_defaults: &[glow::Texture; 5],
    has_env_map: bool,
    env_mip_count: f32,
    bind_transmission: bool,
    selected_node: i32,
    focus_sphere: [f32; 4],
    focus_field_enabled: bool,
) {
    let selection_tint = if selected_node >= 0 {
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
        [
            if focus_field_enabled { 1.0 } else { 0.0 },
            selected_node as f32,
            0.0,
            0.0,
        ],
    );
    renderer.pbr_ubo().bind(gl);

    for (unit, &texture) in texture_defaults.iter().enumerate() {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        }
    }
    let bind_texture = |unit: u32, index: Option<u32>| {
        if let Some(index) = index {
            let texture = texture_cache.get(Some(index as usize));
            unsafe {
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            }
        }
    };
    bind_texture(0, material.textures.base_color);
    bind_texture(1, material.textures.metallic_roughness);
    bind_texture(2, material.textures.normal);
    bind_texture(3, material.textures.emissive);
    bind_texture(4, material.textures.occlusion);
    if bind_transmission {
        bind_texture(10, material.textures.transmission);
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

#[derive(Default)]
struct RenderTimingDiagnostics {
    previous_frame_started_ms: Option<f64>,
    frame_delta_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    render_cpu_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RenderTimingSnapshot {
    frame_delta_ms: TimingDistributionSnapshot,
    render_cpu_ms: TimingDistributionSnapshot,
}

impl RenderTimingDiagnostics {
    fn observe(&mut self, started_ms: f64, finished_ms: f64) {
        if let Some(previous_ms) = self.previous_frame_started_ms {
            self.frame_delta_ms.record(started_ms - previous_ms);
        }
        self.previous_frame_started_ms = Some(started_ms);
        self.render_cpu_ms.record(finished_ms - started_ms);
    }

    fn clear(&mut self) {
        self.previous_frame_started_ms = None;
        self.frame_delta_ms.clear();
        self.render_cpu_ms.clear();
    }

    fn snapshot(&self) -> RenderTimingSnapshot {
        RenderTimingSnapshot {
            frame_delta_ms: self.frame_delta_ms.snapshot(),
            render_cpu_ms: self.render_cpu_ms.snapshot(),
        }
    }
}

#[derive(Default)]
struct IncrementalRootGroupShadow {
    enabled: bool,
    adjacency: Option<batch::VertexFaceAdjacency>,
    vertex_max: Vec<u32>,
    face_vertex_lods: Vec<[u32; 3]>,
    vertex_refresh_scratch: batch::ResidentVertexLodRefreshScratch,
    index: batch::RetainedRootGroupIndex,
    groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    comparisons: u64,
    exact_comparisons: u64,
    mismatched_comparisons: u64,
    full_rebuilds: u64,
    incremental_refreshes: u64,
    last_seed_faces: usize,
    last_affected_faces: usize,
    last_dirty_buckets: usize,
    last_rebuilt_members: usize,
    last_mismatched_buckets: usize,
    last_refresh_ms: f64,
    refresh_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    preflight_complete_recommendations: u64,
    preflight_incremental_recommendations: u64,
    certified_complete_recommendations: u64,
    certified_incremental_recommendations: u64,
    last_preflight_decision: Option<batch::RetainedRootGroupingDecision>,
    last_certified_decision: Option<batch::RetainedRootGroupingDecision>,
    certified_complete_candidate_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    certified_incremental_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    last_error: Option<String>,
}

impl IncrementalRootGroupShadow {
    fn set_enabled(&mut self, enabled: bool) {
        if self.enabled == enabled {
            return;
        }
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    fn reset_model(&mut self) {
        let enabled = self.enabled;
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        topology: &quilting_mesh::HalfEdgeMesh,
        residents: &[Option<batch::ResidentLod>],
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        initial: batch::ResidentLod,
        seed_faces: &[usize],
        force_rebuild: bool,
        reference: &BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    ) {
        if !self.enabled {
            return;
        }
        let index_ready = self
            .adjacency
            .as_ref()
            .is_some_and(|adjacency| adjacency.matches(topology))
            && self.index.indexed_faces() == residents.len()
            && self.index.bucket_count() == self.groups.len();
        let preflight = batch::preflight_retained_root_grouping(
            residents.len(),
            seed_faces.len(),
            force_rebuild,
            index_ready,
        );
        if force_rebuild
            || self
                .adjacency
                .as_ref()
                .is_none_or(|adjacency| !adjacency.matches(topology))
        {
            self.adjacency = Some(batch::VertexFaceAdjacency::from_topology(topology));
            self.vertex_max.clear();
            self.face_vertex_lods.clear();
            self.index = batch::RetainedRootGroupIndex::default();
            self.groups.clear();
        }

        let started_ms = browser_now_ms();
        let adjacency = self
            .adjacency
            .as_ref()
            .expect("an enabled incremental root shadow has topology adjacency");
        let affected_faces = batch::refresh_resident_vertex_lods_from_faces(
            residents,
            topology,
            adjacency,
            initial,
            seed_faces,
            &mut self.vertex_max,
            &mut self.face_vertex_lods,
            &mut self.vertex_refresh_scratch,
        );
        let refresh = self.index.refresh_groups_from_faces(
            residents,
            &self.face_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
            affected_faces,
            &mut self.groups,
        );
        let elapsed_ms = browser_now_ms() - started_ms;
        let certified = batch::certify_retained_root_grouping(
            residents.len(),
            seed_faces.len(),
            refresh.affected_faces,
            refresh.rebuilt_members,
            force_rebuild,
            index_ready,
        );
        let mismatched_buckets = reference
            .iter()
            .filter(|(key, members)| self.groups.get(key) != Some(*members))
            .count()
            + self
                .groups
                .keys()
                .filter(|key| !reference.contains_key(key))
                .count();
        let exact = mismatched_buckets == 0;
        self.comparisons = self.comparisons.saturating_add(1);
        self.exact_comparisons = self
            .exact_comparisons
            .saturating_add(u64::from(exact));
        self.mismatched_comparisons = self
            .mismatched_comparisons
            .saturating_add(u64::from(!exact));
        self.full_rebuilds = self
            .full_rebuilds
            .saturating_add(u64::from(refresh.full_rebuild));
        self.incremental_refreshes = self
            .incremental_refreshes
            .saturating_add(u64::from(!refresh.full_rebuild));
        self.last_seed_faces = seed_faces.len();
        self.last_affected_faces = refresh.affected_faces;
        self.last_dirty_buckets = refresh.dirty_buckets;
        self.last_rebuilt_members = refresh.rebuilt_members;
        self.last_mismatched_buckets = mismatched_buckets;
        self.last_refresh_ms = elapsed_ms;
        self.refresh_ms.record(elapsed_ms);
        match preflight.path {
            batch::RetainedRootGroupingPath::Complete => {
                self.preflight_complete_recommendations =
                    self.preflight_complete_recommendations.saturating_add(1);
            }
            batch::RetainedRootGroupingPath::Incremental => {
                self.preflight_incremental_recommendations = self
                    .preflight_incremental_recommendations
                    .saturating_add(1);
            }
        }
        match certified.path {
            batch::RetainedRootGroupingPath::Complete => {
                self.certified_complete_recommendations =
                    self.certified_complete_recommendations.saturating_add(1);
                self.certified_complete_candidate_ms.record(elapsed_ms);
            }
            batch::RetainedRootGroupingPath::Incremental => {
                self.certified_incremental_recommendations = self
                    .certified_incremental_recommendations
                    .saturating_add(1);
                self.certified_incremental_ms.record(elapsed_ms);
            }
        }
        self.last_preflight_decision = Some(preflight);
        self.last_certified_decision = Some(certified);
        self.last_error = (!exact).then(|| {
            format!(
                "incremental root grouping differed in {mismatched_buckets} bucket(s)",
            )
        });
    }

    fn snapshot(&self) -> IncrementalRootGroupShadowSnapshot {
        IncrementalRootGroupShadowSnapshot {
            enabled: self.enabled,
            ready: self.comparisons != 0,
            comparisons: self.comparisons,
            exact_comparisons: self.exact_comparisons,
            mismatched_comparisons: self.mismatched_comparisons,
            full_rebuilds: self.full_rebuilds,
            incremental_refreshes: self.incremental_refreshes,
            indexed_faces: self.index.indexed_faces(),
            buckets: self.index.bucket_count(),
            last_seed_faces: self.last_seed_faces,
            last_affected_faces: self.last_affected_faces,
            last_dirty_buckets: self.last_dirty_buckets,
            last_rebuilt_members: self.last_rebuilt_members,
            last_mismatched_buckets: self.last_mismatched_buckets,
            last_refresh_ms: self.last_refresh_ms,
            refresh_ms: self.refresh_ms.snapshot(),
            preflight_complete_recommendations: self.preflight_complete_recommendations,
            preflight_incremental_recommendations: self.preflight_incremental_recommendations,
            certified_complete_recommendations: self.certified_complete_recommendations,
            certified_incremental_recommendations: self.certified_incremental_recommendations,
            last_preflight_path: self
                .last_preflight_decision
                .map(|decision| decision.path.as_str()),
            last_preflight_reason: self
                .last_preflight_decision
                .map(|decision| decision.reason.as_str()),
            last_certified_path: self
                .last_certified_decision
                .map(|decision| decision.path.as_str()),
            last_certified_reason: self
                .last_certified_decision
                .map(|decision| decision.reason.as_str()),
            certified_complete_candidate_ms: self.certified_complete_candidate_ms.snapshot(),
            certified_incremental_ms: self.certified_incremental_ms.snapshot(),
            adjacency_payload_capacity_bytes: self
                .adjacency
                .as_ref()
                .map_or(0, batch::VertexFaceAdjacency::payload_capacity_bytes),
            index_payload_capacity_bytes: self.index.payload_capacity_bytes(),
            corner_payload_capacity_bytes: self.vertex_max.capacity()
                * std::mem::size_of::<u32>()
                + self.face_vertex_lods.capacity() * std::mem::size_of::<[u32; 3]>(),
            last_error: self.last_error.clone(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct IncrementalRootGroupShadowSnapshot {
    enabled: bool,
    ready: bool,
    comparisons: u64,
    exact_comparisons: u64,
    mismatched_comparisons: u64,
    full_rebuilds: u64,
    incremental_refreshes: u64,
    indexed_faces: usize,
    buckets: usize,
    last_seed_faces: usize,
    last_affected_faces: usize,
    last_dirty_buckets: usize,
    last_rebuilt_members: usize,
    last_mismatched_buckets: usize,
    last_refresh_ms: f64,
    refresh_ms: TimingDistributionSnapshot,
    preflight_complete_recommendations: u64,
    preflight_incremental_recommendations: u64,
    certified_complete_recommendations: u64,
    certified_incremental_recommendations: u64,
    last_preflight_path: Option<&'static str>,
    last_preflight_reason: Option<&'static str>,
    last_certified_path: Option<&'static str>,
    last_certified_reason: Option<&'static str>,
    certified_complete_candidate_ms: TimingDistributionSnapshot,
    certified_incremental_ms: TimingDistributionSnapshot,
    adjacency_payload_capacity_bytes: usize,
    index_payload_capacity_bytes: usize,
    corner_payload_capacity_bytes: usize,
    last_error: Option<String>,
}

#[cfg(feature = "webgpu-backend")]
struct WebGlFrameEvidenceCapture {
    render_call: u64,
    viewport: [u32; 2],
    submission: RenderSubmissionStats,
    focus_postprocess: Option<FocusPostprocessPacket>,
    rgba8_bottom_left: Vec<u8>,
}

#[cfg(feature = "webgpu-backend")]
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendFrameEvidenceReport {
    webgl_render_call: u64,
    webgpu_frame_revision: u64,
    viewport: [u32; 2],
    webgl_submission: RenderSubmissionStats,
    webgpu_logical_submission: RenderSubmissionStats,
    workload_mismatch: RenderSubmissionMismatch,
    webgpu_indirect_draw_calls: u32,
    webgpu_source_instances: u32,
    clear_color: [f32; 4],
    focus_postprocess: Option<FocusPostprocessPacket>,
    image: RenderImageComparison,
}

#[derive(Clone, Copy, Serialize)]
pub(crate) struct ResolvedSurfacePick {
    face: u32,
    node: u32,
    barycentric: [f32; 3],
    source_position: Option<[f64; 3]>,
    output_position: Option<[f64; 3]>,
}

#[cfg(feature = "webgpu-backend")]
struct BackendPickEvidenceCapture {
    webgl_render_call: u64,
    webgpu_frame_revision: u64,
    viewport: [u32; 2],
    pixel: [u32; 2],
    target_epoch: u32,
    expected: Option<RenderPickHit>,
    staged: crate::webgpu_backend::StagedWebGpuPick,
    staging_ms: f64,
    started_ms: f64,
}

#[cfg(feature = "webgpu-backend")]
pub(crate) struct BackendPickAuthorityCapture {
    source_render_call: u64,
    webgpu_frame_revision: u64,
    viewport: [u32; 2],
    staged: crate::webgpu_backend::StagedWebGpuPick,
}

#[cfg(feature = "webgpu-backend")]
impl BackendPickAuthorityCapture {
    pub(crate) fn source_render_call(&self) -> u64 {
        self.source_render_call
    }

    pub(crate) fn webgpu_frame_revision(&self) -> u64 {
        self.webgpu_frame_revision
    }

    pub(crate) fn viewport(&self) -> [u32; 2] {
        self.viewport
    }

    pub(crate) async fn read(
        self,
    ) -> Result<crate::webgpu_backend::WebGpuPickReadback, String> {
        let readback = self.staged.read().await?;
        if readback.source_render_call != self.source_render_call {
            return Err(format!(
                "authoritative backend pick source changed: staged {}, read {}",
                self.source_render_call, readback.source_render_call,
            ));
        }
        if readback.frame_revision != self.webgpu_frame_revision {
            return Err(format!(
                "authoritative backend pick frame changed: staged {}, read {}",
                self.webgpu_frame_revision, readback.frame_revision,
            ));
        }
        Ok(readback)
    }
}

#[cfg(feature = "webgpu-backend")]
#[derive(Clone, Copy)]
struct BackendPickEvidenceStageMetadata {
    webgpu_frame_revision: u64,
    webgl_render_call: u64,
    staging_ms: f64,
}

#[cfg(feature = "webgpu-backend")]
pub(crate) struct BackendPickEvidenceStageReceipt {
    webgl: Option<ResolvedSurfacePick>,
    staged: Result<BackendPickEvidenceStageMetadata, String>,
}

#[cfg(feature = "webgpu-backend")]
impl BackendPickEvidenceStageReceipt {
    pub(crate) fn staged(&self) -> bool {
        self.staged.is_ok()
    }

    pub(crate) fn error(&self) -> Option<&str> {
        self.staged.as_ref().err().map(String::as_str)
    }

    pub(crate) fn into_js(self) -> JsValue {
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"staged".into(), &JsValue::from_bool(self.staged())).ok();
        js_sys::Reflect::set(
            &result,
            &"webgl".into(),
            &self.webgl.map(surface_pick_to_js).unwrap_or(JsValue::NULL),
        )
        .ok();
        match self.staged {
            Ok(metadata) => {
                js_sys::Reflect::set(
                    &result,
                    &"webgpuFrameRevision".into(),
                    &JsValue::from_f64(metadata.webgpu_frame_revision as f64),
                )
                .ok();
                js_sys::Reflect::set(
                    &result,
                    &"webglRenderCall".into(),
                    &JsValue::from_f64(metadata.webgl_render_call as f64),
                )
                .ok();
                js_sys::Reflect::set(
                    &result,
                    &"stagingMs".into(),
                    &JsValue::from_f64(metadata.staging_ms),
                )
                .ok();
            }
            Err(error) => {
                js_sys::Reflect::set(&result, &"error".into(), &JsValue::from_str(&error)).ok();
            }
        }
        result.into()
    }
}

struct MainState {
    renderer: Renderer,
    /// Authoritative drawable size, updated with the GL viewport by mr_resize.
    /// Optional passes consume this instead of synchronously querying GL state.
    viewport_size: (i32, i32),
    texture_cache: TextureCache,
    env_maps: EnvironmentMaps,
    batches: BTreeMap<batch::RenderBatchId, GpuBatch>,
    /// Retained WebGL draw views derived from backend-neutral batch keys.
    /// Rebuilt only when membership or per-node conformal state changes.
    render_batches: Vec<RenderBatch>,
    render_commands_dirty: bool,
    render_command_builds: u64,
    render_calls: u64,
    /// Frames whose visible patch pass was actually submitted through WebGL2.
    /// This excludes preparation, visibility, picking, and overlay-only work.
    webgl_patch_frames: u64,
    #[cfg(feature = "webgpu-backend")]
    /// Changed visible frames submitted directly to the WebGPU surface.
    webgpu_presentation_patch_frames: u64,
    #[cfg(feature = "webgpu-backend")]
    /// Render calls satisfied by retaining an already-current WebGPU surface.
    webgpu_retained_presentation_frames: u64,
    /// Derived backend telemetry. It is never semantic scene state and is
    /// cleared explicitly before a promotion measurement window.
    render_timing: RenderTimingDiagnostics,
    /// Prepared patch records depend on pose, retained batch membership, and
    /// per-entity affine transforms—not on the camera. Camera-dependent
    /// classification lives in a separate one-float visibility stream.
    patch_prepare_dirty: bool,
    patch_prepare_frames: u64,
    skipped_patch_prepare_frames: u64,
    patch_prepare_calls: u64,
    last_prepared_patch_instances: u64,
    last_visibility_mvp: Option<[f32; 16]>,
    last_visibility_command_build: u64,
    patch_visibility_frames: u64,
    skipped_patch_visibility_frames: u64,
    patch_visibility_calls: u64,
    last_visibility_patch_instances: u64,
    pbr_draw_calls: u64,
    pbr_material_updates: u64,
    pbr_vertex_uniform_updates: u64,
    /// Actual indexed patch work submitted by the most recent scene render.
    /// Picking, highlighting, and fullscreen post-processing are separate
    /// auxiliary passes and are deliberately excluded.
    last_render_submission: RenderSubmissionStats,
    /// Saturating session totals for explicit diagnostic queries.
    render_submission_totals: RenderSubmissionStats,
    #[cfg(feature = "webgpu-backend")]
    backend_evidence_requested: bool,
    #[cfg(feature = "webgpu-backend")]
    backend_evidence_capture: Option<WebGlFrameEvidenceCapture>,
    #[cfg(feature = "webgpu-backend")]
    backend_evidence_error: Option<String>,
    #[cfg(feature = "webgpu-backend")]
    backend_pick_evidence: Option<BackendPickEvidenceCapture>,
    #[cfg(feature = "webgpu-backend")]
    backend_pick_evidence_reading: bool,
    /// Opt-in comparison between backend-neutral frame commands and actual
    /// indexed WebGL patch submissions.
    render_shadow: RenderShadowObserver,
    render_scene_dirty: bool,
    /// One immutable extraction epoch shared by command planning, optional
    /// WebGL parity observation, and WebGPU resource publication.
    validated_render_scene: Option<ValidatedRenderScene>,
    render_scene_extractions: u64,
    render_scene_extraction_failures: u64,
    render_scene_error: Option<String>,
    /// Memoized command-affecting identity for the validated scene. Camera,
    /// pose, and uniform-only changes do not rebuild this value.
    render_command_plan: Option<RenderCommandPlan>,
    webgl_pbr_command_plan: Option<WebGlPbrCommandPlan>,
    render_command_plan_builds: u64,
    batch_groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    /// Complete crack-free root grouping for the latest admitted source-face
    /// classification. Adaptive publication keeps this separate from the live
    /// complete frontier so rollback and retained-layer extraction never
    /// confuse two CPU epochs.
    baseline_batch_groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    /// Monotonic identity of the crack-free resident root field. Camera-only
    /// adaptive refreshes do not advance it.
    root_topology_revision: u64,
    /// Exact baseline-root grouping epoch retained across adaptive pose
    /// refreshes: `(root topology revision, batch layout revision)`.
    baseline_group_signature: Option<(u64, u64)>,
    batch_staging: Vec<f32>,
    cached_instances: Vec<f32>,
    /// Raw classifier request before shared-edge and within-face promotion.
    /// Keeping this separate is what permits a former high-detail component to
    /// demote instead of being promoted back by stale resident neighbors.
    requested_face_lods: Vec<Option<batch::ResidentLod>>,
    /// Crack-free promoted topology uploaded for each face.
    resident_face_lods: Vec<Option<batch::ResidentLod>>,
    /// Current C0-continuous visualization LODs, indexed by source face.
    resident_vertex_lods: Vec<[u32; 3]>,
    resident_vertex_lod_scratch: Vec<u32>,
    /// Validated within-face grading policy. Shared-edge equality remains an
    /// independent invariant; this controls only anisotropy/promotion halos.
    lod_grading: batch::FaceLodGrading,
    /// Visibility from the latest worker classification, retained separately
    /// from the drawable standby/resident topology.
    classified_face_visibility: Vec<bool>,
    classified_culled_faces: usize,
    /// Canonical atlas identity shared by worker and same-context classifiers.
    lod_atlas_lookup: Option<LodAtlasLookup>,
    /// Opt-in classifier residency on the renderer's own WebGL context. Until
    /// shadow parity is proven, the worker remains the only live authority.
    same_context_lod: Option<SameContextLod>,
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
    /// Immutable welded identities reused by camera-dependent adaptive
    /// frontiers. Rebuilt only when source geometry changes.
    screen_topology_cache: Option<ScreenMeshTopologyCache>,
    /// Opt-in all-root equivalence gate. It never mutates draw membership.
    adaptive_root_shadow: AdaptiveRootShadow,
    /// Opt-in exact comparison of sparse retained-root reconstruction against
    /// the live complete rebuild. It owns no GPU resources or authority.
    incremental_root_group_shadow: IncrementalRootGroupShadow,
    /// Explicit, bounded live proof that replaces one picked source face with
    /// a reconciled dyadic frontier. Every failed candidate leaves the legacy
    /// root grouping live.
    adaptive_picked: AdaptivePickedRuntime,
    /// Explicit opt-in cutover gate. Shadow staging remains independently
    /// useful when this is false; when true, only order-safe scenes may publish
    /// retained root and overlay GPU layers.
    adaptive_retained_publication_enabled: bool,
    adaptive_retained_publication_error: Option<String>,
    /// Exact mask corresponding to the currently live GPU batch epoch.
    active_suppressed_root_faces: Vec<u32>,
    /// Keep the next adaptive enable/disable handoff rollback-safe even when
    /// the configured mode is being cleared before legacy groups are rebuilt.
    adaptive_batch_transition_pending: bool,
    /// Rust-authoritative stable source address and output-chart walker. The
    /// adjacency layer is built lazily on first attachment.
    surface_runtime: SurfaceRuntime,
    /// Material/node/atlas changes require a rebuild even when face topology
    /// classifications are unchanged.
    batch_layout_dirty: bool,
    /// Monotonic identity for every input that changes render-batch
    /// membership independently of adaptive leaf topology.
    batch_layout_revision: u64,
    batch_update_stats: BatchUpdateStats,
    face_materials: Vec<usize>,
    /// Stable ordinary glTF node index for each source triangle.
    face_nodes: Vec<usize>,
    /// Metadata staged for the next immutable instance upload. Active
    /// picking, focus, welding, and batches continue to observe `face_nodes`
    /// until both CPU embedding and the face-data texture upload succeed.
    pending_face_nodes: Option<Vec<usize>>,
    /// Render-state representative for each source triangle. Ordinary legacy
    /// glTF geometry is world-baked and shares one sentinel state; explicitly
    /// authored/presented nodes remain distinct.
    face_render_nodes: Vec<usize>,
    materials: Vec<PbrMaterial>,
    num_faces: usize,
    render_style: RenderStyle,
    /// Analytic character-matcap profile selected by the browser shell.
    matcap_style: MatcapStyle,
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
    /// Resolved ordinary-world node models admitted through the Rust
    /// application boundary. These override glTF/conformal source packets
    /// without changing the independently retained visibility or opacity.
    authored_node_models: BTreeMap<usize, [f32; 16]>,
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
    focus_postprocess: Option<FocusPostprocessPacket>,
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

struct SameContextLod {
    compute: LodCompute,
    residency: LodModelResidency,
    pending: Option<SameContextLodPending>,
    completed: Option<SameContextLodCompleted>,
    authority_candidate: Option<SameContextLodAuthority>,
    batch_candidate: Option<SameContextLodDecodedCompleted>,
    worker_batch_publication: Option<SameContextLodRequestStamp>,
    worker_batch_snapshot: Option<SameContextLodBatchAuthoritySnapshot>,
    last_authority_stamp: Option<SameContextLodRequestStamp>,
    authority_resident: Vec<f32>,
    authority_packed: Vec<u32>,
    authority_changed_indices: Vec<u32>,
    authority_changed_packed: Vec<u32>,
    /// One byte per latest parity-accepted classifier face. This is a ranking
    /// hint only; topology deltas deliberately ignore it.
    adaptive_face_priorities: Vec<u8>,
    batch_shadow: SameContextLodBatchShadow,
    diagnostics: SameContextLodDiagnostics,
}

impl SameContextLod {
    fn retain_adaptive_face_priorities(&mut self, packed: &[u32]) {
        self.adaptive_face_priorities.clear();
        self.adaptive_face_priorities.reserve(packed.len());
        self.adaptive_face_priorities.extend(packed.iter().map(|&word| {
            unpack_lod_classification_fields(word)
                .expect("packed classifier words were validated at readback")
                .adaptation_priority
        }));
    }

    fn try_compare(&mut self) -> Result<(), String> {
        if self.completed.is_none() || self.authority_candidate.is_none() {
            return Ok(());
        }
        let completed = self.completed.take().expect("completed LOD was checked");
        let authority = self
            .authority_candidate
            .take()
            .expect("authority LOD was checked");
        if !completed.stamp.same_identity(authority.stamp) {
            self.compute.recycle_readback_vector(completed.packed);
            return Err("same-context LOD comparison stamps do not match".to_string());
        }
        let lods = match self.compute.decode_readback_vector(&completed.packed) {
            Ok(lods) => lods,
            Err(error) => {
                self.compute.recycle_readback_vector(completed.packed);
                return Err(error);
            }
        };
        self.diagnostics.legacy_float_decodes =
            self.diagnostics.legacy_float_decodes.saturating_add(1);
        self.diagnostics.last_legacy_float_decode_bytes =
            lods.len().saturating_mul(std::mem::size_of::<f32>());
        let completed_stamp = completed.stamp;
        let completed_packed = completed.packed;
        let completed = SameContextLodDecodedCompleted {
            stamp: completed_stamp,
            lods,
        };
        if let (Some(completed_pose), Some(authority_pose)) =
            (completed.stamp.pose, authority.stamp.pose)
        {
            self.diagnostics.pose_payload_comparisons =
                self.diagnostics.pose_payload_comparisons.saturating_add(1);
            let joint_mismatch = usize::from(
                completed_pose.payload.map(|payload| payload.joint_matrices)
                    != authority_pose.payload.map(|payload| payload.joint_matrices),
            );
            let morph_mismatch = usize::from(
                completed_pose.payload.map(|payload| payload.morph_weights)
                    != authority_pose.payload.map(|payload| payload.morph_weights),
            );
            self.diagnostics.last_joint_matrix_mismatches = joint_mismatch;
            self.diagnostics.last_morph_weight_mismatches = morph_mismatch;
            if joint_mismatch != 0 || morph_mismatch != 0 {
                self.diagnostics.mismatched_pose_payload_comparisons = self
                    .diagnostics
                    .mismatched_pose_payload_comparisons
                .saturating_add(1);
            }
        }
        let completed_fingerprint = exact_f32_slice_fingerprint(&completed.lods);
        let completed_fingerprint = format!(
            "{}:{:016x}",
            completed_fingerprint.0,
            completed_fingerprint.1,
        );
        self.diagnostics.raw_classifier_fingerprint_comparisons = self
            .diagnostics
            .raw_classifier_fingerprint_comparisons
            .saturating_add(1);
        if completed_fingerprint != authority.worker_full_fingerprint {
            self.diagnostics.mismatched_raw_classifier_fingerprints = self
                .diagnostics
                .mismatched_raw_classifier_fingerprints
                .saturating_add(1);
        }
        self.diagnostics.last_same_context_full_fingerprint =
            Some(completed_fingerprint);
        let parity = compare_lod_classifications(&authority.lods, &completed.lods);
        let parity = match parity {
            Ok(parity) => parity,
            Err(error) => {
                self.compute.recycle_decoded_vector(completed.lods);
                self.compute.recycle_readback_vector(completed_packed);
                return Err(error);
            }
        };
        self.diagnostics.comparisons = self.diagnostics.comparisons.saturating_add(1);
        self.diagnostics.last_compared_faces = parity.compared_faces;
        self.diagnostics.last_mismatched_faces = parity.mismatched_faces;
        self.diagnostics.last_mismatched_fields = parity.mismatched_fields;
        self.diagnostics.last_exact = Some(parity.mismatched_fields == 0);
        self.diagnostics.mismatch_examples = parity
            .examples
            .into_iter()
            .map(|mismatch| SameContextLodMismatchSnapshot {
                face: mismatch.face,
                field: mismatch.field,
                expected_bits: mismatch.expected_bits,
                actual_bits: mismatch.actual_bits,
            })
            .collect();
        if parity.mismatched_fields == 0 {
            self.diagnostics.exact_comparisons =
                self.diagnostics.exact_comparisons.saturating_add(1);
            self.retain_adaptive_face_priorities(&completed_packed);
            if let Some(previous) = self.batch_candidate.replace(completed) {
                self.compute.recycle_decoded_vector(previous.lods);
            }
        } else {
            self.diagnostics.mismatched_comparisons =
                self.diagnostics.mismatched_comparisons.saturating_add(1);
            self.adaptive_face_priorities.clear();
            self.compute.recycle_decoded_vector(completed.lods);
        }
        self.compute.recycle_readback_vector(completed_packed);
        Ok(())
    }

    fn cancel_request(&mut self, gl: &glow::Context, request_id: u32) -> bool {
        let mut cancelled = false;
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.stamp.request_id == request_id)
        {
            let pending = self.pending.take().expect("pending LOD was checked");
            unsafe { gl.delete_sync(pending.fence); }
            self.compute.discard_staged_readback(gl, pending.readback);
            cancelled = true;
        }
        if self
            .completed
            .as_ref()
            .is_some_and(|completed| completed.stamp.request_id == request_id)
        {
            let completed = self.completed.take().expect("completed LOD was checked");
            self.compute.recycle_readback_vector(completed.packed);
            cancelled = true;
        }
        if self
            .authority_candidate
            .as_ref()
            .is_some_and(|authority| authority.stamp.request_id == request_id)
        {
            self.authority_candidate = None;
        }
        if self
            .batch_candidate
            .as_ref()
            .is_some_and(|candidate| candidate.stamp.request_id == request_id)
        {
            let candidate = self
                .batch_candidate
                .take()
                .expect("batch candidate was checked");
            self.compute.recycle_decoded_vector(candidate.lods);
            cancelled = true;
        }
        if self
            .worker_batch_publication
            .is_some_and(|stamp| stamp.request_id == request_id)
        {
            self.worker_batch_publication = None;
            self.worker_batch_snapshot = None;
        }
        if self
            .last_authority_stamp
            .is_some_and(|stamp| stamp.request_id == request_id)
        {
            self.last_authority_stamp = None;
        }
        if cancelled {
            self.diagnostics.cancellations = self.diagnostics.cancellations.saturating_add(1);
        }
        cancelled
    }

    fn snapshot(&self) -> SameContextLodShadowSnapshot {
        let state = if self.pending.is_some() {
            "pending-gpu"
        } else if self.completed.is_some() {
            "awaiting-authority"
        } else if self.batch_candidate.is_some() {
            "awaiting-batch-publication"
        } else if self.diagnostics.authoritative_publications != 0 {
            "authority-active"
        } else {
            match self.diagnostics.last_exact {
                Some(true) => "parity-exact",
                Some(false) => "parity-mismatch",
                None => "resident-shadow",
            }
        };
        let (
            readback_buffers,
            readback_vectors,
            decoded_vectors,
            buffer_creations,
            buffer_reallocations,
            vector_creations,
            decoded_vector_creations,
        ) = self.compute.readback_pool_stats();
        SameContextLodShadowSnapshot {
            ready: true,
            state,
            pending_request_id: self.pending.as_ref().map(|pending| pending.stamp.request_id),
            completed_request_id: self
                .completed
                .as_ref()
                .map(|completed| completed.stamp.request_id),
            batch_candidate_request_id: self
                .batch_candidate
                .as_ref()
                .map(|candidate| candidate.stamp.request_id),
            worker_batch_publication_request_id: self
                .worker_batch_publication
                .map(|stamp| stamp.request_id),
            authority_resident_faces: self.authority_resident.len() / FLOATS_PER_FACE_OUTPUT,
            authority_packed_faces: self.authority_packed.len(),
            dispatches: self.diagnostics.dispatches,
            skipped_busy: self.diagnostics.skipped_busy,
            polls: self.diagnostics.polls,
            completions: self.diagnostics.completions,
            authority_updates: self.diagnostics.authority_updates,
            comparisons: self.diagnostics.comparisons,
            exact_comparisons: self.diagnostics.exact_comparisons,
            mismatched_comparisons: self.diagnostics.mismatched_comparisons,
            pose_payload_comparisons: self.diagnostics.pose_payload_comparisons,
            mismatched_pose_payload_comparisons: self
                .diagnostics
                .mismatched_pose_payload_comparisons,
            publication_fingerprint_comparisons: self
                .diagnostics
                .publication_fingerprint_comparisons,
            mismatched_publication_fingerprints: self
                .diagnostics
                .mismatched_publication_fingerprints,
            raw_classifier_fingerprint_comparisons: self
                .diagnostics
                .raw_classifier_fingerprint_comparisons,
            mismatched_raw_classifier_fingerprints: self
                .diagnostics
                .mismatched_raw_classifier_fingerprints,
            legacy_float_decodes: self.diagnostics.legacy_float_decodes,
            last_legacy_float_decode_bytes: self
                .diagnostics
                .last_legacy_float_decode_bytes,
            batch_semantic_comparisons: self.diagnostics.batch_semantic_comparisons,
            exact_batch_semantic_comparisons: self
                .diagnostics
                .exact_batch_semantic_comparisons,
            mismatched_batch_semantic_comparisons: self
                .diagnostics
                .mismatched_batch_semantic_comparisons,
            batch_group_comparisons: self.diagnostics.batch_group_comparisons,
            exact_batch_group_comparisons: self.diagnostics.exact_batch_group_comparisons,
            mismatched_batch_group_comparisons: self
                .diagnostics
                .mismatched_batch_group_comparisons,
            skipped_adaptive_batch_groups: self.diagnostics.skipped_adaptive_batch_groups,
            authoritative_publications: self.diagnostics.authoritative_publications,
            packed_publication_noops: self.diagnostics.packed_publication_noops,
            packed_sparse_publications: self.diagnostics.packed_sparse_publications,
            packed_changed_records: self.diagnostics.packed_changed_records,
            packed_admission_skips: self.diagnostics.packed_admission_skips,
            stale_authoritative_completions: self
                .diagnostics
                .stale_authoritative_completions,
            cancellations: self.diagnostics.cancellations,
            failures: self.diagnostics.failures,
            last_request_id: self.diagnostics.last_request_id,
            last_classified_faces: self.diagnostics.last_classified_faces,
            last_subject_records: self.diagnostics.last_subject_records,
            last_dispatch_ms: self.diagnostics.last_dispatch_ms,
            last_fence_poll_latency_ms: self.diagnostics.last_fence_poll_latency_ms,
            last_readback_ms: self.diagnostics.last_readback_ms,
            last_publication_ms: self.diagnostics.last_publication_ms,
            dispatch_ms: self.diagnostics.dispatch_ms.snapshot(),
            fence_poll_latency_ms: self.diagnostics.fence_poll_latency_ms.snapshot(),
            readback_ms: self.diagnostics.readback_ms.snapshot(),
            publication_ms: self.diagnostics.publication_ms.snapshot(),
            last_readback_bytes: self.diagnostics.last_readback_bytes,
            packed_readback_bytes_per_face: PACKED_LOD_OUTPUT_BYTES_PER_FACE,
            last_compared_faces: self.diagnostics.last_compared_faces,
            last_mismatched_faces: self.diagnostics.last_mismatched_faces,
            last_mismatched_fields: self.diagnostics.last_mismatched_fields,
            last_joint_matrix_mismatches: self.diagnostics.last_joint_matrix_mismatches,
            last_morph_weight_mismatches: self.diagnostics.last_morph_weight_mismatches,
            last_worker_full_fingerprint: self
                .diagnostics
                .last_worker_full_fingerprint
                .clone(),
            last_reconstructed_full_fingerprint: self
                .diagnostics
                .last_reconstructed_full_fingerprint
                .clone(),
            last_same_context_full_fingerprint: self
                .diagnostics
                .last_same_context_full_fingerprint
                .clone(),
            last_requested_mismatches: self.diagnostics.last_requested_mismatches,
            last_resident_mismatches: self.diagnostics.last_resident_mismatches,
            last_visibility_mismatches: self.diagnostics.last_visibility_mismatches,
            last_culled_mismatch: self.diagnostics.last_culled_mismatch,
            last_batch_groups_compared: self.diagnostics.last_batch_groups_compared,
            last_batch_group_mismatches: self.diagnostics.last_batch_group_mismatches,
            last_authoritative_faces: self.diagnostics.last_authoritative_faces,
            last_authoritative_changed_faces: self
                .diagnostics
                .last_authoritative_changed_faces,
            last_packed_publication_unchanged: self
                .diagnostics
                .last_packed_publication_unchanged,
            last_packed_changed_records: self.diagnostics.last_packed_changed_records,
            last_packed_full_snapshot: self.diagnostics.last_packed_full_snapshot,
            last_packed_admission_skipped: self
                .diagnostics
                .last_packed_admission_skipped,
            last_authoritative_pose_revision: self
                .diagnostics
                .last_authoritative_pose_revision,
            last_authoritative_pose_continuity_epoch: self
                .diagnostics
                .last_authoritative_pose_continuity_epoch,
            mismatch_examples: self.diagnostics.mismatch_examples.clone(),
            last_error: self.diagnostics.last_error.clone(),
            readback_buffers,
            readback_vectors,
            decoded_vectors,
            readback_buffer_creations: buffer_creations,
            readback_buffer_reallocations: buffer_reallocations,
            readback_vector_creations: vector_creations,
            decoded_vector_creations,
        }
    }

    fn destroy(mut self, gl: &glow::Context) {
        if let Some(pending) = self.pending.take() {
            unsafe { gl.delete_sync(pending.fence); }
            self.compute.discard_staged_readback(gl, pending.readback);
        }
        if let Some(completed) = self.completed.take() {
            self.compute.recycle_readback_vector(completed.packed);
        }
        if let Some(candidate) = self.batch_candidate.take() {
            self.compute.recycle_decoded_vector(candidate.lods);
        }
        self.compute.destroy(gl);
    }
}

#[derive(Clone, Copy, Debug)]
struct SameContextLodPoseStamp {
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
    payload: Option<SameContextLodPosePayloadFingerprint>,
}

impl SameContextLodPoseStamp {
    fn same_identity(self, other: Self) -> bool {
        self.clip_time_seconds.to_bits() == other.clip_time_seconds.to_bits()
            && self.sample_time_seconds.to_bits() == other.sample_time_seconds.to_bits()
            && self.revision == other.revision
            && self.continuity_epoch == other.continuity_epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SameContextLodF32Fingerprint {
    len: usize,
    hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SameContextLodPosePayloadFingerprint {
    joint_matrices: SameContextLodF32Fingerprint,
    morph_weights: SameContextLodF32Fingerprint,
}

fn same_context_lod_f32_fingerprint(values: &[f32]) -> SameContextLodF32Fingerprint {
    let mut hash = 0xcbf29ce484222325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    SameContextLodF32Fingerprint {
        len: values.len(),
        hash,
    }
}

fn same_context_lod_pose_payload_fingerprint(
    joint_matrices: &[f32],
    morph_weights: &[f32],
) -> SameContextLodPosePayloadFingerprint {
    SameContextLodPosePayloadFingerprint {
        joint_matrices: same_context_lod_f32_fingerprint(joint_matrices),
        morph_weights: same_context_lod_f32_fingerprint(morph_weights),
    }
}

fn same_context_lod_pose_continuity_matches(
    candidate: Option<SameContextLodPoseStamp>,
    retained: Option<(u32, u32)>,
) -> bool {
    match (candidate, retained) {
        (None, None) => true,
        (Some(candidate), Some((_revision, continuity_epoch))) => {
            candidate.continuity_epoch == continuity_epoch
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
struct SameContextLodRequestStamp {
    request_id: u32,
    classified_faces: usize,
    pose: Option<SameContextLodPoseStamp>,
}

impl SameContextLodRequestStamp {
    fn same_identity(self, other: Self) -> bool {
        self.request_id == other.request_id
            && self.classified_faces == other.classified_faces
            && match (self.pose, other.pose) {
                (Some(left), Some(right)) => left.same_identity(right),
                (None, None) => true,
                _ => false,
            }
    }
}

struct SameContextLodPending {
    stamp: SameContextLodRequestStamp,
    fence: glow::Fence,
    readback: StagedLodReadback,
    fence_started_ms: f64,
}

struct SameContextLodCompleted {
    stamp: SameContextLodRequestStamp,
    packed: Vec<u32>,
}

struct SameContextLodDecodedCompleted {
    stamp: SameContextLodRequestStamp,
    lods: Vec<f32>,
}

struct SameContextLodAuthority {
    stamp: SameContextLodRequestStamp,
    lods: Vec<f32>,
    worker_full_fingerprint: String,
}

struct SameContextLodBatchShadow {
    requested: Vec<Option<batch::ResidentLod>>,
    resident: Vec<Option<batch::ResidentLod>>,
    visible: Vec<bool>,
    culled_faces: usize,
    dirty_faces: Vec<usize>,
    balance_scratch: batch::ResidentLodBalanceScratch,
    resident_vertex_lods: Vec<[u32; 3]>,
    resident_vertex_lod_scratch: Vec<u32>,
    batch_groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    prefix_indices: Vec<u32>,
}

struct SameContextLodBatchAuthoritySnapshot {
    stamp: SameContextLodRequestStamp,
    requested: Vec<Option<batch::ResidentLod>>,
    resident: Vec<Option<batch::ResidentLod>>,
    visible: Vec<bool>,
    culled_faces: usize,
    batch_groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
}

impl SameContextLodBatchAuthoritySnapshot {
    fn from_renderer(state: &MainState, stamp: SameContextLodRequestStamp) -> Self {
        Self {
            stamp,
            requested: state.requested_face_lods.clone(),
            resident: state.resident_face_lods.clone(),
            visible: state.classified_face_visibility.clone(),
            culled_faces: state.classified_culled_faces,
            batch_groups: state.batch_groups.clone(),
        }
    }
}

impl SameContextLodBatchShadow {
    fn from_renderer(state: &MainState, num_faces: usize) -> Self {
        let requested = if state.requested_face_lods.len() == num_faces {
            state.requested_face_lods.clone()
        } else {
            vec![None; num_faces]
        };
        let resident = if state.resident_face_lods.len() == num_faces {
            state.resident_face_lods.clone()
        } else {
            vec![None; num_faces]
        };
        let (visible, culled_faces) = if state.classified_face_visibility.len() == num_faces {
            (
                state.classified_face_visibility.clone(),
                state.classified_culled_faces.min(num_faces),
            )
        } else {
            (vec![false; num_faces], num_faces)
        };
        Self {
            requested,
            resident,
            visible,
            culled_faces,
            dirty_faces: Vec::new(),
            balance_scratch: batch::ResidentLodBalanceScratch::default(),
            resident_vertex_lods: Vec::new(),
            resident_vertex_lod_scratch: Vec::new(),
            batch_groups: BTreeMap::new(),
            prefix_indices: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        candidate: &SameContextLodDecodedCompleted,
        num_faces: usize,
        topology: Option<&quilting_mesh::HalfEdgeMesh>,
        grading: batch::FaceLodGrading,
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
    ) -> Result<(), String> {
        let classified_faces = candidate.stamp.classified_faces;
        let face_indices = if classified_faces == num_faces {
            None
        } else {
            let classified_faces = u32::try_from(classified_faces)
                .map_err(|_| "same-context LOD prefix exceeds u32 indexing")?;
            self.prefix_indices.clear();
            self.prefix_indices.extend(0..classified_faces);
            Some(self.prefix_indices.as_slice())
        };
        let standby = bounded_standby_resident_lod();
        let publication = batch::admit_face_lod_publication(
            &candidate.lods,
            face_indices,
            num_faces,
            standby,
            self.culled_faces,
            batch::FaceLodAdmissionBuffers {
                requested: &mut self.requested,
                resident: &mut self.resident,
                visible: &mut self.visible,
                dirty_faces: &mut self.dirty_faces,
            },
        )
        .map_err(|error| error.to_string())?;
        self.culled_faces = publication.culled_faces;
        if let Some(topology) = topology {
            batch::reconcile_resident_lods_from_requests_with_grading(
                &self.requested,
                &mut self.resident,
                topology,
                &self.dirty_faces,
                &mut self.balance_scratch,
                grading,
            );
            batch::rebuild_resident_vertex_lods(
                &self.resident,
                topology,
                standby,
                &mut self.resident_vertex_lod_scratch,
                &mut self.resident_vertex_lods,
            );
        } else {
            for &face in &self.dirty_faces {
                self.resident[face] = self.requested[face];
            }
            self.resident_vertex_lods.resize(num_faces, [1; 3]);
            for (face, vertex_lods) in self.resident_vertex_lods.iter_mut().enumerate() {
                let edges = self.resident[face].unwrap_or(standby).edge_lods();
                *vertex_lods = [
                    edges[1].max(edges[2]),
                    edges[0].max(edges[2]),
                    edges[0].max(edges[1]),
                ];
            }
        }
        batch::group_resident_faces_into(
            &self.resident,
            &self.resident_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            standby,
            &mut self.batch_groups,
        );
        Ok(())
    }
}

#[derive(Default)]
struct SameContextLodDiagnostics {
    dispatches: u64,
    skipped_busy: u64,
    polls: u64,
    completions: u64,
    authority_updates: u64,
    comparisons: u64,
    exact_comparisons: u64,
    mismatched_comparisons: u64,
    pose_payload_comparisons: u64,
    mismatched_pose_payload_comparisons: u64,
    publication_fingerprint_comparisons: u64,
    mismatched_publication_fingerprints: u64,
    raw_classifier_fingerprint_comparisons: u64,
    mismatched_raw_classifier_fingerprints: u64,
    legacy_float_decodes: u64,
    last_legacy_float_decode_bytes: usize,
    batch_semantic_comparisons: u64,
    exact_batch_semantic_comparisons: u64,
    mismatched_batch_semantic_comparisons: u64,
    batch_group_comparisons: u64,
    exact_batch_group_comparisons: u64,
    mismatched_batch_group_comparisons: u64,
    skipped_adaptive_batch_groups: u64,
    authoritative_publications: u64,
    packed_publication_noops: u64,
    packed_sparse_publications: u64,
    packed_changed_records: u64,
    packed_admission_skips: u64,
    stale_authoritative_completions: u64,
    cancellations: u64,
    failures: u64,
    last_request_id: u32,
    last_classified_faces: usize,
    last_subject_records: usize,
    last_dispatch_ms: f64,
    last_fence_poll_latency_ms: f64,
    last_readback_ms: f64,
    last_publication_ms: f64,
    dispatch_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    fence_poll_latency_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    readback_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    publication_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    last_readback_bytes: usize,
    last_compared_faces: usize,
    last_mismatched_faces: usize,
    last_mismatched_fields: usize,
    last_joint_matrix_mismatches: usize,
    last_morph_weight_mismatches: usize,
    last_worker_full_fingerprint: Option<String>,
    last_reconstructed_full_fingerprint: Option<String>,
    last_same_context_full_fingerprint: Option<String>,
    last_requested_mismatches: usize,
    last_resident_mismatches: usize,
    last_visibility_mismatches: usize,
    last_culled_mismatch: bool,
    last_batch_groups_compared: bool,
    last_batch_group_mismatches: usize,
    last_authoritative_faces: usize,
    last_authoritative_changed_faces: usize,
    last_packed_publication_unchanged: bool,
    last_packed_changed_records: usize,
    last_packed_full_snapshot: bool,
    last_packed_admission_skipped: bool,
    last_authoritative_pose_revision: u32,
    last_authoritative_pose_continuity_epoch: u32,
    last_exact: Option<bool>,
    mismatch_examples: Vec<SameContextLodMismatchSnapshot>,
    last_error: Option<String>,
}

impl SameContextLodDiagnostics {
    fn clear_timings(&mut self) {
        self.last_dispatch_ms = 0.0;
        self.last_fence_poll_latency_ms = 0.0;
        self.last_readback_ms = 0.0;
        self.last_publication_ms = 0.0;
        self.dispatch_ms.clear();
        self.fence_poll_latency_ms.clear();
        self.readback_ms.clear();
        self.publication_ms.clear();
    }
}

#[derive(Clone, Serialize)]
struct SameContextLodMismatchSnapshot {
    face: usize,
    field: usize,
    expected_bits: u32,
    actual_bits: u32,
}

#[derive(Serialize)]
struct SameContextLodShadowSnapshot {
    ready: bool,
    state: &'static str,
    pending_request_id: Option<u32>,
    completed_request_id: Option<u32>,
    batch_candidate_request_id: Option<u32>,
    worker_batch_publication_request_id: Option<u32>,
    authority_resident_faces: usize,
    authority_packed_faces: usize,
    dispatches: u64,
    skipped_busy: u64,
    polls: u64,
    completions: u64,
    authority_updates: u64,
    comparisons: u64,
    exact_comparisons: u64,
    mismatched_comparisons: u64,
    pose_payload_comparisons: u64,
    mismatched_pose_payload_comparisons: u64,
    publication_fingerprint_comparisons: u64,
    mismatched_publication_fingerprints: u64,
    raw_classifier_fingerprint_comparisons: u64,
    mismatched_raw_classifier_fingerprints: u64,
    legacy_float_decodes: u64,
    last_legacy_float_decode_bytes: usize,
    batch_semantic_comparisons: u64,
    exact_batch_semantic_comparisons: u64,
    mismatched_batch_semantic_comparisons: u64,
    batch_group_comparisons: u64,
    exact_batch_group_comparisons: u64,
    mismatched_batch_group_comparisons: u64,
    skipped_adaptive_batch_groups: u64,
    authoritative_publications: u64,
    packed_publication_noops: u64,
    packed_sparse_publications: u64,
    packed_changed_records: u64,
    packed_admission_skips: u64,
    stale_authoritative_completions: u64,
    cancellations: u64,
    failures: u64,
    last_request_id: u32,
    last_classified_faces: usize,
    last_subject_records: usize,
    last_dispatch_ms: f64,
    last_fence_poll_latency_ms: f64,
    last_readback_ms: f64,
    last_publication_ms: f64,
    dispatch_ms: TimingDistributionSnapshot,
    fence_poll_latency_ms: TimingDistributionSnapshot,
    readback_ms: TimingDistributionSnapshot,
    publication_ms: TimingDistributionSnapshot,
    last_readback_bytes: usize,
    packed_readback_bytes_per_face: usize,
    last_compared_faces: usize,
    last_mismatched_faces: usize,
    last_mismatched_fields: usize,
    last_joint_matrix_mismatches: usize,
    last_morph_weight_mismatches: usize,
    last_worker_full_fingerprint: Option<String>,
    last_reconstructed_full_fingerprint: Option<String>,
    last_same_context_full_fingerprint: Option<String>,
    last_requested_mismatches: usize,
    last_resident_mismatches: usize,
    last_visibility_mismatches: usize,
    last_culled_mismatch: bool,
    last_batch_groups_compared: bool,
    last_batch_group_mismatches: usize,
    last_authoritative_faces: usize,
    last_authoritative_changed_faces: usize,
    last_packed_publication_unchanged: bool,
    last_packed_changed_records: usize,
    last_packed_full_snapshot: bool,
    last_packed_admission_skipped: bool,
    last_authoritative_pose_revision: u32,
    last_authoritative_pose_continuity_epoch: u32,
    mismatch_examples: Vec<SameContextLodMismatchSnapshot>,
    last_error: Option<String>,
    readback_buffers: usize,
    readback_vectors: usize,
    decoded_vectors: usize,
    readback_buffer_creations: u64,
    readback_buffer_reallocations: u64,
    readback_vector_creations: u64,
    decoded_vector_creations: u64,
}

#[derive(Serialize)]
struct SameContextLodResidencySnapshot {
    ready: bool,
    num_faces: usize,
    num_vertices: u32,
    topology_domains: usize,
    mesh_radius: f32,
    model_fingerprint: String,
    atlas_patches: usize,
    atlas_max_lod: f32,
}

fn clear_same_context_lod(state: &mut MainState) {
    if let Some(residency) = state.same_context_lod.take() {
        residency.destroy(state.renderer.gl());
    }
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

        if let Some(lod) = self.same_context_lod.take() {
            lod.destroy(gl);
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
    adaptive_refresh_calls: u64,
    adaptive_refresh_noops: u64,
    baseline_group_cache_hits: u64,
    baseline_group_cache_misses: u64,
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
    total_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    retain_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    balance_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    bucket_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    vertex_lod_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    render_node_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    group_member_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
    upload_ms: TimingDistribution<RUNTIME_TIMING_WINDOW_CAPACITY>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchUpdateTimingSnapshot {
    total_ms: TimingDistributionSnapshot,
    retain_ms: TimingDistributionSnapshot,
    balance_ms: TimingDistributionSnapshot,
    bucket_ms: TimingDistributionSnapshot,
    vertex_lod_ms: TimingDistributionSnapshot,
    render_node_ms: TimingDistributionSnapshot,
    group_member_ms: TimingDistributionSnapshot,
    upload_ms: TimingDistributionSnapshot,
}

impl BatchUpdateStats {
    fn clear_timings(&mut self) {
        self.total_ms.clear();
        self.retain_ms.clear();
        self.balance_ms.clear();
        self.bucket_ms.clear();
        self.vertex_lod_ms.clear();
        self.render_node_ms.clear();
        self.group_member_ms.clear();
        self.upload_ms.clear();
    }

    fn timing_snapshot(&self) -> BatchUpdateTimingSnapshot {
        BatchUpdateTimingSnapshot {
            total_ms: self.total_ms.snapshot(),
            retain_ms: self.retain_ms.snapshot(),
            balance_ms: self.balance_ms.snapshot(),
            bucket_ms: self.bucket_ms.snapshot(),
            vertex_lod_ms: self.vertex_lod_ms.snapshot(),
            render_node_ms: self.render_node_ms.snapshot(),
            group_member_ms: self.group_member_ms.snapshot(),
            upload_ms: self.upload_ms.snapshot(),
        }
    }
}

fn finish_batch_update_timing(
    stats: &mut BatchUpdateStats,
    total_started_ms: f64,
    upload_started_ms: Option<f64>,
) {
    let finished_ms = browser_now_ms();
    stats
        .upload_ms
        .record(upload_started_ms.map_or(0.0, |started_ms| finished_ms - started_ms));
    stats.total_ms.record(finished_ms - total_started_ms);
}

struct GpuBatch {
    instances: PersistentBatchInstances,
    mesh: MeshBuffers,
    prepare_vao: glow::VertexArray,
    visibility_vao: glow::VertexArray,
    members: Vec<batch::RenderBatchMember>,
    perm_parity: f32,
    material_index: usize,
    render_node_index: usize,
    pose_dirty: bool,
    last_prepared_model: Option<[f32; 16]>,
}

impl GpuBatch {
    fn destroy(self, gl: &glow::Context) {
        unsafe {
            gl.delete_vertex_array(self.prepare_vao);
            gl.delete_vertex_array(self.visibility_vao);
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
        self.pose_dirty = true;
        Ok(())
    }
}

fn fill_batch_instance_data(
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
        if member.leaf_id.depth > instance_layout::BATCH_LEAF_MAX_DEPTH
            || member.leaf_id.domain().is_none()
        {
            return Err(format!(
                "face {face} has unsupported adaptive leaf {}:{}",
                member.leaf_id.depth, member.leaf_id.path,
            ));
        }
        let resident = batch::ResidentLod::from_edge_lods(member.edge_lods);
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
            member.leaf_id.depth as f32,
            member.leaf_id.path as f32,
        ]);
    }
    Ok(required)
}

fn embed_face_node_ids(
    instances: &mut [f32],
    num_faces: usize,
    face_nodes: &[usize],
) -> Result<(), String> {
    let required = num_faces
        .checked_mul(instance_layout::STRIDE)
        .ok_or_else(|| "source-face instance size overflow".to_string())?;
    if instances.len() < required || face_nodes.len() != num_faces {
        return Err(format!(
            "cannot embed {} face nodes into {} floats for {num_faces} faces",
            face_nodes.len(),
            instances.len(),
        ));
    }
    for (face, &node) in face_nodes.iter().enumerate() {
        if node > 16_777_216 {
            return Err(format!("source node {node} exceeds exact f32 identity range"));
        }
        instances[face * instance_layout::STRIDE + instance_layout::offset::NODE_ID] = node as f32;
    }
    Ok(())
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
    let visibility_vao = match create_patch_visibility_input_vao(
        gl,
        &instances.prepared_buf,
        0,
    ) {
        Ok(vao) => vao,
        Err(error) => {
            unsafe { gl.delete_vertex_array(prepare_vao); }
            instances.destroy(gl);
            return Err(error);
        }
    };
    let mesh = match MeshBuffers::from_shared_with_visibility(
        gl,
        tess,
        &instances.prepared_buf,
        0,
        &instances.visibility_buf,
        0,
        members.len() as i32,
    ) {
        Ok(mesh) => mesh,
        Err(error) => {
            unsafe {
                gl.delete_vertex_array(prepare_vao);
                gl.delete_vertex_array(visibility_vao);
            }
            instances.destroy(gl);
            return Err(error);
        }
    };
    Ok(GpuBatch {
        instances,
        mesh,
        prepare_vao,
        visibility_vao,
        members: members.to_vec(),
        perm_parity: key.parity(),
        material_index: key.material_index,
        render_node_index: key.render_node_index,
        pose_dirty: true,
        last_prepared_model: None,
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
    static LOD_DELTA_CURSOR: RefCell<batch::FaceLodDeltaCursor> =
        RefCell::new(batch::FaceLodDeltaCursor::default());
    static TESS_CACHE: RefCell<std::collections::HashMap<[u32; 3], TessBuffers>> =
        RefCell::new(std::collections::HashMap::new());
    /// Owns the immutable buffers referenced by `TESS_CACHE` patch slices.
    static TESS_ATLAS: RefCell<Option<TessAtlasBuffers>> = RefCell::new(None);
    static HYPERSCAPE_RUNTIME: RefCell<Option<BrowserHyperscapeRuntime>> = RefCell::new(None);
}

struct BrowserHyperscapeRuntime {
    runtime: HyperscapeGltfRuntime,
    surface_pins: Vec<ResolvedSurfaceFramePinRequest>,
    last_surface_pin_sample: Option<SurfaceFramePinSampleKey>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AuthoredConformalFrameInput {
    frame_id: ConformalFrameId,
    generators: Vec<WireConformalGenerator>,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedSurfaceFramePinRequest {
    authored: GltfSurfaceFramePinRequest,
    packed_face: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SurfaceFramePinSampleKey {
    pose_stamp: Option<(u32, u32)>,
    target_models: Vec<[f32; 16]>,
}

const IDENTITY_MOBIUS: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
];

/// Draw-state identity for ordinary glTF primitives whose node transforms were
/// already baked into the source positions by the loader. No semantic glTF
/// node can equal this value.
const SHARED_WORLD_BAKED_RENDER_NODE: usize = usize::MAX;

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

    let _ = gl_ctx.get_extension("EXT_color_buffer_float");

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
            webgl_patch_frames: 0,
            #[cfg(feature = "webgpu-backend")]
            webgpu_presentation_patch_frames: 0,
            #[cfg(feature = "webgpu-backend")]
            webgpu_retained_presentation_frames: 0,
            render_timing: RenderTimingDiagnostics::default(),
            patch_prepare_dirty: true,
            patch_prepare_frames: 0,
            skipped_patch_prepare_frames: 0,
            patch_prepare_calls: 0,
            last_prepared_patch_instances: 0,
            last_visibility_mvp: None,
            last_visibility_command_build: 0,
            patch_visibility_frames: 0,
            skipped_patch_visibility_frames: 0,
            patch_visibility_calls: 0,
            last_visibility_patch_instances: 0,
            pbr_draw_calls: 0,
            pbr_material_updates: 0,
            pbr_vertex_uniform_updates: 0,
            last_render_submission: RenderSubmissionStats::default(),
            render_submission_totals: RenderSubmissionStats::default(),
            #[cfg(feature = "webgpu-backend")]
            backend_evidence_requested: false,
            #[cfg(feature = "webgpu-backend")]
            backend_evidence_capture: None,
            #[cfg(feature = "webgpu-backend")]
            backend_evidence_error: None,
            #[cfg(feature = "webgpu-backend")]
            backend_pick_evidence: None,
            #[cfg(feature = "webgpu-backend")]
            backend_pick_evidence_reading: false,
            render_shadow: RenderShadowObserver::default(),
            render_scene_dirty: true,
            validated_render_scene: None,
            render_scene_extractions: 0,
            render_scene_extraction_failures: 0,
            render_scene_error: None,
            render_command_plan: None,
            webgl_pbr_command_plan: None,
            render_command_plan_builds: 0,
            batch_groups: BTreeMap::new(),
            baseline_batch_groups: BTreeMap::new(),
            root_topology_revision: 0,
            baseline_group_signature: None,
            batch_staging: Vec::new(),
            cached_instances: Vec::new(),
            requested_face_lods: Vec::new(),
            resident_face_lods: Vec::new(),
            resident_vertex_lods: Vec::new(),
            resident_vertex_lod_scratch: Vec::new(),
            lod_grading: batch::FaceLodGrading::default(),
            classified_face_visibility: Vec::new(),
            classified_culled_faces: 0,
            lod_atlas_lookup: None,
            same_context_lod: None,
            round_shadow: RoundShadowObserver::default(),
            lod_dirty_faces: Vec::new(),
            lod_balance_scratch: batch::ResidentLodBalanceScratch::default(),
            lod_topology: None,
            screen_topology_cache: None,
            adaptive_root_shadow: AdaptiveRootShadow::default(),
            incremental_root_group_shadow: IncrementalRootGroupShadow::default(),
            adaptive_picked: AdaptivePickedRuntime::default(),
            adaptive_retained_publication_enabled: false,
            adaptive_retained_publication_error: None,
            active_suppressed_root_faces: Vec::new(),
            adaptive_batch_transition_pending: false,
            surface_runtime: SurfaceRuntime::default(),
            batch_layout_dirty: true,
            batch_layout_revision: 1,
            batch_update_stats: BatchUpdateStats::default(),
            face_materials: Vec::new(),
            face_nodes: Vec::new(),
            pending_face_nodes: None,
            face_render_nodes: Vec::new(),
            materials: Vec::new(),
            num_faces: 0,
            render_style: RenderStyle::Pbr,
            matcap_style: MatcapStyle::default(),
            mobius: IDENTITY_MOBIUS,
            mobius_orientation: 1,
            hyperscape_packets: BTreeMap::new(),
            active_hyperscape_camera: None,
            presentation_nodes: BTreeMap::new(),
            authored_node_models: BTreeMap::new(),
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
            focus_postprocess: None,
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

/// Query bounded frame/render distributions. Recording stays entirely below
/// the WASM boundary; serialization and percentile sorting happen only here.
#[wasm_bindgen(js_name = "mr_runtimeTimingDiagnostics")]
pub fn mr_runtime_timing_diagnostics() -> Result<JsValue, JsValue> {
    STATE.with(|state| {
        let state = state.borrow();
        let state = state
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        serde_wasm_bindgen::to_value(&state.render_timing.snapshot())
            .map_err(|error| JsValue::from_str(&error.to_string()))
    })
}

/// Begin a fresh promotion-measurement window without disturbing semantic
/// scene, classifier, residency, or render state.
#[wasm_bindgen(js_name = "mr_resetRuntimeTimingDiagnostics")]
pub fn mr_reset_runtime_timing_diagnostics() -> Result<(), JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        state.render_timing.clear();
        state.batch_update_stats.clear_timings();
        if let Some(lod) = state.same_context_lod.as_mut() {
            lod.diagnostics.clear_timings();
        }
        Ok(())
    })
}

#[wasm_bindgen(js_name = "mr_setRenderMode")]
pub fn mr_set_render_mode(mode: &str) {
    #[cfg(feature = "webgpu-backend")]
    let mut entered_webgpu_diagnostic = false;
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            let next = match mode {
                "pbr" => RenderStyle::Pbr, "matcap" => RenderStyle::Matcap,
                "wire" => RenderStyle::Wire, "normals" => RenderStyle::Normals,
                "both" => RenderStyle::MatcapWire, "lod" => RenderStyle::Lod,
                "stretch" => RenderStyle::Stretch,
                _ => RenderStyle::Pbr,
            };
            #[cfg(feature = "webgpu-backend")]
            {
                entered_webgpu_diagnostic = st.render_style != next
                    && matches!(
                        next,
                        RenderStyle::Matcap
                            | RenderStyle::Normals
                            | RenderStyle::Lod
                            | RenderStyle::Stretch
                    );
            }
            st.render_style = next;
        }
    });
    #[cfg(feature = "webgpu-backend")]
    if entered_webgpu_diagnostic {
        crate::webgpu_backend::force_next_frame();
    }
}

#[wasm_bindgen(js_name = "mr_setMatcapStyle")]
pub fn mr_set_matcap_style(style: &str) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            renderer.matcap_style = match style {
                "aqua" => MatcapStyle::Aqua,
                "citric-acid" => MatcapStyle::CitricAcid,
                "golden-soft" => MatcapStyle::GoldenSoft,
                "soft-studio" => MatcapStyle::SoftStudio,
                _ => MatcapStyle::default(),
            };
        }
    });
}

#[wasm_bindgen(js_name = "mr_setMobius")]
pub fn mr_set_mobius(mobius: &[f32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            let grouping_changed = st.active_hyperscape_camera.is_some();
            st.hyperscape_packets.clear();
            st.active_hyperscape_camera = None;
            for (i, &v) in mobius.iter().take(16).enumerate() { st.mobius[i] = v; }
            let c_len_sq = st.mobius[8..12].iter().map(|value| value * value).sum::<f32>();
            st.mobius_orientation = if c_len_sq > 0.001 { -1 } else { 1 };
            st.render_commands_dirty = true;
            if grouping_changed {
                mark_batch_layout_dirty(st);
            }
        }
    });
}

#[wasm_bindgen(js_name = "mr_setMobiusWithParity")]
pub fn mr_set_mobius_with_parity(mobius: &[f32], orientation_sign: i32) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            let grouping_changed = renderer.active_hyperscape_camera.is_some();
            renderer.hyperscape_packets.clear();
            renderer.active_hyperscape_camera = None;
            for (index, &value) in mobius.iter().take(16).enumerate() {
                renderer.mobius[index] = value;
            }
            renderer.mobius_orientation = if orientation_sign < 0 { -1 } else { 1 };
            renderer.render_commands_dirty = true;
            if grouping_changed {
                mark_batch_layout_dirty(renderer);
            }
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
            let grouping_changed = state.presentation_nodes.keys().ne(next.keys());
            state.presentation_nodes = next;
            state.render_commands_dirty = true;
            if grouping_changed {
                mark_batch_layout_dirty(state);
            }
        }
        true
    })
}

/// Apply resolved ordinary-world transforms already admitted/materialized by
/// the Rust application boundary. Records are `[node, column_major_matrix x16]`.
#[wasm_bindgen(js_name = "mr_setAuthoredNodeTransforms")]
pub fn mr_set_authored_node_transforms(records: &[f32]) -> bool {
    let (records, remainder) = records.as_chunks::<17>();
    if !remainder.is_empty() {
        warn!("Authored node-transform payload has an invalid length");
        return false;
    }
    let mut next = BTreeMap::new();
    for record in records {
        let node = record[0];
        if !node.is_finite()
            || node < 0.0
            || node.fract() != 0.0
            || !record[1..17].iter().all(|value| value.is_finite())
        {
            warn!("Authored node-transform payload contains invalid values");
            return false;
        }
        let mut euclidean_model = [0.0; 16];
        euclidean_model.copy_from_slice(&record[1..]);
        next.insert(node as usize, euclidean_model);
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return false;
        };
        if state.authored_node_models != next {
            let grouping_changed = state.authored_node_models.keys().ne(next.keys());
            state.authored_node_models = next;
            state.render_commands_dirty = true;
            if grouping_changed {
                mark_batch_layout_dirty(state);
            }
        }
        true
    })
}

fn clear_hyperscape_packets() {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            let grouping_changed = renderer.active_hyperscape_camera.is_some();
            renderer.hyperscape_packets.clear();
            renderer.active_hyperscape_camera = None;
            renderer.render_commands_dirty = true;
            if grouping_changed {
                mark_batch_layout_dirty(renderer);
            }
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
        if renderer.active_hyperscape_camera.is_some() != previous_camera.is_some() {
            mark_batch_layout_dirty(renderer);
        }
    });
}

fn conformal_state_for_node(renderer: &MainState, node_index: usize) -> EntityConformalState {
    let state = if let Some(camera) = renderer.active_hyperscape_camera {
        renderer
            .hyperscape_packets
            .get(&(camera, node_index))
            .copied()
            .unwrap_or(EntityConformalState {
                mobius: IDENTITY_MOBIUS,
                orientation_sign: 1,
                euclidean_model: IDENTITY_MATRIX,
            })
    } else {
        let euclidean_model = renderer
            .presentation_nodes
            .get(&node_index)
            .map_or(IDENTITY_MATRIX, |state| state.euclidean_model);
        EntityConformalState {
            mobius: renderer.mobius,
            orientation_sign: renderer.mobius_orientation,
            euclidean_model,
        }
    };
    resolved_node_model(state, renderer.authored_node_models.get(&node_index))
}

fn rebuild_face_render_nodes(renderer: &mut MainState) {
    renderer.face_render_nodes.clear();
    renderer.face_render_nodes.reserve(renderer.face_nodes.len());
    for &node in &renderer.face_nodes {
        let needs_distinct_render_state = renderer.active_hyperscape_camera.is_some()
            || renderer.presentation_nodes.contains_key(&node)
            || renderer.authored_node_models.contains_key(&node);
        renderer.face_render_nodes.push(if needs_distinct_render_state {
            node
        } else {
            SHARED_WORLD_BAKED_RENDER_NODE
        });
    }
}

fn resolved_node_model(
    mut state: EntityConformalState,
    ordinary_world: Option<&[f32; 16]>,
) -> EntityConformalState {
    if let Some(ordinary_world) = ordinary_world {
        state.euclidean_model = *ordinary_world;
    }
    state
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
    let default_material = PbrMaterial::default();
    for (&id, batch) in &renderer.batches {
        let conformal = conformal_state_for_node(renderer, batch.render_node_index);
        let euclidean_orientation = affine_orientation_sign(&conformal.euclidean_model);
        let mut mesh = MeshDraw::from(&batch.mesh);
        if renderer
            .presentation_nodes
            .get(&batch.render_node_index)
            .is_some_and(|state| !state.visible || state.opacity <= 0.0)
        {
            mesh.num_instances = 0;
        }
        let (_, material) = pbr_material_for_index(
            &renderer.materials,
            &default_material,
            batch.material_index,
        );
        render_batches.push(RenderBatch {
            mesh,
            suppress_source_roots: id.layer == batch::RenderBatchLayer::RetainedRoot,
            perm_parity: batch.perm_parity,
            material_index: batch.material_index,
            pbr_class: pbr_draw_class(material),
            render_node_index: batch.render_node_index,
            mobius: conformal.mobius,
            orientation_sign: conformal.orientation_sign * euclidean_orientation,
            euclidean_model: conformal.euclidean_model,
            euclidean_normal: affine_normal_matrix(&conformal.euclidean_model),
        });
    }
    renderer.render_batches = render_batches;
    renderer.render_commands_dirty = false;
    renderer.render_scene_dirty = true;
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
    let mut batches = Vec::with_capacity(renderer.batches.len());
    for ((&id, gpu_batch), render_batch) in
        renderer.batches.iter().zip(&renderer.render_batches)
    {
        if gpu_batch.material_index != render_batch.material_index
            || gpu_batch.render_node_index != render_batch.render_node_index
        {
            return Err("retained GPU batch metadata does not match its render view".to_string());
        }
        let triangle_index_count = u32::try_from(render_batch.mesh.num_tri_indices)
            .map_err(|_| "render batch has a negative triangle index count".to_string())?;
        let line_index_count = u32::try_from(render_batch.mesh.num_line_indices)
            .map_err(|_| "render batch has a negative line index count".to_string())?;
        batches.push(RenderBatchSnapshot {
            id,
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
            pbr_class: render_batch.pbr_class,
        });
    }
    Ok(RenderSceneSnapshot {
        revision: 0,
        materials: renderer.materials.clone(),
        suppressed_root_faces: renderer.active_suppressed_root_faces.clone(),
        batches,
    })
}

fn current_render_view(renderer: &MainState, camera: &Camera) -> RenderView {
    RenderView {
        viewport: [
            renderer.viewport_size.0.max(0) as u32,
            renderer.viewport_size.1.max(0) as u32,
        ],
        mvp: camera.mvp,
        model_view: camera.mv,
        camera_position: camera.camera_pos,
        selected_node: usize::try_from(renderer.selected_node).ok(),
        focus: FocusFieldPacket {
            sphere: renderer.focus_sphere,
            enabled: renderer.focus_field_enabled,
        },
    }
}

fn current_render_frame_options(renderer: &MainState) -> RenderFrameOptions {
    RenderFrameOptions {
        focus_postprocess: renderer.focus_postprocess,
        highlight_face: u32::try_from(renderer.highlight_face).ok(),
        matcap_style: renderer.matcap_style,
    }
}

fn current_render_pose_identity(renderer: &MainState) -> RenderPoseIdentity {
    renderer.surface_runtime.pose_stamp().map_or_else(
        || RenderPoseIdentity::static_asset(renderer.render_command_builds),
        |(revision, continuity_epoch)| {
            RenderPoseIdentity::timed(
                renderer.render_command_builds,
                revision,
                continuity_epoch,
            )
        },
    )
}

fn current_render_frame(
    renderer: &MainState,
    camera: &Camera,
) -> Result<Option<RenderFrame>, String> {
    let Some(plan) = renderer.render_command_plan.as_ref() else {
        return Ok(None);
    };
    RenderFrame::from_command_plan(
        renderer.render_calls,
        current_render_pose_identity(renderer),
        current_render_view(renderer, camera),
        current_render_frame_options(renderer),
        plan,
    )
    .map(Some)
    .map_err(|error| error.to_string())
}

#[cfg(feature = "webgpu-backend")]
fn webgpu_render_style_requested(
    style: RenderStyle,
    live_presentation_requested: bool,
    evidence_requested: bool,
) -> bool {
    quilting_webgpu::supports_patch_presentation_style(style)
        || (style == RenderStyle::Pbr
            && (live_presentation_requested || evidence_requested))
}

#[cfg(feature = "webgpu-backend")]
fn webgpu_frame_requested(renderer: &MainState) -> bool {
    crate::webgpu_backend::frame_contract_required()
        && webgpu_render_style_requested(
            renderer.render_style,
            crate::webgpu_backend::live_presentation_requested(),
            renderer.backend_evidence_requested,
        )
}

#[cfg(feature = "webgpu-backend")]
fn submit_webgpu_frame(
    renderer: &MainState,
    frame: Option<&RenderFrame>,
) -> crate::webgpu_backend::LiveFrameDisposition {
    use crate::webgpu_backend::LiveFrameDisposition;

    if !webgpu_frame_requested(renderer) {
        return LiveFrameDisposition::IncumbentRequired;
    }
    let source_revision = renderer.render_command_builds;
    let scene = match renderer.validated_render_scene.as_ref() {
        Some(scene) if scene.snapshot().revision == source_revision => scene,
        Some(scene) => {
            crate::webgpu_backend::record_frame_prerequisite_failure(format!(
                "shared render scene revision {} does not match source revision {source_revision}",
                scene.snapshot().revision,
            ));
            return LiveFrameDisposition::IncumbentRequired;
        }
        None => {
            crate::webgpu_backend::record_frame_prerequisite_failure(
                renderer
                    .render_scene_error
                    .as_deref()
                    .unwrap_or("shared validated render scene is unavailable"),
            );
            return LiveFrameDisposition::IncumbentRequired;
        }
    };
    if crate::webgpu_backend::needs_scene(source_revision, scene) {
        if let Err(error) = crate::webgpu_backend::replace_scene(
            source_revision,
            scene.clone(),
            &renderer.cached_instances,
        ) {
            crate::webgpu_backend::record_frame_prerequisite_failure(error);
            return LiveFrameDisposition::IncumbentRequired;
        }
    }
    let frame = match frame {
        Some(frame) => frame,
        None => {
            crate::webgpu_backend::record_frame_prerequisite_failure(
                "shared render frame is unavailable",
            );
            return LiveFrameDisposition::IncumbentRequired;
        }
    };
    let (joint_matrices, morph_weights) = match renderer.surface_runtime.current_pose_payload() {
        Ok(pose) => pose,
        Err(error) => {
            crate::webgpu_backend::record_frame_prerequisite_failure(error);
            return LiveFrameDisposition::IncumbentRequired;
        }
    };
    crate::webgpu_backend::submit_frame(
        renderer.render_calls,
        frame,
        &renderer.classified_face_visibility,
        joint_matrices,
        morph_weights,
    )
    .unwrap_or(LiveFrameDisposition::IncumbentRequired)
}

#[cfg(feature = "webgpu-backend")]
fn capture_webgl_frame_evidence(
    state: &MainState,
    style: RenderStyle,
    viewport: (i32, i32),
    render_call: u64,
    camera: &Camera,
    batches: &[RenderBatch],
    incumbent_submission: RenderSubmissionStats,
) -> Result<WebGlFrameEvidenceCapture, String> {
    let renderer = &state.renderer;
    let (width, height) = viewport;
    if width <= 0 || height <= 0 {
        return Err("WebGL image evidence requires a nonzero viewport".to_string());
    }
    let byte_len = usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .and_then(|row| usize::try_from(height).ok().and_then(|height| row.checked_mul(height)))
        .ok_or_else(|| "WebGL image evidence dimensions overflowed".to_string())?;
    let gl = renderer.gl();
    let mut color = None;
    let mut depth = None;
    let mut framebuffer = None;
    let result = unsafe {
        (|| -> Result<WebGlFrameEvidenceCapture, String> {
            let prior_error = gl.get_error();
            if prior_error != glow::NO_ERROR {
                return Err(format!(
                    "WebGL image evidence found preexisting GL error 0x{prior_error:04x}",
                ));
            }
            let color_texture = gl
                .create_texture()
                .map_err(|error| format!("WebGL evidence color texture: {error}"))?;
            color = Some(color_texture);
            gl.bind_texture(glow::TEXTURE_2D, color);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width,
                height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );

            let depth_buffer = gl
                .create_renderbuffer()
                .map_err(|error| format!("WebGL evidence depth buffer: {error}"))?;
            depth = Some(depth_buffer);
            gl.bind_renderbuffer(glow::RENDERBUFFER, depth);
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, width, height);

            let target = gl
                .create_framebuffer()
                .map_err(|error| format!("WebGL evidence framebuffer: {error}"))?;
            framebuffer = Some(target);
            gl.bind_framebuffer(glow::FRAMEBUFFER, framebuffer);
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                color,
                0,
            );
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::DEPTH_ATTACHMENT,
                glow::RENDERBUFFER,
                depth,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                return Err(format!(
                    "WebGL evidence framebuffer is incomplete: 0x{status:04x}",
                ));
            }

            gl.viewport(0, 0, width, height);
            // Preserve the incumbent RGB clear while leaving background alpha
            // available as an explicit fragment-coverage mask for parity
            // evidence.
            let [red, green, blue, _] = DEFAULT_RENDER_CLEAR_COLOR;
            gl.clear_color(red, green, blue, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            let submission = renderer.render(style, camera, batches);
            if submission != incumbent_submission {
                return Err(format!(
                    "WebGL evidence rerender changed submission work: incumbent={incumbent_submission:?}, evidence={submission:?}",
                ));
            }
            if state.highlight_face >= 0 {
                render_highlight_to(
                    gl,
                    state,
                    camera,
                    Some(target),
                    (width, height),
                );
            }
            let mut rgba8_bottom_left = vec![0u8; byte_len];
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            gl.read_pixels(
                0,
                0,
                width,
                height,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut rgba8_bottom_left)),
            );
            let error = gl.get_error();
            if error != glow::NO_ERROR {
                return Err(format!(
                    "WebGL image evidence readback failed with GL error 0x{error:04x}",
                ));
            }
            Ok(WebGlFrameEvidenceCapture {
                render_call,
                viewport: [width as u32, height as u32],
                submission,
                focus_postprocess: None,
                rgba8_bottom_left,
            })
        })()
    };
    unsafe {
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.viewport(0, 0, width, height);
        if let Some(framebuffer) = framebuffer {
            gl.delete_framebuffer(framebuffer);
        }
        if let Some(depth) = depth {
            gl.delete_renderbuffer(depth);
        }
        if let Some(color) = color {
            gl.delete_texture(color);
        }
    }
    result
}

#[cfg(feature = "webgpu-backend")]
fn capture_current_webgl_pbr_frame_evidence(
    gl: &glow::Context,
    viewport: (i32, i32),
    render_call: u64,
    submission: RenderSubmissionStats,
    focus_postprocess: Option<FocusPostprocessPacket>,
) -> Result<WebGlFrameEvidenceCapture, String> {
    let (width, height) = viewport;
    if width <= 0 || height <= 0 {
        return Err("WebGL PBR image evidence requires a nonzero viewport".to_string());
    }
    let byte_len = usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .and_then(|row| usize::try_from(height).ok().and_then(|height| row.checked_mul(height)))
        .ok_or_else(|| "WebGL PBR image evidence dimensions overflowed".to_string())?;
    let mut rgba8_bottom_left = vec![0u8; byte_len];
    unsafe {
        let prior_error = gl.get_error();
        if prior_error != glow::NO_ERROR {
            return Err(format!(
                "WebGL PBR image evidence found preexisting GL error 0x{prior_error:04x}",
            ));
        }
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.read_pixels(
            0,
            0,
            width,
            height,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelPackData::Slice(Some(&mut rgba8_bottom_left)),
        );
        let error = gl.get_error();
        if error != glow::NO_ERROR {
            return Err(format!(
                "WebGL PBR image evidence readback failed with GL error 0x{error:04x}",
            ));
        }
    }
    // The visible default framebuffer has opaque clear alpha, while the
    // parity target uses transparent alpha as its coverage mask. Canonicalize
    // only exact incumbent-clear pixels; ordinary opaque PBR fragments retain
    // their authored alpha.
    canonicalize_incumbent_clear_alpha(
        &mut rgba8_bottom_left,
        [width as u32, height as u32],
        usize::try_from(width).unwrap_or(0).saturating_mul(4),
    )?;
    Ok(WebGlFrameEvidenceCapture {
        render_call,
        viewport: [width as u32, height as u32],
        submission,
        focus_postprocess,
        rgba8_bottom_left,
    })
}

#[cfg(feature = "webgpu-backend")]
fn canonicalize_incumbent_clear_alpha(
    rgba8: &mut [u8],
    size: [u32; 2],
    bytes_per_row: usize,
) -> Result<(), String> {
    let row_bytes = usize::try_from(size[0])
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or_else(|| "backend evidence row size overflowed".to_string())?;
    if bytes_per_row < row_bytes {
        return Err("backend evidence row pitch is smaller than its viewport".to_string());
    }
    let required = usize::try_from(size[1])
        .ok()
        .and_then(|height| height.checked_sub(1))
        .and_then(|last_row| last_row.checked_mul(bytes_per_row))
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| "backend evidence byte length overflowed".to_string())?;
    if rgba8.len() < required {
        return Err("backend evidence image is shorter than its viewport".to_string());
    }
    for y in 0..size[1] as usize {
        let row_start = y * bytes_per_row;
        for pixel in rgba8[row_start..row_start + row_bytes].chunks_exact_mut(4) {
            if pixel[0] == 51 && pixel[1] == 51 && (pixel[2] == 76 || pixel[2] == 77) {
                pixel[3] = 0;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "webgpu-backend")]
fn backend_frame_evidence_supports_composition(
    focus_postprocess: Option<FocusPostprocessPacket>,
) -> bool {
    focus_postprocess.is_none() || crate::webgpu_backend::focus_evidence_prerequisites_ready()
}

#[cfg(feature = "webgpu-backend")]
#[wasm_bindgen(js_name = "mr_requestBackendFrameEvidence")]
pub fn mr_request_backend_frame_evidence() -> Result<bool, JsValue> {
    STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        if state.render_style != RenderStyle::Pbr
            && !quilting_webgpu::supports_patch_presentation_style(state.render_style)
        {
            return Err(JsValue::from_str(
                "backend image evidence requires a WebGPU-supported diagnostic or basic PBR mode",
            ));
        }
        if !crate::webgpu_backend::frame_evidence_ready() {
            return Err(JsValue::from_str(
                "backend image evidence requires a ready WebGPU device",
            ));
        }
        if !backend_frame_evidence_supports_composition(state.focus_postprocess) {
            return Err(JsValue::from_str(
                "focus backend evidence requires WebGPU focus pipelines and environment residency",
            ));
        }
        if state.highlight_face >= 0
            && (state.highlight_prog.is_none()
                || state.fullscreen_aux.is_none()
                || state.pick_fbo.is_none()
                || state.pick_tex.is_none()
                || state.pick_size.0 <= 0
                || state.pick_size.1 <= 0)
        {
            return Err(JsValue::from_str(
                "backend image evidence requires resident WebGL highlight resources",
            ));
        }
        if state.viewport_size.0 <= 0 || state.viewport_size.1 <= 0 {
            return Err(JsValue::from_str(
                "backend image evidence requires a nonzero viewport",
            ));
        }
        if state.render_style == RenderStyle::Pbr {
            if !crate::webgpu_backend::pbr_evidence_ready() {
                return Err(JsValue::from_str(
                    "PBR backend evidence requires a WebGPU device with resident environment maps",
                ));
            }
            // Evidence must preflight the exact structural epoch that the next
            // WebGPU frame will publish. Synchronize pending batch semantics
            // and cross the shared validation boundary once instead of
            // constructing a private request-time snapshot.
            sync_render_batches(state);
            refresh_validated_render_scene(state, true)
                .map_err(|error| JsValue::from_str(&error))?;
            let scene = state
                .validated_render_scene
                .as_ref()
                .filter(|scene| scene.snapshot().revision == state.render_command_builds)
                .ok_or_else(|| {
                    JsValue::from_str(
                        "PBR backend evidence requires the current validated render scene",
                    )
                })?;
            let options = RenderFrameOptions {
                focus_postprocess: state.focus_postprocess,
                highlight_face: u32::try_from(state.highlight_face).ok(),
                matcap_style: state.matcap_style,
            };
            let supported = if options.focus_postprocess.is_some() {
                quilting_webgpu::supports_focus_pbr_frame(scene.snapshot(), options)
            } else {
                quilting_webgpu::supports_basic_pbr_frame(scene.snapshot(), options)
            };
            if !supported {
                return Err(JsValue::from_str(
                    "PBR backend evidence requires opaque materials without transmission or sheen and an exactly supported composition path",
                ));
            }
        }
        state.backend_evidence_requested = true;
        state.backend_evidence_capture = None;
        state.backend_evidence_error = None;
        crate::webgpu_backend::request_frame_evidence();
        Ok(true)
    })
}

#[cfg(feature = "webgpu-backend")]
#[wasm_bindgen(js_name = "mr_compareBackendFrameEvidence")]
pub async fn mr_compare_backend_frame_evidence() -> Result<JsValue, JsValue> {
    let webgl = STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        if let Some(error) = state.backend_evidence_error.take() {
            return Err(error);
        }
        state
            .backend_evidence_capture
            .take()
            .ok_or_else(|| "backend frame evidence has not completed a render".to_string())
    })
    .map_err(|error| JsValue::from_str(&error))?;
    let staged = crate::webgpu_backend::stage_frame_evidence()
        .map_err(|error| JsValue::from_str(&error))?;
    if staged.source_render_call != webgl.render_call || staged.viewport != webgl.viewport {
        return Err(JsValue::from_str(&format!(
            "backend evidence frame mismatch: WebGL call {} {:?}, WebGPU call {} {:?}",
            webgl.render_call, webgl.viewport, staged.source_render_call, staged.viewport,
        )));
    }
    if staged.focus_postprocess != webgl.focus_postprocess {
        return Err(JsValue::from_str(
            "backend evidence focus packets do not match",
        ));
    }
    let mut webgpu = staged
        .image
        .read()
        .await
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    canonicalize_incumbent_clear_alpha(
        &mut webgpu.bytes,
        webgpu.size,
        webgpu.bytes_per_row,
    )
    .map_err(|error| JsValue::from_str(&error))?;
    let webgl_row_bytes = usize::try_from(webgl.viewport[0])
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or_else(|| JsValue::from_str("WebGL evidence row size overflowed"))?;
    let webgl_view = Rgba8ImageView::new(
        webgl.viewport,
        webgl_row_bytes,
        RenderImageOrigin::BottomLeft,
        RenderImageChannelOrder::Rgba,
        &webgl.rgba8_bottom_left,
    )
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let webgpu_view = webgpu
        .view()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let image = compare_render_images(webgl_view, webgpu_view, 0)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let workload_mismatch =
        RenderSubmissionMismatch::between(webgl.submission, staged.logical_submission);
    serde_wasm_bindgen::to_value(&BackendFrameEvidenceReport {
        webgl_render_call: webgl.render_call,
        webgpu_frame_revision: staged.frame_revision,
        viewport: webgl.viewport,
        webgl_submission: webgl.submission,
        webgpu_logical_submission: staged.logical_submission,
        workload_mismatch,
        webgpu_indirect_draw_calls: staged.indirect_draw_calls,
        webgpu_source_instances: staged.source_instances,
        clear_color: DEFAULT_RENDER_CLEAR_COLOR,
        focus_postprocess: webgl.focus_postprocess,
        image,
    })
    .map_err(|error| JsValue::from_str(&error.to_string()))
}

fn refresh_validated_render_scene(
    renderer: &mut MainState,
    backend_scene_required: bool,
) -> Result<(), String> {
    if !renderer.render_shadow.is_enabled() && !backend_scene_required {
        return Ok(());
    }
    if !renderer.render_scene_dirty && renderer.validated_render_scene.is_some() {
        return Ok(());
    }
    let extracted = extract_render_scene(renderer).and_then(|mut scene| {
        scene.revision = renderer.render_command_builds;
        ValidatedRenderScene::new(scene).map_err(|error| error.to_string())
    });
    renderer.render_scene_dirty = false;
    match extracted {
        Ok(scene) => {
            renderer.render_scene_extractions =
                renderer.render_scene_extractions.saturating_add(1);
            renderer.render_scene_error = None;
            if renderer.render_shadow.is_enabled() {
                renderer.render_shadow.replace_scene(scene.clone());
            }
            renderer.validated_render_scene = Some(scene);
            renderer.render_command_plan = None;
            renderer.webgl_pbr_command_plan = None;
            Ok(())
        }
        Err(error) => {
            renderer.render_scene_extraction_failures =
                renderer.render_scene_extraction_failures.saturating_add(1);
            renderer.render_scene_error = Some(error.clone());
            renderer.validated_render_scene = None;
            renderer.render_command_plan = None;
            renderer.webgl_pbr_command_plan = None;
            renderer.render_shadow.record_extraction_error(&error);
            Err(error)
        }
    }
}

fn refresh_render_command_plan(renderer: &mut MainState, backend_plan_required: bool) {
    if !renderer.render_shadow.is_enabled() && !backend_plan_required {
        return;
    }
    let style = renderer.render_style;
    let options = current_render_frame_options(renderer);
    let Some(scene) = renderer.validated_render_scene.as_ref() else {
        return;
    };
    if renderer
        .render_command_plan
        .as_ref()
        .is_some_and(|plan| plan.matches(scene, style, options))
    {
        return;
    }
    match RenderCommandPlan::build(scene, style, options) {
        Ok(plan) => {
            let pbr_plan = if renderer.render_shadow.is_enabled() && style == RenderStyle::Pbr {
                match WebGlPbrCommandPlan::build(
                    plan.execution(),
                    renderer.render_batches.as_slice(),
                ) {
                    Ok(plan) => Some(plan),
                    Err(error) => {
                        renderer.render_shadow.record_execution_fallback(error);
                        None
                    }
                }
            } else {
                None
            };
            renderer.render_command_plan = Some(plan);
            renderer.webgl_pbr_command_plan = pbr_plan;
            renderer.render_command_plan_builds =
                renderer.render_command_plan_builds.saturating_add(1);
        }
        Err(error) => {
            renderer.render_command_plan = None;
            renderer.webgl_pbr_command_plan = None;
            renderer.render_shadow.record_extraction_error(error);
        }
    }
}

fn prepare_render_shadow_observation(
    renderer: &mut MainState,
    frame: Option<&RenderFrame>,
) -> Option<u64> {
    renderer.render_shadow.prepare_observation(frame?)
}

fn observe_render_submission(
    renderer: &mut MainState,
    prepared_revision: Option<u64>,
    frame: Option<&RenderFrame>,
    actual: RenderSubmissionStats,
) {
    if let (Some(prepared_revision), Some(frame)) = (prepared_revision, frame) {
        if prepared_revision == frame.revision {
            renderer.render_shadow.observe_prepared(frame, actual);
        }
    }
}

fn resolve_surface_frame_pin_faces(
    requests: &[GltfSurfaceFramePinRequest],
) -> Result<Vec<ResolvedSurfaceFramePinRequest>, String> {
    STATE.with(|state| {
        let state = state.borrow();
        let renderer = state
            .as_ref()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        requests
            .iter()
            .map(|&authored| {
                let local_face = authored.face as usize;
                let packed_face = packed_face_for_entity_local_face(
                    &renderer.face_nodes,
                    authored.target_node,
                    local_face,
                )
                .ok_or_else(|| {
                    format!(
                        "surface pin {} targets missing entity-local face {} on node {}",
                        authored.pin_index, authored.face, authored.target_node
                    )
                })?;
                Ok(ResolvedSurfaceFramePinRequest {
                    authored,
                    packed_face,
                })
            })
            .collect()
    })
}

fn packed_face_for_entity_local_face(
    face_nodes: &[usize],
    target_node: usize,
    local_face: usize,
) -> Option<usize> {
    face_nodes
        .iter()
        .enumerate()
        .filter_map(|(face, &node)| (node == target_node).then_some(face))
        .nth(local_face)
}

fn sample_surface_frame_pins(
    requests: &[ResolvedSurfaceFramePinRequest],
    previous: Option<&SurfaceFramePinSampleKey>,
) -> Result<Option<(SurfaceFramePinSampleKey, Vec<SurfaceSample>)>, String> {
    if requests.is_empty() {
        return Ok(None);
    }
    STATE.with(|state| {
        let state = state.borrow();
        let renderer = state
            .as_ref()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        let key = SurfaceFramePinSampleKey {
            pose_stamp: renderer.surface_runtime.pose_stamp(),
            target_models: requests
                .iter()
                .map(|request| {
                    conformal_state_for_node(renderer, request.authored.target_node)
                        .euclidean_model
                })
                .collect(),
        };
        if previous == Some(&key) {
            return Ok(None);
        }
        let samples = requests
            .iter()
            .zip(&key.target_models)
            .map(|(request, &model)| {
                if renderer.face_nodes.get(request.packed_face).copied()
                    != Some(request.authored.target_node)
                {
                    return Err(format!(
                        "surface pin {} no longer matches packed face {}",
                        request.authored.pin_index, request.packed_face
                    ));
                }
                renderer.surface_runtime.sample_face_in_parent_chart(
                    &renderer.cached_instances,
                    request.packed_face,
                    request.authored.barycentric,
                    model,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Some((key, samples)))
    })
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
    let surface_pins = match resolve_surface_frame_pin_faces(
        &runtime.surface_frame_pin_requests(),
    ) {
        Ok(surface_pins) => surface_pins,
        Err(error) => {
            web_sys::console::warn_1(&format!("Hyperscape surface pins: {error}").into());
            HYPERSCAPE_RUNTIME.with(|runtime| *runtime.borrow_mut() = None);
            clear_hyperscape_packets();
            return false;
        }
    };
    apply_hyperscape_packets(&runtime.packets_by_node());
    HYPERSCAPE_RUNTIME.with(|slot| {
        *slot.borrow_mut() = Some(BrowserHyperscapeRuntime {
            runtime,
            surface_pins,
            last_surface_pin_sample: None,
        })
    });
    true
}

/// Apply one complete materialized authored conformal-frame projection to the
/// resident Hyperscape runtime. The JSON array is stable-ID keyed and replacing:
/// each item contains `frameId` plus its complete generator word. A duplicate,
/// unknown, invalid, or surface-pin-owned frame leaves the runtime unchanged.
#[wasm_bindgen(js_name = "mr_applyAuthoredConformalFrames")]
pub fn mr_apply_authored_conformal_frames(frames_json: &str) -> bool {
    let frames = match serde_json::from_str::<Vec<AuthoredConformalFrameInput>>(frames_json) {
        Ok(frames) => frames,
        Err(error) => {
            web_sys::console::warn_1(
                &format!("authored conformal frames: invalid JSON: {error}").into(),
            );
            return false;
        }
    };
    let mut transforms = BTreeMap::new();
    for frame in frames {
        if transforms
            .insert(frame.frame_id, frame.generators)
            .is_some()
        {
            web_sys::console::warn_1(
                &format!("authored conformal frames: duplicate ID {}", frame.frame_id).into(),
            );
            return false;
        }
    }
    HYPERSCAPE_RUNTIME.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(browser) = slot.as_mut() else {
            web_sys::console::warn_1(
                &"authored conformal frames: no resident Hyperscape runtime".into(),
            );
            return false;
        };
        if let Err(error) = browser
            .runtime
            .apply_authored_conformal_frame_transforms(&transforms)
        {
            web_sys::console::warn_1(&format!("authored conformal frames: {error}").into());
            return false;
        }
        apply_hyperscape_packets(&browser.runtime.packets_by_node());
        browser.last_surface_pin_sample = None;
        true
    })
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
    let surface_pins = HYPERSCAPE_RUNTIME.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|runtime| {
                (
                    runtime.surface_pins.clone(),
                    runtime.last_surface_pin_sample.clone(),
                )
            })
    });
    let Some((surface_pins, last_surface_pin_sample)) = surface_pins else {
        return JsValue::NULL;
    };
    let sampled_pins = sample_surface_frame_pins(
        &surface_pins,
        last_surface_pin_sample.as_ref(),
    );
    let snapshot = HYPERSCAPE_RUNTIME.with(|slot| {
        let mut browser_runtime = slot.borrow_mut();
        let browser_runtime = browser_runtime.as_mut()?;
        let mut sampling_diagnostic = None;
        match sampled_pins {
            Ok(Some((key, samples))) => {
                if let Err(error) = browser_runtime
                    .runtime
                    .submit_surface_frame_pin_samples(samples)
                {
                    sampling_diagnostic = Some(error.to_string());
                } else {
                    browser_runtime.last_surface_pin_sample = Some(key);
                }
            }
            Ok(None) => {}
            Err(error) => sampling_diagnostic = Some(error),
        }
        browser_runtime
            .runtime
            .tick(Duration::from_secs_f64(delta_seconds));
        let mut diagnostics = browser_runtime.runtime.diagnostics().to_vec();
        if let Some(diagnostic) = sampling_diagnostic {
            diagnostics.push(diagnostic);
        }
        Some((
            browser_runtime.runtime.packets_by_node(),
            diagnostics,
            include_scene_diagnostics.then(|| browser_runtime.runtime.diagnostic_snapshot()),
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

#[allow(clippy::too_many_arguments)]
fn focus_postprocess_packet(
    max_distance: f32,
    blur_strength: f32,
    mode: u32,
    focus: f32,
    bandwidth: f32,
    normalize: bool,
    stretch_min: f32,
    stretch_max: f32,
    blur_passes: u32,
    kawase_passes: u32,
    kawase_offset: f32,
) -> Result<FocusPostprocessPacket, &'static str> {
    let blur_radius_pixels = u16::try_from(max_distance as i64)
        .ok()
        .filter(|radius| f32::from(*radius) == max_distance)
        .ok_or("focus blur radius must be an exact u16")?;
    let mode = u8::try_from(mode)
        .ok()
        .and_then(FocusPostprocessMode::from_wire_index)
        .ok_or("focus postprocess mode is invalid")?;
    let gaussian_passes = u8::try_from(blur_passes)
        .map_err(|_| "Gaussian pass count does not fit u8")?;
    let kawase_passes =
        u8::try_from(kawase_passes).map_err(|_| "Kawase pass count does not fit u8")?;
    FocusPostprocessPacket {
        mode,
        blur_radius_pixels,
        blur_strength,
        focus_coordinate: focus,
        bandwidth,
        normalize_range: normalize,
        stretch_range: [stretch_min, stretch_max],
        gaussian_passes,
        kawase_passes,
        kawase_offset,
    }
    .validate()
    .map_err(|_| "focus postprocess packet is outside the renderer contract")
}

#[wasm_bindgen(js_name = "mr_setFuzzy")]
#[allow(clippy::too_many_arguments)]
pub fn mr_set_fuzzy(
    enabled: bool,
    max_distance: f32,
    blur_strength: f32,
    mode: u32,
    focus: f32,
    bandwidth: f32,
    normalize: bool,
    stretch_min: f32,
    stretch_max: f32,
    blur_passes: u32,
    kawase_passes: u32,
    kawase_offset: f32,
) -> bool {
    let Ok(packet) = focus_postprocess_packet(
        max_distance,
        blur_strength,
        mode,
        focus,
        bandwidth,
        normalize,
        stretch_min,
        stretch_max,
        blur_passes,
        kawase_passes,
        kawase_offset,
    ) else {
        return false;
    };
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return false;
        };
        state.fuzzy_enabled = enabled;
        state.focus_postprocess = enabled.then_some(packet);
        if let Some(fuzzy) = state.fuzzy.as_mut() {
            let mut config = fuzzy.config().clone();
            config.max_distance = f32::from(packet.blur_radius_pixels);
            config.blur_strength = packet.blur_strength;
            config.focus = packet.focus_coordinate;
            config.bandwidth = packet.bandwidth;
            config.normalize = packet.normalize_range;
            config.cpu_stretch_min = packet.stretch_range[0];
            config.cpu_stretch_max = packet.stretch_range[1];
            config.blur_passes = u32::from(packet.gaussian_passes);
            config.kawase_passes = u32::from(packet.kawase_passes);
            config.kawase_offset = packet.kawase_offset;
            config.blur_mode = u32::from(packet.mode.wire_index());
            fuzzy.set_config(config);
        }
        true
    })
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
            for (batch, render_batch) in state.batches.values().zip(&state.render_batches) {
                state.renderer.classify_patch_batch(
                    &camera,
                    render_batch,
                    batch.visibility_vao,
                    batch.instances.visibility_buf,
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
                    false,
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
fn resolve_surface_pick(state: &MainState, face: i32) -> Option<ResolvedSurfacePick> {
    let face = u32::try_from(face).ok()?;
    resolve_surface_address(state, face, state.last_pick_barycentric?)
}

fn resolve_surface_address(
    state: &MainState,
    face: u32,
    barycentric: [f32; 3],
) -> Option<ResolvedSurfacePick> {
    let face_index = usize::try_from(face).ok()?;
    if barycentric
        .into_iter()
        .any(|coordinate| !coordinate.is_finite() || coordinate < -1.0e-4)
        || (barycentric.into_iter().sum::<f32>() - 1.0).abs() > 1.0e-3
    {
        return None;
    }
    let node_index = *state.face_nodes.get(face_index)?;
    let node = u32::try_from(node_index).ok()?;
    let barycentric_f64 = barycentric.map(f64::from);
    let conformal = conformal_state_for_node(state, node_index);
    let source_position = state
        .surface_runtime
        .sample_face_in_parent_chart(
            &state.cached_instances,
            face_index,
            barycentric_f64,
            conformal.euclidean_model,
        )
        .ok()
        .map(|sample| sample.output_position);
    let output_position = state
        .surface_runtime
        .output_patch_for_face(
            &state.cached_instances,
            face_index,
            conformal.mobius,
            conformal.euclidean_model,
        )
        .ok()
        .map(|patch| patch.eval(barycentric_f64[1], barycentric_f64[2]).to_point())
        .filter(|point| point.iter().all(|coordinate| coordinate.is_finite()));
    Some(ResolvedSurfacePick {
        face,
        node,
        barycentric,
        source_position,
        output_position,
    })
}

#[cfg(feature = "webgpu-backend")]
pub(crate) fn resolve_backend_pick_surface(
    hit: crate::webgpu_backend::WebGpuPickHit,
) -> Result<ResolvedSurfacePick, String> {
    STATE.with(|state| {
        let state = state.borrow();
        let state = state
            .as_ref()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        let mut surface = resolve_surface_address(
            state,
            hit.source_face,
            hit.source_barycentric,
        )
        .ok_or_else(|| "backend pick source address is no longer resident".to_string())?;
        if surface.node != hit.packed_node {
            return Err(format!(
                "backend pick node {} no longer owns source face {} (current node {})",
                hit.packed_node, hit.source_face, surface.node,
            ));
        }
        // Preserve the exact source-chart point from the rendered device
        // frame. The output point is deliberately re-evaluated from its stable
        // face/barycentric address against the current animated pose.
        surface.source_position = Some(hit.source_position.map(f64::from));
        Ok(surface)
    })
}

fn surface_pick_to_js(surface: ResolvedSurfacePick) -> JsValue {
    let result = js_sys::Object::new();
    let barycentric = js_sys::Float32Array::new_with_length(3);
    barycentric.copy_from(&surface.barycentric);
    js_sys::Reflect::set(
        &result,
        &"face".into(),
        &JsValue::from_f64(surface.face as f64),
    )
    .ok();
    js_sys::Reflect::set(
        &result,
        &"node".into(),
        &JsValue::from_f64(surface.node as f64),
    )
    .ok();
    js_sys::Reflect::set(&result, &"barycentric".into(), &barycentric).ok();
    for (name, position) in [
        ("source_position", surface.source_position),
        ("output_position", surface.output_position),
    ] {
        if let Some(position) = position {
            let array = js_sys::Float64Array::new_with_length(3);
            array.copy_from(&position);
            js_sys::Reflect::set(&result, &name.into(), &array).ok();
        }
    }
    result.into()
}

#[cfg(feature = "webgpu-backend")]
fn render_pick_hit(
    surface: ResolvedSurfacePick,
    camera_position: [f32; 3],
) -> Option<RenderPickHit> {
    let source_position = surface.source_position?.map(|coordinate| coordinate as f32);
    let output_position = surface.output_position?;
    let squared_distance = output_position
        .into_iter()
        .zip(camera_position)
        .map(|(output, camera)| (output - f64::from(camera)).powi(2))
        .sum::<f64>();
    RenderPickHit::new(
        surface.node,
        surface.face,
        surface.barycentric,
        source_position,
        squared_distance.sqrt() as f32,
    )
    .ok()
}

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
        resolve_surface_pick(state, face)
            .map(surface_pick_to_js)
            .unwrap_or(JsValue::NULL)
    })
}

/// Stage an opt-in backend comparison while returning the incumbent WebGL2
/// surface synchronously. The staged WebGPU query observes retained frame
/// state and never becomes interaction authority.
#[cfg(feature = "webgpu-backend")]
pub(crate) fn stage_backend_pick_evidence(
    mvp: &[f32],
    mv: &[f32],
    camera_pos: &[f32],
    x: i32,
    y: i32,
    target_epoch: u32,
) -> BackendPickEvidenceStageReceipt {
    let started_ms = browser_now_ms();
    // The incumbent picker historically highlights every queried face. Pick
    // evidence is observational and must not turn that debugging side effect
    // into application state (or make an otherwise supported WebGPU focus
    // frame fall back because source-face highlight composition is absent).
    let prior_highlight = STATE.with(|state| {
        state
            .borrow()
            .as_ref()
            .map_or(-1, |state| state.highlight_face)
    });
    let face = mr_pick(mvp, mv, camera_pos, x, y);
    STATE.with(|state| {
        if let Some(state) = state.borrow_mut().as_mut() {
            state.highlight_face = prior_highlight;
        }
    });
    let surface = STATE.with(|state| {
        let state = state.borrow();
        state
            .as_ref()
            .and_then(|state| resolve_surface_pick(state, face))
    });

    let staged = STATE.with(|slot| -> Result<BackendPickEvidenceStageMetadata, String> {
        let mut state = slot.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        if state.backend_pick_evidence.is_some() || state.backend_pick_evidence_reading {
            return Err("a backend pick comparison is already in flight".to_string());
        }
        let pixel = [
            u32::try_from(x).map_err(|_| "pick x is outside the viewport".to_string())?,
            u32::try_from(y).map_err(|_| "pick y is outside the viewport".to_string())?,
        ];
        let viewport = [
            u32::try_from(state.viewport_size.0)
                .map_err(|_| "pick viewport width is invalid".to_string())?,
            u32::try_from(state.viewport_size.1)
                .map_err(|_| "pick viewport height is invalid".to_string())?,
        ];
        if pixel[0] >= viewport[0] || pixel[1] >= viewport[1] {
            return Err("pick pixel is outside the viewport".to_string());
        }
        let camera_position = [
            camera_pos.first().copied().unwrap_or(0.0),
            camera_pos.get(1).copied().unwrap_or(0.0),
            camera_pos.get(2).copied().unwrap_or(0.0),
        ];
        let expected = match surface {
            Some(surface) => Some(render_pick_hit(surface, camera_position).ok_or_else(|| {
                "incumbent pick could not be expressed as render evidence".to_string()
            })?),
            None => None,
        };
        let webgl_render_call = state.render_calls;
        let stage_started_ms = browser_now_ms();
        let webgpu = crate::webgpu_backend::stage_pick(pixel, target_epoch)?;
        let staging_ms = (browser_now_ms() - stage_started_ms).max(0.0);
        if webgpu.source_render_call() != webgl_render_call {
            return Err(format!(
                "backend pick frame mismatch: WebGL2 render call {webgl_render_call}, WebGPU source render call {}",
                webgpu.source_render_call(),
            ));
        }
        let webgpu_frame_revision = webgpu.frame_revision();
        state.backend_pick_evidence = Some(BackendPickEvidenceCapture {
            webgl_render_call,
            webgpu_frame_revision,
            viewport,
            pixel,
            target_epoch,
            expected,
            staged: webgpu,
            staging_ms,
            started_ms,
        });
        Ok(BackendPickEvidenceStageMetadata {
            webgpu_frame_revision,
            webgl_render_call,
            staging_ms,
        })
    });

    BackendPickEvidenceStageReceipt {
        webgl: surface,
        staged,
    }
}

/// Stage one authoritative query only when the retained WebGPU image is the
/// current logical renderer call and viewport. Unlike the parity path above,
/// this does not execute or mutate the incumbent WebGL picker.
#[cfg(feature = "webgpu-backend")]
pub(crate) fn stage_backend_pick_authority(
    pixel: [u32; 2],
    target_epoch: u32,
) -> Result<BackendPickAuthorityCapture, String> {
    let (source_render_call, viewport) = STATE.with(|slot| {
        let state = slot.borrow();
        let state = state
            .as_ref()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        let viewport = [
            u32::try_from(state.viewport_size.0)
                .map_err(|_| "pick viewport width is invalid".to_string())?,
            u32::try_from(state.viewport_size.1)
                .map_err(|_| "pick viewport height is invalid".to_string())?,
        ];
        Ok::<_, String>((state.render_calls, viewport))
    })?;
    let staged = crate::webgpu_backend::stage_pick(pixel, target_epoch)?;
    if staged.source_render_call() != source_render_call {
        return Err(format!(
            "authoritative backend pick requires renderer call {source_render_call}, but WebGPU retains {}",
            staged.source_render_call(),
        ));
    }
    if staged.viewport() != viewport {
        return Err(format!(
            "authoritative backend pick requires viewport {}x{}, but WebGPU retains {}x{}",
            viewport[0], viewport[1], staged.viewport()[0], staged.viewport()[1],
        ));
    }
    Ok(BackendPickAuthorityCapture {
        source_render_call,
        webgpu_frame_revision: staged.frame_revision(),
        viewport,
        staged,
    })
}

#[cfg(feature = "webgpu-backend")]
#[wasm_bindgen(js_name = "mr_stageBackendPickEvidence")]
pub fn mr_stage_backend_pick_evidence(
    mvp: &[f32],
    mv: &[f32],
    camera_pos: &[f32],
    x: i32,
    y: i32,
    target_epoch: u32,
) -> JsValue {
    stage_backend_pick_evidence(mvp, mv, camera_pos, x, y, target_epoch).into_js()
}

/// Read the single bounded backend pick comparison staged by
/// `mr_stageBackendPickEvidence`.
#[cfg(feature = "webgpu-backend")]
pub(crate) async fn read_backend_pick_evidence() -> Result<RenderPickEvidenceReport, String> {
    let capture = STATE.with(|slot| -> Result<BackendPickEvidenceCapture, String> {
        let mut state = slot.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| "renderer is not initialized".to_string())?;
        if state.backend_pick_evidence_reading {
            return Err("a backend pick comparison readback is already in flight".to_string());
        }
        let capture = state
            .backend_pick_evidence
            .take()
            .ok_or_else(|| "no backend pick comparison is staged".to_string())?;
        state.backend_pick_evidence_reading = true;
        Ok(capture)
    })?;

    let BackendPickEvidenceCapture {
        webgl_render_call,
        webgpu_frame_revision,
        viewport,
        pixel,
        target_epoch,
        expected,
        staged,
        staging_ms,
        started_ms,
    } = capture;
    let readback_started_ms = browser_now_ms();
    let report = async {
        let readback = staged.read().await?;
        if readback.frame_revision != webgpu_frame_revision {
            return Err(format!(
                "backend pick revision changed: staged {webgpu_frame_revision}, read {}",
                readback.frame_revision,
            ));
        }
        if readback.source_render_call != webgl_render_call {
            return Err(format!(
                "backend pick source changed: WebGL2 render call {webgl_render_call}, WebGPU read {}",
                readback.source_render_call,
            ));
        }
        let actual = match readback.hit {
            Some(hit) => {
                if hit.target_epoch != target_epoch {
                    return Err(format!(
                        "backend pick target epoch changed: staged {target_epoch}, read {}",
                        hit.target_epoch,
                    ));
                }
                Some(
                    RenderPickHit::new(
                        hit.packed_node,
                        hit.source_face,
                        hit.source_barycentric,
                        hit.source_position,
                        hit.output_distance,
                    )
                    .map_err(|error| error.to_string())?,
                )
            }
            None => None,
        };
        let comparison = RenderPickComparison::between(expected, actual)
            .map_err(|error| error.to_string())?;
        let completed_ms = browser_now_ms();
        Ok(RenderPickEvidenceReport {
            webgl_render_call,
            webgpu_frame_revision,
            viewport,
            pixel,
            target_epoch,
            comparison,
            staging_ms,
            readback_ms: (completed_ms - readback_started_ms).max(0.0),
            total_ms: (completed_ms - started_ms).max(0.0),
        })
    }
    .await;

    STATE.with(|slot| {
        if let Some(state) = slot.borrow_mut().as_mut() {
            state.backend_pick_evidence_reading = false;
        }
    });
    report
}

#[cfg(feature = "webgpu-backend")]
#[wasm_bindgen(js_name = "mr_readBackendPickEvidence")]
pub async fn mr_read_backend_pick_evidence() -> Result<JsValue, JsValue> {
    let report = read_backend_pick_evidence()
        .await
        .map_err(|error| JsValue::from_str(&error))?;
    serde_wasm_bindgen::to_value(&report).map_err(|error| JsValue::from_str(&error.to_string()))
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

fn surface_camera_anchor_camera(
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    parameters: &[f64],
    semantic_target: &[f64],
) -> Result<CameraRig, String> {
    let mut camera = surface_walk_camera(eye, forward, up, parameters)?;
    camera.semantic_target = match semantic_target {
        [] => None,
        values => Some(exact_vector3(values, "surface-anchor semantic target")?),
    };
    camera.validate().map_err(|error| error.to_string())?;
    Ok(camera)
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

fn surface_camera_anchor_snapshot_to_js(
    snapshot: Result<SurfaceCameraAnchorSnapshot, String>,
) -> JsValue {
    match snapshot {
        Ok(snapshot) => serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL),
        Err(error) => {
            warn!("Surface camera anchor: {error}");
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"status".into(), &"error".into()).ok();
            js_sys::Reflect::set(&result, &"error".into(), &error.into()).ok();
            result.into()
        }
    }
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

/// Attach an ordinary orbit/fly/drone camera to one animated material point.
/// Unlike walking, this retains the source face and barycentric coordinate.
#[wasm_bindgen(js_name = "mr_attachSurfaceCameraAnchor")]
#[allow(clippy::too_many_arguments)]
pub fn mr_attach_surface_camera_anchor(
    face: u32,
    barycentric: &[f64],
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    camera_parameters: &[f64],
    semantic_target: &[f64],
) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let Some(&node) = state.face_nodes.get(face as usize) else {
            return surface_camera_anchor_snapshot_to_js(Err(
                "surface face has no node".into(),
            ));
        };
        let request = (|| {
            let barycentric = exact_vector3(barycentric, "surface barycentric")?;
            let camera = surface_camera_anchor_camera(
                eye,
                forward,
                up,
                camera_parameters,
                semantic_target,
            )?;
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
            surface_runtime.attach_camera_anchor(
                cached_instances,
                *num_faces,
                face_nodes,
                face,
                barycentric,
                camera,
                orientation_sign,
                conformal.mobius,
                conformal.euclidean_model,
            )
        })();
        surface_camera_anchor_snapshot_to_js(request)
    })
}

/// Re-evaluate a fixed camera anchor against the latest accepted animation
/// pose. `capture_relative_camera` records a navigation edit before following.
#[wasm_bindgen(js_name = "mr_stepSurfaceCameraAnchor")]
#[allow(clippy::too_many_arguments)]
pub fn mr_step_surface_camera_anchor(
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    camera_parameters: &[f64],
    semantic_target: &[f64],
    capture_relative_camera: bool,
) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let request = (|| {
            let face = state
                .surface_runtime
                .camera_anchor_face()
                .ok_or_else(|| "surface camera anchor is detached".to_string())?;
            let node = *state
                .face_nodes
                .get(face as usize)
                .ok_or_else(|| "attached face has no node".to_string())?;
            let camera = surface_camera_anchor_camera(
                eye,
                forward,
                up,
                camera_parameters,
                semantic_target,
            )?;
            let conformal = conformal_state_for_node(state, node);
            let orientation_sign = conformal.orientation_sign
                * affine_orientation_sign(&conformal.euclidean_model);
            let MainState {
                surface_runtime,
                cached_instances,
                face_nodes,
                ..
            } = state;
            surface_runtime.step_camera_anchor(
                cached_instances,
                face_nodes,
                camera,
                capture_relative_camera,
                orientation_sign,
                conformal.mobius,
                conformal.euclidean_model,
            )
        })();
        surface_camera_anchor_snapshot_to_js(request)
    })
}

/// Rebase fixed-anchor caches after the browser has transported the live
/// camera exactly through a spherical-reflection chart change.
#[wasm_bindgen(js_name = "mr_transportSurfaceCameraAnchorReflection")]
#[allow(clippy::too_many_arguments)]
pub fn mr_transport_surface_camera_anchor_reflection(
    previous_enabled: bool,
    previous_center: &[f64],
    previous_radius: f64,
    next_enabled: bool,
    next_center: &[f64],
    next_radius: f64,
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    camera_parameters: &[f64],
    semantic_target: &[f64],
) -> Result<bool, JsValue> {
    let previous =
        surface_walk_reflection_state(previous_enabled, previous_center, previous_radius)
            .map_err(|error| JsValue::from_str(&error))?;
    let next = surface_walk_reflection_state(next_enabled, next_center, next_radius)
        .map_err(|error| JsValue::from_str(&error))?;
    let camera = surface_camera_anchor_camera(
        eye,
        forward,
        up,
        camera_parameters,
        semantic_target,
    )
    .map_err(|error| JsValue::from_str(&error))?;
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return Ok(false);
        };
        state
            .surface_runtime
            .transport_camera_anchor_between_reflections(previous, next, camera)
            .map_err(|error| JsValue::from_str(&error))
    })
}

#[wasm_bindgen(js_name = "mr_detachSurfaceCameraAnchor")]
pub fn mr_detach_surface_camera_anchor() -> bool {
    STATE.with(|state| {
        state
            .borrow_mut()
            .as_mut()
            .is_some_and(|state| state.surface_runtime.detach_camera_anchor())
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

fn build_screen_topology_cache(
    topology: Option<&quilting_mesh::HalfEdgeMesh>,
) -> Result<ScreenMeshTopologyCache, String> {
    ScreenMeshTopologyCache::from_half_edge_mesh(
        topology.ok_or_else(|| "source LOD topology is unavailable".to_string())?,
    )
    .map_err(|error| format!("could not cache adaptive source topology: {error}"))
}

fn bounded_standby_resident_lod() -> batch::ResidentLod {
    let lod = TESS_CACHE.with(|cache| {
        let cache = cache.borrow();
        cache
            .keys()
            .filter(|lod| lod[0] >= 2)
            .min_by_key(|lod| {
                (
                    lod[0].saturating_mul(lod[1]).saturating_mul(lod[2]),
                    **lod,
                )
            })
            .or_else(|| {
                cache.keys().min_by_key(|lod| {
                    (
                        lod[0].saturating_mul(lod[1]).saturating_mul(lod[2]),
                        **lod,
                    )
                })
            })
            .copied()
    })
    .unwrap_or([1; 3]);
    batch::ResidentLod {
        canonical: lod,
        perm_index: 0,
        parity_bucket: 0,
    }
}

fn compare_adaptive_root_shadow(
    state: &mut MainState,
    initial: batch::ResidentLod,
    source: LegacyBatchDestination,
) {
    let MainState {
        adaptive_root_shadow,
        screen_topology_cache,
        resident_face_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        batch_groups,
        baseline_batch_groups,
        ..
    } = state;
    let Some(topology) = screen_topology_cache.as_ref() else {
        return;
    };
    let groups = match source {
        LegacyBatchDestination::Live => batch_groups,
        LegacyBatchDestination::Baseline => baseline_batch_groups,
    };
    adaptive_root_shadow.compare(
        topology,
        resident_face_lods,
        initial,
        face_materials,
        face_nodes,
        face_render_nodes,
        groups,
    );
}

/// Replace the freshly rebuilt legacy groups only after one complete adaptive
/// candidate has passed every CPU-side invariant and atlas/work budget. Any
/// unavailable input or rejected plan is recorded and leaves those legacy
/// groups untouched.
fn apply_adaptive_screen_plan(
    state: &mut MainState,
    initial: batch::ResidentLod,
    published_groups_are_live: bool,
    reusable_groups: Option<
        &mut BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    >,
) -> bool {
    let Some(config) = state.adaptive_picked.config() else {
        return false;
    };
    let candidate_inputs = (|| -> Result<_, String> {
        if state.screen_topology_cache.is_none() {
            return Err("adaptive source topology is unavailable".to_string());
        }
        let view_projection = state
            .last_visibility_mvp
            .ok_or_else(|| "no rendered camera is available for adaptive planning".to_string())?
            .map(f64::from);
        let viewport = [
            f64::from(state.viewport_size.0.max(1)),
            f64::from(state.viewport_size.1.max(1)),
        ];
        let (max_atlas_lod, atlas_triangle_counts) = TESS_CACHE.with(|cache| -> Result<_, String> {
            let cache = cache.borrow();
            let maximum = cache
                .keys()
                .flat_map(|key| key.iter().copied())
                .max()
                .ok_or_else(|| "tessellation atlas is unavailable".to_string())?;
            let counts = cache
                .iter()
                .map(|(key, patch)| (*key, patch.num_tri_indices.max(0) as u64 / 3))
                .collect::<BTreeMap<_, _>>();
            Ok((maximum, counts))
        })?;
        let (selected_faces, selection_diagnostic, max_partition_leaves) =
            match config.selection {
                AdaptiveScreenSelection::Picked { face } => {
                    if face as usize >= state.num_faces {
                        return Err(format!(
                            "adaptive face {face} is outside the {}-face scene",
                            state.num_faces,
                        ));
                    }
                    (vec![face], None, config.policy.max_leaves)
                }
                AdaptiveScreenSelection::CurrentView {
                    max_faces,
                    max_partition_leaves,
                } => {
                    if state.classified_face_visibility.len() != state.num_faces
                        || state.requested_face_lods.len() != state.num_faces
                        || state.resident_face_lods.len() != state.num_faces
                        || state.requested_face_lods.iter().any(Option::is_none)
                        || state.resident_face_lods.iter().any(Option::is_none)
                    {
                        return Err(
                            "current-view adaptive selection lacks a complete retained classification"
                                .to_string(),
                        );
                    }
                    if state.num_faces > u32::MAX as usize {
                        return Err(
                            "current-view adaptive source identity exceeds u32".to_string(),
                        );
                    }
                    let candidates = (0..state.num_faces).map(|face_index| {
                        let source_face = face_index as u32;
                        // Keep the raw request as the planner's desired edge
                        // topology, but price its root with the reconciled
                        // resident key that actually exists in the bounded
                        // atlas. Raw requests can exceed the atlas grading
                        // ratio (for example 1/1/64) before shared-edge
                        // promotion makes them resident.
                        let requested = state.requested_face_lods[face_index]
                            .expect("complete current-view request classification")
                            .edge_lods();
                        let resident = state.resident_face_lods[face_index]
                            .expect("complete current-view resident classification");
                        let visible = state.classified_face_visibility[face_index];
                        let root_triangles = if visible {
                            atlas_triangle_counts
                                .get(&resident.canonical)
                                .copied()
                                .unwrap_or(0)
                        } else {
                            0
                        };
                        AdaptiveScreenFaceCandidate {
                            source_face,
                            visible,
                            screen_metric_priority: state
                                .same_context_lod
                                .as_ref()
                                .and_then(|lod| {
                                    lod.adaptive_face_priorities.get(face_index).copied()
                                })
                                .unwrap_or(0),
                            requested_lods: requested,
                            root_triangles,
                        }
                    });
                    let selection = select_adaptive_screen_faces(
                        candidates,
                        AdaptiveScreenFaceSelectionPolicy {
                            max_faces,
                            partition_policy: config.policy,
                            max_partition_leaves,
                        },
                    )
                    .map_err(|error| error.to_string())?;
                    (
                        selection.faces,
                        Some(selection.diagnostic),
                        max_partition_leaves,
                    )
                }
            };
        let mut patches = Vec::with_capacity(selected_faces.len());
        for &face in &selected_faces {
            let face_index = face as usize;
            let node = state.face_nodes.get(face_index).copied().unwrap_or(0);
            let conformal = conformal_state_for_node(state, node);
            patches.push(state.surface_runtime.output_patch_for_face(
                &state.cached_instances,
                face_index,
                conformal.mobius,
                conformal.euclidean_model,
            )?);
        }
        Ok((
            selected_faces,
            patches,
            selection_diagnostic,
            max_partition_leaves,
            view_projection,
            viewport,
            state.surface_runtime.pose_stamp(),
            max_atlas_lod,
            atlas_triangle_counts,
        ))
    })();

    let (
        selected_faces,
        patches,
        selection_diagnostic,
        max_partition_leaves,
        view_projection,
        viewport,
        pose_stamp,
        max_atlas_lod,
        atlas_triangle_counts,
    ) = match candidate_inputs {
        Ok(inputs) => inputs,
        Err(error) => {
            state.adaptive_picked.record_fallback(error);
            return false;
        }
    };
    let selected_patches = selected_faces
        .iter()
        .copied()
        .zip(patches.iter())
        .map(|(source_face, transformed_patch)| SelectedScreenPatch {
            source_face,
            transformed_patch,
        })
        .collect::<Vec<_>>();
    let max_face_edge_ratio = state.lod_grading.ratio();
    let batch_layout_revision = state.batch_layout_revision;
    let root_topology_revision = state.root_topology_revision;
    let local_hot_path_allowed = (state
        .adaptive_picked
        .component_publication_enabled()
        || state
            .adaptive_picked
            .neighborhood_publication_enabled())
        && state.adaptive_retained_publication_enabled
        && retained_frontier_order_safe(state);
    let MainState {
        adaptive_picked,
        screen_topology_cache,
        requested_face_lods,
        resident_face_lods,
        resident_vertex_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        batch_groups,
        baseline_batch_groups,
        ..
    } = state;
    let Some(topology) = screen_topology_cache.as_ref() else {
        adaptive_picked.record_fallback("adaptive source topology disappeared");
        return false;
    };
    let groups_reused = match adaptive_picked
        .plan_selected_and_group(
            &selected_patches,
            selection_diagnostic,
            max_partition_leaves,
            &view_projection,
            viewport,
            pose_stamp,
            requested_face_lods,
            resident_face_lods,
            resident_vertex_lods,
            initial,
            topology,
            max_atlas_lod,
            max_face_edge_ratio,
            &atlas_triangle_counts,
            face_materials,
            face_nodes,
            face_render_nodes,
            batch_layout_revision,
            root_topology_revision,
            local_hot_path_allowed,
            published_groups_are_live,
            reusable_groups,
            batch_groups,
        )
    {
        Ok(groups_reused) => groups_reused,
        Err(_) => return false,
    };
    let overlay_reused = match adaptive_picked.stage_pending_overlay(
        batch_layout_revision,
        baseline_batch_groups,
        resident_face_lods,
        resident_vertex_lods,
        face_materials,
        face_nodes,
        face_render_nodes,
        initial,
        &atlas_triangle_counts,
    ) {
        Ok(reused) => reused,
        Err(error) => {
            adaptive_picked.reject_staged_overlay(groups_reused, batch_groups, error);
            return false;
        }
    };
    groups_reused && overlay_reused
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
            state.render_scene_dirty = true;
            sync_render_batches(state);
            let _ = refresh_validated_render_scene(state, false);
            refresh_render_command_plan(state, false);
        } else if !enabled && changed {
            state.validated_render_scene = None;
            state.render_command_plan = None;
            state.webgl_pbr_command_plan = None;
        }
        state.render_shadow.to_js()
    })
}

/// Enable the non-mutating incremental retained-root equivalence gate. The
/// incumbent complete rebuild remains live authority while the sparse closure
/// is compared byte-for-byte and timed below the WASM boundary.
#[wasm_bindgen(js_name = "mr_setIncrementalRootGroupShadowEnabled")]
pub fn mr_set_incremental_root_group_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        state.incremental_root_group_shadow.set_enabled(enabled);
        if enabled
            && state.num_faces > 0
            && !state.batch_layout_dirty
            && !state.batch_groups.is_empty()
        {
            compare_incremental_root_group_shadow(
                state,
                bounded_standby_resident_lod(),
                true,
            );
        }
        serde_wasm_bindgen::to_value(&state.incremental_root_group_shadow.snapshot())
            .unwrap_or(JsValue::NULL)
    })
}

#[wasm_bindgen(js_name = "mr_incrementalRootGroupShadowDiagnostics")]
pub fn mr_incremental_root_group_shadow_diagnostics() -> JsValue {
    STATE.with(|state| {
        state.borrow().as_ref().map_or(JsValue::NULL, |state| {
            serde_wasm_bindgen::to_value(&state.incremental_root_group_shadow.snapshot())
                .unwrap_or(JsValue::NULL)
        })
    })
}

/// Enable the non-mutating all-root adaptive batch equivalence gate.
///
/// This is deliberately independent of the render shadow: it compares the
/// legacy face grouping against the exact adaptive-frontier handoff that a
/// future picked/full-scene mode will use, while leaving current GL batches
/// untouched.
#[wasm_bindgen(js_name = "mr_setAdaptiveRootShadowEnabled")]
pub fn mr_set_adaptive_root_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        state.adaptive_root_shadow.set_enabled(enabled);
        if enabled && state.num_faces > 0 && state.screen_topology_cache.is_none() {
            match build_screen_topology_cache(state.lod_topology.as_ref()) {
                Ok(cache) => state.screen_topology_cache = Some(cache),
                Err(error) => state.adaptive_root_shadow.record_unavailable(error),
            }
        } else if !enabled {
            if !state.adaptive_picked.is_enabled() {
                state.screen_topology_cache = None;
            }
            state.adaptive_root_shadow.reset_topology();
        }
        if enabled
            && state.num_faces > 0
            && !state.batch_layout_dirty
            && !state.batch_groups.is_empty()
        {
            let initial = bounded_standby_resident_lod();
            ensure_baseline_batch_groups(state, initial);
            compare_adaptive_root_shadow(
                state,
                initial,
                LegacyBatchDestination::Baseline,
            );
        }
        serde_wasm_bindgen::to_value(&state.adaptive_root_shadow.snapshot())
            .unwrap_or(JsValue::NULL)
    })
}

#[wasm_bindgen(js_name = "mr_adaptiveRootShadowDiagnostics")]
pub fn mr_adaptive_root_shadow_diagnostics() -> JsValue {
    STATE.with(|state| {
        state.borrow().as_ref().map_or(JsValue::NULL, |state| {
            serde_wasm_bindgen::to_value(&state.adaptive_root_shadow.snapshot())
                .unwrap_or(JsValue::NULL)
        })
    })
}

/// Stage the exact retained-root plus adaptive-overlay composition beside the
/// complete live frontier. This is an observation/cutover gate only: enabling
/// it does not change GPU batches or draw submission.
#[wasm_bindgen(js_name = "mr_setAdaptiveRetainedShadowEnabled")]
pub fn mr_set_adaptive_retained_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        match state
            .adaptive_picked
            .set_retained_shadow_enabled(enabled)
        {
            Ok(changed) => {
                if changed && enabled && state.adaptive_picked.is_enabled() {
                    state.adaptive_batch_transition_pending = true;
                }
                adaptive_picked_snapshot_js(state, None)
            }
            Err(error) => adaptive_screen_diagnostic_error(error),
        }
    })
}

/// Compare an exact source-component plan and sparse overlay beside every
/// complete adaptive publication. Enabling this observation gate also enables
/// the retained-overlay shadow it compares against; neither changes drawing.
#[wasm_bindgen(js_name = "mr_setAdaptiveComponentShadowEnabled")]
pub fn mr_set_adaptive_component_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        match state
            .adaptive_picked
            .set_component_shadow_enabled(enabled)
        {
            Ok(changed) => {
                if changed && enabled && state.adaptive_picked.is_enabled() {
                    state.adaptive_batch_transition_pending = true;
                }
                adaptive_picked_snapshot_js(state, None)
            }
            Err(error) => adaptive_screen_diagnostic_error(error),
        }
    })
}

/// Compare a statically bounded edge neighborhood plus fixed observer roots
/// against the complete adaptive oracle. This read-only mode always keeps the
/// complete plan in the same epoch and never changes GPU publication.
#[wasm_bindgen(js_name = "mr_setAdaptiveNeighborhoodShadowEnabled")]
pub fn mr_set_adaptive_neighborhood_shadow_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        match state
            .adaptive_picked
            .set_neighborhood_shadow_enabled(enabled)
        {
            Ok(changed) => {
                if changed && enabled && state.adaptive_picked.is_enabled() {
                    state.adaptive_batch_transition_pending = true;
                }
                adaptive_picked_snapshot_js(state, None)
            }
            Err(error) => adaptive_screen_diagnostic_error(error),
        }
    })
}

/// Opt into neighborhood-derived retained-layer publication only after the
/// same-epoch complete plan, overlay, atlas residency, and triangle budget all
/// match. A mismatch keeps the complete overlay staged; a GPU failure keeps
/// the prior GPU epoch. This does not yet let a neighborhood skip its oracle.
#[wasm_bindgen(js_name = "mr_setAdaptiveNeighborhoodPublicationEnabled")]
pub fn mr_set_adaptive_neighborhood_publication_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let changed = match state
            .adaptive_picked
            .set_neighborhood_publication_enabled(enabled)
        {
            Ok(changed) => changed,
            Err(error) => return adaptive_screen_diagnostic_error(error),
        };
        let retained_changed = enabled && !state.adaptive_retained_publication_enabled;
        if enabled {
            state.adaptive_retained_publication_enabled = true;
        }
        if changed || retained_changed {
            state.adaptive_retained_publication_error = None;
            if state.adaptive_picked.is_enabled() {
                state.adaptive_batch_transition_pending = true;
            }
        }
        adaptive_picked_snapshot_js(state, None)
    })
}

/// Opt into component-derived retained-layer publication behind the exact
/// complete-plan oracle. A mismatch keeps the complete overlay staged; a GPU
/// failure keeps the prior GPU epoch. Enabling this also opts into retained
/// physical publication, subject to the existing opaque-ordering gate.
#[wasm_bindgen(js_name = "mr_setAdaptiveComponentPublicationEnabled")]
pub fn mr_set_adaptive_component_publication_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let changed = match state
            .adaptive_picked
            .set_component_publication_enabled(enabled)
        {
            Ok(changed) => changed,
            Err(error) => return adaptive_screen_diagnostic_error(error),
        };
        let retained_changed = enabled && !state.adaptive_retained_publication_enabled;
        if enabled {
            state.adaptive_retained_publication_enabled = true;
        }
        if changed || retained_changed {
            state.adaptive_retained_publication_error = None;
            if state.adaptive_picked.is_enabled() {
                state.adaptive_batch_transition_pending = true;
            }
        }
        adaptive_picked_snapshot_js(state, None)
    })
}

/// Opt into transactional retained-layer GPU publication. The persistent
/// overlay shadow is forced on first; scenes containing order-sensitive PBR
/// buckets remain on the complete path and report a diagnostic instead.
#[wasm_bindgen(js_name = "mr_setAdaptiveRetainedPublicationEnabled")]
pub fn mr_set_adaptive_retained_publication_enabled(enabled: bool) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        if state.adaptive_picked.has_pending_publication() {
            return adaptive_screen_diagnostic_error("adaptive publication is already staged");
        }
        if !enabled
            && (state.adaptive_picked.component_publication_enabled()
                || state.adaptive_picked.neighborhood_publication_enabled())
        {
            return adaptive_screen_diagnostic_error(
                "adaptive component or neighborhood publication requires retained GPU publication",
            );
        }
        if enabled {
            if let Err(error) = state.adaptive_picked.set_retained_shadow_enabled(true) {
                return adaptive_screen_diagnostic_error(error);
            }
        }
        let changed = state.adaptive_retained_publication_enabled != enabled;
        state.adaptive_retained_publication_enabled = enabled;
        if changed {
            state.adaptive_retained_publication_error = None;
            if state.adaptive_picked.is_enabled() {
                state.adaptive_batch_transition_pending = true;
            }
        }
        adaptive_picked_snapshot_js(state, None)
    })
}

/// Configure the first live metric-adaptive path for one stable source face.
///
/// The browser must request a normal LOD recomputation after configuration.
/// Until that classification arrives the current root batches remain live.
/// Every candidate is bounded by selected-leaf, scene-leaf, atlas, and
/// triangle budgets; changed GL buckets are staged before atomic publication.
#[wasm_bindgen(js_name = "mr_setAdaptivePickedFace")]
pub fn mr_set_adaptive_picked_face(
    face: u32,
    min_px_per_segment: f64,
    max_px_per_segment: f64,
    max_depth: u32,
    max_selected_leaves: u32,
    max_total_leaves: u32,
    max_triangles: u32,
) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return adaptive_screen_diagnostic_error("renderer is not initialized");
        };
        if face as usize >= state.num_faces {
            return adaptive_screen_diagnostic_error(format!(
                "face {face} is outside the {}-face scene",
                state.num_faces,
            ));
        }
        if !min_px_per_segment.is_finite()
            || min_px_per_segment <= 0.0
            || !max_px_per_segment.is_finite()
            || max_px_per_segment < min_px_per_segment
        {
            return adaptive_screen_diagnostic_error(
                "adaptive pixel ceiling must be finite and at least the positive pixel floor",
            );
        }
        let Ok(max_depth) = u8::try_from(max_depth) else {
            return adaptive_screen_diagnostic_error("adaptive depth is outside u8 range");
        };
        if max_depth > instance_layout::BATCH_LEAF_MAX_DEPTH {
            return adaptive_screen_diagnostic_error(format!(
                "adaptive depth exceeds the packed leaf limit of {}",
                instance_layout::BATCH_LEAF_MAX_DEPTH,
            ));
        }
        if max_selected_leaves == 0
            || max_total_leaves < state.num_faces as u32
            || max_triangles == 0
        {
            return adaptive_screen_diagnostic_error(
                "adaptive budgets require a positive selected/triangle cap and at least one scene leaf per source face",
            );
        }
        if state.screen_topology_cache.is_none() {
            match build_screen_topology_cache(state.lod_topology.as_ref()) {
                Ok(cache) => state.screen_topology_cache = Some(cache),
                Err(error) => return adaptive_screen_diagnostic_error(error),
            }
        }
        state.adaptive_picked.configure(AdaptivePickedConfig {
            selection: AdaptiveScreenSelection::Picked { face },
            min_px_per_segment,
            max_px_per_segment,
            policy: ScreenPartitionPolicy {
                max_depth,
                max_leaves: max_selected_leaves as usize,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: max_total_leaves as usize,
            max_triangles: u64::from(max_triangles),
        });
        state.adaptive_batch_transition_pending = true;
        adaptive_picked_snapshot_js(state, None)
    })
}

/// Configure bounded automatic adaptation of the most expensive roots in the
/// latest retained visible classification. Selection, posed-patch extraction,
/// partitioning, reconciliation, and publication remain Rust-owned.
#[wasm_bindgen(js_name = "mr_setAdaptiveCurrentView")]
pub fn mr_set_adaptive_current_view(
    min_px_per_segment: f64,
    max_px_per_segment: f64,
    max_depth: u32,
    max_selected_faces: u32,
    max_selected_leaves: u32,
    max_partition_leaves: u32,
    max_total_leaves: u32,
    max_triangles: u32,
) -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return adaptive_screen_diagnostic_error("renderer is not initialized");
        };
        if !min_px_per_segment.is_finite()
            || min_px_per_segment <= 0.0
            || !max_px_per_segment.is_finite()
            || max_px_per_segment < min_px_per_segment
        {
            return adaptive_screen_diagnostic_error(
                "adaptive pixel ceiling must be finite and at least the positive pixel floor",
            );
        }
        let Ok(max_depth) = u8::try_from(max_depth) else {
            return adaptive_screen_diagnostic_error("adaptive depth is outside u8 range");
        };
        if max_depth > instance_layout::BATCH_LEAF_MAX_DEPTH {
            return adaptive_screen_diagnostic_error(format!(
                "adaptive depth exceeds the packed leaf limit of {}",
                instance_layout::BATCH_LEAF_MAX_DEPTH,
            ));
        }
        let Ok(source_faces) = u32::try_from(state.num_faces) else {
            return adaptive_screen_diagnostic_error(
                "current-view adaptive source identity exceeds u32",
            );
        };
        if max_selected_faces == 0
            || max_selected_leaves == 0
            || max_partition_leaves < max_selected_leaves
            || max_total_leaves < source_faces
            || max_triangles == 0
        {
            return adaptive_screen_diagnostic_error(
                "current-view adaptive budgets require positive face/leaf/triangle caps, one per-face partition capacity, and at least one scene leaf per source face",
            );
        }
        if state.screen_topology_cache.is_none() {
            match build_screen_topology_cache(state.lod_topology.as_ref()) {
                Ok(cache) => state.screen_topology_cache = Some(cache),
                Err(error) => return adaptive_screen_diagnostic_error(error),
            }
        }
        state.adaptive_picked.configure(AdaptivePickedConfig {
            selection: AdaptiveScreenSelection::CurrentView {
                max_faces: max_selected_faces as usize,
                max_partition_leaves: max_partition_leaves as usize,
            },
            min_px_per_segment,
            max_px_per_segment,
            policy: ScreenPartitionPolicy {
                max_depth,
                max_leaves: max_selected_leaves as usize,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: max_total_leaves as usize,
            max_triangles: u64::from(max_triangles),
        });
        state.adaptive_batch_transition_pending = true;
        adaptive_picked_snapshot_js(state, None)
    })
}

#[wasm_bindgen(js_name = "mr_clearAdaptivePickedFace")]
pub fn mr_clear_adaptive_picked_face() -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let changed = state.adaptive_picked.is_enabled();
        if changed {
            state.adaptive_picked.stage_clear();
        }
        if !state.adaptive_root_shadow.is_enabled() {
            state.screen_topology_cache = None;
        }
        state.adaptive_batch_transition_pending |= changed;
        adaptive_picked_snapshot_js(state, None)
    })
}

#[wasm_bindgen(js_name = "mr_adaptivePickedDiagnostics")]
pub fn mr_adaptive_picked_diagnostics() -> JsValue {
    STATE.with(|state| {
        state.borrow().as_ref().map_or(JsValue::NULL, |state| {
            adaptive_picked_snapshot_js(state, None)
        })
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
            let mut next_face_nodes = st
                .pending_face_nodes
                .clone()
                .unwrap_or_else(|| st.face_nodes.clone());
            if next_face_nodes.len() != num_faces {
                next_face_nodes = vec![0; num_faces];
            }
            let mut next_instances = instances.to_vec();
            if let Err(error) = embed_face_node_ids(
                &mut next_instances,
                num_faces,
                &next_face_nodes,
            ) {
                warn!("Could not embed source-face node identity: {error}");
                return;
            }
            if let Err(error) = st.renderer.upload_face_data_texture(&next_instances, num_faces) {
                warn!("Could not upload immutable source-face data: {error}");
                return;
            }
            clear_same_context_lod(st);
            st.cached_instances = next_instances;
            st.num_faces = num_faces;
            st.face_nodes = next_face_nodes;
            st.pending_face_nodes = None;
            st.patch_prepare_dirty = true;
            st.render_shadow.asset_changed();
            st.render_scene_dirty = true;
            st.presentation_nodes.clear();
            st.authored_node_models.clear();
            st.surface_runtime.reset_geometry();
            st.batch_groups.clear();
            st.baseline_batch_groups.clear();
            st.root_topology_revision = st.root_topology_revision.wrapping_add(1);
            st.baseline_group_signature = None;
            st.batch_staging.clear();
            st.requested_face_lods = vec![None; st.num_faces];
            st.resident_face_lods = vec![None; st.num_faces];
            st.resident_vertex_lods = vec![[1; 3]; st.num_faces];
            st.resident_vertex_lod_scratch.clear();
            st.classified_face_visibility = vec![false; st.num_faces];
            st.classified_culled_faces = st.num_faces;
            st.lod_dirty_faces.clear();
            st.lod_balance_scratch = batch::ResidentLodBalanceScratch::default();
            st.lod_topology = build_instance_lod_topology(
                &st.cached_instances,
                st.num_faces,
                &st.face_nodes,
            );
            st.adaptive_root_shadow.reset_topology();
            st.incremental_root_group_shadow.reset_model();
            // Picked face IDs and their captured animation pose are local to
            // one immutable asset epoch.
            st.adaptive_picked.clear();
            st.adaptive_retained_publication_error = None;
            st.active_suppressed_root_faces.clear();
            st.adaptive_batch_transition_pending = false;
            st.screen_topology_cache = None;
            if st.adaptive_root_shadow.is_enabled() {
                match build_screen_topology_cache(st.lod_topology.as_ref()) {
                    Ok(cache) => st.screen_topology_cache = Some(cache),
                    Err(error) => st.adaptive_root_shadow.record_unavailable(error),
                }
            }
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
            mark_batch_layout_dirty(st);
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
            ("adaptiveBatchRefreshCalls", batch_stats.adaptive_refresh_calls),
            ("adaptiveBatchRefreshNoops", batch_stats.adaptive_refresh_noops),
            ("baselineGroupCacheHits", batch_stats.baseline_group_cache_hits),
            ("baselineGroupCacheMisses", batch_stats.baseline_group_cache_misses),
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
            ("renderSceneExtractions", state.render_scene_extractions),
            (
                "renderSceneExtractionFailures",
                state.render_scene_extraction_failures,
            ),
            ("renderCommandPlanBuilds", state.render_command_plan_builds),
            ("renderCalls", state.render_calls),
            ("webglPatchFrames", state.webgl_patch_frames),
            ("patchPrepareFrames", state.patch_prepare_frames),
            (
                "skippedPatchPrepareFrames",
                state.skipped_patch_prepare_frames,
            ),
            ("patchPrepareCalls", state.patch_prepare_calls),
            (
                "lastPreparedPatchInstances",
                state.last_prepared_patch_instances,
            ),
            ("patchVisibilityFrames", state.patch_visibility_frames),
            (
                "skippedPatchVisibilityFrames",
                state.skipped_patch_visibility_frames,
            ),
            ("patchVisibilityCalls", state.patch_visibility_calls),
            (
                "lastVisibilityPatchInstances",
                state.last_visibility_patch_instances,
            ),
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
        #[cfg(feature = "webgpu-backend")]
        for (name, value) in [
            (
                "webgpuPresentationPatchFrames",
                state.webgpu_presentation_patch_frames,
            ),
            (
                "webgpuRetainedPresentationFrames",
                state.webgpu_retained_presentation_frames,
            ),
        ] {
            js_sys::Reflect::set(
                &result,
                &name.into(),
                &JsValue::from(value as f64),
            )
            .ok();
        }
        js_sys::Reflect::set(
            &result,
            &"batchTiming".into(),
            &serde_wasm_bindgen::to_value(&batch_stats.timing_snapshot())
                .unwrap_or(JsValue::NULL),
        ).ok();
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

struct RootGroupingProbeInputs<'a> {
    residents: &'a [Option<batch::ResidentLod>],
    topology: &'a quilting_mesh::HalfEdgeMesh,
    face_materials: &'a [usize],
    face_nodes: &'a [usize],
    face_render_nodes: &'a [usize],
    initial: batch::ResidentLod,
}

#[derive(Default)]
struct RootGroupingProbeScratch {
    vertex_max: Vec<u32>,
    face_vertex_lods: Vec<[u32; 3]>,
    groups: BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
}

fn probe_separate_root_grouping(
    inputs: &RootGroupingProbeInputs<'_>,
    scratch: &mut RootGroupingProbeScratch,
) -> f64 {
    let started_ms = browser_now_ms();
    batch::rebuild_resident_vertex_lods(
        inputs.residents,
        inputs.topology,
        inputs.initial,
        &mut scratch.vertex_max,
        &mut scratch.face_vertex_lods,
    );
    batch::group_resident_faces_into(
        inputs.residents,
        &scratch.face_vertex_lods,
        inputs.face_materials,
        inputs.face_nodes,
        inputs.face_render_nodes,
        inputs.initial,
        &mut scratch.groups,
    );
    browser_now_ms() - started_ms
}

fn probe_fused_root_grouping(
    inputs: &RootGroupingProbeInputs<'_>,
    scratch: &mut RootGroupingProbeScratch,
) -> f64 {
    let started_ms = browser_now_ms();
    batch::rebuild_resident_vertex_max(
        inputs.residents,
        inputs.topology,
        inputs.initial,
        &mut scratch.vertex_max,
    );
    batch::group_resident_faces_from_vertex_max_into(
        inputs.residents,
        inputs.topology,
        &scratch.vertex_max,
        &mut scratch.face_vertex_lods,
        inputs.face_materials,
        inputs.face_nodes,
        inputs.face_render_nodes,
        inputs.initial,
        &mut scratch.groups,
    );
    browser_now_ms() - started_ms
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RootGroupingProbeSnapshot {
    ok: bool,
    rounds: u32,
    faces: usize,
    separate_ms: Vec<f64>,
    fused_ms: Vec<f64>,
    exact_rounds: u32,
    mismatched_rounds: u32,
}

/// Compare the historical three-walk root rebuild with the fused two-walk
/// implementation over the exact current resident state. This is explicitly
/// on-demand; neither algorithm mutates renderer residency or GPU resources.
#[wasm_bindgen(js_name = "mr_measureRootGrouping")]
pub fn mr_measure_root_grouping(rounds: u32) -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else { return JsValue::NULL };
        let Some(topology) = state.lod_topology.as_ref() else { return JsValue::NULL };
        if state.face_render_nodes.len() != state.num_faces {
            return JsValue::NULL;
        }
        let rounds = rounds.clamp(1, 16);
        let inputs = RootGroupingProbeInputs {
            residents: &state.resident_face_lods,
            topology,
            face_materials: &state.face_materials,
            face_nodes: &state.face_nodes,
            face_render_nodes: &state.face_render_nodes,
            initial: bounded_standby_resident_lod(),
        };
        let mut separate = RootGroupingProbeScratch::default();
        let mut fused = RootGroupingProbeScratch::default();
        // Production retains this scratch across publications. Prime both
        // paths before sampling so allocator growth is not mistaken for the
        // grouping cost that steady-state camera updates actually pay.
        let _ = probe_separate_root_grouping(&inputs, &mut separate);
        let _ = probe_fused_root_grouping(&inputs, &mut fused);
        let mut separate_ms = Vec::with_capacity(rounds as usize);
        let mut fused_ms = Vec::with_capacity(rounds as usize);
        let mut exact_rounds = 0_u32;
        for round in 0..rounds {
            if round % 2 == 0 {
                separate_ms.push(probe_separate_root_grouping(&inputs, &mut separate));
                fused_ms.push(probe_fused_root_grouping(&inputs, &mut fused));
            } else {
                fused_ms.push(probe_fused_root_grouping(&inputs, &mut fused));
                separate_ms.push(probe_separate_root_grouping(&inputs, &mut separate));
            }
            exact_rounds += u32::from(
                separate.vertex_max == fused.vertex_max
                    && separate.face_vertex_lods == fused.face_vertex_lods
                    && separate.groups == fused.groups,
            );
        }
        serde_wasm_bindgen::to_value(&RootGroupingProbeSnapshot {
            ok: exact_rounds == rounds,
            rounds,
            faces: state.num_faces,
            separate_ms,
            fused_ms,
            exact_rounds,
            mismatched_rounds: rounds - exact_rounds,
        }).unwrap_or(JsValue::NULL)
    })
}

fn adaptive_screen_diagnostic_error(message: impl Into<String>) -> JsValue {
    adaptive_browser_value(&serde_json::json!({
        "ok": false,
        "error": message.into(),
    }))
}

fn adaptive_picked_snapshot_js(state: &MainState, refresh_published: Option<bool>) -> JsValue {
    adaptive_browser_value(&AdaptivePickedRefreshSnapshot {
        snapshot: state.adaptive_picked.snapshot(),
        transition_pending: state.adaptive_batch_transition_pending,
        retained_publication_enabled: state.adaptive_retained_publication_enabled,
        retained_publication_active: state
            .batches
            .keys()
            .any(|id| id.layer != batch::RenderBatchLayer::Complete),
        retained_publication_error: state.adaptive_retained_publication_error.as_deref(),
        refresh_published,
    })
}

/// Adaptive diagnostics are a browser-facing control contract, so even the
/// flattened snapshot and error maps must cross the WASM boundary as ordinary
/// JavaScript objects. The default serde-wasm-bindgen serializer represents
/// maps as `Map`; property reads such as `snapshot.enabled` would then be
/// silently undefined and the browser would never schedule the configured
/// adaptive classification.
fn adaptive_browser_value(value: &impl Serialize) -> JsValue {
    value
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true))
        .unwrap_or(JsValue::NULL)
}

/// Measure one current posed patch under the exact renderer camera and
/// conformal state, without changing residency or issuing GPU work.
///
/// Pass `face = -1` to inspect the last picked face. The diagnostic resolves
/// the same local pixel floor/capacity and ceiling/demand band as the live
/// adaptive planner so it can serve as an acceptance oracle rather than a
/// demand-only estimate.
#[wasm_bindgen(js_name = "mr_debugAdaptiveScreenPatch")]
pub fn mr_debug_adaptive_screen_patch(
    face: i32,
    min_px_per_segment: f64,
    max_px_per_segment: f64,
    max_depth: u32,
    max_leaves: u32,
) -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return adaptive_screen_diagnostic_error("renderer is not initialized");
        };
        let resolved_face = if face < 0 { state.highlight_face } else { face };
        let Ok(face_index) = usize::try_from(resolved_face) else {
            return adaptive_screen_diagnostic_error("no picked face is available");
        };
        if face_index >= state.num_faces {
            return adaptive_screen_diagnostic_error(format!(
                "face {face_index} is outside the {}-face scene",
                state.num_faces,
            ));
        }
        let Some(view_projection) = state.last_visibility_mvp else {
            return adaptive_screen_diagnostic_error("no rendered camera is available");
        };
        let Ok(max_depth) = u8::try_from(max_depth) else {
            return adaptive_screen_diagnostic_error("adaptive depth is outside u8 range");
        };
        let policy = ScreenPartitionPolicy {
            max_depth,
            max_leaves: max_leaves as usize,
            ..ScreenPartitionPolicy::default()
        };
        let viewport = [
            state.viewport_size.0.max(1) as f64,
            state.viewport_size.1.max(1) as f64,
        ];
        let view_projection = view_projection.map(f64::from);
        let node = state.face_nodes.get(face_index).copied().unwrap_or(0);
        let conformal = conformal_state_for_node(state, node);
        let patch = match state.surface_runtime.output_patch_for_face(
            &state.cached_instances,
            face_index,
            conformal.mobius,
            conformal.euclidean_model,
        ) {
            Ok(patch) => patch,
            Err(error) => return adaptive_screen_diagnostic_error(error),
        };
        let current_resident_lod = state
            .resident_face_lods
            .get(face_index)
            .and_then(|resident| *resident)
            .map(|resident| resident.edge_lods());
        let Some(source_requested_lod) = state
            .requested_face_lods
            .get(face_index)
            .and_then(|requested| *requested)
            .map(|requested| requested.edge_lods())
        else {
            return adaptive_screen_diagnostic_error(
                "source face has no completed LoD classification",
            );
        };
        let atlas_triangle_counts = TESS_CACHE.with(|cache| {
            cache
                .borrow()
                .iter()
                .map(|(key, patch)| {
                    (*key, patch.num_tri_indices.max(0) as u64 / 3)
                })
                .collect::<BTreeMap<_, _>>()
        });
        let request = AdaptiveScreenRequest {
            face: face_index as u32,
            node: node as u32,
            patch: &patch,
            view_projection,
            viewport,
            min_px_per_segment,
            max_px_per_segment,
            policy,
            max_face_edge_ratio: state.lod_grading.ratio(),
            source_requested_lod,
            current_resident_lod,
            atlas_triangle_counts: &atlas_triangle_counts,
        };
        match measure_adaptive_screen_patch(request) {
            Ok(snapshot) => serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL),
            Err(error) => adaptive_screen_diagnostic_error(error),
        }
    })
}

/// Read-only per-source-node LOD diagnostics for browser regression checks.
///
/// Aggregate scene histograms hide the exact failure mode of mixed-density
/// assets such as the chess set: a handful of broad board patches coexist with
/// tens of thousands of small piece patches. Keep this query off the frame path
/// and derive it only when DevTools or a test explicitly requests it.
#[wasm_bindgen(js_name = "mr_debugResidentLodNodes")]
pub fn mr_debug_resident_lod_nodes() -> JsValue {
    #[derive(Default)]
    struct NodeLodDiagnostics {
        node: u32,
        faces: u32,
        visible_faces: u32,
        missing_requested: u32,
        missing_resident: u32,
        anisotropic_faces: u32,
        max_edge_ratio: u32,
        resident_triangles: u64,
        min_source_edge: f64,
        max_source_edge: f64,
        max_source_edge_aspect: f64,
        requested_lod_histogram: BTreeMap<String, u32>,
        resident_lod_histogram: BTreeMap<String, u32>,
    }

    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct NodeLodDiagnosticsOutput {
        node: u32,
        faces: u32,
        visible_faces: u32,
        missing_requested: u32,
        missing_resident: u32,
        anisotropic_faces: u32,
        max_edge_ratio: u32,
        resident_triangles: u64,
        min_source_edge: f64,
        max_source_edge: f64,
        max_source_edge_aspect: f64,
        requested_lod_histogram: Vec<(String, u32)>,
        resident_lod_histogram: Vec<(String, u32)>,
    }

    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return JsValue::NULL;
        };
        let mut nodes = BTreeMap::<usize, NodeLodDiagnostics>::new();
        TESS_CACHE.with(|cache| {
            let cache = cache.borrow();
            for face in 0..state.num_faces {
                let node = state.face_nodes.get(face).copied().unwrap_or(0);
                let diagnostics = nodes.entry(node).or_insert_with(|| NodeLodDiagnostics {
                    node: node as u32,
                    min_source_edge: f64::INFINITY,
                    max_edge_ratio: 1,
                    ..NodeLodDiagnostics::default()
                });
                diagnostics.faces += 1;
                diagnostics.visible_faces += u32::from(
                    state
                        .classified_face_visibility
                        .get(face)
                        .copied()
                        .unwrap_or(false),
                );

                match state.requested_face_lods.get(face).and_then(|lod| *lod) {
                    Some(requested) => {
                        let [minimum, middle, maximum] = requested.canonical;
                        *diagnostics
                            .requested_lod_histogram
                            .entry(format!("{minimum}/{middle}/{maximum}"))
                            .or_default() += 1;
                    }
                    None => diagnostics.missing_requested += 1,
                }

                match state.resident_face_lods.get(face).and_then(|lod| *lod) {
                    Some(resident) => {
                        let [minimum, middle, maximum] = resident.canonical;
                        *diagnostics
                            .resident_lod_histogram
                            .entry(format!("{minimum}/{middle}/{maximum}"))
                            .or_default() += 1;
                        let ratio = maximum / minimum.max(1);
                        diagnostics.max_edge_ratio = diagnostics.max_edge_ratio.max(ratio);
                        diagnostics.anisotropic_faces += u32::from(ratio > 1);
                        diagnostics.resident_triangles += cache
                            .get(&resident.canonical)
                            .map_or(0, |patch| patch.num_tri_indices.max(0) as u64 / 3);
                    }
                    None => diagnostics.missing_resident += 1,
                }

                let base = face * instance_layout::STRIDE + instance_layout::offset::POSITIONS;
                let position = |corner: usize| -> Option<[f64; 3]> {
                    let offset = base + corner * 4 + 1;
                    Some([
                        *state.cached_instances.get(offset)? as f64,
                        *state.cached_instances.get(offset + 1)? as f64,
                        *state.cached_instances.get(offset + 2)? as f64,
                    ])
                };
                let distance = |a: [f64; 3], b: [f64; 3]| {
                    ((a[0] - b[0]).powi(2)
                        + (a[1] - b[1]).powi(2)
                        + (a[2] - b[2]).powi(2))
                    .sqrt()
                };
                if let (Some(p0), Some(p1), Some(p2)) =
                    (position(0), position(1), position(2))
                {
                    let edges = [distance(p1, p2), distance(p0, p2), distance(p0, p1)];
                    let minimum = edges.into_iter().fold(f64::INFINITY, f64::min);
                    let maximum = edges.into_iter().fold(0.0_f64, f64::max);
                    if minimum.is_finite() && maximum.is_finite() {
                        diagnostics.min_source_edge = diagnostics.min_source_edge.min(minimum);
                        diagnostics.max_source_edge = diagnostics.max_source_edge.max(maximum);
                        diagnostics.max_source_edge_aspect = diagnostics
                            .max_source_edge_aspect
                            .max(maximum / minimum.max(1.0e-30));
                    }
                }
            }
        });
        let output = nodes
            .into_values()
            .map(|mut diagnostics| {
                if !diagnostics.min_source_edge.is_finite() {
                    diagnostics.min_source_edge = 0.0;
                }
                NodeLodDiagnosticsOutput {
                    node: diagnostics.node,
                    faces: diagnostics.faces,
                    visible_faces: diagnostics.visible_faces,
                    missing_requested: diagnostics.missing_requested,
                    missing_resident: diagnostics.missing_resident,
                    anisotropic_faces: diagnostics.anisotropic_faces,
                    max_edge_ratio: diagnostics.max_edge_ratio,
                    resident_triangles: diagnostics.resident_triangles,
                    min_source_edge: diagnostics.min_source_edge,
                    max_source_edge: diagnostics.max_source_edge,
                    max_source_edge_aspect: diagnostics.max_source_edge_aspect,
                    requested_lod_histogram: diagnostics
                        .requested_lod_histogram
                        .into_iter()
                        .collect(),
                    resident_lod_histogram: diagnostics
                        .resident_lod_histogram
                        .into_iter()
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        serde_wasm_bindgen::to_value(&output).unwrap_or(JsValue::NULL)
    })
}

#[wasm_bindgen(js_name = "mr_setFaceMaterials")]
pub fn mr_set_face_materials(materials: &[i32]) {
    STATE.with(|s| {
        if let Some(ref mut st) = *s.borrow_mut() {
            st.face_materials = materials.iter().map(|&m| if m >= 0 { m as usize } else { 0 }).collect();
            mark_batch_layout_dirty(st);
        }
    });
}

#[wasm_bindgen(js_name = "mr_setFaceNodes")]
pub fn mr_set_face_nodes(nodes: &[i32]) {
    STATE.with(|state| {
        if let Some(renderer) = state.borrow_mut().as_mut() {
            let next = nodes
                .iter()
                .map(|&node| if node >= 0 { node as usize } else { 0 })
                .collect::<Vec<_>>();
            if renderer.face_nodes == next {
                renderer.pending_face_nodes = None;
                return;
            }
            // Face-node identity participates in picking, focus, exact seam
            // welding, and render-state grouping. Keep the active epoch
            // untouched until mr_setInstanceData embeds and uploads these IDs
            // together with their matching immutable controls.
            renderer.pending_face_nodes = Some(next);
        }
    });
}

/// Select the measured within-face LOD grading policy.
///
/// A resident atlas replacement retires every batch before publishing its new
/// lookup, which creates a safe live-policy handoff: the browser uploads the
/// replacement atlas, changes this policy while no batch can refer to the old
/// grading, then submits a freshly classified/reconciled batch snapshot. A
/// direct policy mutation with live batches remains rejected.
#[wasm_bindgen(js_name = "mr_setLodGradingRatio")]
pub fn mr_set_lod_grading_ratio(ratio: u32) -> bool {
    let Some(grading) = batch::FaceLodGrading::from_ratio(ratio) else {
        return false;
    };
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else { return false };
        if !state.batches.is_empty() && state.lod_grading != grading {
            web_sys::console::warn_1(
                &"LOD grading ratio changes require a batch-free atlas replacement fence".into(),
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
    let lod_lookup = match prepare_lod_atlas_lookup(
        patches.chunks_exact(7).map(|patch| [patch[0], patch[1], patch[2]]),
    ) {
        Ok(lookup) => lookup,
        Err(error) => {
            web_sys::console::error_1(
                &format!("packed tessellation atlas lookup: {error}").into(),
            );
            return false;
        }
    };

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
        mark_batch_layout_dirty(state);
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
        if let Some(same_context) = state.same_context_lod.as_mut() {
            same_context
                .compute
                .upload_atlas_lut(state.renderer.gl(), &lod_lookup.lut);
        }
        #[cfg(feature = "webgpu-backend")]
        if let Err(error) = crate::webgpu_backend::replace_atlas(
            patches,
            bary,
            tri_idx,
            line_idx,
            &lod_lookup,
        ) {
            web_sys::console::warn_1(
                &format!("WebGPU shadow atlas residency failed: {error}").into(),
            );
        }
        state.lod_atlas_lookup = Some(lod_lookup);
        true
    })
}

/// Prepare a classifier on the renderer's own WebGL context without changing
/// live LOD authority. The browser calls this only for explicit shadow/Rust
/// candidates; the incumbent worker remains untouched and rollback-safe.
fn build_current_prepared_lod_model(
    state: &MainState,
    total_vertices: usize,
    primary_faces: usize,
) -> Result<quilting_renderer::compute::PreparedLodModel, String> {
    let animation = state.surface_runtime.lod_animation_source(total_vertices)?;
    build_composed_lod_model(
        &state.cached_instances,
        &state.face_nodes,
        total_vertices,
        primary_faces,
        animation,
    )
    .and_then(prepare_lod_model)
}

#[wasm_bindgen(js_name = "mr_uploadComposedLodModel")]
pub fn mr_upload_composed_lod_model(
    total_vertices: u32,
    primary_faces: u32,
) -> Result<JsValue, JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let total_vertices = total_vertices as usize;
        let model = build_current_prepared_lod_model(state, total_vertices, primary_faces as usize)
            .map_err(|error| JsValue::from_str(&error))?;
        let model_fingerprint = prepared_lod_model_fingerprint(&model).stable_text();
        if model.residency.num_faces != state.num_faces {
            return Err(JsValue::from_str(
                "same-context LOD residency does not match renderer topology",
            ));
        }
        let atlas = state
            .lod_atlas_lookup
            .clone()
            .ok_or_else(|| JsValue::from_str("renderer LOD atlas is not resident"))?;
        let mut compute = LodCompute::new(state.renderer.gl(), model.residency.num_faces)
            .map_err(|error| JsValue::from_str(&error))?;
        let residency = compute.upload_model(state.renderer.gl(), &model, &atlas.lut);
        let batch_shadow = SameContextLodBatchShadow::from_renderer(state, residency.num_faces);
        clear_same_context_lod(state);
        state.same_context_lod = Some(SameContextLod {
            compute,
            residency,
            pending: None,
            completed: None,
            authority_candidate: None,
            batch_candidate: None,
            worker_batch_publication: None,
            worker_batch_snapshot: None,
            last_authority_stamp: None,
            authority_resident: Vec::new(),
            authority_packed: Vec::new(),
            authority_changed_indices: Vec::new(),
            authority_changed_packed: Vec::new(),
            adaptive_face_priorities: Vec::new(),
            batch_shadow,
            diagnostics: SameContextLodDiagnostics::default(),
        });
        #[cfg(feature = "webgpu-backend")]
        if let Err(error) = crate::webgpu_backend::replace_model(model, &atlas) {
            web_sys::console::warn_1(
                &format!("WebGPU shadow model residency failed: {error}").into(),
            );
        }
        let residency = &state
            .same_context_lod
            .as_ref()
            .expect("same-context LOD residency was just installed")
            .residency;
        serde_wasm_bindgen::to_value(&SameContextLodResidencySnapshot {
            ready: true,
            num_faces: residency.num_faces,
            num_vertices: residency.num_vertices,
            topology_domains: residency.node_first_faces.len(),
            mesh_radius: residency.mesh_radius,
            model_fingerprint,
            atlas_patches: atlas.keys.len(),
            atlas_max_lod: atlas.max_lod,
        })
        .map_err(|error| JsValue::from_str(&error.to_string()))
    })
}

/// Upload only staged WebGPU model residency when the incumbent worker remains
/// the live LOD authority. This avoids constructing a redundant WebGL2
/// same-context classifier solely to feed the backend shadow.
#[cfg(feature = "webgpu-backend")]
#[wasm_bindgen(js_name = "mr_uploadWebGpuComposedModel")]
pub fn mr_upload_webgpu_composed_model(
    total_vertices: u32,
    primary_faces: u32,
) -> Result<JsValue, JsValue> {
    STATE.with(|state| {
        let state = state.borrow();
        let state = state
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let model = build_current_prepared_lod_model(
            state,
            total_vertices as usize,
            primary_faces as usize,
        )
        .map_err(|error| JsValue::from_str(&error))?;
        if model.residency.num_faces != state.num_faces {
            return Err(JsValue::from_str(
                "WebGPU LOD residency does not match renderer topology",
            ));
        }
        let atlas = state
            .lod_atlas_lookup
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer LOD atlas is not resident"))?;
        let admitted = crate::webgpu_backend::replace_model(model, atlas)
            .map_err(|error| JsValue::from_str(&error))?;
        if !admitted {
            return Err(JsValue::from_str("WebGPU backend is not ready"));
        }
        serde_wasm_bindgen::to_value(&crate::webgpu_backend::diagnostics())
            .map_err(|error| JsValue::from_str(&error.to_string()))
    })
}

fn same_context_lod_request_stamp(
    request_id: u32,
    classified_faces: usize,
    animated: bool,
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
) -> Result<SameContextLodRequestStamp, String> {
    if request_id == 0 {
        return Err("same-context LOD request ID must be nonzero".to_string());
    }
    if classified_faces == 0 {
        return Err("same-context LOD request has no classified faces".to_string());
    }
    let pose = if animated {
        validate_pose_stamp(
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
        )?;
        Some(SameContextLodPoseStamp {
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
            payload: None,
        })
    } else {
        None
    };
    Ok(SameContextLodRequestStamp {
        request_id,
        classified_faces,
        pose,
    })
}

fn same_context_lod_authority_stamp(
    request_id: u32,
    classified_faces: usize,
    animated: bool,
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
) -> Result<SameContextLodRequestStamp, String> {
    if request_id != 0 {
        return same_context_lod_request_stamp(
            request_id,
            classified_faces,
            animated,
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
        );
    }
    if classified_faces == 0 {
        return Err("same-context LOD authority has no classified faces".to_string());
    }
    let pose = if animated {
        validate_pose_stamp(
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
        )?;
        Some(SameContextLodPoseStamp {
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
            payload: None,
        })
    } else {
        None
    };
    Ok(SameContextLodRequestStamp {
        request_id,
        classified_faces,
        pose,
    })
}

fn same_context_lod_snapshot_value(lod: &SameContextLod) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&lod.snapshot())
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

fn same_context_lod_slice_mismatches<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    let shared = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count();
    shared + left.len().abs_diff(right.len())
}

fn same_context_lod_group_mismatches(
    left: &BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    right: &BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
) -> usize {
    let changed_or_missing = left
        .iter()
        .filter(|(key, members)| right.get(*key) != Some(*members))
        .count();
    changed_or_missing + right.keys().filter(|key| !left.contains_key(*key)).count()
}

/// Once an exact classifier pair and the matching worker batch publication are
/// both present, dry-run the same Rust resident semantics and compare them with
/// the live worker-applied state. GPU buffers remain untouched.
fn try_compare_same_context_lod_batches(state: &mut MainState) -> Result<(), String> {
    let compare_groups = !state.adaptive_picked.is_enabled()
        && !state.adaptive_batch_transition_pending;
    let MainState {
        same_context_lod,
        lod_topology,
        lod_grading,
        face_materials,
        face_nodes,
        face_render_nodes,
        requested_face_lods,
        resident_face_lods,
        classified_face_visibility,
        classified_culled_faces,
        batch_groups,
        ..
    } = state;
    let Some(lod) = same_context_lod.as_mut() else {
        return Ok(());
    };
    let (Some(candidate_stamp), Some(worker_stamp)) = (
        lod.batch_candidate.as_ref().map(|candidate| candidate.stamp),
        lod.worker_batch_publication,
    ) else {
        return Ok(());
    };
    let candidate = lod
        .batch_candidate
        .take()
        .expect("same-context batch candidate was checked");
    lod.worker_batch_publication = None;
    let worker_snapshot = lod.worker_batch_snapshot.take();
    if lod
        .last_authority_stamp
        .is_some_and(|stamp| stamp.same_identity(worker_stamp))
    {
        lod.last_authority_stamp = None;
    }
    if !candidate_stamp.same_identity(worker_stamp) {
        lod.compute.recycle_decoded_vector(candidate.lods);
        return Err("same-context and worker batch publication stamps do not match".to_string());
    }
    if worker_snapshot
        .as_ref()
        .is_some_and(|snapshot| !snapshot.stamp.same_identity(worker_stamp))
    {
        lod.compute.recycle_decoded_vector(candidate.lods);
        return Err("delayed worker batch snapshot has the wrong stamp".to_string());
    }

    let num_faces = lod.residency.num_faces;
    if let Err(error) = lod.batch_shadow.apply(
        &candidate,
        num_faces,
        lod_topology.as_ref(),
        *lod_grading,
        face_materials,
        face_nodes,
        face_render_nodes,
    ) {
        lod.compute.recycle_decoded_vector(candidate.lods);
        return Err(error);
    }
    let authority_requested = worker_snapshot
        .as_ref()
        .map_or(requested_face_lods.as_slice(), |snapshot| snapshot.requested.as_slice());
    let authority_resident = worker_snapshot
        .as_ref()
        .map_or(resident_face_lods.as_slice(), |snapshot| snapshot.resident.as_slice());
    let authority_visible = worker_snapshot
        .as_ref()
        .map_or(classified_face_visibility.as_slice(), |snapshot| snapshot.visible.as_slice());
    let authority_culled = worker_snapshot
        .as_ref()
        .map_or(*classified_culled_faces, |snapshot| snapshot.culled_faces);
    let authority_groups = worker_snapshot
        .as_ref()
        .map_or(&*batch_groups, |snapshot| &snapshot.batch_groups);
    let requested_mismatches = same_context_lod_slice_mismatches(
        &lod.batch_shadow.requested,
        authority_requested,
    );
    let resident_mismatches = same_context_lod_slice_mismatches(
        &lod.batch_shadow.resident,
        authority_resident,
    );
    let visibility_mismatches = same_context_lod_slice_mismatches(
        &lod.batch_shadow.visible,
        authority_visible,
    );
    let culled_mismatch = lod.batch_shadow.culled_faces != authority_culled;
    let semantic_exact = requested_mismatches == 0
        && resident_mismatches == 0
        && visibility_mismatches == 0
        && !culled_mismatch;
    lod.diagnostics.batch_semantic_comparisons =
        lod.diagnostics.batch_semantic_comparisons.saturating_add(1);
    if semantic_exact {
        lod.diagnostics.exact_batch_semantic_comparisons = lod
            .diagnostics
            .exact_batch_semantic_comparisons
            .saturating_add(1);
    } else {
        lod.diagnostics.mismatched_batch_semantic_comparisons = lod
            .diagnostics
            .mismatched_batch_semantic_comparisons
            .saturating_add(1);
    }
    lod.diagnostics.last_requested_mismatches = requested_mismatches;
    lod.diagnostics.last_resident_mismatches = resident_mismatches;
    lod.diagnostics.last_visibility_mismatches = visibility_mismatches;
    lod.diagnostics.last_culled_mismatch = culled_mismatch;

    lod.diagnostics.last_batch_groups_compared = compare_groups;
    if compare_groups {
        let group_mismatches = same_context_lod_group_mismatches(
            &lod.batch_shadow.batch_groups,
            authority_groups,
        );
        lod.diagnostics.batch_group_comparisons =
            lod.diagnostics.batch_group_comparisons.saturating_add(1);
        if group_mismatches == 0 {
            lod.diagnostics.exact_batch_group_comparisons = lod
                .diagnostics
                .exact_batch_group_comparisons
                .saturating_add(1);
        } else {
            lod.diagnostics.mismatched_batch_group_comparisons = lod
                .diagnostics
                .mismatched_batch_group_comparisons
                .saturating_add(1);
        }
        lod.diagnostics.last_batch_group_mismatches = group_mismatches;
    } else {
        lod.diagnostics.skipped_adaptive_batch_groups = lod
            .diagnostics
            .skipped_adaptive_batch_groups
            .saturating_add(1);
        lod.diagnostics.last_batch_group_mismatches = 0;
    }

    if !semantic_exact {
        lod.batch_shadow.requested.clone_from_slice(authority_requested);
        lod.batch_shadow.resident.clone_from_slice(authority_resident);
        lod.batch_shadow.visible.clone_from_slice(authority_visible);
        lod.batch_shadow.culled_faces = authority_culled;
    }
    lod.compute.recycle_decoded_vector(candidate.lods);
    Ok(())
}

/// Publish one renderer-context classifier completion directly into retained
/// CPU residency and GPU batches. The full classifier vector never crosses
/// the WASM boundary; JavaScript observes only the bounded diagnostic summary.
fn publish_same_context_lod_completion(state: &mut MainState) -> Result<(), String> {
    let (
        mut candidate,
        mut prefix_indices,
        mut changed_indices,
        mut changed_packed,
        resident_faces,
    ) = {
        let lod = state
            .same_context_lod
            .as_mut()
            .ok_or_else(|| "same-context LOD model is not resident".to_string())?;
        let Some(candidate) = lod.completed.take() else {
            return Ok(());
        };
        let resident_faces = lod.residency.num_faces;
        if candidate.stamp.classified_faces == 0
            || candidate.stamp.classified_faces > resident_faces
            || candidate.stamp.classified_faces > u32::MAX as usize
            || candidate.packed.len() != candidate.stamp.classified_faces
        {
            lod.compute.recycle_readback_vector(candidate.packed);
            return Err(
                "same-context authority produced an invalid classified prefix".to_string(),
            );
        }
        (
            candidate,
            std::mem::take(&mut lod.batch_shadow.prefix_indices),
            std::mem::take(&mut lod.authority_changed_indices),
            std::mem::take(&mut lod.authority_changed_packed),
            resident_faces,
        )
    };

    let classified_faces = candidate.stamp.classified_faces;
    if !same_context_lod_pose_continuity_matches(
        candidate.stamp.pose,
        state.surface_runtime.pose_stamp(),
    ) {
        let lod = state
            .same_context_lod
            .as_mut()
            .ok_or_else(|| "same-context LOD model was retired during publication".to_string())?;
        lod.batch_shadow.prefix_indices = prefix_indices;
        lod.authority_changed_indices = changed_indices;
        lod.authority_changed_packed = changed_packed;
        lod.diagnostics.stale_authoritative_completions = lod
            .diagnostics
            .stale_authoritative_completions
            .saturating_add(1);
        lod.compute.recycle_readback_vector(candidate.packed);
        return Ok(());
    }
    state
        .same_context_lod
        .as_mut()
        .expect("same-context LOD residency was retained")
        .retain_adaptive_face_priorities(&candidate.packed);
    let packed_delta = diff_packed_lod_classifications(
        &candidate.packed,
        &state
            .same_context_lod
            .as_ref()
            .expect("same-context LOD residency was retained")
            .authority_packed,
        &mut changed_indices,
        &mut changed_packed,
    );
    let packed_full_snapshot = packed_delta.full_snapshot;
    let packed_changed_records = packed_delta.changed_records;
    let packed_unchanged = packed_delta.is_unchanged();
    let skip_packed_admission = packed_unchanged && !state.batch_layout_dirty;
    let use_sparse_packed = !packed_full_snapshot && !changed_indices.is_empty();
    if skip_packed_admission {
        state.lod_dirty_faces.clear();
        if state.adaptive_picked.is_enabled() || state.adaptive_batch_transition_pending {
            refresh_adaptive_picked_batches(state);
        }
    } else if use_sparse_packed {
        update_batches_in_state(
            state,
            RetainedLodPublication::PackedWords(&changed_packed),
            Some(&changed_indices),
        );
    } else if classified_faces == resident_faces {
        update_batches_in_state(
            state,
            RetainedLodPublication::PackedWords(&candidate.packed),
            None,
        );
    } else {
        let classified_faces_u32 = u32::try_from(classified_faces)
            .expect("same-context authority prefix was range-checked");
        prefix_indices.clear();
        prefix_indices.extend(0..classified_faces_u32);
        update_batches_in_state(
            state,
            RetainedLodPublication::PackedWords(&candidate.packed),
            Some(&prefix_indices),
        );
    }

    let changed_faces = state.lod_dirty_faces.len();
    let pose = candidate.stamp.pose;
    let lod = state
        .same_context_lod
        .as_mut()
        .ok_or_else(|| "same-context LOD model was retired during publication".to_string())?;
    if !packed_unchanged {
        std::mem::swap(&mut lod.authority_packed, &mut candidate.packed);
    }
    lod.batch_shadow.prefix_indices = prefix_indices;
    lod.authority_changed_indices = changed_indices;
    lod.authority_changed_packed = changed_packed;
    lod.diagnostics.authoritative_publications = lod
        .diagnostics
        .authoritative_publications
        .saturating_add(1);
    lod.diagnostics.packed_changed_records = lod
        .diagnostics
        .packed_changed_records
        .saturating_add(packed_changed_records as u64);
    if packed_unchanged {
        lod.diagnostics.packed_publication_noops = lod
            .diagnostics
            .packed_publication_noops
            .saturating_add(1);
    }
    if use_sparse_packed {
        lod.diagnostics.packed_sparse_publications = lod
            .diagnostics
            .packed_sparse_publications
            .saturating_add(1);
    }
    if skip_packed_admission {
        lod.diagnostics.packed_admission_skips = lod
            .diagnostics
            .packed_admission_skips
            .saturating_add(1);
    }
    lod.diagnostics.last_authoritative_faces = classified_faces;
    lod.diagnostics.last_authoritative_changed_faces = changed_faces;
    lod.diagnostics.last_packed_publication_unchanged = packed_unchanged;
    lod.diagnostics.last_packed_changed_records = packed_changed_records;
    lod.diagnostics.last_packed_full_snapshot = packed_full_snapshot;
    lod.diagnostics.last_packed_admission_skipped = skip_packed_admission;
    lod.diagnostics.last_authoritative_pose_revision =
        pose.map_or(0, |pose| pose.revision);
    lod.diagnostics.last_authoritative_pose_continuity_epoch =
        pose.map_or(0, |pose| pose.continuity_epoch);
    lod.compute.recycle_readback_vector(candidate.packed);
    Ok(())
}

fn restore_renderer_after_lod_compute(gl: &glow::Context, viewport: (i32, i32)) {
    unsafe {
        gl.use_program(None);
        gl.bind_vertex_array(None);
        gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.disable(glow::RASTERIZER_DISCARD);
        gl.enable(glow::DEPTH_TEST);
        gl.enable(glow::BLEND);
        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        gl.viewport(0, 0, viewport.0.max(1), viewport.1.max(1));
    }
}

fn validated_lod_dispatch_payload(
    mobius: &[f32],
    subject_states: &[f32],
    density: f32,
    pixel_floor: f32,
    view_projection: &[f32],
    viewport: [f32; 2],
) -> Result<([f32; 16], [f32; 16]), JsValue> {
    if mobius.len() != 16 || view_projection.len() != 16 {
        return Err(JsValue::from_str(
            "LOD dispatch matrices must contain exactly 16 floats",
        ));
    }
    if subject_states.len() % quilting_renderer::compute::SUBJECT_STATE_STRIDE != 0
        || !mobius
            .iter()
            .chain(subject_states)
            .chain(view_projection)
            .all(|value| value.is_finite())
        || !density.is_finite()
        || density <= 0.0
        || !pixel_floor.is_finite()
        || pixel_floor < 0.0
        || viewport.iter().any(|extent| !extent.is_finite() || *extent <= 0.0)
    {
        return Err(JsValue::from_str("LOD dispatch payload is malformed"));
    }
    let mut legacy_mobius = [0.0; 16];
    legacy_mobius.copy_from_slice(mobius);
    let mut view_projection_matrix = [0.0; 16];
    view_projection_matrix.copy_from_slice(view_projection);
    Ok((legacy_mobius, view_projection_matrix))
}

/// Dispatch one current-view classifier epoch directly on WebGPU. A zero face
/// limit replaces the complete resident scene; a nonzero, topology-closed
/// primary prefix refreshes an existing complete device epoch in place.
#[cfg(feature = "webgpu-backend")]
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "mr_dispatchWebGpuLod")]
pub fn mr_dispatch_webgpu_lod(
    animated: bool,
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
    mobius: &[f32],
    subject_states: &[f32],
    face_limit: u32,
    density: f32,
    min_px: f32,
    vp_matrix: &[f32],
    vp_width: f32,
    vp_height: f32,
) -> Result<bool, JsValue> {
    let (legacy_mobius, view_projection) = validated_lod_dispatch_payload(
        mobius,
        subject_states,
        density,
        min_px,
        vp_matrix,
        [vp_width, vp_height],
    )?;
    STATE.with(|state| {
        let state = state.borrow();
        let state = state
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let max_lod = state
            .lod_atlas_lookup
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer LOD atlas is not resident"))?
            .max_lod;
        let resident_faces = state
            .same_context_lod
            .as_ref()
            .map(|lod| lod.residency.num_faces)
            .ok_or_else(|| JsValue::from_str("renderer LOD model is not resident"))?;
        let classified_faces = if face_limit == 0 {
            resident_faces
        } else {
            usize::try_from(face_limit)
                .map_err(|_| JsValue::from_str("WebGPU LOD face limit exceeds usize"))?
        };
        if animated {
            state
                .surface_runtime
                .lod_pose_source(
                    clip_time_seconds,
                    sample_time_seconds,
                    revision,
                    continuity_epoch,
                )
                .map_err(|error| JsValue::from_str(&error))?;
        }
        let (joint_matrices, morph_weights) = state
            .surface_runtime
            .current_pose_payload()
            .map_err(|error| JsValue::from_str(&error))?;
        crate::webgpu_backend::dispatch_lod(
            current_render_pose_identity(state),
            legacy_mobius,
            subject_states,
            classified_faces,
            density,
            min_px,
            max_lod,
            view_projection,
            [vp_width, vp_height],
            joint_matrices,
            morph_weights,
            state.lod_grading,
        )
        .map_err(|error| JsValue::from_str(&error))
    })
}

/// Dispatch the exact worker-equivalent classifier on the renderer WebGL
/// context. This is shadow-only: its result never mutates resident batches.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "mr_dispatchSameContextLod")]
pub fn mr_dispatch_same_context_lod(
    request_id: u32,
    animated: bool,
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
    mobius: &[f32],
    subject_states: &[f32],
    face_limit: u32,
    density: f32,
    min_px: f32,
    vp_matrix: &[f32],
    vp_width: f32,
    vp_height: f32,
) -> Result<bool, JsValue> {
    let (legacy_mobius, vp) = validated_lod_dispatch_payload(
        mobius,
        subject_states,
        density,
        min_px,
        vp_matrix,
        [vp_width, vp_height],
    )?;

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let max_lod = state
            .lod_atlas_lookup
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer LOD atlas is not resident"))?
            .max_lod;
        let viewport = state.viewport_size;
        let MainState {
            renderer,
            surface_runtime,
            same_context_lod,
            ..
        } = state;
        let lod = same_context_lod
            .as_mut()
            .ok_or_else(|| JsValue::from_str("same-context LOD model is not resident"))?;
        if lod.pending.is_some() || lod.completed.is_some() {
            lod.diagnostics.skipped_busy = lod.diagnostics.skipped_busy.saturating_add(1);
            return Ok(false);
        }

        let classified_faces = if face_limit == 0 {
            lod.residency.num_faces
        } else {
            lod.residency.num_faces.min(face_limit as usize)
        };
        let mut stamp = same_context_lod_request_stamp(
            request_id,
            classified_faces,
            animated,
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
        )
        .map_err(|error| JsValue::from_str(&error))?;
        let pose = if animated {
            Some(
                surface_runtime
                    .lod_pose_source(
                        clip_time_seconds,
                        sample_time_seconds,
                        revision,
                        continuity_epoch,
                    )
                    .map_err(|error| JsValue::from_str(&error))?,
            )
        } else {
            None
        };
        if let Some(pose) = pose {
            let retained = SameContextLodPoseStamp {
                clip_time_seconds: pose.clip_time_seconds,
                sample_time_seconds: pose.sample_time_seconds,
                revision: pose.revision,
                continuity_epoch: pose.continuity_epoch,
                payload: None,
            };
            if !stamp.pose.is_some_and(|request| request.same_identity(retained)) {
                return Err(JsValue::from_str(
                    "same-context LOD did not retain the requested renderer pose",
                ));
            }
        }
        let joint_matrices = pose.map_or(&[][..], |pose| pose.joint_matrices);
        let morph_weights = pose.map_or(&[][..], |pose| pose.morph_weights);
        if let Some(pose_stamp) = stamp.pose.as_mut() {
            pose_stamp.payload = Some(same_context_lod_pose_payload_fingerprint(
                joint_matrices,
                morph_weights,
            ));
        }
        let num_joints = u32::try_from(joint_matrices.len() / 16)
            .map_err(|_| JsValue::from_str("same-context joint count exceeds the ABI"))?;
        let num_morph_targets = u32::try_from(morph_weights.len())
            .map_err(|_| JsValue::from_str("same-context morph count exceeds the ABI"))?;

        let dispatch_state = prepare_lod_dispatch_state(
            subject_states,
            &lod.residency,
            classified_faces,
            legacy_mobius,
        );
        let gl = renderer.gl();
        if !joint_matrices.is_empty() {
            lod.compute.upload_joint_matrices(gl, joint_matrices);
        }
        if !morph_weights.is_empty() {
            lod.compute.upload_morph_weights(gl, morph_weights);
        }

        let dispatch_started_ms = browser_now_ms();
        let dispatched = lod.compute.compute_lods(
            gl,
            classified_faces,
            lod.residency.num_vertices,
            num_joints,
            num_morph_targets,
            &dispatch_state.subjects,
            dispatch_state.baseline_mobius,
            dispatch_state.baseline_model,
            dispatch_state.pole,
            dispatch_state.mobius_power,
            dispatch_state.c_norm_sq,
            dispatch_state.has_pole,
            density,
            lod.residency.mesh_radius,
            min_px,
            max_lod,
            &vp,
            vp_width,
            vp_height,
        );
        restore_renderer_after_lod_compute(gl, viewport);
        let dispatched = dispatched.map_err(|error| JsValue::from_str(&error))?;
        if dispatched != classified_faces {
            return Err(JsValue::from_str(
                "same-context LOD dispatch returned an incomplete face domain",
            ));
        }
        let readback = lod
            .compute
            .stage_readback(gl, classified_faces)
            .map_err(|error| JsValue::from_str(&error))?;
        let fence = match unsafe { gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) } {
            Ok(fence) => fence,
            Err(error) => {
                lod.compute.discard_staged_readback(gl, readback);
                return Err(JsValue::from_str(&format!(
                    "same-context LOD fence creation failed: {error}",
                )));
            }
        };
        unsafe { gl.flush(); }
        let fence_started_ms = browser_now_ms();
        lod.pending = Some(SameContextLodPending {
            stamp,
            fence,
            readback,
            fence_started_ms,
        });
        lod.diagnostics.dispatches = lod.diagnostics.dispatches.saturating_add(1);
        lod.diagnostics.last_request_id = request_id;
        lod.diagnostics.last_classified_faces = classified_faces;
        lod.diagnostics.last_subject_records = dispatch_state.subjects.len();
        let dispatch_ms = fence_started_ms - dispatch_started_ms;
        lod.diagnostics.last_dispatch_ms = dispatch_ms;
        lod.diagnostics.dispatch_ms.record(dispatch_ms);
        lod.diagnostics.last_error = None;
        Ok(true)
    })
}

/// Poll one same-context job without blocking the renderer. In authority mode
/// a completed full prefix is published directly into retained Rust batches;
/// shadow mode instead waits for and compares the matching worker publication.
#[wasm_bindgen(js_name = "mr_pollSameContextLod")]
pub fn mr_poll_same_context_lod(authoritative: bool) -> Result<JsValue, JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        {
            let MainState { renderer, same_context_lod, .. } = state;
            let lod = same_context_lod
                .as_mut()
                .ok_or_else(|| JsValue::from_str("same-context LOD model is not resident"))?;
            lod.diagnostics.polls = lod.diagnostics.polls.saturating_add(1);
            let Some(fence) = lod.pending.as_ref().map(|pending| pending.fence) else {
                return same_context_lod_snapshot_value(lod);
            };
            let gl = renderer.gl();
            let status = unsafe { gl.client_wait_sync(fence, 0, 0) };
            if status == glow::TIMEOUT_EXPIRED
                || (status != glow::ALREADY_SIGNALED
                    && status != glow::CONDITION_SATISFIED
                    && status != glow::WAIT_FAILED)
            {
                return same_context_lod_snapshot_value(lod);
            }

            let pending = lod.pending.take().expect("same-context fence was present");
            unsafe { gl.delete_sync(pending.fence); }
            if status == glow::WAIT_FAILED {
                lod.compute.discard_staged_readback(gl, pending.readback);
                lod.diagnostics.failures = lod.diagnostics.failures.saturating_add(1);
                lod.diagnostics.last_error =
                    Some("same-context LOD fence wait failed".to_string());
                return same_context_lod_snapshot_value(lod);
            }
            let ready_ms = browser_now_ms();
            let readback_started_ms = ready_ms;
            let readback_bytes = pending.readback.byte_len();
            let packed = match lod.compute.finish_staged_readback(gl, pending.readback) {
                Ok(classification) => classification,
                Err(error) => {
                    lod.diagnostics.failures = lod.diagnostics.failures.saturating_add(1);
                    lod.diagnostics.last_error = Some(error);
                    return same_context_lod_snapshot_value(lod);
                }
            };
            let readback_finished_ms = browser_now_ms();
            lod.diagnostics.completions = lod.diagnostics.completions.saturating_add(1);
            // This includes browser scheduling until the first signaling poll.
            // It is deliberately not labeled GPU time; WebGL2 exposes no
            // timestamp for the moment the fence became signaled.
            let fence_poll_latency_ms = ready_ms - pending.fence_started_ms;
            let readback_ms = readback_finished_ms - readback_started_ms;
            lod.diagnostics.last_fence_poll_latency_ms = fence_poll_latency_ms;
            lod.diagnostics.last_readback_ms = readback_ms;
            lod.diagnostics
                .fence_poll_latency_ms
                .record(fence_poll_latency_ms);
            lod.diagnostics.readback_ms.record(readback_ms);
            lod.diagnostics.last_readback_bytes = readback_bytes;
            lod.completed = Some(SameContextLodCompleted {
                stamp: pending.stamp,
                packed,
            });
            if !authoritative {
                if let Err(error) = lod.try_compare() {
                    lod.diagnostics.failures = lod.diagnostics.failures.saturating_add(1);
                    lod.diagnostics.last_error = Some(error);
                }
            }
        }
        let publication_started_ms = browser_now_ms();
        let publication = if authoritative {
            publish_same_context_lod_completion(state)
        } else {
            try_compare_same_context_lod_batches(state)
        };
        if authoritative {
            let publication_ms = browser_now_ms() - publication_started_ms;
            if let Some(lod) = state.same_context_lod.as_mut() {
                lod.diagnostics.last_publication_ms = publication_ms;
                lod.diagnostics.publication_ms.record(publication_ms);
            }
        }
        if let Err(error) = publication {
            if let Some(lod) = state.same_context_lod.as_mut() {
                lod.diagnostics.failures = lod.diagnostics.failures.saturating_add(1);
                lod.diagnostics.last_error = Some(error);
            }
        }
        let lod = state
            .same_context_lod
            .as_ref()
            .ok_or_else(|| JsValue::from_str("same-context LOD model is not resident"))?;
        same_context_lod_snapshot_value(lod)
    })
}

/// Reconstruct the admitted worker snapshot and compare it with the matching
/// same-context result. This never applies the shadow payload to batches.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = "mr_recordSameContextLodAuthority")]
pub fn mr_record_same_context_lod_authority(
    request_id: u32,
    animated: bool,
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
    lods: &[f32],
    indices: &[u32],
    full_snapshot: bool,
    classified_faces: u32,
    resident_faces: u32,
    pose_joint_matrices: &[f32],
    pose_morph_weights: &[f32],
    worker_full_fingerprint: &str,
) -> Result<JsValue, JsValue> {
    let mut stamp = same_context_lod_authority_stamp(
        request_id,
        classified_faces as usize,
        animated,
        clip_time_seconds,
        sample_time_seconds,
        revision,
        continuity_epoch,
    )
    .map_err(|error| JsValue::from_str(&error))?;
    if let Some(pose_stamp) = stamp.pose.as_mut() {
        pose_stamp.payload = Some(same_context_lod_pose_payload_fingerprint(
            pose_joint_matrices,
            pose_morph_weights,
        ));
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let lod = state
            .same_context_lod
            .as_mut()
            .ok_or_else(|| JsValue::from_str("same-context LOD model is not resident"))?;
        if resident_faces as usize != lod.residency.num_faces {
            return Err(JsValue::from_str(
                "worker and same-context LOD residency shapes do not match",
            ));
        }
        apply_lod_classification_publication(
            &mut lod.authority_resident,
            lods,
            indices,
            full_snapshot,
            classified_faces as usize,
            resident_faces as usize,
        )
        .map_err(|error| JsValue::from_str(&error))?;
        lod.diagnostics.authority_updates = lod.diagnostics.authority_updates.saturating_add(1);
        lod.last_authority_stamp = (request_id != 0).then_some(stamp);
        let fields = stamp.classified_faces * FLOATS_PER_FACE_OUTPUT;
        let reconstructed = exact_f32_slice_fingerprint(&lod.authority_resident[..fields]);
        let reconstructed = format!("{}:{:016x}", reconstructed.0, reconstructed.1);
        lod.diagnostics.publication_fingerprint_comparisons = lod
            .diagnostics
            .publication_fingerprint_comparisons
            .saturating_add(1);
        if reconstructed != worker_full_fingerprint {
            lod.diagnostics.mismatched_publication_fingerprints = lod
                .diagnostics
                .mismatched_publication_fingerprints
                .saturating_add(1);
        }
        lod.diagnostics.last_worker_full_fingerprint =
            Some(worker_full_fingerprint.to_string());
        lod.diagnostics.last_reconstructed_full_fingerprint = Some(reconstructed);

        let candidate = lod
            .pending
            .as_ref()
            .map(|pending| pending.stamp)
            .or_else(|| lod.completed.as_ref().map(|completed| completed.stamp));
        if let Some(candidate) = candidate.filter(|candidate| candidate.request_id == request_id) {
            if !candidate.same_identity(stamp) {
                return Err(JsValue::from_str(
                    "worker and same-context LOD request stamps do not match",
                ));
            }
            lod.authority_candidate = Some(SameContextLodAuthority {
                stamp,
                lods: lod.authority_resident[..fields].to_vec(),
                worker_full_fingerprint: worker_full_fingerprint.to_string(),
            });
        }
        if let Err(error) = lod.try_compare() {
            lod.diagnostics.failures = lod.diagnostics.failures.saturating_add(1);
            lod.diagnostics.last_error = Some(error);
        }
        same_context_lod_snapshot_value(lod)
    })
}

/// Record that the admitted worker publication has finished mutating live
/// resident/batch state. The matching exact same-context vector can now be
/// dry-run through the shared Rust semantics and compared without touching GPU
/// resources.
#[wasm_bindgen(js_name = "mr_recordSameContextLodBatchPublication")]
pub fn mr_record_same_context_lod_batch_publication(
    request_id: u32,
) -> Result<JsValue, JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state
            .as_mut()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let stamp = state
            .same_context_lod
            .as_ref()
            .and_then(|lod| lod.last_authority_stamp)
            .filter(|stamp| stamp.request_id == request_id)
            .ok_or_else(|| {
                JsValue::from_str("same-context LOD has no matching worker publication")
            })?;
        state
            .same_context_lod
            .as_mut()
            .expect("same-context LOD stamp came from resident state")
            .worker_batch_publication = Some(stamp);
        if let Err(error) = try_compare_same_context_lod_batches(state) {
            if let Some(lod) = state.same_context_lod.as_mut() {
                lod.diagnostics.failures = lod.diagnostics.failures.saturating_add(1);
                lod.diagnostics.last_error = Some(error);
            }
        }
        let needs_delayed_snapshot = state
            .same_context_lod
            .as_ref()
            .is_some_and(|lod| {
                lod.worker_batch_publication
                    .is_some_and(|pending| pending.same_identity(stamp))
                    && lod.worker_batch_snapshot.is_none()
            });
        if needs_delayed_snapshot {
            let snapshot = SameContextLodBatchAuthoritySnapshot::from_renderer(state, stamp);
            state
                .same_context_lod
                .as_mut()
                .expect("same-context LOD publication remains resident")
                .worker_batch_snapshot = Some(snapshot);
        }
        let lod = state
            .same_context_lod
            .as_ref()
            .expect("same-context LOD stamp came from resident state");
        same_context_lod_snapshot_value(lod)
    })
}

/// Cancel one shadow request whose worker authority failed or was retired.
#[wasm_bindgen(js_name = "mr_cancelSameContextLod")]
pub fn mr_cancel_same_context_lod(request_id: u32) -> bool {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else { return false };
        let MainState { renderer, same_context_lod, .. } = state;
        same_context_lod
            .as_mut()
            .is_some_and(|lod| lod.cancel_request(renderer.gl(), request_id))
    })
}

#[wasm_bindgen(js_name = "mr_sameContextLodDiagnostics")]
pub fn mr_same_context_lod_diagnostics() -> Result<JsValue, JsValue> {
    STATE.with(|state| {
        let state = state.borrow();
        let state = state
            .as_ref()
            .ok_or_else(|| JsValue::from_str("renderer is not initialized"))?;
        let lod = state
            .same_context_lod
            .as_ref()
            .ok_or_else(|| JsValue::from_str("same-context LOD model is not resident"))?;
        same_context_lod_snapshot_value(lod)
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
fn decode_texture_reference(enabled: bool, raw_index: f32) -> Option<Option<u32>> {
    if !enabled {
        return Some(None);
    }
    if !raw_index.is_finite()
        || raw_index < 0.0
        || raw_index >= 4_294_967_296.0
        || raw_index.fract() != 0.0
    {
        return None;
    }
    Some(Some(raw_index as u32))
}

fn decode_optional_texture_reference(raw_index: f32) -> Option<Option<u32>> {
    if raw_index.is_finite() && raw_index < 0.0 {
        Some(None)
    } else {
        decode_texture_reference(true, raw_index)
    }
}

fn decode_attenuation_distance(value: f32) -> Option<Option<f32>> {
    if value == f32::INFINITY {
        Some(None)
    } else if value.is_finite() && value > 0.0 {
        Some(Some(value))
    } else {
        None
    }
}

fn decode_pbr_material(d: &[f32]) -> Option<PbrMaterial> {
    if d.len() < MATERIAL_STRIDE
        || d[..47].iter().chain(&d[48..50]).any(|value| !value.is_finite())
    {
        return None;
    }
    let textures = PbrTextureReferences {
        base_color: decode_texture_reference(d[14] > 0.5, d[36])?,
        metallic_roughness: decode_texture_reference(d[15] > 0.5, d[37])?,
        normal: decode_texture_reference(d[16] > 0.5, d[38])?,
        emissive: decode_texture_reference(d[17] > 0.5, d[39])?,
        occlusion: decode_texture_reference(d[18] > 0.5, d[40])?,
        transmission: decode_optional_texture_reference(d[48])?,
    };
    let material = PbrMaterial {
        base_color: [d[0], d[1], d[2], d[3]],
        metallic: d[4],
        roughness: d[5],
        normal_scale: d[6],
        occlusion_strength: d[7],
        alpha_cutoff: d[8],
        alpha_mode: PbrAlphaMode::from_wire_f32(d[9])?,
        unlit: d[10] > 0.5,
        emissive_factor: [d[11], d[12], d[13]],
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
        ior: d[41],
        transmission_factor: d[42],
        thickness_factor: d[43],
        attenuation_color: [d[44], d[45], d[46]],
        attenuation_distance: decode_attenuation_distance(d[47])?,
        double_sided: d[49] > 0.5,
        textures,
    };
    material.validate().ok()?;
    Some(material)
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
            st.render_commands_dirty = true;
            st.render_scene_dirty = true;
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
        state.render_commands_dirty = true;
        state.render_scene_dirty = true;
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
        #[cfg(feature = "webgpu-backend")]
        match crate::webgpu_backend::replace_image_bitmaps(&images) {
            Ok(true) => info!(
                "Mirrored {uploaded}/{} ImageBitmap textures directly to WebGPU",
                images.len()
            ),
            Ok(false) => {}
            Err(error) => warn!(
                "WebGPU ImageBitmap texture mirror failed; retaining WebGL textures: {error}"
            ),
        }
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
                clear_same_context_lod(state);
                state.surface_runtime.set_skinning(&ji, &jw);
                state.patch_prepare_dirty = true;
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
                clear_same_context_lod(state);
                state.surface_runtime.set_morph_targets(deltas, nv, nt);
                state.patch_prepare_dirty = true;
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
            #[cfg(feature = "webgpu-backend")]
            match crate::webgpu_backend::replace_environment_maps(
                prefiltered,
                pf_size,
                irradiance,
                ir_size,
            ) {
                Ok(true) => debug!("Mirrored environment maps into WebGPU"),
                Ok(false) => {}
                Err(error) => warn!(
                    "WebGPU environment mirror rejected; retained incumbent state: {}",
                    error
                ),
            }
        }
    });
}

#[derive(Default)]
struct TransactionalBatchUploadStats {
    retained: usize,
    created: usize,
    reallocated: usize,
    retired: usize,
    uploaded_instances: usize,
}

struct TransactionalBatchUploadFailure {
    message: String,
    missing_atlas_entries: usize,
    failures: usize,
}

/// Stage every changed bucket in fresh GL resources. Existing buckets are not
/// mutated or retired until the whole candidate exists, so allocation,
/// packing, and atlas failures restore the exact previously published epoch.
fn upload_batch_groups_transactionally(
    state: &mut MainState,
    force_upload: bool,
) -> Result<TransactionalBatchUploadStats, TransactionalBatchUploadFailure> {
    let gl = state.renderer.gl();
    let mut previous_batches = std::mem::take(&mut state.batches);
    let mut reused_batches = BTreeMap::<batch::RenderBatchId, GpuBatch>::new();
    let mut staged_batches = BTreeMap::<batch::RenderBatchId, GpuBatch>::new();
    let mut stats = TransactionalBatchUploadStats::default();
    let stride = instance_layout::BATCH_TOPOLOGY_STRIDE;
    let max_batch_size = state.batch_groups.values().map(Vec::len).max().unwrap_or(0);
    state.batch_staging.resize(max_batch_size * stride, 0.0);

    let attempt = TESS_CACHE.with(|cache| -> Result<(), TransactionalBatchUploadFailure> {
        let cache = cache.borrow();
        for (&key, members) in &state.batch_groups {
            let id = batch::RenderBatchId::complete(key);
            let tess = cache.get(&key.lod).ok_or_else(|| TransactionalBatchUploadFailure {
                message: format!("adaptive batch needs missing atlas patch {:?}", key.lod),
                missing_atlas_entries: 1,
                failures: 0,
            })?;
            let membership_changed = previous_batches
                .get(&id)
                .is_none_or(|batch| batch.members != *members);
            if !force_upload && !membership_changed {
                stats.retained += 1;
                let previous = previous_batches.remove(&id).ok_or_else(|| {
                    TransactionalBatchUploadFailure {
                        message: format!("retained adaptive batch {key:?} disappeared"),
                        missing_atlas_entries: 0,
                        failures: 1,
                    }
                })?;
                reused_batches.insert(id, previous);
                continue;
            }

            let batch_floats = fill_batch_instance_data(
                key,
                members,
                &mut state.batch_staging,
            )
            .map_err(|error| TransactionalBatchUploadFailure {
                message: format!("could not pack adaptive batch {key:?}: {error}"),
                missing_atlas_entries: 0,
                failures: 1,
            })?;
            let staged = create_gpu_batch(
                gl,
                tess,
                key,
                members,
                &state.batch_staging[..batch_floats],
            )
            .map_err(|error| TransactionalBatchUploadFailure {
                message: format!("could not stage adaptive batch {key:?}: {error}"),
                missing_atlas_entries: 0,
                failures: 1,
            })?;
            stats.reallocated += usize::from(previous_batches.contains_key(&id));
            stats.created += 1;
            stats.uploaded_instances += members.len();
            staged_batches.insert(id, staged);
        }
        Ok(())
    });

    if let Err(failure) = attempt {
        for (_, batch) in staged_batches {
            batch.destroy(gl);
        }
        previous_batches.append(&mut reused_batches);
        state.batches = previous_batches;
        return Err(failure);
    }

    reused_batches.append(&mut staged_batches);
    stats.retired = previous_batches.len();
    for (_, batch) in previous_batches {
        batch.destroy(gl);
    }
    state.batches = reused_batches;
    if stats.created != 0 || stats.retired != 0 {
        state.render_commands_dirty = true;
    }
    if !state.active_suppressed_root_faces.is_empty() {
        match state.renderer.set_suppressed_source_faces(state.num_faces, &[]) {
            Ok(changed) => {
                if changed != 0 {
                    state.last_visibility_mvp = None;
                }
            }
            Err(error) => {
                warn!("Could not clear inactive retained-root suppression mask: {error}");
            }
        }
        // Complete batches never consult the mask, so even a diagnostic mask
        // cleanup failure cannot alter this successfully published epoch.
        state.active_suppressed_root_faces.clear();
    }
    Ok(stats)
}

/// Transactionally publish the stable complete root grouping plus the sparse
/// adaptive replacement overlay. Existing layered resources are retained by
/// exact `(key, layer, membership)` identity; the suppression mask changes only
/// after every replacement bucket has staged successfully.
fn upload_retained_batch_groups_transactionally(
    state: &mut MainState,
    force_upload: bool,
) -> Result<TransactionalBatchUploadStats, TransactionalBatchUploadFailure> {
    let MainState {
        renderer,
        batches,
        baseline_batch_groups,
        batch_staging,
        adaptive_picked,
        batch_layout_revision,
        num_faces,
        render_commands_dirty,
        last_visibility_mvp,
        active_suppressed_root_faces,
        ..
    } = state;
    let overlay = adaptive_picked
        .pending_overlay_for_publication(*batch_layout_revision)
        .map_err(|message| TransactionalBatchUploadFailure {
            message,
            missing_atlas_entries: 0,
            failures: 1,
        })?;
    let max_batch_size = baseline_batch_groups
        .values()
        .chain(overlay.groups.values())
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let stride = instance_layout::BATCH_TOPOLOGY_STRIDE;
    batch_staging.resize(max_batch_size * stride, 0.0);

    let mut previous_batches = std::mem::take(batches);
    let mut reused_batches = BTreeMap::<batch::RenderBatchId, GpuBatch>::new();
    let mut staged_batches = BTreeMap::<batch::RenderBatchId, GpuBatch>::new();
    let mut stats = TransactionalBatchUploadStats::default();
    let attempt = {
        let gl = renderer.gl();
        TESS_CACHE.with(|cache| -> Result<(), TransactionalBatchUploadFailure> {
            let cache = cache.borrow();
            for (layer, groups) in [
                (
                    batch::RenderBatchLayer::RetainedRoot,
                    &*baseline_batch_groups,
                ),
                (batch::RenderBatchLayer::AdaptiveOverlay, &overlay.groups),
            ] {
                for (&key, members) in groups {
                    let id = batch::RenderBatchId { key, layer };
                    let tess = cache.get(&key.lod).ok_or_else(|| {
                        TransactionalBatchUploadFailure {
                            message: format!(
                                "retained adaptive batch needs missing atlas patch {:?}",
                                key.lod,
                            ),
                            missing_atlas_entries: 1,
                            failures: 0,
                        }
                    })?;
                    let membership_changed = previous_batches
                        .get(&id)
                        .is_none_or(|batch| batch.members != *members);
                    if !force_upload && !membership_changed {
                        stats.retained += 1;
                        let previous = previous_batches.remove(&id).ok_or_else(|| {
                            TransactionalBatchUploadFailure {
                                message: format!(
                                    "retained adaptive batch {id:?} disappeared",
                                ),
                                missing_atlas_entries: 0,
                                failures: 1,
                            }
                        })?;
                        reused_batches.insert(id, previous);
                        continue;
                    }
                    let batch_floats = fill_batch_instance_data(
                        key,
                        members,
                        batch_staging,
                    )
                    .map_err(|error| TransactionalBatchUploadFailure {
                        message: format!("could not pack retained adaptive batch {id:?}: {error}"),
                        missing_atlas_entries: 0,
                        failures: 1,
                    })?;
                    let staged = create_gpu_batch(
                        gl,
                        tess,
                        key,
                        members,
                        &batch_staging[..batch_floats],
                    )
                    .map_err(|error| TransactionalBatchUploadFailure {
                        message: format!("could not stage retained adaptive batch {id:?}: {error}"),
                        missing_atlas_entries: 0,
                        failures: 1,
                    })?;
                    stats.reallocated += usize::from(previous_batches.contains_key(&id));
                    stats.created += 1;
                    stats.uploaded_instances += members.len();
                    staged_batches.insert(id, staged);
                }
            }
            Ok(())
        })
    };

    if let Err(failure) = attempt {
        let gl = renderer.gl();
        for (_, batch) in staged_batches {
            batch.destroy(gl);
        }
        previous_batches.append(&mut reused_batches);
        *batches = previous_batches;
        return Err(failure);
    }
    let changed_mask_bytes = match renderer
        .set_suppressed_source_faces(*num_faces, &overlay.suppressed_faces)
    {
        Ok(changed) => changed,
        Err(message) => {
            let gl = renderer.gl();
            for (_, batch) in staged_batches {
                batch.destroy(gl);
            }
            previous_batches.append(&mut reused_batches);
            *batches = previous_batches;
            return Err(TransactionalBatchUploadFailure {
                message: format!("could not stage retained-root suppression: {message}"),
                missing_atlas_entries: 0,
                failures: 1,
            });
        }
    };

    reused_batches.append(&mut staged_batches);
    stats.retired = previous_batches.len();
    let gl = renderer.gl();
    for (_, batch) in previous_batches {
        batch.destroy(gl);
    }
    *batches = reused_batches;
    active_suppressed_root_faces.clear();
    active_suppressed_root_faces.extend_from_slice(&overlay.suppressed_faces);
    if stats.created != 0 || stats.retired != 0 {
        *render_commands_dirty = true;
    }
    if changed_mask_bytes != 0 {
        *last_visibility_mvp = None;
    }
    Ok(stats)
}

fn mark_batch_layout_dirty(state: &mut MainState) {
    state.batch_layout_dirty = true;
    state.batch_layout_revision = state.batch_layout_revision.wrapping_add(1);
}

fn compare_incremental_root_group_shadow(
    state: &mut MainState,
    initial: batch::ResidentLod,
    force_rebuild: bool,
) {
    if !state.incremental_root_group_shadow.enabled {
        return;
    }
    let Some(topology) = state.lod_topology.as_ref() else {
        state.incremental_root_group_shadow.last_error =
            Some("incremental root grouping needs resident half-edge topology".into());
        return;
    };
    state.incremental_root_group_shadow.observe(
        topology,
        &state.resident_face_lods,
        &state.face_materials,
        &state.face_nodes,
        &state.face_render_nodes,
        initial,
        state.lod_balance_scratch.affected_component_faces(),
        force_rebuild,
        &state.batch_groups,
    );
}

/// Reconstruct the complete source-face grouping from the last coherent
/// classifier snapshot. Adaptive replanning always starts here so any CPU-side
/// rejection can publish a known crack-free root fallback rather than retain a
/// partially assembled frontier.
#[derive(Clone, Copy)]
enum LegacyBatchDestination {
    Live,
    Baseline,
}

fn rebuild_legacy_batch_groups(
    state: &mut MainState,
    initial: batch::ResidentLod,
    destination: LegacyBatchDestination,
) {
    let nf = state.num_faces;
    let render_node_started_ms = browser_now_ms();
    if state.batch_layout_dirty || state.face_render_nodes.len() != state.face_nodes.len() {
        rebuild_face_render_nodes(state);
    }
    state
        .batch_update_stats
        .render_node_ms
        .record(browser_now_ms() - render_node_started_ms);
    let vertex_lod_started_ms = browser_now_ms();
    if let Some(topology) = state.lod_topology.as_ref() {
        batch::rebuild_resident_vertex_lods(
            &state.resident_face_lods,
            topology,
            initial,
            &mut state.resident_vertex_lod_scratch,
            &mut state.resident_vertex_lods,
        );
    } else {
        state.resident_vertex_lods.resize(nf, [1; 3]);
        for (face, vertex_lods) in state.resident_vertex_lods.iter_mut().enumerate() {
            let edges = state.resident_face_lods[face]
                .unwrap_or(initial)
                .edge_lods();
            *vertex_lods = [
                edges[1].max(edges[2]),
                edges[0].max(edges[2]),
                edges[0].max(edges[1]),
            ];
        }
    }
    state
        .batch_update_stats
        .vertex_lod_ms
        .record(browser_now_ms() - vertex_lod_started_ms);
    let group_member_started_ms = browser_now_ms();
    let groups = match destination {
        LegacyBatchDestination::Live => &mut state.batch_groups,
        LegacyBatchDestination::Baseline => &mut state.baseline_batch_groups,
    };
    batch::group_resident_faces_into(
        &state.resident_face_lods,
        &state.resident_vertex_lods,
        &state.face_materials,
        &state.face_nodes,
        &state.face_render_nodes,
        initial,
        groups,
    );
    state
        .batch_update_stats
        .group_member_ms
        .record(browser_now_ms() - group_member_started_ms);
}

/// Reuse the complete root grouping across camera/animation-only adaptive
/// refreshes. A classifier topology change or any material/node/atlas layout
/// change advances one side of the signature and forces an exact rebuild.
fn ensure_baseline_batch_groups(
    state: &mut MainState,
    initial: batch::ResidentLod,
) -> bool {
    let signature = (state.root_topology_revision, state.batch_layout_revision);
    let retained_shape_is_complete = state.resident_vertex_lods.len() == state.num_faces
        && (state.num_faces == 0 || !state.baseline_batch_groups.is_empty());
    if state.baseline_group_signature == Some(signature) && retained_shape_is_complete {
        state.batch_update_stats.baseline_group_cache_hits = state
            .batch_update_stats
            .baseline_group_cache_hits
            .saturating_add(1);
        return true;
    }
    rebuild_legacy_batch_groups(state, initial, LegacyBatchDestination::Baseline);
    state.baseline_group_signature = Some(signature);
    state.batch_update_stats.baseline_group_cache_misses = state
        .batch_update_stats
        .baseline_group_cache_misses
        .saturating_add(1);
    false
}

/// Publish the desired adaptive-or-fallback grouping and update diagnostics as
/// one epoch. The upload routine preserves the exact prior GL map on failure;
/// adaptive diagnostics become active only after the replacement map exists.
#[derive(Clone, Copy)]
struct AdaptiveBatchPublication {
    published: bool,
    resources_changed: bool,
}

fn retained_gpu_publication_active(state: &MainState) -> bool {
    state
        .batches
        .keys()
        .any(|id| id.layer != batch::RenderBatchLayer::Complete)
}

fn retained_frontier_order_safe(state: &MainState) -> bool {
    let default_material = PbrMaterial::default();
    state.batch_groups.keys().all(|key| {
        let (_, material) = pbr_material_for_index(
            &state.materials,
            &default_material,
            key.material_index,
        );
        pbr_draw_class(material) == PbrDrawClass::Opaque
    })
}

fn retained_publication_desired(state: &MainState) -> bool {
    state.adaptive_retained_publication_enabled
        && state.adaptive_picked.has_pending_plan()
        && retained_frontier_order_safe(state)
}

fn publish_adaptive_batch_groups(
    state: &mut MainState,
    force_upload: bool,
    previous_batch_groups: Option<
        BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    >,
) -> AdaptiveBatchPublication {
    let retained_requested = state.adaptive_retained_publication_enabled
        && state.adaptive_picked.has_pending_plan();
    let retained_order_safe = !retained_requested || retained_frontier_order_safe(state);
    let retained_publication = retained_requested && retained_order_safe;
    if retained_requested && !retained_order_safe {
        state.adaptive_retained_publication_error = Some(
            "retained publication requires every active PBR bucket to be opaque".into(),
        );
    }
    let upload = if retained_publication {
        upload_retained_batch_groups_transactionally(state, force_upload)
    } else {
        upload_batch_groups_transactionally(state, force_upload)
    };
    match upload {
        Ok(stats) => {
            if retained_publication {
                state.adaptive_retained_publication_error = None;
            } else if !retained_requested {
                state.adaptive_retained_publication_error = None;
            }
            state
                .adaptive_picked
                .commit_gpu_publication(retained_publication);
            if let Some(previous_batch_groups) = previous_batch_groups {
                state
                    .adaptive_picked
                    .recycle_group_scratch(previous_batch_groups);
            }
            state.batch_update_stats.retained_buckets += stats.retained as u64;
            state.batch_update_stats.created_buckets += stats.created as u64;
            state.batch_update_stats.reallocated_buckets += stats.reallocated as u64;
            state.batch_update_stats.retired_buckets += stats.retired as u64;
            state.batch_update_stats.uploaded_instances += stats.uploaded_instances as u64;
            state.batch_update_stats.last_missing_atlas_entries = 0;
            state.batch_update_stats.last_gpu_failures = 0;
            state.adaptive_batch_transition_pending = false;
            state.batch_layout_dirty = false;
            debug!(
                "Transactionally published {} adaptive GPU batches in {} mode ({} retained, {} staged, {} replacements, {} retired)",
                state.batches.len(),
                if retained_publication { "layered" } else { "complete" },
                stats.retained, stats.created,
                stats.reallocated, stats.retired,
            );
            AdaptiveBatchPublication {
                published: true,
                resources_changed: stats.created != 0 || stats.retired != 0,
            }
        }
        Err(failure) => {
            if retained_publication {
                state.adaptive_retained_publication_error = Some(failure.message.clone());
            }
            warn!(
                "Adaptive GPU batch publication rolled back: {}",
                failure.message,
            );
            state.batch_update_stats.last_missing_atlas_entries =
                failure.missing_atlas_entries as u64;
            state.batch_update_stats.last_gpu_failures = failure.failures as u64;
            if state.adaptive_picked.has_pending_publication() {
                state
                    .adaptive_picked
                    .record_publication_failure(failure.message.clone());
            }
            // The GL transaction restored the previous GPU epoch. Restore its
            // exact CPU membership map as well; a desired candidate is not
            // authoritative until every corresponding resource has published.
            if let Some(previous_batch_groups) = previous_batch_groups {
                state.batch_groups = previous_batch_groups;
                mark_batch_layout_dirty(state);
            }
            AdaptiveBatchPublication {
                published: false,
                resources_changed: false,
            }
        }
    }
}

fn commit_reused_adaptive_groups(state: &mut MainState) {
    let retained_publication = retained_gpu_publication_active(state);
    state
        .adaptive_picked
        .commit_gpu_publication(retained_publication);
    state.adaptive_batch_transition_pending = false;
    state.batch_layout_dirty = false;
    state.batch_update_stats.adaptive_refresh_noops = state
        .batch_update_stats
        .adaptive_refresh_noops
        .saturating_add(1);
    state.batch_update_stats.last_missing_atlas_entries = 0;
    state.batch_update_stats.last_gpu_failures = 0;
}

struct AdaptiveGroupPreparation {
    reused_published: bool,
    previous_groups: Option<
        BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    >,
}

/// Put the complete root classification in the live slot and return the exact
/// formerly live complete epoch for transactional GPU rollback. The baseline
/// scratch is empty afterward and can be rebuilt without aliasing either map.
fn stage_baseline_batch_groups(
    live: &mut BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
    baseline: &mut BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>>,
) -> BTreeMap<batch::RenderBatchKey, Vec<batch::RenderBatchMember>> {
    std::mem::swap(live, baseline);
    std::mem::take(baseline)
}

fn prepare_adaptive_batch_groups(
    state: &mut MainState,
    initial: batch::ResidentLod,
) -> AdaptiveGroupPreparation {
    let enabled = state.adaptive_picked.is_enabled();

    // Keep the latest complete root classification independent of the live
    // complete adaptive frontier. Planning may swap a candidate into the live
    // map, but neither root-shadow observation nor fallback construction may
    // overwrite the exact CPU epoch corresponding to the current GPU map.
    ensure_baseline_batch_groups(state, initial);
    if state.adaptive_root_shadow.is_enabled() {
        compare_adaptive_root_shadow(
            state,
            initial,
            LegacyBatchDestination::Baseline,
        );
    }

    if enabled {
        if apply_adaptive_screen_plan(state, initial, true, None) {
            let retained_desired = retained_publication_desired(state);
            if state.adaptive_retained_publication_enabled {
                state.adaptive_retained_publication_error = if retained_desired {
                    None
                } else {
                    Some(
                        "retained publication requires every active PBR bucket to be opaque"
                            .into(),
                    )
                };
            }
            if retained_gpu_publication_active(state) == retained_desired {
                return AdaptiveGroupPreparation {
                    reused_published: true,
                    previous_groups: None,
                };
            }
            // Complete membership is unchanged, but the physical publication
            // mode is not. Upload the alternate resource layer without a CPU
            // grouping rollback map; failure leaves this live map untouched.
            return AdaptiveGroupPreparation {
                reused_published: false,
                previous_groups: None,
            };
        }
        if state.adaptive_picked.has_pending_plan() {
            return AdaptiveGroupPreparation {
                reused_published: false,
                previous_groups: Some(state.adaptive_picked.take_group_rollback()),
            };
        }
    }

    // Planning was unavailable or rejected, or adaptive mode is being
    // disabled. Stage the independently built roots while retaining the exact
    // formerly live complete map for transactional GPU rollback.
    let previous_groups = stage_baseline_batch_groups(
        &mut state.batch_groups,
        &mut state.baseline_batch_groups,
    );
    AdaptiveGroupPreparation {
        reused_published: false,
        previous_groups: Some(previous_groups),
    }
}

/// Re-evaluate an enabled picked frontier against the latest retained root
/// classification, rendered camera, and accepted animation pose. This covers
/// sparse worker completions whose source-face LOD delta is empty: the root
/// request may be unchanged even though the within-patch screen metric moved.
fn refresh_adaptive_picked_batches(state: &mut MainState) -> bool {
    if !state.adaptive_picked.is_enabled() && !state.adaptive_batch_transition_pending {
        return false;
    }
    let nf = state.num_faces;
    if state.requested_face_lods.len() != nf
        || state.resident_face_lods.len() != nf
        || state.face_nodes.len() != nf
    {
        state.adaptive_picked.record_refresh_failure(
            "retained source classification is incomplete for adaptive refresh",
        );
        return false;
    }
    state.batch_update_stats.calls = state.batch_update_stats.calls.saturating_add(1);
    state.batch_update_stats.adaptive_refresh_calls =
        state.batch_update_stats.adaptive_refresh_calls.saturating_add(1);
    let total_started_ms = browser_now_ms();
    state.batch_update_stats.retain_ms.record(0.0);
    state.batch_update_stats.balance_ms.record(0.0);
    perf_mark("batch-group-start");
    perf_mark("batch-retain-start");
    perf_mark("batch-retain-end");
    perf_measure("batch-retain", "batch-retain-start", "batch-retain-end");
    perf_mark("batch-balance-start");
    perf_mark("batch-balance-end");
    perf_measure(
        "batch-balance",
        "batch-balance-start",
        "batch-balance-end",
    );
    perf_mark("batch-bucket-start");
    let bucket_started_ms = browser_now_ms();
    let initial = bounded_standby_resident_lod();
    let prepared = prepare_adaptive_batch_groups(state, initial);
    state
        .batch_update_stats
        .bucket_ms
        .record(browser_now_ms() - bucket_started_ms);
    perf_mark("batch-bucket-end");
    perf_measure("batch-bucket", "batch-bucket-start", "batch-bucket-end");
    perf_mark("batch-group-end");
    perf_measure("batch-group", "batch-group-start", "batch-group-end");
    state.adaptive_batch_transition_pending = true;
    perf_mark("batch-upload-start");
    let upload_started_ms = browser_now_ms();
    if prepared.reused_published {
        commit_reused_adaptive_groups(state);
        finish_batch_update_timing(
            &mut state.batch_update_stats,
            total_started_ms,
            Some(upload_started_ms),
        );
        perf_mark("batch-upload-end");
        perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");
        return true;
    }
    let publication = publish_adaptive_batch_groups(
        state,
        false,
        prepared.previous_groups,
    );
    if publication.published && !publication.resources_changed {
        state.batch_update_stats.adaptive_refresh_noops = state
            .batch_update_stats
            .adaptive_refresh_noops
            .saturating_add(1);
    }
    finish_batch_update_timing(
        &mut state.batch_update_stats,
        total_started_ms,
        Some(upload_started_ms),
    );
    perf_mark("batch-upload-end");
    perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");
    publication.published
}

#[wasm_bindgen(js_name = "mr_refreshAdaptivePicked")]
pub fn mr_refresh_adaptive_picked() -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return JsValue::NULL;
        };
        let published = refresh_adaptive_picked_batches(state);
        adaptive_picked_snapshot_js(state, Some(published))
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AdaptiveComponentClosureMeasurement {
    ok: bool,
    selected_faces: Vec<u32>,
    source_faces: u64,
    component_closure_faces: u64,
    unaffected_faces: u64,
    source_scan_reduction_percent: f64,
    closure_face_sample: Vec<u32>,
    elapsed_ms: f64,
}

/// Measure the exact welded-vertex source components touched by the currently
/// published adaptive selection. This explicit diagnostic neither replans nor
/// mutates renderer state.
#[wasm_bindgen(js_name = "mr_measureAdaptiveComponentClosure")]
pub fn mr_measure_adaptive_component_closure(max_faces: u32) -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return adaptive_screen_diagnostic_error("renderer is not initialized");
        };
        if state.adaptive_batch_transition_pending {
            return adaptive_screen_diagnostic_error(
                "adaptive batch transition is awaiting publication",
            );
        }
        let selected_faces = state.adaptive_picked.published_faces();
        if selected_faces.is_empty() {
            return adaptive_screen_diagnostic_error(
                "adaptive component closure has no published face selection",
            );
        }
        let Some(topology) = state.screen_topology_cache.as_ref() else {
            return adaptive_screen_diagnostic_error("adaptive source topology is unavailable");
        };
        let started = browser_now_ms();
        let mut closure = Vec::new();
        if let Err(error) = topology.collect_component_closure(
            selected_faces,
            max_faces as usize,
            &mut closure,
        ) {
            return adaptive_screen_diagnostic_error(error.to_string());
        }
        let source_faces = state.num_faces;
        let unaffected_faces = source_faces.saturating_sub(closure.len());
        let source_scan_reduction_percent = if source_faces == 0 {
            0.0
        } else {
            100.0 * unaffected_faces as f64 / source_faces as f64
        };
        adaptive_browser_value(&AdaptiveComponentClosureMeasurement {
            ok: true,
            selected_faces: selected_faces.to_vec(),
            source_faces: source_faces as u64,
            component_closure_faces: closure.len() as u64,
            unaffected_faces: unaffected_faces as u64,
            source_scan_reduction_percent,
            closure_face_sample: closure.into_iter().take(16).collect(),
            elapsed_ms: browser_now_ms() - started,
        })
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AdaptiveFaceNeighborhoodMeasurement {
    ok: bool,
    selected_faces: Vec<u32>,
    source_faces: u64,
    edge_hops: u32,
    reconciliation_faces: u64,
    observed_faces: u64,
    unaffected_faces: u64,
    source_scan_reduction_percent: f64,
    reconciliation_face_sample: Vec<u32>,
    observed_face_sample: Vec<u32>,
    elapsed_ms: f64,
}

/// Measure a candidate local reconciliation neighborhood and its one-ring
/// corner-density observers. This is a read-only sizing diagnostic, not a
/// publication certificate or a replacement for the complete oracle.
#[wasm_bindgen(js_name = "mr_measureAdaptiveFaceNeighborhood")]
pub fn mr_measure_adaptive_face_neighborhood(edge_hops: u32, max_faces: u32) -> JsValue {
    STATE.with(|state| {
        let state = state.borrow();
        let Some(state) = state.as_ref() else {
            return adaptive_screen_diagnostic_error("renderer is not initialized");
        };
        if state.adaptive_batch_transition_pending {
            return adaptive_screen_diagnostic_error(
                "adaptive batch transition is awaiting publication",
            );
        }
        let selected_faces = state.adaptive_picked.published_faces();
        if selected_faces.is_empty() {
            return adaptive_screen_diagnostic_error(
                "adaptive face neighborhood has no published face selection",
            );
        }
        let Some(topology) = state.screen_topology_cache.as_ref() else {
            return adaptive_screen_diagnostic_error("adaptive source topology is unavailable");
        };
        let started = browser_now_ms();
        let mut reconciliation = Vec::new();
        let mut observed = Vec::new();
        let mut scratch = ScreenMeshNeighborhoodScratch::default();
        if let Err(error) = topology.collect_face_neighborhood(
            selected_faces,
            edge_hops,
            max_faces as usize,
            &mut reconciliation,
            &mut observed,
            &mut scratch,
        ) {
            return adaptive_screen_diagnostic_error(error.to_string());
        }
        let source_faces = state.num_faces;
        let unaffected_faces = source_faces.saturating_sub(observed.len());
        let source_scan_reduction_percent = if source_faces == 0 {
            0.0
        } else {
            100.0 * unaffected_faces as f64 / source_faces as f64
        };
        adaptive_browser_value(&AdaptiveFaceNeighborhoodMeasurement {
            ok: true,
            selected_faces: selected_faces.to_vec(),
            source_faces: source_faces as u64,
            edge_hops,
            reconciliation_faces: reconciliation.len() as u64,
            observed_faces: observed.len() as u64,
            unaffected_faces: unaffected_faces as u64,
            source_scan_reduction_percent,
            reconciliation_face_sample: reconciliation.into_iter().take(16).collect(),
            observed_face_sample: observed.into_iter().take(16).collect(),
            elapsed_ms: browser_now_ms() - started,
        })
    })
}

/// Measure the sparse replacement closure for the exact currently published
/// adaptive epoch without changing draw resources or scheduling frame work.
#[wasm_bindgen(js_name = "mr_measureAdaptiveOverlay")]
pub fn mr_measure_adaptive_overlay() -> JsValue {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return adaptive_screen_diagnostic_error("renderer is not initialized");
        };
        if state.adaptive_batch_transition_pending {
            return adaptive_screen_diagnostic_error(
                "adaptive batch transition is awaiting publication",
            );
        }
        let initial = bounded_standby_resident_lod();
        let MainState {
            batch_groups,
            resident_face_lods,
            resident_vertex_lods,
            adaptive_picked,
            batch_layout_revision,
            face_materials,
            face_nodes,
            face_render_nodes,
            ..
        } = state;
        match adaptive_picked.measure_published_overlay(
            *batch_layout_revision,
            batch_groups,
            resident_face_lods,
            resident_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
        ) {
            Ok(measurement) => adaptive_browser_value(&measurement),
            Err(error) => adaptive_screen_diagnostic_error(error),
        }
    })
}

#[wasm_bindgen(js_name = "mr_buildBatches")]
pub fn mr_build_batches(face_lods: &[f32]) {
    update_batches(face_lods, None);
}

/// Admit one worker publication before any of its records reach resident
/// renderer state. Sparse results must extend the exact accepted base; a
/// revision-one full snapshot explicitly resets the stream after model changes,
/// worker recovery, or a future backend cutover.
#[wasm_bindgen(js_name = "mr_acceptLodDeltaSequence")]
pub fn mr_accept_lod_delta_sequence(
    epoch: u32,
    base_revision: u32,
    revision: u32,
    full_snapshot: bool,
) -> Result<bool, JsValue> {
    LOD_DELTA_CURSOR.with(|cursor| {
        cursor.borrow_mut().accept(
            batch::FaceLodDeltaSequence {
                epoch,
                base_revision,
                revision,
            },
            full_snapshot,
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))
    })
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
        update_batches_in_state(
            state,
            RetainedLodPublication::LegacyFloats(face_lods),
            face_indices,
        );
    });
}

#[derive(Clone, Copy)]
enum RetainedLodPublication<'a> {
    LegacyFloats(&'a [f32]),
    PackedWords(&'a [u32]),
}

fn update_batches_in_state(
    state: &mut MainState,
    publication_payload: RetainedLodPublication<'_>,
    face_indices: Option<&[u32]>,
) {
        let total_started_ms = browser_now_ms();
        state.batch_update_stats.calls += 1;

        // Phase 1: bucket sort to get face groupings (fast O(n), no instance data copy)
        perf_mark("batch-group-start");
        perf_mark("batch-retain-start");
        let retain_started_ms = browser_now_ms();
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
        // Keep conservatively offscreen faces at a small drawable standby LOD.
        // The current-pose render GPU can still resurrect one during an
        // asynchronous camera/animation frame, while avoiding the permanent
        // vertex cost and high-detail flash of arbitrary old topology.
        let initial_resident = bounded_standby_resident_lod();
        // A picked plan is view-dependent even when the worker's raw root
        // requests are unchanged. A pending enable/disable also needs one
        // publication even after the config itself has been cleared.
        let mut topology_changed = state.batch_layout_dirty
            || state.adaptive_picked.is_enabled()
            || state.adaptive_batch_transition_pending;
        let buffers = batch::FaceLodAdmissionBuffers {
            requested: &mut state.requested_face_lods,
            resident: &mut state.resident_face_lods,
            visible: &mut state.classified_face_visibility,
            dirty_faces: &mut state.lod_dirty_faces,
        };
        let publication = match publication_payload {
            RetainedLodPublication::LegacyFloats(face_lods) => {
                batch::admit_face_lod_publication(
                    face_lods,
                    face_indices,
                    nf,
                    initial_resident,
                    state.classified_culled_faces,
                    buffers,
                )
            }
            RetainedLodPublication::PackedWords(words) => {
                batch::admit_face_lod_classification_publication(
                    words.len(),
                    face_indices,
                    nf,
                    state.classified_culled_faces,
                    buffers,
                    |record_index| {
                        let fields = unpack_lod_classification_fields(words[record_index])
                            .expect("packed classifier words were validated at readback");
                        fields.into_face_lod_classification()
                    },
                )
            }
        };
        state
            .batch_update_stats
            .retain_ms
            .record(browser_now_ms() - retain_started_ms);
        perf_mark("batch-retain-end");
        perf_measure("batch-retain", "batch-retain-start", "batch-retain-end");
        let publication = match publication {
            Ok(publication) => publication,
            Err(error) => {
                warn!("Rejected malformed LOD publication: {error}");
                state.batch_update_stats.balance_ms.record(0.0);
                state.batch_update_stats.bucket_ms.record(0.0);
                finish_batch_update_timing(
                    &mut state.batch_update_stats,
                    total_started_ms,
                    None,
                );
                perf_mark("batch-group-end");
                perf_measure("batch-group", "batch-group-start", "batch-group-end");
                return;
            }
        };
        let culled = publication.culled_faces;
        state.classified_culled_faces = culled;
        let admitted_topology_changed = publication.topology_changed();
        topology_changed |= admitted_topology_changed;

        perf_mark("batch-balance-start");
        let balance_started_ms = browser_now_ms();
        let lod_corrections = if let Some(topology) = state.lod_topology.as_ref() {
            batch::reconcile_resident_lods_from_requests_with_grading(
                &state.requested_face_lods,
                &mut state.resident_face_lods,
                topology,
                &state.lod_dirty_faces,
                &mut state.lod_balance_scratch,
                state.lod_grading,
            )
        } else {
            for &face in &state.lod_dirty_faces {
                state.resident_face_lods[face] = state.requested_face_lods[face];
            }
            0
        };
        state
            .batch_update_stats
            .balance_ms
            .record(browser_now_ms() - balance_started_ms);
        perf_mark("batch-balance-end");
        perf_measure(
            "batch-balance",
            "batch-balance-start",
            "batch-balance-end",
        );
        let root_topology_changed = admitted_topology_changed || lod_corrections != 0;
        topology_changed |= root_topology_changed;
        if root_topology_changed {
            state.root_topology_revision = state.root_topology_revision.wrapping_add(1);
            state.baseline_group_signature = None;
        }
        state.batch_update_stats.last_culled_faces = culled as u64;
        state.batch_update_stats.last_lod_corrections = lod_corrections as u64;
        state.batch_update_stats.last_missing_atlas_entries = 0;
        state.batch_update_stats.last_gpu_failures = 0;

        if !topology_changed {
            state.batch_update_stats.unchanged_calls += 1;
            state.batch_update_stats.bucket_ms.record(0.0);
            finish_batch_update_timing(
                &mut state.batch_update_stats,
                total_started_ms,
                None,
            );
            perf_mark("batch-bucket-start");
            perf_mark("batch-bucket-end");
            perf_measure("batch-bucket", "batch-bucket-start", "batch-bucket-end");
            perf_mark("batch-group-end");
            perf_measure("batch-group", "batch-group-start", "batch-group-end");
            return;
        }

        let force_upload = state.batch_layout_dirty;
        let transactional_upload =
            state.adaptive_picked.is_enabled() || state.adaptive_batch_transition_pending;
        perf_mark("batch-bucket-start");
        let bucket_started_ms = browser_now_ms();
        let adaptive_preparation = if transactional_upload {
            Some(prepare_adaptive_batch_groups(state, initial_resident))
        } else {
            rebuild_legacy_batch_groups(
                state,
                initial_resident,
                LegacyBatchDestination::Live,
            );
            compare_incremental_root_group_shadow(
                state,
                initial_resident,
                force_upload,
            );
            if state.adaptive_root_shadow.is_enabled() {
                compare_adaptive_root_shadow(
                    state,
                    initial_resident,
                    LegacyBatchDestination::Live,
                );
            }
            None
        };
        state
            .batch_update_stats
            .bucket_ms
            .record(browser_now_ms() - bucket_started_ms);
        perf_mark("batch-bucket-end");
        perf_measure("batch-bucket", "batch-bucket-start", "batch-bucket-end");
        perf_mark("batch-group-end");
        perf_measure("batch-group", "batch-group-start", "batch-group-end");

        // Phase 2: retain batches whose key and ordered (face, permutation)
        // membership remain valid. Only changed buckets repack or upload their
        // source streams; capacity growth rebuilds that bucket alone.
        perf_mark("batch-upload-start");
        let upload_started_ms = browser_now_ms();
        if adaptive_preparation
            .as_ref()
            .is_some_and(|prepared| prepared.reused_published)
        {
            commit_reused_adaptive_groups(state);
            finish_batch_update_timing(
                &mut state.batch_update_stats,
                total_started_ms,
                Some(upload_started_ms),
            );
            perf_mark("batch-upload-end");
            perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");
            return;
        }
        if transactional_upload {
            let previous_batch_groups = adaptive_preparation
                .and_then(|prepared| prepared.previous_groups);
            let _ = publish_adaptive_batch_groups(
                state,
                force_upload,
                previous_batch_groups,
            );
            finish_batch_update_timing(
                &mut state.batch_update_stats,
                total_started_ms,
                Some(upload_started_ms),
            );
            perf_mark("batch-upload-end");
            perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");
            return;
        }
        let gl = state.renderer.gl();
        let mut previous_batches = std::mem::take(&mut state.batches);
        let mut next_batches = BTreeMap::<batch::RenderBatchId, GpuBatch>::new();
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
                let id = batch::RenderBatchId::complete(key);
                let tess = match tc.get(&key.lod) {
                    Some(t) => t,
                    None => { missing += 1; continue; }
                };
                let previous = previous_batches.remove(&id);
                let membership_changed = previous.as_ref()
                    .is_none_or(|batch| batch.members != *members);
                if !force_upload && !membership_changed {
                    retained += 1;
                    next_batches.insert(id, previous.unwrap());
                    continue;
                }

                let batch_floats = match fill_batch_instance_data(
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
                                next_batches.insert(id, gpu_batch);
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
                        next_batches.insert(id, gpu_batch);
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
        finish_batch_update_timing(
            &mut state.batch_update_stats,
            total_started_ms,
            Some(upload_started_ms),
        );
        perf_mark("batch-upload-end");
        perf_measure("batch-gpu-upload", "batch-upload-start", "batch-upload-end");

        debug!(
            "Updated {} GPU batches ({} unchanged, {} uploaded in place, {} created, {} capacity reallocations, {} retired; {} CPU-culled faces, {} LOD corrections, {} atlas entries missing, {} failures)",
            state.batches.len(), retained, updated, created, reallocated, retired,
            culled, lod_corrections, missing, failed,
        );
        state.batch_layout_dirty = missing != 0 || failed != 0;
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
        st.patch_prepare_dirty = true;
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
            clear_same_context_lod(state);
            state.renderer.joint_ubo().clear(state.renderer.gl());
            state.renderer.clear_animation_textures();
            state.surface_runtime.clear_animation();
            state.patch_prepare_dirty = true;
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
        let render_started_ms = browser_now_ms();
        #[cfg(feature = "webgpu-backend")]
        if state.backend_evidence_requested
            && ((state.render_style != RenderStyle::Pbr
                && !quilting_webgpu::supports_patch_presentation_style(state.render_style))
                || !backend_frame_evidence_supports_composition(state.focus_postprocess))
        {
            state.backend_evidence_requested = false;
            state.backend_evidence_error = Some(
                "backend frame evidence became unsupported before its requested render"
                    .to_string(),
            );
        }

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
        #[cfg(feature = "webgpu-backend")]
        let backend_plan_required = webgpu_frame_requested(state);
        #[cfg(feature = "webgpu-backend")]
        let backend_scene_required = backend_plan_required;
        #[cfg(not(feature = "webgpu-backend"))]
        let backend_plan_required = false;
        #[cfg(not(feature = "webgpu-backend"))]
        let backend_scene_required = false;
        let _scene_refresh = refresh_validated_render_scene(state, backend_scene_required);
        #[cfg(feature = "webgpu-backend")]
        if state.backend_evidence_requested && state.render_style == RenderStyle::Pbr {
            let options = RenderFrameOptions {
                focus_postprocess: state.focus_postprocess,
                highlight_face: u32::try_from(state.highlight_face).ok(),
                matcap_style: state.matcap_style,
            };
            let supported = state.validated_render_scene.as_ref().is_some_and(|scene| {
                if options.focus_postprocess.is_some() {
                    quilting_webgpu::supports_focus_pbr_frame(scene.snapshot(), options)
                } else {
                    quilting_webgpu::supports_basic_pbr_frame(scene.snapshot(), options)
                }
            });
            if !supported {
                state.backend_evidence_requested = false;
                state.backend_evidence_error = Some(
                    "PBR backend evidence became unsupported before its requested render"
                        .to_string(),
                );
            }
        }
        refresh_render_command_plan(state, backend_plan_required);
        let render_frame = match current_render_frame(state, &camera) {
            Ok(frame) => frame,
            Err(error) => {
                state.render_shadow.record_extraction_error(&error);
                #[cfg(feature = "webgpu-backend")]
                if backend_plan_required {
                    crate::webgpu_backend::record_frame_prerequisite_failure(error);
                }
                None
            }
        };
        let prepared_render_revision =
            prepare_render_shadow_observation(state, render_frame.as_ref());

        // The explicitly selected WebGPU backend submits before any WebGL
        // frame work. A valid presentation frame needs none of the
        // incumbent's prepared or camera-visibility buffers: `mr_pick`
        // refreshes both for its exact pick camera, while a later fallback or
        // mode switch observes the deliberately retained dirty stamps below.
        #[cfg(feature = "webgpu-backend")]
        match submit_webgpu_frame(state, render_frame.as_ref()) {
            crate::webgpu_backend::LiveFrameDisposition::PresentationSubmitted(
                submission_stats,
            ) => {
                state.webgpu_presentation_patch_frames = state
                    .webgpu_presentation_patch_frames
                    .saturating_add(1);
                observe_render_submission(
                    state,
                    prepared_render_revision,
                    render_frame.as_ref(),
                    submission_stats,
                );
                state.last_render_submission = submission_stats;
                state.render_submission_totals.merge(submission_stats);
                state
                    .render_timing
                    .observe(render_started_ms, browser_now_ms());
                return;
            }
            crate::webgpu_backend::LiveFrameDisposition::PresentationRetained => {
                state.webgpu_retained_presentation_frames = state
                    .webgpu_retained_presentation_frames
                    .saturating_add(1);
                state.last_render_submission = RenderSubmissionStats::default();
                state
                    .render_timing
                    .observe(render_started_ms, browser_now_ms());
                return;
            }
            crate::webgpu_backend::LiveFrameDisposition::ShadowSubmitted(_) => {}
            crate::webgpu_backend::LiveFrameDisposition::IncumbentRequired => {
                if state.backend_evidence_requested {
                    state.backend_evidence_requested = false;
                    state.backend_evidence_capture = None;
                    state.backend_evidence_error = Some(
                        "backend frame evidence was not admitted by the shadow renderer"
                            .to_string(),
                    );
                }
            }
        }

        let prepare_all = state.patch_prepare_dirty;
        if prepare_all {
            // Hidden batches cannot dispatch through a zero-instance render
            // view. Retain their local dirty bit so becoming visible later
            // prepares the newest global animation/source pose.
            for batch in state.batches.values_mut() {
                batch.pose_dirty = true;
            }
        }
        let (prepare_calls, prepared_instances) = state
            .batches
            .values()
            .zip(&state.render_batches)
            .filter(|(batch, render_batch)| {
                render_batch.mesh.num_instances > 0
                    && patch_preparation_needed(
                        prepare_all,
                        batch.pose_dirty,
                        batch.last_prepared_model,
                        render_batch.euclidean_model,
                    )
            })
            .fold((0_u64, 0_u64), |(calls, instances), (_, render_batch)| {
                (
                    calls.saturating_add(1),
                    instances.saturating_add(render_batch.mesh.num_instances as u64),
                )
            });
        let prepare_needed = prepare_calls != 0;
        let visibility_needed = patch_visibility_needed(
            prepare_needed,
            state.last_visibility_mvp,
            state.last_visibility_command_build,
            camera.mvp,
            state.render_command_builds,
        );
        if prepare_needed {
            state.patch_prepare_frames = state.patch_prepare_frames.saturating_add(1);
            state.patch_prepare_calls = state.patch_prepare_calls.saturating_add(prepare_calls);
            state.last_prepared_patch_instances = prepared_instances;
        } else {
            state.skipped_patch_prepare_frames =
                state.skipped_patch_prepare_frames.saturating_add(1);
            state.last_prepared_patch_instances = 0;
        }
        if visibility_needed {
            state.last_visibility_mvp = Some(camera.mvp);
            state.last_visibility_command_build = state.render_command_builds;
            state.patch_visibility_frames = state.patch_visibility_frames.saturating_add(1);
            state.patch_visibility_calls = state
                .patch_visibility_calls
                .saturating_add(state.batches.len() as u64);
            state.last_visibility_patch_instances = state
                .render_batches
                .iter()
                .map(|batch| batch.mesh.num_instances.max(0) as u64)
                .sum();
        } else {
            state.skipped_patch_visibility_frames =
                state.skipped_patch_visibility_frames.saturating_add(1);
            state.last_visibility_patch_instances = 0;
        }
        let use_resolved_pbr_plan = state.render_style == RenderStyle::Pbr
            && prepared_render_revision.is_some()
            && state.webgl_pbr_command_plan.is_some();
        let has_transmission = if use_resolved_pbr_plan {
            state
                .webgl_pbr_command_plan
                .as_ref()
                .is_some_and(WebGlPbrCommandPlan::builds_transmission_pyramid)
        } else {
            state.render_style == RenderStyle::Pbr
                && state
                    .render_batches
                    .iter()
                    .any(|batch| batch.pbr_class == PbrDrawClass::Transmission)
        };
        let focus_postprocess_requested = if use_resolved_pbr_plan {
            state
                .webgl_pbr_command_plan
                .as_ref()
                .is_some_and(WebGlPbrCommandPlan::runs_focus_postprocess)
        } else {
            state.fuzzy_enabled
        };
        if has_transmission && state.blur_program.is_none() {
            match resolve_auxiliary_program(state, AuxiliaryProgram::Blur) {
                Ok(program) => state.blur_program = Some(program),
                Err(error) => {
                    warn!("Transmission blur unavailable; rendering continues: {error}");
                }
            }
        }
        let resolved_pbr_plan = use_resolved_pbr_plan.then(|| {
            state
                .webgl_pbr_command_plan
                .as_ref()
                .expect("resolved PBR plan availability was checked above")
        });
        let render_batches = state.render_batches.as_slice();
        let gl = state.renderer.gl();
        state.renderer.begin_frame();
        state.renderer.joint_ubo().bind(gl);
        state.renderer.bind_vertex_textures();

        // Deform each patch only when its pose/topology changes. Camera motion
        // updates the separate one-float visibility stream below instead of
        // rewriting this complete 52-float record.
        if prepare_needed {
            let renderer = &state.renderer;
            for (gpu_batch, render_batch) in state.batches.values_mut().zip(render_batches) {
                if render_batch.mesh.num_instances > 0
                    && patch_preparation_needed(
                        prepare_all,
                        gpu_batch.pose_dirty,
                        gpu_batch.last_prepared_model,
                        render_batch.euclidean_model,
                    )
                {
                    renderer.prepare_patch_batch(
                        &camera,
                        render_batch,
                        gpu_batch.prepare_vao,
                        gpu_batch.instances.prepared_buf,
                        0,
                    );
                    gpu_batch.pose_dirty = false;
                    gpu_batch.last_prepared_model = Some(render_batch.euclidean_model);
                }
            }
        }
        state.patch_prepare_dirty = false;
        if visibility_needed {
            for (gpu_batch, render_batch) in state.batches.values().zip(render_batches) {
                state.renderer.classify_patch_batch(
                    &camera,
                    render_batch,
                    gpu_batch.visibility_vao,
                    gpu_batch.instances.visibility_buf,
                    0,
                );
            }
        }

        // Mode-specific UBO setup
        let has_env = state.env_maps.prefiltered.is_some();
        let env_mips = state.env_maps.mip_count;
        let white = state.texture_cache.placeholder();
        let black = state.texture_cache.placeholder_black();
        let cube = state.texture_cache.placeholder_cube();

        match state.render_style {
            RenderStyle::Pbr => {
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
                if focus_postprocess_requested && true /* all fuzzy modes use MRT */ {
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
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

                            // Weight texture (attachment 1)
                            let weight_tex = gl.create_texture().unwrap();
                            gl.bind_texture(glow::TEXTURE_2D, Some(weight_tex));
                            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA16F as i32, vw, vh, 0,
                                glow::RGBA, glow::HALF_FLOAT, glow::PixelUnpackData::Slice(None));
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

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
                        let [red, green, blue, alpha] = DEFAULT_RENDER_CLEAR_COLOR;
                        gl.clear_color(red, green, blue, alpha);
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

                // PBR: two-pass rendering — opaque first, then transparent.
                // Opaque materials can still emit fractional alpha for the
                // conformal horizon fade. Accumulate that alpha with standard
                // source-over factors, matching the WebGPU PBR pipeline,
                // rather than applying the color factors to alpha as well.
                unsafe {
                    gl.use_program(Some(state.renderer.programs().pbr));
                    gl.enable(glow::BLEND);
                    gl.blend_func_separate(
                        glow::SRC_ALPHA,
                        glow::ONE_MINUS_SRC_ALPHA,
                        glow::ONE,
                        glow::ONE_MINUS_SRC_ALPHA,
                    );
                }
                let default_mat = PbrMaterial::default();
                let texture_defaults = [white, black, black, black, white];
                let mut submission_stats = RenderSubmissionStats::default();
                let mut material_updates = 0u64;
                let mut vertex_uniform_updates = 0u64;
                let mut active_material: Option<(usize, i32)> = None;
                let mut active_vertex_state: Option<&RenderBatch> = None;

                // Pass 1: opaque non-transmission
                for (batch_index, batch) in webgl_pbr_draws(
                    resolved_pbr_plan,
                    render_batches,
                    RenderPass::PbrOpaque,
                ) {
                    let (material_slot, mat) = pbr_material_for_index(
                        &state.materials,
                        &default_mat,
                        batch.material_index,
                    );

                    // glTF spec: cull back faces for single-sided materials
                    unsafe {
                        if !mat.double_sided {
                            gl.enable(glow::CULL_FACE);
                            gl.cull_face(glow::BACK);
                        } else {
                            gl.disable(glow::CULL_FACE);
                        }
                    }

                    if active_material != Some((material_slot, state.selected_node)) {
                        bind_pbr_material_state(
                            gl,
                            &state.renderer,
                            &state.texture_cache,
                            mat,
                            &texture_defaults,
                            has_env,
                            env_mips,
                            false,
                            state.selected_node,
                            state.focus_sphere,
                            state.focus_field_enabled,
                        );
                        active_material = Some((material_slot, state.selected_node));
                        material_updates += 1;
                    }

                    apply_batch_winding(gl, batch.orientation_sign, batch.perm_parity);
                    if active_vertex_state
                        .is_none_or(|previous| !same_vertex_uniform_state(previous, batch))
                    {
                        let batch_camera = camera_for_batch(&camera, batch);
                        quilting_renderer::pass::upload_batch_ubo(
                            gl, state.renderer.vtx_ubo(), &batch_camera,
                            batch.suppress_source_roots,
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
                        batch_index,
                        RenderPass::PbrOpaque,
                        quilting_core::render::RenderGeometry::Triangles,
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
                for (batch_index, batch) in webgl_pbr_draws(
                    resolved_pbr_plan,
                    render_batches,
                    RenderPass::PbrTransparent,
                ) {
                    let (material_slot, mat) = pbr_material_for_index(
                        &state.materials,
                        &default_mat,
                        batch.material_index,
                    );

                    let is_blend = mat.alpha_mode == PbrAlphaMode::Blend;
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

                    if active_material != Some((material_slot, state.selected_node)) {
                        bind_pbr_material_state(
                            gl,
                            &state.renderer,
                            &state.texture_cache,
                            mat,
                            &texture_defaults,
                            has_env,
                            env_mips,
                            true,
                            state.selected_node,
                            state.focus_sphere,
                            state.focus_field_enabled,
                        );
                        active_material = Some((material_slot, state.selected_node));
                        material_updates += 1;
                    }

                    apply_batch_winding(gl, batch.orientation_sign, batch.perm_parity);
                    if active_vertex_state
                        .is_none_or(|previous| !same_vertex_uniform_state(previous, batch))
                    {
                        let batch_camera = camera_for_batch(&camera, batch);
                        quilting_renderer::pass::upload_batch_ubo(
                            gl, state.renderer.vtx_ubo(), &batch_camera,
                            batch.suppress_source_roots,
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
                        batch_index,
                        RenderPass::PbrTransparent,
                        quilting_core::render::RenderGeometry::Triangles,
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
                if focus_postprocess_requested {
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
                                // Need a separate half-float FBO for the focused weight output.
                                if state.fuzzy_scene_tex.is_none() || state.fuzzy_scene_size != (vw, vh) {
                                    if let Some(old) = state.fuzzy_scene_fbo { gl.delete_framebuffer(old); }
                                    if let Some(old) = state.fuzzy_scene_tex { gl.delete_texture(old); }
                                    let tex = gl.create_texture().unwrap();
                                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA16F as i32, vw, vh, 0,
                                        glow::RGBA, glow::HALF_FLOAT, glow::PixelUnpackData::Slice(None));
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
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
                                // Radial: blit scene color from the default FB to RGBA8.
                                if state.fuzzy_scene_tex.is_none() || state.fuzzy_scene_size != (vw, vh) {
                                    if let Some(old) = state.fuzzy_scene_fbo { gl.delete_framebuffer(old); }
                                    if let Some(old) = state.fuzzy_scene_tex { gl.delete_texture(old); }
                                    let tex = gl.create_texture().unwrap();
                                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, vw, vh, 0,
                                        glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
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
                                    gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA16F as i32, vw, vh, 0,
                                        glow::RGBA, glow::HALF_FLOAT, glow::PixelUnpackData::Slice(None));
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
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

                // Keep selection as a post-style overlay in PBR as well as
                // diagnostic modes. This also gives the backend oracle one
                // complete incumbent composition to compare with WebGPU.
                if state.highlight_face >= 0 && state.highlight_prog.is_some() {
                    render_highlight(state.renderer.gl(), state, &camera);
                }

                #[cfg(feature = "webgpu-backend")]
                if state.backend_evidence_requested {
                    state.backend_evidence_requested = false;
                    match capture_current_webgl_pbr_frame_evidence(
                        gl,
                        state.viewport_size,
                        state.render_calls,
                        submission_stats,
                        state.focus_postprocess,
                    ) {
                        Ok(capture) => {
                            state.backend_evidence_capture = Some(capture);
                            state.backend_evidence_error = None;
                        }
                        Err(error) => {
                            state.backend_evidence_capture = None;
                            state.backend_evidence_error = Some(error);
                        }
                    }
                }

                state.pbr_draw_calls = state
                    .pbr_draw_calls
                    .saturating_add(submission_stats.draw_calls);
                state.webgl_patch_frames = state.webgl_patch_frames.saturating_add(1);
                state.pbr_material_updates += material_updates;
                state.pbr_vertex_uniform_updates += vertex_uniform_updates;
                state.last_render_submission = submission_stats;
                state.render_submission_totals.merge(submission_stats);
                if use_resolved_pbr_plan {
                    state.render_shadow.record_execution_success();
                }
                observe_render_submission(
                    state,
                    prepared_render_revision,
                    render_frame.as_ref(),
                    submission_stats,
                );
                state.renderer.end_frame();
                state
                    .render_timing
                    .observe(render_started_ms, browser_now_ms());
                return;
            }
            RenderStyle::Lod => {
                state
                    .renderer
                    .matcap_ubo()
                    .upload(gl, 0.0, state.matcap_style.as_f32()); // heatmap
                state.renderer.matcap_ubo().bind(gl);
            }
            _ => {
                state
                    .renderer
                    .matcap_ubo()
                    .upload(gl, 1.0, state.matcap_style.as_f32());
                state.renderer.matcap_ubo().bind(gl);
            }
        }

        let resolved_submission = if prepared_render_revision.is_some() {
            render_frame.as_ref().and_then(|frame| {
                let Some(plan) = frame.retained_command_plan() else {
                    state
                        .render_shadow
                        .record_execution_fallback("shared render frame lost its command plan");
                    return None;
                };
                let execution = match frame.execution_with_command_plan(plan) {
                    Ok(execution) => execution,
                    Err(error) => {
                        state.render_shadow.record_execution_fallback(error);
                        return None;
                    }
                };
                match state.renderer.render_diagnostic_execution(
                    execution,
                    &camera,
                    render_batches,
                ) {
                    Ok(submission) => {
                        state.render_shadow.record_execution_success();
                        Some(submission)
                    }
                    Err(error) => {
                        warn!("Resolved WebGL diagnostic frame fell back: {error}");
                        state.render_shadow.record_execution_fallback(error);
                        None
                    }
                }
            })
        } else {
            None
        };
        let submission_stats = resolved_submission.unwrap_or_else(|| {
            state
                .renderer
                .render(state.render_style, &camera, render_batches)
        });
        state.webgl_patch_frames = state.webgl_patch_frames.saturating_add(1);
        #[cfg(feature = "webgpu-backend")]
        if state.backend_evidence_requested {
            state.backend_evidence_requested = false;
            match capture_webgl_frame_evidence(
                state,
                state.render_style,
                state.viewport_size,
                state.render_calls,
                &camera,
                render_batches,
                submission_stats,
            ) {
                Ok(capture) => {
                    state.backend_evidence_capture = Some(capture);
                    state.backend_evidence_error = None;
                }
                Err(error) => {
                    state.backend_evidence_capture = None;
                    state.backend_evidence_error = Some(error);
                }
            }
        }
        observe_render_submission(
            state,
            prepared_render_revision,
            render_frame.as_ref(),
            submission_stats,
        );

        // Highlight pass: overlay picked QB patch with cyan
        if state.highlight_face >= 0 && state.highlight_prog.is_some() {
            render_highlight(state.renderer.gl(), state, &camera);
        }

        state.last_render_submission = submission_stats;
        state.render_submission_totals.merge(submission_stats);
        state.renderer.end_frame();
        state
            .render_timing
            .observe(render_started_ms, browser_now_ms());
    });
}

/// Render highlight overlay for the picked face.
/// Requires pick FBO + highlight program to exist (created in mr_pick).
/// Renders pick pass to FBO, then fullscreen overlay where face ID matches.
fn render_highlight(gl: &glow::Context, state: &MainState, camera: &quilting_renderer::pass::Camera) {
    render_highlight_to(gl, state, camera, None, state.pick_size);
}

/// Render the incumbent pick-texture highlight into an explicit composition
/// target. Visible rendering uses the default framebuffer; backend evidence
/// uses its retained offscreen color/depth target so the complete selected-face
/// composition can be compared with WebGPU.
fn render_highlight_to(
    gl: &glow::Context,
    state: &MainState,
    camera: &quilting_renderer::pass::Camera,
    output_framebuffer: Option<glow::Framebuffer>,
    output_viewport: (i32, i32),
) {
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
                batch.suppress_source_roots,
                1,
                &batch.euclidean_model, &batch.euclidean_normal,
            );
            state.renderer.vtx_ubo().bind(gl);

            gl.bind_vertex_array(Some(batch.mesh.tri_vao));
            gl.draw_elements_instanced(glow::TRIANGLES, batch.mesh.num_tri_indices,
                glow::UNSIGNED_INT, batch.mesh.tri_index_offset, batch.mesh.num_instances);
        }

        // Fullscreen overlay: cyan where pick matches target face
        gl.bind_framebuffer(glow::FRAMEBUFFER, output_framebuffer);
        gl.viewport(0, 0, output_viewport.0, output_viewport.1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen_test]
    fn evidence_clear_alpha_canonicalization_respects_padded_rows() {
        let mut rgba8 = vec![0xff; 32];
        rgba8[0..8].copy_from_slice(&[51, 51, 77, 255, 1, 2, 3, 255]);
        rgba8[16..24].copy_from_slice(&[51, 51, 76, 255, 4, 5, 6, 255]);
        canonicalize_incumbent_clear_alpha(&mut rgba8, [2, 2], 16).unwrap();
        assert_eq!(rgba8[3], 0);
        assert_eq!(rgba8[7], 255);
        assert_eq!(rgba8[19], 0);
        assert_eq!(rgba8[23], 255);
        assert_eq!(&rgba8[8..16], &[0xff; 8]);
        assert_eq!(&rgba8[24..32], &[0xff; 8]);
    }

    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen_test]
    fn live_presentation_requests_resident_pbr_without_enabling_headless_shadow_work() {
        assert!(webgpu_render_style_requested(
            RenderStyle::Normals,
            false,
            false,
        ));
        assert!(!webgpu_render_style_requested(
            RenderStyle::Pbr,
            false,
            false,
        ));
        assert!(webgpu_render_style_requested(
            RenderStyle::Pbr,
            true,
            false,
        ));
        assert!(webgpu_render_style_requested(
            RenderStyle::Pbr,
            false,
            true,
        ));
    }

    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen_test]
    fn backend_evidence_requires_ready_focus_residency() {
        assert!(backend_frame_evidence_supports_composition(None));
        let focus = focus_postprocess_packet(
            11.0, 1.0, 3, 0.5, 0.1, false, 0.5, 0.5, 1, 3, 1.5,
        )
        .unwrap();
        assert!(!backend_frame_evidence_supports_composition(Some(focus)));
    }

    #[wasm_bindgen_test]
    fn focus_postprocess_packet_preserves_exact_backend_semantics() {
        let focus = focus_postprocess_packet(
            48.0, 1.75, 3, 0.25, 0.08, true, 0.2, 4.5, 2, 4, 2.25,
        )
        .unwrap();
        assert_eq!(focus.mode, FocusPostprocessMode::Spheroidal);
        assert_eq!(focus.blur_radius_pixels, 48);
        assert_eq!(focus.focus_coordinate, 0.25);
        assert_eq!(focus.stretch_range, [0.2, 4.5]);
        assert_eq!(focus.gaussian_passes, 2);
        assert_eq!(focus.kawase_passes, 4);
        assert!(focus_postprocess_packet(
            48.5, 1.75, 3, 0.25, 0.08, true, 0.2, 4.5, 2, 4, 2.25,
        )
        .is_err());
    }

    #[wasm_bindgen_test]
    fn admitted_ordinary_world_model_overrides_conformal_source_only() {
        let source = EntityConformalState {
            mobius: IDENTITY_MOBIUS,
            orientation_sign: -1,
            euclidean_model: IDENTITY_MATRIX,
        };
        let mut admitted = IDENTITY_MATRIX;
        admitted[12] = 3.0;
        admitted[13] = -2.0;

        let resolved = resolved_node_model(source, Some(&admitted));

        assert_eq!(resolved.mobius, source.mobius);
        assert_eq!(resolved.orientation_sign, source.orientation_sign);
        assert_eq!(resolved.euclidean_model, admitted);
    }

    #[wasm_bindgen_test]
    fn entity_local_surface_face_resolves_once_into_packed_scene_order() {
        let face_nodes = [4, 9, 4, 7, 4];
        assert_eq!(packed_face_for_entity_local_face(&face_nodes, 4, 0), Some(0));
        assert_eq!(packed_face_for_entity_local_face(&face_nodes, 4, 1), Some(2));
        assert_eq!(packed_face_for_entity_local_face(&face_nodes, 4, 2), Some(4));
        assert_eq!(packed_face_for_entity_local_face(&face_nodes, 4, 3), None);
    }

    #[wasm_bindgen_test]
    fn stable_face_record_keeps_semantic_node_separate_from_render_state() {
        let resident = batch::ResidentLod::uniform(2);
        let key = batch::RenderBatchKey::from_resident(resident, 7, usize::MAX);
        let mut members = [batch::RenderBatchMember {
            face_index: 23,
            leaf_id: quilting_core::screen_partition::ScreenPatchLeafId::ROOT,
            node_index: 91,
            edge_lods: [2; 3],
            permutation_index: 0,
            vertex_lods: [2; 3],
        }];
        let mut staging = [0.0; instance_layout::BATCH_TOPOLOGY_STRIDE];

        let written = fill_batch_instance_data(
            key,
            &members,
            &mut staging,
        )
        .unwrap();

        assert_eq!(written, instance_layout::BATCH_TOPOLOGY_STRIDE);
        assert_eq!(staging[instance_layout::batch_offset::FACE_ID], 23.0);
        assert_eq!(staging[instance_layout::batch_offset::LEAF_DEPTH], 0.0);
        assert_eq!(staging[instance_layout::batch_offset::LEAF_PATH], 0.0);

        members[0].leaf_id = quilting_core::screen_partition::ScreenPatchLeafId {
            depth: instance_layout::BATCH_LEAF_MAX_DEPTH,
            path: (1 << 24) - 1,
        };
        fill_batch_instance_data(
            key,
            &members,
            &mut staging,
        )
        .unwrap();
        assert_eq!(staging[instance_layout::batch_offset::LEAF_DEPTH], 12.0);
        assert_eq!(
            staging[instance_layout::batch_offset::LEAF_PATH],
            16_777_215.0,
        );

        let mut instances = vec![0.0; 24 * instance_layout::STRIDE];
        let mut nodes = vec![0; 24];
        nodes[23] = 91;
        embed_face_node_ids(&mut instances, 24, &nodes).unwrap();
        assert_eq!(
            instances[23 * instance_layout::STRIDE + instance_layout::offset::NODE_ID],
            91.0,
        );
    }

    #[wasm_bindgen_test]
    fn unmatched_worker_publications_advance_the_shadow_baseline_only() {
        assert!(same_context_lod_request_stamp(0, 3, false, 0.0, 0.0, 0, 0).is_err());
        let authority =
            same_context_lod_authority_stamp(0, 3, false, 0.0, 0.0, 0, 0).unwrap();
        assert_eq!(authority.request_id, 0);
        assert_eq!(authority.classified_faces, 3);
        assert!(authority.pose.is_none());
    }

    #[wasm_bindgen_test]
    fn authoritative_lod_accepts_revision_lag_only_inside_one_continuity_epoch() {
        let candidate = SameContextLodPoseStamp {
            clip_time_seconds: 0.25,
            sample_time_seconds: 4.0,
            revision: 7,
            continuity_epoch: 3,
            payload: None,
        };
        assert!(same_context_lod_pose_continuity_matches(
            Some(candidate),
            Some((9, 3)),
        ));
        assert!(!same_context_lod_pose_continuity_matches(
            Some(candidate),
            Some((1, 4)),
        ));
        assert!(!same_context_lod_pose_continuity_matches(
            Some(candidate),
            None,
        ));
        assert!(!same_context_lod_pose_continuity_matches(
            None,
            Some((1, 3)),
        ));
        assert!(same_context_lod_pose_continuity_matches(None, None));
    }

    #[wasm_bindgen_test]
    fn baseline_publication_retains_the_exact_live_rollback_epoch() {
        let root_lod = batch::ResidentLod::uniform(1);
        let adaptive_lod = batch::ResidentLod::uniform(2);
        let root_key = batch::RenderBatchKey::from_resident(root_lod, 0, 0);
        let adaptive_key = batch::RenderBatchKey::from_resident(adaptive_lod, 0, 0);
        let root_member = batch::RenderBatchMember {
            face_index: 7,
            leaf_id: quilting_core::screen_partition::ScreenPatchLeafId::ROOT,
            node_index: 0,
            edge_lods: [1; 3],
            permutation_index: 0,
            vertex_lods: [1; 3],
        };
        let adaptive_member = batch::RenderBatchMember {
            face_index: 7,
            leaf_id: quilting_core::screen_partition::ScreenPatchLeafId {
                depth: 1,
                path: 2,
            },
            node_index: 0,
            edge_lods: [2; 3],
            permutation_index: 0,
            vertex_lods: [2; 3],
        };
        let expected_baseline = BTreeMap::from([(root_key, vec![root_member])]);
        let expected_live = BTreeMap::from([(adaptive_key, vec![adaptive_member])]);
        let mut baseline = expected_baseline.clone();
        let mut live = expected_live.clone();

        let rollback = stage_baseline_batch_groups(&mut live, &mut baseline);

        assert_eq!(live, expected_baseline);
        assert_eq!(rollback, expected_live);
        assert!(baseline.is_empty());
    }
}
