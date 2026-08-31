//! Rollback-safe browser residency for the staged WebGPU backend.
//!
//! This module deliberately owns no semantic scene or browser layout. Exact
//! packed-atlas, prepared-model, extracted-scene, and current-frame inputs can
//! be mirrored into a headless device for parity or presented through an
//! application-supplied canvas. Canvas selection remains an explicit startup
//! decision; a claimed context is never silently repurposed.

use quilting_core::batch::FaceLodGrading;
use quilting_core::material::{
    EnvironmentMapAsset, EnvironmentMapDescriptor, TextureAssetDescriptor, TextureWrapMode,
};
use quilting_core::render::{
    RenderFrame, RenderFrameOptions, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle,
    RenderSubmissionStats, RenderView, ResidentRootDrawDomains, ValidatedRenderScene,
};
use quilting_renderer::compute::{
    prepare_lod_dispatch_state, LodAtlasLookup, PreparedLodModel, WgslLodDispatchMetrics,
};
use quilting_webgpu::{
    resident_root_render_domains, AdaptiveOverlayScene, DiagnosticPatchRenderPipelines,
    FocusPbrRenderResources, LodClassifierDevice, LodClassifierModel, LodPose,
    OffscreenPatchRenderTarget, PackedPatchAtlas, PatchFrameEncoding, PatchPickPipeline,
    PatchPickRequest, PatchPickTarget, PatchPresentationSurface, PatchRenderScene,
    PatchRenderSceneUpdate, PbrEnvironmentMap, PbrTextureTable, PoseUploadPolicy,
    ResidentGeometryBucketScene, ResidentRootPickPipeline, ResidentRootPreparationScene,
    ResidentRootRenderBindings, ResidentRootRenderPipeline, StagedOffscreenImageReadback,
    StagedPatchPickReadback, SurfacePresentation, WebGpuAdapterSummary,
};
use serde::Serialize;
use std::cell::RefCell;

thread_local! {
    static BACKEND: RefCell<WebGpuBackend> = RefCell::new(WebGpuBackend::default());
}

#[derive(Default)]
struct WebGpuBackend {
    state: &'static str,
    device: Option<LodClassifierDevice>,
    adapter: Option<WebGpuAdapterSummary>,
    atlas: Option<PackedPatchAtlas>,
    textures: Option<PbrTextureTable>,
    environment: Option<PbrEnvironmentMap>,
    model: Option<LodClassifierModel>,
    model_source: Option<PreparedLodModel>,
    pipelines: Option<DiagnosticPatchRenderPipelines>,
    pick_pipeline: Option<PatchPickPipeline>,
    pick_target: Option<PatchPickTarget>,
    resident_root_pipeline: Option<ResidentRootRenderPipeline>,
    resident_root_pick_pipeline: Option<ResidentRootPickPipeline>,
    focus: Option<FocusPbrRenderResources>,
    resident_roots: Option<ResidentRootBackend>,
    scene: Option<PatchRenderScene>,
    scene_source_revision: Option<u64>,
    target: Option<OffscreenPatchRenderTarget>,
    presentation: Option<PatchPresentationSurface>,
    /// One explicit parity frame must use the offscreen target even when the
    /// same device owns a live presentation surface.
    frame_evidence_requested: bool,
    /// Cache key for deciding whether the next requested frame can retain the
    /// current presentation. Device LOD dispatch invalidates this without
    /// invalidating the resources of the last frame that actually completed.
    last_frame_input: Option<LiveFrameInput>,
    /// Exact input represented by the retained preparation/visibility/root
    /// buffers. Picking consumes this completed-frame witness even while a
    /// newer device LOD epoch is waiting for its presentation frame.
    last_completed_frame_input: Option<LiveFrameInput>,
    /// Exact input represented by pixels retained on the browser presentation
    /// surface. Offscreen evidence must not overwrite this witness.
    last_presentation_input: Option<LiveFrameInput>,
    /// Whether the most recent requested live presentation frame was either
    /// submitted or proven identical to `last_presentation_input`.
    presentation_frame_admitted: bool,
    last_presentation_style: Option<RenderStyle>,
    frame_evidence: Option<FrameEvidenceMetadata>,
    last_face_visibility_bits: Vec<u32>,
    next_face_visibility_bits: Vec<u32>,
    next_morph_weights: Vec<f32>,
    device_pose_identity: Option<RenderPoseIdentity>,
    patch_pose_uniforms_ready: bool,
    resident_pose_uniforms_ready: bool,
    initialization_attempts: u64,
    atlas_uploads: u64,
    texture_uploads: u64,
    environment_uploads: u64,
    model_uploads: u64,
    scene_uploads: u64,
    scene_rebuilds: u64,
    scene_updates: u64,
    target_rebuilds: u64,
    frame_attempts: u64,
    frames_submitted: u64,
    frames_skipped_unchanged: u64,
    frame_failures: u64,
    visibility_uploads: u64,
    visibility_upload_bytes: u64,
    fallback_pose_uploads: u64,
    fallback_pose_initializations: u64,
    fallback_pose_reuses: u64,
    classifier_pose_uploads: u64,
    classifier_pose_reuses: u64,
    resident_pose_uploads: u64,
    resident_pose_initializations: u64,
    resident_pose_reuses: u64,
    device_lod_dispatches: u64,
    device_lod_frames: u64,
    resident_root_scene_uploads: u64,
    resident_root_scene_reuses: u64,
    resident_root_frames: u64,
    resident_root_fallbacks: u64,
    focus_target_rebuilds: u64,
    focus_frames: u64,
    focus_fallbacks: u64,
    pick_requests: u64,
    pick_submissions: u64,
    pick_failures: u64,
    last_device_lod_epoch: Option<u64>,
    last_frame_revision: u64,
    last_source_render_call: u64,
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_logical_submission: RenderSubmissionStats,
    last_viewport: [u32; 2],
    last_frame_used_resident_roots: bool,
    last_frame_failure: Option<String>,
    last_resident_root_error: Option<String>,
    last_focus_error: Option<String>,
    last_pick_error: Option<String>,
    last_error: Option<String>,
}

struct ResidentRootFocusBackend {
    bindings: ResidentRootRenderBindings,
    overlay: Option<AdaptiveOverlayScene>,
}

struct ResidentRootPbrCandidate {
    bindings: ResidentRootRenderBindings,
    overlay: Option<AdaptiveOverlayScene>,
    focus: Option<ResidentRootFocusBackend>,
    focus_failure: Option<String>,
}

struct ResidentRootBackend {
    domains: ResidentRootDrawDomains,
    preparation: ResidentRootPreparationScene,
    geometry: ResidentGeometryBucketScene,
    bindings: ResidentRootRenderBindings,
    overlay: Option<AdaptiveOverlayScene>,
    focus: Option<ResidentRootFocusBackend>,
}

struct ResidentRootReuseCandidate {
    bindings: ResidentRootRenderBindings,
    overlay: Option<AdaptiveOverlayScene>,
    focus: Option<ResidentRootFocusBackend>,
    focus_failure: Option<String>,
}

#[derive(Clone, Copy)]
struct FrameEvidenceMetadata {
    frame_revision: u64,
    source_render_call: u64,
    viewport: [u32; 2],
    logical_submission: RenderSubmissionStats,
    indirect_draw_calls: u32,
    source_instances: u32,
    focus_postprocess: Option<quilting_core::render::FocusPostprocessPacket>,
}

enum ResidentRootSceneCandidate {
    Unavailable,
    Reuse(Box<ResidentRootReuseCandidate>),
    Replace {
        backend: Box<ResidentRootBackend>,
        focus_failure: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameDestination {
    Offscreen,
    Presentation,
}

fn frame_destination(presentation_ready: bool, frame_evidence_requested: bool) -> FrameDestination {
    if presentation_ready && !frame_evidence_requested {
        FrameDestination::Presentation
    } else {
        FrameDestination::Offscreen
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LiveFrameInput {
    source_revision: u64,
    device_lod_epoch: Option<u64>,
    style: RenderStyle,
    view: RenderView,
    options: RenderFrameOptions,
}

fn render_pose_upload_policy(
    resident: Option<RenderPoseIdentity>,
    requested: RenderPoseIdentity,
    preparation_ready: bool,
) -> PoseUploadPolicy {
    if resident != Some(requested) {
        PoseUploadPolicy::Publish
    } else if !preparation_ready {
        PoseUploadPolicy::PublishPreparation
    } else {
        PoseUploadPolicy::Reuse
    }
}

/// Whether the incumbent renderer still has to submit the visible patch pass
/// after one WebGPU frame attempt. Offscreen shadow work never owns visible
/// pixels, while a successfully presented (or deliberately retained) surface
/// frame does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LiveFrameDisposition {
    IncumbentRequired,
    ShadowSubmitted(RenderSubmissionStats),
    PresentationSubmitted(RenderSubmissionStats),
    PresentationRetained,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WebGpuBackendDiagnostics {
    state: &'static str,
    adapter_name: Option<String>,
    adapter_backend: Option<String>,
    adapter_device_type: Option<String>,
    atlas_ready: bool,
    atlas_entries: usize,
    atlas_vertices: u32,
    textures_ready: bool,
    texture_slots: usize,
    texture_images: usize,
    environment_ready: bool,
    environment_prefiltered_size: u32,
    environment_prefiltered_mips: u32,
    environment_irradiance_size: u32,
    pbr_texture_materials: usize,
    pbr_texture_references: usize,
    pbr_texture_resident_references: usize,
    pbr_texture_unresolved_references: usize,
    pbr_presentation_ready: bool,
    model_ready: bool,
    model_faces: usize,
    scene_ready: bool,
    scene_batches: u32,
    scene_instances: u32,
    target_ready: bool,
    target_viewport: [u32; 2],
    presentation_ready: bool,
    presentation_viewport: [u32; 2],
    presentation_color_format: Option<String>,
    presentation_style: Option<&'static str>,
    presentation_frame_admitted: bool,
    presentation_frames: u64,
    presentation_skips: u64,
    presentation_losses: u64,
    render_shader_module_hits: u64,
    render_shader_module_misses: u64,
    render_shader_module_failed_creations: u64,
    render_shader_module_invalidations: u64,
    render_shader_module_entries: usize,
    focus_postprocess_pipeline_hits: u64,
    focus_postprocess_pipeline_misses: u64,
    focus_postprocess_pipeline_failed_creations: u64,
    focus_postprocess_pipeline_invalidations: u64,
    focus_postprocess_pipeline_entries: usize,
    prepared_patch_pipeline_hits: u64,
    prepared_patch_pipeline_misses: u64,
    prepared_patch_pipeline_failed_creations: u64,
    prepared_patch_pipeline_invalidations: u64,
    prepared_patch_pipeline_entries: usize,
    resident_root_pipeline_hits: u64,
    resident_root_pipeline_misses: u64,
    resident_root_pipeline_failed_creations: u64,
    resident_root_pipeline_invalidations: u64,
    resident_root_pipeline_entries: usize,
    initialization_attempts: u64,
    atlas_uploads: u64,
    texture_uploads: u64,
    environment_uploads: u64,
    model_uploads: u64,
    scene_uploads: u64,
    scene_rebuilds: u64,
    scene_updates: u64,
    target_rebuilds: u64,
    frame_attempts: u64,
    frames_submitted: u64,
    frames_skipped_unchanged: u64,
    frame_failures: u64,
    visibility_uploads: u64,
    visibility_upload_bytes: u64,
    frame_table_uploads: u64,
    frame_table_reuses: u64,
    frame_table_upload_bytes: u64,
    lod_state_uploads: u64,
    lod_state_reuses: u64,
    lod_state_upload_bytes: u64,
    fallback_pose_uploads: u64,
    fallback_pose_initializations: u64,
    fallback_pose_reuses: u64,
    classifier_pose_uploads: u64,
    classifier_pose_reuses: u64,
    device_pose_asset_revision: Option<u64>,
    device_pose_revision: Option<u64>,
    patch_pose_uniforms_ready: bool,
    resident_pose_uploads: u64,
    resident_pose_initializations: u64,
    resident_pose_reuses: u64,
    resident_pose_uniforms_ready: bool,
    device_lod_dispatches: u64,
    device_lod_frames: u64,
    resident_root_pipeline_ready: bool,
    resident_root_scene_ready: bool,
    resident_root_scene_uploads: u64,
    resident_root_scene_reuses: u64,
    resident_root_frames: u64,
    resident_root_fallbacks: u64,
    focus_pipeline_ready: bool,
    focus_scene_ready: bool,
    focus_evidence_ready: bool,
    focus_target_ready: bool,
    focus_target_viewport: [u32; 2],
    focus_target_rebuilds: u64,
    focus_plan_builds: u64,
    focus_plan_reuses: u64,
    focus_uniform_uploads: u64,
    focus_uniform_reuses: u64,
    focus_uniform_upload_bytes: u64,
    focus_frames: u64,
    focus_fallbacks: u64,
    pick_pipeline_ready: bool,
    resident_root_pick_pipeline_ready: bool,
    pick_target_ready: bool,
    pick_frame_ready: bool,
    pick_requests: u64,
    pick_submissions: u64,
    pick_failures: u64,
    last_device_lod_epoch: Option<u64>,
    last_frame_revision: u64,
    last_source_render_call: u64,
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_logical_submission: RenderSubmissionStats,
    last_viewport: [u32; 2],
    last_frame_used_resident_roots: bool,
    last_frame_failure: Option<String>,
    last_resident_root_error: Option<String>,
    last_focus_error: Option<String>,
    last_pick_error: Option<String>,
    last_error: Option<String>,
}

impl WebGpuBackend {
    fn incumbent_required(&mut self) -> LiveFrameDisposition {
        // A retained surface is only evidence for its exact last frame. If the
        // current request cannot be represented, clear that witness so the
        // browser exposes WebGL2 instead of a stale WebGPU image.
        if self.presentation.is_some() {
            self.last_frame_input = None;
            self.last_completed_frame_input = None;
            self.presentation_frame_admitted = false;
        }
        LiveFrameDisposition::IncumbentRequired
    }

    fn diagnostics(&self) -> WebGpuBackendDiagnostics {
        let presentation = self
            .presentation
            .as_ref()
            .map(PatchPresentationSurface::diagnostics);
        let render_shader_memo = self
            .device
            .as_ref()
            .map(LodClassifierDevice::render_shader_memo_diagnostics)
            .unwrap_or_default();
        let focus_postprocess_pipeline_memo = self
            .device
            .as_ref()
            .map(LodClassifierDevice::focus_postprocess_pipeline_memo_diagnostics)
            .unwrap_or_default();
        let prepared_patch_pipeline_memo = self
            .device
            .as_ref()
            .map(LodClassifierDevice::prepared_patch_pipeline_memo_diagnostics)
            .unwrap_or_default();
        let resident_root_pipeline_memo = self
            .device
            .as_ref()
            .map(LodClassifierDevice::resident_root_pipeline_memo_diagnostics)
            .unwrap_or_default();
        let frame_table_memo = self
            .device
            .as_ref()
            .map(LodClassifierDevice::frame_table_memo_diagnostics)
            .unwrap_or_default();
        let lod_state_memo = self
            .device
            .as_ref()
            .map(LodClassifierDevice::lod_state_memo_diagnostics)
            .unwrap_or_default();
        let focus_plan_memo = self
            .focus
            .as_ref()
            .and_then(FocusPbrRenderResources::target)
            .and_then(|target| target.memo_diagnostics().ok())
            .unwrap_or_default();
        let pbr_texture_residency = self
            .scene
            .as_ref()
            .and_then(PatchRenderScene::pbr_texture_residency)
            .unwrap_or_default();
        WebGpuBackendDiagnostics {
            state: if self.state.is_empty() {
                "disabled"
            } else {
                self.state
            },
            adapter_name: self.adapter.as_ref().map(|adapter| adapter.name.clone()),
            adapter_backend: self.adapter.as_ref().map(|adapter| adapter.backend.clone()),
            adapter_device_type: self
                .adapter
                .as_ref()
                .map(|adapter| adapter.device_type.clone()),
            atlas_ready: self.atlas.is_some(),
            atlas_entries: self.atlas.as_ref().map_or(0, PackedPatchAtlas::entry_count),
            atlas_vertices: self
                .atlas
                .as_ref()
                .map_or(0, PackedPatchAtlas::vertex_count),
            textures_ready: self.textures.is_some(),
            texture_slots: self.textures.as_ref().map_or(0, PbrTextureTable::len),
            texture_images: self
                .textures
                .as_ref()
                .map_or(0, PbrTextureTable::occupied_len),
            environment_ready: self.environment.is_some(),
            environment_prefiltered_size: self.environment.as_ref().map_or(0, |environment| {
                environment.descriptor().prefiltered_face_size
            }),
            environment_prefiltered_mips: self
                .environment
                .as_ref()
                .map_or(0, PbrEnvironmentMap::prefiltered_mip_count),
            environment_irradiance_size: self.environment.as_ref().map_or(0, |environment| {
                environment.descriptor().irradiance_face_size
            }),
            pbr_texture_materials: pbr_texture_residency.len(),
            pbr_texture_references: pbr_texture_residency
                .iter()
                .map(|residency| residency.referenced_mask().count_ones() as usize)
                .sum(),
            pbr_texture_resident_references: pbr_texture_residency
                .iter()
                .map(|residency| residency.resident_mask().count_ones() as usize)
                .sum(),
            pbr_texture_unresolved_references: pbr_texture_residency
                .iter()
                .map(|residency| residency.unresolved_mask().count_ones() as usize)
                .sum(),
            pbr_presentation_ready: self.scene.as_ref().is_some_and(|scene| {
                scene.supports_resident_patch_presentation_frame(
                    RenderStyle::Pbr,
                    RenderFrameOptions::default(),
                )
            }),
            model_ready: self.model.is_some(),
            model_faces: self
                .model_source
                .as_ref()
                .map_or(0, |model| model.residency.num_faces),
            scene_ready: self.scene.is_some(),
            scene_batches: self.scene.as_ref().map_or(0, PatchRenderScene::batch_count),
            scene_instances: self.scene.as_ref().map_or(0, PatchRenderScene::patch_count),
            target_ready: self.target.is_some(),
            target_viewport: self
                .target
                .as_ref()
                .map_or([0, 0], OffscreenPatchRenderTarget::size),
            presentation_ready: presentation
                .as_ref()
                .is_some_and(|presentation| presentation.configured),
            presentation_viewport: presentation
                .as_ref()
                .map_or([0, 0], |presentation| presentation.size),
            presentation_color_format: presentation
                .as_ref()
                .map(|presentation| presentation.color_format.clone()),
            presentation_style: self
                .presentation
                .as_ref()
                .and(self.last_presentation_style.map(render_style_name)),
            presentation_frame_admitted: self.presentation_frame_admitted,
            presentation_frames: presentation
                .as_ref()
                .map_or(0, |presentation| presentation.frames_presented),
            presentation_skips: presentation
                .as_ref()
                .map_or(0, |presentation| presentation.frames_skipped),
            presentation_losses: presentation
                .as_ref()
                .map_or(0, |presentation| presentation.surface_losses),
            render_shader_module_hits: render_shader_memo.hits,
            render_shader_module_misses: render_shader_memo.misses,
            render_shader_module_failed_creations: render_shader_memo.failed_creations,
            render_shader_module_invalidations: render_shader_memo.invalidations,
            render_shader_module_entries: render_shader_memo.resident_entries,
            focus_postprocess_pipeline_hits: focus_postprocess_pipeline_memo.hits,
            focus_postprocess_pipeline_misses: focus_postprocess_pipeline_memo.misses,
            focus_postprocess_pipeline_failed_creations: focus_postprocess_pipeline_memo
                .failed_creations,
            focus_postprocess_pipeline_invalidations: focus_postprocess_pipeline_memo.invalidations,
            focus_postprocess_pipeline_entries: focus_postprocess_pipeline_memo.resident_entries,
            prepared_patch_pipeline_hits: prepared_patch_pipeline_memo.hits,
            prepared_patch_pipeline_misses: prepared_patch_pipeline_memo.misses,
            prepared_patch_pipeline_failed_creations: prepared_patch_pipeline_memo.failed_creations,
            prepared_patch_pipeline_invalidations: prepared_patch_pipeline_memo.invalidations,
            prepared_patch_pipeline_entries: prepared_patch_pipeline_memo.resident_entries,
            resident_root_pipeline_hits: resident_root_pipeline_memo.hits,
            resident_root_pipeline_misses: resident_root_pipeline_memo.misses,
            resident_root_pipeline_failed_creations: resident_root_pipeline_memo.failed_creations,
            resident_root_pipeline_invalidations: resident_root_pipeline_memo.invalidations,
            resident_root_pipeline_entries: resident_root_pipeline_memo.resident_entries,
            initialization_attempts: self.initialization_attempts,
            atlas_uploads: self.atlas_uploads,
            texture_uploads: self.texture_uploads,
            environment_uploads: self.environment_uploads,
            model_uploads: self.model_uploads,
            scene_uploads: self.scene_uploads,
            scene_rebuilds: self.scene_rebuilds,
            scene_updates: self.scene_updates,
            target_rebuilds: self.target_rebuilds,
            frame_attempts: self.frame_attempts,
            frames_submitted: self.frames_submitted,
            frames_skipped_unchanged: self.frames_skipped_unchanged,
            frame_failures: self.frame_failures,
            visibility_uploads: self.visibility_uploads,
            visibility_upload_bytes: self.visibility_upload_bytes,
            frame_table_uploads: frame_table_memo.uploads,
            frame_table_reuses: frame_table_memo.reuses,
            frame_table_upload_bytes: frame_table_memo.upload_bytes,
            lod_state_uploads: lod_state_memo.uploads,
            lod_state_reuses: lod_state_memo.reuses,
            lod_state_upload_bytes: lod_state_memo.upload_bytes,
            fallback_pose_uploads: self.fallback_pose_uploads,
            fallback_pose_initializations: self.fallback_pose_initializations,
            fallback_pose_reuses: self.fallback_pose_reuses,
            classifier_pose_uploads: self.classifier_pose_uploads,
            classifier_pose_reuses: self.classifier_pose_reuses,
            device_pose_asset_revision: self
                .device_pose_identity
                .map(|identity| identity.asset_revision),
            device_pose_revision: self
                .device_pose_identity
                .map(|identity| identity.pose_revision),
            patch_pose_uniforms_ready: self.patch_pose_uniforms_ready,
            resident_pose_uploads: self.resident_pose_uploads,
            resident_pose_initializations: self.resident_pose_initializations,
            resident_pose_reuses: self.resident_pose_reuses,
            resident_pose_uniforms_ready: self.resident_pose_uniforms_ready,
            device_lod_dispatches: self.device_lod_dispatches,
            device_lod_frames: self.device_lod_frames,
            resident_root_pipeline_ready: self.resident_root_pipeline.is_some(),
            resident_root_scene_ready: self.resident_roots.is_some(),
            resident_root_scene_uploads: self.resident_root_scene_uploads,
            resident_root_scene_reuses: self.resident_root_scene_reuses,
            resident_root_frames: self.resident_root_frames,
            resident_root_fallbacks: self.resident_root_fallbacks,
            focus_pipeline_ready: self.focus.is_some(),
            focus_scene_ready: self
                .resident_roots
                .as_ref()
                .is_some_and(|roots| roots.focus.is_some()),
            focus_evidence_ready: self.presentation.is_none()
                && self.focus.is_some()
                && self.environment.is_some()
                && self
                    .resident_roots
                    .as_ref()
                    .is_some_and(|roots| roots.focus.is_some()),
            focus_target_ready: self
                .focus
                .as_ref()
                .is_some_and(|focus| focus.target().is_some()),
            focus_target_viewport: self
                .focus
                .as_ref()
                .and_then(FocusPbrRenderResources::target)
                .map_or([0, 0], |target| target.size()),
            focus_target_rebuilds: self.focus_target_rebuilds,
            focus_plan_builds: focus_plan_memo.plan_builds,
            focus_plan_reuses: focus_plan_memo.plan_reuses,
            focus_uniform_uploads: focus_plan_memo.uniform_uploads,
            focus_uniform_reuses: focus_plan_memo.uniform_reuses,
            focus_uniform_upload_bytes: focus_plan_memo.uniform_upload_bytes,
            focus_frames: self.focus_frames,
            focus_fallbacks: self.focus_fallbacks,
            pick_pipeline_ready: self.pick_pipeline.is_some(),
            resident_root_pick_pipeline_ready: self.resident_root_pick_pipeline.is_some(),
            pick_target_ready: self.pick_target.is_some(),
            pick_frame_ready: self.last_completed_frame_input.is_some(),
            pick_requests: self.pick_requests,
            pick_submissions: self.pick_submissions,
            pick_failures: self.pick_failures,
            last_device_lod_epoch: self.last_device_lod_epoch,
            last_frame_revision: self.last_frame_revision,
            last_source_render_call: self.last_source_render_call,
            last_indirect_draw_calls: self.last_indirect_draw_calls,
            last_source_instances: self.last_source_instances,
            last_logical_submission: self.last_logical_submission,
            last_viewport: self.last_viewport,
            last_frame_used_resident_roots: self.last_frame_used_resident_roots,
            last_frame_failure: self.last_frame_failure.clone(),
            last_resident_root_error: self.last_resident_root_error.clone(),
            last_focus_error: self.last_focus_error.clone(),
            last_pick_error: self.last_pick_error.clone(),
            last_error: self.last_error.clone(),
        }
    }

    fn fail(&mut self, error: impl ToString) -> String {
        let error = error.to_string();
        self.state = "failed";
        self.last_error = Some(error.clone());
        error
    }

    fn reject_frame(
        &mut self,
        face_visibility_bits: Vec<u32>,
        morph_weights: Vec<f32>,
        error: impl ToString,
    ) -> String {
        let error = error.to_string();
        self.next_face_visibility_bits = face_visibility_bits;
        self.next_morph_weights = morph_weights;
        self.frame_failures = self.frame_failures.saturating_add(1);
        self.last_completed_frame_input = None;
        self.presentation_frame_admitted = false;
        self.last_frame_failure = Some(error.clone());
        self.last_error = Some(error.clone());
        error
    }

    fn reject_pick(&mut self, error: impl ToString) -> String {
        let error = error.to_string();
        self.pick_failures = self.pick_failures.saturating_add(1);
        self.last_pick_error = Some(error.clone());
        error
    }

    fn publish_resident_roots(
        &mut self,
        candidate: Option<ResidentRootBackend>,
        failure: Option<String>,
    ) {
        self.resident_roots = candidate;
        if self.resident_roots.is_some() {
            self.resident_root_scene_uploads = self.resident_root_scene_uploads.saturating_add(1);
        }
        if let Some(error) = failure {
            self.resident_root_fallbacks = self.resident_root_fallbacks.saturating_add(1);
            self.last_resident_root_error = Some(error);
        } else {
            self.last_resident_root_error = None;
        }
    }

    fn publish_resident_scene(
        &mut self,
        candidate: ResidentRootSceneCandidate,
        failure: Option<String>,
    ) {
        // Reuse may retain root topology while replacing the overlay/focus
        // preparation buffers. Initialize the complete family on its next
        // successful frame rather than guessing which local uniform survived.
        self.resident_pose_uniforms_ready = false;
        match candidate {
            ResidentRootSceneCandidate::Reuse(candidate) if failure.is_none() => {
                let ResidentRootReuseCandidate {
                    bindings,
                    overlay,
                    focus,
                    focus_failure,
                } = *candidate;
                let publication = self
                    .device
                    .as_ref()
                    .zip(self.resident_roots.as_mut())
                    .ok_or_else(|| {
                        "resident root reuse lost its device or retained scene".to_string()
                    })
                    .and_then(|(device, resident)| {
                        device
                            .publish_adaptive_overlay_suppression(
                                &resident.geometry,
                                overlay.as_ref(),
                            )
                            .map_err(|error| error.to_string())?;
                        resident.bindings = bindings;
                        resident.overlay = overlay;
                        resident.focus = focus;
                        Ok(())
                    });
                match publication {
                    Ok(()) => {
                        self.resident_root_scene_reuses =
                            self.resident_root_scene_reuses.saturating_add(1);
                        self.last_resident_root_error = None;
                        self.publish_focus_result(focus_failure);
                    }
                    Err(error) => self.publish_resident_roots(None, Some(error)),
                }
            }
            ResidentRootSceneCandidate::Replace {
                backend,
                focus_failure,
            } if failure.is_none() => {
                self.publish_resident_roots(Some(*backend), None);
                self.publish_focus_result(focus_failure);
            }
            ResidentRootSceneCandidate::Unavailable
            | ResidentRootSceneCandidate::Reuse(_)
            | ResidentRootSceneCandidate::Replace { .. } => {
                self.publish_resident_roots(None, failure);
            }
        }
    }

    fn publish_focus_result(&mut self, failure: Option<String>) {
        if self.focus.is_none() && failure.is_none() {
            return;
        }
        if failure.is_some() {
            self.focus_fallbacks = self.focus_fallbacks.saturating_add(1);
        }
        self.last_focus_error = failure;
    }
}

fn create_pick_resources(
    device: &LodClassifierDevice,
    pipelines: &DiagnosticPatchRenderPipelines,
) -> (
    Option<PatchPickPipeline>,
    Option<PatchPickTarget>,
    Option<String>,
) {
    let Some(compatible) = pipelines.get(RenderStyle::Pbr) else {
        return (
            None,
            None,
            Some("WebGPU pick pipeline requires the prepared PBR layout".to_string()),
        );
    };
    match device.create_patch_pick_pipeline(compatible) {
        Ok(pipeline) => (
            Some(pipeline),
            Some(device.create_patch_pick_target()),
            None,
        ),
        Err(error) => (None, None, Some(error.to_string())),
    }
}

fn create_resident_root_pick_resource(
    device: &LodClassifierDevice,
    root_pipeline: Option<&ResidentRootRenderPipeline>,
    pick_pipeline: Option<&PatchPickPipeline>,
) -> (Option<ResidentRootPickPipeline>, Option<String>) {
    let Some(root_pipeline) = root_pipeline else {
        return (None, None);
    };
    let Some(pick_pipeline) = pick_pipeline else {
        return (None, None);
    };
    match device.create_resident_root_pick_pipeline(root_pipeline, pick_pipeline) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

pub(crate) async fn initialize() -> Result<WebGpuBackendDiagnostics, String> {
    let should_request = BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        match backend.state {
            "ready" => return Ok(false),
            "initializing" => {
                return Err("WebGPU backend initialization is already in progress".to_string());
            }
            _ => {}
        }
        backend.state = "initializing";
        backend.initialization_attempts = backend.initialization_attempts.saturating_add(1);
        backend.last_error = None;
        Ok(true)
    })?;
    if should_request {
        match LodClassifierDevice::request_headless("Hyperscope WebGPU shadow").await {
            Ok((device, adapter)) => {
                let pipelines = device
                    .create_offscreen_diagnostic_patch_render_pipelines()
                    .map_err(|error| error.to_string())
                    .map_err(|error| BACKEND.with(|slot| slot.borrow_mut().fail(error)))?;
                let (pick_pipeline, pick_target, mut pick_error) =
                    create_pick_resources(&device, &pipelines);
                let (resident_root_pipeline, resident_root_error) =
                    match device.create_offscreen_resident_root_render_pipeline() {
                        Ok(pipeline) => (Some(pipeline), None),
                        Err(error) => (None, Some(error.to_string())),
                    };
                let (resident_root_pick_pipeline, resident_root_pick_error) =
                    create_resident_root_pick_resource(
                        &device,
                        resident_root_pipeline.as_ref(),
                        pick_pipeline.as_ref(),
                    );
                if let Some(error) = resident_root_pick_error {
                    pick_error = Some(error);
                }
                let (focus, focus_error) = match device
                    .create_offscreen_focus_pbr_render_resources()
                    .map_err(|error| error.to_string())
                {
                    Ok(focus) => (Some(focus), None),
                    Err(error) => (None, Some(error)),
                };
                let focus_failed = focus_error.is_some();
                BACKEND.with(|slot| {
                    let mut backend = slot.borrow_mut();
                    backend.device = Some(device);
                    backend.adapter = Some(adapter);
                    backend.atlas = None;
                    backend.textures = None;
                    backend.environment = None;
                    backend.model = None;
                    backend.model_source = None;
                    backend.pipelines = Some(pipelines);
                    backend.pick_pipeline = pick_pipeline;
                    backend.pick_target = pick_target;
                    backend.resident_root_pipeline = resident_root_pipeline;
                    backend.resident_root_pick_pipeline = resident_root_pick_pipeline;
                    backend.focus = focus;
                    backend.resident_roots = None;
                    backend.scene = None;
                    backend.scene_source_revision = None;
                    backend.target = None;
                    backend.presentation = None;
                    backend.frame_evidence_requested = false;
                    backend.last_frame_input = None;
                    backend.last_completed_frame_input = None;
                    backend.last_presentation_input = None;
                    backend.presentation_frame_admitted = false;
                    backend.last_presentation_style = None;
                    backend.frame_evidence = None;
                    backend.last_device_lod_epoch = None;
                    backend.last_frame_used_resident_roots = false;
                    backend.last_face_visibility_bits.clear();
                    backend.next_face_visibility_bits.clear();
                    backend.next_morph_weights.clear();
                    backend.device_pose_identity = None;
                    backend.patch_pose_uniforms_ready = false;
                    backend.resident_pose_uniforms_ready = false;
                    if resident_root_error.is_some() {
                        backend.resident_root_fallbacks =
                            backend.resident_root_fallbacks.saturating_add(1);
                    }
                    backend.last_resident_root_error = resident_root_error;
                    backend.last_focus_error = focus_error;
                    backend.last_pick_error = pick_error;
                    if focus_failed {
                        backend.focus_fallbacks = backend.focus_fallbacks.saturating_add(1);
                    }
                    backend.last_logical_submission = RenderSubmissionStats::default();
                    backend.state = "ready";
                    backend.last_error = None;
                });
            }
            Err(error) => {
                return Err(BACKEND.with(|slot| slot.borrow_mut().fail(error)));
            }
        }
    }
    Ok(diagnostics())
}

/// Claim an application-selected, previously unclaimed browser canvas and
/// initialize the live triangle-diagnostic presentation pipelines on a
/// compatible device.
/// A headless device cannot be promoted after the fact because its adapter was
/// not selected against this surface.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn initialize_presentation(
    canvas: web_sys::HtmlCanvasElement,
    size: [u32; 2],
) -> Result<WebGpuBackendDiagnostics, String> {
    let should_request = BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        match backend.state {
            "ready" if backend.presentation.is_some() => return Ok(false),
            "ready" => {
                return Err(
                    "headless WebGPU backend is already initialized; presentation requires a fresh startup"
                        .to_string(),
                );
            }
            "initializing" => {
                return Err("WebGPU backend initialization is already in progress".to_string());
            }
            _ => {}
        }
        backend.state = "initializing";
        backend.initialization_attempts = backend.initialization_attempts.saturating_add(1);
        backend.last_error = None;
        Ok(true)
    })?;
    if !should_request {
        return Ok(diagnostics());
    }

    let request = LodClassifierDevice::request_canvas_presentation(
        canvas,
        size,
        "Hyperscope WebGPU presentation",
    )
    .await;
    let (device, adapter, presentation) = match request {
        Ok(request) => request,
        Err(error) => return Err(BACKEND.with(|slot| slot.borrow_mut().fail(error))),
    };
    let pipelines = device
        .create_diagnostic_patch_render_pipelines(
            presentation.color_format(),
            Some(presentation.depth_format()),
            1,
        )
        .map_err(|error| error.to_string())
        .map_err(|error| BACKEND.with(|slot| slot.borrow_mut().fail(error)))?;
    let (pick_pipeline, pick_target, mut pick_error) = create_pick_resources(&device, &pipelines);
    let (resident_root_pipeline, resident_root_error) = match device
        .create_resident_root_render_pipeline(
            presentation.color_format(),
            Some(presentation.depth_format()),
            1,
        ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let (resident_root_pick_pipeline, resident_root_pick_error) =
        create_resident_root_pick_resource(
            &device,
            resident_root_pipeline.as_ref(),
            pick_pipeline.as_ref(),
        );
    if let Some(error) = resident_root_pick_error {
        pick_error = Some(error);
    }
    let (focus, focus_error) = match device
        .create_focus_pbr_render_resources(presentation.color_format())
        .map_err(|error| error.to_string())
    {
        Ok(focus) => (Some(focus), None),
        Err(error) => (None, Some(error)),
    };
    let focus_failed = focus_error.is_some();
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        backend.device = Some(device);
        backend.adapter = Some(adapter);
        backend.atlas = None;
        backend.textures = None;
        backend.environment = None;
        backend.model = None;
        backend.model_source = None;
        backend.pipelines = Some(pipelines);
        backend.pick_pipeline = pick_pipeline;
        backend.pick_target = pick_target;
        backend.resident_root_pipeline = resident_root_pipeline;
        backend.resident_root_pick_pipeline = resident_root_pick_pipeline;
        backend.focus = focus;
        backend.resident_roots = None;
        backend.scene = None;
        backend.scene_source_revision = None;
        backend.target = None;
        backend.presentation = Some(presentation);
        backend.frame_evidence_requested = false;
        backend.last_frame_input = None;
        backend.last_completed_frame_input = None;
        backend.last_presentation_input = None;
        backend.presentation_frame_admitted = false;
        backend.last_presentation_style = None;
        backend.frame_evidence = None;
        backend.last_device_lod_epoch = None;
        backend.last_frame_used_resident_roots = false;
        backend.last_face_visibility_bits.clear();
        backend.next_face_visibility_bits.clear();
        backend.next_morph_weights.clear();
        backend.device_pose_identity = None;
        backend.patch_pose_uniforms_ready = false;
        backend.resident_pose_uniforms_ready = false;
        if resident_root_error.is_some() {
            backend.resident_root_fallbacks = backend.resident_root_fallbacks.saturating_add(1);
        }
        backend.last_resident_root_error = resident_root_error;
        backend.last_focus_error = focus_error;
        backend.last_pick_error = pick_error;
        if focus_failed {
            backend.focus_fallbacks = backend.focus_fallbacks.saturating_add(1);
        }
        backend.last_logical_submission = RenderSubmissionStats::default();
        backend.state = "ready";
        backend.last_error = None;
    });
    Ok(diagnostics())
}

/// Mirror the incumbent browser-decoded texture table into the ready WebGPU
/// device before JavaScript closes its `ImageBitmap` handles. Missing images
/// remain empty index slots. A rejected replacement leaves the prior table
/// resident and does not disable the WebGL rollback renderer.
#[cfg(target_arch = "wasm32")]
pub(crate) fn replace_image_bitmaps(
    images: &[Option<(web_sys::ImageBitmap, u32, u32)>],
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" {
            return Ok(false);
        }
        let assets = images
            .iter()
            .enumerate()
            .map(|(index, image)| {
                image
                    .as_ref()
                    .map(|(bitmap, wrap_s, wrap_t)| {
                        Ok((
                            texture_asset_descriptor(
                                index,
                                bitmap.width(),
                                bitmap.height(),
                                *wrap_s,
                                *wrap_t,
                            )?,
                            bitmap.clone(),
                        ))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, String>>();
        let assets = match assets {
            Ok(assets) => assets,
            Err(error) => {
                backend.last_error = Some(error.clone());
                return Err(error);
            }
        };
        let result = {
            let WebGpuBackend {
                device,
                pipelines,
                model,
                scene,
                environment,
                resident_root_pipeline,
                focus,
                resident_roots,
                ..
            } = &mut *backend;
            let device = device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let textures = device
                .upload_pbr_image_bitmap_table(&assets)
                .map_err(|error| error.to_string())?;
            let root_candidate = rebuild_resident_root_pbr_scene(
                device,
                model.as_ref(),
                pipelines.as_ref(),
                resident_root_pipeline.as_ref(),
                focus.as_ref(),
                resident_roots.as_ref(),
                scene.as_ref(),
                Some(&textures),
                environment.as_ref(),
            )?;
            if let Some(candidate) = root_candidate.as_ref() {
                let roots = resident_roots
                    .as_ref()
                    .expect("resident PBR rebuild checked root residency");
                device
                    .publish_adaptive_overlay_suppression(
                        &roots.geometry,
                        candidate.overlay.as_ref(),
                    )
                    .map_err(|error| error.to_string())?;
            }
            if let Some(scene) = scene.as_mut() {
                let pipeline = pipelines
                    .as_ref()
                    .and_then(|pipelines| pipelines.get(RenderStyle::Pbr))
                    .ok_or_else(|| "ready WebGPU backend has no PBR pipeline".to_string())?;
                device
                    .replace_patch_render_scene_texture_bindings(pipeline, scene, Some(&textures))
                    .map_err(|error| error.to_string())?;
            }
            let focus_failure = root_candidate
                .as_ref()
                .and_then(|candidate| candidate.focus_failure.clone());
            if let Some(candidate) = root_candidate {
                let roots = resident_roots
                    .as_mut()
                    .expect("resident PBR rebuild checked root residency");
                roots.bindings = candidate.bindings;
                roots.overlay = candidate.overlay;
                roots.focus = candidate.focus;
            }
            Ok::<_, String>((textures, focus_failure))
        };
        match result {
            Ok((textures, focus_failure)) => {
                backend.textures = Some(textures);
                backend.texture_uploads = backend.texture_uploads.saturating_add(1);
                backend.publish_focus_result(focus_failure);
                backend.last_error = None;
                Ok(true)
            }
            Err(error) => {
                backend.last_error = Some(error.clone());
                Err(error)
            }
        }
    })
}

/// Mirror the incumbent RGBA32F IBL payload into one retained WebGPU
/// environment epoch. Validation and both cube uploads complete before either
/// the asset or a replacement scene bind group is published.
pub(crate) fn replace_environment_maps(
    prefiltered: &[f32],
    prefiltered_face_size: u32,
    irradiance: &[f32],
    irradiance_face_size: u32,
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" {
            return Ok(false);
        }
        let prefiltered_mip_count = prefiltered_face_size
            .checked_ilog2()
            .map_or(0, |maximum_mip| maximum_mip + 1);
        let descriptor = EnvironmentMapDescriptor {
            prefiltered_face_size,
            prefiltered_mip_count,
            irradiance_face_size,
        };
        let asset = match EnvironmentMapAsset::new(descriptor, prefiltered, irradiance) {
            Ok(asset) => asset,
            Err(error) => {
                let error = error.to_string();
                backend.last_error = Some(error.clone());
                return Err(error);
            }
        };
        let result = {
            let WebGpuBackend {
                device,
                pipelines,
                model,
                scene,
                textures,
                resident_root_pipeline,
                focus,
                resident_roots,
                ..
            } = &mut *backend;
            let device = device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let environment = device
                .upload_pbr_environment_map(asset)
                .map_err(|error| error.to_string())?;
            let root_candidate = rebuild_resident_root_pbr_scene(
                device,
                model.as_ref(),
                pipelines.as_ref(),
                resident_root_pipeline.as_ref(),
                focus.as_ref(),
                resident_roots.as_ref(),
                scene.as_ref(),
                textures.as_ref(),
                Some(&environment),
            )?;
            if let Some(candidate) = root_candidate.as_ref() {
                let roots = resident_roots
                    .as_ref()
                    .expect("resident PBR rebuild checked root residency");
                device
                    .publish_adaptive_overlay_suppression(
                        &roots.geometry,
                        candidate.overlay.as_ref(),
                    )
                    .map_err(|error| error.to_string())?;
            }
            if let Some(scene) = scene.as_mut() {
                let pipeline = pipelines
                    .as_ref()
                    .and_then(|pipelines| pipelines.get(RenderStyle::Pbr))
                    .ok_or_else(|| "ready WebGPU backend has no PBR pipeline".to_string())?;
                device
                    .replace_patch_render_scene_environment_bindings(
                        pipeline,
                        scene,
                        Some(&environment),
                    )
                    .map_err(|error| error.to_string())?;
            }
            let focus_failure = root_candidate
                .as_ref()
                .and_then(|candidate| candidate.focus_failure.clone());
            if let Some(candidate) = root_candidate {
                let roots = resident_roots
                    .as_mut()
                    .expect("resident PBR rebuild checked root residency");
                roots.bindings = candidate.bindings;
                roots.overlay = candidate.overlay;
                roots.focus = candidate.focus;
            }
            Ok::<_, String>((environment, focus_failure))
        };
        match result {
            Ok((environment, focus_failure)) => {
                backend.environment = Some(environment);
                backend.environment_uploads = backend.environment_uploads.saturating_add(1);
                backend.publish_focus_result(focus_failure);
                backend.last_error = None;
                Ok(true)
            }
            Err(error) => {
                backend.last_error = Some(error.clone());
                Err(error)
            }
        }
    })
}

/// Replace packed atlas and any classifier model that embeds its lookup as one
/// coherent residency epoch. A disabled/initializing backend is inert.
pub(crate) fn replace_atlas(
    patches: &[u32],
    barycentrics: &[f32],
    triangle_indices: &[u32],
    line_indices: &[u32],
    lookup: &LodAtlasLookup,
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" {
            return Ok(false);
        }
        let result = {
            let device = backend
                .device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let atlas = device
                .upload_packed_patch_atlas(patches, barycentrics, triangle_indices, line_indices)
                .map_err(|error| error.to_string())?;
            let model = backend
                .model_source
                .as_ref()
                .map(|source| device.upload_model(source.clone(), lookup))
                .transpose()
                .map_err(|error| error.to_string())?;
            Ok::<_, String>((atlas, model))
        };
        match result {
            Ok((atlas, model)) => {
                backend.atlas = Some(atlas);
                backend.model = model;
                backend.resident_roots = None;
                backend.last_resident_root_error = None;
                backend.scene = None;
                backend.scene_source_revision = None;
                backend.frame_evidence = None;
                backend.last_frame_input = None;
                backend.last_completed_frame_input = None;
                backend.last_presentation_input = None;
                backend.presentation_frame_admitted = false;
                backend.last_device_lod_epoch = None;
                backend.last_face_visibility_bits.clear();
                backend.next_face_visibility_bits.clear();
                backend.next_morph_weights.clear();
                backend.device_pose_identity = None;
                backend.patch_pose_uniforms_ready = false;
                backend.resident_pose_uniforms_ready = false;
                backend.atlas_uploads = backend.atlas_uploads.saturating_add(1);
                if backend.model.is_some() {
                    backend.model_uploads = backend.model_uploads.saturating_add(1);
                }
                backend.last_error = None;
                Ok(true)
            }
            Err(error) => Err(backend.fail(error)),
        }
    })
}

/// Replace the immutable source model only after packed atlas residency exists.
/// The prior model remains live if validation or GPU allocation fails.
pub(crate) fn replace_model(
    source: PreparedLodModel,
    lookup: &LodAtlasLookup,
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" {
            return Ok(false);
        }
        if backend.atlas.is_none() {
            return Err(backend.fail("WebGPU model upload requires packed atlas residency"));
        }
        let model = backend
            .device
            .as_ref()
            .ok_or_else(|| "ready WebGPU backend has no device".to_string())?
            .upload_model(source.clone(), lookup)
            .map_err(|error| error.to_string());
        match model {
            Ok(model) => {
                backend.model = Some(model);
                backend.model_source = Some(source);
                backend.resident_roots = None;
                backend.last_resident_root_error = None;
                backend.scene = None;
                backend.scene_source_revision = None;
                backend.frame_evidence = None;
                backend.last_frame_input = None;
                backend.last_completed_frame_input = None;
                backend.last_presentation_input = None;
                backend.presentation_frame_admitted = false;
                backend.last_device_lod_epoch = None;
                backend.last_face_visibility_bits.clear();
                backend.next_face_visibility_bits.clear();
                backend.next_morph_weights.clear();
                backend.device_pose_identity = None;
                backend.patch_pose_uniforms_ready = false;
                backend.resident_pose_uniforms_ready = false;
                backend.model_uploads = backend.model_uploads.saturating_add(1);
                backend.last_error = None;
                Ok(true)
            }
            Err(error) => Err(backend.fail(error)),
        }
    })
}

/// Classify and reconcile one complete current-view LOD epoch entirely on the
/// WebGPU device. The packed resident result remains owned by the model for a
/// later presentation frame; no staging copy or map is created here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_lod(
    pose_identity: RenderPoseIdentity,
    legacy_mobius: [f32; 16],
    packed_subjects: &[f32],
    density: f32,
    pixel_floor: f32,
    max_lod: f32,
    view_projection: [f32; 16],
    viewport: [f32; 2],
    joint_matrices: &[f32],
    morph_weights: &[f32],
    grading: FaceLodGrading,
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" || backend.model.is_none() {
            return Ok(false);
        }
        if !joint_matrices.len().is_multiple_of(16) {
            return Err("WebGPU LOD joint matrix payload is malformed".to_string());
        }
        let num_joints = u32::try_from(joint_matrices.len() / 16)
            .map_err(|_| "WebGPU LOD joint count exceeds u32".to_string())?;
        let source = backend
            .model_source
            .as_ref()
            .ok_or_else(|| "WebGPU LOD dispatch requires immutable model source".to_string())?;
        let dispatch = prepare_lod_dispatch_state(
            packed_subjects,
            &source.residency,
            source.residency.num_faces,
            legacy_mobius,
        );
        let metrics = WgslLodDispatchMetrics {
            view_projection,
            density,
            pixel_floor,
            max_lod,
            viewport,
            num_joints,
        };
        let pose_upload = if backend.device_pose_identity == Some(pose_identity) {
            PoseUploadPolicy::Reuse
        } else {
            PoseUploadPolicy::Publish
        };
        if pose_upload == PoseUploadPolicy::Publish {
            backend.device_pose_identity = None;
        }
        backend.last_device_lod_epoch = None;
        backend.last_frame_input = None;
        let epoch_result = (|| {
            let WebGpuBackend { device, model, .. } = &mut *backend;
            let device = device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let model = model
                .as_mut()
                .ok_or_else(|| "ready WebGPU backend has no model".to_string())?;
            device.invalidate_resident_lod(model);
            let resident = device
                .classify_and_reconcile_on_device(
                    model,
                    &dispatch,
                    metrics,
                    LodPose {
                        joint_matrices,
                        morph_weights,
                    },
                    pose_upload,
                    grading,
                )
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(resident.classification_epoch())
        })();
        let epoch = match epoch_result {
            Ok(epoch) => epoch,
            Err(error) => {
                backend.last_error = Some(error.clone());
                return Err(error);
            }
        };
        backend.device_lod_dispatches = backend.device_lod_dispatches.saturating_add(1);
        if pose_upload == PoseUploadPolicy::Publish {
            backend.classifier_pose_uploads = backend.classifier_pose_uploads.saturating_add(1);
        } else {
            backend.classifier_pose_reuses = backend.classifier_pose_reuses.saturating_add(1);
        }
        backend.device_pose_identity = Some(pose_identity);
        backend.last_device_lod_epoch = Some(epoch);
        backend.last_frame_input = None;
        backend.last_error = None;
        Ok(true)
    })
}

/// Retire only the camera/pose-dependent device LOD epoch. This exposes the
/// CPU visibility adapter again without disturbing immutable WebGPU residency.
pub(crate) fn invalidate_lod() {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if let (Some(device), Some(model)) = (backend.device.as_ref(), backend.model.as_ref()) {
            device.invalidate_resident_lod(model);
        }
        backend.last_device_lod_epoch = None;
        backend.last_frame_input = None;
    });
}

pub(crate) fn needs_scene(source_revision: u64, scene: &ValidatedRenderScene) -> bool {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        backend.state == "ready"
            && backend.model.is_some()
            && backend.atlas.is_some()
            && (backend.scene_source_revision != Some(source_revision)
                || backend.scene.as_ref().is_none_or(|retained| {
                    !retained.validated_scene().shares_snapshot_with(scene)
                }))
    })
}

/// Whether the live adapter has enough immutable residency to consume the
/// shared backend-neutral scene and command plan. This says nothing about the
/// current render style; unsupported styles still fall through atomically.
pub(crate) fn frame_contract_required() -> bool {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        backend.state == "ready" && backend.model.is_some() && backend.atlas.is_some()
    })
}

/// Whether the selected WebGPU device owns a browser presentation surface.
/// Headless parity/pick devices must not turn ordinary PBR rendering into a
/// continuous shadow workload; an explicit evidence request remains separate.
pub(crate) fn live_presentation_requested() -> bool {
    BACKEND.with(|slot| slot.borrow().presentation.is_some())
}

pub(crate) fn frame_evidence_ready() -> bool {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        backend.state == "ready" && backend.device.is_some()
    })
}

pub(crate) fn pbr_evidence_ready() -> bool {
    frame_evidence_ready() && BACKEND.with(|slot| slot.borrow().environment.is_some())
}

pub(crate) fn focus_evidence_prerequisites_ready() -> bool {
    frame_evidence_ready()
        && BACKEND.with(|slot| {
            let backend = slot.borrow();
            backend.focus.is_some() && backend.environment.is_some()
        })
}

pub(crate) fn record_frame_prerequisite_failure(error: impl ToString) {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state == "ready" {
            let error = error.to_string();
            backend.frame_failures = backend.frame_failures.saturating_add(1);
            backend.presentation_frame_admitted = false;
            backend.last_frame_failure = Some(error.clone());
            backend.last_error = Some(error);
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn build_resident_root_backend(
    device: &LodClassifierDevice,
    model: &LodClassifierModel,
    atlas: &PackedPatchAtlas,
    root_pipeline: &ResidentRootRenderPipeline,
    overlay_pipelines: &DiagnosticPatchRenderPipelines,
    focus_backend: Option<&FocusPbrRenderResources>,
    domains: ResidentRootDrawDomains,
    scene: &RenderSceneSnapshot,
    source_instances: &[f32],
    textures: Option<&PbrTextureTable>,
    environment: Option<&PbrEnvironmentMap>,
) -> Result<(ResidentRootBackend, Option<String>), String> {
    let preparation = device
        .upload_resident_root_preparation_scene(model, scene, source_instances)
        .map_err(|error| error.to_string())?;
    let geometry = device
        .upload_resident_geometry_bucket_scene(model, atlas, preparation.draw_domains())
        .map_err(|error| error.to_string())?;
    let bindings = device
        .create_resident_root_render_bindings_with_pbr(
            root_pipeline,
            &preparation,
            &geometry,
            scene,
            textures,
            environment,
        )
        .map_err(|error| error.to_string())?;
    let overlay = build_adaptive_overlay(
        device,
        model,
        overlay_pipelines,
        &preparation,
        &bindings,
        scene,
        textures,
        environment,
    )?;
    let (focus, focus_failure) = match build_resident_root_focus_backend(
        device,
        model,
        focus_backend,
        &preparation,
        &geometry,
        scene,
        textures,
        environment,
    ) {
        Ok(focus) => (focus, None),
        Err(error) => (None, Some(error)),
    };
    device
        .publish_adaptive_overlay_suppression(&geometry, overlay.as_ref())
        .map_err(|error| error.to_string())?;
    Ok((
        ResidentRootBackend {
            domains,
            preparation,
            geometry,
            bindings,
            overlay,
            focus,
        },
        focus_failure,
    ))
}

#[allow(clippy::too_many_arguments)]
fn rebuild_resident_root_pbr_scene(
    device: &LodClassifierDevice,
    model: Option<&LodClassifierModel>,
    pipelines: Option<&DiagnosticPatchRenderPipelines>,
    pipeline: Option<&ResidentRootRenderPipeline>,
    focus_backend: Option<&FocusPbrRenderResources>,
    roots: Option<&ResidentRootBackend>,
    scene: Option<&PatchRenderScene>,
    textures: Option<&PbrTextureTable>,
    environment: Option<&PbrEnvironmentMap>,
) -> Result<Option<ResidentRootPbrCandidate>, String> {
    let Some(roots) = roots else {
        return Ok(None);
    };
    let pipeline = pipeline
        .ok_or_else(|| "resident PBR binding rebuild lost its root pipeline".to_string())?;
    let model =
        model.ok_or_else(|| "resident PBR binding rebuild lost its model residency".to_string())?;
    let pipelines = pipelines
        .ok_or_else(|| "resident PBR binding rebuild lost its pipeline family".to_string())?;
    let scene =
        scene.ok_or_else(|| "resident PBR binding rebuild lost its render scene".to_string())?;
    let bindings = device
        .create_resident_root_render_bindings_with_pbr(
            pipeline,
            &roots.preparation,
            &roots.geometry,
            scene.scene(),
            textures,
            environment,
        )
        .map_err(|error| error.to_string())?;
    let overlay = build_adaptive_overlay(
        device,
        model,
        pipelines,
        &roots.preparation,
        &bindings,
        scene.scene(),
        textures,
        environment,
    )?;
    let (focus, focus_failure) = match build_resident_root_focus_backend(
        device,
        model,
        focus_backend,
        &roots.preparation,
        &roots.geometry,
        scene.scene(),
        textures,
        environment,
    ) {
        Ok(focus) => (focus, None),
        Err(error) => (None, Some(error)),
    };
    Ok(Some(ResidentRootPbrCandidate {
        bindings,
        overlay,
        focus,
        focus_failure,
    }))
}

#[allow(clippy::too_many_arguments)]
fn build_resident_root_focus_backend(
    device: &LodClassifierDevice,
    model: &LodClassifierModel,
    focus_backend: Option<&FocusPbrRenderResources>,
    preparation: &ResidentRootPreparationScene,
    geometry: &ResidentGeometryBucketScene,
    scene: &RenderSceneSnapshot,
    textures: Option<&PbrTextureTable>,
    environment: Option<&PbrEnvironmentMap>,
) -> Result<Option<ResidentRootFocusBackend>, String> {
    let Some(focus_backend) = focus_backend else {
        return Ok(None);
    };
    let bindings = device
        .create_resident_root_render_bindings_with_pbr(
            focus_backend.root_pipeline(),
            preparation,
            geometry,
            scene,
            textures,
            environment,
        )
        .map_err(|error| error.to_string())?;
    let overlay = device
        .upload_focus_adaptive_overlay_scene_with_pbr_resources_for_roots(
            focus_backend.overlay_pipeline(),
            model,
            preparation,
            &bindings,
            scene,
            textures,
            environment,
        )
        .map_err(|error| error.to_string())?;
    if overlay
        .as_ref()
        .is_some_and(|overlay| !overlay.shares_global_frame_with(&bindings))
    {
        return Err("resident focus overlay lost aggregate-global frame residency".to_string());
    }
    Ok(Some(ResidentRootFocusBackend { bindings, overlay }))
}

fn build_adaptive_overlay(
    device: &LodClassifierDevice,
    model: &LodClassifierModel,
    pipelines: &DiagnosticPatchRenderPipelines,
    preparation: &ResidentRootPreparationScene,
    root_bindings: &ResidentRootRenderBindings,
    scene: &RenderSceneSnapshot,
    textures: Option<&PbrTextureTable>,
    environment: Option<&PbrEnvironmentMap>,
) -> Result<Option<AdaptiveOverlayScene>, String> {
    let layout_pipeline = pipelines
        .get(RenderStyle::Pbr)
        .ok_or_else(|| "WebGPU diagnostic pipeline family lost PBR".to_string())?;
    let overlay = device
        .upload_adaptive_overlay_scene_with_pbr_resources_for_roots(
            layout_pipeline,
            model,
            preparation,
            root_bindings,
            scene,
            textures,
            environment,
        )
        .map_err(|error| error.to_string())?;
    if overlay
        .as_ref()
        .is_some_and(|overlay| !overlay.shares_global_frame_with(root_bindings))
    {
        return Err("resident overlay lost aggregate-global frame residency".to_string());
    }
    Ok(overlay)
}

/// Publish one device-side render scene derived from shared backend-neutral
/// extraction. Shape-compatible epochs update retained buffers in place; a
/// cardinality change allocates an atomic replacement. The previous scene
/// remains usable if packing or allocation fails.
pub(crate) fn replace_scene(
    source_revision: u64,
    scene: ValidatedRenderScene,
    source_instances: &[f32],
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" {
            return Ok(false);
        }
        let next_revision = backend.scene_uploads.saturating_add(1);
        let resident_root_domains = backend
            .model_source
            .as_ref()
            .ok_or_else(|| "WebGPU resident roots require immutable model residency".to_string())
            .and_then(|source| {
                resident_root_render_domains(scene.snapshot(), source.residency.num_faces)
            });
        let resident_root_candidate = match (
            backend.device.as_ref(),
            backend.model.as_ref(),
            backend.atlas.as_ref(),
            backend.resident_root_pipeline.as_ref(),
            backend.pipelines.as_ref(),
            resident_root_domains,
        ) {
            (_, _, _, _, _, Ok(None)) => Ok(ResidentRootSceneCandidate::Unavailable),
            (_, _, _, None, _, Ok(Some(_))) => Ok(ResidentRootSceneCandidate::Unavailable),
            (
                Some(device),
                Some(model),
                _,
                Some(root_pipeline),
                Some(pipelines),
                Ok(Some(domains)),
            ) if backend
                .resident_roots
                .as_ref()
                .is_some_and(|resident| resident.domains == domains) =>
            {
                // Source-face records are immutable for one model residency;
                // atlas/model replacement clears this aggregate. Exact root
                // domains therefore distinguish authored state changes from
                // CPU-only LOD bucket churn without retaining another copy of
                // the 208-byte source record per face.
                let retained = backend
                    .resident_roots
                    .as_ref()
                    .expect("resident root reuse guard checked presence");
                let bindings = device
                    .create_resident_root_render_bindings_with_pbr(
                        root_pipeline,
                        &retained.preparation,
                        &retained.geometry,
                        scene.snapshot(),
                        backend.textures.as_ref(),
                        backend.environment.as_ref(),
                    )
                    .map_err(|error| error.to_string())?;
                let overlay = build_adaptive_overlay(
                    device,
                    model,
                    pipelines,
                    &retained.preparation,
                    &bindings,
                    scene.snapshot(),
                    backend.textures.as_ref(),
                    backend.environment.as_ref(),
                )?;
                let (focus, focus_failure) = match build_resident_root_focus_backend(
                    device,
                    model,
                    backend.focus.as_ref(),
                    &retained.preparation,
                    &retained.geometry,
                    scene.snapshot(),
                    backend.textures.as_ref(),
                    backend.environment.as_ref(),
                ) {
                    Ok(focus) => (focus, None),
                    Err(error) => (None, Some(error)),
                };
                Ok(ResidentRootSceneCandidate::Reuse(Box::new(
                    ResidentRootReuseCandidate {
                        bindings,
                        overlay,
                        focus,
                        focus_failure,
                    },
                )))
            }
            (
                Some(device),
                Some(model),
                Some(atlas),
                Some(root_pipeline),
                Some(overlay_pipelines),
                Ok(Some(domains)),
            ) => build_resident_root_backend(
                device,
                model,
                atlas,
                root_pipeline,
                overlay_pipelines,
                backend.focus.as_ref(),
                domains,
                scene.snapshot(),
                source_instances,
                backend.textures.as_ref(),
                backend.environment.as_ref(),
            )
            .map(
                |(backend, focus_failure)| ResidentRootSceneCandidate::Replace {
                    backend: Box::new(backend),
                    focus_failure,
                },
            ),
            (_, _, _, _, _, Err(error)) => Err(error),
            _ => Err("WebGPU resident roots require model, atlas, and pipeline residency".into()),
        };
        let (resident_root_candidate, resident_root_failure) = match resident_root_candidate {
            Ok(candidate) => (candidate, None),
            Err(error) => (ResidentRootSceneCandidate::Unavailable, Some(error)),
        };
        let update_result = {
            let WebGpuBackend {
                device,
                model,
                scene: retained,
                pipelines,
                textures,
                ..
            } = &mut *backend;
            let pipeline = pipelines
                .as_ref()
                .and_then(|pipelines| pipelines.get(RenderStyle::Pbr));
            match (device.as_ref(), pipeline, model.as_ref(), retained.as_mut()) {
                (Some(device), Some(pipeline), Some(model), Some(retained)) => device
                    .update_validated_patch_render_scene_in_place(
                        pipeline,
                        model,
                        retained,
                        scene,
                        source_instances,
                        textures.as_ref(),
                    )
                    .map_err(|error| error.to_string()),
                _ => Ok(PatchRenderSceneUpdate::ShapeChanged(scene)),
            }
        };
        let scene = match update_result {
            Ok(PatchRenderSceneUpdate::Updated) => {
                backend.scene_source_revision = Some(source_revision);
                backend.scene_uploads = next_revision;
                backend.scene_updates = backend.scene_updates.saturating_add(1);
                backend.frame_evidence = None;
                backend.last_frame_input = None;
                backend.last_completed_frame_input = None;
                backend.last_presentation_input = None;
                backend.presentation_frame_admitted = false;
                backend.next_morph_weights.clear();
                backend.patch_pose_uniforms_ready = false;
                backend.publish_resident_scene(resident_root_candidate, resident_root_failure);
                backend.last_error = None;
                return Ok(true);
            }
            Ok(PatchRenderSceneUpdate::ShapeChanged(scene)) => scene,
            Err(error) => {
                backend.last_error = Some(error.clone());
                return Err(error);
            }
        };
        let result = {
            let device = backend
                .device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let pipeline = backend
                .pipelines
                .as_ref()
                .and_then(|pipelines| pipelines.get(RenderStyle::Pbr))
                .ok_or_else(|| "ready WebGPU backend has no PBR pipeline".to_string())?;
            let model = backend
                .model
                .as_ref()
                .ok_or_else(|| "WebGPU render scene requires model residency".to_string())?;
            let mut candidate = device
                .upload_validated_patch_render_scene(
                    pipeline,
                    model,
                    scene,
                    source_instances,
                    backend.textures.as_ref(),
                )
                .map_err(|error| error.to_string())?;
            if let Some(environment) = backend.environment.as_ref() {
                device
                    .replace_patch_render_scene_environment_bindings(
                        pipeline,
                        &mut candidate,
                        Some(environment),
                    )
                    .map_err(|error| error.to_string())?;
            }
            Ok::<_, String>(candidate)
        };
        match result {
            Ok(scene) => {
                backend.scene = Some(scene);
                backend.scene_source_revision = Some(source_revision);
                backend.scene_uploads = next_revision;
                backend.scene_rebuilds = backend.scene_rebuilds.saturating_add(1);
                backend.frame_evidence = None;
                backend.last_frame_input = None;
                backend.last_completed_frame_input = None;
                backend.last_presentation_input = None;
                backend.presentation_frame_admitted = false;
                backend.last_face_visibility_bits.clear();
                backend.next_face_visibility_bits.clear();
                backend.next_morph_weights.clear();
                backend.patch_pose_uniforms_ready = false;
                backend.publish_resident_scene(resident_root_candidate, resident_root_failure);
                backend.last_error = None;
                Ok(true)
            }
            Err(error) => {
                backend.last_error = Some(error.clone());
                Err(error)
            }
        }
    })
}

/// Execute one current live frame into either the retained offscreen parity
/// target or the explicitly selected presentation surface. Neither path reads
/// pixels back during ordinary rendering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_frame(
    source_render_call: u64,
    frame: &RenderFrame,
    face_visibility: &[bool],
    joint_matrices: &[f32],
    morph_weights: &[f32],
) -> Result<LiveFrameDisposition, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        let style = frame.style;
        let view = frame.view;
        let options = frame.options;
        if backend.state != "ready" {
            return Ok(backend.incumbent_required());
        }
        let frame_evidence_requested = std::mem::take(&mut backend.frame_evidence_requested);
        let destination = frame_destination(
            backend.presentation.is_some(),
            frame_evidence_requested,
        );
        let presentation_frame = destination == FrameDestination::Presentation;
        // Atlas/model/scene residency arrives asynchronously during ordinary
        // application startup. Frames before the first coherent scene are
        // inert lifecycle gaps, not failed render attempts.
        let Some(scene) = backend.scene.as_ref() else {
            return Ok(backend.incumbent_required());
        };
        let focus_frame = options.focus_postprocess.is_some();
        let scene_supported = if focus_frame {
            style == RenderStyle::Pbr && scene.supports_resident_focus_pbr_frame(options)
        } else {
            scene.supports_resident_patch_presentation_frame(style, options)
        };
        if !scene_supported
            || (focus_frame
                && (backend.focus.is_none()
                    || backend
                        .resident_roots
                        .as_ref()
                        .is_none_or(|roots| roots.focus.is_none())))
        {
            return Ok(backend.incumbent_required());
        }
        if view.viewport[0] == 0 || view.viewport[1] == 0 {
            return Ok(backend.incumbent_required());
        }
        backend.frame_attempts = backend.frame_attempts.saturating_add(1);
        let frame_revision = frame.revision;

        let mut face_visibility_bits = std::mem::take(&mut backend.next_face_visibility_bits);
        face_visibility_bits.clear();
        let mut effective_morph_weights = std::mem::take(&mut backend.next_morph_weights);
        effective_morph_weights.clear();
        let frame_validation = backend
            .scene
            .as_ref()
            .ok_or_else(|| "WebGPU render frame requires scene residency".to_string())
            .and_then(|scene| {
                frame
                    .execution(scene.scene())
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            });
        if let Err(error) = frame_validation {
            return Err(backend.reject_frame(
                face_visibility_bits,
                effective_morph_weights,
                error,
            ));
        }
        let (resident_faces, resident_morph_targets) = match backend.model_source.as_ref() {
            Some(model) => (model.residency.num_faces, model.model.num_morph_targets),
            None => {
                return Err(backend.reject_frame(
                    face_visibility_bits,
                    effective_morph_weights,
                    "WebGPU render frame requires immutable model source",
                ));
            }
        };
        let device_lod_epoch = backend
            .device
            .as_ref()
            .zip(backend.model.as_ref())
            .and_then(|(device, model)| device.latest_resident_lod(model))
            .map(|resident| resident.classification_epoch());
        let resident_root_frame = device_lod_epoch.is_some()
            && backend.resident_roots.as_ref().is_some_and(|roots| {
                if focus_frame {
                    roots.focus.as_ref().is_some_and(|focus| {
                        let adaptive_layer_supported = focus
                            .overlay
                            .as_ref()
                            .is_none_or(AdaptiveOverlayScene::supports_resident_basic_pbr);
                        focus
                            .bindings
                            .supports_resident_root_frame(style, adaptive_layer_supported)
                    })
                } else {
                    let adaptive_layer_supported = roots
                        .overlay
                        .as_ref()
                        .is_none_or(AdaptiveOverlayScene::supports_resident_basic_pbr);
                    roots
                        .bindings
                        .supports_resident_root_frame(style, adaptive_layer_supported)
                }
            });
        if focus_frame && !resident_root_frame {
            return Ok(backend.incumbent_required());
        }
        if device_lod_epoch.is_none() {
            if face_visibility.len() != resident_faces {
                return Err(backend.reject_frame(
                    face_visibility_bits,
                    effective_morph_weights,
                    format!(
                        "WebGPU face visibility has {} values; expected {resident_faces}",
                        face_visibility.len(),
                    ),
                ));
            }
            face_visibility_bits.resize(resident_faces.div_ceil(32), 0);
            for (face, &visible) in face_visibility.iter().enumerate() {
                if visible {
                    face_visibility_bits[face / 32] |= 1 << (face % 32);
                }
            }
        }
        if morph_weights.len() > resident_morph_targets {
            return Err(backend.reject_frame(
                face_visibility_bits,
                effective_morph_weights,
                "WebGPU morph weight payload exceeds resident targets",
            ));
        }

        if !joint_matrices.len().is_multiple_of(16) {
            return Err(backend.reject_frame(
                face_visibility_bits,
                effective_morph_weights,
                "WebGPU joint matrix payload is malformed",
            ));
        }
        let num_joints = match u32::try_from(joint_matrices.len() / 16) {
            Ok(num_joints) => num_joints,
            Err(_) => {
                return Err(backend.reject_frame(
                    face_visibility_bits,
                    effective_morph_weights,
                    "WebGPU joint count exceeds u32",
                ));
            }
        };
        let scene_source_revision = backend.scene_source_revision.unwrap_or(0);
        let frame_input = LiveFrameInput {
            source_revision: scene_source_revision,
            device_lod_epoch,
            style,
            view,
            options,
        };
        let preparation_pose_ready = if resident_root_frame {
            backend.resident_pose_uniforms_ready
        } else {
            backend.patch_pose_uniforms_ready
        };
        let pose_upload = render_pose_upload_policy(
            backend.device_pose_identity,
            frame.pose,
            preparation_pose_ready,
        );
        let pose_upload_required = pose_upload != PoseUploadPolicy::Reuse;
        let unchanged = backend.last_frame_input == Some(frame_input)
            && (device_lod_epoch.is_some()
                || backend.last_face_visibility_bits == face_visibility_bits)
            && !pose_upload_required;
        if unchanged {
            backend.next_face_visibility_bits = face_visibility_bits;
            backend.next_morph_weights = effective_morph_weights;
            backend.frames_skipped_unchanged = backend.frames_skipped_unchanged.saturating_add(1);
            if presentation_frame
                && backend.last_presentation_input == Some(frame_input)
                && backend.last_frame_revision != 0
            {
                // No GPU work is necessary, but the retained presentation and
                // pick resources are also an exact rendering of this newer
                // equivalent source call. Advance that coherence witness so a
                // pick between frames is not rejected solely because the
                // browser kept asking us to present identical state.
                backend.last_source_render_call = source_render_call;
                backend.presentation_frame_admitted = true;
                backend.last_frame_failure = None;
                backend.last_error = None;
                return Ok(LiveFrameDisposition::PresentationRetained);
            }
            if presentation_frame {
                backend.presentation_frame_admitted = false;
            }
            return Ok(LiveFrameDisposition::IncumbentRequired);
        }
        if pose_upload.should_publish_dynamic() {
            effective_morph_weights.extend_from_slice(morph_weights);
            effective_morph_weights.resize(resident_morph_targets, 0.0);
        }
        let visibility_upload_required = device_lod_epoch.is_none()
            && backend.last_face_visibility_bits != face_visibility_bits;

        if presentation_frame {
            let resize = {
                let WebGpuBackend {
                    device,
                    presentation,
                    ..
                } = &mut *backend;
                let device = device
                    .as_ref()
                    .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
                presentation
                    .as_mut()
                    .ok_or_else(|| "WebGPU presentation surface disappeared".to_string())?
                    .resize(device.device(), view.viewport)
                    .map_err(|error| error.to_string())
            };
            if let Err(error) = resize {
                return Err(backend.reject_frame(
                    face_visibility_bits,
                    effective_morph_weights,
                    error,
                ));
            }
        } else if backend
            .target
            .as_ref()
            .is_none_or(|target| target.size() != view.viewport)
        {
            let target = backend
                .device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())
                .and_then(|device| {
                    device
                        .create_offscreen_patch_render_target(view.viewport)
                        .map_err(|error| error.to_string())
                });
            let target = match target {
                Ok(target) => target,
                Err(error) => {
                    return Err(backend.reject_frame(
                        face_visibility_bits,
                        effective_morph_weights,
                        error,
                    ));
                }
            };
            backend.target = Some(target);
            backend.frame_evidence = None;
            backend.target_rebuilds = backend.target_rebuilds.saturating_add(1);
        }
        if focus_frame {
            let target_rebuilt = {
                let WebGpuBackend { device, focus, .. } = &mut *backend;
                let device = device
                    .as_ref()
                    .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
                let focus = focus
                    .as_mut()
                    .ok_or_else(|| "WebGPU focus frame lost its pipeline family".to_string())?;
                focus
                    .ensure_target(device, view.viewport)
                    .map_err(|error| error.to_string())
            };
            let target_rebuilt = match target_rebuilt {
                Ok(target_rebuilt) => target_rebuilt,
                Err(error) => {
                    backend.focus_fallbacks = backend.focus_fallbacks.saturating_add(1);
                    backend.last_focus_error = Some(error.clone());
                    return Err(backend.reject_frame(
                        face_visibility_bits,
                        effective_morph_weights,
                        error,
                    ));
                }
            };
            if target_rebuilt {
                backend.focus_target_rebuilds = backend.focus_target_rebuilds.saturating_add(1);
            }
            backend.last_focus_error = None;
        }
        if pose_upload.should_publish_dynamic() {
            backend.device_pose_identity = None;
        }
        if pose_upload.should_publish_preparation() {
            if resident_root_frame {
                backend.resident_pose_uniforms_ready = false;
            } else {
                backend.patch_pose_uniforms_ready = false;
            }
        }
        let prepared_frame = (|| {
            let device = backend
                .device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let model = backend
                .model
                .as_ref()
                .ok_or_else(|| "WebGPU render frame requires model residency".to_string())?;
            let scene = backend
                .scene
                .as_ref()
                .ok_or_else(|| "WebGPU render frame requires scene residency".to_string())?;
            if !resident_root_frame && pose_upload_required {
                device
                    .write_patch_render_pose_state(
                        model,
                        scene,
                        LodPose {
                            joint_matrices,
                            morph_weights: &effective_morph_weights,
                        },
                        num_joints,
                        pose_upload,
                    )
                    .map_err(|error| error.to_string())?;
            }
            if visibility_upload_required {
                device
                    .write_patch_render_face_visibility_bits(scene, &face_visibility_bits)
                    .map_err(|error| error.to_string())?;
            }
            Ok::<_, String>(frame)
        })();
        if prepared_frame.is_ok() && !resident_root_frame {
            match pose_upload {
                PoseUploadPolicy::Publish => {
                    backend.device_pose_identity = Some(frame.pose);
                    backend.patch_pose_uniforms_ready = true;
                    backend.fallback_pose_uploads =
                        backend.fallback_pose_uploads.saturating_add(1);
                }
                PoseUploadPolicy::PublishPreparation => {
                    backend.patch_pose_uniforms_ready = true;
                    backend.fallback_pose_initializations =
                        backend.fallback_pose_initializations.saturating_add(1);
                }
                PoseUploadPolicy::Reuse => {
                    backend.fallback_pose_reuses =
                        backend.fallback_pose_reuses.saturating_add(1);
                }
            }
        }
        let result =
            prepared_frame.and_then(|frame| {
                if presentation_frame {
                    let WebGpuBackend {
                        device,
                        model,
                        pipelines,
                        resident_root_pipeline,
                        focus,
                        resident_roots,
                        scene,
                        atlas,
                        presentation,
                        ..
                    } = &mut *backend;
                    let device = device
                        .as_ref()
                        .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
                    let pipelines = pipelines.as_ref().ok_or_else(|| {
                        "ready WebGPU backend has no diagnostic pipelines".to_string()
                    })?;
                    let scene = scene.as_ref().ok_or_else(|| {
                        "WebGPU render frame requires scene residency".to_string()
                    })?;
                    let atlas = atlas.as_ref().ok_or_else(|| {
                        "WebGPU render frame requires atlas residency".to_string()
                    })?;
                    let presentation = presentation
                        .as_mut()
                        .ok_or_else(|| "WebGPU presentation surface disappeared".to_string())?;
                    let presented = if resident_root_frame {
                        let model = model.as_ref().ok_or_else(|| {
                            "WebGPU resident root frame requires model residency".to_string()
                        })?;
                        let resident = device.latest_resident_lod(model).ok_or_else(|| {
                            "WebGPU resident root frame lost its LOD epoch".to_string()
                        })?;
                        let roots = resident_roots.as_ref().ok_or_else(|| {
                            "WebGPU resident root frame lost its scene residency".to_string()
                        })?;
                        let pose = LodPose {
                            joint_matrices,
                            morph_weights: &effective_morph_weights,
                        };
                        if focus_frame {
                            let focus = focus.as_ref().ok_or_else(|| {
                                "WebGPU focus frame lost its pipeline family".to_string()
                            })?;
                            let focus_roots = roots.focus.as_ref().ok_or_else(|| {
                                "WebGPU focus frame lost its scene residency".to_string()
                            })?;
                            let focus_target = focus.target().ok_or_else(|| {
                                "WebGPU focus frame lost its postprocess target".to_string()
                            })?;
                            match device.present_focus_resident_adaptive(
                                presentation,
                                frame,
                                scene.scene(),
                                model,
                                &resident,
                                &roots.preparation,
                                &roots.geometry,
                                focus.root_pipeline(),
                                &focus_roots.bindings,
                                focus.overlay_pipeline(),
                                focus_roots.overlay.as_ref(),
                                atlas,
                                focus.postprocess_pipelines(),
                                focus_target,
                                pose,
                                num_joints,
                                pose_upload,
                                true,
                            ) {
                                Ok(SurfacePresentation::Presented(encoding)) => {
                                    Ok(SurfacePresentation::Presented(encoding.scene))
                                }
                                Ok(SurfacePresentation::Skipped(reason)) => {
                                    Ok(SurfacePresentation::Skipped(reason))
                                }
                                Ok(SurfacePresentation::RecreateRequired) => {
                                    Ok(SurfacePresentation::RecreateRequired)
                                }
                                Err(error) => Err(error),
                            }
                        } else {
                            let root_pipeline =
                                resident_root_pipeline.as_ref().ok_or_else(|| {
                                    "WebGPU resident root frame lost its pipeline residency"
                                        .to_string()
                                })?;
                            device.present_resident_adaptive(
                                presentation,
                                frame,
                                scene.scene(),
                                model,
                                &resident,
                                &roots.preparation,
                                &roots.geometry,
                                root_pipeline,
                                &roots.bindings,
                                pipelines,
                                roots.overlay.as_ref(),
                                atlas,
                                pose,
                                num_joints,
                                pose_upload,
                                true,
                            )
                        }
                    } else if let Some(resident) = model
                        .as_ref()
                        .and_then(|model| device.latest_resident_lod(model))
                    {
                        device.present_supported_patch_scene_with_resident_lod_visibility(
                            presentation,
                            frame,
                            pipelines,
                            scene,
                            &resident,
                            atlas,
                            true,
                        )
                    } else {
                        device.present_supported_patch_scene_with_face_visibility(
                            presentation,
                            frame,
                            pipelines,
                            scene,
                            atlas,
                            true,
                        )
                    }
                    .map_err(|error| error.to_string())?;
                    match presented {
                        SurfacePresentation::Presented(encoding) => Ok(Some(encoding)),
                        SurfacePresentation::Skipped(_) => Ok(None),
                        SurfacePresentation::RecreateRequired => Err(
                            "WebGPU presentation surface was lost; reload or use gfx=webgl2"
                                .to_string(),
                        ),
                    }
                } else {
                    let device = backend
                        .device
                        .as_ref()
                        .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
                    let pipelines = backend.pipelines.as_ref().ok_or_else(|| {
                        "ready WebGPU backend has no diagnostic pipelines".to_string()
                    })?;
                    let scene = backend.scene.as_ref().ok_or_else(|| {
                        "WebGPU render frame requires scene residency".to_string()
                    })?;
                    let atlas = backend.atlas.as_ref().ok_or_else(|| {
                        "WebGPU render frame requires atlas residency".to_string()
                    })?;
                    let target = backend
                        .target
                        .as_ref()
                        .ok_or_else(|| "WebGPU render frame requires a target".to_string())?;
                    let resident = backend
                        .model
                        .as_ref()
                        .and_then(|model| device.latest_resident_lod(model));
                    let encoding = if resident_root_frame {
                        let model = backend.model.as_ref().ok_or_else(|| {
                            "WebGPU resident root frame requires model residency".to_string()
                        })?;
                        let resident = resident.ok_or_else(|| {
                            "WebGPU resident root frame lost its LOD epoch".to_string()
                        })?;
                        let roots = backend.resident_roots.as_ref().ok_or_else(|| {
                            "WebGPU resident root frame lost its scene residency".to_string()
                        })?;
                        let pose = LodPose {
                            joint_matrices,
                            morph_weights: &effective_morph_weights,
                        };
                        if focus_frame {
                            let focus = backend.focus.as_ref().ok_or_else(|| {
                                "WebGPU focus frame lost its pipeline family".to_string()
                            })?;
                            let focus_roots = roots.focus.as_ref().ok_or_else(|| {
                                "WebGPU focus frame lost its scene residency".to_string()
                            })?;
                            let focus_target = focus.target().ok_or_else(|| {
                                "WebGPU focus frame lost its postprocess target".to_string()
                            })?;
                            device
                                .render_offscreen_focus_resident_adaptive(
                                    frame,
                                    scene.scene(),
                                    model,
                                    &resident,
                                    &roots.preparation,
                                    &roots.geometry,
                                    focus.root_pipeline(),
                                    &focus_roots.bindings,
                                    focus.overlay_pipeline(),
                                    focus_roots.overlay.as_ref(),
                                    atlas,
                                    focus.postprocess_pipelines(),
                                    focus_target,
                                    target,
                                    pose,
                                    num_joints,
                                    pose_upload,
                                    true,
                                )
                                .map(|encoding| encoding.scene)
                        } else {
                            let root_pipeline = backend
                                .resident_root_pipeline
                                .as_ref()
                                .ok_or_else(|| {
                                    "WebGPU resident root frame lost its pipeline residency"
                                        .to_string()
                                })?;
                            device.render_offscreen_resident_adaptive(
                                frame,
                                scene.scene(),
                                model,
                                &resident,
                                &roots.preparation,
                                &roots.geometry,
                                root_pipeline,
                                &roots.bindings,
                                pipelines,
                                roots.overlay.as_ref(),
                                atlas,
                                target,
                                pose,
                                num_joints,
                                pose_upload,
                                true,
                            )
                        }
                    } else if let Some(resident) = resident {
                        device
                            .render_offscreen_supported_patch_scene_with_resident_lod_visibility_and_webgl_clear(
                                frame,
                                pipelines,
                                scene,
                                &resident,
                                atlas,
                                target,
                                true,
                            )
                    } else if style == RenderStyle::Normals {
                        let pipeline = pipelines.get(RenderStyle::Normals).ok_or_else(|| {
                            "ready WebGPU backend has no normals pipeline".to_string()
                        })?;
                        device.render_offscreen_normals_patch_scene_with_webgl_clear(
                            frame, pipeline, scene, atlas, target, true,
                        )
                    } else if style == RenderStyle::Pbr {
                        device.render_offscreen_supported_patch_scene_with_webgl_clear(
                            frame, pipelines, scene, atlas, target, true,
                        )
                    } else {
                        device.render_offscreen_supported_patch_scene_with_face_visibility(
                            frame, pipelines, scene, atlas, target, true,
                        )
                    };
                    encoding.map(Some).map_err(|error| error.to_string())
                }
            });
        match result {
            Ok(Some(PatchFrameEncoding {
                logical_submission,
                indirect_draw_calls,
                source_instance_count,
            })) => {
                backend.frames_submitted = backend.frames_submitted.saturating_add(1);
                if visibility_upload_required {
                    backend.visibility_uploads = backend.visibility_uploads.saturating_add(1);
                    backend.visibility_upload_bytes = backend
                        .visibility_upload_bytes
                        .saturating_add((face_visibility_bits.len() as u64).saturating_mul(4));
                }
                backend.last_frame_revision = frame_revision;
                backend.last_source_render_call = source_render_call;
                backend.last_indirect_draw_calls = indirect_draw_calls;
                backend.last_source_instances = source_instance_count;
                backend.last_logical_submission = logical_submission;
                backend.last_viewport = view.viewport;
                backend.last_frame_used_resident_roots = resident_root_frame;
                if device_lod_epoch.is_some() {
                    backend.device_lod_frames = backend.device_lod_frames.saturating_add(1);
                    backend.last_face_visibility_bits.clear();
                } else if visibility_upload_required {
                    std::mem::swap(
                        &mut backend.last_face_visibility_bits,
                        &mut face_visibility_bits,
                    );
                }
                backend.next_face_visibility_bits = face_visibility_bits;
                backend.next_morph_weights = effective_morph_weights;
                backend.last_frame_input = Some(frame_input);
                backend.last_completed_frame_input = Some(frame_input);
                if presentation_frame {
                    backend.last_presentation_input = Some(frame_input);
                    backend.presentation_frame_admitted = true;
                    backend.last_presentation_style = Some(style);
                } else {
                    backend.presentation_frame_admitted =
                        backend.last_presentation_input == Some(frame_input);
                    backend.frame_evidence = Some(FrameEvidenceMetadata {
                        frame_revision,
                        source_render_call,
                        viewport: view.viewport,
                        logical_submission,
                        indirect_draw_calls,
                        source_instances: source_instance_count,
                        focus_postprocess: options.focus_postprocess,
                    });
                }
                backend.device_pose_identity = Some(frame.pose);
                if resident_root_frame {
                    backend.resident_root_frames = backend.resident_root_frames.saturating_add(1);
                    backend.resident_pose_uniforms_ready = true;
                    match pose_upload {
                        PoseUploadPolicy::Publish => {
                            backend.resident_pose_uploads =
                                backend.resident_pose_uploads.saturating_add(1);
                        }
                        PoseUploadPolicy::PublishPreparation => {
                            backend.resident_pose_initializations =
                                backend.resident_pose_initializations.saturating_add(1);
                        }
                        PoseUploadPolicy::Reuse => {
                            backend.resident_pose_reuses =
                                backend.resident_pose_reuses.saturating_add(1);
                        }
                    }
                } else {
                    backend.patch_pose_uniforms_ready = true;
                }
                if focus_frame {
                    backend.focus_frames = backend.focus_frames.saturating_add(1);
                    backend.last_focus_error = None;
                }
                backend.last_frame_failure = None;
                backend.last_error = None;
                Ok(if presentation_frame {
                    LiveFrameDisposition::PresentationSubmitted(logical_submission)
                } else {
                    LiveFrameDisposition::ShadowSubmitted(logical_submission)
                })
            }
            Ok(None) => {
                let presentation_matches =
                    backend.last_presentation_input == Some(frame_input);
                backend.next_face_visibility_bits = face_visibility_bits;
                backend.next_morph_weights = effective_morph_weights;
                if presentation_frame
                    && backend.last_frame_revision != 0
                    && presentation_matches
                {
                    backend.last_source_render_call = source_render_call;
                    backend.presentation_frame_admitted = true;
                    backend.last_frame_failure = None;
                    backend.last_error = None;
                    Ok(LiveFrameDisposition::PresentationRetained)
                } else {
                    if presentation_frame {
                        backend.presentation_frame_admitted = false;
                    }
                    Ok(LiveFrameDisposition::IncumbentRequired)
                }
            }
            Err(error) => {
                Err(backend.reject_frame(face_visibility_bits, effective_morph_weights, error))
            }
        }
    })
}

pub(crate) struct StagedWebGpuFrameEvidence {
    pub(crate) frame_revision: u64,
    pub(crate) source_render_call: u64,
    pub(crate) viewport: [u32; 2],
    pub(crate) logical_submission: RenderSubmissionStats,
    pub(crate) indirect_draw_calls: u32,
    pub(crate) source_instances: u32,
    pub(crate) focus_postprocess: Option<quilting_core::render::FocusPostprocessPacket>,
    pub(crate) image: StagedOffscreenImageReadback,
}

/// One opt-in query staged after the latest coherent prepared-patch frame.
/// The readback owns its device resources, so no thread-local backend borrow
/// survives across the asynchronous map.
pub(crate) struct StagedWebGpuPick {
    frame_revision: u64,
    source_render_call: u64,
    readback: StagedPatchPickReadback,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WebGpuPickHit {
    pub(crate) target_epoch: u32,
    pub(crate) packed_node: u32,
    pub(crate) source_face: u32,
    pub(crate) source_barycentric: [f32; 3],
    pub(crate) source_position: [f32; 3],
    pub(crate) output_distance: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WebGpuPickReadback {
    pub(crate) frame_revision: u64,
    pub(crate) source_render_call: u64,
    pub(crate) hit: Option<WebGpuPickHit>,
}

impl StagedWebGpuPick {
    pub(crate) fn frame_revision(&self) -> u64 {
        self.frame_revision
    }

    pub(crate) fn source_render_call(&self) -> u64 {
        self.source_render_call
    }

    pub(crate) async fn read(self) -> Result<WebGpuPickReadback, String> {
        let hit = self
            .readback
            .read()
            .await
            .map_err(|error| error.to_string())?
            .map(|sample| WebGpuPickHit {
                target_epoch: sample.target_epoch,
                packed_node: sample.packed_node,
                source_face: sample.source_face,
                source_barycentric: sample.source_barycentric,
                source_position: sample.source_position,
                output_distance: sample.output_distance,
            });
        Ok(WebGpuPickReadback {
            frame_revision: self.frame_revision,
            source_render_call: self.source_render_call,
            hit,
        })
    }
}

/// Stage a one-pixel query against the latest completed prepared-patch frame.
/// Resident frames consume their current root buckets and sparse overlay;
/// ordinary frames consume their current compacted scene. Neither path reruns
/// LoD, visibility, preparation, or compaction.
pub(crate) fn stage_pick(pixel: [u32; 2], target_epoch: u32) -> Result<StagedWebGpuPick, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        backend.pick_requests = backend.pick_requests.saturating_add(1);
        if backend.state != "ready"
            || backend.last_frame_revision == 0
            || backend.last_completed_frame_input.is_none()
        {
            return Err(backend.reject_pick("WebGPU picking requires one completed coherent frame"));
        }
        let request = match PatchPickRequest::new(backend.last_viewport, pixel, target_epoch) {
            Ok(request) => request,
            Err(error) => return Err(backend.reject_pick(error)),
        };
        let resident_frame = backend.last_frame_used_resident_roots;
        let focus_frame = backend
            .last_completed_frame_input
            .is_some_and(|frame| frame.options.focus_postprocess.is_some());
        let result = (|| {
            let WebGpuBackend {
                device,
                atlas,
                pick_pipeline,
                pick_target,
                resident_root_pick_pipeline,
                resident_roots,
                scene,
                ..
            } = &*backend;
            let device = device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let atlas = atlas
                .as_ref()
                .ok_or_else(|| "WebGPU picking requires atlas residency".to_string())?;
            let pipeline = pick_pipeline
                .as_ref()
                .ok_or_else(|| "WebGPU picking has no retained pipeline".to_string())?;
            let target = pick_target
                .as_ref()
                .ok_or_else(|| "WebGPU picking has no retained target".to_string())?;
            let scene = scene
                .as_ref()
                .ok_or_else(|| "WebGPU picking requires scene residency".to_string())?;
            let readback = if resident_frame {
                let root_pipeline = resident_root_pick_pipeline.as_ref().ok_or_else(|| {
                    "WebGPU resident picking has no retained root pipeline".to_string()
                })?;
                let roots = resident_roots.as_ref().ok_or_else(|| {
                    "WebGPU resident picking requires root scene residency".to_string()
                })?;
                let (root_bindings, overlay) = if focus_frame {
                    let focus = roots.focus.as_ref().ok_or_else(|| {
                        "WebGPU resident focus picking lost its binding epoch".to_string()
                    })?;
                    (&focus.bindings, focus.overlay.as_ref())
                } else {
                    (&roots.bindings, roots.overlay.as_ref())
                };
                device.stage_resident_adaptive_pick(
                    root_pipeline,
                    pipeline,
                    scene.scene(),
                    &roots.preparation,
                    &roots.geometry,
                    root_bindings,
                    overlay,
                    atlas,
                    target,
                    request,
                )
            } else {
                device.stage_patch_render_scene_pick(pipeline, scene, atlas, target, request)
            }
            .map_err(|error| error.to_string())?;
            Ok::<_, String>(readback)
        })();
        match result {
            Ok(readback) => {
                backend.pick_submissions = backend.pick_submissions.saturating_add(1);
                backend.last_pick_error = None;
                Ok(StagedWebGpuPick {
                    frame_revision: backend.last_frame_revision,
                    source_render_call: backend.last_source_render_call,
                    readback,
                })
            }
            Err(error) => Err(backend.reject_pick(error)),
        }
    })
}

/// Stage a one-shot copy of the latest completed shadow frame. Queue ordering
/// guarantees that the copy follows its render submission. No thread-local
/// borrow survives the async map performed by the caller.
pub(crate) fn stage_frame_evidence() -> Result<StagedWebGpuFrameEvidence, String> {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        if backend.state != "ready" || backend.last_frame_revision == 0 {
            return Err("WebGPU image evidence requires one completed frame".to_string());
        }
        let device = backend
            .device
            .as_ref()
            .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
        let target = backend
            .target
            .as_ref()
            .ok_or_else(|| "WebGPU image evidence requires an offscreen target".to_string())?;
        let evidence = backend.frame_evidence.ok_or_else(|| {
            "WebGPU image evidence requires one completed offscreen frame".to_string()
        })?;
        if target.size() != evidence.viewport {
            return Err("WebGPU image evidence target does not match the last frame".to_string());
        }
        let image = device
            .stage_offscreen_patch_render_target_image(target)
            .map_err(|error| error.to_string())?;
        Ok(StagedWebGpuFrameEvidence {
            frame_revision: evidence.frame_revision,
            source_render_call: evidence.source_render_call,
            viewport: evidence.viewport,
            logical_submission: evidence.logical_submission,
            indirect_draw_calls: evidence.indirect_draw_calls,
            source_instances: evidence.source_instances,
            focus_postprocess: evidence.focus_postprocess,
            image,
        })
    })
}

/// Require the next semantically supported frame to execute even if all live
/// inputs compare equal to the prior shadow. Used only by explicit one-shot
/// parity capture so frame identity cannot accidentally refer to stale pixels.
pub(crate) fn force_next_frame() {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state == "ready" {
            backend.last_frame_input = None;
        }
    });
}

/// Route exactly the next coherent backend frame through the retained
/// offscreen target. A live surface keeps displaying its last admitted frame
/// while WebGL renders the matching oracle into its own evidence target.
pub(crate) fn request_frame_evidence() {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state == "ready" {
            backend.frame_evidence_requested = true;
            backend.frame_evidence = None;
            backend.last_frame_input = None;
        }
    });
}

pub(crate) fn diagnostics() -> WebGpuBackendDiagnostics {
    BACKEND.with(|slot| slot.borrow().diagnostics())
}

fn texture_asset_descriptor(
    index: usize,
    width: u32,
    height: u32,
    wrap_s: u32,
    wrap_t: u32,
) -> Result<TextureAssetDescriptor, String> {
    let wrap_s = TextureWrapMode::from_gl_enum(wrap_s)
        .ok_or_else(|| format!("PBR texture {index} has unsupported S wrap {wrap_s:#x}"))?;
    let wrap_t = TextureWrapMode::from_gl_enum(wrap_t)
        .ok_or_else(|| format!("PBR texture {index} has unsupported T wrap {wrap_t:#x}"))?;
    let descriptor = TextureAssetDescriptor {
        width,
        height,
        wrap_s,
        wrap_t,
    };
    descriptor
        .validate()
        .map_err(|error| format!("PBR texture {index}: {error}"))?;
    Ok(descriptor)
}

fn render_style_name(style: RenderStyle) -> &'static str {
    match style {
        RenderStyle::Pbr => "pbr",
        RenderStyle::Matcap => "matcap",
        RenderStyle::Wire => "wire",
        RenderStyle::Normals => "normals",
        RenderStyle::MatcapWire => "both",
        RenderStyle::Lod => "lod",
        RenderStyle::Stretch => "stretch",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[test]
    fn presentation_health_starts_unadmitted_and_is_explicitly_serialized() {
        let diagnostics = serde_json::to_value(WebGpuBackend::default().diagnostics()).unwrap();
        assert_eq!(diagnostics["presentationFrameAdmitted"], false);
    }

    #[test]
    fn explicit_evidence_routes_a_live_device_offscreen() {
        assert_eq!(
            frame_destination(true, false),
            FrameDestination::Presentation,
        );
        assert_eq!(frame_destination(true, true), FrameDestination::Offscreen,);
        assert_eq!(frame_destination(false, true), FrameDestination::Offscreen,);
    }

    #[test]
    fn render_pose_policy_distinguishes_device_pose_from_local_uniforms() {
        let pose = RenderPoseIdentity::timed(5, 7, 11);
        assert_eq!(
            render_pose_upload_policy(None, pose, false),
            PoseUploadPolicy::Publish,
        );
        assert_eq!(
            render_pose_upload_policy(Some(pose), pose, false),
            PoseUploadPolicy::PublishPreparation,
        );
        assert_eq!(
            render_pose_upload_policy(Some(pose), pose, true),
            PoseUploadPolicy::Reuse,
        );
    }

    #[test]
    fn pick_readback_serialization_keeps_epoch_and_source_chart_semantics() {
        let readback = WebGpuPickReadback {
            frame_revision: 17,
            source_render_call: 23,
            hit: Some(WebGpuPickHit {
                target_epoch: 29,
                packed_node: 7,
                source_face: 42,
                source_barycentric: [0.125, 0.25, 0.625],
                source_position: [1.5, -2.0, 0.75],
                output_distance: 4.25,
            }),
        };
        let json = serde_json::to_value(readback).unwrap();
        assert_eq!(json["frameRevision"], 17);
        assert_eq!(json["sourceRenderCall"], 23);
        assert_eq!(json["hit"]["targetEpoch"], 29);
        assert_eq!(json["hit"]["packedNode"], 7);
        assert_eq!(json["hit"]["sourceFace"], 42);
        assert_eq!(
            json["hit"]["sourceBarycentric"],
            serde_json::json!([0.125, 0.25, 0.625])
        );
        assert_eq!(
            json["hit"]["sourcePosition"],
            serde_json::json!([1.5, -2.0, 0.75])
        );
        assert_eq!(json["hit"]["outputDistance"], 4.25);
    }

    #[wasm_bindgen_test]
    fn browser_texture_descriptor_adapts_only_the_legacy_gl_wire_values() {
        let descriptor = texture_asset_descriptor(
            3,
            17,
            9,
            TextureWrapMode::GL_REPEAT,
            TextureWrapMode::GL_MIRRORED_REPEAT,
        )
        .unwrap();
        assert_eq!(descriptor.width, 17);
        assert_eq!(descriptor.height, 9);
        assert_eq!(descriptor.wrap_s, TextureWrapMode::Repeat);
        assert_eq!(descriptor.wrap_t, TextureWrapMode::MirroredRepeat);
        assert!(
            texture_asset_descriptor(3, 0, 9, TextureWrapMode::GL_REPEAT, 0)
                .unwrap_err()
                .contains("unsupported T wrap")
        );
        assert!(texture_asset_descriptor(
            3,
            0,
            9,
            TextureWrapMode::GL_REPEAT,
            TextureWrapMode::GL_REPEAT,
        )
        .unwrap_err()
        .contains("dimensions must be nonzero"));
    }
}
