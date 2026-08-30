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
    RenderSubmissionStats, RenderView, ResidentRootDrawDomains,
};
use quilting_renderer::compute::{
    prepare_lod_dispatch_state, LodAtlasLookup, PreparedLodModel, WgslLodDispatchMetrics,
};
use quilting_webgpu::{
    resident_root_render_domains, AdaptiveOverlayScene, DiagnosticPatchRenderPipelines,
    LodClassifierDevice, LodClassifierModel, LodPose, OffscreenPatchRenderTarget, PackedPatchAtlas,
    PatchFrameEncoding, PatchPresentationSurface, PatchRenderScene, PatchRenderSceneUpdate,
    PbrEnvironmentMap, PbrTextureTable, ResidentGeometryBucketScene, ResidentRootPreparationScene,
    ResidentRootRenderBindings, ResidentRootRenderPipeline, StagedOffscreenImageReadback,
    SurfacePresentation, WebGpuAdapterSummary,
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
    resident_root_pipeline: Option<ResidentRootRenderPipeline>,
    resident_roots: Option<ResidentRootBackend>,
    scene: Option<PatchRenderScene>,
    scene_source_revision: Option<u64>,
    target: Option<OffscreenPatchRenderTarget>,
    presentation: Option<PatchPresentationSurface>,
    last_frame_input: Option<LiveFrameInput>,
    last_face_visibility_bits: Vec<u32>,
    next_face_visibility_bits: Vec<u32>,
    last_joint_matrices: Vec<f32>,
    last_morph_weights: Vec<f32>,
    next_morph_weights: Vec<f32>,
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
    device_lod_dispatches: u64,
    device_lod_frames: u64,
    resident_root_scene_uploads: u64,
    resident_root_scene_reuses: u64,
    resident_root_frames: u64,
    resident_root_fallbacks: u64,
    last_device_lod_epoch: Option<u64>,
    last_frame_revision: u64,
    last_source_render_call: u64,
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_logical_submission: RenderSubmissionStats,
    last_viewport: [u32; 2],
    last_frame_failure: Option<String>,
    last_resident_root_error: Option<String>,
    last_error: Option<String>,
}

struct ResidentRootBackend {
    domains: ResidentRootDrawDomains,
    preparation: ResidentRootPreparationScene,
    geometry: ResidentGeometryBucketScene,
    bindings: ResidentRootRenderBindings,
    overlay: Option<AdaptiveOverlayScene>,
}

enum ResidentRootSceneCandidate {
    Unavailable,
    Reuse {
        bindings: ResidentRootRenderBindings,
        overlay: Option<AdaptiveOverlayScene>,
    },
    Replace(ResidentRootBackend),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LiveFrameInput {
    source_revision: u64,
    device_lod_epoch: Option<u64>,
    style: RenderStyle,
    view: RenderView,
    options: RenderFrameOptions,
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
    presentation_frames: u64,
    presentation_skips: u64,
    presentation_losses: u64,
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
    device_lod_dispatches: u64,
    device_lod_frames: u64,
    resident_root_pipeline_ready: bool,
    resident_root_scene_ready: bool,
    resident_root_scene_uploads: u64,
    resident_root_scene_reuses: u64,
    resident_root_frames: u64,
    resident_root_fallbacks: u64,
    last_device_lod_epoch: Option<u64>,
    last_frame_revision: u64,
    last_source_render_call: u64,
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_logical_submission: RenderSubmissionStats,
    last_viewport: [u32; 2],
    last_frame_failure: Option<String>,
    last_resident_root_error: Option<String>,
    last_error: Option<String>,
}

impl WebGpuBackend {
    fn diagnostics(&self) -> WebGpuBackendDiagnostics {
        let presentation = self
            .presentation
            .as_ref()
            .map(PatchPresentationSurface::diagnostics);
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
            presentation_style: self.presentation.as_ref().and_then(|_| {
                self.last_frame_input
                    .map(|frame| render_style_name(frame.style))
            }),
            presentation_frames: presentation
                .as_ref()
                .map_or(0, |presentation| presentation.frames_presented),
            presentation_skips: presentation
                .as_ref()
                .map_or(0, |presentation| presentation.frames_skipped),
            presentation_losses: presentation
                .as_ref()
                .map_or(0, |presentation| presentation.surface_losses),
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
            device_lod_dispatches: self.device_lod_dispatches,
            device_lod_frames: self.device_lod_frames,
            resident_root_pipeline_ready: self.resident_root_pipeline.is_some(),
            resident_root_scene_ready: self.resident_roots.is_some(),
            resident_root_scene_uploads: self.resident_root_scene_uploads,
            resident_root_scene_reuses: self.resident_root_scene_reuses,
            resident_root_frames: self.resident_root_frames,
            resident_root_fallbacks: self.resident_root_fallbacks,
            last_device_lod_epoch: self.last_device_lod_epoch,
            last_frame_revision: self.last_frame_revision,
            last_source_render_call: self.last_source_render_call,
            last_indirect_draw_calls: self.last_indirect_draw_calls,
            last_source_instances: self.last_source_instances,
            last_logical_submission: self.last_logical_submission,
            last_viewport: self.last_viewport,
            last_frame_failure: self.last_frame_failure.clone(),
            last_resident_root_error: self.last_resident_root_error.clone(),
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
        self.last_frame_failure = Some(error.clone());
        self.last_error = Some(error.clone());
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
        match candidate {
            ResidentRootSceneCandidate::Reuse { bindings, overlay } if failure.is_none() => {
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
                        Ok(())
                    });
                match publication {
                    Ok(()) => {
                        self.resident_root_scene_reuses =
                            self.resident_root_scene_reuses.saturating_add(1);
                        self.last_resident_root_error = None;
                    }
                    Err(error) => self.publish_resident_roots(None, Some(error)),
                }
            }
            ResidentRootSceneCandidate::Replace(candidate) if failure.is_none() => {
                self.publish_resident_roots(Some(candidate), None);
            }
            ResidentRootSceneCandidate::Unavailable
            | ResidentRootSceneCandidate::Reuse { .. }
            | ResidentRootSceneCandidate::Replace(_) => {
                self.publish_resident_roots(None, failure);
            }
        }
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
                let (resident_root_pipeline, resident_root_error) =
                    match device.create_offscreen_resident_root_render_pipeline() {
                        Ok(pipeline) => (Some(pipeline), None),
                        Err(error) => (None, Some(error.to_string())),
                    };
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
                    backend.resident_root_pipeline = resident_root_pipeline;
                    backend.resident_roots = None;
                    backend.scene = None;
                    backend.scene_source_revision = None;
                    backend.target = None;
                    backend.presentation = None;
                    backend.last_frame_input = None;
                    backend.last_device_lod_epoch = None;
                    backend.last_face_visibility_bits.clear();
                    backend.next_face_visibility_bits.clear();
                    backend.last_joint_matrices.clear();
                    backend.last_morph_weights.clear();
                    backend.next_morph_weights.clear();
                    if resident_root_error.is_some() {
                        backend.resident_root_fallbacks =
                            backend.resident_root_fallbacks.saturating_add(1);
                    }
                    backend.last_resident_root_error = resident_root_error;
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
    let (resident_root_pipeline, resident_root_error) = match device
        .create_resident_root_render_pipeline(
            presentation.color_format(),
            Some(presentation.depth_format()),
            1,
        ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    };
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
        backend.resident_root_pipeline = resident_root_pipeline;
        backend.resident_roots = None;
        backend.scene = None;
        backend.scene_source_revision = None;
        backend.target = None;
        backend.presentation = Some(presentation);
        backend.last_frame_input = None;
        backend.last_device_lod_epoch = None;
        backend.last_face_visibility_bits.clear();
        backend.next_face_visibility_bits.clear();
        backend.last_joint_matrices.clear();
        backend.last_morph_weights.clear();
        backend.next_morph_weights.clear();
        if resident_root_error.is_some() {
            backend.resident_root_fallbacks = backend.resident_root_fallbacks.saturating_add(1);
        }
        backend.last_resident_root_error = resident_root_error;
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
                scene,
                environment,
                resident_root_pipeline,
                resident_roots,
                ..
            } = &mut *backend;
            let device = device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let textures = device
                .upload_pbr_image_bitmap_table(&assets)
                .map_err(|error| error.to_string())?;
            let root_bindings = rebuild_resident_root_pbr_bindings(
                device,
                resident_root_pipeline.as_ref(),
                resident_roots.as_ref(),
                scene.as_ref(),
                Some(&textures),
                environment.as_ref(),
            )?;
            if let Some(scene) = scene.as_mut() {
                let pipeline = pipelines
                    .as_ref()
                    .and_then(|pipelines| pipelines.get(RenderStyle::Pbr))
                    .ok_or_else(|| "ready WebGPU backend has no PBR pipeline".to_string())?;
                device
                    .replace_patch_render_scene_texture_bindings(pipeline, scene, Some(&textures))
                    .map_err(|error| error.to_string())?;
            }
            if let Some(bindings) = root_bindings {
                resident_roots
                    .as_mut()
                    .expect("resident root binding rebuild checked scene presence")
                    .bindings = bindings;
            }
            Ok::<_, String>(textures)
        };
        match result {
            Ok(textures) => {
                backend.textures = Some(textures);
                backend.texture_uploads = backend.texture_uploads.saturating_add(1);
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
                scene,
                textures,
                resident_root_pipeline,
                resident_roots,
                ..
            } = &mut *backend;
            let device = device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let environment = device
                .upload_pbr_environment_map(asset)
                .map_err(|error| error.to_string())?;
            let root_bindings = rebuild_resident_root_pbr_bindings(
                device,
                resident_root_pipeline.as_ref(),
                resident_roots.as_ref(),
                scene.as_ref(),
                textures.as_ref(),
                Some(&environment),
            )?;
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
            if let Some(bindings) = root_bindings {
                resident_roots
                    .as_mut()
                    .expect("resident root binding rebuild checked scene presence")
                    .bindings = bindings;
            }
            Ok::<_, String>(environment)
        };
        match result {
            Ok(environment) => {
                backend.environment = Some(environment);
                backend.environment_uploads = backend.environment_uploads.saturating_add(1);
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
                backend.last_frame_input = None;
                backend.last_device_lod_epoch = None;
                backend.last_face_visibility_bits.clear();
                backend.next_face_visibility_bits.clear();
                backend.next_morph_weights.clear();
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
                backend.last_frame_input = None;
                backend.last_device_lod_epoch = None;
                backend.last_face_visibility_bits.clear();
                backend.next_face_visibility_bits.clear();
                backend.next_morph_weights.clear();
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
            let classification = device
                .classify_on_device(
                    model,
                    &dispatch,
                    metrics,
                    LodPose {
                        joint_matrices,
                        morph_weights,
                    },
                )
                .map_err(|error| error.to_string())?;
            Ok::<_, String>(
                device
                    .reconcile_resident_lod_on_device(&classification, grading)
                    .classification_epoch(),
            )
        })();
        let epoch = match epoch_result {
            Ok(epoch) => epoch,
            Err(error) => {
                backend.last_error = Some(error.clone());
                return Err(error);
            }
        };
        backend.device_lod_dispatches = backend.device_lod_dispatches.saturating_add(1);
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

pub(crate) fn needs_scene(source_revision: u64) -> bool {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        backend.state == "ready"
            && backend.model.is_some()
            && backend.atlas.is_some()
            && (backend.scene.is_none() || backend.scene_source_revision != Some(source_revision))
    })
}

pub(crate) fn shadow_evidence_ready() -> bool {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        backend.state == "ready" && backend.presentation.is_none()
    })
}

pub(crate) fn pbr_evidence_ready() -> bool {
    shadow_evidence_ready() && BACKEND.with(|slot| slot.borrow().environment.is_some())
}

pub(crate) fn record_frame_prerequisite_failure(error: impl ToString) {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state == "ready" {
            let error = error.to_string();
            backend.frame_failures = backend.frame_failures.saturating_add(1);
            backend.last_frame_failure = Some(error.clone());
            backend.last_error = Some(error);
        }
    });
}

fn build_resident_root_backend(
    device: &LodClassifierDevice,
    model: &LodClassifierModel,
    atlas: &PackedPatchAtlas,
    root_pipeline: &ResidentRootRenderPipeline,
    overlay_pipelines: &DiagnosticPatchRenderPipelines,
    domains: ResidentRootDrawDomains,
    scene: &RenderSceneSnapshot,
    source_instances: &[f32],
    textures: Option<&PbrTextureTable>,
    environment: Option<&PbrEnvironmentMap>,
) -> Result<ResidentRootBackend, String> {
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
    let overlay = build_adaptive_overlay(device, model, overlay_pipelines, &preparation, scene)?;
    device
        .publish_adaptive_overlay_suppression(&geometry, overlay.as_ref())
        .map_err(|error| error.to_string())?;
    Ok(ResidentRootBackend {
        domains,
        preparation,
        geometry,
        bindings,
        overlay,
    })
}

fn rebuild_resident_root_pbr_bindings(
    device: &LodClassifierDevice,
    pipeline: Option<&ResidentRootRenderPipeline>,
    roots: Option<&ResidentRootBackend>,
    scene: Option<&PatchRenderScene>,
    textures: Option<&PbrTextureTable>,
    environment: Option<&PbrEnvironmentMap>,
) -> Result<Option<ResidentRootRenderBindings>, String> {
    let Some(roots) = roots else {
        return Ok(None);
    };
    let pipeline = pipeline
        .ok_or_else(|| "resident PBR binding rebuild lost its root pipeline".to_string())?;
    let scene =
        scene.ok_or_else(|| "resident PBR binding rebuild lost its render scene".to_string())?;
    device
        .create_resident_root_render_bindings_with_pbr(
            pipeline,
            &roots.preparation,
            &roots.geometry,
            scene.scene(),
            textures,
            environment,
        )
        .map(Some)
        .map_err(|error| error.to_string())
}

fn build_adaptive_overlay(
    device: &LodClassifierDevice,
    model: &LodClassifierModel,
    pipelines: &DiagnosticPatchRenderPipelines,
    preparation: &ResidentRootPreparationScene,
    scene: &RenderSceneSnapshot,
) -> Result<Option<AdaptiveOverlayScene>, String> {
    let layout_pipeline = pipelines
        .get(RenderStyle::Normals)
        .ok_or_else(|| "WebGPU diagnostic pipeline family lost normals".to_string())?;
    device
        .upload_adaptive_overlay_scene(layout_pipeline, model, preparation, scene)
        .map_err(|error| error.to_string())
}

/// Publish one device-side render scene derived from shared backend-neutral
/// extraction. Shape-compatible epochs update retained buffers in place; a
/// cardinality change allocates an atomic replacement. The previous scene
/// remains usable if packing or allocation fails.
pub(crate) fn replace_scene(
    source_revision: u64,
    scene: RenderSceneSnapshot,
    source_instances: &[f32],
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" {
            return Ok(false);
        }
        let next_revision = backend.scene_uploads.saturating_add(1);
        let mut scene = scene;
        scene.revision = next_revision;
        let resident_root_domains = backend
            .model_source
            .as_ref()
            .ok_or_else(|| "WebGPU resident roots require immutable model residency".to_string())
            .and_then(|source| resident_root_render_domains(&scene, source.residency.num_faces));
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
                        &scene,
                        backend.textures.as_ref(),
                        backend.environment.as_ref(),
                    )
                    .map_err(|error| error.to_string())?;
                let overlay = build_adaptive_overlay(
                    device,
                    model,
                    pipelines,
                    &retained.preparation,
                    &scene,
                )?;
                Ok(ResidentRootSceneCandidate::Reuse { bindings, overlay })
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
                domains,
                &scene,
                source_instances,
                backend.textures.as_ref(),
                backend.environment.as_ref(),
            )
            .map(ResidentRootSceneCandidate::Replace),
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
                    .update_patch_render_scene_in_place(
                        pipeline,
                        model,
                        retained,
                        scene,
                        source_instances,
                        next_revision,
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
                backend.last_frame_input = None;
                backend.next_morph_weights.clear();
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
                .upload_patch_render_scene(
                    pipeline,
                    model,
                    scene,
                    source_instances,
                    next_revision,
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
                backend.last_frame_input = None;
                backend.last_face_visibility_bits.clear();
                backend.next_face_visibility_bits.clear();
                backend.next_morph_weights.clear();
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
    style: RenderStyle,
    view: RenderView,
    options: RenderFrameOptions,
    face_visibility: &[bool],
    joint_matrices: &[f32],
    morph_weights: &[f32],
) -> Result<LiveFrameDisposition, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        let shadow_only_pbr = style == RenderStyle::Pbr && backend.presentation.is_none();
        if backend.state != "ready"
            || (!quilting_webgpu::supports_patch_presentation_style(style) && !shadow_only_pbr)
        {
            return Ok(LiveFrameDisposition::IncumbentRequired);
        }
        // Atlas/model/scene residency arrives asynchronously during ordinary
        // application startup. Frames before the first coherent scene are
        // inert lifecycle gaps, not failed render attempts.
        if backend.scene.is_none() {
            return Ok(LiveFrameDisposition::IncumbentRequired);
        }
        if style == RenderStyle::Pbr
            && !backend
                .scene
                .as_ref()
                .expect("scene presence checked above")
                .supports_resident_basic_pbr_frame(options)
        {
            return Ok(LiveFrameDisposition::IncumbentRequired);
        }
        if view.viewport[0] == 0 || view.viewport[1] == 0 {
            return Ok(LiveFrameDisposition::IncumbentRequired);
        }
        backend.frame_attempts = backend.frame_attempts.saturating_add(1);
        let frame_revision = backend.frame_attempts;

        let mut face_visibility_bits = std::mem::take(&mut backend.next_face_visibility_bits);
        face_visibility_bits.clear();
        let mut effective_morph_weights = std::mem::take(&mut backend.next_morph_weights);
        effective_morph_weights.clear();
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
                let adaptive_layer_supported = roots
                    .overlay
                    .as_ref()
                    .is_none_or(AdaptiveOverlayScene::supports_resident_untextured_pbr);
                roots
                    .bindings
                    .supports_resident_root_frame(style, adaptive_layer_supported)
            });
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
        effective_morph_weights.extend_from_slice(morph_weights);
        effective_morph_weights.resize(resident_morph_targets, 0.0);

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
        let unchanged = backend.last_frame_input == Some(frame_input)
            && (device_lod_epoch.is_some()
                || backend.last_face_visibility_bits == face_visibility_bits)
            && backend.last_joint_matrices == joint_matrices
            && backend.last_morph_weights == effective_morph_weights;
        if unchanged {
            backend.next_face_visibility_bits = face_visibility_bits;
            backend.next_morph_weights = effective_morph_weights;
            backend.frames_skipped_unchanged = backend.frames_skipped_unchanged.saturating_add(1);
            return Ok(
                if backend.presentation.is_some() && backend.last_frame_revision != 0 {
                    LiveFrameDisposition::PresentationRetained
                } else {
                    LiveFrameDisposition::IncumbentRequired
                },
            );
        }
        let visibility_upload_required = device_lod_epoch.is_none()
            && backend.last_face_visibility_bits != face_visibility_bits;

        if backend.presentation.is_some() {
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
            backend.target_rebuilds = backend.target_rebuilds.saturating_add(1);
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
            let frame = RenderFrame::build(
                frame_revision,
                RenderPoseIdentity {
                    asset_revision: scene_source_revision,
                    pose_revision: frame_revision,
                },
                style,
                view,
                options,
                scene.scene(),
            )
            .map_err(|error| error.to_string())?;
            if !resident_root_frame {
                device
                    .write_patch_render_pose_state(
                        model,
                        scene,
                        LodPose {
                            joint_matrices,
                            morph_weights: &effective_morph_weights,
                        },
                        num_joints,
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
        let result =
            prepared_frame.and_then(|frame| {
                if backend.presentation.is_some() {
                    let WebGpuBackend {
                        device,
                        model,
                        pipelines,
                        resident_root_pipeline,
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
                        let root_pipeline = resident_root_pipeline.as_ref().ok_or_else(|| {
                            "WebGPU resident root frame lost its pipeline residency".to_string()
                        })?;
                        let pose = LodPose {
                            joint_matrices,
                            morph_weights: &effective_morph_weights,
                        };
                        if style == RenderStyle::Pbr {
                            device.present_resident_roots(
                                presentation,
                                &frame,
                                scene.scene(),
                                model,
                                &resident,
                                &roots.preparation,
                                &roots.geometry,
                                root_pipeline,
                                &roots.bindings,
                                atlas,
                                pose,
                                num_joints,
                                true,
                            )
                        } else {
                            device.present_resident_adaptive(
                                presentation,
                                &frame,
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
                                true,
                            )
                        }
                    } else if let Some(resident) = model
                        .as_ref()
                        .and_then(|model| device.latest_resident_lod(model))
                    {
                        device.present_supported_patch_scene_with_resident_lod_visibility(
                            presentation,
                            &frame,
                            pipelines,
                            scene,
                            &resident,
                            atlas,
                            true,
                        )
                    } else {
                        device.present_supported_patch_scene_with_face_visibility(
                            presentation,
                            &frame,
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
                        let root_pipeline = backend.resident_root_pipeline.as_ref().ok_or_else(|| {
                            "WebGPU resident root frame lost its pipeline residency".to_string()
                        })?;
                        let pose = LodPose {
                            joint_matrices,
                            morph_weights: &effective_morph_weights,
                        };
                        if style == RenderStyle::Pbr {
                            device.render_offscreen_resident_roots(
                                &frame,
                                scene.scene(),
                                model,
                                &resident,
                                &roots.preparation,
                                &roots.geometry,
                                root_pipeline,
                                &roots.bindings,
                                atlas,
                                target,
                                pose,
                                num_joints,
                                true,
                            )
                        } else {
                            device.render_offscreen_resident_adaptive(
                                &frame,
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
                                true,
                            )
                        }
                    } else if let Some(resident) = resident {
                        device
                            .render_offscreen_supported_patch_scene_with_resident_lod_visibility_and_webgl_clear(
                                &frame,
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
                            &frame, pipeline, scene, atlas, target, true,
                        )
                    } else if style == RenderStyle::Pbr {
                        device.render_offscreen_supported_patch_scene_with_webgl_clear(
                            &frame, pipelines, scene, atlas, target, true,
                        )
                    } else {
                        device.render_offscreen_supported_patch_scene_with_face_visibility(
                            &frame, pipelines, scene, atlas, target, true,
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
                backend.last_joint_matrices.clear();
                backend
                    .last_joint_matrices
                    .extend_from_slice(joint_matrices);
                std::mem::swap(
                    &mut backend.last_morph_weights,
                    &mut effective_morph_weights,
                );
                backend.next_morph_weights = effective_morph_weights;
                backend.last_frame_input = Some(frame_input);
                if resident_root_frame {
                    backend.resident_root_frames = backend.resident_root_frames.saturating_add(1);
                }
                backend.last_error = None;
                Ok(if backend.presentation.is_some() {
                    LiveFrameDisposition::PresentationSubmitted(logical_submission)
                } else {
                    LiveFrameDisposition::ShadowSubmitted(logical_submission)
                })
            }
            Ok(None) => {
                let retained_style_matches = backend
                    .last_frame_input
                    .is_some_and(|last_frame| last_frame.style == style);
                backend.next_face_visibility_bits = face_visibility_bits;
                backend.next_morph_weights = effective_morph_weights;
                Ok(
                    if backend.presentation.is_some()
                        && backend.last_frame_revision != 0
                        && retained_style_matches
                    {
                        LiveFrameDisposition::PresentationRetained
                    } else {
                        LiveFrameDisposition::IncumbentRequired
                    },
                )
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
    pub(crate) image: StagedOffscreenImageReadback,
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
        if target.size() != backend.last_viewport {
            return Err("WebGPU image evidence target does not match the last frame".to_string());
        }
        let image = device
            .stage_offscreen_patch_render_target_image(target)
            .map_err(|error| error.to_string())?;
        Ok(StagedWebGpuFrameEvidence {
            frame_revision: backend.last_frame_revision,
            source_render_call: backend.last_source_render_call,
            viewport: backend.last_viewport,
            logical_submission: backend.last_logical_submission,
            indirect_draw_calls: backend.last_indirect_draw_calls,
            source_instances: backend.last_source_instances,
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
