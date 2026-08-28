//! Rollback-safe browser residency for the staged WebGPU backend.
//!
//! This module deliberately owns no semantic scene or browser layout. Exact
//! packed-atlas, prepared-model, extracted-scene, and current-frame inputs can
//! be mirrored into a headless device for parity or presented through an
//! application-supplied canvas. Canvas selection remains an explicit startup
//! decision; a claimed context is never silently repurposed.

use quilting_core::render::{
    RenderFrame, RenderFrameOptions, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle,
    RenderSubmissionStats, RenderView,
};
use quilting_renderer::compute::{LodAtlasLookup, PreparedLodModel};
use quilting_webgpu::{
    DiagnosticPatchRenderPipelines, LodClassifierDevice, LodClassifierModel, LodPose,
    OffscreenPatchRenderTarget, PackedPatchAtlas, PatchFrameEncoding, PatchPresentationSurface,
    PatchRenderScene, PatchRenderSceneUpdate, StagedOffscreenImageReadback, SurfacePresentation,
    WebGpuAdapterSummary,
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
    model: Option<LodClassifierModel>,
    model_source: Option<PreparedLodModel>,
    pipelines: Option<DiagnosticPatchRenderPipelines>,
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
    last_frame_revision: u64,
    last_source_render_call: u64,
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_logical_submission: RenderSubmissionStats,
    last_viewport: [u32; 2],
    last_frame_failure: Option<String>,
    last_error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LiveFrameInput {
    source_revision: u64,
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
    last_frame_revision: u64,
    last_source_render_call: u64,
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_logical_submission: RenderSubmissionStats,
    last_viewport: [u32; 2],
    last_frame_failure: Option<String>,
    last_error: Option<String>,
}

impl WebGpuBackend {
    fn diagnostics(&self) -> WebGpuBackendDiagnostics {
        let presentation = self
            .presentation
            .as_ref()
            .map(PatchPresentationSurface::diagnostics);
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
            last_frame_revision: self.last_frame_revision,
            last_source_render_call: self.last_source_render_call,
            last_indirect_draw_calls: self.last_indirect_draw_calls,
            last_source_instances: self.last_source_instances,
            last_logical_submission: self.last_logical_submission,
            last_viewport: self.last_viewport,
            last_frame_failure: self.last_frame_failure.clone(),
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
                BACKEND.with(|slot| {
                    let mut backend = slot.borrow_mut();
                    backend.device = Some(device);
                    backend.adapter = Some(adapter);
                    backend.atlas = None;
                    backend.model = None;
                    backend.model_source = None;
                    backend.pipelines = Some(pipelines);
                    backend.scene = None;
                    backend.scene_source_revision = None;
                    backend.target = None;
                    backend.presentation = None;
                    backend.last_frame_input = None;
                    backend.last_face_visibility_bits.clear();
                    backend.next_face_visibility_bits.clear();
                    backend.last_joint_matrices.clear();
                    backend.last_morph_weights.clear();
                    backend.next_morph_weights.clear();
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
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        backend.device = Some(device);
        backend.adapter = Some(adapter);
        backend.atlas = None;
        backend.model = None;
        backend.model_source = None;
        backend.pipelines = Some(pipelines);
        backend.scene = None;
        backend.scene_source_revision = None;
        backend.target = None;
        backend.presentation = Some(presentation);
        backend.last_frame_input = None;
        backend.last_face_visibility_bits.clear();
        backend.next_face_visibility_bits.clear();
        backend.last_joint_matrices.clear();
        backend.last_morph_weights.clear();
        backend.next_morph_weights.clear();
        backend.last_logical_submission = RenderSubmissionStats::default();
        backend.state = "ready";
        backend.last_error = None;
    });
    Ok(diagnostics())
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
                backend.scene = None;
                backend.scene_source_revision = None;
                backend.last_frame_input = None;
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
                backend.scene = None;
                backend.scene_source_revision = None;
                backend.last_frame_input = None;
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

pub(crate) fn needs_scene(source_revision: u64) -> bool {
    BACKEND.with(|slot| {
        let backend = slot.borrow();
        backend.state == "ready"
            && backend.model.is_some()
            && backend.atlas.is_some()
            && (backend.scene.is_none() || backend.scene_source_revision != Some(source_revision))
    })
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
        let update_result = {
            let WebGpuBackend {
                device,
                model,
                scene: retained,
                ..
            } = &mut *backend;
            match (device.as_ref(), model.as_ref(), retained.as_mut()) {
                (Some(device), Some(model), Some(retained)) => device
                    .update_patch_render_scene_in_place(
                        model,
                        retained,
                        scene,
                        source_instances,
                        next_revision,
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
                .and_then(|pipelines| pipelines.get(RenderStyle::Normals))
                .ok_or_else(|| "ready WebGPU backend has no normals pipeline".to_string())?;
            let model = backend
                .model
                .as_ref()
                .ok_or_else(|| "WebGPU render scene requires model residency".to_string())?;
            device
                .upload_patch_render_scene(pipeline, model, scene, source_instances, next_revision)
                .map_err(|error| error.to_string())
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
        if backend.state != "ready"
            || !matches!(
                style,
                RenderStyle::Matcap
                    | RenderStyle::Normals
                    | RenderStyle::Lod
                    | RenderStyle::Stretch
            )
        {
            return Ok(LiveFrameDisposition::IncumbentRequired);
        }
        // Atlas/model/scene residency arrives asynchronously during ordinary
        // application startup. Frames before the first coherent scene are
        // inert lifecycle gaps, not failed render attempts.
        if backend.scene.is_none() {
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
            style,
            view,
            options,
        };
        let unchanged = backend.last_frame_input == Some(frame_input)
            && backend.last_face_visibility_bits == face_visibility_bits
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
        let visibility_upload_required = backend.last_face_visibility_bits != face_visibility_bits;

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
                        pipelines,
                        scene,
                        atlas,
                        presentation,
                        ..
                    } = &mut *backend;
                    let device = device
                        .as_ref()
                        .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
                    let pipeline = pipelines
                        .as_ref()
                        .and_then(|pipelines| pipelines.get(style))
                        .ok_or_else(|| format!("ready WebGPU backend has no {style:?} pipeline"))?;
                    let scene = scene.as_ref().ok_or_else(|| {
                        "WebGPU render frame requires scene residency".to_string()
                    })?;
                    let atlas = atlas.as_ref().ok_or_else(|| {
                        "WebGPU render frame requires atlas residency".to_string()
                    })?;
                    let presentation = presentation
                        .as_mut()
                        .ok_or_else(|| "WebGPU presentation surface disappeared".to_string())?;
                    match device
                        .present_diagnostic_patch_scene_with_face_visibility(
                            presentation,
                            &frame,
                            pipeline,
                            scene,
                            atlas,
                            true,
                        )
                        .map_err(|error| error.to_string())?
                    {
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
                    let pipeline = backend
                        .pipelines
                        .as_ref()
                        .and_then(|pipelines| pipelines.get(style))
                        .ok_or_else(|| format!("ready WebGPU backend has no {style:?} pipeline"))?;
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
                    let encoding = if style == RenderStyle::Normals {
                        device.render_offscreen_normals_patch_scene_with_webgl_clear(
                            &frame, pipeline, scene, atlas, target, true,
                        )
                    } else {
                        device.render_offscreen_diagnostic_patch_scene_with_face_visibility(
                            &frame, pipeline, scene, atlas, target, true,
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
                if visibility_upload_required {
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
                backend.last_error = None;
                Ok(if backend.presentation.is_some() {
                    LiveFrameDisposition::PresentationSubmitted(logical_submission)
                } else {
                    LiveFrameDisposition::IncumbentRequired
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
