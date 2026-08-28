//! Rollback-safe browser residency for the staged WebGPU backend.
//!
//! This module deliberately owns no semantic scene or canvas. The incumbent
//! WebGL2 renderer remains authoritative while exact packed-atlas, prepared
//! model, extracted scene, and current frame inputs are mirrored into a
//! headless WebGPU device. The first live frame target remains offscreen until
//! image/workload parity permits an explicit presentation-surface cutover.

use quilting_core::render::{
    RenderFrame, RenderFrameOptions, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle,
    RenderView,
};
use quilting_renderer::compute::{LodAtlasLookup, PreparedLodModel};
use quilting_webgpu::{
    LodClassifierDevice, LodClassifierModel, LodPose, OffscreenPatchRenderTarget, PackedPatchAtlas,
    PatchFrameEncoding, PatchRenderPipeline, PatchRenderScene, PatchRenderSceneUpdate,
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
    pipeline: Option<PatchRenderPipeline>,
    scene: Option<PatchRenderScene>,
    scene_source_revision: Option<u64>,
    target: Option<OffscreenPatchRenderTarget>,
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
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
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
    last_indirect_draw_calls: u32,
    last_source_instances: u32,
    last_viewport: [u32; 2],
    last_frame_failure: Option<String>,
    last_error: Option<String>,
}

impl WebGpuBackend {
    fn diagnostics(&self) -> WebGpuBackendDiagnostics {
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
            last_indirect_draw_calls: self.last_indirect_draw_calls,
            last_source_instances: self.last_source_instances,
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
                let pipeline = device
                    .create_offscreen_patch_render_pipeline()
                    .map_err(|error| error.to_string())
                    .map_err(|error| BACKEND.with(|slot| slot.borrow_mut().fail(error)))?;
                BACKEND.with(|slot| {
                    let mut backend = slot.borrow_mut();
                    backend.device = Some(device);
                    backend.adapter = Some(adapter);
                    backend.atlas = None;
                    backend.model = None;
                    backend.model_source = None;
                    backend.pipeline = Some(pipeline);
                    backend.scene = None;
                    backend.scene_source_revision = None;
                    backend.target = None;
                    backend.last_frame_input = None;
                    backend.last_face_visibility_bits.clear();
                    backend.next_face_visibility_bits.clear();
                    backend.last_joint_matrices.clear();
                    backend.last_morph_weights.clear();
                    backend.next_morph_weights.clear();
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
                .pipeline
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no render pipeline".to_string())?;
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

/// Execute one current live frame into an offscreen target. This is a measured
/// shadow only: the WebGL2 canvas remains authoritative and no pixel readback
/// occurs on the frame path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn submit_frame(
    style: RenderStyle,
    view: RenderView,
    options: RenderFrameOptions,
    face_visibility: &[bool],
    joint_matrices: &[f32],
    morph_weights: &[f32],
) -> Result<bool, String> {
    BACKEND.with(|slot| {
        let mut backend = slot.borrow_mut();
        if backend.state != "ready" || style != RenderStyle::Normals {
            return Ok(false);
        }
        // Atlas/model/scene residency arrives asynchronously during ordinary
        // application startup. Frames before the first coherent scene are
        // inert lifecycle gaps, not failed render attempts.
        if backend.scene.is_none() {
            return Ok(false);
        }
        if view.viewport[0] == 0 || view.viewport[1] == 0 {
            return Ok(false);
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
            return Ok(false);
        }
        let visibility_upload_required = backend.last_face_visibility_bits != face_visibility_bits;

        if backend
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
        let result = (|| {
            let device = backend
                .device
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no device".to_string())?;
            let pipeline = backend
                .pipeline
                .as_ref()
                .ok_or_else(|| "ready WebGPU backend has no render pipeline".to_string())?;
            let model = backend
                .model
                .as_ref()
                .ok_or_else(|| "WebGPU render frame requires model residency".to_string())?;
            let atlas = backend
                .atlas
                .as_ref()
                .ok_or_else(|| "WebGPU render frame requires atlas residency".to_string())?;
            let scene = backend
                .scene
                .as_ref()
                .ok_or_else(|| "WebGPU render frame requires scene residency".to_string())?;
            let target = backend
                .target
                .as_ref()
                .ok_or_else(|| "WebGPU render frame requires a target".to_string())?;
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
            device
                .render_offscreen_normals_patch_scene_with_face_visibility(
                    &frame, pipeline, scene, atlas, target, true,
                )
                .map_err(|error| error.to_string())
        })();
        match result {
            Ok(PatchFrameEncoding {
                indirect_draw_calls,
                source_instance_count,
                ..
            }) => {
                backend.frames_submitted = backend.frames_submitted.saturating_add(1);
                if visibility_upload_required {
                    backend.visibility_uploads = backend.visibility_uploads.saturating_add(1);
                    backend.visibility_upload_bytes = backend
                        .visibility_upload_bytes
                        .saturating_add((face_visibility_bits.len() as u64).saturating_mul(4));
                }
                backend.last_frame_revision = frame_revision;
                backend.last_indirect_draw_calls = indirect_draw_calls;
                backend.last_source_instances = source_instance_count;
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
                Ok(true)
            }
            Err(error) => {
                Err(backend.reject_frame(face_visibility_bits, effective_morph_weights, error))
            }
        }
    })
}

pub(crate) fn diagnostics() -> WebGpuBackendDiagnostics {
    BACKEND.with(|slot| slot.borrow().diagnostics())
}
