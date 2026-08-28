//! Rollback-safe browser residency for the staged WebGPU backend.
//!
//! This module deliberately owns no semantic scene or canvas. The incumbent
//! WebGL2 renderer remains authoritative while exact packed-atlas and prepared
//! model inputs are mirrored into a headless WebGPU device. Promotion later
//! attaches these retained resources to the same extracted RenderFrame path.

use quilting_renderer::compute::{LodAtlasLookup, PreparedLodModel};
use quilting_webgpu::{
    LodClassifierDevice, LodClassifierModel, PackedPatchAtlas, WebGpuAdapterSummary,
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
    initialization_attempts: u64,
    atlas_uploads: u64,
    model_uploads: u64,
    last_error: Option<String>,
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
    initialization_attempts: u64,
    atlas_uploads: u64,
    model_uploads: u64,
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
            initialization_attempts: self.initialization_attempts,
            atlas_uploads: self.atlas_uploads,
            model_uploads: self.model_uploads,
            last_error: self.last_error.clone(),
        }
    }

    fn fail(&mut self, error: impl ToString) -> String {
        let error = error.to_string();
        self.state = "failed";
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
            Ok((device, adapter)) => BACKEND.with(|slot| {
                let mut backend = slot.borrow_mut();
                backend.device = Some(device);
                backend.adapter = Some(adapter);
                backend.atlas = None;
                backend.model = None;
                backend.model_source = None;
                backend.state = "ready";
                backend.last_error = None;
            }),
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
                backend.model_uploads = backend.model_uploads.saturating_add(1);
                backend.last_error = None;
                Ok(true)
            }
            Err(error) => Err(backend.fail(error)),
        }
    })
}

pub(crate) fn diagnostics() -> WebGpuBackendDiagnostics {
    BACKEND.with(|slot| slot.borrow().diagnostics())
}
