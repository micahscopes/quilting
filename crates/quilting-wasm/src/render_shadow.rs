//! Thin WASM adapter for the backend-neutral render parity observer.
//!
//! High-rate comparisons remain entirely in Rust. JavaScript receives only a
//! bounded diagnostic snapshot when it explicitly calls the exported query.

use quilting_core::render::{
    RenderContractError, RenderFrame, RenderFrameOptions, RenderParityDiagnostics,
    RenderParityObserver, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle,
    RenderSubmissionStats, RenderView,
};
use serde::Serialize;
use wasm_bindgen::JsValue;

#[derive(Debug, Default)]
pub(crate) struct RenderShadowObserver {
    parity: RenderParityObserver,
    asset_revision: u64,
    pose_revision: u64,
    scene_revision: u64,
    frame_revision: u64,
    extraction_errors: u64,
    observation_errors: u64,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RenderShadowDiagnostics {
    asset_revision: u64,
    pose_revision: u64,
    scene_revision: u64,
    frame_revision: u64,
    extraction_errors: u64,
    observation_errors: u64,
    last_error: Option<String>,
    parity: RenderParityDiagnostics,
}

impl RenderShadowObserver {
    pub(crate) fn is_enabled(&self) -> bool {
        self.parity.is_enabled()
    }

    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        if self.parity.is_enabled() == enabled {
            return;
        }
        self.parity.set_enabled(enabled);
        self.extraction_errors = 0;
        self.observation_errors = 0;
        self.last_error = None;
    }

    pub(crate) fn asset_changed(&mut self) {
        self.asset_revision = self.asset_revision.saturating_add(1);
        self.pose_revision = 0;
    }

    pub(crate) fn pose_changed(&mut self) {
        self.pose_revision = self.pose_revision.saturating_add(1);
    }

    pub(crate) fn replace_scene(&mut self, mut scene: RenderSceneSnapshot) {
        if !self.is_enabled() {
            return;
        }
        let revision = self.scene_revision.saturating_add(1);
        scene.revision = revision;
        match self.parity.replace_scene(scene) {
            Ok(()) => {
                self.scene_revision = revision;
                self.last_error = None;
            }
            Err(error) => self.record_extraction_error(error),
        }
    }

    pub(crate) fn record_extraction_error(&mut self, error: impl ToString) {
        if !self.is_enabled() {
            return;
        }
        self.extraction_errors = self.extraction_errors.saturating_add(1);
        self.last_error = Some(error.to_string());
    }

    pub(crate) fn observe(
        &mut self,
        style: RenderStyle,
        view: RenderView,
        options: RenderFrameOptions,
        actual: RenderSubmissionStats,
    ) {
        if !self.is_enabled() {
            return;
        }
        let revision = self.frame_revision.saturating_add(1);
        let built = self
            .parity
            .scene()
            .ok_or(RenderContractError::ObserverSceneUnavailable)
            .and_then(|scene| {
                RenderFrame::build(
                    revision,
                    RenderPoseIdentity {
                        asset_revision: self.asset_revision,
                        pose_revision: self.pose_revision,
                    },
                    style,
                    view,
                    options,
                    scene,
                )
            });
        let result = built.and_then(|frame| self.parity.observe(&frame, actual));
        match result {
            Ok(_) => {
                self.frame_revision = revision;
                self.last_error = None;
            }
            Err(error) => {
                self.observation_errors = self.observation_errors.saturating_add(1);
                self.last_error = Some(error.to_string());
            }
        }
    }

    pub(crate) fn to_js(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&RenderShadowDiagnostics {
            asset_revision: self.asset_revision,
            pose_revision: self.pose_revision,
            scene_revision: self.scene_revision,
            frame_revision: self.frame_revision,
            extraction_errors: self.extraction_errors,
            observation_errors: self.observation_errors,
            last_error: self.last_error.clone(),
            parity: self.parity.diagnostics(),
        })
        .unwrap_or(JsValue::NULL)
    }
}
