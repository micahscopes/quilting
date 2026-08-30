//! Thin WASM adapter for the backend-neutral render parity observer.
//!
//! High-rate comparisons remain entirely in Rust. JavaScript receives only a
//! bounded diagnostic snapshot when it explicitly calls the exported query.

use quilting_core::render::{
    RenderCommandPlan, RenderContractError, RenderFrameOptions, RenderParityDiagnostics,
    RenderParityObserver, RenderStyle, RenderSubmissionStats, RenderView, ValidatedRenderScene,
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
    resolved_execution_frames: u64,
    resolved_execution_fallbacks: u64,
    last_execution_error: Option<String>,
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
    resolved_execution_frames: u64,
    resolved_execution_fallbacks: u64,
    last_execution_error: Option<String>,
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
        self.resolved_execution_frames = 0;
        self.resolved_execution_fallbacks = 0;
        self.last_execution_error = None;
        self.last_error = None;
    }

    pub(crate) fn asset_changed(&mut self) {
        self.asset_revision = self.asset_revision.saturating_add(1);
        self.pose_revision = 0;
    }

    pub(crate) fn pose_changed(&mut self) {
        self.pose_revision = self.pose_revision.saturating_add(1);
    }

    pub(crate) fn replace_scene(&mut self, scene: ValidatedRenderScene) {
        if !self.is_enabled() {
            return;
        }
        let revision = scene.snapshot().revision;
        match self.parity.replace_validated_scene(scene) {
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

    pub(crate) fn prepare_observation(
        &mut self,
        plan: &RenderCommandPlan,
        style: RenderStyle,
        view: RenderView,
        options: RenderFrameOptions,
    ) -> Option<u64> {
        if !self.is_enabled() {
            return None;
        }
        let revision = self.frame_revision.saturating_add(1);
        let result = self
            .parity
            .validated_scene()
            .ok_or(RenderContractError::ObserverSceneUnavailable)
            .and_then(|scene| plan.validate_for(scene, style, options))
            .and_then(|()| view.validate())
            .and_then(|()| {
                options
                    .focus_postprocess
                    .map_or(Ok(()), |focus| focus.validate().map(|_| ()))
            });
        match result {
            Ok(()) => Some(revision),
            Err(error) => {
                self.record_observation_error(error);
                None
            }
        }
    }

    pub(crate) fn observe_prepared(
        &mut self,
        frame_revision: u64,
        plan: &RenderCommandPlan,
        actual: RenderSubmissionStats,
    ) {
        if !self.is_enabled() {
            return;
        }
        match self
            .parity
            .observe_execution(frame_revision, plan.execution(), actual)
        {
            Ok(_) => {
                self.frame_revision = frame_revision;
                self.last_error = None;
            }
            Err(error) => self.record_observation_error(error),
        }
    }

    fn record_observation_error(&mut self, error: impl ToString) {
        self.observation_errors = self.observation_errors.saturating_add(1);
        self.last_error = Some(error.to_string());
    }

    pub(crate) fn record_execution_success(&mut self) {
        self.resolved_execution_frames = self.resolved_execution_frames.saturating_add(1);
        self.last_execution_error = None;
    }

    pub(crate) fn record_execution_fallback(&mut self, error: impl ToString) {
        self.resolved_execution_fallbacks = self.resolved_execution_fallbacks.saturating_add(1);
        self.last_execution_error = Some(error.to_string());
    }

    pub(crate) fn to_js(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&RenderShadowDiagnostics {
            asset_revision: self.asset_revision,
            pose_revision: self.pose_revision,
            scene_revision: self.scene_revision,
            frame_revision: self.frame_revision,
            extraction_errors: self.extraction_errors,
            observation_errors: self.observation_errors,
            resolved_execution_frames: self.resolved_execution_frames,
            resolved_execution_fallbacks: self.resolved_execution_fallbacks,
            last_execution_error: self.last_execution_error.clone(),
            last_error: self.last_error.clone(),
            parity: self.parity.diagnostics(),
        })
        .unwrap_or(JsValue::NULL)
    }
}
