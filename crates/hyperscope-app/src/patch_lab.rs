//! Rust-authoritative state and effect protocol for the educational patch lab.
//!
//! Geometry construction and tessellation can run in a browser worker, a
//! native task, or a future WebGPU preparation lane. The reducer owns the
//! controls, generations, coalescing, animation clock, and stale-completion
//! policy so those adapters cannot disagree about which result is current.

pub use quilting_core::educational::{PatchLabField, PatchLabShape};

use crate::{AppEffect, ReduceError, RenderSettings};

pub const PATCH_LAB_PHASE_TURN_MICRORADIANS: u32 = 6_283_185;
pub const PATCH_LAB_ANIMATION_RATE_MICRORADIANS_PER_SECOND: f64 = 1_350_000.0;
pub const PATCH_LAB_ANIMATION_SAMPLE_SECONDS: f64 = 0.09;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabControls {
    pub shape: PatchLabShape,
    pub field: PatchLabField,
    pub manual_edge_exponents: [u8; 3],
    pub min_exponent: u8,
    pub max_exponent: u8,
    pub phase_microradians: u32,
    pub bend_percent: u8,
    pub grid: u8,
    pub animate: bool,
}

impl Default for PatchLabControls {
    fn default() -> Self {
        Self {
            shape: PatchLabShape::Triangle,
            field: PatchLabField::ManualEdges,
            manual_edge_exponents: [3, 4, 4],
            min_exponent: 1,
            max_exponent: 6,
            phase_microradians: 0,
            bend_percent: 55,
            grid: 8,
            animate: false,
        }
    }
}

impl PatchLabControls {
    pub fn normalized(mut self, atlas_exponent: u8) -> Self {
        let atlas_exponent = atlas_exponent.min(9);
        self.phase_microradians %= PATCH_LAB_PHASE_TURN_MICRORADIANS;
        self.bend_percent = self.bend_percent.min(100);
        self.grid = self.grid.clamp(1, 32);
        self.min_exponent = self.min_exponent.min(atlas_exponent);
        self.max_exponent = self.max_exponent.clamp(self.min_exponent, atlas_exponent);
        self.manual_edge_exponents = self
            .manual_edge_exponents
            .map(|value| value.clamp(self.min_exponent, self.max_exponent));
        if self.shape != PatchLabShape::Triangle && self.field == PatchLabField::ManualEdges {
            self.field = PatchLabField::Wave;
        }
        self
    }

    pub fn phase_radians(self) -> f64 {
        f64::from(self.phase_microradians) / 1_000_000.0
    }

    fn geometry_key(self) -> PatchLabGeometryKey {
        PatchLabGeometryKey {
            shape: self.shape,
            grid: if self.shape == PatchLabShape::Plane {
                self.grid
            } else {
                1
            },
            bend_percent: if self.shape == PatchLabShape::Triangle {
                self.bend_percent
            } else {
                0
            },
        }
    }

    fn lod_parameters(self, render: RenderSettings) -> PatchLabLodParameters {
        PatchLabLodParameters {
            field: self.field,
            phase_microradians: self.phase_microradians,
            min_exponent: self.min_exponent,
            max_exponent: self.max_exponent,
            manual_edge_exponents: self.manual_edge_exponents,
            atlas_exponent: render.atlas_exponent,
            max_face_edge_ratio: render.max_face_edge_ratio,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabSessionIntent {
    pub active: bool,
    pub controls: PatchLabControls,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabGeometryKey {
    pub shape: PatchLabShape,
    pub grid: u8,
    pub bend_percent: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabLodParameters {
    pub field: PatchLabField,
    pub phase_microradians: u32,
    pub min_exponent: u8,
    pub max_exponent: u8,
    pub manual_edge_exponents: [u8; 3],
    pub atlas_exponent: u8,
    pub max_face_edge_ratio: u8,
}

impl PatchLabLodParameters {
    pub fn phase_radians(self) -> f64 {
        f64::from(self.phase_microradians) / 1_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "type", rename_all = "snake_case"))]
pub enum PatchLabEffect {
    BuildGeometry {
        job_id: u64,
        geometry: PatchLabGeometryKey,
    },
    CancelGeometry {
        job_id: u64,
    },
    DiscardGeometry {
        geometry_job_id: u64,
    },
    EvaluateLod {
        job_id: u64,
        geometry_job_id: u64,
        parameters: PatchLabLodParameters,
    },
    CancelLod {
        job_id: u64,
        geometry_job_id: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "status", rename_all = "snake_case"))]
pub enum PatchLabGeometryOutcome {
    Built { vertex_count: u32, face_count: u32 },
    Failed(PatchLabFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabGeometryCompletion {
    pub job_id: u64,
    pub outcome: PatchLabGeometryOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabHistogramBin {
    pub edge_subdivisions: [u32; 3],
    pub face_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabLodSummary {
    pub requested_first_face: Option<[u32; 3]>,
    pub resident_first_face: Option<[u32; 3]>,
    pub promoted_faces: u32,
    pub promoted_edges: u32,
    pub shared_edges: u32,
    pub shared_edge_mismatches: u32,
    pub max_face_edge_ratio: u32,
    pub rendered_triangles: u64,
    pub histogram: Vec<PatchLabHistogramBin>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "status", rename_all = "snake_case"))]
pub enum PatchLabLodOutcome {
    Evaluated(PatchLabLodSummary),
    Failed(PatchLabFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabLodCompletion {
    pub job_id: u64,
    pub geometry_job_id: u64,
    pub outcome: PatchLabLodOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "type", rename_all = "snake_case"))]
pub enum PatchLabCompletion {
    Geometry(PatchLabGeometryCompletion),
    Lod(PatchLabLodCompletion),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabGeometryReadModel {
    pub job_id: u64,
    pub geometry: PatchLabGeometryKey,
    pub vertex_count: u32,
    pub face_count: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct PatchLabReadModel {
    pub active: bool,
    pub controls: PatchLabControls,
    pub pending_geometry_job: Option<u64>,
    pub installed_geometry: Option<PatchLabGeometryReadModel>,
    pub pending_lod_job: Option<u64>,
    pub lod_dirty: bool,
    pub latest_lod: Option<PatchLabLodSummary>,
    pub last_error: Option<PatchLabFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingGeometry {
    job_id: u64,
    geometry: PatchLabGeometryKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingLod {
    job_id: u64,
    geometry_job_id: u64,
    parameters: PatchLabLodParameters,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PatchLabRuntime {
    active: bool,
    controls: PatchLabControls,
    next_job_id: Option<u64>,
    pending_geometry: Option<PendingGeometry>,
    installed_geometry: Option<PatchLabGeometryReadModel>,
    pending_lod: Option<PendingLod>,
    lod_dirty: bool,
    latest_lod: Option<PatchLabLodSummary>,
    last_error: Option<PatchLabFailure>,
    animation_epoch_seconds: f64,
    animation_epoch_phase_microradians: u32,
    next_animation_sample_seconds: f64,
}

impl Default for PatchLabRuntime {
    fn default() -> Self {
        Self {
            active: false,
            controls: PatchLabControls::default(),
            next_job_id: Some(0),
            pending_geometry: None,
            installed_geometry: None,
            pending_lod: None,
            lod_dirty: false,
            latest_lod: None,
            last_error: None,
            animation_epoch_seconds: 0.0,
            animation_epoch_phase_microradians: 0,
            next_animation_sample_seconds: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PatchLabCompletionDisposition {
    Applied,
    Stale(String),
}

impl PatchLabRuntime {
    pub(crate) fn summary(&self) -> (bool, Option<u64>, Option<u64>) {
        (
            self.active,
            self.pending_geometry.map(|pending| pending.job_id),
            self.pending_lod.map(|pending| pending.job_id),
        )
    }

    pub(crate) fn read_model(&self) -> PatchLabReadModel {
        PatchLabReadModel {
            active: self.active,
            controls: self.controls,
            pending_geometry_job: self.pending_geometry.map(|pending| pending.job_id),
            installed_geometry: self.installed_geometry,
            pending_lod_job: self.pending_lod.map(|pending| pending.job_id),
            lod_dirty: self.lod_dirty,
            latest_lod: self.latest_lod.clone(),
            last_error: self.last_error.clone(),
        }
    }

    pub(crate) fn apply_session(
        &mut self,
        intent: PatchLabSessionIntent,
        render: RenderSettings,
        now_seconds: f64,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        let controls = intent.controls.normalized(render.atlas_exponent);
        let was_active = self.active;
        let old_controls = self.controls;
        let old_geometry = old_controls.geometry_key();
        let new_geometry = controls.geometry_key();

        if !intent.active {
            self.cancel_pending(effects);
            if let Some(installed) = self.installed_geometry.take() {
                effects.push(AppEffect::PatchLab(PatchLabEffect::DiscardGeometry {
                    geometry_job_id: installed.job_id,
                }));
            }
            self.active = false;
            self.controls = controls;
            self.lod_dirty = false;
            self.latest_lod = None;
            self.last_error = None;
            self.reset_animation_epoch(now_seconds);
            return Ok(());
        }

        self.active = true;
        self.controls = controls;
        if !was_active
            || old_controls.phase_microradians != controls.phase_microradians
            || old_controls.animate != controls.animate
        {
            self.reset_animation_epoch(now_seconds);
        }

        if !was_active || old_geometry != new_geometry {
            self.replace_geometry(new_geometry, effects)?;
        } else if old_controls != controls {
            self.request_lod(render, effects)?;
        }
        Ok(())
    }

    pub(crate) fn render_settings_changed(
        &mut self,
        render: RenderSettings,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        let normalized = self.controls.normalized(render.atlas_exponent);
        self.controls = normalized;
        if self.active {
            self.request_lod(render, effects)?;
        }
        Ok(())
    }

    pub(crate) fn advance(
        &mut self,
        now_seconds: f64,
        render: RenderSettings,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        if !self.active || !self.controls.animate {
            return Ok(());
        }
        self.controls.phase_microradians = self.phase_at(now_seconds);
        if now_seconds + f64::EPSILON >= self.next_animation_sample_seconds {
            self.next_animation_sample_seconds = now_seconds + PATCH_LAB_ANIMATION_SAMPLE_SECONDS;
            self.request_lod(render, effects)?;
        }
        Ok(())
    }

    pub(crate) fn complete(
        &mut self,
        completion: PatchLabCompletion,
        render: RenderSettings,
        effects: &mut Vec<AppEffect>,
    ) -> Result<PatchLabCompletionDisposition, ReduceError> {
        match completion {
            PatchLabCompletion::Geometry(completion) => {
                self.complete_geometry(completion, render, effects)
            }
            PatchLabCompletion::Lod(completion) => self.complete_lod(completion, render, effects),
        }
    }

    fn complete_geometry(
        &mut self,
        completion: PatchLabGeometryCompletion,
        render: RenderSettings,
        effects: &mut Vec<AppEffect>,
    ) -> Result<PatchLabCompletionDisposition, ReduceError> {
        let Some(pending) = self.pending_geometry else {
            return Ok(PatchLabCompletionDisposition::Stale(format!(
                "ignored patch-lab geometry completion for inactive job {}",
                completion.job_id
            )));
        };
        if pending.job_id != completion.job_id {
            return Ok(PatchLabCompletionDisposition::Stale(format!(
                "ignored patch-lab geometry completion {} while job {} is active",
                completion.job_id, pending.job_id
            )));
        }
        self.pending_geometry = None;
        match completion.outcome {
            PatchLabGeometryOutcome::Built {
                vertex_count,
                face_count,
            } => {
                self.installed_geometry = Some(PatchLabGeometryReadModel {
                    job_id: pending.job_id,
                    geometry: pending.geometry,
                    vertex_count,
                    face_count,
                });
                self.last_error = None;
                self.request_lod(render, effects)?;
            }
            PatchLabGeometryOutcome::Failed(error) => {
                self.installed_geometry = None;
                self.latest_lod = None;
                self.last_error = Some(error);
            }
        }
        Ok(PatchLabCompletionDisposition::Applied)
    }

    fn complete_lod(
        &mut self,
        completion: PatchLabLodCompletion,
        render: RenderSettings,
        effects: &mut Vec<AppEffect>,
    ) -> Result<PatchLabCompletionDisposition, ReduceError> {
        let Some(pending) = self.pending_lod else {
            return Ok(PatchLabCompletionDisposition::Stale(format!(
                "ignored patch-lab LOD completion for inactive job {}",
                completion.job_id
            )));
        };
        if pending.job_id != completion.job_id
            || pending.geometry_job_id != completion.geometry_job_id
        {
            return Ok(PatchLabCompletionDisposition::Stale(format!(
                "ignored patch-lab LOD completion {}:{} while job {}:{} is active",
                completion.geometry_job_id,
                completion.job_id,
                pending.geometry_job_id,
                pending.job_id
            )));
        }
        self.pending_lod = None;
        match completion.outcome {
            PatchLabLodOutcome::Evaluated(summary) => {
                self.latest_lod = Some(summary);
                self.last_error = None;
            }
            PatchLabLodOutcome::Failed(error) => {
                self.last_error = Some(error);
            }
        }
        if self.lod_dirty {
            self.lod_dirty = false;
            self.request_lod(render, effects)?;
        }
        Ok(PatchLabCompletionDisposition::Applied)
    }

    fn replace_geometry(
        &mut self,
        geometry: PatchLabGeometryKey,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        self.cancel_pending(effects);
        if let Some(installed) = self.installed_geometry.take() {
            effects.push(AppEffect::PatchLab(PatchLabEffect::DiscardGeometry {
                geometry_job_id: installed.job_id,
            }));
        }
        self.latest_lod = None;
        self.last_error = None;
        let job_id = self.allocate_job()?;
        self.pending_geometry = Some(PendingGeometry { job_id, geometry });
        effects.push(AppEffect::PatchLab(PatchLabEffect::BuildGeometry {
            job_id,
            geometry,
        }));
        Ok(())
    }

    fn request_lod(
        &mut self,
        render: RenderSettings,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        let Some(geometry) = self.installed_geometry else {
            return Ok(());
        };
        let parameters = self.controls.lod_parameters(render);
        if let Some(pending) = self.pending_lod {
            if pending.geometry_job_id == geometry.job_id && pending.parameters == parameters {
                return Ok(());
            }
            self.lod_dirty = true;
            return Ok(());
        }
        let job_id = self.allocate_job()?;
        self.pending_lod = Some(PendingLod {
            job_id,
            geometry_job_id: geometry.job_id,
            parameters,
        });
        effects.push(AppEffect::PatchLab(PatchLabEffect::EvaluateLod {
            job_id,
            geometry_job_id: geometry.job_id,
            parameters,
        }));
        Ok(())
    }

    fn cancel_pending(&mut self, effects: &mut Vec<AppEffect>) {
        if let Some(pending) = self.pending_lod.take() {
            effects.push(AppEffect::PatchLab(PatchLabEffect::CancelLod {
                job_id: pending.job_id,
                geometry_job_id: pending.geometry_job_id,
            }));
        }
        if let Some(pending) = self.pending_geometry.take() {
            effects.push(AppEffect::PatchLab(PatchLabEffect::CancelGeometry {
                job_id: pending.job_id,
            }));
        }
        self.lod_dirty = false;
    }

    fn allocate_job(&mut self) -> Result<u64, ReduceError> {
        let job_id = self.next_job_id.ok_or(ReduceError::PatchLabJobExhausted)?;
        self.next_job_id = job_id.checked_add(1);
        Ok(job_id)
    }

    fn reset_animation_epoch(&mut self, now_seconds: f64) {
        self.animation_epoch_seconds = now_seconds;
        self.animation_epoch_phase_microradians = self.controls.phase_microradians;
        self.next_animation_sample_seconds = now_seconds;
    }

    fn phase_at(&self, now_seconds: f64) -> u32 {
        let elapsed = (now_seconds - self.animation_epoch_seconds).max(0.0);
        let advance = (elapsed * PATCH_LAB_ANIMATION_RATE_MICRORADIANS_PER_SECOND).round() as u64;
        ((u64::from(self.animation_epoch_phase_microradians) + advance)
            % u64::from(PATCH_LAB_PHASE_TURN_MICRORADIANS)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AppEvent, AppStore, CommitDisposition, EffectCompletion, FrameTick, SemanticAction,
    };

    fn active_session(controls: PatchLabControls) -> PatchLabSessionIntent {
        PatchLabSessionIntent {
            active: true,
            controls,
        }
    }

    fn geometry_built(job_id: u64) -> AppEvent {
        AppEvent::EffectCompleted(EffectCompletion::PatchLab(PatchLabCompletion::Geometry(
            PatchLabGeometryCompletion {
                job_id,
                outcome: PatchLabGeometryOutcome::Built {
                    vertex_count: 3,
                    face_count: 1,
                },
            },
        )))
    }

    fn lod_evaluated(job_id: u64, geometry_job_id: u64) -> AppEvent {
        AppEvent::EffectCompleted(EffectCompletion::PatchLab(PatchLabCompletion::Lod(
            PatchLabLodCompletion {
                job_id,
                geometry_job_id,
                outcome: PatchLabLodOutcome::Evaluated(PatchLabLodSummary {
                    requested_first_face: Some([8, 16, 16]),
                    resident_first_face: Some([8, 16, 16]),
                    promoted_faces: 0,
                    promoted_edges: 0,
                    shared_edges: 0,
                    shared_edge_mismatches: 0,
                    max_face_edge_ratio: 2,
                    rendered_triangles: 256,
                    histogram: vec![PatchLabHistogramBin {
                        edge_subdivisions: [8, 16, 16],
                        face_count: 1,
                    }],
                }),
            },
        )))
    }

    #[test]
    fn activation_builds_geometry_then_requests_exact_lod() {
        let store = AppStore::default();
        let controls = PatchLabControls::default();
        let (_, activation) = store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(controls)))
            .unwrap();
        let geometry = controls.geometry_key();
        assert_eq!(
            activation.effects,
            vec![AppEffect::PatchLab(PatchLabEffect::BuildGeometry {
                job_id: 0,
                geometry,
            })]
        );
        assert_eq!(store.patch_lab_snapshot().pending_geometry_job, Some(0));

        let installed = store.dispatch(geometry_built(0)).unwrap();
        assert_eq!(
            installed.effects,
            vec![AppEffect::PatchLab(PatchLabEffect::EvaluateLod {
                job_id: 1,
                geometry_job_id: 0,
                parameters: controls.lod_parameters(RenderSettings::default()),
            })]
        );
        let read = store.patch_lab_snapshot();
        assert_eq!(read.pending_geometry_job, None);
        assert_eq!(read.pending_lod_job, Some(1));
        assert_eq!(read.installed_geometry.unwrap().geometry, geometry);
    }

    #[test]
    fn rapid_lod_edits_coalesce_and_stale_completion_is_ignored() {
        let store = AppStore::default();
        let controls = PatchLabControls::default();
        store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(controls)))
            .unwrap();
        store.dispatch(geometry_built(0)).unwrap();

        let newest = PatchLabControls {
            field: PatchLabField::Radial,
            phase_microradians: 2_250_000,
            ..controls
        };
        let (_, first_edit) = store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(
                PatchLabControls {
                    field: PatchLabField::Wave,
                    ..controls
                },
            )))
            .unwrap();
        let (_, second_edit) = store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(newest)))
            .unwrap();
        assert!(first_edit.effects.is_empty());
        assert!(second_edit.effects.is_empty());
        assert!(store.patch_lab_snapshot().lod_dirty);

        let completion = store.dispatch(lod_evaluated(1, 0)).unwrap();
        assert_eq!(
            completion.effects,
            vec![AppEffect::PatchLab(PatchLabEffect::EvaluateLod {
                job_id: 2,
                geometry_job_id: 0,
                parameters: newest.lod_parameters(RenderSettings::default()),
            })]
        );
        let stale = store.dispatch(lod_evaluated(1, 0)).unwrap();
        assert_eq!(stale.disposition, CommitDisposition::IgnoredStale);
        assert_eq!(store.patch_lab_snapshot().pending_lod_job, Some(2));
    }

    #[test]
    fn geometry_edit_cancels_prior_generation_before_replacement() {
        let store = AppStore::default();
        let controls = PatchLabControls::default();
        store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(controls)))
            .unwrap();
        store.dispatch(geometry_built(0)).unwrap();

        let plane = PatchLabControls {
            shape: PatchLabShape::Plane,
            field: PatchLabField::Wave,
            grid: 12,
            ..controls
        };
        let (_, changed) = store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(plane)))
            .unwrap();
        assert_eq!(
            changed.effects,
            vec![
                AppEffect::PatchLab(PatchLabEffect::CancelLod {
                    job_id: 1,
                    geometry_job_id: 0,
                }),
                AppEffect::PatchLab(PatchLabEffect::DiscardGeometry { geometry_job_id: 0 }),
                AppEffect::PatchLab(PatchLabEffect::BuildGeometry {
                    job_id: 2,
                    geometry: plane.geometry_key(),
                }),
            ]
        );
    }

    #[test]
    fn animation_phase_is_independent_of_frame_partitioning() {
        let one_frame = AppStore::default();
        let many_frames = AppStore::default();
        let controls = PatchLabControls {
            animate: true,
            ..PatchLabControls::default()
        };
        for store in [&one_frame, &many_frames] {
            store
                .dispatch_semantic(SemanticAction::SetPatchLab(active_session(controls)))
                .unwrap();
        }
        one_frame
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 1.0,
                delta_seconds: 1.0,
            }))
            .unwrap();
        for step in 1..=10 {
            many_frames
                .dispatch(AppEvent::Frame(FrameTick {
                    elapsed_seconds: f64::from(step) / 10.0,
                    delta_seconds: 0.1,
                }))
                .unwrap();
        }
        one_frame.flush_read_models();
        many_frames.flush_read_models();
        assert_eq!(
            one_frame.patch_lab_snapshot().controls.phase_microradians,
            many_frames.patch_lab_snapshot().controls.phase_microradians
        );
        assert_eq!(
            one_frame.patch_lab_snapshot().controls.phase_microradians,
            1_350_000
        );
    }

    #[test]
    fn atlas_and_grading_changes_clamp_controls_and_request_new_lod() {
        let store = AppStore::default();
        let controls = PatchLabControls::default();
        store
            .dispatch_semantic(SemanticAction::SetPatchLab(active_session(controls)))
            .unwrap();
        store.dispatch(geometry_built(0)).unwrap();
        store.dispatch(lod_evaluated(1, 0)).unwrap();

        let render = RenderSettings {
            atlas_exponent: 3,
            max_face_edge_ratio: 4,
            ..RenderSettings::default()
        };
        let (_, changed) = store
            .dispatch_semantic(SemanticAction::SetRenderSettings(render))
            .unwrap();
        let normalized = controls.normalized(3);
        assert_eq!(store.patch_lab_snapshot().controls, normalized);
        assert_eq!(
            changed.effects,
            vec![AppEffect::PatchLab(PatchLabEffect::EvaluateLod {
                job_id: 2,
                geometry_job_id: 0,
                parameters: normalized.lod_parameters(render),
            })]
        );
    }
}
