//! Read-only renderer oracle for metric-adaptive QB patch work.
//!
//! This module owns no browser or GPU state. The WebGL adapter supplies one
//! current output-chart patch, camera, active atlas counts, and policy. Keeping
//! the measurement pure prevents experimental refinement policy from leaking
//! into the live frame renderer before its workload and failure modes are
//! understood.

use std::collections::BTreeMap;

use quilting_core::batch::{
    group_resident_screen_leaves_into, RenderBatchKey, RenderBatchMember, ResidentLod,
};
use quilting_core::patch::{QBPatchDomain, QBTriPatch};
use quilting_core::permutation::canonical_form;
use quilting_core::screen_leaf_lod::{
    reconcile_screen_leaf_lods, ScreenLeafLodResult, ScreenLeafTopology,
    ScreenMeshLeafFrontier, ScreenMeshLeafLodScratch, ScreenMeshLeafTopology,
    ScreenMeshTopologyCache,
};
use quilting_core::screen_partition::{
    diagnose_screen_patch, partition_screen_patch, ScreenPartitionPolicy, ScreenPatchDiagnostic,
    ScreenPatchLeafId, ScreenPatchLeafStatus,
};
use quilting_core::screen_plan::{
    inherited_source_edge_lods, plan_adaptive_screen_mesh,
    AdaptiveScreenFaceSelectionDiagnostic, AdaptiveScreenMeshPlanDiagnostic,
    AdaptiveScreenMeshPlanRequest, SelectedScreenPatch,
};
use serde::Serialize;

use crate::round_shadow::browser_now_ms;

/// Disabled-by-default parity observer for the production batch handoff.
///
/// An all-root adaptive frontier must be exactly equivalent to legacy
/// source-face grouping before any dyadic leaf is allowed to change live
/// rendering. The observer retains its frontier, grouping map, and work
/// buffers so repeated classifications measure the real steady-state path
/// rather than allocator noise.
#[derive(Default)]
pub(crate) struct AdaptiveRootShadow {
    enabled: bool,
    frontier: Option<ScreenMeshLeafFrontier>,
    root_lods: Vec<[u32; 3]>,
    lod_scratch: ScreenMeshLeafLodScratch,
    groups: BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    comparisons: u64,
    matches: u64,
    mismatches: u64,
    failures: u64,
    last_match: Option<bool>,
    last_error: Option<String>,
    last_legacy_batches: u64,
    last_shadow_batches: u64,
    last_legacy_instances: u64,
    last_shadow_instances: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AdaptiveRootShadowSnapshot<'a> {
    enabled: bool,
    state: &'static str,
    comparisons: u64,
    matches: u64,
    mismatches: u64,
    failures: u64,
    last_match: Option<bool>,
    last_error: Option<&'a str>,
    last_legacy_batches: u64,
    last_shadow_batches: u64,
    last_legacy_instances: u64,
    last_shadow_instances: u64,
}

impl AdaptiveRootShadow {
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.last_match = None;
        self.last_error = None;
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Invalidate source-topology-dependent state. Batch keys from a previous
    /// asset are not meaningful for the replacement and must not leak into
    /// diagnostics if rebuilding its frontier fails.
    pub(crate) fn reset_topology(&mut self) {
        self.frontier = None;
        self.root_lods.clear();
        self.groups.clear();
        self.last_match = None;
        self.last_error = None;
        self.last_legacy_batches = 0;
        self.last_shadow_batches = 0;
        self.last_legacy_instances = 0;
        self.last_shadow_instances = 0;
    }

    pub(crate) fn record_unavailable(&mut self, error: impl Into<String>) {
        if !self.enabled {
            return;
        }
        self.failures = self.failures.saturating_add(1);
        self.last_match = None;
        self.last_error = Some(error.into());
        self.last_legacy_batches = 0;
        self.last_shadow_batches = 0;
        self.last_legacy_instances = 0;
        self.last_shadow_instances = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compare(
        &mut self,
        topology: &ScreenMeshTopologyCache,
        residents: &[Option<ResidentLod>],
        initial: ResidentLod,
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        legacy_groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    ) {
        if !self.enabled {
            return;
        }
        self.comparisons = self.comparisons.saturating_add(1);
        let comparison = (|| -> Result<bool, String> {
            if self
                .frontier
                .as_ref()
                .is_none_or(|frontier| frontier.leaves().len() != residents.len())
            {
                let leaves = (0..residents.len())
                    .map(|source_face| {
                        Ok(ScreenMeshLeafTopology {
                            source_face: u32::try_from(source_face)
                                .map_err(|_| "adaptive root face identity exceeds u32")?,
                            id: quilting_core::screen_partition::ScreenPatchLeafId::ROOT,
                            domain: QBPatchDomain::FULL,
                        })
                    })
                    .collect::<Result<Vec<_>, &str>>()?;
                self.frontier = Some(
                    ScreenMeshLeafFrontier::build(&leaves, topology)
                        .map_err(|error| error.to_string())?,
                );
            }
            self.root_lods.clear();
            self.root_lods.extend(
                residents
                    .iter()
                    .map(|resident| resident.unwrap_or(initial).edge_lods()),
            );
            group_resident_screen_leaves_into(
                self.frontier
                    .as_ref()
                    .ok_or_else(|| "adaptive root frontier is unavailable".to_string())?,
                &self.root_lods,
                face_materials,
                face_nodes,
                face_render_nodes,
                &mut self.lod_scratch,
                &mut self.groups,
            )
            .map_err(|error| error.to_string())?;
            Ok(self.groups == *legacy_groups)
        })();

        self.last_legacy_batches = legacy_groups.len() as u64;
        self.last_shadow_batches = self.groups.len() as u64;
        self.last_legacy_instances = legacy_groups
            .values()
            .map(|members| members.len() as u64)
            .sum();
        self.last_shadow_instances = self
            .groups
            .values()
            .map(|members| members.len() as u64)
            .sum();
        match comparison {
            Ok(matches) => {
                self.last_match = Some(matches);
                self.last_error = None;
                if matches {
                    self.matches = self.matches.saturating_add(1);
                } else {
                    self.mismatches = self.mismatches.saturating_add(1);
                }
            }
            Err(error) => {
                self.failures = self.failures.saturating_add(1);
                self.last_match = None;
                self.last_error = Some(error);
            }
        }
    }

    pub(crate) fn snapshot(&self) -> AdaptiveRootShadowSnapshot<'_> {
        let state = if !self.enabled {
            "disabled"
        } else if self.last_error.is_some() {
            "error"
        } else {
            match self.last_match {
                Some(true) => "matching",
                Some(false) => "mismatch",
                None => "awaiting-classification",
            }
        };
        AdaptiveRootShadowSnapshot {
            enabled: self.enabled,
            state,
            comparisons: self.comparisons,
            matches: self.matches,
            mismatches: self.mismatches,
            failures: self.failures,
            last_match: self.last_match,
            last_error: self.last_error.as_deref(),
            last_legacy_batches: self.last_legacy_batches,
            last_shadow_batches: self.last_shadow_batches,
            last_legacy_instances: self.last_legacy_instances,
            last_shadow_instances: self.last_shadow_instances,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum AdaptiveScreenSelection {
    Picked {
        face: u32,
    },
    CurrentView {
        max_faces: usize,
        max_partition_leaves: usize,
    },
}

impl AdaptiveScreenSelection {
    pub(crate) fn picked_face(self) -> Option<u32> {
        match self {
            Self::Picked { face } => Some(face),
            Self::CurrentView { .. } => None,
        }
    }

    fn mode(self) -> &'static str {
        match self {
            Self::Picked { .. } => "picked",
            Self::CurrentView { .. } => "current-view",
        }
    }

    fn max_faces(self) -> usize {
        match self {
            Self::Picked { .. } => 1,
            Self::CurrentView { max_faces, .. } => max_faces,
        }
    }

    fn max_partition_leaves(self, per_face_leaves: usize) -> usize {
        match self {
            Self::Picked { .. } => per_face_leaves,
            Self::CurrentView {
                max_partition_leaves,
                ..
            } => max_partition_leaves,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AdaptivePickedConfig {
    pub selection: AdaptiveScreenSelection,
    pub min_px_per_segment: f64,
    pub max_px_per_segment: f64,
    pub policy: ScreenPartitionPolicy,
    pub max_total_leaves: usize,
    pub max_triangles: u64,
}

/// Explicit, bounded live proof for one picked face in the latest accepted
/// pose. Each completed plan records the exact pose revision it measured.
///
/// Candidate groups are assembled off to the side and swapped with the
/// renderer's legacy groups only after partitioning, welded reconciliation,
/// atlas validation, and work-budget validation all succeed.
#[derive(Default)]
pub(crate) struct AdaptivePickedRuntime {
    config: Option<AdaptivePickedConfig>,
    source_lod_scratch: Vec<[u32; 3]>,
    /// Exact welded incidence for the last stable dyadic leaf topology. Camera
    /// and animation changes commonly alter only requested LoDs, so rebuilding
    /// this mesh-sized structure on every refresh is avoidable work.
    frontier: Option<ScreenMeshLeafFrontier>,
    frontier_cache_hits: u64,
    frontier_cache_misses: u64,
    reconciliation_cache: AdaptiveReconciliationCache,
    last_timings: AdaptivePlanTimings,
    lod_scratch: ScreenMeshLeafLodScratch,
    candidate_groups: BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    attempts: u64,
    installs: u64,
    fallbacks: u64,
    last_error: Option<String>,
    last_publication_error: Option<String>,
    last_plan: Option<AdaptiveScreenMeshPlanDiagnostic>,
    last_selection: Option<AdaptiveScreenFaceSelectionDiagnostic>,
    last_triangles: u64,
    last_shared_edge_promotions: u64,
    last_grading_promotions: u64,
    last_reconciliation_iterations: u64,
    last_pose_stamp: Option<(u32, u32)>,
    last_published_faces: Vec<u32>,
    pending_plan: Option<AdaptiveScreenMeshPlanDiagnostic>,
    pending_selection: Option<AdaptiveScreenFaceSelectionDiagnostic>,
    pending_selected_faces: Vec<u32>,
    pending_fallback_error: Option<String>,
    pending_legacy: bool,
    pending_triangles: u64,
    pending_shared_edge_promotions: u64,
    pending_grading_promotions: u64,
    pending_reconciliation_iterations: u64,
    pending_pose_stamp: Option<(u32, u32)>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AdaptivePickedSnapshot<'a> {
    enabled: bool,
    state: &'static str,
    mode: Option<&'static str>,
    face: Option<u32>,
    max_selected_faces: Option<u64>,
    min_px_per_segment: Option<f64>,
    max_px_per_segment: Option<f64>,
    max_depth: Option<u8>,
    max_selected_leaves: Option<u64>,
    max_partition_leaves: Option<u64>,
    max_total_leaves: Option<u64>,
    max_triangles: Option<u64>,
    pose_revision: Option<u32>,
    pose_continuity_epoch: Option<u32>,
    published_face: Option<u32>,
    published_faces: &'a [u32],
    attempts: u64,
    installs: u64,
    fallbacks: u64,
    last_error: Option<&'a str>,
    last_publication_error: Option<&'a str>,
    last_plan: Option<AdaptiveScreenMeshPlanDiagnosticSnapshot>,
    last_selection: Option<AdaptiveScreenFaceSelectionDiagnosticSnapshot>,
    last_triangles: u64,
    last_shared_edge_promotions: u64,
    last_grading_promotions: u64,
    last_reconciliation_iterations: u64,
    frontier_cache_hits: u64,
    frontier_cache_misses: u64,
    reconciliation_cache_hits: u64,
    reconciliation_cache_misses: u64,
    last_timings: AdaptivePlanTimings,
}

#[derive(Default)]
struct AdaptiveReconciliationCache {
    requested_lods: Vec<[u32; 3]>,
    max_face_edge_ratio: u32,
    max_atlas_lod: u32,
    result: Option<ScreenLeafLodResult>,
    hits: u64,
    misses: u64,
}

impl AdaptiveReconciliationCache {
    fn invalidate(&mut self) {
        self.requested_lods.clear();
        self.result = None;
    }

    fn resolve<'a>(
        &'a mut self,
        frontier: &ScreenMeshLeafFrontier,
        requested_lods: &[[u32; 3]],
        max_face_edge_ratio: u32,
        max_atlas_lod: u32,
    ) -> Result<&'a ScreenLeafLodResult, String> {
        let matches = self.result.is_some()
            && self.max_face_edge_ratio == max_face_edge_ratio
            && self.max_atlas_lod == max_atlas_lod
            && self.requested_lods == requested_lods;
        if matches {
            self.hits = self.hits.saturating_add(1);
        } else {
            let result = frontier
                .reconcile_lods(requested_lods, max_face_edge_ratio, max_atlas_lod)
                .map_err(|error| error.to_string())?;
            self.requested_lods.clear();
            self.requested_lods.extend_from_slice(requested_lods);
            self.max_face_edge_ratio = max_face_edge_ratio;
            self.max_atlas_lod = max_atlas_lod;
            self.result = Some(result);
            self.misses = self.misses.saturating_add(1);
        }
        self.result
            .as_ref()
            .ok_or_else(|| "adaptive reconciliation cache is unavailable".to_string())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct AdaptivePlanTimings {
    total_ms: f64,
    mesh_plan_ms: f64,
    frontier_ms: f64,
    reconcile_ms: f64,
    atlas_work_ms: f64,
    group_ms: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AdaptivePickedRefreshSnapshot<'a> {
    #[serde(flatten)]
    pub snapshot: AdaptivePickedSnapshot<'a>,
    pub transition_pending: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_published: Option<bool>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct AdaptiveScreenMeshPlanDiagnosticSnapshot {
    source_faces: u32,
    selected_faces: u32,
    selected_leaves: u32,
    total_leaves: u32,
    split_nodes: u32,
    max_depth_reached: u8,
    unmet_leaves: u32,
    saturated_metric_leaves: u32,
    omitted_culled_leaves: u32,
    boundary_fallback_faces: u32,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
struct AdaptiveScreenFaceSelectionDiagnosticSnapshot {
    examined_faces: u64,
    visible_faces: u64,
    selected_faces: u64,
    partition_face_capacity: u64,
    omitted_by_capacity: u64,
    selected_root_triangles: u64,
}

impl From<AdaptiveScreenFaceSelectionDiagnostic> for AdaptiveScreenFaceSelectionDiagnosticSnapshot {
    fn from(diagnostic: AdaptiveScreenFaceSelectionDiagnostic) -> Self {
        Self {
            examined_faces: diagnostic.examined_faces,
            visible_faces: diagnostic.visible_faces,
            selected_faces: diagnostic.selected_faces,
            partition_face_capacity: diagnostic.partition_face_capacity,
            omitted_by_capacity: diagnostic.omitted_by_capacity,
            selected_root_triangles: diagnostic.selected_root_triangles,
        }
    }
}

impl From<AdaptiveScreenMeshPlanDiagnostic> for AdaptiveScreenMeshPlanDiagnosticSnapshot {
    fn from(diagnostic: AdaptiveScreenMeshPlanDiagnostic) -> Self {
        Self {
            source_faces: diagnostic.source_faces,
            selected_faces: diagnostic.selected_faces,
            selected_leaves: diagnostic.selected_leaves,
            total_leaves: diagnostic.total_leaves,
            split_nodes: diagnostic.split_nodes,
            max_depth_reached: diagnostic.max_depth_reached,
            unmet_leaves: diagnostic.unmet_leaves,
            saturated_metric_leaves: diagnostic.saturated_metric_leaves,
            omitted_culled_leaves: diagnostic.omitted_culled_leaves,
            boundary_fallback_faces: diagnostic.boundary_fallback_faces,
        }
    }
}

impl AdaptivePickedRuntime {
    pub(crate) fn configure(&mut self, config: AdaptivePickedConfig) {
        self.config = Some(config);
        self.clear_pending_publication();
    }

    pub(crate) fn clear(&mut self) {
        self.config = None;
        self.last_error = None;
        self.last_publication_error = None;
        self.last_plan = None;
        self.last_selection = None;
        self.last_triangles = 0;
        self.last_shared_edge_promotions = 0;
        self.last_grading_promotions = 0;
        self.last_reconciliation_iterations = 0;
        self.last_pose_stamp = None;
        self.last_published_faces.clear();
        self.frontier = None;
        self.frontier_cache_hits = 0;
        self.frontier_cache_misses = 0;
        self.reconciliation_cache = AdaptiveReconciliationCache::default();
        self.last_timings = AdaptivePlanTimings::default();
        self.candidate_groups.clear();
        self.clear_pending_publication();
    }

    /// Request a transactional return to the retained legacy root grouping.
    /// Published diagnostics remain intact until that handoff succeeds.
    pub(crate) fn stage_clear(&mut self) {
        self.config = None;
        self.candidate_groups.clear();
        self.clear_pending_publication();
        self.pending_legacy = true;
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    pub(crate) fn has_pending_publication(&self) -> bool {
        self.pending_plan.is_some() || self.pending_fallback_error.is_some() || self.pending_legacy
    }

    pub(crate) fn config(&self) -> Option<AdaptivePickedConfig> {
        self.config
    }

    pub(crate) fn record_fallback(&mut self, error: impl Into<String>) {
        if self.config.is_none() {
            return;
        }
        self.attempts = self.attempts.saturating_add(1);
        self.fallbacks = self.fallbacks.saturating_add(1);
        self.clear_pending_publication();
        self.pending_fallback_error = Some(error.into());
    }

    fn clear_pending_publication(&mut self) {
        self.pending_plan = None;
        self.pending_selection = None;
        self.pending_fallback_error = None;
        self.pending_legacy = false;
        self.pending_triangles = 0;
        self.pending_shared_edge_promotions = 0;
        self.pending_grading_promotions = 0;
        self.pending_reconciliation_iterations = 0;
        self.pending_pose_stamp = None;
        self.pending_selected_faces.clear();
    }

    /// Commit diagnostics only after the renderer has published every staged
    /// GL bucket. CPU grouping alone is not a visible install.
    pub(crate) fn commit_publication(&mut self) {
        if let Some(plan) = self.pending_plan.take() {
            self.installs = self.installs.saturating_add(1);
            self.last_error = None;
            self.last_plan = Some(plan);
            self.last_selection = self.pending_selection.take();
            self.last_triangles = self.pending_triangles;
            self.last_shared_edge_promotions = self.pending_shared_edge_promotions;
            self.last_grading_promotions = self.pending_grading_promotions;
            self.last_reconciliation_iterations = self.pending_reconciliation_iterations;
            self.last_pose_stamp = self.pending_pose_stamp;
            std::mem::swap(
                &mut self.last_published_faces,
                &mut self.pending_selected_faces,
            );
        } else if let Some(error) = self.pending_fallback_error.take() {
            self.last_error = Some(error);
            self.last_plan = None;
            self.last_selection = None;
            self.last_triangles = 0;
            self.last_shared_edge_promotions = 0;
            self.last_grading_promotions = 0;
            self.last_reconciliation_iterations = 0;
            self.last_pose_stamp = None;
            self.last_published_faces.clear();
        } else if self.pending_legacy {
            self.last_error = None;
            self.last_plan = None;
            self.last_selection = None;
            self.last_triangles = 0;
            self.last_shared_edge_promotions = 0;
            self.last_grading_promotions = 0;
            self.last_reconciliation_iterations = 0;
            self.last_pose_stamp = None;
            self.last_published_faces.clear();
        } else {
            return;
        }
        self.last_publication_error = None;
        self.clear_pending_publication();
    }

    pub(crate) fn record_publication_failure(&mut self, error: impl Into<String>) {
        if self.pending_plan.is_some() || self.pending_fallback_error.is_some() {
            self.fallbacks = self.fallbacks.saturating_add(1);
        }
        let retry_legacy = self.pending_legacy;
        self.clear_pending_publication();
        self.pending_legacy = retry_legacy;
        self.last_publication_error = Some(error.into());
    }

    pub(crate) fn record_refresh_failure(&mut self, error: impl Into<String>) {
        if self.config.is_none() && !self.pending_legacy {
            return;
        }
        self.attempts = self.attempts.saturating_add(1);
        self.fallbacks = self.fallbacks.saturating_add(1);
        let retry_legacy = self.pending_legacy;
        self.clear_pending_publication();
        self.pending_legacy = retry_legacy;
        self.last_publication_error = Some(error.into());
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_and_group(
        &mut self,
        transformed_patch: &QBTriPatch,
        view_projection: &[f64; 16],
        viewport: [f64; 2],
        current_pose_stamp: Option<(u32, u32)>,
        source_requests: &[Option<ResidentLod>],
        standby: ResidentLod,
        topology: &ScreenMeshTopologyCache,
        max_atlas_lod: u32,
        max_face_edge_ratio: u32,
        atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        live_groups: &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    ) -> Result<(), String> {
        let config = self
            .config
            .ok_or_else(|| "picked adaptive screen mode is disabled".to_string())?;
        let face = config
            .selection
            .picked_face()
            .ok_or_else(|| "adaptive screen mode is not a picked-face selection".to_string())?;
        let selected = [SelectedScreenPatch {
            source_face: face,
            transformed_patch,
        }];
        self.plan_selected_and_group(
            &selected,
            None,
            config.policy.max_leaves,
            view_projection,
            viewport,
            current_pose_stamp,
            source_requests,
            standby,
            topology,
            max_atlas_lod,
            max_face_edge_ratio,
            atlas_triangle_counts,
            face_materials,
            face_nodes,
            face_render_nodes,
            live_groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_selected_and_group(
        &mut self,
        selected_patches: &[SelectedScreenPatch<'_>],
        selection_diagnostic: Option<AdaptiveScreenFaceSelectionDiagnostic>,
        max_partition_leaves: usize,
        view_projection: &[f64; 16],
        viewport: [f64; 2],
        current_pose_stamp: Option<(u32, u32)>,
        source_requests: &[Option<ResidentLod>],
        standby: ResidentLod,
        topology: &ScreenMeshTopologyCache,
        max_atlas_lod: u32,
        max_face_edge_ratio: u32,
        atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        live_groups: &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    ) -> Result<(), String> {
        if self.has_pending_publication() {
            return Err("adaptive plan is already awaiting publication".into());
        }
        self.attempts = self.attempts.saturating_add(1);
        self.clear_pending_publication();
        let Some(config) = self.config else {
            return Err("adaptive screen mode is disabled".into());
        };
        let attempt = (|| -> Result<_, String> {
            let total_start = browser_now_ms();
            self.source_lod_scratch.clear();
            self.source_lod_scratch.extend(
                source_requests
                    .iter()
                    .map(|request| request.unwrap_or(standby).edge_lods()),
            );
            let mesh_plan_start = browser_now_ms();
            let plan = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
                selected_patches,
                view_projection,
                viewport,
                min_px_per_segment: config.min_px_per_segment,
                max_px_per_segment: config.max_px_per_segment,
                policy: config.policy,
                max_atlas_lod,
                retain_culled_leaves: true,
                max_partition_leaves,
                max_total_leaves: config.max_total_leaves,
                source_requested_lods: &self.source_lod_scratch,
            })
            .map_err(|error| error.to_string())?;
            let mesh_plan_ms = browser_now_ms() - mesh_plan_start;
            let frontier_start = browser_now_ms();
            let frontier_matches = self
                .frontier
                .as_ref()
                .is_some_and(|frontier| frontier.leaves() == plan.leaves.as_slice());
            if frontier_matches {
                self.frontier_cache_hits = self.frontier_cache_hits.saturating_add(1);
            } else {
                let frontier = ScreenMeshLeafFrontier::build(&plan.leaves, topology)
                    .map_err(|error| error.to_string())?;
                self.frontier = Some(frontier);
                self.reconciliation_cache.invalidate();
                self.frontier_cache_misses = self.frontier_cache_misses.saturating_add(1);
            }
            let frontier = self
                .frontier
                .as_ref()
                .ok_or_else(|| "adaptive frontier cache is unavailable".to_string())?;
            let frontier_ms = browser_now_ms() - frontier_start;
            let reconcile_start = browser_now_ms();
            let reconciled = self.reconciliation_cache.resolve(
                frontier,
                &plan.requested_lods,
                max_face_edge_ratio,
                max_atlas_lod,
            )?;
            let reconcile_ms = browser_now_ms() - reconcile_start;
            let atlas_work_start = browser_now_ms();
            let mut triangles = 0u64;
            for (leaf_index, lods) in reconciled.resident.iter().copied().enumerate() {
                let key = canonical_form(lods).res;
                let triangle_count = atlas_triangle_counts.get(&key).ok_or_else(|| {
                    format!("adaptive leaf {leaf_index} needs missing atlas patch {key:?}")
                })?;
                triangles = triangles.saturating_add(*triangle_count);
            }
            if triangles > config.max_triangles {
                return Err(format!(
                    "adaptive plan needs {triangles} triangles; budget is {}",
                    config.max_triangles,
                ));
            }
            let atlas_work_ms = browser_now_ms() - atlas_work_start;
            let group_start = browser_now_ms();
            group_resident_screen_leaves_into(
                frontier,
                &reconciled.resident,
                face_materials,
                face_nodes,
                face_render_nodes,
                &mut self.lod_scratch,
                &mut self.candidate_groups,
            )
            .map_err(|error| error.to_string())?;
            let group_ms = browser_now_ms() - group_start;
            let timings = AdaptivePlanTimings {
                total_ms: browser_now_ms() - total_start,
                mesh_plan_ms,
                frontier_ms,
                reconcile_ms,
                atlas_work_ms,
                group_ms,
            };
            let reconciliation = (
                reconciled.shared_edge_promotions as u64,
                reconciled.grading_promotions as u64,
                reconciled.iterations as u64,
            );
            Ok((
                plan.diagnostic,
                plan.selected_faces,
                reconciliation,
                triangles,
                timings,
            ))
        })();

        match attempt {
            Ok((diagnostic, selected_faces, reconciliation, triangles, timings)) => {
                std::mem::swap(live_groups, &mut self.candidate_groups);
                self.pending_plan = Some(diagnostic);
                self.pending_selection = selection_diagnostic;
                self.pending_selected_faces = selected_faces;
                self.pending_triangles = triangles;
                self.pending_shared_edge_promotions = reconciliation.0;
                self.pending_grading_promotions = reconciliation.1;
                self.pending_reconciliation_iterations = reconciliation.2;
                self.pending_pose_stamp = current_pose_stamp;
                self.last_timings = timings;
                Ok(())
            }
            Err(error) => {
                self.fallbacks = self.fallbacks.saturating_add(1);
                self.clear_pending_publication();
                self.pending_fallback_error = Some(error.clone());
                Err(error)
            }
        }
    }

    pub(crate) fn snapshot(&self) -> AdaptivePickedSnapshot<'_> {
        let state = if self.config.is_none() {
            if self.pending_legacy && self.last_publication_error.is_some() {
                "rollback-disable"
            } else if self.pending_legacy {
                "staged-disable"
            } else {
                "disabled"
            }
        } else if self.pending_plan.is_some() || self.pending_fallback_error.is_some() {
            "staged"
        } else if self.last_publication_error.is_some() {
            if self.last_plan.is_some() {
                "rollback-active"
            } else {
                "rollback-fallback"
            }
        } else if self.last_error.is_some() {
            "fallback"
        } else if self.last_plan.is_some() {
            "active"
        } else {
            "configured"
        };
        let published_face = match self.last_published_faces.as_slice() {
            [face] => Some(*face),
            _ => None,
        };
        AdaptivePickedSnapshot {
            enabled: self.config.is_some(),
            state,
            mode: self.config.map(|config| config.selection.mode()),
            face: self
                .config
                .and_then(|config| config.selection.picked_face()),
            max_selected_faces: self
                .config
                .map(|config| config.selection.max_faces() as u64),
            min_px_per_segment: self.config.map(|config| config.min_px_per_segment),
            max_px_per_segment: self.config.map(|config| config.max_px_per_segment),
            max_depth: self.config.map(|config| config.policy.max_depth),
            max_selected_leaves: self.config.map(|config| config.policy.max_leaves as u64),
            max_partition_leaves: self.config.map(|config| {
                config
                    .selection
                    .max_partition_leaves(config.policy.max_leaves) as u64
            }),
            max_total_leaves: self.config.map(|config| config.max_total_leaves as u64),
            max_triangles: self.config.map(|config| config.max_triangles),
            pose_revision: self
                .pending_pose_stamp
                .or(self.last_pose_stamp)
                .map(|stamp| stamp.0),
            pose_continuity_epoch: self
                .pending_pose_stamp
                .or(self.last_pose_stamp)
                .map(|stamp| stamp.1),
            published_face,
            published_faces: &self.last_published_faces,
            attempts: self.attempts,
            installs: self.installs,
            fallbacks: self.fallbacks,
            last_error: self.last_error.as_deref(),
            last_publication_error: self.last_publication_error.as_deref(),
            last_plan: self.last_plan.map(Into::into),
            last_selection: self.last_selection.map(Into::into),
            last_triangles: self.last_triangles,
            last_shared_edge_promotions: self.last_shared_edge_promotions,
            last_grading_promotions: self.last_grading_promotions,
            last_reconciliation_iterations: self.last_reconciliation_iterations,
            frontier_cache_hits: self.frontier_cache_hits,
            frontier_cache_misses: self.frontier_cache_misses,
            reconciliation_cache_hits: self.reconciliation_cache.hits,
            reconciliation_cache_misses: self.reconciliation_cache.misses,
            last_timings: self.last_timings,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AdaptiveScreenRequest<'a> {
    pub face: u32,
    pub node: u32,
    pub patch: &'a QBTriPatch,
    pub view_projection: [f64; 16],
    pub viewport: [f64; 2],
    pub min_px_per_segment: f64,
    pub max_px_per_segment: f64,
    pub policy: ScreenPartitionPolicy,
    pub max_face_edge_ratio: u32,
    pub source_requested_lod: [u32; 3],
    pub current_resident_lod: Option<[u32; 3]>,
    pub atlas_triangle_counts: &'a BTreeMap<[u32; 3], u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScreenMetricDiagnosticSnapshot {
    edge_arc_px: [f64; 3],
    directional_extent_px: [f64; 3],
    max_edge_arc_px: f64,
    stretch_range: [f64; 2],
    stretch_ratio: f64,
    area_scale_range: [f64; 2],
    area_ratio: f64,
    min_denominator_norm_sq: f64,
    unprojectable: bool,
}

impl From<ScreenPatchDiagnostic> for ScreenMetricDiagnosticSnapshot {
    fn from(diagnostic: ScreenPatchDiagnostic) -> Self {
        Self {
            edge_arc_px: diagnostic.edge_arc_px,
            directional_extent_px: diagnostic.directional_extent_px,
            max_edge_arc_px: diagnostic.max_edge_arc_px,
            stretch_range: diagnostic.stretch_range,
            stretch_ratio: diagnostic.stretch_ratio,
            area_scale_range: diagnostic.area_scale_range,
            area_ratio: diagnostic.area_ratio,
            min_denominator_norm_sq: diagnostic.min_denominator_norm_sq,
            unprojectable: diagnostic.unprojectable,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AdaptiveScreenPatchSnapshot {
    pub ok: bool,
    pub quality_met: bool,
    pub face: u32,
    pub node: u32,
    pub viewport: [u32; 2],
    pub min_px_per_segment: f64,
    pub max_px_per_segment: f64,
    pub max_atlas_lod: u32,
    pub max_face_edge_ratio: u32,
    root_metric: ScreenMetricDiagnosticSnapshot,
    pub single_patch_requested_lod: [u32; 3],
    pub single_patch_resident_lod: [u32; 3],
    pub single_patch_triangles: Option<u64>,
    pub current_resident_lod: Option<[u32; 3]>,
    pub current_resident_triangles: Option<u64>,
    pub leaves: u32,
    pub split_nodes: u32,
    pub max_depth_reached: u8,
    pub unmet_leaves: u32,
    pub saturated_metric_leaves: u32,
    pub boundary_fallback_faces: u32,
    leaf_status_histogram: Vec<(&'static str, u32)>,
    leaf_depth_histogram: Vec<(u8, u32)>,
    requested_lod_histogram: Vec<(String, u32)>,
    resident_lod_histogram: Vec<(String, u32)>,
    pub reconciliation_iterations: u32,
    pub shared_edge_promotions: u32,
    pub grading_promotions: u32,
    pub max_absolute_exponent: u8,
    pub worst_leaf_stretch_ratio: f64,
    pub worst_leaf_area_ratio: f64,
    /// Conservative local derivative/edge-arc extent divided by resident LoD.
    /// This is the planning metric, not a claim about every final atlas edge.
    pub max_local_metric_segment_px: f64,
    pub min_local_metric_segment_px: f64,
    pub atlas_keys: u32,
    pub missing_atlas_keys: Vec<[u32; 3]>,
    pub adaptive_instances: u64,
    pub adaptive_triangles: u64,
}

fn lod_histogram(lods: &[[u32; 3]]) -> Vec<(String, u32)> {
    let mut histogram = BTreeMap::<String, u32>::new();
    for [a, b, c] in lods {
        *histogram.entry(format!("{a}/{b}/{c}")).or_default() += 1;
    }
    histogram.into_iter().collect()
}

pub(crate) fn measure_adaptive_screen_patch(
    request: AdaptiveScreenRequest<'_>,
) -> Result<AdaptiveScreenPatchSnapshot, String> {
    if !request.min_px_per_segment.is_finite()
        || request.min_px_per_segment <= 0.0
        || !request.max_px_per_segment.is_finite()
        || request.max_px_per_segment < request.min_px_per_segment
    {
        return Err("pixel band must be finite, positive, and ordered".into());
    }
    let max_atlas_lod = request
        .atlas_triangle_counts
        .keys()
        .flat_map(|key| key.iter().copied())
        .max()
        .ok_or_else(|| "tessellation atlas is unavailable".to_string())?;
    let root_metric = diagnose_screen_patch(
        request.patch,
        &request.view_projection,
        request.viewport,
        request.policy.arc_options,
    );
    let partition = partition_screen_patch(
        request.patch,
        &request.view_projection,
        request.viewport,
        request.policy,
    )
    .map_err(|error| error.to_string())?;
    let unresolved_boundary_leaves = partition
        .leaves
        .iter()
        .filter(|leaf| leaf.status.is_drawable() && leaf.metric_diagnostic.is_none())
        .count();
    let boundary_fallback_faces = u32::from(unresolved_boundary_leaves != 0);
    let single_patch_requested_lod = if boundary_fallback_faces != 0 {
        request.source_requested_lod
    } else {
        root_metric
            .resolve_subdivision_band(
                request.source_requested_lod,
                request.min_px_per_segment,
                request.max_px_per_segment,
                max_atlas_lod,
            )
            .ok_or_else(|| {
                "root screen metric cannot resolve the requested pixel band".to_string()
            })?
            .requested
    };
    let single_patch_lod = reconcile_screen_leaf_lods(
        &[ScreenLeafTopology {
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        }],
        &[single_patch_requested_lod],
        request.max_face_edge_ratio,
        max_atlas_lod,
    )
    .map_err(|error| error.to_string())?
    .resident[0];

    let drawable_leaves = partition
        .leaves
        .iter()
        .filter(|leaf| leaf.status.is_drawable())
        .collect::<Vec<_>>();
    let mut saturated_metric_leaves = 0u32;
    let (topology, requested) = if boundary_fallback_faces != 0 {
        (
            vec![ScreenLeafTopology {
                id: ScreenPatchLeafId::ROOT,
                domain: QBPatchDomain::FULL,
            }],
            vec![request.source_requested_lod],
        )
    } else {
        let requested = drawable_leaves
            .iter()
            .map(|leaf| {
                let diagnostic = leaf
                    .metric_diagnostic
                    .expect("finite drawable leaves were checked above");
                let inherited = inherited_source_edge_lods(
                    leaf.id,
                    leaf.restricted.domain,
                    request.source_requested_lod,
                )
                .ok();
                let band = inherited.and_then(|inherited| {
                    diagnostic.resolve_subdivision_band(
                        inherited,
                        request.min_px_per_segment,
                        request.max_px_per_segment,
                        max_atlas_lod,
                    )
                });
                match band {
                    Some(band) => {
                        saturated_metric_leaves = saturated_metric_leaves
                            .saturating_add(u32::from(band.saturated));
                        band.requested
                    }
                    None => {
                        saturated_metric_leaves = saturated_metric_leaves.saturating_add(1);
                        [max_atlas_lod; 3]
                    }
                }
            })
            .collect::<Vec<_>>();
        let topology = drawable_leaves
            .iter()
            .map(|leaf| ScreenLeafTopology::from(*leaf))
            .collect::<Vec<_>>();
        (topology, requested)
    };
    let reconciled = reconcile_screen_leaf_lods(
        &topology,
        &requested,
        request.max_face_edge_ratio,
        max_atlas_lod,
    )
    .map_err(|error| error.to_string())?;

    let mut status_histogram = BTreeMap::<&'static str, u32>::new();
    let mut depth_histogram = BTreeMap::<u8, u32>::new();
    let mut worst_leaf_stretch_ratio = 1.0_f64;
    let mut worst_leaf_area_ratio = 1.0_f64;
    let mut max_local_metric_segment_px = 0.0_f64;
    let mut min_local_metric_segment_px = f64::INFINITY;
    for leaf in &partition.leaves {
        let status = match leaf.status {
            ScreenPatchLeafStatus::Accepted => "accepted",
            ScreenPatchLeafStatus::BelowPixelExtent => "belowPixelExtent",
            ScreenPatchLeafStatus::FullyFaded => "fullyFaded",
            ScreenPatchLeafStatus::OutsideFrustum => "outsideFrustum",
            ScreenPatchLeafStatus::BoundaryDepthLimit => "boundaryDepthLimit",
            ScreenPatchLeafStatus::BoundaryLeafBudget => "boundaryLeafBudget",
            ScreenPatchLeafStatus::DepthLimit => "depthLimit",
            ScreenPatchLeafStatus::LeafBudget => "leafBudget",
        };
        *status_histogram.entry(status).or_default() += 1;
        *depth_histogram.entry(leaf.id.depth).or_default() += 1;
    }
    if boundary_fallback_faces == 0 {
        for (leaf, resident) in drawable_leaves.iter().zip(&reconciled.resident) {
            let Some(diagnostic) = leaf.metric_diagnostic else {
                continue;
            };
            worst_leaf_stretch_ratio = worst_leaf_stretch_ratio.max(diagnostic.stretch_ratio);
            worst_leaf_area_ratio = worst_leaf_area_ratio.max(diagnostic.area_ratio);
            for edge in 0..3 {
                let extent =
                    diagnostic.edge_arc_px[edge].max(diagnostic.directional_extent_px[edge]);
                let segment = extent / f64::from(resident[edge]);
                if segment.is_finite() {
                    max_local_metric_segment_px = max_local_metric_segment_px.max(segment);
                    if segment > 0.0 {
                        min_local_metric_segment_px = min_local_metric_segment_px.min(segment);
                    }
                }
            }
        }
    }
    if !min_local_metric_segment_px.is_finite() {
        min_local_metric_segment_px = 0.0;
    }

    let mut adaptive_triangles = 0u64;
    let mut missing_atlas = BTreeMap::<[u32; 3], ()>::new();
    let mut atlas_keys = BTreeMap::<[u32; 3], ()>::new();
    for resident in &reconciled.resident {
        let key = canonical_form(*resident).res;
        atlas_keys.insert(key, ());
        if let Some(triangles) = request.atlas_triangle_counts.get(&key) {
            adaptive_triangles = adaptive_triangles.saturating_add(*triangles);
        } else {
            missing_atlas.insert(key, ());
        }
    }
    let triangle_count = |lod: [u32; 3]| {
        request
            .atlas_triangle_counts
            .get(&canonical_form(lod).res)
            .copied()
    };
    let missing_atlas_keys = missing_atlas.into_keys().collect::<Vec<_>>();
    let quality_met = boundary_fallback_faces == 0
        && partition.unmet_leaves == 0
        && saturated_metric_leaves == 0
        && missing_atlas_keys.is_empty()
        && max_local_metric_segment_px <= request.max_px_per_segment * (1.0 + 1.0e-9)
        && (min_local_metric_segment_px == 0.0
            || min_local_metric_segment_px >= request.min_px_per_segment * (1.0 - 1.0e-9));

    Ok(AdaptiveScreenPatchSnapshot {
        ok: true,
        quality_met,
        face: request.face,
        node: request.node,
        viewport: [request.viewport[0] as u32, request.viewport[1] as u32],
        min_px_per_segment: request.min_px_per_segment,
        max_px_per_segment: request.max_px_per_segment,
        max_atlas_lod,
        max_face_edge_ratio: request.max_face_edge_ratio,
        root_metric: root_metric.into(),
        single_patch_requested_lod,
        single_patch_resident_lod: single_patch_lod,
        single_patch_triangles: triangle_count(single_patch_lod),
        current_resident_lod: request.current_resident_lod,
        current_resident_triangles: request.current_resident_lod.and_then(triangle_count),
        leaves: partition.leaves.len() as u32,
        split_nodes: partition.split_nodes as u32,
        max_depth_reached: partition.max_depth_reached,
        unmet_leaves: partition.unmet_leaves as u32,
        saturated_metric_leaves,
        boundary_fallback_faces,
        leaf_status_histogram: status_histogram.into_iter().collect(),
        leaf_depth_histogram: depth_histogram.into_iter().collect(),
        requested_lod_histogram: lod_histogram(&requested),
        resident_lod_histogram: lod_histogram(&reconciled.resident),
        reconciliation_iterations: reconciled.iterations as u32,
        shared_edge_promotions: reconciled.shared_edge_promotions as u32,
        grading_promotions: reconciled.grading_promotions as u32,
        max_absolute_exponent: reconciled.max_absolute_exponent,
        worst_leaf_stretch_ratio,
        worst_leaf_area_ratio,
        max_local_metric_segment_px,
        min_local_metric_segment_px,
        atlas_keys: atlas_keys.len() as u32,
        missing_atlas_keys,
        adaptive_instances: reconciled.resident.len() as u64,
        adaptive_triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    const PERSPECTIVE: [f64; 16] = [
        0.974_278_579,
        0.0,
        0.0,
        0.0,
        0.0,
        1.732_050_808,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.000_200_02,
        -1.0,
        0.0,
        0.0,
        -0.020_002_000_2,
        0.0,
    ];

    #[wasm_bindgen_test]
    fn measurement_oracle_uses_the_live_floor_capacity_resolver() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let atlas_triangle_counts = BTreeMap::from([
            ([1, 1, 2], 2),
            // Establish the same maximum atlas level as the live renderer.
            ([16, 32, 64], 1),
        ]);
        let snapshot = measure_adaptive_screen_patch(AdaptiveScreenRequest {
            face: 0,
            node: 0,
            patch: &patch,
            view_projection: IDENTITY,
            viewport: [200.0, 200.0],
            min_px_per_segment: 60.0,
            max_px_per_segment: 90.0,
            policy: ScreenPartitionPolicy::default(),
            max_face_edge_ratio: 4,
            source_requested_lod: [64; 3],
            current_resident_lod: Some([64; 3]),
            atlas_triangle_counts: &atlas_triangle_counts,
        })
        .unwrap();

        assert_eq!(snapshot.single_patch_requested_lod, [2, 1, 1]);
        assert_eq!(snapshot.saturated_metric_leaves, 1);
        assert!(!snapshot.quality_met);
        assert!(snapshot.min_local_metric_segment_px >= snapshot.min_px_per_segment);
        assert!(snapshot.max_local_metric_segment_px > snapshot.max_px_per_segment);
    }

    #[wasm_bindgen_test]
    fn measurement_oracle_reports_the_live_source_root_boundary_fallback() {
        let patch = QBTriPatch::flat(
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [0.0, 0.4, 1.0],
        );
        let atlas_triangle_counts = BTreeMap::from([([16, 32, 64], 1)]);
        let snapshot = measure_adaptive_screen_patch(AdaptiveScreenRequest {
            face: 0,
            node: 0,
            patch: &patch,
            view_projection: PERSPECTIVE,
            viewport: [1600.0, 900.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 32.0,
            policy: ScreenPartitionPolicy {
                max_depth: 2,
                max_leaves: 16,
                ignore_below_px: 0.0,
                ..ScreenPartitionPolicy::default()
            },
            max_face_edge_ratio: 4,
            source_requested_lod: [64; 3],
            current_resident_lod: Some([64; 3]),
            atlas_triangle_counts: &atlas_triangle_counts,
        })
        .unwrap();
        assert_eq!(snapshot.boundary_fallback_faces, 1);
        assert_eq!(snapshot.single_patch_requested_lod, [64; 3]);
        assert_eq!(snapshot.single_patch_resident_lod, [64; 3]);
        assert_eq!(
            snapshot.requested_lod_histogram,
            vec![("64/64/64".into(), 1)]
        );
        assert_eq!(
            snapshot.resident_lod_histogram,
            vec![("64/64/64".into(), 1)]
        );
        assert_eq!(snapshot.adaptive_instances, 1);
        assert!(!snapshot.quality_met);
    }

    #[wasm_bindgen_test]
    fn picked_runtime_replans_and_publishes_new_pose_revisions() {
        let positions = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let source =
            quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(&positions, &[[0, 1, 2]]);
        let topology = ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let patch = QBTriPatch::flat(positions[0], positions[1], positions[2]);
        let resident = ResidentLod {
            canonical: [1; 3],
            perm_index: 0,
            parity_bucket: 0,
        };
        let mut atlas_triangles = BTreeMap::new();
        atlas_triangles.insert([1; 3], 1);
        let mut groups = BTreeMap::new();
        let mut runtime = AdaptivePickedRuntime::default();
        runtime.configure(AdaptivePickedConfig {
            selection: AdaptiveScreenSelection::Picked { face: 0 },
            min_px_per_segment: 1.0,
            max_px_per_segment: 10_000.0,
            policy: ScreenPartitionPolicy {
                max_depth: 0,
                max_leaves: 1,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: 1,
            max_triangles: 1,
        });

        let mut plan = |runtime: &mut AdaptivePickedRuntime, stamp, grading_ratio| {
            runtime
                .plan_and_group(
                    &patch,
                    &IDENTITY,
                    [640.0, 480.0],
                    Some(stamp),
                    &[Some(resident)],
                    resident,
                    &topology,
                    1,
                    grading_ratio,
                    &atlas_triangles,
                    &[0],
                    &[0],
                    &[0],
                    &mut groups,
                )
                .unwrap();
        };

        plan(&mut runtime, (7, 2), 4);
        assert_eq!(runtime.snapshot().state, "staged");
        assert_eq!(runtime.snapshot().frontier_cache_hits, 0);
        assert_eq!(runtime.snapshot().frontier_cache_misses, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_hits, 0);
        assert_eq!(runtime.snapshot().reconciliation_cache_misses, 1);
        runtime.commit_publication();
        assert_eq!(runtime.snapshot().pose_revision, Some(7));
        assert_eq!(runtime.snapshot().published_faces, [0]);
        assert_eq!(runtime.snapshot().last_plan.unwrap().selected_faces, 1);

        plan(&mut runtime, (8, 2), 4);
        assert_eq!(runtime.snapshot().frontier_cache_hits, 1);
        assert_eq!(runtime.snapshot().frontier_cache_misses, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_hits, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_misses, 1);
        runtime.record_publication_failure("synthetic upload rollback");
        let rolled_back = runtime.snapshot();
        assert_eq!(rolled_back.state, "rollback-active");
        assert_eq!(rolled_back.pose_revision, Some(7));
        assert_eq!(rolled_back.installs, 1);
        assert_eq!(
            rolled_back.last_publication_error,
            Some("synthetic upload rollback"),
        );

        plan(&mut runtime, (8, 2), 4);
        assert_eq!(runtime.snapshot().frontier_cache_hits, 2);
        assert_eq!(runtime.snapshot().frontier_cache_misses, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_hits, 2);
        assert_eq!(runtime.snapshot().reconciliation_cache_misses, 1);
        runtime.commit_publication();
        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.state, "active");
        assert_eq!(snapshot.pose_revision, Some(8));
        assert_eq!(snapshot.pose_continuity_epoch, Some(2));
        assert_eq!(runtime.snapshot().installs, 2);

        // A grading-policy change is a real reconciliation input even when
        // this one-leaf fixture happens to produce the same resident LoDs.
        plan(&mut runtime, (9, 2), 2);
        assert_eq!(runtime.snapshot().frontier_cache_hits, 3);
        assert_eq!(runtime.snapshot().reconciliation_cache_hits, 2);
        assert_eq!(runtime.snapshot().reconciliation_cache_misses, 2);
        runtime.commit_publication();
        assert_eq!(runtime.snapshot().pose_revision, Some(9));
        assert_eq!(runtime.snapshot().installs, 3);

        runtime.stage_clear();
        let staged_disable = runtime.snapshot();
        assert_eq!(staged_disable.state, "staged-disable");
        assert_eq!(staged_disable.published_face, Some(0));
        assert_eq!(staged_disable.published_faces, [0]);
        assert_eq!(staged_disable.pose_revision, Some(9));
        runtime.record_refresh_failure("synthetic disable refresh rejection");
        let rolled_back_disable = runtime.snapshot();
        assert_eq!(rolled_back_disable.state, "rollback-disable");
        assert_eq!(rolled_back_disable.published_face, Some(0));
        assert_eq!(
            rolled_back_disable.last_publication_error,
            Some("synthetic disable refresh rejection"),
        );
        assert!(runtime.has_pending_publication());
        runtime.commit_publication();
        let disabled = runtime.snapshot();
        assert_eq!(disabled.state, "disabled");
        assert_eq!(disabled.published_face, None);
        assert!(disabled.published_faces.is_empty());
        assert_eq!(disabled.pose_revision, None);
    }

    #[wasm_bindgen_test]
    fn runtime_groups_one_transactional_multi_face_frontier() {
        let positions = [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [2, 1, 3]],
        );
        let topology = ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let patches = [
            QBTriPatch::flat(positions[0], positions[1], positions[2]),
            QBTriPatch::flat(positions[2], positions[1], positions[3]),
        ];
        let selected = [
            SelectedScreenPatch {
                source_face: 1,
                transformed_patch: &patches[1],
            },
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &patches[0],
            },
        ];
        let resident = ResidentLod {
            canonical: [1; 3],
            perm_index: 0,
            parity_bucket: 0,
        };
        let atlas_triangles = BTreeMap::from([([1; 3], 1)]);
        let mut groups = BTreeMap::new();
        let mut runtime = AdaptivePickedRuntime::default();
        runtime.configure(AdaptivePickedConfig {
            selection: AdaptiveScreenSelection::Picked { face: 0 },
            min_px_per_segment: 1.0,
            max_px_per_segment: 10_000.0,
            policy: ScreenPartitionPolicy {
                max_depth: 0,
                max_leaves: 1,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: 2,
            max_triangles: 2,
        });

        runtime
            .plan_selected_and_group(
                &selected,
                Some(AdaptiveScreenFaceSelectionDiagnostic {
                    examined_faces: 2,
                    visible_faces: 2,
                    selected_faces: 2,
                    partition_face_capacity: 2,
                    omitted_by_capacity: 0,
                    selected_root_triangles: 2,
                }),
                2,
                &IDENTITY,
                [640.0, 480.0],
                Some((3, 1)),
                &[Some(resident); 2],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0; 2],
                &[0; 2],
                &[0; 2],
                &mut groups,
            )
            .unwrap();
        assert_eq!(groups.values().map(Vec::len).sum::<usize>(), 2);
        assert_eq!(runtime.snapshot().state, "staged");

        let staged_groups = groups.clone();
        let second_plan = runtime.plan_selected_and_group(
            &selected,
            None,
            2,
            &IDENTITY,
            [640.0, 480.0],
            Some((4, 1)),
            &[Some(resident); 2],
            resident,
            &topology,
            1,
            4,
            &atlas_triangles,
            &[0; 2],
            &[0; 2],
            &[0; 2],
            &mut groups,
        );
        assert_eq!(
            second_plan.unwrap_err(),
            "adaptive plan is already awaiting publication",
        );
        assert_eq!(groups, staged_groups);
        assert_eq!(runtime.snapshot().state, "staged");
        assert_eq!(runtime.snapshot().pose_revision, Some(3));
        assert_eq!(runtime.snapshot().attempts, 1);

        runtime.commit_publication();
        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.state, "active");
        assert_eq!(snapshot.published_face, None);
        assert_eq!(snapshot.published_faces, [0, 1]);
        assert_eq!(snapshot.last_plan.unwrap().selected_faces, 2);
        assert_eq!(snapshot.last_selection.unwrap().selected_faces, 2);
        assert_eq!(snapshot.last_triangles, 2);
    }

    #[wasm_bindgen_test]
    fn runtime_keeps_other_faces_adaptive_when_one_boundary_falls_back() {
        let positions = [
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [0.0, 0.4, -2.0],
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [-0.3, 0.3, -2.0],
        ];
        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [3, 4, 5]],
        );
        let topology = ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let boundary = QBTriPatch::flat(
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [0.0, 0.4, 1.0],
        );
        let finite = QBTriPatch::flat(positions[3], positions[4], positions[5]);
        let selected = [
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &boundary,
            },
            SelectedScreenPatch {
                source_face: 1,
                transformed_patch: &finite,
            },
        ];
        let resident = ResidentLod::uniform(1);
        let atlas_triangles = BTreeMap::from([([1; 3], 1)]);
        let mut groups = BTreeMap::new();
        let mut runtime = AdaptivePickedRuntime::default();
        runtime.configure(AdaptivePickedConfig {
            selection: AdaptiveScreenSelection::CurrentView {
                max_faces: 2,
                max_partition_leaves: 2,
            },
            min_px_per_segment: 16.0,
            max_px_per_segment: 32.0,
            policy: ScreenPartitionPolicy {
                max_depth: 0,
                max_leaves: 1,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: 2,
            max_triangles: 2,
        });

        runtime
            .plan_selected_and_group(
                &selected,
                Some(AdaptiveScreenFaceSelectionDiagnostic {
                    examined_faces: 2,
                    visible_faces: 2,
                    selected_faces: 2,
                    partition_face_capacity: 2,
                    omitted_by_capacity: 0,
                    selected_root_triangles: 2,
                }),
                2,
                &PERSPECTIVE,
                [1600.0, 900.0],
                Some((1, 0)),
                &[Some(resident); 2],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0, 0],
                &[0, 1],
                &[0, 1],
                &mut groups,
            )
            .unwrap();
        runtime.commit_publication();

        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.published_faces, [1]);
        let plan = snapshot.last_plan.unwrap();
        assert_eq!(plan.selected_faces, 1);
        assert_eq!(plan.boundary_fallback_faces, 1);
        let members = groups.values().flatten().collect::<Vec<_>>();
        assert_eq!(members.len(), 2);
        assert_eq!(
            members
                .iter()
                .map(|member| member.face_index)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([0, 1]),
        );
    }
}
