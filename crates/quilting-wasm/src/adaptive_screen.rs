//! Read-only renderer oracle for metric-adaptive QB patch work.
//!
//! This module owns no browser or GPU state. The WebGL adapter supplies one
//! current output-chart patch, camera, active atlas counts, and policy. Keeping
//! the measurement pure prevents experimental refinement policy from leaking
//! into the live frame renderer before its workload and failure modes are
//! understood.

use std::collections::{BTreeMap, BTreeSet};

use quilting_core::batch::{
    group_resident_faces_into, group_resident_screen_leaves_into,
    group_resident_screen_component_overlay_into, group_resident_screen_overlay_into,
    measure_adaptive_render_work, AdaptiveRenderOverlay, RenderBatchKey, RenderBatchMember,
    ResidentLod,
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
    inherited_source_edge_lods, plan_adaptive_screen_components_into,
    plan_adaptive_screen_mesh_into, AdaptiveScreenComponentPlan,
    AdaptiveScreenFaceSelectionDiagnostic, AdaptiveScreenMeshPlan,
    AdaptiveScreenMeshPlanDiagnostic, AdaptiveScreenMeshPlanRequest, SelectedScreenPatch,
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
    plan_scratch: AdaptiveScreenMeshPlan,
    frontier_cache_hits: u64,
    frontier_cache_misses: u64,
    reconciliation_cache: AdaptiveReconciliationCache,
    published_group_signature: Option<AdaptiveGroupSignature>,
    pending_group_signature: Option<AdaptiveGroupSignature>,
    group_cache_hits: u64,
    group_cache_misses: u64,
    last_timings: AdaptivePlanTimings,
    component_shadow: AdaptiveComponentShadow,
    component_publication_enabled: bool,
    pending_component_publication: bool,
    pending_component_authority: bool,
    published_component_publication: bool,
    component_publication_installs: u64,
    component_authority_changed_installs: u64,
    component_publication_fallbacks: u64,
    component_publication_last_error: Option<String>,
    lod_scratch: ScreenMeshLeafLodScratch,
    retained_shadow_enabled: bool,
    pending_overlay: AdaptiveRenderOverlay,
    pending_overlay_signature: Option<AdaptiveGroupSignature>,
    pending_overlay_reuses_published: bool,
    published_overlay: AdaptiveRenderOverlay,
    published_overlay_signature: Option<AdaptiveGroupSignature>,
    retained_shadow_cache_hits: u64,
    retained_shadow_cache_misses: u64,
    last_retained_shadow_stage_ms: f64,
    overlay_shadow: AdaptiveRenderOverlay,
    overlay_baseline_groups: BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    overlay_member_scratch: Vec<RenderBatchMember>,
    overlay_complete_scratch: Vec<RenderBatchMember>,
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
    group_cache_hits: u64,
    group_cache_misses: u64,
    component_shadow_enabled: bool,
    component_shadow_state: &'static str,
    component_shadow_attempts: u64,
    component_shadow_matches: u64,
    component_shadow_mismatches: u64,
    component_shadow_frontier_cache_hits: u64,
    component_shadow_frontier_cache_misses: u64,
    component_shadow_reconciliation_cache_hits: u64,
    component_shadow_reconciliation_cache_misses: u64,
    component_certified_reuses: u64,
    component_certified_misses: u64,
    component_authority_oracle_interval: u64,
    component_authority_oracle_samples: u64,
    component_authority_oracle_skips: u64,
    component_authority_sample_age: u64,
    component_authority_revoked: bool,
    component_authority_revocations: u64,
    component_authority_changed_installs: u64,
    component_shadow_last_error: Option<&'a str>,
    component_shadow_last: Option<AdaptiveComponentShadowComparison>,
    component_publication_enabled: bool,
    component_publication_state: &'static str,
    component_publication_installs: u64,
    component_publication_fallbacks: u64,
    component_publication_last_error: Option<&'a str>,
    retained_shadow_enabled: bool,
    retained_shadow_state: &'static str,
    retained_shadow_suppressed_faces: u64,
    retained_shadow_groups: u64,
    retained_shadow_members: u64,
    retained_shadow_cache_hits: u64,
    retained_shadow_cache_misses: u64,
    last_retained_shadow_stage_ms: f64,
    last_timings: AdaptivePlanTimings,
}

const COMPONENT_AUTHORITY_ORACLE_INTERVAL: u64 = 16;
const COMPONENT_AUTHORITY_MAX_FACES: usize = 4_096;
const COMPONENT_AUTHORITY_LARGE_SCENE_FRACTION: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveGroupAuthority {
    Complete,
    Component,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdaptiveGroupSignature {
    authority: AdaptiveGroupAuthority,
    generation: u64,
    batch_layout_revision: u64,
}

#[derive(Default)]
struct AdaptiveReconciliationCache {
    requested_lods: Vec<[u32; 3]>,
    max_face_edge_ratio: u32,
    max_atlas_lod: u32,
    result: Option<ScreenLeafLodResult>,
    generation: u64,
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
    ) -> Result<(&'a ScreenLeafLodResult, u64, bool), String> {
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
            self.generation = self
                .generation
                .checked_add(1)
                .ok_or_else(|| "adaptive reconciliation generation overflow".to_string())?;
            self.misses = self.misses.saturating_add(1);
        }
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| "adaptive reconciliation cache is unavailable".to_string())?;
        Ok((result, self.generation, matches))
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

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct AdaptiveComponentShadowComparison {
    source_faces: u64,
    component_faces: u64,
    unaffected_faces: u64,
    complete_leaves: u64,
    component_leaves: u64,
    frontier_reused: bool,
    reconciliation_reused: bool,
    diagnostic_match: bool,
    selected_faces_match: bool,
    leaf_mismatches: u64,
    request_mismatches: u64,
    resident_mismatches: u64,
    vertex_mismatches: u64,
    plan_exact: bool,
    overlay_compared: bool,
    suppressed_faces_match: bool,
    overlay_groups_match: bool,
    atlas_residency_valid: bool,
    baseline_triangles: u64,
    suppressed_root_triangles: u64,
    overlay_triangles: u64,
    composed_triangles: u64,
    complete_triangles: u64,
    triangle_budget_match: bool,
    oracle_sampled: bool,
    authoritative: bool,
    exact: bool,
    plan_ms: f64,
    frontier_ms: f64,
    reconcile_ms: f64,
    compare_ms: f64,
    overlay_ms: f64,
    total_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdaptiveComponentStageSignature {
    reconciliation_generation: u64,
    root_topology_revision: u64,
    batch_layout_revision: u64,
}

type AdaptiveReconciliationDiagnostic = (u64, u64, u64);

#[derive(Clone, Copy, Debug)]
struct AdaptiveCertifiedComponent {
    group_signature: AdaptiveGroupSignature,
    triangles: u64,
    reconciliation: AdaptiveReconciliationDiagnostic,
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveChangedComponentAuthority {
    group_signature: AdaptiveGroupSignature,
    reconciliation: AdaptiveReconciliationDiagnostic,
}

#[derive(Clone, Debug)]
struct AdaptiveComponentCertificate {
    stage_signature: AdaptiveComponentStageSignature,
    diagnostic: AdaptiveScreenMeshPlanDiagnostic,
    selected_faces: Vec<u32>,
    full_group_signature: AdaptiveGroupSignature,
    complete_triangles: u64,
    complete_reconciliation: AdaptiveReconciliationDiagnostic,
}

#[derive(Default)]
struct AdaptiveComponentShadow {
    enabled: bool,
    attempts: u64,
    matches: u64,
    mismatches: u64,
    last_error: Option<String>,
    last: Option<AdaptiveComponentShadowComparison>,
    plan: AdaptiveScreenComponentPlan,
    frontier: Option<ScreenMeshLeafFrontier>,
    frontier_cache_hits: u64,
    frontier_cache_misses: u64,
    reconciliation_cache: AdaptiveReconciliationCache,
    complete_lod_scratch: ScreenMeshLeafLodScratch,
    component_lod_scratch: ScreenMeshLeafLodScratch,
    overlay: AdaptiveRenderOverlay,
    staged_signature: Option<AdaptiveComponentStageSignature>,
    certificate: Option<AdaptiveComponentCertificate>,
    certified_reuses: u64,
    certified_misses: u64,
    authority_basis: Option<(u64, u64)>,
    authority_revoked: Option<(u64, u64)>,
    authority_generation: u64,
    authority_changed_since_oracle: u64,
    authority_oracle_samples: u64,
    authority_oracle_skips: u64,
    authority_revocations: u64,
}

impl AdaptiveComponentShadow {
    fn set_enabled(&mut self, enabled: bool) -> bool {
        let changed = self.enabled != enabled;
        self.enabled = enabled;
        if changed {
            self.last_error = None;
            self.last = None;
            self.frontier = None;
            self.frontier_cache_hits = 0;
            self.frontier_cache_misses = 0;
            self.reconciliation_cache = AdaptiveReconciliationCache::default();
            self.staged_signature = None;
            self.certificate = None;
            self.certified_reuses = 0;
            self.certified_misses = 0;
            self.authority_basis = None;
            self.authority_revoked = None;
            self.authority_generation = 0;
            self.authority_changed_since_oracle = 0;
            self.authority_oracle_samples = 0;
            self.authority_oracle_skips = 0;
            self.authority_revocations = 0;
        }
        changed
    }

    fn state(&self) -> &'static str {
        if !self.enabled {
            "disabled"
        } else if self.last_error.is_some() {
            "error"
        } else if self
            .last
            .is_some_and(|comparison| {
                comparison.authoritative && !comparison.oracle_sampled && !comparison.exact
            })
        {
            "authoritative-unsampled"
        } else if self.last.is_some_and(|comparison| comparison.exact) {
            "match"
        } else if self
            .last
            .is_some_and(|comparison| !comparison.plan_exact || comparison.overlay_compared)
        {
            "mismatch"
        } else if self.last.is_some() {
            "awaiting-overlay"
        } else {
            "awaiting-plan"
        }
    }

    fn revoke_staged_authority(&mut self) {
        self.certificate = None;
        let Some(signature) = self.staged_signature else {
            return;
        };
        let epoch = (
            signature.root_topology_revision,
            signature.batch_layout_revision,
        );
        if self.authority_revoked != Some(epoch) {
            self.authority_revocations = self.authority_revocations.saturating_add(1);
        }
        self.authority_revoked = Some(epoch);
    }

    fn record_exact_oracle_match(&mut self) {
        let Some(signature) = self.staged_signature else {
            return;
        };
        self.authority_basis = Some((
            signature.root_topology_revision,
            signature.batch_layout_revision,
        ));
        self.authority_changed_since_oracle = 0;
        self.authority_oracle_samples = self.authority_oracle_samples.saturating_add(1);
    }

    fn reset_authority_basis(&mut self) {
        self.certificate = None;
        self.authority_basis = None;
        self.authority_revoked = None;
        self.authority_changed_since_oracle = 0;
    }

    fn current_authority_revoked(&self) -> bool {
        let current = self
            .staged_signature
            .map(|signature| {
                (
                    signature.root_topology_revision,
                    signature.batch_layout_revision,
                )
            })
            .or(self.authority_basis);
        current.is_some() && current == self.authority_revoked
    }

    fn changed_authority_signature(
        &mut self,
    ) -> Result<Option<AdaptiveChangedComponentAuthority>, String> {
        let Some(stage_signature) = self.staged_signature else {
            return Ok(None);
        };
        let epoch = (
            stage_signature.root_topology_revision,
            stage_signature.batch_layout_revision,
        );
        if self.authority_basis != Some(epoch) || self.authority_revoked == Some(epoch) {
            return Ok(None);
        }
        let source_faces = self.plan.mesh.diagnostic.source_faces as usize;
        let component_faces = self.plan.component_faces.len();
        let bounded = component_faces != 0
            && component_faces < source_faces
            && component_faces <= COMPONENT_AUTHORITY_MAX_FACES
            && (source_faces <= COMPONENT_AUTHORITY_MAX_FACES
                || component_faces
                    .checked_mul(COMPONENT_AUTHORITY_LARGE_SCENE_FRACTION)
                    .is_some_and(|weighted| weighted <= source_faces));
        if !bounded {
            return Ok(None);
        }
        let next_since_oracle = self.authority_changed_since_oracle.saturating_add(1);
        if next_since_oracle >= COMPONENT_AUTHORITY_ORACLE_INTERVAL {
            return Ok(None);
        }
        let resident = self
            .reconciliation_cache
            .result
            .as_ref()
            .ok_or_else(|| "adaptive component reconciliation disappeared".to_string())?;
        self.authority_generation = self
            .authority_generation
            .checked_add(1)
            .ok_or_else(|| "adaptive component authority generation overflow".to_string())?;
        self.authority_changed_since_oracle = next_since_oracle;
        self.authority_oracle_skips = self.authority_oracle_skips.saturating_add(1);
        Ok(Some(AdaptiveChangedComponentAuthority {
            group_signature: AdaptiveGroupSignature {
                authority: AdaptiveGroupAuthority::Component,
                generation: self.authority_generation,
                batch_layout_revision: stage_signature.batch_layout_revision,
            },
            reconciliation: (
                resident.shared_edge_promotions as u64,
                resident.grading_promotions as u64,
                resident.iterations as u64,
            ),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_plan(
        &mut self,
        request: AdaptiveScreenMeshPlanRequest<'_>,
        topology: &ScreenMeshTopologyCache,
        max_face_edge_ratio: u32,
        max_atlas_lod: u32,
        root_topology_revision: u64,
        batch_layout_revision: u64,
    ) -> Result<Option<AdaptiveCertifiedComponent>, String> {
        if !self.enabled {
            return Ok(None);
        }
        self.attempts = self.attempts.saturating_add(1);
        self.last_error = None;
        self.last = None;
        self.staged_signature = None;
        let total_start = browser_now_ms();
        let plan_start = browser_now_ms();
        if let Err(error) = plan_adaptive_screen_components_into(
            request,
            topology,
            request.source_requested_lods.len(),
            &mut self.plan,
        ) {
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some(error.to_string());
            return Err(error.to_string());
        }
        let plan_ms = browser_now_ms() - plan_start;
        let frontier_start = browser_now_ms();
        let frontier_reused = self
            .frontier
            .as_ref()
            .is_some_and(|frontier| frontier.leaves() == self.plan.mesh.leaves.as_slice());
        if frontier_reused {
            self.frontier_cache_hits = self.frontier_cache_hits.saturating_add(1);
        } else {
            let component_frontier = match ScreenMeshLeafFrontier::build(
                &self.plan.mesh.leaves,
                topology,
            ) {
                Ok(frontier) => frontier,
                Err(error) => {
                    self.mismatches = self.mismatches.saturating_add(1);
                    self.last_error = Some(error.to_string());
                    return Err(error.to_string());
                }
            };
            self.frontier = Some(component_frontier);
            self.reconciliation_cache.invalidate();
            self.frontier_cache_misses = self.frontier_cache_misses.saturating_add(1);
        }
        let Some(component_frontier) = self.frontier.as_ref() else {
            self.revoke_staged_authority();
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some("adaptive component shadow frontier disappeared".into());
            return Err("adaptive component shadow frontier disappeared".into());
        };
        let frontier_ms = browser_now_ms() - frontier_start;
        let reconcile_start = browser_now_ms();
        let (_, reconciliation_generation, reconciliation_reused) = match self
            .reconciliation_cache
            .resolve(
                component_frontier,
                &self.plan.mesh.requested_lods,
                max_face_edge_ratio,
                max_atlas_lod,
            ) {
            Ok(resident) => resident,
            Err(error) => {
                self.mismatches = self.mismatches.saturating_add(1);
                self.last_error = Some(error.clone());
                return Err(error);
            }
        };
        let reconcile_ms = browser_now_ms() - reconcile_start;
        let stage_signature = AdaptiveComponentStageSignature {
            reconciliation_generation,
            root_topology_revision,
            batch_layout_revision,
        };
        self.staged_signature = Some(stage_signature);
        let certified = self.certificate.as_ref().filter(|certificate| {
            self.authority_revoked
                != Some((
                    stage_signature.root_topology_revision,
                    stage_signature.batch_layout_revision,
                ))
                && certificate.stage_signature == stage_signature
                && certificate.diagnostic == self.plan.mesh.diagnostic
                && certificate.selected_faces == self.plan.mesh.selected_faces
        });
        let certified_values = certified.map(|certificate| AdaptiveCertifiedComponent {
            group_signature: certificate.full_group_signature,
            triangles: certificate.complete_triangles,
            reconciliation: certificate.complete_reconciliation,
        });
        let source_faces = request.source_requested_lods.len();
        let component_faces = self.plan.component_faces.len();
        self.last = Some(AdaptiveComponentShadowComparison {
            source_faces: source_faces as u64,
            component_faces: component_faces as u64,
            unaffected_faces: source_faces.saturating_sub(component_faces) as u64,
            complete_leaves: certified
                .map(|certificate| certificate.diagnostic.total_leaves as u64)
                .unwrap_or(0),
            component_leaves: self.plan.mesh.leaves.len() as u64,
            frontier_reused,
            reconciliation_reused,
            diagnostic_match: certified.is_some(),
            selected_faces_match: certified.is_some(),
            leaf_mismatches: 0,
            request_mismatches: 0,
            resident_mismatches: 0,
            vertex_mismatches: 0,
            plan_exact: certified.is_some(),
            overlay_compared: false,
            suppressed_faces_match: false,
            overlay_groups_match: false,
            atlas_residency_valid: false,
            baseline_triangles: 0,
            suppressed_root_triangles: 0,
            overlay_triangles: 0,
            composed_triangles: 0,
            complete_triangles: certified
                .map(|certificate| certificate.complete_triangles)
                .unwrap_or(0),
            triangle_budget_match: false,
            oracle_sampled: false,
            authoritative: certified.is_some(),
            exact: false,
            plan_ms,
            frontier_ms,
            reconcile_ms,
            compare_ms: 0.0,
            overlay_ms: 0.0,
            total_ms: browser_now_ms() - total_start,
        });
        Ok(certified_values)
    }

    fn compare_staged_plan(
        &mut self,
        complete_plan: &AdaptiveScreenMeshPlan,
        complete_frontier: &ScreenMeshLeafFrontier,
        complete_resident: &ScreenLeafLodResult,
    ) {
        let Some(comparison) = self.last.as_mut() else {
            return;
        };
        let Some(component_frontier) = self.frontier.as_ref() else {
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some("adaptive component shadow frontier disappeared".into());
            return;
        };
        let Some(component_resident) = self.reconciliation_cache.result.as_ref() else {
            self.revoke_staged_authority();
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some("adaptive component shadow reconciliation disappeared".into());
            return;
        };
        let compare_start = browser_now_ms();
        let complete_vertices = match complete_frontier.rebuild_vertex_lods_into(
            &complete_resident.resident,
            &mut self.complete_lod_scratch,
        ) {
            Ok(vertices) => vertices,
            Err(error) => {
                self.revoke_staged_authority();
                self.mismatches = self.mismatches.saturating_add(1);
                self.last_error = Some(error.to_string());
                return;
            }
        };
        let component_vertices = match component_frontier.rebuild_vertex_lods_into(
            &component_resident.resident,
            &mut self.component_lod_scratch,
        ) {
            Ok(vertices) => vertices,
            Err(error) => {
                self.revoke_staged_authority();
                self.mismatches = self.mismatches.saturating_add(1);
                self.last_error = Some(error.to_string());
                return;
            }
        };
        let diagnostic_match = self.plan.mesh.diagnostic == complete_plan.diagnostic;
        let selected_faces_match = self.plan.mesh.selected_faces == complete_plan.selected_faces;
        let mut leaf_mismatches = 0usize;
        let mut request_mismatches = 0usize;
        let mut resident_mismatches = 0usize;
        let mut vertex_mismatches = 0usize;
        let mut component_index = 0usize;
        for (complete_index, complete_leaf) in complete_plan.leaves.iter().copied().enumerate() {
            if self
                .plan
                .component_faces
                .binary_search(&complete_leaf.source_face)
                .is_err()
            {
                continue;
            }
            let component_leaf = self.plan.mesh.leaves.get(component_index).copied();
            leaf_mismatches += usize::from(component_leaf != Some(complete_leaf));
            request_mismatches += usize::from(
                self.plan.mesh.requested_lods.get(component_index)
                    != complete_plan.requested_lods.get(complete_index),
            );
            resident_mismatches += usize::from(
                component_resident.resident.get(component_index)
                    != complete_resident.resident.get(complete_index),
            );
            vertex_mismatches += usize::from(
                component_vertices.get(component_index) != complete_vertices.get(complete_index),
            );
            component_index += 1;
        }
        let unvisited = self.plan.mesh.leaves.len().saturating_sub(component_index);
        leaf_mismatches += unvisited;
        request_mismatches += unvisited;
        resident_mismatches += unvisited;
        vertex_mismatches += unvisited;
        let plan_exact = diagnostic_match
            && selected_faces_match
            && leaf_mismatches == 0
            && request_mismatches == 0
            && resident_mismatches == 0
            && vertex_mismatches == 0;
        let compare_ms = browser_now_ms() - compare_start;
        comparison.complete_leaves = complete_plan.leaves.len() as u64;
        comparison.diagnostic_match = diagnostic_match;
        comparison.selected_faces_match = selected_faces_match;
        comparison.leaf_mismatches = leaf_mismatches as u64;
        comparison.request_mismatches = request_mismatches as u64;
        comparison.resident_mismatches = resident_mismatches as u64;
        comparison.vertex_mismatches = vertex_mismatches as u64;
        comparison.plan_exact = plan_exact;
        comparison.oracle_sampled = true;
        comparison.compare_ms = compare_ms;
        comparison.total_ms += compare_ms;
        if !plan_exact {
            self.revoke_staged_authority();
            self.mismatches = self.mismatches.saturating_add(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_overlay(
        &mut self,
        complete_overlay: &AdaptiveRenderOverlay,
        baseline_groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        baseline_residents: &[Option<ResidentLod>],
        baseline_vertex_lods: &[[u32; 3]],
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        initial: ResidentLod,
        atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
        complete_triangles: u64,
        full_group_signature: AdaptiveGroupSignature,
        complete_reconciliation: (u64, u64, u64),
    ) {
        if !self.enabled {
            return;
        }
        let Some(comparison) = self.last.as_mut() else {
            return;
        };
        if comparison.overlay_compared || !comparison.plan_exact {
            return;
        }
        let Some(frontier) = self.frontier.as_ref() else {
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some("adaptive component shadow frontier disappeared".into());
            return;
        };
        let Some(resident) = self.reconciliation_cache.result.as_ref() else {
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some("adaptive component shadow reconciliation disappeared".into());
            return;
        };
        let overlay_start = browser_now_ms();
        if let Err(error) = group_resident_screen_component_overlay_into(
            frontier,
            &resident.resident,
            &self.plan.component_faces,
            baseline_residents,
            baseline_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
            &mut self.component_lod_scratch,
            &mut self.overlay,
        ) {
            self.revoke_staged_authority();
            self.mismatches = self.mismatches.saturating_add(1);
            self.last_error = Some(error.to_string());
            return;
        }
        comparison.overlay_ms = browser_now_ms() - overlay_start;
        comparison.total_ms += comparison.overlay_ms;
        comparison.overlay_compared = true;
        comparison.suppressed_faces_match =
            self.overlay.suppressed_faces == complete_overlay.suppressed_faces;
        comparison.overlay_groups_match = self.overlay.groups == complete_overlay.groups;
        let work = match measure_adaptive_render_work(
            baseline_groups,
            &self.overlay,
            baseline_residents,
            initial,
            atlas_triangle_counts,
        ) {
            Ok(work) => work,
            Err(error) => {
                self.revoke_staged_authority();
                self.mismatches = self.mismatches.saturating_add(1);
                self.last_error = Some(error.to_string());
                return;
            }
        };
        comparison.atlas_residency_valid = true;
        comparison.baseline_triangles = work.baseline_triangles;
        comparison.suppressed_root_triangles = work.suppressed_root_triangles;
        comparison.overlay_triangles = work.overlay_triangles;
        comparison.composed_triangles = work.composed_triangles;
        comparison.complete_triangles = complete_triangles;
        comparison.triangle_budget_match = work.composed_triangles == complete_triangles;
        comparison.exact = comparison.plan_exact
            && comparison.suppressed_faces_match
            && comparison.overlay_groups_match
            && comparison.atlas_residency_valid
            && comparison.triangle_budget_match;
        if comparison.exact {
            comparison.authoritative = true;
            self.matches = self.matches.saturating_add(1);
            if comparison.oracle_sampled {
                let stage_signature = self
                    .staged_signature
                    .expect("sampled component comparison has a stage signature");
                self.certificate = Some(AdaptiveComponentCertificate {
                    stage_signature,
                    diagnostic: self.plan.mesh.diagnostic,
                    selected_faces: self.plan.mesh.selected_faces.clone(),
                    full_group_signature,
                    complete_triangles,
                    complete_reconciliation,
                });
                self.record_exact_oracle_match();
            }
        } else {
            self.revoke_staged_authority();
            self.mismatches = self.mismatches.saturating_add(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_authoritative_overlay(
        &mut self,
        target: &mut AdaptiveRenderOverlay,
        baseline_groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        baseline_residents: &[Option<ResidentLod>],
        baseline_vertex_lods: &[[u32; 3]],
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        initial: ResidentLod,
        atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
        max_triangles: u64,
    ) -> Result<u64, String> {
        if !self.enabled {
            return Err("adaptive component authority is disabled".into());
        }
        let frontier = self
            .frontier
            .as_ref()
            .ok_or_else(|| "adaptive component frontier disappeared".to_string())?;
        let resident = self
            .reconciliation_cache
            .result
            .as_ref()
            .ok_or_else(|| "adaptive component reconciliation disappeared".to_string())?;
        let overlay_start = browser_now_ms();
        if let Err(error) = group_resident_screen_component_overlay_into(
            frontier,
            &resident.resident,
            &self.plan.component_faces,
            baseline_residents,
            baseline_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
            &mut self.component_lod_scratch,
            &mut self.overlay,
        ) {
            self.revoke_staged_authority();
            return Err(error.to_string());
        }
        let work = match measure_adaptive_render_work(
            baseline_groups,
            &self.overlay,
            baseline_residents,
            initial,
            atlas_triangle_counts,
        ) {
            Ok(work) => work,
            Err(error) => {
                self.revoke_staged_authority();
                return Err(error.to_string());
            }
        };
        if work.composed_triangles > max_triangles {
            return Err(format!(
                "adaptive component authority needs {} triangles; budget is {max_triangles}",
                work.composed_triangles,
            ));
        }
        let Some(comparison) = self.last.as_mut() else {
            self.revoke_staged_authority();
            return Err("adaptive component authority lost its staged diagnostic".into());
        };
        comparison.overlay_ms = browser_now_ms() - overlay_start;
        comparison.total_ms += comparison.overlay_ms;
        comparison.overlay_compared = false;
        comparison.atlas_residency_valid = true;
        comparison.baseline_triangles = work.baseline_triangles;
        comparison.suppressed_root_triangles = work.suppressed_root_triangles;
        comparison.overlay_triangles = work.overlay_triangles;
        comparison.composed_triangles = work.composed_triangles;
        comparison.complete_triangles = 0;
        comparison.triangle_budget_match = true;
        comparison.oracle_sampled = false;
        comparison.authoritative = true;
        comparison.exact = false;
        std::mem::swap(target, &mut self.overlay);
        Ok(work.composed_triangles)
    }

    fn swap_exact_overlay(
        &mut self,
        target: &mut AdaptiveRenderOverlay,
    ) -> Result<(), String> {
        if let Some(error) = self.last_error.as_deref() {
            return Err(error.to_string());
        }
        if !self.last.is_some_and(|comparison| comparison.exact) {
            return Err("adaptive component publication did not match the complete oracle".into());
        }
        std::mem::swap(target, &mut self.overlay);
        Ok(())
    }
}

/// On-demand workload evidence for retaining unchanged root batches while
/// publishing only the exact adaptive replacement closure. Measuring this is
/// deliberately explicit: rebuilding leaf corner densities is real work and
/// must not silently become part of every animation frame.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AdaptiveOverlayMeasurement {
    ok: bool,
    retained_shadow_reused: bool,
    source_faces: u64,
    frontier_leaves: u64,
    complete_groups: u64,
    retained_root_faces: u64,
    suppressed_faces: u64,
    suppressed_face_sample: Vec<u32>,
    overlay_groups: u64,
    overlay_members: u64,
    overlay_root_members: u64,
    overlay_dyadic_members: u64,
    avoided_publication_members: u64,
    publication_member_reduction_percent: f64,
    membership_match: bool,
    membership_mismatch_groups: u64,
    membership_mismatch_members: u64,
    layer_order_match: bool,
    layer_order_mismatch_groups: u64,
    layer_order_mismatch_materials: Vec<usize>,
    overlay_extraction_ms: f64,
    parity_ms: f64,
    elapsed_ms: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AdaptivePickedRefreshSnapshot<'a> {
    #[serde(flatten)]
    pub snapshot: AdaptivePickedSnapshot<'a>,
    pub transition_pending: bool,
    pub retained_publication_enabled: bool,
    pub retained_publication_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retained_publication_error: Option<&'a str>,
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
    priority_candidates: u64,
    selected_max_priority: u8,
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
            priority_candidates: diagnostic.priority_candidates,
            selected_max_priority: diagnostic.selected_max_priority,
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
        self.plan_scratch = AdaptiveScreenMeshPlan::default();
        self.frontier_cache_hits = 0;
        self.frontier_cache_misses = 0;
        self.reconciliation_cache = AdaptiveReconciliationCache::default();
        self.published_group_signature = None;
        self.pending_group_signature = None;
        self.group_cache_hits = 0;
        self.group_cache_misses = 0;
        self.last_timings = AdaptivePlanTimings::default();
        self.component_shadow = AdaptiveComponentShadow::default();
        self.component_publication_enabled = false;
        self.pending_component_publication = false;
        self.pending_component_authority = false;
        self.published_component_publication = false;
        self.component_publication_installs = 0;
        self.component_authority_changed_installs = 0;
        self.component_publication_fallbacks = 0;
        self.component_publication_last_error = None;
        self.retained_shadow_enabled = false;
        self.pending_overlay = AdaptiveRenderOverlay::default();
        self.pending_overlay_signature = None;
        self.pending_overlay_reuses_published = false;
        self.published_overlay = AdaptiveRenderOverlay::default();
        self.published_overlay_signature = None;
        self.retained_shadow_cache_hits = 0;
        self.retained_shadow_cache_misses = 0;
        self.last_retained_shadow_stage_ms = 0.0;
        self.overlay_shadow = AdaptiveRenderOverlay::default();
        self.overlay_baseline_groups.clear();
        self.overlay_member_scratch.clear();
        self.overlay_complete_scratch.clear();
        self.candidate_groups.clear();
        self.clear_pending_publication();
    }

    /// Request a transactional return to the retained legacy root grouping.
    /// Published diagnostics remain intact until that handoff succeeds.
    pub(crate) fn stage_clear(&mut self) {
        self.config = None;
        self.clear_pending_publication();
        self.pending_legacy = true;
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    pub(crate) fn has_pending_publication(&self) -> bool {
        self.pending_plan.is_some() || self.pending_fallback_error.is_some() || self.pending_legacy
    }

    pub(crate) fn has_pending_plan(&self) -> bool {
        self.pending_plan.is_some()
    }

    pub(crate) fn take_group_rollback(
        &mut self,
    ) -> BTreeMap<RenderBatchKey, Vec<RenderBatchMember>> {
        std::mem::take(&mut self.candidate_groups)
    }

    pub(crate) fn recycle_group_scratch(
        &mut self,
        groups: BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    ) {
        self.candidate_groups = groups;
    }

    pub(crate) fn config(&self) -> Option<AdaptivePickedConfig> {
        self.config
    }

    pub(crate) fn published_faces(&self) -> &[u32] {
        &self.last_published_faces
    }

    pub(crate) fn set_retained_shadow_enabled(&mut self, enabled: bool) -> Result<bool, String> {
        if self.has_pending_publication() {
            return Err("adaptive publication is already staged".into());
        }
        if !enabled && self.component_shadow.enabled {
            return Err("adaptive component shadow requires retained overlay shadow".into());
        }
        let changed = self.retained_shadow_enabled != enabled;
        self.retained_shadow_enabled = enabled;
        self.pending_overlay_signature = None;
        self.pending_overlay_reuses_published = false;
        if !enabled {
            self.published_overlay_signature = None;
        }
        Ok(changed)
    }

    pub(crate) fn set_component_shadow_enabled(&mut self, enabled: bool) -> Result<bool, String> {
        if self.has_pending_publication() {
            return Err("adaptive publication is already staged".into());
        }
        if !enabled && self.component_publication_enabled {
            return Err("adaptive component publication requires its exact shadow oracle".into());
        }
        if enabled {
            self.set_retained_shadow_enabled(true)?;
        }
        Ok(self.component_shadow.set_enabled(enabled))
    }

    pub(crate) fn set_component_publication_enabled(
        &mut self,
        enabled: bool,
    ) -> Result<bool, String> {
        if self.has_pending_publication() {
            return Err("adaptive publication is already staged".into());
        }
        if enabled {
            self.set_component_shadow_enabled(true)?;
        }
        let changed = self.component_publication_enabled != enabled;
        self.component_publication_enabled = enabled;
        self.pending_component_publication = false;
        self.pending_component_authority = false;
        self.component_publication_last_error = None;
        if changed {
            // Force one freshly component-derived overlay across the cutover
            // boundary instead of inheriting an equal complete-shadow cache.
            self.published_overlay_signature = None;
            self.component_shadow.reset_authority_basis();
        }
        Ok(changed)
    }

    pub(crate) fn component_publication_enabled(&self) -> bool {
        self.component_publication_enabled
    }

    fn component_publication_state(&self) -> &'static str {
        if !self.component_publication_enabled {
            "disabled"
        } else if self.pending_component_publication {
            "staged"
        } else if self.published_component_publication {
            "active"
        } else if self.component_publication_last_error.is_some() {
            "complete-fallback"
        } else {
            "awaiting-exact-component"
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage_pending_overlay(
        &mut self,
        batch_layout_revision: u64,
        baseline_groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        baseline_residents: &[Option<ResidentLod>],
        baseline_vertex_lods: &[[u32; 3]],
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        initial: ResidentLod,
        atlas_triangle_counts: &BTreeMap<[u32; 3], u64>,
    ) -> Result<bool, String> {
        if !self.retained_shadow_enabled {
            self.pending_overlay_signature = None;
            self.pending_overlay_reuses_published = false;
            self.last_retained_shadow_stage_ms = 0.0;
            return Ok(false);
        }
        if self.pending_plan.is_none() {
            return Err("adaptive overlay has no staged plan".into());
        }
        let signature = self
            .pending_group_signature
            .ok_or_else(|| "staged adaptive group identity is unavailable".to_string())?;
        if signature.batch_layout_revision != batch_layout_revision {
            return Err("staged adaptive group identity does not match the batch layout".into());
        }
        if self.pending_component_authority {
            if signature.authority != AdaptiveGroupAuthority::Component {
                return Err("component authority staged a non-component group identity".into());
            }
            let max_triangles = self
                .config
                .ok_or_else(|| "adaptive screen mode is disabled".to_string())?
                .max_triangles;
            let stage_start = browser_now_ms();
            let triangles = self.component_shadow.stage_authoritative_overlay(
                &mut self.pending_overlay,
                baseline_groups,
                baseline_residents,
                baseline_vertex_lods,
                face_materials,
                face_nodes,
                face_render_nodes,
                initial,
                atlas_triangle_counts,
                max_triangles,
            )?;
            self.pending_triangles = triangles;
            self.pending_overlay_signature = Some(signature);
            self.pending_overlay_reuses_published = false;
            self.pending_component_publication = true;
            self.component_publication_last_error = None;
            self.retained_shadow_cache_misses =
                self.retained_shadow_cache_misses.saturating_add(1);
            self.last_retained_shadow_stage_ms = browser_now_ms() - stage_start;
            return Ok(false);
        }
        if self.published_overlay_signature == Some(signature) {
            self.pending_overlay_signature = Some(signature);
            self.pending_overlay_reuses_published = true;
            self.retained_shadow_cache_hits = self.retained_shadow_cache_hits.saturating_add(1);
            self.last_retained_shadow_stage_ms = 0.0;
            self.component_shadow.compare_overlay(
                &self.published_overlay,
                baseline_groups,
                baseline_residents,
                baseline_vertex_lods,
                face_materials,
                face_nodes,
                face_render_nodes,
                initial,
                atlas_triangle_counts,
                self.pending_triangles,
                signature,
                (
                    self.pending_shared_edge_promotions,
                    self.pending_grading_promotions,
                    self.pending_reconciliation_iterations,
                ),
            );
            self.pending_component_publication = self.component_publication_enabled
                && self.published_component_publication;
            if self.component_publication_enabled && !self.pending_component_publication {
                self.component_publication_fallbacks =
                    self.component_publication_fallbacks.saturating_add(1);
                self.component_publication_last_error = Some(
                    "exact component overlay was not the published cache epoch".into(),
                );
            }
            return Ok(true);
        }
        let stage_start = browser_now_ms();
        let frontier = self
            .frontier
            .as_ref()
            .ok_or_else(|| "staged adaptive frontier is unavailable".to_string())?;
        let resident_edge_lods = &self
            .reconciliation_cache
            .result
            .as_ref()
            .ok_or_else(|| "staged adaptive reconciliation is unavailable".to_string())?
            .resident;
        group_resident_screen_overlay_into(
            frontier,
            resident_edge_lods,
            baseline_residents,
            baseline_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
            &mut self.lod_scratch,
            &mut self.pending_overlay,
        )
        .map_err(|error| error.to_string())?;
        self.pending_overlay_signature = Some(signature);
        self.pending_overlay_reuses_published = false;
        self.retained_shadow_cache_misses = self.retained_shadow_cache_misses.saturating_add(1);
        self.last_retained_shadow_stage_ms = browser_now_ms() - stage_start;
        self.component_shadow.compare_overlay(
            &self.pending_overlay,
            baseline_groups,
            baseline_residents,
            baseline_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
            atlas_triangle_counts,
            self.pending_triangles,
            signature,
            (
                self.pending_shared_edge_promotions,
                self.pending_grading_promotions,
                self.pending_reconciliation_iterations,
            ),
        );
        if self.component_publication_enabled {
            match self
                .component_shadow
                .swap_exact_overlay(&mut self.pending_overlay)
            {
                Ok(()) => {
                    self.pending_component_publication = true;
                    self.component_publication_last_error = None;
                }
                Err(error) => {
                    self.pending_component_publication = false;
                    self.component_publication_fallbacks =
                        self.component_publication_fallbacks.saturating_add(1);
                    self.component_publication_last_error = Some(error);
                }
            }
        }
        Ok(false)
    }

    pub(crate) fn reject_staged_overlay(
        &mut self,
        groups_reused: bool,
        live_groups: &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        error: impl Into<String>,
    ) {
        if !groups_reused {
            std::mem::swap(live_groups, &mut self.candidate_groups);
        }
        self.fallbacks = self.fallbacks.saturating_add(1);
        self.clear_pending_publication();
        self.pending_fallback_error = Some(error.into());
    }

    pub(crate) fn pending_overlay_for_publication(
        &self,
        batch_layout_revision: u64,
    ) -> Result<&AdaptiveRenderOverlay, String> {
        let signature = self
            .pending_overlay_signature
            .ok_or_else(|| "retained adaptive overlay is not staged".to_string())?;
        if signature.batch_layout_revision != batch_layout_revision
            || self.pending_group_signature != Some(signature)
        {
            return Err("retained adaptive overlay does not match the staged batch epoch".into());
        }
        if self.pending_overlay_reuses_published {
            if self.published_overlay_signature != Some(signature) {
                return Err("retained adaptive overlay cache identity is stale".into());
            }
            Ok(&self.published_overlay)
        } else {
            Ok(&self.pending_overlay)
        }
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
        self.pending_group_signature = None;
        self.pending_overlay_signature = None;
        self.pending_overlay_reuses_published = false;
        self.pending_component_publication = false;
        self.pending_component_authority = false;
        self.pending_selected_faces.clear();
    }

    fn commit_pending_overlay(&mut self) {
        let Some(signature) = self.pending_overlay_signature.take() else {
            self.published_overlay_signature = None;
            return;
        };
        if !self.pending_overlay_reuses_published {
            std::mem::swap(&mut self.published_overlay, &mut self.pending_overlay);
        }
        self.published_overlay_signature = Some(signature);
        self.pending_overlay_reuses_published = false;
    }

    fn clear_published_overlay(&mut self) {
        self.published_overlay_signature = None;
        self.published_component_publication = false;
    }

    /// Commit diagnostics only after the renderer has published every staged
    /// GL bucket. CPU grouping alone is not a visible install.
    #[cfg(test)]
    pub(crate) fn commit_publication(&mut self) {
        self.commit_publication_mode(true);
    }

    pub(crate) fn commit_gpu_publication(&mut self, retained_gpu_publication: bool) {
        self.commit_publication_mode(retained_gpu_publication);
    }

    fn commit_publication_mode(&mut self, retained_gpu_publication: bool) {
        let component_candidate = self.pending_component_publication;
        let component_authority = self.pending_component_authority;
        if let Some(plan) = self.pending_plan.take() {
            self.commit_pending_overlay();
            self.published_component_publication =
                retained_gpu_publication && component_candidate;
            if self.published_component_publication {
                self.component_publication_installs =
                    self.component_publication_installs.saturating_add(1);
                if component_authority {
                    self.component_authority_changed_installs = self
                        .component_authority_changed_installs
                        .saturating_add(1);
                }
            } else if component_candidate {
                self.component_publication_fallbacks =
                    self.component_publication_fallbacks.saturating_add(1);
                self.component_publication_last_error = Some(
                    "retained GPU publication was unavailable; complete GPU epoch committed"
                        .into(),
                );
            }
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
            let pending_signature = self.pending_group_signature.take();
            if pending_signature
                .is_some_and(|signature| signature.authority == AdaptiveGroupAuthority::Complete)
            {
                self.published_group_signature = pending_signature;
            }
        } else if let Some(error) = self.pending_fallback_error.take() {
            self.clear_published_overlay();
            self.last_error = Some(error);
            self.last_plan = None;
            self.last_selection = None;
            self.last_triangles = 0;
            self.last_shared_edge_promotions = 0;
            self.last_grading_promotions = 0;
            self.last_reconciliation_iterations = 0;
            self.last_pose_stamp = None;
            self.last_published_faces.clear();
            self.published_group_signature = None;
        } else if self.pending_legacy {
            self.clear_published_overlay();
            self.last_error = None;
            self.last_plan = None;
            self.last_selection = None;
            self.last_triangles = 0;
            self.last_shared_edge_promotions = 0;
            self.last_grading_promotions = 0;
            self.last_reconciliation_iterations = 0;
            self.last_pose_stamp = None;
            self.last_published_faces.clear();
            self.published_group_signature = None;
        } else {
            return;
        }
        self.last_publication_error = None;
        self.clear_pending_publication();
    }

    pub(crate) fn record_publication_failure(&mut self, error: impl Into<String>) {
        let error = error.into();
        if self.pending_plan.is_some() || self.pending_fallback_error.is_some() {
            self.fallbacks = self.fallbacks.saturating_add(1);
        }
        if self.pending_component_publication {
            self.component_publication_fallbacks =
                self.component_publication_fallbacks.saturating_add(1);
            self.component_publication_last_error = Some(error.clone());
        }
        let retry_legacy = self.pending_legacy;
        self.clear_pending_publication();
        self.pending_legacy = retry_legacy;
        self.last_publication_error = Some(error);
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

    /// Extract sparse replacement workload from the exact currently published
    /// adaptive epoch. Staged and rolled-back candidates are rejected because
    /// their cached frontier does not describe the live GPU batch map.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn measure_published_overlay(
        &mut self,
        batch_layout_revision: u64,
        complete_groups: &BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        baseline_residents: &[Option<ResidentLod>],
        baseline_vertex_lods: &[[u32; 3]],
        face_materials: &[usize],
        face_nodes: &[usize],
        face_render_nodes: &[usize],
        initial: ResidentLod,
    ) -> Result<AdaptiveOverlayMeasurement, String> {
        if self.config.is_none() {
            return Err("adaptive screen mode is disabled".into());
        }
        if self.has_pending_publication() {
            return Err("adaptive screen publication is still staged".into());
        }
        if let Some(error) = self.last_publication_error.as_deref() {
            return Err(format!("adaptive screen publication rolled back: {error}"));
        }
        if let Some(error) = self.last_error.as_deref() {
            return Err(format!("adaptive screen renderer is using its fallback: {error}"));
        }
        if self.last_plan.is_none() {
            return Err("adaptive screen frontier has not been published".into());
        }
        if self
            .published_overlay_signature
            .is_some_and(|signature| signature.authority == AdaptiveGroupAuthority::Component)
        {
            return Err(
                "published adaptive epoch is awaiting its next complete grouping oracle".into(),
            );
        }
        let signature = self
            .published_group_signature
            .ok_or_else(|| "published adaptive group identity is unavailable".to_string())?;
        if signature.authority != AdaptiveGroupAuthority::Complete
            || signature.generation != self.reconciliation_cache.generation
            || signature.batch_layout_revision != batch_layout_revision
        {
            return Err("cached adaptive frontier is not the published batch epoch".into());
        }
        let frontier = self
            .frontier
            .as_ref()
            .ok_or_else(|| "published adaptive frontier is unavailable".to_string())?;
        let resident_edge_lods = &self
            .reconciliation_cache
            .result
            .as_ref()
            .ok_or_else(|| "published adaptive reconciliation is unavailable".to_string())?
            .resident;

        let start = browser_now_ms();
        let retained_shadow_reused = self.published_overlay_signature == Some(signature);
        if !retained_shadow_reused {
            group_resident_screen_overlay_into(
                frontier,
                resident_edge_lods,
                baseline_residents,
                baseline_vertex_lods,
                face_materials,
                face_nodes,
                face_render_nodes,
                initial,
                &mut self.lod_scratch,
                &mut self.overlay_shadow,
            )
            .map_err(|error| error.to_string())?;
        }
        let overlay_extraction_ms = browser_now_ms() - start;
        let overlay = if retained_shadow_reused {
            &self.published_overlay
        } else {
            &self.overlay_shadow
        };

        let parity_start = browser_now_ms();
        group_resident_faces_into(
            baseline_residents,
            baseline_vertex_lods,
            face_materials,
            face_nodes,
            face_render_nodes,
            initial,
            &mut self.overlay_baseline_groups,
        );
        let keys = complete_groups
            .keys()
            .chain(self.overlay_baseline_groups.keys())
            .chain(overlay.groups.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        let mut membership_mismatch_groups = 0usize;
        let mut membership_mismatch_members = 0usize;
        let mut layer_order_mismatch_groups = 0usize;
        let mut layer_order_mismatch_materials = BTreeSet::new();
        for key in keys {
            self.overlay_member_scratch.clear();
            if let Some(roots) = self.overlay_baseline_groups.get(&key) {
                self.overlay_member_scratch.extend(
                    roots.iter().copied().filter(|member| {
                        overlay
                            .suppressed_faces
                            .binary_search(&member.face_index)
                            .is_err()
                    }),
                );
            }
            if let Some(replacements) = overlay.groups.get(&key) {
                self.overlay_member_scratch
                    .extend_from_slice(replacements);
            }
            let complete = complete_groups.get(&key).map(Vec::as_slice).unwrap_or(&[]);
            if self.overlay_member_scratch != complete {
                layer_order_mismatch_groups += 1;
                layer_order_mismatch_materials.insert(key.material_index);
            }

            self.overlay_member_scratch
                .sort_unstable_by_key(|member| member.patch_id());
            self.overlay_complete_scratch.clear();
            self.overlay_complete_scratch.extend_from_slice(complete);
            self.overlay_complete_scratch
                .sort_unstable_by_key(|member| member.patch_id());
            if self.overlay_member_scratch != self.overlay_complete_scratch {
                membership_mismatch_groups += 1;
                membership_mismatch_members += self
                    .overlay_member_scratch
                    .iter()
                    .zip(&self.overlay_complete_scratch)
                    .filter(|(left, right)| left != right)
                    .count()
                    + self
                        .overlay_member_scratch
                        .len()
                        .abs_diff(self.overlay_complete_scratch.len());
            }
        }
        let parity_ms = browser_now_ms() - parity_start;
        let elapsed_ms = browser_now_ms() - start;

        let source_faces = baseline_residents.len();
        let suppressed_faces = overlay.suppressed_faces.len();
        let overlay_members = overlay
            .groups
            .values()
            .map(Vec::len)
            .sum::<usize>();
        let overlay_root_members = overlay
            .groups
            .values()
            .flatten()
            .filter(|member| member.leaf_id == ScreenPatchLeafId::ROOT)
            .count();
        let frontier_leaves = frontier.leaves().len();
        let retained_root_faces = source_faces.saturating_sub(suppressed_faces);
        let composed_members = retained_root_faces.saturating_add(overlay_members);
        if composed_members != frontier_leaves {
            return Err(format!(
                "sparse overlay composes {composed_members} members, expected {frontier_leaves}",
            ));
        }
        let avoided_publication_members = frontier_leaves.saturating_sub(overlay_members);
        let publication_member_reduction_percent = if frontier_leaves == 0 {
            0.0
        } else {
            100.0 * avoided_publication_members as f64 / frontier_leaves as f64
        };

        Ok(AdaptiveOverlayMeasurement {
            ok: true,
            retained_shadow_reused,
            source_faces: source_faces as u64,
            frontier_leaves: frontier_leaves as u64,
            complete_groups: complete_groups.len() as u64,
            retained_root_faces: retained_root_faces as u64,
            suppressed_faces: suppressed_faces as u64,
            suppressed_face_sample: overlay
                .suppressed_faces
                .iter()
                .copied()
                .take(16)
                .collect(),
            overlay_groups: overlay.groups.len() as u64,
            overlay_members: overlay_members as u64,
            overlay_root_members: overlay_root_members as u64,
            overlay_dyadic_members: overlay_members.saturating_sub(overlay_root_members) as u64,
            avoided_publication_members: avoided_publication_members as u64,
            publication_member_reduction_percent,
            membership_match: membership_mismatch_groups == 0,
            membership_mismatch_groups: membership_mismatch_groups as u64,
            membership_mismatch_members: membership_mismatch_members as u64,
            layer_order_match: layer_order_mismatch_groups == 0,
            layer_order_mismatch_groups: layer_order_mismatch_groups as u64,
            layer_order_mismatch_materials: layer_order_mismatch_materials
                .into_iter()
                .collect(),
            overlay_extraction_ms,
            parity_ms,
            elapsed_ms,
        })
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
            0,
            0,
            false,
            false,
            None,
            live_groups,
        )
        .map(|_| ())
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
        batch_layout_revision: u64,
        root_topology_revision: u64,
        component_hot_path_allowed: bool,
        published_groups_are_live: bool,
        mut reusable_groups: Option<
            &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
        >,
        live_groups: &mut BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>,
    ) -> Result<bool, String> {
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
            let plan_request = AdaptiveScreenMeshPlanRequest {
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
            };
            let (component_plan_staged, certified_component) =
                match self.component_shadow.stage_plan(
                    plan_request,
                    topology,
                    max_face_edge_ratio,
                    max_atlas_lod,
                    root_topology_revision,
                    batch_layout_revision,
                ) {
                    Ok(certificate) => (self.component_shadow.enabled, certificate),
                    Err(_) => (false, None),
                };
            if component_hot_path_allowed {
                let reusable_certificate = certified_component.filter(
                    |certificate| {
                        published_groups_are_live
                            && self.published_group_signature
                                == Some(certificate.group_signature)
                            && self.published_overlay_signature
                                == Some(certificate.group_signature)
                    },
                );
                if let Some(certificate) = reusable_certificate {
                    self.component_shadow.certified_reuses =
                        self.component_shadow.certified_reuses.saturating_add(1);
                    self.group_cache_hits = self.group_cache_hits.saturating_add(1);
                    let component = &self.component_shadow.plan.mesh;
                    let comparison = self
                        .component_shadow
                        .last
                        .expect("staged certified component comparison");
                    let timings = AdaptivePlanTimings {
                        total_ms: comparison.total_ms,
                        mesh_plan_ms: comparison.plan_ms,
                        frontier_ms: comparison.frontier_ms,
                        reconcile_ms: comparison.reconcile_ms,
                        atlas_work_ms: 0.0,
                        group_ms: 0.0,
                    };
                    return Ok((
                        component.diagnostic,
                        component.selected_faces.clone(),
                        certificate.reconciliation,
                        certificate.group_signature,
                        true,
                        certificate.triangles,
                        timings,
                    ));
                }
                self.component_shadow.certified_misses =
                    self.component_shadow.certified_misses.saturating_add(1);
                if certified_component.is_none() {
                    if let Some(authority) =
                        self.component_shadow.changed_authority_signature()?
                    {
                        self.pending_component_authority = true;
                        let component = &self.component_shadow.plan.mesh;
                        let comparison = self
                            .component_shadow
                            .last
                            .expect("staged component-authority diagnostic");
                        let timings = AdaptivePlanTimings {
                            total_ms: comparison.total_ms,
                            mesh_plan_ms: comparison.plan_ms,
                            frontier_ms: comparison.frontier_ms,
                            reconcile_ms: comparison.reconcile_ms,
                            atlas_work_ms: 0.0,
                            group_ms: 0.0,
                        };
                        return Ok((
                            component.diagnostic,
                            component.selected_faces.clone(),
                            authority.reconciliation,
                            authority.group_signature,
                            true,
                            0,
                            timings,
                        ));
                    }
                }
            }
            plan_adaptive_screen_mesh_into(plan_request, &mut self.plan_scratch)
                .map_err(|error| error.to_string())?;
            let plan = &self.plan_scratch;
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
            let (reconciled, reconciliation_generation, _) = self
                .reconciliation_cache
                .resolve(
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
            let group_signature = AdaptiveGroupSignature {
                authority: AdaptiveGroupAuthority::Complete,
                generation: reconciliation_generation,
                batch_layout_revision,
            };
            let groups_reused = if self.published_group_signature == Some(group_signature) {
                if published_groups_are_live {
                    self.group_cache_hits = self.group_cache_hits.saturating_add(1);
                    true
                } else if let Some(groups) = reusable_groups.as_deref_mut() {
                    std::mem::swap(live_groups, groups);
                    self.group_cache_hits = self.group_cache_hits.saturating_add(1);
                    true
                } else {
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
                    self.group_cache_misses = self.group_cache_misses.saturating_add(1);
                    false
                }
            } else {
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
                self.group_cache_misses = self.group_cache_misses.saturating_add(1);
                false
            };
            let group_ms = browser_now_ms() - group_start;
            let timings = AdaptivePlanTimings {
                total_ms: browser_now_ms() - total_start,
                mesh_plan_ms,
                frontier_ms,
                reconcile_ms,
                atlas_work_ms,
                group_ms,
            };
            if component_plan_staged {
                self.component_shadow
                    .compare_staged_plan(plan, frontier, reconciled);
            }
            let reconciliation = (
                reconciled.shared_edge_promotions as u64,
                reconciled.grading_promotions as u64,
                reconciled.iterations as u64,
            );
            Ok((
                plan.diagnostic,
                plan.selected_faces.clone(),
                reconciliation,
                group_signature,
                groups_reused,
                triangles,
                timings,
            ))
        })();

        match attempt {
            Ok((
                diagnostic,
                selected_faces,
                reconciliation,
                group_signature,
                groups_reused,
                triangles,
                timings,
            )) => {
                if !groups_reused {
                    std::mem::swap(live_groups, &mut self.candidate_groups);
                }
                self.pending_plan = Some(diagnostic);
                self.pending_selection = selection_diagnostic;
                self.pending_selected_faces = selected_faces;
                self.pending_triangles = triangles;
                self.pending_shared_edge_promotions = reconciliation.0;
                self.pending_grading_promotions = reconciliation.1;
                self.pending_reconciliation_iterations = reconciliation.2;
                self.pending_pose_stamp = current_pose_stamp;
                self.pending_group_signature = Some(group_signature);
                self.last_timings = timings;
                Ok(groups_reused)
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
        let retained_shadow_state = if !self.retained_shadow_enabled {
            "disabled"
        } else if self.pending_overlay_signature.is_some() {
            "staged"
        } else if self.published_overlay_signature.is_some() {
            "published"
        } else {
            "awaiting-plan"
        };
        let retained_shadow_suppressed_faces = self
            .published_overlay_signature
            .map_or(0, |_| self.published_overlay.suppressed_faces.len() as u64);
        let retained_shadow_groups = self
            .published_overlay_signature
            .map_or(0, |_| self.published_overlay.groups.len() as u64);
        let retained_shadow_members = self.published_overlay_signature.map_or(0, |_| {
            self.published_overlay
                .groups
                .values()
                .map(Vec::len)
                .sum::<usize>() as u64
        });
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
            group_cache_hits: self.group_cache_hits,
            group_cache_misses: self.group_cache_misses,
            component_shadow_enabled: self.component_shadow.enabled,
            component_shadow_state: self.component_shadow.state(),
            component_shadow_attempts: self.component_shadow.attempts,
            component_shadow_matches: self.component_shadow.matches,
            component_shadow_mismatches: self.component_shadow.mismatches,
            component_shadow_frontier_cache_hits: self.component_shadow.frontier_cache_hits,
            component_shadow_frontier_cache_misses: self.component_shadow.frontier_cache_misses,
            component_shadow_reconciliation_cache_hits: self
                .component_shadow
                .reconciliation_cache
                .hits,
            component_shadow_reconciliation_cache_misses: self
                .component_shadow
                .reconciliation_cache
                .misses,
            component_certified_reuses: self.component_shadow.certified_reuses,
            component_certified_misses: self.component_shadow.certified_misses,
            component_authority_oracle_interval: COMPONENT_AUTHORITY_ORACLE_INTERVAL,
            component_authority_oracle_samples: self.component_shadow.authority_oracle_samples,
            component_authority_oracle_skips: self.component_shadow.authority_oracle_skips,
            component_authority_sample_age: self
                .component_shadow
                .authority_changed_since_oracle,
            component_authority_revoked: self.component_shadow.current_authority_revoked(),
            component_authority_revocations: self.component_shadow.authority_revocations,
            component_authority_changed_installs: self.component_authority_changed_installs,
            component_shadow_last_error: self.component_shadow.last_error.as_deref(),
            component_shadow_last: self.component_shadow.last,
            component_publication_enabled: self.component_publication_enabled,
            component_publication_state: self.component_publication_state(),
            component_publication_installs: self.component_publication_installs,
            component_publication_fallbacks: self.component_publication_fallbacks,
            component_publication_last_error: self
                .component_publication_last_error
                .as_deref(),
            retained_shadow_enabled: self.retained_shadow_enabled,
            retained_shadow_state,
            retained_shadow_suppressed_faces,
            retained_shadow_groups,
            retained_shadow_members,
            retained_shadow_cache_hits: self.retained_shadow_cache_hits,
            retained_shadow_cache_misses: self.retained_shadow_cache_misses,
            last_retained_shadow_stage_ms: self.last_retained_shadow_stage_ms,
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
        let complete_groups = quilting_core::batch::group_resident_faces(
            &[Some(resident)],
            &[[1; 3]],
            &[0],
            &[0],
            &[0],
            resident,
        );
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
        runtime.set_retained_shadow_enabled(true).unwrap();

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
            runtime
                .stage_pending_overlay(
                    0,
                    &complete_groups,
                    &[Some(resident)],
                    &[[1; 3]],
                    &[0],
                    &[0],
                    &[0],
                    resident,
                    &atlas_triangles,
                )
                .unwrap();
        };

        plan(&mut runtime, (7, 2), 4);
        assert_eq!(runtime.snapshot().state, "staged");
        assert!(runtime.pending_overlay_for_publication(0).is_ok());
        assert_eq!(runtime.snapshot().frontier_cache_hits, 0);
        assert_eq!(runtime.snapshot().frontier_cache_misses, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_hits, 0);
        assert_eq!(runtime.snapshot().reconciliation_cache_misses, 1);
        runtime.commit_publication();
        assert!(runtime.pending_overlay_for_publication(0).is_err());
        assert_eq!(runtime.snapshot().pose_revision, Some(7));
        assert_eq!(runtime.snapshot().published_faces, [0]);
        assert_eq!(runtime.snapshot().last_plan.unwrap().selected_faces, 1);
        assert!(runtime.snapshot().retained_shadow_enabled);
        assert_eq!(runtime.snapshot().retained_shadow_state, "published");
        assert_eq!(runtime.snapshot().retained_shadow_suppressed_faces, 0);
        assert_eq!(runtime.snapshot().retained_shadow_cache_hits, 0);
        assert_eq!(runtime.snapshot().retained_shadow_cache_misses, 1);
        let overlay = runtime
            .measure_published_overlay(
                0,
                &complete_groups,
                &[Some(resident)],
                &[[1; 3]],
                &[0],
                &[0],
                &[0],
                resident,
            )
            .unwrap();
        assert!(overlay.ok);
        assert!(overlay.retained_shadow_reused);
        assert_eq!(overlay.frontier_leaves, 1);
        assert_eq!(overlay.retained_root_faces, 1);
        assert_eq!(overlay.suppressed_faces, 0);
        assert_eq!(overlay.overlay_members, 0);
        assert_eq!(overlay.avoided_publication_members, 1);
        assert!(overlay.membership_match);
        assert_eq!(overlay.membership_mismatch_groups, 0);
        assert!(overlay.layer_order_match);
        assert_eq!(overlay.layer_order_mismatch_groups, 0);

        plan(&mut runtime, (8, 2), 4);
        assert_eq!(runtime.snapshot().frontier_cache_hits, 1);
        assert_eq!(runtime.snapshot().frontier_cache_misses, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_hits, 1);
        assert_eq!(runtime.snapshot().reconciliation_cache_misses, 1);
        assert_eq!(
            runtime
                .measure_published_overlay(
                    0,
                    &complete_groups,
                    &[Some(resident)],
                    &[[1; 3]],
                    &[0],
                    &[0],
                    &[0],
                    resident,
                )
                .unwrap_err(),
            "adaptive screen publication is still staged",
        );
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
        assert_eq!(disabled.retained_shadow_state, "awaiting-plan");
        assert_eq!(disabled.retained_shadow_members, 0);
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
                    priority_candidates: 0,
                    selected_max_priority: 0,
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
                0,
                0,
                false,
                false,
                None,
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
            0,
            0,
            false,
            false,
            None,
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

        let reused = runtime
            .plan_selected_and_group(
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
                0,
                0,
                false,
                true,
                None,
                &mut groups,
            )
            .unwrap();
        assert!(reused);
        assert_eq!(groups, staged_groups);
        assert_eq!(runtime.snapshot().group_cache_hits, 1);
        assert_eq!(runtime.snapshot().group_cache_misses, 1);
        runtime.commit_publication();

        let reused = runtime
            .plan_selected_and_group(
                &selected,
                None,
                2,
                &IDENTITY,
                [640.0, 480.0],
                Some((5, 1)),
                &[Some(resident); 2],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0; 2],
                &[0; 2],
                &[0; 2],
                1,
                0,
                false,
                true,
                None,
                &mut groups,
            )
            .unwrap();
        assert!(!reused);
        assert_eq!(groups, staged_groups);
        let previous_groups = runtime.take_group_rollback();
        assert_eq!(previous_groups, staged_groups);
        assert_eq!(runtime.snapshot().group_cache_hits, 1);
        assert_eq!(runtime.snapshot().group_cache_misses, 2);
        runtime.commit_publication();
    }

    #[wasm_bindgen_test]
    fn changed_component_authority_samples_at_a_bounded_interval() {
        let mut shadow = AdaptiveComponentShadow {
            enabled: true,
            staged_signature: Some(AdaptiveComponentStageSignature {
                reconciliation_generation: 1,
                root_topology_revision: 7,
                batch_layout_revision: 9,
            }),
            authority_basis: Some((7, 9)),
            ..AdaptiveComponentShadow::default()
        };
        shadow.plan.mesh.diagnostic.source_faces = 100;
        shadow.plan.component_faces.extend(0..10);
        shadow.reconciliation_cache.result = Some(ScreenLeafLodResult {
            resident: vec![[1; 3]; 10],
            iterations: 2,
            shared_edge_promotions: 3,
            grading_promotions: 4,
            max_absolute_exponent: 0,
        });

        for _ in 1..COMPONENT_AUTHORITY_ORACLE_INTERVAL {
            let authority = shadow
                .changed_authority_signature()
                .unwrap()
                .expect("bounded changed component should skip the complete oracle");
            assert_eq!(
                authority.group_signature.authority,
                AdaptiveGroupAuthority::Component,
            );
            assert_eq!(authority.reconciliation, (3, 4, 2));
        }
        assert!(shadow.changed_authority_signature().unwrap().is_none());
        assert_eq!(
            shadow.authority_oracle_skips,
            COMPONENT_AUTHORITY_ORACLE_INTERVAL - 1,
        );
        shadow.record_exact_oracle_match();
        assert_eq!(shadow.authority_oracle_samples, 1);
        assert_eq!(shadow.authority_changed_since_oracle, 0);

        shadow.plan.component_faces = (0..100).collect();
        assert!(shadow.changed_authority_signature().unwrap().is_none());
    }

    #[wasm_bindgen_test]
    fn component_shadow_matches_a_complete_publication_across_disconnected_sources() {
        let positions = [
            [-0.8, -0.6, 0.0],
            [-0.2, -0.6, 0.0],
            [-0.5, 0.0, 0.0],
            [-0.8, 0.6, 0.0],
            [-0.2, 0.6, 0.0],
            [0.2, -0.5, 0.0],
            [0.8, -0.5, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [2, 3, 4], [5, 6, 7]],
        );
        let topology = ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let patch = QBTriPatch::flat(positions[0], positions[1], positions[2]);
        let selected = [SelectedScreenPatch {
            source_face: 0,
            transformed_patch: &patch,
        }];
        let resident = ResidentLod::uniform(1);
        let atlas_triangles = BTreeMap::from([([1; 3], 1)]);
        let baseline_groups = quilting_core::batch::group_resident_faces(
            &[Some(resident); 3],
            &[[1; 3]; 3],
            &[0; 3],
            &[0; 3],
            &[0; 3],
            resident,
        );
        let mut groups = BTreeMap::new();
        let mut runtime = AdaptivePickedRuntime::default();
        runtime.configure(AdaptivePickedConfig {
            selection: AdaptiveScreenSelection::Picked { face: 0 },
            min_px_per_segment: 1.0,
            max_px_per_segment: 10.0,
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: 6,
            max_triangles: 6,
        });
        runtime.set_component_publication_enabled(true).unwrap();
        assert_eq!(
            runtime.set_retained_shadow_enabled(false).unwrap_err(),
            "adaptive component shadow requires retained overlay shadow",
        );

        runtime
            .plan_selected_and_group(
                &selected,
                None,
                4,
                &IDENTITY,
                [640.0, 480.0],
                Some((3, 1)),
                &[Some(resident); 3],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0; 3],
                &[0; 3],
                &[0; 3],
                0,
                0,
                false,
                false,
                None,
                &mut groups,
            )
            .unwrap();
        runtime
            .stage_pending_overlay(
                0,
                &baseline_groups,
                &[Some(resident); 3],
                &[[1; 3]; 3],
                &[0; 3],
                &[0; 3],
                &[0; 3],
                resident,
                &atlas_triangles,
            )
            .unwrap();

        let snapshot = runtime.snapshot();
        assert!(snapshot.retained_shadow_enabled);
        assert!(snapshot.component_shadow_enabled);
        assert!(snapshot.component_publication_enabled);
        assert_eq!(snapshot.component_publication_state, "staged");
        assert_eq!(snapshot.component_publication_installs, 0);
        assert_eq!(snapshot.component_publication_fallbacks, 0);
        assert_eq!(snapshot.component_publication_last_error, None);
        assert_eq!(snapshot.component_shadow_state, "match");
        assert_eq!(snapshot.component_shadow_attempts, 1);
        assert_eq!(snapshot.component_shadow_matches, 1);
        assert_eq!(snapshot.component_shadow_mismatches, 0);
        assert_eq!(snapshot.component_shadow_last_error, None);
        let comparison = snapshot.component_shadow_last.unwrap();
        assert_eq!(comparison.source_faces, 3);
        assert_eq!(comparison.component_faces, 2);
        assert_eq!(comparison.unaffected_faces, 1);
        assert_eq!(comparison.complete_leaves, 6);
        assert_eq!(comparison.component_leaves, 5);
        assert!(comparison.plan_exact);
        assert!(comparison.overlay_compared);
        assert!(comparison.atlas_residency_valid);
        assert_eq!(comparison.baseline_triangles, 3);
        // The selected face becomes four dyadic leaves. Its welded neighbour
        // remains a root, but its C0 corner-density record changes and is
        // therefore replaced in the overlay as well.
        assert_eq!(comparison.suppressed_root_triangles, 2);
        assert_eq!(comparison.overlay_triangles, 5);
        assert_eq!(comparison.composed_triangles, 6);
        assert_eq!(comparison.complete_triangles, 6);
        assert!(comparison.triangle_budget_match);
        assert!(comparison.exact);

        runtime.commit_publication();
        let published = runtime.snapshot();
        assert_eq!(published.component_publication_state, "active");
        assert_eq!(published.component_publication_installs, 1);
        let reused_groups = runtime
            .plan_selected_and_group(
                &selected,
                None,
                4,
                &IDENTITY,
                [640.0, 480.0],
                Some((4, 1)),
                &[Some(resident); 3],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0; 3],
                &[0; 3],
                &[0; 3],
                0,
                0,
                true,
                true,
                None,
                &mut groups,
            )
            .unwrap();
        assert!(reused_groups);
        runtime
            .stage_pending_overlay(
                0,
                &baseline_groups,
                &[Some(resident); 3],
                &[[1; 3]; 3],
                &[0; 3],
                &[0; 3],
                &[0; 3],
                resident,
                &atlas_triangles,
            )
            .unwrap();
        let repeated = runtime.snapshot();
        assert_eq!(repeated.component_shadow_state, "match");
        assert_eq!(repeated.component_shadow_attempts, 2);
        assert_eq!(repeated.component_shadow_matches, 2);
        assert_eq!(repeated.component_shadow_mismatches, 0);
        assert_eq!(repeated.component_shadow_frontier_cache_hits, 1);
        assert_eq!(repeated.component_shadow_frontier_cache_misses, 1);
        assert_eq!(repeated.component_shadow_reconciliation_cache_hits, 1);
        assert_eq!(repeated.component_shadow_reconciliation_cache_misses, 1);
        assert_eq!(repeated.component_certified_reuses, 1);
        assert_eq!(repeated.component_certified_misses, 0);
        assert_eq!(repeated.frontier_cache_hits, 0);
        assert_eq!(repeated.frontier_cache_misses, 1);
        assert_eq!(repeated.reconciliation_cache_hits, 0);
        assert_eq!(repeated.reconciliation_cache_misses, 1);
        let repeated_comparison = repeated.component_shadow_last.unwrap();
        assert!(repeated_comparison.frontier_reused);
        assert!(repeated_comparison.reconciliation_reused);
        assert!(repeated_comparison.exact);
        assert_eq!(repeated.component_publication_state, "staged");
        runtime.commit_publication();
        let republished = runtime.snapshot();
        assert_eq!(republished.component_publication_state, "active");
        assert_eq!(republished.component_publication_installs, 2);
        assert_eq!(republished.component_publication_fallbacks, 0);

        let disconnected_patch = QBTriPatch::flat(positions[5], positions[6], positions[7]);
        let changed = [SelectedScreenPatch {
            source_face: 2,
            transformed_patch: &disconnected_patch,
        }];
        let changed_groups_reused = runtime
            .plan_selected_and_group(
                &changed,
                None,
                4,
                &IDENTITY,
                [640.0, 480.0],
                Some((5, 1)),
                &[Some(resident); 3],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0; 3],
                &[0; 3],
                &[0; 3],
                0,
                0,
                true,
                true,
                None,
                &mut groups,
            )
            .unwrap();
        assert!(changed_groups_reused);
        let changed_staged = runtime.snapshot();
        assert_eq!(changed_staged.component_authority_oracle_samples, 1);
        assert_eq!(changed_staged.component_authority_oracle_skips, 1);
        assert_eq!(changed_staged.component_authority_sample_age, 1);
        assert!(!changed_staged.component_authority_revoked);
        assert_eq!(changed_staged.frontier_cache_hits, 0);
        assert_eq!(changed_staged.frontier_cache_misses, 1);
        assert_eq!(changed_staged.reconciliation_cache_hits, 0);
        assert_eq!(changed_staged.reconciliation_cache_misses, 1);
        runtime
            .stage_pending_overlay(
                0,
                &baseline_groups,
                &[Some(resident); 3],
                &[[1; 3]; 3],
                &[0; 3],
                &[0; 3],
                &[0; 3],
                resident,
                &atlas_triangles,
            )
            .unwrap();
        let changed_overlay = runtime.snapshot();
        assert_eq!(
            changed_overlay.component_shadow_state,
            "authoritative-unsampled",
        );
        let changed_comparison = changed_overlay.component_shadow_last.unwrap();
        assert!(changed_comparison.authoritative);
        assert!(!changed_comparison.oracle_sampled);
        assert!(!changed_comparison.exact);
        assert_eq!(changed_comparison.composed_triangles, 6);
        runtime.commit_publication();
        let changed_published = runtime.snapshot();
        assert_eq!(changed_published.component_publication_state, "active");
        assert_eq!(changed_published.component_publication_installs, 3);
        assert_eq!(changed_published.component_authority_changed_installs, 1);
        assert_eq!(
            runtime.published_group_signature.unwrap().authority,
            AdaptiveGroupAuthority::Complete,
        );
        assert_eq!(
            runtime.published_overlay_signature.unwrap().authority,
            AdaptiveGroupAuthority::Component,
        );
        assert_eq!(
            runtime
                .measure_published_overlay(
                    0,
                    &groups,
                    &[Some(resident); 3],
                    &[[1; 3]; 3],
                    &[0; 3],
                    &[0; 3],
                    &[0; 3],
                    resident,
                )
                .unwrap_err(),
            "published adaptive epoch is awaiting its next complete grouping oracle",
        );

        runtime.set_component_publication_enabled(false).unwrap();
        runtime.set_component_publication_enabled(true).unwrap();
        runtime
            .plan_selected_and_group(
                &selected,
                None,
                4,
                &IDENTITY,
                [640.0, 480.0],
                Some((5, 1)),
                &[Some(resident); 3],
                resident,
                &topology,
                1,
                4,
                &atlas_triangles,
                &[0; 3],
                &[0; 3],
                &[0; 3],
                0,
                0,
                false,
                true,
                None,
                &mut groups,
            )
            .unwrap();
        // Corrupt only the component certificate after planning. The complete
        // overlay remains staged and must become the physical fallback.
        runtime.component_shadow.plan.component_faces.clear();
        runtime
            .stage_pending_overlay(
                0,
                &baseline_groups,
                &[Some(resident); 3],
                &[[1; 3]; 3],
                &[0; 3],
                &[0; 3],
                &[0; 3],
                resident,
                &atlas_triangles,
            )
            .unwrap();
        runtime.commit_publication();
        let fallback = runtime.snapshot();
        assert_eq!(fallback.component_publication_state, "complete-fallback");
        assert_eq!(fallback.component_publication_installs, 3);
        assert_eq!(fallback.component_publication_fallbacks, 1);
        assert!(fallback.component_publication_last_error.is_some());
        assert_eq!(fallback.component_authority_revocations, 1);
        assert!(fallback.component_authority_revoked);
        assert!(runtime.component_shadow.certificate.is_none());
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
                    priority_candidates: 0,
                    selected_max_priority: 0,
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
                0,
                0,
                false,
                false,
                None,
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
