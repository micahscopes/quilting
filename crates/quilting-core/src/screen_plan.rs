//! Bounded scene plans for metric-adaptive QB patch restriction.
//!
//! A live renderer may adapt a bounded set of selected source faces while
//! retaining one root leaf for every other face. Keeping this assembly
//! backend-neutral makes the resulting leaf/request stream reusable by WebGL2
//! and WebGPU and prevents browser state from becoming a second geometry
//! authority.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};

use crate::patch::{QBPatchDomain, QBTriPatch};
use crate::screen_leaf_lod::{ScreenLeafLodError, ScreenMeshLeafTopology, ScreenMeshTopologyCache};
use crate::screen_partition::{
    partition_screen_patch, ScreenPartitionError, ScreenPartitionPolicy, ScreenPatchLeafId,
};

#[derive(Clone, Copy, Debug)]
pub struct PickedScreenMeshPlanRequest<'a> {
    pub selected_face: u32,
    pub transformed_patch: &'a QBTriPatch,
    pub view_projection: &'a [f64; 16],
    pub viewport: [f64; 2],
    /// Existing screen-attenuation floor. Each leaf recomputes the greatest
    /// power-of-two LoD that fits this local raster capacity, allowing a root
    /// request driven by one high-dilation region to demote elsewhere.
    pub min_px_per_segment: f64,
    /// Optional tessellation-density ceiling, in pixels per local atlas edge
    /// segment. This supplies a local quality demand inside the floor's
    /// capacity; a power-of-two band gap is reported as metric saturation.
    pub max_px_per_segment: f64,
    pub policy: ScreenPartitionPolicy,
    pub max_atlas_lod: u32,
    /// Keep leaves classified as fully faded/outside the sampled view at
    /// their inherited source density. Live renderers use this so camera
    /// motion between asynchronous classifications cannot expose a hole; a
    /// frozen diagnostic/export may omit them to measure the exact view.
    pub retain_culled_leaves: bool,
    /// Scene-wide instance bound after replacing the selected root.
    pub max_total_leaves: usize,
    /// Raw, unpromoted source-face requests in logical A/B/C edge order.
    pub source_requested_lods: &'a [[u32; 3]],
}

#[derive(Clone, Copy, Debug)]
pub struct SelectedScreenPatch<'a> {
    pub source_face: u32,
    pub transformed_patch: &'a QBTriPatch,
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveScreenMeshPlanRequest<'a> {
    /// Distinct source faces to replace with metric-adaptive dyadic leaves.
    pub selected_patches: &'a [SelectedScreenPatch<'a>],
    pub view_projection: &'a [f64; 16],
    pub viewport: [f64; 2],
    pub min_px_per_segment: f64,
    pub max_px_per_segment: f64,
    pub policy: ScreenPartitionPolicy,
    pub max_atlas_lod: u32,
    pub retain_culled_leaves: bool,
    /// Conservative scene-wide bound on the sum of per-face partition
    /// frontier capacities before culled leaves are filtered from the output.
    pub max_partition_leaves: usize,
    pub max_total_leaves: usize,
    pub source_requested_lods: &'a [[u32; 3]],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PickedScreenMeshPlanDiagnostic {
    pub source_faces: u32,
    pub selected_leaves: u32,
    pub total_leaves: u32,
    pub split_nodes: u32,
    pub max_depth_reached: u8,
    pub unmet_leaves: u32,
    pub saturated_metric_leaves: u32,
    pub omitted_culled_leaves: u32,
    pub boundary_fallback_faces: u32,
}

#[derive(Clone, Debug)]
pub struct PickedScreenMeshPlan {
    pub selected_face: u32,
    pub leaves: Vec<ScreenMeshLeafTopology>,
    pub requested_lods: Vec<[u32; 3]>,
    pub diagnostic: PickedScreenMeshPlanDiagnostic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveScreenMeshPlanDiagnostic {
    pub source_faces: u32,
    pub selected_faces: u32,
    pub selected_leaves: u32,
    pub total_leaves: u32,
    pub split_nodes: u32,
    pub max_depth_reached: u8,
    pub unmet_leaves: u32,
    pub saturated_metric_leaves: u32,
    pub omitted_culled_leaves: u32,
    /// Selected faces retained as source roots because a drawable camera/fade
    /// boundary remained unresolved within the bounded partition policy.
    pub boundary_fallback_faces: u32,
}

#[derive(Clone, Debug, Default)]
pub struct AdaptiveScreenMeshPlan {
    pub selected_faces: Vec<u32>,
    pub leaves: Vec<ScreenMeshLeafTopology>,
    pub requested_lods: Vec<[u32; 3]>,
    pub diagnostic: AdaptiveScreenMeshPlanDiagnostic,
}

impl AdaptiveScreenMeshPlan {
    fn clear_retain_capacity(&mut self) {
        self.selected_faces.clear();
        self.leaves.clear();
        self.requested_lods.clear();
        self.diagnostic = AdaptiveScreenMeshPlanDiagnostic::default();
    }
}

/// One independently recomputable adaptive plan plus the exact welded-source
/// components it covers. The mesh plan emits only these source faces, while
/// its diagnostic and leaf budget continue to describe the composed scene
/// including retained roots outside the closure.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveScreenComponentPlan {
    pub component_faces: Vec<u32>,
    pub mesh: AdaptiveScreenMeshPlan,
}

impl AdaptiveScreenComponentPlan {
    fn clear_retain_capacity(&mut self) {
        self.component_faces.clear();
        self.mesh.clear_retain_capacity();
    }
}

/// One source-ordered root face considered for current-view adaptation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveScreenFaceCandidate {
    pub source_face: u32,
    pub visible: bool,
    /// Quantized within-patch screen-metric variation / pole-proximity hint.
    /// This ranks where subdivision is useful; root atlas cost remains the
    /// secondary workload priority.
    pub screen_metric_priority: u8,
    pub requested_lods: [u32; 3],
    /// Exact triangle count of the candidate's current root atlas patch.
    pub root_triangles: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveScreenFaceSelectionPolicy {
    pub max_faces: usize,
    /// The same partition policy that will be passed to the mesh planner.
    pub partition_policy: ScreenPartitionPolicy,
    pub max_partition_leaves: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveScreenFaceSelectionDiagnostic {
    pub examined_faces: u64,
    pub visible_faces: u64,
    pub selected_faces: u64,
    pub partition_face_capacity: u64,
    pub omitted_by_capacity: u64,
    pub priority_candidates: u64,
    pub selected_max_priority: u8,
    pub selected_root_triangles: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveScreenFaceSelection {
    /// Selected stable source identities in canonical ascending order.
    pub faces: Vec<u32>,
    pub diagnostic: AdaptiveScreenFaceSelectionDiagnostic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveScreenFaceSelectionError {
    InvalidPolicy,
    NonCanonicalSourceOrder { previous: u32, current: u32 },
    InvalidRequestedLod { face: u32, edge: usize },
    MissingRootTriangles { face: u32 },
}

impl std::fmt::Display for AdaptiveScreenFaceSelectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(formatter, "adaptive face selection policy is invalid"),
            Self::NonCanonicalSourceOrder { previous, current } => write!(
                formatter,
                "adaptive candidates are not strictly source ordered: {previous} then {current}",
            ),
            Self::InvalidRequestedLod { face, edge } => write!(
                formatter,
                "adaptive candidate face {face} edge {edge} has an invalid LoD",
            ),
            Self::MissingRootTriangles { face } => write!(
                formatter,
                "adaptive candidate face {face} has no resident root triangle count",
            ),
        }
    }
}

impl std::error::Error for AdaptiveScreenFaceSelectionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RankedAdaptiveScreenFace {
    source_face: u32,
    screen_metric_priority: u8,
    root_triangles: u64,
    maximum_lod: u32,
    lod_exponent_sum: u32,
}

impl Ord for RankedAdaptiveScreenFace {
    fn cmp(&self, other: &Self) -> Ordering {
        self.screen_metric_priority
            .cmp(&other.screen_metric_priority)
            .then_with(|| self.root_triangles.cmp(&other.root_triangles))
            .then_with(|| self.maximum_lod.cmp(&other.maximum_lod))
            .then_with(|| self.lod_exponent_sum.cmp(&other.lod_exponent_sum))
            // A lower stable source identity wins an otherwise exact tie.
            .then_with(|| other.source_face.cmp(&self.source_face))
    }
}

impl PartialOrd for RankedAdaptiveScreenFace {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Select the most distortion-sensitive currently visible roots with bounded
/// scratch, using current root cost to rank otherwise equal candidates.
///
/// Candidates must be strictly ordered by stable source identity. The
/// selector keeps only the best `k` entries in a min-heap, so a large scene
/// costs O(n log k) time and O(k) memory. The returned identities are sorted
/// back into source order for deterministic pose extraction and planning.
pub fn select_adaptive_screen_faces(
    candidates: impl IntoIterator<Item = AdaptiveScreenFaceCandidate>,
    policy: AdaptiveScreenFaceSelectionPolicy,
) -> Result<AdaptiveScreenFaceSelection, AdaptiveScreenFaceSelectionError> {
    if policy.max_faces == 0
        || policy.partition_policy.max_leaves == 0
        || policy.max_partition_leaves < policy.partition_policy.max_leaves
    {
        return Err(AdaptiveScreenFaceSelectionError::InvalidPolicy);
    }
    let partition_face_capacity =
        (policy.max_partition_leaves / policy.partition_policy.max_leaves).min(policy.max_faces);
    let mut selected = BinaryHeap::<Reverse<RankedAdaptiveScreenFace>>::new();
    let mut previous_face = None;
    let mut examined_faces = 0u64;
    let mut visible_faces = 0u64;
    let mut priority_candidates = 0u64;

    for candidate in candidates {
        examined_faces = examined_faces.saturating_add(1);
        if let Some(previous) = previous_face {
            if candidate.source_face <= previous {
                return Err(AdaptiveScreenFaceSelectionError::NonCanonicalSourceOrder {
                    previous,
                    current: candidate.source_face,
                });
            }
        }
        previous_face = Some(candidate.source_face);
        if !candidate.visible {
            continue;
        }
        visible_faces = visible_faces.saturating_add(1);
        priority_candidates = priority_candidates
            .saturating_add(u64::from(candidate.screen_metric_priority != 0));
        for (edge, lod) in candidate.requested_lods.into_iter().enumerate() {
            if !lod.is_power_of_two() {
                return Err(AdaptiveScreenFaceSelectionError::InvalidRequestedLod {
                    face: candidate.source_face,
                    edge,
                });
            }
        }
        if candidate.root_triangles == 0 {
            return Err(AdaptiveScreenFaceSelectionError::MissingRootTriangles {
                face: candidate.source_face,
            });
        }

        let ranked = RankedAdaptiveScreenFace {
            source_face: candidate.source_face,
            screen_metric_priority: candidate.screen_metric_priority,
            root_triangles: candidate.root_triangles,
            maximum_lod: candidate.requested_lods.into_iter().max().unwrap_or(1),
            lod_exponent_sum: candidate
                .requested_lods
                .into_iter()
                .map(u32::trailing_zeros)
                .sum(),
        };
        if selected.len() < partition_face_capacity {
            selected.push(Reverse(ranked));
        } else if selected.peek().is_some_and(|smallest| ranked > smallest.0) {
            selected.pop();
            selected.push(Reverse(ranked));
        }
    }

    let mut ranked = selected
        .into_iter()
        .map(|Reverse(candidate)| candidate)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|candidate| candidate.source_face);
    let selected_root_triangles = ranked.iter().fold(0u64, |total, candidate| {
        total.saturating_add(candidate.root_triangles)
    });
    let selected_max_priority = ranked
        .iter()
        .map(|candidate| candidate.screen_metric_priority)
        .max()
        .unwrap_or(0);
    let selected_faces = ranked.len() as u64;
    Ok(AdaptiveScreenFaceSelection {
        faces: ranked
            .into_iter()
            .map(|candidate| candidate.source_face)
            .collect(),
        diagnostic: AdaptiveScreenFaceSelectionDiagnostic {
            examined_faces,
            visible_faces,
            selected_faces,
            partition_face_capacity: partition_face_capacity as u64,
            omitted_by_capacity: visible_faces.saturating_sub(selected_faces),
            priority_candidates,
            selected_max_priority,
            selected_root_triangles,
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickedScreenMeshPlanError {
    EmptyScene,
    InvalidSelectedFace,
    DuplicateSelectedFace { face: u32 },
    InvalidPixelBand,
    ImpossiblePixelBand,
    InvalidAtlasLod,
    InvalidSourceLod { face: usize, edge: usize },
    InvalidLeafDomain,
    UnmetPartition { leaves: usize },
    UnresolvedMetric,
    PartitionBudgetExceeded { requested: usize, maximum: usize },
    LeafBudgetExceeded { requested: usize, maximum: usize },
    Partition(ScreenPartitionError),
    ComponentClosure(ScreenLeafLodError),
    CountOverflow,
}

/// Representation-neutral name for the compatibility error type retained by
/// the original one-face planner API.
pub type AdaptiveScreenMeshPlanError = PickedScreenMeshPlanError;

impl std::fmt::Display for PickedScreenMeshPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyScene => write!(formatter, "adaptive screen plan requires a source scene"),
            Self::InvalidSelectedFace => write!(formatter, "selected adaptive face is unavailable"),
            Self::DuplicateSelectedFace { face } => {
                write!(
                    formatter,
                    "adaptive face {face} was selected more than once"
                )
            }
            Self::InvalidPixelBand => {
                write!(
                    formatter,
                    "adaptive pixel floor and ceiling must be finite and positive"
                )
            }
            Self::ImpossiblePixelBand => write!(
                formatter,
                "adaptive pixel ceiling must be at least the attenuation floor",
            ),
            Self::InvalidAtlasLod => write!(formatter, "adaptive atlas cap is not a power of two"),
            Self::InvalidSourceLod { face, edge } => {
                write!(
                    formatter,
                    "source face {face} edge {edge} has an invalid LoD"
                )
            }
            Self::InvalidLeafDomain => {
                write!(formatter, "adaptive leaf has an invalid dyadic domain")
            }
            Self::UnmetPartition { leaves } => {
                write!(
                    formatter,
                    "adaptive partition left {leaves} unresolved leaves"
                )
            }
            Self::UnresolvedMetric => {
                write!(
                    formatter,
                    "drawable adaptive leaf has no finite screen metric"
                )
            }
            Self::PartitionBudgetExceeded { requested, maximum } => write!(
                formatter,
                "adaptive planning may examine {requested} leaves; work budget is {maximum}",
            ),
            Self::LeafBudgetExceeded { requested, maximum } => write!(
                formatter,
                "adaptive plan needs {requested} leaves; scene budget is {maximum}",
            ),
            Self::Partition(error) => write!(formatter, "adaptive partition failed: {error}"),
            Self::ComponentClosure(error) => {
                write!(formatter, "adaptive component closure failed: {error}")
            }
            Self::CountOverflow => {
                write!(formatter, "adaptive scene plan exceeds u32 identity limits")
            }
        }
    }
}

impl std::error::Error for PickedScreenMeshPlanError {}

impl From<ScreenPartitionError> for PickedScreenMeshPlanError {
    fn from(error: ScreenPartitionError) -> Self {
        Self::Partition(error)
    }
}

impl From<ScreenLeafLodError> for PickedScreenMeshPlanError {
    fn from(error: ScreenLeafLodError) -> Self {
        Self::ComponentClosure(error)
    }
}

fn validate_source_lods(
    source_requested_lods: &[[u32; 3]],
    max_atlas_lod: u32,
) -> Result<(), PickedScreenMeshPlanError> {
    for (face, lods) in source_requested_lods.iter().copied().enumerate() {
        validate_source_face_lods(face, lods, max_atlas_lod)?;
    }
    Ok(())
}

fn validate_source_face_lods(
    face: usize,
    lods: [u32; 3],
    max_atlas_lod: u32,
) -> Result<(), PickedScreenMeshPlanError> {
    for (edge, lod) in lods.into_iter().enumerate() {
        if !lod.is_power_of_two() || lod > max_atlas_lod {
            return Err(PickedScreenMeshPlanError::InvalidSourceLod { face, edge });
        }
    }
    Ok(())
}

/// Convert a root face's directional request into one dyadic leaf's local
/// A/B/C edge order.
///
/// A depth-`d` leaf spans `1 / 2^d` of a source-direction interval, so its
/// inherited local exponent is `source_exponent - d`, clamped at zero. The
/// central child rotates the local edge ordering; detecting the source
/// barycentric coordinate that is constant along each local edge handles that
/// permutation exactly.
pub fn inherited_source_edge_lods(
    leaf_id: ScreenPatchLeafId,
    domain: QBPatchDomain,
    source_edge_lods: [u32; 3],
) -> Result<[u32; 3], PickedScreenMeshPlanError> {
    if leaf_id.domain() != Some(domain) || source_edge_lods.iter().any(|lod| !lod.is_power_of_two())
    {
        return Err(PickedScreenMeshPlanError::InvalidLeafDomain);
    }
    let edge_corners = [(1usize, 2usize), (0, 2), (0, 1)];
    let mut inherited = [1u32; 3];
    for (local_edge, (first, second)) in edge_corners.into_iter().enumerate() {
        let a = domain.corners[first];
        let b = domain.corners[second];
        let source_edge = (0..3)
            .find(|axis| a[*axis] == b[*axis])
            .ok_or(PickedScreenMeshPlanError::InvalidLeafDomain)?;
        let source_exponent = source_edge_lods[source_edge].trailing_zeros();
        let local_exponent = source_exponent.saturating_sub(u32::from(leaf_id.depth));
        inherited[local_edge] = 1u32 << local_exponent;
    }
    Ok(inherited)
}

/// Build a bounded current-view frontier that adapts one or more selected faces
/// and retains every other source face as one root leaf. Selected leaves proven
/// fully faded or outside the supplied static view are either omitted for a
/// frozen-view measurement or retained at inherited density for a live view.
///
/// The output is raw request state. Shared-edge promotion and within-leaf
/// grading are intentionally deferred to [`crate::screen_leaf_lod`] so every
/// backend uses one reconciliation implementation and can demote cleanly.
/// A renderer must still reject reconciliation overflow, missing atlas keys,
/// and an excessive reconciled vertex/triangle/byte workload before swapping
/// this plan into live residency.
pub fn plan_adaptive_screen_mesh(
    request: AdaptiveScreenMeshPlanRequest<'_>,
) -> Result<AdaptiveScreenMeshPlan, AdaptiveScreenMeshPlanError> {
    let mut output = AdaptiveScreenMeshPlan::default();
    plan_adaptive_screen_mesh_into(request, &mut output)?;
    Ok(output)
}

/// Rebuild a bounded adaptive plan while retaining the caller's allocation.
///
/// On failure the output is empty, so a caller cannot accidentally publish a
/// preceding or partially rebuilt plan. Its vector capacities remain
/// available for the next measured pose.
pub fn plan_adaptive_screen_mesh_into(
    request: AdaptiveScreenMeshPlanRequest<'_>,
    output: &mut AdaptiveScreenMeshPlan,
) -> Result<(), AdaptiveScreenMeshPlanError> {
    output.clear_retain_capacity();
    let result = plan_adaptive_screen_mesh_reusing(request, None, output);
    if result.is_err() {
        output.clear_retain_capacity();
    }
    result
}

/// Build only the exact welded-source components touched by the selected
/// patches. The topology cache supplies the certificate; callers cannot pass
/// an arbitrary halo that might omit a shared edge or corner-density peer.
///
/// Unaffected source roots are not copied into `mesh.leaves`, but remain part
/// of the scene-wide leaf budget and diagnostic. On failure both retained
/// output allocations are cleared without exposing a partial closure or plan.
pub fn plan_adaptive_screen_components(
    request: AdaptiveScreenMeshPlanRequest<'_>,
    topology: &ScreenMeshTopologyCache,
    max_component_faces: usize,
) -> Result<AdaptiveScreenComponentPlan, AdaptiveScreenMeshPlanError> {
    let mut output = AdaptiveScreenComponentPlan::default();
    plan_adaptive_screen_components_into(request, topology, max_component_faces, &mut output)?;
    Ok(output)
}

/// Allocation-retaining form of [`plan_adaptive_screen_components`].
pub fn plan_adaptive_screen_components_into(
    request: AdaptiveScreenMeshPlanRequest<'_>,
    topology: &ScreenMeshTopologyCache,
    max_component_faces: usize,
    output: &mut AdaptiveScreenComponentPlan,
) -> Result<(), AdaptiveScreenMeshPlanError> {
    output.clear_retain_capacity();
    if topology.source_face_count() != request.source_requested_lods.len() {
        return Err(ScreenLeafLodError::LengthMismatch.into());
    }
    let closure = topology.collect_component_closure_from_faces(
        request
            .selected_patches
            .iter()
            .map(|selected| selected.source_face),
        max_component_faces,
        &mut output.component_faces,
    );
    if let Err(error) = closure {
        output.clear_retain_capacity();
        return Err(error.into());
    }
    if output.component_faces.is_empty() {
        output.clear_retain_capacity();
        return Err(PickedScreenMeshPlanError::EmptyScene);
    }
    let result =
        plan_adaptive_screen_mesh_reusing(request, Some(&output.component_faces), &mut output.mesh);
    if result.is_err() {
        output.clear_retain_capacity();
    }
    result
}

fn plan_adaptive_screen_mesh_reusing(
    request: AdaptiveScreenMeshPlanRequest<'_>,
    component_faces: Option<&[u32]>,
    output: &mut AdaptiveScreenMeshPlan,
) -> Result<(), AdaptiveScreenMeshPlanError> {
    if request.source_requested_lods.is_empty() {
        return Err(PickedScreenMeshPlanError::EmptyScene);
    }
    let mut selected_patches = BTreeMap::new();
    for selected in request.selected_patches {
        if selected.source_face as usize >= request.source_requested_lods.len() {
            return Err(PickedScreenMeshPlanError::InvalidSelectedFace);
        }
        if selected_patches
            .insert(selected.source_face, selected.transformed_patch)
            .is_some()
        {
            return Err(PickedScreenMeshPlanError::DuplicateSelectedFace {
                face: selected.source_face,
            });
        }
    }
    if !request.min_px_per_segment.is_finite()
        || request.min_px_per_segment <= 0.0
        || !request.max_px_per_segment.is_finite()
        || request.max_px_per_segment <= 0.0
    {
        return Err(PickedScreenMeshPlanError::InvalidPixelBand);
    }
    if request.max_px_per_segment < request.min_px_per_segment {
        return Err(PickedScreenMeshPlanError::ImpossiblePixelBand);
    }
    if !request.max_atlas_lod.is_power_of_two() {
        return Err(PickedScreenMeshPlanError::InvalidAtlasLod);
    }
    if let Some(component_faces) = component_faces {
        for &source_face in component_faces {
            let face = source_face as usize;
            let lods = request
                .source_requested_lods
                .get(face)
                .copied()
                .ok_or(PickedScreenMeshPlanError::InvalidSelectedFace)?;
            validate_source_face_lods(face, lods, request.max_atlas_lod)?;
        }
    } else {
        validate_source_lods(request.source_requested_lods, request.max_atlas_lod)?;
    }

    let maximum_partition_leaves = request
        .selected_patches
        .len()
        .checked_mul(request.policy.max_leaves)
        .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
    if request.max_partition_leaves == 0 || maximum_partition_leaves > request.max_partition_leaves
    {
        return Err(PickedScreenMeshPlanError::PartitionBudgetExceeded {
            requested: maximum_partition_leaves,
            maximum: request.max_partition_leaves,
        });
    }

    let fixed_roots = request
        .source_requested_lods
        .len()
        .checked_sub(selected_patches.len())
        .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
    let planned_source_faces = component_faces
        .map(<[u32]>::len)
        .unwrap_or(request.source_requested_lods.len());
    let planned_fixed_roots = planned_source_faces
        .checked_sub(selected_patches.len())
        .ok_or(PickedScreenMeshPlanError::CountOverflow)?;

    // Process selected patches in source order and write a viable plan
    // directly. The full metric partition is dropped before the next face is
    // examined, bounding peak memory to one per-face partition plus the
    // retained-output budget. If the output budget is crossed, keep counting
    // bounded partitions without retaining more records so the final error
    // reports the exact required leaf count.
    let AdaptiveScreenMeshPlan {
        selected_faces,
        leaves,
        requested_lods,
        diagnostic,
    } = output;
    selected_faces.reserve(selected_patches.len());
    let retained_capacity = request
        .max_total_leaves
        .min(planned_fixed_roots.saturating_add(request.max_partition_leaves));
    leaves.reserve(retained_capacity);
    requested_lods.reserve(retained_capacity);
    let mut output_viable =
        request.max_total_leaves != 0 && fixed_roots <= request.max_total_leaves;
    let mut selected_leaves = 0usize;
    let mut split_nodes = 0usize;
    let mut max_depth_reached = 0u8;
    let mut unmet_leaves = 0usize;
    let mut saturated_metric_leaves = 0u32;
    let mut omitted_culled_leaves = 0usize;
    let mut fallback_roots = 0usize;
    let mut boundary_fallback_faces = 0usize;

    for planned_face_index in 0..planned_source_faces {
        let source_face_u32 = component_faces
            .map_or_else(
                || u32::try_from(planned_face_index),
                |faces| Ok(faces[planned_face_index]),
            )
            .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
        let source_face = source_face_u32 as usize;
        let source_lods = request.source_requested_lods[source_face];
        let Some(transformed_patch) = selected_patches.get(&source_face_u32) else {
            if output_viable {
                leaves.push(ScreenMeshLeafTopology {
                    source_face: source_face_u32,
                    id: ScreenPatchLeafId::ROOT,
                    domain: QBPatchDomain::FULL,
                });
                requested_lods.push(source_lods);
            }
            continue;
        };
        let partition = partition_screen_patch(
            transformed_patch,
            request.view_projection,
            request.viewport,
            request.policy,
        )?;
        let unresolved_boundary_leaves = partition
            .leaves
            .iter()
            .filter(|leaf| leaf.status.is_drawable() && leaf.metric_diagnostic.is_none())
            .count();
        split_nodes = split_nodes
            .checked_add(partition.split_nodes)
            .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
        max_depth_reached = max_depth_reached.max(partition.max_depth_reached);
        unmet_leaves = unmet_leaves
            .checked_add(partition.unmet_leaves)
            .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
        if unresolved_boundary_leaves != 0 {
            boundary_fallback_faces = boundary_fallback_faces
                .checked_add(1)
                .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
            fallback_roots = fallback_roots
                .checked_add(1)
                .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
            let next_total_leaves = fixed_roots
                .checked_add(fallback_roots)
                .and_then(|roots| roots.checked_add(selected_leaves))
                .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
            output_viable &= next_total_leaves <= request.max_total_leaves;
            if output_viable {
                leaves.push(ScreenMeshLeafTopology {
                    source_face: source_face_u32,
                    id: ScreenPatchLeafId::ROOT,
                    domain: QBPatchDomain::FULL,
                });
                requested_lods.push(source_lods);
            }
            continue;
        }
        selected_faces.push(source_face_u32);

        let included_leaf_count = partition
            .leaves
            .iter()
            .filter(|leaf| request.retain_culled_leaves || leaf.status.is_drawable())
            .count();
        omitted_culled_leaves = omitted_culled_leaves
            .checked_add(partition.leaves.len().saturating_sub(included_leaf_count))
            .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
        let next_selected_leaves = selected_leaves
            .checked_add(included_leaf_count)
            .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
        let next_total_leaves = fixed_roots
            .checked_add(fallback_roots)
            .and_then(|roots| roots.checked_add(next_selected_leaves))
            .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
        output_viable &= next_total_leaves <= request.max_total_leaves;

        if output_viable {
            for leaf in partition
                .leaves
                .iter()
                .filter(|leaf| request.retain_culled_leaves || leaf.status.is_drawable())
            {
                let inherited =
                    inherited_source_edge_lods(leaf.id, leaf.restricted.domain, source_lods)?;
                let local_request = if leaf.status.is_drawable() {
                    let diagnostic = leaf
                        .metric_diagnostic
                        .ok_or(PickedScreenMeshPlanError::UnresolvedMetric)?;
                    let band = diagnostic
                        .resolve_subdivision_band(
                            inherited,
                            request.min_px_per_segment,
                            request.max_px_per_segment,
                            request.max_atlas_lod,
                        )
                        .ok_or(PickedScreenMeshPlanError::UnresolvedMetric)?;
                    if band.saturated {
                        // A discrete power-of-two level cannot satisfy both
                        // sides of this local pixel band. Preserve the existing
                        // pixel-floor authority (never add known subpixel
                        // density) and report the unsatisfied quality ceiling.
                        saturated_metric_leaves = saturated_metric_leaves.saturating_add(1);
                    }
                    band.requested
                } else {
                    // A live-view plan retains culled leaves because the render
                    // GPU can resurrect them before the asynchronous classifier.
                    // With no finite local metric, inherited density is the
                    // only safe self-describing standby.
                    inherited
                };
                leaves.push(ScreenMeshLeafTopology::from_leaf(source_face_u32, leaf));
                requested_lods.push(local_request);
            }
        }
        selected_leaves = next_selected_leaves;
    }

    let total_capacity = fixed_roots
        .checked_add(fallback_roots)
        .and_then(|roots| roots.checked_add(selected_leaves))
        .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
    if !output_viable {
        return Err(PickedScreenMeshPlanError::LeafBudgetExceeded {
            requested: total_capacity,
            maximum: request.max_total_leaves,
        });
    }

    let source_faces = u32::try_from(request.source_requested_lods.len())
        .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    let selected_face_count = u32::try_from(selected_faces.len())
        .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    let selected_leaf_count =
        u32::try_from(selected_leaves).map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    let total_leaves =
        u32::try_from(total_capacity).map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    *diagnostic = AdaptiveScreenMeshPlanDiagnostic {
        source_faces,
        selected_faces: selected_face_count,
        selected_leaves: selected_leaf_count,
        total_leaves,
        split_nodes: u32::try_from(split_nodes)
            .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
        max_depth_reached,
        unmet_leaves: u32::try_from(unmet_leaves)
            .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
        saturated_metric_leaves,
        omitted_culled_leaves: u32::try_from(omitted_culled_leaves)
            .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
        boundary_fallback_faces: u32::try_from(boundary_fallback_faces)
            .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
    };
    Ok(())
}

/// Compatibility wrapper for callers that adapt exactly one picked face.
pub fn plan_picked_screen_mesh(
    request: PickedScreenMeshPlanRequest<'_>,
) -> Result<PickedScreenMeshPlan, PickedScreenMeshPlanError> {
    let selected = [SelectedScreenPatch {
        source_face: request.selected_face,
        transformed_patch: request.transformed_patch,
    }];
    let plan = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
        selected_patches: &selected,
        view_projection: request.view_projection,
        viewport: request.viewport,
        min_px_per_segment: request.min_px_per_segment,
        max_px_per_segment: request.max_px_per_segment,
        policy: request.policy,
        max_atlas_lod: request.max_atlas_lod,
        retain_culled_leaves: request.retain_culled_leaves,
        max_partition_leaves: request.policy.max_leaves.max(1),
        max_total_leaves: request.max_total_leaves,
        source_requested_lods: request.source_requested_lods,
    })?;
    Ok(PickedScreenMeshPlan {
        selected_face: request.selected_face,
        leaves: plan.leaves,
        requested_lods: plan.requested_lods,
        diagnostic: PickedScreenMeshPlanDiagnostic {
            source_faces: plan.diagnostic.source_faces,
            selected_leaves: plan.diagnostic.selected_leaves,
            total_leaves: plan.diagnostic.total_leaves,
            split_nodes: plan.diagnostic.split_nodes,
            max_depth_reached: plan.diagnostic.max_depth_reached,
            unmet_leaves: plan.diagnostic.unmet_leaves,
            saturated_metric_leaves: plan.diagnostic.saturated_metric_leaves,
            omitted_culled_leaves: plan.diagnostic.omitted_culled_leaves,
            boundary_fallback_faces: plan.diagnostic.boundary_fallback_faces,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::{Mobius, Quat};

    #[test]
    fn face_selector_keeps_the_most_expensive_visible_roots() {
        let selection = select_adaptive_screen_faces(
            [
                AdaptiveScreenFaceCandidate {
                    source_face: 0,
                    visible: true,
                    screen_metric_priority: 0,
                    requested_lods: [2, 2, 2],
                    root_triangles: 8,
                },
                AdaptiveScreenFaceCandidate {
                    source_face: 1,
                    visible: false,
                    screen_metric_priority: 0,
                    requested_lods: [64, 64, 64],
                    root_triangles: 8_192,
                },
                AdaptiveScreenFaceCandidate {
                    source_face: 2,
                    visible: true,
                    screen_metric_priority: 0,
                    requested_lods: [8, 8, 8],
                    root_triangles: 128,
                },
                AdaptiveScreenFaceCandidate {
                    source_face: 3,
                    visible: true,
                    screen_metric_priority: 0,
                    requested_lods: [4, 4, 4],
                    root_triangles: 32,
                },
            ],
            AdaptiveScreenFaceSelectionPolicy {
                max_faces: 2,
                partition_policy: ScreenPartitionPolicy {
                    max_leaves: 8,
                    ..ScreenPartitionPolicy::default()
                },
                max_partition_leaves: 32,
            },
        )
        .unwrap();

        assert_eq!(selection.faces, [2, 3]);
        assert_eq!(selection.diagnostic.examined_faces, 4);
        assert_eq!(selection.diagnostic.visible_faces, 3);
        assert_eq!(selection.diagnostic.selected_faces, 2);
        assert_eq!(selection.diagnostic.partition_face_capacity, 2);
        assert_eq!(selection.diagnostic.omitted_by_capacity, 1);
        assert_eq!(selection.diagnostic.selected_root_triangles, 160);
    }

    #[test]
    fn face_selector_prefers_screen_metric_risk_over_equal_root_cost_ordering() {
        let selection = select_adaptive_screen_faces(
            [
                AdaptiveScreenFaceCandidate {
                    source_face: 0,
                    visible: true,
                    screen_metric_priority: 0,
                    requested_lods: [64; 3],
                    root_triangles: 8_192,
                },
                AdaptiveScreenFaceCandidate {
                    source_face: 1,
                    visible: true,
                    screen_metric_priority: 200,
                    requested_lods: [2; 3],
                    root_triangles: 8,
                },
            ],
            AdaptiveScreenFaceSelectionPolicy {
                max_faces: 1,
                partition_policy: ScreenPartitionPolicy {
                    max_leaves: 8,
                    ..ScreenPartitionPolicy::default()
                },
                max_partition_leaves: 8,
            },
        )
        .unwrap();

        assert_eq!(selection.faces, [1]);
        assert_eq!(selection.diagnostic.priority_candidates, 1);
        assert_eq!(selection.diagnostic.selected_max_priority, 200);
        assert_eq!(selection.diagnostic.selected_root_triangles, 8);
    }

    #[test]
    fn face_selector_uses_partition_capacity_and_stable_ties() {
        let candidates = (0..4).map(|source_face| AdaptiveScreenFaceCandidate {
            source_face,
            visible: true,
            screen_metric_priority: 0,
            requested_lods: [4, 4, 4],
            root_triangles: 32,
        });
        let selection = select_adaptive_screen_faces(
            candidates,
            AdaptiveScreenFaceSelectionPolicy {
                max_faces: 4,
                partition_policy: ScreenPartitionPolicy {
                    max_leaves: 8,
                    ..ScreenPartitionPolicy::default()
                },
                max_partition_leaves: 16,
            },
        )
        .unwrap();

        assert_eq!(selection.faces, [0, 1]);
        assert_eq!(selection.diagnostic.partition_face_capacity, 2);
        assert_eq!(selection.diagnostic.omitted_by_capacity, 2);
    }

    #[test]
    fn face_selector_rejects_noncanonical_or_invalid_visible_candidates() {
        let policy = AdaptiveScreenFaceSelectionPolicy {
            max_faces: 1,
            partition_policy: ScreenPartitionPolicy {
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_partition_leaves: 4,
        };
        let duplicate = select_adaptive_screen_faces(
            [
                AdaptiveScreenFaceCandidate {
                    source_face: 1,
                    visible: false,
                    screen_metric_priority: 0,
                    requested_lods: [1; 3],
                    root_triangles: 1,
                },
                AdaptiveScreenFaceCandidate {
                    source_face: 1,
                    visible: true,
                    screen_metric_priority: 0,
                    requested_lods: [1; 3],
                    root_triangles: 1,
                },
            ],
            policy,
        )
        .unwrap_err();
        assert_eq!(
            duplicate,
            AdaptiveScreenFaceSelectionError::NonCanonicalSourceOrder {
                previous: 1,
                current: 1,
            },
        );

        let invalid = select_adaptive_screen_faces(
            [AdaptiveScreenFaceCandidate {
                source_face: 0,
                visible: true,
                screen_metric_priority: 0,
                requested_lods: [1, 3, 1],
                root_triangles: 1,
            }],
            policy,
        )
        .unwrap_err();
        assert_eq!(
            invalid,
            AdaptiveScreenFaceSelectionError::InvalidRequestedLod { face: 0, edge: 1 },
        );
    }

    #[test]
    fn face_selector_can_return_an_empty_bounded_view() {
        let selection = select_adaptive_screen_faces(
            [AdaptiveScreenFaceCandidate {
                source_face: 0,
                visible: false,
                screen_metric_priority: 0,
                requested_lods: [1; 3],
                root_triangles: 0,
            }],
            AdaptiveScreenFaceSelectionPolicy {
                max_faces: 1,
                partition_policy: ScreenPartitionPolicy {
                    max_leaves: 4,
                    ..ScreenPartitionPolicy::default()
                },
                max_partition_leaves: 4,
            },
        )
        .unwrap();
        assert!(selection.faces.is_empty());
        assert_eq!(selection.diagnostic.visible_faces, 0);
        assert_eq!(selection.diagnostic.omitted_by_capacity, 0);
    }

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

    #[test]
    fn inherited_density_scales_and_rotates_with_the_dyadic_child() {
        let children = QBPatchDomain::FULL.quarter();
        let corner = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let centre = ScreenPatchLeafId::ROOT.child(3).unwrap();

        assert_eq!(
            inherited_source_edge_lods(corner, children[0], [2, 4, 8]).unwrap(),
            [1, 2, 4],
        );
        assert_eq!(
            inherited_source_edge_lods(centre, children[3], [2, 4, 8]).unwrap(),
            [4, 1, 2],
        );
    }

    #[test]
    fn empty_automatic_selection_is_an_all_root_plan() {
        let plan = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
            selected_patches: &[],
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 1,
            max_total_leaves: 2,
            source_requested_lods: &[[2, 4, 8], [8, 4, 2]],
        })
        .unwrap();

        assert!(plan.selected_faces.is_empty());
        assert_eq!(plan.diagnostic.selected_faces, 0);
        assert_eq!(plan.diagnostic.selected_leaves, 0);
        assert_eq!(plan.diagnostic.total_leaves, 2);
        assert!(plan
            .leaves
            .iter()
            .all(|leaf| leaf.id == ScreenPatchLeafId::ROOT));
        assert_eq!(plan.requested_lods, [[2, 4, 8], [8, 4, 2]]);
    }

    #[test]
    fn adaptive_plan_into_reuses_storage_and_clears_failed_output() {
        let source_lods = [[2, 4, 8]; 32];
        let request = AdaptiveScreenMeshPlanRequest {
            selected_patches: &[],
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 1,
            max_total_leaves: source_lods.len(),
            source_requested_lods: &source_lods,
        };
        let mut output = AdaptiveScreenMeshPlan::default();
        plan_adaptive_screen_mesh_into(request, &mut output).unwrap();
        let capacities = (
            output.selected_faces.capacity(),
            output.leaves.capacity(),
            output.requested_lods.capacity(),
        );

        plan_adaptive_screen_mesh_into(request, &mut output).unwrap();
        assert_eq!(output.leaves.len(), source_lods.len());
        assert_eq!(
            capacities,
            (
                output.selected_faces.capacity(),
                output.leaves.capacity(),
                output.requested_lods.capacity(),
            ),
        );

        let error = plan_adaptive_screen_mesh_into(
            AdaptiveScreenMeshPlanRequest {
                max_atlas_lod: 3,
                ..request
            },
            &mut output,
        )
        .unwrap_err();
        assert_eq!(error, PickedScreenMeshPlanError::InvalidAtlasLod);
        assert!(output.selected_faces.is_empty());
        assert!(output.leaves.is_empty());
        assert!(output.requested_lods.is_empty());
        assert_eq!(
            output.diagnostic,
            AdaptiveScreenMeshPlanDiagnostic::default(),
        );
        assert_eq!(
            capacities,
            (
                output.selected_faces.capacity(),
                output.leaves.capacity(),
                output.requested_lods.capacity(),
            ),
        );
    }

    #[test]
    fn picked_plan_replaces_only_the_selected_root() {
        let patch = QBTriPatch::flat([-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]);
        let policy = ScreenPartitionPolicy {
            min_depth: 1,
            max_depth: 1,
            max_leaves: 4,
            ..ScreenPartitionPolicy::default()
        };
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 1,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 10_000.0,
            policy,
            max_atlas_lod: 64,
            retain_culled_leaves: false,
            max_total_leaves: 16,
            source_requested_lods: &[[2, 2, 2], [2, 4, 8], [4, 4, 4]],
        })
        .unwrap();

        assert_eq!(plan.diagnostic.source_faces, 3);
        assert_eq!(plan.diagnostic.selected_leaves, 4);
        assert_eq!(plan.diagnostic.total_leaves, 6);
        assert_eq!(plan.leaves[0].source_face, 0);
        assert_eq!(plan.leaves[0].id, ScreenPatchLeafId::ROOT);
        assert_eq!(plan.requested_lods[0], [2, 2, 2]);
        assert!(plan.leaves[1..5]
            .iter()
            .all(|leaf| leaf.source_face == 1 && leaf.id.depth == 1));
        assert_eq!(plan.requested_lods[1], [1, 2, 4]);
        assert_eq!(plan.requested_lods[4], [4, 1, 2]);
        assert_eq!(plan.leaves[5].source_face, 2);
        assert_eq!(plan.requested_lods[5], [4, 4, 4]);
    }

    #[test]
    fn local_pixel_capacity_demotes_low_scale_children_after_root_demand() {
        let positions = [
            [-0.5, -0.5, -2.0],
            [0.5, -0.5, -2.0],
            [-0.5, 0.5, -8.0],
        ];
        let patch = QBTriPatch::flat(positions[0], positions[1], positions[2]);
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &PERSPECTIVE,
            viewport: [1600.0, 900.0],
            min_px_per_segment: 32.0,
            max_px_per_segment: 10_000.0,
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_stretch_ratio: 1.0e12,
                max_area_ratio: 1.0e12,
                ignore_below_px: 0.0,
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_total_leaves: 4,
            source_requested_lods: &[[64; 3]],
        })
        .unwrap();
        assert_eq!(plan.leaves.len(), 4);
        assert!(plan
            .requested_lods
            .iter()
            .flatten()
            .any(|lod| *lod < 32));

        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2]],
        );
        let topology =
            crate::screen_leaf_lod::ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let frontier =
            crate::screen_leaf_lod::ScreenMeshLeafFrontier::build(&plan.leaves, &topology).unwrap();
        let reconciled = frontier
            .reconcile_lods(&plan.requested_lods, 4, 64)
            .unwrap();
        let absolute_maxima = reconciled
            .resident
            .iter()
            .zip(&plan.leaves)
            .map(|(lods, leaf)| {
                lods
                    .iter()
                    .map(|lod| lod.trailing_zeros() + u32::from(leaf.id.depth))
                    .max()
                    .unwrap()
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert!(absolute_maxima.len() > 1);
        assert!(absolute_maxima.iter().all(|exponent| *exponent < 6));
    }

    #[test]
    fn sphere_reflection_frontier_localizes_the_root_peak_after_reconciliation() {
        let positions = [
            [-1.0, -1.0, -3.0],
            [1.0, -1.0, -3.0],
            [-1.0, 1.0, -3.0],
        ];
        let patch = QBTriPatch::flat(positions[0], positions[1], positions[2]).transform(
            &Mobius::sphere_reflection(Quat::from_point(-0.2, -0.1, -2.8), 1.0),
        );
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &PERSPECTIVE,
            viewport: [1600.0, 900.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 32.0,
            policy: ScreenPartitionPolicy {
                max_depth: 5,
                max_leaves: 64,
                ignore_below_px: 0.0,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_total_leaves: 64,
            source_requested_lods: &[[64; 3]],
        })
        .unwrap();
        assert!(plan.leaves.len() > 1);
        assert!(plan.leaves.len() <= 64);
        assert!(plan.diagnostic.unmet_leaves > 0);

        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2]],
        );
        let topology =
            crate::screen_leaf_lod::ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let frontier =
            crate::screen_leaf_lod::ScreenMeshLeafFrontier::build(&plan.leaves, &topology).unwrap();
        let reconciled = frontier
            .reconcile_lods(&plan.requested_lods, 4, 64)
            .unwrap();
        let absolute_exponents = reconciled
            .resident
            .iter()
            .zip(&plan.leaves)
            .flat_map(|(lods, leaf)| {
                lods
                    .map(|lod| lod.trailing_zeros() + u32::from(leaf.id.depth))
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert!(absolute_exponents.len() > 1);
        assert!(absolute_exponents.iter().any(|exponent| *exponent < 6));
    }

    #[test]
    fn unresolved_camera_boundary_falls_back_to_its_source_root() {
        let patch = QBTriPatch::flat(
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [0.0, 0.4, 1.0],
        );
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &PERSPECTIVE,
            viewport: [1600.0, 900.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 32.0,
            policy: ScreenPartitionPolicy {
                max_depth: 2,
                max_leaves: 16,
                ignore_below_px: 0.0,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_total_leaves: 16,
            source_requested_lods: &[[64; 3]],
        })
        .unwrap();
        assert_eq!(plan.leaves.len(), 1);
        assert_eq!(plan.leaves[0].id, ScreenPatchLeafId::ROOT);
        assert_eq!(plan.requested_lods, [[64; 3]]);
        assert_eq!(plan.diagnostic.selected_leaves, 0);
        assert_eq!(plan.diagnostic.boundary_fallback_faces, 1);
        assert!(plan.diagnostic.unmet_leaves > 0);
    }

    #[test]
    fn one_boundary_face_does_not_discard_other_adaptive_faces() {
        let boundary = QBTriPatch::flat(
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [0.0, 0.4, 1.0],
        );
        let finite = QBTriPatch::flat(
            [-0.3, -0.3, -2.0],
            [0.3, -0.3, -2.0],
            [-0.3, 0.3, -2.0],
        );
        let selected = [
            SelectedScreenPatch {
                source_face: 1,
                transformed_patch: &finite,
            },
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &boundary,
            },
        ];
        let plan = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
            selected_patches: &selected,
            view_projection: &PERSPECTIVE,
            viewport: [1600.0, 900.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 32.0,
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_leaves: 4,
                ignore_below_px: 0.0,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 8,
            max_total_leaves: 5,
            source_requested_lods: &[[64; 3], [8; 3]],
        })
        .unwrap();

        assert_eq!(plan.selected_faces, [1]);
        assert_eq!(plan.diagnostic.selected_faces, 1);
        assert_eq!(plan.diagnostic.boundary_fallback_faces, 1);
        assert_eq!(plan.diagnostic.selected_leaves, 4);
        assert_eq!(plan.diagnostic.total_leaves, 5);
        assert_eq!(plan.leaves[0].source_face, 0);
        assert_eq!(plan.leaves[0].id, ScreenPatchLeafId::ROOT);
        assert!(plan.leaves[1..]
            .iter()
            .all(|leaf| leaf.source_face == 1 && leaf.id.depth == 1));
    }

    #[test]
    fn unrepresentable_local_pixel_band_preserves_floor_and_reports_saturation() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [200.0, 200.0],
            min_px_per_segment: 60.0,
            max_px_per_segment: 90.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_total_leaves: 1,
            source_requested_lods: &[[64; 3]],
        })
        .unwrap();

        // The two 100px cardinal edges have no power-of-two subdivision
        // count satisfying both 60px <= segment <= 90px. Keep the existing
        // floor authoritative, report the ceiling miss, and avoid silently
        // manufacturing subpixel density.
        assert_eq!(plan.requested_lods, [[2, 1, 1]]);
        assert_eq!(plan.diagnostic.saturated_metric_leaves, 1);
    }

    #[test]
    fn multi_face_plan_builds_one_weldable_frontier() {
        let positions = [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let first = QBTriPatch::flat(positions[0], positions[1], positions[2]);
        let second = QBTriPatch::flat(positions[2], positions[1], positions[3]);
        let selected = [
            SelectedScreenPatch {
                source_face: 1,
                transformed_patch: &second,
            },
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &first,
            },
        ];
        let plan = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
            selected_patches: &selected,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 10_000.0,
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 8,
            max_total_leaves: 8,
            source_requested_lods: &[[2, 4, 8], [8, 4, 2]],
        })
        .unwrap();

        assert_eq!(plan.selected_faces, [0, 1]);
        assert_eq!(plan.diagnostic.source_faces, 2);
        assert_eq!(plan.diagnostic.selected_faces, 2);
        assert_eq!(plan.diagnostic.selected_leaves, 8);
        assert_eq!(plan.diagnostic.total_leaves, 8);
        assert!(plan.leaves[..4]
            .iter()
            .all(|leaf| leaf.source_face == 0 && leaf.id.depth == 1));
        assert!(plan.leaves[4..]
            .iter()
            .all(|leaf| leaf.source_face == 1 && leaf.id.depth == 1));

        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [2, 1, 3]],
        );
        let topology =
            crate::screen_leaf_lod::ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let frontier =
            crate::screen_leaf_lod::ScreenMeshLeafFrontier::build(&plan.leaves, &topology).unwrap();
        let reconciled = frontier
            .reconcile_lods(&plan.requested_lods, 4, 64)
            .unwrap();
        assert_eq!(reconciled.resident.len(), 8);
    }

    #[test]
    fn component_plan_matches_complete_plan_over_its_certified_closure() {
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
        let triangles = [[0, 1, 2], [2, 3, 4], [5, 6, 7]];
        let source =
            quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(&positions, &triangles);
        let topology =
            crate::screen_leaf_lod::ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let patch = QBTriPatch::flat(positions[0], positions[1], positions[2]);
        let selected = [SelectedScreenPatch {
            source_face: 0,
            transformed_patch: &patch,
        }];
        let source_lods = [[2, 4, 8], [8, 2, 4], [32, 16, 8]];
        let request = AdaptiveScreenMeshPlanRequest {
            selected_patches: &selected,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 10_000.0,
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 4,
            max_total_leaves: 6,
            source_requested_lods: &source_lods,
        };

        let complete = plan_adaptive_screen_mesh(request).unwrap();
        let component = plan_adaptive_screen_components(request, &topology, 2).unwrap();
        assert_eq!(component.component_faces, [0, 1]);
        assert_eq!(component.mesh.diagnostic, complete.diagnostic);
        assert_eq!(component.mesh.diagnostic.source_faces, 3);
        assert_eq!(component.mesh.diagnostic.total_leaves, 6);
        assert_eq!(component.mesh.leaves.len(), 5);

        let complete_frontier =
            crate::screen_leaf_lod::ScreenMeshLeafFrontier::build(&complete.leaves, &topology)
                .unwrap();
        let component_frontier = crate::screen_leaf_lod::ScreenMeshLeafFrontier::build(
            &component.mesh.leaves,
            &topology,
        )
        .unwrap();
        let complete_resident = complete_frontier
            .reconcile_lods(&complete.requested_lods, 4, 64)
            .unwrap();
        let component_resident = component_frontier
            .reconcile_lods(&component.mesh.requested_lods, 4, 64)
            .unwrap();
        let complete_vertices = complete_frontier
            .rebuild_vertex_lods(&complete_resident.resident)
            .unwrap();
        let component_vertices = component_frontier
            .rebuild_vertex_lods(&component_resident.resident)
            .unwrap();
        let complete_component = complete
            .leaves
            .iter()
            .copied()
            .zip(complete.requested_lods.iter().copied())
            .zip(complete_resident.resident.iter().copied())
            .zip(complete_vertices.iter().copied())
            .filter(|(((leaf, _), _), _)| {
                component
                    .component_faces
                    .binary_search(&leaf.source_face)
                    .is_ok()
            })
            .collect::<Vec<_>>();
        let local_component = component
            .mesh
            .leaves
            .iter()
            .copied()
            .zip(component.mesh.requested_lods.iter().copied())
            .zip(component_resident.resident.iter().copied())
            .zip(component_vertices.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(local_component, complete_component);
    }

    #[test]
    fn component_plan_fails_closed_before_exceeding_its_source_budget() {
        let positions = [
            [-0.8, -0.6, 0.0],
            [-0.2, -0.6, 0.0],
            [-0.5, 0.0, 0.0],
            [-0.8, 0.6, 0.0],
            [-0.2, 0.6, 0.0],
        ];
        let source = quilting_mesh::HalfEdgeMesh::from_triangles_welded_exact(
            &positions,
            &[[0, 1, 2], [2, 3, 4]],
        );
        let topology =
            crate::screen_leaf_lod::ScreenMeshTopologyCache::from_half_edge_mesh(&source).unwrap();
        let patch = QBTriPatch::flat(positions[0], positions[1], positions[2]);
        let selected = [SelectedScreenPatch {
            source_face: 0,
            transformed_patch: &patch,
        }];
        let request = AdaptiveScreenMeshPlanRequest {
            selected_patches: &selected,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 256,
            max_total_leaves: 512,
            source_requested_lods: &[[1; 3]; 2],
        };
        let mut output = AdaptiveScreenComponentPlan {
            component_faces: vec![99],
            mesh: AdaptiveScreenMeshPlan {
                selected_faces: vec![99],
                leaves: vec![ScreenMeshLeafTopology {
                    source_face: 99,
                    id: ScreenPatchLeafId::ROOT,
                    domain: QBPatchDomain::FULL,
                }],
                requested_lods: vec![[1; 3]],
                diagnostic: AdaptiveScreenMeshPlanDiagnostic::default(),
            },
        };

        let error =
            plan_adaptive_screen_components_into(request, &topology, 1, &mut output).unwrap_err();
        assert_eq!(
            error,
            PickedScreenMeshPlanError::ComponentClosure(
                ScreenLeafLodError::ComponentClosureBudgetExceeded {
                    required: 2,
                    maximum: 1,
                },
            ),
        );
        assert!(output.component_faces.is_empty());
        assert!(output.mesh.selected_faces.is_empty());
        assert!(output.mesh.leaves.is_empty());
        assert!(output.mesh.requested_lods.is_empty());
    }

    #[test]
    fn multi_face_plan_rejects_duplicate_source_identity() {
        let patch = QBTriPatch::flat([-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]);
        let selected = [
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &patch,
            },
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &patch,
            },
        ];
        let error = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
            selected_patches: &selected,
            view_projection: &IDENTITY,
            // Structural identity is validated before numeric policy fields,
            // so a bad duplicate cannot be hidden by another request error.
            viewport: [f64::NAN, 0.0],
            min_px_per_segment: 0.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 16,
            max_total_leaves: 16,
            source_requested_lods: &[[1; 3]],
        })
        .unwrap_err();
        assert_eq!(
            error,
            PickedScreenMeshPlanError::DuplicateSelectedFace { face: 0 },
        );
    }

    #[test]
    fn multi_face_plan_rejects_partition_work_above_the_global_bound() {
        let first = QBTriPatch::flat([-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]);
        let second = QBTriPatch::flat([-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]);
        let selected = [
            SelectedScreenPatch {
                source_face: 0,
                transformed_patch: &first,
            },
            SelectedScreenPatch {
                source_face: 1,
                transformed_patch: &second,
            },
        ];
        let error = plan_adaptive_screen_mesh(AdaptiveScreenMeshPlanRequest {
            selected_patches: &selected,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy {
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_partition_leaves: 7,
            max_total_leaves: 32,
            source_requested_lods: &[[1; 3]; 2],
        })
        .unwrap_err();

        assert_eq!(
            error,
            PickedScreenMeshPlanError::PartitionBudgetExceeded {
                requested: 8,
                maximum: 7,
            },
        );
    }

    #[test]
    fn picked_plan_rejects_invalid_identity_before_numeric_fields() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]);
        let error = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 1,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [f64::NAN, 0.0],
            min_px_per_segment: 0.0,
            max_px_per_segment: -1.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 3,
            retain_culled_leaves: false,
            max_total_leaves: 0,
            source_requested_lods: &[[3; 3]],
        })
        .unwrap_err();

        assert_eq!(error, PickedScreenMeshPlanError::InvalidSelectedFace);
    }

    #[test]
    fn picked_plan_validates_partition_before_reporting_exact_leaf_budget() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]);
        let request = PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [f64::NAN, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_total_leaves: 1,
            source_requested_lods: &[[1; 3]; 3],
        };
        assert_eq!(
            plan_picked_screen_mesh(request).unwrap_err(),
            PickedScreenMeshPlanError::Partition(ScreenPartitionError::InvalidProjection),
        );

        let error = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            viewport: [640.0, 480.0],
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            ..request
        })
        .unwrap_err();
        assert_eq!(
            error,
            PickedScreenMeshPlanError::LeafBudgetExceeded {
                requested: 6,
                maximum: 1,
            },
        );
    }

    #[test]
    fn picked_plan_rejects_non_atlas_source_requests() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]);
        let error = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 16.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: false,
            max_total_leaves: 16,
            source_requested_lods: &[[3, 1, 1]],
        })
        .unwrap_err();

        assert_eq!(
            error,
            PickedScreenMeshPlanError::InvalidSourceLod { face: 0, edge: 0 },
        );
    }

    #[test]
    fn picked_plan_omits_currently_culled_selected_leaves() {
        let patch = QBTriPatch::flat([4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0]);
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: false,
            max_total_leaves: 16,
            source_requested_lods: &[[64, 64, 64]],
        })
        .unwrap();

        assert!(plan.leaves.is_empty());
        assert!(plan.requested_lods.is_empty());
        assert_eq!(plan.diagnostic.selected_leaves, 0);
        assert_eq!(plan.diagnostic.omitted_culled_leaves, 1);
    }

    #[test]
    fn live_plan_retains_culled_selected_leaves_at_inherited_density() {
        let patch = QBTriPatch::flat([4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0]);
        let plan = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 16.0,
            max_px_per_segment: 64.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: true,
            max_total_leaves: 16,
            source_requested_lods: &[[8, 4, 2]],
        })
        .unwrap();

        assert_eq!(plan.leaves.len(), 1);
        assert_eq!(plan.leaves[0].id, ScreenPatchLeafId::ROOT);
        assert_eq!(plan.requested_lods, [[8, 4, 2]]);
        assert_eq!(plan.diagnostic.selected_leaves, 1);
        assert_eq!(plan.diagnostic.omitted_culled_leaves, 0);
    }

    #[test]
    fn picked_plan_rejects_impossible_pixel_band_and_scene_leaf_budget() {
        let patch = QBTriPatch::flat([-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]);
        let request = PickedScreenMeshPlanRequest {
            selected_face: 0,
            transformed_patch: &patch,
            view_projection: &IDENTITY,
            viewport: [640.0, 480.0],
            min_px_per_segment: 32.0,
            max_px_per_segment: 16.0,
            policy: ScreenPartitionPolicy::default(),
            max_atlas_lod: 64,
            retain_culled_leaves: false,
            max_total_leaves: 16,
            source_requested_lods: &[[2, 2, 2]],
        };
        assert_eq!(
            plan_picked_screen_mesh(request).unwrap_err(),
            PickedScreenMeshPlanError::ImpossiblePixelBand,
        );

        let error = plan_picked_screen_mesh(PickedScreenMeshPlanRequest {
            min_px_per_segment: 16.0,
            max_px_per_segment: 10_000.0,
            policy: ScreenPartitionPolicy {
                min_depth: 1,
                max_depth: 1,
                max_leaves: 4,
                ..ScreenPartitionPolicy::default()
            },
            max_total_leaves: 3,
            ..request
        })
        .unwrap_err();
        assert_eq!(
            error,
            PickedScreenMeshPlanError::LeafBudgetExceeded {
                requested: 4,
                maximum: 3,
            },
        );
    }
}
