//! Bounded scene plans for metric-adaptive QB patch restriction.
//!
//! A live WebGL2 proof may adapt one selected source face while retaining one
//! root leaf for every other face. Keeping this assembly backend-neutral makes
//! the resulting leaf/request stream reusable by WebGPU and prevents browser
//! state from becoming a second geometry authority.

use crate::patch::{QBPatchDomain, QBTriPatch};
use crate::screen_leaf_lod::ScreenMeshLeafTopology;
use crate::screen_partition::{
    partition_screen_patch, ScreenPartitionError, ScreenPartitionPolicy, ScreenPatchLeafId,
};

#[derive(Clone, Copy, Debug)]
pub struct PickedScreenMeshPlanRequest<'a> {
    pub selected_face: u32,
    pub transformed_patch: &'a QBTriPatch,
    pub view_projection: &'a [f64; 16],
    pub viewport: [f64; 2],
    /// Existing screen-attenuation floor. Raw source requests already obey
    /// this cap, but retaining it here lets the planner reject an impossible
    /// requested band rather than silently choosing one side.
    pub min_px_per_segment: f64,
    /// Optional tessellation-density ceiling, in pixels per local atlas edge
    /// segment. This adds detail; it is distinct from the classifier's
    /// pixels-per-subtriangle floor, which can only remove detail.
    pub max_px_per_segment: f64,
    pub policy: ScreenPartitionPolicy,
    pub max_atlas_lod: u32,
    /// Scene-wide instance bound after replacing the selected root.
    pub max_total_leaves: usize,
    /// Raw, unpromoted source-face requests in logical A/B/C edge order.
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
}

#[derive(Clone, Debug)]
pub struct PickedScreenMeshPlan {
    pub selected_face: u32,
    pub leaves: Vec<ScreenMeshLeafTopology>,
    pub requested_lods: Vec<[u32; 3]>,
    pub diagnostic: PickedScreenMeshPlanDiagnostic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickedScreenMeshPlanError {
    EmptyScene,
    InvalidSelectedFace,
    InvalidPixelBand,
    ImpossiblePixelBand,
    InvalidAtlasLod,
    InvalidSourceLod { face: usize, edge: usize },
    InvalidLeafDomain,
    UnmetPartition { leaves: usize },
    UnresolvedMetric,
    LeafBudgetExceeded { requested: usize, maximum: usize },
    Partition(ScreenPartitionError),
    CountOverflow,
}

impl std::fmt::Display for PickedScreenMeshPlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyScene => write!(formatter, "adaptive screen plan requires a source scene"),
            Self::InvalidSelectedFace => write!(formatter, "selected adaptive face is unavailable"),
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
            Self::LeafBudgetExceeded { requested, maximum } => write!(
                formatter,
                "adaptive plan needs {requested} leaves; scene budget is {maximum}",
            ),
            Self::Partition(error) => write!(formatter, "adaptive partition failed: {error}"),
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

fn validate_source_lods(
    source_requested_lods: &[[u32; 3]],
    max_atlas_lod: u32,
) -> Result<(), PickedScreenMeshPlanError> {
    for (face, lods) in source_requested_lods.iter().copied().enumerate() {
        for (edge, lod) in lods.into_iter().enumerate() {
            if !lod.is_power_of_two() || lod > max_atlas_lod {
                return Err(PickedScreenMeshPlanError::InvalidSourceLod { face, edge });
            }
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

/// Build a bounded current-view frontier that adapts one selected face and
/// retains every other source face as one root leaf. Selected leaves proven
/// fully faded or outside the supplied static view are omitted; a renderer
/// must rebuild before treating a changed camera as authoritative.
///
/// The output is raw request state. Shared-edge promotion and within-leaf
/// grading are intentionally deferred to [`crate::screen_leaf_lod`] so every
/// backend uses one reconciliation implementation and can demote cleanly.
/// A renderer must still reject reconciliation overflow, missing atlas keys,
/// and an excessive reconciled vertex/triangle/byte workload before swapping
/// this plan into live residency.
pub fn plan_picked_screen_mesh(
    request: PickedScreenMeshPlanRequest<'_>,
) -> Result<PickedScreenMeshPlan, PickedScreenMeshPlanError> {
    if request.source_requested_lods.is_empty() {
        return Err(PickedScreenMeshPlanError::EmptyScene);
    }
    let selected_face = request.selected_face as usize;
    if selected_face >= request.source_requested_lods.len() {
        return Err(PickedScreenMeshPlanError::InvalidSelectedFace);
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
    validate_source_lods(request.source_requested_lods, request.max_atlas_lod)?;

    let partition = partition_screen_patch(
        request.transformed_patch,
        request.view_projection,
        request.viewport,
        request.policy,
    )?;
    if partition.unmet_leaves != 0 {
        return Err(PickedScreenMeshPlanError::UnmetPartition {
            leaves: partition.unmet_leaves,
        });
    }
    let drawable_leaves = partition
        .leaves
        .iter()
        .filter(|leaf| leaf.status.is_drawable())
        .collect::<Vec<_>>();
    let total_capacity = request
        .source_requested_lods
        .len()
        .checked_sub(1)
        .and_then(|roots| roots.checked_add(drawable_leaves.len()))
        .ok_or(PickedScreenMeshPlanError::CountOverflow)?;
    if request.max_total_leaves == 0 || total_capacity > request.max_total_leaves {
        return Err(PickedScreenMeshPlanError::LeafBudgetExceeded {
            requested: total_capacity,
            maximum: request.max_total_leaves,
        });
    }
    let mut leaves = Vec::with_capacity(total_capacity);
    let mut requested_lods = Vec::with_capacity(total_capacity);
    let mut saturated_metric_leaves = 0u32;

    for (source_face, source_lods) in request.source_requested_lods.iter().copied().enumerate() {
        let source_face_u32 =
            u32::try_from(source_face).map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
        if source_face != selected_face {
            leaves.push(ScreenMeshLeafTopology {
                source_face: source_face_u32,
                id: ScreenPatchLeafId::ROOT,
                domain: QBPatchDomain::FULL,
            });
            requested_lods.push(source_lods);
            continue;
        }

        for leaf in &drawable_leaves {
            let inherited =
                inherited_source_edge_lods(leaf.id, leaf.restricted.domain, source_lods)?;
            let diagnostic = leaf
                .metric_diagnostic
                .ok_or(PickedScreenMeshPlanError::UnresolvedMetric)?;
            let screen = diagnostic
                .edge_subdivision_demand(request.max_px_per_segment, request.max_atlas_lod)
                .ok_or_else(|| {
                    saturated_metric_leaves = saturated_metric_leaves.saturating_add(1);
                    PickedScreenMeshPlanError::UnresolvedMetric
                })?;
            leaves.push(ScreenMeshLeafTopology::from_leaf(source_face_u32, leaf));
            requested_lods.push(std::array::from_fn(|edge| {
                inherited[edge].max(screen[edge])
            }));
        }
    }

    let source_faces = u32::try_from(request.source_requested_lods.len())
        .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    let selected_leaves = u32::try_from(drawable_leaves.len())
        .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    let total_leaves =
        u32::try_from(leaves.len()).map_err(|_| PickedScreenMeshPlanError::CountOverflow)?;
    Ok(PickedScreenMeshPlan {
        selected_face: request.selected_face,
        leaves,
        requested_lods,
        diagnostic: PickedScreenMeshPlanDiagnostic {
            source_faces,
            selected_leaves,
            total_leaves,
            split_nodes: u32::try_from(partition.split_nodes)
                .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
            max_depth_reached: partition.max_depth_reached,
            unmet_leaves: u32::try_from(partition.unmet_leaves)
                .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
            saturated_metric_leaves,
            omitted_culled_leaves: u32::try_from(
                partition.leaves.len().saturating_sub(drawable_leaves.len()),
            )
            .map_err(|_| PickedScreenMeshPlanError::CountOverflow)?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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
