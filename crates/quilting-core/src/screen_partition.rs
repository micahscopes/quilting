//! Metric-driven adaptive restriction of QB patches.
//!
//! A single edge-LOD triple cannot counter a large change of projected scale
//! inside one inverted patch: it must allocate for the worst region everywhere.
//! This module is the backend-neutral CPU oracle for splitting only such a
//! patch into exact rational children. Each leaf can still use the existing
//! tessellation atlas and edge reconciliation. WebGL2 can consume a bounded
//! leaf stream; WebGPU can compact the same logical leaves in compute.

use crate::patch::{QBPatchDomain, QBTriPatch, RestrictedQBTriPatch};
use crate::quaternion::{Quat, SINGULARITY_NORM_SQ};
use crate::screen_metric::{
    patch_screen_edge_arc, patch_screen_metric, PatchEdge, ScreenArcOptions,
};

const METRIC_SAMPLES: [[f64; 3]; 7] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenPartitionPolicy {
    pub min_depth: u8,
    pub max_depth: u8,
    /// Greatest allowed ratio of local maximum principal stretch.
    pub max_stretch_ratio: f64,
    /// Greatest allowed ratio of local projected area density.
    pub max_area_ratio: f64,
    /// Patches below this projected edge extent do not merit refinement.
    pub ignore_below_px: f64,
    /// Hard bound for a WebGL2-compatible retained leaf stream.
    pub max_leaves: usize,
    pub arc_options: ScreenArcOptions,
}

impl Default for ScreenPartitionPolicy {
    fn default() -> Self {
        Self {
            min_depth: 0,
            max_depth: 5,
            max_stretch_ratio: 1.5,
            max_area_ratio: 2.25,
            ignore_below_px: 2.0,
            max_leaves: 1024,
            arc_options: ScreenArcOptions {
                tolerance_px: 0.1,
                min_depth: 2,
                max_depth: 12,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenPatchDiagnostic {
    pub edge_arc_px: [f64; 3],
    /// Largest sampled pixel derivative in each logical edge direction over
    /// the solid patch. This catches an interior dilation that boundary arc
    /// length alone cannot see.
    pub directional_extent_px: [f64; 3],
    pub max_edge_arc_px: f64,
    pub stretch_range: [f64; 2],
    pub stretch_ratio: f64,
    pub area_scale_range: [f64; 2],
    pub area_ratio: f64,
    /// Exact minimum of the rational denominator norm over the solid patch.
    pub min_denominator_norm_sq: f64,
    pub unprojectable: bool,
}

impl ScreenPatchDiagnostic {
    fn below_pixel_extent(&self, policy: ScreenPartitionPolicy) -> bool {
        !self.unprojectable && self.max_edge_arc_px <= policy.ignore_below_px
    }

    fn meets_metric_limits(&self, policy: ScreenPartitionPolicy) -> bool {
        !self.unprojectable
            && self.stretch_ratio <= policy.max_stretch_ratio
            && self.area_ratio <= policy.max_area_ratio
    }

    /// Per-edge lower LoD demand for an optional maximum-pixels-per-segment
    /// target. This is deliberately distinct from the existing pixel *floor*,
    /// which is an upper capacity and must never add tessellation.
    pub fn edge_subdivision_demand(
        &self,
        max_px_per_segment: f64,
        max_lod: u32,
    ) -> Option<[u32; 3]> {
        if self.unprojectable
            || !max_px_per_segment.is_finite()
            || max_px_per_segment <= 0.0
            || max_lod == 0
        {
            return None;
        }
        let lod_cap = 1u32 << (31 - max_lod.leading_zeros());
        Some(std::array::from_fn(|edge| {
            let length = self.edge_arc_px[edge].max(self.directional_extent_px[edge]);
            let requested = if length <= max_px_per_segment {
                1
            } else {
                2f64.powf((length / max_px_per_segment).log2().ceil())
                    .min(u32::MAX as f64) as u32
            };
            requested.clamp(1, lod_cap)
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenPatchLeafStatus {
    Accepted,
    BelowPixelExtent,
    DepthLimit,
    LeafBudget,
}

/// Deterministic identity inside one source face's dyadic restriction tree.
/// Pair this with the stable source face ID; do not promote it to scene/entity
/// identity or persist it independently of the refinement policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScreenPatchLeafId {
    pub depth: u8,
    /// Two child bits per level, oldest level in the most significant used
    /// pair. Depth is stored separately so root and repeated child zero remain
    /// distinct.
    pub path: u32,
}

impl ScreenPatchLeafId {
    pub const ROOT: Self = Self { depth: 0, path: 0 };

    pub fn child(self, child_index: u8) -> Option<Self> {
        if child_index >= 4 || self.depth >= 16 {
            return None;
        }
        Some(Self {
            depth: self.depth + 1,
            path: (self.path << 2) | u32::from(child_index),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScreenPatchLeaf {
    pub id: ScreenPatchLeafId,
    pub restricted: RestrictedQBTriPatch,
    pub diagnostic: ScreenPatchDiagnostic,
    pub status: ScreenPatchLeafStatus,
}

#[derive(Clone, Debug)]
pub struct ScreenPatchPartition {
    pub leaves: Vec<ScreenPatchLeaf>,
    pub split_nodes: usize,
    pub max_depth_reached: u8,
    pub unmet_leaves: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenPartitionError {
    InvalidPolicy,
    InvalidProjection,
}

impl std::fmt::Display for ScreenPartitionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPolicy => write!(formatter, "invalid screen patch partition policy"),
            Self::InvalidProjection => write!(formatter, "invalid screen projection or viewport"),
        }
    }
}

impl std::error::Error for ScreenPartitionError {}

fn finite_ratio(minimum: f64, maximum: f64) -> f64 {
    if maximum <= 1.0e-12 {
        1.0
    } else if minimum <= 1.0e-12 {
        f64::INFINITY
    } else {
        maximum / minimum
    }
}

fn segment_origin_distance_sq(a: Quat, b: Quat) -> f64 {
    let direction = b - a;
    let parameter = (-a.dot(direction) / direction.norm_sq().max(1.0e-300)).clamp(0.0, 1.0);
    (a + direction * parameter).norm_sq()
}

/// Exact closest approach of the affine quaternion denominator triangle to
/// zero. This closes the principal hole in any finite screen-metric stencil:
/// a narrow Möbius pole between samples is detected before refinement policy is
/// allowed to accept the patch.
fn min_denominator_norm_sq(patch: &QBTriPatch) -> f64 {
    let [a, b, c] = patch.weights;
    let ab = b - a;
    let ac = c - a;
    let scale = ab.norm_sq().max(ac.norm_sq()).max(1.0e-300);
    let gram = ab.norm_sq() * ac.norm_sq() - ab.dot(ac).powi(2);
    if gram <= 1.0e-24 * scale * scale {
        return segment_origin_distance_sq(a, b)
            .min(segment_origin_distance_sq(a, c))
            .min(segment_origin_distance_sq(b, c));
    }

    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a.norm_sq();
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b.norm_sq();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3))).norm_sq();
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c.norm_sq();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6))).norm_sq();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = c - b;
        return (b + bc * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))).norm_sq();
    }
    let inverse_sum = 1.0 / (va + vb + vc);
    (a + ab * (vb * inverse_sum) + ac * (vc * inverse_sum)).norm_sq()
}

pub fn diagnose_screen_patch(
    transformed_patch: &QBTriPatch,
    view_projection: &[f64; 16],
    viewport: [f64; 2],
    arc_options: ScreenArcOptions,
) -> ScreenPatchDiagnostic {
    // Exact differentials are sampled at corners, edge midpoints, and centre.
    // The denominator minimum and edge integrals are continuous-domain checks;
    // stretch/area ratios remain a measured refinement oracle rather than a
    // formal supremum bound between samples.
    let mut min_stretch = f64::INFINITY;
    let mut max_stretch: f64 = 0.0;
    let mut min_area = f64::INFINITY;
    let mut max_area: f64 = 0.0;
    let mut directional_extent_px = [0.0f64; 3];
    let min_denominator_norm_sq = min_denominator_norm_sq(transformed_patch);
    let mut unprojectable = min_denominator_norm_sq <= SINGULARITY_NORM_SQ;
    for barycentric in METRIC_SAMPLES {
        let metric = patch_screen_metric(
            transformed_patch,
            barycentric[1],
            barycentric[2],
            view_projection,
            viewport,
        );
        let Some(metric) = metric else {
            unprojectable = true;
            continue;
        };
        min_stretch = min_stretch.min(metric.principal_stretch[1]);
        max_stretch = max_stretch.max(metric.principal_stretch[1]);
        min_area = min_area.min(metric.area_scale);
        max_area = max_area.max(metric.area_scale);
        for (edge, direction) in [[-1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
            .into_iter()
            .enumerate()
        {
            directional_extent_px[edge] = directional_extent_px[edge].max(metric.length(direction));
        }
    }

    let mut edge_arc_px = [0.0; 3];
    for (index, edge) in [PatchEdge::A, PatchEdge::B, PatchEdge::C]
        .into_iter()
        .enumerate()
    {
        match patch_screen_edge_arc(
            transformed_patch,
            edge,
            view_projection,
            viewport,
            arc_options,
        ) {
            Ok(arc) if !arc.tolerance_unmet => edge_arc_px[index] = arc.length_px,
            _ => unprojectable = true,
        }
    }

    if !min_stretch.is_finite() {
        min_stretch = 0.0;
    }
    if !min_area.is_finite() {
        min_area = 0.0;
    }
    ScreenPatchDiagnostic {
        edge_arc_px,
        directional_extent_px,
        max_edge_arc_px: edge_arc_px.into_iter().fold(0.0, f64::max),
        stretch_range: [min_stretch, max_stretch],
        stretch_ratio: if unprojectable {
            f64::INFINITY
        } else {
            finite_ratio(min_stretch, max_stretch)
        },
        area_scale_range: [min_area, max_area],
        area_ratio: if unprojectable {
            f64::INFINITY
        } else {
            finite_ratio(min_area, max_area)
        },
        min_denominator_norm_sq,
        unprojectable,
    }
}

fn valid_policy(policy: ScreenPartitionPolicy) -> bool {
    policy.min_depth <= policy.max_depth
        && policy.max_depth <= 12
        && policy.max_stretch_ratio.is_finite()
        && policy.max_stretch_ratio >= 1.0
        && policy.max_area_ratio.is_finite()
        && policy.max_area_ratio >= 1.0
        && policy.ignore_below_px.is_finite()
        && policy.ignore_below_px >= 0.0
        && policy.max_leaves > 0
        && policy.arc_options.tolerance_px.is_finite()
        && policy.arc_options.tolerance_px > 0.0
        && policy.arc_options.min_depth <= policy.arc_options.max_depth
        && policy.arc_options.max_depth <= 24
}

/// Partition an already transformed patch into exact rational leaves.
///
/// The returned domains always refer to the original source patch. Refinement
/// is deterministic and bounded; leaves marked `DepthLimit` or `LeafBudget`
/// retain diagnostics recording that their requested sampled quality was not
/// met. An exact denominator minimum prevents an interior pole being missed by
/// the finite metric stencil.
pub fn partition_screen_patch(
    transformed_patch: &QBTriPatch,
    view_projection: &[f64; 16],
    viewport: [f64; 2],
    policy: ScreenPartitionPolicy,
) -> Result<ScreenPatchPartition, ScreenPartitionError> {
    if !valid_policy(policy) {
        return Err(ScreenPartitionError::InvalidPolicy);
    }
    if viewport
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || view_projection.iter().any(|value| !value.is_finite())
    {
        return Err(ScreenPartitionError::InvalidProjection);
    }

    let root = transformed_patch.restrict(QBPatchDomain::FULL);
    let mut stack = vec![(root, ScreenPatchLeafId::ROOT)];
    let mut leaves = Vec::new();
    let mut split_nodes = 0usize;
    let mut max_depth_reached = 0u8;
    let mut unmet_leaves = 0usize;

    while let Some((restricted, id)) = stack.pop() {
        let depth = id.depth;
        max_depth_reached = max_depth_reached.max(depth);
        let diagnostic = diagnose_screen_patch(
            &restricted.patch,
            view_projection,
            viewport,
            policy.arc_options,
        );
        let below_pixel_extent = diagnostic.below_pixel_extent(policy);
        let quality_met = diagnostic.meets_metric_limits(policy);
        let must_split = depth < policy.min_depth || (!below_pixel_extent && !quality_met);
        let can_split_depth = depth < policy.max_depth;
        let can_split_budget = leaves.len() + stack.len() + 4 <= policy.max_leaves;

        if must_split && can_split_depth && can_split_budget {
            split_nodes += 1;
            let children = restricted.quarter();
            for (child_index, child) in children.into_iter().enumerate().rev() {
                let child_id = id
                    .child(child_index as u8)
                    .expect("partition depth validation keeps leaf paths representable");
                stack.push((child, child_id));
            }
            continue;
        }

        let status = if below_pixel_extent && depth >= policy.min_depth {
            ScreenPatchLeafStatus::BelowPixelExtent
        } else if quality_met && depth >= policy.min_depth {
            ScreenPatchLeafStatus::Accepted
        } else if !can_split_depth {
            unmet_leaves += 1;
            ScreenPatchLeafStatus::DepthLimit
        } else {
            unmet_leaves += 1;
            ScreenPatchLeafStatus::LeafBudget
        };
        leaves.push(ScreenPatchLeaf {
            id,
            restricted,
            diagnostic,
            status,
        });
    }

    Ok(ScreenPatchPartition {
        leaves,
        split_nodes,
        max_depth_reached,
        unmet_leaves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::{Mobius, Quat};

    fn identity_matrix() -> [f64; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn perspective() -> [f64; 16] {
        [
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
        ]
    }

    fn domain_area(domain: QBPatchDomain) -> f64 {
        let uv = domain
            .corners
            .map(|barycentric| [barycentric[1], barycentric[2]]);
        ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
            - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[0][0]))
            .abs()
            * 0.5
    }

    #[test]
    fn uniform_orthographic_patch_remains_one_leaf() {
        let patch = QBTriPatch::flat([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let partition = partition_screen_patch(
            &patch,
            &identity_matrix(),
            [1000.0; 2],
            ScreenPartitionPolicy::default(),
        )
        .unwrap();
        assert_eq!(partition.leaves.len(), 1);
        assert_eq!(partition.split_nodes, 0);
        assert_eq!(partition.unmet_leaves, 0);
        assert_eq!(partition.leaves[0].status, ScreenPatchLeafStatus::Accepted);
        assert_eq!(partition.leaves[0].id, ScreenPatchLeafId::ROOT);
    }

    #[test]
    fn dyadic_leaf_paths_are_depth_qualified_and_bounded() {
        let root = ScreenPatchLeafId::ROOT;
        let zero = root.child(0).unwrap();
        let nested_zero = zero.child(0).unwrap();
        assert_ne!(root, zero);
        assert_ne!(zero, nested_zero);
        assert_eq!(nested_zero, ScreenPatchLeafId { depth: 2, path: 0 });
        assert!(root.child(4).is_none());
        let mut deepest = root;
        for child in 0..16 {
            deepest = deepest.child(child % 4).unwrap();
        }
        assert!(deepest.child(0).is_none());
    }

    #[test]
    fn pixel_ceiling_demand_rounds_up_without_changing_floor_semantics() {
        let diagnostic = ScreenPatchDiagnostic {
            edge_arc_px: [63.0, 65.0, 257.0],
            directional_extent_px: [129.0, 1.0, 1.0],
            max_edge_arc_px: 257.0,
            stretch_range: [1.0, 1.0],
            stretch_ratio: 1.0,
            area_scale_range: [1.0, 1.0],
            area_ratio: 1.0,
            min_denominator_norm_sq: 1.0,
            unprojectable: false,
        };
        assert_eq!(
            diagnostic.edge_subdivision_demand(64.0, 512),
            Some([4, 2, 8]),
        );
        // A non-power-of-two atlas cap admits only its resident lower power.
        assert_eq!(diagnostic.edge_subdivision_demand(64.0, 6), Some([4, 2, 4]),);
    }

    #[test]
    fn inverted_patch_refines_until_local_metric_variation_is_bounded() {
        let source = QBTriPatch::flat([-0.8, -0.5, -3.2], [0.9, -0.4, -3.0], [-0.3, 0.8, -3.4]);
        let patch = source.transform(&Mobius::sphere_reflection(
            Quat::from_point(0.1, -0.1, -2.2),
            0.8,
        ));
        let policy = ScreenPartitionPolicy {
            max_depth: 8,
            max_stretch_ratio: 1.35,
            max_area_ratio: 1.8,
            ignore_below_px: 0.0,
            max_leaves: 65_536,
            ..ScreenPartitionPolicy::default()
        };
        let root =
            diagnose_screen_patch(&patch, &perspective(), [1600.0, 900.0], policy.arc_options);
        let partition =
            partition_screen_patch(&patch, &perspective(), [1600.0, 900.0], policy).unwrap();
        assert!(partition.leaves.len() > 1);
        assert!(partition.leaves.len() < 512);
        assert_eq!(partition.leaves.len(), 1 + 3 * partition.split_nodes);
        assert_eq!(partition.unmet_leaves, 0);
        let identities = partition
            .leaves
            .iter()
            .map(|leaf| leaf.id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(identities.len(), partition.leaves.len());
        let worst_stretch = partition
            .leaves
            .iter()
            .map(|leaf| leaf.diagnostic.stretch_ratio)
            .fold(1.0, f64::max);
        let worst_area = partition
            .leaves
            .iter()
            .map(|leaf| leaf.diagnostic.area_ratio)
            .fold(1.0, f64::max);
        assert!(worst_stretch <= policy.max_stretch_ratio);
        assert!(worst_area <= policy.max_area_ratio);
        assert!(worst_stretch < root.stretch_ratio);
        assert!(worst_area < root.area_ratio);
        let covered_area = partition
            .leaves
            .iter()
            .map(|leaf| domain_area(leaf.restricted.domain))
            .sum::<f64>();
        assert!((covered_area - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn leaf_budget_is_explicit_and_never_exceeded() {
        let patch = QBTriPatch::flat([-1.0, -1.0, -2.0], [1.0, -1.0, -8.0], [-1.0, 1.0, -2.0]);
        let policy = ScreenPartitionPolicy {
            min_depth: 4,
            max_depth: 8,
            max_leaves: 10,
            ..ScreenPartitionPolicy::default()
        };
        let partition =
            partition_screen_patch(&patch, &perspective(), [1600.0, 900.0], policy).unwrap();
        assert!(partition.leaves.len() <= policy.max_leaves);
        assert!(partition.unmet_leaves > 0);
        assert!(partition
            .leaves
            .iter()
            .any(|leaf| leaf.status == ScreenPatchLeafStatus::LeafBudget));
    }

    #[test]
    fn exact_denominator_minimum_finds_an_unsampled_interior_pole() {
        let source = QBTriPatch::flat([-1.0, -1.0, -3.0], [1.0, -1.0, -3.0], [-1.0, 1.0, -3.0]);
        let transform = Mobius::sphere_reflection(Quat::from_point(-0.2, -0.1, -3.0), 1.0);
        let patch = source.transform(&transform);
        let diagnostic = diagnose_screen_patch(
            &patch,
            &perspective(),
            [1600.0, 900.0],
            ScreenArcOptions::default(),
        );
        assert!(diagnostic.min_denominator_norm_sq <= SINGULARITY_NORM_SQ);
        assert!(diagnostic.unprojectable);

        let policy = ScreenPartitionPolicy {
            max_depth: 2,
            max_leaves: 64,
            ..ScreenPartitionPolicy::default()
        };
        let partition =
            partition_screen_patch(&patch, &perspective(), [1600.0, 900.0], policy).unwrap();
        assert!(partition.unmet_leaves > 0);
        assert!(partition.leaves.iter().any(|leaf| {
            leaf.diagnostic.min_denominator_norm_sq <= SINGULARITY_NORM_SQ
                && leaf.status == ScreenPatchLeafStatus::DepthLimit
        }));
    }
}
