//! Read-only renderer oracle for metric-adaptive QB patch work.
//!
//! This module owns no browser or GPU state. The WebGL adapter supplies one
//! current output-chart patch, camera, active atlas counts, and policy. Keeping
//! the measurement pure prevents experimental refinement policy from leaking
//! into the live frame renderer before its workload and failure modes are
//! understood.

use std::collections::BTreeMap;

use quilting_core::patch::{QBPatchDomain, QBTriPatch};
use quilting_core::permutation::canonical_form;
use quilting_core::screen_leaf_lod::{reconcile_screen_leaf_lods, ScreenLeafTopology};
use quilting_core::screen_partition::{
    diagnose_screen_patch, partition_screen_patch, ScreenPartitionPolicy, ScreenPatchDiagnostic,
    ScreenPatchLeafId, ScreenPatchLeafStatus,
};
use serde::Serialize;

#[derive(Clone, Copy)]
pub(crate) struct AdaptiveScreenRequest<'a> {
    pub face: u32,
    pub node: u32,
    pub patch: &'a QBTriPatch,
    pub view_projection: [f64; 16],
    pub viewport: [f64; 2],
    pub max_px_per_segment: f64,
    pub policy: ScreenPartitionPolicy,
    pub max_face_edge_ratio: u32,
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
    if !request.max_px_per_segment.is_finite() || request.max_px_per_segment <= 0.0 {
        return Err("maximum pixels per segment must be finite and positive".into());
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

    let single_patch_requested_lod = root_metric
        .edge_subdivision_demand(request.max_px_per_segment, max_atlas_lod)
        .unwrap_or([max_atlas_lod; 3]);
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

    let mut saturated_metric_leaves = 0u32;
    let requested = partition
        .leaves
        .iter()
        .map(|leaf| {
            leaf.diagnostic
                .edge_subdivision_demand(request.max_px_per_segment, max_atlas_lod)
                .unwrap_or_else(|| {
                    saturated_metric_leaves += 1;
                    [max_atlas_lod; 3]
                })
        })
        .collect::<Vec<_>>();
    let topology = partition
        .leaves
        .iter()
        .map(ScreenLeafTopology::from)
        .collect::<Vec<_>>();
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
    for (leaf, resident) in partition.leaves.iter().zip(&reconciled.resident) {
        let status = match leaf.status {
            ScreenPatchLeafStatus::Accepted => "accepted",
            ScreenPatchLeafStatus::BelowPixelExtent => "belowPixelExtent",
            ScreenPatchLeafStatus::DepthLimit => "depthLimit",
            ScreenPatchLeafStatus::LeafBudget => "leafBudget",
        };
        *status_histogram.entry(status).or_default() += 1;
        *depth_histogram.entry(leaf.id.depth).or_default() += 1;
        worst_leaf_stretch_ratio = worst_leaf_stretch_ratio.max(leaf.diagnostic.stretch_ratio);
        worst_leaf_area_ratio = worst_leaf_area_ratio.max(leaf.diagnostic.area_ratio);
        for edge in 0..3 {
            let extent =
                leaf.diagnostic.edge_arc_px[edge].max(leaf.diagnostic.directional_extent_px[edge]);
            let segment = extent / f64::from(resident[edge]);
            if segment.is_finite() {
                max_local_metric_segment_px = max_local_metric_segment_px.max(segment);
                if segment > 0.0 {
                    min_local_metric_segment_px = min_local_metric_segment_px.min(segment);
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
    let quality_met = partition.unmet_leaves == 0
        && saturated_metric_leaves == 0
        && missing_atlas_keys.is_empty()
        && max_local_metric_segment_px <= request.max_px_per_segment;

    Ok(AdaptiveScreenPatchSnapshot {
        ok: true,
        quality_met,
        face: request.face,
        node: request.node,
        viewport: [request.viewport[0] as u32, request.viewport[1] as u32],
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
        adaptive_instances: partition.leaves.len() as u64,
        adaptive_triangles,
    })
}
