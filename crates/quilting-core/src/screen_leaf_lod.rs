//! Crack-free LoD reconciliation across adaptive dyadic QB leaves.
//!
//! A coarse leaf edge can meet two or more finer leaf edges. Their local edge
//! LoDs are not directly comparable: the invariant is the absolute dyadic
//! resolution `leaf_depth + log2(local_lod)`. This module groups overlapping
//! collinear spans, promotes each group to one absolute resolution, applies the
//! selected within-leaf grading ratio, and iterates to a fixed point.

use crate::atlas::TessellationAtlas;
use crate::patch::QBPatchDomain;
use crate::permutation::canonical_form;
use crate::screen_partition::{ScreenPatchLeaf, ScreenPatchLeafId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenLeafTopology {
    pub id: ScreenPatchLeafId,
    pub domain: QBPatchDomain,
}

impl From<&ScreenPatchLeaf> for ScreenLeafTopology {
    fn from(leaf: &ScreenPatchLeaf) -> Self {
        Self {
            id: leaf.id,
            domain: leaf.restricted.domain,
        }
    }
}

/// One adaptive leaf located in the welded authored-mesh topology. Source
/// identity remains separate from the ephemeral dyadic path so two faces may
/// choose different refinement trees while still negotiating their common
/// physical boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenMeshLeafTopology {
    pub source_face: u32,
    pub id: ScreenPatchLeafId,
    pub domain: QBPatchDomain,
}

impl ScreenMeshLeafTopology {
    pub fn from_leaf(source_face: u32, leaf: &ScreenPatchLeaf) -> Self {
        Self {
            source_face,
            id: leaf.id,
            domain: leaf.restricted.domain,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScreenLeafLodResult {
    pub resident: Vec<[u32; 3]>,
    pub iterations: usize,
    pub shared_edge_promotions: usize,
    pub grading_promotions: usize,
    pub max_absolute_exponent: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScreenLeafAtlasWork {
    pub instances: u64,
    pub vertices: u64,
    pub triangles: u64,
}

impl ScreenLeafLodResult {
    /// Exact atlas work implied by the reconciled leaves, before material/pass
    /// multiplication. This gives WebGL2 and WebGPU the same workload oracle.
    pub fn atlas_work(
        &self,
        atlas: &TessellationAtlas,
    ) -> Result<ScreenLeafAtlasWork, ScreenLeafLodError> {
        let mut work = ScreenLeafAtlasWork::default();
        for (leaf_index, edge_lods) in self.resident.iter().copied().enumerate() {
            let key = canonical_form(edge_lods).res;
            let Some(entry) = atlas.patches.get(&key) else {
                return Err(ScreenLeafLodError::MissingAtlasPatch { leaf_index, key });
            };
            work.instances = work.instances.saturating_add(1);
            work.vertices = work.vertices.saturating_add(entry.vertex_count as u64);
            work.triangles = work.triangles.saturating_add(entry.triangle_count as u64);
        }
        Ok(work)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenLeafLodError {
    LengthMismatch,
    InvalidTopology {
        leaf_index: usize,
    },
    InvalidLod {
        leaf_index: usize,
        edge_index: usize,
    },
    InvalidGradingRatio,
    AtlasCapExceeded {
        leaf_index: usize,
        edge_index: usize,
        required_lod: u32,
        max_lod: u32,
    },
    MissingAtlasPatch {
        leaf_index: usize,
        key: [u32; 3],
    },
}

impl std::fmt::Display for ScreenLeafLodError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch => write!(formatter, "leaf topology and LoD lengths differ"),
            Self::InvalidTopology { leaf_index } => {
                write!(formatter, "adaptive leaf {leaf_index} has a non-dyadic domain")
            }
            Self::InvalidLod {
                leaf_index,
                edge_index,
            } => write!(
                formatter,
                "adaptive leaf {leaf_index} edge {edge_index} has an invalid LoD"
            ),
            Self::InvalidGradingRatio => write!(formatter, "invalid leaf grading ratio"),
            Self::AtlasCapExceeded {
                leaf_index,
                edge_index,
                required_lod,
                max_lod,
            } => write!(
                formatter,
                "adaptive leaf {leaf_index} edge {edge_index} needs LoD {required_lod}, atlas cap is {max_lod}"
            ),
            Self::MissingAtlasPatch { leaf_index, key } => write!(
                formatter,
                "adaptive leaf {leaf_index} needs missing atlas patch {key:?}"
            ),
        }
    }
}

impl std::error::Error for ScreenLeafLodError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LineKey {
    constant_axis: u8,
    constant_numerator: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MeshLineKey {
    Interior {
        source_face: u32,
        constant_axis: u8,
        constant_numerator: u32,
    },
    SourceBoundary {
        canonical_half_edge: u32,
    },
}

#[derive(Clone, Copy, Debug)]
struct EdgeSpan {
    leaf_index: usize,
    edge_index: usize,
    start: u32,
    end: u32,
    depth: u8,
}

#[derive(Clone, Copy, Debug)]
struct QuantizedEdgeSpan {
    constant_axis: u8,
    constant_numerator: u32,
    endpoints: [[u32; 3]; 2],
    span: EdgeSpan,
}

fn quantized_edge_spans(
    leaves: impl IntoIterator<Item = (usize, ScreenPatchLeafId, QBPatchDomain)>,
    global_depth: u8,
) -> Result<Vec<QuantizedEdgeSpan>, ScreenLeafLodError> {
    if global_depth > 16 {
        return Err(ScreenLeafLodError::InvalidTopology { leaf_index: 0 });
    }
    let mut edges = Vec::new();
    let edge_corners = [(1usize, 2usize), (0, 2), (0, 1)];
    for (leaf_index, id, domain) in leaves {
        if id.depth > global_depth || id.domain() != Some(domain) {
            return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
        }
        let denominator = 1u32 << id.depth;
        let global_scale = 1u32 << (global_depth - id.depth);
        let mut corners = [[0u32; 3]; 3];
        for (corner_index, barycentric) in domain.corners.into_iter().enumerate() {
            let mut sum = 0u32;
            for coordinate in 0..3 {
                let scaled = barycentric[coordinate] * f64::from(denominator);
                let rounded = scaled.round();
                if !scaled.is_finite()
                    || rounded < 0.0
                    || rounded > f64::from(denominator)
                    || (scaled - rounded).abs() > 1.0e-9
                {
                    return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
                }
                let numerator = rounded as u32;
                corners[corner_index][coordinate] = numerator * global_scale;
                sum += numerator;
            }
            if sum != denominator {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            }
        }

        for (edge_index, (first, second)) in edge_corners.into_iter().enumerate() {
            let a = corners[first];
            let b = corners[second];
            let Some(constant_axis) = (0..3).find(|axis| a[*axis] == b[*axis]) else {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            };
            let parameter_axis = (constant_axis + 1) % 3;
            let start = a[parameter_axis].min(b[parameter_axis]);
            let end = a[parameter_axis].max(b[parameter_axis]);
            if start == end || end - start != global_scale {
                return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
            }
            edges.push(QuantizedEdgeSpan {
                constant_axis: constant_axis as u8,
                constant_numerator: a[constant_axis],
                endpoints: [a, b],
                span: EdgeSpan {
                    leaf_index,
                    edge_index,
                    start,
                    end,
                    depth: id.depth,
                },
            });
        }
    }
    Ok(edges)
}

fn topology_lines(
    leaves: &[ScreenLeafTopology],
) -> Result<BTreeMap<LineKey, Vec<EdgeSpan>>, ScreenLeafLodError> {
    let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap_or(0);
    let mut lines = BTreeMap::<LineKey, Vec<EdgeSpan>>::new();
    let edges = quantized_edge_spans(
        leaves
            .iter()
            .enumerate()
            .map(|(index, leaf)| (index, leaf.id, leaf.domain)),
        global_depth,
    )?;
    for edge in edges {
        lines
            .entry(LineKey {
                constant_axis: edge.constant_axis,
                constant_numerator: edge.constant_numerator,
            })
            .or_default()
            .push(edge.span);
    }
    for spans in lines.values_mut() {
        spans.sort_by_key(|span| (span.start, span.end, span.leaf_index, span.edge_index));
    }
    Ok(lines)
}

fn mesh_topology_lines(
    leaves: &[ScreenMeshLeafTopology],
    source_topology: &quilting_mesh::HalfEdgeMesh,
) -> Result<BTreeMap<MeshLineKey, Vec<EdgeSpan>>, ScreenLeafLodError> {
    let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap_or(0);
    let global_denominator = 1u32
        .checked_shl(u32::from(global_depth))
        .ok_or(ScreenLeafLodError::InvalidTopology { leaf_index: 0 })?;
    let edges = quantized_edge_spans(
        leaves
            .iter()
            .enumerate()
            .map(|(index, leaf)| (index, leaf.id, leaf.domain)),
        global_depth,
    )?;
    let mut lines = BTreeMap::<MeshLineKey, Vec<EdgeSpan>>::new();
    for edge in edges {
        let leaf = &leaves[edge.span.leaf_index];
        if leaf.source_face >= source_topology.num_faces {
            return Err(ScreenLeafLodError::InvalidTopology {
                leaf_index: edge.span.leaf_index,
            });
        }
        let (key, span) = if edge.constant_numerator == 0 {
            let source_edge = usize::from(edge.constant_axis);
            let half_edges = source_topology.face_half_edges(leaf.source_face);
            // Half-edge i runs from source corner i to i+1; logical edge A/B/C
            // is opposite source corner 0/1/2 respectively.
            let half_edge = half_edges[(source_edge + 1) % 3];
            let twin = source_topology.twin(half_edge);
            let canonical_half_edge = twin.map_or(half_edge, |other| half_edge.min(other));
            let parameter_axis = (source_edge + 2) % 3;
            let mut first = edge.endpoints[0][parameter_axis];
            let mut second = edge.endpoints[1][parameter_axis];
            if half_edge != canonical_half_edge {
                first = global_denominator - first;
                second = global_denominator - second;
            }
            let mut span = edge.span;
            span.start = first.min(second);
            span.end = first.max(second);
            (
                MeshLineKey::SourceBoundary {
                    canonical_half_edge,
                },
                span,
            )
        } else {
            (
                MeshLineKey::Interior {
                    source_face: leaf.source_face,
                    constant_axis: edge.constant_axis,
                    constant_numerator: edge.constant_numerator,
                },
                edge.span,
            )
        };
        lines.entry(key).or_default().push(span);
    }
    for spans in lines.values_mut() {
        spans.sort_by_key(|span| (span.start, span.end, span.leaf_index, span.edge_index));
    }
    Ok(lines)
}

fn lod_exponent(lod: u32) -> Option<u8> {
    lod.is_power_of_two().then_some(lod.trailing_zeros() as u8)
}

fn required_local_lod(
    absolute_exponent: u8,
    span: EdgeSpan,
    max_lod: u32,
) -> Result<u32, ScreenLeafLodError> {
    let local_exponent = absolute_exponent
        .checked_sub(span.depth)
        .expect("component maximum includes every member");
    let required_lod = 1u32.checked_shl(u32::from(local_exponent)).ok_or(
        ScreenLeafLodError::AtlasCapExceeded {
            leaf_index: span.leaf_index,
            edge_index: span.edge_index,
            required_lod: u32::MAX,
            max_lod,
        },
    )?;
    if required_lod > max_lod {
        return Err(ScreenLeafLodError::AtlasCapExceeded {
            leaf_index: span.leaf_index,
            edge_index: span.edge_index,
            required_lod,
            max_lod,
        });
    }
    Ok(required_lod)
}

fn reconcile_component(
    component: &[EdgeSpan],
    resident: &mut [[u32; 3]],
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    let mut absolute_exponent = 0u8;
    for span in component {
        let lod = resident[span.leaf_index][span.edge_index];
        let exponent = lod_exponent(lod).ok_or(ScreenLeafLodError::InvalidLod {
            leaf_index: span.leaf_index,
            edge_index: span.edge_index,
        })?;
        absolute_exponent = absolute_exponent.max(span.depth.saturating_add(exponent));
    }
    let mut promotions = 0;
    for span in component {
        let required = required_local_lod(absolute_exponent, *span, max_lod)?;
        let current = &mut resident[span.leaf_index][span.edge_index];
        if *current < required {
            *current = required;
            promotions += 1;
        }
    }
    Ok(promotions)
}

fn reconcile_lines<K: Ord>(
    lines: &BTreeMap<K, Vec<EdgeSpan>>,
    resident: &mut [[u32; 3]],
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    let mut promotions = 0;
    for spans in lines.values() {
        let mut component_start = 0usize;
        while component_start < spans.len() {
            let mut component_end = component_start + 1;
            let mut covered_until = spans[component_start].end;
            while component_end < spans.len() && spans[component_end].start < covered_until {
                covered_until = covered_until.max(spans[component_end].end);
                component_end += 1;
            }
            promotions +=
                reconcile_component(&spans[component_start..component_end], resident, max_lod)?;
            component_start = component_end;
        }
    }
    Ok(promotions)
}

fn apply_grading(
    resident: &mut [[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<usize, ScreenLeafLodError> {
    let mut promotions = 0;
    for (leaf_index, lods) in resident.iter_mut().enumerate() {
        let largest = *lods.iter().max().expect("three edge LoDs");
        let minimum = (largest / max_face_edge_ratio).max(1);
        for (edge_index, lod) in lods.iter_mut().enumerate() {
            if *lod < minimum {
                if minimum > max_lod {
                    return Err(ScreenLeafLodError::AtlasCapExceeded {
                        leaf_index,
                        edge_index,
                        required_lod: minimum,
                        max_lod,
                    });
                }
                *lod = minimum;
                promotions += 1;
            }
        }
    }
    Ok(promotions)
}

fn reconcile_lods<K: Ord>(
    depths: &[u8],
    lines: &BTreeMap<K, Vec<EdgeSpan>>,
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    if depths.len() != requested.len() {
        return Err(ScreenLeafLodError::LengthMismatch);
    }
    if max_face_edge_ratio < 2
        || !max_face_edge_ratio.is_power_of_two()
        || max_lod == 0
        || !max_lod.is_power_of_two()
    {
        return Err(ScreenLeafLodError::InvalidGradingRatio);
    }
    for (leaf_index, lods) in requested.iter().enumerate() {
        for (edge_index, lod) in lods.iter().copied().enumerate() {
            if !lod.is_power_of_two() || lod > max_lod {
                return Err(ScreenLeafLodError::InvalidLod {
                    leaf_index,
                    edge_index,
                });
            }
        }
    }

    let mut resident = requested.to_vec();
    let mut iterations = 0usize;
    let mut shared_edge_promotions = 0usize;
    let mut grading_promotions = 0usize;
    loop {
        iterations += 1;
        let shared = reconcile_lines(lines, &mut resident, max_lod)?;
        let graded = apply_grading(&mut resident, max_face_edge_ratio, max_lod)?;
        shared_edge_promotions += shared;
        grading_promotions += graded;
        if shared == 0 && graded == 0 {
            break;
        }
    }

    let max_absolute_exponent = depths
        .iter()
        .zip(&resident)
        .flat_map(|(&depth, lods)| lods.map(|lod| depth.saturating_add(lod.trailing_zeros() as u8)))
        .max()
        .unwrap_or(0);
    Ok(ScreenLeafLodResult {
        resident,
        iterations,
        shared_edge_promotions,
        grading_promotions,
        max_absolute_exponent,
    })
}

/// Reconcile local leaf LoDs across one-to-many shared edges inside one source
/// patch and apply within-leaf atlas grading. Inputs and outputs use logical
/// edge order A/B/C.
pub fn reconcile_screen_leaf_lods(
    leaves: &[ScreenLeafTopology],
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    let lines = topology_lines(leaves)?;
    let depths = leaves.iter().map(|leaf| leaf.id.depth).collect::<Vec<_>>();
    reconcile_lods(&depths, &lines, requested, max_face_edge_ratio, max_lod)
}

/// Reconcile adaptive leaves across both their source-face interiors and the
/// welded authored-mesh boundaries. Different neighboring faces may choose
/// different dyadic trees; overlapping physical edge intervals still receive
/// one absolute sampling resolution.
pub fn reconcile_screen_mesh_leaf_lods(
    leaves: &[ScreenMeshLeafTopology],
    requested: &[[u32; 3]],
    source_topology: &quilting_mesh::HalfEdgeMesh,
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    let lines = mesh_topology_lines(leaves, source_topology)?;
    let depths = leaves.iter().map(|leaf| leaf.id.depth).collect::<Vec<_>>();
    reconcile_lods(&depths, &lines, requested, max_face_edge_ratio, max_lod)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::BuildMode;
    use crate::sampling::PatchConfig;

    fn mixed_depth_partition() -> Vec<ScreenLeafTopology> {
        let root = QBPatchDomain::FULL.quarter();
        let mut leaves = Vec::new();
        for child_index in 0..3u8 {
            leaves.push(ScreenLeafTopology {
                id: ScreenPatchLeafId::ROOT.child(child_index).unwrap(),
                domain: root[child_index as usize],
            });
        }
        let centre_id = ScreenPatchLeafId::ROOT.child(3).unwrap();
        for (grandchild_index, local) in QBPatchDomain::FULL.quarter().into_iter().enumerate() {
            leaves.push(ScreenLeafTopology {
                id: centre_id.child(grandchild_index as u8).unwrap(),
                domain: root[3].compose(local),
            });
        }
        leaves
    }

    fn assert_overlaps_match(leaves: &[ScreenLeafTopology], lods: &[[u32; 3]]) {
        let lines = topology_lines(leaves).unwrap();
        for spans in lines.values() {
            for (index, a) in spans.iter().enumerate() {
                for b in &spans[index + 1..] {
                    if a.start < b.end && b.start < a.end {
                        let absolute_a =
                            a.depth + lods[a.leaf_index][a.edge_index].trailing_zeros() as u8;
                        let absolute_b =
                            b.depth + lods[b.leaf_index][b.edge_index].trailing_zeros() as u8;
                        assert_eq!(absolute_a, absolute_b, "overlapping spans {a:?} / {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn coarse_edges_promote_to_match_multiple_fine_neighbors() {
        let leaves = mixed_depth_partition();
        let result =
            reconcile_screen_leaf_lods(&leaves, &vec![[1; 3]; leaves.len()], 4, 512).unwrap();
        assert!(result.shared_edge_promotions > 0);
        assert_overlaps_match(&leaves, &result.resident);
        for lods in &result.resident {
            let minimum = *lods.iter().min().unwrap();
            let maximum = *lods.iter().max().unwrap();
            assert!(maximum <= 4 * minimum);
        }
        let mut keys = result
            .resident
            .iter()
            .copied()
            .map(|lods| canonical_form(lods).res)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        let mut levels = keys.iter().flatten().copied().collect::<Vec<_>>();
        levels.sort_unstable();
        levels.dedup();
        let atlas = TessellationAtlas::build_for_keys(
            &levels,
            &keys,
            &PatchConfig::default(),
            BuildMode::Hierarchical,
        );
        let work = result.atlas_work(&atlas).unwrap();
        assert_eq!(work.instances, leaves.len() as u64);
        assert!(work.vertices >= work.instances * 3);
        assert!(work.triangles >= work.instances);
    }

    #[test]
    fn screen_demand_and_grading_reach_one_fixed_point() {
        let leaves = mixed_depth_partition();
        let mut requested = vec![[1; 3]; leaves.len()];
        requested[3][1] = 16;
        let result = reconcile_screen_leaf_lods(&leaves, &requested, 2, 512).unwrap();
        assert!(result.iterations >= 2);
        assert!(result.shared_edge_promotions > 0);
        assert!(result.grading_promotions > 0);
        assert_overlaps_match(&leaves, &result.resident);
        for lods in result.resident {
            let minimum = *lods.iter().min().unwrap();
            let maximum = *lods.iter().max().unwrap();
            assert!(maximum <= 2 * minimum);
        }
    }

    #[test]
    fn atlas_overflow_is_reported_instead_of_dropping_a_leaf() {
        let root = ScreenLeafTopology {
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        };
        let deep_domain = (0..10).fold(QBPatchDomain::FULL, |domain, _| {
            domain.compose(QBPatchDomain::FULL.quarter()[0])
        });
        let deep = ScreenLeafTopology {
            id: (0..10).fold(ScreenPatchLeafId::ROOT, |id, _| id.child(0).unwrap()),
            domain: deep_domain,
        };
        let error =
            reconcile_screen_leaf_lods(&[root, deep], &[[1; 3], [512; 3]], 4, 512).unwrap_err();
        assert!(matches!(error, ScreenLeafLodError::AtlasCapExceeded { .. }));
    }

    #[test]
    fn different_face_trees_reconcile_over_welded_source_edges() {
        let source_topology =
            quilting_mesh::HalfEdgeMesh::from_triangles(4, &[[0, 1, 2], [2, 1, 3]]);
        let quarters = QBPatchDomain::FULL.quarter();
        let mut leaves = (0..4u8)
            .map(|child| ScreenMeshLeafTopology {
                source_face: 0,
                id: ScreenPatchLeafId::ROOT.child(child).unwrap(),
                domain: quarters[child as usize],
            })
            .collect::<Vec<_>>();
        leaves.push(ScreenMeshLeafTopology {
            source_face: 1,
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        });
        let mut requested = vec![[1; 3]; leaves.len()];
        // Face 1 logical edge C is the complete physical edge shared with
        // face 0 logical edge A, which is represented by two depth-1 spans.
        requested[4][2] = 8;

        let result =
            reconcile_screen_mesh_leaf_lods(&leaves, &requested, &source_topology, 4, 512).unwrap();
        assert!(result.shared_edge_promotions >= 2);
        assert_eq!(result.resident[4][2], 8);

        let lines = mesh_topology_lines(&leaves, &source_topology).unwrap();
        for spans in lines.values() {
            for (index, a) in spans.iter().enumerate() {
                for b in &spans[index + 1..] {
                    if a.start < b.end && b.start < a.end {
                        let absolute_a = a.depth
                            + result.resident[a.leaf_index][a.edge_index].trailing_zeros() as u8;
                        let absolute_b = b.depth
                            + result.resident[b.leaf_index][b.edge_index].trailing_zeros() as u8;
                        assert_eq!(absolute_a, absolute_b, "overlapping spans {a:?} / {b:?}");
                    }
                }
            }
        }
    }
}
