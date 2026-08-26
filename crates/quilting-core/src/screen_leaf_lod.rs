//! Crack-free LoD reconciliation across adaptive dyadic QB leaves.
//!
//! A coarse leaf edge can meet two or more finer leaf edges. Their local edge
//! LoDs are not directly comparable: the invariant is the absolute dyadic
//! resolution `leaf_depth + log2(local_lod)`. This module groups overlapping
//! collinear spans, promotes each group to one absolute resolution, applies the
//! selected within-leaf grading ratio, and iterates to a fixed point.

use crate::patch::QBPatchDomain;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScreenLeafLodResult {
    pub resident: Vec<[u32; 3]>,
    pub iterations: usize,
    pub shared_edge_promotions: usize,
    pub grading_promotions: usize,
    pub max_absolute_exponent: u8,
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
        }
    }
}

impl std::error::Error for ScreenLeafLodError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LineKey {
    constant_axis: u8,
    constant_numerator: u32,
}

#[derive(Clone, Copy, Debug)]
struct EdgeSpan {
    leaf_index: usize,
    edge_index: usize,
    start: u32,
    end: u32,
    depth: u8,
}

fn topology_lines(
    leaves: &[ScreenLeafTopology],
) -> Result<BTreeMap<LineKey, Vec<EdgeSpan>>, ScreenLeafLodError> {
    let global_depth = leaves.iter().map(|leaf| leaf.id.depth).max().unwrap_or(0);
    if global_depth > 16 {
        return Err(ScreenLeafLodError::InvalidTopology { leaf_index: 0 });
    }
    let mut lines = BTreeMap::<LineKey, Vec<EdgeSpan>>::new();
    let edge_corners = [(1usize, 2usize), (0, 2), (0, 1)];
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        if leaf.id.depth > global_depth {
            return Err(ScreenLeafLodError::InvalidTopology { leaf_index });
        }
        let denominator = 1u32 << leaf.id.depth;
        let global_scale = 1u32 << (global_depth - leaf.id.depth);
        let mut corners = [[0u32; 3]; 3];
        for (corner_index, barycentric) in leaf.domain.corners.into_iter().enumerate() {
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
            lines
                .entry(LineKey {
                    constant_axis: constant_axis as u8,
                    constant_numerator: a[constant_axis],
                })
                .or_default()
                .push(EdgeSpan {
                    leaf_index,
                    edge_index,
                    start,
                    end,
                    depth: leaf.id.depth,
                });
        }
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

fn reconcile_lines(
    lines: &BTreeMap<LineKey, Vec<EdgeSpan>>,
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

/// Reconcile local leaf LoDs across one-to-many shared edges and within-leaf
/// atlas grading. Inputs and outputs use logical edge order A/B/C.
pub fn reconcile_screen_leaf_lods(
    leaves: &[ScreenLeafTopology],
    requested: &[[u32; 3]],
    max_face_edge_ratio: u32,
    max_lod: u32,
) -> Result<ScreenLeafLodResult, ScreenLeafLodError> {
    if leaves.len() != requested.len() {
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

    let lines = topology_lines(leaves)?;
    let mut resident = requested.to_vec();
    let mut iterations = 0usize;
    let mut shared_edge_promotions = 0usize;
    let mut grading_promotions = 0usize;
    loop {
        iterations += 1;
        let shared = reconcile_lines(&lines, &mut resident, max_lod)?;
        let graded = apply_grading(&mut resident, max_face_edge_ratio, max_lod)?;
        shared_edge_promotions += shared;
        grading_promotions += graded;
        if shared == 0 && graded == 0 {
            break;
        }
    }

    let max_absolute_exponent = leaves
        .iter()
        .zip(&resident)
        .flat_map(|(leaf, lods)| {
            lods.map(|lod| leaf.id.depth.saturating_add(lod.trailing_zeros() as u8))
        })
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

#[cfg(test)]
mod tests {
    use super::*;

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
        for lods in result.resident {
            let minimum = *lods.iter().min().unwrap();
            let maximum = *lods.iter().max().unwrap();
            assert!(maximum <= 4 * minimum);
        }
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
}
