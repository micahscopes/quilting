//! Backend-neutral source-chart bounds for renderer instance streams.
//!
//! These bounds deliberately sit after ordinary Euclidean node models and
//! before any conformal map. Selection, focus fitting, navigation scale,
//! picking, and every graphics backend can therefore share one coordinate
//! contract without importing browser or GPU state.

use crate::instance_layout;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceFocusBound {
    pub center: [f64; 3],
    pub radius: f64,
    pub vertex_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceNodeFocusBound {
    pub node: usize,
    pub center: [f64; 3],
    pub radius: f64,
    pub vertex_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourceFocusBoundsSnapshot {
    pub nodes: Vec<SourceNodeFocusBound>,
    pub scene: Option<SourceFocusBound>,
}

struct BoundScratch {
    min: [f64; 3],
    max: [f64; 3],
    center: [f64; 3],
    radius_squared: f64,
    vertex_ids: HashSet<u32>,
}

impl BoundScratch {
    fn new() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
            center: [0.0; 3],
            radius_squared: 0.0,
            vertex_ids: HashSet::new(),
        }
    }

    fn include(&mut self, point: [f64; 3], vertex_id: u32) {
        self.vertex_ids.insert(vertex_id);
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(point[axis]);
            self.max[axis] = self.max[axis].max(point[axis]);
        }
    }

    fn finish_center(&mut self) {
        for axis in 0..3 {
            self.center[axis] = (self.min[axis] + self.max[axis]) * 0.5;
        }
    }

    fn include_radius(&mut self, point: [f64; 3]) {
        let delta = [
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        ];
        self.radius_squared = self
            .radius_squared
            .max(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    }

    fn bound(&self) -> SourceFocusBound {
        SourceFocusBound {
            center: self.center,
            radius: self.radius_squared.sqrt(),
            vertex_count: self.vertex_ids.len(),
        }
    }
}

fn apply_affine(matrix: [f32; 16], point: [f64; 3]) -> Option<[f64; 3]> {
    let transformed = [
        f64::from(matrix[0]) * point[0]
            + f64::from(matrix[4]) * point[1]
            + f64::from(matrix[8]) * point[2]
            + f64::from(matrix[12]),
        f64::from(matrix[1]) * point[0]
            + f64::from(matrix[5]) * point[1]
            + f64::from(matrix[9]) * point[2]
            + f64::from(matrix[13]),
        f64::from(matrix[2]) * point[0]
            + f64::from(matrix[6]) * point[1]
            + f64::from(matrix[10]) * point[2]
            + f64::from(matrix[14]),
    ];
    transformed
        .iter()
        .all(|value| value.is_finite())
        .then_some(transformed)
}

/// Compute exact AABB-centred spheres after each visible node's affine model.
///
/// A `None` model excludes that node. This lets render adapters apply their
/// own visibility policy without moving browser-, presentation-, or backend-
/// specific state into the geometry reducer.
pub fn source_focus_bounds(
    instances: &[f32],
    face_nodes: &[usize],
    node_models: &BTreeMap<usize, Option<[f32; 16]>>,
) -> Result<SourceFocusBoundsSnapshot, String> {
    let face_count = face_nodes.len();
    let required = face_count
        .checked_mul(instance_layout::STRIDE)
        .ok_or_else(|| "source focus-bound size overflow".to_string())?;
    if instances.len() < required {
        return Err(format!(
            "source focus bounds received {} floats for {face_count} faces; expected {required}",
            instances.len(),
        ));
    }

    let mut nodes = BTreeMap::<usize, BoundScratch>::new();
    let mut scene = BoundScratch::new();
    for (face, &node) in face_nodes.iter().enumerate() {
        let Some(matrix) = node_models.get(&node).copied().flatten() else {
            continue;
        };
        let record =
            &instances[face * instance_layout::STRIDE..(face + 1) * instance_layout::STRIDE];
        for corner in 0..3 {
            let offset = instance_layout::offset::POSITIONS + corner * 4;
            let point = [
                f64::from(record[offset + 1]),
                f64::from(record[offset + 2]),
                f64::from(record[offset + 3]),
            ];
            let Some(point) = apply_affine(matrix, point) else {
                continue;
            };
            let vertex = record[offset];
            let vertex_id = if vertex.is_finite() && vertex >= 0.0 {
                vertex as u32
            } else {
                u32::try_from(face * 3 + corner).unwrap_or(u32::MAX)
            };
            nodes
                .entry(node)
                .or_insert_with(BoundScratch::new)
                .include(point, vertex_id);
            scene.include(point, vertex_id);
        }
    }
    nodes.retain(|_, scratch| !scratch.vertex_ids.is_empty());
    for scratch in nodes.values_mut() {
        scratch.finish_center();
    }
    if !scene.vertex_ids.is_empty() {
        scene.finish_center();
    }

    for (face, &node) in face_nodes.iter().enumerate() {
        let Some(scratch) = nodes.get_mut(&node) else {
            continue;
        };
        let Some(matrix) = node_models.get(&node).copied().flatten() else {
            continue;
        };
        let record =
            &instances[face * instance_layout::STRIDE..(face + 1) * instance_layout::STRIDE];
        for corner in 0..3 {
            let offset = instance_layout::offset::POSITIONS + corner * 4;
            let point = [
                f64::from(record[offset + 1]),
                f64::from(record[offset + 2]),
                f64::from(record[offset + 3]),
            ];
            let Some(point) = apply_affine(matrix, point) else {
                continue;
            };
            scratch.include_radius(point);
            scene.include_radius(point);
        }
    }

    let node_bounds: Vec<_> = nodes
        .into_iter()
        .map(|(node, scratch)| {
            let bound = scratch.bound();
            SourceNodeFocusBound {
                node,
                center: bound.center,
                radius: bound.radius,
                vertex_count: bound.vertex_count,
            }
        })
        .collect();
    let scene_vertex_count = node_bounds.iter().map(|bound| bound.vertex_count).sum();
    let scene = (!scene.vertex_ids.is_empty()).then(|| {
        let mut bound = scene.bound();
        bound.vertex_count = scene_vertex_count;
        bound
    });
    Ok(SourceFocusBoundsSnapshot {
        nodes: node_bounds,
        scene,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    fn face(points: [[f32; 3]; 3], first_vertex: u32) -> Vec<f32> {
        let mut record = vec![0.0; instance_layout::STRIDE];
        for (corner, point) in points.into_iter().enumerate() {
            let offset = instance_layout::offset::POSITIONS + corner * 4;
            record[offset] = (first_vertex + corner as u32) as f32;
            record[offset + 1..offset + 4].copy_from_slice(&point);
        }
        record
    }

    #[test]
    fn bounds_share_affine_and_visibility_state() {
        let mut instances = face([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], 0);
        instances.extend(face(
            [
                [100.0, 100.0, 0.0],
                [102.0, 100.0, 0.0],
                [102.0, 102.0, 0.0],
            ],
            3,
        ));
        let mut models = BTreeMap::new();
        // Rotate 90 degrees around Z after non-uniform scaling, then translate.
        models.insert(
            4,
            Some([
                0.0, 2.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 5.0, 6.0, 1.0,
            ]),
        );
        models.insert(9, None);

        let bounds = source_focus_bounds(&instances, &[4, 9], &models).unwrap();
        assert_eq!(bounds.nodes.len(), 1);
        assert_eq!(bounds.nodes[0].node, 4);
        assert_eq!(bounds.nodes[0].center, [4.0, 5.0, 6.0]);
        assert!((bounds.nodes[0].radius - 13.0_f64.sqrt()).abs() < 1.0e-12);
        assert_eq!(bounds.nodes[0].vertex_count, 3);
        assert_eq!(bounds.scene.as_ref().unwrap().center, [4.0, 5.0, 6.0]);
        assert_eq!(bounds.scene.as_ref().unwrap().vertex_count, 3);
    }

    #[test]
    fn bounds_reject_truncated_instance_payloads() {
        let mut models = BTreeMap::new();
        models.insert(0, Some(IDENTITY));
        assert!(source_focus_bounds(&[], &[0], &models)
            .unwrap_err()
            .contains("expected"));
    }
}
