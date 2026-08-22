//! CPU-side surface samples for the backend-neutral Hyperscape walker.
//!
//! The renderer keeps the immutable 52-float face records; this module borrows
//! them rather than retaining a second chess-scale copy. Adjacency is built
//! lazily on first attachment. Posed control points follow the shader order:
//! morph, skin, ordinary affine model, rational QB evaluation, then Möbius.

use std::collections::BTreeMap;

use hyperscape::{
    StableEntityId, SurfaceAddress, SurfaceAdvance, SurfaceAttachment, SurfaceDetachReason,
    SurfaceField, SurfaceSample, SurfaceWalker, SurfaceWalkerStatus, TriangleAdjacency,
};
use quilting_core::instance_layout;
use quilting_core::patch::QBTriPatch;
use quilting_core::{Mobius, Quat};
use serde::Serialize;
use uuid::Uuid;

#[derive(Debug, Default)]
pub(crate) struct SurfaceRuntime {
    walker: SurfaceWalker,
    adjacency: Option<TriangleAdjacency>,
    joint_indices: Vec<[u16; 4]>,
    joint_weights: Vec<[f32; 4]>,
    morph_deltas: Vec<f32>,
    morph_num_vertices: usize,
    morph_num_targets: usize,
    joint_matrices: Vec<f32>,
    morph_weights: Vec<f32>,
    previous_output_position: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SurfaceRuntimeSnapshot {
    pub status: &'static str,
    pub detach_reason: Option<&'static str>,
    pub node: Option<u32>,
    pub face: Option<u32>,
    pub barycentric: Option<[f64; 3]>,
    pub output_position: Option<[f64; 3]>,
    pub output_normal: Option<[f64; 3]>,
    pub eye_position: Option<[f64; 3]>,
    pub projected_output_velocity: [f64; 3],
    pub condition_number: f64,
    pub substeps: u32,
    pub edge_crossings: u32,
}

impl SurfaceRuntime {
    pub fn reset_geometry(&mut self) {
        self.walker = SurfaceWalker::default();
        self.adjacency = None;
        self.previous_output_position = None;
    }

    pub fn set_skinning(&mut self, joint_indices: &[[u16; 4]], joint_weights: &[[f32; 4]]) {
        self.joint_indices.clear();
        self.joint_indices.extend_from_slice(joint_indices);
        self.joint_weights.clear();
        self.joint_weights.extend_from_slice(joint_weights);
    }

    pub fn set_morph_targets(&mut self, deltas: &[f32], num_vertices: usize, num_targets: usize) {
        self.morph_deltas.clear();
        self.morph_deltas.extend_from_slice(deltas);
        self.morph_num_vertices = num_vertices;
        self.morph_num_targets = num_targets;
    }

    pub fn set_pose(&mut self, joint_matrices: &[f32], morph_weights: &[f32]) {
        self.joint_matrices.clear();
        self.joint_matrices.extend_from_slice(joint_matrices);
        self.morph_weights.clear();
        self.morph_weights.extend_from_slice(morph_weights);
    }

    pub fn clear_animation(&mut self) {
        self.joint_indices.clear();
        self.joint_weights.clear();
        self.morph_deltas.clear();
        self.morph_num_vertices = 0;
        self.morph_num_targets = 0;
        self.joint_matrices.clear();
        self.morph_weights.clear();
        self.previous_output_position = None;
    }

    pub fn attachment_face(&self) -> Option<u32> {
        self.walker
            .attachment()
            .map(|attachment| attachment.address.face)
    }

    pub fn attach(
        &mut self,
        instances: &[f32],
        num_faces: usize,
        face_nodes: &[usize],
        face: u32,
        barycentric: [f64; 3],
        eye_height: f64,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<SurfaceRuntimeSnapshot, String> {
        if face as usize >= num_faces || face_nodes.len() != num_faces {
            return Err("surface face is outside the current model".into());
        }
        if self.adjacency.is_none() {
            self.adjacency = Some(build_adjacency(instances, num_faces)?);
        }
        let node = face_nodes[face as usize];
        let entity = runtime_entity_id(node);
        let address = SurfaceAddress::new(entity, face, barycentric)
            .map_err(|error| format!("invalid surface address: {error:?}"))?;
        let attachment = SurfaceAttachment::new(address, eye_height)
            .map_err(|error| format!("invalid surface attachment: {error:?}"))?;
        self.walker.attach(attachment);
        self.previous_output_position = None;
        self.step_relative(
            instances,
            face_nodes,
            0.0,
            [0.0; 3],
            mobius,
            euclidean_model,
        )
    }

    pub fn detach(&mut self) -> SurfaceRuntimeSnapshot {
        self.walker.detach(SurfaceDetachReason::Manual);
        self.previous_output_position = None;
        detached_snapshot(SurfaceDetachReason::Manual)
    }

    pub fn step_relative(
        &mut self,
        instances: &[f32],
        face_nodes: &[usize],
        delta_seconds: f64,
        relative_output_velocity: [f64; 3],
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<SurfaceRuntimeSnapshot, String> {
        let attachment = self
            .walker
            .attachment()
            .ok_or_else(|| "surface walker is detached".to_string())?;
        let node = *face_nodes
            .get(attachment.address.face as usize)
            .ok_or_else(|| "surface face has no node identity".to_string())?;
        let previous_output_position = self.previous_output_position;
        let adjacency = self
            .adjacency
            .as_ref()
            .ok_or_else(|| "surface topology is unavailable".to_string())?;

        let mut field = RuntimeSurfaceField {
            instances,
            face_nodes,
            adjacency,
            joint_indices: &self.joint_indices,
            joint_weights: &self.joint_weights,
            morph_deltas: &self.morph_deltas,
            morph_num_vertices: self.morph_num_vertices,
            morph_num_targets: self.morph_num_targets,
            joint_matrices: &self.joint_matrices,
            morph_weights: &self.morph_weights,
            mobius: mobius_from_array(mobius),
            euclidean_model,
            surface_velocity: [0.0; 3],
        };
        let current_start = field
            .sample(attachment.address)
            .ok_or_else(|| "could not sample attached surface".to_string())?;
        if delta_seconds > 1.0e-9 {
            if let Some(previous) = previous_output_position {
                field.surface_velocity = scale(
                    sub(current_start.output_position, previous),
                    1.0 / delta_seconds,
                );
            }
        }
        let desired_absolute_velocity = add(relative_output_velocity, field.surface_velocity);
        let advance = self
            .walker
            .advance(delta_seconds, desired_absolute_velocity, &mut field);
        self.previous_output_position = advance.contact.map(|contact| contact.output_position);
        Ok(snapshot_from_advance(advance, node))
    }
}

struct RuntimeSurfaceField<'a> {
    instances: &'a [f32],
    face_nodes: &'a [usize],
    adjacency: &'a TriangleAdjacency,
    joint_indices: &'a [[u16; 4]],
    joint_weights: &'a [[f32; 4]],
    morph_deltas: &'a [f32],
    morph_num_vertices: usize,
    morph_num_targets: usize,
    joint_matrices: &'a [f32],
    morph_weights: &'a [f32],
    mobius: Mobius,
    euclidean_model: [f32; 16],
    surface_velocity: [f64; 3],
}

impl SurfaceField for RuntimeSurfaceField<'_> {
    fn sample(&mut self, address: SurfaceAddress) -> Option<SurfaceSample> {
        let patch = self.patch(address.face as usize)?;
        let u = address.barycentric[1];
        let v = address.barycentric[2];
        let epsilon = 1.0e-5;
        let point = patch.eval(u, v).to_point();
        let point_u = patch.eval(u + epsilon, v).to_point();
        let point_v = patch.eval(u, v + epsilon).to_point();
        let tangent_u = scale(sub(point_u, point), 1.0 / epsilon);
        let tangent_v = scale(sub(point_v, point), 1.0 / epsilon);
        [point, tangent_u, tangent_v]
            .into_iter()
            .flatten()
            .all(f64::is_finite)
            .then_some(SurfaceSample {
                output_position: point,
                tangent_u,
                tangent_v,
                surface_velocity: self.surface_velocity,
            })
    }

    fn cross_edge(
        &mut self,
        address_on_edge: SurfaceAddress,
        opposite_corner: usize,
    ) -> Option<SurfaceAddress> {
        let source_node = *self.face_nodes.get(address_on_edge.face as usize)?;
        let source_position = self.sample(address_on_edge)?.output_position;
        let next = self
            .adjacency
            .cross_edge(address_on_edge, opposite_corner)?;
        if *self.face_nodes.get(next.face as usize)? != source_node {
            return None;
        }

        // glTF commonly duplicates vertices at UV/normal seams. The render
        // topology already welds exact rest positions for LOD continuity, so
        // mirror that policy here, but reject a crossing if animation or QB
        // data makes the two posed output-chart edge points discontinuous.
        let target_position = self.sample(next)?.output_position;
        let separation = length(sub(target_position, source_position));
        let scale = 1.0 + length(source_position).max(length(target_position));
        (separation <= 1.0e-7 * scale).then_some(next)
    }
}

impl RuntimeSurfaceField<'_> {
    fn patch(&self, face: usize) -> Option<QBTriPatch> {
        let base = face.checked_mul(instance_layout::STRIDE)?;
        let record = self.instances.get(base..base + instance_layout::STRIDE)?;
        let mut positions = [Quat::ZERO; 3];
        let mut weights = [Quat::ZERO; 3];
        for corner in 0..3 {
            let position_offset = instance_layout::offset::POSITIONS + corner * 4;
            let vertex = record[position_offset];
            if !vertex.is_finite() || vertex < 0.0 || vertex.fract() != 0.0 {
                return None;
            }
            let rest = [
                record[position_offset + 1] as f64,
                record[position_offset + 2] as f64,
                record[position_offset + 3] as f64,
            ];
            let posed = self.posed_position(vertex as usize, rest)?;
            let modeled = apply_affine(self.euclidean_model, posed)?;
            positions[corner] = Quat::from_point(modeled[0], modeled[1], modeled[2]);

            let weight_offset = instance_layout::offset::WEIGHTS + corner * 4;
            weights[corner] = Quat::new(
                record[weight_offset] as f64,
                record[weight_offset + 1] as f64,
                record[weight_offset + 2] as f64,
                record[weight_offset + 3] as f64,
            );
        }
        Some(QBTriPatch::new(positions, weights).transform(&self.mobius))
    }

    fn posed_position(&self, vertex: usize, rest: [f64; 3]) -> Option<[f64; 3]> {
        let mut position = rest;
        if self.morph_num_targets > 0
            && self.morph_num_vertices > vertex
            && !self.morph_deltas.is_empty()
        {
            for target in 0..self.morph_num_targets.min(self.morph_weights.len()) {
                let weight = self.morph_weights[target] as f64;
                if weight.abs() < 1.0e-6 {
                    continue;
                }
                let offset = (target * self.morph_num_vertices + vertex).checked_mul(3)?;
                let delta = self.morph_deltas.get(offset..offset + 3)?;
                for axis in 0..3 {
                    position[axis] += weight * delta[axis] as f64;
                }
            }
        }

        let num_joints = self.joint_matrices.len() / 16;
        if num_joints == 0 {
            return finite3(position).then_some(position);
        }
        let indices = *self.joint_indices.get(vertex)?;
        let weights = *self.joint_weights.get(vertex)?;
        let mut skinned = [0.0; 3];
        for influence in 0..4 {
            let weight = weights[influence] as f64;
            let joint = indices[influence] as usize;
            if weight < 1.0e-6 || joint >= num_joints {
                continue;
            }
            let matrix = self.joint_matrices.get(joint * 16..joint * 16 + 16)?;
            let transformed = [
                matrix[0] as f64 * position[0]
                    + matrix[4] as f64 * position[1]
                    + matrix[8] as f64 * position[2]
                    + matrix[12] as f64,
                matrix[1] as f64 * position[0]
                    + matrix[5] as f64 * position[1]
                    + matrix[9] as f64 * position[2]
                    + matrix[13] as f64,
                matrix[2] as f64 * position[0]
                    + matrix[6] as f64 * position[1]
                    + matrix[10] as f64 * position[2]
                    + matrix[14] as f64,
            ];
            for axis in 0..3 {
                skinned[axis] += weight * transformed[axis];
            }
        }
        finite3(skinned).then_some(skinned)
    }
}

fn build_adjacency(instances: &[f32], num_faces: usize) -> Result<TriangleAdjacency, String> {
    let required = num_faces
        .checked_mul(instance_layout::STRIDE)
        .ok_or_else(|| "surface topology size overflow".to_string())?;
    if instances.len() < required {
        return Err("surface instance data is truncated".into());
    }
    let mut faces = Vec::with_capacity(num_faces);
    let mut welded_vertices = BTreeMap::<[u32; 3], u64>::new();
    let mut next_welded_vertex = 0_u64;
    for face in 0..num_faces {
        let base = face * instance_layout::STRIDE + instance_layout::offset::POSITIONS;
        let mut vertices = [0_u64; 3];
        for corner in 0..3 {
            let vertex = instances[base + corner * 4];
            if !vertex.is_finite() || vertex < 0.0 || vertex.fract() != 0.0 {
                return Err(format!(
                    "surface face {face} has an invalid vertex identity"
                ));
            }
            let rest = [
                instances[base + corner * 4 + 1],
                instances[base + corner * 4 + 2],
                instances[base + corner * 4 + 3],
            ];
            if !rest.into_iter().all(f32::is_finite) {
                return Err(format!("surface face {face} has an invalid rest position"));
            }
            let key = rest.map(|component| {
                // Exact-position welding should regard signed zero as the same
                // coordinate without merging any genuinely distinct points.
                if component == 0.0 {
                    0.0_f32.to_bits()
                } else {
                    component.to_bits()
                }
            });
            vertices[corner] = *welded_vertices.entry(key).or_insert_with(|| {
                let identity = next_welded_vertex;
                next_welded_vertex += 1;
                identity
            });
        }
        faces.push(vertices);
    }
    Ok(TriangleAdjacency::new(faces))
}

fn runtime_entity_id(node: usize) -> StableEntityId {
    // Ordinary glTF nodes do not yet carry durable UUIDs. This identity is
    // stable for the lifetime of one loaded model and is replaced by authored
    // StableEntityId data when available through Hyperscape interchange.
    StableEntityId(Uuid::from_u128(node as u128 + 1))
}

fn mobius_from_array(values: [f32; 16]) -> Mobius {
    let q = |offset: usize| {
        Quat::new(
            values[offset] as f64,
            values[offset + 1] as f64,
            values[offset + 2] as f64,
            values[offset + 3] as f64,
        )
    };
    Mobius::new(q(0), q(4), q(8), q(12))
}

fn apply_affine(matrix: [f32; 16], point: [f64; 3]) -> Option<[f64; 3]> {
    let output = [
        matrix[0] as f64 * point[0]
            + matrix[4] as f64 * point[1]
            + matrix[8] as f64 * point[2]
            + matrix[12] as f64,
        matrix[1] as f64 * point[0]
            + matrix[5] as f64 * point[1]
            + matrix[9] as f64 * point[2]
            + matrix[13] as f64,
        matrix[2] as f64 * point[0]
            + matrix[6] as f64 * point[1]
            + matrix[10] as f64 * point[2]
            + matrix[14] as f64,
    ];
    finite3(output).then_some(output)
}

fn snapshot_from_advance(advance: SurfaceAdvance, node: usize) -> SurfaceRuntimeSnapshot {
    let detach_reason = match advance.status {
        SurfaceWalkerStatus::Attached => None,
        SurfaceWalkerStatus::Detached(reason) => Some(detach_reason_name(reason)),
    };
    SurfaceRuntimeSnapshot {
        status: if detach_reason.is_some() {
            "detached"
        } else {
            "attached"
        },
        detach_reason,
        node: advance.contact.map(|_| node as u32),
        face: advance.contact.map(|contact| contact.address.face),
        barycentric: advance.contact.map(|contact| contact.address.barycentric),
        output_position: advance.contact.map(|contact| contact.output_position),
        output_normal: advance.contact.map(|contact| contact.output_normal),
        eye_position: advance.contact.map(|contact| contact.eye_position),
        projected_output_velocity: advance.projected_output_velocity,
        condition_number: advance.condition_number,
        substeps: advance.substeps,
        edge_crossings: advance.edge_crossings,
    }
}

fn detached_snapshot(reason: SurfaceDetachReason) -> SurfaceRuntimeSnapshot {
    SurfaceRuntimeSnapshot {
        status: "detached",
        detach_reason: Some(detach_reason_name(reason)),
        node: None,
        face: None,
        barycentric: None,
        output_position: None,
        output_normal: None,
        eye_position: None,
        projected_output_velocity: [0.0; 3],
        condition_number: f64::INFINITY,
        substeps: 0,
        edge_crossings: 0,
    }
}

fn detach_reason_name(reason: SurfaceDetachReason) -> &'static str {
    match reason {
        SurfaceDetachReason::Manual => "manual",
        SurfaceDetachReason::InvalidInput => "invalid-input",
        SurfaceDetachReason::SampleUnavailable => "sample-unavailable",
        SurfaceDetachReason::IllConditioned => "ill-conditioned",
        SurfaceDetachReason::Boundary => "boundary",
        SurfaceDetachReason::IterationLimit => "iteration-limit",
    }
}

fn finite3(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn length(value: [f64; 3]) -> f64 {
    value
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_instances() -> Vec<f32> {
        let mut instances = vec![0.0; instance_layout::STRIDE];
        for (corner, point) in [
            [0.0_f32, 0.0, 0.0],
            [1.0_f32, 0.0, 0.0],
            [0.0_f32, 1.0, 0.0],
        ]
        .into_iter()
        .enumerate()
        {
            let offset = instance_layout::offset::POSITIONS + corner * 4;
            instances[offset] = corner as f32;
            instances[offset + 1..offset + 4].copy_from_slice(&point);
            instances[instance_layout::offset::WEIGHTS + corner * 4] = 1.0;
        }
        instances
    }

    #[test]
    fn runtime_samples_affine_and_mobius_output_chart() {
        let instances = triangle_instances();
        let mut runtime = SurfaceRuntime::default();
        let translation = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0,
        ];
        let mobius_translation = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let snapshot = runtime
            .attach(
                &instances,
                1,
                &[7],
                0,
                [0.5, 0.25, 0.25],
                0.1,
                mobius_translation,
                translation,
            )
            .unwrap();
        assert_eq!(snapshot.status, "attached");
        let position = snapshot.output_position.unwrap();
        assert!((position[0] - 2.75).abs() < 1.0e-9);
        assert!((position[1] - 0.25).abs() < 1.0e-9);
    }

    #[test]
    fn runtime_cpu_skinning_matches_column_major_shader_order() {
        let instances = triangle_instances();
        let mut runtime = SurfaceRuntime::default();
        runtime.set_skinning(&[[0; 4]; 3], &[[1.0, 0.0, 0.0, 0.0]; 3]);
        runtime.set_pose(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0,
            ],
            &[],
        );
        let snapshot = runtime
            .attach(
                &instances,
                1,
                &[0],
                0,
                [0.5, 0.25, 0.25],
                0.1,
                [
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                ],
                [
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            )
            .unwrap();
        assert!((snapshot.output_position.unwrap()[2] - 2.0).abs() < 1.0e-9);
    }
}
