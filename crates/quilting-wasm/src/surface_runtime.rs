//! CPU-side surface samples for the backend-neutral Hyperscape walker.
//!
//! The renderer keeps the immutable 52-float face records; this module borrows
//! them rather than retaining a second chess-scale copy. Adjacency is built
//! lazily on first attachment. Posed control points follow the shader order:
//! morph, skin, ordinary affine model, rational QB evaluation, then Möbius.

use std::collections::BTreeMap;

use hyperscape::{
    AnimatedSurfaceAnchor, CameraRig, SphereReflectionState, StableEntityId, SurfaceAddress,
    SurfaceAdvance, SurfaceAttachment, SurfaceDetachReason, SurfaceField, SurfaceSample,
    SurfaceWalkAttachRequest, SurfaceWalkControls, SurfaceWalkInput, SurfaceWalkMetrics,
    SurfaceWalkRuntime, SurfaceWalkStepRequest, SurfaceWalkUpdate, SurfaceWalker,
    SurfaceWalkerStatus, TransitionEasing, TriangleAdjacency,
};
use quilting_core::instance_layout;
use quilting_core::patch::QBTriPatch;
use quilting_core::{Mobius, Quat};
use quilting_renderer::compute::LodAnimationSource;
use quilting_round_index::PatchControl;
use serde::Serialize;
use uuid::Uuid;

const MIN_POSE_SAMPLE_DELTA_SECONDS: f64 = 1.0e-9;

pub(crate) fn validate_pose_stamp(
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
) -> Result<(), String> {
    if !clip_time_seconds.is_finite() || !sample_time_seconds.is_finite() {
        return Err("animation pose clocks must be finite".into());
    }
    if revision == 0 {
        return Err("animation pose revision must be nonzero".into());
    }
    if continuity_epoch == 0 {
        return Err("animation pose continuity epoch must be nonzero".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TimedPoseSample {
    clip_time_seconds: f64,
    sample_time_seconds: f64,
    revision: u32,
    continuity_epoch: u32,
    continuous: bool,
}

/// Exact retained pose payload accepted by the main renderer.
///
/// Same-context LOD must borrow this payload rather than evaluating animation
/// independently or accepting a browser-owned matrix copy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LodPoseSource<'a> {
    pub joint_matrices: &'a [f32],
    pub morph_weights: &'a [f32],
    pub clip_time_seconds: f64,
    pub sample_time_seconds: f64,
    pub revision: u32,
    pub continuity_epoch: u32,
}

#[derive(Debug, Clone, Copy, Default)]
struct SurfaceVelocityTracker {
    previous_output_position: Option<[f64; 3]>,
    previous_pose_revision: Option<u32>,
    previous_pose_continuity_epoch: Option<u32>,
    previous_pose_sample_time_seconds: Option<f64>,
    velocity: [f64; 3],
    pose_sample_delta_seconds: Option<f64>,
    velocity_rebased: bool,
    suppress_once: bool,
}

impl SurfaceVelocityTracker {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn begin_chart_transport(&mut self, transported_position: Option<[f64; 3]>) {
        self.previous_output_position = transported_position;
        self.velocity = [0.0; 3];
        self.pose_sample_delta_seconds = None;
        self.velocity_rebased = transported_position.is_some();
        self.suppress_once = transported_position.is_some();
    }

    fn observe(
        &mut self,
        current_output_position: [f64; 3],
        pose: Option<TimedPoseSample>,
    ) -> [f64; 3] {
        if self.suppress_once {
            self.velocity = [0.0; 3];
            self.pose_sample_delta_seconds = None;
            self.velocity_rebased = true;
            return self.velocity;
        }
        let Some(pose) = pose else {
            self.velocity = [0.0; 3];
            self.pose_sample_delta_seconds = None;
            self.velocity_rebased = false;
            return self.velocity;
        };
        if self.previous_pose_revision == Some(pose.revision)
            && self.previous_pose_continuity_epoch == Some(pose.continuity_epoch)
        {
            self.velocity = [0.0; 3];
            self.pose_sample_delta_seconds = None;
            self.velocity_rebased = false;
            return self.velocity;
        }
        let sample_delta = self
            .previous_pose_sample_time_seconds
            .map(|previous| pose.sample_time_seconds - previous)
            .filter(|delta| delta.is_finite() && *delta > MIN_POSE_SAMPLE_DELTA_SECONDS);
        if pose.continuous && self.previous_pose_continuity_epoch == Some(pose.continuity_epoch) {
            if let (Some(previous), Some(sample_delta)) =
                (self.previous_output_position, sample_delta)
            {
                self.velocity = scale(sub(current_output_position, previous), 1.0 / sample_delta);
                self.pose_sample_delta_seconds = Some(sample_delta);
                self.velocity_rebased = false;
                return self.velocity;
            }
        }
        self.velocity = [0.0; 3];
        self.pose_sample_delta_seconds = None;
        self.velocity_rebased = true;
        self.velocity
    }

    fn commit(&mut self, output_position: Option<[f64; 3]>, pose: Option<TimedPoseSample>) {
        self.previous_output_position = output_position;
        self.previous_pose_revision = pose.map(|sample| sample.revision);
        self.previous_pose_continuity_epoch = pose.map(|sample| sample.continuity_epoch);
        self.previous_pose_sample_time_seconds = pose.map(|sample| sample.sample_time_seconds);
        self.suppress_once = false;
        if output_position.is_none() {
            self.velocity = [0.0; 3];
            self.pose_sample_delta_seconds = None;
            self.velocity_rebased = false;
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct SurfaceRuntime {
    walker: SurfaceWalker,
    composed: SurfaceWalkRuntime,
    camera_anchor: Option<AnimatedSurfaceAnchor>,
    adjacency: Option<TriangleAdjacency>,
    joint_indices: Vec<[u16; 4]>,
    joint_weights: Vec<[f32; 4]>,
    morph_deltas: Vec<f32>,
    morph_num_vertices: usize,
    morph_num_targets: usize,
    joint_matrices: Vec<f32>,
    morph_weights: Vec<f32>,
    pose_sample: Option<TimedPoseSample>,
    legacy_velocity: SurfaceVelocityTracker,
    composed_velocity: SurfaceVelocityTracker,
    camera_anchor_velocity: SurfaceVelocityTracker,
    legacy_eye_height: f64,
    attachment_orientation_sign: i8,
    composed_attachment_orientation_sign: i8,
    camera_anchor_orientation_sign: i8,
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
    pub pose_sample: Option<SurfacePoseSampleSnapshot>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct SurfacePoseSampleSnapshot {
    pub clip_time_seconds: f64,
    pub sample_time_seconds: f64,
    pub revision: u32,
    pub continuity_epoch: u32,
    pub continuous: bool,
    pub sample_delta_seconds: Option<f64>,
    pub velocity_rebased: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct SurfaceWalkCameraSnapshot {
    pub eye: [f64; 3],
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub forward: [f64; 3],
    pub control_distance: f64,
    pub semantic_target: Option<[f64; 3]>,
    pub vertical_fov_radians: f64,
    pub near: f64,
    pub far: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SurfaceCameraAnchorSnapshot {
    pub status: &'static str,
    pub detach_reason: Option<&'static str>,
    pub node: Option<u32>,
    pub face: Option<u32>,
    pub barycentric: Option<[f64; 3]>,
    pub output_position: Option<[f64; 3]>,
    pub output_normal: Option<[f64; 3]>,
    pub tangent_right: Option<[f64; 3]>,
    pub tangent_forward: Option<[f64; 3]>,
    pub surface_velocity: Option<[f64; 3]>,
    pub camera: SurfaceWalkCameraSnapshot,
    pub pose_sample: Option<SurfacePoseSampleSnapshot>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct SurfaceWalkMetricsSnapshot {
    pub body_scale: f64,
    pub radii_per_second: f64,
    pub speed: f64,
    pub eye_height: f64,
    pub near: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ComposedSurfaceWalkSnapshot {
    pub status: &'static str,
    pub phase: &'static str,
    pub detach_reason: Option<&'static str>,
    pub node: Option<u32>,
    pub face: Option<u32>,
    pub barycentric: Option<[f64; 3]>,
    pub output_position: Option<[f64; 3]>,
    pub output_normal: Option<[f64; 3]>,
    pub surface_velocity: Option<[f64; 3]>,
    pub projected_output_velocity: [f64; 3],
    pub desired_output_velocity: Option<[f64; 3]>,
    pub condition_number: Option<f64>,
    pub substeps: u32,
    pub edge_crossings: u32,
    pub camera: SurfaceWalkCameraSnapshot,
    pub target_camera: Option<SurfaceWalkCameraSnapshot>,
    pub filtered_position: Option<[f64; 3]>,
    pub filtered_normal: Option<[f64; 3]>,
    pub tangent_forward: Option<[f64; 3]>,
    pub relative_pitch_radians: Option<f64>,
    pub metrics: Option<SurfaceWalkMetricsSnapshot>,
    pub anchor_transition_remaining_seconds: Option<f64>,
    pub pose_sample: Option<SurfacePoseSampleSnapshot>,
}

/// Atomic renderer-adapter result for a change of spherical-reflection chart.
/// Both topology paths and their animation-history samples are committed
/// together, so the following posed frame cannot double-flip the attachment or
/// infer a spurious surface velocity from coordinates in two different charts.
#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct SurfaceWalkReflectionTransportSnapshot {
    pub legacy_attached: bool,
    pub composed_attached: bool,
    pub composed_follower_transported: bool,
    pub normal_side_flipped: bool,
    pub anchor_transition_cancelled: bool,
    pub legacy_previous_position_transported: bool,
    pub composed_previous_position_transported: bool,
}

impl SurfaceRuntime {
    pub fn reset_geometry(&mut self) {
        self.walker = SurfaceWalker::default();
        self.composed = SurfaceWalkRuntime::default();
        self.camera_anchor = None;
        self.adjacency = None;
        self.legacy_velocity.reset();
        self.composed_velocity.reset();
        self.camera_anchor_velocity.reset();
        self.legacy_eye_height = 0.0;
        self.attachment_orientation_sign = 1;
        self.composed_attachment_orientation_sign = 1;
        self.camera_anchor_orientation_sign = 1;
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

    /// Borrow the immutable animation payload already retained for walking and
    /// patch preparation. A context-local LOD classifier must consume this
    /// exact source instead of asking the browser to manufacture another copy.
    pub fn lod_animation_source(
        &self,
        expected_vertices: usize,
    ) -> Result<LodAnimationSource<'_>, String> {
        let skinning = match (self.joint_indices.len(), self.joint_weights.len()) {
            (0, 0) => None,
            (indices, weights) if indices == expected_vertices && weights == expected_vertices => {
                Some((self.joint_indices.as_slice(), self.joint_weights.as_slice()))
            }
            _ => return Err("renderer LOD skinning source does not match resident vertices".into()),
        };
        let expected_morph_scalars = self
            .morph_num_targets
            .checked_mul(expected_vertices)
            .and_then(|count| count.checked_mul(3))
            .ok_or_else(|| "renderer LOD morph source size overflow".to_string())?;
        if self.morph_deltas.len() != expected_morph_scalars
            || (self.morph_num_targets > 0 && self.morph_num_vertices != expected_vertices)
        {
            return Err("renderer LOD morph source does not match resident vertices".into());
        }
        Ok(LodAnimationSource {
            primary_vertices: expected_vertices,
            joint_indices: skinning.map(|(indices, _)| indices),
            joint_weights: skinning.map(|(_, weights)| weights),
            morph_deltas: &self.morph_deltas,
            num_morph_targets: self.morph_num_targets,
        })
    }

    /// Borrow the exact accepted renderer pose matching one classifier request.
    pub fn lod_pose_source(
        &self,
        clip_time_seconds: f64,
        sample_time_seconds: f64,
        revision: u32,
        continuity_epoch: u32,
    ) -> Result<LodPoseSource<'_>, String> {
        validate_pose_stamp(
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
        )?;
        let pose = self
            .pose_sample
            .ok_or_else(|| "renderer has no accepted animation pose".to_string())?;
        let exact = pose.clip_time_seconds.to_bits() == clip_time_seconds.to_bits()
            && pose.sample_time_seconds.to_bits() == sample_time_seconds.to_bits()
            && pose.revision == revision
            && pose.continuity_epoch == continuity_epoch;
        if !exact {
            return Err("LOD request does not match the accepted renderer pose".to_string());
        }
        if self.joint_matrices.len() % 16 != 0 {
            return Err("renderer joint matrix payload is malformed".to_string());
        }
        if self.morph_weights.len() > self.morph_num_targets {
            return Err("renderer morph weight payload exceeds resident targets".to_string());
        }
        Ok(LodPoseSource {
            joint_matrices: &self.joint_matrices,
            morph_weights: &self.morph_weights,
            clip_time_seconds: pose.clip_time_seconds,
            sample_time_seconds: pose.sample_time_seconds,
            revision: pose.revision,
            continuity_epoch: pose.continuity_epoch,
        })
    }

    #[cfg(test)]
    pub fn set_pose(&mut self, joint_matrices: &[f32], morph_weights: &[f32]) {
        self.joint_matrices.clear();
        self.joint_matrices.extend_from_slice(joint_matrices);
        self.morph_weights.clear();
        self.morph_weights.extend_from_slice(morph_weights);
        self.pose_sample = None;
        self.legacy_velocity.velocity = [0.0; 3];
        self.composed_velocity.velocity = [0.0; 3];
    }

    /// Install a renderer pose together with its exact clip time and a
    /// monotonic semantic sample clock. The clip clock may wrap normally; the
    /// sample clock must increase for a continuous animation interval. An
    /// ordinary model transform must remain stable between these samples;
    /// reflection chart edits use `transport_between_reflections` instead.
    pub fn set_timed_pose(
        &mut self,
        joint_matrices: &[f32],
        morph_weights: &[f32],
        clip_time_seconds: f64,
        sample_time_seconds: f64,
        revision: u32,
        continuity_epoch: u32,
    ) -> Result<bool, String> {
        validate_pose_stamp(
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
        )?;
        let previous_pose = self.pose_sample;
        let continuous = if let Some(previous) = previous_pose {
            if continuity_epoch < previous.continuity_epoch {
                return Ok(false);
            }
            if continuity_epoch == previous.continuity_epoch {
                if revision <= previous.revision {
                    return Ok(false);
                }
                if sample_time_seconds <= previous.sample_time_seconds {
                    return Err("animation pose sample time must increase within an epoch".into());
                }
                true
            } else {
                false
            }
        } else {
            false
        };
        self.joint_matrices.clear();
        self.joint_matrices.extend_from_slice(joint_matrices);
        self.morph_weights.clear();
        self.morph_weights.extend_from_slice(morph_weights);
        self.pose_sample = Some(TimedPoseSample {
            clip_time_seconds,
            sample_time_seconds,
            revision,
            continuity_epoch,
            continuous,
        });
        Ok(true)
    }

    pub fn clear_animation(&mut self) {
        self.joint_indices.clear();
        self.joint_weights.clear();
        self.morph_deltas.clear();
        self.morph_num_vertices = 0;
        self.morph_num_targets = 0;
        self.joint_matrices.clear();
        self.morph_weights.clear();
        self.pose_sample = None;
        self.legacy_velocity.reset();
        self.composed_velocity.reset();
        self.camera_anchor_velocity.reset();
        self.attachment_orientation_sign = 1;
        self.composed_attachment_orientation_sign = 1;
        self.camera_anchor_orientation_sign = 1;
    }

    /// Exact semantic pose identity currently backing CPU patch
    /// reconstruction. Static geometry has no stamp. Retained adaptive plans
    /// capture this pair so a later animation sample cannot silently reuse a
    /// partition measured against different controls.
    pub fn pose_stamp(&self) -> Option<(u32, u32)> {
        self.pose_sample
            .map(|sample| (sample.revision, sample.continuity_epoch))
    }

    pub fn attachment_face(&self) -> Option<u32> {
        self.walker
            .attachment()
            .map(|attachment| attachment.address.face)
    }

    pub fn composed_attachment_face(&self) -> Option<u32> {
        self.composed
            .attachment()
            .map(|attachment| attachment.address.face)
    }

    pub fn camera_anchor_face(&self) -> Option<u32> {
        self.camera_anchor
            .map(|anchor| anchor.attachment().address.face)
    }

    /// Carry every output-chart surface-walk cache through
    /// `next ∘ previous⁻¹` as one transaction. Large immutable geometry and
    /// pose buffers remain borrowed by the runtime and are never cloned here.
    pub fn transport_between_reflections(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
    ) -> Result<SurfaceWalkReflectionTransportSnapshot, String> {
        let legacy_attached = self.walker.attachment().is_some();
        if previous == next {
            return Ok(SurfaceWalkReflectionTransportSnapshot {
                legacy_attached,
                composed_attached: self.composed.is_active(),
                composed_follower_transported: false,
                normal_side_flipped: false,
                anchor_transition_cancelled: false,
                legacy_previous_position_transported: false,
                composed_previous_position_transported: false,
            });
        }
        let parity_changed = previous.orientation_sign() != next.orientation_sign();
        let mut legacy_walker = self.walker.clone();
        if legacy_attached && parity_changed {
            legacy_walker.flip_normal_side();
        }

        let mut composed = self.composed.clone();
        let composed_outcome = composed
            .transport_between_reflections(previous, next)
            .map_err(|error| error.to_string())?;
        let legacy_previous_position = transport_optional_position(
            self.legacy_velocity.previous_output_position,
            previous,
            next,
        )?;
        let composed_previous_position = transport_optional_position(
            self.composed_velocity.previous_output_position,
            previous,
            next,
        )?;
        let orientation_delta = previous.orientation_sign() * next.orientation_sign();

        self.walker = legacy_walker;
        self.composed = composed;
        self.legacy_velocity
            .begin_chart_transport(legacy_previous_position);
        self.composed_velocity
            .begin_chart_transport(composed_previous_position);
        self.attachment_orientation_sign *= orientation_delta;
        self.composed_attachment_orientation_sign *= orientation_delta;

        Ok(SurfaceWalkReflectionTransportSnapshot {
            legacy_attached,
            composed_attached: composed_outcome.attached,
            composed_follower_transported: composed_outcome.follower_transported,
            normal_side_flipped: (legacy_attached && parity_changed)
                || composed_outcome.normal_side_flipped,
            anchor_transition_cancelled: composed_outcome.anchor_transition_cancelled,
            legacy_previous_position_transported: self
                .legacy_velocity
                .previous_output_position
                .is_some(),
            composed_previous_position_transported: self
                .composed_velocity
                .previous_output_position
                .is_some(),
        })
    }

    /// Reconstruct the exact posed source controls used by the GPU animation
    /// path, without applying an ordinary model matrix or Möbius map. This is
    /// the shared CPU boundary for surface walking and animated spatial-index
    /// refits; callers provide the pose captured with their own GPU job so an
    /// asynchronous result is never compared against a newer render pose.
    pub fn patch_controls_for_pose(
        &self,
        instances: &[f32],
        num_faces: usize,
        joint_matrices: &[f32],
        morph_weights: &[f32],
    ) -> Result<Vec<PatchControl>, String> {
        let required = num_faces
            .checked_mul(instance_layout::STRIDE)
            .ok_or_else(|| "posed patch buffer size overflow".to_string())?;
        if instances.len() < required {
            return Err(format!(
                "posed patch buffer has {} floats; expected at least {required}",
                instances.len()
            ));
        }
        let pose = PoseData {
            joint_indices: &self.joint_indices,
            joint_weights: &self.joint_weights,
            morph_deltas: &self.morph_deltas,
            morph_num_vertices: self.morph_num_vertices,
            morph_num_targets: self.morph_num_targets,
            joint_matrices,
            morph_weights,
        };
        (0..num_faces)
            .map(|face| {
                let record = &instances
                    [face * instance_layout::STRIDE..(face + 1) * instance_layout::STRIDE];
                let mut positions = [[0.0; 3]; 3];
                let mut weights = [[0.0; 4]; 3];
                for corner in 0..3 {
                    let position_offset = instance_layout::offset::POSITIONS + corner * 4;
                    let vertex = checked_vertex_index(record[position_offset])?;
                    let rest = [
                        record[position_offset + 1] as f64,
                        record[position_offset + 2] as f64,
                        record[position_offset + 3] as f64,
                    ];
                    positions[corner] = pose
                        .posed_position(vertex, rest)
                        .ok_or_else(|| format!("could not pose face {face} corner {corner}"))?;
                    let weight_offset = instance_layout::offset::WEIGHTS + corner * 4;
                    weights[corner] = std::array::from_fn(|component| {
                        f64::from(record[weight_offset + component])
                    });
                }
                Ok(PatchControl {
                    face: face as u32,
                    positions,
                    weights,
                })
            })
            .collect()
    }

    /// Reconstruct one patch in the exact output chart consumed by rendering.
    ///
    /// This deliberately follows the shader order—current morph/skin pose,
    /// ordinary affine model, then Möbius transformation—so camera-dependent
    /// CPU oracles can measure the same surface without rebuilding every face.
    pub fn output_patch_for_face(
        &self,
        instances: &[f32],
        face: usize,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<QBTriPatch, String> {
        let pose = PoseData {
            joint_indices: &self.joint_indices,
            joint_weights: &self.joint_weights,
            morph_deltas: &self.morph_deltas,
            morph_num_vertices: self.morph_num_vertices,
            morph_num_targets: self.morph_num_targets,
            joint_matrices: &self.joint_matrices,
            morph_weights: &self.morph_weights,
        };
        output_patch_for_pose(
            instances,
            face,
            pose,
            mobius_from_array(mobius),
            euclidean_model,
        )
        .ok_or_else(|| format!("could not reconstruct output patch for face {face}"))
    }

    pub fn attach(
        &mut self,
        instances: &[f32],
        num_faces: usize,
        face_nodes: &[usize],
        face: u32,
        barycentric: [f64; 3],
        eye_height: f64,
        camera_position: [f64; 3],
        orientation_sign: i8,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<SurfaceRuntimeSnapshot, String> {
        if face as usize >= num_faces || face_nodes.len() != num_faces {
            return Err("surface face is outside the current model".into());
        }
        if !eye_height.is_finite() {
            return Err("surface eye height must be finite".into());
        }
        if self.adjacency.is_none() {
            self.adjacency = Some(build_adjacency(instances, num_faces, face_nodes)?);
        }
        let node = face_nodes[face as usize];
        let entity = runtime_entity_id(node);
        let address = SurfaceAddress::new(entity, face, barycentric)
            .map_err(|error| format!("invalid surface address: {error:?}"))?;
        let normal_sign = {
            let adjacency = self
                .adjacency
                .as_ref()
                .ok_or_else(|| "surface topology is unavailable".to_string())?;
            let mut field = RuntimeSurfaceField {
                instances,
                face_nodes,
                adjacency,
                pose: PoseData {
                    joint_indices: &self.joint_indices,
                    joint_weights: &self.joint_weights,
                    morph_deltas: &self.morph_deltas,
                    morph_num_vertices: self.morph_num_vertices,
                    morph_num_targets: self.morph_num_targets,
                    joint_matrices: &self.joint_matrices,
                    morph_weights: &self.morph_weights,
                },
                mobius: mobius_from_array(mobius),
                euclidean_model,
                surface_velocity: [0.0; 3],
            };
            let sample = field
                .sample(address)
                .ok_or_else(|| "could not sample attached surface".to_string())?;
            let normal = sample
                .normal()
                .ok_or_else(|| "attached surface has no stable normal".to_string())?;
            if dot(sub(camera_position, sample.output_position), normal) < 0.0 {
                -1
            } else {
                1
            }
        };
        let attachment = SurfaceAttachment::with_normal_sign(address, normal_sign)
            .map_err(|error| format!("invalid surface attachment: {error:?}"))?;
        self.camera_anchor = None;
        self.camera_anchor_velocity.reset();
        self.camera_anchor_orientation_sign = 1;
        self.walker.attach(attachment);
        self.legacy_velocity.reset();
        self.legacy_eye_height = eye_height;
        self.attachment_orientation_sign = normalize_orientation_sign(orientation_sign);
        self.step_relative(
            instances,
            face_nodes,
            0.0,
            [0.0; 3],
            orientation_sign,
            mobius,
            euclidean_model,
        )
    }

    pub fn detach(&mut self) -> SurfaceRuntimeSnapshot {
        self.walker.detach(SurfaceDetachReason::Manual);
        self.composed.detach(SurfaceDetachReason::Manual);
        self.camera_anchor = None;
        self.legacy_velocity.reset();
        self.composed_velocity.reset();
        self.camera_anchor_velocity.reset();
        self.legacy_eye_height = 0.0;
        self.attachment_orientation_sign = 1;
        self.composed_attachment_orientation_sign = 1;
        self.camera_anchor_orientation_sign = 1;
        detached_snapshot(SurfaceDetachReason::Manual)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attach_camera_anchor(
        &mut self,
        instances: &[f32],
        num_faces: usize,
        face_nodes: &[usize],
        face: u32,
        barycentric: [f64; 3],
        camera: CameraRig,
        orientation_sign: i8,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<SurfaceCameraAnchorSnapshot, String> {
        if face as usize >= num_faces || face_nodes.len() != num_faces {
            return Err("surface face is outside the current model".into());
        }
        camera.validate().map_err(|error| error.to_string())?;
        if self.adjacency.is_none() {
            self.adjacency = Some(build_adjacency(instances, num_faces, face_nodes)?);
        }
        let node = face_nodes[face as usize];
        let address = SurfaceAddress::new(runtime_entity_id(node), face, barycentric)
            .map_err(|error| format!("invalid surface address: {error:?}"))?;
        let adjacency = self
            .adjacency
            .as_ref()
            .ok_or_else(|| "surface topology is unavailable".to_string())?;
        let mut field = RuntimeSurfaceField {
            instances,
            face_nodes,
            adjacency,
            pose: PoseData {
                joint_indices: &self.joint_indices,
                joint_weights: &self.joint_weights,
                morph_deltas: &self.morph_deltas,
                morph_num_vertices: self.morph_num_vertices,
                morph_num_targets: self.morph_num_targets,
                joint_matrices: &self.joint_matrices,
                morph_weights: &self.morph_weights,
            },
            mobius: mobius_from_array(mobius),
            euclidean_model,
            surface_velocity: [0.0; 3],
        };
        let sample = field
            .sample(address)
            .ok_or_else(|| "could not sample camera anchor".to_string())?;
        let normal = sample
            .normal()
            .ok_or_else(|| "camera anchor has no stable normal".to_string())?;
        let normal_sign = if dot(sub(camera.eye, sample.output_position), normal) < 0.0 {
            -1
        } else {
            1
        };
        let attachment = SurfaceAttachment::with_normal_sign(address, normal_sign)
            .map_err(|error| format!("invalid surface attachment: {error:?}"))?;
        let anchor = AnimatedSurfaceAnchor::attach(attachment, sample, camera)
            .map_err(|error| error.to_string())?;

        self.walker.detach(SurfaceDetachReason::Manual);
        self.composed.detach(SurfaceDetachReason::Manual);
        self.legacy_velocity.reset();
        self.composed_velocity.reset();
        self.camera_anchor_velocity.reset();
        self.camera_anchor_velocity
            .commit(Some(sample.output_position), self.pose_sample);
        self.camera_anchor = Some(anchor);
        self.camera_anchor_orientation_sign = normalize_orientation_sign(orientation_sign);
        Ok(camera_anchor_snapshot(
            anchor,
            sample,
            camera,
            node,
            pose_sample_snapshot(self.pose_sample, self.camera_anchor_velocity),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_camera_anchor(
        &mut self,
        instances: &[f32],
        face_nodes: &[usize],
        camera: CameraRig,
        capture_relative_camera: bool,
        orientation_sign: i8,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<SurfaceCameraAnchorSnapshot, String> {
        let next_orientation_sign = normalize_orientation_sign(orientation_sign);
        if self.camera_anchor_orientation_sign != next_orientation_sign {
            return Err("camera anchor reflection transport is required before sampling".into());
        }
        let mut anchor = self
            .camera_anchor
            .ok_or_else(|| "surface camera anchor is detached".to_string())?;
        if capture_relative_camera {
            anchor
                .recapture_camera(camera)
                .map_err(|error| error.to_string())?;
        }
        let address = anchor.attachment().address;
        let node = *face_nodes
            .get(address.face as usize)
            .ok_or_else(|| "surface face has no node identity".to_string())?;
        let adjacency = self
            .adjacency
            .as_ref()
            .ok_or_else(|| "surface topology is unavailable".to_string())?;
        let mut field = RuntimeSurfaceField {
            instances,
            face_nodes,
            adjacency,
            pose: PoseData {
                joint_indices: &self.joint_indices,
                joint_weights: &self.joint_weights,
                morph_deltas: &self.morph_deltas,
                morph_num_vertices: self.morph_num_vertices,
                morph_num_targets: self.morph_num_targets,
                joint_matrices: &self.joint_matrices,
                morph_weights: &self.morph_weights,
            },
            mobius: mobius_from_array(mobius),
            euclidean_model,
            surface_velocity: [0.0; 3],
        };
        let initial_sample = field
            .sample(address)
            .ok_or_else(|| "could not sample camera anchor".to_string())?;
        let mut velocity_tracker = self.camera_anchor_velocity;
        field.surface_velocity =
            velocity_tracker.observe(initial_sample.output_position, self.pose_sample);
        let sample = field
            .sample(address)
            .ok_or_else(|| "could not resample camera anchor velocity".to_string())?;
        let followed = anchor
            .follow_sample(sample)
            .map_err(|error| error.to_string())?;
        velocity_tracker.commit(Some(sample.output_position), self.pose_sample);

        self.camera_anchor = Some(anchor);
        self.camera_anchor_velocity = velocity_tracker;
        Ok(camera_anchor_snapshot(
            anchor,
            sample,
            followed.camera,
            node,
            pose_sample_snapshot(self.pose_sample, self.camera_anchor_velocity),
        ))
    }

    pub fn transport_camera_anchor_between_reflections(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
        transported_camera: CameraRig,
    ) -> Result<bool, String> {
        let Some(mut anchor) = self.camera_anchor else {
            return Ok(false);
        };
        let previous_position = transport_optional_position(
            self.camera_anchor_velocity.previous_output_position,
            previous,
            next,
        )?;
        anchor
            .rebase_after_reflection_transport(previous, next, transported_camera)
            .map_err(|error| error.to_string())?;
        self.camera_anchor = Some(anchor);
        self.camera_anchor_velocity
            .begin_chart_transport(previous_position);
        self.camera_anchor_orientation_sign *=
            previous.orientation_sign() * next.orientation_sign();
        Ok(true)
    }

    pub fn detach_camera_anchor(&mut self) -> bool {
        let detached = self.camera_anchor.take().is_some();
        self.camera_anchor_velocity.reset();
        self.camera_anchor_orientation_sign = 1;
        detached
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attach_composed(
        &mut self,
        instances: &[f32],
        num_faces: usize,
        face_nodes: &[usize],
        face: u32,
        barycentric: [f64; 3],
        camera: CameraRig,
        scene_radius: f64,
        controls: SurfaceWalkControls,
        transition_duration_seconds: f64,
        orientation_sign: i8,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<ComposedSurfaceWalkSnapshot, String> {
        if face as usize >= num_faces || face_nodes.len() != num_faces {
            return Err("surface face is outside the current model".into());
        }
        if self.adjacency.is_none() {
            self.adjacency = Some(build_adjacency(instances, num_faces, face_nodes)?);
        }
        let node = face_nodes[face as usize];
        let address = SurfaceAddress::new(runtime_entity_id(node), face, barycentric)
            .map_err(|error| format!("invalid surface address: {error:?}"))?;
        let adjacency = self
            .adjacency
            .as_ref()
            .ok_or_else(|| "surface topology is unavailable".to_string())?;
        let mut field = RuntimeSurfaceField {
            instances,
            face_nodes,
            adjacency,
            pose: PoseData {
                joint_indices: &self.joint_indices,
                joint_weights: &self.joint_weights,
                morph_deltas: &self.morph_deltas,
                morph_num_vertices: self.morph_num_vertices,
                morph_num_targets: self.morph_num_targets,
                joint_matrices: &self.joint_matrices,
                morph_weights: &self.morph_weights,
            },
            mobius: mobius_from_array(mobius),
            euclidean_model,
            surface_velocity: [0.0; 3],
        };
        let mut camera = camera;
        let update = self
            .composed
            .attach(
                &mut camera,
                SurfaceWalkAttachRequest {
                    address,
                    normal_sign: None,
                    scene_radius,
                    controls,
                    transition_duration_seconds,
                    transition_easing: TransitionEasing::SmootherStep,
                },
                &mut field,
            )
            .map_err(|error| error.to_string())?;
        self.camera_anchor = None;
        self.camera_anchor_velocity.reset();
        self.camera_anchor_orientation_sign = 1;
        self.composed_velocity.reset();
        self.composed_velocity.commit(
            update
                .advance
                .contact
                .map(|contact| contact.output_position),
            self.pose_sample,
        );
        self.composed_attachment_orientation_sign = normalize_orientation_sign(orientation_sign);
        Ok(composed_snapshot(
            update,
            node,
            pose_sample_snapshot(self.pose_sample, self.composed_velocity),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_composed(
        &mut self,
        instances: &[f32],
        face_nodes: &[usize],
        delta_seconds: f64,
        camera: CameraRig,
        scene_radius: f64,
        controls: SurfaceWalkControls,
        input: SurfaceWalkInput,
        orient: bool,
        capture_relative_view: bool,
        orientation_sign: i8,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<ComposedSurfaceWalkSnapshot, String> {
        let next_orientation_sign = normalize_orientation_sign(orientation_sign);
        let mut candidate = self.composed.clone();
        if self.composed_attachment_orientation_sign != next_orientation_sign {
            candidate.flip_normal_side();
        }
        let attachment = candidate
            .attachment()
            .ok_or_else(|| "composed surface walker is detached".to_string())?;
        let Some(&node) = face_nodes.get(attachment.address.face as usize) else {
            return Ok(self.commit_composed_detach(
                candidate,
                camera,
                next_orientation_sign,
                SurfaceDetachReason::SampleUnavailable,
            ));
        };
        let Some(adjacency) = self.adjacency.as_ref() else {
            return Ok(self.commit_composed_detach(
                candidate,
                camera,
                next_orientation_sign,
                SurfaceDetachReason::SampleUnavailable,
            ));
        };
        let mut field = RuntimeSurfaceField {
            instances,
            face_nodes,
            adjacency,
            pose: PoseData {
                joint_indices: &self.joint_indices,
                joint_weights: &self.joint_weights,
                morph_deltas: &self.morph_deltas,
                morph_num_vertices: self.morph_num_vertices,
                morph_num_targets: self.morph_num_targets,
                joint_matrices: &self.joint_matrices,
                morph_weights: &self.morph_weights,
            },
            mobius: mobius_from_array(mobius),
            euclidean_model,
            surface_velocity: [0.0; 3],
        };
        let mut velocity_tracker = self.composed_velocity;
        if let Some(current_start) = field.sample(attachment.address) {
            field.surface_velocity =
                velocity_tracker.observe(current_start.output_position, self.pose_sample);
        }
        let mut camera = camera;
        let update = candidate
            .step(
                &mut camera,
                SurfaceWalkStepRequest {
                    delta_seconds,
                    scene_radius,
                    controls,
                    input,
                    orient,
                    capture_relative_view,
                },
                &mut field,
            )
            .map_err(|error| error.to_string())?;
        self.composed = candidate;
        self.composed_attachment_orientation_sign = next_orientation_sign;
        velocity_tracker.commit(
            update
                .advance
                .contact
                .map(|contact| contact.output_position),
            self.pose_sample,
        );
        self.composed_velocity = velocity_tracker;
        Ok(composed_snapshot(
            update,
            node,
            pose_sample_snapshot(self.pose_sample, self.composed_velocity),
        ))
    }

    fn commit_composed_detach(
        &mut self,
        mut candidate: SurfaceWalkRuntime,
        camera: CameraRig,
        orientation_sign: i8,
        reason: SurfaceDetachReason,
    ) -> ComposedSurfaceWalkSnapshot {
        candidate.detach(reason);
        self.composed = candidate;
        self.composed_velocity.reset();
        self.composed_attachment_orientation_sign = orientation_sign;
        composed_snapshot(
            SurfaceWalkUpdate {
                advance: detached_advance(reason),
                motion: None,
                target_frame: None,
                camera,
                anchor_transition_remaining_seconds: None,
            },
            0,
            None,
        )
    }

    pub fn step_relative(
        &mut self,
        instances: &[f32],
        face_nodes: &[usize],
        delta_seconds: f64,
        relative_output_velocity: [f64; 3],
        orientation_sign: i8,
        mobius: [f32; 16],
        euclidean_model: [f32; 16],
    ) -> Result<SurfaceRuntimeSnapshot, String> {
        let next_orientation_sign = normalize_orientation_sign(orientation_sign);
        if self.attachment_orientation_sign != next_orientation_sign {
            self.walker.flip_normal_side();
            self.attachment_orientation_sign = next_orientation_sign;
        }
        let attachment = self
            .walker
            .attachment()
            .ok_or_else(|| "surface walker is detached".to_string())?;
        let node = *face_nodes
            .get(attachment.address.face as usize)
            .ok_or_else(|| "surface face has no node identity".to_string())?;
        let adjacency = self
            .adjacency
            .as_ref()
            .ok_or_else(|| "surface topology is unavailable".to_string())?;

        let mut field = RuntimeSurfaceField {
            instances,
            face_nodes,
            adjacency,
            pose: PoseData {
                joint_indices: &self.joint_indices,
                joint_weights: &self.joint_weights,
                morph_deltas: &self.morph_deltas,
                morph_num_vertices: self.morph_num_vertices,
                morph_num_targets: self.morph_num_targets,
                joint_matrices: &self.joint_matrices,
                morph_weights: &self.morph_weights,
            },
            mobius: mobius_from_array(mobius),
            euclidean_model,
            surface_velocity: [0.0; 3],
        };
        let current_start = field
            .sample(attachment.address)
            .ok_or_else(|| "could not sample attached surface".to_string())?;
        let mut velocity_tracker = self.legacy_velocity;
        field.surface_velocity =
            velocity_tracker.observe(current_start.output_position, self.pose_sample);
        let desired_absolute_velocity = add(relative_output_velocity, field.surface_velocity);
        let advance = self
            .walker
            .advance(delta_seconds, desired_absolute_velocity, &mut field);
        velocity_tracker.commit(
            advance.contact.map(|contact| contact.output_position),
            self.pose_sample,
        );
        self.legacy_velocity = velocity_tracker;
        Ok(snapshot_from_advance(
            advance,
            node,
            self.legacy_eye_height,
            pose_sample_snapshot(self.pose_sample, self.legacy_velocity),
        ))
    }
}

struct RuntimeSurfaceField<'a> {
    instances: &'a [f32],
    face_nodes: &'a [usize],
    adjacency: &'a TriangleAdjacency,
    pose: PoseData<'a>,
    mobius: Mobius,
    euclidean_model: [f32; 16],
    surface_velocity: [f64; 3],
}

impl SurfaceField for RuntimeSurfaceField<'_> {
    fn sample(&mut self, address: SurfaceAddress) -> Option<SurfaceSample> {
        let patch = self.patch(address.face as usize)?;
        let u = address.barycentric[1];
        let v = address.barycentric[2];
        let differential = patch.eval_differential(u, v);
        let point = differential.position;
        let tangent_u = differential.tangent_u;
        let tangent_v = differential.tangent_v;
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
        output_patch_for_pose(
            self.instances,
            face,
            self.pose,
            self.mobius,
            self.euclidean_model,
        )
    }
}

fn output_patch_for_pose(
    instances: &[f32],
    face: usize,
    pose: PoseData<'_>,
    mobius: Mobius,
    euclidean_model: [f32; 16],
) -> Option<QBTriPatch> {
    let base = face.checked_mul(instance_layout::STRIDE)?;
    let record = instances.get(base..base + instance_layout::STRIDE)?;
    let mut positions = [Quat::ZERO; 3];
    let mut weights = [Quat::ZERO; 3];
    for corner in 0..3 {
        let position_offset = instance_layout::offset::POSITIONS + corner * 4;
        let vertex = checked_vertex_index(record[position_offset]).ok()?;
        let rest = [
            record[position_offset + 1] as f64,
            record[position_offset + 2] as f64,
            record[position_offset + 3] as f64,
        ];
        let posed = pose.posed_position(vertex, rest)?;
        let modeled = apply_affine(euclidean_model, posed)?;
        positions[corner] = Quat::from_point(modeled[0], modeled[1], modeled[2]);

        let weight_offset = instance_layout::offset::WEIGHTS + corner * 4;
        weights[corner] = Quat::new(
            record[weight_offset] as f64,
            record[weight_offset + 1] as f64,
            record[weight_offset + 2] as f64,
            record[weight_offset + 3] as f64,
        );
    }
    Some(QBTriPatch::new(positions, weights).transform(&mobius))
}

#[derive(Clone, Copy)]
struct PoseData<'a> {
    joint_indices: &'a [[u16; 4]],
    joint_weights: &'a [[f32; 4]],
    morph_deltas: &'a [f32],
    morph_num_vertices: usize,
    morph_num_targets: usize,
    joint_matrices: &'a [f32],
    morph_weights: &'a [f32],
}

impl PoseData<'_> {
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
        let mut applied_weight = 0.0;
        for influence in 0..4 {
            let weight = weights[influence] as f64;
            let joint = indices[influence] as usize;
            if weight < 1.0e-6 || joint >= num_joints {
                continue;
            }
            applied_weight += weight;
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
        let output = if applied_weight > 1.0e-6 {
            skinned
        } else {
            position
        };
        finite3(output).then_some(output)
    }
}

fn checked_vertex_index(vertex: f32) -> Result<usize, String> {
    if vertex.is_finite() && vertex >= 0.0 && vertex.fract() == 0.0 {
        Ok(vertex as usize)
    } else {
        Err("patch has an invalid vertex identity".to_string())
    }
}

fn build_adjacency(
    instances: &[f32],
    num_faces: usize,
    face_nodes: &[usize],
) -> Result<TriangleAdjacency, String> {
    let required = num_faces
        .checked_mul(instance_layout::STRIDE)
        .ok_or_else(|| "surface topology size overflow".to_string())?;
    if instances.len() < required || face_nodes.len() != num_faces {
        return Err("surface instance data is truncated".into());
    }
    let mut faces = Vec::with_capacity(num_faces);
    let mut welded_vertices = BTreeMap::<(usize, [u32; 3]), u64>::new();
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
            vertices[corner] = *welded_vertices
                .entry((face_nodes[face], key))
                .or_insert_with(|| {
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

fn snapshot_from_advance(
    advance: SurfaceAdvance,
    node: usize,
    eye_height: f64,
    pose_sample: Option<SurfacePoseSampleSnapshot>,
) -> SurfaceRuntimeSnapshot {
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
        eye_position: advance.contact.map(|contact| {
            add(
                contact.output_position,
                scale(contact.output_normal, eye_height),
            )
        }),
        projected_output_velocity: advance.projected_output_velocity,
        condition_number: advance.condition_number,
        substeps: advance.substeps,
        edge_crossings: advance.edge_crossings,
        pose_sample,
    }
}

fn composed_snapshot(
    update: SurfaceWalkUpdate,
    node: usize,
    pose_sample: Option<SurfacePoseSampleSnapshot>,
) -> ComposedSurfaceWalkSnapshot {
    let contact = update.advance.contact;
    let detach_reason = match update.advance.status {
        SurfaceWalkerStatus::Attached => None,
        SurfaceWalkerStatus::Detached(reason) => Some(detach_reason_name(reason)),
    };
    let transition_active = update.anchor_transition_remaining_seconds.is_some();
    ComposedSurfaceWalkSnapshot {
        status: if detach_reason.is_some() {
            "detached"
        } else {
            "attached"
        },
        phase: if detach_reason.is_some() {
            "detached"
        } else if transition_active {
            "anchoring"
        } else {
            "walking"
        },
        detach_reason,
        node: contact.map(|_| node as u32),
        face: contact.map(|value| value.address.face),
        barycentric: contact.map(|value| value.address.barycentric),
        output_position: contact.map(|value| value.output_position),
        output_normal: contact.map(|value| value.output_normal),
        surface_velocity: contact.map(|value| value.surface_velocity),
        projected_output_velocity: update.advance.projected_output_velocity,
        desired_output_velocity: update.motion.map(|motion| motion.desired_output_velocity),
        condition_number: (update.advance.status == SurfaceWalkerStatus::Attached)
            .then_some(update.advance.condition_number),
        substeps: update.advance.substeps,
        edge_crossings: update.advance.edge_crossings,
        camera: camera_snapshot(update.camera),
        target_camera: update
            .target_frame
            .map(|frame| camera_snapshot(frame.camera)),
        filtered_position: update.target_frame.map(|frame| frame.filtered_position),
        filtered_normal: update.target_frame.map(|frame| frame.filtered_normal),
        tangent_forward: update.target_frame.and_then(|frame| frame.tangent_forward),
        relative_pitch_radians: update
            .target_frame
            .and_then(|frame| frame.relative_pitch_radians),
        metrics: update
            .target_frame
            .map(|frame| metrics_snapshot(frame.metrics)),
        anchor_transition_remaining_seconds: update.anchor_transition_remaining_seconds,
        pose_sample,
    }
}

fn camera_anchor_snapshot(
    anchor: AnimatedSurfaceAnchor,
    sample: SurfaceSample,
    camera: CameraRig,
    node: usize,
    pose_sample: Option<SurfacePoseSampleSnapshot>,
) -> SurfaceCameraAnchorSnapshot {
    let frame = anchor.frame();
    let attachment = anchor.attachment();
    SurfaceCameraAnchorSnapshot {
        status: "attached",
        detach_reason: None,
        node: Some(node as u32),
        face: Some(attachment.address.face),
        barycentric: Some(attachment.address.barycentric),
        output_position: Some(frame.origin),
        output_normal: Some(frame.normal),
        tangent_right: Some(frame.tangent_right),
        tangent_forward: Some(frame.tangent_forward),
        surface_velocity: Some(sample.surface_velocity),
        camera: camera_snapshot(camera),
        pose_sample,
    }
}

fn pose_sample_snapshot(
    pose: Option<TimedPoseSample>,
    tracker: SurfaceVelocityTracker,
) -> Option<SurfacePoseSampleSnapshot> {
    pose.map(|sample| SurfacePoseSampleSnapshot {
        clip_time_seconds: sample.clip_time_seconds,
        sample_time_seconds: sample.sample_time_seconds,
        revision: sample.revision,
        continuity_epoch: sample.continuity_epoch,
        continuous: sample.continuous,
        sample_delta_seconds: tracker.pose_sample_delta_seconds,
        velocity_rebased: tracker.velocity_rebased,
    })
}

fn camera_snapshot(camera: CameraRig) -> SurfaceWalkCameraSnapshot {
    let basis = camera.basis();
    SurfaceWalkCameraSnapshot {
        eye: camera.eye,
        right: basis.right,
        up: basis.up,
        forward: basis.forward,
        control_distance: camera.control_distance,
        semantic_target: camera.semantic_target,
        vertical_fov_radians: camera.lens.vertical_fov_radians,
        near: camera.lens.near,
        far: camera.lens.far,
    }
}

fn metrics_snapshot(metrics: SurfaceWalkMetrics) -> SurfaceWalkMetricsSnapshot {
    SurfaceWalkMetricsSnapshot {
        body_scale: metrics.body_scale,
        radii_per_second: metrics.radii_per_second,
        speed: metrics.speed,
        eye_height: metrics.eye_height,
        near: metrics.near,
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
        pose_sample: None,
    }
}

fn detached_advance(reason: SurfaceDetachReason) -> SurfaceAdvance {
    SurfaceAdvance {
        status: SurfaceWalkerStatus::Detached(reason),
        contact: None,
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

fn normalize_orientation_sign(sign: i8) -> i8 {
    if sign < 0 {
        -1
    } else {
        1
    }
}

fn transport_optional_position(
    position: Option<[f64; 3]>,
    previous: SphereReflectionState,
    next: SphereReflectionState,
) -> Result<Option<[f64; 3]>, String> {
    position
        .map(|position| {
            previous
                .transport_point_and_directions::<0>(next, position, [])
                .map(|transport| transport.point)
                .map_err(|error| error.to_string())
        })
        .transpose()
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

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
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
    use hyperscape::{CameraBasis, FocusSphere, PerspectiveLens};
    use wasm_bindgen_test::wasm_bindgen_test;

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

    fn identity_matrix() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn identity_mobius() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]
    }

    fn sphere_reflection_mobius(center: [f64; 3], radius: f64) -> [f32; 16] {
        let center_squared = center.into_iter().map(|value| value * value).sum::<f64>();
        [
            0.0,
            center[0] as f32,
            center[1] as f32,
            center[2] as f32,
            (center_squared - radius * radius) as f32,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -center[0] as f32,
            -center[1] as f32,
            -center[2] as f32,
        ]
    }

    const QUAD_POINTS: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];

    fn two_triangle_instances(faces: [[usize; 3]; 2]) -> Vec<f32> {
        let mut instances = vec![0.0; 2 * instance_layout::STRIDE];
        for (face_index, face) in faces.into_iter().enumerate() {
            let base = face_index * instance_layout::STRIDE;
            for (corner, vertex) in face.into_iter().enumerate() {
                let position_offset = base + instance_layout::offset::POSITIONS + corner * 4;
                instances[position_offset] = vertex as f32;
                instances[position_offset + 1..position_offset + 4]
                    .copy_from_slice(&QUAD_POINTS[vertex]);
                instances[base + instance_layout::offset::WEIGHTS + corner * 4] = 1.0;
            }
        }
        instances
    }

    fn face_barycentric(face: [usize; 3], vertex_weights: [f32; 4]) -> [f64; 3] {
        face.map(|vertex| f64::from(vertex_weights[vertex]))
    }

    fn vertex_weights(face: [usize; 3], barycentric: [f64; 3]) -> [f64; 4] {
        let mut weights = [0.0; 4];
        for (corner, vertex) in face.into_iter().enumerate() {
            weights[vertex] = barycentric[corner];
        }
        weights
    }

    fn camera_crossing_shared_edge(
        face: [usize; 3],
        barycentric: [f64; 3],
        mobius: [f32; 16],
    ) -> CameraRig {
        let patch = QBTriPatch::flat(
            QUAD_POINTS[face[0]].map(f64::from),
            QUAD_POINTS[face[1]].map(f64::from),
            QUAD_POINTS[face[2]].map(f64::from),
        )
        .transform(&mobius_from_array(mobius));
        let address = SurfaceAddress::new(runtime_entity_id(0), 0, barycentric).unwrap();
        let differential = patch.eval_differential(address.barycentric[1], address.barycentric[2]);
        let vertex_rates = [-2.0, 1.0, 1.0, 0.0];
        let local_rates = face.map(|vertex| vertex_rates[vertex]);
        let forward = add(
            scale(differential.tangent_u, local_rates[1]),
            scale(differential.tangent_v, local_rates[2]),
        );
        let forward = scale(forward, 1.0 / length(forward));
        let normal = [
            differential.tangent_u[1] * differential.tangent_v[2]
                - differential.tangent_u[2] * differential.tangent_v[1],
            differential.tangent_u[2] * differential.tangent_v[0]
                - differential.tangent_u[0] * differential.tangent_v[2],
            differential.tangent_u[0] * differential.tangent_v[1]
                - differential.tangent_u[1] * differential.tangent_v[0],
        ];
        let normal = scale(normal, 1.0 / length(normal));
        CameraRig::new(
            add(differential.position, normal),
            CameraBasis::from_forward_up(forward, normal).unwrap(),
            1.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap()
    }

    fn assert_near3(left: [f64; 3], right: [f64; 3], tolerance: f64) {
        assert!(
            length(sub(left, right)) <= tolerance,
            "left={left:?}, right={right:?}, tolerance={tolerance}"
        );
    }

    #[test]
    fn one_face_output_patch_matches_render_transform_order() {
        let instances = triangle_instances();
        let runtime = SurfaceRuntime::default();
        let model = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.25, -0.5, -3.0, 1.0,
        ];
        let reflection = sphere_reflection_mobius([0.1, -0.2, -2.0], 1.3);
        let reconstructed = runtime
            .output_patch_for_face(&instances, 0, reflection, model)
            .unwrap();
        let expected = QBTriPatch::flat(
            [0.25, -0.5, -3.0],
            [2.25, -0.5, -3.0],
            [0.25, 2.5, -3.0],
        )
        .transform(&mobius_from_array(reflection));

        for barycentric in [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 1.0, 0.0]] {
            assert_near3(
                reconstructed
                    .eval(barycentric[1], barycentric[2])
                    .to_point(),
                expected.eval(barycentric[1], barycentric[2]).to_point(),
                1.0e-12,
            );
        }
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
                [2.75, 0.25, 1.0],
                1,
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
                [0.25, 0.25, 3.0],
                1,
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

    #[wasm_bindgen_test]
    fn fixed_camera_anchor_follows_pose_and_recaptures_navigation_edits() {
        let instances = triangle_instances();
        let mut runtime = SurfaceRuntime::default();
        runtime.set_morph_targets(
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            3,
            1,
        );
        runtime
            .set_timed_pose(&[], &[0.0], 0.0, 10.0, 1, 1)
            .unwrap();
        let camera = CameraRig::new(
            [0.25, 0.25, 2.0],
            CameraBasis::CANONICAL,
            2.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap();
        let attached = runtime
            .attach_camera_anchor(
                &instances,
                1,
                &[0],
                0,
                [0.5, 0.25, 0.25],
                camera,
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        assert_near3(attached.output_position.unwrap(), [0.25, 0.25, 0.0], 1.0e-10);
        assert_near3(attached.camera.eye, camera.eye, 1.0e-10);

        runtime
            .set_timed_pose(&[], &[1.0], 1.0, 10.0 + 1.0 / 60.0, 2, 1)
            .unwrap();
        let moved = runtime
            .step_camera_anchor(
                &instances,
                &[0],
                camera,
                false,
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        assert_near3(moved.output_position.unwrap(), [1.25, 0.25, 0.0], 1.0e-10);
        assert_near3(moved.camera.eye, [1.25, 0.25, 2.0], 1.0e-10);

        let current_anchor = runtime.camera_anchor.unwrap();
        let mut edited = current_anchor
            .relative_camera()
            .resolve(current_anchor.frame())
            .unwrap();
        edited.eye[1] += 0.75;
        let recaptured = runtime
            .step_camera_anchor(
                &instances,
                &[0],
                edited,
                true,
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        assert_near3(recaptured.camera.eye, edited.eye, 1.0e-10);

        runtime
            .set_timed_pose(&[], &[0.0], 0.0, 20.0, 1, 2)
            .unwrap();
        let returned = runtime
            .step_camera_anchor(
                &instances,
                &[0],
                edited,
                false,
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        assert_near3(returned.output_position.unwrap(), [0.25, 0.25, 0.0], 1.0e-10);
        assert_near3(returned.camera.eye, [0.25, 1.0, 2.0], 1.0e-10);
        assert_eq!(returned.face, attached.face);
        assert_eq!(returned.barycentric, attached.barycentric);
    }

    #[test]
    fn attachment_chooses_camera_side_and_preserves_it_through_inversion_parity() {
        let mut instances = triangle_instances();
        for corner in 0..3 {
            instances[instance_layout::offset::POSITIONS + corner * 4 + 3] = 2.0;
        }
        let barycentric = [0.5, 0.25, 0.25];
        let source_point = Quat::from_point(0.25, 0.25, 2.0);

        let mut below = SurfaceRuntime::default();
        let below_snapshot = below
            .attach(
                &instances,
                1,
                &[0],
                0,
                barycentric,
                0.1,
                [0.25, 0.25, 1.0],
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        assert!(below_snapshot.output_normal.unwrap()[2] < -0.999);

        let mut runtime = SurfaceRuntime::default();
        let attached = runtime
            .attach(
                &instances,
                1,
                &[0],
                0,
                barycentric,
                0.1,
                [0.25, 0.25, 3.0],
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        assert!(attached.output_normal.unwrap()[2] > 0.999);

        let inverted = runtime
            .step_relative(
                &instances,
                &[0],
                0.0,
                [0.0; 3],
                -1,
                [
                    0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                identity_matrix(),
            )
            .unwrap();
        let expected_point = Mobius::inversion().apply(source_point).to_point();
        let actual_point = inverted.output_position.unwrap();
        assert!(length(sub(actual_point, expected_point)) < 1.0e-9);

        let epsilon = 1.0e-6;
        let mapped_above = Mobius::inversion()
            .apply(source_point + Quat::from_point(0.0, 0.0, epsilon))
            .to_point();
        let expected_side = scale(sub(mapped_above, expected_point), 1.0 / epsilon);
        let expected_side = scale(expected_side, 1.0 / length(expected_side));
        let actual_side = inverted.output_normal.unwrap();
        assert!(dot(expected_side, actual_side) > 1.0 - 1.0e-8);
    }

    #[wasm_bindgen_test]
    fn reflection_transport_carries_legacy_side_and_velocity_history_once() {
        let mut instances = triangle_instances();
        for corner in 0..3 {
            instances[instance_layout::offset::POSITIONS + corner * 4 + 3] = 2.0;
        }
        let mut runtime = SurfaceRuntime::default();
        runtime.set_morph_targets(&[0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0], 3, 1);
        runtime
            .set_timed_pose(&[], &[0.0], 0.0, 10.0, 1, 1)
            .unwrap();
        let address = [0.5, 0.25, 0.25];
        let mut camera = CameraRig::new(
            [0.25, 0.25, 3.0],
            CameraBasis::from_forward_up([1.0, 0.0, -1.0], [0.0, 1.0, 0.0]).unwrap(),
            3.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap();
        let attached = runtime
            .attach(
                &instances,
                1,
                &[0],
                0,
                address,
                0.1,
                camera.eye,
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        runtime
            .attach_composed(
                &instances,
                1,
                &[0],
                0,
                address,
                camera,
                1.0,
                SurfaceWalkControls::default(),
                1.0,
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        let source_position = attached.output_position.unwrap();
        let legacy_before = runtime.walker.attachment().unwrap();
        let composed_before = runtime.composed.attachment().unwrap();
        let center = [0.1, -0.2, 0.3];
        let radius = 1.1;
        let inversion = SphereReflectionState::Sphere(FocusSphere::new(center, radius).unwrap());
        let mobius = sphere_reflection_mobius(center, radius);

        let outcome = runtime
            .transport_between_reflections(SphereReflectionState::Identity, inversion)
            .unwrap();
        assert!(outcome.legacy_attached);
        assert!(outcome.composed_attached);
        assert!(outcome.composed_follower_transported);
        assert!(outcome.normal_side_flipped);
        assert!(outcome.anchor_transition_cancelled);
        assert!(outcome.legacy_previous_position_transported);
        assert!(outcome.composed_previous_position_transported);
        assert_eq!(
            runtime.walker.attachment().unwrap().normal_sign,
            -legacy_before.normal_sign
        );
        assert_eq!(
            runtime.composed.attachment().unwrap().normal_sign,
            -composed_before.normal_sign
        );
        assert_eq!(
            runtime.walker.attachment().unwrap().address,
            legacy_before.address
        );
        assert_eq!(
            runtime.composed.attachment().unwrap().address,
            composed_before.address
        );
        assert_eq!(runtime.attachment_orientation_sign, -1);
        assert_eq!(runtime.composed_attachment_orientation_sign, -1);
        assert!(runtime.legacy_velocity.suppress_once);
        assert!(runtime.composed_velocity.suppress_once);
        let expected = inversion
            .transport_point_and_directions::<0>(
                SphereReflectionState::Identity,
                runtime.legacy_velocity.previous_output_position.unwrap(),
                [],
            )
            .unwrap()
            .point;
        assert!(length(sub(expected, source_position)) < 1.0e-10);

        camera
            .transport_between_reflections(SphereReflectionState::Identity, inversion)
            .unwrap();
        let legacy_static = runtime
            .step_relative(
                &instances,
                &[0],
                1.0 / 60.0,
                [0.0; 3],
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        let composed_static = runtime
            .step_composed(
                &instances,
                &[0],
                1.0 / 60.0,
                camera,
                1.0,
                SurfaceWalkControls::default(),
                SurfaceWalkInput::default(),
                true,
                false,
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        assert_eq!(
            legacy_static.barycentric.unwrap(),
            legacy_before.address.barycentric
        );
        assert_eq!(
            composed_static.barycentric.unwrap(),
            composed_before.address.barycentric
        );
        assert!(length(legacy_static.projected_output_velocity) < 1.0e-12);
        assert!(length(composed_static.surface_velocity.unwrap()) < 1.0e-12);
        assert!(legacy_static.pose_sample.unwrap().velocity_rebased);
        assert!(composed_static.pose_sample.unwrap().velocity_rebased);
        assert_eq!(
            runtime.walker.attachment().unwrap().normal_sign,
            -legacy_before.normal_sign
        );
        assert_eq!(
            runtime.composed.attachment().unwrap().normal_sign,
            -composed_before.normal_sign
        );

        runtime
            .set_timed_pose(&[], &[0.2], 0.2, 10.2, 2, 1)
            .unwrap();
        let legacy_animated = runtime
            .step_relative(
                &instances,
                &[0],
                1.0 / 1000.0,
                [0.0; 3],
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        let composed_animated = runtime
            .step_composed(
                &instances,
                &[0],
                1.0 / 3.0,
                camera,
                1.0,
                SurfaceWalkControls::default(),
                SurfaceWalkInput::default(),
                true,
                false,
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        assert_eq!(
            legacy_animated.barycentric.unwrap(),
            legacy_before.address.barycentric
        );
        assert_eq!(
            composed_animated.barycentric.unwrap(),
            composed_before.address.barycentric
        );
        let composed_velocity = composed_animated.surface_velocity.unwrap();
        assert!(length(composed_velocity) > 1.0e-6);
        assert!(
            length(sub(
                legacy_animated.projected_output_velocity,
                composed_velocity,
            )) < 1.0e-9
        );
        for pose in [
            legacy_animated.pose_sample.unwrap(),
            composed_animated.pose_sample.unwrap(),
        ] {
            assert_eq!(pose.revision, 2);
            assert_eq!(pose.continuity_epoch, 1);
            assert!(pose.continuous);
            assert!((pose.sample_delta_seconds.unwrap() - 0.2).abs() < 1.0e-12);
            assert!(!pose.velocity_rebased);
        }

        let legacy_held = runtime
            .step_relative(
                &instances,
                &[0],
                1.0 / 60.0,
                [0.0; 3],
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        let composed_held = runtime
            .step_composed(
                &instances,
                &[0],
                1.0 / 60.0,
                camera,
                1.0,
                SurfaceWalkControls::default(),
                SurfaceWalkInput::default(),
                true,
                false,
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        assert!(length(legacy_held.projected_output_velocity) < 1.0e-12);
        assert!(length(composed_held.surface_velocity.unwrap()) < 1.0e-12);
        assert!(legacy_held
            .pose_sample
            .unwrap()
            .sample_delta_seconds
            .is_none());
        assert!(composed_held
            .pose_sample
            .unwrap()
            .sample_delta_seconds
            .is_none());

        assert!(runtime
            .set_timed_pose(&[], &[0.25], 0.3, 10.3, 3, 1)
            .unwrap());
        assert!(runtime
            .set_timed_pose(&[], &[0.4], 0.5, 10.5, 4, 1)
            .unwrap());
        let legacy_coalesced = runtime
            .step_relative(
                &instances,
                &[0],
                1.0 / 500.0,
                [0.0; 3],
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        let composed_coalesced = runtime
            .step_composed(
                &instances,
                &[0],
                0.25,
                camera,
                1.0,
                SurfaceWalkControls::default(),
                SurfaceWalkInput::default(),
                true,
                false,
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        let coalesced_velocity = composed_coalesced.surface_velocity.unwrap();
        assert!(length(coalesced_velocity) > 1.0e-6);
        assert!(
            length(sub(
                legacy_coalesced.projected_output_velocity,
                coalesced_velocity,
            )) < 1.0e-9
        );
        for pose in [
            legacy_coalesced.pose_sample.unwrap(),
            composed_coalesced.pose_sample.unwrap(),
        ] {
            assert_eq!(pose.revision, 4);
            assert!((pose.sample_delta_seconds.unwrap() - 0.3).abs() < 1.0e-12);
        }

        runtime.set_timed_pose(&[], &[0.3], 0.0, 0.0, 1, 2).unwrap();
        let legacy_rebased = runtime
            .step_relative(
                &instances,
                &[0],
                1.0 / 60.0,
                [0.0; 3],
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        let composed_rebased = runtime
            .step_composed(
                &instances,
                &[0],
                1.0 / 60.0,
                camera,
                1.0,
                SurfaceWalkControls::default(),
                SurfaceWalkInput::default(),
                true,
                false,
                -1,
                mobius,
                identity_matrix(),
            )
            .unwrap();
        assert!(length(legacy_rebased.projected_output_velocity) < 1.0e-12);
        assert!(length(composed_rebased.surface_velocity.unwrap()) < 1.0e-12);
        assert!(legacy_rebased.pose_sample.unwrap().velocity_rebased);
        assert!(composed_rebased.pose_sample.unwrap().velocity_rebased);

        runtime
            .transport_between_reflections(inversion, SphereReflectionState::Identity)
            .unwrap();
        assert_eq!(
            runtime.walker.attachment().unwrap().normal_sign,
            legacy_before.normal_sign
        );
        assert_eq!(runtime.attachment_orientation_sign, 1);
    }

    #[wasm_bindgen_test]
    fn reflection_transport_rolls_back_if_velocity_history_hits_a_pole() {
        let instances = triangle_instances();
        let mut runtime = SurfaceRuntime::default();
        let attached = runtime
            .attach(
                &instances,
                1,
                &[0],
                0,
                [0.5, 0.25, 0.25],
                0.1,
                [0.25, 0.25, 3.0],
                1,
                identity_mobius(),
                identity_matrix(),
            )
            .unwrap();
        let pole = attached.output_position.unwrap();
        let before_attachment = runtime.walker.attachment();
        let before_position = runtime.legacy_velocity.previous_output_position;
        let before_orientation = runtime.attachment_orientation_sign;
        let inversion = SphereReflectionState::Sphere(FocusSphere::new(pole, 1.0).unwrap());

        assert!(runtime
            .transport_between_reflections(SphereReflectionState::Identity, inversion)
            .is_err());
        assert_eq!(runtime.walker.attachment(), before_attachment);
        assert_eq!(
            runtime.legacy_velocity.previous_output_position,
            before_position
        );
        assert_eq!(runtime.attachment_orientation_sign, before_orientation);
    }

    #[wasm_bindgen_test]
    fn timed_pose_rejects_stale_or_invalid_packets_atomically() {
        let mut runtime = SurfaceRuntime::default();
        assert!(runtime
            .set_timed_pose(&[1.0, 2.0], &[0.25], 0.5, 10.0, 4, 3)
            .unwrap());
        let accepted_pose = runtime.pose_sample;
        let accepted_matrices = runtime.joint_matrices.clone();
        let accepted_morphs = runtime.morph_weights.clone();

        assert!(!runtime
            .set_timed_pose(&[9.0], &[0.9], 0.6, 10.1, 4, 3)
            .unwrap());
        assert!(!runtime
            .set_timed_pose(&[8.0], &[0.8], 0.7, 10.2, 100, 2)
            .unwrap());
        assert!(runtime
            .set_timed_pose(&[7.0], &[0.7], 0.8, 10.0, 5, 3)
            .is_err());
        assert!(runtime
            .set_timed_pose(&[6.0], &[0.6], f64::NAN, 10.3, 5, 3)
            .is_err());
        assert!(runtime
            .set_timed_pose(&[5.0], &[0.5], 0.9, 10.3, 0, 3)
            .is_err());

        assert_eq!(runtime.pose_sample, accepted_pose);
        assert_eq!(runtime.joint_matrices, accepted_matrices);
        assert_eq!(runtime.morph_weights, accepted_morphs);
    }

    #[wasm_bindgen_test]
    fn float32_near_edge_crossing_matches_composed_runtime_across_permutations() {
        let source_faces = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
        let target_faces = [[1, 3, 2], [3, 2, 1], [2, 1, 3]];
        let epsilon = f32::EPSILON;
        let source_vertex_weights = [epsilon, 0.55_f32, 1.0_f32 - epsilon - 0.55_f32, 0.0];
        let controls = SurfaceWalkControls {
            base_radii_per_second: 0.1,
            smoothing_seconds: 0.0,
            ..SurfaceWalkControls::default()
        };
        let maps = [
            (identity_mobius(), 1),
            (sphere_reflection_mobius([0.1, -0.2, 0.3], 1.1), -1),
        ];

        for (mobius, orientation_sign) in maps {
            let mut reference_position: Option<[f64; 3]> = None;
            let mut reference_weights: Option<[f64; 4]> = None;
            for source_face in source_faces {
                for target_face in target_faces {
                    let faces = [source_face, target_face];
                    let instances = two_triangle_instances(faces);
                    let barycentric = face_barycentric(source_face, source_vertex_weights);
                    let camera = camera_crossing_shared_edge(source_face, barycentric, mobius);

                    let mut legacy = SurfaceRuntime::default();
                    let attached = legacy
                        .attach(
                            &instances,
                            2,
                            &[0, 0],
                            0,
                            barycentric,
                            controls.base_eye_height,
                            camera.eye,
                            orientation_sign,
                            mobius,
                            identity_matrix(),
                        )
                        .unwrap();
                    assert_eq!(attached.status, "attached");

                    let mut composed = SurfaceRuntime::default();
                    let composed_attached = composed
                        .attach_composed(
                            &instances,
                            2,
                            &[0, 0],
                            0,
                            barycentric,
                            camera,
                            1.0,
                            controls,
                            0.0,
                            orientation_sign,
                            mobius,
                            identity_matrix(),
                        )
                        .unwrap();
                    assert_eq!(composed_attached.status, "attached");

                    let composed_step = composed
                        .step_composed(
                            &instances,
                            &[0, 0],
                            0.01,
                            camera,
                            1.0,
                            controls,
                            SurfaceWalkInput {
                                forward_axis: 1.0,
                                ..SurfaceWalkInput::default()
                            },
                            false,
                            false,
                            orientation_sign,
                            mobius,
                            identity_matrix(),
                        )
                        .unwrap();
                    let desired_velocity = composed_step.desired_output_velocity.unwrap();
                    let legacy_velocity = desired_velocity.map(|value| f64::from(value as f32));
                    let legacy_step = legacy
                        .step_relative(
                            &instances,
                            &[0, 0],
                            0.01,
                            legacy_velocity,
                            orientation_sign,
                            mobius,
                            identity_matrix(),
                        )
                        .unwrap();

                    assert_eq!(legacy_step.status, "attached");
                    assert_eq!(composed_step.status, "attached");
                    assert_eq!(legacy_step.face, Some(1));
                    assert_eq!(composed_step.face, Some(1));
                    assert_eq!(legacy_step.edge_crossings, 1);
                    assert_eq!(composed_step.edge_crossings, 1);
                    assert_near3(
                        legacy_step.output_position.unwrap(),
                        composed_step.output_position.unwrap(),
                        1.0e-9,
                    );
                    assert_near3(
                        legacy_step.projected_output_velocity,
                        composed_step.projected_output_velocity,
                        f64::from(f32::EPSILON),
                    );
                    for (legacy_value, composed_value) in legacy_step
                        .barycentric
                        .unwrap()
                        .into_iter()
                        .zip(composed_step.barycentric.unwrap())
                    {
                        assert!((legacy_value - composed_value).abs() <= 1.0e-9);
                    }

                    let position = composed_step.output_position.unwrap();
                    let weights = vertex_weights(target_face, composed_step.barycentric.unwrap());
                    if let Some(reference) = reference_position {
                        assert_near3(position, reference, 1.0e-10);
                    } else {
                        reference_position = Some(position);
                    }
                    if let Some(reference) = reference_weights {
                        for (actual, expected) in weights.into_iter().zip(reference) {
                            assert!((actual - expected).abs() <= 1.0e-10);
                        }
                    } else {
                        reference_weights = Some(weights);
                    }
                }
            }
        }
    }

    #[test]
    fn exact_external_pose_reconstructs_patch_controls_without_mutating_walk_pose() {
        let instances = triangle_instances();
        let mut runtime = SurfaceRuntime::default();
        runtime.set_skinning(&[[0; 4]; 3], &[[1.0, 0.0, 0.0, 0.0]; 3]);
        let joint_matrices = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0,
        ];
        let controls = runtime
            .patch_controls_for_pose(&instances, 1, &joint_matrices, &[])
            .unwrap();
        assert_eq!(controls.len(), 1);
        assert!(controls[0]
            .positions
            .iter()
            .all(|position| (position[2] - 2.0).abs() < 1.0e-9));
        assert_eq!(controls[0].weights, [[1.0, 0.0, 0.0, 0.0]; 3]);
    }

    #[test]
    fn lod_animation_source_borrows_the_exact_renderer_payload() {
        let mut runtime = SurfaceRuntime::default();
        let joint_indices = [[0, 1, 2, 3]; 3];
        let joint_weights = [[0.25; 4]; 3];
        let morph_deltas = [0.125; 9];
        runtime.set_skinning(&joint_indices, &joint_weights);
        runtime.set_morph_targets(&morph_deltas, 3, 1);

        let source = runtime.lod_animation_source(3).unwrap();
        assert_eq!(source.primary_vertices, 3);
        assert_eq!(source.joint_indices, Some(joint_indices.as_slice()));
        assert_eq!(source.joint_weights, Some(joint_weights.as_slice()));
        assert_eq!(source.morph_deltas, &morph_deltas);
        assert_eq!(source.num_morph_targets, 1);
    }

    #[test]
    fn lod_animation_source_rejects_a_stale_geometry_epoch() {
        let mut runtime = SurfaceRuntime::default();
        runtime.set_skinning(&[[0; 4]; 2], &[[0.0; 4]; 2]);
        assert_eq!(
            runtime.lod_animation_source(3).unwrap_err(),
            "renderer LOD skinning source does not match resident vertices",
        );
    }

    #[test]
    fn lod_pose_source_requires_the_exact_accepted_renderer_stamp() {
        let mut runtime = SurfaceRuntime::default();
        runtime.set_morph_targets(&[0.0; 9], 3, 1);
        let matrices = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        runtime
            .set_timed_pose(&matrices, &[0.5], 0.75, 12.5, 7, 3)
            .unwrap();

        let source = runtime.lod_pose_source(0.75, 12.5, 7, 3).unwrap();
        assert_eq!(source.joint_matrices, &matrices);
        assert_eq!(source.morph_weights, &[0.5]);
        assert_eq!(source.clip_time_seconds, 0.75);
        assert_eq!(source.sample_time_seconds, 12.5);
        assert_eq!(source.revision, 7);
        assert_eq!(source.continuity_epoch, 3);
        assert!(runtime.lod_pose_source(0.5, 12.5, 7, 3).is_err());
        assert!(runtime.lod_pose_source(0.75, 12.5, 8, 3).is_err());
    }
}
