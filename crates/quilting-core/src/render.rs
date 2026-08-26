//! Backend-neutral render extraction.
//!
//! [`RenderSceneSnapshot`] changes only when retained batch membership or
//! entity/material state changes. [`RenderFrame`] carries the high-rate view,
//! pose identity, and ordered logical commands. Neither type contains a GL
//! handle, WebGPU resource, DOM object, or platform callback.

use crate::batch::{RenderBatchKey, RenderBatchMember};
use crate::permutation::perm_sign;
use serde::Serialize;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderEntityTransform {
    pub mobius: [f32; 16],
    pub orientation_sign: i8,
    pub euclidean_model: [f32; 16],
    pub euclidean_normal: [f32; 16],
}

impl RenderEntityTransform {
    fn validate(self) -> Result<(), RenderContractError> {
        if !matches!(self.orientation_sign, -1 | 1)
            || !finite(self.mobius)
            || !finite(self.euclidean_model)
            || !finite(self.euclidean_normal)
        {
            return Err(RenderContractError::InvalidTransform);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PbrDrawClass {
    Opaque,
    Blend,
    Transmission,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderBatchSnapshot {
    pub key: RenderBatchKey,
    pub members: Vec<RenderBatchMember>,
    /// Backend-neutral cardinality of the resident tessellation entry. Both
    /// WebGL2 and WebGPU consume the same indexed atlas topology.
    pub triangle_index_count: u32,
    pub line_index_count: u32,
    pub transform: RenderEntityTransform,
    /// Presentation/layer visibility. Disabled batches remain resident and
    /// therefore keep their member list while commands carry zero instances.
    pub enabled: bool,
    pub pbr_class: PbrDrawClass,
}

impl RenderBatchSnapshot {
    pub fn active_instance_count(&self) -> Result<u32, RenderContractError> {
        if self.enabled {
            self.members
                .len()
                .try_into()
                .map_err(|_| RenderContractError::InstanceCountOverflow)
        } else {
            Ok(0)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSceneSnapshot {
    pub revision: u64,
    pub batches: Vec<RenderBatchSnapshot>,
}

impl RenderSceneSnapshot {
    pub fn validate(&self) -> Result<(), RenderContractError> {
        let mut previous = None;
        let mut patches = BTreeSet::new();
        for (batch_index, batch) in self.batches.iter().enumerate() {
            if previous.is_some_and(|key| key >= batch.key) {
                return Err(RenderContractError::BatchOrder { batch_index });
            }
            previous = Some(batch.key);
            if batch.key.parity_bucket > 1
                || batch
                    .key
                    .lod
                    .into_iter()
                    .any(|lod| lod == 0 || !lod.is_power_of_two())
            {
                return Err(RenderContractError::InvalidBatchKey { batch_index });
            }
            if batch.triangle_index_count == 0
                || !batch.triangle_index_count.is_multiple_of(3)
                || !batch.line_index_count.is_multiple_of(2)
            {
                return Err(RenderContractError::InvalidBatchGeometry { batch_index });
            }
            batch.transform.validate()?;
            batch.active_instance_count()?;
            for member in &batch.members {
                let member_lod = crate::batch::ResidentLod::from_edge_lods(member.edge_lods);
                if member.permutation_index >= 6
                    || member_lod.canonical != batch.key.lod
                    || member_lod.perm_index.min(5) as u8 != member.permutation_index
                    || usize::from(perm_sign(member.permutation_index as usize) < 0)
                        != usize::from(batch.key.parity_bucket)
                    || member
                        .vertex_lods
                        .into_iter()
                        .any(|lod| lod == 0 || !lod.is_power_of_two())
                    || member.leaf_id.domain().is_none()
                {
                    return Err(RenderContractError::InvalidBatchMember {
                        batch_index,
                        face_index: member.face_index,
                    });
                }
                if !patches.insert(member.patch_id()) {
                    if member.leaf_id == crate::screen_partition::ScreenPatchLeafId::ROOT {
                        return Err(RenderContractError::DuplicateFace(member.face_index));
                    }
                    return Err(RenderContractError::DuplicatePatch {
                        face_index: member.face_index,
                        leaf_depth: member.leaf_id.depth,
                        leaf_path: member.leaf_id.path,
                    });
                }
            }
        }
        for &(face_index, leaf_id) in &patches {
            for ancestor_depth in 0..leaf_id.depth {
                let ancestor = leaf_id
                    .ancestor_at_depth(ancestor_depth)
                    .expect("validated leaf has every shallower ancestor");
                if patches.contains(&(face_index, ancestor)) {
                    return Err(RenderContractError::OverlappingPatch {
                        face_index,
                        ancestor_depth,
                        ancestor_path: ancestor.path,
                        descendant_depth: leaf_id.depth,
                        descendant_path: leaf_id.path,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusFieldPacket {
    pub sphere: [f32; 4],
    pub enabled: bool,
}

impl Default for FocusFieldPacket {
    fn default() -> Self {
        Self {
            sphere: [0.0, 0.0, 0.0, 1.0],
            enabled: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderView {
    pub viewport: [u32; 2],
    pub mvp: [f32; 16],
    pub model_view: [f32; 16],
    pub camera_position: [f32; 3],
    pub selected_node: Option<usize>,
    pub focus: FocusFieldPacket,
}

impl RenderView {
    pub fn validate(self) -> Result<(), RenderContractError> {
        if !finite(self.mvp)
            || !finite(self.model_view)
            || !finite(self.camera_position)
            || !finite(self.focus.sphere)
            || self.focus.sphere[3] <= 0.0
        {
            return Err(RenderContractError::InvalidView);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderPoseIdentity {
    pub asset_revision: u64,
    pub pose_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    Pbr,
    Matcap,
    Wire,
    Normals,
    MatcapWire,
    Lod,
    Stretch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPass {
    PbrOpaque,
    PbrTransparent,
    Matcap,
    Wire,
    Normals,
    Lod,
    Stretch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderGeometry {
    Triangles,
    Lines,
}

/// Backend-neutral accounting for indexed patch draws actually submitted by a
/// renderer. Logical [`RenderCommand`]s predict this work; WebGL2 and WebGPU
/// implementations populate the counters at their submission boundary so a
/// shadow observer can compare intent with execution without a GPU readback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderSubmissionStats {
    /// Indexed patch draw calls issued to the backend, including zero-instance
    /// and invalid calls.
    pub draw_calls: u64,
    /// Draws whose instance count is zero. These issue no primitives and are
    /// tracked separately because retained hidden batches should eventually be
    /// skipped before reaching the backend.
    pub zero_instance_draw_calls: u64,
    /// Draws rejected by the recorder because a backend count was negative or
    /// otherwise could not be represented by the shared contract.
    pub invalid_draw_calls: u64,
    /// Sum of non-zero instances supplied to valid indexed draws.
    pub submitted_instances: u64,
    /// Triangle primitives implied by valid index and instance counts.
    pub triangles: u64,
    /// Line primitives implied by valid index and instance counts.
    pub lines: u64,
}

/// Field-level difference between expected and actual submission work. Keeping
/// this bounded avoids retaining per-frame command streams or copying them to
/// a platform adapter just to diagnose parity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderSubmissionMismatch {
    pub draw_calls: bool,
    pub zero_instance_draw_calls: bool,
    pub invalid_draw_calls: bool,
    pub submitted_instances: bool,
    pub triangles: bool,
    pub lines: bool,
}

impl RenderSubmissionMismatch {
    pub fn between(expected: RenderSubmissionStats, actual: RenderSubmissionStats) -> Self {
        Self {
            draw_calls: expected.draw_calls != actual.draw_calls,
            zero_instance_draw_calls: expected.zero_instance_draw_calls
                != actual.zero_instance_draw_calls,
            invalid_draw_calls: expected.invalid_draw_calls != actual.invalid_draw_calls,
            submitted_instances: expected.submitted_instances != actual.submitted_instances,
            triangles: expected.triangles != actual.triangles,
            lines: expected.lines != actual.lines,
        }
    }

    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderParityComparison {
    pub frame_revision: u64,
    pub scene_revision: u64,
    pub expected: RenderSubmissionStats,
    pub actual: RenderSubmissionStats,
    pub mismatch: RenderSubmissionMismatch,
}

impl RenderParityComparison {
    pub fn is_match(self) -> bool {
        self.mismatch.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderParityDiagnostics {
    pub enabled: bool,
    pub scene_revision: Option<u64>,
    pub scene_rebuilds: u64,
    pub frames_observed: u64,
    pub matching_frames: u64,
    pub mismatching_frames: u64,
    pub last: Option<RenderParityComparison>,
}

/// Opt-in bounded observer shared by concrete renderer backends. Disabled
/// observers retain no scene clone and perform no per-frame work.
#[derive(Debug, Default)]
pub struct RenderParityObserver {
    enabled: bool,
    scene: Option<RenderSceneSnapshot>,
    scene_rebuilds: u64,
    frames_observed: u64,
    matching_frames: u64,
    mismatching_frames: u64,
    last: Option<RenderParityComparison>,
}

impl RenderParityObserver {
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled == enabled {
            return;
        }
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Validate before replacing the retained scene, so a failed extraction
    /// never destroys the last comparison oracle.
    pub fn replace_scene(&mut self, scene: RenderSceneSnapshot) -> Result<(), RenderContractError> {
        if !self.enabled {
            return Err(RenderContractError::ObserverDisabled);
        }
        scene.validate()?;
        self.scene = Some(scene);
        self.scene_rebuilds = self.scene_rebuilds.saturating_add(1);
        Ok(())
    }

    pub fn scene(&self) -> Option<&RenderSceneSnapshot> {
        self.scene.as_ref()
    }

    pub fn observe(
        &mut self,
        frame: &RenderFrame,
        actual: RenderSubmissionStats,
    ) -> Result<RenderParityComparison, RenderContractError> {
        if !self.enabled {
            return Err(RenderContractError::ObserverDisabled);
        }
        let scene = self
            .scene
            .as_ref()
            .ok_or(RenderContractError::ObserverSceneUnavailable)?;
        let expected = frame.expected_submission_stats(scene)?;
        let comparison = RenderParityComparison {
            frame_revision: frame.revision,
            scene_revision: scene.revision,
            expected,
            actual,
            mismatch: RenderSubmissionMismatch::between(expected, actual),
        };
        self.frames_observed = self.frames_observed.saturating_add(1);
        if comparison.is_match() {
            self.matching_frames = self.matching_frames.saturating_add(1);
        } else {
            self.mismatching_frames = self.mismatching_frames.saturating_add(1);
        }
        self.last = Some(comparison);
        Ok(comparison)
    }

    pub fn diagnostics(&self) -> RenderParityDiagnostics {
        RenderParityDiagnostics {
            enabled: self.enabled,
            scene_revision: self.scene.as_ref().map(|scene| scene.revision),
            scene_rebuilds: self.scene_rebuilds,
            frames_observed: self.frames_observed,
            matching_frames: self.matching_frames,
            mismatching_frames: self.mismatching_frames,
            last: self.last,
        }
    }
}

impl RenderSubmissionStats {
    /// Record one valid indexed patch draw. Incomplete trailing indices do not
    /// form a primitive, matching indexed triangle/line assembly.
    pub fn record_indexed_draw(
        &mut self,
        geometry: RenderGeometry,
        index_count: u32,
        instance_count: u32,
    ) {
        self.draw_calls = self.draw_calls.saturating_add(1);
        if instance_count == 0 {
            self.zero_instance_draw_calls = self.zero_instance_draw_calls.saturating_add(1);
            return;
        }
        let instances = u64::from(instance_count);
        self.submitted_instances = self.submitted_instances.saturating_add(instances);
        let primitive_width = match geometry {
            RenderGeometry::Triangles => 3,
            RenderGeometry::Lines => 2,
        };
        let primitives = u64::from(index_count / primitive_width).saturating_mul(instances);
        match geometry {
            RenderGeometry::Triangles => {
                self.triangles = self.triangles.saturating_add(primitives);
            }
            RenderGeometry::Lines => {
                self.lines = self.lines.saturating_add(primitives);
            }
        }
    }

    /// Record a draw call whose signed backend counts could not be represented
    /// by the shared non-negative contract.
    pub fn record_invalid_draw(&mut self) {
        self.draw_calls = self.draw_calls.saturating_add(1);
        self.invalid_draw_calls = self.invalid_draw_calls.saturating_add(1);
    }

    /// Accumulate another submission interval without allowing diagnostic
    /// counters to wrap during a long-running session.
    pub fn merge(&mut self, other: Self) {
        self.draw_calls = self.draw_calls.saturating_add(other.draw_calls);
        self.zero_instance_draw_calls = self
            .zero_instance_draw_calls
            .saturating_add(other.zero_instance_draw_calls);
        self.invalid_draw_calls = self
            .invalid_draw_calls
            .saturating_add(other.invalid_draw_calls);
        self.submitted_instances = self
            .submitted_instances
            .saturating_add(other.submitted_instances);
        self.triangles = self.triangles.saturating_add(other.triangles);
        self.lines = self.lines.saturating_add(other.lines);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderCommand {
    PreparePatches {
        batch_index: u32,
        instance_count: u32,
    },
    DrawPatches {
        batch_index: u32,
        instance_count: u32,
        pass: RenderPass,
        geometry: RenderGeometry,
    },
    BuildTransmissionPyramid,
    FocusPostProcess,
    HighlightFace {
        face_index: u32,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderFrameOptions {
    pub focus_postprocess: bool,
    pub highlight_face: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderFrame {
    pub revision: u64,
    pub scene_revision: u64,
    pub pose: RenderPoseIdentity,
    pub style: RenderStyle,
    pub view: RenderView,
    pub options: RenderFrameOptions,
    pub commands: Vec<RenderCommand>,
}

impl RenderFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        revision: u64,
        pose: RenderPoseIdentity,
        style: RenderStyle,
        view: RenderView,
        options: RenderFrameOptions,
        scene: &RenderSceneSnapshot,
    ) -> Result<Self, RenderContractError> {
        scene.validate()?;
        view.validate()?;
        let commands = expected_commands(style, options, &scene.batches)?;
        Ok(Self {
            revision,
            scene_revision: scene.revision,
            pose,
            style,
            view,
            options,
            commands,
        })
    }

    pub fn validate(&self, scene: &RenderSceneSnapshot) -> Result<(), RenderContractError> {
        scene.validate()?;
        self.view.validate()?;
        if self.scene_revision != scene.revision {
            return Err(RenderContractError::SceneRevisionMismatch {
                frame: self.scene_revision,
                scene: scene.revision,
            });
        }
        if self.commands != expected_commands(self.style, self.options, &scene.batches)? {
            return Err(RenderContractError::CommandSequenceMismatch);
        }
        Ok(())
    }

    /// Derive the indexed patch work implied by this validated frame. This is
    /// the backend-independent side of render shadowing; concrete renderers
    /// record an actual [`RenderSubmissionStats`] at their draw boundary.
    pub fn expected_submission_stats(
        &self,
        scene: &RenderSceneSnapshot,
    ) -> Result<RenderSubmissionStats, RenderContractError> {
        self.validate(scene)?;
        let mut stats = RenderSubmissionStats::default();
        for command in &self.commands {
            let RenderCommand::DrawPatches {
                batch_index,
                instance_count,
                geometry,
                ..
            } = *command
            else {
                continue;
            };
            let batch = scene
                .batches
                .get(batch_index as usize)
                .ok_or(RenderContractError::CommandBatchMissing { batch_index })?;
            let index_count = match geometry {
                RenderGeometry::Triangles => batch.triangle_index_count,
                RenderGeometry::Lines => batch.line_index_count,
            };
            stats.record_indexed_draw(geometry, index_count, instance_count);
        }
        Ok(stats)
    }
}

fn expected_commands(
    style: RenderStyle,
    options: RenderFrameOptions,
    batches: &[RenderBatchSnapshot],
) -> Result<Vec<RenderCommand>, RenderContractError> {
    let mut commands = Vec::with_capacity(batches.len().saturating_mul(3).saturating_add(3));
    for (batch_index, batch) in batches.iter().enumerate() {
        commands.push(RenderCommand::PreparePatches {
            batch_index: batch_index
                .try_into()
                .map_err(|_| RenderContractError::BatchCountOverflow)?,
            instance_count: batch.active_instance_count()?,
        });
    }

    match style {
        RenderStyle::Pbr => {
            append_draw_commands(
                &mut commands,
                batches,
                RenderPass::PbrOpaque,
                RenderGeometry::Triangles,
                &|batch| batch.pbr_class == PbrDrawClass::Opaque,
            )?;
            if batches
                .iter()
                .any(|batch| batch.pbr_class == PbrDrawClass::Transmission)
            {
                commands.push(RenderCommand::BuildTransmissionPyramid);
            }
            append_draw_commands(
                &mut commands,
                batches,
                RenderPass::PbrTransparent,
                RenderGeometry::Triangles,
                &|batch| batch.pbr_class != PbrDrawClass::Opaque,
            )?;
            if options.focus_postprocess {
                commands.push(RenderCommand::FocusPostProcess);
            }
        }
        RenderStyle::Matcap => append_draw_commands(
            &mut commands,
            batches,
            RenderPass::Matcap,
            RenderGeometry::Triangles,
            &|_| true,
        )?,
        RenderStyle::Wire => append_draw_commands(
            &mut commands,
            batches,
            RenderPass::Wire,
            RenderGeometry::Lines,
            &|_| true,
        )?,
        RenderStyle::Normals => append_draw_commands(
            &mut commands,
            batches,
            RenderPass::Normals,
            RenderGeometry::Triangles,
            &|_| true,
        )?,
        RenderStyle::MatcapWire => {
            append_draw_commands(
                &mut commands,
                batches,
                RenderPass::Matcap,
                RenderGeometry::Triangles,
                &|_| true,
            )?;
            append_draw_commands(
                &mut commands,
                batches,
                RenderPass::Wire,
                RenderGeometry::Lines,
                &|_| true,
            )?;
        }
        RenderStyle::Lod => append_draw_commands(
            &mut commands,
            batches,
            RenderPass::Lod,
            RenderGeometry::Triangles,
            &|_| true,
        )?,
        RenderStyle::Stretch => append_draw_commands(
            &mut commands,
            batches,
            RenderPass::Stretch,
            RenderGeometry::Triangles,
            &|_| true,
        )?,
    }
    if style != RenderStyle::Pbr {
        if let Some(face_index) = options.highlight_face {
            commands.push(RenderCommand::HighlightFace { face_index });
        }
    }
    Ok(commands)
}

fn append_draw_commands(
    commands: &mut Vec<RenderCommand>,
    batches: &[RenderBatchSnapshot],
    pass: RenderPass,
    geometry: RenderGeometry,
    include: &dyn Fn(&RenderBatchSnapshot) -> bool,
) -> Result<(), RenderContractError> {
    for (batch_index, batch) in batches
        .iter()
        .enumerate()
        .filter(|(_, batch)| include(batch))
    {
        commands.push(RenderCommand::DrawPatches {
            batch_index: batch_index
                .try_into()
                .map_err(|_| RenderContractError::BatchCountOverflow)?,
            instance_count: batch.active_instance_count()?,
            pass,
            geometry,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderContractError {
    InvalidTransform,
    InvalidView,
    InvalidBatchKey { batch_index: usize },
    InvalidBatchGeometry { batch_index: usize },
    InvalidBatchMember { batch_index: usize, face_index: u32 },
    BatchOrder { batch_index: usize },
    DuplicateFace(u32),
    DuplicatePatch {
        face_index: u32,
        leaf_depth: u8,
        leaf_path: u32,
    },
    OverlappingPatch {
        face_index: u32,
        ancestor_depth: u8,
        ancestor_path: u32,
        descendant_depth: u8,
        descendant_path: u32,
    },
    BatchCountOverflow,
    InstanceCountOverflow,
    SceneRevisionMismatch { frame: u64, scene: u64 },
    CommandSequenceMismatch,
    CommandBatchMissing { batch_index: u32 },
    ObserverDisabled,
    ObserverSceneUnavailable,
}

impl fmt::Display for RenderContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransform => formatter.write_str("render transform is invalid"),
            Self::InvalidView => formatter.write_str("render view is invalid"),
            Self::InvalidBatchKey { batch_index } => {
                write!(formatter, "render batch {batch_index} has an invalid key")
            }
            Self::InvalidBatchGeometry { batch_index } => write!(
                formatter,
                "render batch {batch_index} has invalid index cardinalities"
            ),
            Self::InvalidBatchMember {
                batch_index,
                face_index,
            } => write!(
                formatter,
                "render batch {batch_index} has an invalid member for face {face_index}"
            ),
            Self::BatchOrder { batch_index } => write!(
                formatter,
                "render batch {batch_index} is duplicate or out of canonical order"
            ),
            Self::DuplicateFace(face) => {
                write!(
                    formatter,
                    "source face {face} occurs in multiple render batches"
                )
            }
            Self::DuplicatePatch {
                face_index,
                leaf_depth,
                leaf_path,
            } => write!(
                formatter,
                "adaptive leaf {leaf_depth}:{leaf_path} of source face {face_index} occurs more than once"
            ),
            Self::OverlappingPatch {
                face_index,
                ancestor_depth,
                ancestor_path,
                descendant_depth,
                descendant_path,
            } => write!(
                formatter,
                "adaptive leaf {ancestor_depth}:{ancestor_path} overlaps descendant {descendant_depth}:{descendant_path} on source face {face_index}"
            ),
            Self::BatchCountOverflow => formatter.write_str("render batch count exceeds u32"),
            Self::InstanceCountOverflow => formatter.write_str("render instance count exceeds u32"),
            Self::SceneRevisionMismatch { frame, scene } => write!(
                formatter,
                "render frame scene revision {frame} does not match snapshot {scene}"
            ),
            Self::CommandSequenceMismatch => {
                formatter.write_str("render command sequence is not canonical")
            }
            Self::CommandBatchMissing { batch_index } => {
                write!(
                    formatter,
                    "render command references missing batch {batch_index}"
                )
            }
            Self::ObserverDisabled => formatter.write_str("render parity observer is disabled"),
            Self::ObserverSceneUnavailable => {
                formatter.write_str("render parity observer has no scene snapshot")
            }
        }
    }
}

impl Error for RenderContractError {}

fn finite<const N: usize>(values: [f32; N]) -> bool {
    values.into_iter().all(f32::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    const IDENTITY_MOBIUS: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ];

    fn transform() -> RenderEntityTransform {
        RenderEntityTransform {
            mobius: IDENTITY_MOBIUS,
            orientation_sign: 1,
            euclidean_model: IDENTITY,
            euclidean_normal: IDENTITY,
        }
    }

    fn batch(
        material_index: usize,
        face_index: u32,
        pbr_class: PbrDrawClass,
        enabled: bool,
    ) -> RenderBatchSnapshot {
        RenderBatchSnapshot {
            key: RenderBatchKey {
                lod: [2, 2, 2],
                parity_bucket: 0,
                material_index,
                render_node_index: 0,
            },
            members: vec![RenderBatchMember {
                face_index,
                leaf_id: crate::screen_partition::ScreenPatchLeafId::ROOT,
                node_index: 0,
                edge_lods: [2; 3],
                permutation_index: 0,
                vertex_lods: [2; 3],
            }],
            triangle_index_count: 6,
            line_index_count: 6,
            transform: transform(),
            enabled,
            pbr_class,
        }
    }

    fn scene() -> RenderSceneSnapshot {
        RenderSceneSnapshot {
            revision: 7,
            batches: vec![
                batch(0, 0, PbrDrawClass::Opaque, true),
                batch(1, 1, PbrDrawClass::Blend, false),
                batch(2, 2, PbrDrawClass::Transmission, true),
            ],
        }
    }

    fn view() -> RenderView {
        RenderView {
            viewport: [1280, 720],
            mvp: IDENTITY,
            model_view: IDENTITY,
            camera_position: [0.0, 0.0, 3.0],
            selected_node: Some(0),
            focus: FocusFieldPacket::default(),
        }
    }

    #[test]
    fn pbr_frame_orders_prepare_opaque_resolve_transparent_and_focus() {
        let scene = scene();
        let frame = RenderFrame::build(
            11,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::Pbr,
            view(),
            RenderFrameOptions {
                focus_postprocess: true,
                highlight_face: Some(0),
            },
            &scene,
        )
        .unwrap();
        assert_eq!(frame.commands.len(), 8);
        assert!(matches!(
            frame.commands[0],
            RenderCommand::PreparePatches {
                batch_index: 0,
                instance_count: 1
            }
        ));
        assert!(matches!(
            frame.commands[1],
            RenderCommand::PreparePatches {
                batch_index: 1,
                instance_count: 0
            }
        ));
        assert!(matches!(
            frame.commands[3],
            RenderCommand::DrawPatches {
                batch_index: 0,
                pass: RenderPass::PbrOpaque,
                ..
            }
        ));
        assert_eq!(frame.commands[4], RenderCommand::BuildTransmissionPyramid);
        assert_eq!(frame.commands[7], RenderCommand::FocusPostProcess);
        assert!(!frame
            .commands
            .iter()
            .any(|command| matches!(command, RenderCommand::HighlightFace { .. })));
        assert_eq!(
            frame.expected_submission_stats(&scene).unwrap(),
            RenderSubmissionStats {
                draw_calls: 3,
                zero_instance_draw_calls: 1,
                invalid_draw_calls: 0,
                submitted_instances: 2,
                triangles: 4,
                lines: 0,
            }
        );
        frame.validate(&scene).unwrap();
    }

    #[test]
    fn matcap_wire_has_two_ordered_draw_passes_and_highlight() {
        let mut scene = scene();
        scene.batches.truncate(2);
        let frame = RenderFrame::build(
            12,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::MatcapWire,
            view(),
            RenderFrameOptions {
                focus_postprocess: false,
                highlight_face: Some(4),
            },
            &scene,
        )
        .unwrap();
        assert_eq!(frame.commands.len(), 7);
        assert!(matches!(
            frame.commands[2],
            RenderCommand::DrawPatches {
                pass: RenderPass::Matcap,
                ..
            }
        ));
        assert!(matches!(
            frame.commands[4],
            RenderCommand::DrawPatches {
                pass: RenderPass::Wire,
                ..
            }
        ));
        assert_eq!(
            frame.commands[6],
            RenderCommand::HighlightFace { face_index: 4 }
        );
        assert_eq!(
            frame.expected_submission_stats(&scene).unwrap(),
            RenderSubmissionStats {
                draw_calls: 4,
                zero_instance_draw_calls: 2,
                invalid_draw_calls: 0,
                submitted_instances: 2,
                triangles: 2,
                lines: 3,
            }
        );
    }

    #[test]
    fn validation_rejects_duplicate_faces_bad_permutations_and_stale_commands() {
        let mut duplicate = scene();
        duplicate.batches[1].members[0].face_index = 0;
        assert_eq!(
            duplicate.validate(),
            Err(RenderContractError::DuplicateFace(0))
        );

        let mut bad_permutation = scene();
        bad_permutation.batches[0].members[0].permutation_index = 6;
        assert!(matches!(
            bad_permutation.validate(),
            Err(RenderContractError::InvalidBatchMember { .. })
        ));

        let scene = scene();
        let mut frame = RenderFrame::build(
            13,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::Matcap,
            view(),
            RenderFrameOptions::default(),
            &scene,
        )
        .unwrap();
        frame.commands.pop();
        assert_eq!(
            frame.validate(&scene),
            Err(RenderContractError::CommandSequenceMismatch)
        );
        frame.scene_revision = 8;
        assert!(matches!(
            frame.validate(&scene),
            Err(RenderContractError::SceneRevisionMismatch { .. })
        ));
    }

    #[test]
    fn validation_distinguishes_adaptive_leaves_from_authored_faces() {
        let mut adaptive = scene();
        let first = crate::screen_partition::ScreenPatchLeafId::ROOT
            .child(0)
            .unwrap();
        let second = crate::screen_partition::ScreenPatchLeafId::ROOT
            .child(1)
            .unwrap();
        adaptive.batches[0].members[0].leaf_id = first;
        let first_member = adaptive.batches[0].members[0];
        adaptive.batches[0].members.push(RenderBatchMember {
            leaf_id: second,
            ..first_member
        });
        assert_eq!(adaptive.validate(), Ok(()));

        adaptive.batches[0].members[1].leaf_id = first;
        assert_eq!(
            adaptive.validate(),
            Err(RenderContractError::DuplicatePatch {
                face_index: 0,
                leaf_depth: first.depth,
                leaf_path: first.path,
            })
        );

        let mut overlapping = scene();
        let root_member = overlapping.batches[0].members[0];
        overlapping.batches[0].members.push(RenderBatchMember {
            leaf_id: first,
            ..root_member
        });
        assert_eq!(
            overlapping.validate(),
            Err(RenderContractError::OverlappingPatch {
                face_index: 0,
                ancestor_depth: 0,
                ancestor_path: 0,
                descendant_depth: first.depth,
                descendant_path: first.path,
            })
        );
    }

    #[test]
    fn submission_stats_distinguish_work_zero_instances_and_invalid_draws() {
        let mut stats = RenderSubmissionStats::default();
        stats.record_indexed_draw(RenderGeometry::Triangles, 7, 3);
        stats.record_indexed_draw(RenderGeometry::Lines, 8, 2);
        stats.record_indexed_draw(RenderGeometry::Triangles, 12, 0);
        stats.record_invalid_draw();

        assert_eq!(
            stats,
            RenderSubmissionStats {
                draw_calls: 4,
                zero_instance_draw_calls: 1,
                invalid_draw_calls: 1,
                submitted_instances: 5,
                triangles: 6,
                lines: 8,
            }
        );

        let mut total = RenderSubmissionStats {
            draw_calls: u64::MAX,
            ..RenderSubmissionStats::default()
        };
        total.merge(stats);
        assert_eq!(total.draw_calls, u64::MAX);
        assert_eq!(total.triangles, 6);
    }

    #[test]
    fn parity_observer_retains_only_valid_scenes_and_bounded_diagnostics() {
        let scene = scene();
        let frame = RenderFrame::build(
            21,
            RenderPoseIdentity {
                asset_revision: 3,
                pose_revision: 5,
            },
            RenderStyle::Pbr,
            view(),
            RenderFrameOptions::default(),
            &scene,
        )
        .unwrap();
        let expected = frame.expected_submission_stats(&scene).unwrap();
        let mut observer = RenderParityObserver::default();
        assert_eq!(
            observer.observe(&frame, expected),
            Err(RenderContractError::ObserverDisabled)
        );

        observer.set_enabled(true);
        assert_eq!(
            observer.observe(&frame, expected),
            Err(RenderContractError::ObserverSceneUnavailable)
        );
        observer.replace_scene(scene.clone()).unwrap();
        assert!(observer.observe(&frame, expected).unwrap().is_match());

        let mut actual = expected;
        actual.submitted_instances += 1;
        let mismatch = observer.observe(&frame, actual).unwrap();
        assert!(mismatch.mismatch.submitted_instances);
        assert!(!mismatch.mismatch.draw_calls);
        assert_eq!(
            observer.diagnostics(),
            RenderParityDiagnostics {
                enabled: true,
                scene_revision: Some(7),
                scene_rebuilds: 1,
                frames_observed: 2,
                matching_frames: 1,
                mismatching_frames: 1,
                last: Some(mismatch),
            }
        );

        let mut invalid = scene;
        invalid.batches[0].triangle_index_count = 4;
        assert!(observer.replace_scene(invalid).is_err());
        assert_eq!(observer.diagnostics().scene_revision, Some(7));

        observer.set_enabled(false);
        assert_eq!(observer.diagnostics(), RenderParityDiagnostics::default());
    }
}
