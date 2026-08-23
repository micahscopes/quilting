//! Backend-neutral render extraction.
//!
//! [`RenderSceneSnapshot`] changes only when retained batch membership or
//! entity/material state changes. [`RenderFrame`] carries the high-rate view,
//! pose identity, and ordered logical commands. Neither type contains a GL
//! handle, WebGPU resource, DOM object, or platform callback.

use crate::batch::{RenderBatchKey, RenderBatchMember};
use crate::permutation::perm_sign;
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
        let mut faces = BTreeSet::new();
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
            batch.transform.validate()?;
            batch.active_instance_count()?;
            for member in &batch.members {
                if member.permutation_index >= 6
                    || usize::from(perm_sign(member.permutation_index as usize) < 0)
                        != usize::from(batch.key.parity_bucket)
                    || member
                        .vertex_lods
                        .into_iter()
                        .any(|lod| lod == 0 || !lod.is_power_of_two())
                {
                    return Err(RenderContractError::InvalidBatchMember {
                        batch_index,
                        face_index: member.face_index,
                    });
                }
                if !faces.insert(member.face_index) {
                    return Err(RenderContractError::DuplicateFace(member.face_index));
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
    InvalidBatchMember { batch_index: usize, face_index: u32 },
    BatchOrder { batch_index: usize },
    DuplicateFace(u32),
    BatchCountOverflow,
    InstanceCountOverflow,
    SceneRevisionMismatch { frame: u64, scene: u64 },
    CommandSequenceMismatch,
}

impl fmt::Display for RenderContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransform => formatter.write_str("render transform is invalid"),
            Self::InvalidView => formatter.write_str("render view is invalid"),
            Self::InvalidBatchKey { batch_index } => {
                write!(formatter, "render batch {batch_index} has an invalid key")
            }
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
            Self::BatchCountOverflow => formatter.write_str("render batch count exceeds u32"),
            Self::InstanceCountOverflow => formatter.write_str("render instance count exceeds u32"),
            Self::SceneRevisionMismatch { frame, scene } => write!(
                formatter,
                "render frame scene revision {frame} does not match snapshot {scene}"
            ),
            Self::CommandSequenceMismatch => {
                formatter.write_str("render command sequence is not canonical")
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
                node_index: 0,
            },
            members: vec![RenderBatchMember {
                face_index,
                permutation_index: 0,
                vertex_lods: [2; 3],
            }],
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
}
