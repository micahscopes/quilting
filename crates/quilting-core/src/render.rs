//! Backend-neutral render extraction.
//!
//! [`RenderSceneSnapshot`] changes only when retained batch membership or
//! entity/material state changes. [`RenderFrame`] carries the high-rate view,
//! pose identity, and ordered logical commands. Neither type contains a GL
//! handle, WebGPU resource, DOM object, or platform callback.

use crate::batch::{RenderBatchId, RenderBatchLayer, RenderBatchMember};
use crate::material::PbrMaterial;
use crate::permutation::perm_sign;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

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
    pub id: RenderBatchId,
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
    /// Authored material table addressed by `RenderBatchKey::material_index`.
    /// Empty tables retain the glTF default-material semantics.
    pub materials: Vec<PbrMaterial>,
    /// Exact roots omitted from the logical retained layer and replaced by
    /// adaptive overlay members. Physical WebGL2 root batches may still
    /// dispatch these instances with a zero visibility scalar; WebGPU may
    /// compact them away.
    pub suppressed_root_faces: Vec<u32>,
    pub batches: Vec<RenderBatchSnapshot>,
}

/// Immutable render state shared by every retained source root in one draw
/// domain. Atlas topology and permutation parity are deliberately absent:
/// those change with LOD and are appended to this domain on the GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidentRootDrawDomain {
    pub material_index: usize,
    pub render_node_index: usize,
    pub pbr_class: PbrDrawClass,
    pub transform: RenderEntityTransform,
    pub enabled: bool,
}

/// Dense rows for only the material/render domains actually present in a
/// retained root scene. The row per source face is immutable across LOD
/// changes, avoiding an atlas × material × node Cartesian allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentRootDrawDomains {
    pub domains: Vec<ResidentRootDrawDomain>,
    pub face_domain_rows: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentRootDrawDomainError {
    InvalidScene(String),
    FaceCountOverflow,
    FaceOutOfBounds { face_index: u32, face_count: usize },
    DuplicateRootFace(u32),
    MissingRootFace(u32),
    ConflictingDomainState {
        material_index: usize,
        render_node_index: usize,
    },
}

impl fmt::Display for ResidentRootDrawDomainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScene(error) => write!(formatter, "invalid render scene: {error}"),
            Self::FaceCountOverflow => write!(formatter, "resident root face count exceeds u32"),
            Self::FaceOutOfBounds {
                face_index,
                face_count,
            } => write!(
                formatter,
                "resident root face {face_index} exceeds the {face_count}-face domain",
            ),
            Self::DuplicateRootFace(face_index) => {
                write!(formatter, "resident root face {face_index} appears more than once")
            }
            Self::MissingRootFace(face_index) => {
                write!(formatter, "resident root face {face_index} has no draw domain")
            }
            Self::ConflictingDomainState {
                material_index,
                render_node_index,
            } => write!(
                formatter,
                "resident root domain material {material_index}, render node {render_node_index} has conflicting state",
            ),
        }
    }
}

impl Error for ResidentRootDrawDomainError {}

impl ResidentRootDrawDomains {
    /// Extract a complete source-face mapping while retaining only observed
    /// material/render domains. Adaptive leaves do not create root domains;
    /// their sparse draw layer is planned independently.
    pub fn build(
        scene: &RenderSceneSnapshot,
        face_count: usize,
    ) -> Result<Self, ResidentRootDrawDomainError> {
        scene
            .validate()
            .map_err(|error| ResidentRootDrawDomainError::InvalidScene(error.to_string()))?;
        u32::try_from(face_count).map_err(|_| ResidentRootDrawDomainError::FaceCountOverflow)?;

        type DomainKey = (usize, usize);
        let mut states = BTreeMap::<DomainKey, ResidentRootDrawDomain>::new();
        let mut face_keys = vec![None::<DomainKey>; face_count];
        for batch in &scene.batches {
            if batch.id.layer == RenderBatchLayer::AdaptiveOverlay {
                continue;
            }
            if !batch
                .members
                .iter()
                .any(|member| member.leaf_id == crate::screen_partition::ScreenPatchLeafId::ROOT)
            {
                continue;
            }
            let key = (
                batch.id.key.material_index,
                batch.id.key.render_node_index,
            );
            let domain = ResidentRootDrawDomain {
                material_index: key.0,
                render_node_index: key.1,
                pbr_class: batch.pbr_class,
                transform: batch.transform,
                enabled: batch.enabled,
            };
            match states.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(domain);
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() != &domain =>
                {
                    return Err(ResidentRootDrawDomainError::ConflictingDomainState {
                        material_index: key.0,
                        render_node_index: key.1,
                    });
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
            for member in &batch.members {
                if member.leaf_id != crate::screen_partition::ScreenPatchLeafId::ROOT {
                    continue;
                }
                let face = member.face_index as usize;
                let Some(destination) = face_keys.get_mut(face) else {
                    return Err(ResidentRootDrawDomainError::FaceOutOfBounds {
                        face_index: member.face_index,
                        face_count,
                    });
                };
                if destination.replace(key).is_some() {
                    return Err(ResidentRootDrawDomainError::DuplicateRootFace(
                        member.face_index,
                    ));
                }
            }
        }

        let domains = states.values().copied().collect::<Vec<_>>();
        if domains.len() > u32::MAX as usize {
            return Err(ResidentRootDrawDomainError::FaceCountOverflow);
        }
        let rows = states
            .keys()
            .copied()
            .enumerate()
            .map(|(row, key)| (key, row as u32))
            .collect::<BTreeMap<_, _>>();
        let face_domain_rows = face_keys
            .into_iter()
            .enumerate()
            .map(|(face, key)| {
                let key = key.ok_or(ResidentRootDrawDomainError::MissingRootFace(face as u32))?;
                Ok(rows[&key])
            })
            .collect::<Result<Vec<_>, ResidentRootDrawDomainError>>()?;
        Ok(Self {
            domains,
            face_domain_rows,
        })
    }
}

impl RenderSceneSnapshot {
    pub fn validate(&self) -> Result<(), RenderContractError> {
        for (material_index, material) in self.materials.iter().enumerate() {
            if material.validate().is_err() {
                return Err(RenderContractError::InvalidMaterial { material_index });
            }
        }
        if self
            .suppressed_root_faces
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(RenderContractError::SuppressedFaceOrder);
        }
        let suppressed = self
            .suppressed_root_faces
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let uses_complete = self
            .batches
            .iter()
            .any(|batch| batch.id.layer == RenderBatchLayer::Complete);
        let uses_retained = self
            .batches
            .iter()
            .any(|batch| batch.id.layer != RenderBatchLayer::Complete);
        if uses_complete && uses_retained {
            return Err(RenderContractError::MixedBatchLayers);
        }
        if uses_complete && !suppressed.is_empty() {
            return Err(RenderContractError::UnexpectedSuppression);
        }

        let mut previous = None;
        let mut patches = BTreeSet::new();
        let mut suppressed_roots = BTreeSet::new();
        let mut overlay_faces = BTreeSet::new();
        for (batch_index, batch) in self.batches.iter().enumerate() {
            if previous.is_some_and(|id| id >= batch.id) {
                return Err(RenderContractError::BatchOrder { batch_index });
            }
            previous = Some(batch.id);
            if batch.id.key.parity_bucket > 1
                || batch
                    .id
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
                    || member_lod.canonical != batch.id.key.lod
                    || member_lod.perm_index.min(5) as u8 != member.permutation_index
                    || usize::from(perm_sign(member.permutation_index as usize) < 0)
                        != usize::from(batch.id.key.parity_bucket)
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
                match batch.id.layer {
                    RenderBatchLayer::Complete => {}
                    RenderBatchLayer::RetainedRoot => {
                        if member.leaf_id
                            != crate::screen_partition::ScreenPatchLeafId::ROOT
                        {
                            return Err(RenderContractError::InvalidLayerMember {
                                batch_index,
                                face_index: member.face_index,
                            });
                        }
                        if suppressed.contains(&member.face_index) {
                            suppressed_roots.insert(member.face_index);
                            continue;
                        }
                    }
                    RenderBatchLayer::AdaptiveOverlay => {
                        if !suppressed.contains(&member.face_index) {
                            return Err(RenderContractError::UnmaskedAdaptiveReplacement(
                                member.face_index,
                            ));
                        }
                        overlay_faces.insert(member.face_index);
                    }
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
        if let Some(face_index) = suppressed
            .difference(&suppressed_roots)
            .next()
            .copied()
        {
            return Err(RenderContractError::MissingSuppressedRoot(face_index));
        }
        if let Some(face_index) = suppressed.difference(&overlay_faces).next().copied() {
            return Err(RenderContractError::MissingAdaptiveReplacement(face_index));
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

/// One backend-neutral range in the stable source-instance stream and its
/// corresponding compacted survivor stream. WebGPU can write the survivor
/// indices into storage and copy the compacted fields directly into per-bucket
/// indirect arguments; WebGL2 retains its degenerate-vertex fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactedRenderBatchRange {
    pub batch_index: u32,
    pub source_first_instance: u32,
    pub source_instance_count: u32,
    pub compacted_first_instance: u32,
    pub compacted_instance_count: u32,
}

/// Exact WebGPU/WebGL indexed-indirect field order. Portable records keep
/// `first_instance` zero because nonzero indirect values require WebGPU's
/// optional `indirect-first-instance` feature. The compacted prefix remains in
/// [`CompactedRenderBatchRange`] for a vertex stage to add explicitly. A packed
/// atlas may patch `first_index` and `base_vertex` without changing compaction.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexedIndirectArguments {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Deterministic CPU oracle for same-pose GPU visibility compaction.
/// `compacted_source_instances` stores flattened source-instance IDs in stable
/// batch/member order; it does not copy patch records or tessellated geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisibilityCompactionPlan {
    pub scene_revision: u64,
    pub source_instance_count: u32,
    pub compacted_source_instances: Vec<u32>,
    pub batches: Vec<CompactedRenderBatchRange>,
}

impl VisibilityCompactionPlan {
    pub fn build(
        scene: &RenderSceneSnapshot,
        source_visibility: &[u8],
    ) -> Result<Self, RenderContractError> {
        scene.validate()?;
        let expected_instances = scene.batches.iter().try_fold(0usize, |total, batch| {
            total
                .checked_add(batch.members.len())
                .ok_or(RenderContractError::InstanceCountOverflow)
        })?;
        if source_visibility.len() != expected_instances {
            return Err(RenderContractError::VisibilityLengthMismatch {
                expected: expected_instances,
                actual: source_visibility.len(),
            });
        }

        let source_instance_count = expected_instances
            .try_into()
            .map_err(|_| RenderContractError::InstanceCountOverflow)?;
        let mut compacted_source_instances = Vec::new();
        let mut batches = Vec::with_capacity(scene.batches.len());
        let mut source_first_instance = 0usize;
        for (batch_index, batch) in scene.batches.iter().enumerate() {
            let compacted_first_instance = compacted_source_instances.len();
            for (member_index, member) in batch.members.iter().enumerate() {
                let source_instance = source_first_instance + member_index;
                let visibility = source_visibility[source_instance];
                if visibility > 1 {
                    return Err(RenderContractError::InvalidVisibilityValue {
                        source_instance,
                        value: visibility,
                    });
                }
                let suppressed_root = batch.id.layer == RenderBatchLayer::RetainedRoot
                    && scene
                        .suppressed_root_faces
                        .binary_search(&member.face_index)
                        .is_ok();
                if batch.enabled && visibility == 1 && !suppressed_root {
                    compacted_source_instances.push(
                        source_instance
                            .try_into()
                            .map_err(|_| RenderContractError::InstanceCountOverflow)?,
                    );
                }
            }
            let source_instance_count = batch
                .members
                .len()
                .try_into()
                .map_err(|_| RenderContractError::InstanceCountOverflow)?;
            let compacted_instance_count = compacted_source_instances
                .len()
                .checked_sub(compacted_first_instance)
                .expect("compaction output length is monotonic")
                .try_into()
                .map_err(|_| RenderContractError::InstanceCountOverflow)?;
            batches.push(CompactedRenderBatchRange {
                batch_index: batch_index
                    .try_into()
                    .map_err(|_| RenderContractError::BatchCountOverflow)?,
                source_first_instance: source_first_instance
                    .try_into()
                    .map_err(|_| RenderContractError::InstanceCountOverflow)?,
                source_instance_count,
                compacted_first_instance: compacted_first_instance
                    .try_into()
                    .map_err(|_| RenderContractError::InstanceCountOverflow)?,
                compacted_instance_count,
            });
            source_first_instance += batch.members.len();
        }
        Ok(Self {
            scene_revision: scene.revision,
            source_instance_count,
            compacted_source_instances,
            batches,
        })
    }

    /// Produce one exact five-word indexed-indirect record per canonical batch.
    /// The backend may issue only the records selected by the active draw pass;
    /// zero-instance records remain useful for a fixed-size GPU argument table.
    pub fn indexed_indirect_arguments(
        &self,
        scene: &RenderSceneSnapshot,
        geometry: RenderGeometry,
    ) -> Result<Vec<IndexedIndirectArguments>, RenderContractError> {
        scene.validate()?;
        if self.scene_revision != scene.revision {
            return Err(RenderContractError::CompactionSceneRevisionMismatch {
                plan: self.scene_revision,
                scene: scene.revision,
            });
        }
        if self.batches.len() != scene.batches.len() {
            return Err(RenderContractError::CompactionBatchShapeMismatch);
        }
        self.batches
            .iter()
            .zip(&scene.batches)
            .enumerate()
            .map(|(batch_index, (range, batch))| {
                if range.batch_index as usize != batch_index
                    || range.source_instance_count as usize != batch.members.len()
                {
                    return Err(RenderContractError::CompactionBatchShapeMismatch);
                }
                let index_count = match geometry {
                    RenderGeometry::Triangles => batch.triangle_index_count,
                    RenderGeometry::Lines => batch.line_index_count,
                };
                Ok(IndexedIndirectArguments {
                    index_count,
                    instance_count: range.compacted_instance_count,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusFieldPacket {
    pub sphere: [f32; 4],
    pub enabled: bool,
}

/// Backend-neutral scalar field used to build a focus-composition mask.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusPostprocessMode {
    DepthOfField,
    #[default]
    ConformalStretch,
    Hybrid,
    Spheroidal,
}

impl FocusPostprocessMode {
    pub const fn wire_index(self) -> u8 {
        match self {
            Self::DepthOfField => 0,
            Self::ConformalStretch => 1,
            Self::Hybrid => 2,
            Self::Spheroidal => 3,
        }
    }

    pub const fn from_wire_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::DepthOfField),
            1 => Some(Self::ConformalStretch),
            2 => Some(Self::Hybrid),
            3 => Some(Self::Spheroidal),
            _ => None,
        }
    }
}

/// Fully resolved focus-composition request consumed by a render backend.
///
/// `RenderFrameOptions::focus_postprocess == None` is the effective disabled
/// state. Spheroidal coordinates are resolved from Hyperscape focus state
/// before extraction, so this packet contains no application or navigation
/// authority and can be consumed identically by WebGL2 and WebGPU.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FocusPostprocessPacket {
    pub mode: FocusPostprocessMode,
    pub blur_radius_pixels: u16,
    pub blur_strength: f32,
    pub focus_coordinate: f32,
    pub bandwidth: f32,
    pub normalize_range: bool,
    pub stretch_range: [f32; 2],
    pub gaussian_passes: u8,
    pub kawase_passes: u8,
    pub kawase_offset: f32,
}

/// Canonical opaque frame clear shared by every Quilting render backend.
///
/// Background authorship can later promote this value into scene/frame state;
/// until then, keeping the incumbent color here prevents WebGL2, WebGPU, focus
/// composition, and parity targets from silently inventing different policy.
/// Every channel is also an exact RGBA8 code point; a half-code value such as
/// `0.3 * 255 = 76.5` may be quantized differently by WebGL2 and WebGPU.
pub const DEFAULT_RENDER_CLEAR_COLOR: [f32; 4] = [0.2, 0.2, 77.0 / 255.0, 1.0];

impl FocusPostprocessPacket {
    pub fn validate(self) -> Result<Self, RenderContractError> {
        let valid_stretch_range = finite(self.stretch_range)
            && (!self.normalize_range || self.stretch_range[0] <= self.stretch_range[1]);
        if !(4..=128).contains(&self.blur_radius_pixels)
            || !self.blur_strength.is_finite()
            || !(0.1..=3.0).contains(&self.blur_strength)
            || !self.focus_coordinate.is_finite()
            || !(0.0..=1.0).contains(&self.focus_coordinate)
            || !self.bandwidth.is_finite()
            || !(0.01..=0.5).contains(&self.bandwidth)
            || !valid_stretch_range
            || !(1..=4).contains(&self.gaussian_passes)
            || self.kawase_passes > 4
            || !self.kawase_offset.is_finite()
            || !(0.1..=3.0).contains(&self.kawase_offset)
        {
            return Err(RenderContractError::InvalidFocusPostprocess);
        }
        Ok(self)
    }
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

impl RenderPoseIdentity {
    /// Identity for an unanimated asset revision.
    pub const fn static_asset(asset_revision: u64) -> Self {
        Self {
            asset_revision,
            pose_revision: 0,
        }
    }

    /// Losslessly combine an animation continuity epoch and its local pose
    /// revision. A restarted clip may reuse a local revision, but not the
    /// resulting backend identity.
    pub const fn timed(asset_revision: u64, revision: u32, continuity_epoch: u32) -> Self {
        Self {
            asset_revision,
            pose_revision: ((continuity_epoch as u64) << 32) | revision as u64,
        }
    }
}

/// Whether a backend must resolve a batch's source pose again. The global and
/// batch-local dirty flags carry topology/pose invalidation; ordinary affine
/// entity motion is compared explicitly because it is part of the prepared
/// patch record. The conformal map remains a later evaluation parameter.
pub fn patch_preparation_needed(
    global_dirty: bool,
    batch_dirty: bool,
    last_model: Option<[f32; 16]>,
    model: [f32; 16],
) -> bool {
    global_dirty || batch_dirty || last_model != Some(model)
}

/// Whether conservative visibility must be resolved for the current prepared
/// patches. `residency_revision` identifies the backend-neutral batch command
/// set, not a backend resource or command buffer.
pub fn patch_visibility_needed(
    pose_prepared: bool,
    last_mvp: Option<[f32; 16]>,
    last_residency_revision: u64,
    mvp: [f32; 16],
    residency_revision: u64,
) -> bool {
    pose_prepared
        || last_mvp != Some(mvp)
        || last_residency_revision != residency_revision
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderStyle {
    Pbr,
    Matcap,
    Wire,
    Normals,
    MatcapWire,
    Lod,
    Stretch,
}

impl RenderStyle {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Pbr => "pbr",
            Self::Matcap => "matcap",
            Self::Wire => "wire",
            Self::Normals => "normals",
            Self::MatcapWire => "matcap_wire",
            Self::Lod => "lod",
            Self::Stretch => "stretch",
        }
    }

    pub fn from_wire_name(name: &str) -> Option<Self> {
        match name {
            "pbr" => Some(Self::Pbr),
            "matcap" => Some(Self::Matcap),
            "wire" => Some(Self::Wire),
            "normals" => Some(Self::Normals),
            "matcap_wire" => Some(Self::MatcapWire),
            "lod" => Some(Self::Lod),
            "stretch" => Some(Self::Stretch),
            _ => None,
        }
    }
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

/// Backend-neutral selection applied by one ordered draw pass. Material and
/// resource binding remain backend work; this only defines which retained
/// batches participate in the pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBatchSelection {
    All,
    PbrOpaque,
    PbrNonOpaque,
}

impl RenderBatchSelection {
    pub fn includes(self, pbr_class: PbrDrawClass) -> bool {
        match self {
            Self::All => true,
            Self::PbrOpaque => pbr_class == PbrDrawClass::Opaque,
            Self::PbrNonOpaque => pbr_class != PbrDrawClass::Opaque,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderDrawPassPlan {
    pub pass: RenderPass,
    pub geometry: RenderGeometry,
    pub batches: RenderBatchSelection,
}

const PBR_DRAW_PASSES: [RenderDrawPassPlan; 2] = [
    RenderDrawPassPlan {
        pass: RenderPass::PbrOpaque,
        geometry: RenderGeometry::Triangles,
        batches: RenderBatchSelection::PbrOpaque,
    },
    RenderDrawPassPlan {
        pass: RenderPass::PbrTransparent,
        geometry: RenderGeometry::Triangles,
        batches: RenderBatchSelection::PbrNonOpaque,
    },
];
const MATCAP_DRAW_PASSES: [RenderDrawPassPlan; 1] = [RenderDrawPassPlan {
    pass: RenderPass::Matcap,
    geometry: RenderGeometry::Triangles,
    batches: RenderBatchSelection::All,
}];
const WIRE_DRAW_PASSES: [RenderDrawPassPlan; 1] = [RenderDrawPassPlan {
    pass: RenderPass::Wire,
    geometry: RenderGeometry::Lines,
    batches: RenderBatchSelection::All,
}];
const NORMAL_DRAW_PASSES: [RenderDrawPassPlan; 1] = [RenderDrawPassPlan {
    pass: RenderPass::Normals,
    geometry: RenderGeometry::Triangles,
    batches: RenderBatchSelection::All,
}];
const MATCAP_WIRE_DRAW_PASSES: [RenderDrawPassPlan; 2] = [
    MATCAP_DRAW_PASSES[0],
    WIRE_DRAW_PASSES[0],
];
const LOD_DRAW_PASSES: [RenderDrawPassPlan; 1] = [RenderDrawPassPlan {
    pass: RenderPass::Lod,
    geometry: RenderGeometry::Triangles,
    batches: RenderBatchSelection::All,
}];
const STRETCH_DRAW_PASSES: [RenderDrawPassPlan; 1] = [RenderDrawPassPlan {
    pass: RenderPass::Stretch,
    geometry: RenderGeometry::Triangles,
    batches: RenderBatchSelection::All,
}];

/// Canonical ordered draw-pass plan shared by command extraction and concrete
/// render backends. It deliberately excludes API objects and resource state.
pub fn render_draw_passes(style: RenderStyle) -> &'static [RenderDrawPassPlan] {
    match style {
        RenderStyle::Pbr => &PBR_DRAW_PASSES,
        RenderStyle::Matcap => &MATCAP_DRAW_PASSES,
        RenderStyle::Wire => &WIRE_DRAW_PASSES,
        RenderStyle::Normals => &NORMAL_DRAW_PASSES,
        RenderStyle::MatcapWire => &MATCAP_WIRE_DRAW_PASSES,
        RenderStyle::Lod => &LOD_DRAW_PASSES,
        RenderStyle::Stretch => &STRETCH_DRAW_PASSES,
    }
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
    /// Deterministic rolling fingerprint of ordered patch submissions. Unlike
    /// the aggregate counters, this distinguishes pass and batch reordering.
    #[serde(serialize_with = "serialize_u64_hex")]
    pub draw_sequence_hash: u64,
}

fn serialize_u64_hex<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&format!("{value:016x}"))
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
    pub draw_sequence: bool,
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
            draw_sequence: expected.draw_sequence_hash != actual.draw_sequence_hash,
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
    scene: Option<ValidatedRenderScene>,
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
        let scene = ValidatedRenderScene::new(scene)?;
        self.replace_validated_scene(scene)
    }

    pub fn replace_validated_scene(
        &mut self,
        scene: ValidatedRenderScene,
    ) -> Result<(), RenderContractError> {
        if !self.enabled {
            return Err(RenderContractError::ObserverDisabled);
        }
        self.scene = Some(scene);
        self.scene_rebuilds = self.scene_rebuilds.saturating_add(1);
        Ok(())
    }

    pub fn scene(&self) -> Option<&RenderSceneSnapshot> {
        self.scene.as_ref().map(ValidatedRenderScene::snapshot)
    }

    pub fn validated_scene(&self) -> Option<&ValidatedRenderScene> {
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
        let (scene_revision, expected) = {
            let scene = self
                .scene
                .as_ref()
                .ok_or(RenderContractError::ObserverSceneUnavailable)?;
            (
                scene.snapshot().revision,
                frame.execution(scene.snapshot())?.submission_stats(),
            )
        };
        Ok(self.record_comparison(frame.revision, scene_revision, expected, actual))
    }

    pub fn observe_execution(
        &mut self,
        frame_revision: u64,
        execution: RenderExecution<'_, '_>,
        actual: RenderSubmissionStats,
    ) -> Result<RenderParityComparison, RenderContractError> {
        if !self.enabled {
            return Err(RenderContractError::ObserverDisabled);
        }
        let scene_revision = {
            let scene = self
                .scene
                .as_ref()
                .ok_or(RenderContractError::ObserverSceneUnavailable)?;
            if !scene.contains_snapshot(execution.scene()) {
                return Err(RenderContractError::ExecutionSceneMismatch);
            }
            scene.snapshot().revision
        };
        let expected = execution.submission_stats();
        Ok(self.record_comparison(frame_revision, scene_revision, expected, actual))
    }

    fn record_comparison(
        &mut self,
        frame_revision: u64,
        scene_revision: u64,
        expected: RenderSubmissionStats,
        actual: RenderSubmissionStats,
    ) -> RenderParityComparison {
        let comparison = RenderParityComparison {
            frame_revision,
            scene_revision,
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
        comparison
    }

    pub fn diagnostics(&self) -> RenderParityDiagnostics {
        RenderParityDiagnostics {
            enabled: self.enabled,
            scene_revision: self.scene.as_ref().map(|scene| scene.snapshot().revision),
            scene_rebuilds: self.scene_rebuilds,
            frames_observed: self.frames_observed,
            matching_frames: self.matching_frames,
            mismatching_frames: self.mismatching_frames,
            last: self.last,
        }
    }
}

impl RenderSubmissionStats {
    /// Record one valid indexed patch draw with its exact logical command
    /// identity. This is the allocation-free execution oracle used to compare
    /// backend submission order with [`RenderFrame::commands`].
    pub fn record_patch_draw(
        &mut self,
        batch_index: u32,
        pass: RenderPass,
        geometry: RenderGeometry,
        index_count: u32,
        instance_count: u32,
    ) {
        self.append_draw_fingerprint(draw_fingerprint(
            batch_index,
            pass,
            geometry,
            index_count,
            instance_count,
        ));
        self.record_indexed_draw_counts(geometry, index_count, instance_count);
    }

    /// Record one valid indexed patch draw. Incomplete trailing indices do not
    /// form a primitive, matching indexed triangle/line assembly. Callers that
    /// know the logical pass and batch should prefer [`Self::record_patch_draw`].
    pub fn record_indexed_draw(
        &mut self,
        geometry: RenderGeometry,
        index_count: u32,
        instance_count: u32,
    ) {
        self.append_draw_fingerprint(draw_fingerprint(
            u32::MAX,
            RenderPass::Matcap,
            geometry,
            index_count,
            instance_count,
        ));
        self.record_indexed_draw_counts(geometry, index_count, instance_count);
    }

    fn record_indexed_draw_counts(
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
        self.append_draw_fingerprint(INVALID_DRAW_FINGERPRINT);
        self.draw_calls = self.draw_calls.saturating_add(1);
        self.invalid_draw_calls = self.invalid_draw_calls.saturating_add(1);
    }

    fn append_draw_fingerprint(&mut self, fingerprint: u64) {
        self.draw_sequence_hash = self
            .draw_sequence_hash
            .wrapping_mul(DRAW_SEQUENCE_BASE)
            .wrapping_add(fingerprint);
    }

    /// Accumulate another submission interval without allowing diagnostic
    /// counters to wrap during a long-running session.
    pub fn merge(&mut self, other: Self) {
        self.draw_sequence_hash = self
            .draw_sequence_hash
            .wrapping_mul(wrapping_pow(DRAW_SEQUENCE_BASE, other.draw_calls))
            .wrapping_add(other.draw_sequence_hash);
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
    /// Resolve the current source pose into the resident patch-instance
    /// representation. Backends may elide this logical command when the pose,
    /// topology, and entity transform match an already prepared result.
    PreparePatches {
        batch_index: u32,
        instance_count: u32,
    },
    /// Classify complete posed patches against the guarded current view.
    /// WebGL2 lowers this to the retained one-float visibility stream; WebGPU
    /// may instead compact visible instances and produce indirect arguments.
    ResolveVisibility {
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

/// Texture-free matcap profiles shared by render backends and route adapters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum MatcapStyle {
    Aqua = 0,
    #[default]
    CitricAcid = 1,
    GoldenSoft = 2,
    SoftStudio = 3,
}

impl MatcapStyle {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    pub const fn as_f32(self) -> f32 {
        self.as_u32() as f32
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RenderFrameOptions {
    pub focus_postprocess: Option<FocusPostprocessPacket>,
    pub highlight_face: Option<u32>,
    pub matcap_style: MatcapStyle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderFrame {
    pub revision: u64,
    pub scene_revision: u64,
    pub pose: RenderPoseIdentity,
    pub style: RenderStyle,
    pub view: RenderView,
    pub options: RenderFrameOptions,
    pub commands: Arc<[RenderCommand]>,
    command_plan: Option<RenderCommandPlan>,
}

/// An immutable scene that has passed the complete semantic contract once.
/// Clones share the exact snapshot allocation, allowing command plans and
/// backends to retain rollback-safe scene epochs without copying member tables.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedRenderScene {
    snapshot: Arc<RenderSceneSnapshot>,
}

impl ValidatedRenderScene {
    pub fn new(snapshot: RenderSceneSnapshot) -> Result<Self, RenderContractError> {
        snapshot.validate()?;
        Ok(Self {
            snapshot: Arc::new(snapshot),
        })
    }

    pub fn snapshot(&self) -> &RenderSceneSnapshot {
        &self.snapshot
    }

    pub fn shares_snapshot_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.snapshot, &other.snapshot)
    }

    pub fn contains_snapshot(&self, snapshot: &RenderSceneSnapshot) -> bool {
        std::ptr::eq(self.snapshot(), snapshot)
    }
}

impl TryFrom<RenderSceneSnapshot> for ValidatedRenderScene {
    type Error = RenderContractError;

    fn try_from(snapshot: RenderSceneSnapshot) -> Result<Self, Self::Error> {
        Self::new(snapshot)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderCommandPlanKey {
    pub style: RenderStyle,
    pub focus_postprocess: bool,
    pub highlight_face: Option<u32>,
}

impl RenderCommandPlanKey {
    pub fn new(style: RenderStyle, options: RenderFrameOptions) -> Self {
        Self {
            style,
            focus_postprocess: options.focus_postprocess.is_some(),
            highlight_face: options.highlight_face,
        }
    }
}

/// Immutable low-rate command identity bound to one validated scene epoch.
/// View matrices, pose samples, focus parameters, and matcap uniforms do not
/// rebuild this plan unless they change command presence or batch cardinality.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderCommandPlan {
    key: RenderCommandPlanKey,
    scene: ValidatedRenderScene,
    commands: Arc<[RenderCommand]>,
}

impl RenderCommandPlan {
    pub fn build(
        scene: &ValidatedRenderScene,
        style: RenderStyle,
        options: RenderFrameOptions,
    ) -> Result<Self, RenderContractError> {
        if let Some(focus) = options.focus_postprocess {
            focus.validate()?;
        }
        Ok(Self {
            key: RenderCommandPlanKey::new(style, options),
            scene: scene.clone(),
            commands: expected_commands(style, options, &scene.snapshot().batches)?.into(),
        })
    }

    pub fn key(&self) -> RenderCommandPlanKey {
        self.key
    }

    pub fn scene(&self) -> &ValidatedRenderScene {
        &self.scene
    }

    pub fn commands(&self) -> &[RenderCommand] {
        &self.commands
    }

    pub fn matches(
        &self,
        scene: &ValidatedRenderScene,
        style: RenderStyle,
        options: RenderFrameOptions,
    ) -> bool {
        self.scene.shares_snapshot_with(scene)
            && self.key == RenderCommandPlanKey::new(style, options)
    }

    pub fn validate_for(
        &self,
        scene: &ValidatedRenderScene,
        style: RenderStyle,
        options: RenderFrameOptions,
    ) -> Result<(), RenderContractError> {
        if self.matches(scene, style, options) {
            Ok(())
        } else {
            Err(RenderContractError::CommandPlanMismatch)
        }
    }

    pub fn execution(&self) -> RenderExecution<'_, '_> {
        RenderExecution {
            commands: &self.commands,
            scene: self.scene.snapshot(),
        }
    }
}

/// One validated logical command resolved against the immutable scene it
/// addresses. Backend resources deliberately remain outside this value; a
/// renderer uses `batch_index` to select its retained device allocation while
/// consuming the canonical batch metadata and draw cardinality from here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolvedRenderCommand<'scene> {
    PreparePatches {
        batch_index: u32,
        batch: &'scene RenderBatchSnapshot,
        instance_count: u32,
    },
    ResolveVisibility {
        batch_index: u32,
        batch: &'scene RenderBatchSnapshot,
        instance_count: u32,
    },
    DrawPatches {
        batch_index: u32,
        batch: &'scene RenderBatchSnapshot,
        instance_count: u32,
        pass: RenderPass,
        geometry: RenderGeometry,
        index_count: u32,
    },
    BuildTransmissionPyramid,
    FocusPostProcess,
    HighlightFace {
        face_index: u32,
    },
}

/// Validated, allocation-free execution view over a frame or retained command
/// plan.
///
/// Constructing this view proves the scene revision and entire command stream
/// before a backend can touch device state. Iteration then resolves batch
/// references and index counts without backend-specific pass reconstruction.
#[derive(Debug, Clone, Copy)]
pub struct RenderExecution<'frame, 'scene> {
    commands: &'frame [RenderCommand],
    scene: &'scene RenderSceneSnapshot,
}

impl<'frame, 'scene> RenderExecution<'frame, 'scene> {
    pub fn iter(self) -> RenderExecutionIter<'frame, 'scene> {
        RenderExecutionIter {
            commands: self.commands.iter(),
            scene: self.scene,
        }
    }

    pub fn batches(self) -> &'scene [RenderBatchSnapshot] {
        &self.scene.batches
    }

    pub fn scene(self) -> &'scene RenderSceneSnapshot {
        self.scene
    }

    /// Derive the exact logical indexed work without revalidating or allocating.
    pub fn submission_stats(self) -> RenderSubmissionStats {
        let mut stats = RenderSubmissionStats::default();
        for command in self {
            let ResolvedRenderCommand::DrawPatches {
                batch_index,
                instance_count,
                pass,
                geometry,
                index_count,
                ..
            } = command
            else {
                continue;
            };
            stats.record_patch_draw(
                batch_index,
                pass,
                geometry,
                index_count,
                instance_count,
            );
        }
        stats
    }
}

impl<'frame, 'scene> IntoIterator for RenderExecution<'frame, 'scene> {
    type Item = ResolvedRenderCommand<'scene>;
    type IntoIter = RenderExecutionIter<'frame, 'scene>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct RenderExecutionIter<'frame, 'scene> {
    commands: std::slice::Iter<'frame, RenderCommand>,
    scene: &'scene RenderSceneSnapshot,
}

impl<'scene> Iterator for RenderExecutionIter<'_, 'scene> {
    type Item = ResolvedRenderCommand<'scene>;

    fn next(&mut self) -> Option<Self::Item> {
        let command = *self.commands.next()?;
        Some(match command {
            RenderCommand::PreparePatches {
                batch_index,
                instance_count,
            } => ResolvedRenderCommand::PreparePatches {
                batch_index,
                batch: &self.scene.batches[batch_index as usize],
                instance_count,
            },
            RenderCommand::ResolveVisibility {
                batch_index,
                instance_count,
            } => ResolvedRenderCommand::ResolveVisibility {
                batch_index,
                batch: &self.scene.batches[batch_index as usize],
                instance_count,
            },
            RenderCommand::DrawPatches {
                batch_index,
                instance_count,
                pass,
                geometry,
            } => {
                let batch = &self.scene.batches[batch_index as usize];
                let index_count = match geometry {
                    RenderGeometry::Triangles => batch.triangle_index_count,
                    RenderGeometry::Lines => batch.line_index_count,
                };
                ResolvedRenderCommand::DrawPatches {
                    batch_index,
                    batch,
                    instance_count,
                    pass,
                    geometry,
                    index_count,
                }
            }
            RenderCommand::BuildTransmissionPyramid => {
                ResolvedRenderCommand::BuildTransmissionPyramid
            }
            RenderCommand::FocusPostProcess => ResolvedRenderCommand::FocusPostProcess,
            RenderCommand::HighlightFace { face_index } => {
                ResolvedRenderCommand::HighlightFace { face_index }
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.commands.size_hint()
    }
}

impl ExactSizeIterator for RenderExecutionIter<'_, '_> {}
impl std::iter::FusedIterator for RenderExecutionIter<'_, '_> {}

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
        if let Some(focus) = options.focus_postprocess {
            focus.validate()?;
        }
        let commands = expected_commands(style, options, &scene.batches)?.into();
        Ok(Self {
            revision,
            scene_revision: scene.revision,
            pose,
            style,
            view,
            options,
            commands,
            command_plan: None,
        })
    }

    /// Build one high-rate frame by borrowing the immutable command identity
    /// of a retained low-rate plan. View, pose, and uniform-only options may
    /// change every frame without allocating or rebuilding the command stream.
    #[allow(clippy::too_many_arguments)]
    pub fn from_command_plan(
        revision: u64,
        pose: RenderPoseIdentity,
        view: RenderView,
        options: RenderFrameOptions,
        plan: &RenderCommandPlan,
    ) -> Result<Self, RenderContractError> {
        view.validate()?;
        if let Some(focus) = options.focus_postprocess {
            focus.validate()?;
        }
        let style = plan.key.style;
        plan.validate_for(plan.scene(), style, options)?;
        Ok(Self {
            revision,
            scene_revision: plan.scene().snapshot().revision,
            pose,
            style,
            view,
            options,
            commands: Arc::clone(&plan.commands),
            command_plan: Some(plan.clone()),
        })
    }

    /// The immutable command/scene provenance retained by
    /// [`Self::from_command_plan`]. Legacy frames deliberately return `None`
    /// and continue through the independent full-validation oracle.
    pub fn retained_command_plan(&self) -> Option<&RenderCommandPlan> {
        self.command_plan.as_ref()
    }

    pub fn validate(&self, scene: &RenderSceneSnapshot) -> Result<(), RenderContractError> {
        scene.validate()?;
        self.view.validate()?;
        if let Some(focus) = self.options.focus_postprocess {
            focus.validate()?;
        }
        if self.scene_revision != scene.revision {
            return Err(RenderContractError::SceneRevisionMismatch {
                frame: self.scene_revision,
                scene: scene.revision,
            });
        }
        if self.commands.as_ref()
            != expected_commands(self.style, self.options, &scene.batches)?.as_slice()
        {
            return Err(RenderContractError::CommandSequenceMismatch);
        }
        Ok(())
    }

    /// Validate a frame against the exact retained plan that created it.
    ///
    /// Unlike [`Self::validate`], this does not revalidate the immutable scene
    /// or allocate an expected command vector. Pointer identity proves that
    /// the frame still carries the plan's canonical command allocation; the
    /// key admits only uniform-only changes such as matcap profile or focus
    /// field values.
    pub fn validate_with_command_plan(
        &self,
        plan: &RenderCommandPlan,
    ) -> Result<(), RenderContractError> {
        self.view.validate()?;
        if let Some(focus) = self.options.focus_postprocess {
            focus.validate()?;
        }
        let scene_revision = plan.scene().snapshot().revision;
        if self.scene_revision != scene_revision {
            return Err(RenderContractError::SceneRevisionMismatch {
                frame: self.scene_revision,
                scene: scene_revision,
            });
        }
        plan.validate_for(plan.scene(), self.style, self.options)?;
        if !Arc::ptr_eq(&self.commands, &plan.commands) {
            return Err(RenderContractError::CommandPlanMismatch);
        }
        Ok(())
    }

    /// Validate once and expose the exact backend-neutral command execution
    /// order with all scene-dependent batch metadata resolved.
    pub fn execution<'frame, 'scene>(
        &'frame self,
        scene: &'scene RenderSceneSnapshot,
    ) -> Result<RenderExecution<'frame, 'scene>, RenderContractError> {
        if let Some(plan) = self.command_plan.as_ref() {
            if !plan.scene().contains_snapshot(scene) {
                return Err(RenderContractError::ExecutionSceneMismatch);
            }
            self.validate_with_command_plan(plan)?;
            return Ok(RenderExecution {
                commands: &self.commands,
                scene,
            });
        }
        self.validate(scene)?;
        Ok(RenderExecution {
            commands: &self.commands,
            scene,
        })
    }

    /// Expose the retained plan's allocation-free execution after proving the
    /// high-rate frame still belongs to that exact scene/command epoch.
    pub fn execution_with_command_plan<'plan>(
        &self,
        plan: &'plan RenderCommandPlan,
    ) -> Result<RenderExecution<'plan, 'plan>, RenderContractError> {
        self.validate_with_command_plan(plan)?;
        Ok(plan.execution())
    }

    /// Derive the indexed patch work implied by this validated frame. This is
    /// the backend-independent side of render shadowing; concrete renderers
    /// record an actual [`RenderSubmissionStats`] at their draw boundary.
    pub fn expected_submission_stats(
        &self,
        scene: &RenderSceneSnapshot,
    ) -> Result<RenderSubmissionStats, RenderContractError> {
        Ok(self.execution(scene)?.submission_stats())
    }
}

const DRAW_SEQUENCE_BASE: u64 = 0x9e37_79b1_85eb_ca87;
const INVALID_DRAW_FINGERPRINT: u64 = 0xd1ff_ffff_ffff_ffff;

fn draw_fingerprint(
    batch_index: u32,
    pass: RenderPass,
    geometry: RenderGeometry,
    index_count: u32,
    instance_count: u32,
) -> u64 {
    let pass = match pass {
        RenderPass::PbrOpaque => 0_u64,
        RenderPass::PbrTransparent => 1,
        RenderPass::Matcap => 2,
        RenderPass::Wire => 3,
        RenderPass::Normals => 4,
        RenderPass::Lod => 5,
        RenderPass::Stretch => 6,
    };
    let geometry = match geometry {
        RenderGeometry::Triangles => 0_u64,
        RenderGeometry::Lines => 1,
    };
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in [0xd1, pass as u8, geometry as u8]
        .into_iter()
        .chain(batch_index.to_le_bytes())
        .chain(index_count.to_le_bytes())
        .chain(instance_count.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn wrapping_pow(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

fn expected_commands(
    style: RenderStyle,
    options: RenderFrameOptions,
    batches: &[RenderBatchSnapshot],
) -> Result<Vec<RenderCommand>, RenderContractError> {
    let mut commands = Vec::with_capacity(batches.len().saturating_mul(4).saturating_add(3));
    for (batch_index, batch) in batches.iter().enumerate() {
        commands.push(RenderCommand::PreparePatches {
            batch_index: batch_index
                .try_into()
                .map_err(|_| RenderContractError::BatchCountOverflow)?,
            instance_count: batch.active_instance_count()?,
        });
    }
    for (batch_index, batch) in batches.iter().enumerate() {
        commands.push(RenderCommand::ResolveVisibility {
            batch_index: batch_index
                .try_into()
                .map_err(|_| RenderContractError::BatchCountOverflow)?,
            instance_count: batch.active_instance_count()?,
        });
    }

    let has_transmission = style == RenderStyle::Pbr
        && batches
            .iter()
            .any(|batch| batch.pbr_class == PbrDrawClass::Transmission);
    for draw_pass in render_draw_passes(style) {
        if draw_pass.pass == RenderPass::PbrTransparent && has_transmission {
            commands.push(RenderCommand::BuildTransmissionPyramid);
        }
        append_draw_commands(&mut commands, batches, *draw_pass)?;
    }
    if style == RenderStyle::Pbr && options.focus_postprocess.is_some() {
        commands.push(RenderCommand::FocusPostProcess);
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
    draw_pass: RenderDrawPassPlan,
) -> Result<(), RenderContractError> {
    for (batch_index, batch) in batches
        .iter()
        .enumerate()
        .filter(|(_, batch)| draw_pass.batches.includes(batch.pbr_class))
    {
        commands.push(RenderCommand::DrawPatches {
            batch_index: batch_index
                .try_into()
                .map_err(|_| RenderContractError::BatchCountOverflow)?,
            instance_count: batch.active_instance_count()?,
            pass: draw_pass.pass,
            geometry: draw_pass.geometry,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderContractError {
    InvalidTransform,
    InvalidView,
    InvalidFocusPostprocess,
    InvalidMaterial { material_index: usize },
    InvalidBatchKey { batch_index: usize },
    InvalidBatchGeometry { batch_index: usize },
    InvalidBatchMember { batch_index: usize, face_index: u32 },
    InvalidLayerMember { batch_index: usize, face_index: u32 },
    BatchOrder { batch_index: usize },
    SuppressedFaceOrder,
    MixedBatchLayers,
    UnexpectedSuppression,
    MissingSuppressedRoot(u32),
    MissingAdaptiveReplacement(u32),
    UnmaskedAdaptiveReplacement(u32),
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
    VisibilityLengthMismatch { expected: usize, actual: usize },
    InvalidVisibilityValue { source_instance: usize, value: u8 },
    CompactionSceneRevisionMismatch { plan: u64, scene: u64 },
    CompactionBatchShapeMismatch,
    SceneRevisionMismatch { frame: u64, scene: u64 },
    CommandSequenceMismatch,
    CommandPlanMismatch,
    ExecutionSceneMismatch,
    ObserverDisabled,
    ObserverSceneUnavailable,
}

impl fmt::Display for RenderContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransform => formatter.write_str("render transform is invalid"),
            Self::InvalidView => formatter.write_str("render view is invalid"),
            Self::InvalidFocusPostprocess => {
                formatter.write_str("focus postprocess packet is invalid")
            }
            Self::InvalidMaterial { material_index } => {
                write!(formatter, "render material {material_index} is invalid")
            }
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
            Self::InvalidLayerMember {
                batch_index,
                face_index,
            } => write!(
                formatter,
                "render batch {batch_index} has an invalid layered member for face {face_index}"
            ),
            Self::BatchOrder { batch_index } => write!(
                formatter,
                "render batch {batch_index} is duplicate or out of canonical order"
            ),
            Self::SuppressedFaceOrder => {
                formatter.write_str("suppressed root faces are not strictly increasing")
            }
            Self::MixedBatchLayers => {
                formatter.write_str("complete and retained render layers cannot be mixed")
            }
            Self::UnexpectedSuppression => {
                formatter.write_str("complete render batches cannot suppress source roots")
            }
            Self::MissingSuppressedRoot(face_index) => write!(
                formatter,
                "suppressed source face {face_index} has no retained root instance"
            ),
            Self::MissingAdaptiveReplacement(face_index) => write!(
                formatter,
                "suppressed source face {face_index} has no adaptive replacement"
            ),
            Self::UnmaskedAdaptiveReplacement(face_index) => write!(
                formatter,
                "adaptive replacement for source face {face_index} is not masked in the root layer"
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
            Self::VisibilityLengthMismatch { expected, actual } => write!(
                formatter,
                "visibility stream has {actual} entries; expected {expected}"
            ),
            Self::InvalidVisibilityValue {
                source_instance,
                value,
            } => write!(
                formatter,
                "visibility stream entry {source_instance} has non-binary value {value}"
            ),
            Self::CompactionSceneRevisionMismatch { plan, scene } => write!(
                formatter,
                "visibility compaction revision {plan} does not match scene {scene}"
            ),
            Self::CompactionBatchShapeMismatch => {
                formatter.write_str("visibility compaction batch shape does not match the scene")
            }
            Self::SceneRevisionMismatch { frame, scene } => write!(
                formatter,
                "render frame scene revision {frame} does not match snapshot {scene}"
            ),
            Self::CommandSequenceMismatch => {
                formatter.write_str("render command sequence is not canonical")
            }
            Self::CommandPlanMismatch => {
                formatter.write_str("render command plan does not match the retained scene or options")
            }
            Self::ExecutionSceneMismatch => {
                formatter.write_str("render execution does not reference the observed scene epoch")
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

    #[test]
    fn default_clear_is_exactly_quantized_in_rgba8() {
        let codes = DEFAULT_RENDER_CLEAR_COLOR.map(|channel| channel * 255.0);
        assert_eq!(codes, [51.0, 51.0, 77.0, 255.0]);
    }

    #[test]
    fn timed_pose_identity_includes_continuity_epoch() {
        assert_eq!(RenderPoseIdentity::static_asset(7).pose_revision, 0);
        assert_eq!(
            RenderPoseIdentity::timed(7, 11, 3),
            RenderPoseIdentity {
                asset_revision: 7,
                pose_revision: (3u64 << 32) | 11,
            },
        );
        assert_ne!(
            RenderPoseIdentity::timed(7, 11, 3),
            RenderPoseIdentity::timed(7, 11, 4),
        );
    }

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
            id: RenderBatchId::complete(crate::batch::RenderBatchKey {
                lod: [2, 2, 2],
                parity_bucket: 0,
                material_index,
                render_node_index: 0,
            }),
            members: vec![member(
                face_index,
                crate::screen_partition::ScreenPatchLeafId::ROOT,
            )],
            triangle_index_count: 6,
            line_index_count: 6,
            transform: transform(),
            enabled,
            pbr_class,
        }
    }

    fn member(
        face_index: u32,
        leaf_id: crate::screen_partition::ScreenPatchLeafId,
    ) -> RenderBatchMember {
        RenderBatchMember {
            face_index,
            leaf_id,
            node_index: 0,
            edge_lods: [2; 3],
            permutation_index: 0,
            vertex_lods: [2; 3],
        }
    }

    fn scene() -> RenderSceneSnapshot {
        RenderSceneSnapshot {
            revision: 7,
            materials: Vec::new(),
            suppressed_root_faces: Vec::new(),
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
    fn pose_preparation_and_visibility_have_independent_revisions() {
        let mut mvp = [0.0; 16];
        mvp[0] = 1.0;
        assert!(!patch_preparation_needed(false, false, Some(mvp), mvp));
        assert!(patch_preparation_needed(true, false, Some(mvp), mvp));
        assert!(patch_preparation_needed(false, true, Some(mvp), mvp));
        assert!(patch_preparation_needed(false, false, None, mvp));
        assert!(!patch_visibility_needed(false, Some(mvp), 7, mvp, 7));
        assert!(patch_visibility_needed(true, Some(mvp), 7, mvp, 7));
        assert!(patch_visibility_needed(false, None, 7, mvp, 7));
        assert!(patch_visibility_needed(false, Some(mvp), 6, mvp, 7));
        let mut moved = mvp;
        moved[12] = 0.25;
        assert!(!patch_preparation_needed(false, false, Some(mvp), mvp));
        let mut moved_model = mvp;
        moved_model[13] = 0.5;
        assert!(patch_preparation_needed(false, false, Some(mvp), moved_model));
        assert!(patch_visibility_needed(false, Some(mvp), 7, moved, 7));
    }

    #[test]
    fn visibility_compaction_is_stable_and_emits_indirect_counts() {
        assert_eq!(std::mem::size_of::<IndexedIndirectArguments>(), 20);
        assert_eq!(std::mem::align_of::<IndexedIndirectArguments>(), 4);
        let mut scene = scene();
        scene.batches[0].members.push(member(
            3,
            crate::screen_partition::ScreenPatchLeafId::ROOT,
        ));
        let plan = VisibilityCompactionPlan::build(&scene, &[1, 0, 1, 1]).unwrap();
        assert_eq!(plan.source_instance_count, 4);
        assert_eq!(plan.compacted_source_instances, [0, 3]);
        assert_eq!(
            plan.batches,
            [
                CompactedRenderBatchRange {
                    batch_index: 0,
                    source_first_instance: 0,
                    source_instance_count: 2,
                    compacted_first_instance: 0,
                    compacted_instance_count: 1,
                },
                CompactedRenderBatchRange {
                    batch_index: 1,
                    source_first_instance: 2,
                    source_instance_count: 1,
                    compacted_first_instance: 1,
                    compacted_instance_count: 0,
                },
                CompactedRenderBatchRange {
                    batch_index: 2,
                    source_first_instance: 3,
                    source_instance_count: 1,
                    compacted_first_instance: 1,
                    compacted_instance_count: 1,
                },
            ],
        );
        let arguments = plan
            .indexed_indirect_arguments(&scene, RenderGeometry::Triangles)
            .unwrap();
        assert_eq!(
            arguments,
            [
                IndexedIndirectArguments {
                    index_count: 6,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                },
                IndexedIndirectArguments {
                    index_count: 6,
                    instance_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                },
                IndexedIndirectArguments {
                    index_count: 6,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                },
            ],
        );
    }

    #[test]
    fn visibility_compaction_replaces_suppressed_roots_with_visible_leaves() {
        let mut roots = batch(0, 0, PbrDrawClass::Opaque, true);
        roots.id.layer = RenderBatchLayer::RetainedRoot;
        roots.members.push(member(
            1,
            crate::screen_partition::ScreenPatchLeafId::ROOT,
        ));
        let mut overlay = batch(0, 0, PbrDrawClass::Opaque, true);
        overlay.id.layer = RenderBatchLayer::AdaptiveOverlay;
        overlay.members[0].leaf_id = crate::screen_partition::ScreenPatchLeafId::ROOT
            .child(0)
            .unwrap();
        let scene = RenderSceneSnapshot {
            revision: 11,
            materials: Vec::new(),
            suppressed_root_faces: vec![0],
            batches: vec![roots, overlay],
        };
        let plan = VisibilityCompactionPlan::build(&scene, &[1, 1, 1]).unwrap();
        assert_eq!(plan.compacted_source_instances, [1, 2]);
        assert_eq!(plan.batches[0].compacted_instance_count, 1);
        assert_eq!(plan.batches[1].compacted_instance_count, 1);
    }

    #[test]
    fn visibility_compaction_rejects_bad_streams_and_stale_scenes() {
        let scene = scene();
        assert_eq!(
            VisibilityCompactionPlan::build(&scene, &[1, 0]),
            Err(RenderContractError::VisibilityLengthMismatch {
                expected: 3,
                actual: 2,
            }),
        );
        assert_eq!(
            VisibilityCompactionPlan::build(&scene, &[1, 2, 1]),
            Err(RenderContractError::InvalidVisibilityValue {
                source_instance: 1,
                value: 2,
            }),
        );
        let mut plan = VisibilityCompactionPlan::build(&scene, &[1, 1, 1]).unwrap();
        let mut newer_scene = scene.clone();
        newer_scene.revision += 1;
        assert_eq!(
            plan.indexed_indirect_arguments(&newer_scene, RenderGeometry::Lines),
            Err(RenderContractError::CompactionSceneRevisionMismatch {
                plan: scene.revision,
                scene: newer_scene.revision,
            }),
        );
        plan.batches.pop();
        assert_eq!(
            plan.indexed_indirect_arguments(&scene, RenderGeometry::Lines),
            Err(RenderContractError::CompactionBatchShapeMismatch),
        );
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
                focus_postprocess: Some(FocusPostprocessPacket {
                    mode: FocusPostprocessMode::Spheroidal,
                    blur_radius_pixels: 11,
                    blur_strength: 1.0,
                    focus_coordinate: 0.5,
                    bandwidth: 0.1,
                    normalize_range: false,
                    stretch_range: [0.5, 0.5],
                    gaussian_passes: 1,
                    kawase_passes: 3,
                    kawase_offset: 1.5,
                }),
                highlight_face: Some(0),
                ..RenderFrameOptions::default()
            },
            &scene,
        )
        .unwrap();
        assert_eq!(frame.commands.len(), 11);
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
            RenderCommand::ResolveVisibility {
                batch_index: 0,
                instance_count: 1
            }
        ));
        assert!(matches!(
            frame.commands[4],
            RenderCommand::ResolveVisibility {
                batch_index: 1,
                instance_count: 0
            }
        ));
        assert!(matches!(
            frame.commands[6],
            RenderCommand::DrawPatches {
                batch_index: 0,
                pass: RenderPass::PbrOpaque,
                ..
            }
        ));
        assert_eq!(frame.commands[7], RenderCommand::BuildTransmissionPyramid);
        assert_eq!(frame.commands[10], RenderCommand::FocusPostProcess);
        assert!(!frame
            .commands
            .iter()
            .any(|command| matches!(command, RenderCommand::HighlightFace { .. })));
        let mut expected = RenderSubmissionStats::default();
        expected.record_patch_draw(0, RenderPass::PbrOpaque, RenderGeometry::Triangles, 6, 1);
        expected.record_patch_draw(
            1,
            RenderPass::PbrTransparent,
            RenderGeometry::Triangles,
            6,
            0,
        );
        expected.record_patch_draw(
            2,
            RenderPass::PbrTransparent,
            RenderGeometry::Triangles,
            6,
            1,
        );
        assert_eq!(frame.expected_submission_stats(&scene).unwrap(), expected);
        frame.validate(&scene).unwrap();
    }

    #[test]
    fn validated_execution_resolves_exact_batches_counts_and_command_order() {
        let scene = scene();
        let frame = RenderFrame::build(
            12,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::Pbr,
            view(),
            RenderFrameOptions {
                focus_postprocess: Some(FocusPostprocessPacket {
                    mode: FocusPostprocessMode::Spheroidal,
                    blur_radius_pixels: 11,
                    blur_strength: 1.0,
                    focus_coordinate: 0.5,
                    bandwidth: 0.1,
                    normalize_range: false,
                    stretch_range: [0.5, 0.5],
                    gaussian_passes: 1,
                    kawase_passes: 3,
                    kawase_offset: 1.5,
                }),
                ..RenderFrameOptions::default()
            },
            &scene,
        )
        .unwrap();

        let commands = frame.execution(&scene).unwrap().into_iter().collect::<Vec<_>>();
        assert_eq!(commands.len(), frame.commands.len());
        assert!(matches!(
            commands[1],
            ResolvedRenderCommand::PreparePatches {
                batch_index: 1,
                batch,
                instance_count: 0,
            } if std::ptr::eq(batch, &scene.batches[1])
        ));
        assert!(matches!(
            commands[6],
            ResolvedRenderCommand::DrawPatches {
                batch_index: 0,
                batch,
                instance_count: 1,
                pass: RenderPass::PbrOpaque,
                geometry: RenderGeometry::Triangles,
                index_count: 6,
            } if std::ptr::eq(batch, &scene.batches[0])
        ));
        assert_eq!(
            commands[7],
            ResolvedRenderCommand::BuildTransmissionPyramid
        );
        assert_eq!(commands[10], ResolvedRenderCommand::FocusPostProcess);
    }

    #[test]
    fn execution_rejects_mutated_commands_before_iteration() {
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
        Arc::make_mut(&mut frame.commands).swap(0, 1);

        assert_eq!(
            frame.execution(&scene).unwrap_err(),
            RenderContractError::CommandSequenceMismatch
        );
    }

    #[test]
    fn retained_command_plan_binds_one_validated_scene_without_hot_path_rebuilds() {
        let scene = scene();
        let validated = ValidatedRenderScene::new(scene.clone()).unwrap();
        let options = RenderFrameOptions::default();
        let plan = RenderCommandPlan::build(&validated, RenderStyle::Matcap, options).unwrap();
        assert!(plan.matches(&validated, RenderStyle::Matcap, options));
        assert_eq!(
            plan.validate_for(&validated, RenderStyle::Matcap, options),
            Ok(())
        );
        assert!(plan.scene().shares_snapshot_with(&validated));

        let frame = RenderFrame::build(
            14,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::Matcap,
            view(),
            options,
            &scene,
        )
        .unwrap();
        assert_eq!(plan.commands(), frame.commands.as_ref());
        assert_eq!(
            plan.execution().submission_stats(),
            frame.expected_submission_stats(&scene).unwrap()
        );

        let planned_frame = RenderFrame::from_command_plan(
            15,
            frame.pose,
            frame.view,
            options,
            &plan,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&planned_frame.commands, &plan.commands));
        assert!(planned_frame
            .retained_command_plan()
            .is_some_and(|retained| retained.scene().shares_snapshot_with(plan.scene())));
        assert_eq!(
            planned_frame
                .execution_with_command_plan(&plan)
                .unwrap()
                .submission_stats(),
            plan.execution().submission_stats(),
        );
        assert_eq!(
            planned_frame
                .execution(validated.snapshot())
                .unwrap()
                .submission_stats(),
            plan.execution().submission_stats(),
        );

        let mut uniform_only_frame_options = options;
        uniform_only_frame_options.matcap_style = MatcapStyle::SoftStudio;
        let uniform_only_frame = RenderFrame::from_command_plan(
            16,
            planned_frame.pose,
            planned_frame.view,
            uniform_only_frame_options,
            &plan,
        )
        .unwrap();
        assert!(Arc::ptr_eq(
            &planned_frame.commands,
            &uniform_only_frame.commands,
        ));

        assert_eq!(
            frame.execution_with_command_plan(&plan).unwrap_err(),
            RenderContractError::CommandPlanMismatch,
        );

        let equivalent_snapshot = scene.clone();
        assert_eq!(
            planned_frame.execution(&equivalent_snapshot).unwrap_err(),
            RenderContractError::ExecutionSceneMismatch,
        );

        let mut uniform_only = options;
        uniform_only.matcap_style = MatcapStyle::GoldenSoft;
        assert!(plan.matches(&validated, RenderStyle::Matcap, uniform_only));
        assert_eq!(
            plan.validate_for(&validated, RenderStyle::Matcap, uniform_only),
            Ok(())
        );

        let mut different_commands = options;
        different_commands.highlight_face = Some(7);
        assert_eq!(
            plan.validate_for(&validated, RenderStyle::Matcap, different_commands),
            Err(RenderContractError::CommandPlanMismatch)
        );

        let equivalent_but_distinct = ValidatedRenderScene::new(scene).unwrap();
        assert!(!plan.matches(
            &equivalent_but_distinct,
            RenderStyle::Matcap,
            options,
        ));
        assert_eq!(
            plan.validate_for(
                &equivalent_but_distinct,
                RenderStyle::Matcap,
                options,
            ),
            Err(RenderContractError::CommandPlanMismatch)
        );
        let clone = plan.clone();
        assert!(clone.scene().shares_snapshot_with(plan.scene()));
        assert!(std::ptr::eq(
            clone.commands().as_ptr(),
            plan.commands().as_ptr()
        ));
    }

    #[test]
    fn validated_scene_and_command_plan_fail_before_retention() {
        let mut invalid = scene();
        invalid.batches[0].triangle_index_count = 5;
        assert!(matches!(
            ValidatedRenderScene::new(invalid),
            Err(RenderContractError::InvalidBatchGeometry { batch_index: 0 })
        ));

        let validated = ValidatedRenderScene::new(scene()).unwrap();
        let mut options = RenderFrameOptions::default();
        options.focus_postprocess = Some(FocusPostprocessPacket {
            mode: FocusPostprocessMode::Spheroidal,
            blur_radius_pixels: 11,
            blur_strength: f32::NAN,
            focus_coordinate: 0.5,
            bandwidth: 0.1,
            normalize_range: false,
            stretch_range: [0.5, 0.5],
            gaussian_passes: 1,
            kawase_passes: 3,
            kawase_offset: 1.5,
        });
        assert_eq!(
            RenderCommandPlan::build(&validated, RenderStyle::Pbr, options).unwrap_err(),
            RenderContractError::InvalidFocusPostprocess
        );
    }

    #[test]
    fn render_frame_rejects_invalid_focus_postprocess_before_backend_dispatch() {
        let scene = scene();
        let result = RenderFrame::build(
            12,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::Pbr,
            view(),
            RenderFrameOptions {
                focus_postprocess: Some(FocusPostprocessPacket {
                    mode: FocusPostprocessMode::ConformalStretch,
                    blur_radius_pixels: 11,
                    blur_strength: f32::NAN,
                    focus_coordinate: 0.5,
                    bandwidth: 0.1,
                    normalize_range: false,
                    stretch_range: [0.5, 0.5],
                    gaussian_passes: 1,
                    kawase_passes: 3,
                    kawase_offset: 1.5,
                }),
                ..RenderFrameOptions::default()
            },
            &scene,
        );
        assert!(matches!(
            result,
            Err(RenderContractError::InvalidFocusPostprocess)
        ));
    }

    #[test]
    fn focus_postprocess_packet_serialization_preserves_evidence_semantics() {
        let packet = FocusPostprocessPacket {
            mode: FocusPostprocessMode::Hybrid,
            blur_radius_pixels: 23,
            blur_strength: 1.25,
            focus_coordinate: 0.6,
            bandwidth: 0.08,
            normalize_range: true,
            stretch_range: [0.15, 4.5],
            gaussian_passes: 1,
            kawase_passes: 3,
            kawase_offset: 1.75,
        };
        let encoded = serde_json::to_string(&packet).unwrap();
        let decoded = serde_json::from_str::<FocusPostprocessPacket>(&encoded).unwrap();
        assert_eq!(decoded, packet);
        decoded.validate().unwrap();
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
                focus_postprocess: None,
                highlight_face: Some(4),
                ..RenderFrameOptions::default()
            },
            &scene,
        )
        .unwrap();
        assert_eq!(frame.commands.len(), 9);
        assert!(matches!(
            frame.commands[4],
            RenderCommand::DrawPatches {
                pass: RenderPass::Matcap,
                ..
            }
        ));
        assert!(matches!(
            frame.commands[6],
            RenderCommand::DrawPatches {
                pass: RenderPass::Wire,
                ..
            }
        ));
        assert_eq!(
            frame.commands[8],
            RenderCommand::HighlightFace { face_index: 4 }
        );
        let mut expected = RenderSubmissionStats::default();
        expected.record_patch_draw(0, RenderPass::Matcap, RenderGeometry::Triangles, 6, 1);
        expected.record_patch_draw(1, RenderPass::Matcap, RenderGeometry::Triangles, 6, 0);
        expected.record_patch_draw(0, RenderPass::Wire, RenderGeometry::Lines, 6, 1);
        expected.record_patch_draw(1, RenderPass::Wire, RenderGeometry::Lines, 6, 0);
        assert_eq!(frame.expected_submission_stats(&scene).unwrap(), expected);
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
        frame.commands = frame.commands[..frame.commands.len() - 1].into();
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
    fn scene_validation_rejects_invalid_authored_materials() {
        let mut invalid = scene();
        let mut material = PbrMaterial::default();
        material.ior = f32::NAN;
        invalid.materials.push(material);
        assert_eq!(
            invalid.validate(),
            Err(RenderContractError::InvalidMaterial { material_index: 0 }),
        );
    }

    #[test]
    fn retained_layers_validate_logical_replacement_and_physical_dispatch() {
        let mut roots = batch(0, 0, PbrDrawClass::Opaque, true);
        roots.id.layer = RenderBatchLayer::RetainedRoot;
        roots.members.push(RenderBatchMember {
            face_index: 1,
            ..roots.members[0]
        });
        let mut overlay = batch(0, 0, PbrDrawClass::Opaque, true);
        overlay.id.layer = RenderBatchLayer::AdaptiveOverlay;
        let first = crate::screen_partition::ScreenPatchLeafId::ROOT
            .child(0)
            .unwrap();
        let second = crate::screen_partition::ScreenPatchLeafId::ROOT
            .child(1)
            .unwrap();
        overlay.members[0].leaf_id = first;
        overlay.members.push(RenderBatchMember {
            leaf_id: second,
            ..overlay.members[0]
        });
        let retained = RenderSceneSnapshot {
            revision: 8,
            materials: Vec::new(),
            suppressed_root_faces: vec![0],
            batches: vec![roots, overlay],
        };
        assert_eq!(retained.validate(), Ok(()));

        let frame = RenderFrame::build(
            13,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 9,
            },
            RenderStyle::Matcap,
            view(),
            RenderFrameOptions::default(),
            &retained,
        )
        .unwrap();
        assert_eq!(
            frame
                .commands
                .iter()
                .filter_map(|command| match command {
                    RenderCommand::DrawPatches { instance_count, .. } => {
                        Some(*instance_count)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![2, 2],
        );

        let mut missing_overlay = retained.clone();
        missing_overlay.batches.pop();
        assert_eq!(
            missing_overlay.validate(),
            Err(RenderContractError::MissingAdaptiveReplacement(0)),
        );

        let mut unmasked = retained.clone();
        unmasked.suppressed_root_faces.clear();
        assert_eq!(
            unmasked.validate(),
            Err(RenderContractError::UnmaskedAdaptiveReplacement(0)),
        );

        let mut mixed = retained;
        mixed.batches[0].id.layer = RenderBatchLayer::Complete;
        assert_eq!(
            mixed.validate(),
            Err(RenderContractError::MixedBatchLayers),
        );
    }

    #[test]
    fn resident_root_draw_domains_are_sparse_and_source_complete() {
        let mut low = batch(7, 0, PbrDrawClass::Opaque, true);
        low.id.layer = RenderBatchLayer::RetainedRoot;
        low.id.key.render_node_index = 2;
        low.id.key.lod = [1; 3];
        low.members[0].edge_lods = [1; 3];
        low.members[0].vertex_lods = [1; 3];

        let mut high = batch(7, 2, PbrDrawClass::Opaque, true);
        high.id.layer = RenderBatchLayer::RetainedRoot;
        high.id.key.render_node_index = 2;
        high.id.key.lod = [4; 3];
        high.members[0].edge_lods = [4; 3];
        high.members[0].vertex_lods = [4; 3];

        let mut sparse_material = batch(1_000_000, 1, PbrDrawClass::Blend, false);
        sparse_material.id.layer = RenderBatchLayer::RetainedRoot;
        sparse_material.id.key.render_node_index = 9;

        let scene = RenderSceneSnapshot {
            revision: 11,
            materials: Vec::new(),
            suppressed_root_faces: Vec::new(),
            batches: vec![low, high, sparse_material],
        };
        let extracted = ResidentRootDrawDomains::build(&scene, 3).unwrap();
        assert_eq!(extracted.domains.len(), 2);
        assert_eq!(extracted.face_domain_rows, [0, 1, 0]);
        assert_eq!(extracted.domains[0].material_index, 7);
        assert_eq!(extracted.domains[0].render_node_index, 2);
        assert_eq!(extracted.domains[1].material_index, 1_000_000);
        assert!(!extracted.domains[1].enabled);
    }

    #[test]
    fn resident_root_draw_domains_reject_conflicting_state_and_missing_faces() {
        let mut first = batch(3, 0, PbrDrawClass::Opaque, true);
        first.id.layer = RenderBatchLayer::RetainedRoot;
        let mut conflicting = batch(3, 1, PbrDrawClass::Blend, true);
        conflicting.id.layer = RenderBatchLayer::RetainedRoot;
        conflicting.id.key.lod = [4; 3];
        conflicting.members[0].edge_lods = [4; 3];
        conflicting.members[0].vertex_lods = [4; 3];
        let conflict_scene = RenderSceneSnapshot {
            revision: 12,
            materials: Vec::new(),
            suppressed_root_faces: Vec::new(),
            batches: vec![first.clone(), conflicting],
        };
        assert_eq!(
            ResidentRootDrawDomains::build(&conflict_scene, 2),
            Err(ResidentRootDrawDomainError::ConflictingDomainState {
                material_index: 3,
                render_node_index: 0,
            })
        );

        let missing_scene = RenderSceneSnapshot {
            revision: 13,
            materials: Vec::new(),
            suppressed_root_faces: Vec::new(),
            batches: vec![first],
        };
        assert_eq!(
            ResidentRootDrawDomains::build(&missing_scene, 2),
            Err(ResidentRootDrawDomainError::MissingRootFace(1))
        );
    }

    #[test]
    fn submission_stats_distinguish_work_zero_instances_and_invalid_draws() {
        let mut stats = RenderSubmissionStats::default();
        stats.record_indexed_draw(RenderGeometry::Triangles, 7, 3);
        stats.record_indexed_draw(RenderGeometry::Lines, 8, 2);
        stats.record_indexed_draw(RenderGeometry::Triangles, 12, 0);
        stats.record_invalid_draw();

        assert_eq!(stats.draw_calls, 4);
        assert_eq!(stats.zero_instance_draw_calls, 1);
        assert_eq!(stats.invalid_draw_calls, 1);
        assert_eq!(stats.submitted_instances, 5);
        assert_eq!(stats.triangles, 6);
        assert_eq!(stats.lines, 8);
        assert_ne!(stats.draw_sequence_hash, 0);

        let mut prefix = RenderSubmissionStats::default();
        prefix.record_indexed_draw(RenderGeometry::Triangles, 7, 3);
        prefix.record_indexed_draw(RenderGeometry::Lines, 8, 2);
        let mut suffix = RenderSubmissionStats::default();
        suffix.record_indexed_draw(RenderGeometry::Triangles, 12, 0);
        suffix.record_invalid_draw();
        prefix.merge(suffix);
        assert_eq!(prefix, stats);

        let mut reordered = RenderSubmissionStats::default();
        reordered.record_indexed_draw(RenderGeometry::Lines, 8, 2);
        reordered.record_indexed_draw(RenderGeometry::Triangles, 7, 3);
        reordered.record_indexed_draw(RenderGeometry::Triangles, 12, 0);
        reordered.record_invalid_draw();
        assert_ne!(reordered.draw_sequence_hash, stats.draw_sequence_hash);
        let reordered_mismatch = RenderSubmissionMismatch::between(stats, reordered);
        assert!(reordered_mismatch.draw_sequence);
        assert!(!reordered_mismatch.draw_calls);
        assert!(!reordered_mismatch.submitted_instances);
        assert!(!reordered_mismatch.triangles);
        assert!(!reordered_mismatch.lines);

        let serialized = serde_json::to_value(stats).unwrap();
        assert_eq!(
            serialized["drawSequenceHash"],
            format!("{:016x}", stats.draw_sequence_hash),
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

    #[test]
    fn parity_observer_accepts_only_execution_from_its_retained_scene_epoch() {
        let scene = ValidatedRenderScene::new(scene()).unwrap();
        let plan = RenderCommandPlan::build(
            &scene,
            RenderStyle::Matcap,
            RenderFrameOptions::default(),
        )
        .unwrap();
        let expected = plan.execution().submission_stats();
        let mut observer = RenderParityObserver::default();
        observer.set_enabled(true);
        observer.replace_validated_scene(scene.clone()).unwrap();
        assert!(observer
            .observe_execution(22, plan.execution(), expected)
            .unwrap()
            .is_match());

        let distinct = ValidatedRenderScene::new(scene.snapshot().clone()).unwrap();
        let distinct_plan = RenderCommandPlan::build(
            &distinct,
            RenderStyle::Matcap,
            RenderFrameOptions::default(),
        )
        .unwrap();
        assert_eq!(
            observer.observe_execution(23, distinct_plan.execution(), expected),
            Err(RenderContractError::ExecutionSceneMismatch)
        );
        assert_eq!(observer.diagnostics().frames_observed, 1);
    }

    #[test]
    fn render_style_has_a_stable_backend_neutral_wire_spelling() {
        let styles = [
            (RenderStyle::Pbr, "pbr"),
            (RenderStyle::Matcap, "matcap"),
            (RenderStyle::Wire, "wire"),
            (RenderStyle::Normals, "normals"),
            (RenderStyle::MatcapWire, "matcap_wire"),
            (RenderStyle::Lod, "lod"),
            (RenderStyle::Stretch, "stretch"),
        ];
        for (style, name) in styles {
            let encoded = format!("\"{name}\"");
            assert_eq!(style.wire_name(), name);
            assert_eq!(RenderStyle::from_wire_name(name), Some(style));
            assert_eq!(serde_json::to_string(&style).unwrap(), encoded);
            assert_eq!(serde_json::from_str::<RenderStyle>(&encoded).unwrap(), style);
        }
        assert_eq!(RenderStyle::from_wire_name("both"), None);
        assert_eq!(RenderStyle::from_wire_name("browser_magic"), None);
    }
}
