//! Staged WebGPU execution for Quilting's shared render contract.
//!
//! This crate remains outside the current WebGL2 authority path while backend
//! parity is established. It consumes backend-neutral `RenderSceneSnapshot`
//! and `RenderFrame` values, retains device pipelines/buffers, and can own an
//! explicitly supplied presentation surface. Semantic scene, FRP, replicated
//! state, browser layout, and canvas selection remain application concerns.

mod adaptive_overlay;
mod focus_postprocess;
mod functional_pipeline;
mod pbr_environment;
mod pbr_resources;
mod pipeline_lowering;
mod picking;
mod portable_texture_atlas;
mod prepared_patch_pipeline;
mod presentation;
mod resident_roots;

pub use adaptive_overlay::{
    AdaptiveOverlayFrameEncoding, AdaptiveOverlayScene, FocusResidentAdaptiveFrameEncoding,
    ResidentAdaptiveFrameEncoding,
};
pub use focus_postprocess::{
    focus_postprocess_pipeline_descriptors, FocusPostprocessEncoding,
    FocusPostprocessMemoDiagnostics, FocusPostprocessPipelines, FocusPostprocessTarget,
    FocusRawFieldImage, StagedFocusRawFieldReadback,
};
pub use pbr_environment::{PbrEnvironmentBindings, PbrEnvironmentMap};
pub use pbr_resources::{
    PbrMaterialTextureBindings, PbrMaterialTextureResidency, PbrTextureTable, PbrTextureTableUpdate,
};
pub use picking::{
    PatchPickEncoding, PatchPickPipeline, PatchPickRequest, PatchPickSample, PatchPickTarget,
    ResidentRootPickPipeline, StagedPatchPickReadback,
};
pub use portable_texture_atlas::{
    PortableTextureAtlasLimits, PortableTextureAtlasPlacement, PortableTextureAtlasPlan,
    PortableTextureAtlasPlanError,
};
pub use prepared_patch_pipeline::{
    prepared_patch_pipeline_descriptors, PreparedPatchPipelineDescriptorError,
};
pub use presentation::{
    PatchPresentationSurface, PresentationSkipReason, SurfacePresentation,
    SurfacePresentationDiagnostics,
};
pub use resident_roots::{
    resident_root_pipeline_descriptors, resident_root_render_domains,
    supports_resident_root_render_scene, supports_resident_root_render_style,
    FocusResidentRootFrameEncoding, ResidentGeometryBucketOutput, ResidentGeometryBucketScene,
    ResidentRootDrawDomainOutput, ResidentRootDrawDomainScene, ResidentRootFrameEncoding,
    ResidentRootPreparationScene, ResidentRootRenderBindings, ResidentRootRenderPipeline,
    ResidentRootTopologyScene,
};

use futures_channel::oneshot;
use quilting_core::batch::{
    FaceLodGrading, RenderBatchId, RenderBatchKey, RenderBatchLayer, RenderBatchMember,
};
use quilting_core::instance_layout::InstanceWriter;
use quilting_core::material::{
    pbr_material_for_index, EnvironmentMapAsset, EnvironmentMapDescriptor, PbrAlphaMode,
    PbrMaterial, Rgba8TextureAsset, TextureAssetDescriptor, TextureWrapMode,
};
use quilting_core::render::{
    render_draw_passes, FocusFieldPacket, PbrDrawClass, RenderBatchSnapshot,
    RenderCommandPlan, RenderEntityTransform, RenderFrame, RenderFrameOptions, RenderGeometry,
    RenderPass, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle, RenderSubmissionStats,
    RenderView, ResolvedRenderCommand, ResidentRootDrawDomain, ResidentRootDrawDomains,
    ValidatedRenderScene,
};
use quilting_core::render_evidence::{
    render_image_signature, RenderImageChannelOrder, RenderImageOrigin, RenderImageSignature,
    Rgba8ImageView,
};
use quilting_core::render_memo::{DeviceMemo, DeviceMemoDiagnostics};
use quilting_core::render_pipeline::{
    self as functional, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderStage, ShaderTarget,
};
use quilting_core::screen_partition::ScreenPatchLeafId;
use quilting_renderer::compute::{
    pack_lod_classification, pack_wgsl_adaptive_overlay_scene_words, pack_wgsl_lod_atlas_words,
    pack_wgsl_lod_dispatch_words, pack_wgsl_lod_model_words,
    pack_wgsl_lod_subject_words_with_layout, pack_wgsl_patch_preparation_scene_words,
    pack_wgsl_resident_root_preparation_scene_words, pack_wgsl_root_eligibility_bits,
    pack_wgsl_source_visibility_words, pack_wgsl_visibility_compaction_scene_words,
    prepare_lod_atlas_lookup, prepare_lod_model, reconcile_and_pack_wgsl_lod_pass2,
    reconcile_and_pack_wgsl_resident_lods, wgsl_resident_geometry_bucket_oracle_words_with_domains,
    wgsl_resident_root_topology_oracle_words, LodAtlasLookup, LodDispatchState, LodModelData,
    PreparedLodModel, WgslAdaptiveOverlayPreparationSceneWords, WgslLodDispatchMetrics,
    WgslLodSubjectLayout, WgslPatchPreparationSceneWords, WgslResidentRootPreparationSceneWords,
    WgslVisibilityCompactionSceneWords,
};
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const LOD_WORKGROUP_SIZE: u32 = 64;
const RESIDENT_LOD_RECONCILIATION_PASSES: usize = 10;
const DISPATCH_UNIFORM_BYTES: u64 = 272;
const SUBJECT_RECORD_BYTES: u64 = 160;
const JOINT_MATRIX_BYTES: u64 = 64;
const PASS1_RECORD_BYTES: u64 = 16;
const PACKED_RECORD_BYTES: u64 = 4;
const VISIBILITY_UNIFORM_BYTES: u64 = 16;
const FACE_VISIBILITY_UNIFORM_BYTES: u64 = 16;
const VISIBILITY_BATCH_RECORD_BYTES: u64 = 16;
const VISIBILITY_RANGE_RECORD_BYTES: u64 = 20;
const INDEXED_INDIRECT_RECORD_BYTES: u64 = 20;
const RESIDENT_ATLAS_DRAW_RECORD_BYTES: u64 = 16;
const RESIDENT_BUCKET_RANGE_RECORD_BYTES: u64 = 20;
const MAX_RESIDENT_GEOMETRY_BUCKETS: u32 = 510;
const DRAW_BATCH_INDEX_BYTES: u64 = 16;
const PATCH_PREPARE_UNIFORM_BYTES: u64 = 16;
const PATCH_TOPOLOGY_RECORD_BYTES: u64 = 48;
const PREPARED_PATCH_RECORD_WORDS: usize = 52;
const PREPARED_PATCH_RECORD_BYTES: u64 = 208;
const PATCH_SUBJECT_RECORD_BYTES: u64 = 128;
const PATCH_SUBJECT_RECORD_WORDS: usize = 32;
const PATCH_RENDER_GLOBAL_WORDS: usize = 44;
const PATCH_RENDER_GLOBAL_BYTES: u64 = 176;
const PATCH_RENDER_DOMAIN_WORDS: usize = 20;
const PATCH_RENDER_DOMAIN_BYTES: u64 = 80;
const PATCH_PBR_MATERIAL_WORDS: usize = 40;
const PATCH_PBR_MATERIAL_BYTES: u64 = 160;

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

fn translation_matrix(x: f32, y: f32, z: f32) -> [f32; 16] {
    let mut matrix = identity_matrix();
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    matrix
}

fn patch_preparation_conformance_words() -> (
    WgslPatchPreparationSceneWords,
    Vec<[u32; PREPARED_PATCH_RECORD_WORDS]>,
) {
    let bits = |value: f32| value.to_bits();
    let mut source = [0u32; PREPARED_PATCH_RECORD_WORDS];
    for (corner, position) in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        .into_iter()
        .enumerate()
    {
        let offset = corner * 4;
        source[offset] = bits(corner as f32);
        for (word, value) in source[offset + 1..offset + 4].iter_mut().zip(position) {
            *word = bits(value);
        }
        source[12 + offset] = bits(1.0);
        source[40 + offset + 2] = bits(1.0);
    }
    source[43] = bits(7.0);
    for (word, value) in source[32..38]
        .iter_mut()
        .zip([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    {
        *word = bits(value);
    }

    let topology_record = |lod: [f32; 4], face: [f32; 4], leaf: [f32; 2]| {
        let mut record = [0u32; 12];
        for (word, value) in record[..4].iter_mut().zip(lod) {
            *word = bits(value);
        }
        for (word, value) in record[4..8].iter_mut().zip(face) {
            *word = bits(value);
        }
        for (word, value) in record[8..10].iter_mut().zip(leaf) {
            *word = bits(value);
        }
        record
    };
    let topology = vec![
        topology_record([8.0, 4.0, 2.0, 5.0], [0.0, 4.0, 8.0, 16.0], [0.0, 0.0]),
        topology_record([2.0, 4.0, 1.0, 3.0], [0.0, 4.0, 8.0, 8.0], [1.0, 3.0]),
    ];
    let mut subject = [0u32; 32];
    for (word, value) in subject[..16].iter_mut().zip(identity_matrix()) {
        *word = bits(value);
    }
    let mut normal_model = identity_matrix();
    normal_model[10] = 2.0;
    for (word, value) in subject[16..].iter_mut().zip(normal_model) {
        *word = bits(value);
    }

    let mut root = source;
    for (corner, position) in [[1.25, 0.0, 2.0], [2.25, 0.0, 2.0], [1.25, 1.0, 2.0]]
        .into_iter()
        .enumerate()
    {
        let offset = corner * 4;
        for (word, value) in root[offset + 1..offset + 4].iter_mut().zip(position) {
            *word = bits(value);
        }
    }
    for (word, value) in root[24..28].iter_mut().zip([8.0, 4.0, 2.0, 5.0]) {
        *word = bits(value);
    }
    for (word, value) in root[28..32].iter_mut().zip([4.0, 8.0, 16.0, 0.0]) {
        *word = bits(value);
    }
    root[38] = bits(0.0);
    root[39] = bits(1.0);
    for word in [42, 46, 50] {
        root[word] = bits(2.0);
    }

    let mut child = source;
    for (corner, tagged_position) in [
        [-2.0, 1.75, 0.0, 2.0],
        [3.0, 1.75, 0.5, 2.0],
        [0.0, 1.25, 0.5, 2.0],
    ]
    .into_iter()
    .enumerate()
    {
        let offset = corner * 4;
        for (word, value) in child[offset..offset + 4].iter_mut().zip(tagged_position) {
            *word = bits(value);
        }
    }
    for (word, value) in child[24..28].iter_mut().zip([2.0, 4.0, 1.0, 3.0]) {
        *word = bits(value);
    }
    for (word, value) in child[28..32].iter_mut().zip([4.0, 8.0, 8.0, 0.0]) {
        *word = bits(value);
    }
    for (word, value) in child[32..38].iter_mut().zip([0.5, 0.0, 0.5, 0.5, 0.0, 0.5]) {
        *word = bits(value);
    }
    child[38] = bits(0.0);
    child[39] = bits(1.0);
    for word in [42, 46, 50] {
        child[word] = bits(2.0);
    }

    (
        WgslPatchPreparationSceneWords {
            uniform: [2, 3, 0, 1],
            topology,
            source_faces: vec![source],
            subjects: vec![subject],
        },
        vec![root, child],
    )
}

#[derive(Debug)]
pub enum LodWebGpuError {
    Shader(String),
    Payload(String),
    Conformance(String),
    Mapping(String),
    Poll(String),
}

impl std::fmt::Display for LodWebGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shader(message) => write!(formatter, "WebGPU LOD shader: {message}"),
            Self::Payload(message) => write!(formatter, "WebGPU LOD payload: {message}"),
            Self::Conformance(message) => write!(formatter, "WebGPU LOD conformance: {message}"),
            Self::Mapping(message) => write!(formatter, "WebGPU LOD mapping: {message}"),
            Self::Poll(message) => write!(formatter, "WebGPU LOD poll: {message}"),
        }
    }
}

impl std::error::Error for LodWebGpuError {}

/// Dynamic pose values uploaded for one classifier dispatch.
#[derive(Clone, Copy, Debug, Default)]
pub struct LodPose<'a> {
    /// Column-major 4x4 matrices, exactly `metrics.num_joints * 16` floats.
    pub joint_matrices: &'a [f32],
    /// Exactly one weight per immutable morph target.
    pub morph_weights: &'a [f32],
}

/// Whether a retained preparation family must publish its dynamic pose inputs
/// before encoding. `Reuse` is valid only while the caller retains the exact
/// model and preparation epoch that received the previous successful publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoseUploadPolicy {
    /// Publish shared dynamic pose buffers and preparation-local uniforms.
    Publish,
    /// Reuse the shared dynamic pose while initializing local preparation
    /// uniforms for a newly retained scene family.
    PublishPreparation,
    /// Reuse both shared and preparation-local pose state.
    Reuse,
}

impl PoseUploadPolicy {
    pub const fn should_publish_dynamic(self) -> bool {
        matches!(self, Self::Publish)
    }

    pub const fn should_publish_preparation(self) -> bool {
        !matches!(self, Self::Reuse)
    }
}

/// Bounded evidence returned after the shared device conformance matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodDeviceConformance {
    pub full_pipeline_words: usize,
    pub resident_lod_words: usize,
    pub resident_visibility_words: usize,
    pub resident_bucket_words: usize,
    pub resident_root_topology_words: usize,
    pub resident_root_prepared_words: usize,
    pub resident_root_domain_words: usize,
    pub resident_adaptive_rendered_pixels: usize,
    pub resident_adaptive_image: RenderImageSignature,
    pub resident_root_indirect_draws: usize,
    pub adaptive_overlay_patches: usize,
    pub adaptive_overlay_indirect_draws: usize,
    pub coherence_words: usize,
    pub prepared_patch_words: usize,
    pub rendered_patch_pixels: usize,
    pub shared_frame_draws: usize,
    pub compacted_source_words: usize,
    pub compacted_range_words: usize,
    pub indirect_argument_words: usize,
    pub indirect_draws: usize,
}

/// Stable diagnostics for an adapter selected without a presentation surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebGpuAdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

/// Diagnostic copy of the exact same-device visibility outputs. The retained
/// GPU buffers remain suitable for direct storage/indirect consumption; this
/// owned projection exists only for conformance gates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibilityCompactionOutput {
    pub compacted_source_instances: Vec<u32>,
    pub compacted_ranges: Vec<[u32; 5]>,
    pub triangle_indirect_arguments: Vec<[u32; 5]>,
    pub line_indirect_arguments: Vec<[u32; 5]>,
}

/// Frame-global dynamic values consumed once by every prepared-surface domain.
/// Matrices use the same column-major convention as WebGL2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchRenderGlobal {
    pub mvp: [f32; 16],
    pub mv: [f32; 16],
    pub use_qb: bool,
    pub matcap_style: quilting_core::render::MatcapStyle,
    pub selected_node: Option<u32>,
    pub selected_face: Option<u32>,
    pub camera_position: [f32; 3],
    pub focus: FocusFieldPacket,
}

impl PatchRenderGlobal {
    pub fn from_render_frame(frame: &RenderFrame, use_qb: bool) -> Self {
        Self {
            mvp: frame.view.mvp,
            mv: frame.view.model_view,
            use_qb,
            matcap_style: frame.options.matcap_style,
            selected_node: frame
                .view
                .selected_node
                .and_then(|node| node.try_into().ok()),
            selected_face: frame.options.highlight_face,
            camera_position: frame.view.camera_position,
            focus: frame.view.focus,
        }
    }

    fn to_words(self) -> Result<[u32; PATCH_RENDER_GLOBAL_WORDS], LodWebGpuError> {
        if self
            .mvp
            .into_iter()
            .chain(self.mv)
            .chain(self.camera_position)
            .chain(self.focus.sphere)
            .any(|value| !value.is_finite())
            || self.focus.sphere[3] <= 0.0
        {
            return Err(LodWebGpuError::Payload(
                "patch render frame contains non-finite values".to_string(),
            ));
        }
        let mut words = [0u32; PATCH_RENDER_GLOBAL_WORDS];
        for (word, value) in words[..16].iter_mut().zip(self.mvp) {
            *word = value.to_bits();
        }
        for (word, value) in words[16..32].iter_mut().zip(self.mv) {
            *word = value.to_bits();
        }
        words[32] = u32::from(self.use_qb);
        words[33] = self.matcap_style.as_u32();
        words[34] = self.selected_node.unwrap_or(u32::MAX);
        words[35] = self.selected_face.unwrap_or(u32::MAX);
        for (word, value) in words[36..39].iter_mut().zip(self.camera_position) {
            *word = value.to_bits();
        }
        words[39] = f32::from(u8::from(self.focus.enabled)).to_bits();
        for (word, value) in words[40..44].iter_mut().zip(self.focus.sphere) {
            *word = value.to_bits();
        }
        Ok(words)
    }
}

/// Batch/domain-local values. Affine model/normal state was already consumed
/// by patch preparation; only the conformal map and material row vary here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchRenderDomain {
    pub mobius: [f32; 16],
    pub material_slot: u32,
}

impl PatchRenderDomain {
    pub fn from_transform(transform: RenderEntityTransform, material_slot: u32) -> Self {
        Self {
            mobius: transform.mobius,
            material_slot,
        }
    }

    fn to_words(self) -> Result<[u32; PATCH_RENDER_DOMAIN_WORDS], LodWebGpuError> {
        if self.mobius.into_iter().any(|value| !value.is_finite()) {
            return Err(LodWebGpuError::Payload(
                "patch render domain contains non-finite values".to_string(),
            ));
        }
        let mut words = [0u32; PATCH_RENDER_DOMAIN_WORDS];
        for (word, value) in words[..16].iter_mut().zip(self.mobius) {
            *word = value.to_bits();
        }
        words[16] = self.material_slot;
        Ok(words)
    }
}

/// Lossless semantic pair used by compatibility fixtures and callers that
/// prepare one complete retained domain at a time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchRenderFrame {
    pub global: PatchRenderGlobal,
    pub domain: PatchRenderDomain,
}

impl PatchRenderFrame {
    pub fn from_render_frame(
        frame: &RenderFrame,
        batch: &RenderBatchSnapshot,
        use_qb: bool,
    ) -> Self {
        Self::from_render_frame_with_material_slot(frame, batch, use_qb, 0)
    }

    pub fn from_render_frame_with_material_slot(
        frame: &RenderFrame,
        batch: &RenderBatchSnapshot,
        use_qb: bool,
        material_slot: u32,
    ) -> Self {
        Self {
            global: PatchRenderGlobal::from_render_frame(frame, use_qb),
            domain: PatchRenderDomain::from_transform(batch.transform, material_slot),
        }
    }

    pub fn from_transform(
        frame: &RenderFrame,
        transform: RenderEntityTransform,
        use_qb: bool,
    ) -> Self {
        Self {
            global: PatchRenderGlobal::from_render_frame(frame, use_qb),
            domain: PatchRenderDomain::from_transform(transform, 0),
        }
    }
}

fn patch_pbr_material_words(material: &PbrMaterial) -> [u32; PATCH_PBR_MATERIAL_WORDS] {
    let mut words = [0u32; PATCH_PBR_MATERIAL_WORDS];
    for (word, value) in words[..4].iter_mut().zip(material.base_color) {
        *word = value.to_bits();
    }
    for (word, value) in words[4..7].iter_mut().zip(material.emissive_factor) {
        *word = value.to_bits();
    }
    words[7] = material.metallic.to_bits();
    for (word, value) in words[8..12].iter_mut().zip([
        material.roughness,
        material.alpha_cutoff,
        material.alpha_mode.as_f32(),
        material.ior,
    ]) {
        *word = value.to_bits();
    }
    words[12] = u32::from(material.unlit);
    words[13] = u32::from(material.double_sided);
    words[14] = u32::from(material.has_specular);
    words[15] = u32::from(material.has_sheen)
        | (pbr_resources::pbr_texture_reference_mask(material.textures) << 1);
    for (word, value) in words[16..19].iter_mut().zip(material.specular_color) {
        *word = value.to_bits();
    }
    words[19] = material.normal_scale.to_bits();
    for (word, value) in words[20..23].iter_mut().zip(material.sheen_color) {
        *word = value.to_bits();
    }
    words[23] = material.sheen_roughness.to_bits();
    for (word, value) in words[24..28].iter_mut().zip(
        material
            .normal_uv_scale
            .into_iter()
            .chain(material.normal_uv_offset),
    ) {
        *word = value.to_bits();
    }
    for (word, value) in words[28..32].iter_mut().zip([
        material.normal_uv_rotation,
        material.occlusion_strength,
        material.base_uv_scale[0],
        material.base_uv_scale[1],
    ]) {
        *word = value.to_bits();
    }
    for (word, value) in words[32..36].iter_mut().zip([
        material.base_uv_rotation,
        material.transmission_factor,
        material.thickness_factor,
        material.attenuation_distance.unwrap_or(0.0),
    ]) {
        *word = value.to_bits();
    }
    for (word, value) in words[36..39].iter_mut().zip(material.attenuation_color) {
        *word = value.to_bits();
    }
    words
}

fn patch_pbr_material_table_words(materials: &[PbrMaterial]) -> Result<Vec<u32>, LodWebGpuError> {
    let default_material = PbrMaterial::default();
    let material_count = materials.len().max(1);
    let mut words = Vec::with_capacity(
        material_count
            .checked_mul(PATCH_PBR_MATERIAL_WORDS)
            .ok_or_else(|| LodWebGpuError::Payload("PBR material table is too large".into()))?,
    );
    if materials.is_empty() {
        words.extend(patch_pbr_material_words(&default_material));
    } else {
        for material in materials {
            material
                .validate()
                .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
            words.extend(patch_pbr_material_words(material));
        }
    }
    Ok(words)
}

fn patch_pbr_material_slot(
    materials: &[PbrMaterial],
    requested: usize,
) -> Result<u32, LodWebGpuError> {
    let default_material = PbrMaterial::default();
    let (material_slot, _) = pbr_material_for_index(materials, &default_material, requested);
    if material_slot == usize::MAX {
        Ok(0)
    } else {
        u32::try_from(material_slot).map_err(|_| {
            LodWebGpuError::Payload("resolved PBR material slot exceeds u32".to_string())
        })
    }
}

#[cfg(test)]
mod patch_pbr_material_tests {
    use super::*;
    use quilting_core::material::{PbrAlphaMode, PbrTextureReferences};

    #[test]
    fn pose_upload_policy_separates_dynamic_and_preparation_publication() {
        assert!(PoseUploadPolicy::Publish.should_publish_dynamic());
        assert!(PoseUploadPolicy::Publish.should_publish_preparation());
        assert!(!PoseUploadPolicy::PublishPreparation.should_publish_dynamic());
        assert!(PoseUploadPolicy::PublishPreparation.should_publish_preparation());
        assert!(!PoseUploadPolicy::Reuse.should_publish_dynamic());
        assert!(!PoseUploadPolicy::Reuse.should_publish_preparation());
    }

    #[test]
    fn retained_frame_table_publishes_once_and_reuses_exact_words() {
        let mut table = RetainedFrameTable::new(
            PATCH_RENDER_DOMAIN_WORDS,
            PATCH_RENDER_DOMAIN_WORDS,
            PATCH_RENDER_DOMAIN_BYTES,
        );
        let mut row = [0u32; PATCH_RENDER_DOMAIN_WORDS];

        let mut changed = table.begin_update();
        changed |= table.replace_row(0, &row);
        assert_eq!(
            table.commit(changed),
            FrameTablePublication::Upload {
                bytes: PATCH_RENDER_DOMAIN_BYTES,
            },
        );

        let mut changed = table.begin_update();
        changed |= table.replace_row(0, &row);
        assert_eq!(table.commit(changed), FrameTablePublication::Reuse);

        row[17] = 9;
        let mut changed = table.begin_update();
        changed |= table.replace_row(0, &row);
        assert_eq!(
            table.commit(changed),
            FrameTablePublication::Upload {
                bytes: PATCH_RENDER_DOMAIN_BYTES,
            },
        );

        table.invalidate();
        let mut changed = table.begin_update();
        changed |= table.replace_row(0, &row);
        assert_eq!(
            table.commit(changed),
            FrameTablePublication::Upload {
                bytes: PATCH_RENDER_DOMAIN_BYTES,
            },
        );
    }

    #[test]
    fn retained_lod_state_memos_global_and_subject_words_independently() {
        let mut state = RetainedLodDispatchState::new(2);
        assert_eq!(
            state.commit_uniform([0; 68]),
            FrameTablePublication::Upload {
                bytes: DISPATCH_UNIFORM_BYTES,
            },
        );
        assert_eq!(
            state.commit_subject_scratch(),
            FrameTablePublication::Upload {
                bytes: 2 * SUBJECT_RECORD_BYTES,
            },
        );
        assert_eq!(state.commit_uniform([0; 68]), FrameTablePublication::Reuse);
        assert_eq!(state.commit_subject_scratch(), FrameTablePublication::Reuse);

        let mut changed_uniform = [0; 68];
        changed_uniform[40] = 1.0f32.to_bits();
        state.subject_scratch[1][36] = 2.0f32.to_bits();
        assert_eq!(
            state.commit_uniform(changed_uniform),
            FrameTablePublication::Upload {
                bytes: DISPATCH_UNIFORM_BYTES,
            },
        );
        assert_eq!(
            state.commit_subject_scratch(),
            FrameTablePublication::Upload {
                bytes: 2 * SUBJECT_RECORD_BYTES,
            },
        );
        assert_eq!(state.subject_words[1][36], 2.0f32.to_bits());
    }

    #[test]
    fn patch_frame_keeps_source_face_selection_separate_from_node_and_material() {
        let mut mobius = identity_mobius();
        mobius[5] = 0.625;
        let frame = PatchRenderFrame {
            global: PatchRenderGlobal {
                mvp: identity_matrix(),
                mv: translation_matrix(1.0, 2.0, 3.0),
                use_qb: true,
                matcap_style: quilting_core::render::MatcapStyle::GoldenSoft,
                selected_node: Some(23),
                selected_face: Some(29),
                camera_position: [4.0, 5.0, 6.0],
                focus: FocusFieldPacket {
                    sphere: [7.0, 8.0, 9.0, 10.0],
                    enabled: true,
                },
            },
            domain: PatchRenderDomain {
                mobius,
                material_slot: 17,
            },
        };
        let global = frame.global.to_words().unwrap();
        let domain = frame.domain.to_words().unwrap();
        assert_eq!(global.len(), PATCH_RENDER_GLOBAL_WORDS);
        assert_eq!(domain.len(), PATCH_RENDER_DOMAIN_WORDS);
        assert_eq!(global[32..36], [1, 2, 23, 29]);
        assert_eq!(global[36..39], [4.0, 5.0, 6.0].map(f32::to_bits));
        assert_eq!(global[39], 1.0f32.to_bits());
        assert_eq!(global[40..44], [7.0, 8.0, 9.0, 10.0].map(f32::to_bits));
        assert_eq!(domain[..16], mobius.map(f32::to_bits));
        assert_eq!(domain[16..20], [17, 0, 0, 0]);
        assert_eq!(PATCH_RENDER_GLOBAL_BYTES + PATCH_RENDER_DOMAIN_BYTES, 256);
    }

    #[test]
    fn authored_material_words_match_the_shader_record() {
        let mut material = PbrMaterial::default();
        material.base_color = [0.1, 0.2, 0.3, 0.4];
        material.emissive_factor = [0.5, 0.6, 0.7];
        material.metallic = 0.8;
        material.roughness = 0.9;
        material.alpha_cutoff = 0.25;
        material.alpha_mode = PbrAlphaMode::Mask;
        material.ior = 1.33;
        material.unlit = true;
        material.double_sided = true;
        material.has_specular = true;
        material.has_sheen = true;
        material.specular_color = [0.11, 0.22, 0.33];
        material.sheen_color = [0.44, 0.55, 0.66];
        material.sheen_roughness = 0.77;
        material.normal_scale = 0.45;
        material.occlusion_strength = 0.67;
        material.normal_uv_scale = [1.2, 1.3];
        material.normal_uv_offset = [0.14, 0.15];
        material.normal_uv_rotation = 0.16;
        material.base_uv_scale = [1.7, 1.8];
        material.base_uv_rotation = 0.19;
        material.transmission_factor = 0.21;
        material.thickness_factor = 0.22;
        material.attenuation_distance = Some(2.3);
        material.attenuation_color = [0.24, 0.25, 0.26];
        material.textures = PbrTextureReferences {
            base_color: Some(3),
            normal: Some(4),
            occlusion: Some(5),
            ..Default::default()
        };
        let words = patch_pbr_material_words(&material);
        assert_eq!(
            words.len() * std::mem::size_of::<u32>(),
            PATCH_PBR_MATERIAL_BYTES as usize,
        );
        assert_eq!(words[0..4], material.base_color.map(f32::to_bits));
        assert_eq!(words[4..7], material.emissive_factor.map(f32::to_bits));
        assert_eq!(words[7], material.metallic.to_bits());
        assert_eq!(words[8], material.roughness.to_bits());
        assert_eq!(words[9], material.alpha_cutoff.to_bits());
        assert_eq!(words[10], 1.0f32.to_bits());
        assert_eq!(words[11], material.ior.to_bits());
        assert_eq!(words[12..15], [1, 1, 1]);
        assert_eq!(words[15], 1 | (0b01_0101 << 1));
        assert_eq!(words[16..19], material.specular_color.map(f32::to_bits),);
        assert_eq!(words[19], material.normal_scale.to_bits());
        assert_eq!(words[20..23], material.sheen_color.map(f32::to_bits));
        assert_eq!(words[23], material.sheen_roughness.to_bits());
        assert_eq!(words[24..26], material.normal_uv_scale.map(f32::to_bits));
        assert_eq!(words[26..28], material.normal_uv_offset.map(f32::to_bits));
        assert_eq!(words[28], material.normal_uv_rotation.to_bits());
        assert_eq!(words[29], material.occlusion_strength.to_bits());
        assert_eq!(words[30..32], material.base_uv_scale.map(f32::to_bits));
        assert_eq!(words[32], material.base_uv_rotation.to_bits());
        assert_eq!(words[33], material.transmission_factor.to_bits());
        assert_eq!(words[34], material.thickness_factor.to_bits());
        assert_eq!(words[35], 2.3f32.to_bits());
        assert_eq!(words[36..39], material.attenuation_color.map(f32::to_bits));
    }

    #[test]
    fn empty_material_table_uploads_one_default_record() {
        let words = patch_pbr_material_table_words(&[]).unwrap();
        assert_eq!(words.len(), PATCH_PBR_MATERIAL_WORDS);
        assert_eq!(words, patch_pbr_material_words(&PbrMaterial::default()));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchWinding {
    CounterClockwise,
    Clockwise,
}

/// Canonical atlas buffers selected for one retained render batch. Ownership
/// stays with the backend atlas cache; a frame merely borrows the exact entry
/// named by `RenderBatchId::key`.
pub struct PatchAtlasDraw<'a> {
    pub barycentric_buffer: &'a wgpu::Buffer,
    pub index_buffer: &'a wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    /// First index element in the packed global index buffer. The bound slice
    /// starts here so the portable indirect record can retain `first_index=0`.
    pub first_index: u32,
    pub index_count: u32,
}

/// Element ranges for one canonical entry in Hyperscope's packed global atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedPatchAtlasEntry {
    pub triangle_first_index: u32,
    pub triangle_index_count: u32,
    pub line_first_index: u32,
    pub line_index_count: u32,
}

/// WebGPU ownership of the exact packed atlas already consumed by WebGL2.
/// One global barycentric buffer and two global index buffers serve every
/// canonical key; lightweight range views are rebuilt without GPU allocation.
pub struct PackedPatchAtlas {
    barycentric_buffer: wgpu::Buffer,
    triangle_index_buffer: wgpu::Buffer,
    line_index_buffer: wgpu::Buffer,
    entries: BTreeMap<[u32; 3], PackedPatchAtlasEntry>,
    keys: Vec<[u32; 3]>,
    vertex_count: u32,
    triangle_index_count: u32,
    line_index_count: u32,
}

/// Attachments for the current WebGPU patch pass. Clearing is explicit so an
/// application can compose Quilting into an existing render graph without the
/// backend inventing frame ownership.
pub struct PatchRenderTarget<'a> {
    pub color_view: &'a wgpu::TextureView,
    pub resolve_target: Option<&'a wgpu::TextureView>,
    pub depth_stencil_view: Option<&'a wgpu::TextureView>,
    pub clear_color: Option<wgpu::Color>,
    pub clear_depth: Option<f32>,
}

#[derive(Clone, Copy)]
struct FocusFrameTarget<'a> {
    color_view: &'a wgpu::TextureView,
    depth_stencil_view: &'a wgpu::TextureView,
    size: [u32; 2],
}

/// Backend-owned offscreen attachments for live shadow rendering. This keeps
/// texture formats and raw `wgpu` handles out of the WASM/application layer;
/// promotion can later replace this target with a surface view while reusing
/// the same pipeline and frame executor.
pub struct OffscreenPatchRenderTarget {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    size: [u32; 2],
}

/// One explicit, diagnostic-only copy of a completed offscreen frame. Staging
/// is synchronous and ordered on the render queue; mapping remains async and
/// owns all resources needed after the live backend borrow is released.
pub struct StagedOffscreenImageReadback {
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    buffer: wgpu::Buffer,
    size: [u32; 2],
    bytes_per_row: usize,
    byte_len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffscreenRgba8Image {
    pub size: [u32; 2],
    pub bytes_per_row: usize,
    pub bytes: Vec<u8>,
}

impl OffscreenRgba8Image {
    pub fn view(&self) -> Result<Rgba8ImageView<'_>, LodWebGpuError> {
        Rgba8ImageView::new(
            self.size,
            self.bytes_per_row,
            RenderImageOrigin::TopLeft,
            RenderImageChannelOrder::Rgba,
            &self.bytes,
        )
        .map_err(|error| LodWebGpuError::Payload(error.to_string()))
    }
}

impl StagedOffscreenImageReadback {
    pub async fn read(self) -> Result<OffscreenRgba8Image, LodWebGpuError> {
        let slice = self.buffer.slice(..self.byte_len);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| LodWebGpuError::Poll(error.to_string()))?;
        receiver
            .await
            .map_err(|_| LodWebGpuError::Mapping("map callback was canceled".to_string()))?
            .map_err(|error| LodWebGpuError::Mapping(error.to_string()))?;
        let view = slice.get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        self.buffer.unmap();
        Ok(OffscreenRgba8Image {
            size: self.size,
            bytes_per_row: self.bytes_per_row,
            bytes,
        })
    }
}

/// CPU-visible facts about encoding one shared frame. Compacted survivor
/// counts deliberately remain device-local; delayed telemetry may observe
/// them later without turning this result into a readback boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchFrameEncoding {
    pub logical_submission: RenderSubmissionStats,
    pub indirect_draw_calls: u32,
    pub source_instance_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FocusPatchFrameEncoding {
    pub scene: PatchFrameEncoding,
    pub postprocess: FocusPostprocessEncoding,
}

/// Exact queue traffic caused by retained per-batch/domain frame tables.
/// Counts live on the device so presentation skips and fallback paths remain
/// observable without threading diagnostic state through render semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameTableMemoDiagnostics {
    pub uploads: u64,
    pub reuses: u64,
    pub upload_bytes: u64,
}

/// Exact queue traffic for classifier-global dispatch words and retained
/// per-subject transform rows. Each classification records one outcome for
/// each half, independently of dynamic pose publication.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LodStateMemoDiagnostics {
    pub uploads: u64,
    pub reuses: u64,
    pub upload_bytes: u64,
}

/// Device-local pipelines shared by every uploaded classifier model.
pub struct LodClassifierDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    next_model_identity: Mutex<u64>,
    render_shader_modules: Mutex<DeviceMemo<ShaderModuleDescriptor, wgpu::ShaderModule>>,
    focus_postprocess_render_pipelines:
        Mutex<DeviceMemo<Vec<RenderPipelineDescriptor>, FocusPostprocessPipelines>>,
    prepared_patch_render_pipelines: Mutex<
        DeviceMemo<
            Vec<RenderPipelineDescriptor>,
            (Vec<PatchRenderPipeline>, Option<PatchRenderPipeline>),
        >,
    >,
    resident_root_render_pipelines:
        Mutex<DeviceMemo<Vec<RenderPipelineDescriptor>, ResidentRootRenderPipeline>>,
    pass1_pipeline: wgpu::ComputePipeline,
    pass2_pipeline: wgpu::ComputePipeline,
    resident_seed_pipeline: wgpu::ComputePipeline,
    resident_reconcile_2_to_1_pipeline: wgpu::ComputePipeline,
    resident_reconcile_4_to_1_pipeline: wgpu::ComputePipeline,
    resident_pack_pipeline: wgpu::ComputePipeline,
    patch_prepare_pipeline: wgpu::ComputePipeline,
    prepared_visibility_pipeline: wgpu::ComputePipeline,
    visibility_expand_pipeline: wgpu::ComputePipeline,
    lod_visibility_expand_pipeline: wgpu::ComputePipeline,
    resident_bucket_histogram_pipeline: wgpu::ComputePipeline,
    resident_bucket_prefix_pipeline: wgpu::ComputePipeline,
    resident_bucket_scan_pipeline: wgpu::ComputePipeline,
    resident_bucket_scatter_pipeline: wgpu::ComputePipeline,
    resident_root_visibility_pipeline: wgpu::ComputePipeline,
    resident_root_vertex_clear_pipeline: wgpu::ComputePipeline,
    resident_root_vertex_accumulate_pipeline: wgpu::ComputePipeline,
    resident_root_topology_pipeline: wgpu::ComputePipeline,
    visibility_count_pipeline: wgpu::ComputePipeline,
    visibility_scan_pipeline: wgpu::ComputePipeline,
    visibility_scatter_pipeline: wgpu::ComputePipeline,
    frame_table_uploads: AtomicU64,
    frame_table_reuses: AtomicU64,
    frame_table_upload_bytes: AtomicU64,
    lod_state_uploads: AtomicU64,
    lod_state_reuses: AtomicU64,
    lod_state_upload_bytes: AtomicU64,
}

/// Retained device buffers for one immutable prepared model and atlas lookup.
pub struct LodClassifierModel {
    identity: u64,
    atlas_keys: Vec<[u32; 3]>,
    prepared: PreparedLodModel,
    classification_epoch: u64,
    joint_capacity: usize,
    subject_rows: usize,
    subject_layout: WgslLodSubjectLayout,
    uniform: wgpu::Buffer,
    faces: wgpu::Buffer,
    skinning: wgpu::Buffer,
    joint_matrices: wgpu::Buffer,
    morph_deltas: wgpu::Buffer,
    morph_weights: wgpu::Buffer,
    subject_states: wgpu::Buffer,
    lod_state: Mutex<RetainedLodDispatchState>,
    packed_records: wgpu::Buffer,
    pass1_bind_group: wgpu::BindGroup,
    pass2_bind_group: wgpu::BindGroup,
    resident: ResidentLodReconciliationBuffers,
    resident_epoch: Cell<Option<(u64, FaceLodGrading)>>,
}

struct RetainedLodDispatchState {
    uniform_words: [u32; 68],
    subject_words: Vec<[u32; 40]>,
    subject_scratch: Vec<[u32; 40]>,
    uniform_published: bool,
    subjects_published: bool,
}

impl RetainedLodDispatchState {
    fn new(subject_rows: usize) -> Self {
        Self {
            uniform_words: [0; 68],
            subject_words: vec![[0; 40]; subject_rows],
            subject_scratch: vec![[0; 40]; subject_rows],
            uniform_published: false,
            subjects_published: false,
        }
    }

    fn commit_uniform(&mut self, words: [u32; 68]) -> FrameTablePublication {
        if self.uniform_published && self.uniform_words == words {
            FrameTablePublication::Reuse
        } else {
            self.uniform_words = words;
            self.uniform_published = true;
            FrameTablePublication::Upload {
                bytes: DISPATCH_UNIFORM_BYTES,
            }
        }
    }

    fn commit_subject_scratch(&mut self) -> FrameTablePublication {
        if self.subjects_published && self.subject_words == self.subject_scratch {
            FrameTablePublication::Reuse
        } else {
            std::mem::swap(&mut self.subject_words, &mut self.subject_scratch);
            self.subjects_published = true;
            FrameTablePublication::Upload {
                bytes: self.subject_words.len() as u64 * SUBJECT_RECORD_BYTES,
            }
        }
    }
}

struct ResidentLodReconciliationBuffers {
    packed_records: wgpu::Buffer,
    seed_bind_group: wgpu::BindGroup,
    reconcile_2_to_1_forward_bind_group: wgpu::BindGroup,
    reconcile_2_to_1_backward_bind_group: wgpu::BindGroup,
    reconcile_4_to_1_forward_bind_group: wgpu::BindGroup,
    reconcile_4_to_1_backward_bind_group: wgpu::BindGroup,
    pack_bind_group: wgpu::BindGroup,
}

/// One encoded, device-resident LOD classification result. The packed words
/// remain owned by the uploaded model and can be bound directly by downstream
/// compute passes. The mutable model borrow prevents another classification
/// from overwriting this epoch while a consumer is encoding against it.
pub struct DeviceLodClassification<'model> {
    model: &'model LodClassifierModel,
    face_count: u32,
    epoch: u64,
}

impl DeviceLodClassification<'_> {
    pub fn packed_records_buffer(&self) -> &wgpu::Buffer {
        &self.model.packed_records
    }

    pub fn face_count(&self) -> u32 {
        self.face_count
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Crack-free, within-face-graded resident topology derived from one exact
/// classifier epoch without a CPU publication or staging copy.
pub struct DeviceResidentLod<'classification> {
    packed_records: &'classification wgpu::Buffer,
    model_identity: u64,
    face_count: u32,
    classification_epoch: u64,
    grading: FaceLodGrading,
}

impl DeviceResidentLod<'_> {
    pub fn packed_records_buffer(&self) -> &wgpu::Buffer {
        self.packed_records
    }

    pub fn face_count(&self) -> u32 {
        self.face_count
    }

    pub fn classification_epoch(&self) -> u64 {
        self.classification_epoch
    }

    pub fn grading(&self) -> FaceLodGrading {
        self.grading
    }
}

/// Retained current-scene topology and prepared-patch output. Immutable source
/// faces and affine subject rows are uploaded once; only pose buffers and the
/// joint-count word change with animation.
pub struct PatchPreparationScene {
    patch_count: u32,
    subject_count: u32,
    uniform_words: [u32; 4],
    uniform: wgpu::Buffer,
    topology: wgpu::Buffer,
    source_faces: Arc<wgpu::Buffer>,
    subjects: wgpu::Buffer,
    prepared_records: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Retained WebGPU graphics pipelines for one prepared QB surface mode.
/// Winding is pipeline state in WebGPU, so both variants share one explicit
/// bind-group layout instead of recompiling or mutating state per batch.
#[derive(Clone)]
pub struct PatchRenderPipeline {
    kind: prepared_patch_pipeline::PreparedPatchPipelineKind,
    geometry: RenderGeometry,
    bind_group_layout: wgpu::BindGroupLayout,
    pbr_texture_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pbr_environment_bind_group_layout: Option<wgpu::BindGroupLayout>,
    counter_clockwise: wgpu::RenderPipeline,
    clockwise: wgpu::RenderPipeline,
}

/// Dedicated two-attachment PBR pipeline used only by focus composition.
/// Keeping the ordinary single-target pipeline inaccessible through this
/// wrapper prevents a focus pipeline from being submitted to a one-attachment
/// render pass by mistake.
pub struct FocusPbrPatchRenderPipeline {
    inner: PatchRenderPipeline,
    color_format: wgpu::TextureFormat,
    raw_field_format: wgpu::TextureFormat,
}

impl FocusPbrPatchRenderPipeline {
    pub const fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub const fn raw_field_format(&self) -> wgpu::TextureFormat {
        self.raw_field_format
    }
}

/// Retained WebGPU resources for the complete focus render graph. The root
/// and adaptive-overlay passes always target the fixed intermediate MRT; only
/// the final composer pipeline depends on the presentation format. The
/// viewport target is replaced atomically when its extent changes.
pub struct FocusPbrRenderResources {
    root_pipeline: ResidentRootRenderPipeline,
    overlay_pipeline: FocusPbrPatchRenderPipeline,
    postprocess_pipelines: FocusPostprocessPipelines,
    target: Option<FocusPostprocessTarget>,
}

impl FocusPbrRenderResources {
    pub const fn root_pipeline(&self) -> &ResidentRootRenderPipeline {
        &self.root_pipeline
    }

    pub const fn overlay_pipeline(&self) -> &FocusPbrPatchRenderPipeline {
        &self.overlay_pipeline
    }

    pub const fn postprocess_pipelines(&self) -> &FocusPostprocessPipelines {
        &self.postprocess_pipelines
    }

    pub const fn target(&self) -> Option<&FocusPostprocessTarget> {
        self.target.as_ref()
    }

    /// Ensure one target for the current viewport. Allocation finishes before
    /// replacing the previous target, so an error leaves the last coherent
    /// render graph resident. Returns whether a replacement was published.
    pub fn ensure_target(
        &mut self,
        device: &LodClassifierDevice,
        size: [u32; 2],
    ) -> Result<bool, LodWebGpuError> {
        if self.target.as_ref().is_some_and(|target| target.size() == size) {
            return Ok(false);
        }
        let target = device.create_focus_postprocess_target(size, &self.postprocess_pipelines)?;
        self.target = Some(target);
        Ok(true)
    }
}

impl PatchRenderPipeline {
    fn style(&self) -> Option<RenderStyle> {
        match self.kind {
            prepared_patch_pipeline::PreparedPatchPipelineKind::Style(style) => Some(style),
            prepared_patch_pipeline::PreparedPatchPipelineKind::Highlight => None,
        }
    }

    fn uses_pbr_bindings(&self) -> bool {
        self.style() == Some(RenderStyle::Pbr)
    }
}

/// Retained diagnostic pipelines sharing one shader module and bind-group
/// layout. Triangle, line, and composite frames can switch without rebuilding
/// scene bindings or duplicating geometry residency.
pub struct DiagnosticPatchRenderPipelines {
    pbr: PatchRenderPipeline,
    normals: PatchRenderPipeline,
    matcap: PatchRenderPipeline,
    lod: PatchRenderPipeline,
    stretch: PatchRenderPipeline,
    wire: PatchRenderPipeline,
    highlight: PatchRenderPipeline,
}

/// Whether the retained WebGPU patch renderer can present this style without
/// falling back to the incumbent backend. Keep live adapters on this single
/// capability predicate so a newly added pipeline cannot be stranded behind
/// a stale front-door mode list.
pub const fn supports_patch_presentation_style(style: RenderStyle) -> bool {
    matches!(
        style,
        RenderStyle::Matcap
            | RenderStyle::Wire
            | RenderStyle::Normals
            | RenderStyle::MatcapWire
            | RenderStyle::Lod
            | RenderStyle::Stretch
    )
}

/// Whether the current authored-factor and material-texture PBR pipeline can
/// render a shared scene within its declared feature subset. Unlike the static
/// diagnostic-style predicate above, this depends on coherent scene residency.
pub fn supports_basic_pbr_frame(scene: &RenderSceneSnapshot, options: RenderFrameOptions) -> bool {
    validate_basic_pbr_frame(scene, options).is_ok()
}

/// Whether the authored PBR subset can execute through the focus MRT and
/// retained post-processing schedule. This is intentionally distinct from
/// [`supports_basic_pbr_frame`]: callers must never admit a focus packet to a
/// single-target PBR pipeline or silently discard the composition request.
pub fn supports_focus_pbr_frame(scene: &RenderSceneSnapshot, options: RenderFrameOptions) -> bool {
    validate_focus_pbr_frame(scene, options).is_ok()
}

fn validate_basic_pbr_frame(
    scene: &RenderSceneSnapshot,
    options: RenderFrameOptions,
) -> Result<(), LodWebGpuError> {
    validate_pbr_material_subset(scene)?;
    if options.focus_postprocess.is_some() {
        return Err(LodWebGpuError::Payload(
            "basic WebGPU PBR does not yet support focus post-processing".to_string(),
        ));
    }
    Ok(())
}

fn validate_focus_pbr_frame(
    scene: &RenderSceneSnapshot,
    options: RenderFrameOptions,
) -> Result<(), LodWebGpuError> {
    validate_pbr_material_subset(scene)?;
    if options.focus_postprocess.is_none() {
        return Err(LodWebGpuError::Payload(
            "focus WebGPU PBR requires a focus postprocess packet".to_string(),
        ));
    }
    Ok(())
}

fn validate_pbr_material_subset(scene: &RenderSceneSnapshot) -> Result<(), LodWebGpuError> {
    scene
        .validate()
        .map_err(|error| LodWebGpuError::Payload(format!("render scene contract: {error}")))?;
    if scene.batches.is_empty() {
        return Err(LodWebGpuError::Payload(
            "basic WebGPU PBR requires a nonempty scene".to_string(),
        ));
    }
    let default_material = PbrMaterial::default();
    for (batch_index, batch) in scene.batches.iter().enumerate() {
        if batch.pbr_class != PbrDrawClass::Opaque {
            return Err(LodWebGpuError::Payload(format!(
                "basic WebGPU PBR batch {batch_index} is not opaque",
            )));
        }
        let (_, material) = pbr_material_for_index(
            &scene.materials,
            &default_material,
            batch.id.key.material_index,
        );
        if material.alpha_mode == PbrAlphaMode::Blend || material.transmission_factor > 0.0 {
            return Err(LodWebGpuError::Payload(format!(
                "basic WebGPU PBR batch {batch_index} requires blending or transmission",
            )));
        }
        if material.has_sheen {
            return Err(LodWebGpuError::Payload(format!(
                "basic WebGPU PBR batch {batch_index} requires sheen",
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameTablePublication {
    Upload { bytes: u64 },
    Reuse,
}

struct RetainedFrameTable {
    words: Vec<u32>,
    row_words: usize,
    byte_len: u64,
    published: bool,
}

impl RetainedFrameTable {
    fn new(word_count: usize, row_words: usize, byte_len: u64) -> Self {
        debug_assert_eq!(word_count.saturating_mul(4) as u64, byte_len);
        debug_assert!(row_words != 0 && word_count.is_multiple_of(row_words));
        Self {
            words: vec![0; word_count],
            row_words,
            byte_len,
            published: false,
        }
    }

    fn begin_update(&self) -> bool {
        !self.published
    }

    fn replace_row(&mut self, row: usize, next: &[u32]) -> bool {
        debug_assert_eq!(next.len(), self.row_words);
        let start = row * self.row_words;
        let destination = &mut self.words[start..start + self.row_words];
        if destination == next {
            false
        } else {
            destination.copy_from_slice(next);
            true
        }
    }

    fn invalidate(&mut self) {
        self.published = false;
    }

    fn commit(&mut self, changed: bool) -> FrameTablePublication {
        self.published = true;
        if changed {
            FrameTablePublication::Upload {
                bytes: self.byte_len,
            }
        } else {
            FrameTablePublication::Reuse
        }
    }
}

pub(crate) struct PatchRenderGlobalResidency {
    buffer: wgpu::Buffer,
    table: Mutex<RetainedFrameTable>,
}

/// Scene/frame bindings shared by every atlas bucket and indirect batch draw.
pub struct PatchRenderBindings {
    domain_count: u32,
    material_count: u32,
    global_frame: Arc<PatchRenderGlobalResidency>,
    domains: wgpu::Buffer,
    materials: wgpu::Buffer,
    material_textures: Option<PbrMaterialTextureBindings>,
    pbr_environment: Option<PbrEnvironmentBindings>,
    domain_table: Mutex<RetainedFrameTable>,
    bind_group: wgpu::BindGroup,
}

/// One coherent retained scene for prepared-patch rendering. Applications
/// replace this aggregate atomically when extracted batch membership changes;
/// frame code cannot accidentally combine preparation, compaction, and bind
/// groups from different scene epochs.
pub struct PatchRenderScene {
    model_identity: u64,
    scene: ValidatedRenderScene,
    patches: PatchPreparationScene,
    visibility: VisibilityCompactionScene,
    face_visibility: FaceVisibilityExpansionScene,
    bindings: PatchRenderBindings,
}

/// Outcome of attempting to publish a new extracted topology into retained
/// scene allocations. Shape changes return ownership of the validated input so
/// the caller can construct an atomic replacement without cloning it.
pub enum PatchRenderSceneUpdate {
    Updated,
    ShapeChanged(ValidatedRenderScene),
}

/// Compact per-face CPU/shadow visibility and its retained expansion bindings.
/// A future same-device classifier can replace the bitset producer while
/// preserving the topology-to-source expansion contract.
struct FaceVisibilityExpansionScene {
    face_count: u32,
    word_count: u32,
    bits: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    lod_bind_group: wgpu::BindGroup,
}

/// Retained scene shape and output buffers for deterministic visibility
/// compaction. Only `source_visibility` changes with the current pose.
pub struct VisibilityCompactionScene {
    batch_count: u32,
    source_count: u32,
    batch_index_uniform_stride: u32,
    batch_index_uniform: wgpu::Buffer,
    batches: wgpu::Buffer,
    source_eligibility: wgpu::Buffer,
    source_visibility: wgpu::Buffer,
    compacted_source_instances: wgpu::Buffer,
    compacted_ranges: wgpu::Buffer,
    triangle_indirect_arguments: wgpu::Buffer,
    line_indirect_arguments: wgpu::Buffer,
    count_bind_group: wgpu::BindGroup,
    scan_bind_group: wgpu::BindGroup,
    scatter_bind_group: wgpu::BindGroup,
}

impl LodClassifierDevice {
    /// Request a compute/render device without claiming a window or canvas.
    /// Hyperscope uses this for rollback-safe shadow residency before the
    /// explicit backend switch owns a presentation surface.
    pub async fn request_headless(
        label: &str,
    ) -> Result<(Self, WebGpuAdapterSummary), LodWebGpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|error| {
                LodWebGpuError::Payload(format!("WebGPU adapter request failed: {error}"))
            })?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                ..Default::default()
            })
            .await
            .map_err(|error| {
                LodWebGpuError::Payload(format!("WebGPU device request failed: {error}"))
            })?;
        let summary = WebGpuAdapterSummary {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        };
        Ok((Self::new(device, queue)?, summary))
    }

    /// Compile and retain the two flattened WGSL pipelines on an existing
    /// device. Device creation and adapter policy stay with the application.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, LodWebGpuError> {
        let pass1_source = quilting_shaders::compile_lod_pass1_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass2_source = quilting_shaders::compile_lod_pass2_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let resident_source = quilting_shaders::compile_lod_resident_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let patch_prepare_source = quilting_shaders::compile_patch_prepare_compute_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let prepared_visibility_source = quilting_shaders::compile_prepared_visibility_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_expand_source = quilting_shaders::compile_visibility_expand_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let lod_visibility_expand_source =
            quilting_shaders::compile_lod_visibility_expand_wgsl()
                .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_count_source = quilting_shaders::compile_visibility_count_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_scan_source = quilting_shaders::compile_visibility_scan_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_scatter_source = quilting_shaders::compile_visibility_scatter_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let resident_buckets_source = quilting_shaders::compile_resident_buckets_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let resident_root_visibility_source =
            quilting_shaders::compile_resident_root_visibility_wgsl()
                .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let resident_root_topology_source = quilting_shaders::compile_resident_root_topology_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass1_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass one"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass1_source)),
        });
        let pass2_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass two"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass2_source)),
        });
        let resident_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting resident LOD reconciliation"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(resident_source)),
        });
        let patch_prepare_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting patch preparation"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(patch_prepare_source)),
        });
        let prepared_visibility_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quilting prepared patch visibility"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(prepared_visibility_source)),
            });
        let visibility_expand_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting face visibility expansion"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_expand_source)),
        });
        let lod_visibility_expand_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quilting resident LOD visibility expansion"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(lod_visibility_expand_source)),
            });
        let visibility_count_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting visibility count"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_count_source)),
        });
        let visibility_scan_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting visibility scan"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_scan_source)),
        });
        let visibility_scatter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting visibility scatter"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_scatter_source)),
        });
        let resident_buckets_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting resident geometry buckets"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(resident_buckets_source)),
        });
        let resident_root_visibility_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quilting resident root visibility"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(resident_root_visibility_source)),
            });
        let resident_root_topology_module =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quilting resident root topology"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(resident_root_topology_source)),
            });
        let pass1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("quilting LOD pass one"),
            layout: None,
            module: &pass1_module,
            entry_point: Some(quilting_shaders::LOD_PASS1_DEVICE_ENTRY_POINT),
            compilation_options: Default::default(),
            cache: None,
        });
        let pass2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("quilting LOD pass two"),
            layout: None,
            module: &pass2_module,
            entry_point: Some(quilting_shaders::LOD_PASS2_DEVICE_ENTRY_POINT),
            compilation_options: Default::default(),
            cache: None,
        });
        let resident_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &resident_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let resident_seed_pipeline = resident_pipeline(
            "quilting resident LOD seed",
            quilting_shaders::LOD_RESIDENT_SEED_DEVICE_ENTRY_POINT,
        );
        let resident_reconcile_2_to_1_pipeline = resident_pipeline(
            "quilting resident LOD 2:1 reconciliation",
            quilting_shaders::LOD_RESIDENT_RECONCILE_2_TO_1_DEVICE_ENTRY_POINT,
        );
        let resident_reconcile_4_to_1_pipeline = resident_pipeline(
            "quilting resident LOD 4:1 reconciliation",
            quilting_shaders::LOD_RESIDENT_RECONCILE_4_TO_1_DEVICE_ENTRY_POINT,
        );
        let resident_pack_pipeline = resident_pipeline(
            "quilting resident LOD packing",
            quilting_shaders::LOD_RESIDENT_PACK_DEVICE_ENTRY_POINT,
        );
        let patch_prepare_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting patch preparation"),
                layout: None,
                module: &patch_prepare_module,
                entry_point: Some(quilting_shaders::PATCH_PREPARE_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let prepared_visibility_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting prepared patch visibility"),
                layout: None,
                module: &prepared_visibility_module,
                entry_point: Some(quilting_shaders::PREPARED_VISIBILITY_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let visibility_expand_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting face visibility expansion"),
                layout: None,
                module: &visibility_expand_module,
                entry_point: Some(quilting_shaders::VISIBILITY_EXPAND_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let lod_visibility_expand_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting resident LOD visibility expansion"),
                layout: None,
                module: &lod_visibility_expand_module,
                entry_point: Some(quilting_shaders::LOD_VISIBILITY_EXPAND_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let visibility_count_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting visibility count"),
                layout: None,
                module: &visibility_count_module,
                entry_point: Some(quilting_shaders::VISIBILITY_COUNT_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let visibility_scan_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting visibility scan"),
                layout: None,
                module: &visibility_scan_module,
                entry_point: Some(quilting_shaders::VISIBILITY_SCAN_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let visibility_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting visibility scatter"),
                layout: None,
                module: &visibility_scatter_module,
                entry_point: Some(quilting_shaders::VISIBILITY_SCATTER_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let resident_bucket_histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting resident geometry bucket histogram"),
                layout: None,
                module: &resident_buckets_module,
                entry_point: Some(quilting_shaders::RESIDENT_BUCKET_HISTOGRAM_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let resident_bucket_prefix_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting resident geometry bucket chunk prefix"),
                layout: None,
                module: &resident_buckets_module,
                entry_point: Some(quilting_shaders::RESIDENT_BUCKET_PREFIX_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let resident_bucket_scan_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting resident geometry bucket scan"),
                layout: None,
                module: &resident_buckets_module,
                entry_point: Some(quilting_shaders::RESIDENT_BUCKET_SCAN_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let resident_bucket_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting resident geometry bucket scatter"),
                layout: None,
                module: &resident_buckets_module,
                entry_point: Some(quilting_shaders::RESIDENT_BUCKET_SCATTER_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let resident_root_visibility_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting resident root visibility"),
                layout: None,
                module: &resident_root_visibility_module,
                entry_point: Some(quilting_shaders::RESIDENT_ROOT_VISIBILITY_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let resident_root_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &resident_root_topology_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let resident_root_vertex_clear_pipeline = resident_root_pipeline(
            "quilting resident root vertex LOD clear",
            quilting_shaders::RESIDENT_ROOT_VERTEX_CLEAR_DEVICE_ENTRY_POINT,
        );
        let resident_root_vertex_accumulate_pipeline = resident_root_pipeline(
            "quilting resident root vertex LOD accumulation",
            quilting_shaders::RESIDENT_ROOT_VERTEX_ACCUMULATE_DEVICE_ENTRY_POINT,
        );
        let resident_root_topology_pipeline = resident_root_pipeline(
            "quilting resident root topology emission",
            quilting_shaders::RESIDENT_ROOT_TOPOLOGY_DEVICE_ENTRY_POINT,
        );
        Ok(Self {
            device,
            queue,
            next_model_identity: Mutex::new(1),
            render_shader_modules: Mutex::new(DeviceMemo::new(0)),
            focus_postprocess_render_pipelines: Mutex::new(DeviceMemo::new(0)),
            prepared_patch_render_pipelines: Mutex::new(DeviceMemo::new(0)),
            resident_root_render_pipelines: Mutex::new(DeviceMemo::new(0)),
            pass1_pipeline,
            pass2_pipeline,
            resident_seed_pipeline,
            resident_reconcile_2_to_1_pipeline,
            resident_reconcile_4_to_1_pipeline,
            resident_pack_pipeline,
            patch_prepare_pipeline,
            prepared_visibility_pipeline,
            visibility_expand_pipeline,
            lod_visibility_expand_pipeline,
            resident_bucket_histogram_pipeline,
            resident_bucket_prefix_pipeline,
            resident_bucket_scan_pipeline,
            resident_bucket_scatter_pipeline,
            resident_root_visibility_pipeline,
            resident_root_vertex_clear_pipeline,
            resident_root_vertex_accumulate_pipeline,
            resident_root_topology_pipeline,
            visibility_count_pipeline,
            visibility_scan_pipeline,
            visibility_scatter_pipeline,
            frame_table_uploads: AtomicU64::new(0),
            frame_table_reuses: AtomicU64::new(0),
            frame_table_upload_bytes: AtomicU64::new(0),
            lod_state_uploads: AtomicU64::new(0),
            lod_state_reuses: AtomicU64::new(0),
            lod_state_upload_bytes: AtomicU64::new(0),
        })
    }

    /// Borrow the application-supplied device for attachment and atlas
    /// residency creation. Quilting retains pipelines, not ownership of the
    /// surrounding render graph.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the application-supplied queue. Frame uploads are ordered before
    /// the caller's subsequent command submission on this same queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn record_frame_table_publication(&self, publication: FrameTablePublication) {
        match publication {
            FrameTablePublication::Upload { bytes } => {
                self.frame_table_uploads.fetch_add(1, Ordering::Relaxed);
                self.frame_table_upload_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            FrameTablePublication::Reuse => {
                self.frame_table_reuses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Device-lifetime frame-table traffic. A reuse means the exact retained
    /// words remained valid and no `Queue::write_buffer` call was issued.
    pub fn frame_table_memo_diagnostics(&self) -> FrameTableMemoDiagnostics {
        FrameTableMemoDiagnostics {
            uploads: self.frame_table_uploads.load(Ordering::Relaxed),
            reuses: self.frame_table_reuses.load(Ordering::Relaxed),
            upload_bytes: self.frame_table_upload_bytes.load(Ordering::Relaxed),
        }
    }

    fn record_lod_state_publication(&self, publication: FrameTablePublication) {
        match publication {
            FrameTablePublication::Upload { bytes } => {
                self.lod_state_uploads.fetch_add(1, Ordering::Relaxed);
                self.lod_state_upload_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
            }
            FrameTablePublication::Reuse => {
                self.lod_state_reuses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Device-lifetime classifier-state traffic. Dynamic pose arrays are
    /// reported separately by the browser's pose-identity diagnostics.
    pub fn lod_state_memo_diagnostics(&self) -> LodStateMemoDiagnostics {
        LodStateMemoDiagnostics {
            uploads: self.lod_state_uploads.load(Ordering::Relaxed),
            reuses: self.lod_state_reuses.load(Ordering::Relaxed),
            upload_bytes: self.lod_state_upload_bytes.load(Ordering::Relaxed),
        }
    }

    /// Lower one immutable composable WGSL descriptor once per device. The
    /// raw entry source and complete compiler catalog form the cache key;
    /// flattening and `wgpu` module creation occur only on a miss.
    pub(crate) fn memoized_render_shader_module(
        &self,
        label: &'static str,
        source: &'static str,
        representative_entry_point: &'static str,
        compile: impl FnOnce() -> Result<String, Box<dyn std::error::Error>>,
    ) -> Result<wgpu::ShaderModule, LodWebGpuError> {
        let descriptor = ShaderModuleDescriptor::new(
            label,
            source,
            quilting_shaders::compiler_catalog_revision(),
            ShaderStage::Vertex,
            representative_entry_point,
            ShaderTarget::Wgsl,
            Vec::new(),
        )
        .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let mut modules = self.render_shader_modules.lock().map_err(|_| {
            LodWebGpuError::Payload("WebGPU render shader memo was poisoned".to_string())
        })?;
        let module = modules.get_or_try_insert_with(descriptor, |descriptor| {
            let source = compile().map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
            Ok::<_, LodWebGpuError>(self.device.create_shader_module(
                wgpu::ShaderModuleDescriptor {
                    label: Some(descriptor.label()),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
                },
            ))
        })?;
        Ok(module.clone())
    }

    /// Observable cache state for functional render diagnostics and tests.
    pub fn render_shader_memo_diagnostics(&self) -> DeviceMemoDiagnostics {
        self.render_shader_modules
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .diagnostics()
    }

    /// Observable retained resident-root pipeline state. The memo is scoped to
    /// this device, so dropping the device releases every cached WebGPU object
    /// without a process-global registry or hidden invalidation channel.
    pub fn resident_root_pipeline_memo_diagnostics(&self) -> DeviceMemoDiagnostics {
        self.resident_root_render_pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .diagnostics()
    }

    /// Observable cache state for the focus composer pipeline family. Its key
    /// is the complete backend-neutral descriptor vector, not a WebGPU handle
    /// or an application-owned focus setting.
    pub fn focus_postprocess_pipeline_memo_diagnostics(&self) -> DeviceMemoDiagnostics {
        self.focus_postprocess_render_pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .diagnostics()
    }

    /// Observable cache state for prepared/adaptive patch graphics families.
    /// Exact backend-neutral descriptor vectors are retained for this device;
    /// unsupported nonportable formats deliberately bypass the memo.
    pub fn prepared_patch_pipeline_memo_diagnostics(&self) -> DeviceMemoDiagnostics {
        self.prepared_patch_render_pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .diagnostics()
    }

    /// Upload Hyperscope's existing seven-word packed-atlas metadata and three
    /// shared geometry arrays. This is the WebGPU counterpart of WebGL2's
    /// `TessAtlasBuffers`; it deliberately preserves the same canonical keys
    /// and global index values.
    pub fn upload_packed_patch_atlas(
        &self,
        patches: &[u32],
        barycentrics: &[f32],
        triangle_indices: &[u32],
        line_indices: &[u32],
    ) -> Result<PackedPatchAtlas, LodWebGpuError> {
        if patches.is_empty() || !patches.len().is_multiple_of(7) {
            return Err(LodWebGpuError::Payload(
                "packed WebGPU atlas metadata must contain nonempty seven-word records".to_string(),
            ));
        }
        if barycentrics.is_empty()
            || !barycentrics.len().is_multiple_of(3)
            || barycentrics.iter().any(|value| !value.is_finite())
        {
            return Err(LodWebGpuError::Payload(
                "packed WebGPU atlas barycentrics must contain finite vec3 records".to_string(),
            ));
        }
        let vertex_count = u32::try_from(barycentrics.len() / 3).map_err(|_| {
            LodWebGpuError::Payload("packed WebGPU atlas vertex count exceeds u32".to_string())
        })?;
        let triangle_index_count = u32::try_from(triangle_indices.len()).map_err(|_| {
            LodWebGpuError::Payload(
                "packed WebGPU atlas triangle index count exceeds u32".to_string(),
            )
        })?;
        let line_index_count = u32::try_from(line_indices.len()).map_err(|_| {
            LodWebGpuError::Payload("packed WebGPU atlas line index count exceeds u32".to_string())
        })?;
        if triangle_indices
            .iter()
            .chain(line_indices)
            .any(|&index| index >= vertex_count)
        {
            return Err(LodWebGpuError::Payload(
                "packed WebGPU atlas contains an out-of-range global vertex index".to_string(),
            ));
        }

        let keys = patches
            .chunks_exact(7)
            .map(|patch| [patch[0], patch[1], patch[2]])
            .collect::<Vec<_>>();
        let lookup = prepare_lod_atlas_lookup(keys.iter().copied()).map_err(|error| {
            LodWebGpuError::Payload(format!("packed WebGPU atlas keys: {error}"))
        })?;
        let mut entries = BTreeMap::new();
        for (entry_index, patch) in patches.chunks_exact(7).enumerate() {
            let key = [patch[0], patch[1], patch[2]];
            let triangle_end = patch[3].checked_add(patch[4]).ok_or_else(|| {
                LodWebGpuError::Payload(format!(
                    "packed WebGPU atlas entry {entry_index} triangle range overflows",
                ))
            })?;
            let line_end = patch[5].checked_add(patch[6]).ok_or_else(|| {
                LodWebGpuError::Payload(format!(
                    "packed WebGPU atlas entry {entry_index} line range overflows",
                ))
            })?;
            if triangle_end > triangle_index_count || line_end > line_index_count {
                return Err(LodWebGpuError::Payload(format!(
                    "packed WebGPU atlas entry {entry_index} range exceeds its index buffer",
                )));
            }
            if entries
                .insert(
                    key,
                    PackedPatchAtlasEntry {
                        triangle_first_index: patch[3],
                        triangle_index_count: patch[4],
                        line_first_index: patch[5],
                        line_index_count: patch[6],
                    },
                )
                .is_some()
            {
                return Err(LodWebGpuError::Payload(format!(
                    "packed WebGPU atlas repeats canonical key {key:?}",
                )));
            }
        }

        let barycentric_buffer = buffer_init_or_zero(
            &self.device,
            "packed patch atlas barycentrics",
            bytemuck::cast_slice(barycentrics),
            wgpu::BufferUsages::VERTEX,
        );
        let triangle_index_buffer = buffer_init_or_zero(
            &self.device,
            "packed patch atlas triangle indices",
            bytemuck::cast_slice(triangle_indices),
            wgpu::BufferUsages::INDEX,
        );
        let line_index_buffer = buffer_init_or_zero(
            &self.device,
            "packed patch atlas line indices",
            bytemuck::cast_slice(line_indices),
            wgpu::BufferUsages::INDEX,
        );
        Ok(PackedPatchAtlas {
            barycentric_buffer,
            triangle_index_buffer,
            line_index_buffer,
            entries,
            keys: lookup.keys,
            vertex_count,
            triangle_index_count,
            line_index_count,
        })
    }

    async fn run_resident_lod_conformance(
        &self,
    ) -> Result<(usize, usize, usize, usize), LodWebGpuError> {
        let face_count = 10usize;
        let mut positions = vec![0.0, 0.0, 0.0];
        for point in 0..=face_count {
            let angle = point as f32 * std::f32::consts::TAU / (face_count + 1) as f32;
            positions.extend_from_slice(&[angle.cos(), angle.sin(), 0.0]);
        }
        let faces = (0..face_count)
            .map(|face| [0, face as u32 + 1, face as u32 + 2])
            .collect::<Vec<_>>();
        let vertex_count = positions.len() / 3;
        let mut source_instances = vec![0.0; face_count * PREPARED_PATCH_RECORD_WORDS];
        for (face, vertices) in faces.iter().copied().enumerate() {
            let mut writer = InstanceWriter::new(&mut source_instances, face);
            for (corner, vertex) in vertices.into_iter().enumerate() {
                writer.set_position(
                    corner,
                    vertex,
                    [
                        positions[vertex as usize * 3],
                        positions[vertex as usize * 3 + 1],
                        positions[vertex as usize * 3 + 2],
                    ],
                );
                writer.set_normal(corner, [0.0, 0.0, 1.0]);
            }
            writer.set_face_id(face as u32);
            writer.set_node_id(0);
        }
        let source_face_words = source_instances
            .chunks_exact(PREPARED_PATCH_RECORD_WORDS)
            .map(|source| std::array::from_fn(|word| source[word].to_bits()))
            .collect::<Vec<_>>();
        let prepared = prepare_lod_model(LodModelData {
            positions,
            faces,
            joint_indices: vec![[0; 4]; vertex_count],
            joint_weights: vec![[0.0; 4]; vertex_count],
            morph_deltas: Vec::new(),
            num_morph_targets: 0,
            face_nodes: vec![0; face_count],
        })
        .map_err(LodWebGpuError::Payload)?;
        let model_words = pack_wgsl_lod_model_words(&prepared).map_err(LodWebGpuError::Payload)?;
        let mut atlas_keys = Vec::with_capacity(220);
        for a in 0..=9 {
            for b in a..=9 {
                for c in b..=9 {
                    atlas_keys.push([1u32 << a, 1u32 << b, 1u32 << c]);
                }
            }
        }
        let atlas = prepare_lod_atlas_lookup(atlas_keys).map_err(LodWebGpuError::Payload)?;
        let atlas_words = pack_wgsl_lod_atlas_words(&atlas);
        let request = |exponent: u32, visible: bool, priority| {
            pack_lod_classification([exponent; 3], 0, visible.then_some(0), priority)
                .expect("conformance request is inside the packed ABI")
        };
        let mut requested = vec![request(0, true, 0); face_count];
        requested[0] = request(9, true, 17);
        requested[2] = request(0, false, 23);

        let mut model = self.upload_model(prepared, &atlas)?;
        let face_subject_rows = (0..face_count)
            .map(|face| (face % 2) as u32)
            .collect::<Vec<_>>();
        let mut subject_zero = [0u32; PATCH_SUBJECT_RECORD_WORDS];
        let mut subject_one = [0u32; PATCH_SUBJECT_RECORD_WORDS];
        for (word, value) in subject_zero[..16].iter_mut().zip(identity_matrix()) {
            *word = value.to_bits();
        }
        for (word, value) in subject_zero[16..].iter_mut().zip(identity_matrix()) {
            *word = value.to_bits();
        }
        subject_one.copy_from_slice(&subject_zero);
        subject_one[12] = 0.25f32.to_bits();
        let root_words = WgslResidentRootPreparationSceneWords {
            uniform: [face_count as u32, vertex_count as u32, 0, 0],
            source_faces: source_face_words,
            subjects: vec![subject_zero, subject_one],
            face_subject_rows: face_subject_rows.clone(),
        };
        let draw_domains = ResidentRootDrawDomains {
            domains: vec![
                ResidentRootDrawDomain {
                    material_index: 3,
                    render_node_index: 1,
                    pbr_class: PbrDrawClass::Opaque,
                    transform: RenderEntityTransform {
                        mobius: identity_mobius(),
                        orientation_sign: 1,
                        euclidean_model: identity_matrix(),
                        euclidean_normal: identity_matrix(),
                    },
                    enabled: true,
                },
                ResidentRootDrawDomain {
                    material_index: 1_000_000,
                    render_node_index: 9,
                    pbr_class: PbrDrawClass::Blend,
                    transform: RenderEntityTransform {
                        mobius: identity_mobius(),
                        orientation_sign: -1,
                        euclidean_model: identity_matrix(),
                        euclidean_normal: identity_matrix(),
                    },
                    enabled: true,
                },
            ],
            face_domain_rows: face_subject_rows.clone(),
        };
        let root_preparation =
            self.upload_resident_root_preparation_words(&model, root_words.clone(), draw_domains)?;
        let actual_domains = self
            .resident_root_draw_domains_for_diagnostics(&root_preparation.draw_domains)
            .await?;
        let expected_domains = ResidentRootDrawDomainOutput {
            face_domain_rows: face_subject_rows.clone(),
            domain_records: vec![[3, 1, 0, 1], [1_000_000, 9, 1, 3]],
        };
        if actual_domains != expected_domains {
            return Err(LodWebGpuError::Conformance(format!(
                "resident root draw-domain mismatch: expected {expected_domains:?}, got {actual_domains:?}",
            )));
        }
        let compared_domain_words =
            actual_domains.face_domain_rows.len() + actual_domains.domain_records.len() * 4;
        let mut dispatch_words = [0u32; 68];
        dispatch_words[64] = face_count as u32;
        self.queue
            .write_buffer(&model.uniform, 0, bytemuck::cast_slice(&dispatch_words));
        self.queue
            .write_buffer(&model.packed_records, 0, bytemuck::cast_slice(&requested));
        model.classification_epoch = 1;
        let classification = DeviceLodClassification {
            model: &model,
            face_count: face_count as u32,
            epoch: 1,
        };
        let mut compared_words = 0usize;
        let mut compared_topology_words = 0usize;
        let mut compared_prepared_words = 0usize;
        for grading in [FaceLodGrading::TwoToOne, FaceLodGrading::FourToOne] {
            let expected = reconcile_and_pack_wgsl_resident_lods(
                &requested,
                &model_words.adjacency,
                &atlas_words,
                grading,
            )
            .map_err(LodWebGpuError::Conformance)?;
            let resident = self.reconcile_resident_lod_on_device(&classification, grading);
            let actual = self.read_resident_lod_for_diagnostics(&resident).await?;
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected);
                return Err(LodWebGpuError::Conformance(format!(
                    "resident {grading:?} chain mismatch at {mismatch:?}: expected {expected:?}, got {actual:?}",
                )));
            }
            let expected_topology = wgsl_resident_root_topology_oracle_words(
                &model.prepared,
                &expected,
                &face_subject_rows,
                2,
            )
            .map_err(LodWebGpuError::Conformance)?;
            let actual_topology = self
                .resident_root_topology_for_diagnostics(&root_preparation.topology, &resident)
                .await?;
            if actual_topology != expected_topology.topology {
                let mismatch = actual_topology
                    .iter()
                    .zip(&expected_topology.topology)
                    .position(|(actual, expected)| actual != expected);
                return Err(LodWebGpuError::Conformance(format!(
                    "resident root {grading:?} topology mismatch at face {mismatch:?}",
                )));
            }
            let ordinary_scene = self.upload_patch_preparation_scene(
                &model,
                WgslPatchPreparationSceneWords {
                    uniform: root_words.uniform,
                    topology: expected_topology.topology.clone(),
                    source_faces: root_words.source_faces.clone(),
                    subjects: root_words.subjects.clone(),
                },
            )?;
            let ordinary_prepared = self
                .prepare_patches(&model, &ordinary_scene, LodPose::default(), 0)
                .await?;
            let resident_prepared = self
                .prepare_resident_roots_for_diagnostics(
                    &model,
                    &root_preparation,
                    &resident,
                    LodPose::default(),
                    0,
                )
                .await?;
            if resident_prepared != ordinary_prepared {
                let mismatch = resident_prepared
                    .iter()
                    .zip(&ordinary_prepared)
                    .position(|(actual, expected)| actual != expected);
                return Err(LodWebGpuError::Conformance(format!(
                    "resident root {grading:?} preparation mismatch at face {mismatch:?}",
                )));
            }
            compared_words += actual.len();
            compared_topology_words += actual_topology.len() * 12;
            compared_prepared_words += resident_prepared.len() * PREPARED_PATCH_RECORD_WORDS;
        }
        Ok((
            compared_words,
            compared_topology_words,
            compared_prepared_words,
            compared_domain_words,
        ))
    }

    /// Prove the direct chain from packed resident words through root topology,
    /// sparse dyadic replacement preparation, domain-aware bucketing, shared
    /// global-atlas indirect drawing, and composited rasterization without an
    /// intermediate map or duplicated source-face buffer.
    async fn validate_resident_adaptive_render_conformance(
        &self,
        presentation: Option<&mut PatchPresentationSurface>,
    ) -> Result<(usize, RenderImageSignature, usize, usize, usize), LodWebGpuError> {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 32;
        const PADDED_BYTES_PER_ROW: u32 = 256;

        let positions = vec![
            -0.9, -0.7, 0.0, -0.1, -0.7, 0.0, -0.5, 0.5, 0.0, 0.1, -0.7, 0.0, 0.9, -0.7, 0.0, 0.5,
            0.5, 0.0,
        ];
        let prepared = prepare_lod_model(LodModelData {
            positions: positions.clone(),
            faces: vec![[0, 1, 2], [3, 4, 5]],
            joint_indices: vec![[0; 4]; 6],
            joint_weights: vec![[0.0; 4]; 6],
            morph_deltas: Vec::new(),
            num_morph_targets: 0,
            face_nodes: vec![7, 8],
        })
        .map_err(LodWebGpuError::Conformance)?;
        let atlas_lookup =
            prepare_lod_atlas_lookup([[1, 1, 1]]).map_err(LodWebGpuError::Conformance)?;
        let mut model = self.upload_model(prepared, &atlas_lookup)?;
        let packed_atlas = self.upload_packed_patch_atlas(
            &[1, 1, 1, 0, 3, 0, 6],
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[0, 1, 2],
            &[0, 1, 1, 2, 2, 0],
        )?;
        let transform = RenderEntityTransform {
            mobius: identity_mobius(),
            // The effective-parity bucket must absorb this reversal.
            orientation_sign: -1,
            euclidean_model: identity_matrix(),
            euclidean_normal: identity_matrix(),
        };
        let render_scene = RenderSceneSnapshot {
            revision: 101,
            materials: Vec::new(),
            suppressed_root_faces: vec![0],
            batches: vec![
                RenderBatchSnapshot {
                    id: RenderBatchId {
                        key: RenderBatchKey {
                            lod: [1; 3],
                            parity_bucket: 0,
                            material_index: 0,
                            render_node_index: 8,
                        },
                        layer: RenderBatchLayer::RetainedRoot,
                    },
                    members: vec![RenderBatchMember {
                        face_index: 0,
                        leaf_id: ScreenPatchLeafId::ROOT,
                        node_index: 7,
                        edge_lods: [1; 3],
                        permutation_index: 0,
                        vertex_lods: [1; 3],
                    }],
                    triangle_index_count: 3,
                    line_index_count: 6,
                    transform,
                    enabled: true,
                    pbr_class: PbrDrawClass::Opaque,
                },
                RenderBatchSnapshot {
                    id: RenderBatchId {
                        key: RenderBatchKey {
                            lod: [1; 3],
                            parity_bucket: 0,
                            material_index: 0,
                            render_node_index: 8,
                        },
                        layer: RenderBatchLayer::AdaptiveOverlay,
                    },
                    members: vec![RenderBatchMember {
                        face_index: 0,
                        leaf_id: ScreenPatchLeafId::ROOT.child(3).ok_or_else(|| {
                            LodWebGpuError::Conformance(
                                "resident overlay conformance child is missing".to_string(),
                            )
                        })?,
                        node_index: 7,
                        edge_lods: [1; 3],
                        permutation_index: 0,
                        vertex_lods: [1; 3],
                    }],
                    triangle_index_count: 3,
                    line_index_count: 6,
                    transform,
                    enabled: true,
                    pbr_class: PbrDrawClass::Opaque,
                },
                RenderBatchSnapshot {
                    id: RenderBatchId {
                        key: RenderBatchKey {
                            lod: [1; 3],
                            parity_bucket: 0,
                            material_index: 1,
                            render_node_index: 9,
                        },
                        layer: RenderBatchLayer::RetainedRoot,
                    },
                    members: vec![RenderBatchMember {
                        face_index: 1,
                        leaf_id: ScreenPatchLeafId::ROOT,
                        node_index: 8,
                        edge_lods: [1; 3],
                        permutation_index: 0,
                        vertex_lods: [1; 3],
                    }],
                    triangle_index_count: 3,
                    line_index_count: 6,
                    transform,
                    enabled: true,
                    pbr_class: PbrDrawClass::Opaque,
                },
            ],
        };
        let faces = [[0u32, 1, 2], [3, 4, 5]];
        let mut source_instances = vec![0.0; 2 * PREPARED_PATCH_RECORD_WORDS];
        for (face, vertices) in faces.into_iter().enumerate() {
            let mut writer = InstanceWriter::new(&mut source_instances, face);
            for (corner, vertex) in vertices.into_iter().enumerate() {
                writer.set_position(
                    corner,
                    vertex,
                    [
                        positions[vertex as usize * 3],
                        positions[vertex as usize * 3 + 1],
                        positions[vertex as usize * 3 + 2],
                    ],
                );
                writer.set_normal(corner, [0.0, 0.0, 1.0]);
            }
            writer.set_face_id(face as u32);
            writer.set_node_id(7 + face as u32);
        }
        let preparation =
            self.upload_resident_root_preparation_scene(&model, &render_scene, &source_instances)?;
        let geometry = self.upload_resident_geometry_bucket_scene(
            &model,
            &packed_atlas,
            &preparation.draw_domains,
        )?;
        let color_format = presentation
            .as_deref()
            .map_or(wgpu::TextureFormat::Rgba8Unorm, |surface| {
                surface.color_format()
            });
        let pipeline = self.create_resident_root_render_pipeline(
            color_format,
            Some(wgpu::TextureFormat::Depth24Plus),
            1,
        )?;
        let overlay_pipelines = self.create_diagnostic_patch_render_pipelines(
            color_format,
            Some(wgpu::TextureFormat::Depth24Plus),
            1,
        )?;
        let overlay_layout_pipeline = overlay_pipelines
            .get(RenderStyle::Normals)
            .expect("diagnostic pipeline family contains normals");
        let mut invalid_overlay_scene = render_scene.clone();
        invalid_overlay_scene
            .batches
            .retain(|batch| batch.id.layer != RenderBatchLayer::AdaptiveOverlay);
        if self
            .upload_adaptive_overlay_scene(
                overlay_layout_pipeline,
                &model,
                &preparation,
                &invalid_overlay_scene,
            )
            .is_ok()
            || !geometry
                .suppressed_faces
                .lock()
                .map_err(|_| {
                    LodWebGpuError::Conformance(
                        "resident suppression conformance lock was poisoned".to_string(),
                    )
                })?
                .is_empty()
        {
            return Err(LodWebGpuError::Conformance(
                "failed adaptive allocation mutated resident root suppression".to_string(),
            ));
        }
        let overlay = self
            .upload_adaptive_overlay_scene(
                overlay_layout_pipeline,
                &model,
                &preparation,
                &render_scene,
            )?
            .ok_or_else(|| {
                LodWebGpuError::Conformance(
                    "resident render conformance lost its adaptive overlay".to_string(),
                )
            })?;
        if !Arc::ptr_eq(
            &overlay.patches.source_faces,
            &preparation.patches.source_faces,
        ) {
            return Err(LodWebGpuError::Conformance(
                "adaptive overlay duplicated the resident source-face buffer".to_string(),
            ));
        }
        self.publish_adaptive_overlay_suppression(&geometry, Some(&overlay))?;
        let foreign_domains = self.upload_resident_root_draw_domain_scene(&model, &render_scene)?;
        let foreign_geometry =
            self.upload_resident_geometry_bucket_scene(&model, &packed_atlas, &foreign_domains)?;
        match self.create_resident_root_render_bindings(&pipeline, &preparation, &foreign_geometry)
        {
            Ok(_) => {
                return Err(LodWebGpuError::Conformance(
                    "resident root render accepted a foreign domain epoch".to_string(),
                ));
            }
            Err(error) if error.to_string().contains("incompatible retained domains") => {}
            Err(error) => return Err(error),
        }
        let bindings =
            self.create_resident_root_render_bindings(&pipeline, &preparation, &geometry)?;
        if bindings.supports_resident_untextured_pbr() {
            return Err(LodWebGpuError::Conformance(
                "resident root PBR accepted placeholder environment residency".to_string(),
            ));
        }

        self.write_lod_classification_state(
            &model,
            &LodDispatchState {
                subjects: Vec::new(),
                baseline_mobius: identity_mobius(),
                baseline_model: identity_matrix(),
                pole: [0.0; 4],
                mobius_power: 0.0,
                c_norm_sq: 0.0,
                has_pole: 0.0,
            },
            WgslLodDispatchMetrics {
                view_projection: identity_matrix(),
                density: 1.0,
                pixel_floor: 0.0,
                max_lod: atlas_lookup.max_lod,
                viewport: [WIDTH as f32, HEIGHT as f32],
                num_joints: 0,
            },
            LodPose::default(),
            PoseUploadPolicy::Publish,
        )?;
        let frame_scene = ValidatedRenderScene::new(render_scene.clone())
            .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
        let frame_options = RenderFrameOptions::default();
        let frame_plan =
            RenderCommandPlan::build(&frame_scene, RenderStyle::MatcapWire, frame_options)
                .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
        let frame = RenderFrame::from_command_plan(
            17,
            RenderPoseIdentity {
                asset_revision: 3,
                pose_revision: 5,
            },
            RenderView {
                viewport: [WIDTH, HEIGHT],
                mvp: identity_matrix(),
                model_view: identity_matrix(),
                camera_position: [0.0, 0.0, 4.0],
                selected_node: None,
                focus: FocusFieldPacket::default(),
            },
            frame_options,
            &frame_plan,
        )
        .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;

        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resident root render conformance target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resident adaptive conformance depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let readback_bytes = u64::from(PADDED_BYTES_PER_ROW) * u64::from(HEIGHT);
        let readback = gpu_buffer(
            &self.device,
            "resident root render conformance readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident root direct render conformance"),
            });
        let encoding = {
            let classification = self.encode_lod_classification(&mut model, &mut encoder)?;
            let resident = self.encode_resident_lod_reconciliation(
                &classification,
                FaceLodGrading::TwoToOne,
                &mut encoder,
            );
            self.encode_resident_adaptive(
                &mut encoder,
                &frame,
                frame_plan.scene().snapshot(),
                classification.model,
                &resident,
                &preparation,
                &geometry,
                &pipeline,
                &bindings,
                &overlay_pipelines,
                Some(&overlay),
                &packed_atlas,
                PatchRenderTarget {
                    color_view: &target_view,
                    resolve_target: None,
                    depth_stencil_view: Some(&depth_view),
                    clear_color: Some(wgpu::Color::TRANSPARENT),
                    clear_depth: Some(1.0),
                },
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )?
        };
        let overlay_encoding = encoding.overlay.ok_or_else(|| {
            LodWebGpuError::Conformance("adaptive overlay was not encoded".to_string())
        })?;
        if encoding.roots
            != (ResidentRootFrameEncoding {
                indirect_draw_calls: 4,
                source_face_count: 2,
            })
            || overlay_encoding
                != (AdaptiveOverlayFrameEncoding {
                    indirect_draw_calls: 2,
                    source_patch_count: 1,
                })
        {
            return Err(LodWebGpuError::Conformance(format!(
                "resident adaptive render encoding mismatch: {encoding:?}",
            )));
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(PADDED_BYTES_PER_ROW),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        if let Some(error) = error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "resident root direct render failed validation: {error}",
            )));
        }
        let pixels = self.readback_words(&readback, readback_bytes).await?;
        let channel_order = match color_format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                RenderImageChannelOrder::Rgba
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                RenderImageChannelOrder::Bgra
            }
            _ => {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident adaptive image evidence requires RGBA8/BGRA8, got {color_format:?}",
                )));
            }
        };
        let image = Rgba8ImageView::new(
            [WIDTH, HEIGHT],
            PADDED_BYTES_PER_ROW as usize,
            RenderImageOrigin::TopLeft,
            channel_order,
            bytemuck::cast_slice(&pixels),
        )
        .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
        let image_signature = render_image_signature(image, 0);
        let rendered = usize::try_from(image_signature.covered_pixels).map_err(|_| {
            LodWebGpuError::Conformance(
                "resident adaptive coverage exceeds address space".to_string(),
            )
        })?;
        if !(32..WIDTH as usize * HEIGHT as usize).contains(&rendered) {
            return Err(LodWebGpuError::Conformance(format!(
                "resident root render produced an implausible {rendered}-pixel footprint",
            )));
        }
        let words_per_row = PADDED_BYTES_PER_ROW as usize / std::mem::size_of::<u32>();
        let covered_pixel = |x_range: std::ops::Range<usize>| {
            pixels
                .chunks_exact(words_per_row)
                .take(HEIGHT as usize)
                .enumerate()
                .find_map(|(y, row)| {
                    x_range
                        .clone()
                        .find(|&x| row[x] != 0)
                        .map(|x| [x as u32, y as u32])
                })
        };
        // The left source root is suppressed and can only be visible through
        // its sparse dyadic replacement. The right source face remains a
        // resident root. Querying one proven rasterized pixel in each half
        // therefore exercises both passes and their shared one-pixel depth.
        let overlay_pixel = covered_pixel(0..WIDTH as usize / 2).ok_or_else(|| {
            LodWebGpuError::Conformance(
                "resident adaptive image has no covered overlay pixel".to_string(),
            )
        })?;
        let root_pixel = covered_pixel(WIDTH as usize / 2..WIDTH as usize).ok_or_else(|| {
            LodWebGpuError::Conformance(
                "resident adaptive image has no covered root pixel".to_string(),
            )
        })?;
        let pick_pipeline = self.create_patch_pick_pipeline(overlay_layout_pipeline)?;
        let root_pick_pipeline =
            self.create_resident_root_pick_pipeline(&pipeline, &pick_pipeline)?;
        let pick_target = self.create_patch_pick_target();
        for (pixel, epoch, expected_face, expected_node) in
            [(overlay_pixel, 92, 0, 7), (root_pixel, 93, 1, 8)]
        {
            let request = PatchPickRequest::new([WIDTH, HEIGHT], pixel, epoch)?;
            let staged = self.stage_resident_adaptive_pick(
                &root_pick_pipeline,
                &pick_pipeline,
                &render_scene,
                &preparation,
                &geometry,
                &bindings,
                Some(&overlay),
                &packed_atlas,
                &pick_target,
                request,
            )?;
            if staged.encoding().indirect_draw_calls != 3 {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident adaptive pick encoded {} indirect draws; expected 3",
                    staged.encoding().indirect_draw_calls,
                )));
            }
            let picked = staged.read().await?.ok_or_else(|| {
                LodWebGpuError::Conformance(format!(
                    "resident adaptive covered pixel {pixel:?} produced no pick",
                ))
            })?;
            if picked.target_epoch != epoch
                || picked.source_face != expected_face
                || picked.packed_node != expected_node
                || (picked.source_barycentric.into_iter().sum::<f32>() - 1.0).abs() > 1.0e-5
                || picked.output_distance <= 0.0
            {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident adaptive pick returned an incoherent sample: {picked:?}",
                )));
            }
        }
        let environment_descriptor = EnvironmentMapDescriptor {
            prefiltered_face_size: 1,
            prefiltered_mip_count: 1,
            irradiance_face_size: 1,
        };
        let environment_error = |error: quilting_core::material::EnvironmentMapAssetError| {
            LodWebGpuError::Conformance(error.to_string())
        };
        let prefiltered = vec![
            0.5;
            environment_descriptor
                .prefiltered_rgba32f_len()
                .map_err(environment_error)?
        ];
        let irradiance = vec![
            0.5;
            environment_descriptor
                .irradiance_rgba32f_len()
                .map_err(environment_error)?
        ];
        let environment = self.upload_pbr_environment_map(
            EnvironmentMapAsset::new(environment_descriptor, &prefiltered, &irradiance)
                .map_err(environment_error)?,
        )?;
        // This conformance transform deliberately reports reversed
        // orientation while retaining identity coordinates to exercise parity
        // bucketing. Make the PBR material double-sided so the raster proof is
        // independent of that intentionally inconsistent fixture metadata.
        let mut untextured_material = PbrMaterial {
            base_color: [0.0, 0.0, 0.0, 1.0],
            double_sided: true,
            ..PbrMaterial::default()
        };
        untextured_material.roughness = 1.0;
        let mut pbr_material = PbrMaterial {
            double_sided: true,
            ..PbrMaterial::default()
        };
        pbr_material.textures.base_color = Some(0);
        let mut pbr_render_scene = render_scene.clone();
        pbr_render_scene.materials = vec![untextured_material, pbr_material];
        let pbr_texture_descriptor = TextureAssetDescriptor {
            width: 1,
            height: 1,
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::Repeat,
        };
        let pbr_texture_pixels = [0, 0, 255, 255];
        let pbr_textures = self.upload_pbr_texture_table(&[Rgba8TextureAsset::new(
            pbr_texture_descriptor,
            &pbr_texture_pixels,
        )
        .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?])?;
        let pbr_bindings = self.create_resident_root_render_bindings_with_pbr(
            &pipeline,
            &preparation,
            &geometry,
            &pbr_render_scene,
            Some(&pbr_textures),
            Some(&environment),
        )?;
        if pbr_bindings.supports_resident_untextured_pbr()
            || !pbr_bindings.supports_resident_basic_pbr()
            || pbr_bindings
                .pbr_texture_residency()
                .is_none_or(|residency| {
                    residency.len() != 2
                        || residency[0].referenced_mask() != 0
                        || residency[1].resident_mask() != pbr_resources::PBR_BASE_COLOR_TEXTURE_BIT
                })
        {
            return Err(LodWebGpuError::Conformance(
                "resident root PBR rejected exact multi-material portable texture residency"
                    .to_string(),
            ));
        }
        let pbr_overlay_pipeline = overlay_pipelines
            .get(RenderStyle::Pbr)
            .expect("diagnostic pipeline family contains PBR");
        let pbr_overlay = self
            .upload_adaptive_overlay_scene_with_pbr_resources(
                pbr_overlay_pipeline,
                &model,
                &preparation,
                &pbr_render_scene,
                Some(&pbr_textures),
                Some(&environment),
            )?
            .ok_or_else(|| {
                LodWebGpuError::Conformance(
                    "resident PBR conformance lost its adaptive overlay".to_string(),
                )
            })?;
        if pbr_overlay.supports_resident_untextured_pbr()
            || !pbr_overlay.supports_resident_basic_pbr()
        {
            return Err(LodWebGpuError::Conformance(
                "adaptive PBR rejected exact material-batched texture residency".to_string(),
            ));
        }
        if !pbr_bindings.supports_resident_root_frame(RenderStyle::Pbr, true)
            || pbr_bindings.supports_resident_root_frame(RenderStyle::Pbr, false)
        {
            return Err(LodWebGpuError::Conformance(
                "resident root PBR overlay capability was not conservative".to_string(),
            ));
        }
        let pbr_frame = RenderFrame::build(
            18,
            RenderPoseIdentity {
                asset_revision: 3,
                pose_revision: 6,
            },
            RenderStyle::Pbr,
            frame.view,
            RenderFrameOptions::default(),
            &pbr_render_scene,
        )
        .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
        let pbr_error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut pbr_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident adaptive PBR render conformance"),
            });
        let pbr_encoding = {
            let classification = self.encode_lod_classification(&mut model, &mut pbr_encoder)?;
            let resident = self.encode_resident_lod_reconciliation(
                &classification,
                FaceLodGrading::TwoToOne,
                &mut pbr_encoder,
            );
            self.encode_resident_adaptive(
                &mut pbr_encoder,
                &pbr_frame,
                &pbr_render_scene,
                classification.model,
                &resident,
                &preparation,
                &geometry,
                &pipeline,
                &pbr_bindings,
                &overlay_pipelines,
                Some(&pbr_overlay),
                &packed_atlas,
                PatchRenderTarget {
                    color_view: &target_view,
                    resolve_target: None,
                    depth_stencil_view: Some(&depth_view),
                    clear_color: Some(wgpu::Color::TRANSPARENT),
                    clear_depth: Some(1.0),
                },
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )?
        };
        if pbr_encoding.roots.source_face_count != 2
            || pbr_encoding.roots.indirect_draw_calls != 2
            || pbr_encoding.overlay
                != Some(AdaptiveOverlayFrameEncoding {
                    indirect_draw_calls: 1,
                    source_patch_count: 1,
                })
        {
            return Err(LodWebGpuError::Conformance(format!(
                "resident adaptive PBR encoding mismatch: {pbr_encoding:?}",
            )));
        }
        pbr_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(PADDED_BYTES_PER_ROW),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([pbr_encoder.finish()]);
        if let Some(error) = pbr_error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "resident adaptive PBR failed validation: {error}",
            )));
        }
        let pbr_pixels = self.readback_words(&readback, readback_bytes).await?;
        let pbr_image = Rgba8ImageView::new(
            [WIDTH, HEIGHT],
            PADDED_BYTES_PER_ROW as usize,
            RenderImageOrigin::TopLeft,
            channel_order,
            bytemuck::cast_slice(&pbr_pixels),
        )
        .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
        let pbr_signature = render_image_signature(pbr_image, 0);
        if pbr_signature.covered_pixels == 0
            || pbr_signature.covered_pixels >= u64::from(WIDTH) * u64::from(HEIGHT)
            || pbr_signature.channel_sums[2] <= pbr_signature.channel_sums[0] * 2
        {
            return Err(LodWebGpuError::Conformance(format!(
                "resident adaptive PBR produced implausible image evidence: {pbr_signature:?}",
            )));
        }
        let focus_root_pipeline = self.create_offscreen_resident_root_render_pipeline()?;
        let focus_overlay_pipeline = self.create_offscreen_focus_pbr_patch_render_pipeline()?;
        let focus_root_bindings = self.create_resident_root_render_bindings_with_pbr(
            &focus_root_pipeline,
            &preparation,
            &geometry,
            &pbr_render_scene,
            Some(&pbr_textures),
            Some(&environment),
        )?;
        let focus_overlay = self
            .upload_focus_adaptive_overlay_scene_with_pbr_resources(
                &focus_overlay_pipeline,
                &model,
                &preparation,
                &pbr_render_scene,
                Some(&pbr_textures),
                Some(&environment),
            )?
            .ok_or_else(|| {
                LodWebGpuError::Conformance(
                    "resident focus conformance lost its adaptive overlay".to_string(),
                )
            })?;
        self.publish_adaptive_overlay_suppression(&geometry, Some(&focus_overlay))?;
        let focus_pipelines =
            self.create_focus_postprocess_pipelines(wgpu::TextureFormat::Rgba8Unorm)?;
        let focus_target =
            self.create_focus_postprocess_target([WIDTH, HEIGHT], &focus_pipelines)?;
        let focus_output = self.create_offscreen_patch_render_target([WIDTH, HEIGHT])?;
        let mut focus_view = pbr_frame.view;
        focus_view.focus = FocusFieldPacket {
            sphere: [0.0, 0.0, 0.0, 1.0],
            enabled: true,
        };
        let focus_frame = RenderFrame::build(
            19,
            pbr_frame.pose,
            RenderStyle::Pbr,
            focus_view,
            RenderFrameOptions {
                focus_postprocess: Some(quilting_core::render::FocusPostprocessPacket {
                    mode: quilting_core::render::FocusPostprocessMode::Spheroidal,
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
            &pbr_render_scene,
        )
        .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
        let focus_error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut focus_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resident adaptive focus render conformance"),
                });
        let focus_encoding = {
            let classification = self.encode_lod_classification(&mut model, &mut focus_encoder)?;
            let resident = self.encode_resident_lod_reconciliation(
                &classification,
                FaceLodGrading::TwoToOne,
                &mut focus_encoder,
            );
            self.encode_focus_resident_adaptive(
                &mut focus_encoder,
                &focus_frame,
                &pbr_render_scene,
                classification.model,
                &resident,
                &preparation,
                &geometry,
                &focus_root_pipeline,
                &focus_root_bindings,
                &focus_overlay_pipeline,
                Some(&focus_overlay),
                &packed_atlas,
                &focus_pipelines,
                &focus_target,
                &focus_output,
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )?
        };
        if focus_encoding.scene.roots.source_face_count != 2
            || focus_encoding.scene.roots.indirect_draw_calls != 2
            || focus_encoding.scene.overlay
                != Some(AdaptiveOverlayFrameEncoding {
                    indirect_draw_calls: 1,
                    source_patch_count: 1,
                })
            || focus_encoding.postprocess.render_passes != 8
        {
            return Err(LodWebGpuError::Conformance(format!(
                "resident adaptive focus encoding mismatch: {focus_encoding:?}",
            )));
        }
        self.queue.submit([focus_encoder.finish()]);
        if let Some(error) = focus_error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "resident adaptive focus failed validation: {error}",
            )));
        }
        let focus_raw = self
            .stage_focus_raw_field_image(&focus_target)?
            .read()
            .await?;
        if focus_raw.covered_texels() == 0 {
            return Err(LodWebGpuError::Conformance(
                "resident adaptive focus raw field has no coverage".to_string(),
            ));
        }
        let focus_image = self
            .stage_offscreen_patch_render_target_image(&focus_output)?
            .read()
            .await?;
        let focus_signature = render_image_signature(focus_image.view()?, 0);
        if focus_signature.covered_pixels == 0 {
            return Err(LodWebGpuError::Conformance(
                "resident adaptive focus composition has no coverage".to_string(),
            ));
        }
        if let Some(surface) = presentation {
            let surface_size = surface.size();
            if surface_size[0] == 0 || surface_size[1] == 0 {
                return Err(LodWebGpuError::Conformance(
                    "resident adaptive presentation surface is suspended".to_string(),
                ));
            }
            self.write_lod_classification_state(
                &model,
                &LodDispatchState {
                    subjects: Vec::new(),
                    baseline_mobius: identity_mobius(),
                    baseline_model: identity_matrix(),
                    pole: [0.0; 4],
                    mobius_power: 0.0,
                    c_norm_sq: 0.0,
                    has_pole: 0.0,
                },
                WgslLodDispatchMetrics {
                    view_projection: identity_matrix(),
                    density: 1.0,
                    pixel_floor: 0.0,
                    max_lod: atlas_lookup.max_lod,
                    viewport: [surface_size[0] as f32, surface_size[1] as f32],
                    num_joints: 0,
                },
                LodPose::default(),
                PoseUploadPolicy::Publish,
            )?;
            let presentation_frame = RenderFrame::build(
                18,
                RenderPoseIdentity {
                    asset_revision: 3,
                    pose_revision: 6,
                },
                RenderStyle::MatcapWire,
                RenderView {
                    viewport: surface_size,
                    mvp: identity_matrix(),
                    model_view: identity_matrix(),
                    camera_position: [0.0, 0.0, 4.0],
                    selected_node: None,
                    focus: FocusFieldPacket::default(),
                },
                RenderFrameOptions::default(),
                &render_scene,
            )
            .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
            let presented = surface.present_with(
                self,
                "resident adaptive browser presentation conformance",
                |encoder, target| {
                    let classification = self.encode_lod_classification(&mut model, encoder)?;
                    let resident = self.encode_resident_lod_reconciliation(
                        &classification,
                        FaceLodGrading::TwoToOne,
                        encoder,
                    );
                    self.encode_resident_adaptive(
                        encoder,
                        &presentation_frame,
                        &render_scene,
                        classification.model,
                        &resident,
                        &preparation,
                        &geometry,
                        &pipeline,
                        &bindings,
                        &overlay_pipelines,
                        Some(&overlay),
                        &packed_atlas,
                        PatchRenderTarget {
                            color_view: target.color_view,
                            resolve_target: target.resolve_target,
                            depth_stencil_view: target.depth_stencil_view,
                            clear_color: Some(wgpu::Color {
                                r: 0.015,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            clear_depth: Some(1.0),
                        },
                        LodPose::default(),
                        0,
                        PoseUploadPolicy::Publish,
                        true,
                    )
                },
            )?;
            let SurfacePresentation::Presented(presented_encoding) = presented else {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident adaptive surface frame was not presented: {presented:?}",
                )));
            };
            if presented_encoding != encoding {
                return Err(LodWebGpuError::Conformance(format!(
                    "offscreen/presentation encoding mismatch: offscreen={encoding:?}, \
                     presentation={presented_encoding:?}",
                )));
            }

            let surface_focus_pipelines =
                self.create_focus_postprocess_pipelines(surface.color_format())?;
            let surface_focus_target =
                self.create_focus_postprocess_target(surface_size, &surface_focus_pipelines)?;
            let mut surface_focus_view = focus_view;
            surface_focus_view.viewport = surface_size;
            let surface_focus_frame = RenderFrame::build(
                20,
                pbr_frame.pose,
                RenderStyle::Pbr,
                surface_focus_view,
                RenderFrameOptions {
                    focus_postprocess: focus_frame.options.focus_postprocess,
                    ..RenderFrameOptions::default()
                },
                &pbr_render_scene,
            )
            .map_err(|error| LodWebGpuError::Conformance(error.to_string()))?;
            let surface_focus_error_scope =
                self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mut surface_focus_prepare =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("resident adaptive focus presentation preparation"),
                    });
            let surface_focus_classification =
                self.encode_lod_classification(&mut model, &mut surface_focus_prepare)?;
            let surface_focus_resident = self.encode_resident_lod_reconciliation(
                &surface_focus_classification,
                FaceLodGrading::TwoToOne,
                &mut surface_focus_prepare,
            );
            self.queue.submit([surface_focus_prepare.finish()]);
            let surface_focus_presented = self.present_focus_resident_adaptive(
                surface,
                &surface_focus_frame,
                &pbr_render_scene,
                surface_focus_classification.model,
                &surface_focus_resident,
                &preparation,
                &geometry,
                &focus_root_pipeline,
                &focus_root_bindings,
                &focus_overlay_pipeline,
                Some(&focus_overlay),
                &packed_atlas,
                &surface_focus_pipelines,
                &surface_focus_target,
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )?;
            let SurfacePresentation::Presented(surface_focus_encoding) = surface_focus_presented
            else {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident adaptive focus surface frame was not presented: \
                     {surface_focus_presented:?}",
                )));
            };
            let expected_surface_focus_encoding = FocusPatchFrameEncoding {
                scene: PatchFrameEncoding {
                    logical_submission: focus_encoding.scene.logical_submission,
                    indirect_draw_calls: focus_encoding
                        .scene
                        .roots
                        .indirect_draw_calls
                        .saturating_add(
                            focus_encoding
                                .scene
                                .overlay
                                .map_or(0, |overlay| overlay.indirect_draw_calls),
                        ),
                    source_instance_count: focus_encoding
                        .scene
                        .roots
                        .source_face_count
                        .saturating_add(
                            focus_encoding
                                .scene
                                .overlay
                                .map_or(0, |overlay| overlay.source_patch_count),
                        ),
                },
                postprocess: focus_encoding.postprocess,
            };
            if surface_focus_encoding != expected_surface_focus_encoding {
                return Err(LodWebGpuError::Conformance(format!(
                    "offscreen/presentation focus encoding mismatch: \
                     offscreen={expected_surface_focus_encoding:?}, \
                     presentation={surface_focus_encoding:?}",
                )));
            }
            if let Some(error) = surface_focus_error_scope.pop().await {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident adaptive focus presentation failed validation: {error}",
                )));
            }
        }
        Ok((
            rendered,
            image_signature,
            encoding.roots.indirect_draw_calls as usize,
            overlay_encoding.source_patch_count as usize,
            overlay_encoding.indirect_draw_calls as usize,
        ))
    }

    async fn run_resident_visibility_conformance(
        &self,
        model: &mut LodClassifierModel,
        atlas: &LodAtlasLookup,
        patches: &PatchPreparationScene,
    ) -> Result<usize, LodWebGpuError> {
        let source_count = patches.patch_count;
        let visibility =
            self.upload_visibility_compaction_scene(WgslVisibilityCompactionSceneWords {
                uniform: [1, source_count, 0, 0],
                batches: vec![[0, source_count, 3, 6]],
                source_eligibility: vec![1; source_count as usize],
            })?;
        let expansion = self.create_face_visibility_expansion_scene(
            model,
            patches,
            &visibility,
            model.prepared.residency.num_faces,
        )?;
        let packed_atlas = self.upload_packed_patch_atlas(
            &[1, 1, 2, 0, 3, 0, 0],
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[0, 1, 2],
            &[],
        )?;
        let draw_domains = self.upload_resident_root_draw_domains(
            model.identity,
            ResidentRootDrawDomains {
                domains: vec![ResidentRootDrawDomain {
                    material_index: 0,
                    render_node_index: 0,
                    pbr_class: PbrDrawClass::Opaque,
                    transform: RenderEntityTransform {
                        mobius: identity_mobius(),
                        orientation_sign: 1,
                        euclidean_model: identity_matrix(),
                        euclidean_normal: identity_matrix(),
                    },
                    enabled: true,
                }],
                face_domain_rows: vec![0; model.prepared.residency.num_faces],
            },
        )?;
        let bucket_scene =
            self.upload_resident_geometry_bucket_scene(model, &packed_atlas, &draw_domains)?;
        let output_bytes = u64::from(source_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Conformance(
                    "resident visibility conformance output is too large".to_string(),
                )
            })?;
        let readback = gpu_buffer(
            &self.device,
            "resident visibility conformance readback",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let range_readback = gpu_buffer(
            &self.device,
            "resident visibility range conformance readback",
            VISIBILITY_RANGE_RECORD_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let indirect_readback = gpu_buffer(
            &self.device,
            "resident visibility indirect conformance readback",
            INDEXED_INDIRECT_RECORD_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let metrics = WgslLodDispatchMetrics {
            view_projection: identity_matrix(),
            density: 1.0,
            pixel_floor: 0.0,
            max_lod: atlas.max_lod,
            viewport: [1024.0, 1024.0],
            num_joints: 0,
        };
        let visible_dispatch = LodDispatchState {
            subjects: Vec::new(),
            baseline_mobius: identity_mobius(),
            baseline_model: identity_matrix(),
            pole: [0.0; 4],
            mobius_power: 0.0,
            c_norm_sq: 0.0,
            has_pole: 0.0,
        };
        let mut hidden_dispatch = visible_dispatch.clone();
        hidden_dispatch.baseline_model = translation_matrix(100.0, 0.0, 0.0);
        let mut compared_words = 0usize;
        for (dispatch, expected) in [(&visible_dispatch, 1u32), (&hidden_dispatch, 0u32)] {
            let classification = self.classify_on_device(
                model,
                dispatch,
                metrics,
                LodPose {
                    joint_matrices: &[],
                    morph_weights: &[0.0],
                },
                PoseUploadPolicy::Publish,
            )?;
            let resident =
                self.reconcile_resident_lod_on_device(&classification, FaceLodGrading::TwoToOne);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resident visibility conformance encoder"),
                });
            self.encode_resident_lod_visibility_expansion(
                &expansion,
                patches.patch_count,
                &mut encoder,
            );
            self.encode_visibility_compaction(&visibility, &mut encoder);
            encoder.copy_buffer_to_buffer(
                &visibility.source_visibility,
                0,
                &readback,
                0,
                output_bytes,
            );
            encoder.copy_buffer_to_buffer(
                &visibility.compacted_ranges,
                0,
                &range_readback,
                0,
                VISIBILITY_RANGE_RECORD_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &visibility.triangle_indirect_arguments,
                0,
                &indirect_readback,
                0,
                INDEXED_INDIRECT_RECORD_BYTES,
            );
            self.queue.submit([encoder.finish()]);
            let actual = self.readback_words(&readback, output_bytes).await?;
            let expected_words = vec![expected; source_count as usize];
            if actual != expected_words {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident visibility expansion mismatch: expected {expected_words:?}, got {actual:?}",
                )));
            }
            let survivor_count = expected.saturating_mul(source_count);
            let range = self
                .readback_words(&range_readback, VISIBILITY_RANGE_RECORD_BYTES)
                .await?;
            let expected_range = [0, 0, source_count, 0, survivor_count];
            if range != expected_range {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident visibility compacted range mismatch: expected {expected_range:?}, got {range:?}",
                )));
            }
            let indirect = self
                .readback_words(&indirect_readback, INDEXED_INDIRECT_RECORD_BYTES)
                .await?;
            let expected_indirect = [3, survivor_count, 0, 0, 0];
            if indirect != expected_indirect {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident visibility indirect arguments mismatch: expected {expected_indirect:?}, got {indirect:?}",
                )));
            }
            let bucketed = self
                .resident_geometry_buckets_for_diagnostics(&bucket_scene, &resident, &[])
                .await?;
            let expected_faces = if expected == 0 { vec![] } else { vec![0] };
            if bucketed.compacted_faces != expected_faces
                || bucketed
                    .triangle_indirect_arguments
                    .iter()
                    .map(|arguments| arguments[1])
                    .sum::<u32>()
                    != expected
            {
                return Err(LodWebGpuError::Conformance(format!(
                    "classifier-to-root-bucket mismatch: expected visibility {expected}, got {bucketed:?}",
                )));
            }
            compared_words += actual.len();
        }
        Ok(compared_words)
    }

    /// Run the same minimum exact matrix on native and browser devices.
    ///
    /// This covers one complete animated-capable two-pass dispatch; exact 2:1
    /// and 4:1 resident closure across a maximum-range ten-face chain; all S3
    /// permutations; visible-only neighbor promotion; invisible records;
    /// priorities; multiple atlas keys in the coherence pass; and exact
    /// source-stable root bucketing across three 64-face chunks.
    pub async fn run_conformance_matrix(&self) -> Result<LodDeviceConformance, LodWebGpuError> {
        self.run_conformance_matrix_impl(None).await
    }

    /// Run the exact matrix and direct its composed resident/adaptive case to
    /// a real presentation surface using the same pipelines and device state.
    pub async fn run_conformance_matrix_with_surface(
        &self,
        presentation: &mut PatchPresentationSurface,
    ) -> Result<LodDeviceConformance, LodWebGpuError> {
        self.run_conformance_matrix_impl(Some(presentation)).await
    }

    async fn run_conformance_matrix_impl(
        &self,
        presentation: Option<&mut PatchPresentationSurface>,
    ) -> Result<LodDeviceConformance, LodWebGpuError> {
        let prepared = prepare_lod_model(LodModelData {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![[0, 1, 2]],
            joint_indices: vec![[0; 4]; 3],
            joint_weights: vec![[0.0; 4]; 3],
            morph_deltas: Vec::new(),
            num_morph_targets: 0,
            face_nodes: vec![0],
        })
        .map_err(LodWebGpuError::Payload)?;
        let atlas = prepare_lod_atlas_lookup([[1, 1, 2]]).map_err(LodWebGpuError::Payload)?;
        let model_words = pack_wgsl_lod_model_words(&prepared).map_err(LodWebGpuError::Payload)?;
        let atlas_words = pack_wgsl_lod_atlas_words(&atlas);
        let expected = reconcile_and_pack_wgsl_lod_pass2(
            &[[1.0, 0.0, 0.0, 1.0]],
            &model_words.adjacency,
            &atlas_words,
        )
        .map_err(LodWebGpuError::Payload)?;
        let dispatch = LodDispatchState {
            subjects: Vec::new(),
            baseline_mobius: identity_mobius(),
            baseline_model: identity_matrix(),
            pole: [0.0; 4],
            mobius_power: 0.0,
            c_norm_sq: 0.0,
            has_pole: 0.0,
        };
        let metrics = WgslLodDispatchMetrics {
            view_projection: identity_matrix(),
            density: 1.0,
            pixel_floor: 0.0,
            max_lod: atlas.max_lod,
            viewport: [1024.0, 1024.0],
            num_joints: 2,
        };
        let mut joint_matrices = Vec::with_capacity(32);
        joint_matrices.extend_from_slice(&identity_matrix());
        joint_matrices.extend_from_slice(&identity_matrix());
        let mut resident = self.upload_model(prepared, &atlas)?;
        let actual = {
            let output = self.classify_on_device(
                &mut resident,
                &dispatch,
                metrics,
                LodPose {
                    joint_matrices: &joint_matrices,
                    morph_weights: &[],
                },
                PoseUploadPolicy::Publish,
            )?;
            if output.face_count() != 1 || output.epoch() != 1 {
                return Err(LodWebGpuError::Conformance(format!(
                    "device-resident classification identity mismatch: faces={}, epoch={}",
                    output.face_count(),
                    output.epoch(),
                )));
            }
            let actual = self
                .read_lod_classification_for_diagnostics(&output)
                .await?;
            for grading in [FaceLodGrading::TwoToOne, FaceLodGrading::FourToOne] {
                let resident = self.reconcile_resident_lod_on_device(&output, grading);
                if resident.face_count() != output.face_count()
                    || resident.classification_epoch() != output.epoch()
                    || resident.grading() != grading
                {
                    return Err(LodWebGpuError::Conformance(
                        "device-resident LOD reconciliation identity mismatch".to_string(),
                    ));
                }
                let resident_words = self.read_resident_lod_for_diagnostics(&resident).await?;
                if resident_words != expected {
                    return Err(LodWebGpuError::Conformance(format!(
                        "resident {grading:?} mismatch: expected {expected:?}, got {resident_words:?}",
                    )));
                }
            }
            actual
        };
        if actual != expected {
            return Err(LodWebGpuError::Conformance(format!(
                "full pipeline mismatch: expected {expected:?}, got {actual:?}"
            )));
        }
        let next_output = self.classify_on_device(
            &mut resident,
            &dispatch,
            metrics,
            LodPose {
                joint_matrices: &joint_matrices,
                morph_weights: &[],
            },
            PoseUploadPolicy::Publish,
        )?;
        if next_output.epoch() != 2 {
            return Err(LodWebGpuError::Conformance(format!(
                "device-resident classification epoch did not advance: {}",
                next_output.epoch(),
            )));
        }
        let (
            resident_lod_words,
            resident_root_topology_words,
            resident_root_prepared_words,
            resident_root_domain_words,
        ) = self.run_resident_lod_conformance().await?;
        let (
            resident_adaptive_rendered_pixels,
            resident_adaptive_image,
            resident_root_indirect_draws,
            adaptive_overlay_patches,
            adaptive_overlay_indirect_draws,
        ) = self
            .validate_resident_adaptive_render_conformance(presentation)
            .await?;
        let resident_bucket_words = self.run_resident_geometry_bucket_conformance().await?;
        let full_pipeline_words = actual.len();

        let pass1 = [
            [1.0, 2.0, 3.0, 11.0],
            [1.0, 3.0, 2.0, 12.0],
            [2.0, 1.0, 3.0, 13.0],
            [3.0, 1.0, 2.0, 14.0],
            [2.0, 3.0, 1.0, 15.0],
            [3.0, 2.0, 1.0, 16.0],
            [4.0, 5.0, 6.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 4.0, 2.0],
            [8.0, 8.0, 8.0, 0.0],
        ];
        let mut adjacency = vec![[u32::MAX, 0, 0, 0]; pass1.len() * 3];
        adjacency[7 * 3] = [8, 2, 0, 0];
        adjacency[7 * 3 + 1] = [9, 0, 0, 0];
        let mut atlas_lut = vec![u8::MAX as u32; 1_200];
        atlas_lut[321] = 17;
        atlas_lut[411] = 29;
        atlas_lut[654] = 31;
        let expected = reconcile_and_pack_wgsl_lod_pass2(&pass1, &adjacency, &atlas_lut)
            .map_err(LodWebGpuError::Payload)?;
        let actual = self
            .reconcile_conformance_records(&pass1, &adjacency, &atlas_lut)
            .await?;
        if actual != expected {
            return Err(LodWebGpuError::Conformance(format!(
                "coherence mismatch: expected {expected:?}, got {actual:?}"
            )));
        }

        let patch_prepared = prepare_lod_model(LodModelData {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![[0, 1, 2]],
            joint_indices: vec![[0; 4]; 3],
            joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
            morph_deltas: vec![0.25, 0.5, 0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.0],
            num_morph_targets: 1,
            face_nodes: vec![0],
        })
        .map_err(LodWebGpuError::Payload)?;
        let mut patch_model = self.upload_model(patch_prepared, &atlas)?;
        let (patch_words, expected_patches) = patch_preparation_conformance_words();
        let patch_scene = self.upload_patch_preparation_scene(&patch_model, patch_words)?;
        let patch_joint = translation_matrix(1.0, -0.5, 2.0);
        let prepared_patches = self
            .prepare_patches(
                &patch_model,
                &patch_scene,
                LodPose {
                    joint_matrices: &patch_joint,
                    morph_weights: &[1.0],
                },
                1,
            )
            .await?;
        if prepared_patches != expected_patches {
            return Err(LodWebGpuError::Conformance(format!(
                "patch preparation mismatch: expected {expected_patches:?}, got \
                 {prepared_patches:?}"
            )));
        }
        let prepared_patch_words = prepared_patches.len() * PREPARED_PATCH_RECORD_WORDS;
        let resident_visibility_words = self
            .run_resident_visibility_conformance(&mut patch_model, &atlas, &patch_scene)
            .await?;
        let (rendered_patch_pixels, shared_frame_draws) = self
            .validate_patch_render_conformance(
                &patch_model,
                &patch_scene,
                LodPose {
                    joint_matrices: &patch_joint,
                    morph_weights: &[1.0],
                },
                1,
            )
            .await?;

        let batch_zero_count = 130u32;
        let mut source_visibility = Vec::with_capacity(137);
        source_visibility.extend((0..batch_zero_count).map(|index| u8::from(index % 3 != 0)));
        source_visibility.extend([1, 1, 1, 1, 0, 1, 1]);
        let compaction_words = WgslVisibilityCompactionSceneWords {
            uniform: [3, 137, 0, 0],
            batches: vec![[0, 130, 6, 8], [130, 3, 12, 16], [133, 4, 18, 24]],
            source_eligibility: [vec![1; 130], vec![0; 3], vec![1; 4]].concat(),
        };
        let mut expected_sources = (0..batch_zero_count)
            .filter(|index| index % 3 != 0)
            .collect::<Vec<_>>();
        expected_sources.extend([133, 135, 136]);
        let expected_ranges = vec![[0, 0, 130, 0, 86], [1, 130, 3, 86, 0], [2, 133, 4, 86, 3]];
        let expected_indirect = vec![[6, 86, 0, 0, 0], [12, 0, 0, 0, 0], [18, 3, 0, 0, 0]];
        let expected_line_indirect = vec![[8, 86, 0, 0, 0], [16, 0, 0, 0, 0], [24, 3, 0, 0, 0]];
        let mut compaction = self.upload_visibility_compaction_scene(compaction_words)?;
        let compacted = self
            .compact_visibility(&mut compaction, &source_visibility)
            .await?;
        if compacted.compacted_source_instances != expected_sources
            || compacted.compacted_ranges != expected_ranges
            || compacted.triangle_indirect_arguments != expected_indirect
            || compacted.line_indirect_arguments != expected_line_indirect
        {
            return Err(LodWebGpuError::Conformance(format!(
                "visibility compaction mismatch: expected sources {expected_sources:?}, ranges \
                 {expected_ranges:?}, triangle indirect {expected_indirect:?}, line indirect \
                 {expected_line_indirect:?}; got {compacted:?}"
            )));
        }
        let indirect_draws = self
            .validate_indirect_draw_conformance(&compaction, &source_visibility)
            .await?;

        Ok(LodDeviceConformance {
            full_pipeline_words,
            resident_lod_words,
            resident_visibility_words,
            resident_bucket_words,
            resident_root_topology_words,
            resident_root_prepared_words,
            resident_root_domain_words,
            resident_adaptive_rendered_pixels,
            resident_adaptive_image,
            resident_root_indirect_draws,
            adaptive_overlay_patches,
            adaptive_overlay_indirect_draws,
            coherence_words: actual.len(),
            prepared_patch_words,
            rendered_patch_pixels,
            shared_frame_draws,
            compacted_source_words: compacted.compacted_source_instances.len(),
            compacted_range_words: compacted.compacted_ranges.len() * 5,
            indirect_argument_words: (compacted.triangle_indirect_arguments.len()
                + compacted.line_indirect_arguments.len())
                * 5,
            indirect_draws,
        })
    }

    /// Upload immutable source faces plus retained scene topology and affine
    /// transforms. The output order matches visibility compaction's stable
    /// source-instance order and can be consumed directly by a render vertex
    /// stage through [`PatchPreparationScene::prepared_records_buffer`].
    pub fn upload_patch_preparation_scene(
        &self,
        model: &LodClassifierModel,
        words: WgslPatchPreparationSceneWords,
    ) -> Result<PatchPreparationScene, LodWebGpuError> {
        let patch_count = words.uniform[0];
        let num_morph_targets = u32::try_from(model.prepared.model.num_morph_targets)
            .map_err(|_| LodWebGpuError::Payload("patch morph target count exceeds u32".into()))?;
        if words.uniform[1] != model.prepared.residency.num_vertices
            || words.uniform[2] != 0
            || words.uniform[3] != num_morph_targets
            || words.topology.len() != patch_count as usize
            || words.source_faces.len() != model.prepared.residency.num_faces
            || words.subjects.is_empty()
        {
            return Err(LodWebGpuError::Payload(
                "patch preparation scene shape is malformed".to_string(),
            ));
        }
        for (patch_index, topology) in words.topology.iter().enumerate() {
            let float_words: [f32; 10] = std::array::from_fn(|word| f32::from_bits(topology[word]));
            let face = float_words[4];
            let leaf_depth = float_words[8];
            let leaf_path = float_words[9];
            let integral_nonnegative = |value: f32| value >= 0.0 && value.fract() == 0.0;
            if float_words.iter().any(|value| !value.is_finite())
                || float_words[..8]
                    .iter()
                    .any(|&value| !integral_nonnegative(value))
                || float_words[3] > 5.0
                || face as usize >= words.source_faces.len()
                || !integral_nonnegative(leaf_depth)
                || leaf_depth > 12.0
                || !integral_nonnegative(leaf_path)
                || leaf_path >= (1u32 << (2 * leaf_depth as u32)) as f32
                || topology[10] as usize >= words.subjects.len()
                || topology[11] != 0
            {
                return Err(LodWebGpuError::Payload(format!(
                    "patch preparation topology {patch_index} is malformed",
                )));
            }
        }
        for (face_index, (source, expected_vertices)) in words
            .source_faces
            .iter()
            .zip(&model.prepared.model.faces)
            .enumerate()
        {
            for (corner, &expected_vertex) in expected_vertices.iter().enumerate() {
                let encoded_vertex = f32::from_bits(source[corner * 4]);
                if !encoded_vertex.is_finite()
                    || encoded_vertex < 0.0
                    || encoded_vertex.fract() != 0.0
                    || encoded_vertex as u32 != expected_vertex
                {
                    return Err(LodWebGpuError::Payload(format!(
                        "patch preparation source face {face_index} corner {corner} does not match the resident model",
                    )));
                }
            }
        }
        if words
            .source_faces
            .iter()
            .flatten()
            .chain(words.subjects.iter().flatten())
            .copied()
            .map(f32::from_bits)
            .any(|value| !value.is_finite())
        {
            return Err(LodWebGpuError::Payload(
                "patch preparation source contains non-finite values".to_string(),
            ));
        }

        debug_assert_eq!(
            std::mem::size_of_val(&words.uniform) as u64,
            PATCH_PREPARE_UNIFORM_BYTES,
        );
        debug_assert!(words
            .topology
            .iter()
            .all(|record| { std::mem::size_of_val(record) as u64 == PATCH_TOPOLOGY_RECORD_BYTES }));
        debug_assert!(words
            .source_faces
            .iter()
            .all(|record| { std::mem::size_of_val(record) as u64 == PREPARED_PATCH_RECORD_BYTES }));
        debug_assert!(words
            .subjects
            .iter()
            .all(|record| { std::mem::size_of_val(record) as u64 == PATCH_SUBJECT_RECORD_BYTES }));

        let inert_topology = [[0u32; 12]];
        let topology = buffer_init_or_zero(
            &self.device,
            "patch preparation topology",
            if words.topology.is_empty() {
                bytemuck::cast_slice(&inert_topology)
            } else {
                bytemuck::cast_slice(&words.topology)
            },
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.allocate_patch_preparation_scene(
            model,
            words.uniform,
            topology,
            &words.source_faces,
            &words.subjects,
        )
    }

    fn allocate_patch_preparation_scene(
        &self,
        model: &LodClassifierModel,
        uniform_words: [u32; 4],
        topology: wgpu::Buffer,
        source_face_words: &[[u32; PREPARED_PATCH_RECORD_WORDS]],
        subject_words: &[[u32; PATCH_SUBJECT_RECORD_WORDS]],
    ) -> Result<PatchPreparationScene, LodWebGpuError> {
        let source_faces = Arc::new(buffer_init_or_zero(
            &self.device,
            "patch preparation source faces",
            bytemuck::cast_slice(source_face_words),
            wgpu::BufferUsages::STORAGE,
        ));
        self.allocate_patch_preparation_scene_with_source(
            model,
            uniform_words,
            topology,
            source_faces,
            subject_words,
        )
    }

    fn allocate_patch_preparation_scene_with_source(
        &self,
        model: &LodClassifierModel,
        uniform_words: [u32; 4],
        topology: wgpu::Buffer,
        source_faces: Arc<wgpu::Buffer>,
        subject_words: &[[u32; PATCH_SUBJECT_RECORD_WORDS]],
    ) -> Result<PatchPreparationScene, LodWebGpuError> {
        let patch_count = uniform_words[0];
        let subject_count = u32::try_from(subject_words.len())
            .map_err(|_| LodWebGpuError::Payload("patch subject count exceeds u32".into()))?;
        let uniform = buffer_init_or_zero(
            &self.device,
            "patch preparation uniform",
            bytemuck::cast_slice(&uniform_words),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let subjects = buffer_init_or_zero(
            &self.device,
            "patch preparation subjects",
            bytemuck::cast_slice(subject_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let prepared_bytes = u64::from(patch_count)
            .checked_mul(PREPARED_PATCH_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("prepared patch buffer is too large".to_string())
            })?;
        let prepared_records = gpu_buffer(
            &self.device,
            "prepared patch records",
            prepared_bytes.max(PREPARED_PATCH_RECORD_BYTES),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let layout = self.patch_prepare_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("patch preparation bindings"),
            layout: &layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &topology),
                bind(2, source_faces.as_ref()),
                bind(3, &model.skinning),
                bind(4, &model.joint_matrices),
                bind(5, &model.morph_deltas),
                bind(6, &model.morph_weights),
                bind(7, &subjects),
                bind(8, &prepared_records),
            ],
        });
        Ok(PatchPreparationScene {
            patch_count,
            subject_count,
            uniform_words,
            uniform,
            topology,
            source_faces,
            subjects,
            prepared_records,
            bind_group,
        })
    }

    fn write_dynamic_pose(
        &self,
        model: &LodClassifierModel,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<(), LodWebGpuError> {
        let num_joints = num_joints as usize;
        if pose.joint_matrices.len() != num_joints.saturating_mul(16) {
            return Err(LodWebGpuError::Payload(
                "joint pose does not match the dispatch joint count".to_string(),
            ));
        }
        if pose.morph_weights.len() != model.prepared.model.num_morph_targets {
            return Err(LodWebGpuError::Payload(
                "morph weights do not match immutable targets".to_string(),
            ));
        }
        let resident_joint_floats = num_joints.min(model.joint_capacity) * 16;
        if resident_joint_floats != 0 {
            self.queue.write_buffer(
                &model.joint_matrices,
                0,
                bytemuck::cast_slice(&pose.joint_matrices[..resident_joint_floats]),
            );
        }
        if !pose.morph_weights.is_empty() {
            self.queue.write_buffer(
                &model.morph_weights,
                0,
                bytemuck::cast_slice(pose.morph_weights),
            );
        }
        Ok(())
    }

    /// Upload current pose state and the dynamic joint-count word. Applications
    /// may then encode preparation and all subsequent passes in one command
    /// encoder without any CPU-visible copy or map.
    pub fn write_patch_pose(
        &self,
        model: &LodClassifierModel,
        scene: &PatchPreparationScene,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<(), LodWebGpuError> {
        self.write_dynamic_pose(model, pose, num_joints)?;
        self.write_patch_joint_count(scene, num_joints);
        Ok(())
    }

    fn write_patch_joint_count(&self, scene: &PatchPreparationScene, num_joints: u32) {
        let mut uniform_words = scene.uniform_words;
        uniform_words[2] = num_joints;
        self.queue
            .write_buffer(&scene.uniform, 0, bytemuck::cast_slice(&uniform_words));
    }

    pub fn encode_patch_preparation(
        &self,
        scene: &PatchPreparationScene,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if scene.patch_count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quilting patch preparation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.patch_prepare_pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.dispatch_workgroups(scene.patch_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
    }

    /// No-readback convenience submission for an authoritative backend.
    pub fn prepare_patches_on_device(
        &self,
        model: &LodClassifierModel,
        scene: &PatchPreparationScene,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<(), LodWebGpuError> {
        self.write_patch_pose(model, scene, pose, num_joints)?;
        if scene.patch_count == 0 {
            return Ok(());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting patch preparation"),
            });
        self.encode_patch_preparation(scene, &mut encoder);
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Diagnostic wrapper over the same encoder path. The temporary staging
    /// buffer exists only for native/browser parity gates.
    pub async fn prepare_patches(
        &self,
        model: &LodClassifierModel,
        scene: &PatchPreparationScene,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<Vec<[u32; PREPARED_PATCH_RECORD_WORDS]>, LodWebGpuError> {
        self.write_patch_pose(model, scene, pose, num_joints)?;
        if scene.patch_count == 0 {
            return Ok(Vec::new());
        }
        let prepared_bytes = u64::from(scene.patch_count) * PREPARED_PATCH_RECORD_BYTES;
        let readback = gpu_buffer(
            &self.device,
            "prepared patch diagnostic readback",
            prepared_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting patch preparation diagnostic"),
            });
        self.encode_patch_preparation(scene, &mut encoder);
        encoder.copy_buffer_to_buffer(&scene.prepared_records, 0, &readback, 0, prepared_bytes);
        self.queue.submit([encoder.finish()]);
        words_to_patch_records(self.readback_words(&readback, prepared_bytes).await?)
    }

    /// Create retained normals-mode QB graphics pipelines for one attachment
    /// configuration. Both winding variants share an explicit layout, making
    /// one scene bind group valid for every batch.
    pub fn create_patch_render_pipeline(
        &self,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Result<PatchRenderPipeline, LodWebGpuError> {
        self.create_diagnostic_patch_render_pipeline(
            RenderStyle::Normals,
            color_format,
            depth_format,
            sample_count,
        )
    }

    /// Create one single-pass pipeline over the shared prepared QB vertex
    /// path. Non-opaque PBR and composite styles remain explicit rather than
    /// silently selecting the wrong resources or draw sequence.
    pub fn create_diagnostic_patch_render_pipeline(
        &self,
        style: RenderStyle,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Result<PatchRenderPipeline, LodWebGpuError> {
        self.create_diagnostic_patch_render_pipelines_for(
            &[style],
            color_format,
            depth_format,
            sample_count,
            false,
            None,
        )?
        .0
        .pop()
        .ok_or_else(|| LodWebGpuError::Payload("diagnostic pipeline set was empty".to_string()))
    }

    /// Create the complete currently supported diagnostic pipeline family.
    /// Every member borrows the same binding-layout identity, so a live scene
    /// changes style by selecting a pipeline—not by republishing resources.
    pub fn create_diagnostic_patch_render_pipelines(
        &self,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Result<DiagnosticPatchRenderPipelines, LodWebGpuError> {
        let (mut pipelines, highlight) = self.create_diagnostic_patch_render_pipelines_for(
            &[
                RenderStyle::Pbr,
                RenderStyle::Normals,
                RenderStyle::Matcap,
                RenderStyle::Lod,
                RenderStyle::Stretch,
                RenderStyle::Wire,
            ],
            color_format,
            depth_format,
            sample_count,
            true,
            None,
        )?;
        let highlight = highlight.expect("requested highlight pipeline");
        let wire = pipelines.pop().expect("requested wire pipeline");
        let stretch = pipelines.pop().expect("requested stretch pipeline");
        let lod = pipelines.pop().expect("requested LOD pipeline");
        let matcap = pipelines.pop().expect("requested matcap pipeline");
        let normals = pipelines.pop().expect("requested normals pipeline");
        let pbr = pipelines.pop().expect("requested PBR pipeline");
        Ok(DiagnosticPatchRenderPipelines {
            pbr,
            normals,
            matcap,
            lod,
            stretch,
            wire,
            highlight,
        })
    }

    fn create_diagnostic_patch_render_pipelines_for(
        &self,
        styles: &[RenderStyle],
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
        include_highlight: bool,
        pbr_raw_field_format: Option<wgpu::TextureFormat>,
    ) -> Result<(Vec<PatchRenderPipeline>, Option<PatchRenderPipeline>), LodWebGpuError> {
        let functional_color_format =
            crate::pipeline_lowering::functional_texture_format(color_format);
        let functional_depth_format = depth_format.map_or(Some(None), |format| {
            crate::pipeline_lowering::functional_texture_format(format).map(Some)
        });
        let functional_raw_field_format = pbr_raw_field_format.map_or(Some(None), |format| {
            crate::pipeline_lowering::functional_texture_format(format).map(Some)
        });
        if let (
            Some(functional_color_format),
            Some(functional_depth_format),
            Some(functional_raw_field_format),
        ) = (
            functional_color_format,
            functional_depth_format,
            functional_raw_field_format,
        ) {
            let descriptors = prepared_patch_pipeline_descriptors(
                styles,
                functional_color_format,
                functional_depth_format,
                sample_count,
                include_highlight,
                functional_raw_field_format,
            )
            .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
            let mut pipelines = self.prepared_patch_render_pipelines.lock().map_err(|_| {
                LodWebGpuError::Payload(
                    "WebGPU prepared-patch render pipeline memo was poisoned".to_string(),
                )
            })?;
            let family = pipelines.get_or_try_insert_with(descriptors, |descriptors| {
                self.build_diagnostic_patch_render_pipelines_for(
                    styles,
                    color_format,
                    depth_format,
                    sample_count,
                    include_highlight,
                    pbr_raw_field_format,
                    Some(descriptors.as_slice()),
                )
            })?;
            return Ok(family.clone());
        }
        self.build_diagnostic_patch_render_pipelines_for(
            styles,
            color_format,
            depth_format,
            sample_count,
            include_highlight,
            pbr_raw_field_format,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_diagnostic_patch_render_pipelines_for(
        &self,
        styles: &[RenderStyle],
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
        include_highlight: bool,
        pbr_raw_field_format: Option<wgpu::TextureFormat>,
        descriptors: Option<&[RenderPipelineDescriptor]>,
    ) -> Result<(Vec<PatchRenderPipeline>, Option<PatchRenderPipeline>), LodWebGpuError> {
        if sample_count == 0 || !sample_count.is_power_of_two() {
            return Err(LodWebGpuError::Payload(
                "patch render sample count must be a nonzero power of two".to_string(),
            ));
        }
        let passes = prepared_patch_pipeline::prepared_patch_pipeline_passes(
            styles,
            include_highlight,
            pbr_raw_field_format.is_some(),
        )
        .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        let expected_descriptor_count = passes
            .iter()
            .copied()
            .map(prepared_patch_pipeline::PreparedPatchPipelinePass::descriptor_count)
            .sum::<usize>();
        if descriptors.is_some_and(|descriptors| descriptors.len() != expected_descriptor_count) {
            return Err(LodWebGpuError::Payload(format!(
                "functional prepared-patch family must contain {expected_descriptor_count} variants",
            )));
        }
        let module = self.memoized_render_shader_module(
            "quilting prepared QB render",
            quilting_shaders::sources::PATCH_RENDER_DEVICE,
            quilting_shaders::PATCH_RENDER_DEVICE_VERTEX_ENTRY_POINT,
            quilting_shaders::compile_patch_render_device_wgsl,
        )?;
        let uses_pbr_family = passes
            .iter()
            .copied()
            .any(prepared_patch_pipeline::PreparedPatchPipelinePass::uses_pbr);
        let (bind_group_layout, pbr_texture_bind_group_layout, pbr_environment_bind_group_layout) =
            if let Some(descriptors) = descriptors {
                let first_layout = descriptors
                    .first()
                    .ok_or_else(|| {
                        LodWebGpuError::Payload(
                            "functional prepared-patch pipeline family is empty".to_string(),
                        )
                    })?
                    .layout();
                let root_layout = first_layout.groups().first().ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "functional prepared-patch pipeline has no root bind group".to_string(),
                    )
                })?;
                if descriptors
                    .iter()
                    .any(|descriptor| descriptor.layout().groups().first() != Some(root_layout))
                {
                    return Err(LodWebGpuError::Payload(
                        "functional prepared-patch root layouts are inconsistent".to_string(),
                    ));
                }
                let pbr_layout = if uses_pbr_family {
                    Some(
                        descriptors
                            .iter()
                            .find_map(|descriptor| {
                                let entry = descriptor.program().fragment()?.entry_point();
                                matches!(
                                    entry,
                                    quilting_shaders::PATCH_RENDER_DEVICE_PBR_ENTRY_POINT
                                        | quilting_shaders::PATCH_RENDER_DEVICE_PBR_FOCUS_ENTRY_POINT
                                )
                                .then_some(descriptor.layout())
                            })
                            .ok_or_else(|| {
                                LodWebGpuError::Payload(
                                    "functional prepared-patch PBR layout is missing".to_string(),
                                )
                            })?,
                    )
                } else {
                    None
                };
                if descriptors.iter().any(|descriptor| {
                    let Some(fragment) = descriptor.program().fragment() else {
                        return true;
                    };
                    let uses_pbr = matches!(
                        fragment.entry_point(),
                        quilting_shaders::PATCH_RENDER_DEVICE_PBR_ENTRY_POINT
                            | quilting_shaders::PATCH_RENDER_DEVICE_PBR_FOCUS_ENTRY_POINT
                    );
                    if uses_pbr {
                        descriptor.layout().groups().len() != 3
                            || Some(descriptor.layout()) != pbr_layout
                    } else {
                        descriptor.layout().groups().len() != 1
                    }
                }) {
                    return Err(LodWebGpuError::Payload(
                        "functional prepared-patch pipeline layouts are inconsistent".to_string(),
                    ));
                }
                (
                    crate::pipeline_lowering::bind_group_layout(
                        &self.device,
                        "quilting prepared QB render bindings",
                        root_layout,
                    ),
                    pbr_layout.map(|layout| {
                        crate::pipeline_lowering::bind_group_layout(
                            &self.device,
                            "quilting PBR material texture bindings",
                            &layout.groups()[1],
                        )
                    }),
                    pbr_layout.map(|layout| {
                        crate::pipeline_lowering::bind_group_layout(
                            &self.device,
                            "quilting PBR environment binding layout",
                            &layout.groups()[2],
                        )
                    }),
                )
            } else {
                let bind_group_layout =
                    self.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("quilting prepared QB render bindings"),
                            entries: &[
                                render_buffer_layout_visible(
                                    0,
                                    wgpu::BufferBindingType::Storage { read_only: true },
                                    PATCH_RENDER_GLOBAL_BYTES,
                                    false,
                                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ),
                                render_buffer_layout_visible(
                                    1,
                                    wgpu::BufferBindingType::Storage { read_only: true },
                                    PATCH_RENDER_DOMAIN_BYTES,
                                    false,
                                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ),
                                render_buffer_layout_visible(
                                    2,
                                    wgpu::BufferBindingType::Storage { read_only: true },
                                    PREPARED_PATCH_RECORD_BYTES,
                                    false,
                                    wgpu::ShaderStages::VERTEX,
                                ),
                                render_buffer_layout_visible(
                                    3,
                                    wgpu::BufferBindingType::Storage { read_only: true },
                                    PACKED_RECORD_BYTES,
                                    false,
                                    wgpu::ShaderStages::VERTEX,
                                ),
                                render_buffer_layout_visible(
                                    4,
                                    wgpu::BufferBindingType::Storage { read_only: true },
                                    VISIBILITY_RANGE_RECORD_BYTES,
                                    false,
                                    wgpu::ShaderStages::VERTEX,
                                ),
                                render_buffer_layout_visible(
                                    5,
                                    wgpu::BufferBindingType::Uniform,
                                    DRAW_BATCH_INDEX_BYTES,
                                    true,
                                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                                ),
                                render_buffer_layout_visible(
                                    6,
                                    wgpu::BufferBindingType::Storage { read_only: true },
                                    PATCH_PBR_MATERIAL_BYTES,
                                    false,
                                    wgpu::ShaderStages::FRAGMENT,
                                ),
                            ],
                        });
                (
                    bind_group_layout,
                    uses_pbr_family
                        .then(|| pbr_resources::create_pbr_texture_bind_group_layout(&self.device)),
                    uses_pbr_family.then(|| {
                        pbr_environment::create_pbr_environment_bind_group_layout(&self.device)
                    }),
                )
            };
        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let mut pipelines = Vec::with_capacity(passes.len());
        let mut descriptor_index = 0usize;
        for pass in passes {
            let kind = pass.kind;
            let geometry = pass.geometry;
            let fragment_entry_point = pass.fragment_entry_point;
            let uses_pbr_bindings = pass.uses_pbr();
            let pipeline_depth_stencil = depth_stencil.clone().map(|mut state| {
                state.depth_write_enabled = Some(pass.depth_write);
                state
            });
            let pipeline_layout = if uses_pbr_bindings {
                let pbr_texture_bind_group_layout = pbr_texture_bind_group_layout
                    .as_ref()
                    .expect("PBR pipeline request creates its texture layout");
                let pbr_environment_bind_group_layout = pbr_environment_bind_group_layout
                    .as_ref()
                    .expect("PBR pipeline request creates its environment layout");
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("quilting prepared QB PBR pipeline layout"),
                        bind_group_layouts: &[
                            Some(&bind_group_layout),
                            Some(pbr_texture_bind_group_layout),
                            Some(pbr_environment_bind_group_layout),
                        ],
                        immediate_size: 0,
                    })
            } else {
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("quilting prepared QB diagnostic pipeline layout"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    })
            };
            let create = |index: usize,
                          front_face: wgpu::FrontFace|
             -> Result<wgpu::RenderPipeline, LodWebGpuError> {
                if let Some(descriptors) = descriptors {
                    let descriptor = &descriptors[index];
                    let fragment = descriptor.program().fragment().ok_or_else(|| {
                        LodWebGpuError::Payload(
                            "functional prepared-patch pipeline is missing a fragment stage"
                                .to_string(),
                        )
                    })?;
                    let expected_topology = match geometry {
                        RenderGeometry::Triangles => functional::PrimitiveTopology::TriangleList,
                        RenderGeometry::Lines => functional::PrimitiveTopology::LineList,
                    };
                    let expected_front_face = match front_face {
                        wgpu::FrontFace::Ccw => functional::FrontFace::CounterClockwise,
                        wgpu::FrontFace::Cw => functional::FrontFace::Clockwise,
                    };
                    if fragment.entry_point() != fragment_entry_point
                        || descriptor.primitive().topology != expected_topology
                        || descriptor.primitive().front_face != expected_front_face
                    {
                        return Err(LodWebGpuError::Payload(
                            "functional prepared-patch pipeline order is inconsistent".to_string(),
                        ));
                    }
                    return crate::pipeline_lowering::render_pipeline(
                        &self.device,
                        "quilting prepared QB diagnostic",
                        &pipeline_layout,
                        &module,
                        descriptor,
                    );
                }
                let mut targets = vec![Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })];
                if uses_pbr_bindings {
                    if let Some(format) = pbr_raw_field_format {
                        targets.push(Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }));
                    }
                }
                Ok(self
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("quilting prepared QB diagnostic"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &module,
                            entry_point: Some(
                                quilting_shaders::PATCH_RENDER_DEVICE_VERTEX_ENTRY_POINT,
                            ),
                            compilation_options: Default::default(),
                            buffers: &[wgpu::VertexBufferLayout {
                                array_stride: 12,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &attributes,
                            }],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: match geometry {
                                RenderGeometry::Triangles => wgpu::PrimitiveTopology::TriangleList,
                                RenderGeometry::Lines => wgpu::PrimitiveTopology::LineList,
                            },
                            front_face,
                            cull_mode: None,
                            ..Default::default()
                        },
                        depth_stencil: pipeline_depth_stencil.clone(),
                        multisample: wgpu::MultisampleState {
                            count: sample_count,
                            ..Default::default()
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &module,
                            entry_point: Some(fragment_entry_point),
                            compilation_options: Default::default(),
                            targets: &targets,
                        }),
                        multiview_mask: None,
                        cache: None,
                    }))
            };
            let counter_clockwise = create(descriptor_index, wgpu::FrontFace::Ccw)?;
            descriptor_index += 1;
            let clockwise = match geometry {
                RenderGeometry::Triangles => {
                    let pipeline = create(descriptor_index, wgpu::FrontFace::Cw)?;
                    descriptor_index += 1;
                    pipeline
                }
                RenderGeometry::Lines => counter_clockwise.clone(),
            };
            pipelines.push(PatchRenderPipeline {
                kind,
                geometry,
                bind_group_layout: bind_group_layout.clone(),
                pbr_texture_bind_group_layout: uses_pbr_bindings.then(|| {
                    pbr_texture_bind_group_layout
                        .as_ref()
                        .expect("PBR pipeline request creates its texture layout")
                        .clone()
                }),
                pbr_environment_bind_group_layout: uses_pbr_bindings.then(|| {
                    pbr_environment_bind_group_layout
                        .as_ref()
                        .expect("PBR pipeline request creates its environment layout")
                        .clone()
                }),
                counter_clockwise,
                clockwise,
            });
        }
        let highlight =
            include_highlight.then(|| pipelines.pop().expect("requested highlight pipeline"));
        Ok((pipelines, highlight))
    }

    /// Create the focus-only PBR MRT pipeline. The first attachment receives
    /// scene color; the second receives the raw stretch/depth/spheroidal
    /// payload consumed by [`FocusPostprocessPipelines`].
    pub fn create_focus_pbr_patch_render_pipeline(
        &self,
        color_format: wgpu::TextureFormat,
        raw_field_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Result<FocusPbrPatchRenderPipeline, LodWebGpuError> {
        let (mut pipelines, highlight) = self.create_diagnostic_patch_render_pipelines_for(
            &[RenderStyle::Pbr],
            color_format,
            depth_format,
            sample_count,
            false,
            Some(raw_field_format),
        )?;
        debug_assert!(highlight.is_none());
        let inner = pipelines.pop().ok_or_else(|| {
            LodWebGpuError::Payload("focus PBR pipeline set was empty".to_string())
        })?;
        Ok(FocusPbrPatchRenderPipeline {
            inner,
            color_format,
            raw_field_format,
        })
    }

    pub fn create_offscreen_focus_pbr_patch_render_pipeline(
        &self,
    ) -> Result<FocusPbrPatchRenderPipeline, LodWebGpuError> {
        self.create_focus_pbr_patch_render_pipeline(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba16Float,
            Some(wgpu::TextureFormat::Depth24Plus),
            1,
        )
    }

    /// Build one retained focus render graph for the requested final output
    /// format. No partially constructed family is published on failure.
    pub fn create_focus_pbr_render_resources(
        &self,
        output_format: wgpu::TextureFormat,
    ) -> Result<FocusPbrRenderResources, LodWebGpuError> {
        Ok(FocusPbrRenderResources {
            root_pipeline: self.create_offscreen_resident_root_render_pipeline()?,
            overlay_pipeline: self.create_offscreen_focus_pbr_patch_render_pipeline()?,
            postprocess_pipelines: self.create_focus_postprocess_pipelines(output_format)?,
            target: None,
        })
    }

    pub fn create_offscreen_focus_pbr_render_resources(
        &self,
    ) -> Result<FocusPbrRenderResources, LodWebGpuError> {
        self.create_focus_pbr_render_resources(wgpu::TextureFormat::Rgba8Unorm)
    }

    /// Create the fixed-format live shadow pipeline without leaking backend
    /// texture enums into the application adapter.
    pub fn create_offscreen_patch_render_pipeline(
        &self,
    ) -> Result<PatchRenderPipeline, LodWebGpuError> {
        self.create_patch_render_pipeline(
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth24Plus),
            1,
        )
    }

    /// Create the fixed-format offscreen diagnostic family without exposing
    /// backend texture enums to application adapters.
    pub fn create_offscreen_diagnostic_patch_render_pipelines(
        &self,
    ) -> Result<DiagnosticPatchRenderPipelines, LodWebGpuError> {
        self.create_diagnostic_patch_render_pipelines(
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth24Plus),
            1,
        )
    }

    pub fn create_offscreen_patch_render_target(
        &self,
        size: [u32; 2],
    ) -> Result<OffscreenPatchRenderTarget, LodWebGpuError> {
        if size[0] == 0 || size[1] == 0 {
            return Err(LodWebGpuError::Payload(
                "offscreen patch target dimensions must be nonzero".to_string(),
            ));
        }
        let limit = self.device.limits().max_texture_dimension_2d;
        if size[0] > limit || size[1] > limit {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen patch target {}x{} exceeds device limit {limit}",
                size[0], size[1],
            )));
        }
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let color = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("quilting live shadow color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("quilting live shadow depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Ok(OffscreenPatchRenderTarget {
            color_view: color.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_view: depth.create_view(&wgpu::TextureViewDescriptor::default()),
            color,
            _depth: depth,
            size,
        })
    }

    /// Stage an explicit parity/diagnostic image from the most recently
    /// submitted offscreen frame. Ordinary rendering never calls this method.
    pub fn stage_offscreen_patch_render_target_image(
        &self,
        target: &OffscreenPatchRenderTarget,
    ) -> Result<StagedOffscreenImageReadback, LodWebGpuError> {
        let unpadded_bytes_per_row = target.size[0].checked_mul(4).ok_or_else(|| {
            LodWebGpuError::Payload("offscreen image row size overflowed".to_string())
        })?;
        let bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| {
                LodWebGpuError::Payload("offscreen image padded row size overflowed".to_string())
            })?;
        let byte_len = u64::from(bytes_per_row)
            .checked_mul(u64::from(target.size[1]))
            .ok_or_else(|| {
                LodWebGpuError::Payload("offscreen image readback size overflowed".to_string())
            })?;
        let buffer = gpu_buffer(
            &self.device,
            "quilting offscreen image evidence readback",
            byte_len,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting offscreen image evidence copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(target.size[1]),
                },
            },
            wgpu::Extent3d {
                width: target.size[0],
                height: target.size[1],
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        Ok(StagedOffscreenImageReadback {
            #[cfg(not(target_arch = "wasm32"))]
            device: self.device.clone(),
            buffer,
            size: target.size,
            bytes_per_row: bytes_per_row as usize,
            byte_len,
        })
    }

    /// Bind the output of preparation and stable visibility compaction
    /// directly. Empty scenes need no render bindings and are rejected here.
    pub fn create_patch_render_bindings(
        &self,
        pipeline: &PatchRenderPipeline,
        scene: &RenderSceneSnapshot,
        patches: &PatchPreparationScene,
        visibility: &VisibilityCompactionScene,
        textures: Option<&PbrTextureTable>,
    ) -> Result<PatchRenderBindings, LodWebGpuError> {
        self.create_patch_render_bindings_with_environment(
            pipeline, scene, patches, visibility, textures, None, None,
        )
    }

    fn create_patch_render_global_residency(
        &self,
        label: &'static str,
    ) -> Arc<PatchRenderGlobalResidency> {
        Arc::new(PatchRenderGlobalResidency {
            buffer: gpu_buffer(
                &self.device,
                label,
                PATCH_RENDER_GLOBAL_BYTES,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            table: Mutex::new(RetainedFrameTable::new(
                PATCH_RENDER_GLOBAL_WORDS,
                PATCH_RENDER_GLOBAL_WORDS,
                PATCH_RENDER_GLOBAL_BYTES,
            )),
        })
    }

    pub(crate) fn create_patch_render_bindings_with_environment(
        &self,
        pipeline: &PatchRenderPipeline,
        scene: &RenderSceneSnapshot,
        patches: &PatchPreparationScene,
        visibility: &VisibilityCompactionScene,
        textures: Option<&PbrTextureTable>,
        environment: Option<&PbrEnvironmentMap>,
        shared_global_frame: Option<Arc<PatchRenderGlobalResidency>>,
    ) -> Result<PatchRenderBindings, LodWebGpuError> {
        if patches.patch_count == 0
            || visibility.source_count == 0
            || visibility.batch_count == 0
            || patches.patch_count != visibility.source_count
        {
            return Err(LodWebGpuError::Payload(
                "patch render bindings require one nonempty shared source-instance domain"
                    .to_string(),
            ));
        }
        let domain_bytes = u64::from(visibility.batch_count)
            .checked_mul(PATCH_RENDER_DOMAIN_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("patch render domain table is too large".to_string())
            })?;
        let domain_word_count = usize::try_from(visibility.batch_count)
            .ok()
            .and_then(|count| count.checked_mul(PATCH_RENDER_DOMAIN_WORDS))
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "patch render domain staging table exceeds address space".to_string(),
                )
            })?;
        let global_frame = shared_global_frame.unwrap_or_else(|| {
            self.create_patch_render_global_residency("patch render global frame")
        });
        let domains = gpu_buffer(
            &self.device,
            "patch render domain table",
            domain_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let material_words = patch_pbr_material_table_words(&scene.materials)?;
        let material_count = u32::try_from(scene.materials.len().max(1))
            .map_err(|_| LodWebGpuError::Payload("PBR material count exceeds u32".to_string()))?;
        let materials = buffer_init_or_zero(
            &self.device,
            "patch authored PBR material table",
            bytemuck::cast_slice(&material_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let material_textures = if pipeline.uses_pbr_bindings() {
            Some(self.create_pbr_material_texture_bindings(pipeline, &scene.materials, textures)?)
        } else {
            None
        };
        let pbr_environment = if pipeline.uses_pbr_bindings() {
            Some(self.create_pbr_environment_bindings(pipeline, environment)?)
        } else {
            None
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting prepared QB render bindings"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                bind(0, &global_frame.buffer),
                bind(1, &domains),
                bind(2, &patches.prepared_records),
                bind(3, &visibility.compacted_source_instances),
                bind(4, &visibility.compacted_ranges),
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &visibility.batch_index_uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(DRAW_BATCH_INDEX_BYTES),
                    }),
                },
                bind(6, &materials),
            ],
        });
        Ok(PatchRenderBindings {
            domain_count: visibility.batch_count,
            material_count,
            global_frame,
            domains,
            materials,
            material_textures,
            pbr_environment,
            domain_table: Mutex::new(RetainedFrameTable::new(
                domain_word_count,
                PATCH_RENDER_DOMAIN_WORDS,
                domain_bytes,
            )),
            bind_group,
        })
    }

    fn create_face_visibility_expansion_scene(
        &self,
        model: &LodClassifierModel,
        patches: &PatchPreparationScene,
        visibility: &VisibilityCompactionScene,
        face_count: usize,
    ) -> Result<FaceVisibilityExpansionScene, LodWebGpuError> {
        let face_count = u32::try_from(face_count)
            .map_err(|_| LodWebGpuError::Payload("face visibility count exceeds u32".into()))?;
        let word_count = face_count.div_ceil(32);
        let uniform_words = [patches.patch_count, face_count, word_count, 0];
        debug_assert_eq!(
            std::mem::size_of_val(&uniform_words) as u64,
            FACE_VISIBILITY_UNIFORM_BYTES,
        );
        let uniform = buffer_init_or_zero(
            &self.device,
            "face visibility expansion uniform",
            bytemuck::cast_slice(&uniform_words),
            wgpu::BufferUsages::UNIFORM,
        );
        let bit_bytes = u64::from(word_count).saturating_mul(PACKED_RECORD_BYTES);
        let bits = gpu_buffer(
            &self.device,
            "packed face visibility bits",
            bit_bytes.max(PACKED_RECORD_BYTES),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let layout = self.visibility_expand_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("face visibility expansion bindings"),
            layout: &layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &patches.topology),
                bind(2, &bits),
                bind(3, &visibility.source_visibility),
            ],
        });
        let lod_layout = self.lod_visibility_expand_pipeline.get_bind_group_layout(0);
        let lod_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident LOD visibility expansion bindings"),
            layout: &lod_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &patches.topology),
                bind(2, &model.resident.packed_records),
                bind(3, &visibility.source_visibility),
            ],
        });
        Ok(FaceVisibilityExpansionScene {
            face_count,
            word_count,
            bits,
            bind_group,
            lod_bind_group,
        })
    }

    /// Upload every immutable resource derived from one backend-neutral scene
    /// as a single replacement candidate. Construction failure leaves the
    /// caller's previous aggregate untouched.
    pub fn upload_patch_render_scene(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        mut scene: RenderSceneSnapshot,
        source_instances: &[f32],
        revision: u64,
        textures: Option<&PbrTextureTable>,
    ) -> Result<PatchRenderScene, LodWebGpuError> {
        scene.revision = revision;
        let scene = ValidatedRenderScene::new(scene).map_err(|error| {
            LodWebGpuError::Payload(format!("retained WebGPU render scene: {error}"))
        })?;
        self.upload_validated_patch_render_scene(
            pipeline,
            model,
            scene,
            source_instances,
            textures,
        )
    }

    /// Upload one scene that has already crossed the backend-neutral semantic
    /// validation boundary. The retained WebGPU aggregate shares the exact
    /// scene allocation with extraction, command planning, and parity
    /// observers instead of cloning and revalidating its member tables.
    pub fn upload_validated_patch_render_scene(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        scene: ValidatedRenderScene,
        source_instances: &[f32],
        textures: Option<&PbrTextureTable>,
    ) -> Result<PatchRenderScene, LodWebGpuError> {
        let snapshot = scene.snapshot();
        let patch_words =
            pack_wgsl_patch_preparation_scene_words(&model.prepared, snapshot, source_instances)
                .map_err(LodWebGpuError::Payload)?;
        let visibility_words =
            pack_wgsl_visibility_compaction_scene_words(snapshot)
                .map_err(LodWebGpuError::Payload)?;
        let patches = self.upload_patch_preparation_scene(model, patch_words)?;
        let visibility = self.upload_visibility_compaction_scene(visibility_words)?;
        let face_visibility = self.create_face_visibility_expansion_scene(
            model,
            &patches,
            &visibility,
            model.prepared.residency.num_faces,
        )?;
        let bindings = self.create_patch_render_bindings(
            pipeline,
            snapshot,
            &patches,
            &visibility,
            textures,
        )?;
        Ok(PatchRenderScene {
            model_identity: model.identity,
            scene,
            patches,
            visibility,
            face_visibility,
            bindings,
        })
    }

    /// Focus-PBR counterpart to [`Self::upload_patch_render_scene`]. The
    /// wrapper keeps the two-attachment pipeline type distinct while reusing
    /// the same retained semantic scene and binding machinery.
    pub fn upload_focus_pbr_patch_render_scene(
        &self,
        pipeline: &FocusPbrPatchRenderPipeline,
        model: &LodClassifierModel,
        scene: RenderSceneSnapshot,
        source_instances: &[f32],
        revision: u64,
        textures: Option<&PbrTextureTable>,
    ) -> Result<PatchRenderScene, LodWebGpuError> {
        self.upload_patch_render_scene(
            &pipeline.inner,
            model,
            scene,
            source_instances,
            revision,
            textures,
        )
    }

    /// Publish a new extracted topology into an existing scene allocation when
    /// its patch, subject, and batch cardinalities are unchanged. All queue
    /// writes precede the next frame encoder on the same device queue; callers
    /// retain full aggregate replacement as the shape-change fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn update_patch_render_scene_in_place(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        retained: &mut PatchRenderScene,
        mut scene: RenderSceneSnapshot,
        source_instances: &[f32],
        revision: u64,
        textures: Option<&PbrTextureTable>,
    ) -> Result<PatchRenderSceneUpdate, LodWebGpuError> {
        scene.revision = revision;
        let scene = ValidatedRenderScene::new(scene).map_err(|error| {
            LodWebGpuError::Payload(format!("retained WebGPU render scene update: {error}"))
        })?;
        self.update_validated_patch_render_scene_in_place(
            pipeline,
            model,
            retained,
            scene,
            source_instances,
            textures,
        )
    }

    /// Publish an already validated scene into retained allocations when its
    /// device shape is unchanged. Shape changes return the same validated
    /// allocation so the caller can atomically upload a replacement without
    /// cloning or revalidating the snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn update_validated_patch_render_scene_in_place(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        retained: &mut PatchRenderScene,
        scene: ValidatedRenderScene,
        source_instances: &[f32],
        textures: Option<&PbrTextureTable>,
    ) -> Result<PatchRenderSceneUpdate, LodWebGpuError> {
        let snapshot = scene.snapshot();
        let patch_words =
            pack_wgsl_patch_preparation_scene_words(&model.prepared, snapshot, source_instances)
                .map_err(LodWebGpuError::Payload)?;
        let visibility_words =
            pack_wgsl_visibility_compaction_scene_words(snapshot).map_err(LodWebGpuError::Payload)?;
        let material_words = patch_pbr_material_table_words(&snapshot.materials)?;
        let same_shape = model.identity == retained.model_identity
            && patch_words.uniform[0] == retained.patches.patch_count
            && patch_words.topology.len() == retained.patches.patch_count as usize
            && patch_words.subjects.len() == retained.patches.subject_count as usize
            && visibility_words.uniform[0] == retained.visibility.batch_count
            && visibility_words.uniform[1] == retained.visibility.source_count
            && visibility_words.batches.len() == retained.visibility.batch_count as usize
            && visibility_words.source_eligibility.len()
                == retained.visibility.source_count as usize
            && snapshot.materials.len().max(1) == retained.bindings.material_count as usize;
        if !same_shape {
            return Ok(PatchRenderSceneUpdate::ShapeChanged(scene));
        }
        if patch_words.uniform[1..] != retained.patches.uniform_words[1..]
            || visibility_words.uniform[2..] != [0, 0]
        {
            return Err(LodWebGpuError::Payload(
                "in-place WebGPU scene update changed immutable model words".to_string(),
            ));
        }
        let material_textures = if pipeline.uses_pbr_bindings() {
            Some(self.create_pbr_material_texture_bindings(
                pipeline,
                &scene.snapshot().materials,
                textures,
            )?)
        } else {
            None
        };

        if !patch_words.topology.is_empty() {
            self.queue.write_buffer(
                &retained.patches.topology,
                0,
                bytemuck::cast_slice(&patch_words.topology),
            );
        }
        self.queue.write_buffer(
            &retained.patches.subjects,
            0,
            bytemuck::cast_slice(&patch_words.subjects),
        );
        if !visibility_words.batches.is_empty() {
            self.queue.write_buffer(
                &retained.visibility.batches,
                0,
                bytemuck::cast_slice(&visibility_words.batches),
            );
        }
        if !visibility_words.source_eligibility.is_empty() {
            self.queue.write_buffer(
                &retained.visibility.source_eligibility,
                0,
                bytemuck::cast_slice(&visibility_words.source_eligibility),
            );
        }
        self.queue.write_buffer(
            &retained.bindings.materials,
            0,
            bytemuck::cast_slice(&material_words),
        );
        if let Some(material_textures) = material_textures {
            retained.bindings.material_textures = Some(material_textures);
        }
        retained.scene = scene;
        Ok(PatchRenderSceneUpdate::Updated)
    }

    /// Atomically refresh only the material-to-texture bind groups after an
    /// independently decoded texture table is published. Frame, topology, and
    /// material buffers retain their allocations.
    pub fn replace_patch_render_scene_texture_bindings(
        &self,
        pipeline: &PatchRenderPipeline,
        retained: &mut PatchRenderScene,
        textures: Option<&PbrTextureTable>,
    ) -> Result<(), LodWebGpuError> {
        let candidate = self.create_pbr_material_texture_bindings(
            pipeline,
            &retained.scene.snapshot().materials,
            textures,
        )?;
        if candidate.material_count() != retained.bindings.material_count {
            return Err(LodWebGpuError::Payload(
                "PBR texture binding material count does not match retained scene".to_string(),
            ));
        }
        retained.bindings.material_textures = Some(candidate);
        Ok(())
    }

    /// Atomically refresh only the PBR image-based-lighting bind group. A
    /// missing environment deliberately publishes the analytical fallback
    /// binding instead of leaving group two unbound.
    pub fn replace_patch_render_scene_environment_bindings(
        &self,
        pipeline: &PatchRenderPipeline,
        retained: &mut PatchRenderScene,
        environment: Option<&PbrEnvironmentMap>,
    ) -> Result<(), LodWebGpuError> {
        retained.bindings.pbr_environment =
            Some(self.create_pbr_environment_bindings(pipeline, environment)?);
        Ok(())
    }

    /// Upload current pose and visibility inputs for one coherent retained
    /// scene. These queue writes precede the caller's subsequent encoder
    /// submission on the same queue.
    pub fn write_patch_render_scene_state(
        &self,
        model: &LodClassifierModel,
        scene: &PatchRenderScene,
        pose: LodPose<'_>,
        num_joints: u32,
        source_visibility: &[u8],
    ) -> Result<(), LodWebGpuError> {
        self.write_patch_render_pose_state(
            model,
            scene,
            pose,
            num_joints,
            PoseUploadPolicy::Publish,
        )?;
        self.write_source_visibility(&scene.visibility, source_visibility)
    }

    pub fn write_patch_render_pose_state(
        &self,
        model: &LodClassifierModel,
        scene: &PatchRenderScene,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
    ) -> Result<(), LodWebGpuError> {
        if model.identity != scene.model_identity {
            return Err(LodWebGpuError::Payload(
                "patch render scene belongs to a different WebGPU model".to_string(),
            ));
        }
        if pose_upload.should_publish_dynamic() {
            self.write_dynamic_pose(model, pose, num_joints)?;
        }
        if pose_upload.should_publish_preparation() {
            self.write_patch_joint_count(&scene.patches, num_joints);
        }
        Ok(())
    }

    /// Upload a compact face-indexed visibility bitset. Expansion into current
    /// flattened patch order occurs on-device after patch preparation.
    pub fn write_patch_render_face_visibility_bits(
        &self,
        scene: &PatchRenderScene,
        words: &[u32],
    ) -> Result<(), LodWebGpuError> {
        let expected = scene.face_visibility.word_count as usize;
        if words.len() != expected {
            return Err(LodWebGpuError::Payload(format!(
                "face visibility bitset has {} words; expected {expected}",
                words.len(),
            )));
        }
        let tail_bits = scene.face_visibility.face_count % 32;
        if tail_bits != 0
            && words
                .last()
                .is_some_and(|word| word & !((1u32 << tail_bits) - 1) != 0)
        {
            return Err(LodWebGpuError::Payload(
                "face visibility bitset has nonzero padding".to_string(),
            ));
        }
        if !words.is_empty() {
            self.queue
                .write_buffer(&scene.face_visibility.bits, 0, bytemuck::cast_slice(words));
        }
        Ok(())
    }

    fn encode_face_visibility_expansion(
        &self,
        scene: &PatchRenderScene,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if scene.patches.patch_count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quilting face visibility expansion"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.visibility_expand_pipeline);
        pass.set_bind_group(0, &scene.face_visibility.bind_group, &[]);
        pass.dispatch_workgroups(scene.patches.patch_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
    }

    fn encode_resident_lod_visibility_expansion(
        &self,
        scene: &FaceVisibilityExpansionScene,
        patch_count: u32,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if patch_count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quilting resident LOD visibility expansion"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.lod_visibility_expand_pipeline);
        pass.set_bind_group(0, &scene.lod_bind_group, &[]);
        pass.dispatch_workgroups(patch_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
    }

    /// Project the final device-resident classifier visibility into the
    /// retained scene's current flattened patch order. The scene and result
    /// must originate from the same uploaded model; this prevents a bind group
    /// retained for one model from silently consuming another model's face
    /// domain. The resulting storage buffer feeds visibility compaction
    /// directly and is never mapped in production.
    pub fn encode_patch_render_resident_lod_visibility(
        &self,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), LodWebGpuError> {
        if scene.model_identity != resident.model_identity {
            return Err(LodWebGpuError::Payload(
                "resident LOD visibility belongs to a different WebGPU model".to_string(),
            ));
        }
        if scene.face_visibility.face_count != resident.face_count {
            return Err(LodWebGpuError::Payload(format!(
                "resident LOD visibility has {} faces; render scene expects {}",
                resident.face_count, scene.face_visibility.face_count,
            )));
        }
        self.encode_resident_lod_visibility_expansion(
            &scene.face_visibility,
            scene.patches.patch_count,
            encoder,
        );
        Ok(())
    }

    /// Diagnostic-only exact readback for validating the compact visibility
    /// adapter. Production rendering encodes the same pass directly into
    /// compaction and never calls this method.
    pub async fn expand_face_visibility_for_diagnostics(
        &self,
        scene: &PatchRenderScene,
        words: &[u32],
    ) -> Result<Vec<u32>, LodWebGpuError> {
        self.write_patch_render_face_visibility_bits(scene, words)?;
        let output_bytes = u64::from(scene.visibility.source_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("expanded visibility readback is too large".to_string())
            })?;
        if output_bytes == 0 {
            return Ok(Vec::new());
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expanded face visibility diagnostic readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("expanded face visibility diagnostic encoder"),
            });
        self.encode_face_visibility_expansion(scene, &mut encoder);
        encoder.copy_buffer_to_buffer(
            &scene.visibility.source_visibility,
            0,
            &readback,
            0,
            output_bytes,
        );
        self.queue.submit([encoder.finish()]);
        self.readback_words(&readback, output_bytes).await
    }

    /// Diagnostic-only readback of the direct resident-LOD visibility adapter.
    /// Production rendering calls
    /// [`Self::encode_patch_render_resident_lod_visibility`] inside its frame
    /// encoder and leaves the result device-resident for compaction.
    pub async fn expand_resident_lod_visibility_for_diagnostics(
        &self,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let output_bytes = u64::from(scene.visibility.source_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("resident LOD visibility readback is too large".to_string())
            })?;
        if output_bytes == 0 {
            return Ok(Vec::new());
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident LOD visibility diagnostic readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident LOD visibility diagnostic encoder"),
            });
        self.encode_patch_render_resident_lod_visibility(scene, resident, &mut encoder)?;
        encoder.copy_buffer_to_buffer(
            &scene.visibility.source_visibility,
            0,
            &readback,
            0,
            output_bytes,
        );
        self.queue.submit([encoder.finish()]);
        self.readback_words(&readback, output_bytes).await
    }

    /// Encode one supported single-pass diagnostic frame from a coherent scene
    /// aggregate and the retained packed atlas.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_diagnostic_patch_render_scene<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        pipeline: &'resource PatchRenderPipeline,
        scene: &'resource PatchRenderScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        self.encode_diagnostic_render_frame(
            encoder,
            frame,
            scene.scene.snapshot(),
            pipeline,
            &scene.bindings,
            &scene.patches,
            &scene.visibility,
            atlas,
            target,
            use_qb,
            encode_visibility,
        )
    }

    /// Encode a supported single- or multi-pass diagnostic frame from one
    /// coherent retained scene aggregate.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_supported_patch_render_scene<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        pipelines: &'resource DiagnosticPatchRenderPipelines,
        scene: &'resource PatchRenderScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        self.encode_supported_render_frame(
            encoder,
            frame,
            scene.scene.snapshot(),
            pipelines,
            &scene.bindings,
            &scene.patches,
            &scene.visibility,
            atlas,
            target,
            use_qb,
            encode_visibility,
        )
    }

    /// Compatibility wrapper for the first promoted diagnostic mode.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_normals_patch_render_scene<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        pipeline: &'resource PatchRenderPipeline,
        scene: &'resource PatchRenderScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        require_normals_frame_pipeline(frame, pipeline)?;
        self.encode_diagnostic_patch_render_scene(
            encoder,
            frame,
            pipeline,
            scene,
            atlas,
            target,
            use_qb,
            encode_visibility,
        )
    }

    /// Submit one complete offscreen normals frame with no CPU readback. Pose
    /// and visibility queue writes must already have been issued through
    /// [`Self::write_patch_render_scene_state`].
    pub fn render_offscreen_normals_patch_scene(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        require_normals_frame_pipeline(frame, pipeline)?;
        self.render_offscreen_diagnostic_patch_scene_impl(
            frame,
            pipeline,
            scene,
            atlas,
            target,
            use_qb,
            false,
            wgpu::Color::TRANSPARENT,
        )
    }

    /// Encode one complete offscreen focus frame: PBR scene/raw MRT followed
    /// by the retained Rust-scheduled focus composition. All resources remain
    /// on the same device and no readback is introduced.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_focus_pbr_patch_render_scene<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        pipeline: &'resource FocusPbrPatchRenderPipeline,
        focus_pipelines: &FocusPostprocessPipelines,
        scene: &'resource PatchRenderScene,
        atlas: &'resource PackedPatchAtlas,
        focus_target: &FocusPostprocessTarget,
        output_target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<FocusPatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        self.encode_focus_pbr_patch_render_scene_impl(
            encoder,
            frame,
            pipeline,
            focus_pipelines,
            scene,
            atlas,
            focus_target,
            output_target,
            use_qb,
            encode_visibility,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_focus_pbr_patch_render_scene_impl<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        pipeline: &'resource FocusPbrPatchRenderPipeline,
        focus_pipelines: &FocusPostprocessPipelines,
        scene: &'resource PatchRenderScene,
        atlas: &'resource PackedPatchAtlas,
        focus_target: &FocusPostprocessTarget,
        output_target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<FocusPatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        if frame.style != RenderStyle::Pbr {
            return Err(LodWebGpuError::Payload(format!(
                "focus WebGPU pipeline cannot render {:?} frame",
                frame.style,
            )));
        }
        if frame.view.viewport != focus_target.size() || frame.view.viewport != output_target.size {
            return Err(LodWebGpuError::Payload(format!(
                "focus/output targets {:?}/{:?} do not match frame viewport {:?}",
                focus_target.size(),
                output_target.size,
                frame.view.viewport,
            )));
        }
        if pipeline.color_format != focus_target.scene_color_format()
            || pipeline.raw_field_format != focus_target.raw_field_format()
        {
            return Err(LodWebGpuError::Payload(
                "focus PBR pipeline formats do not match retained target".to_string(),
            ));
        }
        let packet = frame.options.focus_postprocess.ok_or_else(|| {
            LodWebGpuError::Payload(
                "focus WebGPU frame has no postprocess packet after validation".to_string(),
            )
        })?;
        let scene_encoding = self.encode_render_frame_with_pipelines(
            encoder,
            frame,
            scene.scene.snapshot(),
            &scene.bindings,
            &scene.patches,
            &scene.visibility,
            atlas,
            PatchRenderTarget {
                color_view: focus_target.scene_color_view(),
                resolve_target: None,
                depth_stencil_view: Some(&output_target.depth_view),
                clear_color: Some(wgpu::Color::TRANSPARENT),
                clear_depth: Some(1.0),
            },
            Some((
                focus_target.raw_field_view(),
                Some(wgpu::Color::TRANSPARENT),
            )),
            use_qb,
            None,
            |pass, geometry| {
                if pass == RenderPass::PbrOpaque && geometry == RenderGeometry::Triangles {
                    Ok(&pipeline.inner)
                } else {
                    Err(LodWebGpuError::Payload(format!(
                        "focus WebGPU frame cannot lower {pass:?}/{geometry:?}",
                    )))
                }
            },
            encode_visibility,
        )?;
        let postprocess = self.encode_focus_postprocess(
            encoder,
            focus_pipelines,
            focus_target,
            &output_target.color_view,
            packet,
        )?;
        Ok(FocusPatchFrameEncoding {
            scene: scene_encoding,
            postprocess,
        })
    }

    /// Submit the complete focus frame using the scene's retained per-face
    /// visibility adapter. This convenience remains no-readback.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_focus_pbr_patch_scene_with_face_visibility(
        &self,
        frame: &RenderFrame,
        pipeline: &FocusPbrPatchRenderPipeline,
        focus_pipelines: &FocusPostprocessPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        focus_target: &FocusPostprocessTarget,
        output_target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<FocusPatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_focus_pbr_patch_scene_impl(
            frame,
            pipeline,
            focus_pipelines,
            scene,
            atlas,
            focus_target,
            output_target,
            use_qb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_offscreen_focus_pbr_patch_scene_impl(
        &self,
        frame: &RenderFrame,
        pipeline: &FocusPbrPatchRenderPipeline,
        focus_pipelines: &FocusPostprocessPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        focus_target: &FocusPostprocessTarget,
        output_target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<FocusPatchFrameEncoding, LodWebGpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting complete offscreen focus frame"),
            });
        let encoding = self.encode_focus_pbr_patch_render_scene_impl(
            &mut encoder,
            frame,
            pipeline,
            focus_pipelines,
            scene,
            atlas,
            focus_target,
            output_target,
            use_qb,
            |encoder, _, _| {
                self.encode_face_visibility_expansion(scene, encoder);
                Ok(())
            },
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(encoding)
    }

    /// Submit a live offscreen normals frame whose current visibility arrives
    /// as the compact per-face bitset retained by this scene.
    pub fn render_offscreen_normals_patch_scene_with_face_visibility(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        require_normals_frame_pipeline(frame, pipeline)?;
        self.render_offscreen_diagnostic_patch_scene_impl(
            frame,
            pipeline,
            scene,
            atlas,
            target,
            use_qb,
            true,
            wgpu::Color::TRANSPARENT,
        )
    }

    /// Live parity variant whose clear color matches the incumbent WebGL2
    /// framebuffer. It remains a no-readback render call.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_normals_patch_scene_with_face_visibility_and_clear(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        clear_color: wgpu::Color,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        require_normals_frame_pipeline(frame, pipeline)?;
        self.render_offscreen_diagnostic_patch_scene_impl(
            frame,
            pipeline,
            scene,
            atlas,
            target,
            use_qb,
            true,
            clear_color,
        )
    }

    /// Incumbent-parity convenience for the WebGL2 renderer's canonical RGB
    /// clear. Transparent alpha makes the diagnostic background an explicit
    /// fragment-coverage mask without changing the visible RGB comparison.
    pub fn render_offscreen_normals_patch_scene_with_webgl_clear(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        require_normals_frame_pipeline(frame, pipeline)?;
        self.render_offscreen_diagnostic_patch_scene_impl(
            frame,
            pipeline,
            scene,
            atlas,
            target,
            use_qb,
            true,
            wgpu::Color {
                r: 0.2,
                g: 0.2,
                b: 0.3,
                a: 0.0,
            },
        )
    }

    /// Present a live single-pass diagnostic frame directly to a Rust-owned
    /// browser surface. The same frame, scene bindings, visibility expansion,
    /// and draw encoder are used by the offscreen parity path.
    pub fn present_diagnostic_patch_scene_with_face_visibility(
        &self,
        surface: &mut PatchPresentationSurface,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        use_qb: bool,
    ) -> Result<SurfacePresentation<PatchFrameEncoding>, LodWebGpuError> {
        surface.present_with(
            self,
            "quilting live presentation frame",
            |encoder, mut target| {
                target.clear_color = Some(wgpu::Color {
                    r: 0.2,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                });
                target.clear_depth = Some(1.0);
                self.encode_diagnostic_patch_render_scene(
                    encoder,
                    frame,
                    pipeline,
                    scene,
                    atlas,
                    target,
                    use_qb,
                    |encoder, _, _| {
                        self.encode_face_visibility_expansion(scene, encoder);
                        Ok(())
                    },
                )
            },
        )
    }

    /// Present any supported diagnostic frame, including matcap plus wire,
    /// from one preparation/visibility result and one surface submission.
    pub fn present_supported_patch_scene_with_face_visibility(
        &self,
        surface: &mut PatchPresentationSurface,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        use_qb: bool,
    ) -> Result<SurfacePresentation<PatchFrameEncoding>, LodWebGpuError> {
        surface.present_with(
            self,
            "quilting live presentation frame",
            |encoder, mut target| {
                target.clear_color = Some(wgpu::Color {
                    r: 0.2,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                });
                target.clear_depth = Some(1.0);
                self.encode_supported_patch_render_scene(
                    encoder,
                    frame,
                    pipelines,
                    scene,
                    atlas,
                    target,
                    use_qb,
                    |encoder, _, _| {
                        self.encode_face_visibility_expansion(scene, encoder);
                        Ok(())
                    },
                )
            },
        )
    }

    /// Present one supported frame from the latest device-resident,
    /// crack-free classifier epoch. Classification, visibility expansion,
    /// stable compaction, and indirect drawing remain ordered on the device;
    /// no mesh-sized visibility payload crosses the CPU boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn present_supported_patch_scene_with_resident_lod_visibility(
        &self,
        surface: &mut PatchPresentationSurface,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
        atlas: &PackedPatchAtlas,
        use_qb: bool,
    ) -> Result<SurfacePresentation<PatchFrameEncoding>, LodWebGpuError> {
        surface.present_with(
            self,
            "quilting device-resident LOD presentation frame",
            |encoder, mut target| {
                target.clear_color = Some(wgpu::Color {
                    r: 0.2,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                });
                target.clear_depth = Some(1.0);
                self.encode_supported_patch_render_scene(
                    encoder,
                    frame,
                    pipelines,
                    scene,
                    atlas,
                    target,
                    use_qb,
                    |encoder, _, _| {
                        self.encode_patch_render_resident_lod_visibility(scene, resident, encoder)
                    },
                )
            },
        )
    }

    /// Compatibility wrapper for normals-only presentation callers.
    pub fn present_normals_patch_scene_with_face_visibility(
        &self,
        surface: &mut PatchPresentationSurface,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        use_qb: bool,
    ) -> Result<SurfacePresentation<PatchFrameEncoding>, LodWebGpuError> {
        require_normals_frame_pipeline(frame, pipeline)?;
        self.present_diagnostic_patch_scene_with_face_visibility(
            surface, frame, pipeline, scene, atlas, use_qb,
        )
    }

    /// Submit a supported single-pass diagnostic frame with compact per-face
    /// visibility to a retained offscreen target.
    pub fn render_offscreen_diagnostic_patch_scene_with_face_visibility(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_diagnostic_patch_scene_impl(
            frame,
            pipeline,
            scene,
            atlas,
            target,
            use_qb,
            true,
            wgpu::Color::TRANSPARENT,
        )
    }

    /// Submit one single-pass diagnostic frame from a device-resident LOD
    /// epoch without staging visibility through CPU memory.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_diagnostic_patch_scene_with_resident_lod_visibility(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        if frame.view.viewport != target.size {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen target {:?} does not match frame viewport {:?}",
                target.size, frame.view.viewport,
            )));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting device-resident LOD diagnostic frame"),
            });
        let encoding = self.encode_diagnostic_patch_render_scene(
            &mut encoder,
            frame,
            pipeline,
            scene,
            atlas,
            PatchRenderTarget {
                color_view: &target.color_view,
                resolve_target: None,
                depth_stencil_view: Some(&target.depth_view),
                clear_color: Some(wgpu::Color::TRANSPARENT),
                clear_depth: Some(1.0),
            },
            use_qb,
            |encoder, _, _| {
                self.encode_patch_render_resident_lod_visibility(scene, resident, encoder)
            },
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(encoding)
    }

    /// Submit any supported diagnostic frame to the retained offscreen target
    /// without a readback. Composite styles still prepare and compact once.
    pub fn render_offscreen_supported_patch_scene_with_face_visibility(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_supported_patch_scene_with_face_visibility_and_clear(
            frame,
            pipelines,
            scene,
            atlas,
            target,
            use_qb,
            wgpu::Color::TRANSPARENT,
        )
    }

    /// Submit any supported diagnostic frame from a device-resident LOD
    /// epoch. This is the rollback-independent counterpart to the compact
    /// CPU-bitset adapter above and performs no readback.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_supported_patch_scene_with_resident_lod_visibility(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_supported_patch_scene_with_resident_lod_visibility_and_clear(
            frame,
            pipelines,
            scene,
            resident,
            atlas,
            target,
            use_qb,
            wgpu::Color::TRANSPARENT,
        )
    }

    /// Device-resident LOD variant with an explicit parity clear color.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_supported_patch_scene_with_resident_lod_visibility_and_clear(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        clear_color: wgpu::Color,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        if frame.view.viewport != target.size {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen target {:?} does not match frame viewport {:?}",
                target.size, frame.view.viewport,
            )));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting device-resident LOD offscreen frame"),
            });
        let encoding = self.encode_supported_patch_render_scene(
            &mut encoder,
            frame,
            pipelines,
            scene,
            atlas,
            PatchRenderTarget {
                color_view: &target.color_view,
                resolve_target: None,
                depth_stencil_view: Some(&target.depth_view),
                clear_color: Some(clear_color),
                clear_depth: Some(1.0),
            },
            use_qb,
            |encoder, _, _| {
                self.encode_patch_render_resident_lod_visibility(scene, resident, encoder)
            },
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(encoding)
    }

    /// Device-resident LOD convenience matching the incumbent WebGL2 clear.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_supported_patch_scene_with_resident_lod_visibility_and_webgl_clear(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        resident: &DeviceResidentLod<'_>,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_supported_patch_scene_with_resident_lod_visibility_and_clear(
            frame,
            pipelines,
            scene,
            resident,
            atlas,
            target,
            use_qb,
            wgpu::Color {
                r: 0.2,
                g: 0.2,
                b: 0.3,
                a: 0.0,
            },
        )
    }

    /// Submit a supported style with an explicit parity clear while preserving
    /// the shared preparation, visibility, and draw-command path.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_supported_patch_scene_with_face_visibility_and_clear(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        clear_color: wgpu::Color,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_supported_patch_scene_with_face_visibility_impl(
            frame,
            pipelines,
            scene,
            atlas,
            target,
            use_qb,
            clear_color,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_offscreen_supported_patch_scene_with_face_visibility_impl(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        clear_color: wgpu::Color,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        if frame.view.viewport != target.size {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen target {:?} does not match frame viewport {:?}",
                target.size, frame.view.viewport,
            )));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting live offscreen diagnostic frame"),
            });
        let encoding = self.encode_render_frame_with_pipelines(
            &mut encoder,
            frame,
            scene.scene.snapshot(),
            &scene.bindings,
            &scene.patches,
            &scene.visibility,
            atlas,
            PatchRenderTarget {
                color_view: &target.color_view,
                resolve_target: None,
                depth_stencil_view: Some(&target.depth_view),
                clear_color: Some(clear_color),
                clear_depth: Some(1.0),
            },
            None,
            use_qb,
            Some(&pipelines.highlight),
            |pass, geometry| pipelines.get_for_pass(pass, geometry),
            |encoder, _, _| {
                self.encode_face_visibility_expansion(scene, encoder);
                Ok(())
            },
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(encoding)
    }

    /// Backend-neutral convenience for exact incumbent RGB with transparent
    /// diagnostic coverage alpha.
    pub fn render_offscreen_supported_patch_scene_with_webgl_clear(
        &self,
        frame: &RenderFrame,
        pipelines: &DiagnosticPatchRenderPipelines,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        self.render_offscreen_supported_patch_scene_with_face_visibility_and_clear(
            frame,
            pipelines,
            scene,
            atlas,
            target,
            use_qb,
            wgpu::Color {
                r: 0.2,
                g: 0.2,
                b: 0.3,
                a: 0.0,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_offscreen_diagnostic_patch_scene_impl(
        &self,
        frame: &RenderFrame,
        pipeline: &PatchRenderPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        use_qb: bool,
        expand_face_visibility: bool,
        clear_color: wgpu::Color,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        if frame.view.viewport != target.size {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen target {:?} does not match frame viewport {:?}",
                target.size, frame.view.viewport,
            )));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting live offscreen shadow frame"),
            });
        let encoding = self.encode_diagnostic_patch_render_scene(
            &mut encoder,
            frame,
            pipeline,
            scene,
            atlas,
            PatchRenderTarget {
                color_view: &target.color_view,
                resolve_target: None,
                depth_stencil_view: Some(&target.depth_view),
                clear_color: Some(clear_color),
                clear_depth: Some(1.0),
            },
            use_qb,
            |encoder, _, _| {
                if expand_face_visibility {
                    self.encode_face_visibility_expansion(scene, encoder);
                }
                Ok(())
            },
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(encoding)
    }

    /// Upload one exact global frame plus one local conformal/material domain
    /// per retained batch. Both tables are written before command submission;
    /// scenes with distinct Möbius maps need no queue writes between draws.
    pub fn write_patch_render_frames(
        &self,
        bindings: &PatchRenderBindings,
        frames: &[PatchRenderFrame],
    ) -> Result<(), LodWebGpuError> {
        let Some(first) = frames.first() else {
            return Err(LodWebGpuError::Payload(
                "patch render frame table must be nonempty".to_string(),
            ));
        };
        if frames.iter().any(|frame| frame.global != first.global) {
            return Err(LodWebGpuError::Payload(
                "patch render domains disagree on frame-global state".to_string(),
            ));
        }
        self.write_patch_render_frame_parts(
            bindings,
            first.global,
            frames.iter().map(|frame| Ok(frame.domain)),
        )
    }

    fn write_patch_render_global(
        &self,
        residency: &PatchRenderGlobalResidency,
        global: PatchRenderGlobal,
    ) -> Result<(), LodWebGpuError> {
        let global_words = global.to_words()?;
        let mut global_table = residency.table.lock().map_err(|_| {
            LodWebGpuError::Payload("patch global-frame staging lock was poisoned".to_string())
        })?;
        let mut global_changed = global_table.begin_update();
        global_changed |= global_table.replace_row(0, &global_words);
        let global_publication = global_table.commit(global_changed);
        if matches!(global_publication, FrameTablePublication::Upload { .. }) {
            self.queue.write_buffer(
                &residency.buffer,
                0,
                bytemuck::cast_slice(global_table.words.as_slice()),
            );
        }
        self.record_frame_table_publication(global_publication);
        Ok(())
    }

    fn write_patch_render_frame_parts(
        &self,
        bindings: &PatchRenderBindings,
        global: PatchRenderGlobal,
        domains: impl ExactSizeIterator<Item = Result<PatchRenderDomain, LodWebGpuError>>,
    ) -> Result<(), LodWebGpuError> {
        if domains.len() != bindings.domain_count as usize {
            return Err(LodWebGpuError::Payload(format!(
                "patch render domain table has {} records; expected {}",
                domains.len(),
                bindings.domain_count,
            )));
        }
        self.write_patch_render_global(&bindings.global_frame, global)?;

        let mut domain_table = bindings.domain_table.lock().map_err(|_| {
            LodWebGpuError::Payload("patch render-domain staging lock was poisoned".to_string())
        })?;
        let mut domains_changed = domain_table.begin_update();
        for (row, domain) in domains.enumerate() {
            let words = match domain.and_then(PatchRenderDomain::to_words) {
                Ok(words) => words,
                Err(error) => {
                    domain_table.invalidate();
                    return Err(error);
                }
            };
            domains_changed |= domain_table.replace_row(row, &words);
        }
        debug_assert_eq!(
            std::mem::size_of_val(domain_table.words.as_slice()) as u64,
            u64::from(bindings.domain_count) * PATCH_RENDER_DOMAIN_BYTES,
        );
        let domain_publication = domain_table.commit(domains_changed);
        if matches!(domain_publication, FrameTablePublication::Upload { .. }) {
            self.queue.write_buffer(
                &bindings.domains,
                0,
                bytemuck::cast_slice(domain_table.words.as_slice()),
            );
        }
        self.record_frame_table_publication(domain_publication);
        Ok(())
    }

    /// Narrow compatibility helper for single-batch conformance fixtures.
    /// Production scenes must upload the complete table atomically with
    /// [`Self::write_patch_render_frames`].
    pub fn write_patch_render_frame(
        &self,
        bindings: &PatchRenderBindings,
        frame: PatchRenderFrame,
    ) -> Result<(), LodWebGpuError> {
        self.write_patch_render_frames(bindings, &[frame])
    }

    /// Encode the supported prepared rational-QB subset of a shared frame. The
    /// canonical command sequence proves that every logical prepare precedes
    /// every visibility resolution, so each scene-wide compute phase is safely
    /// coalesced once.
    ///
    /// `encode_visibility` runs after patch preparation and before compaction.
    /// It may dispatch a same-device classifier into
    /// [`VisibilityCompactionScene::source_visibility_buffer`], or be a no-op
    /// when the caller already uploaded an exact current-pose stream. No map or
    /// copy to CPU memory occurs here.
    #[allow(clippy::too_many_arguments)]
    fn encode_render_frame_with_pipelines<'resource, VisibilityProducer, PipelineResolver>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        bindings: &'resource PatchRenderBindings,
        patches: &'resource PatchPreparationScene,
        visibility: &'resource VisibilityCompactionScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        raw_field_target: Option<(&'resource wgpu::TextureView, Option<wgpu::Color>)>,
        use_qb: bool,
        highlight_pipeline: Option<&'resource PatchRenderPipeline>,
        resolve_pipeline: PipelineResolver,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        PipelineResolver: Fn(
            RenderPass,
            RenderGeometry,
        ) -> Result<&'resource PatchRenderPipeline, LodWebGpuError>,
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        let execution = frame
            .execution(scene)
            .map_err(|error| LodWebGpuError::Payload(format!("render frame contract: {error}")))?;
        if frame.style == RenderStyle::Pbr {
            if raw_field_target.is_some() {
                validate_focus_pbr_frame(scene, frame.options)?;
            } else {
                validate_basic_pbr_frame(scene, frame.options)?;
            }
        }
        if scene.batches.is_empty() {
            return Err(LodWebGpuError::Payload(
                "WebGPU patch renderer requires a nonempty retained scene".to_string(),
            ));
        }
        let batch_count = u32::try_from(scene.batches.len()).map_err(|_| {
            LodWebGpuError::Payload("render scene batch count exceeds u32".to_string())
        })?;
        let source_instance_count = scene.batches.iter().try_fold(0u32, |total, batch| {
            let count = u32::try_from(batch.members.len()).map_err(|_| {
                LodWebGpuError::Payload("render batch instance count exceeds u32".to_string())
            })?;
            total.checked_add(count).ok_or_else(|| {
                LodWebGpuError::Payload("render scene instance count exceeds u32".to_string())
            })
        })?;
        if patches.patch_count != source_instance_count
            || visibility.source_count != source_instance_count
            || visibility.batch_count != batch_count
            || bindings.domain_count != batch_count
            || bindings.material_count as usize != scene.materials.len().max(1)
            || (frame.style == RenderStyle::Pbr
                && bindings
                    .material_textures
                    .as_ref()
                    .is_none_or(|textures| textures.material_count() != bindings.material_count))
        {
            return Err(LodWebGpuError::Payload(format!(
                "WebGPU render residency does not match scene: batches={batch_count}, instances={source_instance_count}, patch_records={}, visibility={}/{}, domains={}, materials={}/{}",
                patches.patch_count,
                visibility.batch_count,
                visibility.source_count,
                bindings.domain_count,
                bindings.material_count,
                scene.materials.len().max(1),
            )));
        }

        // Keep unsupported commands explicit even after shared validation, so
        // extending RenderFrame cannot silently lower to the wrong pipeline.
        for command in execution {
            match command {
                ResolvedRenderCommand::PreparePatches { .. }
                | ResolvedRenderCommand::ResolveVisibility { .. } => {}
                ResolvedRenderCommand::DrawPatches {
                    batch_index,
                    batch,
                    pass,
                    geometry,
                    index_count,
                    ..
                } => {
                    let pipeline = resolve_pipeline(pass, geometry)?;
                    if pipeline.geometry != geometry {
                        return Err(LodWebGpuError::Payload(format!(
                            "WebGPU {pass:?} pipeline uses {:?}, but the frame requires {geometry:?}",
                            pipeline.geometry,
                        )));
                    }
                    let draw = atlas.draw(batch.id.key.lod, geometry).ok_or_else(|| {
                        LodWebGpuError::Payload(format!(
                            "packed WebGPU atlas is missing batch {batch_index} key {:?} for {geometry:?}",
                            batch.id.key.lod,
                        ))
                    })?;
                    if draw.index_count != index_count {
                        return Err(LodWebGpuError::Payload(format!(
                            "WebGPU atlas batch {batch_index} has {} {geometry:?} indices; scene requires {index_count}",
                            draw.index_count,
                        )));
                    }
                    let index_width = match draw.index_format {
                        wgpu::IndexFormat::Uint16 => 2,
                        wgpu::IndexFormat::Uint32 => 4,
                    };
                    let first_index_byte = u64::from(draw.first_index) * index_width;
                    let required_index_bytes = u64::from(draw.index_count)
                        .checked_mul(index_width)
                        .and_then(|bytes| first_index_byte.checked_add(bytes))
                        .ok_or_else(|| {
                            LodWebGpuError::Payload(format!(
                                "WebGPU atlas batch {batch_index} {geometry:?} range exceeds u64",
                            ))
                        })?;
                    if draw.index_buffer.size() < required_index_bytes {
                        return Err(LodWebGpuError::Payload(format!(
                            "WebGPU atlas batch {batch_index} {geometry:?} buffer is {} bytes; packed range ends at {required_index_bytes}",
                            draw.index_buffer.size(),
                        )));
                    }
                }
                ResolvedRenderCommand::HighlightFace { .. } if highlight_pipeline.is_some() => {}
                unsupported => {
                    return Err(LodWebGpuError::Payload(format!(
                        "WebGPU patch renderer does not yet support command {unsupported:?}",
                    )));
                }
            }
        }

        self.write_patch_render_frame_parts(
            bindings,
            PatchRenderGlobal::from_render_frame(frame, use_qb),
            scene.batches.iter().map(|batch| {
                let material_slot =
                    patch_pbr_material_slot(&scene.materials, batch.id.key.material_index)?;
                Ok(PatchRenderDomain::from_transform(
                    batch.transform,
                    material_slot,
                ))
            }),
        )?;

        self.encode_patch_preparation(patches, encoder);
        encode_visibility(encoder, patches, visibility)?;
        self.encode_visibility_compaction(visibility, encoder);

        let mut indirect_draw_calls = 0u32;
        {
            let color_load = target
                .clear_color
                .map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear);
            let depth_stencil_attachment =
                target
                    .depth_stencil_view
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: target
                                .clear_depth
                                .map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    });
            let primary_attachment = Some(wgpu::RenderPassColorAttachment {
                view: target.color_view,
                depth_slice: None,
                resolve_target: target.resolve_target,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            });
            let has_raw_field_target = raw_field_target.is_some();
            let raw_attachment = raw_field_target.map(|(view, clear)| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                        store: wgpu::StoreOp::Store,
                    },
                })
            });
            let color_attachments = [primary_attachment, raw_attachment.flatten()];
            let color_attachments = if has_raw_field_target {
                &color_attachments[..]
            } else {
                &color_attachments[..1]
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quilting shared diagnostic frame"),
                color_attachments,
                depth_stencil_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for command in execution {
                let ResolvedRenderCommand::DrawPatches {
                    batch_index,
                    batch,
                    pass,
                    geometry,
                    ..
                } = command
                else {
                    continue;
                };
                let draw = atlas.draw(batch.id.key.lod, geometry).ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                        "packed WebGPU atlas lost batch {batch_index} key {:?} for {geometry:?}",
                        batch.id.key.lod,
                    ))
                })?;
                let pipeline = resolve_pipeline(pass, geometry)?;
                let permutation_sign = if batch.id.key.parity_bucket == 0 {
                    1
                } else {
                    -1
                };
                let winding = if batch.transform.orientation_sign * permutation_sign < 0 {
                    PatchWinding::Clockwise
                } else {
                    PatchWinding::CounterClockwise
                };
                let material_slot =
                    patch_pbr_material_slot(&scene.materials, batch.id.key.material_index)?;
                pipeline.draw_batch(
                    &mut render_pass,
                    bindings,
                    visibility,
                    draw,
                    batch_index,
                    material_slot,
                    winding,
                )?;
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
            if execution
                .into_iter()
                .any(|command| matches!(command, ResolvedRenderCommand::HighlightFace { .. }))
            {
                let highlight_pipeline = highlight_pipeline.ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "WebGPU patch renderer has no selection highlight pipeline".to_string(),
                    )
                })?;
                for (batch_index, batch) in scene.batches.iter().enumerate() {
                    let draw = atlas
                        .draw(batch.id.key.lod, RenderGeometry::Triangles)
                        .ok_or_else(|| {
                            LodWebGpuError::Payload(format!(
                                "packed WebGPU atlas lost highlight batch {batch_index} key {:?}",
                                batch.id.key.lod,
                            ))
                        })?;
                    let permutation_sign = if batch.id.key.parity_bucket == 0 {
                        1
                    } else {
                        -1
                    };
                    let winding = if batch.transform.orientation_sign * permutation_sign < 0 {
                        PatchWinding::Clockwise
                    } else {
                        PatchWinding::CounterClockwise
                    };
                    highlight_pipeline.draw_batch(
                        &mut render_pass,
                        bindings,
                        visibility,
                        draw,
                        batch_index as u32,
                        0,
                        winding,
                    )?;
                    indirect_draw_calls = indirect_draw_calls.saturating_add(1);
                }
            }
        }

        let logical_submission = execution.submission_stats();
        Ok(PatchFrameEncoding {
            logical_submission,
            indirect_draw_calls,
            source_instance_count,
        })
    }

    /// Encode one single-pass diagnostic frame with a style-matched pipeline.
    /// Composite frames use [`Self::encode_supported_render_frame`] so each
    /// draw command can select its canonical triangle or line pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_diagnostic_render_frame<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        pipeline: &'resource PatchRenderPipeline,
        bindings: &'resource PatchRenderBindings,
        patches: &'resource PatchPreparationScene,
        visibility: &'resource VisibilityCompactionScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        if pipeline.style() != Some(frame.style) || frame.style == RenderStyle::MatcapWire {
            return Err(LodWebGpuError::Payload(format!(
                "single WebGPU patch pipeline {:?} cannot render {:?} frame",
                pipeline.style(),
                frame.style,
            )));
        }
        self.encode_render_frame_with_pipelines(
            encoder,
            frame,
            scene,
            bindings,
            patches,
            visibility,
            atlas,
            target,
            None,
            use_qb,
            None,
            |_, _| Ok(pipeline),
            encode_visibility,
        )
    }

    /// Encode any currently supported diagnostic style, including the
    /// matcap-plus-wire composite, through one preparation and visibility pass.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_supported_render_frame<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        pipelines: &'resource DiagnosticPatchRenderPipelines,
        bindings: &'resource PatchRenderBindings,
        patches: &'resource PatchPreparationScene,
        visibility: &'resource VisibilityCompactionScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        self.encode_render_frame_with_pipelines(
            encoder,
            frame,
            scene,
            bindings,
            patches,
            visibility,
            atlas,
            target,
            None,
            use_qb,
            Some(&pipelines.highlight),
            |pass, geometry| pipelines.get_for_pass(pass, geometry),
            encode_visibility,
        )
    }

    /// Compatibility entry point for the first promoted mode. New
    /// single-pass diagnostic callers should use
    /// [`Self::encode_diagnostic_render_frame`] with a style-matched pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_normals_render_frame<'resource, VisibilityProducer>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        pipeline: &'resource PatchRenderPipeline,
        bindings: &'resource PatchRenderBindings,
        patches: &'resource PatchPreparationScene,
        visibility: &'resource VisibilityCompactionScene,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        use_qb: bool,
        encode_visibility: VisibilityProducer,
    ) -> Result<PatchFrameEncoding, LodWebGpuError>
    where
        VisibilityProducer: FnOnce(
            &mut wgpu::CommandEncoder,
            &PatchPreparationScene,
            &VisibilityCompactionScene,
        ) -> Result<(), LodWebGpuError>,
    {
        if frame.style != RenderStyle::Normals || pipeline.style() != Some(RenderStyle::Normals) {
            return Err(LodWebGpuError::Payload(
                "normals render entry point requires a normals frame and pipeline".to_string(),
            ));
        }
        self.encode_diagnostic_render_frame(
            encoder,
            frame,
            scene,
            pipeline,
            bindings,
            patches,
            visibility,
            atlas,
            target,
            use_qb,
            encode_visibility,
        )
    }

    /// Upload retained batch shape and static eligibility once. Output buffers
    /// include `INDIRECT` usage now, so a later render pass can consume the
    /// generated arguments without changing this residency contract.
    pub fn upload_visibility_compaction_scene(
        &self,
        words: WgslVisibilityCompactionSceneWords,
    ) -> Result<VisibilityCompactionScene, LodWebGpuError> {
        let batch_count = words.uniform[0];
        let source_count = words.uniform[1];
        if words.uniform[2..] != [0, 0]
            || words.batches.len() != batch_count as usize
            || words.source_eligibility.len() != source_count as usize
            || words.source_eligibility.iter().any(|&value| value > 1)
        {
            return Err(LodWebGpuError::Payload(
                "visibility compaction scene shape is malformed".to_string(),
            ));
        }
        let mut expected_first = 0u32;
        for batch in &words.batches {
            if batch[0] != expected_first
                || !batch[2].is_multiple_of(3)
                || !batch[3].is_multiple_of(2)
            {
                return Err(LodWebGpuError::Payload(
                    "visibility compaction batch ranges or geometry counts are not canonical"
                        .to_string(),
                ));
            }
            expected_first = expected_first.checked_add(batch[1]).ok_or_else(|| {
                LodWebGpuError::Payload(
                    "visibility compaction source count exceeds u32".to_string(),
                )
            })?;
        }
        if expected_first != source_count {
            return Err(LodWebGpuError::Payload(
                "visibility compaction batches do not cover the source stream".to_string(),
            ));
        }

        let uniform = buffer_init_or_zero(
            &self.device,
            "visibility compaction uniform",
            bytemuck::cast_slice(&words.uniform),
            wgpu::BufferUsages::UNIFORM,
        );
        let batches = buffer_init_or_zero(
            &self.device,
            "visibility compaction batches",
            bytemuck::cast_slice(&words.batches),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let source_eligibility = buffer_init_or_zero(
            &self.device,
            "visibility compaction eligibility",
            bytemuck::cast_slice(&words.source_eligibility),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let source_bytes = u64::from(source_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility source buffer is too large".to_string())
            })?;
        let batch_count_bytes = u64::from(batch_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility batch count buffer is too large".to_string())
            })?;
        let range_bytes = u64::from(batch_count)
            .checked_mul(VISIBILITY_RANGE_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility range buffer is too large".to_string())
            })?;
        let indirect_bytes = u64::from(batch_count)
            .checked_mul(INDEXED_INDIRECT_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility indirect buffer is too large".to_string())
            })?;
        debug_assert_eq!(
            std::mem::size_of_val(&words.uniform) as u64,
            VISIBILITY_UNIFORM_BYTES
        );
        debug_assert!(words
            .batches
            .iter()
            .all(|record| std::mem::size_of_val(record) as u64 == VISIBILITY_BATCH_RECORD_BYTES));

        let uniform_alignment = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(1);
        let batch_index_uniform_stride = u32::try_from(
            DRAW_BATCH_INDEX_BYTES.div_ceil(u64::from(uniform_alignment))
                * u64::from(uniform_alignment),
        )
        .map_err(|_| LodWebGpuError::Payload("draw batch-index stride exceeds u32".to_string()))?;
        let batch_index_bytes = u64::from(batch_count)
            .checked_mul(u64::from(batch_index_uniform_stride))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                LodWebGpuError::Payload("draw batch-index table is too large".to_string())
            })?;
        let mut batch_index_words = vec![0u8; batch_index_bytes];
        for batch_index in 0..batch_count {
            let offset = batch_index as usize * batch_index_uniform_stride as usize;
            batch_index_words[offset..offset + 4].copy_from_slice(&batch_index.to_le_bytes());
        }
        let batch_index_uniform = buffer_init_or_zero(
            &self.device,
            "visibility draw batch indices",
            &batch_index_words,
            wgpu::BufferUsages::UNIFORM,
        );

        let source_visibility = gpu_buffer(
            &self.device,
            "visibility compaction current visibility",
            source_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let batch_counts = gpu_buffer(
            &self.device,
            "visibility compaction batch counts",
            batch_count_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let compacted_source_instances = gpu_buffer(
            &self.device,
            "visibility compacted source instances",
            source_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let compacted_ranges = gpu_buffer(
            &self.device,
            "visibility compacted ranges",
            range_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let triangle_indirect_arguments = gpu_buffer(
            &self.device,
            "visibility triangle indirect arguments",
            indirect_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );
        let line_indirect_arguments = gpu_buffer(
            &self.device,
            "visibility line indirect arguments",
            indirect_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );
        let count_layout = self.visibility_count_pipeline.get_bind_group_layout(0);
        let count_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility count bindings"),
            layout: &count_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &batches),
                bind(2, &source_eligibility),
                bind(3, &source_visibility),
                bind(4, &batch_counts),
            ],
        });
        let scan_layout = self.visibility_scan_pipeline.get_bind_group_layout(0);
        let scan_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility scan bindings"),
            layout: &scan_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &batches),
                bind(2, &batch_counts),
                bind(3, &compacted_ranges),
                bind(4, &triangle_indirect_arguments),
                bind(5, &line_indirect_arguments),
            ],
        });
        let scatter_layout = self.visibility_scatter_pipeline.get_bind_group_layout(0);
        let scatter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility scatter bindings"),
            layout: &scatter_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &batches),
                bind(2, &source_eligibility),
                bind(3, &source_visibility),
                bind(4, &compacted_ranges),
                bind(5, &compacted_source_instances),
            ],
        });

        Ok(VisibilityCompactionScene {
            batch_count,
            source_count,
            batch_index_uniform_stride,
            batch_index_uniform,
            batches,
            source_eligibility,
            source_visibility,
            compacted_source_instances,
            compacted_ranges,
            triangle_indirect_arguments,
            line_indirect_arguments,
            count_bind_group,
            scan_bind_group,
            scatter_bind_group,
        })
    }

    /// Validate and upload a CPU/shadow visibility fixture. A same-device
    /// producer can write [`VisibilityCompactionScene::source_visibility_buffer`]
    /// directly and skip this transfer.
    pub fn write_source_visibility(
        &self,
        scene: &VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<(), LodWebGpuError> {
        let visibility = pack_wgsl_source_visibility_words(source_visibility, scene.source_count)
            .map_err(LodWebGpuError::Payload)?;
        if !visibility.is_empty() {
            self.queue.write_buffer(
                &scene.source_visibility,
                0,
                bytemuck::cast_slice(&visibility),
            );
        }
        Ok(())
    }

    /// Append count, deterministic batch scan, and stable parallel scatter to
    /// an application-owned command encoder. A caller can encode its visibility
    /// producer before this and indirect render passes after this, preserving
    /// one ordered GPU submission with no map/copy boundary.
    pub fn encode_visibility_compaction(
        &self,
        scene: &VisibilityCompactionScene,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if scene.batch_count == 0 {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting visibility count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.visibility_count_pipeline);
            pass.set_bind_group(0, &scene.count_bind_group, &[]);
            pass.dispatch_workgroups(scene.batch_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting visibility scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.visibility_scan_pipeline);
            pass.set_bind_group(0, &scene.scan_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting visibility scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.visibility_scatter_pipeline);
            pass.set_bind_group(0, &scene.scatter_bind_group, &[]);
            pass.dispatch_workgroups(scene.batch_count, 1, 1);
        }
    }

    /// Convenience submission for CPU/shadow visibility input. This method
    /// returns immediately after queue submission and performs no readback.
    pub fn compact_visibility_on_device(
        &self,
        scene: &VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<(), LodWebGpuError> {
        self.write_source_visibility(scene, source_visibility)?;
        if scene.batch_count == 0 {
            return Ok(());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting visibility compaction"),
            });
        self.encode_visibility_compaction(scene, &mut encoder);
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Diagnostic wrapper around the same no-readback encoder path. Temporary
    /// staging buffers exist only for conformance calls and are not retained by
    /// live scene residency.
    pub async fn compact_visibility(
        &self,
        scene: &mut VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<VisibilityCompactionOutput, LodWebGpuError> {
        self.write_source_visibility(scene, source_visibility)?;
        if scene.batch_count == 0 {
            return Ok(VisibilityCompactionOutput {
                compacted_source_instances: Vec::new(),
                compacted_ranges: Vec::new(),
                triangle_indirect_arguments: Vec::new(),
                line_indirect_arguments: Vec::new(),
            });
        }

        let source_bytes = u64::from(scene.source_count) * PACKED_RECORD_BYTES;
        let range_bytes = u64::from(scene.batch_count) * VISIBILITY_RANGE_RECORD_BYTES;
        let indirect_bytes = u64::from(scene.batch_count) * INDEXED_INDIRECT_RECORD_BYTES;
        let source_readback = gpu_buffer(
            &self.device,
            "visibility source diagnostic readback",
            source_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let range_readback = gpu_buffer(
            &self.device,
            "visibility range diagnostic readback",
            range_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let triangle_indirect_readback = gpu_buffer(
            &self.device,
            "visibility triangle indirect diagnostic readback",
            indirect_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let line_indirect_readback = gpu_buffer(
            &self.device,
            "visibility line indirect diagnostic readback",
            indirect_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting visibility diagnostic compaction"),
            });
        self.encode_visibility_compaction(scene, &mut encoder);
        if source_bytes != 0 {
            encoder.copy_buffer_to_buffer(
                &scene.compacted_source_instances,
                0,
                &source_readback,
                0,
                source_bytes,
            );
        }
        encoder.copy_buffer_to_buffer(&scene.compacted_ranges, 0, &range_readback, 0, range_bytes);
        encoder.copy_buffer_to_buffer(
            &scene.triangle_indirect_arguments,
            0,
            &triangle_indirect_readback,
            0,
            indirect_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &scene.line_indirect_arguments,
            0,
            &line_indirect_readback,
            0,
            indirect_bytes,
        );
        self.queue.submit([encoder.finish()]);

        let compacted_ranges = words_to_five_records(
            self.readback_words(&range_readback, range_bytes).await?,
            "visibility range",
        )?;
        let triangle_indirect_arguments = words_to_five_records(
            self.readback_words(&triangle_indirect_readback, indirect_bytes)
                .await?,
            "visibility triangle indirect",
        )?;
        let line_indirect_arguments = words_to_five_records(
            self.readback_words(&line_indirect_readback, indirect_bytes)
                .await?,
            "visibility line indirect",
        )?;
        let survivor_count = compacted_ranges
            .last()
            .map_or(0u32, |range| range[3].saturating_add(range[4]));
        if survivor_count > scene.source_count {
            return Err(LodWebGpuError::Conformance(
                "visibility compaction emitted too many survivors".to_string(),
            ));
        }
        let mut compacted_source_instances = if source_bytes == 0 {
            Vec::new()
        } else {
            self.readback_words(&source_readback, source_bytes).await?
        };
        compacted_source_instances.truncate(survivor_count as usize);
        Ok(VisibilityCompactionOutput {
            compacted_source_instances,
            compacted_ranges,
            triangle_indirect_arguments,
            line_indirect_arguments,
        })
    }

    /// Device proof that animated preparation, stable compaction, vertex
    /// pulling, shared QB evaluation, rasterization, and indirect arguments
    /// can execute in one submission without an intermediate CPU map.
    async fn validate_patch_render_conformance(
        &self,
        model: &LodClassifierModel,
        patches: &PatchPreparationScene,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<(usize, usize), LodWebGpuError> {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 32;
        const PADDED_BYTES_PER_ROW: u32 = 256;

        let normal_model = {
            let mut matrix = identity_matrix();
            matrix[10] = 2.0;
            matrix
        };
        let batch = |material_index,
                     layer,
                     lod,
                     edge_lods,
                     parity_bucket,
                     face_index,
                     leaf_id,
                     permutation_index,
                     vertex_lods| RenderBatchSnapshot {
            id: RenderBatchId {
                key: RenderBatchKey {
                    lod,
                    parity_bucket,
                    material_index,
                    render_node_index: 0,
                },
                layer,
            },
            members: vec![RenderBatchMember {
                face_index,
                leaf_id,
                node_index: 7,
                edge_lods,
                permutation_index,
                vertex_lods,
            }],
            triangle_index_count: 3,
            line_index_count: 6,
            transform: RenderEntityTransform {
                mobius: identity_mobius(),
                orientation_sign: 1,
                euclidean_model: identity_matrix(),
                euclidean_normal: normal_model,
            },
            enabled: true,
            pbr_class: PbrDrawClass::Opaque,
        };
        let render_scene = RenderSceneSnapshot {
            revision: 73,
            materials: Vec::new(),
            suppressed_root_faces: vec![0],
            batches: vec![
                batch(
                    0,
                    RenderBatchLayer::RetainedRoot,
                    [2, 4, 8],
                    [8, 4, 2],
                    1,
                    0,
                    ScreenPatchLeafId::ROOT,
                    5,
                    [4, 8, 16],
                ),
                batch(
                    1,
                    RenderBatchLayer::AdaptiveOverlay,
                    [1, 2, 4],
                    [2, 4, 1],
                    0,
                    0,
                    ScreenPatchLeafId::ROOT.child(3).ok_or_else(|| {
                        LodWebGpuError::Conformance(
                            "missing adaptive conformance child".to_string(),
                        )
                    })?,
                    3,
                    [4, 8, 8],
                ),
            ],
        };
        let mvp = [
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, -0.25, 0.5, 1.0,
        ];
        let render_frame = RenderFrame::build(
            11,
            RenderPoseIdentity {
                asset_revision: 5,
                pose_revision: 9,
            },
            RenderStyle::Normals,
            RenderView {
                viewport: [WIDTH, HEIGHT],
                mvp,
                model_view: identity_matrix(),
                camera_position: [0.0, 0.0, 4.0],
                selected_node: None,
                focus: FocusFieldPacket::default(),
            },
            RenderFrameOptions::default(),
            &render_scene,
        )
        .map_err(|error| {
            LodWebGpuError::Conformance(format!("shared render-frame fixture is invalid: {error}",))
        })?;
        let visibility_words = WgslVisibilityCompactionSceneWords {
            uniform: [2, patches.patch_count, 0, 0],
            batches: vec![[0, 1, 3, 6], [1, 1, 3, 6]],
            source_eligibility: vec![0, 1],
        };
        let visibility = self.upload_visibility_compaction_scene(visibility_words)?;
        let pipeline =
            self.create_patch_render_pipeline(wgpu::TextureFormat::Rgba8Unorm, None, 1)?;
        let bindings = self.create_patch_render_bindings(
            &pipeline,
            &render_scene,
            patches,
            &visibility,
            None,
        )?;
        self.write_patch_pose(model, patches, pose, num_joints)?;
        let source_visibility = vec![1; patches.patch_count as usize];
        self.write_source_visibility(&visibility, &source_visibility)?;

        let barycentrics = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // The degenerate prefix proves packed-atlas slicing: reading from
        // element zero would produce no plausible rasterized footprint.
        let indices = [0u32, 0, 0, 0, 1, 2];
        let packed_atlas = self.upload_packed_patch_atlas(
            &[2, 4, 8, 3, 3, 0, 0, 1, 2, 4, 3, 3, 0, 0],
            &barycentrics,
            &indices,
            &[],
        )?;
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("patch render conformance target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let readback_bytes = u64::from(PADDED_BYTES_PER_ROW) * u64::from(HEIGHT);
        let readback = gpu_buffer(
            &self.device,
            "patch render conformance readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("patch prepare compact render conformance"),
            });
        let encoding = self.encode_normals_render_frame(
            &mut encoder,
            &render_frame,
            &render_scene,
            &pipeline,
            &bindings,
            patches,
            &visibility,
            &packed_atlas,
            PatchRenderTarget {
                color_view: &target_view,
                resolve_target: None,
                depth_stencil_view: None,
                clear_color: Some(wgpu::Color::TRANSPARENT),
                clear_depth: None,
            },
            true,
            |_, _, _| Ok(()),
        )?;
        if encoding.indirect_draw_calls != 2
            || encoding.source_instance_count != 2
            || encoding.logical_submission
                != render_frame
                    .expected_submission_stats(&render_scene)
                    .map_err(|error| {
                        LodWebGpuError::Conformance(format!(
                            "shared render-frame stats are invalid: {error}",
                        ))
                    })?
        {
            return Err(LodWebGpuError::Conformance(format!(
                "shared render-frame encoding mismatch: {encoding:?}",
            )));
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(PADDED_BYTES_PER_ROW),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        if let Some(error) = error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "patch prepare/compact/render submission failed validation: {error}"
            )));
        }
        let words_per_row = PADDED_BYTES_PER_ROW as usize / std::mem::size_of::<u32>();
        let pixels = self.readback_words(&readback, readback_bytes).await?;
        let rendered_pixels = pixels
            .chunks_exact(words_per_row)
            .take(HEIGHT as usize)
            .enumerate()
            .flat_map(|(y, row)| {
                row[..WIDTH as usize]
                    .iter()
                    .enumerate()
                    .filter_map(move |(x, &pixel)| (pixel != 0).then_some([x as u32, y as u32]))
            })
            .collect::<Vec<_>>();
        let rendered = rendered_pixels.len();
        if !(8..WIDTH as usize * HEIGHT as usize).contains(&rendered) {
            return Err(LodWebGpuError::Conformance(format!(
                "patch render produced an implausible {rendered}-pixel footprint"
            )));
        }

        // Query a pixel proven to be covered by the ordinary render pass. The
        // second submission deliberately reuses its prepared records, frame
        // table, compacted indirect arguments, and packed atlas.
        let pick_error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pick_pipeline = self.create_patch_pick_pipeline(&pipeline)?;
        let pick_target = self.create_patch_pick_target();
        let pick_request = PatchPickRequest::new([WIDTH, HEIGHT], rendered_pixels[0], 91)?;
        let mut pick_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("patch render one-pixel pick conformance"),
                });
        let staged_pick = self.encode_patch_pick(
            &mut pick_encoder,
            &pick_pipeline,
            &render_scene,
            &bindings,
            patches,
            &visibility,
            &packed_atlas,
            &pick_target,
            pick_request,
        )?;
        if staged_pick.encoding().indirect_draw_calls != 2 {
            return Err(LodWebGpuError::Conformance(format!(
                "patch pick encoded {} indirect draws; expected 2",
                staged_pick.encoding().indirect_draw_calls,
            )));
        }
        self.queue.submit([pick_encoder.finish()]);
        if let Some(error) = pick_error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "patch pick submission failed validation: {error}"
            )));
        }
        let picked = staged_pick.read().await?.ok_or_else(|| {
            LodWebGpuError::Conformance("rendered patch pixel produced no pick".to_string())
        })?;
        if picked.target_epoch != 91
            || picked.packed_node != 7
            || picked.source_face != 0
            || (picked.source_barycentric.into_iter().sum::<f32>() - 1.0).abs() > 1.0e-5
            || picked.output_distance <= 0.0
        {
            return Err(LodWebGpuError::Conformance(format!(
                "patch pick returned an incoherent sample: {picked:?}"
            )));
        }
        Ok((rendered, encoding.indirect_draw_calls as usize))
    }

    /// Prove that portable zero-based arguments emitted by compaction can be
    /// consumed immediately by real indexed-indirect draws on this device.
    /// This is part of the shared native/browser conformance matrix only.
    async fn validate_indirect_draw_conformance(
        &self,
        scene: &VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<usize, LodWebGpuError> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visibility indirect conformance shader"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                    struct DrawBatchIndex {
                        batch_index: u32,
                        _padding_0: u32,
                        _padding_1: u32,
                        _padding_2: u32,
                    }

                    struct CompactedBatchRange {
                        batch_index: u32,
                        source_first_instance: u32,
                        source_instance_count: u32,
                        compacted_first_instance: u32,
                        compacted_instance_count: u32,
                    }

                    @group(0) @binding(0) var<uniform> draw_batch: DrawBatchIndex;
                    @group(0) @binding(1) var<storage, read> ranges: array<CompactedBatchRange>;
                    @group(0) @binding(2) var<storage, read> compacted_sources: array<u32>;

                    @vertex
                    fn vertex_main(
                        @builtin(vertex_index) vertex: u32,
                        @builtin(instance_index) local_instance: u32,
                    ) -> @builtin(position) vec4<f32> {
                        let positions = array<vec2<f32>, 3>(
                            vec2<f32>(-0.5, -0.5),
                            vec2<f32>(0.5, -0.5),
                            vec2<f32>(0.0, 0.5),
                        );
                        let range = ranges[draw_batch.batch_index];
                        let compacted_index = range.compacted_first_instance + local_instance;
                        let source_instance = compacted_sources[compacted_index];
                        let source_band = f32(source_instance % 11u) / 10.0;
                        let position = positions[vertex % 3u] * 0.08
                            + vec2<f32>(source_band * 1.8 - 0.9, 0.0);
                        return vec4<f32>(position, 0.0, 1.0);
                    }

                    @fragment
                    fn fragment_main() -> @location(0) vec4<f32> {
                        return vec4<f32>(0.25, 0.75, 1.0, 1.0);
                    }
                "#
                    .into(),
                ),
            });
        let draw_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visibility compacted draw layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: std::num::NonZeroU64::new(DRAW_BATCH_INDEX_BYTES),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visibility compacted draw pipeline layout"),
                bind_group_layouts: &[Some(&draw_layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visibility indirect conformance pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let draw_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility compacted draw bindings"),
            layout: &draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scene.batch_index_uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(DRAW_BATCH_INDEX_BYTES),
                    }),
                },
                bind(1, &scene.compacted_ranges),
                bind(2, &scene.compacted_source_instances),
            ],
        });
        let indices = [0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visibility indirect conformance indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility indirect conformance target"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        self.write_source_visibility(scene, source_visibility)?;
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visibility compaction to indirect draw conformance"),
            });
        self.encode_visibility_compaction(scene, &mut encoder);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("visibility indirect conformance pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for batch_index in 0..scene.batch_count {
                pass.set_bind_group(
                    0,
                    &draw_bind_group,
                    &[batch_index * scene.batch_index_uniform_stride],
                );
                pass.draw_indexed_indirect(
                    &scene.triangle_indirect_arguments,
                    u64::from(batch_index) * INDEXED_INDIRECT_RECORD_BYTES,
                );
            }
        }
        let submission = self.queue.submit([encoder.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| LodWebGpuError::Poll(error.to_string()))?;
        #[cfg(target_arch = "wasm32")]
        let _ = submission;
        if let Some(error) = error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "compaction-to-indirect submission failed validation: {error}"
            )));
        }
        Ok(scene.batch_count as usize)
    }

    fn allocate_model_identity(&self) -> Result<u64, LodWebGpuError> {
        let mut next = self.next_model_identity.lock().map_err(|_| {
            LodWebGpuError::Payload("WebGPU model identity lock was poisoned".to_string())
        })?;
        let identity = *next;
        *next = next.checked_add(1).ok_or_else(|| {
            LodWebGpuError::Payload("WebGPU model identity overflowed".to_string())
        })?;
        Ok(identity)
    }

    /// Upload immutable geometry/topology once and allocate retained dynamic
    /// buffers. Diagnostic readback is allocated only by an explicit
    /// conformance call; the resident model owns device-local results.
    pub fn upload_model(
        &self,
        prepared: PreparedLodModel,
        atlas: &LodAtlasLookup,
    ) -> Result<LodClassifierModel, LodWebGpuError> {
        let identity = self.allocate_model_identity()?;
        let words = pack_wgsl_lod_model_words(&prepared).map_err(LodWebGpuError::Payload)?;
        let atlas_words = pack_wgsl_lod_atlas_words(atlas);
        let face_count = prepared.residency.num_faces;
        let subject_rows = words.subject_layout.len().max(1);
        let joint_capacity = prepared
            .model
            .joint_indices
            .iter()
            .zip(&prepared.model.joint_weights)
            .flat_map(|(indices, weights)| indices.iter().zip(weights))
            .filter(|(_, weight)| **weight >= 1e-6)
            .map(|(joint, _)| usize::from(*joint) + 1)
            .max()
            .unwrap_or(1);

        let faces = storage_buffer(&self.device, "LOD faces", &words.faces, false);
        let positions = storage_buffer(&self.device, "LOD positions", &words.positions, false);
        let skinning = storage_buffer(&self.device, "LOD skinning", &words.skinning, false);
        let zero_record = [[0u32; 4]];
        let morph_source = if words.morph_deltas.is_empty() {
            zero_record.as_slice()
        } else {
            words.morph_deltas.as_slice()
        };
        let morph_deltas = storage_buffer(&self.device, "LOD morph deltas", morph_source, false);
        let adjacency = storage_buffer(&self.device, "LOD adjacency", &words.adjacency, false);
        let atlas_lut = storage_buffer(&self.device, "LOD atlas lookup", &atlas_words, false);

        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD dispatch uniform"),
            size: DISPATCH_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let joint_matrices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD joint matrices"),
            size: joint_capacity as u64 * JOINT_MATRIX_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let morph_weights = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD morph weights"),
            size: prepared.model.num_morph_targets.max(1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let subject_states = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD subject states"),
            size: subject_rows as u64 * SUBJECT_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pass1_records = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD pass one records"),
            size: face_count as u64 * PASS1_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let packed_record_bytes = face_count as u64 * PACKED_RECORD_BYTES;
        let packed_records = gpu_buffer(
            &self.device,
            "LOD packed records",
            packed_record_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let resident_ping = gpu_buffer(
            &self.device,
            "LOD resident reconciliation ping",
            packed_record_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let resident_pong = gpu_buffer(
            &self.device,
            "LOD resident reconciliation pong",
            packed_record_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let resident_packed_records = gpu_buffer(
            &self.device,
            "LOD resident packed records",
            packed_record_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let pass1_layout = self.pass1_pipeline.get_bind_group_layout(0);
        let pass1_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD pass one bindings"),
            layout: &pass1_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &faces),
                bind(2, &positions),
                bind(3, &skinning),
                bind(4, &joint_matrices),
                bind(5, &morph_deltas),
                bind(6, &morph_weights),
                bind(7, &subject_states),
                bind(8, &pass1_records),
            ],
        });
        let pass2_layout = self.pass2_pipeline.get_bind_group_layout(0);
        let pass2_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD pass two bindings"),
            layout: &pass2_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &pass1_records),
                bind(2, &adjacency),
                bind(3, &atlas_lut),
                bind(4, &packed_records),
            ],
        });
        let resident_seed_layout = self.resident_seed_pipeline.get_bind_group_layout(0);
        let resident_seed_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD resident seed bindings"),
            layout: &resident_seed_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &packed_records),
                bind(4, &resident_ping),
            ],
        });
        let resident_reconcile_bind_groups = |pipeline: &wgpu::ComputePipeline, label| {
            let layout = pipeline.get_bind_group_layout(0);
            let forward = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    bind(0, &uniform),
                    bind(2, &adjacency),
                    bind(4, &resident_ping),
                    bind(5, &resident_pong),
                ],
            });
            let backward = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    bind(0, &uniform),
                    bind(2, &adjacency),
                    bind(4, &resident_pong),
                    bind(5, &resident_ping),
                ],
            });
            (forward, backward)
        };
        let (
            resident_reconcile_2_to_1_forward_bind_group,
            resident_reconcile_2_to_1_backward_bind_group,
        ) = resident_reconcile_bind_groups(
            &self.resident_reconcile_2_to_1_pipeline,
            "LOD resident 2:1 reconcile bindings",
        );
        let (
            resident_reconcile_4_to_1_forward_bind_group,
            resident_reconcile_4_to_1_backward_bind_group,
        ) = resident_reconcile_bind_groups(
            &self.resident_reconcile_4_to_1_pipeline,
            "LOD resident 4:1 reconcile bindings",
        );
        let resident_pack_layout = self.resident_pack_pipeline.get_bind_group_layout(0);
        let resident_pack_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD resident pack bindings"),
            layout: &resident_pack_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &packed_records),
                bind(3, &atlas_lut),
                bind(4, &resident_ping),
                bind(6, &resident_packed_records),
            ],
        });
        let resident = ResidentLodReconciliationBuffers {
            packed_records: resident_packed_records,
            seed_bind_group: resident_seed_bind_group,
            reconcile_2_to_1_forward_bind_group: resident_reconcile_2_to_1_forward_bind_group,
            reconcile_2_to_1_backward_bind_group: resident_reconcile_2_to_1_backward_bind_group,
            reconcile_4_to_1_forward_bind_group: resident_reconcile_4_to_1_forward_bind_group,
            reconcile_4_to_1_backward_bind_group: resident_reconcile_4_to_1_backward_bind_group,
            pack_bind_group: resident_pack_bind_group,
        };

        Ok(LodClassifierModel {
            identity,
            atlas_keys: atlas.keys.clone(),
            prepared,
            classification_epoch: 0,
            joint_capacity,
            subject_rows,
            subject_layout: words.subject_layout,
            uniform,
            faces,
            skinning,
            joint_matrices,
            morph_deltas,
            morph_weights,
            subject_states,
            lod_state: Mutex::new(RetainedLodDispatchState::new(subject_rows)),
            packed_records,
            pass1_bind_group,
            pass2_bind_group,
            resident,
            resident_epoch: Cell::new(None),
        })
    }

    /// Publish or reuse the current pose, then upload dispatch metrics and
    /// authored subject state. This changes no immutable model shape and
    /// performs no command encoding or readback.
    pub fn write_lod_classification_state(
        &self,
        model: &LodClassifierModel,
        dispatch: &LodDispatchState,
        metrics: WgslLodDispatchMetrics,
        pose: LodPose<'_>,
        pose_upload: PoseUploadPolicy,
    ) -> Result<(), LodWebGpuError> {
        if pose_upload.should_publish_dynamic() {
            self.write_dynamic_pose(model, pose, metrics.num_joints)?;
        }
        let uniform_words = pack_wgsl_lod_dispatch_words(&model.prepared, dispatch, metrics)
            .map_err(LodWebGpuError::Payload)?;
        let mut lod_state = model.lod_state.lock().map_err(|_| {
            LodWebGpuError::Payload("LOD state staging lock was poisoned".to_string())
        })?;
        pack_wgsl_lod_subject_words_with_layout(
            &model.subject_layout,
            dispatch,
            &mut lod_state.subject_scratch,
        )
        .map_err(LodWebGpuError::Payload)?;
        if lod_state.subject_scratch.len() != model.subject_rows {
            return Err(LodWebGpuError::Payload(
                "subject table changed immutable shape".to_string(),
            ));
        }
        let uniform_publication = lod_state.commit_uniform(uniform_words);
        let subject_publication = lod_state.commit_subject_scratch();
        if matches!(uniform_publication, FrameTablePublication::Upload { .. }) {
            self.queue.write_buffer(
                &model.uniform,
                0,
                bytemuck::cast_slice(&lod_state.uniform_words),
            );
        }
        if matches!(subject_publication, FrameTablePublication::Upload { .. }) {
            self.queue.write_buffer(
                &model.subject_states,
                0,
                bytemuck::cast_slice(&lod_state.subject_words),
            );
        }
        self.record_lod_state_publication(uniform_publication);
        self.record_lod_state_publication(subject_publication);
        Ok(())
    }

    /// Append both classifier passes to an application-owned encoder and
    /// return the exact device-local packed output. A downstream reconciliation
    /// pass can consume this handle in the same encoder and queue submission.
    pub fn encode_lod_classification<'model>(
        &self,
        model: &'model mut LodClassifierModel,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<DeviceLodClassification<'model>, LodWebGpuError> {
        let epoch = model.classification_epoch.checked_add(1).ok_or_else(|| {
            LodWebGpuError::Payload("LOD classification epoch overflowed".to_string())
        })?;
        let groups = (model.prepared.residency.num_faces as u32).div_ceil(LOD_WORKGROUP_SIZE);
        if groups != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting LOD pass one"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass1_pipeline);
            pass.set_bind_group(0, &model.pass1_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if groups != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting LOD pass two"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass2_pipeline);
            pass.set_bind_group(0, &model.pass2_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        model.classification_epoch = epoch;
        model.resident_epoch.set(None);
        let face_count = model.prepared.residency.num_faces as u32;
        Ok(DeviceLodClassification {
            model: &*model,
            face_count,
            epoch,
        })
    }

    /// Append the exact resident shared-edge and within-face grading closure
    /// after one classifier output. Ten bounded Jacobi passes reach the least
    /// fixed point over the classifier's four-bit exponent lattice; the final
    /// pass re-canonicalizes topology and performs the resident atlas lookup.
    pub fn encode_resident_lod_reconciliation<'model>(
        &self,
        classification: &DeviceLodClassification<'model>,
        grading: FaceLodGrading,
        encoder: &mut wgpu::CommandEncoder,
    ) -> DeviceResidentLod<'model> {
        let model = classification.model;
        let groups = classification.face_count.div_ceil(LOD_WORKGROUP_SIZE);
        if groups != 0 {
            let (reconcile_pipeline, reconcile_forward_bind_group, reconcile_backward_bind_group) =
                match grading {
                    FaceLodGrading::TwoToOne => (
                        &self.resident_reconcile_2_to_1_pipeline,
                        &model.resident.reconcile_2_to_1_forward_bind_group,
                        &model.resident.reconcile_2_to_1_backward_bind_group,
                    ),
                    FaceLodGrading::FourToOne => (
                        &self.resident_reconcile_4_to_1_pipeline,
                        &model.resident.reconcile_4_to_1_forward_bind_group,
                        &model.resident.reconcile_4_to_1_backward_bind_group,
                    ),
                };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident LOD reconciliation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_seed_pipeline);
            pass.set_bind_group(0, &model.resident.seed_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);

            pass.set_pipeline(reconcile_pipeline);
            for iteration in 0..RESIDENT_LOD_RECONCILIATION_PASSES {
                let bind_group = if iteration.is_multiple_of(2) {
                    reconcile_forward_bind_group
                } else {
                    reconcile_backward_bind_group
                };
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }

            pass.set_pipeline(&self.resident_pack_pipeline);
            pass.set_bind_group(0, &model.resident.pack_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        model
            .resident_epoch
            .set(Some((classification.epoch, grading)));
        DeviceResidentLod {
            packed_records: &model.resident.packed_records,
            model_identity: model.identity,
            face_count: classification.face_count,
            classification_epoch: classification.epoch,
            grading,
        }
    }

    /// Borrow the most recently reconciled classifier epoch without copying
    /// its packed records. A subsequent classification invalidates this view
    /// until reconciliation publishes another complete resident epoch.
    pub fn latest_resident_lod<'model>(
        &self,
        model: &'model LodClassifierModel,
    ) -> Option<DeviceResidentLod<'model>> {
        let (classification_epoch, grading) = model.resident_epoch.get()?;
        Some(DeviceResidentLod {
            packed_records: &model.resident.packed_records,
            model_identity: model.identity,
            face_count: model.prepared.residency.num_faces as u32,
            classification_epoch,
            grading,
        })
    }

    /// Retire a resident visibility epoch when its camera/pose request is no
    /// longer coherent. Immutable model and scene residency stay intact.
    pub fn invalidate_resident_lod(&self, model: &LodClassifierModel) {
        model.resident_epoch.set(None);
    }

    /// Submit resident reconciliation without staging or readback. Callers
    /// building a larger render graph should prefer
    /// [`Self::encode_resident_lod_reconciliation`].
    pub fn reconcile_resident_lod_on_device<'model>(
        &self,
        classification: &DeviceLodClassification<'model>,
        grading: FaceLodGrading,
    ) -> DeviceResidentLod<'model> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting resident LOD reconciliation"),
            });
        let output = self.encode_resident_lod_reconciliation(classification, grading, &mut encoder);
        self.queue.submit([encoder.finish()]);
        output
    }

    /// Upload state, execute both classifier passes, and leave their packed
    /// output on the device. This is the authoritative fast path: it submits no
    /// copy to a staging buffer and never maps GPU memory.
    pub fn classify_on_device<'model>(
        &self,
        model: &'model mut LodClassifierModel,
        dispatch: &LodDispatchState,
        metrics: WgslLodDispatchMetrics,
        pose: LodPose<'_>,
        pose_upload: PoseUploadPolicy,
    ) -> Result<DeviceLodClassification<'model>, LodWebGpuError> {
        self.write_lod_classification_state(model, dispatch, metrics, pose, pose_upload)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting LOD classifier"),
            });
        let output = self.encode_lod_classification(model, &mut encoder)?;
        self.queue.submit([encoder.finish()]);
        Ok(output)
    }

    /// Publish classifier inputs, classify, and reconcile one resident LOD
    /// epoch in a single ordered GPU submission. The lower-level encode and
    /// submit helpers remain available for diagnostics and custom graphs.
    #[allow(clippy::too_many_arguments)]
    pub fn classify_and_reconcile_on_device<'model>(
        &self,
        model: &'model mut LodClassifierModel,
        dispatch: &LodDispatchState,
        metrics: WgslLodDispatchMetrics,
        pose: LodPose<'_>,
        pose_upload: PoseUploadPolicy,
        grading: FaceLodGrading,
    ) -> Result<DeviceResidentLod<'model>, LodWebGpuError> {
        self.write_lod_classification_state(model, dispatch, metrics, pose, pose_upload)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting classified resident LOD graph"),
            });
        let classification = self.encode_lod_classification(model, &mut encoder)?;
        let resident =
            self.encode_resident_lod_reconciliation(&classification, grading, &mut encoder);
        self.queue.submit([encoder.finish()]);
        Ok(resident)
    }

    /// Copy a device-resident result into temporary staging for conformance or
    /// diagnostics. Live rendering must bind the output buffer directly.
    pub async fn read_lod_classification_for_diagnostics(
        &self,
        output: &DeviceLodClassification<'_>,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let readback_bytes = u64::from(output.face_count) * PACKED_RECORD_BYTES;
        if readback_bytes == 0 {
            return Ok(Vec::new());
        }
        let readback = gpu_buffer(
            &self.device,
            "LOD temporary diagnostic readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting LOD diagnostic copy"),
            });
        encoder.copy_buffer_to_buffer(
            output.packed_records_buffer(),
            0,
            &readback,
            0,
            readback_bytes,
        );
        self.queue.submit([encoder.finish()]);
        self.readback_words(&readback, readback_bytes).await
    }

    /// Explicit full diagnostic readback of reconciled resident topology.
    pub async fn read_resident_lod_for_diagnostics(
        &self,
        output: &DeviceResidentLod<'_>,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let readback_bytes = u64::from(output.face_count) * PACKED_RECORD_BYTES;
        if readback_bytes == 0 {
            return Ok(Vec::new());
        }
        let readback = gpu_buffer(
            &self.device,
            "LOD resident temporary diagnostic readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting resident LOD diagnostic copy"),
            });
        encoder.copy_buffer_to_buffer(
            output.packed_records_buffer(),
            0,
            &readback,
            0,
            readback_bytes,
        );
        self.queue.submit([encoder.finish()]);
        self.readback_words(&readback, readback_bytes).await
    }

    /// Diagnostic compatibility wrapper around [`Self::classify_on_device`].
    /// It is intentionally unsuitable for an authoritative render loop.
    pub async fn classify(
        &self,
        model: &mut LodClassifierModel,
        dispatch: &LodDispatchState,
        metrics: WgslLodDispatchMetrics,
        pose: LodPose<'_>,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let output = self.classify_on_device(
            model,
            dispatch,
            metrics,
            pose,
            PoseUploadPolicy::Publish,
        )?;
        self.read_lod_classification_for_diagnostics(&output).await
    }

    /// Execute only the coherence/packing pass over caller-supplied
    /// intermediates. This diagnostic boundary exists to compare the device
    /// shader with the independent CPU pass-two oracle over a broad fixture
    /// matrix; it allocates temporary buffers and is not a runtime fast path.
    pub async fn reconcile_conformance_records(
        &self,
        pass1_records: &[[f32; 4]],
        adjacency: &[[u32; 4]],
        atlas_lut: &[u32],
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let face_count = pass1_records.len();
        if face_count == 0
            || adjacency.len() != face_count.saturating_mul(3)
            || atlas_lut.len() < 1_200
            || atlas_lut.iter().any(|&index| index > u8::MAX as u32)
        {
            return Err(LodWebGpuError::Payload(
                "pass-two conformance payload is malformed".to_string(),
            ));
        }
        let face_count_u32 = u32::try_from(face_count).map_err(|_| {
            LodWebGpuError::Payload("pass-two conformance face count exceeds u32".to_string())
        })?;
        let mut uniform_words = [0u32; 68];
        uniform_words[64] = face_count_u32;
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LOD pass two conformance uniform"),
                contents: bytemuck::cast_slice(&uniform_words),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pass1 = storage_buffer(
            &self.device,
            "LOD pass two conformance records",
            pass1_records,
            false,
        );
        let adjacency = storage_buffer(
            &self.device,
            "LOD pass two conformance adjacency",
            adjacency,
            false,
        );
        let atlas = storage_buffer(
            &self.device,
            "LOD pass two conformance atlas",
            atlas_lut,
            false,
        );
        let output_bytes = face_count as u64 * PACKED_RECORD_BYTES;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD pass two conformance output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD pass two conformance readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = self.pass2_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD pass two conformance bindings"),
            layout: &layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &pass1),
                bind(2, &adjacency),
                bind(3, &atlas),
                bind(4, &output),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LOD pass two conformance encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LOD pass two conformance dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass2_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(face_count_u32.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
        self.queue.submit([encoder.finish()]);

        self.readback_words(&readback, output_bytes).await
    }

    async fn readback_words(
        &self,
        readback: &wgpu::Buffer,
        readback_bytes: u64,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let slice = readback.slice(..readback_bytes);

        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| LodWebGpuError::Poll(error.to_string()))?;
        receiver
            .await
            .map_err(|_| LodWebGpuError::Mapping("map callback was canceled".to_string()))?
            .map_err(|error| LodWebGpuError::Mapping(error.to_string()))?;
        let view = slice.get_mapped_range();
        let packed = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        readback.unmap();
        Ok(packed)
    }
}

impl VisibilityCompactionScene {
    pub fn batch_count(&self) -> u32 {
        self.batch_count
    }

    pub fn source_count(&self) -> u32 {
        self.source_count
    }

    /// Device-aligned stride for one 16-byte batch-index uniform record.
    pub fn batch_index_uniform_stride(&self) -> u32 {
        self.batch_index_uniform_stride
    }

    /// Static table whose record at `batch * stride` names that batch. Render
    /// passes bind one record dynamically while ranges remain GPU-generated.
    pub fn batch_index_uniform_buffer(&self) -> &wgpu::Buffer {
        &self.batch_index_uniform
    }

    /// One `u32` visibility flag per stable source instance. A culling or LOD
    /// compute pass may bind this as writable storage before compaction.
    pub fn source_visibility_buffer(&self) -> &wgpu::Buffer {
        &self.source_visibility
    }

    /// Stable source-instance indirection consumed by a render vertex stage.
    pub fn compacted_source_instances_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_source_instances
    }

    /// One 20-byte source/compacted range record per retained batch.
    pub fn compacted_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_ranges
    }

    /// One exact triangle `DrawIndexedIndirect` record per retained batch.
    pub fn triangle_indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.triangle_indirect_arguments
    }

    /// One exact line `DrawIndexedIndirect` record per retained batch.
    pub fn line_indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.line_indirect_arguments
    }
}

impl PatchPreparationScene {
    pub fn patch_count(&self) -> u32 {
        self.patch_count
    }

    /// Canonical 208-byte records written by the preparation pass. A render
    /// pipeline can bind this directly as read-only storage without readback.
    pub fn prepared_records_buffer(&self) -> &wgpu::Buffer {
        &self.prepared_records
    }
}

impl PatchRenderScene {
    pub fn scene(&self) -> &RenderSceneSnapshot {
        self.scene.snapshot()
    }

    pub fn validated_scene(&self) -> &ValidatedRenderScene {
        &self.scene
    }

    /// Create one immutable low-rate command plan for this exact retained
    /// scene epoch. Cloned plans and frames share both scene and command
    /// allocations; style/command-presence changes deliberately require a new
    /// plan.
    pub fn command_plan(
        &self,
        style: RenderStyle,
        options: RenderFrameOptions,
    ) -> Result<RenderCommandPlan, LodWebGpuError> {
        RenderCommandPlan::build(&self.scene, style, options).map_err(|error| {
            LodWebGpuError::Payload(format!("retained WebGPU command plan: {error}"))
        })
    }

    pub fn patch_count(&self) -> u32 {
        self.patches.patch_count
    }

    pub fn batch_count(&self) -> u32 {
        self.visibility.batch_count
    }

    pub fn pbr_texture_residency(&self) -> Option<&[PbrMaterialTextureResidency]> {
        self.bindings
            .material_textures
            .as_ref()
            .map(PbrMaterialTextureBindings::residency)
    }

    pub fn pbr_environment_bindings(&self) -> Option<&PbrEnvironmentBindings> {
        self.bindings.pbr_environment.as_ref()
    }

    /// Whether this retained epoch can execute the declared basic PBR subset
    /// without semantic lowering or placeholder substitution.
    pub fn supports_resident_basic_pbr_frame(&self, options: RenderFrameOptions) -> bool {
        supports_basic_pbr_frame(self.scene.snapshot(), options)
            && self.pbr_texture_residency().is_some_and(|residency| {
                residency
                    .iter()
                    .all(|material| material.unresolved_mask() == 0)
            })
            && self
                .pbr_environment_bindings()
                .is_some_and(PbrEnvironmentBindings::is_resident)
    }

    /// Whether this coherent scene epoch can execute the focus-aware PBR
    /// path without lowering authored materials or substituting resources.
    pub fn supports_resident_focus_pbr_frame(&self, options: RenderFrameOptions) -> bool {
        supports_focus_pbr_frame(self.scene.snapshot(), options)
            && self.pbr_texture_residency().is_some_and(|residency| {
                residency
                    .iter()
                    .all(|material| material.unresolved_mask() == 0)
            })
            && self
                .pbr_environment_bindings()
                .is_some_and(PbrEnvironmentBindings::is_resident)
    }

    /// Whether this coherent scene epoch can be presented without semantic
    /// lowering. Diagnostic styles are unconditional; PBR is admitted only
    /// after its authored subset, texture table, and environment are exact.
    pub fn supports_resident_patch_presentation_frame(
        &self,
        style: RenderStyle,
        options: RenderFrameOptions,
    ) -> bool {
        supports_patch_presentation_style(style)
            || (style == RenderStyle::Pbr && self.supports_resident_basic_pbr_frame(options))
    }
}

impl OffscreenPatchRenderTarget {
    pub fn size(&self) -> [u32; 2] {
        self.size
    }
}

impl PackedPatchAtlas {
    pub fn entry(&self, key: [u32; 3]) -> Option<PackedPatchAtlasEntry> {
        self.entries.get(&key).copied()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    pub fn triangle_index_count(&self) -> u32 {
        self.triangle_index_count
    }

    pub fn line_index_count(&self) -> u32 {
        self.line_index_count
    }

    pub fn draw(&self, key: [u32; 3], geometry: RenderGeometry) -> Option<PatchAtlasDraw<'_>> {
        match geometry {
            RenderGeometry::Triangles => self.triangle_draw(key),
            RenderGeometry::Lines => self.line_draw(key),
        }
    }

    pub fn triangle_draw(&self, key: [u32; 3]) -> Option<PatchAtlasDraw<'_>> {
        let entry = self.entry(key)?;
        Some(PatchAtlasDraw {
            barycentric_buffer: &self.barycentric_buffer,
            index_buffer: &self.triangle_index_buffer,
            index_format: wgpu::IndexFormat::Uint32,
            first_index: entry.triangle_first_index,
            index_count: entry.triangle_index_count,
        })
    }

    pub fn line_draw(&self, key: [u32; 3]) -> Option<PatchAtlasDraw<'_>> {
        let entry = self.entry(key)?;
        Some(PatchAtlasDraw {
            barycentric_buffer: &self.barycentric_buffer,
            index_buffer: &self.line_index_buffer,
            index_format: wgpu::IndexFormat::Uint32,
            first_index: entry.line_first_index,
            index_count: entry.line_index_count,
        })
    }

    /// Resolve the exact packed-atlas entry required by every extracted batch.
    /// The returned views borrow the three retained global buffers and perform
    /// no GPU allocation or geometry copy.
    pub fn triangle_draws_for_scene(
        &self,
        scene: &RenderSceneSnapshot,
    ) -> Result<Vec<PatchAtlasDraw<'_>>, LodWebGpuError> {
        scene
            .validate()
            .map_err(|error| LodWebGpuError::Payload(format!("render scene contract: {error}")))?;
        scene
            .batches
            .iter()
            .enumerate()
            .map(|(batch_index, batch)| {
                let draw = self.triangle_draw(batch.id.key.lod).ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                        "packed WebGPU atlas is missing batch {batch_index} key {:?}",
                        batch.id.key.lod,
                    ))
                })?;
                if draw.index_count != batch.triangle_index_count {
                    return Err(LodWebGpuError::Payload(format!(
                        "packed WebGPU atlas key {:?} has {} triangle indices; batch {batch_index} requires {}",
                        batch.id.key.lod,
                        draw.index_count,
                        batch.triangle_index_count,
                    )));
                }
                Ok(draw)
            })
            .collect()
    }
}

fn require_normals_frame_pipeline(
    frame: &RenderFrame,
    pipeline: &PatchRenderPipeline,
) -> Result<(), LodWebGpuError> {
    if frame.style == RenderStyle::Normals && pipeline.style() == Some(RenderStyle::Normals) {
        Ok(())
    } else {
        Err(LodWebGpuError::Payload(
            "normals render entry point requires a normals frame and pipeline".to_string(),
        ))
    }
}

impl PatchRenderPipeline {
    /// Encode one compacted indirect QB batch. The caller selects the atlas
    /// buffers corresponding to that batch's canonical LOD key.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_batch<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        bindings: &'pass PatchRenderBindings,
        visibility: &'pass VisibilityCompactionScene,
        atlas: PatchAtlasDraw<'pass>,
        batch_index: u32,
        material_slot: u32,
        winding: PatchWinding,
    ) -> Result<(), LodWebGpuError> {
        if batch_index >= visibility.batch_count {
            return Err(LodWebGpuError::Payload(
                "patch render batch index is out of range".to_string(),
            ));
        }
        let dynamic_offset = batch_index
            .checked_mul(visibility.batch_index_uniform_stride)
            .ok_or_else(|| {
                LodWebGpuError::Payload("patch render batch offset exceeds u32".to_string())
            })?;
        let pipeline = match winding {
            PatchWinding::CounterClockwise => &self.counter_clockwise,
            PatchWinding::Clockwise => &self.clockwise,
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings.bind_group, &[dynamic_offset]);
        if self.uses_pbr_bindings() {
            let material_textures = bindings
                .material_textures
                .as_ref()
                .ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "PBR material texture bindings are not resident".to_string(),
                    )
                })?
                .bind_group(material_slot)
                .ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                        "PBR material texture slot {material_slot} is not resident",
                    ))
                })?;
            pass.set_bind_group(1, material_textures, &[]);
            let environment = bindings.pbr_environment.as_ref().ok_or_else(|| {
                LodWebGpuError::Payload("PBR environment bindings are not resident".to_string())
            })?;
            pass.set_bind_group(2, environment.bind_group(), &[]);
        }
        pass.set_vertex_buffer(0, atlas.barycentric_buffer.slice(..));
        let index_width = match atlas.index_format {
            wgpu::IndexFormat::Uint16 => 2,
            wgpu::IndexFormat::Uint32 => 4,
        };
        let index_byte_offset = u64::from(atlas.first_index) * index_width;
        pass.set_index_buffer(
            atlas.index_buffer.slice(index_byte_offset..),
            atlas.index_format,
        );
        let indirect_arguments = match self.geometry {
            RenderGeometry::Triangles => &visibility.triangle_indirect_arguments,
            RenderGeometry::Lines => &visibility.line_indirect_arguments,
        };
        pass.draw_indexed_indirect(
            indirect_arguments,
            u64::from(batch_index) * INDEXED_INDIRECT_RECORD_BYTES,
        );
        Ok(())
    }
}

impl DiagnosticPatchRenderPipelines {
    pub fn get(&self, style: RenderStyle) -> Option<&PatchRenderPipeline> {
        match style {
            RenderStyle::Pbr => Some(&self.pbr),
            RenderStyle::Normals => Some(&self.normals),
            RenderStyle::Matcap => Some(&self.matcap),
            RenderStyle::Lod => Some(&self.lod),
            RenderStyle::Stretch => Some(&self.stretch),
            RenderStyle::Wire => Some(&self.wire),
            _ => None,
        }
    }

    fn get_for_pass(
        &self,
        pass: RenderPass,
        geometry: RenderGeometry,
    ) -> Result<&PatchRenderPipeline, LodWebGpuError> {
        let pipeline = match pass {
            RenderPass::PbrOpaque => &self.pbr,
            RenderPass::Matcap => &self.matcap,
            RenderPass::Wire => &self.wire,
            RenderPass::Normals => &self.normals,
            RenderPass::Lod => &self.lod,
            RenderPass::Stretch => &self.stretch,
            RenderPass::PbrTransparent => {
                return Err(LodWebGpuError::Payload(
                    "WebGPU patch renderer does not yet support non-opaque PBR".to_string(),
                ));
            }
        };
        if pipeline.geometry != geometry {
            return Err(LodWebGpuError::Payload(format!(
                "WebGPU {pass:?} pipeline uses {:?}, but command requires {geometry:?}",
                pipeline.geometry,
            )));
        }
        Ok(pipeline)
    }
}

fn words_to_patch_records(
    words: Vec<u32>,
) -> Result<Vec<[u32; PREPARED_PATCH_RECORD_WORDS]>, LodWebGpuError> {
    if !words.len().is_multiple_of(PREPARED_PATCH_RECORD_WORDS) {
        return Err(LodWebGpuError::Mapping(format!(
            "prepared-patch readback does not contain {PREPARED_PATCH_RECORD_WORDS}-word records"
        )));
    }
    Ok(words
        .chunks_exact(PREPARED_PATCH_RECORD_WORDS)
        .map(|record| std::array::from_fn(|word| record[word]))
        .collect())
}

fn words_to_five_records(words: Vec<u32>, label: &str) -> Result<Vec<[u32; 5]>, LodWebGpuError> {
    if !words.len().is_multiple_of(5) {
        return Err(LodWebGpuError::Mapping(format!(
            "{label} readback does not contain five-word records"
        )));
    }
    Ok(words
        .chunks_exact(5)
        .map(|record| [record[0], record[1], record[2], record[3], record[4]])
        .collect())
}

fn words_to_twelve_records(words: Vec<u32>) -> Result<Vec<[u32; 12]>, LodWebGpuError> {
    if !words.len().is_multiple_of(12) {
        return Err(LodWebGpuError::Mapping(
            "resident root topology readback does not contain twelve-word records".to_string(),
        ));
    }
    Ok(words
        .chunks_exact(12)
        .map(|record| std::array::from_fn(|word| record[word]))
        .collect())
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn render_buffer_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    min_binding_size: u64,
    has_dynamic_offset: bool,
) -> wgpu::BindGroupLayoutEntry {
    render_buffer_layout_visible(
        binding,
        ty,
        min_binding_size,
        has_dynamic_offset,
        wgpu::ShaderStages::VERTEX,
    )
}

fn render_buffer_layout_visible(
    binding: u32,
    ty: wgpu::BufferBindingType,
    min_binding_size: u64,
    has_dynamic_offset: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size: std::num::NonZeroU64::new(min_binding_size),
        },
        count: None,
    }
}

fn buffer_init_or_zero(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let zero = [0u8; 4];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: if contents.is_empty() { &zero } else { contents },
        usage,
    })
}

fn gpu_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(PACKED_RECORD_BYTES),
        usage,
        mapped_at_creation: false,
    })
}

fn storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    records: &[T],
    copy_dst: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_dst {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(records),
        usage,
    })
}

/// Request a browser WebGPU device and execute the same exact matrix as the
/// native hardware gate. Enabled only for the standalone browser harness.
#[cfg(all(target_arch = "wasm32", feature = "browser-conformance"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_browser_lod_conformance(
    canvas: web_sys::HtmlCanvasElement,
) -> Result<String, wasm_bindgen::JsValue> {
    let size = [canvas.width(), canvas.height()];
    let (classifier, adapter, mut presentation) = LodClassifierDevice::request_canvas_presentation(
        canvas,
        size,
        "quilting browser LOD conformance",
    )
    .await
    .map_err(browser_error)?;
    let report = classifier
        .run_conformance_matrix_with_surface(&mut presentation)
        .await
        .map_err(browser_error)?;
    let surface = presentation.diagnostics();
    Ok(format!(
        "adapter={} backend={} surface={}x{} surface_format={} surface_presented={} surface_reconfigurations={} full_pipeline_words={} resident_lod_words={} resident_visibility_words={} resident_bucket_words={} resident_root_topology_words={} resident_root_prepared_words={} resident_root_domain_words={} resident_adaptive_rendered_pixels={} resident_adaptive_image_hash={:016x} resident_root_indirect_draws={} adaptive_overlay_patches={} adaptive_overlay_indirect_draws={} coherence_words={} \
         prepared_patch_words={} rendered_patch_pixels={} shared_frame_draws={} compacted_source_words={} \
         compacted_range_words={} indirect_argument_words={} indirect_draws={}",
        adapter.name,
        adapter.backend,
        surface.size[0],
        surface.size[1],
        surface.color_format,
        surface.frames_presented,
        surface.reconfigurations,
        report.full_pipeline_words,
        report.resident_lod_words,
        report.resident_visibility_words,
        report.resident_bucket_words,
        report.resident_root_topology_words,
        report.resident_root_prepared_words,
        report.resident_root_domain_words,
        report.resident_adaptive_rendered_pixels,
        report.resident_adaptive_image.rgba8_hash,
        report.resident_root_indirect_draws,
        report.adaptive_overlay_patches,
        report.adaptive_overlay_indirect_draws,
        report.coherence_words,
        report.prepared_patch_words,
        report.rendered_patch_pixels,
        report.shared_frame_draws,
        report.compacted_source_words,
        report.compacted_range_words,
        report.indirect_argument_words,
        report.indirect_draws,
    ))
}

#[cfg(all(target_arch = "wasm32", feature = "browser-conformance"))]
fn browser_error(error: impl std::fmt::Display) -> wasm_bindgen::JsValue {
    wasm_bindgen::JsValue::from_str(&error.to_string())
}
