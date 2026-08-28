//! Shadow WebGPU execution for Quilting's two-pass LOD classifier.
//!
//! This crate is intentionally outside the current WebGL2 authority path. It
//! consumes the exact backend-neutral payload frozen by `quilting-renderer`,
//! retains device pipelines and model buffers, and returns packed words for
//! conformance comparison. It owns no scene, FRP, or replicated state.

use futures_channel::oneshot;
use quilting_renderer::compute::{
    pack_wgsl_lod_atlas_words, pack_wgsl_lod_dispatch_words, pack_wgsl_lod_model_words,
    pack_wgsl_lod_subject_words, pack_wgsl_source_visibility_words, prepare_lod_atlas_lookup,
    prepare_lod_model, reconcile_and_pack_wgsl_lod_pass2, LodAtlasLookup, LodDispatchState,
    LodModelData, PreparedLodModel, WgslLodDispatchMetrics, WgslPatchPreparationSceneWords,
    WgslVisibilityCompactionSceneWords,
};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const LOD_WORKGROUP_SIZE: u32 = 64;
const DISPATCH_UNIFORM_BYTES: u64 = 272;
const SUBJECT_RECORD_BYTES: u64 = 160;
const JOINT_MATRIX_BYTES: u64 = 64;
const PASS1_RECORD_BYTES: u64 = 16;
const PACKED_RECORD_BYTES: u64 = 4;
const VISIBILITY_UNIFORM_BYTES: u64 = 16;
const VISIBILITY_BATCH_RECORD_BYTES: u64 = 16;
const VISIBILITY_RANGE_RECORD_BYTES: u64 = 20;
const INDEXED_INDIRECT_RECORD_BYTES: u64 = 20;
const DRAW_BATCH_INDEX_BYTES: u64 = 16;
const PATCH_PREPARE_UNIFORM_BYTES: u64 = 16;
const PATCH_TOPOLOGY_RECORD_BYTES: u64 = 48;
const PREPARED_PATCH_RECORD_WORDS: usize = 52;
const PREPARED_PATCH_RECORD_BYTES: u64 = 208;
const PATCH_SUBJECT_RECORD_BYTES: u64 = 128;
const PATCH_RENDER_FRAME_WORDS: usize = 56;
const PATCH_RENDER_FRAME_BYTES: u64 = 224;

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
        topology_record([2.0, 4.0, 8.0, 5.0], [0.0, 3.0, 6.0, 9.0], [0.0, 0.0]),
        topology_record([1.0, 2.0, 4.0, 3.0], [0.0, 4.0, 5.0, 6.0], [1.0, 3.0]),
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
    for (word, value) in root[24..28].iter_mut().zip([2.0, 4.0, 8.0, 5.0]) {
        *word = bits(value);
    }
    for (word, value) in root[28..32].iter_mut().zip([3.0, 6.0, 9.0, 0.0]) {
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
    for (word, value) in child[24..28].iter_mut().zip([1.0, 2.0, 4.0, 3.0]) {
        *word = bits(value);
    }
    for (word, value) in child[28..32].iter_mut().zip([4.0, 5.0, 6.0, 0.0]) {
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

/// Bounded evidence returned after the shared device conformance matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodDeviceConformance {
    pub full_pipeline_words: usize,
    pub coherence_words: usize,
    pub prepared_patch_words: usize,
    pub rendered_patch_pixels: usize,
    pub compacted_source_words: usize,
    pub compacted_range_words: usize,
    pub indirect_argument_words: usize,
    pub indirect_draws: usize,
}

/// Diagnostic copy of the exact same-device visibility outputs. The retained
/// GPU buffers remain suitable for direct storage/indirect consumption; this
/// owned projection exists only for conformance gates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibilityCompactionOutput {
    pub compacted_source_instances: Vec<u32>,
    pub compacted_ranges: Vec<[u32; 5]>,
    pub indirect_arguments: Vec<[u32; 5]>,
}

/// Backend-neutral dynamic values consumed by the WebGPU prepared-surface
/// evaluator. Matrices use the same column-major convention as WebGL2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchRenderFrame {
    pub mvp: [f32; 16],
    pub mv: [f32; 16],
    pub use_qb: bool,
    pub mobius: [f32; 16],
    pub camera_position: [f32; 3],
}

impl PatchRenderFrame {
    fn to_words(self) -> Result<[u32; PATCH_RENDER_FRAME_WORDS], LodWebGpuError> {
        if self
            .mvp
            .into_iter()
            .chain(self.mv)
            .chain(self.mobius)
            .chain(self.camera_position)
            .any(|value| !value.is_finite())
        {
            return Err(LodWebGpuError::Payload(
                "patch render frame contains non-finite values".to_string(),
            ));
        }
        let mut words = [0u32; PATCH_RENDER_FRAME_WORDS];
        for (word, value) in words[..16].iter_mut().zip(self.mvp) {
            *word = value.to_bits();
        }
        for (word, value) in words[16..32].iter_mut().zip(self.mv) {
            *word = value.to_bits();
        }
        words[32] = u32::from(self.use_qb);
        for (word, value) in words[36..52].iter_mut().zip(self.mobius) {
            *word = value.to_bits();
        }
        for (word, value) in words[52..55].iter_mut().zip(self.camera_position) {
            *word = value.to_bits();
        }
        Ok(words)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchWinding {
    CounterClockwise,
    Clockwise,
}

/// Device-local pipelines shared by every uploaded classifier model.
pub struct LodClassifierDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass1_pipeline: wgpu::ComputePipeline,
    pass2_pipeline: wgpu::ComputePipeline,
    patch_prepare_pipeline: wgpu::ComputePipeline,
    visibility_count_pipeline: wgpu::ComputePipeline,
    visibility_scan_pipeline: wgpu::ComputePipeline,
    visibility_scatter_pipeline: wgpu::ComputePipeline,
}

/// Retained device buffers for one immutable prepared model and atlas lookup.
pub struct LodClassifierModel {
    prepared: PreparedLodModel,
    joint_capacity: usize,
    subject_rows: usize,
    uniform: wgpu::Buffer,
    skinning: wgpu::Buffer,
    joint_matrices: wgpu::Buffer,
    morph_deltas: wgpu::Buffer,
    morph_weights: wgpu::Buffer,
    subject_states: wgpu::Buffer,
    packed_records: wgpu::Buffer,
    readback: wgpu::Buffer,
    pass1_bind_group: wgpu::BindGroup,
    pass2_bind_group: wgpu::BindGroup,
}

/// Retained current-scene topology and prepared-patch output. Immutable source
/// faces and affine subject rows are uploaded once; only pose buffers and the
/// joint-count word change with animation.
pub struct PatchPreparationScene {
    patch_count: u32,
    uniform_words: [u32; 4],
    uniform: wgpu::Buffer,
    prepared_records: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Retained WebGPU graphics pipelines for the first shared QB surface mode.
/// Winding is pipeline state in WebGPU, so both variants share one explicit
/// bind-group layout instead of recompiling or mutating state per batch.
pub struct PatchRenderPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    counter_clockwise: wgpu::RenderPipeline,
    clockwise: wgpu::RenderPipeline,
}

/// Scene/frame bindings shared by every atlas bucket and indirect batch draw.
pub struct PatchRenderBindings {
    frame: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Retained scene shape and output buffers for deterministic visibility
/// compaction. Only `source_visibility` changes with the current pose.
pub struct VisibilityCompactionScene {
    batch_count: u32,
    source_count: u32,
    batch_index_uniform_stride: u32,
    batch_index_uniform: wgpu::Buffer,
    source_visibility: wgpu::Buffer,
    compacted_source_instances: wgpu::Buffer,
    compacted_ranges: wgpu::Buffer,
    indirect_arguments: wgpu::Buffer,
    count_bind_group: wgpu::BindGroup,
    scan_bind_group: wgpu::BindGroup,
    scatter_bind_group: wgpu::BindGroup,
}

impl LodClassifierDevice {
    /// Compile and retain the two flattened WGSL pipelines on an existing
    /// device. Device creation and adapter policy stay with the application.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, LodWebGpuError> {
        let pass1_source = quilting_shaders::compile_lod_pass1_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass2_source = quilting_shaders::compile_lod_pass2_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let patch_prepare_source = quilting_shaders::compile_patch_prepare_compute_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_count_source = quilting_shaders::compile_visibility_count_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_scan_source = quilting_shaders::compile_visibility_scan_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_scatter_source = quilting_shaders::compile_visibility_scatter_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass1_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass one"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass1_source)),
        });
        let pass2_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass two"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass2_source)),
        });
        let patch_prepare_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting patch preparation"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(patch_prepare_source)),
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
        let patch_prepare_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting patch preparation"),
                layout: None,
                module: &patch_prepare_module,
                entry_point: Some(quilting_shaders::PATCH_PREPARE_DEVICE_ENTRY_POINT),
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
        Ok(Self {
            device,
            queue,
            pass1_pipeline,
            pass2_pipeline,
            patch_prepare_pipeline,
            visibility_count_pipeline,
            visibility_scan_pipeline,
            visibility_scatter_pipeline,
        })
    }

    /// Run the same minimum exact matrix on native and browser devices.
    ///
    /// This covers one complete animated-capable two-pass dispatch plus all S3
    /// permutations, visible-only neighbor promotion, invisible records,
    /// priorities, and multiple atlas keys in the coherence pass.
    pub async fn run_conformance_matrix(&self) -> Result<LodDeviceConformance, LodWebGpuError> {
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
        let actual = self
            .classify(
                &mut resident,
                &dispatch,
                metrics,
                LodPose {
                    joint_matrices: &joint_matrices,
                    morph_weights: &[],
                },
            )
            .await?;
        if actual != expected {
            return Err(LodWebGpuError::Conformance(format!(
                "full pipeline mismatch: expected {expected:?}, got {actual:?}"
            )));
        }
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
        let patch_model = self.upload_model(patch_prepared, &atlas)?;
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
        let rendered_patch_pixels = self
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
            batches: vec![[0, 130, 6, 0], [130, 3, 12, 0], [133, 4, 18, 0]],
            source_eligibility: [vec![1; 130], vec![0; 3], vec![1; 4]].concat(),
        };
        let mut expected_sources = (0..batch_zero_count)
            .filter(|index| index % 3 != 0)
            .collect::<Vec<_>>();
        expected_sources.extend([133, 135, 136]);
        let expected_ranges = vec![[0, 0, 130, 0, 86], [1, 130, 3, 86, 0], [2, 133, 4, 86, 3]];
        let expected_indirect = vec![[6, 86, 0, 0, 0], [12, 0, 0, 0, 0], [18, 3, 0, 0, 0]];
        let mut compaction = self.upload_visibility_compaction_scene(compaction_words)?;
        let compacted = self
            .compact_visibility(&mut compaction, &source_visibility)
            .await?;
        if compacted.compacted_source_instances != expected_sources
            || compacted.compacted_ranges != expected_ranges
            || compacted.indirect_arguments != expected_indirect
        {
            return Err(LodWebGpuError::Conformance(format!(
                "visibility compaction mismatch: expected sources {expected_sources:?}, ranges \
                 {expected_ranges:?}, indirect {expected_indirect:?}; got {compacted:?}"
            )));
        }
        let indirect_draws = self
            .validate_indirect_draw_conformance(&compaction, &source_visibility)
            .await?;

        Ok(LodDeviceConformance {
            full_pipeline_words,
            coherence_words: actual.len(),
            prepared_patch_words,
            rendered_patch_pixels,
            compacted_source_words: compacted.compacted_source_instances.len(),
            compacted_range_words: compacted.compacted_ranges.len() * 5,
            indirect_argument_words: compacted.indirect_arguments.len() * 5,
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

        let uniform = buffer_init_or_zero(
            &self.device,
            "patch preparation uniform",
            bytemuck::cast_slice(&words.uniform),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let inert_topology = [[0u32; 12]];
        let topology = buffer_init_or_zero(
            &self.device,
            "patch preparation topology",
            if words.topology.is_empty() {
                bytemuck::cast_slice(&inert_topology)
            } else {
                bytemuck::cast_slice(&words.topology)
            },
            wgpu::BufferUsages::STORAGE,
        );
        let source_faces = buffer_init_or_zero(
            &self.device,
            "patch preparation source faces",
            bytemuck::cast_slice(&words.source_faces),
            wgpu::BufferUsages::STORAGE,
        );
        let subjects = buffer_init_or_zero(
            &self.device,
            "patch preparation subjects",
            bytemuck::cast_slice(&words.subjects),
            wgpu::BufferUsages::STORAGE,
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
                bind(2, &source_faces),
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
            uniform_words: words.uniform,
            uniform,
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
        let mut uniform_words = scene.uniform_words;
        uniform_words[2] = num_joints;
        self.queue
            .write_buffer(&scene.uniform, 0, bytemuck::cast_slice(&uniform_words));
        Ok(())
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
        if sample_count == 0 {
            return Err(LodWebGpuError::Payload(
                "patch render sample count must be nonzero".to_string(),
            ));
        }
        let source = quilting_shaders::compile_patch_render_device_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quilting prepared QB render"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
            });
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("quilting prepared QB render bindings"),
                    entries: &[
                        render_buffer_layout(
                            0,
                            wgpu::BufferBindingType::Uniform,
                            PATCH_RENDER_FRAME_BYTES,
                            false,
                        ),
                        render_buffer_layout(
                            1,
                            wgpu::BufferBindingType::Storage { read_only: true },
                            PREPARED_PATCH_RECORD_BYTES,
                            false,
                        ),
                        render_buffer_layout(
                            2,
                            wgpu::BufferBindingType::Storage { read_only: true },
                            PACKED_RECORD_BYTES,
                            false,
                        ),
                        render_buffer_layout(
                            3,
                            wgpu::BufferBindingType::Storage { read_only: true },
                            VISIBILITY_RANGE_RECORD_BYTES,
                            false,
                        ),
                        render_buffer_layout(
                            4,
                            wgpu::BufferBindingType::Uniform,
                            DRAW_BATCH_INDEX_BYTES,
                            true,
                        ),
                    ],
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quilting prepared QB render pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let create = |front_face| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("quilting prepared QB normals"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(quilting_shaders::PATCH_RENDER_DEVICE_VERTEX_ENTRY_POINT),
                        compilation_options: Default::default(),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &attributes,
                        }],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: depth_stencil.clone(),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(
                            quilting_shaders::PATCH_RENDER_DEVICE_NORMALS_ENTRY_POINT,
                        ),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let counter_clockwise = create(wgpu::FrontFace::Ccw);
        let clockwise = create(wgpu::FrontFace::Cw);
        Ok(PatchRenderPipeline {
            bind_group_layout,
            counter_clockwise,
            clockwise,
        })
    }

    /// Bind the output of preparation and stable visibility compaction
    /// directly. Empty scenes need no render bindings and are rejected here.
    pub fn create_patch_render_bindings(
        &self,
        pipeline: &PatchRenderPipeline,
        patches: &PatchPreparationScene,
        visibility: &VisibilityCompactionScene,
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
        let frame = gpu_buffer(
            &self.device,
            "patch render frame",
            PATCH_RENDER_FRAME_BYTES,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting prepared QB render bindings"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                bind(0, &frame),
                bind(1, &patches.prepared_records),
                bind(2, &visibility.compacted_source_instances),
                bind(3, &visibility.compacted_ranges),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &visibility.batch_index_uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(DRAW_BATCH_INDEX_BYTES),
                    }),
                },
            ],
        });
        Ok(PatchRenderBindings { frame, bind_group })
    }

    pub fn write_patch_render_frame(
        &self,
        bindings: &PatchRenderBindings,
        frame: PatchRenderFrame,
    ) -> Result<(), LodWebGpuError> {
        let words = frame.to_words()?;
        debug_assert_eq!(
            std::mem::size_of_val(&words) as u64,
            PATCH_RENDER_FRAME_BYTES
        );
        self.queue
            .write_buffer(&bindings.frame, 0, bytemuck::cast_slice(&words));
        Ok(())
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
            if batch[0] != expected_first || batch[3] != 0 {
                return Err(LodWebGpuError::Payload(
                    "visibility compaction batch ranges are not canonical".to_string(),
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
            wgpu::BufferUsages::STORAGE,
        );
        let source_eligibility = buffer_init_or_zero(
            &self.device,
            "visibility compaction eligibility",
            bytemuck::cast_slice(&words.source_eligibility),
            wgpu::BufferUsages::STORAGE,
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
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        let indirect_arguments = gpu_buffer(
            &self.device,
            "visibility indirect arguments",
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
                bind(4, &indirect_arguments),
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
            source_visibility,
            compacted_source_instances,
            compacted_ranges,
            indirect_arguments,
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
                indirect_arguments: Vec::new(),
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
        let indirect_readback = gpu_buffer(
            &self.device,
            "visibility indirect diagnostic readback",
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
            &scene.indirect_arguments,
            0,
            &indirect_readback,
            0,
            indirect_bytes,
        );
        self.queue.submit([encoder.finish()]);

        let compacted_ranges = words_to_five_records(
            self.readback_words(&range_readback, range_bytes).await?,
            "visibility range",
        )?;
        let indirect_arguments = words_to_five_records(
            self.readback_words(&indirect_readback, indirect_bytes)
                .await?,
            "visibility indirect",
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
            indirect_arguments,
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
    ) -> Result<usize, LodWebGpuError> {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 32;
        const PADDED_BYTES_PER_ROW: u32 = 256;

        let visibility_words = WgslVisibilityCompactionSceneWords {
            uniform: [1, patches.patch_count, 0, 0],
            batches: vec![[0, patches.patch_count, 3, 0]],
            source_eligibility: vec![1; patches.patch_count as usize],
        };
        let visibility = self.upload_visibility_compaction_scene(visibility_words)?;
        let pipeline =
            self.create_patch_render_pipeline(wgpu::TextureFormat::Rgba8Unorm, None, 1)?;
        let bindings = self.create_patch_render_bindings(&pipeline, patches, &visibility)?;
        let mvp = [
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, -0.25, 0.5, 1.0,
        ];
        self.write_patch_render_frame(
            &bindings,
            PatchRenderFrame {
                mvp,
                mv: identity_matrix(),
                use_qb: true,
                mobius: identity_mobius(),
                camera_position: [0.0, 0.0, 4.0],
            },
        )?;
        self.write_patch_pose(model, patches, pose, num_joints)?;
        let source_visibility = vec![1; patches.patch_count as usize];
        self.write_source_visibility(&visibility, &source_visibility)?;

        let barycentrics = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let barycentric_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("patch render conformance barycentrics"),
                    contents: bytemuck::cast_slice(&barycentrics),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        let indices = [0u32, 1, 2];
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("patch render conformance indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
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
        self.encode_patch_preparation(patches, &mut encoder);
        self.encode_visibility_compaction(&visibility, &mut encoder);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("patch render conformance pass"),
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
            pipeline.draw_batch(
                &mut pass,
                &bindings,
                &visibility,
                &barycentric_buffer,
                &index_buffer,
                wgpu::IndexFormat::Uint32,
                0,
                PatchWinding::CounterClockwise,
            )?;
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
        let rendered = pixels
            .chunks_exact(words_per_row)
            .take(HEIGHT as usize)
            .flat_map(|row| &row[..WIDTH as usize])
            .filter(|&&pixel| pixel != 0)
            .count();
        if !(8..WIDTH as usize * HEIGHT as usize).contains(&rendered) {
            return Err(LodWebGpuError::Conformance(format!(
                "patch render produced an implausible {rendered}-pixel footprint"
            )));
        }
        Ok(rendered)
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
                    &scene.indirect_arguments,
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

    /// Upload immutable geometry/topology once and allocate retained dynamic
    /// buffers. The current implementation deliberately keeps readback for
    /// shadow conformance; an authoritative backend will consume packed words
    /// on-device instead.
    pub fn upload_model(
        &self,
        prepared: PreparedLodModel,
        atlas: &LodAtlasLookup,
    ) -> Result<LodClassifierModel, LodWebGpuError> {
        let words = pack_wgsl_lod_model_words(&prepared).map_err(LodWebGpuError::Payload)?;
        let atlas_words = pack_wgsl_lod_atlas_words(atlas);
        let face_count = prepared.residency.num_faces;
        let subject_rows = words
            .faces
            .iter()
            .map(|face| face[3] as usize)
            .max()
            .map_or(1, |row| row + 1);
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
        let packed_records = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD packed records"),
            size: face_count as u64 * PACKED_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD diagnostic readback"),
            size: face_count as u64 * PACKED_RECORD_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        Ok(LodClassifierModel {
            prepared,
            joint_capacity,
            subject_rows,
            uniform,
            skinning,
            joint_matrices,
            morph_deltas,
            morph_weights,
            subject_states,
            packed_records,
            readback,
            pass1_bind_group,
            pass2_bind_group,
        })
    }

    /// Execute both passes in one command buffer and return the diagnostic
    /// packed words only after the device signals the map callback.
    pub async fn classify(
        &self,
        model: &mut LodClassifierModel,
        dispatch: &LodDispatchState,
        metrics: WgslLodDispatchMetrics,
        pose: LodPose<'_>,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        self.write_dynamic_pose(model, pose, metrics.num_joints)?;
        let subject_words = pack_wgsl_lod_subject_words(&model.prepared, dispatch)
            .map_err(LodWebGpuError::Payload)?;
        if subject_words.len() != model.subject_rows {
            return Err(LodWebGpuError::Payload(
                "subject table changed immutable shape".to_string(),
            ));
        }
        let uniform_words = pack_wgsl_lod_dispatch_words(&model.prepared, dispatch, metrics)
            .map_err(LodWebGpuError::Payload)?;
        self.queue
            .write_buffer(&model.uniform, 0, bytemuck::cast_slice(&uniform_words));
        self.queue.write_buffer(
            &model.subject_states,
            0,
            bytemuck::cast_slice(&subject_words),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting LOD classifier"),
            });
        let groups = (model.prepared.residency.num_faces as u32).div_ceil(LOD_WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting LOD pass one"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass1_pipeline);
            pass.set_bind_group(0, &model.pass1_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting LOD pass two"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass2_pipeline);
            pass.set_bind_group(0, &model.pass2_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let readback_bytes = model.prepared.residency.num_faces as u64 * PACKED_RECORD_BYTES;
        encoder.copy_buffer_to_buffer(&model.packed_records, 0, &model.readback, 0, readback_bytes);
        self.queue.submit([encoder.finish()]);

        self.readback_words(&model.readback, readback_bytes).await
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

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
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

    /// One exact 20-byte `DrawIndexedIndirect` record per retained batch.
    pub fn indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_arguments
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

impl PatchRenderPipeline {
    /// Encode one compacted indirect QB batch. The caller selects the atlas
    /// buffers corresponding to that batch's canonical LOD key.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_batch<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
        bindings: &'pass PatchRenderBindings,
        visibility: &'pass VisibilityCompactionScene,
        barycentric_buffer: &'pass wgpu::Buffer,
        index_buffer: &'pass wgpu::Buffer,
        index_format: wgpu::IndexFormat,
        batch_index: u32,
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
        pass.set_vertex_buffer(0, barycentric_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), index_format);
        pass.draw_indexed_indirect(
            &visibility.indirect_arguments,
            u64::from(batch_index) * INDEXED_INDIRECT_RECORD_BYTES,
        );
        Ok(())
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
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
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
pub async fn run_browser_lod_conformance() -> Result<String, wasm_bindgen::JsValue> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(browser_error)?;
    let adapter_info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("quilting browser LOD conformance"),
            ..Default::default()
        })
        .await
        .map_err(browser_error)?;
    let classifier = LodClassifierDevice::new(device, queue).map_err(browser_error)?;
    let report = classifier
        .run_conformance_matrix()
        .await
        .map_err(browser_error)?;
    Ok(format!(
        "adapter={} backend={:?} full_pipeline_words={} coherence_words={} \
         prepared_patch_words={} rendered_patch_pixels={} compacted_source_words={} \
         compacted_range_words={} indirect_argument_words={} indirect_draws={}",
        adapter_info.name,
        adapter_info.backend,
        report.full_pipeline_words,
        report.coherence_words,
        report.prepared_patch_words,
        report.rendered_patch_pixels,
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
